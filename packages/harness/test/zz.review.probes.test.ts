/*
 * TEMPORARY review probes — not for committing.
 *
 * Each probe pins down one hypothesis about the entry-lifecycle change under
 * review. They are written against the CURRENT implementation; where a probe's
 * expectation differs from what the implementation does, the test name says
 * which is which.
 */
import { createAssistantMessageEventStream, Message } from "@codeworksh/aikit";
import { DateTime, Deferred, Effect, Exit, Fiber, Layer, Option, Schema } from "effect";
import { SqlClient } from "effect/unstable/sql";
import { describe, expect } from "vite-plus/test";
import { Context } from "../src/context/context.ts";
import { Control } from "../src/control.ts";
import { Database } from "../src/db/db.ts";
import { Event } from "../src/event/event.ts";
import { EventList } from "../src/event/list.ts";
import { RunnerExecute } from "../src/runner/execute.ts";
import { RunnerExecution } from "../src/runner/execution.ts";
import { LLM } from "../src/runner/llm.ts";
import { Loop } from "../src/runner/loop.ts";
import { SandboxController } from "../src/sandbox/control.ts";
import { SandboxDriver } from "../src/sandbox/driver.ts";
import { FakeSandboxDriver } from "../src/sandbox/drivers/fake.ts";
import { SandboxIO } from "../src/sandbox/io.ts";
import { AbsolutePath } from "../src/schema.ts";
import { SessionInput } from "../src/session/input/input.ts";
import { SessionLive } from "../src/session/live.ts";
import { SessionMessageSchema } from "../src/session/message/schema.ts";
import { SessionProjector } from "../src/session/projector.ts";
import type { SessionSchema } from "../src/session/schema.ts";
import { Session } from "../src/session/session.ts";
import { State } from "../src/state/state.ts";
import { ToolProgress } from "../src/tools/progress.ts";
import * as Tool from "../src/tools/tool.ts";
import { testEffect } from "./utils/effect.ts";

/* ------------------------------------------------------------------ */
/* Projection-level harness (mirrors session.llm.projection.test.ts)   */
/* ------------------------------------------------------------------ */

const projectionLayer = SessionLive.layer.pipe(
	Layer.provideMerge(Event.layer),
	Layer.provideMerge(Database.layer(":memory:")),
);

const projectionSetup = Effect.gen(function* () {
	const sql = yield* SqlClient.SqlClient;
	yield* sql`INSERT OR IGNORE INTO project (id, name, created_at, updated_at) VALUES ('local','local',0,0)`;
	const sessions = yield* Session.Service;
	const session = yield* sessions.create({
		projectId: "local",
		slug: "probe",
		directory: AbsolutePath.make("/repo"),
		title: "T",
		tag: "test",
		sandboxInstanceId: "local" as never,
	});
	return { sessions, events: yield* Event.Service, sessionId: session.id };
});

const usageTotals = Effect.gen(function* () {
	const sql = yield* SqlClient.SqlClient;
	const rows = yield* sql`SELECT cost, tokens_input FROM session`;
	return rows[0] as { cost: number; tokensInput: number };
});

const probeMessageId = SessionMessageSchema.ID.create();

const probeAssistant = (overrides: Partial<Message.AssistantMessage> = {}): Message.AssistantMessage => ({
	messageId: probeMessageId,
	role: "assistant",
	protocol: "anthropic",
	provider: { id: "anthropic", name: "Anthropic", source: "custom", env: [] },
	model: "claude-test",
	usage: {
		input: 11,
		output: 22,
		cacheRead: 3,
		cacheWrite: 4,
		totalTokens: 33,
		cost: { input: 0.1, output: 0.2, cacheRead: 0, cacheWrite: 0, total: 0.3 },
	},
	stopReason: "stop",
	time: { created: 10, completed: 20 },
	parts: [{ type: "text", text: "the answer" }],
	...overrides,
});

const probeCall = {
	type: "toolCall",
	callID: "call_1",
	name: "bash",
	arguments: { command: "pwd" },
	status: "pending",
	time: { start: 1, end: 2 },
} as const satisfies Message.ToolCallPendingPart;

const startDraft = Effect.fnUntraced(function* (sessionId: SessionSchema.ID) {
	const events = yield* Event.Service;
	yield* events.publish(EventList.LLMStarted, {
		sessionId,
		messageId: probeMessageId,
		timestamp: DateTime.makeUnsafe(0),
		message: probeAssistant({
			parts: [],
			stopReason: "aborted",
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
		}),
	});
	yield* events.publish(EventList.LLMToolCallFinalized, {
		sessionId,
		messageId: probeMessageId,
		timestamp: DateTime.makeUnsafe(0),
		partIndex: 0,
		callId: probeCall.callID,
		toolName: probeCall.name,
		part: probeCall,
	});
});

describe("probe A — a second terminal on a toolUse draft (still unfinalized by state)", () => {
	const { effect: it } = testEffect(projectionLayer);

	it("is accepted and charges usage twice", () =>
		Effect.gen(function* () {
			const { sessions, events, sessionId } = yield* projectionSetup;
			yield* startDraft(sessionId);
			const terminal = {
				sessionId,
				messageId: probeMessageId,
				timestamp: DateTime.makeUnsafe(0),
				reason: "toolUse",
				message: probeAssistant({ stopReason: "toolUse", parts: [probeCall] }),
			} as const;
			yield* events.publish(EventList.LLMEnded, terminal);
			expect((yield* sessions.path(sessionId))[0]!.entry.state).toBe("draft");
			expect((yield* usageTotals).tokensInput).toBe(11);

			// The guard that rejected a second `stop` terminal (state !== "draft")
			// does not fire here, because a toolUse terminal LEAVES the entry draft.
			const second = yield* events
				.publish(EventList.LLMEnded, { ...terminal, timestamp: DateTime.makeUnsafe(1) })
				.pipe(Effect.exit);

			expect(Exit.isSuccess(second)).toBe(true);
			expect((yield* usageTotals).tokensInput).toBe(22); // charged twice
			expect((yield* sessions.path(sessionId))[0]!.entry.state).toBe("draft");
		}));
});

describe("probe B — ToolFailed naming a call the entry does not have", () => {
	const { effect: it } = testEffect(projectionLayer);

	it("is silently absorbed rather than dying", () =>
		Effect.gen(function* () {
			const { sessions, events, sessionId } = yield* projectionSetup;
			yield* startDraft(sessionId);

			const exit = yield* events
				.publish(EventList.ToolFailed, {
					sessionId,
					messageId: probeMessageId,
					timestamp: DateTime.makeUnsafe(0),
					partIndex: 0,
					callId: "call_that_does_not_exist",
					toolName: "bash",
					status: "skipped",
					error: "probe",
				})
				.pipe(Effect.exit);

			// The pre-change projector died here ("ToolFailed names call ... which
			// entry ... does not have"). The settled part below shows nothing moved.
			expect(Exit.isSuccess(exit)).toBe(true);
			const [entry] = yield* sessions.path(sessionId);
			expect(Option.getOrNull(entry!.parts[0]!.status)).toBe("pending");
		}));
});

describe("probe C — TurnFailed settles calls that never started", () => {
	const { effect: it } = testEffect(projectionLayer);

	it("marks a never-started pending call 'aborted', not 'skipped'", () =>
		Effect.gen(function* () {
			const { sessions, events, sessionId } = yield* projectionSetup;
			yield* startDraft(sessionId);
			// The terminal landed (entry stays draft); the batch never ran — the
			// interrupt-between-terminal-and-batch case the spec calls D-12's fifth
			// path, closed by the TurnFailed projector rather than the sweep.
			yield* events.publish(EventList.LLMEnded, {
				sessionId,
				messageId: probeMessageId,
				timestamp: DateTime.makeUnsafe(0),
				reason: "toolUse",
				message: probeAssistant({ stopReason: "toolUse", parts: [probeCall] }),
			});

			yield* events.publish(EventList.TurnFailed, {
				sessionId,
				timestamp: DateTime.makeUnsafe(1),
				turn: 1,
				reason: "interrupted",
			});

			const [entry] = yield* sessions.path(sessionId);
			const settled = JSON.parse(entry!.parts[0]!.data) as Message.ToolCallTerminalPart;
			// §11.3: `pending` proves the handler never started -> `skipped`.
			// The TurnFailed projector settles everything it finds as `aborted`.
			expect(settled.status).toBe("aborted");
			expect(entry!.entry.state).toBe("committed");
		}));
});

/* ------------------------------------------------------------------ */
/* Loop-level harness (mirrors the continuation-ceiling test)          */
/* ------------------------------------------------------------------ */

const fake = FakeSandboxDriver.make(SandboxDriver.Name.make("review-probe-fake"));

const probeLoopAssistant = (
	input: LLM.Input,
	index: number,
	overrides: Partial<Message.AssistantMessage> = {},
): Message.AssistantMessage =>
	Message.createAssistantMessage({
		messageId: `assistant_${index}`,
		role: "assistant",
		protocol: "openai",
		provider: { id: input.provider, name: input.provider, source: "custom", env: [] },
		model: input.model,
		usage: {
			input: 1,
			output: 1,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 2,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		time: { created: index, completed: index },
		parts: [{ type: "text", text: `response ${index}` }],
		...overrides,
	});

const gateRelease = Deferred.makeUnsafe<void>();
let gateArmed = true;

const gate = Tool.register(
	Tool.implement(
		Tool.define({
			name: "gate",
			description: "Blocks the first call until the test releases it.",
			promptSnippet: "A test tool.",
			parameters: Schema.Struct({ label: Schema.String }),
			success: Schema.String,
		}),
		(params) =>
			Effect.gen(function* () {
				if (gateArmed) {
					gateArmed = false;
					const progress = yield* ToolProgress;
					yield* progress.report({ content: [{ type: "text", text: `gate holding ${params.label}` }] });
					yield* Deferred.await(gateRelease);
				}
				return params.label;
			}),
	),
);

const insatiableGate = (contexts: Message.Context[]): LLM.Open => {
	let responseIndex = 0;
	return (input) =>
		Effect.sync(() => {
			responseIndex += 1;
			contexts.push(input.context);
			const part = {
				type: "toolCall",
				callID: `call_${responseIndex}`,
				name: "gate",
				arguments: { label: `gate_${responseIndex}` },
				status: "pending",
				time: { start: responseIndex, end: responseIndex },
			} as const satisfies Message.ToolCallPendingPart;
			const message = probeLoopAssistant(input, responseIndex, { stopReason: "toolUse", parts: [part] });
			const events = createAssistantMessageEventStream();
			events.push({ type: "start", partial: message });
			events.push({ type: "toolcall.final", partIndex: 0, toolCall: part, partial: message });
			events.push({ type: "done", reason: "toolUse", message });
			return events;
		});
};

const loopSeed = Effect.fnUntraced(function* (slug: string) {
	const sql = yield* SqlClient.SqlClient;
	const sessions = yield* Session.Service;
	const controller = yield* SandboxController.Controller;
	yield* sql`INSERT OR IGNORE INTO project (id, name, created_at, updated_at) VALUES ('p', 'p', 0, 0)`;
	const instance = yield* controller.create({
		driver: fake.driver,
		config: { defaultCwd: SandboxDriver.AbsolutePath.make("/") },
	});
	yield* Effect.flatMap(SandboxIO.FileSystem, (fs) => fs.mkdir("/repo", { recursive: true })).pipe(
		Effect.provide(Layer.unwrap(Effect.succeed(controller.mount(instance.id)))),
		Effect.scoped,
	);
	const session = yield* sessions.create({
		projectId: "p",
		slug: `${slug}-${crypto.randomUUID()}`,
		directory: AbsolutePath.make("/repo"),
		title: "probe",
		sandboxInstanceId: instance.id,
	});
	return session.id;
});

const probeAdmit = Effect.fnUntraced(function* (input: {
	readonly id: string;
	readonly sessionId: SessionSchema.ID;
	readonly delivery: "steer" | "followUp";
}) {
	const inputs = yield* SessionInput.make;
	return yield* inputs.admit({
		id: SessionMessageSchema.ID.make(input.id),
		sessionId: input.sessionId,
		prompt: { text: input.id },
		delivery: input.delivery,
	});
});

describe("probe D — a steer admitted mid-chain and the continuation counter", () => {
	const contexts: Message.Context[] = [];
	const database = Database.layer(":memory:");
	const request = LLM.make(insatiableGate(contexts));
	const sandbox = SandboxController.layer().pipe(Layer.provide(SandboxDriver.layer(fake.driver)));
	const layer = Control.layer.pipe(
		Layer.provideMerge(
			RunnerExecute.layer.pipe(
				Layer.provide(
					Loop.layer({ request }).pipe(Layer.provide(State.layer({ maxContinuations: 3, tools: [gate] }))),
				),
			),
		),
		Layer.provideMerge(Context.layer),
		Layer.provideMerge(SessionProjector.layer),
		Layer.provideMerge(Session.layer),
		Layer.provideMerge(Event.layer),
		Layer.provideMerge(sandbox),
		Layer.provideMerge(database),
	);

	testEffect(layer).live(
		"does not reset the counter on user input delivered mid-chain",
		Effect.gen(function* () {
			contexts.length = 0;
			gateArmed = true;
			const sql = yield* SqlClient.SqlClient;
			const execution = yield* RunnerExecution.Service;
			const events = yield* Event.Service;
			const sessions = yield* Session.Service;
			const sessionId = yield* loopSeed("steer-reset");
			yield* probeAdmit({ id: "msg_start", sessionId, delivery: "steer" });

			const holding = yield* Deferred.make<void>();
			yield* events.listen((event) =>
				event.type === "session.tool.execution.updated"
					? Deferred.succeed(holding, undefined).pipe(Effect.asVoid)
					: Effect.void,
			);
			const waiting = yield* execution.resume(sessionId).pipe(Effect.forkChild);
			yield* Deferred.await(holding);

			// Turn 1's batch is mid-flight: the steer lands inside the exchange and
			// is promoted at the next turn boundary.
			yield* probeAdmit({ id: "msg_midchain", sessionId, delivery: "steer" });
			Deferred.doneUnsafe(gateRelease, Effect.void);
			yield* Fiber.await(waiting);

			// The steer was delivered — turn 2's request carries it.
			const sawSteer = contexts.some((context) =>
				context.messages.some(
					(message) => message.role === "user" && JSON.stringify(message.parts).includes("msg_midchain"),
				),
			);
			expect(sawSteer).toBe(true);

			// Spec: "user input resets it". Implementation: 3 assistants and a halt —
			// exactly what no steer at all would have produced.
			const path = yield* sessions.path(sessionId);
			expect(path.filter((item) => item.entry.type === "assistant")).toHaveLength(3);
			const durable = yield* sql`SELECT type FROM event WHERE aggregate_id = ${sessionId} ORDER BY seq`;
			expect(
				(durable as ReadonlyArray<{ type: string }>).filter((row) => row.type === "session.exchange.halted.1"),
			).toHaveLength(1);
			expect(path.at(-1)!.entry.type).toBe("synthetic");
		}),
		{ timeout: 15_000 },
	);
});
