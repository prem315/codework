import { createAssistantMessageEventStream, Message } from "@codeworksh/aikit";
import { Cause, DateTime, Deferred, Effect, Exit, Fiber, Layer, Option, Schema } from "effect";
import { SqlClient, SqlSchema } from "effect/unstable/sql";
import { describe, expect } from "vite-plus/test";
import { Context } from "../src/context/context.ts";
import { Control } from "../src/control.ts";
import { Database } from "../src/db/db.ts";
import { SessionInputRow } from "../src/db/schema.sql.ts";
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
import { SessionMessageSchema } from "../src/session/message/schema.ts";
import { SessionProjector } from "../src/session/projector.ts";
import { SessionSchema } from "../src/session/schema.ts";
import { ToolProgress } from "../src/tools/progress.ts";
import * as Tool from "../src/tools/tool.ts";
import { Session } from "../src/session/session.ts";
import { State } from "../src/state/state.ts";
import { testEffect } from "./utils/effect.ts";

const assistant = (
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

const immediateOpen = (contexts: Message.Context[] = []): LLM.Open => {
	let responseIndex = 0;
	return (input) =>
		Effect.sync(() => {
			responseIndex += 1;
			contexts.push(input.context);
			const message = assistant(input, responseIndex);
			const events = createAssistantMessageEventStream();
			events.push({ type: "start", partial: message });
			events.push({ type: "text.start", partIndex: 0, partial: message });
			events.push({ type: "text.delta", partIndex: 0, delta: `response ${responseIndex}`, partial: message });
			events.push({ type: "text.end", partIndex: 0, content: `response ${responseIndex}`, partial: message });
			events.push({ type: "done", reason: "stop", message });
			return events;
		});
};

/*
 * A virtual namespace, not the host: `RunnerExecute` now mounts the session
 * sandbox and proves its directory exists, so these tests need a filesystem they
 * can create that directory in. Mounting the host instead would make every case
 * walk the real tree through Project, which is both slow and not what is under
 * test here.
 */
const fake = FakeSandboxDriver.make(SandboxDriver.Name.make("runner-loop-fake"));

const runtime = (options: { readonly open?: LLM.Open; readonly contexts?: Message.Context[] } = {}) => {
	const database = Database.layer(":memory:");
	const request = LLM.make(options.open ?? immediateOpen(options.contexts));
	const sandbox = SandboxController.layer().pipe(Layer.provide(SandboxDriver.layer(fake.driver)));
	return Control.layer.pipe(
		Layer.provideMerge(
			RunnerExecute.layer.pipe(Layer.provide(Loop.layer({ request }).pipe(Layer.provide(State.layer())))),
		),
		Layer.provideMerge(Context.layer),
		Layer.provideMerge(SessionProjector.layer),
		Layer.provideMerge(Session.layer),
		Layer.provideMerge(Event.layer),
		Layer.provideMerge(sandbox),
		Layer.provideMerge(database),
	);
};

const inputsFor = (sql: SqlClient.SqlClient) =>
	SqlSchema.findAll({
		Request: Schema.String,
		Result: SessionInputRow,
		execute: (sessionId) => sql`
			SELECT * FROM session_input WHERE session_id = ${sessionId} ORDER BY admitted_seq
		`,
	});

const logFor = (sql: SqlClient.SqlClient) =>
	SqlSchema.findAll({
		Request: Schema.String,
		Result: Schema.Struct({ type: Schema.String }),
		execute: (sessionId) => sql`SELECT type FROM event WHERE aggregate_id = ${sessionId} ORDER BY seq`,
	});

const delivered = Effect.fnUntraced(function* (sessionId: string) {
	const sql = yield* SqlClient.SqlClient;
	const rows = yield* inputsFor(sql)(sessionId);
	return rows.map((row) => ({ id: row.id, promotedSeq: Option.getOrNull(row.promotedSeq) }));
});

const seedSession = Effect.fnUntraced(function* (slug = "runner-loop") {
	const sql = yield* SqlClient.SqlClient;
	const sessions = yield* Session.Service;
	const controller = yield* SandboxController.Controller;
	yield* sql`INSERT OR IGNORE INTO project (id, name, created_at, updated_at) VALUES ('p', 'p', 0, 0)`;
	// The mount must find this directory, so create it before the drain does.
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
		title: "runner loop",
		sandboxInstanceId: instance.id,
	});
	return session.id;
});

const admit = Effect.fnUntraced(function* (input: {
	readonly id: string;
	readonly sessionId: SessionSchema.ID;
	readonly delivery: "steer" | "followUp";
	readonly text?: string;
}) {
	const inputs = yield* SessionInput.make;
	return yield* inputs.admit({
		id: SessionMessageSchema.ID.make(input.id),
		sessionId: input.sessionId,
		prompt: { text: input.text ?? input.id },
		delivery: input.delivery,
	});
});

describe("runner loop — aikit input/output", () => {
	const contexts: Message.Context[] = [];
	const { effect: it } = testEffect(runtime({ contexts }));

	it(
		"drains steers before follow-ups and persists each terminal assistant",
		Effect.gen(function* () {
			contexts.length = 0;
			const execution = yield* RunnerExecution.Service;
			const sessions = yield* Session.Service;
			const sessionId = yield* seedSession();

			yield* admit({ id: "msg_follow", sessionId, delivery: "followUp" });
			yield* admit({ id: "msg_steer", sessionId, delivery: "steer" });
			yield* execution.resume(sessionId);

			const rows = yield* delivered(sessionId);
			const bySeq = [...rows].sort((left, right) => (left.promotedSeq ?? 0) - (right.promotedSeq ?? 0));
			expect(bySeq.map((row) => row.id)).toEqual(["msg_steer", "msg_follow"]);
			expect((yield* sessions.path(sessionId)).map((item) => item.entry.type)).toEqual([
				"user",
				"assistant",
				"user",
				"assistant",
			]);

			// Every request is rebuilt from durable history. The second request sees
			// the first terminal assistant and the newly promoted follow-up.
			expect(contexts.map((context) => context.messages.map((message) => message.role))).toEqual([
				["user"],
				["user", "assistant", "user"],
			]);
		}),
	);

	it(
		"brackets each turn, asks with State's prompt and tools, and keeps TurnStarted out of the log",
		Effect.gen(function* () {
			contexts.length = 0;
			const sql = yield* SqlClient.SqlClient;
			const execution = yield* RunnerExecution.Service;
			const events = yield* Event.Service;
			const sessionId = yield* seedSession();

			const starts: number[] = [];
			const ends: Array<{ turn: number; continuation: boolean }> = [];
			yield* events.listen((event) => {
				const data = event.data as { readonly turn: number; readonly continuation?: boolean };
				if (event.type === "session.turn.started") return Effect.sync(() => starts.push(data.turn));
				if (event.type === "session.turn.ended") {
					return Effect.sync(() => ends.push({ turn: data.turn, continuation: data.continuation === true }));
				}
				return Effect.void;
			});

			yield* admit({ id: "msg_bracket", sessionId, delivery: "steer" });
			yield* execution.resume(sessionId);

			// One turn, opened once and closed once. Nothing asked for a second.
			expect(starts).toEqual([1]);
			expect(ends).toEqual([{ turn: 1, continuation: false }]);

			// The whole durable shape of one turn, in order: the prompt, its
			// promotion, the assistant allocation, one completed block, the terminal
			// that promotes the entry, and the turn boundary that closes it.
			// TurnStarted precedes every fact it could summarize, so it never reaches
			// the log; neither do the starts and deltas between block boundaries.
			const durable = yield* logFor(sql)(sessionId);
			expect(durable.map((row) => row.type)).toEqual([
				"session.next.prompt.admitted.1",
				"session.next.prompt.promoted.1",
				"session.llm.started.1",
				"session.llm.text.end.1",
				"session.llm.ended.1",
				"session.turn.ended.1",
			]);

			// The request is State's, not the loop's: its prompt names the mounted
			// working directory and its tools are the resolved registry's wire view.
			const request = contexts[0];
			expect(request?.systemPrompt).toContain("Current working directory: /repo");
			expect(request?.tools?.map((tool) => tool.name)).toEqual(["bash"]);
		}),
	);

	it(
		"does not promote an input twice; an explicit resume still performs one forced request",
		Effect.gen(function* () {
			const execution = yield* RunnerExecution.Service;
			const sessions = yield* Session.Service;
			const sessionId = yield* seedSession();
			yield* admit({ id: "msg_once", sessionId, delivery: "steer" });

			yield* execution.resume(sessionId);
			yield* execution.resume(sessionId);

			expect(yield* delivered(sessionId)).toEqual([{ id: "msg_once", promotedSeq: 1 }]);
			expect((yield* sessions.path(sessionId)).map((item) => item.entry.type)).toEqual([
				"user",
				"assistant",
				"assistant",
			]);
		}),
	);

	it(
		"keeps promotion and output scoped per session",
		Effect.gen(function* () {
			const execution = yield* RunnerExecution.Service;
			const sessions = yield* Session.Service;
			const first = yield* seedSession("first");
			const second = yield* seedSession("second");
			yield* admit({ id: "msg_first", sessionId: first, delivery: "steer" });
			yield* admit({ id: "msg_second", sessionId: second, delivery: "steer" });

			yield* execution.resume(first);
			yield* execution.resume(second);

			expect(yield* delivered(first)).toEqual([{ id: "msg_first", promotedSeq: 1 }]);
			expect(yield* delivered(second)).toEqual([{ id: "msg_second", promotedSeq: 1 }]);
			expect((yield* sessions.path(first)).map((item) => item.entry.type)).toEqual(["user", "assistant"]);
			expect((yield* sessions.path(second)).map((item) => item.entry.type)).toEqual(["user", "assistant"]);
		}),
	);

	it(
		"promotes all current steers as one request, then follow-ups one per request",
		Effect.gen(function* () {
			const execution = yield* RunnerExecution.Service;
			const sessions = yield* Session.Service;
			const sessionId = yield* seedSession();
			yield* admit({ id: "msg_s1", sessionId, delivery: "steer" });
			yield* admit({ id: "msg_f1", sessionId, delivery: "followUp" });
			yield* admit({ id: "msg_s2", sessionId, delivery: "steer" });
			yield* admit({ id: "msg_f2", sessionId, delivery: "followUp" });
			yield* admit({ id: "msg_s3", sessionId, delivery: "steer" });

			yield* execution.resume(sessionId);

			const path = yield* sessions.path(sessionId);
			expect(path.map((item) => item.entry.type)).toEqual([
				"user",
				"user",
				"user",
				"assistant",
				"user",
				"assistant",
				"user",
				"assistant",
			]);
			expect(path.filter((item) => item.entry.type === "user").map((item) => item.entry.id)).toEqual([
				"msg_s1",
				"msg_s2",
				"msg_s3",
				"msg_f1",
				"msg_f2",
			]);
		}),
	);

	it(
		"fails for a session that does not exist",
		Effect.gen(function* () {
			const execution = yield* RunnerExecution.Service;
			const exit = yield* execution.resume(SessionSchema.ID.create()).pipe(Effect.exit);
			expect(Exit.isFailure(exit)).toBe(true);
		}),
	);
});

describe("runner loop — provider interruption", () => {
	let responseIndex = 0;
	const open: LLM.Open = (input, signal) =>
		Effect.sync(() => {
			responseIndex += 1;
			const partial = assistant(input, responseIndex, { parts: [{ type: "text", text: "partial" }] });
			const events = createAssistantMessageEventStream();
			events.push({ type: "start", partial });
			events.push({ type: "text.delta", partIndex: 0, delta: "partial", partial });

			const fail = () => {
				const failed = assistant(input, responseIndex, {
					stopReason: "aborted",
					errorMessage: "Request was aborted",
					parts: [{ type: "text", text: "partial" }],
				});
				events.push({ type: "error", reason: "aborted", error: failed });
			};
			if (signal.aborted) fail();
			else signal.addEventListener("abort", fail, { once: true });
			return events;
		});
	const { live: it } = testEffect(runtime({ open }));

	it(
		"aborts aikit, durably projects its terminal failure, then propagates interruption",
		Effect.gen(function* () {
			const execution = yield* RunnerExecution.Service;
			const events = yield* Event.Service;
			const sessions = yield* Session.Service;
			const sessionId = yield* seedSession();
			yield* admit({ id: "msg_interrupted", sessionId, delivery: "steer" });

			const streaming = yield* Deferred.make<void>();
			yield* events.listen((event) =>
				event.type === "session.llm.text.delta"
					? Deferred.succeed(streaming, undefined).pipe(Effect.asVoid)
					: Effect.void,
			);
			const waiting = yield* execution.resume(sessionId).pipe(Effect.forkChild);
			yield* Deferred.await(streaming);

			yield* execution.interrupt(sessionId);
			const exit = yield* Fiber.await(waiting);
			expect(Exit.isFailure(exit) && Cause.hasInterrupts(exit.cause)).toBe(true);

			const path = yield* sessions.path(sessionId);
			expect(path.map((item) => item.entry.type)).toEqual(["user", "assistant"]);
			expect(JSON.parse(path[1]!.entry.data)).toMatchObject({ stopReason: "aborted" });
			expect(JSON.parse(path[1]!.parts[0]!.data)).toEqual({ type: "text", text: "partial" });
			expect(Array.from(yield* execution.active)).toEqual([]);
		}),
		{ timeout: 10_000 },
	);
});

describe("runner loop — provider failure", () => {
	const open: LLM.Open = (input) =>
		Effect.sync(() => {
			const failed = assistant(input, 1, {
				stopReason: "error",
				errorMessage: "provider failed",
				parts: [{ type: "text", text: "partial" }],
			});
			const events = createAssistantMessageEventStream();
			events.push({ type: "start", partial: failed });
			events.push({ type: "error", reason: "error", error: failed });
			return events;
		});
	const { effect: it } = testEffect(runtime({ open }));

	it(
		"commits the failed assistant before failing the turn",
		Effect.gen(function* () {
			const execution = yield* RunnerExecution.Service;
			const sessions = yield* Session.Service;
			const sessionId = yield* seedSession();
			yield* admit({ id: "msg_failed", sessionId, delivery: "steer" });

			const exit = yield* execution.resume(sessionId).pipe(Effect.exit);
			expect(Exit.isFailure(exit)).toBe(true);

			const path = yield* sessions.path(sessionId);
			expect(path.map((item) => item.entry.type)).toEqual(["user", "assistant"]);
			expect(JSON.parse(path[1]!.entry.data)).toMatchObject({
				stopReason: "error",
				errorMessage: "provider failed",
			});
		}),
	);
});

describe("runner loop — tool execution", () => {
	const toolCall = {
		type: "toolCall",
		callID: "call_1",
		name: "bash",
		arguments: { command: "pwd" },
		status: "pending",
		time: { start: 1, end: 2 },
	} as const satisfies Message.ToolCallPendingPart;

	/**
	 * Ask for one tool on the first response, then answer plainly on the second.
	 * The second response is what proves the continuation happened without any
	 * new prompt.
	 */
	const open = (contexts: Message.Context[]): LLM.Open => {
		let responseIndex = 0;
		return (input) =>
			Effect.sync(() => {
				responseIndex += 1;
				contexts.push(input.context);
				const events = createAssistantMessageEventStream();
				if (responseIndex === 1) {
					const message = assistant(input, 1, { stopReason: "toolUse", parts: [toolCall] });
					events.push({ type: "start", partial: message });
					events.push({ type: "toolcall.start", partIndex: 0, partial: message });
					events.push({ type: "toolcall.final", partIndex: 0, toolCall, partial: message });
					events.push({ type: "done", reason: "toolUse", message });
					return events;
				}
				const message = assistant(input, 2, { parts: [{ type: "text", text: "it is /repo" }] });
				events.push({ type: "start", partial: message });
				events.push({ type: "text.end", partIndex: 0, content: "it is /repo", partial: message });
				events.push({ type: "done", reason: "stop", message });
				return events;
			});
	};

	const contexts: Message.Context[] = [];
	const { effect: it } = testEffect(runtime({ open: open(contexts) }));

	it(
		"runs the call against the mounted sandbox, settles it durably, and continues the turn",
		Effect.gen(function* () {
			contexts.length = 0;
			const sql = yield* SqlClient.SqlClient;
			const execution = yield* RunnerExecution.Service;
			const sessions = yield* Session.Service;
			const sessionId = yield* seedSession();
			yield* admit({ id: "msg_tool", sessionId, delivery: "steer" });

			yield* execution.resume(sessionId);

			// One prompt, two assistant responses: the second turn was demanded by
			// the tool results, not by any new input.
			const path = yield* sessions.path(sessionId);
			expect(path.map((item) => item.entry.type)).toEqual(["user", "assistant", "assistant"]);

			// The call really executed, in the session's own namespace — `pwd` can
			// only answer /repo from inside the mount.
			const settled = JSON.parse(path[1]!.parts[0]!.data) as Message.ToolCallCompletedPart;
			expect(settled.status).toBe("completed");
			expect(Option.getOrNull(path[1]!.parts[0]!.status)).toBe("completed");
			expect(JSON.stringify(settled.result.content)).toContain("/repo");

			// pending -> running -> terminal, each its own durable fact, bracketed by
			// the turn that scheduled them.
			const durable = (yield* logFor(sql)(sessionId)).map((row) => row.type);
			expect(durable).toEqual([
				"session.next.prompt.admitted.1",
				"session.next.prompt.promoted.1",
				"session.llm.started.1",
				"session.llm.toolcall.finalized.1",
				"session.llm.ended.1",
				"session.tool.execution.started.1",
				"session.tool.execution.ended.1",
				"session.turn.ended.1",
				"session.llm.started.1",
				"session.llm.text.end.1",
				"session.llm.ended.1",
				"session.turn.ended.1",
			]);

			// The second request carries the settled call back to the model, which is
			// the whole reason the turn continued.
			const second = contexts[1];
			expect(second?.messages.map((message) => message.role)).toEqual(["user", "assistant"]);
			const parts = second?.messages[1]?.parts ?? [];
			expect(parts.map((part) => part.type)).toEqual(["toolCall"]);
		}),
	);
});

describe("runner loop — recovery sweep", () => {
	const contexts: Message.Context[] = [];
	const { effect: it } = testEffect(runtime({ contexts }));

	/**
	 * What a hard kill leaves behind: the assistant entry allocated at
	 * `LLMStarted`, a call finalized into it, and no terminal. A graceful
	 * interrupt never looks like this — aikit's terminal carries the whole
	 * message, so that path closes its own turn in place.
	 */
	const killedTurn = Effect.fnUntraced(function* (sessionId: SessionSchema.ID, callStatus: "pending" | "running") {
		const events = yield* Event.Service;
		const messageId = SessionMessageSchema.ID.make("assistant_killed");
		const envelope = Message.createAssistantMessage({
			messageId,
			role: "assistant",
			protocol: "openai",
			provider: { id: "openai", name: "openai", source: "custom", env: [] },
			model: "gpt-5.5",
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "aborted",
			time: { created: 1, completed: 1 },
			parts: [],
		});
		const call = {
			type: "toolCall",
			callID: "call_killed",
			name: "bash",
			arguments: { command: "pwd" },
			status: "pending",
			time: { start: 1, end: 2 },
		} as const satisfies Message.ToolCallPendingPart;

		yield* events.publish(EventList.LLMStarted, {
			sessionId,
			timestamp: DateTime.makeUnsafe(0),
			messageId,
			message: envelope,
		});
		yield* events.publish(EventList.LLMToolCallFinalized, {
			sessionId,
			timestamp: DateTime.makeUnsafe(0),
			messageId,
			partIndex: 0,
			callId: call.callID,
			toolName: call.name,
			part: call,
		});
		// A kill after `ToolExecutionStarted` committed leaves the call `running`.
		if (callStatus === "running") {
			yield* events.publish(EventList.ToolExecutionStarted, {
				sessionId,
				timestamp: DateTime.makeUnsafe(0),
				messageId,
				partIndex: 0,
				callId: call.callID,
				toolName: call.name,
				part: { ...call, status: "running" },
			});
		}
		return messageId;
	});

	it(
		"settles a call that never started as skipped, before asking anything new",
		Effect.gen(function* () {
			contexts.length = 0;
			const sql = yield* SqlClient.SqlClient;
			const execution = yield* RunnerExecution.Service;
			const sessions = yield* Session.Service;
			const sessionId = yield* seedSession();
			yield* killedTurn(sessionId, "pending");
			yield* admit({ id: "msg_after_kill", sessionId, delivery: "steer" });

			yield* execution.resume(sessionId);

			const path = yield* sessions.path(sessionId);
			const settled = JSON.parse(path[0]!.parts[0]!.data) as Message.ToolCallSkippedPart;
			// `pending` proves it never ran, so the honest settlement is `skipped` —
			// never a re-execution, because a mutating tool would then run twice.
			expect(settled.status).toBe("skipped");
			expect(Option.getOrNull(path[0]!.parts[0]!.status)).toBe("skipped");
			expect(JSON.stringify(settled.result.content)).toContain("never executed");

			// The sweep runs once per drain, before both loops: the failure is
			// durable ahead of the prompt this drain came to deliver.
			const durable = (yield* logFor(sql)(sessionId)).map((row) => row.type);
			expect(durable.indexOf("session.tool.failed.1")).toBeLessThan(
				durable.indexOf("session.next.prompt.promoted.1"),
			);

			// And the model sees a well-formed conversation: the settled call replays
			// as a tool call with its result.
			expect(contexts[0]?.messages.map((message) => message.role)).toEqual(["assistant", "user"]);
		}),
	);

	it(
		"finds the killed assistant behind a non-message leaf",
		Effect.gen(function* () {
			const execution = yield* RunnerExecution.Service;
			const events = yield* Event.Service;
			const sessions = yield* Session.Service;
			const sessionId = yield* seedSession("leaf-behind-config");
			yield* killedTurn(sessionId, "running");
			// A model switch after the kill. The leaf is the tree cursor, so it moves
			// to this entry and the assistant sits behind it — asking the leaf would
			// miss exactly the case the sweep exists for.
			yield* sessions.append({
				id: "config_after_kill",
				sessionId,
				seq: (yield* events.latestSequence(sessionId)) + 1,
				type: "configChange",
				data: JSON.stringify({ thinkingLevel: "low" }),
			});
			yield* admit({ id: "msg_behind_config", sessionId, delivery: "steer" });

			yield* execution.resume(sessionId);

			const killed = yield* sessions.entry("assistant_killed");
			const settled = JSON.parse(Option.getOrThrow(killed).parts[0]!.data) as Message.ToolCallAbortedPart;
			expect(settled.status).toBe("aborted");
		}),
	);

	it(
		"settles a call that may have run as aborted",
		Effect.gen(function* () {
			const execution = yield* RunnerExecution.Service;
			const sessions = yield* Session.Service;
			const sessionId = yield* seedSession("running-kill");
			yield* killedTurn(sessionId, "running");
			yield* admit({ id: "msg_after_running_kill", sessionId, delivery: "steer" });

			yield* execution.resume(sessionId);

			const path = yield* sessions.path(sessionId);
			const settled = JSON.parse(path[0]!.parts[0]!.data) as Message.ToolCallAbortedPart;
			// `running` is a claim that a side effect may already have happened, so
			// the settlement says interrupted rather than skipped.
			expect(settled.status).toBe("aborted");
			expect(JSON.stringify(settled.result.content)).toContain("interrupted");
		}),
	);
});

describe("runner loop — the request State builds", () => {
	/** Capture the bag the loop hands aikit, then answer trivially. */
	const capturing = (bags: LLM.RequestOptions[], models: Array<{ provider: string; model: string }>): LLM.Open => {
		let responseIndex = 0;
		return (input) =>
			Effect.sync(() => {
				responseIndex += 1;
				bags.push(input.options);
				models.push({ provider: input.provider, model: input.model });
				const message = assistant(input, responseIndex);
				const events = createAssistantMessageEventStream();
				events.push({ type: "start", partial: message });
				events.push({ type: "text.end", partIndex: 0, content: `response ${responseIndex}`, partial: message });
				events.push({ type: "done", reason: "stop", message });
				return events;
			});
	};

	const withState = (agent: State.Options) => {
		const bags: LLM.RequestOptions[] = [];
		const models: Array<{ provider: string; model: string }> = [];
		const database = Database.layer(":memory:");
		const request = LLM.make(capturing(bags, models));
		const sandbox = SandboxController.layer().pipe(Layer.provide(SandboxDriver.layer(fake.driver)));
		const layer = Control.layer.pipe(
			Layer.provideMerge(
				RunnerExecute.layer.pipe(Layer.provide(Loop.layer({ request }).pipe(Layer.provide(State.layer(agent))))),
			),
			Layer.provideMerge(Context.layer),
			Layer.provideMerge(SessionProjector.layer),
			Layer.provideMerge(Session.layer),
			Layer.provideMerge(Event.layer),
			Layer.provideMerge(sandbox),
			Layer.provideMerge(database),
		);
		return { bags, models, layer };
	};

	const drive = Effect.fnUntraced(function* (id: string) {
		const execution = yield* RunnerExecution.Service;
		const sessionId = yield* seedSession(id);
		yield* admit({ id: `msg_${id}`, sessionId, delivery: "steer" });
		yield* execution.resume(sessionId);
		return sessionId;
	});

	const passthrough = withState({
		thinkingLevel: "high",
		maxTokens: 4_096,
		providerOptions: { openai: { reasoningSummary: "auto" } },
		factoryOptions: { organization: "acme" },
	});
	testEffect(passthrough.layer).effect(
		"carries the caller's options to aikit unaltered, and adds only what it owns",
		Effect.gen(function* () {
			yield* drive("passthrough");

			const bag = passthrough.bags[0];
			expect(bag?.maxTokens).toBe(4_096);
			expect(bag?.providerOptions).toEqual({ openai: { reasoningSummary: "auto" } });
			expect(bag?.factoryOptions).toEqual({ organization: "acme" });
			// The thinking level is spelled `reasoning` on the wire — one value, not
			// two spellings free to drift.
			expect(bag?.reasoning).toBe("high");
			// The two fields State withholds, supplied by the loop that owns them.
			expect(bag?.sessionId).toEqual(expect.any(String));
			// `signal` is bound to the scope that aborts the transport; a caller value
			// would silently break cancellation, so it is never in the bag State built.
			expect(bag && "signal" in bag).toBe(false);
		}),
	);

	const thinkingOff = withState({ thinkingLevel: "off" });
	testEffect(thinkingOff.layer).effect(
		"omits reasoning entirely when thinking is off",
		Effect.gen(function* () {
			yield* drive("thinking-off");
			// aikit types `reasoning` as ActiveThinkingLevel, which has no `off`
			// member, so "no thinking" is expressed by absence rather than a value.
			expect(thinkingOff.bags[0] && "reasoning" in thinkingOff.bags[0]).toBe(false);
		}),
	);

	const defaults = withState({});
	testEffect(defaults.layer).effect(
		"leaves maxTokens unset so aikit derives it from the model",
		Effect.gen(function* () {
			yield* drive("defaults");
			// aikit's `applyDefaultMaxTokens` reads the model's own maxTokens and
			// contextWindow. A value here would override a model-aware number with a
			// fixed guess — which is what the deleted `runtimeOptions` used to do.
			expect(defaults.bags[0]?.maxTokens).toBeUndefined();
			expect(defaults.models[0]).toEqual({ provider: "openai", model: "gpt-5.5" });
		}),
	);

	const overridden = withState({ provider: "openai", model: "gpt-5.5" });
	testEffect(overridden.layer).effect(
		"lets a durable model change in the session beat the State default",
		Effect.gen(function* () {
			const execution = yield* RunnerExecution.Service;
			const events = yield* Event.Service;
			const sessions = yield* Session.Service;
			const sessionId = yield* seedSession("model-override");
			// A durable model switch, which also leaves a non-message leaf — the
			// shape the recovery sweep has to read as "nothing in flight".
			yield* sessions.append({
				id: "config_override",
				sessionId,
				seq: (yield* events.latestSequence(sessionId)) + 1,
				type: "configChange",
				data: JSON.stringify({ model: { providerId: "anthropic", modelId: "claude-sonnet-5" } }),
			});
			yield* admit({ id: "msg_override", sessionId, delivery: "steer" });

			yield* execution.resume(sessionId);

			// State holds the agent's configured default; the session may have moved
			// off it mid-conversation, and that change is durable in its history.
			expect(overridden.models[0]).toEqual({ provider: "anthropic", model: "claude-sonnet-5" });
		}),
	);
});

describe("runner loop — scheduling and tool failures", () => {
	/** Records entry and exit so a test can see whether two calls overlapped. */
	const trace: string[] = [];
	const SlowParams = Schema.Struct({ label: Schema.String });
	const slow = Tool.register(
		Tool.implement(
			Tool.define({
				name: "slow",
				description: "Record entry and exit around a yield point.",
				promptSnippet: "A test tool.",
				parameters: SlowParams,
				success: Schema.String,
			}),
			(params) =>
				Effect.gen(function* () {
					trace.push(`enter:${params.label}`);
					yield* Effect.sleep("20 millis");
					trace.push(`exit:${params.label}`);
					return params.label;
				}),
		),
	);

	const call = (label: string, index: number) =>
		({
			type: "toolCall",
			callID: `call_${label}`,
			name: "slow",
			arguments: { label },
			status: "pending",
			time: { start: index, end: index },
		}) as const satisfies Message.ToolCallPendingPart;

	/** Ask for two calls at once, then answer. */
	const twoCalls = (): LLM.Open => {
		let responseIndex = 0;
		return (input) =>
			Effect.sync(() => {
				responseIndex += 1;
				const events = createAssistantMessageEventStream();
				if (responseIndex === 1) {
					const parts = [call("a", 1), call("b", 2)];
					const message = assistant(input, 1, { stopReason: "toolUse", parts: [...parts] });
					events.push({ type: "start", partial: message });
					for (const [partIndex, part] of parts.entries()) {
						events.push({ type: "toolcall.final", partIndex, toolCall: part, partial: message });
					}
					events.push({ type: "done", reason: "toolUse", message });
					return events;
				}
				const message = assistant(input, 2, { parts: [{ type: "text", text: "both done" }] });
				events.push({ type: "start", partial: message });
				events.push({ type: "text.end", partIndex: 0, content: "both done", partial: message });
				events.push({ type: "done", reason: "stop", message });
				return events;
			});
	};

	const scheduled = (mode: "sequential" | "parallel") => {
		const database = Database.layer(":memory:");
		const request = LLM.make(twoCalls());
		const sandbox = SandboxController.layer().pipe(Layer.provide(SandboxDriver.layer(fake.driver)));
		return Control.layer.pipe(
			Layer.provideMerge(
				RunnerExecute.layer.pipe(
					Layer.provide(
						Loop.layer({ request }).pipe(Layer.provide(State.layer({ toolExecution: mode, tools: [slow] }))),
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
	};

	const runBoth = Effect.fnUntraced(function* (slug: string) {
		const execution = yield* RunnerExecution.Service;
		const sessions = yield* Session.Service;
		const sessionId = yield* seedSession(slug);
		yield* admit({ id: `msg_${slug}`, sessionId, delivery: "steer" });
		yield* execution.resume(sessionId);
		const path = yield* sessions.path(sessionId);
		return path[1]!.parts.map((part) => JSON.parse(part.data) as Message.ToolCallTerminalPart);
	});

	testEffect(scheduled("sequential")).live(
		"sequential mode settles one call at a time, in partIndex order",
		Effect.gen(function* () {
			trace.length = 0;
			const parts = yield* runBoth("sequential");

			expect(trace).toEqual(["enter:a", "exit:a", "enter:b", "exit:b"]);
			// Source order regardless: each call settles at its own slot.
			expect(parts.map((part) => part.callID)).toEqual(["call_a", "call_b"]);
			expect(parts.every((part) => part.status === "completed")).toBe(true);
		}),
	);

	testEffect(scheduled("parallel")).live(
		"parallel mode overlaps the calls and still replays them in source order",
		Effect.gen(function* () {
			trace.length = 0;
			const parts = yield* runBoth("parallel");

			// Both entered before either exited — the whole difference between the
			// modes is a concurrency argument.
			expect(trace.slice(0, 2).sort()).toEqual(["enter:a", "enter:b"]);
			// A terminal only ever writes to the slot its own call already occupies,
			// so completion order cannot disturb the array.
			expect(parts.map((part) => part.callID)).toEqual(["call_a", "call_b"]);
			expect(parts.every((part) => part.status === "completed")).toBe(true);
		}),
	);

	testEffect(scheduled("parallel")).live(
		"gives concurrent terminals distinct aggregate sequences",
		Effect.gen(function* () {
			trace.length = 0;
			const sql = yield* SqlClient.SqlClient;
			const execution = yield* RunnerExecution.Service;
			const sessionId = yield* seedSession("parallel-seq");
			yield* admit({ id: "msg_parallel_seq", sessionId, delivery: "steer" });
			yield* execution.resume(sessionId);

			// Two calls settling at once commit through one aggregate, so their
			// sequences have to be distinct and gapless however they interleave —
			// that is what lets the log be replayed in commit order at all.
			const rows = yield* SqlSchema.findAll({
				Request: Schema.String,
				Result: Schema.Struct({ seq: Schema.Int }),
				execute: (id) => sql`SELECT seq FROM event WHERE aggregate_id = ${id} ORDER BY seq`,
			})(sessionId);
			const seqs = rows.map((row) => row.seq);
			expect(new Set(seqs).size).toBe(seqs.length);
			expect(seqs).toEqual(Array.from({ length: seqs.length }, (_, index) => seqs[0]! + index));
		}),
	);

	/** One call to a tool nobody registered, then a plain answer. */
	const unknownTool = (): LLM.Open => {
		let responseIndex = 0;
		return (input) =>
			Effect.sync(() => {
				responseIndex += 1;
				const events = createAssistantMessageEventStream();
				if (responseIndex === 1) {
					const part = {
						type: "toolCall",
						callID: "call_ghost",
						name: "ghost",
						arguments: {},
						status: "pending",
						time: { start: 1, end: 1 },
					} as const satisfies Message.ToolCallPendingPart;
					const message = assistant(input, 1, { stopReason: "toolUse", parts: [part] });
					events.push({ type: "start", partial: message });
					events.push({ type: "toolcall.final", partIndex: 0, toolCall: part, partial: message });
					events.push({ type: "done", reason: "toolUse", message });
					return events;
				}
				const message = assistant(input, 2, { parts: [{ type: "text", text: "sorry" }] });
				events.push({ type: "start", partial: message });
				events.push({ type: "text.end", partIndex: 0, content: "sorry", partial: message });
				events.push({ type: "done", reason: "stop", message });
				return events;
			});
	};

	const { effect: it } = testEffect(runtime({ open: unknownTool() }));
	it(
		"turns an unknown tool into a model-visible error rather than failing the run",
		Effect.gen(function* () {
			const execution = yield* RunnerExecution.Service;
			const sessions = yield* Session.Service;
			const sessionId = yield* seedSession("unknown-tool");
			yield* admit({ id: "msg_ghost", sessionId, delivery: "steer" });

			// The drain succeeds: a tool the registry does not have is the model's
			// mistake to see and correct, not the loop's to crash on.
			yield* execution.resume(sessionId);

			const path = yield* sessions.path(sessionId);
			const settled = JSON.parse(path[1]!.parts[0]!.data) as Message.ToolCallErrorPart;
			expect(settled.status).toBe("error");
			expect(JSON.stringify(settled.result.content)).toContain("Unknown tool");
			// And the turn continued, so the model got to answer with that in hand.
			expect(path.map((item) => item.entry.type)).toEqual(["user", "assistant", "assistant"]);
		}),
	);
});

describe("runner loop — State is pinned per exchange", () => {
	const renders: string[] = [];
	const toolThenAnswer = (): LLM.Open => {
		let responseIndex = 0;
		return (input) =>
			Effect.sync(() => {
				responseIndex += 1;
				const events = createAssistantMessageEventStream();
				if (responseIndex === 1) {
					const part = {
						type: "toolCall",
						callID: "call_pin",
						name: "bash",
						arguments: { command: "pwd" },
						status: "pending",
						time: { start: 1, end: 1 },
					} as const satisfies Message.ToolCallPendingPart;
					const message = assistant(input, 1, { stopReason: "toolUse", parts: [part] });
					events.push({ type: "start", partial: message });
					events.push({ type: "toolcall.final", partIndex: 0, toolCall: part, partial: message });
					events.push({ type: "done", reason: "toolUse", message });
					return events;
				}
				const message = assistant(input, responseIndex, { parts: [{ type: "text", text: "ok" }] });
				events.push({ type: "start", partial: message });
				events.push({ type: "text.end", partIndex: 0, content: "ok", partial: message });
				events.push({ type: "done", reason: "stop", message });
				return events;
			});
	};

	const database = Database.layer(":memory:");
	const request = LLM.make(toolThenAnswer());
	const sandbox = SandboxController.layer().pipe(Layer.provide(SandboxDriver.layer(fake.driver)));
	// The override runs once per snapshot, so counting its renders counts the
	// snapshots without reaching inside the loop.
	const agent: State.Options = {
		promptSystemOverride: (input) => {
			renders.push(input.systemPrompt);
			return input.systemPrompt;
		},
	};
	const layer = Control.layer.pipe(
		Layer.provideMerge(
			RunnerExecute.layer.pipe(Layer.provide(Loop.layer({ request }).pipe(Layer.provide(State.layer(agent))))),
		),
		Layer.provideMerge(Context.layer),
		Layer.provideMerge(SessionProjector.layer),
		Layer.provideMerge(Session.layer),
		Layer.provideMerge(Event.layer),
		Layer.provideMerge(sandbox),
		Layer.provideMerge(database),
	);

	testEffect(layer).effect(
		"captures State once per exchange, and once more for a follow-up",
		Effect.gen(function* () {
			renders.length = 0;
			const execution = yield* RunnerExecution.Service;
			const sessions = yield* Session.Service;
			const sessionId = yield* seedSession("pinned");
			yield* admit({ id: "msg_pin", sessionId, delivery: "steer" });
			yield* admit({ id: "msg_pin_follow", sessionId, delivery: "followUp" });

			yield* execution.resume(sessionId);

			// Two exchanges: the steer, whose two turns share one snapshot, then the
			// follow-up, which opens a new one. Four provider requests, two captures.
			const path = yield* sessions.path(sessionId);
			expect(path.map((item) => item.entry.type)).toEqual(["user", "assistant", "assistant", "user", "assistant"]);
			expect(renders).toHaveLength(2);
		}),
	);
});

describe("runner loop — interruption during a tool batch", () => {
	const started: string[] = [];
	const BlockParams = Schema.Struct({ label: Schema.String });
	/** Reports progress, then blocks until interrupted. */
	const blocking = Tool.register(
		Tool.implement(
			Tool.define({
				name: "block",
				description: "Report progress, then block forever.",
				promptSnippet: "A test tool.",
				parameters: BlockParams,
				success: Schema.String,
			}),
			(params) =>
				Effect.gen(function* () {
					started.push(params.label);
					const progress = yield* ToolProgress;
					yield* progress.report({ content: [{ type: "text", text: `partial from ${params.label}` }] });
					yield* Effect.never;
					return params.label;
				}),
		),
	);

	const pending = (label: string, index: number) =>
		({
			type: "toolCall",
			callID: `call_${label}`,
			name: "block",
			arguments: { label },
			status: "pending",
			time: { start: index, end: index },
		}) as const satisfies Message.ToolCallPendingPart;

	const twoBlockingCalls = (): LLM.Open => (input) =>
		Effect.sync(() => {
			const parts = [pending("a", 1), pending("b", 2)];
			const message = assistant(input, 1, { stopReason: "toolUse", parts: [...parts] });
			const events = createAssistantMessageEventStream();
			events.push({ type: "start", partial: message });
			for (const [partIndex, part] of parts.entries()) {
				events.push({ type: "toolcall.final", partIndex, toolCall: part, partial: message });
			}
			events.push({ type: "done", reason: "toolUse", message });
			return events;
		});

	const database = Database.layer(":memory:");
	const request = LLM.make(twoBlockingCalls());
	const sandbox = SandboxController.layer().pipe(Layer.provide(SandboxDriver.layer(fake.driver)));
	const layer = Control.layer.pipe(
		Layer.provideMerge(
			RunnerExecute.layer.pipe(
				Layer.provide(
					Loop.layer({ request }).pipe(
						Layer.provide(State.layer({ toolExecution: "sequential", tools: [blocking] })),
					),
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
		"commits the running call as aborted with its last partial, and skips the queued one",
		Effect.gen(function* () {
			started.length = 0;
			const execution = yield* RunnerExecution.Service;
			const events = yield* Event.Service;
			const sessions = yield* Session.Service;
			const sessionId = yield* seedSession("interrupt-batch");
			yield* admit({ id: "msg_interrupt_batch", sessionId, delivery: "steer" });

			const running = yield* Deferred.make<void>();
			yield* events.listen((event) =>
				event.type === "session.tool.execution.updated"
					? Deferred.succeed(running, undefined).pipe(Effect.asVoid)
					: Effect.void,
			);
			const waiting = yield* execution.resume(sessionId).pipe(Effect.forkChild);
			yield* Deferred.await(running);

			yield* execution.interrupt(sessionId);
			const exit = yield* Fiber.await(waiting);
			expect(Exit.isFailure(exit) && Cause.hasInterrupts(exit.cause)).toBe(true);

			// Sequential mode means only the first call ever entered its handler.
			expect(started).toEqual(["a"]);

			const path = yield* sessions.path(sessionId);
			const parts = path[1]!.parts.map((part) => JSON.parse(part.data) as Message.ToolCallTerminalPart);

			// The running call is settled with what it had already produced. Effect
			// unwinds an external interrupt through the handler, so this arrives from
			// the batch's finalizer rather than from the executor's own terminal.
			expect(parts[0]?.status).toBe("aborted");
			expect(JSON.stringify(parts[0]?.result.content)).toContain("partial from a");

			// The queued call is settled without executing — `pending` proved it never
			// started, so it says skipped rather than aborted.
			expect(parts[1]?.status).toBe("skipped");
			expect(JSON.stringify(parts[1]?.result.content)).toContain("never executed");

			// Nothing is left open for the next drain to find.
			expect(yield* sessions.unsettled(sessionId)).toEqual([]);
		}),
		{ timeout: 10_000 },
	);
});

describe("runner loop — terminals other than the happy path", () => {
	const call = (index: number) =>
		({
			type: "toolCall",
			callID: `call_${index}`,
			name: "bash",
			arguments: { command: "pwd" },
			status: "pending",
			time: { start: index, end: index },
		}) as const satisfies Message.ToolCallPendingPart;

	/** One response, ended however the case asks for, carrying `parts` tool calls. */
	const endingWith = (
		reason: "stop" | "length" | "toolUse",
		outcome: "ended" | "failed",
		failure: "aborted" | "error",
		callCount: number,
	): LLM.Open => {
		let responseIndex = 0;
		return (input) =>
			Effect.sync(() => {
				responseIndex += 1;
				const events = createAssistantMessageEventStream();
				if (responseIndex > 1) {
					const answer = assistant(input, responseIndex, { parts: [{ type: "text", text: "done" }] });
					events.push({ type: "start", partial: answer });
					events.push({ type: "text.end", partIndex: 0, content: "done", partial: answer });
					events.push({ type: "done", reason: "stop", message: answer });
					return events;
				}
				const parts = Array.from({ length: callCount }, (_, index) => call(index));
				const stopReason = outcome === "ended" ? reason : failure;
				const message = assistant(input, 1, {
					stopReason,
					...(outcome === "failed" ? { errorMessage: `provider ${failure}` } : {}),
					parts: [...parts],
				});
				events.push({ type: "start", partial: message });
				for (const [partIndex, part] of parts.entries()) {
					events.push({ type: "toolcall.final", partIndex, toolCall: part, partial: message });
				}
				if (outcome === "ended") events.push({ type: "done", reason, message });
				else events.push({ type: "error", reason: failure, error: message });
				return events;
			});
	};

	const runtimeFor = (open: LLM.Open) => {
		const database = Database.layer(":memory:");
		const request = LLM.make(open);
		const sandbox = SandboxController.layer().pipe(Layer.provide(SandboxDriver.layer(fake.driver)));
		return Control.layer.pipe(
			Layer.provideMerge(
				RunnerExecute.layer.pipe(Layer.provide(Loop.layer({ request }).pipe(Layer.provide(State.layer())))),
			),
			Layer.provideMerge(Context.layer),
			Layer.provideMerge(SessionProjector.layer),
			Layer.provideMerge(Session.layer),
			Layer.provideMerge(Event.layer),
			Layer.provideMerge(sandbox),
			Layer.provideMerge(database),
		);
	};

	const settledCalls = Effect.fnUntraced(function* (slug: string, expectFailure: boolean) {
		const execution = yield* RunnerExecution.Service;
		const sessions = yield* Session.Service;
		const sessionId = yield* seedSession(slug);
		yield* admit({ id: `msg_${slug}`, sessionId, delivery: "steer" });
		const exit = yield* execution.resume(sessionId).pipe(Effect.exit);
		expect(Exit.isFailure(exit)).toBe(expectFailure);
		const path = yield* sessions.path(sessionId);
		return {
			path,
			parts: path[1]!.parts.map((part) => JSON.parse(part.data) as Message.ToolCallTerminalPart),
			unsettled: yield* sessions.unsettled(sessionId),
		};
	});

	testEffect(runtimeFor(endingWith("length", "ended", "error", 1))).effect(
		"skips the calls of a truncated response rather than running them",
		Effect.gen(function* () {
			const { parts, path, unsettled } = yield* settledCalls("length", false);
			// Truncated arguments can still parse and validate while meaning
			// something else, so none of them is safe to execute.
			expect(parts[0]?.status).toBe("skipped");
			expect(JSON.stringify(parts[0]?.result.content)).toContain("output limit");
			// `pwd` never ran, and the turn settled rather than continuing.
			expect(JSON.stringify(parts[0]?.result.content)).not.toContain("/repo");
			expect(path.map((item) => item.entry.type)).toEqual(["user", "assistant"]);
			expect(unsettled).toEqual([]);
		}),
	);

	testEffect(runtimeFor(endingWith("stop", "failed", "error", 1))).effect(
		"settles the calls of a failed response as error, at the point of failure",
		Effect.gen(function* () {
			const { parts, unsettled } = yield* settledCalls("failed", true);
			// Left open, these would tell the model nothing until it asked again.
			expect(parts[0]?.status).toBe("error");
			expect(JSON.stringify(parts[0]?.result.content)).toContain("response failed");
			expect(unsettled).toEqual([]);
		}),
	);

	testEffect(runtimeFor(endingWith("stop", "failed", "aborted", 1))).effect(
		"settles the calls of an aborted response as aborted",
		Effect.gen(function* () {
			const { parts } = yield* settledCalls("aborted", true);
			// Aborted is something we did; error is something the provider did.
			expect(parts[0]?.status).toBe("aborted");
			expect(JSON.stringify(parts[0]?.result.content)).toContain("was aborted");
		}),
	);

	testEffect(runtimeFor(endingWith("stop", "ended", "error", 1))).effect(
		"runs the calls of a stop that contradicts itself, and continues",
		Effect.gen(function* () {
			const { parts, path } = yield* settledCalls("stop-with-calls", false);
			// The provider said it was finished and asked for a tool anyway. The
			// concrete tool lifecycle wins: the call is durable and canonical.
			expect(parts[0]?.status).toBe("completed");
			expect(JSON.stringify(parts[0]?.result.content)).toContain("/repo");
			expect(path.map((item) => item.entry.type)).toEqual(["user", "assistant", "assistant"]);
		}),
	);

	testEffect(runtimeFor(endingWith("toolUse", "ended", "error", 0))).effect(
		"fails a toolUse terminal that named no tools, rather than spinning",
		Effect.gen(function* () {
			const execution = yield* RunnerExecution.Service;
			const sessionId = yield* seedSession("empty-tooluse");
			yield* admit({ id: "msg_empty_tooluse", sessionId, delivery: "steer" });

			// Nothing to execute and nothing to continue on: a broken stream, not an
			// empty turn. Continuing would loop forever.
			const exit = yield* execution.resume(sessionId).pipe(Effect.exit);
			expect(Exit.isFailure(exit)).toBe(true);
		}),
	);
});

describe("runner loop — what a killed turn leaves behind", () => {
	const contexts: Message.Context[] = [];
	const { effect: it } = testEffect(runtime({ contexts }));

	it(
		"a zero-part assistant contributes nothing to the next request",
		Effect.gen(function* () {
			contexts.length = 0;
			const execution = yield* RunnerExecution.Service;
			const events = yield* Event.Service;
			const sessions = yield* Session.Service;
			const sessionId = yield* seedSession("zero-part");

			// A kill after LLMStarted but before any block completed: the entry was
			// allocated and nothing ever filled it.
			const messageId = SessionMessageSchema.ID.make("assistant_empty");
			yield* events.publish(EventList.LLMStarted, {
				sessionId,
				timestamp: DateTime.makeUnsafe(0),
				messageId,
				message: Message.createAssistantMessage({
					messageId,
					role: "assistant",
					protocol: "openai",
					provider: { id: "openai", name: "openai", source: "custom", env: [] },
					model: "gpt-5.5",
					usage: {
						input: 0,
						output: 0,
						cacheRead: 0,
						cacheWrite: 0,
						totalTokens: 0,
						cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
					},
					stopReason: "aborted",
					time: { created: 1, completed: 1 },
					parts: [],
				}),
			});
			yield* admit({ id: "msg_after_zero", sessionId, delivery: "steer" });

			yield* execution.resume(sessionId);

			// It stays in the timeline as a truthful record that an attempt died,
			// and it is well-formed: decodable, empty, and carrying no content into
			// the next request. Whether the provider adapter drops an empty assistant
			// outright is aikit's business, not ours.
			const path = yield* sessions.path(sessionId);
			expect(path[0]!.entry.type).toBe("assistant");
			expect(path[0]!.parts).toEqual([]);
			const empty = contexts[0]?.messages.find((message) => message.role === "assistant");
			expect(empty?.parts).toEqual([]);
			expect(contexts[0]?.messages.map((message) => message.role)).toEqual(["assistant", "user"]);
		}),
	);

	it(
		"an unsettled pending call survives assembly intact, so aikit can rewrite it",
		Effect.gen(function* () {
			contexts.length = 0;
			const sessions = yield* Session.Service;
			const context = yield* Context.Service;
			const events = yield* Event.Service;
			const sessionId = yield* seedSession("unsettled-floor");
			const messageId = SessionMessageSchema.ID.make("assistant_unsettled");
			yield* events.publish(EventList.LLMStarted, {
				sessionId,
				timestamp: DateTime.makeUnsafe(0),
				messageId,
				message: Message.createAssistantMessage({
					messageId,
					role: "assistant",
					protocol: "openai",
					provider: { id: "openai", name: "openai", source: "custom", env: [] },
					model: "gpt-5.5",
					usage: {
						input: 0,
						output: 0,
						cacheRead: 0,
						cacheWrite: 0,
						totalTokens: 0,
						cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
					},
					stopReason: "aborted",
					time: { created: 1, completed: 1 },
					parts: [],
				}),
			});
			yield* events.publish(EventList.LLMToolCallFinalized, {
				sessionId,
				timestamp: DateTime.makeUnsafe(0),
				messageId,
				partIndex: 0,
				callId: "call_floor",
				toolName: "bash",
				part: {
					type: "toolCall",
					callID: "call_floor",
					name: "bash",
					arguments: {},
					status: "pending",
					time: { start: 1, end: 1 },
				},
			});

			// Deliberately no sweep and no settlement — this is the floor underneath
			// every failure path, not the happy path.
			expect(yield* sessions.unsettled(sessionId)).toHaveLength(1);
			const assembled = yield* context.assemble(sessionId);
			const parts = assembled.messages[0]?.parts ?? [];
			const call = parts.find((part) => part.type === "toolCall");
			/*
			 * Our half of the floor: the call comes back out of storage `pending` and
			 * structurally complete. aikit's `transformMessages` is what rewrites an
			 * unresolved call to a synthetic `skipped` on the way to the provider
			 * (`message/message.ts:367-372`) — that is its guarantee to keep, and it
			 * can only keep it if assembly hands over a well-formed call.
			 */
			expect(call && "status" in call ? call.status : undefined).toBe("pending");
			expect(call && "callID" in call ? call.callID : undefined).toBe("call_floor");
			expect(call && "name" in call ? call.name : undefined).toBe("bash");
		}),
	);
});

describe("runner loop — remaining Phase 1 assertions", () => {
	const call = (label: string, index: number) =>
		({
			type: "toolCall",
			callID: `call_${label}`,
			name: "bash",
			arguments: { command: "pwd" },
			status: "pending",
			time: { start: index, end: index },
		}) as const satisfies Message.ToolCallPendingPart;

	/** Ask for a tool `rounds` times, then answer. */
	const chain = (rounds: number): LLM.Open => {
		let responseIndex = 0;
		return (input) =>
			Effect.sync(() => {
				responseIndex += 1;
				const events = createAssistantMessageEventStream();
				if (responseIndex <= rounds) {
					const part = call(`r${responseIndex}`, responseIndex);
					const message = assistant(input, responseIndex, { stopReason: "toolUse", parts: [part] });
					events.push({ type: "start", partial: message });
					events.push({ type: "toolcall.final", partIndex: 0, toolCall: part, partial: message });
					events.push({ type: "done", reason: "toolUse", message });
					return events;
				}
				const message = assistant(input, responseIndex, { parts: [{ type: "text", text: "done" }] });
				events.push({ type: "start", partial: message });
				events.push({ type: "text.end", partIndex: 0, content: "done", partial: message });
				events.push({ type: "done", reason: "stop", message });
				return events;
			});
	};

	const { effect: it } = testEffect(runtime({ open: chain(3) }));

	it(
		"an N-round chain opens and closes N turns, and keeps one mount for all of them",
		Effect.gen(function* () {
			const execution = yield* RunnerExecution.Service;
			const events = yield* Event.Service;
			const sessions = yield* Session.Service;
			const sessionId = yield* seedSession("chain");

			const starts: number[] = [];
			yield* events.listen((event) =>
				event.type === "session.turn.started"
					? Effect.sync(() => starts.push((event.data as { readonly turn: number }).turn))
					: Effect.void,
			);
			yield* admit({ id: "msg_chain", sessionId, delivery: "steer" });
			yield* execution.resume(sessionId);

			// Three tool rounds plus the answer. TurnStarted is live, so it is
			// counted from the listener rather than the durable log — the bracket is
			// per turn, not per chain.
			expect(starts).toEqual([1, 2, 3, 4]);
			const path = yield* sessions.path(sessionId);
			expect(path.map((item) => item.entry.type)).toEqual([
				"user",
				"assistant",
				"assistant",
				"assistant",
				"assistant",
			]);

			// One mount for the whole drain: every call ran in the same namespace,
			// which only holds if the mount outlived all four turns.
			const settled = path
				.flatMap((entry) => entry.parts)
				.filter((part) => part.type === "toolCall")
				.map((part) => JSON.parse(part.data) as Message.ToolCallCompletedPart);
			expect(settled).toHaveLength(3);
			expect(settled.every((part) => JSON.stringify(part.result.content).includes("/repo"))).toBe(true);
		}),
	);
});

describe("runner loop — sessions do not share a sandbox", () => {
	/** Write a file, then read the directory back. */
	const writeThenList = (command: string): LLM.Open => {
		let responseIndex = 0;
		return (input) =>
			Effect.sync(() => {
				responseIndex += 1;
				const events = createAssistantMessageEventStream();
				if (responseIndex === 1) {
					const part = {
						type: "toolCall",
						callID: "call_fs",
						name: "bash",
						arguments: { command },
						status: "pending",
						time: { start: 1, end: 1 },
					} as const satisfies Message.ToolCallPendingPart;
					const message = assistant(input, 1, { stopReason: "toolUse", parts: [part] });
					events.push({ type: "start", partial: message });
					events.push({ type: "toolcall.final", partIndex: 0, toolCall: part, partial: message });
					events.push({ type: "done", reason: "toolUse", message });
					return events;
				}
				const message = assistant(input, responseIndex, { parts: [{ type: "text", text: "ok" }] });
				events.push({ type: "start", partial: message });
				events.push({ type: "text.end", partIndex: 0, content: "ok", partial: message });
				events.push({ type: "done", reason: "stop", message });
				return events;
			});
	};

	const isolated = (command: string) => {
		const database = Database.layer(":memory:");
		const request = LLM.make(writeThenList(command));
		const sandbox = SandboxController.layer().pipe(Layer.provide(SandboxDriver.layer(fake.driver)));
		return Control.layer.pipe(
			Layer.provideMerge(
				RunnerExecute.layer.pipe(Layer.provide(Loop.layer({ request }).pipe(Layer.provide(State.layer())))),
			),
			Layer.provideMerge(Context.layer),
			Layer.provideMerge(SessionProjector.layer),
			Layer.provideMerge(Session.layer),
			Layer.provideMerge(Event.layer),
			Layer.provideMerge(sandbox),
			Layer.provideMerge(database),
		);
	};

	const bashOutput = Effect.fnUntraced(function* (slug: string) {
		const execution = yield* RunnerExecution.Service;
		const sessions = yield* Session.Service;
		const sessionId = yield* seedSession(slug);
		yield* admit({ id: `msg_${slug}`, sessionId, delivery: "steer" });
		yield* execution.resume(sessionId);
		const path = yield* sessions.path(sessionId);
		const part = JSON.parse(path[1]!.parts[0]!.data) as Message.ToolCallTerminalPart;
		return JSON.stringify(part.result.content);
	});

	// Each session gets its own instance from `seedSession`, so each mount is a
	// separate namespace even though both sit at /repo.
	testEffect(isolated("echo mine > marker.txt && ls")).effect(
		"one session's writes are its own",
		Effect.gen(function* () {
			expect(yield* bashOutput("fs-writer")).toContain("marker.txt");
		}),
	);

	testEffect(isolated("ls")).effect(
		"another session at the same path sees none of them",
		Effect.gen(function* () {
			// Same directory, different namespace: the file written above does not
			// exist here, and neither does any shell state that produced it.
			expect(yield* bashOutput("fs-reader")).not.toContain("marker.txt");
		}),
	);
});

describe("runner loop — a failing prompt override never reaches the provider", () => {
	let requests = 0;
	const counting: LLM.Open = (input) =>
		Effect.sync(() => {
			requests += 1;
			const message = assistant(input, requests);
			const events = createAssistantMessageEventStream();
			events.push({ type: "start", partial: message });
			events.push({ type: "done", reason: "stop", message });
			return events;
		});

	const database = Database.layer(":memory:");
	const request = LLM.make(counting);
	const sandbox = SandboxController.layer().pipe(Layer.provide(SandboxDriver.layer(fake.driver)));
	const layer = Control.layer.pipe(
		Layer.provideMerge(
			RunnerExecute.layer.pipe(
				Layer.provide(
					Loop.layer({ request }).pipe(
						Layer.provide(
							State.layer({
								promptSystemOverride: () => {
									throw new Error("override exploded");
								},
							}),
						),
					),
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

	testEffect(layer).effect(
		"fails the drain typed, before any request goes out",
		Effect.gen(function* () {
			requests = 0;
			const execution = yield* RunnerExecution.Service;
			const sessions = yield* Session.Service;
			const sessionId = yield* seedSession("override-fails");
			yield* admit({ id: "msg_override_fails", sessionId, delivery: "steer" });

			const exit = yield* execution.resume(sessionId).pipe(Effect.exit);
			expect(Exit.isFailure(exit)).toBe(true);
			// A caller's callback failing is a caller bug, but one the session should
			// report and survive rather than crash on — and the turn fails before any
			// provider request goes out.
			expect(requests).toBe(0);
			// State is captured before the prompt is promoted, so nothing was written.
			expect(yield* sessions.path(sessionId)).toEqual([]);
		}),
	);
});
