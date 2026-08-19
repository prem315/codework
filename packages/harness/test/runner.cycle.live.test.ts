import "./utils/env.ts";

import type { Message } from "@codeworksh/aikit";
import { Effect, Layer, Queue, Schema } from "effect";
import { SqlClient, SqlSchema } from "effect/unstable/sql";
import { describe, expect, it as vitestIt } from "vite-plus/test";
import { Context } from "../src/context/context.ts";
import { ContextCodec } from "../src/context/codec.ts";
import { Control } from "../src/control.ts";
import { Database } from "../src/db/db.ts";
import { Event } from "../src/event/event.ts";
import { RunnerExecute } from "../src/runner/execute.ts";
import { Loop } from "../src/runner/loop.ts";
import { SandboxController } from "../src/sandbox/control.ts";
import { SandboxDriver } from "../src/sandbox/driver.ts";
import { FakeSandboxDriver } from "../src/sandbox/drivers/fake.ts";
import { SandboxIO } from "../src/sandbox/io.ts";
import { AbsolutePath } from "../src/schema.ts";
import { SessionLive } from "../src/session/live.ts";
import type { SessionSchema } from "../src/session/schema.ts";
import { Session } from "../src/session/session.ts";
import { State } from "../src/state/state.ts";

/*
 * Live in the provider dimension only. The drain mounts the session sandbox and
 * proves its directory exists, and a virtual namespace gives this test a
 * filesystem it can create that directory in — mounting the host instead would
 * make Project walk the real tree for a case that is about OpenAI, not about
 * files.
 */
const fake = FakeSandboxDriver.make(SandboxDriver.Name.make("runner-cycle-fake"));

/*
 * What `runtimeOptions` used to hardcode inside `LLM.open` before step C moved
 * the request bag to State. This case asserts that thinking survives the durable
 * round-trip, and a summary only exists if one was asked for — so the ask is now
 * stated here, which is also the end-to-end proof that a caller's provider
 * options reach aikit through State.
 */
const agent = {
	thinkingLevel: "high",
	providerOptions: { openai: { reasoningSummary: "auto" } },
} as const satisfies State.Options;

const runtime = Control.layer.pipe(
	Layer.provideMerge(RunnerExecute.layer.pipe(Layer.provide(Loop.layer().pipe(Layer.provide(State.layer(agent)))))),
	Layer.provideMerge(Context.layer),
	Layer.provideMerge(SessionLive.layer),
	Layer.provideMerge(Event.layer),
	Layer.provideMerge(SandboxController.layer().pipe(Layer.provide(SandboxDriver.layer(fake.driver)))),
	Layer.provideMerge(Database.layer(":memory:")),
);

const openaiLiveIt = process.env.OPENAI_API_KEY ? vitestIt : vitestIt.skip;

const eventTypesFor = (sql: SqlClient.SqlClient) =>
	SqlSchema.findAll({
		Request: Schema.String,
		Result: Schema.Struct({ type: Schema.String }),
		execute: (sessionId) => sql`SELECT type FROM event WHERE aggregate_id = ${sessionId} ORDER BY seq`,
	});

const inputStateFor = (sql: SqlClient.SqlClient) =>
	SqlSchema.findOne({
		Request: Schema.String,
		Result: Schema.Struct({ count: Schema.Int, pending: Schema.Int }),
		execute: (sessionId) => sql`
			SELECT
				COUNT(*) AS count,
				SUM(CASE WHEN promoted_seq IS NULL THEN 1 ELSE 0 END) AS pending
			FROM session_input
			WHERE session_id = ${sessionId}
		`,
	});

const textOf = (message: Message.Message): string =>
	message.parts
		.flatMap((part) => (part.type === "text" ? [part.text] : []))
		.join("\n")
		.trim();

const thinkingOf = (message: Message.Message): string =>
	message.parts
		.flatMap((part) => (part.type === "thinking" ? [part.thinking] : []))
		.join("\n")
		.trim();

/**
 * The transcript's shape, not its length.
 *
 * A turn is one assistant response plus any tool calls it makes, and an exchange
 * is one or more turns — so a prompt now produces at least one assistant entry
 * and possibly several. Asserting strict user/assistant alternation would be
 * asserting that the model never uses the tools it is offered, which is not a
 * property of the durable cycle this case is about.
 */
const shape = (entries: ReadonlyArray<{ readonly entry: { readonly type: string } }>) => ({
	prompts: entries.filter((item) => item.entry.type === "user").length,
	last: entries.at(-1)?.entry.type,
	alternates: entries.every((item, index) =>
		index === 0
			? item.entry.type === "user"
			: item.entry.type !== "user" || entries[index - 1]?.entry.type !== "user",
	),
});

const waitUntilIdle = (control: Control.Interface, sessionId: SessionSchema.ID) =>
	Effect.gen(function* () {
		while ((yield* control.active).has(sessionId)) yield* Effect.sleep("10 millis");
	}).pipe(Effect.timeout("120 seconds"));

const historyPrompts = [
	"We are building a continuity test around a fictional moon named Velora. In one short sentence, state the color of its sky.",
	"In one short sentence, name Velora's capital and its defining landmark.",
	"In one short sentence, introduce an archivist who lives in that capital.",
	"In one short sentence, state the archivist's central problem.",
	"In one short sentence, name the object that might solve that problem.",
] as const;

describe("runner cycle — OpenAI live", () => {
	openaiLiveIt(
		"runs durable input through Loop and aikit, preserves an interrupted response, then recovers",
		{ timeout: 480_000 },
		() =>
			Effect.runPromise(
				Effect.gen(function* () {
					const sql = yield* SqlClient.SqlClient;
					const control = yield* Control.Service;
					const context = yield* Context.Service;
					const events = yield* Event.Service;
					const sessions = yield* Session.Service;

					yield* sql`INSERT OR IGNORE INTO project (id, name, created_at, updated_at) VALUES ('local','local',0,0)`;
					const controller = yield* SandboxController.Controller;
					const instance = yield* controller.create({
						driver: fake.driver,
						config: { defaultCwd: SandboxDriver.AbsolutePath.make("/") },
					});
					// The mount must find this directory, so create it before the drain does.
					yield* Effect.flatMap(SandboxIO.FileSystem, (fs) => fs.mkdir("/repo", { recursive: true })).pipe(
						Effect.provide(Layer.unwrap(Effect.succeed(controller.mount(instance.id)))),
						Effect.scoped,
					);
					const session = yield* sessions.create({
						projectId: "local",
						slug: `runner-cycle-${crypto.randomUUID()}`,
						directory: AbsolutePath.make("/repo"),
						title: "Live runner cycle",
						tag: "test",
						sandboxInstanceId: instance.id,
					});

					const terminals: Array<"ended" | "failed"> = [];
					const removeTerminalListener = yield* events.listen((event) => {
						if (typeof event.data !== "object" || event.data === null) return Effect.void;
						const data = event.data as Record<string, unknown>;
						if (data.sessionId !== session.id) return Effect.void;
						if (event.type === "session.llm.ended") return Effect.sync(() => terminals.push("ended"));
						if (event.type === "session.llm.failed") return Effect.sync(() => terminals.push("failed"));
						return Effect.void;
					});
					yield* Effect.addFinalizer(() => removeTerminalListener);

					const readLastAssistant = Effect.fnUntraced(function* () {
						const path = yield* sessions.path(session.id);
						const last = path.at(-1);
						if (last === undefined) return yield* Effect.die("session path is empty after an LLM terminal event");
						const message = yield* ContextCodec.decodeMessage(last);
						if (message.role !== "assistant") {
							return yield* Effect.die(`last session entry is ${message.role}, not assistant`);
						}
						return message;
					});

					/*
					 * Every terminal an exchange produced, not just its first.
					 *
					 * An exchange is one or more turns and each turn publishes its own
					 * terminal, so consuming a single one per prompt would leave the rest
					 * behind and hand the next assertion a stale outcome. Waiting for one
					 * terminal proves the drain started; waiting for idle proves it
					 * finished.
					 */
					const settle = Effect.fnUntraced(function* () {
						yield* Effect.gen(function* () {
							while (terminals.length === 0) yield* Effect.sleep("10 millis");
						}).pipe(Effect.timeout("120 seconds"));
						yield* waitUntilIdle(control, session.id);
						const settled = [...terminals];
						terminals.length = 0;
						return settled;
					});

					const ask = Effect.fnUntraced(function* (prompt: string) {
						yield* control.prompt({ sessionId: session.id, prompt: { text: prompt } });
						const settled = yield* settle();
						expect(settled.every((outcome) => outcome === "ended")).toBe(true);
						return yield* readLastAssistant();
					});

					const persistedThinking: string[] = [];
					for (const [index, prompt] of historyPrompts.entries()) {
						const assistant = yield* ask(prompt);
						const thinking = thinkingOf(assistant);
						if (thinking.length > 0) persistedThinking.push(thinking);
						const path = yield* sessions.path(session.id);
						// One prompt admitted per ask, each answered, and no two prompts
						// ever adjacent — the transcript stays well-formed however many
						// turns the model took to answer.
						expect(shape(path)).toEqual({ prompts: index + 1, last: "assistant", alternates: true });
						yield* Effect.logInfo("live conversation pair", {
							turn: index + 1,
							user: prompt,
							thinking,
							assistant: textOf(assistant),
						});
					}
					expect(persistedThinking.length).toBeGreaterThan(0);

					const textDeltas = yield* Queue.unbounded<string>();
					const removeDeltaListener = yield* events.listen((event) => {
						if (typeof event.data !== "object" || event.data === null) return Effect.void;
						const data = event.data as Record<string, unknown>;
						if (data.sessionId !== session.id || typeof data.delta !== "string" || data.delta.length === 0) {
							return Effect.void;
						}
						if (event.type === "session.llm.text.delta") {
							return Queue.offer(textDeltas, data.delta).pipe(Effect.asVoid);
						}
						return Effect.void;
					});
					yield* Effect.addFinalizer(() => removeDeltaListener);

					const storyPrompt =
						"Using every established detail about Velora, write a vivid story of exactly 100 words. Do not preface or explain it.";
					yield* control.prompt({ sessionId: session.id, prompt: { text: storyPrompt } });
					const firstTextDelta = yield* Queue.take(textDeltas).pipe(Effect.timeout("120 seconds"));
					yield* control.interrupt(session.id);
					const interruptedTerminals = yield* settle();

					// The interrupt lands on whichever turn was streaming, so the turns
					// before it may have settled normally — what matters is that the
					// exchange ends failed.
					expect(interruptedTerminals.at(-1)).toBe("failed");
					expect(Array.from(yield* control.active)).toEqual([]);
					const interruptedPath = yield* sessions.path(session.id);
					expect(shape(interruptedPath)).toEqual({ prompts: 6, last: "assistant", alternates: true });
					const aborted = yield* readLastAssistant();
					const abortedText = textOf(aborted);
					const abortedThinking = thinkingOf(aborted);
					expect(aborted.stopReason).toBe("aborted");
					expect(abortedText.length).toBeGreaterThan(0);

					const eventTypes = yield* eventTypesFor(sql)(session.id);
					expect(eventTypes.at(-1)?.type).toBe("session.llm.failed.1");
					yield* Effect.logInfo("live interrupted response", {
						turn: 6,
						user: storyPrompt,
						firstTextDelta,
						persistedThinking: abortedThinking,
						persistedText: abortedText,
						stopReason: aborted.stopReason,
						durableTerminal: eventTypes.at(-1)?.type,
					});

					const recoveryPrompt =
						"The previous generation was interrupted. In exactly five words, confirm that you can continue.";
					const recovered = yield* ask(recoveryPrompt);
					expect(recovered.stopReason).toBe("stop");

					const finalPath = yield* sessions.path(session.id);
					expect(shape(finalPath)).toEqual({ prompts: 7, last: "assistant", alternates: true });
					const snapshot = yield* context.assemble(session.id);
					expect(snapshot.messages).toHaveLength(finalPath.length);
					// The aborted response survives in history, and the recovery after it
					// completed — the two facts the interruption half of this case exists
					// to prove.
					expect(snapshot.messages.filter((message) => message.role === "assistant")).toContainEqual(
						expect.objectContaining({ stopReason: "aborted" }),
					);
					expect(snapshot.messages.at(-1)).toMatchObject({ role: "assistant", stopReason: "stop" });

					const inputState = yield* inputStateFor(sql)(session.id);
					expect(inputState).toEqual({ count: 7, pending: 0 });
					const finalEventTypes = yield* eventTypesFor(sql)(session.id);
					/*
					 * Counted by type rather than totalled: the number of block
					 * completions is the model's to choose, so a total would assert on
					 * how verbose the response happened to be. What is fixed is one
					 * allocation per exchange, one terminal per exchange, and a turn
					 * boundary for every exchange that reached one — the interrupted turn
					 * failed before it could close.
					 */
					const tally = (type: string) => finalEventTypes.filter((row) => row.type === type).length;
					expect(tally("session.next.prompt.promoted.1")).toBe(7);
					expect(tally("session.llm.failed.1")).toBe(1);
					// One allocation per turn, one terminal per turn, one boundary per
					// turn that reached one — the interrupted turn failed before it could
					// close, so the boundaries trail the allocations by exactly that one.
					expect(tally("session.llm.started.1")).toBe(finalPath.length - 7);
					expect(tally("session.llm.ended.1")).toBe(tally("session.llm.started.1") - 1);
					expect(tally("session.turn.ended.1")).toBe(tally("session.llm.ended.1"));
					// Every call the harness began, it also settled.
					expect(tally("session.tool.execution.ended.1")).toBe(tally("session.tool.execution.started.1"));
					expect(finalEventTypes.at(-1)?.type).toBe("session.turn.ended.1");
					yield* Effect.logInfo("live recovery response", {
						turn: 7,
						user: recoveryPrompt,
						assistant: textOf(recovered),
						pathEntries: finalPath.length,
						durableEvents: finalEventTypes.length,
					});
				}).pipe(Effect.scoped, Effect.provide(runtime)),
			),
	);
});
