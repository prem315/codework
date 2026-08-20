import type { Message } from "@codeworksh/aikit";
import { DateTime, Effect, Exit, Layer, Option } from "effect";
import { SqlClient } from "effect/unstable/sql";
import { describe, expect } from "vite-plus/test";
import { Database } from "../src/db/db.ts";
import { Event } from "../src/event/event.ts";
import { EventList } from "../src/event/list.ts";
import { SandboxInstance } from "../src/sandbox/instance.ts";
import { AbsolutePath } from "../src/schema.ts";
import { SessionLive } from "../src/session/live.ts";
import { SessionMessageSchema } from "../src/session/message/schema.ts";
import type { SessionSchema } from "../src/session/schema.ts";
import { Session } from "../src/session/session.ts";
import { testEffect } from "./utils/effect.ts";

/**
 * The output half on its own: publish the durable LLM events and assert what
 * reaches `session_entry`. No loop, no publisher, no provider — the whole point
 * of splitting the phases is that this side is provable without them.
 */
const layer = SessionLive.layer.pipe(Layer.provideMerge(Event.layer), Layer.provideMerge(Database.layer(":memory:")));
const { effect: it } = testEffect(layer);

const setup = Effect.gen(function* () {
	const sql = yield* SqlClient.SqlClient;
	yield* sql`INSERT OR IGNORE INTO project (id, name, created_at, updated_at) VALUES ('local','local',0,0)`;
	const sessions = yield* Session.Service;
	const session = yield* sessions.create({
		projectId: "local",
		slug: "llm",
		directory: AbsolutePath.make("/repo"),
		title: "T",
		tag: "test",
		sandboxInstanceId: SandboxInstance.ID.local,
	});
	return { sessions, events: yield* Event.Service, sessionId: session.id };
});

const messageId = SessionMessageSchema.ID.create();

/** A terminal assistant message, shaped exactly as aikit would hand it over. */
const assistant = (overrides: Partial<Message.AssistantMessage> = {}): Message.AssistantMessage => ({
	messageId,
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

/**
 * The envelope `LLMStarted` creates the entry from: identity and provenance,
 * no content, no usage, `stopReason: "aborted"` until a terminal promotes it.
 */
const creation = (): Message.AssistantMessage =>
	assistant({
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
	});

/** Allocate the entry, the way every response now begins. */
const start = Effect.fnUntraced(function* (sessionId: SessionSchema.ID) {
	const events = yield* Event.Service;
	return yield* events.publish(EventList.LLMStarted, {
		sessionId,
		messageId,
		timestamp: DateTime.makeUnsafe(0),
		message: creation(),
	});
});

const usageTotals = Effect.gen(function* () {
	const sql = yield* SqlClient.SqlClient;
	// The client is configured with `transformResultNames: snakeToCamel`, so the
	// columns come back camelCased even from a raw query.
	const rows = yield* sql`SELECT cost, tokens_input, tokens_output, tokens_cache_read FROM session`;
	return rows[0] as { cost: number; tokensInput: number; tokensOutput: number; tokensCacheRead: number };
});

const eventCount = Effect.gen(function* () {
	const sql = yield* SqlClient.SqlClient;
	return (yield* sql`SELECT id FROM event`).length;
});

describe("LLM terminal projection", () => {
	it("creates the entry at session.llm.started, empty and aborted", () =>
		Effect.gen(function* () {
			const { sessions, sessionId } = yield* setup;
			const published = yield* start(sessionId);

			const path = yield* sessions.path(sessionId);
			expect(path.length).toBe(1);
			const [entry] = path;
			// The entry is the message: same id, and positioned by the event that
			// produced it rather than by a counter of its own.
			expect(entry!.entry.id).toBe(messageId);
			expect(entry!.entry.type).toBe("assistant");
			expect(entry!.entry.seq).toBe(published.durable?.seq);
			// The empty array the block completions fill, marked so a crash here
			// leaves a truthful record rather than a response that looks finished.
			expect(entry!.parts).toEqual([]);
			expect(JSON.parse(entry!.entry.data)).toMatchObject({ stopReason: "aborted" });
			expect((yield* usageTotals).tokensInput).toBe(0);
		}));

	it("writes a completed block at its partIndex, before any terminal", () =>
		Effect.gen(function* () {
			const { sessions, events, sessionId } = yield* setup;
			yield* start(sessionId);

			yield* events.publish(EventList.LLMThinkingEnd, {
				sessionId,
				messageId,
				timestamp: DateTime.makeUnsafe(0),
				partIndex: 0,
				// The signature rides along: it is what a provider needs to continue
				// this reasoning thread on the next turn.
				part: { type: "thinking", thinking: "hmm", thinkingSignature: "sig" },
			});
			yield* events.publish(EventList.LLMTextEnd, {
				sessionId,
				messageId,
				timestamp: DateTime.makeUnsafe(0),
				partIndex: 1,
				part: { type: "text", text: "the answer" },
			});

			const [entry] = yield* sessions.path(sessionId);
			expect(entry!.parts.map((part) => part.type)).toEqual(["thinking", "text"]);
			expect(JSON.parse(entry!.parts[0]!.data)).toEqual({
				type: "thinking",
				thinking: "hmm",
				thinkingSignature: "sig",
			});
			// Still aborted: a block completing is not a response settling.
			expect(JSON.parse(entry!.entry.data)).toMatchObject({ stopReason: "aborted" });
		}));

	it("writes a finalized tool call as a pending part with its columns promoted", () =>
		Effect.gen(function* () {
			const { sessions, events, sessionId } = yield* setup;
			yield* start(sessionId);
			const part = {
				type: "toolCall",
				callID: "call_1",
				name: "bash",
				arguments: { command: "pwd" },
				status: "pending",
				time: { start: 1, end: 2 },
			} as const satisfies Message.ToolCallPendingPart;

			yield* events.publish(EventList.LLMToolCallFinalized, {
				sessionId,
				messageId,
				timestamp: DateTime.makeUnsafe(0),
				partIndex: 0,
				callId: part.callID,
				toolName: part.name,
				part,
			});

			const [entry] = yield* sessions.path(sessionId);
			expect(JSON.parse(entry!.parts[0]!.data)).toEqual(part);
			// `pending` is the whole point: the model finished asking, and the
			// transition into `running` belongs to ToolExecutionStarted alone.
			expect(Option.getOrNull(entry!.parts[0]!.status)).toBe("pending");
			expect(Option.getOrNull(entry!.parts[0]!.callId)).toBe("call_1");
			expect(Option.getOrNull(entry!.parts[0]!.toolName)).toBe("bash");
			expect(JSON.parse(entry!.entry.data)).toMatchObject({ stopReason: "aborted" });
		}));

	it("promotes the entry at session.llm.ended and reconciles its parts", () =>
		Effect.gen(function* () {
			const { sessions, events, sessionId } = yield* setup;
			yield* start(sessionId);
			yield* events.publish(EventList.LLMTextEnd, {
				sessionId,
				messageId,
				timestamp: DateTime.makeUnsafe(0),
				partIndex: 0,
				part: { type: "text", text: "the ans" },
			});

			yield* events.publish(EventList.LLMEnded, {
				sessionId,
				messageId,
				timestamp: DateTime.makeUnsafe(0),
				reason: "stop",
				message: assistant(),
			});

			const path = yield* sessions.path(sessionId);
			expect(path.length).toBe(1);
			const [entry] = path;
			// One entry, promoted in place — the terminal does not append a second.
			expect(entry!.entry.id).toBe(messageId);
			expect(entry!.parts.map((part) => part.type)).toEqual(["text"]);
			// The terminal is authoritative, so it overwrites the slot the block
			// completion filled rather than landing beside it.
			expect(JSON.parse(entry!.parts[0]!.data)).toEqual({ type: "text", text: "the answer" });
			expect(JSON.parse(entry!.entry.data)).toMatchObject({ role: "assistant", stopReason: "stop" });
		}));

	it("rejects a terminal whose message disagrees with its own reason", () =>
		Effect.gen(function* () {
			const { sessions, events, sessionId } = yield* setup;
			yield* start(sessionId);

			// The event says how the stream ended and the message says how it ended.
			// A disagreement means the publisher and aikit no longer describe the
			// same response, and the envelope about to be written is the one that
			// would be believed forever after.
			const exit = yield* events
				.publish(EventList.LLMEnded, {
					sessionId,
					messageId,
					timestamp: DateTime.makeUnsafe(0),
					reason: "toolUse",
					message: assistant({ stopReason: "stop" }),
				})
				.pipe(Effect.exit);
			expect(Exit.isFailure(exit)).toBe(true);

			// And the rejection took the durable event down with it: the entry is
			// still the unfinalized placeholder.
			const [entry] = yield* sessions.path(sessionId);
			expect(JSON.parse(entry!.entry.data)).toMatchObject({ stopReason: "aborted" });
		}));

	it("creates the entry as a draft and settles it at the terminal", () =>
		Effect.gen(function* () {
			const { sessions, events, sessionId } = yield* setup;
			yield* start(sessionId);

			// A draft is the one entry type that is not complete when it is written.
			expect((yield* sessions.path(sessionId))[0]!.entry.state).toBe("draft");

			yield* events.publish(EventList.LLMEnded, {
				sessionId,
				messageId,
				timestamp: DateTime.makeUnsafe(0),
				reason: "stop",
				message: assistant(),
			});

			// `stop`, `length`, and `toolUse` all commit; which it was stays in the
			// envelope rather than being smeared across two vocabularies.
			const [entry] = yield* sessions.path(sessionId);
			expect(entry!.entry.state).toBe("committed");
			expect(JSON.parse(entry!.entry.data)).toMatchObject({ stopReason: "stop" });
		}));

	it("settles a failed response as aborted or error, not as committed", () =>
		Effect.gen(function* () {
			const { sessions, events, sessionId } = yield* setup;
			yield* start(sessionId);

			yield* events.publish(EventList.LLMFailed, {
				sessionId,
				messageId,
				timestamp: DateTime.makeUnsafe(0),
				reason: "aborted",
				message: assistant({ stopReason: "aborted", errorMessage: "interrupted" }),
			});

			// The column answers "did it end badly" without parsing the envelope,
			// which `stopReason` alone could never do — a placeholder carries
			// `aborted` too.
			expect((yield* sessions.path(sessionId))[0]!.entry.state).toBe("aborted");
		}));

	it("rejects a second, different terminal on an already aborted entry", () =>
		Effect.gen(function* () {
			const { sessions, events, sessionId } = yield* setup;
			yield* start(sessionId);
			const failed = assistant({ stopReason: "aborted", errorMessage: "interrupted" });
			yield* events.publish(EventList.LLMFailed, {
				sessionId,
				messageId,
				timestamp: DateTime.makeUnsafe(0),
				reason: "aborted",
				message: failed,
			});

			/*
			 * This is what `stopReason` could not catch. An aborted terminal writes
			 * the same value the draft placeholder held, so finality was unanswerable
			 * and a second terminal was accepted — and its usage charged again.
			 */
			const second = yield* events
				.publish(EventList.LLMEnded, {
					sessionId,
					messageId,
					timestamp: DateTime.makeUnsafe(1),
					reason: "stop",
					message: assistant(),
				})
				.pipe(Effect.exit);
			expect(Exit.isFailure(second)).toBe(true);
			expect((yield* sessions.path(sessionId))[0]!.entry.state).toBe("aborted");
			expect((yield* usageTotals).tokensInput).toBe(11);
		}));

	it("rejects any second finalization, identical or not", () =>
		Effect.gen(function* () {
			const { sessions, events, sessionId } = yield* setup;
			yield* start(sessionId);
			const terminal = {
				sessionId,
				messageId,
				timestamp: DateTime.makeUnsafe(0),
				reason: "stop",
				message: assistant(),
			} as const;
			yield* events.publish(EventList.LLMEnded, terminal);

			/*
			 * Not even an identical redelivery. The commit path rejects a repeated
			 * event id before projectors run, and nothing publishes the same logical
			 * event twice — so a second one means an assumption broke, and absorbing
			 * it is how a double-charge would stay invisible.
			 */
			const replay = yield* events
				.publish(EventList.LLMEnded, { ...terminal, timestamp: DateTime.makeUnsafe(1) })
				.pipe(Effect.exit);
			expect(Exit.isFailure(replay)).toBe(true);

			// The rejection took the durable event with it: one entry, charged once.
			expect((yield* usageTotals).tokensInput).toBe(11);
			expect((yield* sessions.path(sessionId)).length).toBe(1);
			expect(JSON.parse((yield* sessions.path(sessionId))[0]!.entry.data)).toMatchObject({ stopReason: "stop" });
		}));

	it("stores a failed response the same way, keeping what the model produced", () =>
		Effect.gen(function* () {
			const { sessions, events, sessionId } = yield* setup;
			yield* start(sessionId);

			yield* events.publish(EventList.LLMFailed, {
				sessionId,
				messageId,
				timestamp: DateTime.makeUnsafe(0),
				reason: "aborted",
				// aikit reports an abort as a complete message carrying everything
				// generated before it, so the partial text is history, not a loss.
				message: assistant({
					stopReason: "aborted",
					errorMessage: "Request was aborted",
					parts: [{ type: "text", text: "half an ans" }],
				}),
			});

			const path = yield* sessions.path(sessionId);
			expect(path.map((item) => item.entry.type)).toEqual(["assistant"]);
			expect(JSON.parse(path[0]!.entry.data)).toMatchObject({
				stopReason: "aborted",
				errorMessage: "Request was aborted",
			});
			expect(JSON.parse(path[0]!.parts[0]!.data)).toEqual({ type: "text", text: "half an ans" });
		}));

	it("charges usage once, from the terminal message", () =>
		Effect.gen(function* () {
			const { events, sessionId } = yield* setup;
			// Creation charges nothing; only the terminal carries final usage.
			yield* start(sessionId);
			expect((yield* usageTotals).tokensInput).toBe(0);

			yield* events.publish(EventList.LLMEnded, {
				sessionId,
				messageId,
				timestamp: DateTime.makeUnsafe(0),
				reason: "stop",
				message: assistant(),
			});

			const totals = yield* usageTotals;
			expect(totals.tokensInput).toBe(11);
			expect(totals.tokensOutput).toBe(22);
			expect(totals.tokensCacheRead).toBe(3);
			expect(totals.cost).toBeCloseTo(0.3);
		}));

	it("rolls back the entry, the event row, and the sequence when the ids disagree", () =>
		Effect.gen(function* () {
			const { sessions, events, sessionId } = yield* setup;
			yield* start(sessionId);
			const before = yield* events.latestSequence(sessionId);

			// The event names one message, the payload names another. Storing this
			// would put the entry under an id its own envelope contradicts.
			const failure = yield* events
				.publish(EventList.LLMEnded, {
					sessionId,
					messageId,
					timestamp: DateTime.makeUnsafe(0),
					reason: "stop",
					message: assistant({ messageId: SessionMessageSchema.ID.create() }),
				})
				.pipe(Effect.exit);

			expect(failure._tag).toBe("Failure");
			// A projector runs inside the commit, so its rejection takes the event and
			// the sequence allocation down with it — the entry stays as `LLMStarted`
			// left it, unfinalized, and no second event row survives.
			const [entry] = yield* sessions.path(sessionId);
			expect(JSON.parse(entry!.entry.data)).toMatchObject({ stopReason: "aborted" });
			expect(yield* eventCount).toBe(1);
			expect(yield* events.latestSequence(sessionId)).toBe(before);
		}));

	it("lands after a user entry, so the transcript reads in order", () =>
		Effect.gen(function* () {
			const { sessions, events, sessionId } = yield* setup;
			const userId = SessionMessageSchema.ID.create();

			yield* events.publish(EventList.Prompted, {
				sessionId,
				messageId: userId,
				timestamp: DateTime.makeUnsafe(0),
				prompt: { text: "ask" },
				delivery: "steer",
			});
			yield* start(sessionId);
			yield* events.publish(EventList.LLMEnded, {
				sessionId,
				messageId,
				timestamp: DateTime.makeUnsafe(0),
				reason: "stop",
				message: assistant(),
			});

			const path = yield* sessions.path(sessionId);
			expect(path.map((item) => item.entry.id)).toEqual([userId, messageId]);
			expect(path.map((item) => item.entry.type)).toEqual(["user", "assistant"]);
			// Both positions come from the same log, so the assistant is strictly above
			// the prompt it answers.
			expect(path[1]!.entry.seq).toBeGreaterThan(path[0]!.entry.seq);
		}));
});
