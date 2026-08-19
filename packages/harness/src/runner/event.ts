/*
 * @file Translates one aikit provider stream into Harness events.
 *
 * Request-scoped by construction: `make` is an Effect, not a `Layer` or a
 * `Context.Service`. Those are graph-lifetime, so a service here would share one
 * request's latches with every other request on the process. Every
 * `yield* make(...)` allocates its own.
 *
 * The LLM boundary sequences the stream and hands each event over; this
 * module decides what the event means. It writes nothing itself -- the two
 * durable events reach `session_entry` through the projectors in
 * `session/projector.ts`.
 *
 * There are no fragment buffers here, which is the main structural difference
 * from the reference implementation this file is named after. Its provider
 * events carry only fragments, so it has to reassemble text across deltas before
 * it can persist a block. Every aikit event already carries a complete assistant
 * message, and the terminal one carries the whole response -- including whatever
 * was generated before an abort -- so there is nothing to reassemble and nothing
 * to flush.
 */

import { Model, type Event as AikitEvent, type Message } from "@codeworksh/aikit";
import { DateTime, Effect } from "effect";
import { Event } from "../event/event.ts";
import { EventList } from "../event/list.ts";
import { SessionMessageSchema } from "../session/message/schema.ts";
import type { SessionSchema } from "../session/schema.ts";
import { Runner } from "./run.ts";

/** How the response settled. The loop branches on this, never on `result()`. */
export type Terminal =
	| {
			readonly outcome: "ended";
			readonly reason: "stop" | "length" | "toolUse";
			readonly message: Message.AssistantMessage;
	  }
	| {
			readonly outcome: "failed";
			readonly reason: "aborted" | "error";
			readonly message: Message.AssistantMessage;
	  };

export interface Publisher {
	/** Interpret one provider event. Durable ones commit before this returns. */
	readonly publish: (event: AikitEvent.LLMMessageEvent) => Effect.Effect<void, Runner.LLMStreamError>;
	/**
	 * Record a failure that will produce no aikit terminal -- an unresolvable
	 * model, or a stream that threw before it opened.
	 *
	 * The assistant entry is created first, so an unreachable provider still
	 * lands on the timeline as a failed assistant rather than vanishing into a
	 * typed error and a log line. A no-op once the response has settled.
	 */
	readonly failAssistant: (input: {
		readonly reason: "aborted" | "error";
		readonly errorMessage: string;
	}) => Effect.Effect<void, Runner.LLMStreamError>;
	/** How the response settled; fails if the stream ended without terminating. */
	readonly terminal: Effect.Effect<Terminal, Runner.LLMStreamError>;
}

const zeroUsage: Message.Usage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

const knownProtocols: ReadonlySet<string> = new Set(Object.values(Model.KnownProviderEnum));

/**
 * Best-effort protocol for an envelope no provider ever produced.
 *
 * Every real response carries aikit's own resolved protocol. This is the other
 * case: the request never reached a provider, so nothing resolved one, and the
 * envelope still has to name it. `openai-compatible` is the honest default for
 * a provider id aikit does not recognize -- it is what an unknown provider
 * would have spoken had it answered.
 */
const protocolOf = (provider: string): Model.KnownProviderEnum =>
	knownProtocols.has(provider) ? (provider as Model.KnownProviderEnum) : Model.KnownProviderEnum.openaiCompatible;

export const make = Effect.fn("LLMEventPublisher.make")(function* (input: {
	readonly sessionId: SessionSchema.ID;
	readonly provider: string;
	readonly model: string;
}) {
	const events = yield* Event.Service;

	let messageId: SessionMessageSchema.ID | undefined;
	let settled: Terminal | undefined;

	const fault = (reason: string) => new Runner.LLMStreamError({ sessionId: input.sessionId, reason });

	/**
	 * Every event names the message it belongs to. The first one to arrive fixes
	 * the identity for the whole response; a later disagreement means two
	 * responses are interleaving on one stream, which would silently file half of
	 * one under the other.
	 *
	 * Reading `messageId` off `partial` is safe even though `partial` itself is
	 * not: aikit mutates that object in place as the stream advances, but the id
	 * is fixed when the message is created and never moves.
	 */
	const identify = (observed: string) =>
		Effect.suspend(() => {
			const id = SessionMessageSchema.ID.from(observed);
			if (messageId === undefined) {
				messageId = id;
				return Effect.succeed(id);
			}
			return messageId === id
				? Effect.succeed(messageId)
				: Effect.fail(fault(`message id changed mid-response: "${messageId}" then "${id}"`));
		});

	/**
	 * The envelope the assistant entry is created from: identity and provenance,
	 * no content.
	 *
	 * `parts` is emptied because block completions fill them one slot at a time,
	 * and `usage` is zeroed because usage is charged exactly once, at
	 * finalization -- a partial that already carried a count would be charged
	 * twice. `stopReason` is `aborted` so a crash mid-stream leaves an entry that
	 * is already correctly marked; the terminal promotes it.
	 *
	 * Nested objects are copied rather than shared. aikit rewrites `partial` in
	 * place as the stream advances, so anything this envelope keeps by reference
	 * would keep changing under the durable value.
	 */
	const creationEnvelope = (source: Message.AssistantMessage): Message.AssistantMessage => ({
		...source,
		provider: { ...source.provider },
		time: { ...source.time },
		usage: { ...zeroUsage, cost: { ...zeroUsage.cost } },
		parts: [],
		stopReason: "aborted",
	});

	/**
	 * Create the assistant entry, once.
	 *
	 * Every path that needs an assistant goes through here, so there is one
	 * creation site rather than a normal path plus a projector fallback -- the
	 * projectors stay pure projections with no conditional create. Later calls
	 * return the id the first one minted.
	 */
	const startAssistant = (
		seed?: Message.AssistantMessage,
	): Effect.Effect<SessionMessageSchema.ID, Runner.LLMStreamError> =>
		Effect.gen(function* () {
			// A terminal still has its identity checked: arriving under a different
			// message id means two responses are interleaving on one stream.
			if (messageId !== undefined) return seed === undefined ? messageId : yield* identify(seed.messageId);
			const now = yield* DateTime.now;
			const created = DateTime.toEpochMillis(now);
			const envelope = creationEnvelope(
				seed ?? {
					messageId: SessionMessageSchema.ID.create(),
					role: "assistant",
					protocol: protocolOf(input.provider),
					provider: { id: input.provider, name: input.provider, source: "custom", env: [] },
					model: input.model,
					usage: zeroUsage,
					stopReason: "aborted",
					time: { created, completed: created },
					parts: [],
				},
			);
			const id = yield* identify(envelope.messageId);
			yield* events.publish(EventList.LLMStarted, {
				sessionId: input.sessionId,
				timestamp: now,
				messageId: id,
				message: envelope,
			});
			return id;
		});

	/**
	 * The assistant a part event belongs to. Dying is the point: a part with no
	 * assistant means the stream is broken and there is nothing to salvage,
	 * whereas a terminal with no assistant is an ordinary failure carrying a
	 * complete message to record.
	 */
	const requireAssistant = (type: string, observed: string) =>
		Effect.suspend(() =>
			messageId === undefined
				? Effect.die(`llm: "${type}" arrived before the assistant was started`)
				: identify(observed),
		);

	const failAssistant = (failure: {
		readonly reason: "aborted" | "error";
		readonly errorMessage: string;
	}): Effect.Effect<void, Runner.LLMStreamError> =>
		Effect.gen(function* () {
			if (settled !== undefined) return;
			const id = yield* startAssistant();
			const now = yield* DateTime.now;
			const completed = DateTime.toEpochMillis(now);
			const message: Message.AssistantMessage = {
				messageId: id,
				role: "assistant",
				protocol: protocolOf(input.provider),
				provider: { id: input.provider, name: input.provider, source: "custom", env: [] },
				model: input.model,
				usage: zeroUsage,
				stopReason: failure.reason,
				errorMessage: failure.errorMessage,
				time: { created: completed, completed },
				parts: [],
			};
			yield* events.publish(EventList.LLMFailed, {
				sessionId: input.sessionId,
				timestamp: now,
				messageId: id,
				reason: failure.reason,
				message,
			});
			settled = { outcome: "failed", reason: failure.reason, message };
		});

	/*
	 * The finished block a `*.end` event refers to.
	 *
	 * aikit's end events carry only the block's text, but it writes
	 * `textSignature` / `thinkingSignature` onto the block immediately before
	 * pushing (`aikit/llm/stream.ts:253-282`), and those signatures are what a
	 * provider needs to continue a reasoning thread across turns. A durable write
	 * built from the scalar would drop them, leaving a resumed session unable to
	 * continue its own thinking.
	 *
	 * Reading `partial` is safe here and nowhere else: a completed block is the
	 * one thing in that object aikit will not touch again. It is copied anyway,
	 * because the array around it keeps growing.
	 */
	const textBlock = (partial: Message.AssistantMessage, partIndex: number) =>
		Effect.suspend(() => {
			const block = partial.parts[partIndex];
			return block?.type === "text"
				? Effect.succeed({ ...block })
				: Effect.fail(fault(`"text.end" names part ${partIndex}, which is not a text block`));
		});

	const thinkingBlock = (partial: Message.AssistantMessage, partIndex: number) =>
		Effect.suspend(() => {
			const block = partial.parts[partIndex];
			return block?.type === "thinking"
				? Effect.succeed({ ...block })
				: Effect.fail(fault(`"thinking.end" names part ${partIndex}, which is not a thinking block`));
		});

	const publish = (event: AikitEvent.LLMMessageEvent): Effect.Effect<void, Runner.LLMStreamError> =>
		Effect.gen(function* () {
			// Exactly one terminal per response. A second, or anything after one,
			// would append a second assistant entry for the same request.
			if (settled !== undefined) {
				return yield* fault(`received "${event.type}" after the response had already terminated`);
			}
			const timestamp = yield* DateTime.now;
			const base = { sessionId: input.sessionId, timestamp };

			switch (event.type) {
				case "start": {
					yield* startAssistant(event.partial);
					return;
				}

				/*
				 * Live only, all of them. The terminal message contains every finished
				 * block, so persisting these would write the same text twice and, worse,
				 * leave an entry that a crash could strand mid-response -- which
				 * `Context.assemble` would then replay to the model as a complete turn.
				 *
				 * Only scalars computed at push time are carried. `partial` is excluded
				 * deliberately: it is one object aikit rewrites in place and pushes by
				 * reference into a queue, so by the time this consumer reads a queued
				 * event the value has already moved on.
				 */
				case "text.start": {
					yield* events.publish(EventList.LLMTextStart, {
						...base,
						messageId: yield* requireAssistant(event.type, event.partial.messageId),
						partIndex: event.partIndex,
					});
					return;
				}
				case "text.delta": {
					yield* events.publish(EventList.LLMTextDelta, {
						...base,
						messageId: yield* requireAssistant(event.type, event.partial.messageId),
						partIndex: event.partIndex,
						delta: event.delta,
					});
					return;
				}
				case "text.end": {
					yield* events.publish(EventList.LLMTextEnd, {
						...base,
						messageId: yield* requireAssistant(event.type, event.partial.messageId),
						partIndex: event.partIndex,
						part: yield* textBlock(event.partial, event.partIndex),
					});
					return;
				}
				case "thinking.start": {
					yield* events.publish(EventList.LLMThinkingStart, {
						...base,
						messageId: yield* requireAssistant(event.type, event.partial.messageId),
						partIndex: event.partIndex,
					});
					return;
				}
				case "thinking.delta": {
					yield* events.publish(EventList.LLMThinkingDelta, {
						...base,
						messageId: yield* requireAssistant(event.type, event.partial.messageId),
						partIndex: event.partIndex,
						delta: event.delta,
					});
					return;
				}
				case "thinking.end": {
					yield* events.publish(EventList.LLMThinkingEnd, {
						...base,
						messageId: yield* requireAssistant(event.type, event.partial.messageId),
						partIndex: event.partIndex,
						part: yield* thinkingBlock(event.partial, event.partIndex),
					});
					return;
				}

				case "toolcall.start": {
					yield* events.publish(EventList.LLMToolCallStarted, {
						...base,
						messageId: yield* requireAssistant(event.type, event.partial.messageId),
						partIndex: event.partIndex,
					});
					return;
				}
				case "toolcall.delta": {
					yield* events.publish(EventList.LLMToolCallDelta, {
						...base,
						messageId: yield* requireAssistant(event.type, event.partial.messageId),
						partIndex: event.partIndex,
						delta: event.delta,
					});
					return;
				}
				case "toolcall.end": {
					yield* events.publish(EventList.LLMToolCallEnded, {
						...base,
						messageId: yield* requireAssistant(event.type, event.partial.messageId),
						partIndex: event.partIndex,
						callId: event.toolCall.callID,
						toolName: event.toolCall.name,
					});
					return;
				}

				/*
				 * The third block completion, and the only durable tool-call event. The
				 * call is written at its `partIndex` with canonical arguments and status
				 * `pending`: the model has finished asking, and nothing has run.
				 */
				case "toolcall.final": {
					const messageId = yield* requireAssistant(event.type, event.partial.messageId);
					const part = event.toolCall;
					if (part.status !== "pending") {
						return yield* fault(
							`"toolcall.final" for call "${part.callID}" arrived with status "${part.status}"; nothing has executed it yet`,
						);
					}
					yield* events.publish(EventList.LLMToolCallFinalized, {
						...base,
						messageId,
						partIndex: event.partIndex,
						callId: part.callID,
						toolName: part.name,
						part: { ...part },
					});
					return;
				}

				case "done": {
					yield* events.publish(EventList.LLMEnded, {
						...base,
						messageId: yield* startAssistant(event.message),
						reason: event.reason,
						message: event.message,
					});
					// Latched after the commit, so it records what is durable rather than
					// what was attempted.
					settled = { outcome: "ended", reason: event.reason, message: event.message };
					return;
				}

				/*
				 * aikit calls the payload `error`, but it is a complete assistant
				 * message: an abort arrives here carrying every block produced before
				 * the interrupt. Publishing it is what keeps that work, so the durable
				 * path for a failure is the same one a success takes.
				 */
				case "error": {
					yield* events.publish(EventList.LLMFailed, {
						...base,
						messageId: yield* startAssistant(event.error),
						reason: event.reason,
						message: event.error,
					});
					settled = { outcome: "failed", reason: event.reason, message: event.error };
					return;
				}
			}
		});

	const terminal: Effect.Effect<Terminal, Runner.LLMStreamError> = Effect.suspend(() =>
		settled === undefined
			? Effect.fail(fault("the provider stream ended without a terminal event"))
			: Effect.succeed(settled),
	);

	return { publish, failAssistant, terminal } satisfies Publisher;
});

export * as LLMEventPublisher from "./event.ts";
