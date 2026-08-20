import type { Message } from "@codeworksh/aikit";
import { Context, Effect, Layer, Option } from "effect";
import { SessionSchema } from "../session/schema.ts";
import { Session } from "../session/session.ts";
import { assemblePath, type Snapshot } from "./assemble.ts";
import { decodeMessage } from "./codec.ts";
import type { ContextDecodeError } from "./errors.ts";

export type ContextReadError = Session.SessionNotFoundError | ContextDecodeError;

export interface Interface {
	readonly assemble: (sessionId: SessionSchema.ID) => Effect.Effect<Snapshot, ContextReadError>;
	/**
	 * The newest assistant message on the session's active path.
	 *
	 * Deliberately not "the leaf, if it is a message". `leaf_entry_id` is the tree
	 * cursor -- where the next append attaches -- and every entry type moves it,
	 * so a model switch between turns leaves a `configChange` at the leaf with the
	 * assistant behind it. The question the recovery sweep asks is which assistant
	 * might still hold unresolved calls, and only this answers it.
	 *
	 * `None` when the path has no assistant yet.
	 *
	 * This belongs on Context because Context already owns decoding a
	 * `session_entry` into an aikit message; this is that same decode for one
	 * entry instead of a path.
	 */
	readonly latestAssistant: (
		sessionId: SessionSchema.ID,
	) => Effect.Effect<Option.Option<Message.AssistantMessage>, ContextReadError>;
	/**
	 * The session's unfinished assistant, decoded.
	 *
	 * Only the recovery sweep wants this, and only a hard kill produces one: a
	 * turn that reached any terminal settles its own draft. `None` is the answer
	 * on every drain that follows a clean turn.
	 */
	readonly currentDraft: (
		sessionId: SessionSchema.ID,
	) => Effect.Effect<Option.Option<Message.AssistantMessage>, ContextReadError>;
}

export class Service extends Context.Service<Service, Interface>()("@codeworksh/harness/context/context/Service") {}

export const layer = Layer.effect(
	Service,
	Effect.gen(function* () {
		const sessions = yield* Session.Service;
		const assemble = Effect.fn("Context.assemble")(function* (sessionId: SessionSchema.ID) {
			const found = yield* sessions.get(sessionId);
			if (Option.isNone(found)) return yield* new Session.SessionNotFoundError({ sessionId });
			return yield* assemblePath(sessionId, yield* sessions.path(sessionId));
		});
		const latestAssistant = Effect.fn("Context.latestAssistant")(function* (sessionId: SessionSchema.ID) {
			const found = yield* sessions.latestAssistant(sessionId);
			if (Option.isNone(found)) return Option.none<Message.AssistantMessage>();
			const message = yield* decodeMessage(found.value);
			// The row said `assistant`, so a decode that disagrees means the entry
			// type and its envelope have diverged -- corruption, not a case to handle.
			if (message.role !== "assistant") {
				return yield* Effect.die(`entry ${found.value.entry.id} is typed assistant but decodes as ${message.role}`);
			}
			return Option.some(message);
		});

		const currentDraft = Effect.fn("Context.currentDraft")(function* (sessionId: SessionSchema.ID) {
			const found = yield* sessions.latestDraft(sessionId);
			if (Option.isNone(found)) return Option.none<Message.AssistantMessage>();
			const message = yield* decodeMessage(found.value);
			if (message.role !== "assistant") {
				return yield* Effect.die(`entry ${found.value.entry.id} is a draft but decodes as ${message.role}`);
			}
			return Option.some(message);
		});

		return Service.of({ assemble, latestAssistant, currentDraft });
	}),
);

export * from "./assemble.ts";
export * from "./codec.ts";
export * from "./errors.ts";

export * as Context from "./context.ts";
