/*
 * @file Registers the durable-input projectors with the event service.
 *
 * Provides no service. Its only purpose is the registration side effect, which
 * must happen exactly once and before anything publishes -- `Layer` gives both,
 * because a layer is built once per graph, at startup.
 *
 * Registering from `SessionInput.make` instead would run once per consumer, and
 * a projector registered twice runs twice on one event: the second insert hits
 * ON CONFLICT, raises LifecycleConflict inside the commit, and rolls the event
 * back. Every admission would fail, blaming a conflict that does not exist.
 *
 * Nothing type-errors if this layer is left out of the graph. `admit` publishes
 * happily and writes no row, so the wiring is worth asserting in a test.
 *
 * Only durable definitions are registered here, and that is not a stylistic
 * choice: `Event.project` hands its callback to `commitDurableEvent`, which only
 * runs for events that have a transaction. Registering a projector for a live
 * definition -- any `session.llm.*` except the two terminals -- compiles, looks
 * wired, and never fires.
 *
 * Live events are not lost, they take the other path. `notify` publishes every
 * event to the typed and firehose pubsubs, so deltas and block boundaries are
 * already streaming; what does not exist yet is a `subscribe` on
 * `Event.Interface` for a consumer to read them from. That is the hook a future
 * streamer wants -- a stream it subscribes to, not a callback per event type --
 * so nothing needs registering here for it.
 */

import { Message } from "@codeworksh/aikit";
import { DateTime, Effect, Layer, Option } from "effect";
import { ContextCodec } from "../context/codec.ts";
import { Event } from "../event/event.ts";
import { EventList } from "../event/list.ts";
import { SessionInput } from "./input/input.ts";
import type { SessionMessageSchema } from "./message/schema.ts";
import type { SessionSchema } from "./schema.ts";
import { Session } from "./session.ts";

export const layer = Layer.effectDiscard(
	Effect.gen(function* () {
		const events = yield* Event.Service;
		const input = yield* SessionInput.make;
		const sessions = yield* Session.Service;

		/*
		 * Two fields that could disagree: the event names the message it is about,
		 * and the message names itself. The codec requires the entry id to equal
		 * the message id, so a disagreement would silently store the entry under
		 * one id while its envelope claims another.
		 */
		const sameMessage = (message: Message.AssistantMessage, messageId: SessionMessageSchema.ID) =>
			message.messageId === messageId
				? Effect.void
				: Effect.die(`assistant message id "${message.messageId}" does not match event message id "${messageId}"`);

		/**
		 * The response allocates its entry here, before it has produced anything.
		 *
		 * This is what lets a block completion project a part on its own: a part
		 * needs an owning entry, and the terminal is far too late to be the first
		 * writer. The envelope arrives with no parts and `stopReason: "aborted"`,
		 * so a crash mid-stream leaves an assistant that is already correctly
		 * marked and carries whatever blocks had completed -- rather than nothing
		 * at all.
		 */
		const createAssistant = Effect.fn("SessionProjector.createAssistant")(function* (input: {
			readonly sessionId: SessionSchema.ID;
			readonly messageId: SessionMessageSchema.ID;
			readonly message: Message.AssistantMessage;
			readonly seq: number;
			readonly metadata?: Record<string, string>;
		}) {
			yield* sameMessage(input.message, input.messageId);
			const encoded = yield* ContextCodec.encodeMessage(input.message).pipe(Effect.orDie);
			yield* sessions
				.append({
					id: input.messageId,
					sessionId: input.sessionId,
					seq: input.seq,
					...encoded,
					...(input.metadata === undefined ? {} : { metadata: input.metadata }),
				})
				.pipe(Effect.orDie);
		});

		/**
		 * The terminal promotes the entry to its real reason and charges usage.
		 *
		 * Ended and failed take the same path deliberately. aikit names the failed
		 * payload `error`, but it is a complete assistant message carrying whatever
		 * was generated before the failure -- its `stopReason` and `errorMessage`
		 * are what record that it failed, not its absence from the conversation.
		 *
		 * The terminal's parts are authoritative, so they are written over the
		 * slots the block completions filled. In Phase 1 that is a no-op by
		 * construction: tools execute strictly after the terminal, so every tool
		 * part is still `pending` and matches what aikit carries. It stops being a
		 * no-op in Phase 2.
		 */
		const finalizeAssistant = Effect.fn("SessionProjector.finalizeAssistant")(function* (input: {
			readonly sessionId: SessionSchema.ID;
			readonly messageId: SessionMessageSchema.ID;
			readonly message: Message.AssistantMessage;
			readonly reason: Message.AssistantMessage["stopReason"];
		}) {
			yield* sameMessage(input.message, input.messageId);
			/*
			 * The same class of check as the id above, for the same reason. The event
			 * says how the stream ended and the message says how it ended; a
			 * disagreement means the publisher and aikit no longer describe the same
			 * response, and the envelope about to be written is the one that would be
			 * believed forever after.
			 */
			if (input.message.stopReason !== input.reason) {
				return yield* Effect.die(
					`assistant stopReason "${input.message.stopReason}" does not match terminal reason "${input.reason}"`,
				);
			}
			const encoded = yield* ContextCodec.encodeMessage(input.message).pipe(Effect.orDie);
			yield* sessions
				.finalizeAssistant({
					sessionId: input.sessionId,
					entryId: input.messageId,
					data: encoded.data,
					parts: encoded.parts,
				})
				.pipe(Effect.orDie);
		});

		/**
		 * One completed block, written into the slot aikit assigned it.
		 *
		 * A `toolCall` block goes through the same path as text and thinking, which
		 * is the point: `encodePart` derives the promoted `status` / `callId` /
		 * `toolName` columns from the part JSON, so the indexed copies cannot drift
		 * from the authoritative value the way a restated literal could.
		 */
		const writeBlock = Effect.fn("SessionProjector.writeBlock")(function* (input: {
			readonly sessionId: SessionSchema.ID;
			readonly messageId: SessionMessageSchema.ID;
			readonly partIndex: number;
			readonly part: Message.TextContent | Message.ThinkingContent | Message.ToolCallPendingPart;
		}) {
			const encoded = yield* ContextCodec.encodePart({
				messageId: input.messageId,
				role: "assistant",
				part: input.part,
			});
			yield* sessions.upsertPart({
				sessionId: input.sessionId,
				entryId: input.messageId,
				partIndex: input.partIndex,
				...encoded,
			});
		}, Effect.orDie);

		yield* events.project(EventList.PromptAdmitted, (event) =>
			Effect.gen(function* () {
				// The sequence is assigned by the commit itself, so a durable event
				// arriving without one means it never went through the transaction.
				if (event.durable === undefined)
					return yield* Effect.die("PromptAdmitted is missing its aggregate sequence");
				yield* input.projectAdmitted({
					admittedSeq: event.durable.seq,
					id: event.data.messageId,
					sessionId: event.data.sessionId,
					prompt: event.data.prompt,
					delivery: event.data.delivery,
					timeCreated: event.data.timestamp,
				});
			}),
		);

		// Publishing `Prompted` is what promotes an input, so this registration is
		// what makes `promoteSteers` / `promoteFollowUp` take effect at all.
		yield* events.project(EventList.Prompted, (event) =>
			Effect.gen(function* () {
				if (event.durable === undefined) return yield* Effect.die("Prompted is missing its aggregate sequence");
				yield* input.projectPrompted({
					promotedSeq: event.durable.seq,
					id: event.data.messageId,
					sessionId: event.data.sessionId,
					prompt: event.data.prompt,
					delivery: event.data.delivery,
					timeCreated: event.data.timestamp,
				});
				// Promotion is what puts a prompt into the conversation, so the append
				// belongs in this commit: "promoted but absent from history" is then
				// unrepresentable rather than a window someone has to reconcile.
				// Materialize the canonical conversation message now. Context later
				// rehydrates this envelope + parts; aikit performs only the target-model
				// conversion when Loop calls the provider.
				const encoded = yield* ContextCodec.encodeMessage(
					Message.createUserMessage({
						messageId: event.data.messageId,
						role: "user",
						time: { created: DateTime.toEpochMillis(event.data.timestamp) },
						parts: [{ type: "text", text: event.data.prompt.text }],
					}),
				).pipe(Effect.orDie);
				yield* sessions
					.append({
						id: event.data.messageId,
						sessionId: event.data.sessionId,
						seq: event.durable.seq,
						...encoded,
						// Rides the in-memory payload; the event row never carried it, so
						// the entry is the only place it can outlive the publish.
						...(event.metadata === undefined ? {} : { metadata: event.metadata }),
					})
					.pipe(Effect.orDie);
			}),
		);

		yield* events.project(EventList.LLMStarted, (event) =>
			Effect.gen(function* () {
				if (event.durable === undefined) return yield* Effect.die("LLMStarted is missing its aggregate sequence");
				yield* createAssistant({
					sessionId: event.data.sessionId,
					messageId: event.data.messageId,
					message: event.data.message,
					seq: event.durable.seq,
					...(event.metadata === undefined ? {} : { metadata: event.metadata }),
				});
			}),
		);

		yield* events.project(EventList.LLMTextEnd, (event) =>
			writeBlock({
				sessionId: event.data.sessionId,
				messageId: event.data.messageId,
				partIndex: event.data.partIndex,
				part: event.data.part,
			}),
		);

		yield* events.project(EventList.LLMThinkingEnd, (event) =>
			writeBlock({
				sessionId: event.data.sessionId,
				messageId: event.data.messageId,
				partIndex: event.data.partIndex,
				part: event.data.part,
			}),
		);

		yield* events.project(EventList.LLMToolCallFinalized, (event) =>
			writeBlock({
				sessionId: event.data.sessionId,
				messageId: event.data.messageId,
				partIndex: event.data.partIndex,
				part: event.data.part,
			}),
		);

		yield* events.project(EventList.LLMEnded, (event) =>
			finalizeAssistant({
				sessionId: event.data.sessionId,
				messageId: event.data.messageId,
				message: event.data.message,
				reason: event.data.reason,
			}),
		);

		yield* events.project(EventList.LLMFailed, (event) =>
			finalizeAssistant({
				sessionId: event.data.sessionId,
				messageId: event.data.messageId,
				message: event.data.message,
				reason: event.data.reason,
			}),
		);

		/*
		 * `running` is committed before the handler runs and the terminal after it
		 * returns, so the two transitions are the same write at the same address
		 * with a different part. The projector does not distinguish them because
		 * the part JSON already carries its own status.
		 */
		yield* events.project(EventList.ToolExecutionStarted, (event) =>
			Effect.gen(function* () {
				const data = yield* ContextCodec.encodePart({
					messageId: event.data.messageId,
					role: "assistant",
					part: event.data.part,
				});
				yield* sessions.beginToolCall({
					sessionId: event.data.sessionId,
					entryId: event.data.messageId,
					callId: event.data.callId,
					toolName: event.data.toolName,
					data: data.data,
				});
			}).pipe(Effect.orDie),
		);

		yield* events.project(EventList.ToolExecutionEnded, (event) =>
			Effect.gen(function* () {
				const data = yield* ContextCodec.encodePart({
					messageId: event.data.messageId,
					role: "assistant",
					part: event.data.part,
				});
				yield* sessions.settleToolCall({
					sessionId: event.data.sessionId,
					entryId: event.data.messageId,
					callId: event.data.callId,
					toolName: event.data.toolName,
					status: event.data.part.status,
					data: data.data,
				});
			}).pipe(Effect.orDie),
		);

		/*
		 * Recovery settlement. No executor outcome exists, so the terminal part is
		 * synthesized here from the stored call plus the error -- which is why this
		 * is not `ToolExecutionEnded`: that event carries a part `Executor.handle`
		 * built, and hand-constructing one for a call that never ran is exactly the
		 * ad-hoc construction the split exists to prevent.
		 *
		 * The stored call is read rather than carried on the event because the
		 * event's whole claim is that nothing produced a part for it. Its identity
		 * and arguments already live in the row.
		 */
		yield* events.project(EventList.ToolFailed, (event) =>
			Effect.gen(function* () {
				const rows = yield* sessions.toolCalls(event.data.messageId);
				const row = rows.find((candidate) => Option.getOrUndefined(candidate.callId) === event.data.callId);
				if (row === undefined) {
					return yield* Effect.die(
						`ToolFailed names call "${event.data.callId}", which entry ${event.data.messageId} does not have`,
					);
				}
				const base = yield* ContextCodec.decodeToolCallBase(row);
				/*
				 * A tool that was interrupted mid-run usually produced something first,
				 * and that output is more useful to the model than the reason it
				 * stopped. Keep both: the partial when there is one, the reason always.
				 */
				const reported = event.data.partial?.partial?.content ?? [];
				const settled: Message.ToolCallTerminalPart = {
					...base,
					status: event.data.status,
					result: {
						content: [...reported, { type: "text", text: event.data.error }],
						isError: true,
					},
				};
				const encoded = yield* ContextCodec.encodePart({
					messageId: event.data.messageId,
					role: "assistant",
					part: settled,
				});
				yield* sessions.settleToolCall({
					sessionId: event.data.sessionId,
					entryId: event.data.messageId,
					callId: event.data.callId,
					toolName: event.data.toolName,
					status: event.data.status,
					data: encoded.data,
				});
			}).pipe(Effect.orDie),
		);
	}),
);

export * as SessionProjector from "./projector.ts";
