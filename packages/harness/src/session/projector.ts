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
import { DateTime, Effect, Layer, Option, Schema } from "effect";
import { ContextCodec } from "../context/codec.ts";
import { Event } from "../event/event.ts";
import { EventList } from "../event/list.ts";
import { SessionInput } from "./input/input.ts";
import { SessionMessageSchema } from "./message/schema.ts";
import type { SessionSchema } from "./schema.ts";
import { Session } from "./session.ts";

/**
 * How a terminal disposes of the calls it will not run, by what it reported.
 *
 * Absent means "leave them pending" — `toolUse` and `stop` both hand their calls
 * to the loop. `length` is the interesting one: the output was cut off
 * mid-generation, so arguments that still parse and validate may mean something
 * else entirely, and none of them is safe to execute.
 */
const unrunnable: Partial<
	Record<
		Message.AssistantMessage["stopReason"],
		{ readonly status: "skipped" | "aborted" | "error"; readonly error: string }
	>
> = {
	/*
	 * The wording carries the whole signal. `stopReason` is never sent to a
	 * provider — only text, thinking, and tool-call parts are — so the model is
	 * never told its last response was truncated. This tool result is the only
	 * place it can learn that, and the only place it can be told what to do about
	 * it. Pi's phrasing, for the same reason.
	 */
	length: {
		status: "skipped",
		error: "Tool call was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.",
	},
	// Aborted is something we did; error is something the provider did. The
	// distinction is what a reader of the settled part gets to see.
	aborted: { status: "aborted", error: "Tool call was not executed: the response was aborted." },
	error: { status: "error", error: "Tool call was not executed: the response failed." },
};

/** The synthetic envelope a halt writes, as the assembler will read it back. */
const encodeHaltData = Schema.encodeEffect(
	Schema.fromJsonString(
		Schema.Struct({ messageId: Schema.String, customType: Schema.String, display: Schema.Boolean }),
	),
);

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
					// The one entry type that is not complete when it is written. It
					// stays `draft` until a terminal settles it, or until the recovery
					// sweep abandons it.
					state: "draft",
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
		/**
		 * Terminal parts for calls that never ran, written from the stored call.
		 *
		 * No executor outcome exists for any of these, so the part is synthesized
		 * here rather than handed over — the same construction `ToolFailed` uses,
		 * which is why both go through this one function.
		 */
		const settleOpenCalls = Effect.fn("SessionProjector.settleOpenCalls")(function* (input: {
			readonly sessionId: SessionSchema.ID;
			readonly messageId: SessionMessageSchema.ID;
			readonly status: "skipped" | "aborted" | "error";
			readonly error: string;
			/** Progress the harness saw before the call stopped, when it saw any. */
			readonly partial?: Message.ToolCallRunningPart;
			/** Restrict to one call; omitted settles every open call on the entry. */
			readonly callId?: string;
		}) {
			const rows = yield* sessions.toolCalls(input.messageId);
			for (const row of rows) {
				const open = Option.getOrUndefined(row.status);
				if (open !== "pending" && open !== "running") continue;
				const callId = Option.getOrUndefined(row.callId);
				const toolName = Option.getOrUndefined(row.toolName);
				if (callId === undefined || toolName === undefined) continue;
				if (input.callId !== undefined && callId !== input.callId) continue;

				const base = yield* ContextCodec.decodeToolCallBase(row);
				// A tool interrupted mid-run usually produced something first, and that
				// is more useful to the model than the reason it stopped. Keep both.
				const reported = input.partial?.partial?.content ?? [];
				const settled: Message.ToolCallTerminalPart = {
					...base,
					status: input.status,
					result: { content: [...reported, { type: "text", text: input.error }], isError: true },
				};
				const encoded = yield* ContextCodec.encodePart({
					messageId: input.messageId,
					role: "assistant",
					part: settled,
				});
				yield* sessions.settleToolCall({
					sessionId: input.sessionId,
					entryId: input.messageId,
					callId,
					toolName,
					status: input.status,
					data: encoded.data,
				});
			}
		});

		const finalizeAssistant = Effect.fn("SessionProjector.finalizeAssistant")(function* (input: {
			readonly sessionId: SessionSchema.ID;
			readonly messageId: SessionMessageSchema.ID;
			readonly message: Message.AssistantMessage;
			readonly reason: Message.AssistantMessage["stopReason"];
			readonly state: Session.EntryState;
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
					state: input.state,
					data: encoded.data,
					parts: encoded.parts,
				})
				.pipe(Effect.orDie);

			/*
			 * Settle the calls this terminal will never run, here, in the transaction
			 * that settles the entry.
			 *
			 * The loop used to do this afterwards, publishing one event per call, so
			 * a crash between the terminal and its own policy left an entry marked
			 * settled with calls still open. Deciding it once, where the reason is
			 * already in hand, makes the whole turn's settlement atomic — and takes
			 * the reason switch out of the loop, which now only has to run whatever
			 * is still pending.
			 *
			 * `toolUse` and `stop` settle nothing: their calls are the loop's to run.
			 * A `stop` carrying calls is a provider contradiction, and the concrete
			 * tool lifecycle is the authoritative one.
			 */
			const disposition = unrunnable[input.reason];
			if (disposition !== undefined) {
				yield* settleOpenCalls({
					sessionId: input.sessionId,
					messageId: input.messageId,
					...disposition,
				});
			}
		});

		/**
		 * Where a terminal leaves the entry.
		 *
		 * A turn is one assistant response *plus its tool calls and results*, so a
		 * response that hands calls to the loop has not finished the turn — the
		 * entry stays `draft` and `TurnEnded` commits it once every call is
		 * terminal. Anything else is settled here: a failure never runs its calls,
		 * and a truncated response has just had them settled above.
		 *
		 * Calling a `toolUse` entry `committed` was the earlier behaviour and it was
		 * a false claim — it also made the recovery sweep, which looks for drafts,
		 * blind to a kill landing between the terminal and the batch.
		 */
		const settledState = (
			message: Message.AssistantMessage,
			reason: Message.AssistantMessage["stopReason"],
		): Session.EntryState => {
			if (reason === "aborted") return "aborted";
			if (reason === "error") return "error";
			if (unrunnable[reason] !== undefined) return "committed";
			return message.parts.some((part) => part.type === "toolCall") ? "draft" : "committed";
		};

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
			// `stop`, `length`, and `toolUse` all settle the entry the same way; how
			// it ended stays in the envelope's `stopReason`.
			finalizeAssistant({
				sessionId: event.data.sessionId,
				messageId: event.data.messageId,
				message: event.data.message,
				reason: event.data.reason,
				state: settledState(event.data.message, event.data.reason),
			}).pipe(Effect.orDie),
		);

		yield* events.project(EventList.LLMFailed, (event) =>
			finalizeAssistant({
				sessionId: event.data.sessionId,
				messageId: event.data.messageId,
				message: event.data.message,
				reason: event.data.reason,
				state: settledState(event.data.message, event.data.reason),
			}).pipe(Effect.orDie),
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
		/*
		 * The turn is over: the response is written and every call it made is
		 * terminal. This is the only thing that commits an entry a terminal left
		 * draft, which is what makes "committed" mean the whole turn rather than
		 * just the provider's half of it.
		 *
		 * Nothing to do for a turn whose terminal already settled the entry — a
		 * failure, or a response with no calls — so an already-settled entry is not
		 * an error. An entry that does not exist at all is.
		 */
		yield* events.project(EventList.TurnEnded, (event) =>
			sessions
				.closeDraft({
					sessionId: event.data.sessionId,
					entryId: event.data.messageId,
					state: "committed",
				})
				// `false` is ordinary — a terminal that left no tool work settled the
				// entry itself. A *missing* entry is not, and dies rather than being
				// swallowed by the same ignore.
				.pipe(Effect.asVoid, Effect.orDie),
		);

		/*
		 * Tell the model, not just the log.
		 *
		 * A synthetic entry decodes into a user message and lands in the next
		 * request's context, which is the only way the model can learn the exchange
		 * was cut short — nothing about a halt reaches a provider otherwise. Same
		 * reasoning as the truncated-tool-call wording above.
		 */
		yield* events.project(EventList.ExchangeHalted, (event) =>
			Effect.gen(function* () {
				if (event.durable === undefined) return yield* Effect.die("ExchangeHalted is missing its sequence");
				const text = `Stopped after ${event.data.continuations} consecutive turns without user input. Ask again to continue.`;
				const data = yield* encodeHaltData({
					messageId: event.data.entryId,
					customType: "exchange-halted",
					display: true,
				});
				const part = yield* ContextCodec.encodePart({
					messageId: event.data.entryId,
					role: "user",
					part: { type: "text", text },
				});
				yield* sessions.append({
					id: event.data.entryId,
					sessionId: event.data.sessionId,
					seq: event.durable.seq,
					type: "synthetic",
					data,
					parts: [part],
				});
			}).pipe(Effect.orDie),
		);

		/*
		 * A turn that stopped without finishing: settle what it left open, then
		 * close the entry the way the turn would have.
		 *
		 * Which way is the same question the recovery sweep answers, and `state` is
		 * what makes it answerable — a draft still carrying `stopReason: "aborted"`
		 * never received a terminal, because an aborted terminal settles the entry
		 * rather than leaving it draft. Anything else means the response completed
		 * and only its tools were cut short.
		 *
		 * No draft means the failure happened before an entry existed, or a terminal
		 * already settled it. Both are ordinary, and both are nothing to do.
		 */
		yield* events.project(EventList.TurnFailed, (event) =>
			Effect.gen(function* () {
				const draft = yield* sessions.latestDraft(event.data.sessionId);
				if (Option.isNone(draft)) return;
				const messageId = SessionMessageSchema.ID.from(draft.value.entry.id);
				yield* settleOpenCalls({
					sessionId: event.data.sessionId,
					messageId,
					status: "aborted",
					error: `Tool call was not executed: the turn failed (${event.data.reason}).`,
				});
				const envelope = yield* ContextCodec.decodeMessage(draft.value);
				yield* sessions.closeDraft({
					sessionId: event.data.sessionId,
					entryId: messageId,
					state: envelope.role === "assistant" && envelope.stopReason !== "aborted" ? "committed" : "aborted",
				});
			}).pipe(Effect.orDie),
		);

		yield* events.project(EventList.ToolFailed, (event) =>
			settleOpenCalls({
				sessionId: event.data.sessionId,
				messageId: event.data.messageId,
				callId: event.data.callId,
				status: event.data.status,
				error: event.data.error,
				...(event.data.partial === undefined ? {} : { partial: event.data.partial }),
			}).pipe(Effect.orDie),
		);
	}),
);

export * as SessionProjector from "./projector.ts";
