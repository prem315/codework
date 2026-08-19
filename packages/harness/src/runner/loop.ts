/*
 * @file Effect-native durable session loop.
 *
 * The loop owns delivery and execution order. It promotes durable input, asks
 * Context for the current storage-faithful aikit messages, and sends the aikit
 * provider events through their request-scoped publisher. It never reads or
 * writes Session.Service directly.
 *
 * Three levels, and the split is load-bearing:
 *
 *   run             lanes and the two loops. Promotes nothing, publishes
 *                   nothing, captures State once per exchange.
 *   runTurn         admits the lane's input, then brackets one turn with
 *                   TurnStarted / TurnEnded. Also the retry seam -- a turn that
 *                   overflows the context window is re-attempted here, after
 *                   compaction, without re-opening the bracket.
 *   runTurnAttempt  the turn body: assemble Context, resolve the effective
 *                   model, one provider step. Returns whether the turn's own
 *                   outcome requires another one.
 *
 * A turn is one assistant response plus any tool calls it makes, so the bracket
 * belongs around `runTurnAttempt` and not around the continuation loop: every
 * return from `runTurn` is a completed turn. Bracketing the loop instead would
 * give N starts and one end for an N-round tool chain.
 */

import type { Message } from "@codeworksh/aikit";
import { DateTime, Effect, Exit, Layer, Option } from "effect";
import { ContextCodec } from "../context/codec.ts";
import { Context } from "../context/context.ts";
import { Event } from "../event/event.ts";
import { EventList } from "../event/list.ts";
import { SessionInput } from "../session/input/input.ts";
import { SessionMessageSchema } from "../session/message/schema.ts";
import type { SessionSchema } from "../session/schema.ts";
import { Session } from "../session/session.ts";
import { State } from "../state/state.ts";
import { LLMEventPublisher } from "./event.ts";
import { LLM } from "./llm.ts";
import { Runner } from "./run.ts";

export interface Options {
	/** Deterministic test seam; production resolves and streams through aikit. */
	readonly request?: LLM.Request;
}

/**
 * What starts a turn.
 *
 * The two halves are genuinely independent, and conflating them is a live bug:
 * a tool continuation admits any steer that arrived mid-turn, but it happens
 * whether or not one did. Only a turn the lane itself triggered is cancelled
 * when that lane turns out to have nothing.
 */
interface Admission {
	/** Lane whose pending input this turn promotes, if any. */
	readonly lane: "steer" | "followUp" | undefined;
	/** When true, the turn happens only if the lane actually promoted something. */
	readonly requires: boolean;
}

/**
 * The aikit bag for one request: State's pinned options plus the two fields
 * State withholds.
 *
 * `reasoning` is dropped when thinking is `"off"`. aikit types it as
 * `Model.ActiveThinkingLevel`, which has no `off` member, so "no thinking" is
 * expressed by absence rather than by a value.
 */
const requestOptions = (
	snapshot: State.Snapshot,
	thinkingLevel: State.Snapshot["thinkingLevel"],
): LLM.RequestOptions => ({
	...snapshot.request,
	sessionId: snapshot.sessionId,
	...(thinkingLevel === "off" ? {} : { reasoning: thinkingLevel }),
});

export const layer = (options: Options = {}) =>
	Layer.effect(
		Runner.Service,
		Effect.gen(function* () {
			const context = yield* Context.Service;
			const state = yield* State.Service;
			const events = yield* Event.Service;
			const sessions = yield* Session.Service;
			const inputs = yield* SessionInput.make;
			const requestLLM = options.request ?? LLM.run;

			/**
			 * Execute one committed call.
			 *
			 * `running` is committed before the handler is invoked and never after,
			 * so the durable record always says "a side effect may have happened
			 * from here" rather than claiming a call ran that could not have.
			 *
			 * The handler swallows almost everything by design: an unknown tool name
			 * and invalid arguments come back as model-visible terminal errors, and
			 * an interruption comes back as an `aborted` outcome carrying whatever
			 * progress the tool had reported. All four settle through one event.
			 */
			/**
			 * One unresolved call, as the two facts settling it needs.
			 *
			 * Taken from the promoted columns rather than the part JSON: the shutdown
			 * path runs while the fiber is being torn down, and a decode there is work
			 * done at the worst possible moment.
			 */
			interface Unresolved {
				readonly partIndex: number;
				readonly callId: string;
				readonly toolName: string;
				readonly status: "pending" | "running";
			}

			/**
			 * Why a turn stopped with calls still open. Each answers the settlement
			 * question differently, and none of them re-executes anything.
			 */
			type Ending = "interrupted" | "truncated" | "aborted" | "failed";

			/**
			 * How one open call settles, given what ended the turn.
			 *
			 * The interrupted case is the only one that reads the prior status,
			 * because it is the only one where the prior status carries information:
			 * a live process watched these calls, so it knows which had started. The
			 * others describe something that happened to the whole response.
			 */
			const settlement = (ending: Ending, prior: "pending" | "running") => {
				switch (ending) {
					case "interrupted":
						return prior === "pending"
							? ({
									status: "skipped",
									error: "Tool call was never executed: the session ended before it started.",
								} as const)
							: ({
									status: "aborted",
									error: "Tool execution was interrupted: the session ended while it was running.",
								} as const);
					case "truncated":
						return {
							status: "skipped",
							error: "Tool call was not executed: the response hit its output limit, so the arguments may be incomplete.",
						} as const;
					case "aborted":
						return { status: "aborted", error: "Tool call was not executed: the response was aborted." } as const;
					case "failed":
						return { status: "error", error: "Tool call was not executed: the response failed." } as const;
				}
			};

			/**
			 * Settle calls that never produced an executor outcome.
			 *
			 * Shared by the two places that need it — the drain-start sweep and
			 * shutdown — because both answer the same question, and the prior status
			 * means the same thing in each: `pending` proves the handler never
			 * started, `running` admits a side effect may already have happened.
			 * Neither is ever re-executed.
			 */
			const failCalls = Effect.fn("Loop.failCalls")(function* (input: {
				readonly sessionId: SessionSchema.ID;
				readonly messageId: SessionMessageSchema.ID;
				readonly calls: ReadonlyArray<Unresolved>;
				readonly ending: Ending;
				/** Last progress per call, when this process was the one watching. */
				readonly partials?: ReadonlyMap<string, Message.ToolCallRunningPart>;
			}) {
				for (const call of input.calls) {
					const { status, error } = settlement(input.ending, call.status);
					const partial = input.partials?.get(call.callId);
					yield* events.publish(EventList.ToolFailed, {
						sessionId: input.sessionId,
						timestamp: yield* DateTime.now,
						messageId: input.messageId,
						partIndex: call.partIndex,
						callId: call.callId,
						toolName: call.toolName,
						status,
						error,
						...(partial === undefined ? {} : { partial }),
					});
				}
			});

			/** Settle every open call on one entry. Returns how many there were. */
			const closeOpenCalls = Effect.fn("Loop.closeOpenCalls")(function* (input: {
				readonly sessionId: SessionSchema.ID;
				readonly messageId: SessionMessageSchema.ID;
				readonly ending: Ending;
			}) {
				const calls = yield* unresolvedCalls(input.messageId);
				yield* failCalls({ ...input, calls });
				return calls.length;
			});

			/** Unresolved calls on one entry, straight from the promoted columns. */
			const unresolvedCalls = Effect.fn("Loop.unresolvedCalls")(function* (messageId: SessionMessageSchema.ID) {
				const rows = yield* sessions.toolCalls(messageId);
				return rows.flatMap((row): ReadonlyArray<Unresolved> => {
					const status = Option.getOrUndefined(row.status);
					if (status !== "pending" && status !== "running") return [];
					const callId = Option.getOrUndefined(row.callId);
					const toolName = Option.getOrUndefined(row.toolName);
					return callId === undefined || toolName === undefined
						? []
						: [{ partIndex: row.partIndex, callId, toolName, status }];
				});
			});

			const runCall = Effect.fn("Loop.runCall")(function* (input: {
				readonly snapshot: State.Snapshot;
				readonly messageId: SessionMessageSchema.ID;
				readonly partIndex: number;
				readonly call: Message.ToolCallPendingPart;
				/** Written as progress arrives, read by the shutdown finalizer. */
				readonly partials: Map<string, Message.ToolCallRunningPart>;
			}) {
				const { snapshot, messageId, partIndex, call } = input;
				const identity = {
					sessionId: snapshot.sessionId,
					messageId,
					partIndex,
					callId: call.callID,
					toolName: call.name,
				};
				const { status: _pending, ...execution } = call;

				yield* events.publish(EventList.ToolExecutionStarted, {
					...identity,
					timestamp: yield* DateTime.now,
					part: { ...execution, status: "running" },
				});

				/*
				 * The handler stays interruptible; its terminal commit does not.
				 *
				 * `Executor.handle` already turns an interrupted handler into an
				 * `aborted` outcome carrying the last progress partial, so the work is
				 * done by the time we get here — but without the mask that commit
				 * races the same interrupt that produced it, and loses. One chance to
				 * write it down is the whole point.
				 */
				const handled = snapshot.tools.handle(execution, {
					onProgress: (event) =>
						DateTime.now.pipe(
							Effect.tap(() => Effect.sync(() => input.partials.set(call.callID, event.toolCall))),
							Effect.flatMap((timestamp) =>
								events.publish(EventList.ToolExecutionUpdated, {
									...identity,
									timestamp,
									part: event.toolCall,
								}),
							),
						),
				});

				/*
				 * Effect unwinds an external interrupt straight through `Effect.exit`,
				 * so a call killed mid-flight never reaches this commit however it is
				 * wrapped — the only code that runs during unwinding is a finalizer.
				 * That is where an interrupted call gets settled, in `runToolBatch`,
				 * carrying the progress this loop recorded on the way.
				 *
				 * The mask still earns its place, for a narrower window: an interrupt
				 * arriving after the handler returned and before its outcome is written
				 * down.
				 */
				yield* Effect.uninterruptibleMask((restore) =>
					Effect.gen(function* () {
						// `restore` keeps the handler itself interruptible; without it a
						// blocking tool could not be stopped at all.
						const outcome = yield* restore(handled);
						yield* events.publish(EventList.ToolExecutionEnded, {
							...identity,
							timestamp: yield* DateTime.now,
							part: outcome,
						});
					}),
				);
			});

			/**
			 * The turn's tool batch: re-read, then schedule.
			 *
			 * Read from the committed projection rather than from the terminal aikit
			 * message held in memory. It is the same data, but reading it from
			 * storage is what makes execution resumable and keeps one rule — execute
			 * what is durable — true in every phase.
			 *
			 * Every call is known before any of them starts, so the whole difference
			 * between the two modes is a concurrency argument. Parts keep source
			 * order regardless, because each settles at its own `(entryId, callId)`.
			 */
			const runToolBatch = Effect.fn("Loop.runToolBatch")(function* (
				snapshot: State.Snapshot,
				messageId: SessionMessageSchema.ID,
			) {
				/*
				 * Only what is still open. `toolCalls` returns every call on the entry,
				 * and `decodeToolCall` accepts only `pending` — so an unfiltered read
				 * would fail on any entry whose calls had already settled. Nothing does
				 * that today; the retry seam this file's header promises for `runTurn`
				 * would, and would fail with a decode error rather than a no-op.
				 */
				const rows = (yield* sessions.toolCalls(messageId)).filter(
					(row) => Option.getOrUndefined(row.status) === "pending",
				);
				const calls = yield* Effect.forEach(rows, (row) =>
					ContextCodec.decodeToolCall(row).pipe(Effect.map((call) => ({ partIndex: row.partIndex, call }))),
				);
				if (calls.length === 0) return false;

				/*
				 * Interruption stops admission by construction: `forEach` starts no
				 * further call once the fiber is interrupted, and in sequential mode
				 * that is most of the batch. Those calls are still `pending` rows, so
				 * the shutdown path settles them here rather than leaving them for the
				 * next drain — the turn closes where it was interrupted.
				 *
				 * The finalizer re-reads instead of tracking what it started: a call
				 * whose own terminal committed under the mask above is no longer
				 * `pending`, so re-reading cannot double-settle it.
				 */
				const partials = new Map<string, Message.ToolCallRunningPart>();
				yield* Effect.forEach(
					calls,
					({ partIndex, call }) => runCall({ snapshot, messageId, partIndex, call, partials }),
					{ concurrency: snapshot.toolExecution === "sequential" ? 1 : "unbounded", discard: true },
				).pipe(
					/*
					 * On exit rather than on interrupt. `forEach` interrupts its siblings
					 * when one child *fails* too, and that path leaves the same
					 * half-settled rows an interrupt does — settling only one of the two
					 * would strand the other set for the next drain, which is exactly
					 * what closing the turn in place is supposed to prevent.
					 */
					Effect.onExit((exit) =>
						Exit.isSuccess(exit)
							? Effect.void
							: unresolvedCalls(messageId).pipe(
									Effect.flatMap((unresolved) =>
										failCalls({
											sessionId: snapshot.sessionId,
											messageId,
											calls: unresolved,
											ending: "interrupted",
											partials,
										}),
									),
									Effect.orDie,
								),
					),
				);
				return true;
			});

			/**
			 * Settle whatever a previous drain left open, before asking anything new.
			 *
			 * Only a hard kill can leave work here. A graceful interrupt closes its own
			 * turn in place: aikit's terminal `error` carries the whole message, so
			 * `LLMFailed` finalizes the envelope and the batch's own `aborted` outcomes
			 * settle the calls. A process that dies mid-stream emits none of that, and
			 * all that survives is the assistant entry created at `LLMStarted` plus
			 * whatever blocks completed before the kill.
			 *
			 * The newest assistant, not the whole path: a turn only completes with
			 * `TurnEnded`, which is published after every call it made has settled, so
			 * an unresolved call can only be on that one. `Session.unsettled` would be
			 * wrong here for a second reason — it spans branches, including abandoned
			 * ones.
			 *
			 * And not the leaf either. The leaf is the tree cursor, so a model switch
			 * between turns puts a `configChange` there with the assistant behind it;
			 * asking the leaf would miss exactly the case this sweep exists for.
			 *
			 * Settled, never re-executed. The process may have died after Bash wrote to
			 * the filesystem but before the terminal committed, so replay would give
			 * at-least-once execution for a mutating tool.
			 */
			const failInterruptedTools = Effect.fn("Loop.failInterruptedTools")(function* (sessionId: SessionSchema.ID) {
				const found = yield* context.latestAssistant(sessionId);
				if (Option.isNone(found)) return;
				const messageId = SessionMessageSchema.ID.from(found.value.messageId);
				yield* failCalls({
					sessionId,
					messageId,
					ending: "interrupted",
					calls: yield* unresolvedCalls(messageId),
				});
			});

			const runTurnAttempt = Effect.fn("Loop.runTurnAttempt")(function* (snapshot: State.Snapshot) {
				const sessionId = snapshot.sessionId;

				// This is our toLLMMessages boundary: Context reads native durable
				// history and returns aikit's canonical Message.Context values. On a
				// continuation it reads exactly the context the previous TurnEnded
				// closed -- the assistant message with its tool-call parts settled.
				const assembled = yield* context.assemble(sessionId);

				// State holds the agent's configured defaults; a session may have moved
				// off them mid-conversation, and that change is durable in its history.
				const provider = assembled.config.model?.providerId ?? snapshot.provider;
				const model = assembled.config.model?.modelId ?? snapshot.model;
				const thinkingLevel = assembled.config.thinkingLevel ?? snapshot.thinkingLevel;

				const request: Message.Context = {
					systemPrompt: snapshot.systemPrompt,
					messages: [...assembled.messages],
					tools: [...snapshot.tools.wire],
				};

				const publisher = yield* LLMEventPublisher.make({ sessionId, provider, model }).pipe(
					Effect.provideService(Event.Service, events),
				);
				const terminal = yield* requestLLM({
					sessionId,
					context: request,
					provider,
					model,
					options: requestOptions(snapshot, thinkingLevel),
					publisher,
				});
				yield* Effect.logInfo("loop: aikit response settled").pipe(
					Effect.annotateLogs({
						sessionId,
						messageId: terminal.message.messageId,
						outcome: terminal.outcome,
						reason: terminal.reason,
					}),
				);
				const messageId = SessionMessageSchema.ID.from(terminal.message.messageId);

				/*
				 * A response that failed still leaves its calls open, and leaving them
				 * for the next drain would mean the model is told nothing about them
				 * until it asks again. Settle here, saying which happened: aborted is
				 * something we did, error is something the provider did.
				 */
				if (terminal.outcome === "failed") {
					yield* closeOpenCalls({
						sessionId,
						messageId,
						ending: terminal.reason === "aborted" ? "aborted" : "failed",
					});
					return yield* new Runner.ProviderTurnError({
						provider,
						model,
						cause: new Error(terminal.message.errorMessage ?? `Provider turn ${terminal.reason}`),
					});
				}

				/*
				 * What the terminal says the response was, and what to do about the
				 * calls it left behind. The batch always runs after the terminal is
				 * durable, so the array is complete before a single handler is invoked.
				 */
				switch (terminal.reason) {
					case "toolUse": {
						const ran = yield* runToolBatch(snapshot, messageId);
						if (!ran) {
							// The provider said it wanted tools and named none. Nothing to
							// execute and nothing to continue on, so continuing would spin —
							// this is a broken stream, not an empty turn.
							return yield* new Runner.LLMStreamError({
								sessionId,
								reason: `terminal reason "toolUse" but the response has no tool calls`,
							});
						}
						return { messageId, reason: terminal.reason, needsContinuation: true };
					}

					case "length": {
						/*
						 * The output was cut off mid-generation, so every call in it may
						 * carry truncated arguments — arguments that can still parse and
						 * validate while meaning something else entirely. None are safe to
						 * run, and the model is told so rather than left guessing.
						 */
						yield* closeOpenCalls({ sessionId, messageId, ending: "truncated" });
						return { messageId, reason: terminal.reason, needsContinuation: false };
					}

					case "stop": {
						/*
						 * A `stop` carrying tool calls is a provider contradiction. The
						 * calls are durable and canonical, so the concrete tool lifecycle
						 * is the authoritative one: run them and record the disagreement,
						 * rather than stranding a request the model plainly made.
						 */
						const ran = yield* runToolBatch(snapshot, messageId);
						if (ran) {
							yield* Effect.logWarning("loop: provider finished with stop but requested tools").pipe(
								Effect.annotateLogs({ sessionId, messageId }),
							);
						}
						return { messageId, reason: terminal.reason, needsContinuation: ran };
					}
				}
			});

			const runTurn = Effect.fn("Loop.runTurn")(function* (
				snapshot: State.Snapshot,
				turn: number,
				admission: Admission,
			) {
				const sessionId = snapshot.sessionId;

				// Admitted before the bracket, not inside it: a lane whose input raced
				// away opens no turn at all, and publishing a Started/Ended pair for it
				// would break "count TurnEnded to get the turn index".
				let promoted = 0;
				if (admission.lane !== undefined) {
					// Capture outside the promotion commit: inputs admitted after this
					// position belong to the next provider request.
					const cutoff = yield* events.latestSequence(sessionId);
					if (admission.lane === "followUp") promoted += Number(yield* inputs.promoteFollowUp(sessionId));
					promoted += yield* inputs.promoteSteers(sessionId, cutoff);
					if (promoted === 0 && admission.requires) return { ran: false, needsContinuation: false } as const;
				}

				yield* Effect.logInfo("loop: delivering").pipe(
					Effect.annotateLogs({ sessionId, lane: admission.lane ?? "forced", turn, promoted }),
				);
				yield* events.publish(EventList.TurnStarted, { sessionId, timestamp: yield* DateTime.now, turn });

				const result = yield* runTurnAttempt(snapshot);

				yield* events.publish(EventList.TurnEnded, {
					sessionId,
					timestamp: yield* DateTime.now,
					turn,
					messageId: result.messageId,
					reason: result.reason,
					continuation: result.needsContinuation,
				});
				return { ran: true, needsContinuation: result.needsContinuation } as const;
			});

			const run = Effect.fn("Loop.run")(function* (input: {
				readonly sessionId: SessionSchema.ID;
				readonly force: boolean;
			}) {
				// check enqued steering or followUp messages
				// steer always moves ahead
				const hasSteer = yield* inputs.hasPending(input.sessionId, "steer");
				const hasFollowUp = hasSteer ? false : yield* inputs.hasPending(input.sessionId, "followUp");
				// if nothing then return
				// we don't have anything to process
				if (!input.force && !hasSteer && !hasFollowUp) return;

				// Once per drain, before both loops: the model must not be asked a new
				// question while a call it already requested is still open.
				yield* failInterruptedTools(input.sessionId);

				//
				// promotion that starts the turn
				let admission: Admission = {
					lane: hasSteer ? "steer" : hasFollowUp ? "followUp" : undefined,
					requires: true,
				};
				let shouldRun = input.force || hasSteer || hasFollowUp;

				while (shouldRun) {
					/*
					 * One State capture per exchange. Every turn inside sees the same
					 * prompt, the same tools, and the same provider options, so a
					 * continuation cannot answer a question that was asked of a different
					 * agent. Only a follow-up -- a new outer iteration -- recaptures.
					 */
					const snapshot = yield* state.snapshot(input.sessionId);
					let turn = 0;
					let needsContinuation = true;
					while (needsContinuation) {
						const result = yield* runTurn(snapshot, turn + 1, admission);
						if (result.ran) turn += 1;
						// a continuation is true when we have toolcalls that needs resolving
						// i.e tools were executed; now we need to feed them back
						needsContinuation = result.needsContinuation;
						// else re-run if any steering messages are pending
						if (!needsContinuation) {
							needsContinuation = yield* inputs.hasPending(input.sessionId, "steer");
						}
						// Inside a continuation the turn is already justified. A steer that
						// arrived mid-turn joins the next request; its absence does not
						// cancel a turn the tool results themselves demand.
						admission = { lane: "steer", requires: false };
					}
					//
					// outside the continuation; check for pending followUp
					shouldRun = yield* inputs.hasPending(input.sessionId, "followUp");
					admission = { lane: shouldRun ? "followUp" : undefined, requires: true };
				}
			});

			return Runner.Service.of({ run });
		}),
	);

export * as Loop from "./loop.ts";
