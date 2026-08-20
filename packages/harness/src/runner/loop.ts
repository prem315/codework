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
import { Cause, DateTime, Effect, Exit, Layer, Option } from "effect";
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
				/** Last progress per call, when this process was the one watching. */
				readonly partials?: ReadonlyMap<string, Message.ToolCallRunningPart>;
			}) {
				for (const call of input.calls) {
					// What the prior status proves, and nothing more. `pending` was never
					// started; `running` may already have written to disk.
					const skipped = call.status === "pending";
					const partial = input.partials?.get(call.callId);
					yield* events.publish(EventList.ToolFailed, {
						sessionId: input.sessionId,
						timestamp: yield* DateTime.now,
						messageId: input.messageId,
						partIndex: call.partIndex,
						callId: call.callId,
						toolName: call.toolName,
						status: skipped ? "skipped" : "aborted",
						error: skipped
							? "Tool call was never executed: the session ended before it started."
							: "Tool execution was interrupted: the session ended while it was running.",
						...(partial === undefined ? {} : { partial }),
					});
				}
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
				const found = yield* context.currentDraft(sessionId);
				if (Option.isNone(found)) return;
				const messageId = SessionMessageSchema.ID.from(found.value.messageId);
				yield* failCalls({ sessionId, messageId, calls: yield* unresolvedCalls(messageId) });
				/*
				 * Close it, as the turn itself would have.
				 *
				 * Which way depends on how far the turn got, and `state` is what make
				 * that answerable: a draft whose envelope still says `aborted` never
				 * received a terminal at all, because an aborted terminal settles the
				 * entry rather than leaving it draft. Anything else means the response
				 * completed and only its tools were cut short — that turn earned its
				 * commit.
				 */
				yield* sessions
					.closeDraft({
						sessionId,
						entryId: messageId,
						state: found.value.stopReason === "aborted" ? "aborted" : "committed",
					})
					.pipe(Effect.asVoid, Effect.orDie);
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
				 * The terminal's own projection has already settled every call this
				 * response will never run — a truncated one's arguments cannot be
				 * trusted, a failed one's calls never got a chance — in the same
				 * transaction that settled the entry. So whatever is still pending here
				 * is, by construction, exactly what should execute, and the loop does
				 * not need to know why the turn ended to decide that.
				 *
				 * That is what used to be a five-way switch. It also dissolves the
				 * question of whether a `stop` carrying tool calls should run them:
				 * nothing settled them, so they run.
				 */
				if (terminal.outcome === "failed") {
					return yield* new Runner.ProviderTurnError({
						provider,
						model,
						cause: new Error(terminal.message.errorMessage ?? `Provider turn ${terminal.reason}`),
					});
				}

				const ran = yield* runToolBatch(snapshot, messageId);
				/*
				 * A truncated response continues even though nothing ran. Its calls were
				 * settled with an instruction to re-issue them, and that instruction is
				 * only worth writing if the model gets another turn to act on it —
				 * otherwise the exchange ends with dead calls and no answer.
				 */
				if (terminal.reason === "length") {
					return { messageId, reason: terminal.reason, needsContinuation: true };
				}
				if (terminal.reason === "toolUse" && !ran) {
					// The provider said it wanted tools and named none. Nothing to run
					// and nothing to continue on, so continuing would spin.
					return yield* new Runner.LLMStreamError({
						sessionId,
						reason: `terminal reason "toolUse" but the response has no tool calls`,
					});
				}
				return { messageId, reason: terminal.reason, needsContinuation: ran };
			});

			/**
			 * Stop an exchange the model will not stop on its own.
			 *
			 * Ends the drain normally rather than failing it: a policy limit tripping
			 * is the limit working, and a failure here would read as a defect in every
			 * log and alert that saw it. The durable record and the synthetic entry
			 * are what make the stop legible — to a reader of the log, and to the model
			 * on the next prompt, which would otherwise see a long chain of turns and
			 * no sign that it was cut off.
			 */
			const halt = Effect.fn("Loop.halt")(function* (snapshot: State.Snapshot, continuations: number) {
				yield* Effect.logWarning("loop: exchange halted at the continuation limit").pipe(
					Effect.annotateLogs({ sessionId: snapshot.sessionId, continuations }),
				);
				yield* events.publish(EventList.ExchangeHalted, {
					sessionId: snapshot.sessionId,
					timestamp: yield* DateTime.now,
					entryId: SessionMessageSchema.ID.create(),
					continuations,
				});
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

				/*
				 * A turn that stops without finishing still has to close.
				 *
				 * `TurnEnded` is what commits an entry the terminal deliberately left
				 * `draft`, so stopping between the two left it unfinished until the next
				 * drain's sweep found it — making an interrupt depend on the path meant
				 * for hard kills.
				 *
				 * Interruption is what this is for, not tool failure. A tool that errors
				 * or aborts settles as a terminal part with `isError`, which the model
				 * reads; only a `failureMode: "error"` tool propagates, and none exists.
				 * What remains is a stop landing mid-batch, plus corruption edges.
				 *
				 * The close is a finalizer because that is the only thing that runs
				 * while a fiber unwinds.
				 */
				const result = yield* runTurnAttempt(snapshot).pipe(
					Effect.onExit((exit) =>
						Exit.isSuccess(exit)
							? Effect.void
							: DateTime.now.pipe(
									Effect.flatMap((timestamp) =>
										events.publish(EventList.TurnFailed, {
											sessionId,
											timestamp,
											turn,
											// An interrupted exit *is* a failure exit, so the cause has
											// to be asked what kind it was rather than the exit.
											reason: Cause.hasInterrupts(exit.cause) ? "interrupted" : Cause.pretty(exit.cause),
										}),
									),
									Effect.orDie,
								),
					),
				);

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
					/*
					 * Consecutive turns the *model* asked for. Not the same as `turn`,
					 * which counts every turn in the exchange: a user steering fifty times
					 * is driving, not looping, and bounding that would cut them off.
					 */
					let continuations = 0;
					let needsContinuation = true;
					while (needsContinuation) {
						const result = yield* runTurn(snapshot, turn + 1, admission);
						if (result.ran) turn += 1;
						// a continuation is true when we have toolcalls that needs resolving
						// i.e tools were executed; now we need to feed them back
						needsContinuation = result.needsContinuation;
						continuations = needsContinuation ? continuations + 1 : 0;
						// else re-run if any steering messages are pending
						if (!needsContinuation) {
							// User input resets the leash rather than spending it.
							needsContinuation = yield* inputs.hasPending(input.sessionId, "steer");
						}
						if (needsContinuation && continuations >= snapshot.maxContinuations) {
							yield* halt(snapshot, continuations);
							break;
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
