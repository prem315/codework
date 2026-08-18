/*
 * @file Effect-native durable session loop.
 *
 * The loop owns delivery and execution order. It promotes durable input, asks
 * Context for the current storage-faithful aikit messages, and sends the aikit
 * provider events through their request-scoped publisher. It never reads or
 * writes Session.Service directly.
 */

import type { Message } from "@codeworksh/aikit";
import { Effect, Layer } from "effect";
import { Context } from "../context/context.ts";
import { Event } from "../event/event.ts";
import { SessionInput } from "../session/input/input.ts";
import type { SessionSchema } from "../session/schema.ts";
import { LLMEventPublisher } from "./event.ts";
import { LLM } from "./llm.ts";
import { Runner } from "./run.ts";

export interface Options {
	readonly provider?: string;
	readonly model?: string;
	readonly systemPrompt?: string;
	/** Deterministic test seam; production resolves and streams through aikit. */
	readonly request?: LLM.Request;
}

const defaults = {
	provider: "openai",
	model: "gpt-5.5",
	systemPrompt: "You are a concise coding assistant.",
} as const;

export const layer = (options: Options = {}) =>
	Layer.effect(
		Runner.Service,
		Effect.gen(function* () {
			const context = yield* Context.Service;
			const events = yield* Event.Service;
			const inputs = yield* SessionInput.make;
			const provider = options.provider ?? defaults.provider;
			const model = options.model ?? defaults.model;
			const requestLLM = options.request ?? LLM.run;

			// TODO:
			// add failInterruptedTools definition
			// which will pull from context service: currentLeaf
			// if currentLeaf is type user message then return; do nothing
			// if currentLeaf is type assistant message then iterate over its
			// parts.
			// check if the parts are of tool call, then trigger event for each tool call entry
			// yield* events.publish(SessionEvent.Tool.Failed, {
			// see opencode ref: packages/core/src/session/runner/llm.ts Lines:119-139
			// TODO:
			// requires projection to handle:ToolFailed event in :packages/harness/src/session/projector.ts

			// TODO:
			// have runTurnAttempt; take cues form opencode

			// Note: Call Stack: like opencode
			// run
			// 	runTurn
			//			runTurnAttempt
			//
			// Division of labour:
			//   run             - lanes, promotion, failInterruptedTools. Publishes nothing.
			//   runTurn         - TurnStarted -> runTurnAttempt -> TurnEnded.
			//                     Also the retry seam (opencode wraps runTurnAttempt here to
			//                     re-run a turn after overflow compaction: llm.ts:355-381).
			//   runTurnAttempt  - the turn body: promote input, assemble Context, resolve
			//                     provider/model, one provider step, then the tool batch.
			//                     Returns needsContinuation.

			const runTurn = Effect.fn("Loop.runTurn")(function* (
				sessionId: SessionSchema.ID,
				promotion: "steer" | "followUp" | undefined,
			) {
				let promoted = 0;
				if (promotion !== undefined) {
					// Capture outside the promotion commit: inputs admitted after this
					// position belong to the next provider request.
					const cutoff = yield* events.latestSequence(sessionId);
					if (promotion === "followUp") promoted += Number(yield* inputs.promoteFollowUp(sessionId));
					promoted += yield* inputs.promoteSteers(sessionId, cutoff);
					if (promoted === 0) return { ran: false, promoted: 0 } as const;
				}

				// This is our toLLMMessages boundary: Context reads native durable
				// history and returns aikit's canonical Message.Context values.
				const snapshot = yield* context.assemble(sessionId);
				const requestContext: Message.Context = {
					systemPrompt: options.systemPrompt ?? defaults.systemPrompt,
					messages: [...snapshot.messages],
				};
				const label = promotion ?? "forced";
				yield* Effect.logInfo("loop: delivering").pipe(Effect.annotateLogs({ sessionId, lane: label, promoted }));

				const publisher = yield* LLMEventPublisher.make({ sessionId }).pipe(
					Effect.provideService(Event.Service, events),
				);
				const terminal = yield* requestLLM({
					sessionId,
					context: requestContext,
					provider,
					model,
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
				if (terminal.outcome === "failed") {
					return yield* new Runner.ProviderTurnError({
						provider,
						model,
						cause: new Error(terminal.message.errorMessage ?? `Provider turn ${terminal.reason}`),
					});
				}
				return { ran: true, promoted } as const;
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
				// TODO:
				// before we continue invoke:
				// yield* failInterruptedTools(input.sessionID)
				// this shall use the latest assistant message think arrays last entry
				// pulls session entry and parts and iterates over them and sets them:
				// "Tool Execution Interrupted"
				// - a similar messaging like opencode but formatted for our data model.

				//
				// promotion that starts the turn
				let promotion: "steer" | "followUp" | undefined = hasSteer ? "steer" : hasFollowUp ? "followUp" : undefined;
				let shouldRun = input.force || hasSteer || hasFollowUp;

				while (shouldRun) {
					let needsContinuation = true;
					while (needsContinuation) {
						// Turn Lifecycle: a turn is one assistant response + any tool calls/results.
						//
						// TurnStart/TurnEnd therefore bracket `runTurn` itself, NOT this loop:
						// every `runTurn` return is a completed turn, which is exactly what
						// `needsContinuation` reports. Emitting TurnEnd only when the chain stops
						// would give N starts and one end for an N-round tool chain.
						const result = yield* runTurn(input.sessionId, promotion);
						needsContinuation = result.needsContinuation;
						// check if we its continuation of the same run via `result.needsContinuation`
						// a continuation is true when we have toolcalls that needs resolving
						// i.e tools were executed; now we need to feed them back
						//
						// else re-run if any steering messages are pending
						if (!needsContinuation) {
							needsContinuation = yield* inputs.hasPending(input.sessionId, "steer");
						}
						promotion = "steer"; // promotion retains steer within continuation block
					}
					//
					// outside the continuation; check for pending followUp
					shouldRun = yield* inputs.hasPending(input.sessionId, "followUp");
					promotion = shouldRun ? "followUp" : undefined;
				}
			});

			return Runner.Service.of({ run });
		}),
	);

export * as Loop from "./loop.ts";
