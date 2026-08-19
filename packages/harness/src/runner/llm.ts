/*
 * @file Effect boundary for one aikit LLM request.
 *
 * aikit owns provider normalization and emits an AsyncIterable of canonical LLM
 * events. This module owns the lifecycle that aikit cannot see: Effect fiber
 * interruption aborts the transport, while a scoped consumer stays alive long
 * enough to publish aikit's terminal error event durably.
 */

import {
	stream as aikitStream,
	llm,
	type Event as AikitEvent,
	type Message,
	type Model,
	type OpenAIOptions,
} from "@codeworksh/aikit";
import { Effect, Exit, Fiber, Scope, Stream } from "effect";
import type { SessionSchema } from "../session/schema.ts";
import type { State } from "../state/state.ts";
import { LLMEventPublisher } from "./event.ts";
import { Runner } from "./run.ts";

/**
 * The resolved aikit option bag for one request.
 *
 * State supplies the caller-facing half; Loop adds the two fields State
 * deliberately withholds -- `sessionId`, which the loop owns, and `reasoning`,
 * which it derives from the pinned thinking level. This module adds `signal` and
 * nothing else, because `signal` is bound to the scope that aborts the transport
 * on interruption and a caller value would silently break cancellation.
 */
export type RequestOptions = State.RequestOptions & {
	readonly sessionId: string;
	readonly reasoning?: Model.ActiveThinkingLevel;
};

export interface Input {
	readonly sessionId: SessionSchema.ID;
	readonly context: Message.Context;
	readonly provider: string;
	readonly model: string;
	/** Fully resolved by the loop; this module neither defaults nor overrides it. */
	readonly options: RequestOptions;
}

export interface RequestInput extends Input {
	readonly publisher: LLMEventPublisher.Publisher;
}

export type Open = (
	input: Input,
	signal: AbortSignal,
) => Effect.Effect<AsyncIterable<AikitEvent.LLMMessageEvent>, Runner.ModelNotFoundError | Runner.ProviderTurnError>;

export type Request = (
	input: RequestInput,
) => Effect.Effect<
	LLMEventPublisher.Terminal,
	Runner.ModelNotFoundError | Runner.ProviderTurnError | Runner.LLMStreamError
>;

/** Resolve the configured model and start aikit's provider stream. */
export const open: Open = Effect.fn("LLM.open")(function* (input, signal) {
	const model = yield* Effect.tryPromise({
		try: () => llm(input.provider, input.model),
		catch: (cause) => new Runner.ProviderTurnError({ provider: input.provider, model: input.model, cause }),
	});
	if (model === undefined) {
		return yield* new Runner.ModelNotFoundError({ provider: input.provider, model: input.model });
	}

	return yield* Effect.try({
		/*
		 * The one cast in the request path, and it is deliberately here rather than
		 * in the type of the bag. aikit's option type is generic per protocol
		 * (`Protocol.OptionsFor<TProtocol>`); ours is the erased form, so a
		 * provider-agnostic loop cannot name the concrete one. Naming a protocol at
		 * the call site is the narrowest place to erase that difference -- the same
		 * move the tool registry makes when it discharges a tool's capability `R`.
		 */
		try: () => aikitStream(model, input.context, { ...input.options, signal } as OpenAIOptions),
		catch: (cause) => new Runner.ProviderTurnError({ provider: input.provider, model: input.model, cause }),
	});
});

const openFailureMessage = (cause: Runner.ModelNotFoundError | Runner.ProviderTurnError): string =>
	cause._tag === "Runner.ModelNotFoundError"
		? `Model "${cause.model}" is not available from provider "${cause.provider}"`
		: `Provider "${cause.provider}" could not start a response: ${
				cause.cause instanceof Error ? cause.cause.message : String(cause.cause)
			}`;

/**
 * Build a provider request from an opener. Tests use this with deterministic
 * aikit event streams; production uses `open` above.
 */
export const make = (openStream: Open): Request => {
	const request = Effect.fn("LLM.run")(function* (input: RequestInput) {
		/*
		 * Give the transport signal its own closeable scope. Closing it aborts the
		 * signal immediately without closing this request's outer scope, which must
		 * remain alive while the consumer drains and commits aikit's terminal event.
		 */
		const signalScope = yield* Scope.make();
		yield* Effect.addFinalizer((exit) => Scope.close(signalScope, exit));
		const signal = yield* Effect.abortSignal.pipe(Scope.provide(signalScope));
		const abort = Scope.close(signalScope, Exit.void);

		return yield* Effect.uninterruptibleMask((restore) =>
			Effect.gen(function* () {
				/*
				 * A failure here produced no aikit events at all -- an unresolvable
				 * model, or a transport that threw before the stream opened. Routing it
				 * through the publisher first is what puts it on the timeline as a
				 * failed assistant entry rather than letting it vanish into a typed
				 * error, which is the only durable record a turn that never reached the
				 * provider can leave.
				 */
				const iterable = yield* restore(openStream(input, signal)).pipe(
					Effect.onInterrupt(() => abort),
					Effect.tapError((cause) =>
						input.publisher
							.failAssistant({ reason: "error", errorMessage: openFailureMessage(cause) })
							.pipe(Effect.ignore),
					),
				);
				const consume = Stream.fromAsyncIterable(
					iterable,
					(cause) => new Runner.ProviderTurnError({ provider: input.provider, model: input.model, cause }),
				).pipe(
					Stream.runForEach(input.publisher.publish),
					Effect.andThen(input.publisher.terminal),
					Effect.interruptible,
				);

				/*
				 * The consumer is deliberately a separate scoped fiber. Interrupting the
				 * owning turn only interrupts its wait, not this consumer. The interrupt
				 * handler aborts aikit's transport; aikit then emits its terminal `error`,
				 * which the same consumer publishes before the parent rethrows the original
				 * interruption. No second iterator can steal or lose that terminal event.
				 */
				const consumer = yield* consume.pipe(Effect.forkScoped({ startImmediately: true }));
				const awaited = yield* restore(Fiber.await(consumer)).pipe(
					Effect.onInterrupt(() => abort),
					Effect.exit,
				);

				if (Exit.isFailure(awaited)) {
					const drained = yield* Fiber.await(consumer);
					if (Exit.isFailure(drained)) return yield* Effect.failCause(drained.cause);
					return yield* Effect.failCause(awaited.cause);
				}

				const completed = awaited.value;
				if (Exit.isFailure(completed)) return yield* Effect.failCause(completed.cause);
				return completed.value;
			}),
		);
	});

	return (input) => request(input).pipe(Effect.scoped);
};

export const run = make(open);

export * as LLM from "./llm.ts";
