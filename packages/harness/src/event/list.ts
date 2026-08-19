import { Schema } from "effect";
import { DateTimeUtcFromMillis, NonNegativeInt, optional, PositiveInt } from "../schema.ts";
import { SessionMessageSchema } from "../session/message/schema.ts";
import { PromptSchema } from "../session/prompt/schema.ts";
import { SessionSchema } from "../session/schema.ts";
import { EventSchema } from "./schema.ts";

const baseOptions = {
	timestamp: DateTimeUtcFromMillis,
	sessionId: SessionSchema.ID,
};
const durableOptions = {
	durable: {
		aggregate: "sessionId",
		version: 1,
	},
} as const;

const PromptFields = {
	...baseOptions,
	messageId: SessionMessageSchema.ID,
	prompt: PromptSchema.Prompt,
	delivery: PromptSchema.Delivery,
};

export const PromptAdmitted = EventSchema.define({
	type: "session.next.prompt.admitted",
	...durableOptions,
	schema: PromptFields,
});
export type PromptAdmitted = typeof PromptAdmitted.Type;

export const Prompted = EventSchema.define({
	type: "session.next.prompt.promoted",
	...durableOptions,
	schema: PromptFields,
});
export type Prompted = typeof Prompted.Type;

/*
 * Turn lifecycle. A turn is one assistant response plus any tool calls it makes;
 * `runTurn` brackets `runTurnAttempt` with these two.
 */
const TurnFields = {
	...baseOptions,
	/** Position within the exchange, from 1. Resets when a follow-up opens a new one. */
	turn: PositiveInt,
};

/**
 * Live: a turn was opened.
 *
 * Deliberately not durable. Every field it could carry is already recorded
 * somewhere earlier -- the resolved model in the assistant envelope, the
 * promoted prompts in `Prompted`, the parent entry in `session_entry.parent_id`
 * -- and it projects nothing, because the assistant entry it would describe does
 * not exist yet. It is a boundary marker for a UI, nothing more.
 */
export const TurnStarted = EventSchema.define({
	type: "session.turn.started",
	schema: TurnFields,
});
export type TurnStarted = typeof TurnStarted.Type;

/**
 * Durable: the turn's context, tool results included, is complete.
 *
 * It writes no state -- everything the turn produced is durable by the time it
 * fires -- so it has no projector. It is durable because the boundary itself is
 * worth replaying: it is what lets the log be read as a sequence of turns rather
 * than an undifferentiated stream of parts, and `continuation` is the one fact
 * not recorded anywhere else. The parts say what the model asked for and what
 * came back; only the loop knows whether it decided to go around again.
 *
 * Identity and outcome only, never content: the assistant entry and its parts
 * already are the content, and duplicating them here would create a second
 * source of truth free to drift.
 */
export const TurnEnded = EventSchema.define({
	type: "session.turn.ended",
	...durableOptions,
	schema: {
		...TurnFields,
		messageId: SessionMessageSchema.ID,
		/** The provider's terminal reason for this turn's response. */
		reason: Schema.Literals(["stop", "length", "toolUse"]),
		/** Whether the turn's own outcome requires another turn. */
		continuation: Schema.Boolean,
	},
});
export type TurnEnded = typeof TurnEnded.Type;

// working on it
const LLMFields = {
	...baseOptions,
	messageId: SessionMessageSchema.ID,
};

const LLMPartFields = {
	...LLMFields,
	partIndex: NonNegativeInt,
};

/**
 * aikit `start`, or the publisher's own lazy creation: the assistant entry
 * exists from here on.
 *
 * Durable because it is the **allocation** — every block completion writes a
 * part, and a part needs an owning entry. The envelope it carries is a snapshot
 * with no parts and `stopReason: "aborted"`, promoted to its true reason at
 * `LLMEnded` / `LLMFailed`. A crash mid-stream therefore leaves a correctly
 * marked aborted assistant carrying whatever blocks had completed, instead of
 * discarding the whole response.
 *
 * Its usage is zeroed rather than copied from aikit's partial: usage is charged
 * exactly once, at finalization, and a partial that already carried a count
 * would be charged twice.
 */
export const LLMStarted = EventSchema.define({
	type: "session.llm.started",
	...durableOptions,
	schema: { ...LLMFields, message: EventSchema.AikitAssistantMessage },
});
export type LLMStarted = typeof LLMStarted.Type;

export const LLMTextStart = EventSchema.define({
	type: "session.llm.text.start",
	schema: LLMPartFields,
});
export type LLMTextStart = typeof LLMTextStart.Type;

export const LLMTextDelta = EventSchema.define({
	type: "session.llm.text.delta",
	schema: { ...LLMPartFields, delta: Schema.String },
});
export type LLMTextDelta = typeof LLMTextDelta.Type;

/** Block boundary: the text block is final. Written at its `partIndex`. */
export const LLMTextEnd = EventSchema.define({
	type: "session.llm.text.end",
	...durableOptions,
	schema: { ...LLMPartFields, part: EventSchema.AikitTextPart },
});
export type LLMTextEnd = typeof LLMTextEnd.Type;

export const LLMThinkingStart = EventSchema.define({
	type: "session.llm.thinking.start",
	schema: LLMPartFields,
});
export type LLMThinkingStart = typeof LLMThinkingStart.Type;

export const LLMThinkingDelta = EventSchema.define({
	type: "session.llm.thinking.delta",
	schema: { ...LLMPartFields, delta: Schema.String },
});
export type LLMThinkingDelta = typeof LLMThinkingDelta.Type;

/** Block boundary: the thinking block is final. Written at its `partIndex`. */
export const LLMThinkingEnd = EventSchema.define({
	type: "session.llm.thinking.end",
	...durableOptions,
	schema: { ...LLMPartFields, part: EventSchema.AikitThinkingPart },
});
export type LLMThinkingEnd = typeof LLMThinkingEnd.Type;

/*
 * Tool-call blocks follow the same rule as text and thinking: live while the
 * block streams, durable when it completes. Only `toolcall.final` is a
 * completion.
 *
 * Start and delta carry no call identity, exactly as their text and thinking
 * counterparts carry none. A live consumer addresses a block by `partIndex`,
 * which is the addressing unit everywhere else; the identity that matters
 * downstream is the one written durably at finalization.
 */
export const LLMToolCallStarted = EventSchema.define({
	type: "session.llm.toolcall.started",
	schema: LLMPartFields,
});
export type LLMToolCallStarted = typeof LLMToolCallStarted.Type;

export const LLMToolCallDelta = EventSchema.define({
	type: "session.llm.toolcall.delta",
	schema: { ...LLMPartFields, delta: Schema.String },
});
export type LLMToolCallDelta = typeof LLMToolCallDelta.Type;

/**
 * Live: the provider stopped emitting input for this call.
 *
 * Not a block completion. A provider may emit `toolcall.end` and then a
 * `toolcall.final` that rewrites the arguments (`aikit/llm/stream.ts:294-318`),
 * so this is the last fragment rather than the first fact.
 */
export const LLMToolCallEnded = EventSchema.define({
	type: "session.llm.toolcall.ended",
	schema: { ...LLMPartFields, callId: Schema.String, toolName: Schema.String },
});
export type LLMToolCallEnded = typeof LLMToolCallEnded.Type;

/**
 * Durable block completion: the model finished asking, with canonical
 * arguments. The part is written at its `partIndex`, status `pending`.
 *
 * Two facts a single event would conflate stay apart here. `final` means the
 * arguments are complete; nothing has run. `ToolExecutionStarted` means we began
 * doing, and a side effect may now have happened. Keeping them separate is what
 * makes the recovery table expressible: `pending` provably never ran, `running`
 * may have written to disk.
 */
export const LLMToolCallFinalized = EventSchema.define({
	type: "session.llm.toolcall.finalized",
	...durableOptions,
	schema: {
		...LLMPartFields,
		callId: Schema.String,
		toolName: Schema.String,
		part: EventSchema.AikitToolCallPendingPart,
	},
});
export type LLMToolCallFinalized = typeof LLMToolCallFinalized.Type;

/** aikit `done`: the response completed and `message` is authoritative. */
export const LLMEnded = EventSchema.define({
	type: "session.llm.ended",
	...durableOptions,
	schema: {
		...LLMFields,
		reason: Schema.Literals(["stop", "length", "toolUse"]),
		message: EventSchema.AikitAssistantMessage,
	},
});
export type LLMEnded = typeof LLMEnded.Type;

/**
 * aikit `error`: the response failed or was aborted. aikit names this payload
 * `error`, but it is a complete assistant message carrying everything generated
 * before the failure, so it projects exactly like a successful one.
 */
export const LLMFailed = EventSchema.define({
	type: "session.llm.failed",
	...durableOptions,
	schema: {
		...LLMFields,
		reason: Schema.Literals(["aborted", "error"]),
		message: EventSchema.AikitAssistantMessage,
	},
});
export type LLMFailed = typeof LLMFailed.Type;

/*
 * Tool execution: the harness's own record of what it did, as distinct from
 * what the provider said. `LLMToolCallFinalized` means the model finished
 * asking; these mean we began, and then finished, doing.
 */
const ToolExecutionFields = {
	...baseOptions,
	/** The owning assistant entry. Named for the message because it is one. */
	messageId: SessionMessageSchema.ID,
	partIndex: NonNegativeInt,
	callId: Schema.String,
	toolName: Schema.String,
};

/**
 * Durable: `pending -> running`. The only transition into `running`, and it is
 * always a claim that a side effect may now occur — it is published immediately
 * before the handler is invoked, never after.
 *
 * That is what makes crash recovery expressible: a `pending` call provably never
 * ran, a `running` one may have written to disk.
 *
 * It carries the running part rather than bare identity because the promoted
 * `status` column and the authoritative part JSON have to agree; `decodeParts`
 * re-checks exactly that on the way back.
 */
export const ToolExecutionStarted = EventSchema.define({
	type: "session.tool.execution.started",
	...durableOptions,
	schema: { ...ToolExecutionFields, part: EventSchema.AikitToolCallRunningPart },
});
export type ToolExecutionStarted = typeof ToolExecutionStarted.Type;

/** Live: interim output from a running tool. Best-effort, never load-bearing. */
export const ToolExecutionUpdated = EventSchema.define({
	type: "session.tool.execution.updated",
	schema: { ...ToolExecutionFields, part: EventSchema.AikitToolCallRunningPart },
});
export type ToolExecutionUpdated = typeof ToolExecutionUpdated.Type;

/**
 * Durable: `running -> terminal`. One event for all four outcomes —
 * `completed`, `error`, `skipped`, and `aborted` all come out of the same
 * `Executor.handle` pipeline as ordinary return values, project identically,
 * and already carry their status in the part JSON.
 */
export const ToolExecutionEnded = EventSchema.define({
	type: "session.tool.execution.ended",
	...durableOptions,
	schema: { ...ToolExecutionFields, part: EventSchema.AikitToolCallTerminalPart },
});
export type ToolExecutionEnded = typeof ToolExecutionEnded.Type;

/**
 * Durable: an unresolved call settled without ever producing an executor
 * outcome. Published by the drain-start recovery sweep, and by the tool batch's
 * own shutdown finalizer when a live process is interrupted.
 *
 * Deliberately outside the `session.tool.execution.*` namespace, and deliberately
 * not `ToolExecutionEnded`. That event carries a complete terminal part built by
 * `Executor.handle`; this one exists precisely because no such part was ever
 * built, so it carries identity plus an error and lets the projector synthesize
 * the part. Reusing the other would mean hand-constructing aikit terminal parts
 * for calls that never ran.
 *
 * `status` is chosen by what ended the turn together with what the prior status
 * proved. Under interruption a `pending` call was never started and is
 * `skipped`, while a `running` one may already have written to disk and is
 * `aborted` — the entire payoff of keeping `ToolExecutionStarted` a separate
 * transition. A provider failure settles its calls as `error`, and a truncated
 * response settles them `skipped` because their arguments cannot be trusted.
 */
export const ToolFailed = EventSchema.define({
	type: "session.tool.failed",
	...durableOptions,
	schema: {
		...ToolExecutionFields,
		status: Schema.Literals(["skipped", "aborted", "error"]),
		error: Schema.String,
		/**
		 * The last progress the tool reported, when the harness still had it.
		 *
		 * Present only on the shutdown path, where the loop watched the call run
		 * and holds its partials. The drain-start sweep has none — the process that
		 * held them is gone — which is exactly the difference between an interrupt
		 * and a kill, recorded rather than flattened.
		 */
		partial: optional(EventSchema.AikitToolCallRunningPart),
	},
});
export type ToolFailed = typeof ToolFailed.Type;

/**
 * First event in a forked session's log. The aggregate is the *new* session, so
 * this lands at `baseSeq + 1`: the fork seeds the new aggregate's sequence to
 * the fork point, reserving [0..baseSeq] for the copied entries. Recording it
 * makes the seeded range auditable from the log itself, not only from
 * `session.parentId`.
 */
export const SessionForked = EventSchema.define({
	type: "session.next.forked",
	...durableOptions,
	schema: {
		...baseOptions,
		sourceSessionId: SessionSchema.ID,
		/** Entry in the source session the copy stops at (its leaf, for a clone). */
		sourceEntryId: Schema.String,
		/** Highest sequence carried over; the new log starts above it. */
		baseSeq: NonNegativeInt,
	},
});
export type SessionForked = typeof SessionForked.Type;

export const DurableDefinitions = EventSchema.inventory(
	PromptAdmitted,
	Prompted,
	SessionForked,
	TurnEnded,
	LLMStarted,
	LLMTextEnd,
	LLMThinkingEnd,
	LLMToolCallFinalized,
	LLMEnded,
	LLMFailed,
	ToolExecutionStarted,
	ToolExecutionEnded,
	ToolFailed,
);

export * as EventList from "./list.ts";
