/*
 * @file Codec provides the wire codec schema b/w harness <-> aikit
 * Use helper functions to encode-decode payload over the wire.
 * */
import { Message } from "@codeworksh/aikit";
import { DateTime, Effect, Option, Schema } from "effect";
import {
	validateAikitMessage,
	validateAikitPendingToolCall,
	validateAikitToolCall,
	validateAikitUserMessage,
} from "../schema.ts";
import { SessionSchema } from "../session/schema.ts";
import type { Session } from "../session/session.ts";
import { ContextDecodeError, ContextEncodeError } from "./errors.ts";

export interface EncodedMessage {
	readonly type: "user" | "assistant";
	readonly data: string;
	readonly parts: ReadonlyArray<Session.AppendPart>;
}

const reasonOf = (cause: unknown): string => (cause instanceof Error ? cause.message : String(cause));
const decodeJsonObject = Schema.decodeUnknownEffect(SessionSchema.JsonObject);
const encodeJsonObject = Schema.encodeEffect(SessionSchema.JsonObject);

const parseJson = (data: string, entryId: string, type: string, subject: string) =>
	decodeJsonObject(data).pipe(
		Effect.mapError(
			(cause) =>
				new ContextDecodeError({
					entryId,
					type,
					reason: `${subject} is not valid JSON: ${reasonOf(cause)}`,
				}),
		),
	);

const validateDecoded = (
	value: unknown,
	entryId: string,
	type: string,
): Effect.Effect<Message.Message, ContextDecodeError> =>
	Effect.try({
		try: () => validateAikitMessage(value, `context entry ${entryId}`),
		catch: (cause) => new ContextDecodeError({ entryId, type, reason: reasonOf(cause) }),
	});

export const validateUserMessage = (
	value: unknown,
	entryId: string,
	type: string,
): Effect.Effect<Message.UserMessage, ContextDecodeError> =>
	Effect.try({
		try: () => validateAikitUserMessage(value, `context entry ${entryId}`),
		catch: (cause) => new ContextDecodeError({ entryId, type, reason: reasonOf(cause) }),
	});

type AnyPart = Message.Message["parts"][number];

/**
 * One part as a storable row: verbatim JSON plus the columns promoted out of it.
 *
 * The promotion is not a convenience copy — `status`, `callId`, and `toolName`
 * are indexed, and `decodeParts` re-checks them against the JSON on the way
 * back, so they have to be derived here rather than restated by each caller.
 */
export const encodePart = Effect.fn("Context.encodePart")(function* (input: {
	readonly messageId: string;
	readonly role: "user" | "assistant";
	readonly part: AnyPart;
}): Effect.fn.Return<Session.AppendPart, ContextEncodeError> {
	const { messageId, role, part } = input;
	const data = yield* encodeJsonObject(part).pipe(
		Effect.mapError((cause) => new ContextEncodeError({ messageId, role, reason: cause.message })),
	);
	return {
		type: part.type,
		...(part.type === "toolCall" ? { status: part.status, callId: part.callID, toolName: part.name } : {}),
		data,
	};
});

export const encodeMessage = Effect.fn("Context.encodeMessage")(function* (
	message: Message.Message,
): Effect.fn.Return<EncodedMessage, ContextEncodeError> {
	const validated = yield* Effect.try({
		try: () => validateAikitMessage(message, `context message ${message.messageId}`),
		catch: (cause) =>
			new ContextEncodeError({
				messageId: message.messageId,
				role: message.role,
				reason: reasonOf(cause),
			}),
	});
	const { parts, ...envelope } = validated;
	const data = yield* encodeJsonObject(envelope).pipe(
		Effect.mapError(
			(cause) =>
				new ContextEncodeError({
					messageId: message.messageId,
					role: message.role,
					reason: cause.message,
				}),
		),
	);
	const encodedParts = yield* Effect.forEach(parts, (part) =>
		encodePart({ messageId: validated.messageId, role: validated.role, part }),
	);
	return {
		type: validated.role,
		data,
		parts: encodedParts,
	};
});

const decodeParts = Effect.fn("Context.decodeParts")(function* (
	hydrated: Session.HydratedEntry,
): Effect.fn.Return<ReadonlyArray<unknown>, ContextDecodeError> {
	const { entry } = hydrated;
	const sorted = [...hydrated.parts].sort((left, right) => left.partIndex - right.partIndex);
	const parts: unknown[] = [];
	for (const [index, row] of sorted.entries()) {
		if (row.entryId !== entry.id) {
			return yield* new ContextDecodeError({
				entryId: entry.id,
				type: entry.type,
				reason: `part ${index} belongs to entry ${row.entryId}`,
			});
		}
		/*
		 * Gaps are legal, and this used to reject them.
		 *
		 * `partIndex` is aikit's own addressing — a block's position in the message
		 * it is being assembled into — and a slot is written when its block
		 * completes, not when it is announced. So an entry killed mid-stream can
		 * genuinely hold part 1 and not part 0: the text block at 0 was announced
		 * and never finished while the tool call at 1 finalized. Requiring dense
		 * indexes made that entry unreadable, which broke `latestAssistant` and
		 * `assemble` on precisely the entry the recovery sweep exists to settle —
		 * and, because both are read on every drain, left the session permanently
		 * undrainable.
		 *
		 * Nothing is lost by allowing it. A settled assistant is dense by
		 * construction rather than by inspection: `finalizeAssistant` upserts the
		 * terminal's whole array and deletes anything past its end. This check only
		 * ever fired on entries still in flight.
		 */
		const part = yield* parseJson(row.data, entry.id, entry.type, `part ${index}`);
		if (typeof part !== "object" || part === null || !("type" in part) || part.type !== row.type) {
			return yield* new ContextDecodeError({
				entryId: entry.id,
				type: entry.type,
				reason: `part ${index} type does not match promoted type "${row.type}"`,
			});
		}
		if (row.type === "toolCall") {
			const promotedStatus = Option.getOrUndefined(row.status);
			const promotedCallId = Option.getOrUndefined(row.callId);
			const promotedToolName = Option.getOrUndefined(row.toolName);
			if (
				!("status" in part) ||
				part.status !== promotedStatus ||
				!("callID" in part) ||
				part.callID !== promotedCallId ||
				!("name" in part) ||
				part.name !== promotedToolName
			) {
				return yield* new ContextDecodeError({
					entryId: entry.id,
					type: entry.type,
					reason: `part ${index} tool columns do not match its authoritative JSON`,
				});
			}
		} else if (Option.isSome(row.status) || Option.isSome(row.callId) || Option.isSome(row.toolName)) {
			return yield* new ContextDecodeError({
				entryId: entry.id,
				type: entry.type,
				reason: `part ${index} type "${row.type}" must not carry promoted tool columns`,
			});
		}
		parts.push(part);
	}
	return parts;
});

/**
 * One stored `toolCall` row, back as the value the executor takes.
 *
 * The loop re-reads what it committed rather than trusting the terminal message
 * it holds in memory, so this is the boundary that turns a row back into an
 * aikit part. The status is dropped, not checked: `ExecutionInput` is
 * status-free because `ToolExecutionStarted` commits `running` before the
 * handler runs.
 */
export const decodeToolCall = Effect.fn("Context.decodeToolCall")(function* (row: {
	readonly entryId: string;
	readonly partIndex: number;
	readonly data: string;
}): Effect.fn.Return<Message.ToolCallPendingPart, ContextDecodeError> {
	const part = yield* parseJson(row.data, row.entryId, "assistant", `part ${row.partIndex}`);
	return yield* Effect.try({
		try: () => validateAikitPendingToolCall(part, `context entry ${row.entryId} part ${row.partIndex}`),
		catch: (cause) => new ContextDecodeError({ entryId: row.entryId, type: "assistant", reason: reasonOf(cause) }),
	});
});

/**
 * Any stored `toolCall` row as its identity and arguments, without its outcome.
 *
 * What recovery needs: a call it must settle may be `pending` or `running`, and
 * neither status nor a running `partial` belongs in the terminal part
 * synthesized from it. The result is the same status-free shape execution takes.
 */
export const decodeToolCallBase = Effect.fn("Context.decodeToolCallBase")(function* (row: {
	readonly entryId: string;
	readonly partIndex: number;
	readonly data: string;
}): Effect.fn.Return<Omit<Message.ToolCallPendingPart, "status">, ContextDecodeError> {
	const parsed = yield* parseJson(row.data, row.entryId, "assistant", `part ${row.partIndex}`);
	const call = yield* Effect.try({
		try: () => validateAikitToolCall(parsed, `context entry ${row.entryId} part ${row.partIndex}`),
		catch: (cause) => new ContextDecodeError({ entryId: row.entryId, type: "assistant", reason: reasonOf(cause) }),
	});
	const { status: _status, ...base } = call;
	return "partial" in base || "result" in base
		? (({ partial: _partial, result: _result, ...rest }) => rest)(
				base as typeof base & { partial?: unknown; result?: unknown },
			)
		: base;
});

export const decodeSyntheticMessage = Effect.fn("Context.decodeSyntheticMessage")(function* (
	hydrated: Session.HydratedEntry,
): Effect.fn.Return<Message.UserMessage, ContextDecodeError> {
	const { entry } = hydrated;
	const parts = yield* decodeParts(hydrated);
	return yield* validateUserMessage(
		{
			messageId: entry.id,
			role: "user",
			time: { created: DateTime.toEpochMillis(entry.createdAt) },
			parts,
		},
		entry.id,
		entry.type,
	);
});

export const decodeMessage = Effect.fn("Context.decodeMessage")(function* (
	hydrated: Session.HydratedEntry,
): Effect.fn.Return<Message.Message, ContextDecodeError> {
	const { entry } = hydrated;
	if (entry.type !== "user" && entry.type !== "assistant") {
		return yield* new ContextDecodeError({
			entryId: entry.id,
			type: entry.type,
			reason: "only user and assistant entries contain complete aikit messages",
		});
	}

	const envelope = yield* parseJson(entry.data, entry.id, entry.type, "message envelope");
	const parts = yield* decodeParts(hydrated);

	const message = yield* validateDecoded({ ...(envelope as Record<string, unknown>), parts }, entry.id, entry.type);
	if (message.messageId !== entry.id) {
		return yield* new ContextDecodeError({
			entryId: entry.id,
			type: entry.type,
			reason: `messageId "${message.messageId}" does not match entry id`,
		});
	}
	if (message.role !== entry.type) {
		return yield* new ContextDecodeError({
			entryId: entry.id,
			type: entry.type,
			reason: `message role "${message.role}" does not match entry type`,
		});
	}
	return message;
});

export * as ContextCodec from "./codec.ts";
