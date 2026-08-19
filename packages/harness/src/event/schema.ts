import { Message } from "@codeworksh/aikit";
import { Effect, Schema, type SchemaAST, SchemaGetter, SchemaIssue } from "effect";
import type { Static, TSchema } from "typebox";
import { uuidv7 } from "uuidv7";
import {
	aikitValidator,
	isAikitAssistantMessage,
	optional,
	validateAikitAssistantMessage,
	withStatics,
} from "../schema.ts";

type AikitAssistantPart = Message.AssistantMessage["parts"][number];

/** Undeclared fields used only while aikit is assembling a streamed part. */
const aikitTransientFields: Readonly<Record<AikitAssistantPart["type"], ReadonlyArray<string>>> = {
	text: ["streamId"],
	image: [],
	thinking: ["streamId"],
	toolCall: ["partialJson"],
};

const omit = <T extends object>(value: T, keys: ReadonlyArray<string>): T => {
	if (!keys.some((key) => key in value)) return value;
	const copy = { ...value } as T & Record<string, unknown>;
	for (const key of keys) delete copy[key];
	return copy;
};

const canonicalizeAikitAssistantMessage = (message: Message.AssistantMessage): Message.AssistantMessage => ({
	...message,
	parts: message.parts.map((part) => omit(part, aikitTransientFields[part.type])),
});

const reasonOf = (cause: unknown): string => (cause instanceof Error ? cause.message : String(cause));

/**
 * Effect Schema adapter for one aikit TypeBox value.
 *
 * A durable event that persists an aikit value needs the same three things
 * every time: a refinement for the declaration, a validating transform in both
 * directions, and a canonicalizer that strips fields aikit only uses while a
 * block is streaming. Encode runs the same transform as decode deliberately —
 * the value is stored verbatim, so what goes in must satisfy what comes out.
 */
const aikitValue = <T extends TSchema>(
	schema: T,
	expected: string,
	canonicalize: (value: Static<T>) => Static<T> = (value) => value,
) => {
	const { is, validate } = aikitValidator(schema, expected);
	const transform = (value: unknown, options: SchemaAST.ParseOptions) =>
		Effect.try({
			try: () => canonicalize(validate(value)),
			catch: (cause) => new SchemaIssue.InvalidValue({ message: reasonOf(cause) }, value, options),
		});
	return Schema.Unknown.pipe(
		Schema.decodeTo(Schema.declare<Static<T>>(is, { expected }), {
			decode: SchemaGetter.transformOrFail(transform),
			encode: SchemaGetter.transformOrFail(transform),
		}),
	);
};

const decodeAikitAssistantMessage = (value: unknown, options: SchemaAST.ParseOptions) =>
	Effect.try({
		try: () => canonicalizeAikitAssistantMessage(validateAikitAssistantMessage(value, "aikit assistant message")),
		catch: (cause) => new SchemaIssue.InvalidValue({ message: reasonOf(cause) }, value, options),
	});

const aikitAssistantMessageDeclared = Schema.declare<Message.AssistantMessage>(isAikitAssistantMessage, {
	expected: "aikit AssistantMessage",
});

/**
 * Effect Schema adapter for aikit's TypeBox assistant message. Durable LLM
 * events use it to validate their payload and strip streaming-only fields.
 */
export const AikitAssistantMessage = Schema.Unknown.pipe(
	Schema.decodeTo(aikitAssistantMessageDeclared, {
		decode: SchemaGetter.transformOrFail(decodeAikitAssistantMessage),
		encode: SchemaGetter.transformOrFail(decodeAikitAssistantMessage),
	}),
);
export type AikitAssistantMessage = typeof AikitAssistantMessage.Type;

/**
 * The two block completions that project a part on their own.
 *
 * They carry the finished block rather than its text because aikit sets
 * `textSignature` / `thinkingSignature` on the block *before* pushing the end
 * event (`aikit/llm/stream.ts:253-282`). Those signatures are what a provider
 * needs to continue a multi-turn reasoning thread, so a durable write that
 * dropped them would leave a crashed session unable to resume its own thinking.
 */
export const AikitTextPart = aikitValue(Message.TextContentSchema, "aikit TextContent", (part) =>
	omit(part, aikitTransientFields.text),
);
export type AikitTextPart = typeof AikitTextPart.Type;

export const AikitThinkingPart = aikitValue(Message.ThinkingContentSchema, "aikit ThinkingContent", (part) =>
	omit(part, aikitTransientFields.thinking),
);
export type AikitThinkingPart = typeof AikitThinkingPart.Type;

/**
 * The third block completion: the model has finished asking for a call.
 *
 * Pinned to the `pending` member of aikit's `ToolCall` union rather than the
 * union itself, because that is the whole content of the event. `final` means
 * the arguments are complete and nothing has run -- the transition into
 * `running` belongs to `ToolExecutionStarted`, which is what makes "pending
 * provably never ran" a fact the recovery sweep can rely on.
 */
export const AikitToolCallPendingPart = aikitValue(
	Message.ToolCallPendingPartSchema,
	"aikit pending ToolCall",
	(part) => omit(part, aikitTransientFields.toolCall),
);
export type AikitToolCallPendingPart = typeof AikitToolCallPendingPart.Type;

/** A call the harness has begun executing. Live progress and the durable start. */
export const AikitToolCallRunningPart = aikitValue(
	Message.ToolCallRunningPartSchema,
	"aikit running ToolCall",
	(part) => omit(part, aikitTransientFields.toolCall),
);
export type AikitToolCallRunningPart = typeof AikitToolCallRunningPart.Type;

/**
 * A settled call, in any of its four terminal states.
 *
 * One codec for all of them, matching the single `ToolExecutionEnded` event:
 * `completed`, `error`, `skipped`, and `aborted` all come out of the same
 * `Executor.handle` pipeline as ordinary return values, project identically,
 * and already carry their status in the part JSON.
 */
export const AikitToolCallTerminalPart = aikitValue(
	Message.ToolCallTerminalPartSchema,
	"aikit terminal ToolCall",
	(part) => omit(part, aikitTransientFields.toolCall),
);
export type AikitToolCallTerminalPart = typeof AikitToolCallTerminalPart.Type;

export const ID = Schema.String.pipe(
	Schema.brand("Event.ID"),
	withStatics((schema) => ({
		create: () => schema.make(`evt_${uuidv7()}`),
	})),
);
export type ID = typeof ID.Type;

export type Definition<
	Type extends string = string,
	DataSchema extends Schema.Codec<unknown, unknown> = Schema.Codec<unknown, unknown>,
> = Schema.Top & {
	readonly type: Type;
	readonly durable?: {
		readonly version: number; // event versioning for decoding in later time
		readonly aggregate: string;
	};
	readonly data: DataSchema;
};

export type Data<D extends Definition> = Schema.Schema.Type<D["data"]>;

export type Payload<D extends Definition = Definition> = {
	readonly id: ID;
	readonly type: D["type"];
	readonly data: Data<D>;
	readonly durable?: {
		readonly aggregateId: string;
		readonly seq: number;
		readonly version: number;
	};
	readonly metadata?: Record<string, string>;
};

export function define<
	const Type extends string,
	const Fields extends Readonly<Record<PropertyKey, Schema.Codec<unknown, unknown>>>,
>(input: {
	readonly type: Type;
	readonly durable?: {
		readonly version: number;
		readonly aggregate: string;
	};
	readonly schema: Fields;
}) {
	const data = Schema.Struct(input.schema);
	return Schema.Struct({
		id: ID,
		metadata: optional(Schema.Record(Schema.String, Schema.String)),
		type: Schema.Literal(input.type),
		durable: optional(Schema.Struct({ aggregateId: Schema.String, seq: Schema.Int, version: Schema.Int })),
		data,
	})
		.annotate({ identifier: input.type })
		.pipe(
			withStatics(() => ({
				type: input.type,
				...(input.durable === undefined ? {} : { durable: input.durable }),
				data,
			})),
		) satisfies Definition<Type, typeof data>;
}

// Register all the event definitions.
export function inventory<const Definitions extends ReadonlyArray<Definition>>(...definitions: Definitions) {
	return Object.freeze(definitions);
}

export function versionedType(type: string, version: number) {
	return `${type}.${version}`;
}

export function durable<const Definitions extends ReadonlyArray<Definition>>(definitions: Definitions) {
	return readonlyMap(
		definitions.reduce((result, definition) => {
			if (!definition.durable) return result;
			const key = versionedType(definition.type, definition.durable.version);
			if (result.has(key)) throw new Error(`Duplicate durable event definition for ${key}`);
			result.set(key, definition);
			return result;
		}, new Map<string, Definitions[number]>()),
	);
}

function readonlyMap<Key, Value>(map: Map<Key, Value>): ReadonlyMap<Key, Value> {
	const result: ReadonlyMap<Key, Value> = Object.freeze({
		get size() {
			return map.size;
		},
		entries: () => map.entries(),
		forEach: (callback: (value: Value, key: Key, map: ReadonlyMap<Key, Value>) => void, thisArg?: unknown) =>
			map.forEach((value, key) => callback.call(thisArg, value, key, result)),
		get: (key: Key) => map.get(key),
		has: (key: Key) => map.has(key),
		keys: () => map.keys(),
		values: () => map.values(),
		[Symbol.iterator]: () => map[Symbol.iterator](),
	});
	return result;
}

export function latest(definitions: ReadonlyArray<Definition>) {
	return readonlyMap(
		definitions.reduce((result, definition) => {
			const existing = result.get(definition.type);
			if (!existing) {
				result.set(definition.type, definition);
				return result;
			}
			if (definition.durable && existing.durable && definition.durable.version !== existing.durable.version) {
				if (definition.durable.version > existing.durable.version) result.set(definition.type, definition);
				return result;
			}
			if (definition !== existing) throw new Error(`Duplicate latest event definition for ${definition.type}`);
			return result;
		}, new Map<string, Definition>()),
	);
}

export * as EventSchema from "./schema.ts";
