import { Context, DateTime, Effect, Layer, Option, Result, Schema } from "effect";
import { SqlClient, SqlSchema } from "effect/unstable/sql";
import { uuidv7 } from "uuidv7";
import { Database } from "../db/db.ts";
import {
	type EntryState,
	entryStates,
	type EntryType,
	entryTypes,
	type MessageEntryType,
	messageEntryTypes,
	type PartType,
	partTypes,
	SessionEntryPartRow,
	SessionEntryRow,
	SessionRow,
	type ToolStatus,
	toolStatuses,
} from "../db/schema.sql.ts";
import { Event } from "../event/event.ts";
import { EventList } from "../event/list.ts";
import { SandboxInstance } from "../sandbox/instance.ts";
import type { AbsolutePath } from "../schema.ts";
import { SessionSchema } from "./schema.ts";

export {
	entryStates,
	entryTypes,
	messageEntryTypes,
	partTypes,
	SessionEntryPartRow,
	SessionEntryRow,
	SessionRow,
	toolStatuses,
	type EntryState,
	type EntryType,
	type MessageEntryType,
	type PartType,
	type ToolStatus,
};

export class SessionNotFoundError extends Schema.TaggedError<SessionNotFoundError>()("SessionNotFoundError", {
	sessionId: Schema.String,
}) {}

export class EntryNotFoundError extends Schema.TaggedError<EntryNotFoundError>()("EntryNotFoundError", {
	sessionId: Schema.String,
	entryId: Schema.String,
}) {}

export class ToolCallNotFoundError extends Schema.TaggedError<ToolCallNotFoundError>()("ToolCallNotFoundError", {
	entryId: Schema.String,
	callId: Schema.String,
}) {}

/**
 * A tool-call transition that cannot be reconciled with what is stored.
 *
 * Separate from {@link ToolCallNotFoundError} because the two mean opposite
 * things: "no such call" is a lookup miss, while this is a call whose recorded
 * history disagrees with what is being written over it — a different session, a
 * different tool, or a transition out of a state it has already left. The
 * settlement must roll back rather than win.
 *
 * Replaying the *same* value is not a conflict. A durable event can be projected
 * more than once, and an identical write is how idempotence is supposed to look.
 */
export class ToolCallConflictError extends Schema.TaggedError<ToolCallConflictError>()("ToolCallConflictError", {
	entryId: Schema.String,
	callId: Schema.String,
	reason: Schema.String,
}) {}

// Structural rejection — the data-structure layer's "index out of bounds".
export class InvalidEntryDataError extends Schema.TaggedError<InvalidEntryDataError>()("InvalidEntryDataError", {
	entryId: Schema.String,
	type: Schema.String,
	reason: Schema.String,
}) {}

export interface CreateSession {
	readonly id?: SessionSchema.ID;
	readonly projectId: string;
	readonly parentId?: SessionSchema.ID; // session hierarchy (subagents) or fork lineage
	readonly slug: string;
	readonly directory: AbsolutePath;
	readonly title: string;
	readonly tag?: string;
	/**
	 * The namespace this session's directory lives in. Defaults to the host,
	 * which needs no row; any other namespace must already be registered.
	 */
	readonly sandboxInstanceId?: SandboxInstance.ID;
	readonly metadata?: Readonly<Record<string, string>>;
}

export interface AppendPart {
	readonly id?: string; // uuidv7; generated when omitted
	readonly type: PartType;
	readonly status?: ToolStatus; // toolCall parts only
	readonly callId?: string; // toolCall parts only
	readonly toolName?: string; // toolCall parts only
	readonly data: string; // verbatim aikit part JSON
}

export interface AppendEntry {
	readonly id: string; // uuidv7; == aikit messageId for message types
	/**
	 * Settlement state. Defaults to `committed`, which is right for every entry
	 * that is complete when it is written — only a streamed assistant is created
	 * `draft` and settled later.
	 */
	readonly state?: EntryState;
	readonly sessionId: SessionSchema.ID;
	/**
	 * Position in the session's durable log — the sequence of the event that
	 * produced this entry. Must exceed every existing entry's seq, which the
	 * event log guarantees since sequences only ever advance.
	 */
	readonly seq: number;
	readonly type: EntryType;
	readonly data: string; // JSON: full payload, or message envelope
	readonly parts?: ReadonlyArray<AppendPart>; // order = partIndex; message types only
	readonly parentId?: string; // explicit branch appends only; default = current leaf
	/**
	 * Publish-time context from the event that produced this entry. The log does
	 * not store event metadata, so if it is to survive at all it survives here.
	 */
	readonly metadata?: Readonly<Record<string, string>>;
}

/**
 * Write one part at a known index, creating it or replacing what is there.
 *
 * The addressing unit is `(entryId, partIndex)` — aikit's own — so a block
 * completion writes into the slot the model assigned it, and a terminal that
 * re-states the whole array overwrites slot by slot rather than appending a
 * second copy. The unique index it relies on already exists
 * (`db/migrations.ts:193`).
 */
export interface UpsertPart {
	readonly id?: string; // uuidv7; generated when omitted
	readonly sessionId: SessionSchema.ID;
	readonly entryId: string;
	readonly partIndex: number;
	readonly type: PartType;
	readonly status?: ToolStatus; // toolCall parts only
	readonly callId?: string; // toolCall parts only
	readonly toolName?: string; // toolCall parts only
	readonly data: string; // verbatim aikit part JSON
}

/**
 * Promote an assistant entry from `aborted` to its terminal reason.
 *
 * The counterpart to creating the entry at `LLMStarted`: creation allocates the
 * array and charges nothing, finalization replaces the envelope with the
 * terminal one, reconciles the parts against it, and charges usage — exactly
 * once, because this is the only place that charges it for a streamed response.
 *
 * The envelope is immutable once finalized. A second call is rejected rather
 * than applied, so "the response settled twice" cannot silently double-charge.
 */
export interface FinalizeAssistant {
	readonly sessionId: SessionSchema.ID;
	readonly entryId: string;
	/**
	 * Where the entry lands. Not always settled: a terminal that leaves tool
	 * calls for the loop to run keeps it `draft`, because the turn is not over
	 * until those results are in.
	 */
	readonly state: EntryState;
	readonly data: string; // final aikit envelope JSON, without parts
	readonly parts: ReadonlyArray<AppendPart>; // authoritative; order = partIndex
}

/**
 * `pending -> running`, published immediately before a handler is invoked.
 *
 * Carries the running part rather than only a status because the promoted
 * column and the authoritative JSON have to agree — `decodeParts` re-checks
 * that on the way back, so writing one without the other would make the row
 * unreadable.
 */
export interface BeginToolCall {
	readonly sessionId: SessionSchema.ID; // ownership, checked not assumed
	readonly entryId: string; // owning assistant entry
	readonly callId: string;
	readonly toolName: string; // continuity: the call must still be the same tool
	readonly data: string; // running aikit part JSON, verbatim
}

export interface SettleToolCall {
	readonly sessionId: SessionSchema.ID; // ownership, checked not assumed
	readonly entryId: string; // owning assistant entry
	readonly callId: string; // from the loop's tool.execution.end
	readonly toolName: string; // continuity: the call must still be the same tool
	readonly status: Exclude<ToolStatus, "pending" | "running">;
	readonly data: string; // settled aikit part JSON, verbatim
}

// Fork copies ONE path (root → fork point) into a new session — never sibling
// or abandoned branches.
// clone = fork at the current leaf.
export interface ForkInput {
	readonly sessionId: SessionSchema.ID; // source session
	readonly entryId?: string; // fork point; default = current leaf (clone)
	readonly mode?: "at" | "before"; // default "at"; "before" valid only for user entries
	readonly id?: SessionSchema.ID; // new session id; generated when omitted
	readonly slug: string; // required — slugs are unique
	readonly title?: string; // default: source title
	readonly tag?: string; // default: source tag
}

// Entry row + its part rows, zipped. Message decode happens above this layer.
export interface HydratedEntry {
	readonly entry: SessionEntryRow;
	readonly parts: ReadonlyArray<SessionEntryPartRow>;
}

export interface Interface {
	readonly create: (input: CreateSession) => Effect.Effect<SessionRow>;
	readonly get: (sessionId: SessionSchema.ID) => Effect.Effect<Option.Option<SessionRow>>;
	readonly list: (input: { projectId: string }) => Effect.Effect<SessionRow[]>;
	readonly entry: (entryId: string) => Effect.Effect<Option.Option<HydratedEntry>>;
	/** Active root→leaf path with parts attached. */
	readonly path: (sessionId: SessionSchema.ID) => Effect.Effect<HydratedEntry[]>;
	/** All branches, newest-first, paged by entry seq. */
	readonly timeline: (input: {
		sessionId: SessionSchema.ID;
		cursorSeq?: number;
		limit?: number;
	}) => Effect.Effect<HydratedEntry[]>;
	/** Unsettled toolCall parts (crash recovery / live status). */
	readonly unsettled: (sessionId: SessionSchema.ID) => Effect.Effect<SessionEntryPartRow[]>;
	/**
	 * The newest assistant entry on the active path, with its parts.
	 *
	 * Not the leaf. `leaf_entry_id` is the tree cursor -- where the next append
	 * attaches -- so it moves to whatever was written last, including a config
	 * change or an annotation between turns. Callers asking "which assistant
	 * might still be in flight" want this instead, and it stays correct however
	 * many non-message entries were written after it.
	 *
	 * Path-scoped, so abandoned branches are excluded -- which is why the
	 * session-wide `unsettled` read is not a substitute.
	 */
	readonly latestAssistant: (sessionId: SessionSchema.ID) => Effect.Effect<Option.Option<HydratedEntry>>;
	/**
	 * One entry's toolCall parts in `partIndex` order.
	 *
	 * The loop's re-read: it schedules what is committed, not what the terminal
	 * message it holds in memory happens to say. Same data, but reading it from
	 * storage is what makes execution resumable and keeps one rule — execute what
	 * is durable — true in every phase.
	 */
	readonly toolCalls: (entryId: string) => Effect.Effect<SessionEntryPartRow[]>;
	/**
	 * The session's unfinished assistant, if it has one.
	 *
	 * The recovery sweep's whole question, in one indexed lookup. A session whose
	 * last turn finished cleanly answers `None` without reading an entry or
	 * decoding a message — which is the common case, on every drain.
	 */
	readonly latestDraft: (sessionId: SessionSchema.ID) => Effect.Effect<Option.Option<HydratedEntry>>;
	/**
	 * Settle a draft without touching its envelope.
	 *
	 * What closes a turn once its tool results are in — the envelope was already
	 * written by the terminal, and there is nothing new to say about it. Also how
	 * the recovery sweep closes a draft no terminal will ever settle.
	 *
	 * `false` means the entry exists and was already settled, which is ordinary: a
	 * terminal that left no tool work settles the entry itself. A *missing* entry
	 * is not ordinary and fails, so the two cannot be confused by a caller that
	 * ignores the first.
	 */
	readonly closeDraft: (input: {
		readonly sessionId: SessionSchema.ID;
		readonly entryId: string;
		readonly state: Exclude<EntryState, "draft">;
	}) => Effect.Effect<boolean, EntryNotFoundError>;
	readonly append: (
		input: AppendEntry,
	) => Effect.Effect<SessionEntryRow, SessionNotFoundError | EntryNotFoundError | InvalidEntryDataError>;
	/** Write one part at `(entryId, partIndex)`, creating or replacing it. */
	readonly upsertPart: (input: UpsertPart) => Effect.Effect<void, EntryNotFoundError | InvalidEntryDataError>;
	/** Promote a streamed assistant entry to its terminal envelope; charges usage. */
	readonly finalizeAssistant: (
		input: FinalizeAssistant,
	) => Effect.Effect<void, EntryNotFoundError | InvalidEntryDataError>;
	/** `pending -> running`; the only transition into `running`. */
	readonly beginToolCall: (input: BeginToolCall) => Effect.Effect<void, ToolCallNotFoundError | ToolCallConflictError>;
	readonly settleToolCall: (
		input: SettleToolCall,
	) => Effect.Effect<void, ToolCallNotFoundError | ToolCallConflictError>;
	/** Copy root→fork-point into a new session (`parentId` = source). Clone = fork at leaf. */
	readonly fork: (
		input: ForkInput,
	) => Effect.Effect<SessionRow, SessionNotFoundError | EntryNotFoundError | InvalidEntryDataError>;
	/** Move the leaf cursor; the next append forks a sibling branch. */
	readonly branch: (input: {
		sessionId: SessionSchema.ID;
		entryId: string;
	}) => Effect.Effect<void, EntryNotFoundError>;
	/** Set or clear (null) the bookmark label on an entry. */
	readonly setLabel: (input: {
		sessionId: SessionSchema.ID;
		entryId: string;
		label: string | null;
	}) => Effect.Effect<void, EntryNotFoundError>;
}

export class Service extends Context.Service<Service, Interface>()("@codeworksh/harness/session/session/Service") {}

const messageTypes: ReadonlySet<string> = new Set(messageEntryTypes);

// Structural decode of the persisted envelope's usage; failures surface as
// InvalidEntryDataError, never a defect — callers (Context Manager, UI) can act.
const decodeEnvelopeUsage = Schema.decodeUnknownEffect(SessionSchema.AssistantEnvelopeUsage);
const decodeMessageEnvelopeIdentity = Schema.decodeUnknownEffect(SessionSchema.MessageEnvelopeIdentity);
const decodeCompactionData = Schema.decodeUnknownEffect(SessionSchema.CompactionData);
const decodeJsonObject = Schema.decodeUnknownEffect(SessionSchema.JsonObject);
const encodeJsonObject = Schema.encodeEffect(SessionSchema.JsonObject);

export const layer = Layer.effect(
	Service,
	Effect.gen(function* () {
		const sql = yield* SqlClient.SqlClient;
		const events = yield* Event.Service;
		const findSession = SqlSchema.findOneOption({
			Request: Schema.String,
			Result: SessionRow,
			execute: (id) => sql`SELECT * FROM session WHERE id = ${id}`,
		});

		const selectSessions = SqlSchema.findAll({
			Request: Schema.String,
			Result: SessionRow,
			execute: (projectId) => sql`SELECT * FROM session WHERE project_id = ${projectId} ORDER BY updated_at DESC`,
		});

		const insertSession = SqlSchema.void({
			Request: SessionRow.insert,
			execute: (row) => sql`INSERT INTO session ${sql.insert(row)}`,
		});

		const findEntry = SqlSchema.findOneOption({
			Request: Schema.String,
			Result: SessionEntryRow,
			execute: (id) => sql`SELECT * FROM session_entry WHERE id = ${id}`,
		});

		// seq is supplied by the caller, not computed here: it is the sequence of
		// the event that produced the entry, so the durable log and the tree share
		// one position space.
		//
		// The WHERE NOT EXISTS keeps positions strictly advancing. `max(seq) + 1`
		// used to make that structural; sourcing seq from outside would otherwise
		// downgrade it to an unchecked contract, and a stale position fails
		// silently — `selectPath` orders by seq, so a child would sort above its
		// own parent. Inserting zero rows is the signal, checked by the callers.
		const insertEntry = SqlSchema.findAll({
			Request: SessionEntryRow.insert,
			Result: Schema.Struct({ id: Schema.String }),
			execute: (row) => sql`
				INSERT INTO session_entry (id, session_id, parent_id, seq, type, state, data, label, metadata, created_at, updated_at)
				SELECT
					${row.id}, ${row.sessionId}, ${row.parentId ?? null}, ${row.seq},
					${row.type}, ${row.state}, ${row.data}, ${row.label ?? null}, ${row.metadata ?? null},
					${row.createdAt}, ${row.updatedAt}
				WHERE NOT EXISTS (
					SELECT 1 FROM session_entry WHERE session_id = ${row.sessionId} AND seq >= ${row.seq}
				)
				RETURNING id
			`,
		});

		// Single-row insert: toolCall and non-tool parts have different encoded
		// key sets (FieldOption omits absent keys), so multi-row sql.insert would
		// mix shapes. Parts per message are few; sequential inserts inside the
		// append transaction are fine.
		const insertPart = SqlSchema.void({
			Request: SessionEntryPartRow.insert,
			execute: (row) => sql`INSERT INTO session_entry_part ${sql.insert(row)}`,
		});

		// Root→leaf in one query: a parent always exists before its child, so a
		// parent's seq is always smaller and ORDER BY seq returns root-first.
		const selectPath = SqlSchema.findAll({
			Request: Schema.String,
			Result: SessionEntryRow,
			execute: (leafEntryId) => sql`
				WITH RECURSIVE path AS (
					SELECT * FROM session_entry WHERE id = ${leafEntryId}
					UNION ALL
					SELECT e.* FROM session_entry e
					JOIN path p ON e.id = p.parent_id AND e.session_id = p.session_id
				)
				SELECT * FROM path ORDER BY seq
			`,
		});

		// Recovery/import fallback when session.leaf_entry_id is NULL (§4.1).
		const selectLatestEntry = SqlSchema.findOneOption({
			Request: Schema.String,
			Result: SessionEntryRow,
			execute: (sessionId) => sql`
				SELECT * FROM session_entry WHERE session_id = ${sessionId} ORDER BY seq DESC LIMIT 1
			`,
		});

		const selectTimelinePage = SqlSchema.findAll({
			Request: Schema.Struct({
				sessionId: Schema.String,
				cursorSeq: Schema.NullOr(Schema.Int),
				limit: Schema.Int,
			}),
			Result: SessionEntryRow,
			execute: (r) => sql`
				SELECT * FROM session_entry
				WHERE session_id = ${r.sessionId} AND (${r.cursorSeq} IS NULL OR seq < ${r.cursorSeq})
				ORDER BY seq DESC LIMIT ${r.limit}
			`,
		});

		const selectPartsForEntries = SqlSchema.findAll({
			Request: Schema.Array(Schema.String),
			Result: SessionEntryPartRow,
			execute: (entryIds) => sql`
				SELECT * FROM session_entry_part
				WHERE ${sql.in("entry_id", entryIds)}
				ORDER BY entry_id, part_index
			`,
		});

		const selectUnsettled = SqlSchema.findAll({
			Request: Schema.String,
			Result: SessionEntryPartRow,
			execute: (sessionId) => sql`
				SELECT * FROM session_entry_part
				WHERE session_id = ${sessionId} AND status IN ('pending', 'running')
				ORDER BY entry_id, part_index
			`,
		});

		// The path walk, stopped at the newest assistant. Same recursive CTE as
		// `selectPath`, filtered and limited in SQL rather than by reading the whole
		// path back and scanning it here.
		const selectLatestAssistant = SqlSchema.findOneOption({
			Request: Schema.String,
			Result: SessionEntryRow,
			execute: (leafEntryId) => sql`
				WITH RECURSIVE path AS (
					SELECT * FROM session_entry WHERE id = ${leafEntryId}
					UNION ALL
					SELECT e.* FROM session_entry e
					JOIN path p ON e.id = p.parent_id AND e.session_id = p.session_id
				)
				SELECT * FROM path WHERE type = 'assistant' ORDER BY seq DESC LIMIT 1
			`,
		});

		// Newest first: a session has at most one draft in practice, and taking the
		// newest is the right answer if an older one was ever stranded.
		const selectLatestDraft = SqlSchema.findOneOption({
			Request: Schema.String,
			Result: SessionEntryRow,
			execute: (sessionId) => sql`
				SELECT * FROM session_entry
				WHERE session_id = ${sessionId} AND state = 'draft'
				ORDER BY seq DESC LIMIT 1
			`,
		});

		const selectToolCalls = SqlSchema.findAll({
			Request: Schema.String,
			Result: SessionEntryPartRow,
			execute: (entryId) => sql`
				SELECT * FROM session_entry_part
				WHERE entry_id = ${entryId} AND type = 'toolCall'
				ORDER BY part_index
			`,
		});

		const epochNow = Effect.map(DateTime.now, DateTime.toEpochMillis);

		// Zip rule (§7.2): group parts by entry id, attach while walking entries.
		// entry.type says whether parts are expected; a mismatch is corruption,
		// not an edge case (§10.4).
		const hydrate = (
			entries: ReadonlyArray<SessionEntryRow>,
			parts: ReadonlyArray<SessionEntryPartRow>,
		): HydratedEntry[] => {
			const byEntry = new Map<string, SessionEntryPartRow[]>();
			for (const part of parts) {
				const bucket = byEntry.get(part.entryId);
				if (bucket) bucket.push(part);
				else byEntry.set(part.entryId, [part]);
			}
			return entries.map((entry) => ({ entry, parts: byEntry.get(entry.id) ?? [] }));
		};

		const partsFor = Effect.fnUntraced(function* (entries: ReadonlyArray<SessionEntryRow>) {
			const ids = entries.filter((entry) => messageTypes.has(entry.type)).map((entry) => entry.id);
			if (ids.length === 0) return [];
			return yield* selectPartsForEntries(ids).pipe(Effect.orDie);
		});

		const create = Effect.fn("Session.create")(function* (input: CreateSession) {
			const id = input.id ?? SessionSchema.ID.create();
			const row = yield* SessionRow.insert
				.makeEffect({
					id,
					projectId: input.projectId,
					parentId: Option.fromUndefinedOr(input.parentId),
					slug: input.slug,
					directory: input.directory,
					title: input.title,
					tag: Option.fromUndefinedOr(input.tag),
					sandboxInstanceId: SandboxInstance.toField(input.sandboxInstanceId ?? SandboxInstance.ID.local),
					metadata: Option.fromUndefinedOr(input.metadata as Record<string, string> | undefined),
					leafEntryId: Option.none(),
				})
				.pipe(Effect.orDie);
			yield* insertSession(row).pipe(Effect.orDie);
			const created = yield* findSession(id).pipe(Effect.orDie);
			if (Option.isNone(created)) {
				return yield* Effect.die(new Error("session insert did not persist"));
			}
			return created.value;
		});

		const get = Effect.fn("Session.get")(function* (sessionId: string) {
			return yield* findSession(sessionId).pipe(Effect.orDie);
		});

		const list = Effect.fn("Session.list")(function* (input: { projectId: string }) {
			return yield* selectSessions(input.projectId).pipe(Effect.orDie);
		});

		const entry = Effect.fn("Session.entry")(function* (entryId: string) {
			const found = yield* findEntry(entryId).pipe(Effect.orDie);
			if (Option.isNone(found)) return Option.none<HydratedEntry>();
			const parts = yield* partsFor([found.value]);
			return Option.some(hydrate([found.value], parts)[0]!);
		});

		const resolveLeaf = Effect.fnUntraced(function* (session: SessionRow) {
			if (Option.isSome(session.leafEntryId)) return session.leafEntryId;
			const latest = yield* selectLatestEntry(session.id).pipe(Effect.orDie);
			return Option.map(latest, (row) => row.id);
		});

		const path = Effect.fn("Session.path")(function* (sessionId: string) {
			const session = yield* findSession(sessionId).pipe(Effect.orDie);
			if (Option.isNone(session)) return [];
			const leaf = yield* resolveLeaf(session.value);
			if (Option.isNone(leaf)) return [];
			const entries = yield* selectPath(leaf.value).pipe(Effect.orDie);
			const parts = yield* partsFor(entries);
			return hydrate(entries, parts);
		});

		const timeline = Effect.fn("Session.timeline")(function* (input: {
			sessionId: string;
			cursorSeq?: number;
			limit?: number;
		}) {
			const entries = yield* selectTimelinePage({
				sessionId: input.sessionId,
				cursorSeq: input.cursorSeq ?? null,
				limit: input.limit ?? 50,
			}).pipe(Effect.orDie);
			const parts = yield* partsFor(entries);
			return hydrate(entries, parts);
		});

		const unsettled = Effect.fn("Session.unsettled")(function* (sessionId: string) {
			return yield* selectUnsettled(sessionId).pipe(Effect.orDie);
		});

		const latestAssistant = Effect.fn("Session.latestAssistant")(function* (sessionId: SessionSchema.ID) {
			const session = yield* findSession(sessionId).pipe(Effect.orDie);
			if (Option.isNone(session)) return Option.none<HydratedEntry>();
			const leaf = yield* resolveLeaf(session.value);
			if (Option.isNone(leaf)) return Option.none<HydratedEntry>();
			const found = yield* selectLatestAssistant(leaf.value).pipe(Effect.orDie);
			if (Option.isNone(found)) return Option.none<HydratedEntry>();
			const parts = yield* partsFor([found.value]);
			return Option.some(hydrate([found.value], parts)[0]!);
		});

		const toolCalls = Effect.fn("Session.toolCalls")(function* (entryId: string) {
			return yield* selectToolCalls(entryId).pipe(Effect.orDie);
		});

		const latestDraft = Effect.fn("Session.latestDraft")(function* (sessionId: SessionSchema.ID) {
			const found = yield* selectLatestDraft(sessionId).pipe(Effect.orDie);
			if (Option.isNone(found)) return Option.none<HydratedEntry>();
			const parts = yield* partsFor([found.value]);
			return Option.some(hydrate([found.value], parts)[0]!);
		});

		/*
		 * State only; the envelope is left exactly as it was. A turn closing after
		 * its tools settled has nothing new to say about the message, and a killed
		 * turn produced no terminal to say it with — inventing one would put words
		 * in the provider's mouth.
		 */
		const closeDraft = Effect.fn("Session.closeDraft")(function* (input: {
			readonly sessionId: SessionSchema.ID;
			readonly entryId: string;
			readonly state: Exclude<EntryState, "draft">;
		}) {
			const now = yield* epochNow;
			return yield* sql
				.withTransaction(
					Effect.gen(function* () {
						const owner = yield* findEntry(input.entryId);
						if (Option.isNone(owner) || owner.value.sessionId !== input.sessionId) return undefined;
						if (owner.value.state !== "draft") return false;
						yield* sql`
							UPDATE session_entry SET state = ${input.state}, updated_at = ${now}
							WHERE id = ${input.entryId}
						`;
						return true;
					}),
				)
				.pipe(
					Effect.orDie,
					Effect.flatMap((closed) =>
						closed === undefined
							? new EntryNotFoundError({ sessionId: input.sessionId, entryId: input.entryId })
							: Effect.succeed(closed),
					),
				);
		});

		type AppendTxResult =
			| { readonly _tag: "sessionNotFound" }
			| { readonly _tag: "parentNotFound" }
			| { readonly _tag: "invalidData"; readonly reason: string }
			| { readonly _tag: "inserted"; readonly entry: SessionEntryRow };

		const append = Effect.fn("Session.append")(function* (input: AppendEntry) {
			const parts = input.parts ?? [];
			if (parts.length > 0 && !messageTypes.has(input.type)) {
				return yield* new InvalidEntryDataError({
					entryId: input.id,
					type: input.type,
					reason: `entry type "${input.type}" must not carry parts`,
				});
			}

			// Part data is documented as verbatim aikit JSON and every reader parses
			// it as such, so it is validated here for the same reason the envelope
			// is: an unparseable part is only discovered by whoever reads it back,
			// long after the write that could have rejected it.
			for (const [index, part] of parts.entries()) {
				yield* decodeJsonObject(part.data).pipe(
					Effect.mapError(
						(error) =>
							new InvalidEntryDataError({
								entryId: input.id,
								type: input.type,
								reason: `part ${index} ("${part.type}") is not a JSON object: ${error.message}`,
							}),
					),
				);
			}

			const envelopeIdentity = messageTypes.has(input.type)
				? yield* decodeMessageEnvelopeIdentity(input.data).pipe(
						Effect.mapError(
							(error) =>
								new InvalidEntryDataError({ entryId: input.id, type: input.type, reason: error.message }),
						),
					)
				: undefined;
			if (envelopeIdentity !== undefined && envelopeIdentity.messageId !== input.id) {
				return yield* new InvalidEntryDataError({
					entryId: input.id,
					type: input.type,
					reason: `messageId "${envelopeIdentity.messageId}" does not match entry id`,
				});
			}

			const compaction =
				input.type === "compaction"
					? yield* decodeCompactionData(input.data).pipe(
							Effect.mapError(
								(error) =>
									new InvalidEntryDataError({ entryId: input.id, type: input.type, reason: error.message }),
							),
						)
					: undefined;

			// The envelope is authoritative; aggregate deltas are never accepted
			// independently of the data persisted by this transaction.
			const usage =
				input.type === "assistant"
					? yield* decodeEnvelopeUsage(input.data).pipe(
							Effect.map((envelope) => envelope.usage),
							Effect.mapError(
								(error) =>
									new InvalidEntryDataError({ entryId: input.id, type: input.type, reason: error.message }),
							),
						)
					: undefined;

			const result: AppendTxResult = yield* sql
				.withTransaction(
					Effect.gen(function* () {
						const session = yield* findSession(input.sessionId);
						if (Option.isNone(session)) return { _tag: "sessionNotFound" } as const;
						const currentLeaf = yield* resolveLeaf(session.value);

						// Append anchor: explicit branch target
						// (validated same-session; the composite FK is the schema backstop),
						// else the current leaf resolved with the same fallback reads use (§4.1).
						let parentId: Option.Option<string>;
						if (input.parentId !== undefined) {
							const rows = yield* sql`
								SELECT id FROM session_entry
								WHERE id = ${input.parentId} AND session_id = ${input.sessionId}
							`;
							if (rows.length === 0) return { _tag: "parentNotFound" } as const;
							parentId = Option.some(input.parentId);
						} else {
							parentId = currentLeaf;
						}

						if (compaction !== undefined && compaction.firstKeptEntryId !== null) {
							if (Option.isNone(parentId)) {
								return {
									_tag: "invalidData",
									reason: `firstKeptEntryId "${compaction.firstKeptEntryId}" is not on the compaction path`,
								} as const;
							}
							const compactionPath = yield* selectPath(parentId.value);
							if (!compactionPath.some((entry) => entry.id === compaction.firstKeptEntryId)) {
								return {
									_tag: "invalidData",
									reason: `firstKeptEntryId "${compaction.firstKeptEntryId}" is not on the compaction path`,
								} as const;
							}
						}

						const entryRow = yield* SessionEntryRow.insert.makeEffect({
							id: input.id,
							sessionId: input.sessionId,
							parentId,
							seq: input.seq,
							type: input.type,
							state: input.state ?? "committed",
							data: input.data,
							label: Option.none(),
							metadata: Option.fromUndefinedOr(input.metadata),
						});
						const inserted = yield* insertEntry(entryRow);
						// Keep the aggregate head above what was just written. An entry
						// that arrived without an event still occupies its position, and
						// the next event must land above it or hit the guard below
						// forever.
						yield* events.advance(input.sessionId, input.seq);
						if (inserted.length === 0) {
							return {
								_tag: "invalidData",
								reason: `seq ${input.seq} does not advance past the session's latest entry`,
							} as const;
						}

						for (const [partIndex, part] of parts.entries()) {
							const partRow = yield* SessionEntryPartRow.insert.makeEffect({
								id: part.id ?? uuidv7(),
								entryId: input.id,
								sessionId: input.sessionId,
								partIndex,
								type: part.type,
								status: Option.fromUndefinedOr(part.status),
								callId: Option.fromUndefinedOr(part.callId),
								toolName: Option.fromUndefinedOr(part.toolName),
								data: part.data,
							});
							yield* insertPart(partRow);
						}

						// Advance the cursor; assistant appends bump the usage aggregates
						// in the same statement (envelope usage is final at message.end).
						const now = yield* epochNow;
						yield* sql`
							UPDATE session SET
								leaf_entry_id = ${input.id},
								updated_at = ${now},
								cost = cost + ${usage?.cost.total ?? 0},
								tokens_input = tokens_input + ${usage?.input ?? 0},
								tokens_output = tokens_output + ${usage?.output ?? 0},
								tokens_cache_read = tokens_cache_read + ${usage?.cacheRead ?? 0},
								tokens_cache_write = tokens_cache_write + ${usage?.cacheWrite ?? 0}
							WHERE id = ${input.sessionId}
						`;

						const entry = yield* findEntry(input.id);
						if (Option.isNone(entry)) {
							return yield* Effect.die(new Error("session append: inserted entry did not persist"));
						}
						return { _tag: "inserted", entry: entry.value } as const;
					}),
				)
				.pipe(Effect.orDie);

			switch (result._tag) {
				case "sessionNotFound":
					return yield* new SessionNotFoundError({ sessionId: input.sessionId });
				case "parentNotFound":
					return yield* new EntryNotFoundError({ sessionId: input.sessionId, entryId: input.parentId ?? "" });
				case "invalidData":
					return yield* new InvalidEntryDataError({
						entryId: input.id,
						type: input.type,
						reason: result.reason,
					});
				case "inserted":
					return result.entry;
			}
		});

		// ON CONFLICT over the unique `(entry_id, part_index)` index. `id` is
		// preserved on update: a slot rewritten by the terminal is the same part
		// the block completion wrote, and rotating its row id would break anything
		// holding a reference to it.
		const upsertPartRow = SqlSchema.void({
			Request: SessionEntryPartRow.insert,
			execute: (row) => sql`
				INSERT INTO session_entry_part ${sql.insert(row)}
				ON CONFLICT (entry_id, part_index) DO UPDATE SET
					type = excluded.type,
					status = excluded.status,
					call_id = excluded.call_id,
					tool_name = excluded.tool_name,
					data = excluded.data,
					updated_at = excluded.updated_at
			`,
		});

		type PartTxResult =
			| { readonly _tag: "entryNotFound" }
			| { readonly _tag: "invalidData"; readonly reason: string }
			| { readonly _tag: "written" };

		const writePart = Effect.fnUntraced(function* (input: UpsertPart) {
			const row = yield* SessionEntryPartRow.insert.makeEffect({
				id: input.id ?? uuidv7(),
				entryId: input.entryId,
				sessionId: input.sessionId,
				partIndex: input.partIndex,
				type: input.type,
				status: Option.fromUndefinedOr(input.status),
				callId: Option.fromUndefinedOr(input.callId),
				toolName: Option.fromUndefinedOr(input.toolName),
				data: input.data,
			});
			yield* upsertPartRow(row);
		});

		const upsertPart = Effect.fn("Session.upsertPart")(function* (input: UpsertPart) {
			yield* decodeJsonObject(input.data).pipe(
				Effect.mapError(
					(error) =>
						new InvalidEntryDataError({
							entryId: input.entryId,
							type: input.type,
							reason: `part ${input.partIndex} ("${input.type}") is not a JSON object: ${error.message}`,
						}),
				),
			);

			const result: PartTxResult = yield* sql
				.withTransaction(
					Effect.gen(function* () {
						const owner = yield* findEntry(input.entryId);
						if (Option.isNone(owner) || owner.value.sessionId !== input.sessionId) {
							return { _tag: "entryNotFound" } as const;
						}
						if (!messageTypes.has(owner.value.type)) {
							return {
								_tag: "invalidData",
								reason: `entry type "${owner.value.type}" must not carry parts`,
							} as const;
						}
						yield* writePart(input);
						return { _tag: "written" } as const;
					}),
				)
				.pipe(Effect.orDie);

			if (result._tag === "entryNotFound") {
				return yield* new EntryNotFoundError({ sessionId: input.sessionId, entryId: input.entryId });
			}
			if (result._tag === "invalidData") {
				return yield* new InvalidEntryDataError({
					entryId: input.entryId,
					type: input.type,
					reason: result.reason,
				});
			}
		});

		const finalizeAssistant = Effect.fn("Session.finalizeAssistant")(function* (input: FinalizeAssistant) {
			for (const [index, part] of input.parts.entries()) {
				yield* decodeJsonObject(part.data).pipe(
					Effect.mapError(
						(error) =>
							new InvalidEntryDataError({
								entryId: input.entryId,
								type: "assistant",
								reason: `part ${index} ("${part.type}") is not a JSON object: ${error.message}`,
							}),
					),
				);
			}
			const identity = yield* decodeMessageEnvelopeIdentity(input.data).pipe(
				Effect.mapError(
					(error) =>
						new InvalidEntryDataError({ entryId: input.entryId, type: "assistant", reason: error.message }),
				),
			);
			if (identity.messageId !== input.entryId) {
				return yield* new InvalidEntryDataError({
					entryId: input.entryId,
					type: "assistant",
					reason: `messageId "${identity.messageId}" does not match entry id`,
				});
			}
			const usage = yield* decodeEnvelopeUsage(input.data).pipe(
				Effect.map((envelope) => envelope.usage),
				Effect.mapError(
					(error) =>
						new InvalidEntryDataError({ entryId: input.entryId, type: "assistant", reason: error.message }),
				),
			);

			const result: PartTxResult = yield* sql
				.withTransaction(
					Effect.gen(function* () {
						const owner = yield* findEntry(input.entryId);
						if (Option.isNone(owner) || owner.value.sessionId !== input.sessionId) {
							return { _tag: "entryNotFound" } as const;
						}
						if (owner.value.type !== "assistant") {
							return {
								_tag: "invalidData",
								reason: `entry type "${owner.value.type}" is not an assistant`,
							} as const;
						}
						/*
						 * One question, answerable. Finality used to be inferred from
						 * `stopReason !== "aborted"`, which cannot work: a draft placeholder
						 * and an aborted terminal write the same value, so an aborted turn
						 * could be finalized twice and charged twice. `status` is the
						 * harness's own vocabulary and says exactly this.
						 */
						if (owner.value.state !== "draft") {
							return {
								_tag: "invalidData",
								reason: `assistant is already settled as "${owner.value.state}"`,
							} as const;
						}

						const now = yield* epochNow;
						yield* sql`
							UPDATE session_entry SET data = ${input.data}, state = ${input.state}, updated_at = ${now}
							WHERE id = ${input.entryId}
						`;
						/*
						 * Replace the whole array rather than upsert over it.
						 *
						 * `session_entry_part` has two unique indexes — `(entry_id,
						 * part_index)` and `(entry_id, call_id)` — and an upsert can only
						 * name one conflict target. Writing over the old rows therefore
						 * collided whenever a call's position moved: its own surviving row
						 * still held the `call_id`, the insert hit the *other* index, and
						 * the whole terminal transaction died, permanently and every time.
						 *
						 * `call_id` is the identity; the index is just where it sits. So
						 * the safe write is one that cannot have a stale row to collide
						 * with. This is reachable only at finalization, which happens once
						 * and before any tool runs, so nothing settled is being discarded.
						 */
						yield* sql`DELETE FROM session_entry_part WHERE entry_id = ${input.entryId}`;
						for (const [partIndex, part] of input.parts.entries()) {
							yield* writePart({ ...part, sessionId: input.sessionId, entryId: input.entryId, partIndex });
						}
						yield* sql`
							UPDATE session SET
								updated_at = ${now},
								cost = cost + ${usage.cost.total},
								tokens_input = tokens_input + ${usage.input},
								tokens_output = tokens_output + ${usage.output},
								tokens_cache_read = tokens_cache_read + ${usage.cacheRead},
								tokens_cache_write = tokens_cache_write + ${usage.cacheWrite}
							WHERE id = ${input.sessionId}
						`;
						return { _tag: "written" } as const;
					}),
				)
				.pipe(Effect.orDie);

			if (result._tag === "entryNotFound") {
				return yield* new EntryNotFoundError({ sessionId: input.sessionId, entryId: input.entryId });
			}
			if (result._tag === "invalidData") {
				return yield* new InvalidEntryDataError({
					entryId: input.entryId,
					type: "assistant",
					reason: result.reason,
				});
			}
		});

		/*
		 * Both tool-call transitions write the same two columns at the same
		 * address, so they are one statement with a different status rather than
		 * two near-identical ones free to drift. The transition guard, ownership
		 * check, and conflict rollback belong to the settlement-conflict work, not
		 * here.
		 */
		type MoveTxResult =
			| { readonly _tag: "notFound" }
			| { readonly _tag: "conflict"; readonly reason: string }
			| { readonly _tag: "moved" };

		/**
		 * Both tool-call transitions write the same two columns at the same address,
		 * so they are one statement with a different target status.
		 *
		 * What it validates before writing, and why each matters:
		 *
		 * - **ownership** — the part must belong to the session being written on
		 *   behalf of. Nothing constructs a cross-session write today; the check
		 *   exists so that if something ever does, it fails here rather than
		 *   silently settling a stranger's call.
		 * - **continuity** — the tool name must still be the one the call was
		 *   finalized with. A mismatch means two different calls are being treated
		 *   as one.
		 * - **the transition itself** — a settled call has left the lifecycle, and
		 *   letting a late event write over it would make the terminal a matter of
		 *   arrival order.
		 *
		 * A duplicate is a conflict, not something to absorb. The commit path dies
		 * on a repeated event id before any projector runs (`event/event.ts`), and
		 * nothing publishes the same logical event twice — so a second identical
		 * write means an assumption broke, and failing loudly beats swallowing it.
		 */
		const moveToolCall = Effect.fnUntraced(function* (input: {
			readonly sessionId: SessionSchema.ID;
			readonly entryId: string;
			readonly callId: string;
			readonly toolName: string;
			readonly status: ToolStatus;
			readonly data: string;
			readonly from: ReadonlySet<ToolStatus>;
		}) {
			const result: MoveTxResult = yield* sql
				.withTransaction(
					Effect.gen(function* () {
						const rows = yield* sql`
							SELECT session_id, status, tool_name, data FROM session_entry_part
							WHERE entry_id = ${input.entryId} AND call_id = ${input.callId} AND type = 'toolCall'
						`;
						// The client maps result names to camelCase (`db/db.ts:37`).
						const row = rows[0] as
							| { sessionId: string; status: string; toolName: string | null; data: string }
							| undefined;
						if (row === undefined) return { _tag: "notFound" } as const;

						if (row.sessionId !== input.sessionId) {
							return {
								_tag: "conflict",
								reason: `call belongs to session "${row.sessionId}", not "${input.sessionId}"`,
							} as const;
						}
						if (row.toolName !== null && row.toolName !== input.toolName) {
							return {
								_tag: "conflict",
								reason: `call was finalized as "${row.toolName}" and is being settled as "${input.toolName}"`,
							} as const;
						}
						if (!input.from.has(row.status as ToolStatus)) {
							return {
								_tag: "conflict",
								reason: `call is "${row.status}" and cannot move to "${input.status}"`,
							} as const;
						}

						const now = yield* epochNow;
						yield* sql`
							UPDATE session_entry_part
							SET data = ${input.data}, status = ${input.status}, updated_at = ${now}
							WHERE entry_id = ${input.entryId} AND call_id = ${input.callId} AND type = 'toolCall'
						`;
						return { _tag: "moved" } as const;
					}),
				)
				.pipe(Effect.orDie);

			if (result._tag === "notFound") {
				return yield* new ToolCallNotFoundError({ entryId: input.entryId, callId: input.callId });
			}
			if (result._tag === "conflict") {
				return yield* new ToolCallConflictError({
					entryId: input.entryId,
					callId: input.callId,
					reason: result.reason,
				});
			}
		});

		const pendingOnly: ReadonlySet<ToolStatus> = new Set(["pending"]);
		const unsettledOnly: ReadonlySet<ToolStatus> = new Set(["pending", "running"]);

		const beginToolCall = Effect.fn("Session.beginToolCall")(function* (input: BeginToolCall) {
			yield* moveToolCall({ ...input, status: "running", from: pendingOnly });
		});

		/*
		 * `pending` is accepted as well as `running`: recovery settles calls that
		 * never started, and the interrupted-batch finalizer settles calls that were
		 * never admitted. Both are legitimate `pending -> terminal` moves.
		 */
		const settleToolCall = Effect.fn("Session.settleToolCall")(function* (input: SettleToolCall) {
			yield* moveToolCall({ ...input, from: unsettledOnly });
		});

		type ForkTxResult =
			| { readonly _tag: "sessionNotFound" }
			| { readonly _tag: "entryNotFound"; readonly entryId: string }
			| { readonly _tag: "invalidMode"; readonly entryId: string; readonly type: string }
			| { readonly _tag: "invalidData"; readonly entryId: string; readonly type: string; readonly reason: string }
			| { readonly _tag: "forked"; readonly session: SessionRow };

		// Copied entries get NEW ids (ours are globally unique),
		// so every entry-id reference on the path is remapped:
		// parent edges, message envelopes' messageId, and compaction's
		// `firstKeptEntryId` (when non-null, always on the copied path and mappable).
		// branchSummary.fromEntryId is NOT remapped — it points at an abandoned
		// leaf that is deliberately not copied;
		const rewriteForkedData = Effect.fnUntraced(function* (
			entry: SessionEntryRow,
			newId: string,
			idMap: ReadonlyMap<string, string>,
		) {
			if (!messageTypes.has(entry.type) && entry.type !== "compaction") return entry.data;
			const invalidData = (reason: string) =>
				new InvalidEntryDataError({ entryId: entry.id, type: entry.type, reason });
			const payload: Record<string, unknown> = {
				...(yield* decodeJsonObject(entry.data).pipe(Effect.mapError((error) => invalidData(error.message)))),
			};
			if (messageTypes.has(entry.type)) {
				const identity = yield* decodeMessageEnvelopeIdentity(entry.data).pipe(
					Effect.mapError((error) => invalidData(error.message)),
				);
				if (identity.messageId !== entry.id) {
					return yield* invalidData(`messageId "${identity.messageId}" does not match entry id`);
				}
				payload["messageId"] = newId;
			}
			if (entry.type === "compaction") {
				const compaction = yield* decodeCompactionData(entry.data).pipe(
					Effect.mapError((error) => invalidData(error.message)),
				);
				if (compaction.firstKeptEntryId !== null) {
					const mapped = idMap.get(compaction.firstKeptEntryId);
					if (mapped === undefined) {
						return yield* invalidData(
							`firstKeptEntryId "${compaction.firstKeptEntryId}" is not on the copied path`,
						);
					}
					payload["firstKeptEntryId"] = mapped;
				}
			}
			return yield* encodeJsonObject(payload).pipe(Effect.mapError((error) => invalidData(error.message)));
		});

		const fork = Effect.fn("Session.fork")(function* (input: ForkInput) {
			const mode = input.mode ?? "at";
			const newSessionId = input.id ?? SessionSchema.ID.create();

			const result: ForkTxResult = yield* sql
				.withTransaction(
					Effect.gen(function* () {
						const source = yield* findSession(input.sessionId);
						if (Option.isNone(source)) return { _tag: "sessionNotFound" } as const;

						// Resolve the fork-point entry: explicit (validated in-session)
						// or the current leaf (clone). None = empty source.
						let targetEntry: Option.Option<SessionEntryRow>;
						if (input.entryId !== undefined) {
							const found = yield* findEntry(input.entryId);
							if (Option.isNone(found) || found.value.sessionId !== input.sessionId) {
								return { _tag: "entryNotFound", entryId: input.entryId } as const;
							}
							targetEntry = found;
						} else {
							const leaf = yield* resolveLeaf(source.value);
							targetEntry = Option.isNone(leaf) ? Option.none() : yield* findEntry(leaf.value);
						}

						// mode "before": land just above a user prompt so the caller can
						// re-edit and resubmit it in the fork. Parent may be none (root prompt) — that yields an empty fork.
						let targetId: Option.Option<string>;
						if (Option.isSome(targetEntry) && mode === "before") {
							if (targetEntry.value.type !== "user") {
								return {
									_tag: "invalidMode",
									entryId: targetEntry.value.id,
									type: targetEntry.value.type,
								} as const;
							}
							targetId = targetEntry.value.parentId;
						} else {
							targetId = Option.map(targetEntry, (entry) => entry.id);
						}

						const pathEntries = Option.isSome(targetId) ? yield* selectPath(targetId.value) : [];
						const idMap = new Map(pathEntries.map((entry) => [entry.id, uuidv7()]));

						// Prepare the complete copy before the first write. Returning a tagged
						// validation error after an insert would otherwise commit a partial fork.
						const prepared = yield* Effect.result(
							Effect.forEach(pathEntries, (entry) => {
								const id = idMap.get(entry.id)!;
								return rewriteForkedData(entry, id, idMap).pipe(
									Effect.map((data) => ({ source: entry, id, data })),
								);
							}),
						);
						if (Result.isFailure(prepared)) {
							return {
								_tag: "invalidData",
								entryId: prepared.failure.entryId,
								type: prepared.failure.type,
								reason: prepared.failure.reason,
							} as const;
						}
						const preparedEntries = prepared.success;

						// New session first (prepared entries FK it); leaf set after the copy.
						// Aggregates start at 0 — spend stays recorded in the source.
						const sessionRow = yield* SessionRow.insert.makeEffect({
							id: newSessionId,
							projectId: source.value.projectId,
							parentId: Option.some(source.value.id), // fork lineage
							slug: input.slug,
							directory: source.value.directory,
							title: input.title ?? source.value.title,
							tag: input.tag === undefined ? source.value.tag : Option.some(input.tag),
							// Same directory as the source, so the same sandbox env.
							sandboxInstanceId: source.value.sandboxInstanceId,
							metadata: source.value.metadata,
							leafEntryId: Option.none(),
						});
						yield* insertSession(sessionRow);

						/*
						 * A draft is a turn still being written, or one a kill left behind
						 * before the sweep reached it. Copying it would hand the new
						 * session an unfinished turn whose calls belong to another
						 * session's history — and whose sweep would then settle them
						 * there. An active session is refused a level above this; a
						 * *killed* one is not active, which is exactly the case that
						 * needs the check to be structural.
						 */
						const draft = preparedEntries.find((prepared) => prepared.source.state === "draft");
						if (draft !== undefined) {
							return {
								_tag: "invalidData",
								entryId: draft.source.id,
								type: draft.source.type,
								reason: "cannot fork a path containing an unfinished draft",
							} as const;
						}

						// Positions are carried over verbatim, not renumbered: they are log
						// positions, and preserving them keeps the copy ordered exactly as
						// the source. The new aggregate is then seeded above the highest of
						// them so post-fork events never land underneath.
						for (const prepared of preparedEntries) {
							const entryRow = yield* SessionEntryRow.insert.makeEffect({
								id: prepared.id,
								sessionId: newSessionId,
								parentId: Option.map(prepared.source.parentId, (parent) => idMap.get(parent)!),
								seq: prepared.source.seq,
								type: prepared.source.type,
								state: prepared.source.state,
								data: prepared.data,
								label: prepared.source.label, // annotations ride the copy (§10.12)
								metadata: prepared.source.metadata,
							});
							// Path order means source positions already ascend, so a
							// rejection here would mean the source tree violated its own
							// ordering invariant rather than anything about this copy.
							const copied = yield* insertEntry(entryRow);
							if (copied.length === 0) {
								return yield* Effect.die(
									`fork: copied entry ${prepared.id} at seq ${prepared.source.seq} does not advance`,
								);
							}
						}

						// Parts: new ids, re-keyed entry/session, everything else verbatim —
						// including pending toolCalls (the fork inherits crash recovery).
						const messageEntryIds = pathEntries
							.filter((entry) => messageTypes.has(entry.type))
							.map((entry) => entry.id);
						if (messageEntryIds.length > 0) {
							const parts = yield* selectPartsForEntries(messageEntryIds);
							for (const part of parts) {
								const partRow = yield* SessionEntryPartRow.insert.makeEffect({
									id: uuidv7(),
									entryId: idMap.get(part.entryId)!,
									sessionId: newSessionId,
									partIndex: part.partIndex,
									type: part.type,
									status: part.status,
									callId: part.callId,
									toolName: part.toolName,
									data: part.data,
								});
								yield* insertPart(partRow);
							}
						}

						if (Option.isSome(targetId)) {
							const now = yield* epochNow;
							yield* sql`
								UPDATE session SET leaf_entry_id = ${idMap.get(targetId.value)!}, updated_at = ${now}
								WHERE id = ${newSessionId}
							`;
						}

						// The copied entries hold positions from the source's log, so the
						// new aggregate starts above the highest of them. Without this its
						// first event would land at 0, underneath the copy, and
						// `parent.seq < child.seq` — which `selectPath` orders by — would
						// invert. An empty fork copies nothing and starts at 0 normally.
						if (preparedEntries.length > 0 && Option.isSome(targetId)) {
							const baseSeq = preparedEntries.reduce(
								(highest, prepared) => Math.max(highest, prepared.source.seq),
								0,
							);
							yield* events.advance(newSessionId, baseSeq);
							// Lands at baseSeq + 1, so the seeded range is recorded by the
							// log itself rather than inferable only from `parentId`.
							yield* events.publish(EventList.SessionForked, {
								sessionId: newSessionId,
								sourceSessionId: input.sessionId,
								sourceEntryId: targetId.value,
								baseSeq,
								timestamp: yield* DateTime.now,
							});
						}

						const created = yield* findSession(newSessionId);
						if (Option.isNone(created)) {
							return yield* Effect.die(new Error("session fork: created session did not persist"));
						}
						return { _tag: "forked", session: created.value } as const;
					}),
				)
				.pipe(Effect.orDie);

			switch (result._tag) {
				case "sessionNotFound":
					return yield* new SessionNotFoundError({ sessionId: input.sessionId });
				case "entryNotFound":
					return yield* new EntryNotFoundError({ sessionId: input.sessionId, entryId: result.entryId });
				case "invalidMode":
					return yield* new InvalidEntryDataError({
						entryId: result.entryId,
						type: result.type,
						reason: `fork mode "before" requires a user entry`,
					});
				case "invalidData":
					return yield* new InvalidEntryDataError({
						entryId: result.entryId,
						type: result.type,
						reason: result.reason,
					});
				case "forked":
					return result.session;
			}
		});

		const branch = Effect.fn("Session.branch")(function* (input: { sessionId: string; entryId: string }) {
			const moved = yield* sql
				.withTransaction(
					Effect.gen(function* () {
						const rows = yield* sql`
							SELECT id FROM session_entry WHERE id = ${input.entryId} AND session_id = ${input.sessionId}
						`;
						if (rows.length === 0) return false;
						const now = yield* epochNow;
						yield* sql`
							UPDATE session SET leaf_entry_id = ${input.entryId}, updated_at = ${now}
							WHERE id = ${input.sessionId}
						`;
						return true;
					}),
				)
				.pipe(Effect.orDie);

			if (!moved) {
				return yield* new EntryNotFoundError({ sessionId: input.sessionId, entryId: input.entryId });
			}
		});

		const setLabel = Effect.fn("Session.setLabel")(function* (input: {
			sessionId: string;
			entryId: string;
			label: string | null;
		}) {
			const labeled = yield* sql
				.withTransaction(
					Effect.gen(function* () {
						const rows = yield* sql`
							SELECT id FROM session_entry WHERE id = ${input.entryId} AND session_id = ${input.sessionId}
						`;
						if (rows.length === 0) return false;
						const now = yield* epochNow;
						yield* sql`
							UPDATE session_entry SET label = ${input.label}, updated_at = ${now}
							WHERE id = ${input.entryId}
						`;
						return true;
					}),
				)
				.pipe(Effect.orDie);

			if (!labeled) {
				return yield* new EntryNotFoundError({ sessionId: input.sessionId, entryId: input.entryId });
			}
		});

		return Service.of({
			create,
			get,
			list,
			entry,
			path,
			timeline,
			unsettled,
			latestAssistant,
			latestDraft,
			closeDraft,
			toolCalls,
			append,
			upsertPart,
			finalizeAssistant,
			beginToolCall,
			settleToolCall,
			fork,
			branch,
			setLabel,
		});
	}),
);

// Sessions foreign-key to sandbox_instance, so the host row has to exist before
// a session can record which namespace it runs in.
// Plain database: the host has no row and the namespace foreign key is skipped
// on NULL, so a session needs no sandbox setup to exist.
export const defaultLayer = layer.pipe(Layer.provide(Database.defaultLayer));

export * as Session from "./session.ts";
