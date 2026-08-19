import { Message } from "@codeworksh/aikit";
import { DateTime, Effect, Layer, Option } from "effect";
import { SqlClient } from "effect/unstable/sql";
import { describe, expect } from "vite-plus/test";
import { Context } from "../src/context/context.ts";
import { Database } from "../src/db/db.ts";
import { Event } from "../src/event/event.ts";
import { EventList } from "../src/event/list.ts";
import { SandboxInstance } from "../src/sandbox/instance.ts";
import { AbsolutePath } from "../src/schema.ts";
import { SessionMessageSchema } from "../src/session/message/schema.ts";
import { SessionProjector } from "../src/session/projector.ts";
import { Session } from "../src/session/session.ts";
import { testEffect } from "./utils/effect.ts";

const layer = SessionProjector.layer.pipe(
	Layer.provideMerge(Context.layer),
	Layer.provideMerge(Session.layer),
	Layer.provideMerge(Event.layer),
	Layer.provideMerge(Database.layer(":memory:")),
);
const { effect: it } = testEffect(layer);

describe("context — a killed entry with a gap in its parts", () => {
	it(
		"stays readable, so the sweep that settles it can still run",
		Effect.gen(function* () {
			const sql = yield* SqlClient.SqlClient;
			const sessions = yield* Session.Service;
			const events = yield* Event.Service;
			const context = yield* Context.Service;
			yield* sql`INSERT OR IGNORE INTO project (id, name, created_at, updated_at) VALUES ('p','p',0,0)`;
			const session = yield* sessions.create({
				projectId: "p",
				slug: `sparse-${crypto.randomUUID()}`,
				directory: AbsolutePath.make("/repo"),
				title: "sparse",
				sandboxInstanceId: SandboxInstance.ID.local,
			});
			const messageId = SessionMessageSchema.ID.make("assistant_sparse");
			const envelope = Message.createAssistantMessage({
				messageId,
				role: "assistant",
				protocol: "openai",
				provider: { id: "openai", name: "openai", source: "custom", env: [] },
				model: "gpt-5.5",
				usage: {
					input: 0,
					output: 0,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 0,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "aborted",
				time: { created: 1, completed: 1 },
				parts: [],
			});
			yield* events.publish(EventList.LLMStarted, {
				sessionId: session.id,
				timestamp: DateTime.makeUnsafe(0),
				messageId,
				message: envelope,
			});
			// Text announced at index 0 but never completed; the tool at index 1 does.
			yield* events.publish(EventList.LLMToolCallFinalized, {
				sessionId: session.id,
				timestamp: DateTime.makeUnsafe(0),
				messageId,
				partIndex: 1,
				callId: "call_1",
				toolName: "bash",
				part: {
					type: "toolCall",
					callID: "call_1",
					name: "bash",
					arguments: {},
					status: "pending",
					time: { start: 1, end: 1 },
				},
			});

			// Both reads run on every drain, so a hole here used to make the session
			// permanently undrainable — including for the sweep that would fix it.
			const leaf = yield* context.latestAssistant(session.id);
			expect(Option.isSome(leaf)).toBe(true);
			const parts = Option.isSome(leaf) ? leaf.value.parts : [];
			expect(parts.map((part) => part.type)).toEqual(["toolCall"]);

			const assembled = yield* context.assemble(session.id);
			expect(assembled.messages.map((message) => message.role)).toEqual(["assistant"]);
		}),
	);
});
