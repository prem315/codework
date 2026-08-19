import { Effect, Layer, Schema } from "effect";
import { describe, expect } from "vite-plus/test";
import { Location } from "../src/location/location.ts";
import { ProjectSchema } from "../src/project/schema.ts";
import { SandboxInstance } from "../src/sandbox/instance.ts";
import { Sandbox } from "../src/sandbox/sandbox.ts";
import { AbsolutePath } from "../src/schema.ts";
import { SessionSchema } from "../src/session/schema.ts";
import { State } from "../src/state/state.ts";
import { StatePrompt } from "../src/state/prompt.ts";
import * as Tool from "../src/tools/tool.ts";
import { pendingCall } from "./tools.fixture.ts";
import { testEffect } from "./utils/effect.ts";

const cwd = "/workspace";

/*
 * Location is stubbed, not built. State reads exactly one field from it -- the
 * working directory -- and Location has its own test; building the real Project
 * graph here would test resolution rather than State. The sandbox, by contrast,
 * is real: Bash has to be bound to a shell that actually runs.
 */
const location = Layer.succeed(
	Location.Service,
	Location.Service.of({
		directory: AbsolutePath.make(cwd),
		sandboxInstanceId: SandboxInstance.ID.local,
		project: {
			id: ProjectSchema.ID.make("proj"),
			name: "workspace",
			directory: AbsolutePath.make(cwd),
		},
	}),
);

// `Sandbox.memory()` carries its own controller and driver registry, so the
// mount needs nothing beneath it.
const runtime = (options: State.Options = {}) =>
	State.layer(options).pipe(Layer.provideMerge(location), Layer.provideMerge(Sandbox.memory({ cwd })));

const sessionId = SessionSchema.ID.make("ses_state");

const take = (options: State.Options = {}) => {
	const { effect: it } = testEffect(runtime(options));
	return it;
};

const echo = Tool.provide(
	Tool.make({
		name: "echo",
		description: "echo back",
		promptSnippet: "echo a value",
		parameters: Schema.Struct({ value: Schema.String }),
		success: Schema.String,
		handler: (params) => Effect.succeed(params.value),
	}),
	Layer.empty,
);

describe("State.snapshot", () => {
	const it = take();

	it(
		"registers Bash and advertises it on the wire",
		Effect.gen(function* () {
			const state = yield* State.Service;
			const snapshot = yield* state.snapshot(sessionId);
			expect(snapshot.tools.defs.map((def) => def.name)).toEqual(["bash"]);
			expect(snapshot.tools.wire.map((tool) => tool.name)).toEqual(["bash"]);
		}),
	);

	it(
		"binds Bash to the mounted sandbox, not the host",
		Effect.gen(function* () {
			const state = yield* State.Service;
			const snapshot = yield* state.snapshot(sessionId);
			// The memory sandbox has no /etc/passwd and no host tools; running pwd
			// there proves the handler went to the mount rather than this process.
			const outcome = yield* snapshot.tools.handle(pendingCall("bash", { command: "pwd" }));
			expect(outcome.status).toBe("completed");
			expect(JSON.stringify(outcome.result.content)).toContain(cwd);
		}),
	);

	it(
		"renders the mounted working directory into the prompt",
		Effect.gen(function* () {
			const state = yield* State.Service;
			const snapshot = yield* state.snapshot(sessionId);
			expect(snapshot.systemPrompt).toContain(`Current working directory: ${cwd}`);
			expect(snapshot.systemPrompt).toContain("- bash:");
		}),
	);

	it(
		"captures the sandbox identity and the location",
		Effect.gen(function* () {
			const state = yield* State.Service;
			const snapshot = yield* state.snapshot(sessionId);
			expect(snapshot.sandbox.cwd).toBe(cwd);
			expect(snapshot.location.directory).toBe(cwd);
			expect(snapshot.sessionId).toBe(sessionId);
		}),
	);

	it(
		"applies the documented defaults and leaves maxTokens to aikit",
		Effect.gen(function* () {
			const state = yield* State.Service;
			const snapshot = yield* state.snapshot(sessionId);
			expect(snapshot.provider).toBe("openai");
			expect(snapshot.model).toBe("gpt-5.5");
			expect(snapshot.thinkingLevel).toBe("medium");
			expect(snapshot.toolExecution).toBe("sequential");
			expect(snapshot.request.timeoutMs).toBe(60_000);
			expect(snapshot.request.maxRetries).toBe(0);
			// Unset on purpose: aikit derives it from the model's own limits.
			expect(snapshot.request.maxTokens).toBeUndefined();
		}),
	);
});

describe("State.snapshot — caller options", () => {
	take({ tools: [echo] })(
		"appends caller tools after the built-in",
		Effect.gen(function* () {
			const state = yield* State.Service;
			const snapshot = yield* state.snapshot(sessionId);
			expect(snapshot.tools.defs.map((def) => def.name)).toEqual(["bash", "echo"]);
			expect(snapshot.systemPrompt).toContain("- echo: echo a value");
		}),
	);

	take({ provider: "anthropic", model: "claude", thinkingLevel: "off", toolExecution: "parallel", maxTokens: 99 })(
		"lets caller values win over the defaults",
		Effect.gen(function* () {
			const state = yield* State.Service;
			const snapshot = yield* state.snapshot(sessionId);
			expect(snapshot.provider).toBe("anthropic");
			expect(snapshot.model).toBe("claude");
			expect(snapshot.thinkingLevel).toBe("off");
			expect(snapshot.toolExecution).toBe("parallel");
			expect(snapshot.request.maxTokens).toBe(99);
		}),
	);

	take({ promptCustom: "you are terse" })(
		"replaces the foundation while keeping the tool sections",
		Effect.gen(function* () {
			const state = yield* State.Service;
			const snapshot = yield* state.snapshot(sessionId);
			expect(snapshot.systemPrompt.startsWith("you are terse")).toBe(true);
			expect(snapshot.systemPrompt).not.toContain(StatePrompt.foundation);
			expect(snapshot.systemPrompt).toContain("- bash:");
		}),
	);

	take({ promptSystemAppend: "house rules" })(
		"appends before the working-directory line",
		Effect.gen(function* () {
			const state = yield* State.Service;
			const { systemPrompt } = yield* state.snapshot(sessionId);
			expect(systemPrompt.indexOf("house rules")).toBeLessThan(systemPrompt.indexOf("Current working directory:"));
		}),
	);
});

describe("State.snapshot — promptSystemOverride", () => {
	take({ promptSystemOverride: (input) => `${input.systemPrompt}\n\nappended` })(
		"runs a synchronous override over the rendered prompt",
		Effect.gen(function* () {
			const state = yield* State.Service;
			const { systemPrompt } = yield* state.snapshot(sessionId);
			expect(systemPrompt.endsWith("\n\nappended")).toBe(true);
			expect(systemPrompt).toContain("- bash:");
		}),
	);

	take({ promptSystemOverride: async (input) => `tools: ${input.tools.map((def) => def.name).join(",")}` })(
		"runs an asynchronous override and can rebuild from the resolved tools",
		Effect.gen(function* () {
			const state = yield* State.Service;
			const { systemPrompt } = yield* state.snapshot(sessionId);
			expect(systemPrompt).toBe("tools: bash");
		}),
	);

	take({
		promptSystemOverride: (input) => `${input.provider}/${input.model}/${input.thinkingLevel}/${input.toolExecution}`,
	})(
		"hands the override the runtime facts the prompt was rendered from",
		Effect.gen(function* () {
			const state = yield* State.Service;
			const { systemPrompt } = yield* state.snapshot(sessionId);
			expect(systemPrompt).toBe("openai/gpt-5.5/medium/sequential");
		}),
	);

	take({
		promptSystemOverride: () => {
			throw new Error("boom");
		},
	})(
		"turns a synchronous throw into a typed SnapshotError",
		Effect.gen(function* () {
			const state = yield* State.Service;
			const exit = yield* state.snapshot(sessionId).pipe(Effect.exit);
			const error = yield* Effect.flip(state.snapshot(sessionId));
			expect(exit._tag).toBe("Failure");
			expect(error).toBeInstanceOf(State.SnapshotError);
			expect(error.sessionId).toBe(sessionId);
		}),
	);

	take({ promptSystemOverride: () => Promise.reject(new Error("nope")) })(
		"turns a rejected promise into a typed SnapshotError",
		Effect.gen(function* () {
			const state = yield* State.Service;
			const error = yield* Effect.flip(state.snapshot(sessionId));
			expect(error).toBeInstanceOf(State.SnapshotError);
		}),
	);
});

describe("State.snapshot — execution mode does not reach the prompt", () => {
	const sequential = take({ toolExecution: "sequential" });
	const parallel = take({ toolExecution: "parallel" });
	const prompts: string[] = [];

	sequential(
		"renders under sequential",
		Effect.gen(function* () {
			const state = yield* State.Service;
			prompts.push((yield* state.snapshot(sessionId)).systemPrompt);
		}),
	);

	parallel(
		"renders byte-identically under parallel",
		Effect.gen(function* () {
			const state = yield* State.Service;
			const { systemPrompt, toolExecution } = yield* state.snapshot(sessionId);
			prompts.push(systemPrompt);
			expect(toolExecution).toBe("parallel");
			// Only the snapshot field differs; the prompt does not branch on it.
			expect(prompts[0]).toBe(prompts[1]);
		}),
	);
});

describe("State.snapshot — prompt options composed", () => {
	const fakeBash = Tool.provide(
		Tool.make({
			name: "bash",
			description: "a caller's own bash",
			promptSnippet: "Run a command, the caller's way.",
			parameters: Schema.Struct({ command: Schema.String }),
			success: Schema.String,
			handler: (params) => Effect.succeed(`caller ran: ${params.command}`),
		}),
		Layer.empty,
	);

	take({ tools: [fakeBash] })(
		"lets a caller replace the built-in Bash by name",
		Effect.gen(function* () {
			const state = yield* State.Service;
			const snapshot = yield* state.snapshot(sessionId);

			// Registered after the built-in, so last-registration-wins replaces the
			// implementation while keeping one `bash` on the wire.
			expect(snapshot.tools.wire.map((tool) => tool.name)).toEqual(["bash"]);
			const outcome = yield* snapshot.tools.handle(pendingCall("bash", { command: "pwd" }));
			expect(outcome.status).toBe("completed");
			// The caller's handler ran, not the sandbox-bound one.
			expect(JSON.stringify(outcome.result.content)).toContain("caller ran: pwd");
		}),
	);

	take({
		promptCustom: "You are a haiku generator.",
		promptSystemAppend: "Answer in seventeen syllables.",
		promptSystemOverride: (input) => `<<${input.systemPrompt}>>`,
	})(
		"composes all three in §6.1 order, with the override outermost",
		Effect.gen(function* () {
			const state = yield* State.Service;
			const { systemPrompt } = yield* state.snapshot(sessionId);

			// The override sees the fully rendered chain, so its wrapper proves the
			// other two had already been applied when it ran.
			expect(systemPrompt.startsWith("<<")).toBe(true);
			expect(systemPrompt.endsWith(">>")).toBe(true);
			const rendered = systemPrompt.slice(2, -2);
			expect(rendered).toContain("You are a haiku generator.");
			expect(rendered).not.toContain(StatePrompt.foundation);
			// Order within the chain: tool sections, then the append, then the cwd.
			expect(rendered.indexOf("Available tools:")).toBeLessThan(rendered.indexOf("Answer in seventeen syllables."));
			expect(rendered.indexOf("Answer in seventeen syllables.")).toBeLessThan(
				rendered.indexOf(`Current working directory: ${cwd}`),
			);
		}),
	);

	take({ promptSystemOverride: (input) => `PREFIX\n\n${input.systemPrompt}` })(
		"lets an override prepend as readily as append",
		Effect.gen(function* () {
			const state = yield* State.Service;
			const { systemPrompt } = yield* state.snapshot(sessionId);
			expect(systemPrompt.startsWith("PREFIX")).toBe(true);
			expect(systemPrompt).toContain(StatePrompt.foundation);
		}),
	);
});
