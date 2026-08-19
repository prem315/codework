import { Schema } from "effect";
import { describe, expect, it } from "vite-plus/test";
import { StatePrompt } from "../src/state/prompt.ts";
import * as Tool from "../src/tools/tool.ts";

const def = (input: {
	readonly name: string;
	readonly promptSnippet?: string;
	readonly promptGuidelines?: ReadonlyArray<string>;
}): Tool.AnyToolDef =>
	Tool.define({
		name: input.name,
		description: `${input.name} description`,
		...(input.promptSnippet === undefined ? {} : { promptSnippet: input.promptSnippet }),
		...(input.promptGuidelines === undefined ? {} : { promptGuidelines: input.promptGuidelines }),
		parameters: Schema.Struct({}),
		success: Schema.String,
	});

const build = (input: Partial<StatePrompt.BuildInput> = {}) =>
	StatePrompt.build({ tools: [], directory: "/w", ...input });

describe("StatePrompt.build", () => {
	it("is deterministic for identical inputs", () => {
		const input = {
			tools: [def({ name: "bash", promptSnippet: "run a command", promptGuidelines: ["use bash"] })],
			directory: "/workspace",
			promptSystemAppend: "house rules",
		};
		expect(StatePrompt.build(input)).toBe(StatePrompt.build(input));
	});

	it("renders the foundation, the index, guidelines, and the working directory", () => {
		const prompt = build({
			tools: [def({ name: "bash", promptSnippet: "run a command" })],
			directory: "/workspace",
		});
		expect(prompt.startsWith(StatePrompt.foundation)).toBe(true);
		expect(prompt).toContain("Available tools:\n- bash: run a command");
		expect(prompt).toContain("Guidelines:\n");
		expect(prompt.endsWith("Current working directory: /workspace")).toBe(true);
	});

	it("omits a tool with no promptSnippet from the index but keeps it in guidelines", () => {
		const prompt = build({
			tools: [
				def({ name: "listed", promptSnippet: "shown" }),
				def({ name: "unlisted", promptGuidelines: ["still advises"] }),
			],
		});
		expect(prompt).toContain("- listed: shown");
		expect(prompt).not.toContain("unlisted:");
		expect(prompt).toContain("- still advises");
	});

	it("renders (none) rather than dropping the section when no tool is listed", () => {
		expect(build()).toContain("Available tools:\n(none)");
	});

	it("normalizes and deduplicates guidelines, first occurrence winning", () => {
		const lines = StatePrompt.guidelines([
			def({ name: "a", promptGuidelines: ["  keep   me  ", "", "   "] }),
			def({ name: "b", promptGuidelines: ["keep me", "and me"] }),
		]);
		expect(lines.slice(0, 2)).toEqual(["keep me", "and me"]);
		expect(lines.filter((line) => line === "keep me")).toHaveLength(1);
	});

	it("follows registry order for both the index and the guidelines", () => {
		const prompt = build({
			tools: [
				def({ name: "first", promptSnippet: "1", promptGuidelines: ["g1"] }),
				def({ name: "second", promptSnippet: "2", promptGuidelines: ["g2"] }),
			],
		});
		expect(prompt.indexOf("- first: 1")).toBeLessThan(prompt.indexOf("- second: 2"));
		expect(prompt.indexOf("- g1")).toBeLessThan(prompt.indexOf("- g2"));
	});

	it("puts standing guidelines after tool-contributed ones", () => {
		const lines = StatePrompt.guidelines([def({ name: "a", promptGuidelines: ["tool line"] })]);
		expect(lines[0]).toBe("tool line");
		expect(lines.slice(1)).toEqual([...StatePrompt.standingGuidelines]);
	});

	it("replaces the foundation with promptCustom but keeps the tool sections", () => {
		const prompt = build({
			tools: [def({ name: "bash", promptSnippet: "run a command" })],
			promptCustom: "you are a haiku generator",
		});
		expect(prompt.startsWith("you are a haiku generator")).toBe(true);
		expect(prompt).not.toContain(StatePrompt.foundation);
		// The §6.1 divergence from Pi: a custom prose block must not silently cost
		// the caller the tool index.
		expect(prompt).toContain("- bash: run a command");
		expect(prompt).toContain("Guidelines:\n");
	});

	it("places promptSystemAppend after the tool sections and before the cwd line", () => {
		const prompt = build({
			tools: [def({ name: "bash", promptSnippet: "run a command" })],
			promptSystemAppend: "house rules",
		});
		expect(prompt.indexOf("Guidelines:")).toBeLessThan(prompt.indexOf("house rules"));
		expect(prompt.indexOf("house rules")).toBeLessThan(prompt.indexOf("Current working directory:"));
	});

	it("places promptSystemAppend identically in the promptCustom branch", () => {
		const prompt = build({ promptCustom: "custom", promptSystemAppend: "house rules" });
		expect(prompt.indexOf("Available tools:")).toBeLessThan(prompt.indexOf("house rules"));
		expect(prompt.indexOf("house rules")).toBeLessThan(prompt.indexOf("Current working directory:"));
	});

	it("drops a whitespace-only promptSystemAppend rather than rendering a blank block", () => {
		expect(build({ promptSystemAppend: "   " })).toBe(build());
	});
});
