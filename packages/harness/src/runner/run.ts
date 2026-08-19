/*
 * @file Defines API for actual run implementation (contract)
 * Typically drain wraps the run to be then drained by the coordinator.
 */

import { Context, Effect, Schema } from "effect";
import type { ContextDecodeError, ContextEncodeError } from "../context/errors.ts";
import type { Location } from "../location/location.ts";
import type { SandboxMountError } from "../sandbox/errors.ts";
import type { SandboxFileSystem } from "../sandbox/fs/filesystem.ts";
import { SandboxInstance } from "../sandbox/instance.ts";
import type { SandboxIO } from "../sandbox/io.ts";
import type { ID as SessionId } from "../session/schema.ts";
import type { SessionNotFoundError } from "../session/session.ts";
import type { State } from "../state/state.ts";
import type { ToolExecutionError } from "../tools/error.ts";

/**
 * The session's directory does not exist inside its mounted namespace.
 *
 * Raised before the loop starts, never during it: a session whose working
 * directory is missing cannot resolve a Location, materialize Bash, or run a
 * tool, so failing at the envelope keeps every consumer beneath it free of the
 * "mounted but nowhere" case.
 */
export class SandboxDirectoryNotFoundError extends Schema.TaggedError<SandboxDirectoryNotFoundError>()(
	"Runner.SandboxDirectoryNotFoundError",
	{
		sessionId: Schema.String,
		sandboxInstanceId: SandboxInstance.ID,
		directory: Schema.String,
	},
) {}

export class ModelNotFoundError extends Schema.TaggedError<ModelNotFoundError>()("Runner.ModelNotFoundError", {
	provider: Schema.String,
	model: Schema.String,
}) {}

export class ProviderTurnError extends Schema.TaggedError<ProviderTurnError>()("Runner.ProviderTurnError", {
	provider: Schema.String,
	model: Schema.String,
	cause: Schema.Defect(),
}) {}

export class LLMStreamError extends Schema.TaggedError<LLMStreamError>()("Runner.LLMStreamError", {
	sessionId: Schema.String,
	reason: Schema.String,
}) {}

export type RunError =
	| ModelNotFoundError
	| State.SnapshotError
	/**
	 * A tool declaring `failureMode: "error"` chose to fail the run rather than
	 * report to the model. No Phase 1 tool does, but the choice is the tool
	 * author's, so the channel has to carry it.
	 */
	| ToolExecutionError
	| ProviderTurnError
	| LLMStreamError
	| ContextDecodeError
	| ContextEncodeError
	| SessionNotFoundError
	| SandboxDirectoryNotFoundError
	| SandboxMountError
	| SandboxFileSystem.FileSystemError;

export interface Interface {
	/**
	 * Drains eligible durable work. Explicit runs perform one provider attempt
	 * even when no work is eligible.
	 *
	 * The mount is in the requirements, not acquired here: `RunnerExecute` owns
	 * one mount for the whole drain, and everything below this boundary -- State,
	 * Bash, Location -- reads it. Declaring it makes "mount first" a type-level
	 * fact rather than a convention, exactly as `Location.layer` does one level
	 * down.
	 */
	readonly run: (input: {
		readonly sessionId: SessionId;
		readonly force: boolean;
	}) => Effect.Effect<void, RunError, SandboxIO.Provides | Location.Service>;
}

export class Service extends Context.Service<Service, Interface>()("@codeworksh/harness/runner/run/Service") {}

export * as Runner from "./run.ts";
