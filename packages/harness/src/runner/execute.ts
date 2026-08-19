/*
 * @file Implements execution contract service.
 *
 * This is the infrastructure envelope around a drain: it resolves the session's
 * sandbox, mounts it, proves its working directory exists, builds a Location on
 * that same mount, and only then hands control to the loop.
 *
 * Everything below this boundary -- State, Bash, Location, every tool -- reads a
 * mount it did not acquire. Owning the mount here, once per drain, is what makes
 * that safe: the mount outlives every turn inside the drain, so a Bash handler
 * captured for one turn is still executable when a later turn calls it.
 *
 * It assembles nothing else. Prompts, tools, and provider requests belong to
 * State and the loop.
 */

import { Cause, Effect, Layer, Option } from "effect";
import { SqlClient } from "effect/unstable/sql";
import { RunCoordinator } from "./coordinator.ts";
import { RunnerExecution } from "./execution.ts";
import { Runner } from "./run.ts";

// location
import { Location } from "../location/location.ts";
import { Git } from "../git/git.ts";
import { ProjectCopy } from "../project/copy.ts";
import { Project } from "../project/project.ts";

// sandbox
import { SandboxController } from "../sandbox/control.ts";
import { SandboxInstance } from "../sandbox/instance.ts";
import { SandboxIO } from "../sandbox/io.ts";

// session
import type { ID as SessionId } from "../session/schema.ts";
import { Session } from "../session/session.ts";

export const layer = Layer.effect(
	RunnerExecution.Service,
	Effect.gen(function* () {
		const store = yield* Session.Service;
		const sandbox = yield* SandboxController.Controller;
		// Captured here, not requested inside `drain`: `RunCoordinator.make`
		// requires the drain's `R` channel to be `never`, and a `Runner.Service.use`
		// in the callback would leave the tag in it. The same reason applies to the
		// SQL client below.
		const runner = yield* Runner.Service;
		const sql = yield* SqlClient.SqlClient;

		/*
		 * Location's dependency graph, composed once at layer build rather than per
		 * drain. Its one residual requirement is the mount -- Project reads
		 * `SandboxIO.FileSystem` and `SandboxIO.Current`, Git and Copy read the
		 * shell -- which `withMount` supplies inside each drain.
		 *
		 * `Location.layer()` takes no ref, so the location directory is the mount's
		 * own cwd. That is the invariant the system prompt depends on: there is one
		 * working directory, not a mount cwd and a location directory free to
		 * disagree.
		 */
		const database = Layer.succeed(SqlClient.SqlClient, sql);
		const project = Project.layer.pipe(
			Layer.provide(Layer.merge(Git.layer, ProjectCopy.layer)),
			Layer.provide(database),
		);
		const location = Location.layer().pipe(Layer.provide(project));

		const coordinator = yield* RunCoordinator.make<SessionId, Runner.RunError>({
			drain: Effect.fnUntraced(function* (sessionId: SessionId, force) {
				// `get` returns an Option, which is always truthy as an object -- the
				// emptiness test has to be explicit.
				const session = yield* store.get(sessionId);
				if (Option.isNone(session)) return yield* Effect.die(`Session not found: ${sessionId}`);
				const row = session.value;
				// NULL means the host (§4 of the sandbox model), so a session written
				// before any namespace was registered still resolves.
				const instanceId = SandboxInstance.fromField(row.sandboxInstanceId);

				const mounted = Effect.gen(function* () {
					const current = yield* SandboxIO.Current;
					const fs = yield* SandboxIO.FileSystem;
					/*
					 * Prove the directory before anything downstream assumes it. A
					 * session mounted at a path that does not exist would otherwise fail
					 * later and less legibly: Project walks up from nowhere, Bash runs
					 * with an invalid cwd, and the prompt tells the model it is somewhere
					 * it is not.
					 */
					if (!(yield* fs.exists(current.cwd))) {
						return yield* new Runner.SandboxDirectoryNotFoundError({
							sessionId,
							sandboxInstanceId: instanceId,
							directory: current.cwd,
						});
					}
					return yield* runner.run({ sessionId, force }).pipe(Effect.provide(location));
				});

				/*
				 * `withMount` is `use.pipe(Effect.provide(mount(...)), Effect.scoped)`,
				 * so the mount is built exactly once and released when the drain
				 * settles or is interrupted. Building the mount layer a second time
				 * would take another reference and could attach another transport.
				 */
				return yield* sandbox
					.withMount(instanceId, mounted, { cwd: row.directory })
					.pipe(
						Effect.tapCause((cause) =>
							Cause.hasInterruptsOnly(cause)
								? Effect.void
								: Effect.logError("Failed to drain Session", cause).pipe(Effect.annotateLogs({ sessionId })),
						),
					);
			}),
		});

		return RunnerExecution.Service.of({
			active: coordinator.active,
			interrupt: coordinator.interrupt,
			resume: coordinator.run,
			wake: coordinator.wake,
		});
	}),
);

export * as RunnerExecute from "./execute.ts";
