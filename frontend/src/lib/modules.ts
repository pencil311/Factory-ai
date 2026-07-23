import type { ModuleName, ModuleStatus } from "@/lib/types";

/**
 * Execution levels, mirroring the dependency graph in
 * app/orchestrator/graph.py. Modules sharing a level run concurrently.
 *
 * Held as a static table rather than read from `module_start.level` because
 * the timeline shows all eight rows from the outset — including the ones the
 * router did not select — and an unselected module never emits an event to
 * carry its level.
 */
export const MODULE_LEVELS: readonly { level: number; modules: readonly ModuleName[] }[] = [
  { level: 0, modules: ["RESOLVER"] },
  { level: 1, modules: ["RAG", "PDM"] },
  { level: 2, modules: ["RCA"] },
  { level: 3, modules: ["MAINTENANCE", "SAFETY"] },
  { level: 4, modules: ["INVENTORY"] },
  { level: 5, modules: ["PRODUCTION"] },
];

export const ALL_MODULES: readonly ModuleName[] = MODULE_LEVELS.flatMap((l) => l.modules);

export const MODULE_LABELS: Record<ModuleName, string> = {
  RESOLVER: "Resolver",
  RAG: "Knowledge",
  PDM: "Prediction",
  RCA: "Root cause",
  MAINTENANCE: "Maintenance",
  INVENTORY: "Inventory",
  SAFETY: "Safety",
  PRODUCTION: "Production",
};

/** Shown on idle rows so the timeline explains itself before a run starts. */
export const MODULE_DESCRIPTIONS: Record<ModuleName, string> = {
  RESOLVER: "Confirms which machine is meant",
  RAG: "Retrieves manuals and procedures",
  PDM: "Predicts failure risk and remaining life",
  RCA: "Determines the underlying cause",
  MAINTENANCE: "Writes the repair procedure",
  INVENTORY: "Checks parts availability",
  SAFETY: "Assesses hazards and isolation",
  PRODUCTION: "Estimates downtime and cost",
};

/**
 * How a finished row reads. DEGRADED is not a backend status — it is derived
 * when a module reports non-empty `degraded_inputs`, meaning it ran without
 * one of its inputs. That field only ever arrives on the final `result`
 * payload, so a row settles into DEGRADED at the end of the run.
 */
export type RowStatus = ModuleStatus | "DEGRADED";

export const STATUS_LABELS: Record<RowStatus, string> = {
  OK: "OK",
  PARTIAL: "Partial",
  UNAVAILABLE: "Failed",
  SKIPPED: "Skipped",
  REUSED: "Reused",
  DEGRADED: "Degraded",
};
