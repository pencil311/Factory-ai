import { useCallback, useReducer, useRef } from "react";

import { streamChat } from "@/lib/chat-stream";
import { ALL_MODULES, type RowStatus } from "@/lib/modules";
import type {
  Citation,
  ModuleName,
  OrchestrationResult,
  ResolutionEventData,
  RoutingEventData,
  StreamEvent,
  UserRole,
} from "@/lib/types";

export interface ModuleRow {
  module: ModuleName;
  phase: "idle" | "running" | "finished";
  /** Present once the row has finished. DEGRADED is derived at `result`. */
  status: RowStatus | null;
  summary: string;
  reason: string | null;
  elapsedMs: number | null;
  degradedInputs: string[];
  /** True once `routing` names this module in the plan. */
  selected: boolean;
}

export interface Turn {
  id: string;
  question: string;
  role: UserRole;
  /** Set when the turn was sent with a machine already pinned. */
  machineId?: string;
  phase: "streaming" | "done" | "failed";
  routing: RoutingEventData | null;
  resolution: ResolutionEventData | null;
  modules: Record<ModuleName, ModuleRow>;
  narrative: string;
  citations: Citation[];
  conflicts: string[];
  result: OrchestrationResult | null;
  /** Fatal error for this turn. Recoverable ones become `notice` instead. */
  error: string | null;
  /** A recoverable error the run continued past — shown, never fatal. */
  notice: string | null;
  totalElapsedMs: number | null;
  /** The stream closed without a `done` event. */
  truncated: boolean;
}

/** Eight idle rows. Exported so a screen with no turn yet can render the
 * timeline in its resting state. */
export function idleRows(): Record<ModuleName, ModuleRow> {
  const rows = {} as Record<ModuleName, ModuleRow>;
  for (const module of ALL_MODULES) {
    rows[module] = {
      module,
      phase: "idle",
      status: null,
      summary: "",
      reason: null,
      elapsedMs: null,
      degradedInputs: [],
      selected: false,
    };
  }
  return rows;
}

interface State {
  turns: Turn[];
  sessionId: string | null;
  streaming: boolean;
}

type Action =
  | { type: "ask"; turn: Turn }
  | { type: "event"; turnId: string; event: StreamEvent }
  | { type: "settle"; turnId: string; error?: string }
  | { type: "reset" };

function updateTurn(state: State, turnId: string, fn: (turn: Turn) => Turn): State {
  return { ...state, turns: state.turns.map((t) => (t.id === turnId ? fn(t) : t)) };
}

function patchRow(turn: Turn, module: ModuleName, patch: Partial<ModuleRow>): Turn {
  const existing = turn.modules[module];
  if (!existing) return turn;
  return { ...turn, modules: { ...turn.modules, [module]: { ...existing, ...patch } } };
}

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "ask":
      return { ...state, turns: [...state.turns, action.turn], streaming: true };

    case "reset":
      return { turns: [], sessionId: null, streaming: false };

    case "settle":
      return {
        ...updateTurn(state, action.turnId, (turn) => ({
          ...turn,
          // A stream that ends without `done` still leaves a usable answer on
          // screen; say it was cut short rather than discarding it.
          phase: action.error ? "failed" : turn.phase === "streaming" ? "done" : turn.phase,
          error: action.error ?? turn.error,
          truncated: !action.error && turn.phase === "streaming",
        })),
        streaming: false,
      };

    case "event": {
      const { event, turnId } = action;

      if (event.type === "session") {
        return updateTurn({ ...state, sessionId: event.data.session_id }, turnId, (t) => t);
      }

      return updateTurn(state, turnId, (turn) => {
        switch (event.type) {
          case "routing": {
            const selected = new Set<ModuleName>(event.data.selected_modules);
            const modules = { ...turn.modules };
            for (const name of ALL_MODULES) {
              modules[name] = { ...modules[name], selected: selected.has(name) };
            }
            return { ...turn, routing: event.data, modules };
          }

          case "resolution":
            return { ...turn, resolution: event.data };

          case "module_start":
            return patchRow(turn, event.data.module, { phase: "running" });

          case "module_finish":
            return patchRow(turn, event.data.module, {
              phase: "finished",
              status: event.data.status,
              summary: event.data.summary,
              reason: event.data.reason,
              elapsedMs: event.data.elapsed_ms,
            });

          case "narrative_delta":
            return { ...turn, narrative: turn.narrative + event.data.text };

          case "citation": {
            const key = (c: Citation) => `${c.document_id}:${c.page_number ?? ""}`;
            if (turn.citations.some((c) => key(c) === key(event.data))) return turn;
            return { ...turn, citations: [...turn.citations, event.data] };
          }

          case "conflict":
            return { ...turn, conflicts: [...turn.conflicts, event.data.description] };

          case "result": {
            // `degraded_inputs` reaches the client only here, so this is where
            // a row can finally settle into DEGRADED.
            const modules = { ...turn.modules };
            for (const run of event.data.modules_run) {
              const existing = modules[run.name];
              if (!existing) continue;
              modules[run.name] = {
                ...existing,
                phase: "finished",
                status: run.degraded_inputs.length > 0 ? "DEGRADED" : run.status,
                reason: existing.reason ?? run.reason,
                degradedInputs: run.degraded_inputs,
                elapsedMs: existing.elapsedMs ?? run.elapsed_ms,
              };
            }
            return { ...turn, result: event.data, modules };
          }

          case "error":
            if (event.data.recoverable) {
              // The narrative that follows replaces whatever is on screen —
              // the composed one failed validation and was discarded.
              return { ...turn, notice: event.data.message, narrative: "" };
            }
            return { ...turn, error: event.data.message, phase: "failed" };

          case "done":
            return { ...turn, phase: "done", totalElapsedMs: event.data.total_elapsed_ms };

          default:
            return turn;
        }
      });
    }

    default:
      return state;
  }
}

export interface AskOptions {
  role: UserRole;
  machineId?: string;
  language?: string;
}

export function useChatStream() {
  const [state, dispatch] = useReducer(reducer, {
    turns: [],
    sessionId: null,
    streaming: false,
  });
  const abortRef = useRef<AbortController | null>(null);
  // Read inside `ask` without making it a dependency, so the callback stays
  // stable across turns.
  const sessionRef = useRef<string | null>(null);
  sessionRef.current = state.sessionId;

  const ask = useCallback((question: string, options: AskOptions) => {
    const text = question.trim();
    if (!text) return;

    abortRef.current?.abort();

    const turnId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    // Dispatched before the request so the question is on screen immediately.
    dispatch({
      type: "ask",
      turn: {
        id: turnId,
        question: text,
        role: options.role,
        machineId: options.machineId,
        phase: "streaming",
        routing: null,
        resolution: null,
        modules: idleRows(),
        narrative: "",
        citations: [],
        conflicts: [],
        result: null,
        error: null,
        notice: null,
        totalElapsedMs: null,
        truncated: false,
      },
    });

    const { events, controller } = streamChat({
      message: text,
      user_role: options.role,
      ...(sessionRef.current ? { session_id: sessionRef.current } : {}),
      ...(options.machineId ? { machine_id: options.machineId } : {}),
      ...(options.language ? { language: options.language } : {}),
    });
    abortRef.current = controller;

    void (async () => {
      try {
        for await (const event of events) {
          dispatch({ type: "event", turnId, event });
        }
        dispatch({ type: "settle", turnId });
      } catch (err) {
        if (controller.signal.aborted) {
          dispatch({ type: "settle", turnId });
          return;
        }
        dispatch({
          type: "settle",
          turnId,
          error: err instanceof Error ? err.message : "The request failed.",
        });
      } finally {
        if (abortRef.current === controller) abortRef.current = null;
      }
    })();
  }, []);

  /** Cancels in flight work; the server sees the disconnect and stops too. */
  const stop = useCallback(() => abortRef.current?.abort(), []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    dispatch({ type: "reset" });
  }, []);

  return { ...state, ask, stop, reset };
}
