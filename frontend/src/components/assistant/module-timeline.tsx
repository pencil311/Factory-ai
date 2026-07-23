import { motion, useReducedMotion } from "framer-motion";
import {
  CheckCircle2,
  ChevronDown,
  Circle,
  CircleDotDashed,
  CircleMinus,
  Contrast,
  History,
  XCircle,
} from "lucide-react";
import { useState, type ComponentType } from "react";

import type { ModuleRow } from "@/hooks/use-chat-stream";
import {
  MODULE_DESCRIPTIONS,
  MODULE_LABELS,
  MODULE_LEVELS,
  STATUS_LABELS,
  type RowStatus,
} from "@/lib/modules";
import type { ModuleName, RoutingEventData } from "@/lib/types";
import { cn } from "@/lib/utils";

/**
 * Each finished state carries a glyph AND a word, so the five outcomes stay
 * distinguishable without relying on colour.
 */
const STATUS_STYLES: Record<
  RowStatus,
  { icon: ComponentType<{ className?: string }>; tone: string; weight: string }
> = {
  OK: { icon: CheckCircle2, tone: "text-success", weight: "font-medium" },
  DEGRADED: { icon: CircleDotDashed, tone: "text-info", weight: "font-medium" },
  PARTIAL: { icon: Contrast, tone: "text-warning", weight: "font-medium" },
  UNAVAILABLE: { icon: XCircle, tone: "text-destructive", weight: "font-semibold" },
  SKIPPED: { icon: CircleMinus, tone: "text-muted-foreground", weight: "font-normal" },
  REUSED: { icon: History, tone: "text-muted-foreground", weight: "font-normal" },
};

function formatElapsed(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

export function ModuleTimeline({
  modules,
  routing,
  className,
}: {
  modules: Record<ModuleName, ModuleRow>;
  routing: RoutingEventData | null;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-col", className)}>
      <RoutingHeader routing={routing} />
      <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-4">
        {MODULE_LEVELS.map((group) => (
          <LevelGroup
            key={group.level}
            level={group.level}
            modules={group.modules}
            rows={modules}
          />
        ))}
      </div>
    </div>
  );
}

function RoutingHeader({ routing }: { routing: RoutingEventData | null }) {
  return (
    <div className="border-b border-border/60 px-4 py-3">
      <div className="flex items-baseline justify-between gap-2">
        <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">
          Investigation
        </h2>
        {routing && (
          <span className="font-readout text-[11px] text-muted-foreground">
            {routing.selected_modules.length}/8 modules
          </span>
        )}
      </div>
      {routing ? (
        <>
          <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">
            {routing.reasoning || "No routing rationale was given."}
          </p>
          {routing.selected_modules.length > 0 && (
            <ul className="mt-2 flex flex-wrap gap-1">
              {routing.selected_modules.map((m) => (
                <li
                  key={m}
                  className="rounded-sm border border-border bg-muted px-1.5 py-px text-[10px] font-medium"
                >
                  {MODULE_LABELS[m]}
                </li>
              ))}
            </ul>
          )}
        </>
      ) : (
        <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">
          Eight specialists stand by. Ask a question to dispatch them.
        </p>
      )}
    </div>
  );
}

function LevelGroup({
  level,
  modules,
  rows,
}: {
  level: number;
  modules: readonly ModuleName[];
  rows: Record<ModuleName, ModuleRow>;
}) {
  const concurrent = modules.length > 1;

  return (
    <div className="relative pl-6 pt-3">
      {/* Rail: continuous through the group, so levels read as a sequence and
          the modules bracketed within one read as simultaneous. */}
      <span aria-hidden className="absolute left-[7px] top-0 h-full w-px bg-border" />
      <span
        aria-hidden
        className="absolute left-[3px] top-[18px] size-2.5 rounded-full border-2 border-border bg-background"
      />

      <div className="mb-1 flex items-center gap-1.5">
        <span className="font-readout text-[10px] uppercase tracking-wider text-muted-foreground">
          Level {level}
        </span>
        {concurrent && (
          <span className="rounded-sm bg-muted px-1.5 py-px text-[10px] font-medium text-muted-foreground">
            runs together
          </span>
        )}
      </div>

      <div
        className={cn(
          "space-y-1",
          // A shared bracket makes the concurrency structural, not just a label.
          concurrent && "border-l-2 border-dashed border-border/80 pl-2",
        )}
      >
        {modules.map((name) => (
          <TimelineRow key={name} row={rows[name]} />
        ))}
      </div>
    </div>
  );
}

function TimelineRow({ row }: { row: ModuleRow }) {
  const [open, setOpen] = useState(false);
  const reduceMotion = useReducedMotion();

  const finished = row.phase === "finished" && row.status !== null;
  const style = finished ? STATUS_STYLES[row.status as RowStatus] : null;
  const Icon = style?.icon ?? Circle;

  const details = [
    row.reason,
    row.degradedInputs.length > 0 ? `Ran without: ${row.degradedInputs.join(", ")}.` : null,
  ].filter((d): d is string => Boolean(d));
  const expandable = details.length > 0;

  const dimmed = row.status === "SKIPPED" || (row.phase === "idle" && !row.selected);

  const content = (
    <>
      <span className="mt-0.5 shrink-0">
        {row.phase === "running" ? (
          <motion.span
            className="block size-4 rounded-full border-2 border-primary"
            animate={reduceMotion ? undefined : { opacity: [1, 0.35, 1] }}
            transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
          />
        ) : (
          <motion.span
            key={row.status ?? "idle"}
            initial={finished && !reduceMotion ? { opacity: 0, scale: 0.9 } : false}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.2, ease: [0.4, 0, 0.2, 1] }}
            className="block"
          >
            <Icon className={cn("size-4", style ? style.tone : "text-muted-foreground/40")} />
          </motion.span>
        )}
      </span>

      <span className="min-w-0 flex-1">
        <span className="flex items-baseline justify-between gap-2">
          <span className={cn("truncate text-sm", style?.weight ?? "font-normal")}>
            {MODULE_LABELS[row.module]}
          </span>
          <span className="flex shrink-0 items-center gap-1.5">
            {finished && (
              <span className={cn("text-[10px] uppercase tracking-wide", style?.tone)}>
                {STATUS_LABELS[row.status as RowStatus]}
              </span>
            )}
            {row.elapsedMs !== null && (
              <span className="font-readout text-[10px] text-muted-foreground">
                {formatElapsed(row.elapsedMs)}
              </span>
            )}
            {expandable && (
              <ChevronDown
                className={cn(
                  "size-3 text-muted-foreground transition-transform duration-200",
                  open && "rotate-180",
                )}
              />
            )}
          </span>
        </span>

        <span className="mt-0.5 block text-xs leading-relaxed text-muted-foreground">
          {row.phase === "running" ? "Working…" : row.summary || MODULE_DESCRIPTIONS[row.module]}
        </span>
      </span>
    </>
  );

  return (
    <div className={cn("rounded-md transition-opacity duration-200", dimmed && "opacity-45")}>
      {expandable ? (
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          className="flex w-full gap-2.5 rounded-md p-1.5 text-left hover:bg-muted/60"
        >
          {content}
        </button>
      ) : (
        <div className="flex gap-2.5 p-1.5">{content}</div>
      )}

      {expandable && open && (
        <motion.div
          initial={reduceMotion ? false : { opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.15 }}
          className="ml-6 mr-1.5 mb-1.5 rounded-md bg-muted/70 px-2.5 py-2 text-xs leading-relaxed text-muted-foreground"
        >
          {/* `reason` only. `error_detail` carries raw exception text and is
              never rendered. */}
          {details.map((d) => (
            <p key={d}>{d}</p>
          ))}
        </motion.div>
      )}
    </div>
  );
}
