import { motion, useReducedMotion } from "framer-motion";
import { CircleAlert, FileText, Info, ShieldAlert, TriangleAlert } from "lucide-react";

import { Markdown } from "@/components/assistant/markdown";
import type { Turn } from "@/hooks/use-chat-stream";
import { MODULE_LABELS } from "@/lib/modules";
import type { Citation, ModuleName, ResolutionCandidate } from "@/lib/types";
import { cn } from "@/lib/utils";

export function TurnView({
  turn,
  onSelectMachine,
}: {
  turn: Turn;
  onSelectMachine: (candidate: ResolutionCandidate) => void;
}) {
  const reduceMotion = useReducedMotion();
  const result = turn.result;
  const ambiguous = turn.resolution?.status === "AMBIGUOUS";
  const notFound = turn.resolution?.status === "NOT_FOUND";
  const provisional = result?.provisional ?? false;

  const failed = Object.values(turn.modules).filter(
    (r) => r.status === "UNAVAILABLE" || r.status === "SKIPPED",
  );

  return (
    <article className="mb-10">
      {/* The question, echoed so a scrolled-back transcript stays readable. */}
      <div className="mb-4 flex justify-end">
        <div className="max-w-[85%] rounded-md rounded-br-sm bg-primary px-3.5 py-2 text-sm leading-relaxed text-primary-foreground">
          {turn.question}
        </div>
      </div>

      <motion.div
        initial={reduceMotion ? false : { opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2, ease: [0.4, 0, 0.2, 1] }}
      >
        {result?.safety_critical && <SafetyBanner turn={turn} />}

        {turn.error && (
          <Callout
            tone="destructive"
            icon={CircleAlert}
            title="This request could not be completed"
          >
            {turn.error}
          </Callout>
        )}

        {turn.notice && (
          <Callout tone="muted" icon={Info} title="The answer was recomposed">
            {turn.notice}
          </Callout>
        )}

        {ambiguous && turn.resolution && (
          <Clarification
            question={turn.resolution.clarification_question}
            candidates={turn.resolution.candidates}
            onSelect={onSelectMachine}
          />
        )}

        {notFound && (
          <Callout tone="muted" icon={Info} title="No matching machine">
            {turn.resolution?.clarification_question ??
              "That machine could not be identified. Try its ID, floor name, or an error code."}
          </Callout>
        )}

        {turn.narrative && (
          <div
            className={cn(
              // Provisional spans the whole response: a dashed rule down the
              // entire answer, so the uncertainty is not a footnote.
              provisional && "border-l-2 border-dashed border-warning/70 pl-4",
            )}
          >
            {provisional && (
              <p className="mb-2 flex items-center gap-1.5 text-xs font-medium text-warning">
                <TriangleAlert className="size-3.5 shrink-0" />
                Provisional — evidence was thin, everything below is unconfirmed
              </p>
            )}
            <Markdown content={turn.narrative} />
            {turn.phase === "streaming" && <StreamingCaret />}
          </div>
        )}

        {turn.phase === "streaming" && !turn.narrative && !ambiguous && !notFound && (
          <WorkingIndicator />
        )}

        {turn.conflicts.length > 0 && (
          <Callout tone="warning" icon={TriangleAlert} title="Modules disagreed">
            <ul className="space-y-1">
              {turn.conflicts.map((c) => (
                <li key={c}>{c}</li>
              ))}
            </ul>
          </Callout>
        )}

        {turn.citations.length > 0 && <Sources citations={turn.citations} />}

        {(result || turn.truncated) && <Footer turn={turn} failed={failed.map((f) => f.module)} />}
      </motion.div>
    </article>
  );
}

/**
 * Safety-critical leads the response. Uses the safety token rather than
 * destructive: this fires routinely and is not an error, but it must be
 * impossible to skim past.
 */
function SafetyBanner({ turn }: { turn: Turn }) {
  const blocking = turn.result?.safety?.blocking_conditions ?? [];
  const clearance = turn.result?.safety_clearance_required ?? false;

  return (
    <div className="mb-4 rounded-md border border-safety/40 border-l-4 border-l-safety bg-safety/10 p-3.5">
      <div className="flex items-start gap-2.5">
        <ShieldAlert className="mt-0.5 size-5 shrink-0 text-safety" />
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-semibold tracking-tight">Safety-critical</h3>
          <p className="mt-0.5 text-xs leading-relaxed text-muted-foreground">
            Read the safety section before starting work on this machine.
          </p>
          {blocking.length > 0 && (
            <ul className="mt-2 space-y-1 text-xs leading-relaxed">
              {blocking.map((c) => (
                <li key={c} className="flex gap-1.5">
                  <span aria-hidden className="text-safety">
                    ▪
                  </span>
                  <span>{c}</span>
                </li>
              ))}
            </ul>
          )}
          {clearance && (
            <p className="mt-2 text-xs font-medium">
              A repair may not proceed without safety sign-off.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

/** AMBIGUOUS: the system stopped and asked. A confident question, not an error. */
function Clarification({
  question,
  candidates,
  onSelect,
}: {
  question: string | null;
  candidates: ResolutionCandidate[];
  onSelect: (candidate: ResolutionCandidate) => void;
}) {
  return (
    <div className="mb-4">
      <p className="text-sm font-medium leading-relaxed">
        {question ?? "Which machine did you mean?"}
      </p>
      <div className="mt-3 grid gap-2 sm:grid-cols-2">
        {candidates.map((c) => (
          <button
            key={c.machine_id}
            type="button"
            onClick={() => onSelect(c)}
            className="group rounded-md border border-border bg-card p-3 text-left transition-colors duration-200 hover:border-primary hover:bg-muted/60"
          >
            <div className="flex items-baseline justify-between gap-2">
              <span className="font-readout text-xs font-semibold">{c.machine_id}</span>
              {c.confidence > 0 && (
                <span className="font-readout text-[10px] text-muted-foreground">
                  {Math.round(c.confidence * 100)}%
                </span>
              )}
            </div>
            <div className="mt-1 truncate text-sm">{c.name}</div>
            <div className="mt-0.5 truncate text-xs text-muted-foreground">
              {[c.model, c.line_id, c.status].filter(Boolean).join(" · ")}
            </div>
            {c.matched_by && (
              <div className="mt-1.5 text-[10px] uppercase tracking-wide text-muted-foreground">
                matched by {c.matched_by.replace(/_/g, " ").toLowerCase()}
              </div>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}

function Sources({ citations }: { citations: Citation[] }) {
  return (
    <section className="mt-5 rounded-md border border-border bg-card/60 p-3">
      <h3 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-[0.1em] text-muted-foreground">
        <FileText className="size-3.5" />
        Sources
      </h3>
      <ol className="mt-2 space-y-1.5">
        {citations.map((c, i) => (
          <li key={`${c.document_id}-${i}`} className="flex gap-2 text-xs leading-relaxed">
            <span className="font-readout shrink-0 text-muted-foreground">[{i + 1}]</span>
            <span className="min-w-0">
              <span className="font-medium">{c.title ?? c.document_id}</span>
              {c.section_title && (
                <span className="text-muted-foreground"> · {c.section_title}</span>
              )}
              {c.page_number !== null && (
                <span className="font-readout text-muted-foreground"> · p.{c.page_number}</span>
              )}
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}

/** PARTIAL and provenance: failure must look handled, not broken. */
function Footer({ turn, failed }: { turn: Turn; failed: ModuleName[] }) {
  const result = turn.result;
  const omitted = result?.omitted_for_role ?? [];
  const templated = result?.narrative_source === "TEMPLATE";

  return (
    <div className="mt-5 space-y-2 border-t border-border/60 pt-3">
      {failed.length > 0 && (
        <p className="text-xs leading-relaxed text-muted-foreground">
          Answered without {failed.map((m) => MODULE_LABELS[m]).join(", ")} —{" "}
          {failed.length === 1 ? "that module" : "those modules"} could not report. The rest of the
          investigation completed.
        </p>
      )}

      {omitted.length > 0 && (
        <p className="text-xs leading-relaxed text-muted-foreground">
          Scoped out for your role: {omitted.join(", ")}. Switch role to see{" "}
          {omitted.length === 1 ? "it" : "them"}.
        </p>
      )}

      {turn.truncated && (
        <p className="text-xs leading-relaxed text-warning">
          The connection closed before the run finished. What is shown above is complete as far as
          it got.
        </p>
      )}

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
        {result && (
          <span title={templated ? "Rendered deterministically from structured data" : undefined}>
            {templated ? "Deterministic answer (template)" : "Composed by language model"}
          </span>
        )}
        {result?.language_fallback && <span>Answered in English (template not translated)</span>}
        {turn.totalElapsedMs !== null && (
          <span className="font-readout">{(turn.totalElapsedMs / 1000).toFixed(1)}s</span>
        )}
      </div>
    </div>
  );
}

function Callout({
  tone,
  icon: Icon,
  title,
  children,
}: {
  tone: "destructive" | "warning" | "muted";
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  children: React.ReactNode;
}) {
  const tones = {
    destructive: "border-destructive/40 bg-destructive/10 text-destructive",
    warning: "border-warning/40 bg-warning/10 text-warning",
    muted: "border-border bg-muted/60 text-muted-foreground",
  };
  return (
    <div className={cn("mb-4 rounded-md border p-3", tones[tone])}>
      <div className="flex items-start gap-2">
        <Icon className="mt-0.5 size-4 shrink-0" />
        <div className="min-w-0 flex-1">
          <p className="text-xs font-semibold">{title}</p>
          <div className="mt-1 text-xs leading-relaxed text-foreground/80">{children}</div>
        </div>
      </div>
    </div>
  );
}

function StreamingCaret() {
  const reduceMotion = useReducedMotion();
  return (
    <motion.span
      aria-hidden
      className="ml-0.5 inline-block h-4 w-[2px] translate-y-0.5 bg-primary"
      animate={reduceMotion ? undefined : { opacity: [1, 0.2, 1] }}
      transition={{ duration: 1.1, repeat: Infinity, ease: "easeInOut" }}
    />
  );
}

function WorkingIndicator() {
  const reduceMotion = useReducedMotion();
  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground">
      <motion.span
        className="size-1.5 rounded-full bg-primary"
        animate={reduceMotion ? undefined : { opacity: [1, 0.3, 1] }}
        transition={{ duration: 1.4, repeat: Infinity, ease: "easeInOut" }}
      />
      Investigating…
    </div>
  );
}
