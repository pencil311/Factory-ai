import { createFileRoute } from "@tanstack/react-router";
import { motion, useReducedMotion } from "framer-motion";
import { ChevronDown, ListChecks } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { AppTopbar } from "@/components/app-topbar";
import { Composer } from "@/components/assistant/composer";
import { ModuleTimeline } from "@/components/assistant/module-timeline";
import { TurnView } from "@/components/assistant/turn-view";
import { idleRows, useChatStream, type Turn } from "@/hooks/use-chat-stream";
import { MODULE_LABELS } from "@/lib/modules";
import type { ResolutionCandidate, UserRole } from "@/lib/types";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/app/assistant")({
  head: () => ({
    meta: [
      { title: "Assistant — FactoryPilot" },
      {
        name: "description",
        content: "Report a problem and watch eight specialist modules investigate it in parallel.",
      },
      { property: "og:title", content: "Assistant — FactoryPilot" },
      {
        property: "og:description",
        content: "Report a problem and watch eight specialist modules investigate it in parallel.",
      },
    ],
  }),
  component: AssistantPage,
});

const ROLES: { value: UserRole; label: string; lens: string }[] = [
  { value: "TECHNICIAN", label: "Technician", lens: "Procedures, parts and hazards" },
  { value: "ENGINEER", label: "Engineer", lens: "Causes, evidence and signals" },
  { value: "MANAGER", label: "Manager", lens: "Downtime, cost and impact" },
  { value: "SAFETY_OFFICER", label: "Safety officer", lens: "Hazards, isolation and permits" },
];

function AssistantPage() {
  const { turns, streaming, ask, stop } = useChatStream();
  const [role, setRole] = useState<UserRole>("TECHNICIAN");
  const [language, setLanguage] = useState("en-US");
  const [timelineOpen, setTimelineOpen] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const current = turns[turns.length - 1] ?? null;

  // Follow the conversation as it grows, but only on new turns — not on every
  // narrative token, which would fight the user trying to read.
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [turns.length]);

  const send = (text: string, machineId?: string) =>
    ask(text, { role, machineId, language: language.slice(0, 2) });

  const onSelectMachine = (candidate: ResolutionCandidate) => {
    const question = current?.question ?? "";
    send(question, candidate.machine_id);
  };

  return (
    <>
      <AppTopbar title="Assistant" breadcrumbs={["Intelligence"]} />
      <main className="min-w-0 flex-1 overflow-hidden">
        <div className="grid h-[calc(100dvh-56px)] grid-cols-1 lg:grid-cols-[1fr_340px]">
          {/* Conversation */}
          <div className="flex min-h-0 flex-col">
            <RoleSwitcher role={role} onChange={setRole} />

            {/* Mobile: the timeline collapses to a strip above the answer. */}
            <MobileTimelineStrip
              turn={current}
              open={timelineOpen}
              onToggle={() => setTimelineOpen((o) => !o)}
            />

            <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto">
              <div className="mx-auto max-w-3xl px-4 py-6 md:px-8">
                {turns.length === 0 ? (
                  <EmptyState onPick={send} />
                ) : (
                  turns.map((turn) => (
                    <TurnView key={turn.id} turn={turn} onSelectMachine={onSelectMachine} />
                  ))
                )}
              </div>
            </div>

            <Composer
              streaming={streaming}
              language={language}
              onLanguageChange={setLanguage}
              onSend={send}
              onStop={stop}
            />
          </div>

          {/* Desktop: the timeline is a permanent second pane. */}
          <aside className="hidden min-h-0 border-l border-border bg-card/40 lg:flex lg:flex-col">
            <ModuleTimeline
              modules={current?.modules ?? idleRows()}
              routing={current?.routing ?? null}
              className="min-h-0 flex-1"
            />
          </aside>
        </div>
      </main>
    </>
  );
}

function RoleSwitcher({ role, onChange }: { role: UserRole; onChange: (role: UserRole) => void }) {
  const active = ROLES.find((r) => r.value === role);
  const reduceMotion = useReducedMotion();

  return (
    <div className="border-b border-border px-4 py-2.5 md:px-8">
      <div className="mx-auto flex max-w-3xl flex-wrap items-center gap-x-3 gap-y-1.5">
        <div
          role="radiogroup"
          aria-label="Answer for role"
          className="flex flex-wrap gap-1 rounded-md bg-muted p-0.5"
        >
          {ROLES.map((r) => (
            <button
              key={r.value}
              type="button"
              role="radio"
              aria-checked={role === r.value}
              onClick={() => onChange(r.value)}
              className={cn(
                "relative rounded-sm px-2.5 py-1 text-xs transition-colors duration-200",
                role === r.value
                  ? "text-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {role === r.value && (
                <motion.span
                  layoutId={reduceMotion ? undefined : "role-pill"}
                  transition={{ duration: 0.2, ease: [0.4, 0, 0.2, 1] }}
                  className="absolute inset-0 rounded-sm bg-background"
                />
              )}
              <span className="relative">{r.label}</span>
            </button>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">{active?.lens}</p>
      </div>
    </div>
  );
}

function MobileTimelineStrip({
  turn,
  open,
  onToggle,
}: {
  turn: Turn | null;
  open: boolean;
  onToggle: () => void;
}) {
  const rows = turn ? Object.values(turn.modules) : [];
  const finished = rows.filter((r) => r.phase === "finished").length;
  const running = rows.filter((r) => r.phase === "running");
  const selected = rows.filter((r) => r.selected).length;

  return (
    <div className="border-b border-border lg:hidden">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-4 py-2 text-left"
      >
        <ListChecks className="size-4 shrink-0 text-muted-foreground" />
        <span className="min-w-0 flex-1 truncate text-xs">
          {running.length > 0 ? (
            <>
              Running{" "}
              <span className="font-medium">
                {running.map((r) => MODULE_LABELS[r.module]).join(", ")}
              </span>
            </>
          ) : selected > 0 ? (
            <span className="font-readout">
              {finished}/{selected} modules reported
            </span>
          ) : (
            <span className="text-muted-foreground">Investigation timeline</span>
          )}
        </span>
        <ChevronDown
          className={cn(
            "size-4 shrink-0 text-muted-foreground transition-transform duration-200",
            open && "rotate-180",
          )}
        />
      </button>
      {open && (
        <div className="max-h-[45dvh] overflow-y-auto border-t border-border">
          <ModuleTimeline modules={turn?.modules ?? idleRows()} routing={turn?.routing ?? null} />
        </div>
      )}
    </div>
  );
}

function EmptyState({ onPick }: { onPick: (text: string) => void }) {
  const prompts = useMemo(
    () => [
      "CV-201 is making a grinding noise on the drive end.",
      "What does error E104 on the packing line mean?",
      "Is the spindle on the CNC safe to work on right now?",
      "How much downtime if we repair the conveyor bearing this shift?",
    ],
    [],
  );

  return (
    <div className="py-8">
      <h2 className="text-lg font-semibold tracking-tight">Report a problem</h2>
      <p className="mt-1.5 max-w-prose text-sm leading-relaxed text-muted-foreground">
        Describe what you are seeing in your own words. Eight specialist modules — knowledge
        retrieval, failure prediction, root cause, maintenance, inventory, safety and production
        impact — investigate in parallel. You can watch each one report as it lands.
      </p>
      <div className="mt-5 grid gap-2 sm:grid-cols-2">
        {prompts.map((p) => (
          <button
            key={p}
            type="button"
            onClick={() => onPick(p)}
            className="rounded-md border border-border bg-card p-3 text-left text-sm leading-relaxed transition-colors duration-200 hover:border-primary hover:bg-muted/60"
          >
            {p}
          </button>
        ))}
      </div>
    </div>
  );
}
