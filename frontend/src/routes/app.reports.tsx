import { createFileRoute } from "@tanstack/react-router";
import { Download, FileText, Filter, Plus } from "lucide-react";

import { AppTopbar } from "@/components/app-topbar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

export const Route = createFileRoute("/app/reports")({
  head: () => ({
    meta: [
      { title: "Reports — Iron Sight" },
      { name: "description", content: "Auto-generated reliability, quality and energy reports." },
      { property: "og:title", content: "Reports — Iron Sight" },
      { property: "og:description", content: "Auto-generated reliability, quality and energy reports." },
    ],
  }),
  component: ReportsPage,
});

const reports = [
  { t: "Weekly Reliability Summary", d: "Auto-generated · 2h ago", size: "3.4 MB", tag: "Reliability" },
  { t: "Nordic 01 — Monthly Energy Report", d: "Feb 2026", size: "1.8 MB", tag: "Energy" },
  { t: "Quality Yield — Line A", d: "14-day rolling", size: "2.1 MB", tag: "Quality" },
  { t: "Regulatory Compliance Audit", d: "Q1 2026", size: "5.7 MB", tag: "Compliance" },
  { t: "Downtime Root Causes", d: "Past 30 days", size: "1.2 MB", tag: "Reliability" },
  { t: "Shift Handover — Night A", d: "Today · 06:00", size: "412 KB", tag: "Ops" },
];

function ReportsPage() {
  return (
    <>
      <AppTopbar title="Reports" breadcrumbs={["Intelligence"]} />
      <main className="min-w-0 flex-1 space-y-4 overflow-y-auto p-4 md:p-6">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight">Reports</h2>
            <p className="text-sm text-muted-foreground">Generated and shared across your organization</p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" className="border-white/10 bg-white/[0.03]">
              <Filter className="mr-1.5 size-3.5" /> Filter
            </Button>
            <Button size="sm" className="shadow-glow">
              <Plus className="mr-1.5 size-3.5" /> New report
            </Button>
          </div>
        </div>

        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {reports.map((r) => (
            <div key={r.t} className="group rounded-xl border border-white/10 bg-white/[0.03] p-5 transition hover:border-primary/40">
              <div className="mb-4 flex items-start justify-between">
                <div className="grid size-10 place-items-center rounded-lg bg-primary/15 text-primary">
                  <FileText className="size-5" />
                </div>
                <Badge variant="outline" className="border-white/10 bg-white/[0.03] text-[10px]">{r.tag}</Badge>
              </div>
              <div className="text-base font-semibold tracking-tight">{r.t}</div>
              <div className="mt-1 text-xs text-muted-foreground">{r.d} · {r.size}</div>
              <div className="mt-4 flex gap-2">
                <Button size="sm" variant="outline" className="flex-1 border-white/10 bg-white/[0.03]">
                  <Download className="mr-1.5 size-3.5" /> PDF
                </Button>
                <Button size="sm" variant="ghost">Preview</Button>
              </div>
            </div>
          ))}
        </div>
      </main>
    </>
  );
}
