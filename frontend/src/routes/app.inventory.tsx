import { createFileRoute } from "@tanstack/react-router";
import { Package, Search } from "lucide-react";

import { AppTopbar } from "@/components/app-topbar";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/app/inventory")({
  head: () => ({
    meta: [
      { title: "Inventory — Iron Sight" },
      { name: "description", content: "Spare parts, consumables, and reorder signals." },
      { property: "og:title", content: "Inventory — Iron Sight" },
      { property: "og:description", content: "Spare parts, consumables, and reorder signals." },
    ],
  }),
  component: InventoryPage,
});

const items = [
  { sku: "BR-1147", name: "Bearing 6205-2RS", stock: 12, min: 8, loc: "A-4-2", status: "OK" },
  { sku: "VL-0217", name: "Valve V-217 Actuator", stock: 2, min: 4, loc: "C-1-1", status: "Low" },
  { sku: "OL-9004", name: "Hydraulic Oil ISO 46 · 20L", stock: 34, min: 20, loc: "B-2-4", status: "OK" },
  { sku: "FL-3312", name: "Air Filter A-3312", stock: 0, min: 6, loc: "A-1-3", status: "Out" },
  { sku: "BL-2201", name: "Timing Belt XL-100", stock: 18, min: 10, loc: "B-3-2", status: "OK" },
];

const statusStyle: Record<string, string> = {
  OK: "bg-emerald-500/15 text-emerald-300",
  Low: "bg-yellow-500/15 text-yellow-300",
  Out: "bg-red-500/15 text-red-300",
};

function InventoryPage() {
  return (
    <>
      <AppTopbar title="Inventory" breadcrumbs={["Resources"]} />
      <main className="min-w-0 flex-1 space-y-4 overflow-y-auto p-4 md:p-6">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight">Spare parts & consumables</h2>
          <p className="text-sm text-muted-foreground">1,284 SKUs · 3 reorder alerts</p>
        </div>

        <div className="grid gap-3 md:grid-cols-4">
          {[
            { l: "Total SKUs", v: "1,284" },
            { l: "Reorder needed", v: "3", c: "text-yellow-300" },
            { l: "Out of stock", v: "1", c: "text-red-300" },
            { l: "Value on hand", v: "$412K" },
          ].map((s) => (
            <div key={s.l} className="rounded-xl border border-white/10 bg-white/[0.03] p-4">
              <div className="text-xs text-muted-foreground">{s.l}</div>
              <div className={cn("mt-1 text-2xl font-semibold", s.c)}>{s.v}</div>
            </div>
          ))}
        </div>

        <div className="rounded-xl border border-white/10 bg-white/[0.03]">
          <div className="flex items-center gap-2 border-b border-white/10 p-3">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
              <Input placeholder="Search parts…" className="h-9 border-white/10 bg-background/40 pl-9" />
            </div>
          </div>
          <table className="w-full text-sm">
            <thead className="text-left text-xs text-muted-foreground">
              <tr className="border-b border-white/5">
                <th className="px-4 py-3 font-medium">Part</th>
                <th className="px-4 py-3 font-medium">Stock</th>
                <th className="px-4 py-3 font-medium">Min</th>
                <th className="px-4 py-3 font-medium">Location</th>
                <th className="px-4 py-3 font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {items.map((it) => (
                <tr key={it.sku} className="border-b border-white/5 last:border-0 hover:bg-white/[0.02]">
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-3">
                      <div className="grid size-8 place-items-center rounded-md bg-white/5 ring-1 ring-white/10">
                        <Package className="size-3.5 text-primary" />
                      </div>
                      <div>
                        <div className="font-medium">{it.name}</div>
                        <div className="text-xs text-muted-foreground">{it.sku}</div>
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-3 font-mono">{it.stock}</td>
                  <td className="px-4 py-3 font-mono text-muted-foreground">{it.min}</td>
                  <td className="px-4 py-3 font-mono text-xs">{it.loc}</td>
                  <td className="px-4 py-3">
                    <Badge className={cn("border-transparent", statusStyle[it.status])}>{it.status}</Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </main>
    </>
  );
}
