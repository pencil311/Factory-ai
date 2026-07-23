import { Link, useRouterState } from "@tanstack/react-router";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Bot,
  Boxes,
  Building2,
  ChevronsLeft,
  Cog,
  FileText,
  Gauge,
  LayoutDashboard,
  LifeBuoy,
  Map,
  Package,
  Search,
  Wrench,
  Zap,
} from "lucide-react";
import { useState } from "react";

import { BrandLogo } from "@/components/brand-logo";
import { cn } from "@/lib/utils";

type NavItem = {
  to: string;
  label: string;
  icon: typeof LayoutDashboard;
  exact?: boolean;
  badge?: string;
};

const groups: { label: string; items: NavItem[] }[] = [
  {
    label: "Overview",
    items: [
      { to: "/app", label: "Dashboard", icon: LayoutDashboard, exact: true },
      { to: "/app/analytics", label: "Analytics", icon: BarChart3 },
    ],
  },
  {
    label: "Operations",
    items: [
      { to: "/app/machines", label: "Machines", icon: Cog },
      { to: "/app/twin", label: "Digital Twin", icon: Map },
      { to: "/app/maintenance", label: "Predictive Maintenance", icon: Wrench },
      { to: "/app/alerts", label: "Alert Center", icon: AlertTriangle, badge: "12" },
    ],
  },
  {
    label: "Intelligence",
    items: [
      { to: "/app/assistant", label: "AI Assistant", icon: Bot, badge: "new" },
      { to: "/app/reports", label: "Reports", icon: FileText },
    ],
  },
  {
    label: "Resources",
    items: [
      { to: "/app/inventory", label: "Inventory", icon: Package },
      { to: "/app/energy", label: "Energy", icon: Zap },
    ],
  },
];

export function AppSidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  const isActive = (to: string, exact?: boolean) =>
    exact ? pathname === to : pathname === to || pathname.startsWith(to + "/");

  return (
    <aside
      className={cn(
        "relative z-30 flex h-dvh shrink-0 flex-col border-r border-white/5 bg-sidebar text-sidebar-foreground transition-[width] duration-300",
        collapsed ? "w-[68px]" : "w-64",
      )}
    >
      <div className="flex h-14 items-center justify-between px-3">
        {collapsed ? <BrandLogo compact /> : <BrandLogo />}
        <button
          onClick={() => setCollapsed((c) => !c)}
          className="grid size-7 place-items-center rounded-md text-muted-foreground transition hover:bg-white/5 hover:text-foreground"
          aria-label="Toggle sidebar"
        >
          <ChevronsLeft className={cn("size-4 transition", collapsed && "rotate-180")} />
        </button>
      </div>

      {!collapsed && (
        <div className="px-3 pb-3">
          <div className="flex items-center gap-2 rounded-md border border-white/10 bg-white/[0.03] px-2.5 py-1.5 text-xs text-muted-foreground">
            <Search className="size-3.5" />
            <span>Search…</span>
            <kbd className="ml-auto rounded border border-white/10 bg-white/5 px-1.5 py-0.5 font-mono text-[10px]">
              ⌘K
            </kbd>
          </div>
        </div>
      )}

      <nav className="flex-1 overflow-y-auto px-2 pb-4">
        {groups.map((g) => (
          <div key={g.label} className="mb-4">
            {!collapsed && (
              <div className="mb-1 px-2 text-[10px] font-medium uppercase tracking-[0.16em] text-muted-foreground">
                {g.label}
              </div>
            )}
            <ul className="space-y-0.5">
              {g.items.map((item) => {
                const active = isActive(item.to, item.exact);
                return (
                  <li key={item.to}>
                    <Link
                      to={item.to}
                      className={cn(
                        "group relative flex items-center gap-2.5 rounded-md px-2 py-1.5 text-sm transition",
                        active
                          ? "bg-white/[0.06] text-foreground"
                          : "text-muted-foreground hover:bg-white/[0.04] hover:text-foreground",
                      )}
                    >
                      {active && (
                        <span className="absolute -left-2 top-1/2 h-5 w-0.5 -translate-y-1/2 rounded-r bg-primary" />
                      )}
                      <item.icon className="size-4 shrink-0" />
                      {!collapsed && (
                        <>
                          <span className="truncate">{item.label}</span>
                          {"badge" in item && item.badge && (
                            <span
                              className={cn(
                                "ml-auto rounded-full px-1.5 py-0.5 text-[10px] font-medium",
                                item.badge === "new"
                                  ? "bg-primary/20 text-primary"
                                  : "bg-white/10 text-muted-foreground",
                              )}
                            >
                              {item.badge}
                            </span>
                          )}
                        </>
                      )}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </nav>

      {!collapsed && (
        <div className="border-t border-white/5 p-3">
          <div className="rounded-xl border border-white/10 bg-primary/10 p-3">
            <div className="mb-1 flex items-center gap-2 text-xs">
              <LifeBuoy className="size-3.5 text-primary" />
              <span className="font-medium">Need help?</span>
            </div>
            <p className="text-[11px] leading-relaxed text-muted-foreground">
              Chat with a reliability engineer, 24/7.
            </p>
            <button className="mt-2 w-full rounded-md bg-white/10 px-2 py-1.5 text-xs font-medium transition hover:bg-white/15">
              Open support
            </button>
          </div>
          <div className="mt-3 flex items-center gap-2 rounded-lg px-1 py-2 text-xs text-muted-foreground">
            <div className="grid size-7 place-items-center rounded-full bg-primary text-[10px] font-semibold text-primary-foreground">
              SM
            </div>
            <div className="min-w-0 flex-1">
              <div className="truncate text-foreground">Sofia Marchetti</div>
              <div className="flex items-center gap-1 truncate">
                <Building2 className="size-3" />
                NovaSteel · Nordic 01
              </div>
            </div>
          </div>
        </div>
      )}
    </aside>
  );
}

// Re-exports to avoid tree-shake errors
export const _icons = { Activity, Boxes, Gauge };
