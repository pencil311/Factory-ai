import { Bell, Command, HelpCircle, MessageSquare, Moon, Search, Sun } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";

export function AppTopbar({ title, breadcrumbs }: { title: string; breadcrumbs?: string[] }) {
  const [dark, setDark] = useState(true);

  const toggleTheme = () => {
    setDark((d) => !d);
    document.documentElement.classList.toggle("dark");
  };

  return (
    <header className="sticky top-0 z-20 flex h-14 items-center gap-3 border-b border-white/5 bg-background px-4 md:px-6">
      <div className="min-w-0 flex-1">
        {breadcrumbs && (
          <div className="text-[11px] text-muted-foreground">{breadcrumbs.join(" / ")}</div>
        )}
        <h1 className="truncate text-sm font-semibold tracking-tight">{title}</h1>
      </div>

      <button className="hidden items-center gap-2 rounded-md border border-white/10 bg-white/[0.03] px-3 py-1.5 text-xs text-muted-foreground transition hover:bg-white/[0.06] md:flex">
        <Search className="size-3.5" />
        <span>Search machines, alerts, docs…</span>
        <kbd className="ml-8 flex items-center gap-0.5 rounded border border-white/10 bg-white/5 px-1.5 py-0.5 font-mono text-[10px]">
          <Command className="size-3" />K
        </kbd>
      </button>

      <Button size="sm" variant="ghost" className="gap-1.5 text-xs">
        <MessageSquare className="size-3.5 text-primary" />
        Ask Elena
      </Button>

      <button
        onClick={toggleTheme}
        className={cn(
          "grid size-9 place-items-center rounded-md text-muted-foreground transition hover:bg-white/5 hover:text-foreground",
        )}
        aria-label="Toggle theme"
      >
        {dark ? <Sun className="size-4" /> : <Moon className="size-4" />}
      </button>

      <button
        aria-label="Help"
        className="grid size-9 place-items-center rounded-md text-muted-foreground transition hover:bg-white/5 hover:text-foreground"
      >
        <HelpCircle className="size-4" />
      </button>

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            aria-label="Notifications"
            className="relative grid size-9 place-items-center rounded-md text-muted-foreground transition hover:bg-white/5 hover:text-foreground"
          >
            <Bell className="size-4" />
            <span className="absolute right-1.5 top-1.5 grid h-4 min-w-4 place-items-center rounded-full bg-primary px-1 text-[9px] font-semibold text-primary-foreground">
              3
            </span>
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-80">
          <DropdownMenuLabel>Notifications</DropdownMenuLabel>
          <DropdownMenuSeparator />
          {[
            { t: "Bearing B-1147 vibration spike", d: "2 min ago", c: "warning" },
            { t: "Line A throughput restored", d: "18 min ago", c: "success" },
            { t: "Weekly reliability report ready", d: "1 h ago", c: "info" },
          ].map((n) => (
            <DropdownMenuItem key={n.t} className="flex items-start gap-3 py-2">
              <span
                className={cn(
                  "mt-1 size-1.5 rounded-full",
                  n.c === "warning" && "bg-warning",
                  n.c === "success" && "bg-success",
                  n.c === "info" && "bg-info",
                )}
              />
              <div>
                <div className="text-sm">{n.t}</div>
                <div className="text-xs text-muted-foreground">{n.d}</div>
              </div>
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>

      <div className="ml-1 grid size-8 place-items-center rounded-full bg-primary text-xs font-semibold text-primary-foreground ring-2 ring-white/10">
        SM
      </div>
    </header>
  );
}
