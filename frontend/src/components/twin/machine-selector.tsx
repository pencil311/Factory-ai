import { HEALTH_STYLE } from "@/lib/health-style";
import type { HealthLevel } from "@/lib/health";
import type { Machine } from "@/lib/types";
import { cn } from "@/lib/utils";

export function MachineSelector({
  machines,
  selectedId,
  levels,
  onSelect,
}: {
  machines: Machine[];
  selectedId: string | null;
  levels: Map<string, HealthLevel>;
  onSelect: (machineId: string) => void;
}) {
  return (
    <div className="flex flex-wrap gap-1.5 border-b border-border px-4 py-2.5 md:px-6">
      {machines.map((m) => {
        const level = levels.get(m.machine_id) ?? "unknown";
        const style = HEALTH_STYLE[level];
        const active = selectedId === m.machine_id;
        return (
          <button
            key={m.machine_id}
            type="button"
            onClick={() => onSelect(m.machine_id)}
            className={cn(
              "flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs transition-colors duration-200",
              active
                ? "border-primary bg-primary/10 text-foreground"
                : "border-border bg-card text-muted-foreground hover:text-foreground",
            )}
          >
            <style.icon className={cn("size-3", style.tone)} />
            <span className="font-readout font-medium">{m.machine_id}</span>
            <span className="hidden sm:inline">{m.name}</span>
          </button>
        );
      })}
    </div>
  );
}
