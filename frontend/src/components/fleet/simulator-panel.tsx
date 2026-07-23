import { FlaskConical, Loader2, RefreshCw, X } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import {
  useClearFault,
  useInjectFault,
  useResetSimulator,
  useSimulatorState,
} from "@/hooks/use-fleet-data";
import type { FaultType, Machine } from "@/lib/types";
import { cn } from "@/lib/utils";

const FAULT_TYPES: { value: FaultType; label: string }[] = [
  { value: "BEARING_WEAR", label: "Bearing wear" },
  { value: "MOTOR_OVERHEAT", label: "Motor overheat" },
  { value: "LUBRICATION_LOSS", label: "Lubrication loss" },
  { value: "BELT_MISALIGNMENT", label: "Belt misalignment" },
  { value: "SEAL_LEAK", label: "Seal leak" },
  { value: "TOOL_WEAR", label: "Tool wear" },
];

/**
 * Fault injection for demos and QA — deliberately in the open, not tucked
 * behind a hidden shortcut, because an operator or tester deciding "let's
 * see what a developing bearing fault looks like on the fleet view" is a
 * legitimate, everyday use of this screen.
 */
export function SimulatorPanel({ machines }: { machines: Machine[] }) {
  const simulator = useSimulatorState();
  const injectFault = useInjectFault();
  const clearFault = useClearFault();
  const resetSimulator = useResetSimulator();

  const [machineId, setMachineId] = useState(machines[0]?.machine_id ?? "");
  const [faultType, setFaultType] = useState<FaultType>("BEARING_WEAR");
  const [severity, setSeverity] = useState(0.15);
  const [confirmReset, setConfirmReset] = useState(false);

  const machineName = (id: string) => machines.find((m) => m.machine_id === id)?.name ?? id;

  return (
    <section className="rounded-md border border-border bg-card">
      <header className="flex items-center gap-2 border-b border-border px-4 py-3">
        <FlaskConical className="size-4 text-primary" />
        <h2 className="text-sm font-semibold tracking-tight">Simulator controls</h2>
        <span className="text-xs text-muted-foreground">
          Inject and clear faults on the live fleet — for demos and testing.
        </span>
        {simulator.data && (
          <span className="font-readout ml-auto text-[11px] text-muted-foreground">
            {simulator.data.source} · {simulator.data.time_scale}x
          </span>
        )}
      </header>

      <div className="grid gap-4 p-4 lg:grid-cols-[1fr_1.4fr]">
        {/* Inject */}
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-2">
            <label className="space-y-1 text-xs">
              <span className="text-muted-foreground">Machine</span>
              <Select value={machineId} onValueChange={setMachineId}>
                <SelectTrigger className="h-9">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {machines.map((m) => (
                    <SelectItem key={m.machine_id} value={m.machine_id}>
                      {m.machine_id} — {m.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </label>
            <label className="space-y-1 text-xs">
              <span className="text-muted-foreground">Fault type</span>
              <Select value={faultType} onValueChange={(v) => setFaultType(v as FaultType)}>
                <SelectTrigger className="h-9">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {FAULT_TYPES.map((f) => (
                    <SelectItem key={f.value} value={f.value}>
                      {f.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </label>
          </div>

          <label className="block text-xs">
            <span className="flex items-center justify-between text-muted-foreground">
              <span>Initial severity</span>
              <span className="font-readout text-foreground">{severity.toFixed(2)}</span>
            </span>
            <Slider
              className="mt-2"
              value={[severity]}
              onValueChange={([v]) => setSeverity(v)}
              min={0}
              max={1}
              step={0.05}
            />
          </label>

          <Button
            size="sm"
            className="w-full"
            disabled={!machineId || injectFault.isPending}
            onClick={() => injectFault.mutate({ machineId, faultType, severity })}
          >
            {injectFault.isPending && <Loader2 className="mr-1.5 size-3.5 animate-spin" />}
            Inject fault on {machineId || "—"}
          </Button>
          {injectFault.isError && (
            <p className="text-xs text-destructive">{(injectFault.error as Error).message}</p>
          )}
        </div>

        {/* Live state */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Active faults
            </h3>
            {confirmReset ? (
              <div className="flex items-center gap-1.5 text-xs">
                <span className="text-muted-foreground">Reset entire fleet to hour zero?</span>
                <Button
                  size="sm"
                  variant="destructive"
                  className="h-6 px-2 text-[11px]"
                  disabled={resetSimulator.isPending}
                  onClick={() => {
                    resetSimulator.mutate();
                    setConfirmReset(false);
                  }}
                >
                  Confirm
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-6 px-2 text-[11px]"
                  onClick={() => setConfirmReset(false)}
                >
                  Cancel
                </Button>
              </div>
            ) : (
              <Button
                size="sm"
                variant="outline"
                className="h-6 gap-1 px-2 text-[11px]"
                onClick={() => setConfirmReset(true)}
              >
                <RefreshCw className="size-3" />
                Reset fleet
              </Button>
            )}
          </div>

          <div className="space-y-1.5">
            {simulator.data?.machines
              .filter((m) => m.active_faults.length > 0)
              .map((m) => (
                <div key={m.machine_id} className="rounded-sm bg-muted/50 p-2">
                  <div className="mb-1 flex items-center justify-between text-xs">
                    <span className="font-medium">
                      {m.machine_id} — {machineName(m.machine_id)}
                    </span>
                    <span className="font-readout text-muted-foreground">
                      health {Math.round(m.health * 100)}%
                    </span>
                  </div>
                  {m.active_faults.map((f) => (
                    <div
                      key={f.fault_type}
                      className="flex items-center justify-between gap-2 rounded-sm px-1.5 py-1 text-[11px]"
                    >
                      <span className="min-w-0 flex-1">
                        <span className="font-medium">
                          {FAULT_TYPES.find((ft) => ft.value === f.fault_type)?.label ??
                            f.fault_type}
                        </span>
                        <span className="font-readout ml-1.5 text-muted-foreground">
                          {Math.round(f.severity * 100)}% developed
                        </span>
                      </span>
                      <button
                        type="button"
                        onClick={() =>
                          clearFault.mutate({ machineId: m.machine_id, faultType: f.fault_type })
                        }
                        disabled={clearFault.isPending}
                        className={cn(
                          "flex shrink-0 items-center gap-1 rounded-sm px-1.5 py-0.5 text-[11px] text-muted-foreground transition-colors hover:bg-muted hover:text-foreground",
                        )}
                      >
                        <X className="size-3" />
                        Clear
                      </button>
                    </div>
                  ))}
                </div>
              ))}
            {simulator.data &&
              simulator.data.machines.every((m) => m.active_faults.length === 0) && (
                <p className="rounded-sm bg-muted/50 p-3 text-center text-xs text-muted-foreground">
                  No active faults — the fleet is running clean.
                </p>
              )}
            {!simulator.data && (
              <p className="p-3 text-center text-xs text-muted-foreground">
                Loading simulator state…
              </p>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
