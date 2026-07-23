import { Gauge, RotateCw, ThermometerSun, Waves, Zap } from "lucide-react";
import type { ComponentType as ReactComponentType } from "react";

import { COMPONENT_ICON } from "@/components/twin/component-icons";
import type { NodeSensorReadout } from "@/components/twin/schematic-node";
import { componentLevel } from "@/lib/health";
import { HEALTH_STYLE } from "@/lib/health-style";
import type { Component, SensorType } from "@/lib/types";
import { cn } from "@/lib/utils";

const SENSOR_ICON: Record<SensorType, ReactComponentType<{ className?: string }>> = {
  temperature: ThermometerSun,
  vibration: Waves,
  pressure: Gauge,
  rpm: RotateCw,
  power: Zap,
};

export function ComponentDetail({
  component,
  readouts,
}: {
  component: Component | null;
  readouts: NodeSensorReadout[];
}) {
  if (!component) {
    return (
      <div className="p-4 text-sm text-muted-foreground">
        Select a component to see its live readings.
      </div>
    );
  }

  const level = readouts.length ? componentLevel(readouts.map((r) => r.level)) : "unknown";
  const style = HEALTH_STYLE[level];
  const Icon = COMPONENT_ICON[component.type];

  return (
    <div className="p-4">
      <div className="flex items-center gap-2.5">
        <div
          className={cn(
            "grid size-9 shrink-0 place-items-center rounded-md border-2",
            style.border,
          )}
        >
          <Icon className={cn("size-4", style.tone)} />
        </div>
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold">{component.name}</div>
          <div className="font-readout truncate text-xs text-muted-foreground">
            {component.component_id}
          </div>
        </div>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
        <Field label="Type" value={component.type} />
        <Field label="Part number" value={component.part_number ?? "—"} />
      </div>

      <div className="mt-4 flex items-center gap-1.5">
        <style.icon className={cn("size-3.5", style.tone)} />
        <span className={cn("text-xs font-medium", style.tone)}>{style.label}</span>
      </div>

      <div className="mt-3 space-y-2">
        {readouts.map(({ sensor, reading, level: sLevel, trend }) => {
          const SIcon = SENSOR_ICON[sensor.type];
          const tone = HEALTH_STYLE[sLevel].tone;
          return (
            <div
              key={sensor.sensor_id}
              className="rounded-sm border border-border bg-muted/40 p-2.5"
            >
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5 text-xs">
                  <SIcon className="size-3.5 text-muted-foreground" />
                  <span className="capitalize">{sensor.type}</span>
                </span>
                <span className={cn("font-readout text-sm font-semibold", tone)}>
                  {reading ? reading.value.toFixed(2) : "—"}
                  <span className="ml-0.5 text-xs font-normal text-muted-foreground">
                    {sensor.unit}
                  </span>
                </span>
              </div>
              <div className="mt-1.5 flex items-center justify-between text-[10px] text-muted-foreground">
                <span>
                  Normal ≤{sensor.normal_max} · Warning ≥{sensor.warning_threshold} · Critical ≥
                  {sensor.critical_threshold}
                </span>
                <span className="capitalize">{trend}</span>
              </div>
            </div>
          );
        })}
        {readouts.length === 0 && (
          <p className="rounded-sm border border-dashed border-border p-3 text-center text-xs text-muted-foreground">
            No sensors attached to this component.
          </p>
        )}
      </div>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-sm bg-muted/50 p-2">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-0.5 truncate capitalize">{value}</div>
    </div>
  );
}
