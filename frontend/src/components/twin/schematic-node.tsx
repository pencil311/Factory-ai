import {
  Gauge,
  RotateCw,
  ThermometerSun,
  TrendingDown,
  TrendingUp,
  Waves,
  Zap,
} from "lucide-react";
import type { ComponentType as ReactComponentType } from "react";

import { COMPONENT_ICON } from "@/components/twin/component-icons";
import type { HealthLevel } from "@/lib/health";
import { HEALTH_STYLE } from "@/lib/health-style";
import type { TrendCall } from "@/lib/trend";
import type { Component, Reading, Sensor, SensorType } from "@/lib/types";
import { cn } from "@/lib/utils";

const SENSOR_ICON: Record<SensorType, ReactComponentType<{ className?: string }>> = {
  temperature: ThermometerSun,
  vibration: Waves,
  pressure: Gauge,
  rpm: RotateCw,
  power: Zap,
};

export interface NodeSensorReadout {
  sensor: Sensor;
  reading: Reading | undefined;
  level: HealthLevel;
  trend: TrendCall;
}

export function SchematicNode({
  component,
  level,
  sensors,
  highlighted,
  onSelect,
  style,
}: {
  component: Component;
  level: HealthLevel;
  sensors: NodeSensorReadout[];
  highlighted: boolean;
  onSelect: () => void;
  style: React.CSSProperties;
}) {
  const health = HEALTH_STYLE[level];
  const Icon = COMPONENT_ICON[component.type];

  return (
    <button
      type="button"
      onClick={onSelect}
      style={style}
      className={cn(
        "absolute w-[164px] -translate-x-1/2 rounded-md border-2 bg-card p-2.5 text-left transition-colors duration-200 hover:bg-muted/40",
        health.border,
        highlighted && "ring-2 ring-primary ring-offset-2 ring-offset-background",
      )}
    >
      <div className="flex items-center gap-1.5">
        <Icon className={cn("size-3.5 shrink-0", health.tone)} />
        <span className="truncate text-xs font-semibold">{component.name}</span>
      </div>
      <div className="mt-0.5 truncate text-[10px] text-muted-foreground">
        {component.part_number ?? component.component_id}
      </div>

      {sensors.length > 0 && (
        <div className="mt-2 space-y-1 border-t border-border/60 pt-1.5">
          {sensors.map(({ sensor, reading, level: sLevel, trend }) => {
            const SIcon = SENSOR_ICON[sensor.type];
            const tone = HEALTH_STYLE[sLevel].tone;
            return (
              <div
                key={sensor.sensor_id}
                className="flex items-center justify-between gap-1 text-[11px]"
              >
                <span className="flex min-w-0 items-center gap-1 text-muted-foreground">
                  <SIcon className="size-2.5 shrink-0" />
                  <span className="truncate">{sensor.type}</span>
                </span>
                <span
                  className={cn(
                    "font-readout flex shrink-0 items-center gap-0.5 font-medium",
                    tone,
                  )}
                >
                  {reading ? reading.value.toFixed(1) : "—"}
                  <span className="font-normal text-muted-foreground">{sensor.unit}</span>
                  {trend === "rising" && <TrendingUp className="size-2.5 text-warning" />}
                  {trend === "falling" && <TrendingDown className="size-2.5 text-success" />}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </button>
  );
}
