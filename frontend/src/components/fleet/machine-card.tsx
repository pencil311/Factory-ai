import { Link } from "@tanstack/react-router";
import {
  Activity,
  Gauge,
  Minus,
  RotateCw,
  ThermometerSun,
  TrendingDown,
  TrendingUp,
  Waves,
  Zap,
} from "lucide-react";
import { useMemo } from "react";
import type { ComponentType as ReactComponentType } from "react";

import { Sparkline } from "@/components/fleet/sparkline";
import { useMachineHealth, useSensorHistory } from "@/hooks/use-fleet-data";
import { levelFromSensorHealth, machineLevel, type HealthLevel } from "@/lib/health";
import { HEALTH_STYLE } from "@/lib/health-style";
import { computeSensorTrend } from "@/lib/trend";
import type { FleetEntry, Machine, Reading, SensorType } from "@/lib/types";
import { cn } from "@/lib/utils";

const SENSOR_ICON: Record<SensorType, ReactComponentType<{ className?: string }>> = {
  temperature: ThermometerSun,
  vibration: Waves,
  pressure: Gauge,
  rpm: RotateCw,
  power: Zap,
};

const STATUS_STYLE: Record<Machine["status"], { label: string; className: string }> = {
  running: { label: "Running", className: "bg-success/15 text-success" },
  stopped: { label: "Stopped", className: "bg-muted text-muted-foreground" },
  maintenance: { label: "Maintenance", className: "bg-info/15 text-info" },
  fault: { label: "Fault", className: "bg-destructive/15 text-destructive" },
};

export function MachineCard({
  machine,
  readings,
  prediction,
}: {
  machine: Machine;
  readings: Reading[];
  prediction: FleetEntry | undefined;
}) {
  const health = useMachineHealth(machine.machine_id);

  const worstSensor = useMemo(() => {
    const sensors = health.data?.sensors ?? [];
    if (sensors.length === 0) return null;
    return sensors.reduce((worst, s) => (s.score < worst.score ? s : worst));
  }, [health.data]);

  const worstSensorHistory = useSensorHistory(worstSensor?.sensor_id ?? null);

  const aggregateSensorLevel = levelFromSensorHealth(health.data?.sensors ?? []);
  const level = machineLevel(aggregateSensorLevel, prediction?.prediction?.trend_direction ?? null);
  const style = HEALTH_STYLE[level];
  const status = STATUS_STYLE[machine.status];

  const trend = useMemo(() => {
    if (!worstSensor || !worstSensorHistory.data) return "flat" as const;
    return computeSensorTrend(worstSensorHistory.data.points, worstSensor).direction;
  }, [worstSensor, worstSensorHistory.data]);

  return (
    <Link
      to="/app/twin"
      search={{ machine_id: machine.machine_id }}
      className={cn(
        "block rounded-md border border-l-4 bg-card p-4 transition-colors duration-200 hover:bg-muted/40",
        style.border,
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="font-readout text-xs font-semibold text-muted-foreground">
            {machine.machine_id}
          </div>
          <h3 className="truncate text-sm font-semibold tracking-tight">{machine.name}</h3>
          <p className="truncate text-xs text-muted-foreground">{machine.model}</p>
        </div>
        <span
          className={cn(
            "shrink-0 rounded-sm px-1.5 py-0.5 text-[10px] font-medium",
            status.className,
          )}
        >
          {status.label}
        </span>
      </div>

      <div className="mt-3 flex items-center gap-1.5">
        <style.icon className={cn("size-3.5 shrink-0", style.tone)} />
        <span className={cn("text-xs font-medium", style.tone)}>{style.label}</span>
        {level === "trending" && (
          <span className="text-[11px] text-muted-foreground">
            — thresholds normal, model flags degrading trend
          </span>
        )}
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2">
        {readings.map((r) => {
          const sh = health.data?.sensors.find((s) => s.sensor_id === r.sensor_id);
          const Icon = SENSOR_ICON[r.sensor_type];
          const readingLevel: HealthLevel = sh
            ? sh.status === "CRITICAL"
              ? "critical"
              : sh.status === "WARNING"
                ? "warning"
                : "normal"
            : "unknown";
          const tone = HEALTH_STYLE[readingLevel].tone;
          return (
            <div key={r.sensor_id} className="rounded-sm bg-muted/50 px-2 py-1.5">
              <div className="flex items-center gap-1 text-[10px] uppercase tracking-wide text-muted-foreground">
                <Icon className="size-3" />
                {r.sensor_type}
              </div>
              <div className={cn("font-readout mt-0.5 text-sm font-medium", tone)}>
                {r.value.toFixed(1)}
                <span className="ml-0.5 text-[10px] font-normal text-muted-foreground">
                  {r.unit}
                </span>
              </div>
            </div>
          );
        })}
        {readings.length === 0 && (
          <div className="col-span-2 rounded-sm bg-muted/50 px-2 py-3 text-center text-xs text-muted-foreground">
            No readings yet
          </div>
        )}
      </div>

      {worstSensor && (
        <div className="mt-3">
          <div className="mb-1 flex items-center justify-between text-[10px] text-muted-foreground">
            <span>
              Most-watched signal ·{" "}
              <span className="font-readout text-foreground">{worstSensor.sensor_id}</span>
            </span>
            <TrendGlyph direction={trend} />
          </div>
          <Sparkline
            points={worstSensorHistory.data?.points ?? []}
            warningThreshold={worstSensor.warning_threshold}
            normalMax={worstSensor.normal_max}
            trend={trend}
          />
        </div>
      )}

      {prediction?.prediction && (
        <div className="mt-3 flex items-center justify-between border-t border-border/60 pt-2.5 text-xs">
          <span className="text-muted-foreground">Failure risk</span>
          <span className="font-readout font-medium">
            {Math.round(prediction.prediction.failure_probability * 100)}%
            <span className="ml-2 font-normal text-muted-foreground">
              {Math.round(prediction.prediction.remaining_useful_life_hours)}h RUL
            </span>
          </span>
        </div>
      )}
      {prediction?.error && (
        <div className="mt-3 flex items-center gap-1.5 border-t border-border/60 pt-2.5 text-[11px] text-muted-foreground">
          <Activity className="size-3" />
          {prediction.error}
        </div>
      )}
    </Link>
  );
}

function TrendGlyph({ direction }: { direction: "rising" | "falling" | "flat" }) {
  if (direction === "rising") {
    return (
      <span className="flex items-center gap-0.5 text-warning">
        <TrendingUp className="size-3" /> rising
      </span>
    );
  }
  if (direction === "falling") {
    return (
      <span className="flex items-center gap-0.5 text-success">
        <TrendingDown className="size-3" /> falling
      </span>
    );
  }
  return (
    <span className="flex items-center gap-0.5">
      <Minus className="size-3" /> steady
    </span>
  );
}
