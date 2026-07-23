import type { SensorHealth, SensorStatus, TrendDirection } from "@/lib/types";

/**
 * The severity language shared by the fleet cards and the machine schematic.
 *
 * "trending" is the state the product exists to catch: every sensor reads
 * inside its normal band (no WARNING/CRITICAL), yet the signal is moving the
 * wrong way. It is deliberately a distinct rung from "warning" — not a
 * lesser version of it — because conflating the two would erase the exact
 * gap (visible degradation before any threshold trips) this tool is meant
 * to surface.
 */
export type HealthLevel = "critical" | "warning" | "trending" | "normal" | "unknown";

const RANK: Record<HealthLevel, number> = {
  unknown: 0,
  normal: 1,
  trending: 2,
  warning: 3,
  critical: 4,
};

/** Worse-wins reduction over any number of health levels. */
export function worstLevel(levels: HealthLevel[]): HealthLevel {
  if (levels.length === 0) return "unknown";
  return levels.reduce((worst, l) => (RANK[l] > RANK[worst] ? l : worst));
}

function fromSensorStatus(status: SensorStatus): HealthLevel {
  switch (status) {
    case "CRITICAL":
      return "critical";
    case "WARNING":
      return "warning";
    case "UNKNOWN":
      return "unknown";
    case "NORMAL":
      return "normal";
  }
}

/**
 * A sensor's level from its live status alone, promoted to "trending" when
 * it reads NORMAL but a same-window trend (see lib/trend.ts) shows it
 * climbing toward its own thresholds — the not-yet-a-warning case.
 */
export function sensorLevel(status: SensorStatus, rising: boolean): HealthLevel {
  const base = fromSensorStatus(status);
  return base === "normal" && rising ? "trending" : base;
}

/** A component's level is the worst of the sensors attached to it. */
export function componentLevel(levels: HealthLevel[]): HealthLevel {
  return worstLevel(levels.length ? levels : ["unknown"]);
}

/**
 * A machine card's level. Sensor status (already-crossed thresholds) always
 * outranks the PdM trend — a live CRITICAL reading is never softened by a
 * model saying things are merely trending. But when every sensor is inside
 * its normal band, a DEGRADING prediction from the trained model is real
 * evidence and is surfaced as "trending", not hidden behind a healthy card.
 */
export function machineLevel(
  sensorStatusLevel: HealthLevel,
  trendDirection: TrendDirection | null | undefined,
): HealthLevel {
  if (sensorStatusLevel === "normal" && trendDirection === "DEGRADING") return "trending";
  return sensorStatusLevel;
}

export function levelFromSensorHealth(sensors: SensorHealth[]): HealthLevel {
  return worstLevel(sensors.length ? sensors.map((s) => fromSensorStatus(s.status)) : ["unknown"]);
}
