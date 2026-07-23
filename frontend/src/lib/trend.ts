import type { HistoryPoint } from "@/lib/types";

export type TrendCall = "rising" | "falling" | "flat";

export interface SensorTrend {
  direction: TrendCall;
  /** Signed, in units of (warning_threshold - normal_max) — how far the mean
   * moved between window halves, relative to that sensor's own danger span.
   * Comparable across sensors of wildly different physical scales. */
  ratio: number;
}

/** Below this fraction of the danger span, movement is noise, not a trend. */
const RISING_EPSILON = 0.12;

/**
 * Every sensor in this system is scored against an upper bound only
 * (normal_max / warning_threshold / critical_threshold) — see
 * app/routers/sensors.py:classify — so "rising" uniformly means "moving
 * toward trouble" here, never "falling toward trouble."
 *
 * Compares the mean of the first half of the window against the second
 * half rather than fitting a slope: robust to the single-sample noise a
 * simulated sensor stream carries, and cheap enough to run on every card
 * tick without a regression library.
 */
export function computeSensorTrend(
  points: HistoryPoint[],
  sensor: { normal_max: number; warning_threshold: number },
): SensorTrend {
  if (points.length < 4) return { direction: "flat", ratio: 0 };

  const mid = Math.floor(points.length / 2);
  const mean = (pts: HistoryPoint[]) => pts.reduce((sum, p) => sum + p.value, 0) / pts.length;
  const delta = mean(points.slice(mid)) - mean(points.slice(0, mid));

  const dangerSpan = Math.max(1e-9, sensor.warning_threshold - sensor.normal_max);
  const ratio = delta / dangerSpan;

  if (ratio > RISING_EPSILON) return { direction: "rising", ratio };
  if (ratio < -RISING_EPSILON) return { direction: "falling", ratio };
  return { direction: "flat", ratio };
}
