import { Area, AreaChart, ReferenceLine, ResponsiveContainer, YAxis } from "recharts";

import type { HistoryPoint } from "@/lib/types";
import type { TrendCall } from "@/lib/trend";

/**
 * A sensor's recent history plotted against its own warning threshold — the
 * evidence for "trending badly" made visible: a line climbing toward the
 * dashed reference line while still under it is a fact, not a badge.
 */
export function Sparkline({
  points,
  warningThreshold,
  normalMax,
  trend,
  height = 44,
}: {
  points: HistoryPoint[];
  warningThreshold: number;
  normalMax: number;
  trend: TrendCall;
  height?: number;
}) {
  if (points.length < 2) {
    return (
      <div
        style={{ height }}
        className="flex items-center justify-center text-[10px] text-muted-foreground"
      >
        Gathering history…
      </div>
    );
  }

  const data = points.map((p, i) => ({ i, value: p.value }));
  const strokeVar =
    trend === "rising"
      ? "var(--warning)"
      : trend === "falling"
        ? "var(--success)"
        : "var(--muted-foreground)";

  // Pad the domain so the reference line and the trace both stay clear of
  // the chart edges instead of clipping against them.
  const values = points.map((p) => p.value);
  const lo = Math.min(...values, normalMax * 0.9);
  const hi = Math.max(...values, warningThreshold * 1.05);
  const pad = (hi - lo) * 0.1 || 1;

  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 4, right: 2, bottom: 2, left: 2 }}>
          <YAxis domain={[lo - pad, hi + pad]} hide />
          <ReferenceLine
            y={warningThreshold}
            stroke="var(--warning)"
            strokeDasharray="3 3"
            strokeWidth={1}
          />
          <Area
            type="monotone"
            dataKey="value"
            stroke={strokeVar}
            strokeWidth={1.75}
            fill={strokeVar}
            fillOpacity={0.12}
            isAnimationActive={false}
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
