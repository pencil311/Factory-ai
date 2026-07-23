import { CheckCircle2, CircleDashed, TrendingUp, TriangleAlert, XCircle } from "lucide-react";
import type { ComponentType as ReactComponentType } from "react";

import type { HealthLevel } from "@/lib/health";

export interface HealthStyle {
  label: string;
  icon: ReactComponentType<{ className?: string }>;
  /** Text/icon tone. */
  tone: string;
  /** Left-accent border used on cards; "trending" is dashed rather than
   * solid so it never reads as an already-tripped warning. */
  border: string;
  /** Background tint for chips/badges. */
  chip: string;
}

/**
 * One visual language for severity, shared by the fleet cards and the
 * schematic nodes. Each level carries a distinct icon AND border treatment,
 * not just a colour — "trending" in particular must never look like a
 * softened "normal", so it gets its own glyph (TrendingUp) and a dashed
 * rather than solid border.
 */
export const HEALTH_STYLE: Record<HealthLevel, HealthStyle> = {
  critical: {
    label: "Critical",
    icon: XCircle,
    tone: "text-destructive",
    border: "border-destructive",
    chip: "bg-destructive/15 text-destructive",
  },
  warning: {
    label: "Warning",
    icon: TriangleAlert,
    tone: "text-warning",
    border: "border-warning",
    chip: "bg-warning/15 text-warning",
  },
  trending: {
    label: "Trending",
    icon: TrendingUp,
    tone: "text-warning",
    border: "border-dashed border-warning",
    chip: "bg-warning/10 text-warning",
  },
  normal: {
    label: "Normal",
    icon: CheckCircle2,
    tone: "text-success",
    border: "border-success/50",
    chip: "bg-success/15 text-success",
  },
  unknown: {
    label: "No data",
    icon: CircleDashed,
    tone: "text-muted-foreground",
    border: "border-border",
    chip: "bg-muted text-muted-foreground",
  },
};
