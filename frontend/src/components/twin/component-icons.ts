import {
  Box,
  CircleDot,
  Cog,
  Cpu,
  Cylinder,
  Disc3,
  Droplets,
  HelpCircle,
  Radio,
  RotateCw,
  Settings2,
  SlidersHorizontal,
  Waves,
} from "lucide-react";
import type { ComponentType as ReactComponentType } from "react";

import type { ComponentType } from "@/lib/types";

export const COMPONENT_ICON: Record<ComponentType, ReactComponentType<{ className?: string }>> = {
  motor: Cog,
  bearing: CircleDot,
  pump: Droplets,
  spindle: Disc3,
  gearbox: Settings2,
  belt: Waves,
  valve: SlidersHorizontal,
  cylinder: Cylinder,
  roller: RotateCw,
  controller: Cpu,
  sensor: Radio,
  frame: Box,
  other: HelpCircle,
};
