import { useMemo } from "react";

import {
  useComponents,
  useMachineHealth,
  useSensorHistories,
  useSensors,
} from "@/hooks/use-fleet-data";
import { sensorLevel } from "@/lib/health";
import { computeSensorTrend } from "@/lib/trend";
import { layoutComponents } from "@/lib/tree-layout";
import type { Reading } from "@/lib/types";
import type { NodeSensorReadout } from "@/components/twin/schematic-node";

/**
 * Everything the schematic and the component-detail panel both need for one
 * machine: the laid-out tree and each component's attached sensor readouts.
 * Shared as one hook so the two consumers agree on layout and severity
 * without recomputing (or drifting from) each other's logic — react-query
 * dedupes the underlying network calls regardless of how many components
 * call these hooks.
 */
export function useMachineSchematic(machineId: string, readings: Reading[]) {
  const components = useComponents(machineId);
  const sensors = useSensors(machineId);
  const health = useMachineHealth(machineId);
  const histories = useSensorHistories((sensors.data ?? []).map((s) => s.sensor_id));

  const layout = useMemo(() => layoutComponents(components.data ?? []), [components.data]);

  const readoutsByComponent = useMemo(() => {
    const map = new Map<string, NodeSensorReadout[]>();
    for (const sensor of sensors.data ?? []) {
      if (!sensor.component_id) continue;
      const status =
        health.data?.sensors.find((s) => s.sensor_id === sensor.sensor_id)?.status ?? "UNKNOWN";
      const history = histories.get(sensor.sensor_id);
      const trend = history ? computeSensorTrend(history.points, sensor).direction : "flat";
      const readout: NodeSensorReadout = {
        sensor,
        reading: readings.find((r) => r.sensor_id === sensor.sensor_id),
        level: sensorLevel(status, trend === "rising"),
        trend,
      };
      const list = map.get(sensor.component_id) ?? [];
      list.push(readout);
      map.set(sensor.component_id, list);
    }
    return map;
  }, [sensors.data, health.data, histories, readings]);

  return { components, sensors, health, layout, readoutsByComponent };
}
