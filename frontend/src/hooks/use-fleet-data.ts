import { useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  clearFault,
  getComponents,
  getFleetPredictions,
  getLatestReadings,
  getMachineHealth,
  getMachines,
  getSensorHistory,
  getSensors,
  getSimulatorState,
  injectFault,
  resetSimulator,
} from "@/lib/api";
import type { SensorHistory } from "@/lib/types";

/** Machine metadata rarely changes mid-session; a slow poll is enough to
 * notice a machine added or renamed without refetching on every tick. */
const METADATA_INTERVAL = 30_000;
/** Matches the live-readout cadence the fleet view is built around. */
const LIVE_INTERVAL = 2_000;
/** PdM predictions come from a trained model over a rolling window — they
 * do not move meaningfully within a couple of seconds, so refreshing this
 * fast would just be load without new information. */
const PREDICTION_INTERVAL = 10_000;
/** History-derived trend for the schematic: fresh enough to notice a fault
 * developing, cheap enough to run per-sensor for one selected machine. */
const HISTORY_INTERVAL = 15_000;
const TREND_WINDOW_MINUTES = 20;

export function useMachines() {
  return useQuery({
    queryKey: ["machines"],
    queryFn: getMachines,
    refetchInterval: METADATA_INTERVAL,
  });
}

export function useLatestReadings() {
  return useQuery({
    queryKey: ["sensors", "latest"],
    queryFn: () => getLatestReadings(),
    refetchInterval: LIVE_INTERVAL,
  });
}

export function useMachineHealth(machineId: string) {
  return useQuery({
    queryKey: ["machines", machineId, "health"],
    queryFn: () => getMachineHealth(machineId),
    refetchInterval: LIVE_INTERVAL,
  });
}

/** Health for a fixed set of machines, fetched in parallel — used by the
 * schematic's machine selector, which needs every machine's severity at
 * once rather than just the currently-selected one. */
export function useMachineHealths(machineIds: string[]) {
  const results = useQueries({
    queries: machineIds.map((id) => ({
      queryKey: ["machines", id, "health"],
      queryFn: () => getMachineHealth(id),
      refetchInterval: LIVE_INTERVAL,
    })),
  });
  return new Map(machineIds.map((id, i) => [id, results[i]?.data]));
}

export function useFleetPredictions() {
  return useQuery({
    queryKey: ["pdm", "fleet"],
    queryFn: getFleetPredictions,
    refetchInterval: PREDICTION_INTERVAL,
  });
}

export function useComponents(machineId: string | null) {
  return useQuery({
    queryKey: ["machines", machineId, "components"],
    queryFn: () => getComponents(machineId!),
    enabled: machineId !== null,
    refetchInterval: METADATA_INTERVAL,
  });
}

export function useSensors(machineId: string | null) {
  return useQuery({
    queryKey: ["machines", machineId, "sensors"],
    queryFn: () => getSensors(machineId!),
    enabled: machineId !== null,
    refetchInterval: METADATA_INTERVAL,
  });
}

export function useSensorHistory(sensorId: string | null) {
  return useQuery({
    queryKey: ["sensors", sensorId, "history", TREND_WINDOW_MINUTES],
    queryFn: () => getSensorHistory(sensorId!, TREND_WINDOW_MINUTES),
    enabled: sensorId !== null,
    refetchInterval: HISTORY_INTERVAL,
  });
}

/** One machine's sensors (up to a handful) fetched in parallel — `useQueries`
 * rather than calling `useSensorHistory` in a loop, since the sensor count
 * varies per machine and hooks cannot be called a variable number of times. */
export function useSensorHistories(sensorIds: string[]): Map<string, SensorHistory | undefined> {
  const results = useQueries({
    queries: sensorIds.map((id) => ({
      queryKey: ["sensors", id, "history", TREND_WINDOW_MINUTES],
      queryFn: () => getSensorHistory(id, TREND_WINDOW_MINUTES),
      refetchInterval: HISTORY_INTERVAL,
    })),
  });
  return new Map(sensorIds.map((id, i) => [id, results[i]?.data]));
}

export function useSimulatorState() {
  return useQuery({
    queryKey: ["simulator", "state"],
    queryFn: getSimulatorState,
    refetchInterval: LIVE_INTERVAL,
  });
}

/** Every mutation below invalidates the same live queries: an operator
 * clicking "inject fault" expects the fleet and schematic to reflect it
 * immediately, not on the next 2s tick. */
function useSimulatorMutation<TArgs>(mutationFn: (args: TArgs) => Promise<unknown>) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["simulator"] });
      queryClient.invalidateQueries({ queryKey: ["sensors"] });
      queryClient.invalidateQueries({ queryKey: ["machines"] });
      queryClient.invalidateQueries({ queryKey: ["pdm"] });
    },
  });
}

export function useInjectFault() {
  return useSimulatorMutation((args: { machineId: string; faultType: string; severity: number }) =>
    injectFault(args.machineId, args.faultType, args.severity),
  );
}

export function useClearFault() {
  return useSimulatorMutation((args: { machineId: string; faultType: string }) =>
    clearFault(args.machineId, args.faultType),
  );
}

export function useResetSimulator() {
  return useSimulatorMutation<void>(() => resetSimulator());
}
