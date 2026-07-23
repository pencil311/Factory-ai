import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Loader2 } from "lucide-react";
import { z } from "zod";

import { AppTopbar } from "@/components/app-topbar";
import { ComponentDetail } from "@/components/twin/component-detail";
import { MachineSchematic } from "@/components/twin/machine-schematic";
import { MachineSelector } from "@/components/twin/machine-selector";
import { useLatestReadings, useMachineHealths, useMachines } from "@/hooks/use-fleet-data";
import { useMachineSchematic } from "@/hooks/use-machine-schematic";
import { levelFromSensorHealth } from "@/lib/health";

const twinSearchSchema = z.object({
  machine_id: z.string().optional(),
  component_id: z.string().optional(),
});

export const Route = createFileRoute("/app/twin")({
  validateSearch: twinSearchSchema,
  head: () => ({
    meta: [
      { title: "Machine Schematic — FactoryPilot" },
      {
        name: "description",
        content: "Component-level view of a machine's condition, derived from its live sensors.",
      },
      { property: "og:title", content: "Machine Schematic — FactoryPilot" },
      {
        property: "og:description",
        content: "Component-level view of a machine's condition, derived from its live sensors.",
      },
    ],
  }),
  component: TwinPage,
});

function TwinPage() {
  const search = Route.useSearch();
  const navigate = useNavigate({ from: Route.fullPath });
  const machines = useMachines();
  const readings = useLatestReadings();

  const machineId = search.machine_id ?? machines.data?.[0]?.machine_id ?? null;
  const machineIds = (machines.data ?? []).map((m) => m.machine_id);
  const healths = useMachineHealths(machineIds);

  const machineReadings = (readings.data ?? []).filter((r) => r.machine_id === machineId);
  const schematic = useMachineSchematic(machineId ?? "", machineReadings);

  const selectComponent = (componentId: string) => {
    void navigate({ search: (prev) => ({ ...prev, component_id: componentId }) });
  };

  const selectMachine = (nextMachineId: string) => {
    void navigate({ search: { machine_id: nextMachineId } });
  };

  const selectedComponent = schematic.components.data?.find(
    (c) => c.component_id === search.component_id,
  );
  const selectedReadouts = search.component_id
    ? (schematic.readoutsByComponent.get(search.component_id) ?? [])
    : [];

  return (
    <>
      <AppTopbar title="Machine Schematic" breadcrumbs={["Operations"]} />
      <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
        {machines.isLoading && (
          <div className="flex flex-1 items-center justify-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" />
            Loading fleet…
          </div>
        )}

        {machines.isError && (
          <div className="p-6 text-sm text-destructive">
            Could not reach the API: {(machines.error as Error).message}
          </div>
        )}

        {machines.data && machineId && (
          <>
            <MachineSelector
              machines={machines.data}
              selectedId={machineId}
              levels={
                new Map(
                  machineIds.map((id) => [
                    id,
                    levelFromSensorHealth(healths.get(id)?.sensors ?? []),
                  ]),
                )
              }
              onSelect={selectMachine}
            />
            <div className="grid min-h-0 flex-1 grid-cols-1 xl:grid-cols-[1fr_340px]">
              <MachineSchematic
                layout={schematic.layout}
                readoutsByComponent={schematic.readoutsByComponent}
                isLoading={schematic.components.isLoading || schematic.sensors.isLoading}
                errorMessage={
                  schematic.components.isError
                    ? (schematic.components.error as Error).message
                    : null
                }
                highlightedComponentId={search.component_id ?? null}
                onSelectComponent={selectComponent}
              />
              <aside className="border-t border-border xl:border-l xl:border-t-0">
                <ComponentDetail
                  component={selectedComponent ?? null}
                  readouts={selectedReadouts}
                />
              </aside>
            </div>
          </>
        )}
      </main>
    </>
  );
}
