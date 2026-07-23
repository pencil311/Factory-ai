import { createFileRoute } from "@tanstack/react-router";
import { Loader2 } from "lucide-react";

import { AppTopbar } from "@/components/app-topbar";
import { MachineCard } from "@/components/fleet/machine-card";
import { SimulatorPanel } from "@/components/fleet/simulator-panel";
import { useFleetPredictions, useLatestReadings, useMachines } from "@/hooks/use-fleet-data";

export const Route = createFileRoute("/app/")({
  head: () => ({
    meta: [
      { title: "Fleet — FactoryPilot" },
      {
        name: "description",
        content: "Live status, sensor readings, and failure risk across the fleet.",
      },
      { property: "og:title", content: "Fleet — FactoryPilot" },
      {
        property: "og:description",
        content: "Live status, sensor readings, and failure risk across the fleet.",
      },
    ],
  }),
  component: FleetPage,
});

function FleetPage() {
  const machines = useMachines();
  const readings = useLatestReadings();
  const predictions = useFleetPredictions();

  return (
    <>
      <AppTopbar title="Fleet" breadcrumbs={["Operations"]} />
      <main className="min-w-0 flex-1 space-y-4 overflow-y-auto p-4 md:p-6">
        {machines.isLoading && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" />
            Loading fleet…
          </div>
        )}

        {machines.isError && (
          <div className="rounded-md border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
            Could not reach the API: {(machines.error as Error).message}
          </div>
        )}

        {machines.data && (
          <>
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
              {machines.data.map((machine) => (
                <MachineCard
                  key={machine.machine_id}
                  machine={machine}
                  readings={(readings.data ?? []).filter(
                    (r) => r.machine_id === machine.machine_id,
                  )}
                  prediction={predictions.data?.find((p) => p.machine_id === machine.machine_id)}
                />
              ))}
            </div>

            <SimulatorPanel machines={machines.data} />
          </>
        )}
      </main>
    </>
  );
}
