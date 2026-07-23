import { Loader2 } from "lucide-react";

import { SchematicNode, type NodeSensorReadout } from "@/components/twin/schematic-node";
import { componentLevel } from "@/lib/health";
import type { ForestLayout } from "@/lib/tree-layout";

const PAD = 32;

export function MachineSchematic({
  layout,
  readoutsByComponent,
  isLoading,
  errorMessage,
  highlightedComponentId,
  onSelectComponent,
}: {
  layout: ForestLayout;
  readoutsByComponent: Map<string, NodeSensorReadout[]>;
  isLoading: boolean;
  errorMessage: string | null;
  highlightedComponentId: string | null;
  onSelectComponent: (componentId: string) => void;
}) {
  if (isLoading) {
    return (
      <div className="flex h-full min-h-[400px] items-center justify-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" />
        Loading component tree…
      </div>
    );
  }

  if (errorMessage) {
    return (
      <div className="flex h-full min-h-[400px] items-center justify-center p-6 text-sm text-destructive">
        Could not load components: {errorMessage}
      </div>
    );
  }

  if (layout.nodes.length === 0) {
    return (
      <div className="flex h-full min-h-[400px] items-center justify-center p-6 text-sm text-muted-foreground">
        This machine has no components on record.
      </div>
    );
  }

  return (
    <div className="h-full min-h-[400px] overflow-auto p-6">
      <div
        className="relative"
        style={{ width: layout.width + PAD * 2, height: layout.height + PAD * 2 }}
      >
        <svg
          className="pointer-events-none absolute inset-0"
          width={layout.width + PAD * 2}
          height={layout.height + PAD * 2}
        >
          {layout.nodes.map((node) => {
            if (!node.parentId) return null;
            const parent = layout.nodes.find((n) => n.id === node.parentId);
            if (!parent) return null;
            return (
              <line
                key={`${parent.id}-${node.id}`}
                x1={parent.x + PAD}
                y1={parent.y + PAD + 28}
                x2={node.x + PAD}
                y2={node.y + PAD}
                stroke="var(--border)"
                strokeWidth={1.5}
              />
            );
          })}
        </svg>

        {layout.nodes.map((node) => {
          const componentSensors = readoutsByComponent.get(node.id) ?? [];
          const level = componentSensors.length
            ? componentLevel(componentSensors.map((s) => s.level))
            : "unknown";
          return (
            <SchematicNode
              key={node.id}
              component={node.component}
              level={level}
              sensors={componentSensors}
              highlighted={highlightedComponentId === node.id}
              onSelect={() => onSelectComponent(node.id)}
              style={{ left: node.x + PAD, top: node.y + PAD }}
            />
          );
        })}
      </div>
    </div>
  );
}
