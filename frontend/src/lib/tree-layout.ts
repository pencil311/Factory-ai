import type { Component } from "@/lib/types";

export interface LayoutNode {
  id: string;
  component: Component;
  parentId: string | null;
  depth: number;
  x: number;
  y: number;
}

export interface ForestLayout {
  nodes: LayoutNode[];
  width: number;
  height: number;
}

const SLOT_WIDTH = 190;
const LEVEL_HEIGHT = 132;
const TREE_GAP = 70;
const NODE_HALF_WIDTH = 82;

interface TreeNode {
  id: string;
  component: Component;
  children: TreeNode[];
}

/**
 * `components` is a flat list nesting via `parent_component_id` — not
 * necessarily a single tree. Real machines mix standalone parts (a frame, a
 * belt) with small nested assemblies (a bearing under a roller under a drive
 * unit), so this returns a forest: one tree per component with no parent (or
 * whose declared parent isn't actually in this machine's own list).
 *
 * Also guards against a cycle in `parent_component_id` — which would
 * otherwise leave every member of the cycle parentless-looking on one side
 * and never reached as a root on the other, silently dropping nodes from a
 * machine the system has never seen. Any node not reachable from a genuine
 * root is promoted to a root of its own rather than lost.
 */
function buildForest(components: Component[]): TreeNode[] {
  const byId = new Map(components.map((c) => [c.component_id, c]));
  const childIds = new Map<string, string[]>();

  for (const c of components) {
    const parent = c.parent_component_id;
    if (parent && parent !== c.component_id && byId.has(parent)) {
      const siblings = childIds.get(parent) ?? [];
      siblings.push(c.component_id);
      childIds.set(parent, siblings);
    }
  }

  const isChild = new Set(Array.from(childIds.values()).flat());
  const declaredRoots = components.map((c) => c.component_id).filter((id) => !isChild.has(id));

  const reached = new Set<string>();
  const stack = [...declaredRoots];
  while (stack.length) {
    const id = stack.pop()!;
    if (reached.has(id)) continue;
    reached.add(id);
    for (const child of childIds.get(id) ?? []) stack.push(child);
  }
  const strandedByCycle = components.map((c) => c.component_id).filter((id) => !reached.has(id));
  const roots = [...declaredRoots, ...strandedByCycle];

  function build(id: string, ancestry: ReadonlySet<string>): TreeNode {
    const nextAncestry = new Set(ancestry).add(id);
    const kids = (childIds.get(id) ?? []).filter((childId) => !ancestry.has(childId));
    return {
      id,
      component: byId.get(id)!,
      children: kids.map((childId) => build(childId, nextAncestry)),
    };
  }

  return roots.map((id) => build(id, new Set()));
}

/** Places `node`'s subtree starting at `xStart`; returns the horizontal span
 * it consumed so the caller can advance past it. A parent centers over the
 * span of its children — the classic tidy-tree rule, minus reingold-tilford
 * collision resolution, which this forest's shallow depth doesn't need. */
function place(
  node: TreeNode,
  xStart: number,
  depth: number,
  out: LayoutNode[],
  parentId: string | null,
): number {
  if (node.children.length === 0) {
    out.push({
      id: node.id,
      component: node.component,
      parentId,
      depth,
      x: xStart + SLOT_WIDTH / 2,
      y: depth * LEVEL_HEIGHT,
    });
    return SLOT_WIDTH;
  }

  let cursor = xStart;
  const childCenters: number[] = [];
  for (const child of node.children) {
    const span = place(child, cursor, depth + 1, out, node.id);
    childCenters.push(cursor + span / 2);
    cursor += span;
  }

  out.push({
    id: node.id,
    component: node.component,
    parentId,
    depth,
    x: (childCenters[0] + childCenters[childCenters.length - 1]) / 2,
    y: depth * LEVEL_HEIGHT,
  });
  return cursor - xStart;
}

/** Lays out a machine's component tree with no coordinates baked in
 * anywhere — every position is derived from `parent_component_id` alone, so
 * this renders correctly for a machine's component list it has never seen. */
export function layoutComponents(components: Component[]): ForestLayout {
  if (components.length === 0) return { nodes: [], width: 0, height: 0 };

  const forest = buildForest(components);
  const nodes: LayoutNode[] = [];
  let xCursor = 0;

  for (const root of forest) {
    const span = place(root, xCursor, 0, nodes, null);
    xCursor += span + TREE_GAP;
  }

  const maxDepth = nodes.reduce((max, n) => Math.max(max, n.depth), 0);
  const width = Math.max(0, xCursor - TREE_GAP) + NODE_HALF_WIDTH * 2;
  const height = (maxDepth + 1) * LEVEL_HEIGHT;

  // Shift everything right by one node's half-width so the leftmost node's
  // card isn't clipped by the container edge.
  for (const n of nodes) n.x += NODE_HALF_WIDTH;

  return { nodes, width, height };
}
