import dagre from 'dagre'
import type { Node, Edge } from '@xyflow/react'
import type { GraphNodeData } from './schemaToGraph'

const NODE_WIDTH = 240
const NODE_HEIGHT_BASE = 60
const FIELD_HEIGHT = 24

export function layoutNodes(
  nodes: Node<GraphNodeData>[],
  edges: Edge[],
): Node<GraphNodeData>[] {
  const g = new dagre.graphlib.Graph()
  g.setGraph({ rankdir: 'LR', nodesep: 40, ranksep: 80 })
  g.setDefaultEdgeLabel(() => ({}))

  for (const node of nodes) {
    const fieldCount = node.data.fields?.length ?? 0
    const valueCount = node.data.values?.length ?? 0
    const contentCount = Math.max(fieldCount, valueCount, 1)
    const height = NODE_HEIGHT_BASE + contentCount * FIELD_HEIGHT
    g.setNode(node.id, { width: NODE_WIDTH, height })
  }

  for (const edge of edges) {
    g.setEdge(edge.source, edge.target)
  }

  dagre.layout(g)

  return nodes.map((node) => {
    const pos = g.node(node.id)
    const fieldCount = node.data.fields?.length ?? 0
    const valueCount = node.data.values?.length ?? 0
    const contentCount = Math.max(fieldCount, valueCount, 1)
    const height = NODE_HEIGHT_BASE + contentCount * FIELD_HEIGHT
    return {
      ...node,
      position: {
        x: pos.x - NODE_WIDTH / 2,
        y: pos.y - height / 2,
      },
    }
  })
}
