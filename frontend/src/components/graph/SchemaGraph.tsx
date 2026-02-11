import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  type Node,
  type Edge,
  type NodeTypes,
  useReactFlow,
  ReactFlowProvider,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

import type { JsonSchema } from '../../types/api'
import type { GraphNodeData } from './schemaToGraph'
import { buildFullGraph, expandField } from './schemaToGraph'
import { layoutNodes } from './layoutGraph'
import SentenceTypeNodeComponent from './nodes/SentenceTypeNode'
import ModelNodeComponent from './nodes/ModelNode'
import UnionNodeComponent from './nodes/UnionNode'
import EnumNodeComponent from './nodes/EnumNode'

interface SchemaGraphInnerProps {
  schema: JsonSchema
  title: string
}

function SchemaGraphInner({ schema, title }: SchemaGraphInnerProps) {
  const [nodes, setNodes] = useState<Node<GraphNodeData>[]>([])
  const [edges, setEdges] = useState<Edge[]>([])
  const defsRef = useRef<Record<string, JsonSchema>>({})
  const { fitView } = useReactFlow()

  // Rebuild graph when schema changes
  useEffect(() => {
    const { nodes: initial, edges: initialEdges, defs } = buildFullGraph(schema)
    defsRef.current = defs
    const laid = layoutNodes(initial, initialEdges)
    setNodes(laid)
    setEdges(initialEdges)
    setTimeout(() => fitView({ padding: 0.2 }), 50)
  }, [schema, fitView])

  const handleToggleField = useCallback(
    (nodeId: string, fieldName: string) => {
      setNodes((prev) => {
        const nodeIdx = prev.findIndex((n) => n.id === nodeId)
        if (nodeIdx === -1) return prev

        const node = prev[nodeIdx]
        const expanded = new Set(node.data.expanded)
        const isExpanding = !expanded.has(fieldName)

        if (isExpanding) {
          expanded.add(fieldName)
        } else {
          expanded.delete(fieldName)
        }

        const updatedNode: Node<GraphNodeData> = {
          ...node,
          data: { ...node.data, expanded },
        }
        const next = [...prev]
        next[nodeIdx] = updatedNode

        if (isExpanding) {
          const field = node.data.fields?.find((f) => f.name === fieldName)
          if (field) {
            const existingIds = new Set(next.map((n) => n.id))
            const { nodes: newNodes, edges: newEdges } = expandField(
              nodeId,
              fieldName,
              field,
              defsRef.current,
              existingIds,
            )
            setEdges((e) => {
              const allEdges = [...e, ...newEdges]
              const laid = layoutNodes([...next, ...newNodes], allEdges)
              // We need to set nodes from inside setEdges callback
              // to avoid stale closure — use a ref or schedule
              setTimeout(() => {
                setNodes(laid)
                setTimeout(() => fitView({ padding: 0.2, duration: 200 }), 50)
              }, 0)
              return allEdges
            })
            return [...next, ...newNodes]
          }
        } else {
          // Collapse: remove child nodes and edges from this field
          const removedEdgeIds = new Set<string>()
          const removedNodeIds = new Set<string>()

          setEdges((prevEdges) => {
            // Find edges from this node's field handle
            const toRemove = prevEdges.filter(
              (e) => e.source === nodeId && e.sourceHandle === `field-${fieldName}`,
            )
            const queue = toRemove.map((e) => e.target)
            for (const e of toRemove) removedEdgeIds.add(e.id)

            // BFS to find all descendant nodes
            while (queue.length > 0) {
              const cur = queue.shift()!
              removedNodeIds.add(cur)
              for (const e of prevEdges) {
                if (e.source === cur && !removedEdgeIds.has(e.id)) {
                  removedEdgeIds.add(e.id)
                  if (!removedNodeIds.has(e.target)) {
                    queue.push(e.target)
                  }
                }
              }
            }

            // Keep nodes that are referenced by other edges (shared $defs)
            const keptEdges = prevEdges.filter((e) => !removedEdgeIds.has(e.id))
            const referencedTargets = new Set(keptEdges.map((e) => e.target))
            for (const id of removedNodeIds) {
              if (referencedTargets.has(id)) {
                removedNodeIds.delete(id)
              }
            }

            setTimeout(() => {
              setNodes((curNodes) => {
                const filtered = curNodes.filter((n) => !removedNodeIds.has(n.id))
                const laid = layoutNodes(filtered, keptEdges)
                setTimeout(() => fitView({ padding: 0.2, duration: 200 }), 50)
                return laid
              })
            }, 0)

            return keptEdges
          })
        }

        return next
      })
    },
    [fitView],
  )

  // Wrap node components to inject onToggleField callback
  const nodeTypes: NodeTypes = useMemo(
    () => ({
      sentenceType: (props: { id: string; data: GraphNodeData }) => (
        <SentenceTypeNodeComponent
          data={props.data}
          onToggleField={(f) => handleToggleField(props.id, f)}
        />
      ),
      model: (props: { id: string; data: GraphNodeData }) => (
        <ModelNodeComponent
          data={props.data}
          onToggleField={(f) => handleToggleField(props.id, f)}
        />
      ),
      union: (props: { data: GraphNodeData }) => <UnionNodeComponent data={props.data} />,
      enum: (props: { data: GraphNodeData }) => <EnumNodeComponent data={props.data} />,
    }),
    [handleToggleField],
  )

  return (
    <div className="h-full w-full">
      <div className="border-b border-gray-200 bg-white px-4 py-2">
        <h3 className="text-sm font-semibold text-gray-700">{title}</h3>
        <p className="text-xs text-gray-400">Click + to expand fields, − to collapse</p>
      </div>
      <div className="h-[calc(100%-3rem)]">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          fitView
          proOptions={{ hideAttribution: true }}
          nodesDraggable
          nodesConnectable={false}
          edgesFocusable={false}
        >
          <Background />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
    </div>
  )
}

export default function SchemaGraph(props: SchemaGraphInnerProps) {
  return (
    <ReactFlowProvider>
      <SchemaGraphInner {...props} />
    </ReactFlowProvider>
  )
}
