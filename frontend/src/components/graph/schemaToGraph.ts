import type { Node, Edge } from '@xyflow/react'
import type { JsonSchema } from '../../types/api'

export type GraphNodeType = 'sentenceType' | 'model' | 'union' | 'enum'

export interface FieldInfo {
  name: string
  typeSummary: string
  optional: boolean
  expandable: boolean
  refTarget?: string        // $defs key for direct $ref
  unionTargets?: string[]   // $defs keys for anyOf
  inlineEnum?: (string | number)[]
  enumRef?: string          // $defs key for an enum type
}

export interface GraphNodeData {
  label: string
  nodeType: GraphNodeType
  fields?: FieldInfo[]
  values?: (string | number)[] // for enum nodes
  expanded: Set<string>        // which field names are expanded
  defKey?: string              // $defs key this node came from (for dedup)
  [key: string]: unknown
}

function resolveRef(ref: string): string {
  // "#/$defs/Foo" → "Foo"
  const parts = ref.split('/')
  return parts[parts.length - 1]
}

function isNullType(s: JsonSchema): boolean {
  return s.type === 'null' || s.const === null
}

function summarizeType(field: JsonSchema, defs: Record<string, JsonSchema>): FieldInfo {
  const info: FieldInfo = {
    name: '',
    typeSummary: 'unknown',
    optional: false,
    expandable: false,
  }

  // Direct $ref
  if (field.$ref) {
    const target = resolveRef(field.$ref)
    const resolved = defs[target]
    if (resolved?.enum) {
      info.typeSummary = target
      info.expandable = true
      info.enumRef = target
    } else {
      info.typeSummary = target
      info.expandable = true
      info.refTarget = target
    }
    return info
  }

  // anyOf
  if (field.anyOf) {
    const nonNull = field.anyOf.filter((s) => !isNullType(s))
    const hasNull = field.anyOf.length > nonNull.length
    info.optional = hasNull

    if (nonNull.length === 1) {
      // Optional wrapper: anyOf with one real type + null
      const inner = nonNull[0]
      if (inner.$ref) {
        const target = resolveRef(inner.$ref)
        const resolved = defs[target]
        if (resolved?.enum) {
          info.typeSummary = target + (hasNull ? '?' : '')
          info.expandable = true
          info.enumRef = target
        } else {
          info.typeSummary = target + (hasNull ? '?' : '')
          info.expandable = true
          info.refTarget = target
        }
      } else if (inner.enum) {
        info.typeSummary = 'enum' + (hasNull ? '?' : '')
        info.expandable = true
        info.inlineEnum = inner.enum
      } else {
        info.typeSummary = (inner.type ?? 'unknown') + (hasNull ? '?' : '')
      }
    } else if (nonNull.length >= 2) {
      // Union: anyOf with multiple $ref
      const refs = nonNull.filter((s) => s.$ref).map((s) => resolveRef(s.$ref!))
      if (refs.length === nonNull.length) {
        info.typeSummary = refs.join(' | ')
        info.expandable = true
        info.unionTargets = refs
      } else {
        info.typeSummary = nonNull.map((s) => s.type ?? '?').join(' | ')
      }
    }
    return info
  }

  // Inline enum
  if (field.enum) {
    info.typeSummary = 'enum'
    info.expandable = true
    info.inlineEnum = field.enum
    return info
  }

  // Scalar
  info.typeSummary = field.type ?? 'unknown'
  if (field.default !== undefined) {
    info.typeSummary += ` = ${JSON.stringify(field.default)}`
  }
  return info
}

export function parseFields(schema: JsonSchema, defs: Record<string, JsonSchema>): FieldInfo[] {
  if (!schema.properties) return []
  const required = new Set(schema.required ?? [])
  return Object.entries(schema.properties).map(([name, prop]) => {
    const info = summarizeType(prop, defs)
    info.name = name
    if (!required.has(name) && !info.optional) info.optional = true
    return info
  })
}

// Build initial graph: just the root node (collapsed)
export function buildInitialGraph(schema: JsonSchema): {
  nodes: Node<GraphNodeData>[]
  edges: Edge[]
  defs: Record<string, JsonSchema>
} {
  const defs = schema.$defs ?? {}
  const fields = parseFields(schema, defs)

  const rootNode: Node<GraphNodeData> = {
    id: 'root',
    type: 'sentenceType',
    position: { x: 0, y: 0 },
    data: {
      label: schema.title ?? 'Root',
      nodeType: 'sentenceType',
      fields,
      expanded: new Set(),
    },
  }

  return { nodes: [rootNode], edges: [], defs }
}

// Build fully expanded graph: all expandable fields recursively opened
export function buildFullGraph(schema: JsonSchema): {
  nodes: Node<GraphNodeData>[]
  edges: Edge[]
  defs: Record<string, JsonSchema>
} {
  const defs = schema.$defs ?? {}
  const allNodes: Node<GraphNodeData>[] = []
  const allEdges: Edge[] = []
  const existingIds = new Set<string>()

  function expandNodeFields(nodeId: string, fields: FieldInfo[]) {
    for (const field of fields) {
      if (!field.expandable) continue
      const { nodes: childNodes, edges: childEdges } = expandField(
        nodeId,
        field.name,
        field,
        defs,
        existingIds,
      )
      for (const n of childNodes) {
        existingIds.add(n.id)
        allNodes.push(n)
        // Recursively expand model nodes' fields
        if (n.data.fields) {
          n.data.expanded = new Set(n.data.fields.filter((f) => f.expandable).map((f) => f.name))
          expandNodeFields(n.id, n.data.fields)
        }
      }
      allEdges.push(...childEdges)
    }
  }

  const fields = parseFields(schema, defs)
  const expandedFieldNames = new Set(fields.filter((f) => f.expandable).map((f) => f.name))

  const rootNode: Node<GraphNodeData> = {
    id: 'root',
    type: 'sentenceType',
    position: { x: 0, y: 0 },
    data: {
      label: schema.title ?? 'Root',
      nodeType: 'sentenceType',
      fields,
      expanded: expandedFieldNames,
    },
  }

  existingIds.add('root')
  allNodes.unshift(rootNode)

  expandNodeFields('root', fields)

  return { nodes: allNodes, edges: allEdges, defs }
}

// Expand a field on a node, producing new child nodes + edges
export function expandField(
  parentId: string,
  fieldName: string,
  field: FieldInfo,
  defs: Record<string, JsonSchema>,
  existingNodeIds: Set<string>,
): { nodes: Node<GraphNodeData>[]; edges: Edge[] } {
  const newNodes: Node<GraphNodeData>[] = []
  const newEdges: Edge[] = []

  if (field.unionTargets) {
    // Create a union node, then model nodes for each target
    const unionId = `${parentId}__${fieldName}__union`
    if (!existingNodeIds.has(unionId)) {
      newNodes.push({
        id: unionId,
        type: 'union',
        position: { x: 0, y: 0 },
        data: {
          label: fieldName,
          nodeType: 'union',
          expanded: new Set(),
        },
      })
    }
    newEdges.push({
      id: `${parentId}->${unionId}`,
      source: parentId,
      target: unionId,
      sourceHandle: `field-${fieldName}`,
    })

    for (const target of field.unionTargets) {
      const nodeId = `def__${target}`
      if (!existingNodeIds.has(nodeId)) {
        const defSchema = defs[target]
        if (defSchema) {
          const childFields = parseFields(defSchema, defs)
          if (defSchema.enum) {
            newNodes.push({
              id: nodeId,
              type: 'enum',
              position: { x: 0, y: 0 },
              data: {
                label: target,
                nodeType: 'enum',
                values: defSchema.enum,
                expanded: new Set(),
                defKey: target,
              },
            })
          } else {
            newNodes.push({
              id: nodeId,
              type: 'model',
              position: { x: 0, y: 0 },
              data: {
                label: defSchema.title ?? target,
                nodeType: 'model',
                fields: childFields,
                expanded: new Set(),
                defKey: target,
              },
            })
          }
        }
      }
      newEdges.push({
        id: `${unionId}->${nodeId}`,
        source: unionId,
        target: nodeId,
      })
    }
  } else if (field.refTarget) {
    const nodeId = `def__${field.refTarget}`
    if (!existingNodeIds.has(nodeId)) {
      const defSchema = defs[field.refTarget]
      if (defSchema) {
        const childFields = parseFields(defSchema, defs)
        newNodes.push({
          id: nodeId,
          type: 'model',
          position: { x: 0, y: 0 },
          data: {
            label: defSchema.title ?? field.refTarget,
            nodeType: 'model',
            fields: childFields,
            expanded: new Set(),
            defKey: field.refTarget,
          },
        })
      }
    }
    newEdges.push({
      id: `${parentId}->${nodeId}`,
      source: parentId,
      target: nodeId,
      sourceHandle: `field-${fieldName}`,
    })
  } else if (field.enumRef) {
    const nodeId = `def__${field.enumRef}`
    if (!existingNodeIds.has(nodeId)) {
      const defSchema = defs[field.enumRef]
      if (defSchema?.enum) {
        newNodes.push({
          id: nodeId,
          type: 'enum',
          position: { x: 0, y: 0 },
          data: {
            label: field.enumRef,
            nodeType: 'enum',
            values: defSchema.enum,
            expanded: new Set(),
            defKey: field.enumRef,
          },
        })
      }
    }
    newEdges.push({
      id: `${parentId}->${nodeId}`,
      source: parentId,
      target: nodeId,
      sourceHandle: `field-${fieldName}`,
    })
  } else if (field.inlineEnum) {
    const nodeId = `${parentId}__${fieldName}__enum`
    if (!existingNodeIds.has(nodeId)) {
      newNodes.push({
        id: nodeId,
        type: 'enum',
        position: { x: 0, y: 0 },
        data: {
          label: fieldName,
          nodeType: 'enum',
          values: field.inlineEnum,
          expanded: new Set(),
        },
      })
    }
    newEdges.push({
      id: `${parentId}->${nodeId}`,
      source: parentId,
      target: nodeId,
      sourceHandle: `field-${fieldName}`,
    })
  }

  return { nodes: newNodes, edges: newEdges }
}
