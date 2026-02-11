import { Handle, Position } from '@xyflow/react'
import type { FieldInfo, GraphNodeData } from '../schemaToGraph'

interface Props {
  data: GraphNodeData
  onToggleField?: (fieldName: string) => void
}

export default function ModelNode({ data, onToggleField }: Props) {
  return (
    <div className="min-w-[220px] rounded-lg border-2 border-emerald-400 bg-emerald-50 shadow-md">
      <div className="rounded-t-md bg-emerald-500 px-3 py-2 text-sm font-bold text-white">
        {data.label}
      </div>
      <div className="px-1 py-1">
        {data.fields?.map((f: FieldInfo) => {
          const isExpanded = data.expanded.has(f.name)
          return (
            <div key={f.name} className="group relative flex items-center gap-1 px-2 py-0.5">
              {f.expandable && (
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    onToggleField?.(f.name)
                  }}
                  className="flex h-4 w-4 shrink-0 items-center justify-center rounded text-xs text-gray-400 hover:bg-emerald-100 hover:text-emerald-600"
                >
                  {isExpanded ? '−' : '+'}
                </button>
              )}
              {!f.expandable && <span className="inline-block h-4 w-4" />}
              <span className="text-xs font-medium text-gray-700">{f.name}</span>
              <span className="ml-auto text-xs text-gray-400">
                {f.typeSummary}
                {f.optional && !f.typeSummary.endsWith('?') && '?'}
              </span>
              {f.expandable && (
                <Handle
                  type="source"
                  position={Position.Right}
                  id={`field-${f.name}`}
                  className="!absolute !right-0 !h-2 !w-2 !border-emerald-400 !bg-emerald-400"
                />
              )}
            </div>
          )
        })}
      </div>
      <Handle type="target" position={Position.Left} className="!h-3 !w-3 !border-emerald-400 !bg-emerald-400" />
    </div>
  )
}
