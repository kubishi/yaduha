import { Handle, Position } from '@xyflow/react'
import type { GraphNodeData } from '../schemaToGraph'

interface Props {
  data: GraphNodeData
}

export default function EnumNode({ data }: Props) {
  return (
    <div className="min-w-[140px] rounded-lg border-2 border-amber-400 bg-amber-50 shadow-md">
      <div className="rounded-t-md bg-amber-400 px-3 py-1.5 text-xs font-bold text-amber-900">
        {data.label}
      </div>
      <div className="max-h-40 overflow-y-auto px-2 py-1">
        {data.values?.map((v, i) => (
          <div key={i} className="py-0.5 text-xs text-amber-800">
            {String(v)}
          </div>
        ))}
      </div>
      <Handle type="target" position={Position.Left} className="!h-3 !w-3 !border-amber-400 !bg-amber-400" />
    </div>
  )
}
