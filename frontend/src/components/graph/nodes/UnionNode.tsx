import { Handle, Position } from '@xyflow/react'
import type { GraphNodeData } from '../schemaToGraph'

interface Props {
  data: GraphNodeData
}

export default function UnionNode({ data }: Props) {
  return (
    <div className="flex items-center rounded-full border-2 border-violet-400 bg-violet-50 px-4 py-2 shadow-md">
      <Handle type="target" position={Position.Left} className="!h-3 !w-3 !border-violet-400 !bg-violet-400" />
      <span className="text-xs font-semibold text-violet-700">{data.label}</span>
      <Handle type="source" position={Position.Right} className="!h-3 !w-3 !border-violet-400 !bg-violet-400" />
    </div>
  )
}
