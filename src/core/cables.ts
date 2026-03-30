import type { ArchitectureEdge, ArchitectureGraph, TensorRole } from './ir'

export interface CableEndpointPair {
  sourceId: string
  sourcePort: TensorRole
  sourcePortId?: string
  targetId: string
  targetPort: TensorRole
  targetPortId?: string
}

export interface CableConnectOutcome {
  ok: boolean
  code: 'CABLE_CONNECTED' | 'UNKNOWN_NODE' | 'PORT_TYPE_MISMATCH'
  graph: ArchitectureGraph
  edgeId?: string
  message: string
}

function edgeTargetPort(graph: ArchitectureGraph, edge: ArchitectureEdge): TensorRole {
  if (edge.targetPort) return edge.targetPort as TensorRole
  const source = graph.nodes.find((node) => node.id === edge.source)
  return source?.role === 'attention' ? 'hidden' : (source?.role ?? 'hidden')
}

function uniqueEdgeId(graph: ArchitectureGraph, base: string): string {
  if (!graph.edges.some((edge) => edge.id === base)) return base
  let sequence = 2
  while (graph.edges.some((edge) => edge.id === `${base}-${sequence}`)) sequence += 1
  return `${base}-${sequence}`
}

export function connectCable(graph: ArchitectureGraph, endpoints: CableEndpointPair): CableConnectOutcome {
  const source = graph.nodes.find((node) => node.id === endpoints.sourceId)
  const target = graph.nodes.find((node) => node.id === endpoints.targetId)
  if (!source || !target) {
    return { ok: false, code: 'UNKNOWN_NODE', graph, message: 'Cable endpoint references an unknown atom' }
  }
  if (endpoints.sourcePort !== endpoints.targetPort) {
    return {
      ok: false,
      code: 'PORT_TYPE_MISMATCH',
      graph,
      message: `${endpoints.sourcePort} output cannot plug into ${endpoints.targetPort} input`,
    }
  }

  const targetPortId = endpoints.targetPortId ?? endpoints.targetPort
  const withoutPreviousInput = graph.edges.filter((edge) =>
    !(edge.target === endpoints.targetId && (edge.targetPort ? edge.targetPort === targetPortId : edgeTargetPort(graph, edge) === endpoints.targetPort)),
  )
  const baseId = `${endpoints.sourceId}-${endpoints.targetId}-${targetPortId}`
  const edgeId = uniqueEdgeId({ ...graph, edges: withoutPreviousInput }, baseId)
  const edge: ArchitectureEdge = {
    id: edgeId,
    source: endpoints.sourceId,
    sourcePort: endpoints.sourcePortId ?? endpoints.sourcePort,
    target: endpoints.targetId,
    targetPort: targetPortId,
    label: endpoints.targetPort.toUpperCase(),
  }
  return {
    ok: true,
    code: 'CABLE_CONNECTED',
    graph: { ...graph, edges: [...withoutPreviousInput, edge] },
    edgeId,
    message: `Connected ${endpoints.sourceId}.${endpoints.sourcePort} to ${endpoints.targetId}.${endpoints.targetPort}`,
  }
}

export function disconnectCable(graph: ArchitectureGraph, edgeId: string): ArchitectureGraph {
  return { ...graph, edges: graph.edges.filter((edge) => edge.id !== edgeId) }
}

export function inferredEdgeTargetPort(graph: ArchitectureGraph, edge: ArchitectureEdge): TensorRole {
  return edgeTargetPort(graph, edge)
}
