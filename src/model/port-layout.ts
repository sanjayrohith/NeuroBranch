import type { ArchitectureGraph, ArchitectureNode } from '../core/ir'
import { modelAtomRegistry, type AtomPort } from '../core/model-atoms'

/** Reorder only visually-equivalent input plugs so elastics keep their left-to-right order. */
export function orderedNodeInputPorts(graph: ArchitectureGraph, node: ArchitectureNode): AtomPort[] {
  if (!node.atomId) return []
  const ports = modelAtomRegistry[node.atomId]?.inputs ?? []
  const ordered = [...ports]
  const tensors = new Set(ports.map((port) => port.tensor))

  for (const tensor of tensors) {
    const slots = ports.map((port, index) => ({ port, index })).filter(({ port }) => port.tensor === tensor)
    if (slots.length < 2) continue
    const ranked = slots.map(({ port, index }) => {
      const edge = graph.edges.find((candidate) => candidate.target === node.id && candidate.targetPort === port.id)
      const source = edge ? graph.nodes.find((candidate) => candidate.id === edge.source) : undefined
      return { port, originalIndex: index, sourceX: source?.position.x ?? Number.POSITIVE_INFINITY }
    }).sort((left, right) => left.sourceX - right.sourceX || left.originalIndex - right.originalIndex)
    slots.forEach(({ index }, rank) => { ordered[index] = ranked[rank]!.port })
  }

  return ordered
}
