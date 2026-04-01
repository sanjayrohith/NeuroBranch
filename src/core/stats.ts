import type { ArchitectureGraph, ArchitectureNode } from './ir'

const parameterCounters: Partial<Record<ArchitectureNode['kind'], (node: ArchitectureNode) => number>> = {
  linear: (node) => {
    const input = Number(node.attributes?.inFeatures ?? 0)
    const output = Number(node.attributes?.outFeatures ?? 0)
    const bias = node.attributes?.bias === true ? output : 0
    return input * output + bias
  },
}

function formatCount(value: number): string {
  if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(2)}B`
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(2)}K`
  return String(value)
}

export function deriveGraphStats(graph: ArchitectureGraph) {
  const parameterCount = graph.nodes.reduce(
    (total, node) => total + (parameterCounters[node.kind]?.(node) ?? 0),
    0,
  )

  return {
    parameterCount,
    formattedParameterCount: formatCount(parameterCount),
    nodeCount: graph.nodes.length,
    edgeCount: graph.edges.length,
  }
}
