import type { ArchitectureGraph } from './ir'

const pastelBranches = ['#82bda9', '#91a9d6', '#c99ab8', '#c8ad78', '#9f96cf', '#83b8c8']

/** Color generic fan-out branches until their merge or their next nested fan-out. */
export function parallelEdgeColors(graph: ArchitectureGraph): Map<string, string> {
  const outgoing = new Map(graph.nodes.map((node) => [node.id, [] as typeof graph.edges]))
  const incoming = new Map(graph.nodes.map((node) => [node.id, new Set<string>()]))
  for (const edge of graph.edges) {
    outgoing.get(edge.source)?.push(edge)
    incoming.get(edge.target)?.add(edge.source)
  }
  const colors = new Map<string, string>()
  for (const node of graph.nodes) {
    const branches = outgoing.get(node.id) ?? []
    if (branches.length < 2) continue
    branches.forEach((edge, index) => {
      const color = pastelBranches[index % pastelBranches.length]!
      colors.set(edge.id, color)
      let cursor = edge.target
      const visited = new Set<string>()
      while (!visited.has(cursor)) {
        visited.add(cursor)
        if ((incoming.get(cursor)?.size ?? 0) > 1) break
        const next = outgoing.get(cursor) ?? []
        if (next.length !== 1) break
        colors.set(next[0]!.id, color)
        cursor = next[0]!.target
      }
    })
  }
  return colors
}
