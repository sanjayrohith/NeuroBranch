import type { ArchitectureGraph } from './ir'

export function executionLayers(graph: ArchitectureGraph): string[][] {
  const index = new Map(graph.nodes.map((node, position) => [node.id, position]))
  const indegree = new Map(graph.nodes.map((node) => [node.id, 0]))
  const outgoing = new Map(graph.nodes.map((node) => [node.id, [] as string[]]))

  for (const edge of graph.edges) {
    if (!indegree.has(edge.source) || !indegree.has(edge.target)) throw new Error(`Unknown edge endpoint: ${edge.id}`)
    indegree.set(edge.target, (indegree.get(edge.target) ?? 0) + 1)
    outgoing.get(edge.source)?.push(edge.target)
  }

  const stable = (ids: string[]) => ids.sort((left, right) => (index.get(left) ?? 0) - (index.get(right) ?? 0))
  let current = stable(graph.nodes.filter((node) => indegree.get(node.id) === 0).map((node) => node.id))
  const layers: string[][] = []
  let visited = 0

  while (current.length > 0) {
    layers.push(current)
    visited += current.length
    const next = new Set<string>()
    for (const source of current) {
      for (const target of outgoing.get(source) ?? []) {
        const remaining = (indegree.get(target) ?? 0) - 1
        indegree.set(target, remaining)
        if (remaining === 0) next.add(target)
      }
    }
    current = stable([...next])
  }

  if (visited !== graph.nodes.length) throw new Error('Cannot plan a cyclic graph')
  return layers
}
