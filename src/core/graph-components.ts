import type { ArchitectureGraph } from './ir'

export interface ArchitectureComponent {
  id: string
  label: string
  nodeIds: string[]
  graph: ArchitectureGraph
}

/** Returns stable, undirected connected components for independent architectures on one canvas. */
export function architectureComponents(graph: ArchitectureGraph, knownArchitectures: ArchitectureGraph[] = []): ArchitectureComponent[] {
  const neighbors = new Map(graph.nodes.map((node) => [node.id, new Set<string>()]))
  for (const edge of graph.edges) {
    neighbors.get(edge.source)?.add(edge.target)
    neighbors.get(edge.target)?.add(edge.source)
  }
  const visited = new Set<string>()
  const components: ArchitectureComponent[] = []
  for (const seed of graph.nodes) {
    if (visited.has(seed.id)) continue
    const queue = [seed.id]
    const nodeIds: string[] = []
    visited.add(seed.id)
    while (queue.length > 0) {
      const id = queue.shift()!
      nodeIds.push(id)
      for (const neighbor of neighbors.get(id) ?? []) {
        if (visited.has(neighbor)) continue
        visited.add(neighbor)
        queue.push(neighbor)
      }
    }
    const ids = new Set(nodeIds)
    const nodes = graph.nodes.filter((node) => ids.has(node.id))
    const edges = graph.edges.filter((edge) => ids.has(edge.source) && ids.has(edge.target))
    const source = nodes.find((node) => !edges.some((edge) => edge.target === node.id)) ?? nodes[0]
    const sink = [...nodes].reverse().find((node) => !edges.some((edge) => edge.source === node.id)) ?? nodes.at(-1)
    const exactGroup = graph.groups?.find((group) => group.nodeIds.length === nodeIds.length && group.nodeIds.every((nodeId) => ids.has(nodeId)))
    const known = knownArchitectures
      .filter((candidate) => candidate.nodes.length > 0 && candidate.nodes.every((node) => ids.has(node.id)))
      .sort((left, right) => right.nodes.length - left.nodes.length)[0]
    const metadataName = typeof source?.attributes?.laboArchitectureName === 'string' ? source.attributes.laboArchitectureName : undefined
    const metadataNumber = (key: string) => typeof source?.attributes?.[key] === 'number' ? source.attributes[key] as number : undefined
    const inferredRoute = source?.id === sink?.id ? source?.label : `${source?.label ?? 'Input'} → ${sink?.label ?? 'Output'}`
    const label = exactGroup?.label ?? metadataName ?? known?.name ?? (graph.nodes.length === nodes.length ? graph.name : `Architecture ${components.length + 1} · ${inferredRoute}`)
    const id = `architecture-${components.length + 1}-${source?.id ?? 'graph'}`
    components.push({
      id,
      label,
      nodeIds,
      graph: {
        ...graph,
        id: `${graph.id}-${id}`,
        name: label,
        config: known?.config ?? {
          hiddenSize: metadataNumber('laboArchitectureHiddenSize') ?? graph.config.hiddenSize,
          queryHeads: metadataNumber('laboArchitectureQueryHeads') ?? graph.config.queryHeads,
          keyValueHeads: metadataNumber('laboArchitectureKeyValueHeads') ?? graph.config.keyValueHeads,
          headDim: metadataNumber('laboArchitectureHeadDim') ?? graph.config.headDim,
        },
        nodes,
        edges,
        groups: graph.groups?.filter((group) => group.nodeIds.some((nodeId) => ids.has(nodeId))).map((group) => ({ ...group, nodeIds: group.nodeIds.filter((nodeId) => ids.has(nodeId)) })),
      },
    })
  }
  return components
}
