import type { ArchitectureGraph } from './ir'
import { executionLayers } from './execution-plan'

const cardWidth = 148
const cardHeight = 76
const gridSpacing = 24
const horizontalGap = 72
const verticalGap = 40
const snapUp = (value: number) => Math.ceil(value / gridSpacing) * gridSpacing
const snapNearest = (value: number) => Math.round(value / gridSpacing) * gridSpacing
const horizontalStep = snapUp(cardWidth + horizontalGap)
const verticalStep = snapUp(cardHeight + verticalGap)
const parallelColumnStep = snapUp(cardWidth + 24)
const parallelLengthGap = 32
const maximumParallelSlack = 4
const layoutStartX = 72
const layoutStartY = 72
const componentGap = 168

type Position = { x: number; y: number }
type ParallelBranch = { root: string; nodes: Set<string>; length: number; order: number }
type ParallelRegion = { fork: string; join: string; branches: ParallelBranch[]; span: number }

function mean(values: number[]): number {
  return values.length > 0 ? values.reduce((sum, value) => sum + value, 0) / values.length : 0
}

function connectedComponents(graph: ArchitectureGraph, nodeIds: Set<string>): string[][] {
  const index = new Map(graph.nodes.map((node, position) => [node.id, position]))
  const neighbours = new Map([...nodeIds].map((id) => [id, new Set<string>()]))
  for (const edge of graph.edges) {
    if (nodeIds.has(edge.source) && nodeIds.has(edge.target)) {
      neighbours.get(edge.source)?.add(edge.target)
      neighbours.get(edge.target)?.add(edge.source)
    }
  }
  const pending = new Set(nodeIds)
  const components: string[][] = []
  while (pending.size > 0) {
    const seed = [...pending].sort((left, right) => (index.get(left) ?? 0) - (index.get(right) ?? 0))[0]!
    const queue = [seed]
    const component: string[] = []
    pending.delete(seed)
    while (queue.length > 0) {
      const current = queue.shift()!
      component.push(current)
      for (const neighbour of neighbours.get(current) ?? []) {
        if (!pending.delete(neighbour)) continue
        queue.push(neighbour)
      }
    }
    component.sort((left, right) => (index.get(left) ?? 0) - (index.get(right) ?? 0))
    components.push(component)
  }
  return components
}

function componentGraph(graph: ArchitectureGraph, ids: Set<string>): ArchitectureGraph {
  return {
    ...graph,
    nodes: graph.nodes.filter((node) => ids.has(node.id)),
    edges: graph.edges.filter((edge) => ids.has(edge.source) && ids.has(edge.target)),
    groups: [],
  }
}

/**
 * Assign every node to its latest valid execution rank.
 *
 * An earliest-rank layout makes every parallel branch start beside its source,
 * even when a short branch is not consumed until much later. That creates long
 * empty vertical lanes. Anchoring ranks from the sinks measures each branch by
 * its remaining topological length: the longest continuation stays on the
 * source spine, while shorter branches are delayed and placed left/right near
 * the block that consumes them. Every dependency still flows downward.
 */
function compactTopologyLayers(graph: ArchitectureGraph): { layers: string[][]; slack: Map<string, number> } {
  const topological = executionLayers(graph).flat()
  const incomingIds = new Set(graph.edges.map((edge) => edge.target))
  const earliest = new Map(graph.nodes.map((node) => [node.id, 0]))
  for (const source of topological) {
    const sourceRank = earliest.get(source) ?? 0
    for (const edge of graph.edges.filter((candidate) => candidate.source === source)) {
      earliest.set(edge.target, Math.max(earliest.get(edge.target) ?? 0, sourceRank + 1))
    }
  }
  const depth = Math.max(0, ...earliest.values())
  const distanceToSink = new Map(graph.nodes.map((node) => [node.id, 0]))
  for (const source of [...topological].reverse()) {
    const children = graph.edges.filter((edge) => edge.source === source).map((edge) => edge.target)
    if (children.length > 0) distanceToSink.set(source, Math.max(...children.map((child) => (distanceToSink.get(child) ?? 0) + 1)))
  }
  const layers: string[][] = []
  for (const id of topological) {
    // All independent entry cards are siblings of one invisible super-root.
    // They must therefore start on the same grid row even when their paths to
    // the first merge have different lengths.
    const level = incomingIds.has(id) ? depth - (distanceToSink.get(id) ?? 0) : 0
    ;(layers[level] ??= []).push(id)
  }
  const slack = new Map(graph.nodes.map((node) => [
    node.id,
    Math.max(0, depth - (earliest.get(node.id) ?? 0) - (distanceToSink.get(node.id) ?? 0)),
  ]))
  return { layers: layers.filter(Boolean), slack }
}

function centeredOrderPositions(layers: string[][]): Map<string, number> {
  const positions = new Map<string, number>()
  for (const layer of layers) layer.forEach((id, index) => positions.set(id, index - (layer.length - 1) / 2))
  return positions
}

function reachableDistances(seed: string, outgoing: Map<string, string[]>): Map<string, number> {
  const distances = new Map([[seed, 0]])
  const queue = [seed]
  while (queue.length > 0) {
    const source = queue.shift()!
    const distance = distances.get(source) ?? 0
    for (const target of outgoing.get(source) ?? []) {
      if (distances.has(target)) continue
      distances.set(target, distance + 1)
      queue.push(target)
    }
  }
  return distances
}

/** Find topology-only fork/join regions that can be rendered as nested 2D panels. */
function parallelRegions(graph: ArchitectureGraph): ParallelRegion[] {
  const nodeOrder = new Map(graph.nodes.map((node, index) => [node.id, index]))
  const outgoing = new Map(graph.nodes.map((node) => [node.id, [] as string[]]))
  const incoming = new Map(graph.nodes.map((node) => [node.id, [] as string[]]))
  for (const edge of graph.edges) {
    const targets = outgoing.get(edge.source)
    if (targets && !targets.includes(edge.target)) targets.push(edge.target)
    const sources = incoming.get(edge.target)
    if (sources && !sources.includes(edge.source)) sources.push(edge.source)
  }
  const postDominators = new Map<string, Set<string>>()
  for (const id of executionLayers(graph).flat().reverse()) {
    const children = outgoing.get(id) ?? []
    if (children.length === 0) {
      postDominators.set(id, new Set([id]))
      continue
    }
    const intersection = new Set(postDominators.get(children[0]) ?? [])
    for (const child of children.slice(1)) {
      const childDominators = postDominators.get(child) ?? new Set<string>()
      for (const candidate of [...intersection]) if (!childDominators.has(candidate)) intersection.delete(candidate)
    }
    intersection.add(id)
    postDominators.set(id, intersection)
  }

  const regions: ParallelRegion[] = []
  for (const node of graph.nodes) {
    const directChildren = outgoing.get(node.id) ?? []
    if (directChildren.length < 2) continue
    const childReachability = new Map(directChildren.map((child) => [child, reachableDistances(child, outgoing)]))
    const children = directChildren.filter((child) => !directChildren.some((other) => {
      if (other === child || !childReachability.get(other)?.has(child)) return false
      const reachableFromOther = childReachability.get(other)!
      const bundledInputs = graph.edges.filter((edge) => (
        edge.target === child
        && edge.source !== node.id
        && reachableFromOther.has(edge.source)
      )).length
      // A direct conditioning elastic into a descendant does not create a new
      // visual sibling when that descendant already receives a real bundle
      // from the branch (indices + weights, Q + K, and similar atomics).
      return bundledInputs >= 2
    }))
    if (children.length < 2) continue
    const distances = children.map((child) => reachableDistances(child, outgoing))
    const common = [...distances[0]!.keys()].filter((candidate) => (
      candidate !== node.id
      && children.every((child) => postDominators.get(child)?.has(candidate))
      && (incoming.get(candidate)?.length ?? 0) > 1
    ))
    common.sort((left, right) => {
      const leftMax = Math.max(...distances.map((branch) => branch.get(left) ?? Number.MAX_SAFE_INTEGER))
      const rightMax = Math.max(...distances.map((branch) => branch.get(right) ?? Number.MAX_SAFE_INTEGER))
      if (leftMax !== rightMax) return leftMax - rightMax
      const leftSum = distances.reduce((sum, branch) => sum + (branch.get(left) ?? 0), 0)
      const rightSum = distances.reduce((sum, branch) => sum + (branch.get(right) ?? 0), 0)
      return leftSum - rightSum || (nodeOrder.get(left) ?? 0) - (nodeOrder.get(right) ?? 0)
    })
    const join = common[0]
    if (!join) continue

    const canReachJoin = new Set(reachableDistances(join, incoming).keys())
    const rawBranches = children.map((root, order) => {
      const reachable = reachableDistances(root, outgoing)
      const nodes = new Set([...reachable.keys()].filter((id) => id !== join && canReachJoin.has(id)))
      return { root, nodes, reachable, length: reachable.get(join) ?? 0, order }
    })
    const owner = new Map<string, number>()
    for (const id of new Set(rawBranches.flatMap((branch) => [...branch.nodes]))) {
      const candidates = rawBranches
        .map((branch, index) => ({ index, distance: branch.reachable.get(id) }))
        .filter((candidate): candidate is { index: number; distance: number } => candidate.distance !== undefined)
        .sort((left, right) => left.distance - right.distance || left.index - right.index)
      if (candidates[0]) owner.set(id, candidates[0].index)
    }
    const branches = rawBranches.map((branch) => ({
      root: branch.root,
      length: branch.length,
      order: branch.order,
      nodes: new Set([...branch.nodes].filter((id) => owner.get(id) === branch.order)),
    }))
    regions.push({ fork: node.id, join, branches, span: Math.max(...branches.map((branch) => branch.length)) })
  }
  return regions.sort((left, right) => right.span - left.span || (nodeOrder.get(left.fork) ?? 0) - (nodeOrder.get(right.fork) ?? 0))
}

function parallelAnchors(
  graph: ArchitectureGraph,
  regions: ParallelRegion[],
  levelById: Map<string, number>,
  totalLevels: number,
): { x: Map<string, number>; anchored: Set<string>; fixed: Set<string>; panelLevel: Map<string, number>; spineLocked: Set<string> } {
  const x = new Map(graph.nodes.map((node) => [node.id, 0]))
  const panelLevel = new Map<string, number>()
  const anchored = new Set<string>()
  const outgoing = new Map(graph.nodes.map((node) => [node.id, new Set<string>()]))
  const incoming = new Map(graph.nodes.map((node) => [node.id, [] as string[]]))
  for (const edge of graph.edges) {
    outgoing.get(edge.source)?.add(edge.target)
    incoming.get(edge.target)?.push(edge.source)
  }
  const railThreshold = Math.max(4, Math.ceil(Math.max(1, totalLevels - 1) * 0.65))
  const railBranches = (region: ParallelRegion) => {
    const verticalSpan = Math.max(0, (levelById.get(region.join) ?? 0) - (levelById.get(region.fork) ?? 0))
    const longestBranch = Math.max(...region.branches.map((branch) => branch.nodes.size))
    return region.branches.filter((branch) => (
      verticalSpan >= railThreshold
      && branch.nodes.size < longestBranch
      && ![...branch.nodes].some((id) => (outgoing.get(id)?.size ?? 0) > 1)
    ))
  }
  const regionByFork = new Map(regions.map((region) => [region.fork, region]))
  const incomingIds = new Set(graph.edges.map((edge) => edge.target))
  const roots = graph.nodes.filter((node) => !incomingIds.has(node.id))
  const rootLaneNodes = new Set<string>()
  if (roots.length > 1) {
    const rootWidth = roots.length
    roots.forEach((root, index) => {
      const slot = index - (rootWidth - 1) / 2
      x.set(root.id, slot * parallelColumnStep)
      anchored.add(root.id)
      rootLaneNodes.add(root.id)
    })
    const rootIds = new Set(roots.map((root) => root.id))
    const ancestry = new Map<string, Set<string>>()
    for (const id of executionLayers(graph).flat()) {
      if (rootIds.has(id)) {
        ancestry.set(id, new Set([id]))
        continue
      }
      const inherited = new Set((incoming.get(id) ?? []).flatMap((source) => [...(ancestry.get(source) ?? [])]))
      ancestry.set(id, inherited)
      if (inherited.size !== 1) continue
      const rootId = [...inherited][0]!
      x.set(id, x.get(rootId) ?? 0)
      anchored.add(id)
      rootLaneNodes.add(id)
    }
  }
  const backbone = new Set<string>()
  let cursor = graph.nodes.find((node) => !incomingIds.has(node.id))?.id
  while (cursor && !backbone.has(cursor)) {
    backbone.add(cursor)
    const children = [...(outgoing.get(cursor) ?? [])]
    if (children.length === 0) break
    if (children.length === 1) {
      cursor = children[0]
      continue
    }
    const region = regionByFork.get(cursor)
    if (!region) {
      cursor = children[0]
      continue
    }
    const rails = railBranches(region)
    if (rails.length > 0) {
      const main = [...region.branches]
        .filter((branch) => !rails.includes(branch))
        .sort((left, right) => right.length - left.length || left.order - right.order)[0]
      cursor = main?.root ?? region.join
    } else {
      cursor = region.join
    }
  }
  for (const id of backbone) anchored.add(id)
  const axis = new Set([...backbone, ...rootLaneNodes, ...roots.map((root) => root.id)])
  const spineInsertions: { id: string; atLevel: number }[] = []
  const oneAtomBypassLocks = new Set<string>()
  const regionMembership = new Map<string, number>()
  for (const region of regions) {
    for (const branch of region.branches) {
      for (const id of branch.nodes) regionMembership.set(id, (regionMembership.get(id) ?? 0) + 1)
    }
  }
  const regionWidthMemo = new Map<ParallelRegion, number>()
  const regionNodes = (region: ParallelRegion) => new Set(region.branches.flatMap((branch) => [...branch.nodes]))
  const branchDivWidth = (owner: ParallelRegion, branch: ParallelBranch, trail: Set<ParallelRegion>): number => {
    let width = 1
    for (const candidate of regions) {
      if (candidate === owner || trail.has(candidate) || !branch.nodes.has(candidate.fork)) continue
      const candidateNodes = regionNodes(candidate)
      const contained = [...candidateNodes].every((id) => branch.nodes.has(id) || id === owner.join)
      if (!contained) continue
      width = Math.max(width, regionDivWidth(candidate, new Set([...trail, owner])))
    }
    return width
  }
  function regionDivWidth(region: ParallelRegion, trail = new Set<ParallelRegion>()): number {
    const cached = regionWidthMemo.get(region)
    if (cached !== undefined && trail.size === 0) return cached
    if (trail.has(region)) return 1
    const visibleBranches = region.branches.filter((branch) => branch.nodes.size > 0)
    const width = Math.max(1, visibleBranches.reduce((sum, branch) => sum + branchDivWidth(region, branch, new Set([...trail, region])), 0))
    if (trail.size === 0) regionWidthMemo.set(region, width)
    return width
  }
  for (const region of regions) {
    const base = x.get(region.fork) ?? 0
    const sorted = [...region.branches].sort((left, right) => left.order - right.order)
    const shortest = Math.min(...sorted.map((branch) => branch.length))
    const lengthSpread = Math.min(maximumParallelSlack, Math.max(0, region.span - shortest))
    const columnStep = horizontalStep + lengthSpread * parallelLengthGap
    const localColumnStep = snapUp(Math.max(parallelColumnStep, columnStep * 0.7))
    const rails = railBranches(region)
    const local = sorted.filter((branch) => !rails.includes(branch) && branch.nodes.size > 0)
    if (local.length < 2 && rails.length === 0) continue
    const localWidths = local.map((branch) => branchDivWidth(region, branch, new Set([region])))
    const localWidth = localWidths.reduce((sum, width) => sum + width, 0)
    let slotCursor = -localWidth / 2
    let localSlots = localWidths.map((width) => {
      const center = slotCursor + width / 2
      slotCursor += width
      return center
    })
    if (local.length === 2) {
      // Recursive div width reserves room for descendants; it must not push
      // the two immediate sibling roots out to -1/+1. Every binary fork starts
      // at the atomic half lanes and its nested forks expand from there.
      localSlots = [-0.5, 0.5]
    }
    const localPanelLimit = Math.max(4, Math.floor(totalLevels * 0.35))
    const isLocalPanel = region.span <= localPanelLimit
    const branchMeta = local.map((branch, index) => ({
      branch,
      index,
      levelShift: isLocalPanel && index > 0 && branch.length < region.span ? -1 : 0,
    }))
    branchMeta.forEach(({ branch, index, levelShift }) => {
      for (const id of branch.nodes) {
        const branchStep = local.length === 2 ? parallelColumnStep : localColumnStep
        const offset = (localSlots[index] ?? 0) * branchStep
        // A branch owns its lane until the actual join. Do not recenter its
        // final sibling merely because a separate backbone heuristic selected
        // that atom while walking the graph.
        x.set(id, base + offset)
        if (levelShift < 0) panelLevel.set(id, Math.min(panelLevel.get(id) ?? Number.POSITIVE_INFINITY, (levelById.get(id) ?? 0) + levelShift))
        axis.add(id)
        anchored.add(id)
      }
    })
    rails.forEach((branch, index) => {
      const distance = Math.floor(index / 2) + 1
      const railStep = parallelColumnStep
      const offset = (index % 2 === 0 ? distance : -distance) * railStep
      for (const id of branch.nodes) {
        if (!backbone.has(id)) x.set(id, base + offset)
        axis.add(id)
        anchored.add(id)
      }
      const railOnly = [...branch.nodes]
        .filter((id) => (regionMembership.get(id) ?? 0) === 1)
        .sort((left, right) => (levelById.get(left) ?? 0) - (levelById.get(right) ?? 0))
      const forkLevel = levelById.get(region.fork) ?? 0
      const joinLevel = levelById.get(region.join) ?? forkLevel + 1
      railOnly.forEach((id, railIndex) => {
        if (railOnly.length === 1) {
          // A single atom spanning almost the whole graph is not a floating
          // side card. Insert it on the dominant spine immediately before its
          // first consumer, then shift that consumer and every following row.
          const consumers = outgoing.get(id) ?? new Set<string>()
          const inputParallelLength = Math.max(...region.branches.map((candidateBranch) => candidateBranch.nodes.size))
          const outputParallelLengths = regions
            .filter((candidate) => candidate !== region)
            .filter((candidate) => (
              candidate.branches.filter((candidateBranch) => candidateBranch.nodes.size > 0).length > 1
              && candidate.branches.some((candidateBranch) => [...consumers].some((consumer) => candidateBranch.nodes.has(consumer)))
            ))
            .map((candidate) => Math.max(...candidate.branches.map((candidateBranch) => candidateBranch.nodes.size)))
          const outputParallelLength = Math.max(0, ...outputParallelLengths)
          const railDirection = index % 2 === 0 ? -1 : 1
          // Equal input/output spans form a two-card lane at one half-step.
          // Only an input-dominant parallel pushes the upstream card out to
          // the second half-step before it returns toward the main axis.
          const parallelDepth = outputParallelLength > 0 && inputParallelLength > outputParallelLength ? 1 : 0.5
          x.set(id, base + railDirection * parallelColumnStep * parallelDepth)
          const consumerLevel = Math.min(...[...(outgoing.get(id) ?? [])].map((target) => levelById.get(target) ?? joinLevel))
          spineInsertions.push({ id, atLevel: consumerLevel })
          return
        }
        const ratio = (railIndex + 1) / (railOnly.length + 1)
        panelLevel.set(id, Math.round(forkLevel + (joinLevel - forkLevel) * ratio))
      })
    })
    anchored.add(region.fork)
    anchored.add(region.join)
    axis.add(region.fork)
    axis.add(region.join)
    x.set(region.join, base)
  }

  // Exact one-atom bypass: parent -> child -> join, plus parent -> join.
  // Parent and join remain on the vertical axis; only the enclosed child takes
  // a half-lane. This must not be generalized to long rails or crossed cables.
  for (const parent of graph.nodes) {
    const parentTargets = outgoing.get(parent.id) ?? new Set<string>()
    const candidates = [...parentTargets].filter((child) => {
      const childTargets = outgoing.get(child) ?? new Set<string>()
      const childSources = new Set(incoming.get(child) ?? [])
      if (childTargets.size !== 1 || childSources.size !== 1 || !childSources.has(parent.id)) return false
      const join = [...childTargets][0]!
      return parentTargets.has(join)
    })
    candidates.forEach((child, index) => {
      const direction = index % 2 === 0 ? -1 : 1
      x.set(child, (x.get(parent.id) ?? 0) + direction * parallelColumnStep / 2)
      anchored.add(child)
      axis.add(child)
      oneAtomBypassLocks.add(child)
    })
  }
  const orderedInsertions = [...spineInsertions].sort((left, right) => left.atLevel - right.atLevel || left.id.localeCompare(right.id))
  const insertionIds = new Set(orderedInsertions.map(({ id }) => id))
  const spineLocked = new Set([...insertionIds, ...oneAtomBypassLocks])
  if (orderedInsertions.length > 0) for (const id of backbone) spineLocked.add(id)
  for (const node of graph.nodes) {
    if (insertionIds.has(node.id)) continue
    const level = panelLevel.get(node.id) ?? levelById.get(node.id) ?? 0
    const shift = orderedInsertions.filter(({ atLevel }) => atLevel <= level).length
    if (shift > 0) panelLevel.set(node.id, level + shift)
  }
  orderedInsertions.forEach(({ id, atLevel }, insertionIndex) => {
    const prior = orderedInsertions.slice(0, insertionIndex).filter((insertion) => insertion.atLevel <= atLevel).length
    panelLevel.set(id, atLevel + prior)
    axis.add(id)
    for (const successor of backbone) {
      if ((levelById.get(successor) ?? -1) === atLevel) spineLocked.add(successor)
    }
  })

  // A compact panel may pull a short branch upward into a free side cell. A
  // single horizontal hand-off is readable there, but a multi-elastic bundle
  // (for example Top-K indices + weights) must retain a full vertical row.
  const effectiveLevel = (id: string) => panelLevel.get(id) ?? levelById.get(id) ?? 0
  for (const id of executionLayers(graph).flat()) {
    const parentRequirements = (incoming.get(id) ?? []).map((source) => {
      const sourceLevel = effectiveLevel(source)
      const pairElasticCount = graph.edges.filter((edge) => edge.source === source && edge.target === id).length
      const hasFreeSideCell = Math.abs((x.get(source) ?? 0) - (x.get(id) ?? 0)) >= parallelColumnStep
      return sourceLevel + (pairElasticCount === 1 && hasFreeSideCell ? 0 : 1)
    })
    if (parentRequirements.length === 0) continue
    const minimumLevel = Math.max(...parentRequirements)
    if (effectiveLevel(id) < minimumLevel) panelLevel.set(id, minimumLevel)
  }

  return { x, anchored, fixed: axis, panelLevel, spineLocked }
}

/** Repeated barycentric sweeps reduce crossings without making the result depend on old X/Y coordinates. */
function orderLayers(graph: ArchitectureGraph, layers: string[][]): string[][] {
  const nodeIndex = new Map(graph.nodes.map((node, index) => [node.id, index]))
  const incoming = new Map(graph.nodes.map((node) => [node.id, [] as string[]]))
  const outgoing = new Map(graph.nodes.map((node) => [node.id, [] as string[]]))
  for (const edge of graph.edges) {
    incoming.get(edge.target)?.push(edge.source)
    outgoing.get(edge.source)?.push(edge.target)
  }
  const ordered = layers.map((layer) => [...layer].sort((left, right) => (nodeIndex.get(left) ?? 0) - (nodeIndex.get(right) ?? 0)))

  const sweep = (direction: 'down' | 'up') => {
    const levels = direction === 'down'
      ? Array.from({ length: ordered.length - 1 }, (_, index) => index + 1)
      : Array.from({ length: ordered.length - 1 }, (_, index) => ordered.length - index - 2)
    for (const level of levels) {
      const positions = centeredOrderPositions(ordered)
      const neighbours = direction === 'down' ? incoming : outgoing
      const previous = new Map(ordered[level]!.map((id, index) => [id, index]))
      ordered[level]!.sort((left, right) => {
        const leftNeighbours = neighbours.get(left) ?? []
        const rightNeighbours = neighbours.get(right) ?? []
        const leftScore = leftNeighbours.length > 0 ? mean(leftNeighbours.map((id) => positions.get(id) ?? 0)) : previous.get(left) ?? 0
        const rightScore = rightNeighbours.length > 0 ? mean(rightNeighbours.map((id) => positions.get(id) ?? 0)) : previous.get(right) ?? 0
        if (Math.abs(leftScore - rightScore) > 1e-6) return leftScore - rightScore
        return (nodeIndex.get(left) ?? 0) - (nodeIndex.get(right) ?? 0)
      })
    }
  }

  for (let pass = 0; pass < 6; pass += 1) {
    sweep('down')
    sweep('up')
  }
  return ordered
}

function separatedLayer(ids: string[], desired: Map<string, number>, slack: Map<string, number>, fixed: Set<string>): Map<string, number> {
  const result = new Map<string, number>()
  const gap = (left: string, right: string) => {
    if (fixed.has(left) && fixed.has(right)) return parallelColumnStep
    const pathSlack = Math.min(maximumParallelSlack, Math.max(slack.get(left) ?? 0, slack.get(right) ?? 0))
    return horizontalStep + pathSlack * parallelLengthGap
  }
  const pivot = ids.findIndex((id) => fixed.has(id))
  if (pivot >= 0) {
    const pivotId = ids[pivot]!
    result.set(pivotId, desired.get(pivotId) ?? 0)
    for (let index = pivot - 1; index >= 0; index -= 1) {
      const id = ids[index]!
      const right = ids[index + 1]!
      result.set(id, Math.min(desired.get(id) ?? 0, (result.get(right) ?? 0) - gap(id, right)))
    }
    for (let index = pivot + 1; index < ids.length; index += 1) {
      const id = ids[index]!
      const left = ids[index - 1]!
      result.set(id, Math.max(desired.get(id) ?? 0, (result.get(left) ?? 0) + gap(left, id)))
    }
    return result
  }
  let previous = Number.NEGATIVE_INFINITY
  let previousId: string | undefined
  for (const id of ids) {
    const pathSlack = Math.min(maximumParallelSlack, Math.max(slack.get(id) ?? 0, previousId ? slack.get(previousId) ?? 0 : 0))
    const x = Math.max(desired.get(id) ?? 0, previous + horizontalStep + pathSlack * parallelLengthGap)
    result.set(id, x)
    previous = x
    previousId = id
  }
  const shift = mean([...result.values()]) - mean(ids.map((id) => desired.get(id) ?? 0))
  for (const [id, x] of result) result.set(id, x - shift)
  return result
}

function layoutComponent(graph: ArchitectureGraph): { positions: Map<string, Position>; width: number; height: number } {
  const topology = compactTopologyLayers(graph)
  const layers = orderLayers(graph, topology.layers)
  const levelById = new Map<string, number>()
  layers.forEach((layer, level) => layer.forEach((id) => levelById.set(id, level)))
  const regions = parallelRegions(graph)
  const anchors = parallelAnchors(graph, regions, levelById, layers.length)
  const incoming = new Map(graph.nodes.map((node) => [node.id, [] as string[]]))
  const outgoing = new Map(graph.nodes.map((node) => [node.id, [] as string[]]))
  for (const edge of graph.edges) {
    incoming.get(edge.target)?.push(edge.source)
    outgoing.get(edge.source)?.push(edge.target)
  }
  const x = new Map<string, number>()

  layers.forEach((layer) => {
    const layerOrder = new Map(layer.map((id, index) => [id, index]))
    layer.sort((left, right) => (anchors.x.get(left) ?? 0) - (anchors.x.get(right) ?? 0) || (layerOrder.get(left) ?? 0) - (layerOrder.get(right) ?? 0))
    layer.forEach((id, index) => x.set(id, anchors.anchored.has(id) ? anchors.x.get(id) ?? 0 : (index - (layer.length - 1) / 2) * horizontalStep))
  })

  const alignLayer = (layer: string[], neighbours: Map<string, string[]>) => {
    const desired = new Map<string, number>()
    for (const id of layer) {
      const level = levelById.get(id) ?? 0
      const neighbourX = (neighbours.get(id) ?? [])
        .filter((neighbour) => Math.abs((levelById.get(neighbour) ?? level) - level) === 1)
        .map((neighbour) => x.get(neighbour))
        .filter((value): value is number => value !== undefined)
      const neighbourCenter = neighbourX.length > 0 ? mean(neighbourX) : x.get(id) ?? 0
      const anchor = anchors.x.get(id) ?? 0
      desired.set(id, anchors.anchored.has(id) ? anchor : neighbourCenter)
    }
    for (const [id, position] of separatedLayer(layer, desired, topology.slack, anchors.fixed)) x.set(id, position)
  }

  // Coordinate sweeps keep a branch in one lane across several ranks and let
  // merge points pull their producers together without depending on old XY.
  for (let pass = 0; pass < 6; pass += 1) {
    for (let level = 1; level < layers.length; level += 1) alignLayer(layers[level]!, incoming)
    for (let level = layers.length - 2; level >= 0; level -= 1) alignLayer(layers[level]!, outgoing)
  }

  // Topology panels own their final lanes. In particular, a long singleton
  // rail inserted into the spine must not be pushed sideways again merely
  // because its original latest-valid rank contained another atom.
  for (const id of anchors.spineLocked) {
    const anchor = anchors.x.get(id)
    if (anchor !== undefined) x.set(id, anchor)
  }

  const layerY: number[] = []
  let nextY = 0
  for (let level = 0; level < layers.length; level += 1) {
    layerY[level] = nextY
    nextY += verticalStep
  }
  const positions = new Map<string, Position>()
  const minX = Math.min(...x.values())
  const maxX = Math.max(...x.values()) + cardWidth
  layers.forEach((layer, level) => layer.forEach((id) => {
    positions.set(id, { x: (x.get(id) ?? 0) - minX, y: (anchors.panelLevel.get(id) ?? level) * verticalStep })
  }))
  return { positions, width: maxX - minX, height: Math.max(cardHeight, (layerY.at(-1) ?? 0) + cardHeight) }
}

function collides(position: Position, occupied: Position[]): boolean {
  return occupied.some((other) => Math.abs(other.x - position.x) < cardWidth + 22 && Math.abs(other.y - position.y) < cardHeight + 30)
}

function groupPositions(graph: ArchitectureGraph, positions: Map<string, Position>, arranged: Set<string>) {
  return graph.groups?.map((group) => {
    const children = group.nodeIds.filter((id) => arranged.has(id)).map((id) => positions.get(id)).filter((position): position is Position => Boolean(position))
    if (children.length === 0) return group
    const center = mean(children.map((position) => position.x + cardWidth / 2))
    return { ...group, position: { x: center - 170, y: Math.min(...children.map((position) => position.y)) - 55 } }
  })
}

export function findOpenGraphPosition(graph: ArchitectureGraph): Position {
  if (graph.nodes.length === 0) return { x: layoutStartX, y: layoutStartY }
  const connected = new Set(graph.edges.flatMap((edge) => [edge.source, edge.target]))
  const structural = graph.nodes.filter((node) => connected.has(node.id))
  const baseX = snapNearest(structural.length > 0 ? Math.max(...structural.map((node) => node.position.x)) + horizontalStep + 40 : layoutStartX)
  const baseY = snapNearest(structural.length > 0 ? Math.min(...structural.map((node) => node.position.y)) : layoutStartY)
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 80; row += 1) {
      const candidate = { x: baseX + column * horizontalStep, y: baseY + row * verticalStep }
      if (!collides(candidate, graph.nodes.map((node) => node.position))) return candidate
    }
  }
  return { x: baseX + horizontalStep * 4, y: baseY }
}

/** Deterministic compact DAG layout: Y is the longest-path rank and X follows parallel topology. */
export function layoutArchitectureGraph(graph: ArchitectureGraph, nodeIds?: Iterable<string>): ArchitectureGraph {
  const arranged = new Set(nodeIds ?? graph.nodes.map((node) => node.id))
  if (arranged.size === 0) return graph
  try {
    executionLayers(graph)
  } catch {
    return graph
  }

  const external = graph.nodes.filter((node) => !arranged.has(node.id))
  const occupied = external.map((node) => ({ ...node.position }))
  const positions = new Map<string, Position>()
  const components = connectedComponents(graph, arranged)
  let fullCursorX = layoutStartX

  for (const componentIds of components) {
    const ids = new Set(componentIds)
    const local = layoutComponent(componentGraph(graph, ids))
    const incomingAnchors = graph.edges.filter((edge) => !arranged.has(edge.source) && ids.has(edge.target)).map((edge) => graph.nodes.find((node) => node.id === edge.source)!).filter(Boolean)
    const anchorX = incomingAnchors.length > 0 ? mean(incomingAnchors.map((node) => node.position.x)) : undefined
    const baseY = snapNearest(incomingAnchors.length > 0 ? Math.max(...incomingAnchors.map((node) => node.position.y)) + verticalStep : layoutStartY)
    let baseX = external.length === 0 ? fullCursorX : anchorX !== undefined ? anchorX - local.width / 2 + cardWidth / 2 : Math.max(layoutStartX, ...external.map((node) => node.position.x + cardWidth + componentGap))

    baseX = snapNearest(baseX)
    if (external.length > 0) {
      const candidates = [0, 1, -1, 2, -2, 3, -3, 4]
      const clearOffset = candidates.find((step) => [...local.positions.values()].every((position) => !collides({ x: position.x + baseX + step * horizontalStep, y: position.y + baseY }, occupied))) ?? 4
      baseX += clearOffset * horizontalStep
    }
    for (const [id, position] of local.positions) {
      const placed = { x: snapNearest(position.x + baseX), y: snapNearest(position.y + baseY) }
      positions.set(id, placed)
      occupied.push(placed)
    }
    if (external.length === 0) fullCursorX += local.width + componentGap
  }

  return {
    ...graph,
    nodes: graph.nodes.map((node) => positions.has(node.id) ? { ...node, position: positions.get(node.id)! } : node),
    groups: groupPositions(graph, positions, arranged),
  }
}

/** Pack one or more newly-created architectures to the right of existing user work. */
export function layoutParallelArchitecture(graph: ArchitectureGraph, nodeIds: Iterable<string>): ArchitectureGraph {
  const arranged = new Set(nodeIds)
  if (arranged.size === 0) return graph
  const branch = componentGraph(graph, arranged)
  let arrangedBranch: ArchitectureGraph
  try {
    arrangedBranch = layoutArchitectureGraph(branch)
  } catch {
    return graph
  }
  const existing = graph.nodes.filter((node) => !arranged.has(node.id))
  const branchMinX = Math.min(...arrangedBranch.nodes.map((node) => node.position.x))
  const branchMinY = Math.min(...arrangedBranch.nodes.map((node) => node.position.y))
  const targetX = existing.length > 0 ? Math.max(...existing.map((node) => node.position.x)) + cardWidth + componentGap : layoutStartX
  const targetY = existing.length > 0 ? Math.min(...existing.map((node) => node.position.y)) : layoutStartY
  const positions = new Map(arrangedBranch.nodes.map((node) => [node.id, { x: node.position.x - branchMinX + targetX, y: node.position.y - branchMinY + targetY }]))
  return {
    ...graph,
    nodes: graph.nodes.map((node) => positions.has(node.id) ? { ...node, position: positions.get(node.id)! } : node),
    groups: groupPositions(graph, positions, arranged),
  }
}
