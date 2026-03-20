import { compileRegistryGraph } from './pytorch-compiler'

export type NodeKind =
  | 'input'
  | 'linear'
  | 'split-heads'
  | 'qk-norm'
  | 'rope'
  | 'sdpa'
  | 'merge-heads'
  | 'output'

  | 'gate'
  | 'add'
  | 'concatenate'
  | 'custom-pytorch'
  | 'semantic'

export type TensorRole = 'token-ids' | 'image' | 'video' | 'audio' | 'hidden' | 'query' | 'key' | 'value' | 'attention' | 'output' | 'logits' | 'labels' | 'scalar' | 'routing-logits' | 'expert-indices' | 'routing-weights'

export interface ArchitectureNode {
  id: string
  kind: NodeKind
  label: string
  role: TensorRole
  position: { x: number; y: number }
  attributes?: Record<string, number | string | boolean>
  code?: string
  atomId?: string
  /** Internal typed graph carried by a reusable composite card. Legacy custom cards omit it. */
  customCardGraph?: ArchitectureGraph
}

export interface ArchitectureGroup {
  id: string
  kind: 'qkv-projection'
  label: string
  nodeIds: string[]
  expanded: boolean
  position: { x: number; y: number }
}

export interface ArchitectureEdge {
  id: string
  source: string
  target: string
  sourcePort?: string
  targetPort?: string
  label?: string
}

export interface ArchitectureGraph {
  id: string
  name: string
  architecture: 'gqa' | 'custom'
  config: {
    hiddenSize: number
    queryHeads: number
    keyValueHeads: number
    headDim: number
  }
  nodes: ArchitectureNode[]
  groups?: ArchitectureGroup[]
  edges: ArchitectureEdge[]
  contracts: {
    causal: boolean
    preservesGqaAtZeroGate: boolean
    sdpaCompatible: boolean
    contextualValue: boolean
  }
}

export interface GraphValidation {
  valid: boolean
  errors: string[]
}

export interface CompositeBlock {
  id: string
  label: string
  nodeIds: string[]
  nodes: ArchitectureNode[]
  inputEdges: ArchitectureEdge[]
  outputEdges: ArchitectureEdge[]
}

export function validateGraph(graph: ArchitectureGraph): GraphValidation {
  const errors: string[] = []
  const ids = new Set<string>()

  for (const node of graph.nodes) {
    if (ids.has(node.id)) errors.push(`Duplicate node id: ${node.id}`)
    ids.add(node.id)
  }

  for (const edge of graph.edges) {
    if (!ids.has(edge.source)) errors.push(`Unknown edge source: ${edge.source}`)
    if (!ids.has(edge.target)) errors.push(`Unknown edge target: ${edge.target}`)
  }

  const registryGraph = graph.nodes.every((node) => node.kind === 'input' || node.kind === 'semantic' || node.kind === 'custom-pytorch')
  if (registryGraph && graph.nodes.some((node) => node.kind === 'semantic' || node.kind === 'custom-pytorch')) {
    if (errors.length === 0) {
      try {
        compileRegistryGraph(graph)
      } catch (error) {
        errors.push(error instanceof Error ? error.message : String(error))
      }
    }
    return { valid: errors.length === 0, errors }
  }

  const sdpa = graph.nodes.find((node) => node.kind === 'sdpa')
  if (!sdpa) errors.push('Graph requires an SDPA node')
  for (const role of ['query', 'key', 'value'] as const) {
    const connected = sdpa && graph.edges.some((edge) => {
      const source = graph.nodes.find((node) => node.id === edge.source)
      return edge.target === sdpa.id && source?.role === role
    })
    if (!connected) {
      errors.push(`SDPA requires a connected ${role} input`)
    }
  }

  return { valid: errors.length === 0, errors }
}

export function updateNodeAttributes(
  graph: ArchitectureGraph,
  nodeId: string,
  attributes: Record<string, number | string | boolean>,
): ArchitectureGraph {
  if (!graph.nodes.some((node) => node.id === nodeId)) throw new Error(`Unknown node: ${nodeId}`)
  return {
    ...graph,
    nodes: graph.nodes.map((node) => node.id === nodeId
      ? { ...node, attributes: { ...node.attributes, ...attributes } }
      : node),
  }
}

export function moveNode(
  graph: ArchitectureGraph,
  nodeId: string,
  position: { x: number; y: number },
): ArchitectureGraph {
  if (!graph.nodes.some((node) => node.id === nodeId)) throw new Error(`Unknown node: ${nodeId}`)
  return {
    ...graph,
    nodes: graph.nodes.map((node) => node.id === nodeId ? { ...node, position } : node),
  }
}

export function moveGroup(
  graph: ArchitectureGraph,
  groupId: string,
  position: { x: number; y: number },
): ArchitectureGraph {
  if (!graph.groups?.some((group) => group.id === groupId)) throw new Error(`Unknown group: ${groupId}`)
  return {
    ...graph,
    groups: graph.groups.map((group) => group.id === groupId ? { ...group, position } : group),
  }
}

export function addNode(graph: ArchitectureGraph, node: ArchitectureNode): ArchitectureGraph {
  if (graph.nodes.some((candidate) => candidate.id === node.id)) {
    throw new Error(`Duplicate node id: ${node.id}`)
  }
  return { ...graph, nodes: [...graph.nodes, { ...node }] }
}

export function duplicateNode(
  graph: ArchitectureGraph,
  nodeId: string,
  duplicateId: string,
): ArchitectureGraph {
  const source = graph.nodes.find((node) => node.id === nodeId)
  if (!source) throw new Error(`Unknown node: ${nodeId}`)
  return addNode(graph, {
    ...source,
    id: duplicateId,
    label: `${source.label} copy`,
    position: { x: source.position.x + 24, y: source.position.y + 24 },
    attributes: source.attributes ? { ...source.attributes } : undefined,
  })
}

export function removeNode(graph: ArchitectureGraph, nodeId: string): ArchitectureGraph {
  if (!graph.nodes.some((node) => node.id === nodeId)) throw new Error(`Unknown node: ${nodeId}`)
  return {
    ...graph,
    nodes: graph.nodes.filter((node) => node.id !== nodeId),
    edges: graph.edges.filter((edge) => edge.source !== nodeId && edge.target !== nodeId),
  }
}

export function createCompositeBlock(
  graph: ArchitectureGraph,
  selection: { id: string; label: string; nodeIds: string[] },
): CompositeBlock {
  const selected = new Set(selection.nodeIds)
  const nodes = selection.nodeIds.map((id) => {
    const node = graph.nodes.find((candidate) => candidate.id === id)
    if (!node) throw new Error(`Unknown node in composite block: ${id}`)
    return node
  })

  return {
    ...selection,
    nodes,
    inputEdges: graph.edges.filter((edge) => !selected.has(edge.source) && selected.has(edge.target)),
    outputEdges: graph.edges.filter((edge) => selected.has(edge.source) && !selected.has(edge.target)),
  }
}

function pythonIdentifier(value: string): string {
  return value.replaceAll('-', '_').replace(/[^A-Za-z0-9_]/g, '_')
}

function moduleName(node: ArchitectureNode): string {
  return node.id === 'output' ? 'out_proj' : pythonIdentifier(node.id)
}

function topologicalNodes(graph: ArchitectureGraph): ArchitectureNode[] {
  const indegree = new Map(graph.nodes.map((node) => [node.id, 0]))
  const outgoing = new Map(graph.nodes.map((node) => [node.id, [] as string[]]))
  for (const edge of graph.edges) {
    indegree.set(edge.target, (indegree.get(edge.target) ?? 0) + 1)
    outgoing.get(edge.source)?.push(edge.target)
  }
  const queue = graph.nodes.filter((node) => indegree.get(node.id) === 0)
  const ordered: ArchitectureNode[] = []
  while (queue.length > 0) {
    const node = queue.shift()!
    ordered.push(node)
    for (const target of outgoing.get(node.id) ?? []) {
      const next = (indegree.get(target) ?? 0) - 1
      indegree.set(target, next)
      if (next === 0) queue.push(graph.nodes.find((candidate) => candidate.id === target)!)
    }
  }
  return ordered.length === graph.nodes.length ? ordered : graph.nodes
}

function edgeMarker(edge: ArchitectureEdge): string {
  const sourcePort = edge.sourcePort ?? 'output'
  const targetPort = edge.targetPort ?? 'input'
  return `        # labo:edge=${edge.id} source=${edge.source} target=${edge.target} source_port=${sourcePort} target_port=${targetPort}`
}

function compileGqaForward(graph: ArchitectureGraph): string {
  const values = new Map<string, string>()
  const lines = ['        batch, sequence, _ = hidden_states.shape']
  const incoming = (nodeId: string) => graph.edges.filter((edge) => edge.target === nodeId)

  for (const node of topologicalNodes(graph)) {
    if (node.kind === 'input') {
      values.set(node.id, 'hidden_states')
      continue
    }
    const inputEdges = incoming(node.id)
    for (const edge of inputEdges) lines.push(edgeMarker(edge))
    const firstInput = inputEdges[0] ? values.get(inputEdges[0].source) : undefined

    if (node.kind === 'custom-pytorch' && firstInput) {
      const value = pythonIdentifier(node.id)
      lines.push(`        ${value} = self.${moduleName(node)}(${firstInput})`)
      values.set(node.id, value)
      continue
    }
    if (node.kind === 'linear' && firstInput) {
      if (node.role === 'query' || node.role === 'key' || node.role === 'value') {
        const short = node.role === 'query' ? 'q' : node.role === 'key' ? 'k' : 'v'
        const heads = node.role === 'query' ? 'self.query_heads' : 'self.key_value_heads'
        lines.push(`        ${short}_projected = self.${moduleName(node)}(${firstInput})`)
        lines.push(`        ${short} = ${short}_projected.view(batch, sequence, ${heads}, self.head_dim).transpose(1, 2)`)
        values.set(node.id, short)
      } else {
        const value = pythonIdentifier(node.label || node.id)
        lines.push(`        ${value} = self.${moduleName(node)}(${firstInput})`)
        values.set(node.id, value)
      }
      continue
    }
    if (node.kind === 'sdpa') {
      const byRole = (role: TensorRole) => {
        const edge = inputEdges.find((candidate) => graph.nodes.find((source) => source.id === candidate.source)?.role === role)
        return edge ? values.get(edge.source) : undefined
      }
      const q = byRole('query') ?? 'q'
      const k = byRole('key') ?? 'k'
      const v = byRole('value') ?? 'v'
      lines.push('        repeats = self.query_heads // self.key_value_heads')
      lines.push(`        expanded_k = ${k}.repeat_interleave(repeats, dim=1)`)
      lines.push(`        expanded_v = ${v}.repeat_interleave(repeats, dim=1)`)
      lines.push('        attended = F.scaled_dot_product_attention(')
      lines.push(`            ${q}, expanded_k, expanded_v,`)
      lines.push('            is_causal=True,')
      lines.push('        )')
      lines.push('        context = attended.transpose(1, 2).reshape(batch, sequence, -1)')
      values.set(node.id, 'context')
    }
  }

  const sinks = graph.nodes.filter((node) => !graph.edges.some((edge) => edge.source === node.id) && values.has(node.id))
  const result = values.get(sinks.at(-1)?.id ?? '') ?? values.get('output') ?? 'context'
  lines.push(`        return ${result}`)
  return lines.join('\n')
}

export function compileToPyTorch(graph: ArchitectureGraph): string {
  if (graph.nodes.length > 0 && graph.nodes.every((node) => node.kind === 'input' || node.kind === 'semantic' || node.kind === 'custom-pytorch')) {
    try {
      return compileRegistryGraph(graph, { disconnectedInputs: 'skip' })
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      return `import torch
import torch.nn as nn


class GeneratedInvalidGraph(nn.Module):
    """The semantic graph cannot be lowered until its typed ports are connected."""

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise RuntimeError(${JSON.stringify(message)})
`
    }
  }
  const { hiddenSize, queryHeads, keyValueHeads, headDim } = graph.config
  const validation = validateGraph(graph)
  if (!validation.valid) {
    const declarations = graph.nodes.flatMap((node) => {
      if (node.kind === 'linear') {
        const name = node.id === 'output' ? 'out_proj' : node.id.replaceAll('-', '_')
        const input = Number(node.attributes?.inFeatures ?? 0)
        const output = Number(node.attributes?.outFeatures ?? 0)
        const bias = node.attributes?.bias === true ? 'True' : 'False'
        return [`        # labo:node=${node.id} kind=linear`, `        self.${name} = nn.Linear(${input}, ${output}, bias=${bias})`]
      }
      if (node.kind === 'custom-pytorch' && node.code) {
        return [`        # labo:node=${node.id} kind=custom-pytorch`, `        self.${node.id.replaceAll('-', '_')} = ${node.code}`]
      }
      return []
    }).join('\n')
    return `import torch
import torch.nn as nn


class GeneratedInvalidGraph(nn.Module):
    """Partial program emitted from the current invalid LABO AI graph."""

    def __init__(self):
        super().__init__()
${declarations || '        pass'}

    def forward(self, *args, **kwargs):
        raise RuntimeError("Graph is invalid: ${validation.errors.join('; ')}")
`
  }
  const linear = (nodeId: string, fallbackInput: number, fallbackOutput: number) => {
    const node = graph.nodes.find((candidate) => candidate.id === nodeId)
    return {
      input: Number(node?.attributes?.inFeatures ?? fallbackInput),
      output: Number(node?.attributes?.outFeatures ?? fallbackOutput),
      bias: node?.attributes?.bias === true ? 'True' : 'False',
    }
  }
  const gqaDeclarations = graph.nodes.flatMap((node) => {
    if (node.kind === 'custom-pytorch' && node.code) {
      return [`        # labo:node=${node.id} kind=custom-pytorch`, `        self.${moduleName(node)} = ${node.code}`]
    }
    if (node.kind !== 'linear') return []
    const fallbackOutput = node.role === 'query'
      ? queryHeads * headDim
      : node.role === 'key' || node.role === 'value'
        ? keyValueHeads * headDim
        : hiddenSize
    const spec = linear(node.id, hiddenSize, fallbackOutput)
    return [`        # labo:node=${node.id} kind=linear`, `        self.${moduleName(node)} = nn.Linear(${spec.input}, ${spec.output}, bias=${spec.bias})`]
  }).join('\n')
  return `import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratedGQA(nn.Module):
    """Generated by LABO AI from ${graph.name}."""

    def __init__(self):
        super().__init__()
        self.query_heads = ${queryHeads}
        self.key_value_heads = ${keyValueHeads}
        self.head_dim = ${headDim}
${gqaDeclarations}

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
${compileGqaForward(graph)}
`
}
