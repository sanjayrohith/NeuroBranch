import type { ArchitectureGraph, ArchitectureNode, TensorRole } from './ir'
import { modelAtomRegistry } from './model-atoms'

export interface CustomCardPort {
  id: string
  label: string
  tensor: TensorRole
  rank?: number
  nodeId: string
  nodePortId: string
}

export function tensorRank(role: TensorRole): number | undefined {
  if (role === 'token-ids' || role === 'labels') return 2
  if (role === 'image') return 4
  if (role === 'video') return 5
  if (role === 'audio' || role === 'hidden' || role === 'query' || role === 'key' || role === 'value' || role === 'attention' || role === 'logits') return 3
  return undefined
}

export function customCardInputPorts(graph: ArchitectureGraph): CustomCardPort[] {
  return graph.nodes.filter((node) => node.kind === 'input').map((node) => ({
    id: node.id,
    label: node.label,
    tensor: node.role,
    rank: tensorRank(node.role),
    nodeId: node.id,
    nodePortId: node.role === 'token-ids' ? 'tokenIds' : node.role,
  }))
}

function nodeOutputPorts(node: ArchitectureNode): Array<{ id: string; tensor: TensorRole; rank?: number }> {
  if (node.kind === 'semantic' && node.atomId) return modelAtomRegistry[node.atomId]?.outputs ?? []
  if (node.kind === 'custom-pytorch' && node.customCardGraph) {
    return customCardOutputPorts(node.customCardGraph).map((port) => ({ id: port.id, tensor: port.tensor, rank: port.rank }))
  }
  if (node.kind === 'custom-pytorch') return [{ id: 'output', tensor: node.role, rank: tensorRank(node.role) }]
  return []
}

export function customCardOutputPorts(graph: ArchitectureGraph): CustomCardPort[] {
  const executable = graph.nodes.filter((node) => node.kind === 'semantic' || node.kind === 'custom-pytorch')
  const executableIds = new Set(executable.map((node) => node.id))
  const sinks = executable.filter((node) => !graph.edges.some((edge) => edge.source === node.id && executableIds.has(edge.target)))
  return sinks.flatMap((node) => nodeOutputPorts(node).map((port) => ({
    id: `${node.id}--${port.id}`,
    label: `${node.label} · ${port.id}`,
    tensor: port.tensor,
    rank: port.rank,
    nodeId: node.id,
    nodePortId: port.id,
  })))
}

export function validateCustomCardGraph(graph: ArchitectureGraph): string[] {
  const errors: string[] = []
  const inputs = customCardInputPorts(graph)
  const outputs = customCardOutputPorts(graph)
  if (inputs.length === 0) errors.push('Add at least one graph input.')
  if (!graph.nodes.some((node) => node.kind === 'semantic' || node.kind === 'custom-pytorch')) errors.push('Add at least one executable atom.')
  if (outputs.length === 0) errors.push('The card needs at least one executable output.')
  if (graph.nodes.length > 32) errors.push('Reusable cards are limited to 32 internal atoms.')
  if (graph.nodes.some((node) => node.customCardGraph)) errors.push('Nested reusable cards are not supported yet.')
  return errors
}
