import type { ArchitectureGraph, ArchitectureNode, TensorRole } from './ir'
import { modelAtomRegistry, type AtomPort } from './model-atoms'
import { customCardInputPorts, customCardOutputPorts } from './custom-card-graph'

const inputRanks: Partial<Record<TensorRole, number>> = {
  'token-ids': 2,
  labels: 2,
  image: 4,
  video: 5,
  audio: 3,
  hidden: 3,
}

function portTensor(role: TensorRole): AtomPort['tensor'] {
  return role === 'output' ? 'hidden' : role
}

function outputPorts(node: ArchitectureNode): AtomPort[] {
  if (node.kind === 'input') {
    return [{
      id: node.role === 'token-ids' ? 'tokenIds' : node.role,
      tensor: portTensor(node.role),
      rank: inputRanks[node.role],
    }]
  }
  if (node.atomId && modelAtomRegistry[node.atomId]) return modelAtomRegistry[node.atomId].outputs
  if (node.kind === 'custom-pytorch' && node.customCardGraph) return customCardOutputPorts(node.customCardGraph).map(({ id, tensor, rank }) => ({ id, tensor: portTensor(tensor), rank }))
  return [{ id: 'output', tensor: portTensor(node.role), rank: inputRanks[node.role] }]
}

function inputPorts(node: ArchitectureNode): AtomPort[] {
  if (node.kind === 'input') return []
  if (node.atomId && modelAtomRegistry[node.atomId]) return modelAtomRegistry[node.atomId].inputs
  if (node.kind === 'custom-pytorch' && node.customCardGraph) {
    return customCardInputPorts(node.customCardGraph).map(({ id, tensor, rank }) => ({ id, tensor: portTensor(tensor), rank }))
  }
  if (node.kind === 'custom-pytorch') {
    const tensor = portTensor((node.attributes?.inputRole as TensorRole | undefined) ?? 'hidden')
    return [{ id: node.attributes?.inputRole === 'hidden' || !node.attributes?.inputRole ? 'hidden' : 'input', tensor, rank: inputRanks[tensor] }]
  }
  return []
}

function connectedPortError(graph: ArchitectureGraph, node: ArchitectureNode): string | undefined {
  const expectedInputs = inputPorts(node)
  for (const edge of graph.edges.filter((candidate) => candidate.target === node.id)) {
    const source = graph.nodes.find((candidate) => candidate.id === edge.source)
    if (!source) return `Unknown source ${edge.source} connected to ${node.label}`
    const sourcePort = outputPorts(source).find((port) => port.id === edge.sourcePort)
      ?? (!edge.sourcePort || edge.sourcePort === source.role ? outputPorts(source)[0] : undefined)
    const targetPort = expectedInputs.find((port) => port.id === edge.targetPort)
      ?? (!edge.targetPort && expectedInputs.length === 1 ? expectedInputs[0] : undefined)
    if (!sourcePort) return `Unknown output port ${edge.sourcePort ?? 'output'} on ${source.label}`
    if (!targetPort) return `Unknown input port ${edge.targetPort ?? 'input'} on ${node.label}`
    if (sourcePort.tensor !== targetPort.tensor) {
      return `${sourcePort.tensor} ${source.label}.${sourcePort.id} cannot plug into ${targetPort.tensor} ${node.label}.${targetPort.id}`
    }
    if (sourcePort.rank !== undefined && targetPort.rank !== undefined && sourcePort.rank !== targetPort.rank) {
      return `Rank-${sourcePort.rank} ${source.label}.${sourcePort.id} cannot plug into rank-${targetPort.rank} ${node.label}.${targetPort.id}`
    }
  }
  return undefined
}

function missingInputs(graph: ArchitectureGraph, node: ArchitectureNode): AtomPort[] {
  const definition = node.atomId ? modelAtomRegistry[node.atomId] : undefined
  const inputs: AtomPort[] = definition?.inputs ?? (node.kind === 'custom-pytorch'
    ? node.customCardGraph ? customCardInputPorts(node.customCardGraph).map(({ id, tensor, rank }) => ({ id, tensor: portTensor(tensor), rank })) : [{ id: node.attributes?.inputRole === 'hidden' || !node.attributes?.inputRole ? 'hidden' : 'input', tensor: portTensor((node.attributes?.inputRole as TensorRole | undefined) ?? 'hidden') }]
    : [])
  return inputs.filter((port) => !graph.edges.some((edge) => edge.target === node.id
    && (edge.targetPort === port.id || (!edge.targetPort && inputs.length === 1))))
}

/**
 * Runs typed graph contracts in browsers, which do not ship a local Python
 * process. Electron continues to execute the exact generated PyTorch program.
 */
export async function previewModelAtom(graph: ArchitectureGraph, atomId: string): Promise<{ summary: string }> {
  const node = graph.nodes.find((candidate) => candidate.id === atomId)
  if (!node) throw new Error(`Unknown graph card: ${atomId}`)

  const connectionError = connectedPortError(graph, node)
  if (connectionError) throw new Error(connectionError)

  const missing = missingInputs(graph, node)
  if (missing.length > 0) throw new Error(`Missing ${missing.map((port) => port.id).join(' + ')} input on ${node.label}`)

  const outputs = outputPorts(node).map((port) => `${port.tensor}${port.rank === undefined ? '' : ` · rank ${port.rank}`}`)
  return { summary: `Graph preview · ${node.label} → ${outputs.join(' + ') || 'complete'}` }
}
