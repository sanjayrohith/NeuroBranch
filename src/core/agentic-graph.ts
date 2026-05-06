import { connectCable, inferredEdgeTargetPort } from './cables'
import { executionLayers } from './execution-plan'
import { findOpenGraphPosition, layoutArchitectureGraph, layoutParallelArchitecture } from './graph-placement'
import { addNode, removeNode, type ArchitectureGraph, type ArchitectureNode, type TensorRole } from './ir'
import type { CustomPyTorchCard } from '../model/custom-card'
import { architectureComponents } from './graph-components'
import { modelAtomRegistry, type ModelAtomDefinition } from './model-atoms'
import { validCustomPyTorchModule } from './pytorch-compiler'
import { customCardInputPorts, customCardOutputPorts, validateCustomCardGraph } from './custom-card-graph'

export interface AgentPortSnapshot {
  id: string
  tensor: TensorRole
  rank?: number
}

export interface AgentNodeSnapshot {
  id: string
  atomId?: string
  label: string
  inputs: AgentPortSnapshot[]
  outputs: AgentPortSnapshot[]
}

export interface AgentConnectionProposal {
  sourceId: string
  sourcePortId: string
  targetId: string
  targetPortId: string
  reason: string
}

export interface AgentBlockProposal {
  atomId: string
  nodeId: string
  reason: string
}

export interface AgentCreatedBlockProposal {
  nodeId: string
  label: string
  pytorchModule: string
  inputRole?: TensorRole
  outputRole?: TensorRole
  reason: string
  cardGraph?: ArchitectureGraph
}

export interface AgentUpdatedBlockProposal {
  nodeId: string
  label: string | null
  settings: Record<string, number | string | boolean> | null
  pytorchModule: string | null
  reason: string
}

export interface AgentReplacedBlockProposal {
  nodeId: string
  atomId: string
  reason: string
}

export type AgentGraphAction =
  | { type: 'layout'; scope: 'all' | 'new'; reason: string }
  | { type: 'run'; mode: 'play' | 'step'; reason: string }
  | { type: 'save-preset'; name: string; reason: string }
  | { type: 'export'; kind: 'svg' | 'python' | 'both'; reason: string }
  | { type: 'run-selection'; mode: 'play' | 'step'; nodeIds: string[]; reason: string }

export interface AgentMissingBlock {
  atomId: string | null
  label: string
  reason: string
}

export interface AgentGraphPlan {
  summary: string
  addedBlocks: AgentBlockProposal[]
  createdBlocks: AgentCreatedBlockProposal[]
  connections: AgentConnectionProposal[]
  updatedBlocks?: AgentUpdatedBlockProposal[]
  replacedBlocks?: AgentReplacedBlockProposal[]
  deletedBlocks?: Array<{ nodeId: string; reason: string }>
  movedBlocks?: Array<{ nodeId: string; x: number; y: number; reason: string }>
  actions?: AgentGraphAction[]
  missingBlocks: AgentMissingBlock[]
  warnings: string[]
  toolTrace?: Array<{ tool: string; status: 'accepted' | 'rejected' | 'read'; summary: string }>
}

export type AgentGraphMode = 'extend' | 'parallel'

export interface RejectedAgentBlock {
  block: AgentBlockProposal | AgentCreatedBlockProposal
  reason: string
}

export interface RejectedAgentConnection {
  connection: AgentConnectionProposal
  reason: string
}

export interface AgentGraphPreview {
  graph: ArchitectureGraph
  acceptedBlocks: AgentBlockProposal[]
  acceptedCreatedBlocks: AgentCreatedBlockProposal[]
  rejectedBlocks: RejectedAgentBlock[]
  accepted: AgentConnectionProposal[]
  rejected: RejectedAgentConnection[]
  acceptedActions: AgentGraphAction[]
  rejectedMutations: Array<{ nodeId?: string; action?: AgentGraphAction; reason: string }>
}

interface AgentVirtualAtomic {
  atomId: string
  label: string
  inputs: AgentPortSnapshot[]
  outputs: AgentPortSnapshot[]
  createNode(nodeId: string, position: { x: number; y: number }): ArchitectureNode
}

const agentVirtualAtomics: Record<string, AgentVirtualAtomic> = {
  'token-ids-input': {
    atomId: 'token-ids-input', label: 'Token IDs input', inputs: [], outputs: [{ id: 'tokenIds', tensor: 'token-ids', rank: 2 }],
    createNode: (id, position) => ({ id, kind: 'input', label: 'Token IDs', role: 'token-ids', position }),
  },
  'hidden-state-input': {
    atomId: 'hidden-state-input', label: 'Hidden State input', inputs: [], outputs: [{ id: 'hidden', tensor: 'hidden', rank: 3 }],
    createNode: (id, position) => ({ id, kind: 'input', label: 'Hidden State', role: 'hidden', position }),
  },
  'image-tensor-input': {
    atomId: 'image-tensor-input', label: 'Image Tensor input', inputs: [], outputs: [{ id: 'image', tensor: 'image', rank: 4 }],
    createNode: (id, position) => ({ id, kind: 'input', label: 'Image Tensor', role: 'image', position }),
  },
  'video-tensor-input': {
    atomId: 'video-tensor-input', label: 'Video Tensor input', inputs: [], outputs: [{ id: 'video', tensor: 'video', rank: 5 }],
    createNode: (id, position) => ({ id, kind: 'input', label: 'Video Tensor', role: 'video', position }),
  },
  'audio-waveform-input': {
    atomId: 'audio-waveform-input', label: 'Audio Waveform input', inputs: [], outputs: [{ id: 'audio', tensor: 'audio', rank: 3 }],
    createNode: (id, position) => ({ id, kind: 'input', label: 'Audio Waveform', role: 'audio', position }),
  },
  'training-labels-input': {
    atomId: 'training-labels-input', label: 'Training Labels input', inputs: [], outputs: [{ id: 'labels', tensor: 'labels', rank: 2 }],
    createNode: (id, position) => ({ id, kind: 'input', label: 'Training Labels', role: 'labels', position }),
  },
}

function uniquePorts(ports: AgentPortSnapshot[]): AgentPortSnapshot[] {
  return [...new Map(ports.map((port) => [`${port.id}:${port.tensor}`, port])).values()]
}

function fallbackOutput(node: ArchitectureNode): AgentPortSnapshot {
  if (node.id.toLowerCase().includes('token')) return { id: 'tokenIds', tensor: 'token-ids' }
  return { id: 'output', tensor: node.role === 'attention' ? 'attention' : node.role }
}

function snapshotNode(graph: ArchitectureGraph, node: ArchitectureNode): AgentNodeSnapshot {
  const definition = node.atomId ? modelAtomRegistry[node.atomId] : undefined
  const edgeInputs = graph.edges
    .filter((edge) => edge.target === node.id)
    .map((edge) => ({ id: edge.targetPort ?? inferredEdgeTargetPort(graph, edge), tensor: inferredEdgeTargetPort(graph, edge) }))
  const edgeOutputs = graph.edges
    .filter((edge) => edge.source === node.id)
    .map((edge) => ({ id: edge.sourcePort ?? fallbackOutput(node).id, tensor: sourceTensor(graph, node, edge.sourcePort) }))

  return {
    id: node.id,
    ...(node.atomId ? { atomId: node.atomId } : {}),
    label: node.label,
    inputs: node.kind === 'input' ? [] : definition
      ? definition.inputs.map(({ id, tensor, rank }) => ({ id, tensor, ...(rank ? { rank } : {}) }))
      : node.kind === 'custom-pytorch' ? node.customCardGraph
        ? customCardInputPorts(node.customCardGraph).map(({ id, tensor, rank }) => ({ id, tensor, ...(rank ? { rank } : {}) }))
        : [{ id: node.attributes?.inputRole === 'hidden' || !node.attributes?.inputRole ? 'hidden' : 'input', tensor: (node.attributes?.inputRole as TensorRole | undefined) ?? 'hidden' }] : uniquePorts(edgeInputs),
    outputs: node.kind === 'input'
      ? [{ id: node.role === 'token-ids' ? 'tokenIds' : node.role === 'labels' ? 'labels' : node.role, tensor: node.role, rank: node.role === 'token-ids' || node.role === 'labels' ? 2 : node.role === 'image' ? 4 : node.role === 'video' ? 5 : 3 }]
      : definition
      ? definition.outputs.map(({ id, tensor, rank }) => ({ id, tensor, ...(rank ? { rank } : {}) }))
      : node.kind === 'custom-pytorch' ? node.customCardGraph
        ? customCardOutputPorts(node.customCardGraph).map(({ id, tensor, rank }) => ({ id, tensor, ...(rank ? { rank } : {}) }))
        : [{ id: 'output', tensor: node.role }] : uniquePorts(edgeOutputs.length > 0 ? edgeOutputs : [fallbackOutput(node)]),
  }
}

function sourceTensor(graph: ArchitectureGraph, node: ArchitectureNode, portId?: string): TensorRole {
  const definition = node.atomId ? modelAtomRegistry[node.atomId] : undefined
  const port = definition?.outputs.find((candidate) => candidate.id === portId)
  if (port) return port.tensor
  if (node.id.toLowerCase().includes('token') || portId === 'tokenIds') return 'token-ids'
  const connected = graph.edges.find((edge) => edge.source === node.id && edge.sourcePort === portId)
  if (connected?.targetPort) {
    const target = graph.nodes.find((candidate) => candidate.id === connected.target)
    const targetPort = target?.atomId ? modelAtomRegistry[target.atomId]?.inputs.find((candidate) => candidate.id === connected.targetPort) : undefined
    if (targetPort) return targetPort.tensor
  }
  return node.role === 'attention' ? 'attention' : node.role
}

export function createAgentGraphContext(graph: ArchitectureGraph, mode: AgentGraphMode = 'extend', customCards: CustomPyTorchCard[] = [], editing?: { active: boolean; nodeIds: string[] }) {
  return {
    operationMode: mode,
    editing: { active: editing?.active === true, nodeIds: editing?.nodeIds.filter((nodeId) => graph.nodes.some((node) => node.id === nodeId)) ?? [] },
    graph: {
      id: graph.id,
      name: graph.name,
      config: graph.config,
      nodes: graph.nodes.map((node) => snapshotNode(graph, node)),
      connections: graph.edges.map((edge) => ({
        sourceId: edge.source,
        sourcePortId: edge.sourcePort ?? 'output',
        targetId: edge.target,
        targetPortId: edge.targetPort ?? inferredEdgeTargetPort(graph, edge),
      })),
    },
    availableAtomics: [
      ...Object.values(agentVirtualAtomics).map(({ atomId, label, inputs, outputs }) => ({ atomId, label, inputs, outputs, settings: [] })),
      ...Object.values(modelAtomRegistry).filter((definition) => !definition.composite).map((definition) => ({
        atomId: definition.id,
        label: definition.label,
        inputs: definition.inputs.map(({ id, tensor, rank }) => ({ id, tensor, ...(rank ? { rank } : {}) })),
        outputs: definition.outputs.map(({ id, tensor, rank }) => ({ id, tensor, ...(rank ? { rank } : {}) })),
        settings: definition.settings.map(({ id, type, default: defaultValue, options }) => ({
          id,
          type,
          default: id in graph.config ? graph.config[id as keyof ArchitectureGraph['config']] : defaultValue,
          ...(options ? { options } : {}),
        })),
      })),
    ],
    availableCustomCards: customCards.map((card) => ({
      id: card.id,
      label: card.label,
      code: card.code,
      inputRole: card.inputRole ?? 'hidden',
      outputRole: card.outputRole ?? 'hidden',
      ...(card.graph ? {
        graph: card.graph,
        inputs: customCardInputPorts(card.graph).map(({ id, tensor, rank }) => ({ id, tensor, ...(rank ? { rank } : {}) })),
        outputs: customCardOutputPorts(card.graph).map(({ id, tensor, rank }) => ({ id, tensor, ...(rank ? { rank } : {}) })),
      } : {}),
    })),
    architectures: architectureComponents(graph).map((architecture) => ({ id: architecture.id, label: architecture.label, nodeIds: architecture.nodeIds })),
  }
}

function roleForDefinition(definition: ModelAtomDefinition): TensorRole {
  const output = definition.outputs[0]?.tensor
  if (output === 'query' || output === 'key' || output === 'value' || output === 'token-ids' || output === 'image' || output === 'video' || output === 'audio') return output
  if (output === 'logits' || output === 'scalar') return 'output'
  return 'hidden'
}

function uniqueAgentNodeId(graph: ArchitectureGraph, plan: AgentGraphPlan, base: string): string {
  const used = new Set([...graph.nodes.map((node) => node.id), ...plan.addedBlocks.map((block) => block.nodeId), ...plan.createdBlocks.map((block) => block.nodeId)])
  if (!used.has(base)) return base
  let sequence = 2
  while (used.has(`${base}-${sequence}`)) sequence += 1
  return `${base}-${sequence}`
}

function normalizeAgentPlanNodeIds(graph: ArchitectureGraph, sourcePlan: AgentGraphPlan): AgentGraphPlan {
  const used = new Set(graph.nodes.map((node) => node.id))
  const replacements = new Map<string, string>()
  const normalize = (sourceId: string) => {
    const existing = replacements.get(sourceId)
    if (existing) return existing
    let base = sourceId
      .normalize('NFKD')
      .replace(/[\u0300-\u036f]/g, '')
      .replace(/[^A-Za-z0-9-]+/g, '-')
      .replace(/-+/g, '-')
      .replace(/^-|-$/g, '')
    if (!/^[A-Za-z]/.test(base)) base = `agent-${base || 'card'}`
    base = base.slice(0, 64).replace(/-$/g, '') || 'agent-card'
    let candidate = base
    let sequence = 2
    while (used.has(candidate)) {
      const suffix = `-${sequence++}`
      candidate = `${base.slice(0, 64 - suffix.length).replace(/-$/g, '')}${suffix}`
    }
    used.add(candidate)
    replacements.set(sourceId, candidate)
    return candidate
  }

  for (const block of [...sourcePlan.addedBlocks, ...(sourcePlan.createdBlocks ?? [])]) normalize(block.nodeId)
  const reference = (nodeId: string) => replacements.get(nodeId) ?? nodeId
  return {
    ...sourcePlan,
    addedBlocks: sourcePlan.addedBlocks.map((block) => ({ ...block, nodeId: reference(block.nodeId) })),
    createdBlocks: (sourcePlan.createdBlocks ?? []).map((block) => ({ ...block, nodeId: reference(block.nodeId) })),
    connections: sourcePlan.connections.map((connection) => ({ ...connection, sourceId: reference(connection.sourceId), targetId: reference(connection.targetId) })),
    updatedBlocks: (sourcePlan.updatedBlocks ?? []).map((block) => ({ ...block, nodeId: reference(block.nodeId) })),
    replacedBlocks: (sourcePlan.replacedBlocks ?? []).map((block) => ({ ...block, nodeId: reference(block.nodeId) })),
    deletedBlocks: (sourcePlan.deletedBlocks ?? []).map((block) => ({ ...block, nodeId: reference(block.nodeId) })),
    movedBlocks: (sourcePlan.movedBlocks ?? []).map((block) => ({ ...block, nodeId: reference(block.nodeId) })),
  }
}

/** Deterministically fixes source/sampler omissions that the model incorrectly reports as missing. */
export function repairAgentGraphPlan(graph: ArchitectureGraph, sourcePlan: AgentGraphPlan): AgentGraphPlan {
  const normalizedPlan = normalizeAgentPlanNodeIds(graph, sourcePlan)
  const plan: AgentGraphPlan = {
    ...normalizedPlan,
    addedBlocks: [...normalizedPlan.addedBlocks],
    createdBlocks: [...normalizedPlan.createdBlocks],
    connections: [...normalizedPlan.connections],
    missingBlocks: [...normalizedPlan.missingBlocks],
    warnings: [...normalizedPlan.warnings],
    updatedBlocks: [...(normalizedPlan.updatedBlocks ?? [])],
    replacedBlocks: [...(normalizedPlan.replacedBlocks ?? [])],
    deletedBlocks: [...(normalizedPlan.deletedBlocks ?? [])],
    movedBlocks: [...(normalizedPlan.movedBlocks ?? [])],
    actions: [...(normalizedPlan.actions ?? [])],
    toolTrace: [...(normalizedPlan.toolTrace ?? [])],
  }
  const normalizeCapabilityText = (value: string) => value.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '')
  const samplerPattern = /sampl|echantillonn|decod|token gener|generated token|autoregress|logits[^.]{0,80}token/
  const missingText = () => normalizeCapabilityText(plan.missingBlocks.map((block) => `${block.atomId ?? ''} ${block.label} ${block.reason}`).join(' '))
  const hasIncoming = (nodeId: string, portId: string) => plan.connections.some((connection) => connection.targetId === nodeId && connection.targetPortId === portId)
    || graph.edges.some((edge) => edge.target === nodeId && edge.targetPort === portId)
  const capacity = () => plan.addedBlocks.length + plan.createdBlocks.length < 24

  for (const block of [...plan.addedBlocks]) {
    const definition = modelAtomRegistry[block.atomId]
    for (const input of definition?.inputs ?? []) {
      if (!['token-ids', 'labels', 'image', 'video', 'audio'].includes(input.tensor) || hasIncoming(block.nodeId, input.id)) continue
      const virtualAtomId = input.tensor === 'token-ids' ? 'token-ids-input' : input.tensor === 'labels' ? 'training-labels-input' : input.tensor === 'image' ? 'image-tensor-input' : input.tensor === 'video' ? 'video-tensor-input' : 'audio-waveform-input'
      let source = plan.addedBlocks.find((candidate) => candidate.atomId === virtualAtomId)
      if (!source && capacity()) {
        source = {
          atomId: virtualAtomId,
          nodeId: uniqueAgentNodeId(graph, plan, input.tensor === 'token-ids' ? 'agent-token-ids' : input.tensor === 'labels' ? 'agent-training-labels' : input.tensor === 'image' ? 'agent-image' : input.tensor === 'video' ? 'agent-video' : 'agent-audio'),
          reason: `LABO repaired the missing ${input.tensor} graph source locally.`,
        }
        plan.addedBlocks.push(source)
      }
      if (source) plan.connections.push({
        sourceId: source.nodeId,
        sourcePortId: input.tensor === 'token-ids' ? 'tokenIds' : input.tensor,
        targetId: block.nodeId,
        targetPortId: input.id,
        reason: `LABO connected the available ${input.tensor} source locally.`,
      })
    }
  }

  const samplerClaimedMissing = samplerPattern.test(missingText())

  for (const attention of plan.addedBlocks.filter((block) => ['causal-sdpa', 'eager-causal-attention', 'noncausal-sdpa'].includes(block.atomId))) {
    const direct = plan.connections.filter((connection) => connection.targetId === attention.nodeId && ['q', 'k', 'v'].includes(connection.targetPortId))
    const qkvSource = direct.find((connection) => plan.addedBlocks.some((block) => block.nodeId === connection.sourceId && block.atomId === 'qkv-projection'))?.sourceId
      ?? plan.addedBlocks.findLast((block) => block.atomId === 'qkv-projection')?.nodeId
    if (!qkvSource || !capacity()) continue
    let layout = plan.addedBlocks.find((block) => block.atomId === 'attention-head-layout' && plan.connections.some((connection) => connection.targetId === block.nodeId && connection.sourceId === qkvSource))
    if (!layout) {
      layout = { atomId: 'attention-head-layout', nodeId: uniqueAgentNodeId(graph, plan, 'agent-head-layout'), reason: 'LABO inserted the required rank-3 to rank-4 attention head layout.' }
      plan.addedBlocks.push(layout)
    }
    plan.connections = plan.connections.filter((connection) => !(connection.targetId === attention.nodeId && connection.sourceId === qkvSource && ['q', 'k', 'v'].includes(connection.targetPortId)))
    const portMap = { q: 'qHeads', k: 'kHeads', v: 'vHeads' } as const
    for (const port of ['q', 'k', 'v'] as const) {
      if (!plan.connections.some((connection) => connection.sourceId === qkvSource && connection.sourcePortId === port && connection.targetId === layout!.nodeId && connection.targetPortId === port)) {
        plan.connections.push({ sourceId: qkvSource, sourcePortId: port, targetId: layout.nodeId, targetPortId: port, reason: `Reshape ${port.toUpperCase()} into attention heads.` })
      }
      if (!plan.connections.some((connection) => connection.sourceId === layout!.nodeId && connection.sourcePortId === portMap[port] && connection.targetId === attention.nodeId && connection.targetPortId === port)) {
        plan.connections.push({ sourceId: layout.nodeId, sourcePortId: portMap[port], targetId: attention.nodeId, targetPortId: port, reason: `Feed rank-4 ${port.toUpperCase()} heads into attention.` })
      }
    }
    plan.warnings = plan.warnings.filter((warning) => !/head layout|rank.?3|rank.?4/i.test(warning))
  }

  if (samplerClaimedMissing) {
    for (const head of plan.addedBlocks.filter((block) => block.atomId === 'lm-head')) {
      const alreadyDecoded = plan.connections.some((connection) => connection.sourceId === head.nodeId && connection.sourcePortId === 'logits')
      if (alreadyDecoded || !capacity()) continue
      const nodeId = uniqueAgentNodeId(graph, plan, 'agent-greedy-decoder')
      plan.addedBlocks.push({ atomId: 'greedy-token-decoder', nodeId, reason: 'LABO resolved the requested logits-to-token capability with the native greedy decoder.' })
      plan.connections.push({ sourceId: head.nodeId, sourcePortId: 'logits', targetId: nodeId, targetPortId: 'logits', reason: 'Decode language-model logits into generated Token IDs.' })
    }
  }

  plan.missingBlocks = plan.missingBlocks.filter((block) => {
    const text = normalizeCapabilityText(`${block.atomId ?? ''} ${block.label} ${block.reason}`)
    if (/token[- ]?ids|token ids|tokenizer/.test(text) && plan.addedBlocks.some((candidate) => candidate.atomId === 'token-ids-input')) return false
    if (samplerPattern.test(text) && plan.addedBlocks.some((candidate) => ['greedy-token-decoder', 'top-k-token-sampler', 'multinomial-token-sampler'].includes(candidate.atomId))) return false
    return true
  })
  return plan
}

function agentNode(definition: ModelAtomDefinition, nodeId: string, position: { x: number; y: number }, config: ArchitectureGraph['config']): ArchitectureNode {
  return {
    id: nodeId,
    kind: 'semantic',
    atomId: definition.id,
    label: definition.label,
    role: roleForDefinition(definition),
    position,
    attributes: Object.fromEntries(definition.settings.map((setting) => [
      setting.id,
      setting.id in config ? config[setting.id as keyof ArchitectureGraph['config']] : setting.default,
    ])),
  }
}

function targetHasConnection(graph: ArchitectureGraph, targetId: string, targetPortId: string): boolean {
  return graph.edges.some((edge) => edge.target === targetId && (edge.targetPort ?? inferredEdgeTargetPort(graph, edge)) === targetPortId)
}

export function previewAgentGraphPlan(graph: ArchitectureGraph, plan: AgentGraphPlan, mode: AgentGraphMode = 'extend', options?: { editingNodeIds?: string[] }): AgentGraphPreview {
  let nextGraph = graph
  const existingNodeIds = new Set(graph.nodes.map((node) => node.id))
  const editSelection = options?.editingNodeIds ? new Set(options.editingNodeIds) : undefined
  const acceptedBlocks: AgentBlockProposal[] = []
  const acceptedCreatedBlocks: AgentCreatedBlockProposal[] = []
  const rejectedBlocks: RejectedAgentBlock[] = []
  const accepted: AgentConnectionProposal[] = []
  const rejected: RejectedAgentConnection[] = []
  const acceptedActions: AgentGraphAction[] = []
  const rejectedMutations: AgentGraphPreview['rejectedMutations'] = []

  for (const replacement of plan.replacedBlocks ?? []) {
    const current = nextGraph.nodes.find((node) => node.id === replacement.nodeId)
    const definition = modelAtomRegistry[replacement.atomId]
    if (!current || !definition || definition.composite || (mode === 'parallel' && existingNodeIds.has(replacement.nodeId)) || (editSelection && !editSelection.has(replacement.nodeId))) {
      rejectedMutations.push({ nodeId: replacement.nodeId, reason: 'Replacement card is unknown, composite, or read-only in parallel mode' })
      continue
    }
    const candidate = agentNode(definition, current.id, current.position, nextGraph.config)
    const inputIds = new Set(definition.inputs.map((port) => port.id))
    const outputIds = new Set(definition.outputs.map((port) => port.id))
    const incompatible = nextGraph.edges.some((edge) => edge.target === current.id && !inputIds.has(edge.targetPort ?? inferredEdgeTargetPort(nextGraph, edge)))
      || nextGraph.edges.some((edge) => edge.source === current.id && !outputIds.has(edge.sourcePort ?? 'output'))
    if (incompatible) {
      rejectedMutations.push({ nodeId: replacement.nodeId, reason: 'Replacement does not preserve the connected port contract' })
      continue
    }
    nextGraph = { ...nextGraph, nodes: nextGraph.nodes.map((node) => node.id === current.id ? candidate : node) }
  }

  for (const deletion of plan.deletedBlocks ?? []) {
    if (!nextGraph.nodes.some((node) => node.id === deletion.nodeId)) {
      rejectedMutations.push({ nodeId: deletion.nodeId, reason: 'Card does not exist' })
      continue
    }
    if (mode === 'parallel' && existingNodeIds.has(deletion.nodeId)) {
      rejectedMutations.push({ nodeId: deletion.nodeId, reason: 'Parallel architecture mode cannot delete existing cards' })
      continue
    }
    if (editSelection && !editSelection.has(deletion.nodeId)) {
      rejectedMutations.push({ nodeId: deletion.nodeId, reason: 'Edit Cards can delete only cards in the active selection' })
      continue
    }
    nextGraph = removeNode(nextGraph, deletion.nodeId)
  }

  for (const block of plan.addedBlocks.slice(0, 24)) {
    if (editSelection) {
      rejectedBlocks.push({ block, reason: 'Edit Cards cannot add blocks; switch to Add Blocks' })
      continue
    }
    const virtual = agentVirtualAtomics[block.atomId]
    const definition = modelAtomRegistry[block.atomId]
    if (!definition && !virtual) {
      rejectedBlocks.push({ block, reason: 'Atomic block is not available in the LABO library' })
      continue
    }
    if (definition?.composite) {
      rejectedBlocks.push({ block, reason: 'Composite recipes must be expanded into their atomic blocks' })
      continue
    }
    if (!/^[A-Za-z][A-Za-z0-9-]{0,63}$/.test(block.nodeId)) {
      rejectedBlocks.push({ block, reason: 'Block id must start with a letter and contain only letters, numbers, or hyphens' })
      continue
    }
    if (nextGraph.nodes.some((node) => node.id === block.nodeId)) {
      rejectedBlocks.push({ block, reason: `Block id ${block.nodeId} already exists` })
      continue
    }
    const position = findOpenGraphPosition(nextGraph)
    nextGraph = addNode(nextGraph, virtual ? virtual.createNode(block.nodeId, position) : agentNode(definition!, block.nodeId, position, graph.config))
    acceptedBlocks.push(block)
  }
  if (plan.addedBlocks.length > 24) {
    for (const block of plan.addedBlocks.slice(24)) rejectedBlocks.push({ block, reason: 'A single agent plan can add at most 24 blocks' })
  }

  const remainingBlockCapacity = Math.max(0, 24 - acceptedBlocks.length)
  for (const block of (plan.createdBlocks ?? []).slice(0, Math.min(12, remainingBlockCapacity))) {
    if (editSelection) {
      rejectedBlocks.push({ block, reason: 'Edit Cards cannot create reusable cards; use Reusable Card' })
      continue
    }
    if (!/^[A-Za-z][A-Za-z0-9-]{0,63}$/.test(block.nodeId)) {
      rejectedBlocks.push({ block, reason: 'Generated card id must start with a letter and contain only letters, numbers, or hyphens' })
      continue
    }
    if (nextGraph.nodes.some((node) => node.id === block.nodeId)) {
      rejectedBlocks.push({ block, reason: `Block id ${block.nodeId} already exists` })
      continue
    }
    if (!block.label.trim() || block.label.length > 80) {
      rejectedBlocks.push({ block, reason: 'Generated card must have a short label' })
      continue
    }
    if (block.cardGraph && validateCustomCardGraph(block.cardGraph).length > 0) {
      rejectedBlocks.push({ block, reason: 'Generated composite card graph is invalid' })
      continue
    }
    if (!block.cardGraph && !validCustomPyTorchModule(block.pytorchModule)) {
      rejectedBlocks.push({ block, reason: 'Generated PyTorch card is outside the safe nn.Module subset' })
      continue
    }
    nextGraph = addNode(nextGraph, {
      id: block.nodeId,
      kind: 'custom-pytorch',
      label: block.label.trim(),
      role: block.outputRole ?? 'hidden',
      position: findOpenGraphPosition(nextGraph),
      code: block.pytorchModule.trim(),
      attributes: { inputRole: block.inputRole ?? 'hidden' },
      ...(block.cardGraph ? { customCardGraph: structuredClone(block.cardGraph) } : {}),
    })
    acceptedCreatedBlocks.push(block)
  }
  for (const block of (plan.createdBlocks ?? []).slice(Math.min(12, remainingBlockCapacity))) {
    rejectedBlocks.push({ block, reason: 'A single agent plan can add at most 24 blocks, including 12 generated cards' })
  }

  for (const connection of plan.connections) {
    const source = nextGraph.nodes.find((node) => node.id === connection.sourceId)
    const target = nextGraph.nodes.find((node) => node.id === connection.targetId)
    if (!source || !target) {
      rejected.push({ connection, reason: 'Unknown source or target block' })
      continue
    }
    if (editSelection && (!editSelection.has(source.id) || !editSelection.has(target.id))) {
      rejected.push({ connection, reason: 'Edit Cards can rewire only inside the active selection' })
      continue
    }
    if (mode === 'parallel' && (existingNodeIds.has(source.id) || existingNodeIds.has(target.id))) {
      rejected.push({ connection, reason: 'Parallel architecture mode cannot connect to or modify the existing graph' })
      continue
    }
    if (source.id === target.id) {
      rejected.push({ connection, reason: 'A block cannot connect to itself' })
      continue
    }

    const sourcePort = snapshotNode(nextGraph, source).outputs.find((port) => port.id === connection.sourcePortId)
    const targetPort = snapshotNode(nextGraph, target).inputs.find((port) => port.id === connection.targetPortId)
    if (!sourcePort || !targetPort) {
      rejected.push({ connection, reason: 'Unknown source or target port' })
      continue
    }
    if (sourcePort.tensor !== targetPort.tensor) {
      rejected.push({ connection, reason: `${sourcePort.tensor} cannot plug into ${targetPort.tensor}` })
      continue
    }
    if (sourcePort.rank && targetPort.rank && sourcePort.rank !== targetPort.rank) {
      rejected.push({ connection, reason: `Rank-${sourcePort.rank} ${sourcePort.tensor} cannot plug into rank-${targetPort.rank} ${targetPort.tensor}` })
      continue
    }
    if (targetHasConnection(nextGraph, target.id, targetPort.id)) {
      rejected.push({ connection, reason: `${target.id}.${targetPort.id} is already connected` })
      continue
    }

    const outcome = connectCable(nextGraph, {
      sourceId: source.id,
      sourcePort: sourcePort.tensor,
      sourcePortId: sourcePort.id,
      targetId: target.id,
      targetPort: targetPort.tensor,
      targetPortId: targetPort.id,
    })
    if (!outcome.ok) {
      rejected.push({ connection, reason: outcome.message })
      continue
    }
    try {
      executionLayers(outcome.graph)
    } catch {
      rejected.push({ connection, reason: 'Connection would create a cycle' })
      continue
    }
    nextGraph = outcome.graph
    accepted.push(connection)
  }

  for (const update of plan.updatedBlocks ?? []) {
    const node = nextGraph.nodes.find((candidate) => candidate.id === update.nodeId)
    if (!node || (mode === 'parallel' && existingNodeIds.has(update.nodeId))) {
      rejectedMutations.push({ nodeId: update.nodeId, reason: 'Card is unknown or read-only in parallel mode' })
      continue
    }
    if (editSelection && !editSelection.has(update.nodeId)) {
      rejectedMutations.push({ nodeId: update.nodeId, reason: 'Edit Cards can update only cards in the active selection' })
      continue
    }
    if (update.pytorchModule && (node.kind !== 'custom-pytorch' || !validCustomPyTorchModule(update.pytorchModule))) {
      rejectedMutations.push({ nodeId: update.nodeId, reason: 'PyTorch edit is invalid for this card' })
      continue
    }
    const graphWideSettings = ['hiddenSize', 'queryHeads', 'keyValueHeads', 'headDim']
    const attemptedGraphWideSettings = graphWideSettings.filter((setting) => setting in (update.settings ?? {}))
    if (attemptedGraphWideSettings.length > 0) {
      rejectedMutations.push({ nodeId: update.nodeId, reason: `${attemptedGraphWideSettings.join(', ')} are graph-wide dimensions and cannot be edited on one card` })
      continue
    }
    nextGraph = {
      ...nextGraph,
      nodes: nextGraph.nodes.map((candidate) => candidate.id === update.nodeId ? {
        ...candidate,
        ...(update.label?.trim() ? { label: update.label.trim() } : {}),
        ...(update.settings ? { attributes: { ...candidate.attributes, ...update.settings } } : {}),
        ...(update.pytorchModule ? { code: update.pytorchModule.trim() } : {}),
      } : candidate),
    }
  }

  const addedNodeIds = [...acceptedBlocks, ...acceptedCreatedBlocks].map((block) => block.nodeId)
  const layoutAction = editSelection ? undefined : (plan.actions ?? []).find((action): action is Extract<AgentGraphAction, { type: 'layout' }> => action.type === 'layout')
  if (!editSelection) {
    nextGraph = mode === 'parallel'
      ? layoutParallelArchitecture(nextGraph, addedNodeIds)
      : layoutAction?.scope === 'all' ? layoutArchitectureGraph(nextGraph) : layoutArchitectureGraph(nextGraph, addedNodeIds)
  }

  for (const movement of plan.movedBlocks ?? []) {
    if (!Number.isFinite(movement.x) || !Number.isFinite(movement.y) || (mode === 'parallel' && existingNodeIds.has(movement.nodeId))) {
      rejectedMutations.push({ nodeId: movement.nodeId, reason: 'Card position is invalid or read-only in parallel mode' })
      continue
    }
    if (!nextGraph.nodes.some((node) => node.id === movement.nodeId)) {
      rejectedMutations.push({ nodeId: movement.nodeId, reason: 'Card does not exist' })
      continue
    }
    if (editSelection && !editSelection.has(movement.nodeId)) {
      rejectedMutations.push({ nodeId: movement.nodeId, reason: 'Edit Cards can move only cards in the active selection' })
      continue
    }
    nextGraph = { ...nextGraph, nodes: nextGraph.nodes.map((node) => node.id === movement.nodeId ? { ...node, position: { x: movement.x, y: movement.y } } : node) }
  }
  for (const action of plan.actions ?? []) {
    if (editSelection && action.type === 'layout') rejectedMutations.push({ action, reason: 'Edit Cards cannot relayout the whole graph' })
    else if (editSelection && action.type === 'run') rejectedMutations.push({ action, reason: 'Edit Cards must run the selected subgraph, not the whole graph' })
    else if (action.type === 'run-selection' && editSelection && action.nodeIds.some((nodeId) => !editSelection.has(nodeId))) rejectedMutations.push({ action, reason: 'Edit Cards can run only cards in the active selection' })
    else if (action.type === 'layout' && mode === 'parallel' && action.scope === 'all') rejectedMutations.push({ action, reason: 'Parallel mode cannot lay out existing work' })
    else acceptedActions.push(action)
  }
  return { graph: nextGraph, acceptedBlocks, acceptedCreatedBlocks, rejectedBlocks, accepted, rejected, acceptedActions, rejectedMutations }
}
