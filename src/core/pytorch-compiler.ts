import type { ArchitectureEdge, ArchitectureGraph, ArchitectureNode } from './ir'
import { customCardInputPorts, customCardOutputPorts } from './custom-card-graph'
import { modelAtomRegistry, type ModelAtomDefinition } from './model-atoms'

function identifier(value: string): string {
  const normalized = value.replaceAll('-', '_').replace(/[^A-Za-z0-9_]/g, '_')
  return /^\d/.test(normalized) ? `node_${normalized}` : normalized
}

function pythonValue(value: number | string | boolean): string {
  if (typeof value === 'boolean') return value ? 'True' : 'False'
  if (typeof value === 'string') return `'${value.replaceAll('\\', '\\\\').replaceAll("'", "\\'")}'`
  if (value !== 0 && Math.abs(value) < 1e-4) {
    return value.toExponential().replace(/e([+-])(\d)$/, 'e$10$2')
  }
  return String(value)
}

function orderedNodes(graph: ArchitectureGraph): ArchitectureNode[] {
  const index = new Map(graph.nodes.map((node, position) => [node.id, position]))
  const indegree = new Map(graph.nodes.map((node) => [node.id, 0]))
  const outgoing = new Map(graph.nodes.map((node) => [node.id, [] as string[]]))
  for (const edge of graph.edges) {
    if (!indegree.has(edge.source) || !indegree.has(edge.target)) throw new Error(`Unknown edge endpoint: ${edge.id}`)
    indegree.set(edge.target, (indegree.get(edge.target) ?? 0) + 1)
    outgoing.get(edge.source)?.push(edge.target)
  }
  for (const targets of outgoing.values()) targets.sort((left, right) => (index.get(left) ?? 0) - (index.get(right) ?? 0))
  const queue = graph.nodes.filter((node) => indegree.get(node.id) === 0)
  const result: ArchitectureNode[] = []
  while (queue.length > 0) {
    const node = queue.shift()!
    result.push(node)
    for (const target of outgoing.get(node.id) ?? []) {
      const next = (indegree.get(target) ?? 0) - 1
      indegree.set(target, next)
      if (next === 0) {
        queue.push(graph.nodes.find((candidate) => candidate.id === target)!)
        queue.sort((left, right) => (index.get(left.id) ?? 0) - (index.get(right.id) ?? 0))
      }
    }
  }
  if (result.length !== graph.nodes.length) throw new Error('Cannot compile a cyclic graph')
  return result
}

function inputArgument(node: ArchitectureNode): string {
  if (node.id === 'hidden' || node.id === 'hidden-states' || node.id === 'hidden_states') return 'hidden_states'
  if (['token', 'tokens', 'token-ids', 'token_ids'].includes(node.id.toLowerCase())) return 'token_ids'
  if (['label', 'labels', 'training-labels'].includes(node.id.toLowerCase())) return 'labels'
  if (node.role === 'image') return identifier(node.id).includes('image') ? identifier(node.id) : `${identifier(node.id)}_image`
  if (node.role === 'video') return identifier(node.id).includes('video') ? identifier(node.id) : `${identifier(node.id)}_video`
  if (node.role === 'audio') return identifier(node.id).includes('audio') ? identifier(node.id) : `${identifier(node.id)}_audio`
  return identifier(node.id)
}

function upstreamEmbedding(graph: ArchitectureGraph, nodeId: string): ArchitectureNode | undefined {
  const pending = [nodeId]
  const visited = new Set<string>()
  while (pending.length > 0) {
    const current = pending.shift()!
    if (visited.has(current)) continue
    visited.add(current)
    for (const edge of graph.edges.filter((candidate) => candidate.target === current)) {
      const source = graph.nodes.find((candidate) => candidate.id === edge.source)
      if (source?.atomId === 'token-embedding') return source
      if (source) pending.push(source.id)
    }
  }
  return undefined
}

function sourcePort(graph: ArchitectureGraph, edge: ArchitectureEdge): string {
  if (edge.sourcePort) return edge.sourcePort
  const source = graph.nodes.find((node) => node.id === edge.source)
  if (source?.kind === 'semantic' && source.atomId) return modelAtomRegistry[source.atomId]?.outputs[0]?.id ?? 'output'
  if (source?.kind === 'custom-pytorch' && source.customCardGraph) return customCardOutputPorts(source.customCardGraph)[0]?.id ?? 'output'
  return 'output'
}

function sourceRank(graph: ArchitectureGraph, edge: ArchitectureEdge): number | undefined {
  const source = graph.nodes.find((node) => node.id === edge.source)
  if (!source) return undefined
  if (source.kind === 'input') {
    if (source.role === 'token-ids' || source.role === 'labels') return 2
    if (source.role === 'image') return 4
    if (source.role === 'video') return 5
    if (source.role === 'audio') return 3
    if (source.role === 'hidden') return 3
    return undefined
  }
  if (source.kind === 'custom-pytorch' && source.customCardGraph) return customCardOutputPorts(source.customCardGraph).find((port) => port.id === sourcePort(graph, edge))?.rank
  if (source.kind !== 'semantic' || !source.atomId) return undefined
  return modelAtomRegistry[source.atomId]?.outputs.find((port) => port.id === sourcePort(graph, edge))?.rank
}

function edgeMarker(edge: ArchitectureEdge): string {
  return `# labo:edge=${edge.id} source=${edge.source} target=${edge.target} source_port=${edge.sourcePort ?? 'output'} target_port=${edge.targetPort ?? 'input'}`
}

interface RenderContext {
  module: string
  settings: Record<string, number | string | boolean>
  inputs: Record<string, string>
  outputs: Record<string, string>
}

function render(template: string, context: RenderContext): string {
  return template.replace(/\{\{([^}]+)}}/g, (_, key: string) => {
    if (key === 'module') return context.module
    if (key.startsWith('in:')) {
      const port = key.slice(3)
      if (!context.inputs[port]) throw new Error(`Missing connected input port ${port} on ${context.module}`)
      return context.inputs[port]
    }
    if (key.startsWith('out:')) {
      const port = key.slice(4)
      if (!context.outputs[port]) throw new Error(`Unknown output port ${port} on ${context.module}`)
      return context.outputs[port]
    }
    if (!(key in context.settings)) throw new Error(`Missing PyTorch setting ${key} on ${context.module}`)
    return pythonValue(context.settings[key])
  })
}

function definitionFor(node: ArchitectureNode): ModelAtomDefinition {
  if (node.kind !== 'semantic' || !node.atomId) throw new Error(`Node ${node.id} is not a registry semantic atom`)
  const definition = modelAtomRegistry[node.atomId]
  if (!definition) throw new Error(`Unknown model atom: ${node.atomId}`)
  if (definition.composite) throw new Error(`${definition.label} is a recipe; expand it before PyTorch compilation`)
  return definition
}

interface RegistryCompileOptions {
  disconnectedInputs?: 'error' | 'skip'
}

const customModulePattern = /^nn\.(Linear|RMSNorm|LayerNorm|Dropout|Identity|ReLU|ReLU6|GELU|SiLU|Sigmoid|Tanh|Softplus|ELU|CELU|SELU|LeakyReLU|PReLU|Mish|Hardtanh)\((.*)\)$/
const customLiteralPattern = /^(?:-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?|True|False|None|'[^'\\]*'|"[^"\\]*")$/i

export function validCustomPyTorchModule(code: string): boolean {
  const match = customModulePattern.exec(code.trim())
  if (!match) return false
  const argumentsSource = match[2].trim()
  if (!argumentsSource) return true
  return argumentsSource.split(',').every((argument) => {
    const candidate = argument.trim()
    const assignment = /^([A-Za-z_]\w*)\s*=\s*(.+)$/.exec(candidate)
    return customLiteralPattern.test((assignment?.[2] ?? candidate).trim())
  })
}

export function compileRegistryGraph(graph: ArchitectureGraph, options: RegistryCompileOptions = {}): string {
  const nodes = orderedNodes(graph)
  const inputNodes = nodes.filter((node) => node.kind === 'input')
  const executableNodes = nodes.filter((node) => node.kind === 'semantic' || node.kind === 'custom-pytorch')
  if (executableNodes.length === 0 && inputNodes.length === 0) throw new Error('Registry compiler requires at least one input or executable atom')

  const values = new Map<string, string>()
  for (const node of inputNodes) {
    const argument = inputArgument(node)
    values.set(`${node.id}:output`, argument)
    for (const edge of graph.edges.filter((candidate) => candidate.source === node.id)) {
      values.set(`${node.id}:${edge.sourcePort ?? 'output'}`, argument)
    }
  }

  const declarations: string[] = []
  const forward: string[] = []
  const helpers = new Set<string>()
  const nodeOutputs = new Map<string, string[]>()


  for (const node of executableNodes) {
    if (node.kind === 'custom-pytorch') {
      const module = identifier(node.id)
      if (node.customCardGraph) {
        const inputPorts = customCardInputPorts(node.customCardGraph)
        const outputPorts = customCardOutputPorts(node.customCardGraph)
        if (inputPorts.length === 0 || outputPorts.length === 0) throw new Error(`Composite card ${node.id} has no external ports`)
        const className = `_LaboCard_${module}`
        const nestedSource = compileRegistryGraph(node.customCardGraph).replace('class GeneratedModel(nn.Module):', `class ${className}(nn.Module):`)
        helpers.add(nestedSource)
        declarations.push(`        # labo:node=${node.id} kind=custom-card-graph`)
        declarations.push(`        self.${module} = ${className}()`)
        const incoming = graph.edges.filter((edge) => edge.target === node.id)
        const argumentsByPort = new Map(incoming.map((edge) => [edge.targetPort ?? inputPorts[0]?.id, { edge, value: values.get(`${edge.source}:${sourcePort(graph, edge)}`) }]))
        const missing = inputPorts.find((port) => !argumentsByPort.get(port.id)?.value)
        if (missing) {
          if (options.disconnectedInputs === 'skip') continue
          throw new Error(`Missing connected input port ${missing.id} on ${node.id}`)
        }
        for (const edge of incoming) forward.push(`        ${edgeMarker(edge)}`)
        const outputVariables = outputPorts.map((port) => `${module}_${identifier(port.id)}`)
        const callTarget = outputVariables.length === 1 ? outputVariables[0] : `(${outputVariables.join(', ')})`
        const argumentsList = inputPorts.map((port) => argumentsByPort.get(port.id)!.value).join(', ')
        forward.push(`        ${callTarget} = self.${module}(${argumentsList})`)
        outputPorts.forEach((port, index) => values.set(`${node.id}:${port.id}`, outputVariables[index]))
        values.set(`${node.id}:output`, outputVariables[0])
        values.set(`${node.id}:${node.role}`, outputVariables[0])
        nodeOutputs.set(node.id, outputVariables)
        continue
      }
      const code = node.code?.trim() ?? ''
      if (!validCustomPyTorchModule(code)) throw new Error(`Invalid custom PyTorch module on ${node.id}`)
      declarations.push(`        # labo:node=${node.id} kind=custom-pytorch`)
      declarations.push(`        self.${module} = ${code}`)
      const incoming = graph.edges.filter((edge) => edge.target === node.id)
      const firstInput = incoming.map((edge) => values.get(`${edge.source}:${sourcePort(graph, edge)}`)).find(Boolean)
      if (!firstInput) {
        if (options.disconnectedInputs !== 'skip') throw new Error(`Missing connected input port hidden on ${node.id}`)
        continue
      }
      for (const edge of incoming) forward.push(`        ${edgeMarker(edge)}`)
      const output = `${module}_output`
      forward.push(`        ${output} = self.${module}(${firstInput})`)
      values.set(`${node.id}:output`, output)
      values.set(`${node.id}:hidden`, output)
      values.set(`${node.id}:${node.role}`, output)
      nodeOutputs.set(node.id, [output])
      continue
    }
    const definition = definitionFor(node)
    const module = identifier(node.id)
    const settings: Record<string, number | string | boolean> = {
      ...Object.fromEntries(definition.settings.map((setting) => [setting.id, setting.default])),
      ...graph.config,
      ...node.attributes,
    }
    const inputs: Record<string, string> = {}
    const outputs = Object.fromEntries(definition.outputs.map((port) => [port.id, `${module}_${identifier(port.id)}`]))
    const context: RenderContext = { module, settings, inputs, outputs }

    declarations.push(`        # labo:node=${node.id} atom=${definition.id}`)
    for (const line of definition.lowerings.pytorch.declarations) declarations.push(`        ${render(line, context)}`)
    if (definition.id === 'lm-head' && settings.tieEmbeddingWeights === true) {
      const embedding = upstreamEmbedding(graph, node.id)
      if (embedding) declarations.push(`        self.${module}.weight = self.${identifier(embedding.id)}.weight`)
      else if (options.disconnectedInputs !== 'skip') throw new Error(`LM head ${node.id} requests tied weights but no token embedding exists`)
    }
    for (const helper of definition.lowerings.pytorch.helpers ?? []) helpers.add(helper)

    const incoming = graph.edges.filter((edge) => edge.target === node.id)
    for (const edge of incoming) {
      const targetPort = edge.targetPort ?? definition.inputs[0]?.id
      const value = values.get(`${edge.source}:${sourcePort(graph, edge)}`)
      if (!targetPort) throw new Error(`Cannot resolve edge ${edge.id} into ${node.id}`)
      const expectedRank = definition.inputs.find((port) => port.id === targetPort)?.rank
      const actualRank = sourceRank(graph, edge)
      if (actualRank && expectedRank && actualRank !== expectedRank) {
        throw new Error(`Rank-${actualRank} ${edge.source}.${sourcePort(graph, edge)} cannot plug into rank-${expectedRank} ${node.id}.${targetPort}`)
      }
      if (!value) {
        if (options.disconnectedInputs === 'skip') continue
        throw new Error(`Cannot resolve edge ${edge.id} into ${node.id}`)
      }
      inputs[targetPort] = value
    }
    const missingPort = definition.inputs.find((port) => !inputs[port.id])
    if (missingPort) {
      if (options.disconnectedInputs !== 'skip') throw new Error(`Missing connected input port ${missingPort.id} on ${node.id}`)
      continue
    }
    for (const edge of incoming) forward.push(`        ${edgeMarker(edge)}`)
    for (const line of definition.lowerings.pytorch.forward) forward.push(`        ${render(line, context)}`)
    for (const [port, variable] of Object.entries(outputs)) values.set(`${node.id}:${port}`, variable)
    nodeOutputs.set(node.id, Object.values(outputs))
  }

  const compiledNodeIds = new Set(nodeOutputs.keys())
  const sinks = executableNodes.filter((node) => compiledNodeIds.has(node.id) && !graph.edges.some((edge) => edge.source === node.id && compiledNodeIds.has(edge.target)))
  const results = sinks.flatMap((node) => nodeOutputs.get(node.id) ?? [])
  const fallbackResults = inputNodes.map((node) => inputArgument(node))
  const returnedValues = results.length > 0 ? results : fallbackResults
  if (returnedValues.length === 0) {
    if (options.disconnectedInputs !== 'skip') throw new Error('Compiled graph has no executable output')
    forward.push('        return None')
  } else {
    forward.push(`        return ${returnedValues.length === 1 ? returnedValues[0] : `(${returnedValues.join(', ')})`}`)
  }

  const argumentsList = inputNodes.map((node) => `${inputArgument(node)}: torch.Tensor`).join(', ')
  const helperSource = [...helpers].join('\n\n')
  return `import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n${helperSource ? `\n\n${helperSource}` : ''}\n\n\nclass GeneratedModel(nn.Module):\n    """Generated from the LABO AI semantic atom registry and elastic topology."""\n\n    def __init__(self):\n        super().__init__()\n${declarations.join('\n')}\n\n    def forward(self, ${argumentsList}):\n${forward.join('\n')}\n`
}
