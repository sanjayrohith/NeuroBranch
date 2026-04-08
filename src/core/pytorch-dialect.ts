import {
  addNode,
  removeNode,
  updateNodeAttributes,
  validateGraph,
  type ArchitectureGraph,
  type NodeKind,
} from './ir'
import { modelAtomRegistry, type ModelAtomDefinition } from './model-atoms'

export interface PyTorchDialectDiagnostic {
  code: 'INVALID_LINEAR_DECLARATION' | 'INVALID_SEMANTIC_DECLARATION' | 'UNKNOWN_NODE_MARKER' | 'UNKNOWN_ATOM' | 'UNKNOWN_EDGE_ENDPOINT' | 'UNSUPPORTED_MANAGED_KIND'
  nodeId: string
  message: string
}

export interface PyTorchDialectResult {
  graph: ArchitectureGraph
  validation: ReturnType<typeof validateGraph>
  diagnostics: PyTorchDialectDiagnostic[]
}

interface ManagedBlock {
  nodeId: string
  kind: NodeKind
  atomId?: string
  declarations: string[]
}

interface ManagedEdge {
  id: string
  source: string
  target: string
  sourcePort?: string
  targetPort?: string
}

function managedBlocks(source: string): ManagedBlock[] {
  const marker = /^\s*# labo:node=([^\s]+) (kind|atom)=([^\s]+)\s*$/gm
  const matches = [...source.matchAll(marker)]
  return matches.map((match, index) => {
    const start = (match.index ?? 0) + match[0].length
    const end = matches[index + 1]?.index ?? source.length
    const declarations = source.slice(start, end).split('\n')
      .map((line) => line.trim())
      .filter((line) => line.startsWith('self.'))
    return { nodeId: match[1], kind: match[2] === 'atom' ? 'semantic' : match[3] as NodeKind, atomId: match[2] === 'atom' ? match[3] : undefined, declarations }
  })
}

function pythonLiteral(value: string): number | string | boolean {
  const trimmed = value.trim()
  if (trimmed === 'True') return true
  if (trimmed === 'False') return false
  if ((trimmed.startsWith("'") && trimmed.endsWith("'")) || (trimmed.startsWith('"') && trimmed.endsWith('"'))) return trimmed.slice(1, -1)
  const numeric = Number(trimmed)
  return Number.isNaN(numeric) ? trimmed : numeric
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function settingsFromDeclarations(definition: ModelAtomDefinition, declarations: string[]): Record<string, number | string | boolean> {
  const settingIds = new Set(definition.settings.map((setting) => setting.id))
  const settings: Record<string, number | string | boolean> = {}
  definition.lowerings.pytorch.declarations.forEach((template, index) => {
    const declaration = declarations[index]
    if (!declaration) return
    let cursor = 0
    let pattern = '^'
    const captures: string[] = []
    for (const match of template.matchAll(/\{\{([^}]+)}}/g)) {
      pattern += escapeRegex(template.slice(cursor, match.index))
      const key = match[1]
      if (key === 'module') pattern += '[A-Za-z_][A-Za-z0-9_]*'
      else if (settingIds.has(key)) {
        pattern += '(.+?)'
        captures.push(key)
      } else pattern += '.+?'
      cursor = (match.index ?? 0) + match[0].length
    }
    pattern += `${escapeRegex(template.slice(cursor))}$`
    const parsed = declaration.match(new RegExp(pattern))
    if (!parsed) throw new Error(`Declaration does not match ${definition.id} lowering: ${declaration}`)
    captures.forEach((key, captureIndex) => { settings[key] = pythonLiteral(parsed[captureIndex + 1]) })
  })
  return settings
}

function managedEdges(source: string): ManagedEdge[] {
  const marker = /^\s*# labo:edge=([^\s]+) source=([^\s]+) target=([^\s]+)(?: source_port=([^\s]+) target_port=([^\s]+))?\s*$/gm
  return [...source.matchAll(marker)].map((match) => ({
    id: match[1],
    source: match[2],
    target: match[3],
    sourcePort: match[4],
    targetPort: match[5],
  }))
}

export function parsePyTorchDialect(source: string, currentGraph: ArchitectureGraph): PyTorchDialectResult {
  const blocks = managedBlocks(source)
  const edges = managedEdges(source)
  const diagnostics: PyTorchDialectDiagnostic[] = []
  const foundIds = new Set(blocks.map((block) => block.nodeId))
  let graph = structuredClone(currentGraph)

  const managedExistingNodes = currentGraph.nodes.filter((node) => node.kind === 'linear' || node.kind === 'custom-pytorch' || node.kind === 'semantic')
  for (const node of managedExistingNodes) {
    if (!foundIds.has(node.id)) graph = removeNode(graph, node.id)
  }

  for (const block of blocks) {
    let node = graph.nodes.find((candidate) => candidate.id === block.nodeId)
    if (!node && block.kind === 'semantic' && block.atomId) {
      const definition = modelAtomRegistry[block.atomId]
      if (!definition) {
        diagnostics.push({ code: 'UNKNOWN_ATOM', nodeId: block.nodeId, message: `Unknown LABO AI atom: ${block.atomId}` })
        continue
      }
      let parsedSettings: Record<string, number | string | boolean> = {}
      try {
        parsedSettings = settingsFromDeclarations(definition, block.declarations)
      } catch (error) {
        diagnostics.push({ code: 'INVALID_SEMANTIC_DECLARATION', nodeId: block.nodeId, message: error instanceof Error ? error.message : String(error) })
        continue
      }
      const outputTensor = definition.outputs[0]?.tensor
      const role = outputTensor === 'query' || outputTensor === 'key' || outputTensor === 'value' || outputTensor === 'image' || outputTensor === 'video' || outputTensor === 'audio' ? outputTensor : outputTensor === 'hidden' ? 'hidden' : 'output'
      graph = addNode(graph, {
        id: block.nodeId,
        kind: 'semantic',
        atomId: block.atomId,
        label: definition.label,
        role,
        position: { x: 40 + (graph.nodes.length % 4) * 140, y: 80 + Math.floor(graph.nodes.length / 4) * 100 },
        attributes: { ...Object.fromEntries(definition.settings.map((setting) => [setting.id, setting.default])), ...parsedSettings },
      })
      node = graph.nodes.find((candidate) => candidate.id === block.nodeId)
    }
    if (!node) {
      diagnostics.push({ code: 'UNKNOWN_NODE_MARKER', nodeId: block.nodeId, message: `Unknown LABO AI node marker: ${block.nodeId}` })
      continue
    }
    if (block.kind === 'linear') {
      const parsed = block.declarations[0]?.match(/^self\.[A-Za-z_][A-Za-z0-9_]*\s*=\s*nn\.Linear\((\d+),\s*(\d+),\s*bias=(True|False)\)$/)
      if (!parsed) {
        diagnostics.push({ code: 'INVALID_LINEAR_DECLARATION', nodeId: block.nodeId, message: `Unsupported Linear declaration for ${block.nodeId}` })
        continue
      }
      graph = updateNodeAttributes(graph, block.nodeId, {
        inFeatures: Number(parsed[1]),
        outFeatures: Number(parsed[2]),
        bias: parsed[3] === 'True',
      })
      continue
    }
    if (block.kind === 'custom-pytorch') {
      const assignment = block.declarations[0]?.match(/^self\.[A-Za-z_][A-Za-z0-9_]*\s*=\s*(.+)$/)
      if (assignment) {
        graph = {
          ...graph,
          nodes: graph.nodes.map((candidate) => candidate.id === block.nodeId ? { ...candidate, code: assignment[1] } : candidate),
        }
      }
      continue
    }
    if (block.kind === 'semantic' && block.atomId) {
      const definition = modelAtomRegistry[block.atomId]
      if (!definition || node.atomId !== block.atomId) {
        diagnostics.push({ code: 'UNKNOWN_ATOM', nodeId: block.nodeId, message: `Unknown or mismatched LABO AI atom: ${block.atomId}` })
        continue
      }
      try {
        graph = updateNodeAttributes(graph, block.nodeId, settingsFromDeclarations(definition, block.declarations))
      } catch (error) {
        diagnostics.push({ code: 'INVALID_SEMANTIC_DECLARATION', nodeId: block.nodeId, message: error instanceof Error ? error.message : String(error) })
      }
      continue
    }
    diagnostics.push({ code: 'UNSUPPORTED_MANAGED_KIND', nodeId: block.nodeId, message: `Unsupported managed kind: ${block.kind}` })
  }

  const nodeIds = new Set(graph.nodes.map((node) => node.id))
  graph = {
    ...graph,
    edges: edges.flatMap((edge) => {
      if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) {
        diagnostics.push({
          code: 'UNKNOWN_EDGE_ENDPOINT',
          nodeId: edge.id,
          message: `Unknown endpoint in LABO AI edge marker ${edge.id}: ${edge.source} -> ${edge.target}`,
        })
        return []
      }
      const existing = currentGraph.edges.find((candidate) => candidate.id === edge.id)
      return [{
        ...existing,
        id: edge.id,
        source: edge.source,
        target: edge.target,
        ...(edge.sourcePort && edge.sourcePort !== 'output' ? { sourcePort: edge.sourcePort } : {}),
        ...(edge.targetPort && edge.targetPort !== 'input' ? { targetPort: edge.targetPort } : {}),
      }]
    }),
  }

  return { graph, validation: validateGraph(graph), diagnostics }
}
