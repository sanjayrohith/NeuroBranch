import type { ArchitectureGraph } from './ir'

export const MODEL_WORKSPACE_STORAGE_KEY = 'labo.model-workspace.v1'

export interface ModelPresetDraft {
  graph: ArchitectureGraph
  selectedNodeId: string
}

export interface ModelWorkspaceState {
  activePresetId: string
  drafts: Record<string, ModelPresetDraft>
  userPresets: ArchitectureGraph[]
  updatedAt: number
}

export const emptyModelWorkspace = (): ModelWorkspaceState => ({
  activePresetId: '',
  drafts: {},
  userPresets: [],
  updatedAt: 0,
})

function isGraph(value: unknown): value is ArchitectureGraph {
  if (!value || typeof value !== 'object') return false
  const graph = value as Partial<ArchitectureGraph>
  return typeof graph.id === 'string'
    && typeof graph.name === 'string'
    && (graph.architecture === 'gqa' || graph.architecture === 'custom')
    && Array.isArray(graph.nodes)
    && graph.nodes.every((node) => (
      typeof node?.id === 'string'
      && typeof node?.label === 'string'
      && typeof node?.position?.x === 'number'
      && Number.isFinite(node.position.x)
      && typeof node?.position?.y === 'number'
      && Number.isFinite(node.position.y)
    ))
    && Array.isArray(graph.edges)
    && graph.edges.every((edge) => typeof edge?.id === 'string' && typeof edge?.source === 'string' && typeof edge?.target === 'string')
    && typeof graph.config === 'object'
    && typeof graph.contracts === 'object'
}

function isDraft(value: unknown): value is ModelPresetDraft {
  if (!value || typeof value !== 'object') return false
  const draft = value as Partial<ModelPresetDraft>
  return isGraph(draft.graph) && typeof draft.selectedNodeId === 'string'
}

function sameGraphContent(left: ArchitectureGraph, right: ArchitectureGraph): boolean {
  const content = ({ architecture, nodes, edges, groups, config, contracts }: ArchitectureGraph) => ({ architecture, nodes, edges, groups, config, contracts })
  return JSON.stringify(content(left)) === JSON.stringify(content(right))
}

function browserStorage(): Storage | undefined {
  try {
    return typeof window === 'undefined' ? undefined : window.localStorage
  } catch {
    return undefined
  }
}

export function parseModelWorkspace(value: unknown): ModelWorkspaceState {
  if (!value || typeof value !== 'object') return emptyModelWorkspace()
  const candidate = value as Partial<ModelWorkspaceState>
  const drafts = Object.fromEntries(Object.entries(candidate.drafts ?? {}).filter((entry): entry is [string, ModelPresetDraft] => isDraft(entry[1])))
  const userPresets = Array.isArray(candidate.userPresets) ? candidate.userPresets.filter(isGraph) : []
  const blankDraft = drafts['blank-starter']
  if (blankDraft && userPresets.some((preset) => sameGraphContent(blankDraft.graph, preset))) delete drafts['blank-starter']
  return {
    activePresetId: typeof candidate.activePresetId === 'string' ? candidate.activePresetId : '',
    drafts,
    userPresets,
    updatedAt: typeof candidate.updatedAt === 'number' && Number.isFinite(candidate.updatedAt) ? candidate.updatedAt : 0,
  }
}

export function cloneArchitectureGraph(graph: ArchitectureGraph): ArchitectureGraph {
  return JSON.parse(JSON.stringify(graph)) as ArchitectureGraph
}

export function loadModelWorkspace(storage: Storage | undefined = browserStorage()): ModelWorkspaceState {
  if (!storage) return emptyModelWorkspace()
  try {
    return parseModelWorkspace(JSON.parse(storage.getItem(MODEL_WORKSPACE_STORAGE_KEY) ?? 'null') as unknown)
  } catch {
    return emptyModelWorkspace()
  }
}

export function saveModelWorkspaceCache(workspace: ModelWorkspaceState, storage: Storage | undefined = browserStorage()): boolean {
  const snapshot = { ...workspace, updatedAt: workspace.updatedAt || Date.now() }
  if (!storage) return false
  try {
    storage.setItem(MODEL_WORKSPACE_STORAGE_KEY, JSON.stringify(snapshot))
    return true
  } catch {
    return false
  }
}

export function saveModelWorkspace(workspace: ModelWorkspaceState, storage: Storage | undefined = browserStorage()): boolean {
  return saveModelWorkspaceCache(workspace, storage)
}
