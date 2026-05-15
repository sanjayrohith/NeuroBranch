import { lazy, Suspense, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import {
  Blocks,
  Braces,
  Code2,
  Cpu,
  PanelLeft,
  SplitSquareHorizontal,
  Play,
  Trash2,
  X,
  Zap,
} from 'lucide-react'
import '../App.scss'
import { AtomicPlayer, type AtomicPlayerSnapshot } from '../core/atomic-player'
import { executionLayers } from '../core/execution-plan'
import { architectureComponents } from '../core/graph-components'
import { findOpenGraphPosition, layoutArchitectureGraph, layoutParallelArchitecture } from '../core/graph-placement'
import { addNode, compileToPyTorch, removeNode, validateGraph, type ArchitectureGraph, type ArchitectureNode, type TensorRole } from '../core/ir'
import { connectCable } from '../core/cables'
import { cloneArchitectureGraph, emptyModelWorkspace, loadModelWorkspace, parseModelWorkspace, saveModelWorkspace, saveModelWorkspaceCache, type ModelPresetDraft } from '../core/model-workspace'
import { modelAtomRegistry, type ModelAtomDefinition } from '../core/model-atoms'
import { blankStarterPreset, complexityDeepPreset, gptLikeStarterPreset, tokenMoePreset, trBasicPreset } from '../core/presets'
import { audioEncoderPreset, multimodalImageEditorPreset, videoTransformerPreset, visionTransformerPreset } from '../core/media-presets'
import { researchBpePreset } from '../core/tokenizer-presets'
import { parsePyTorchDialect, type PyTorchDialectDiagnostic } from '../core/pytorch-dialect'
import { validCustomPyTorchModule } from '../core/pytorch-compiler'
import { customCardInputPorts, customCardOutputPorts, validateCustomCardGraph } from '../core/custom-card-graph'
import { deriveGraphStats } from '../core/stats'
import { GraphCanvas } from './GraphCanvas'
import { PythonCodeEditor, PythonCodePreview } from './PythonCodeEditor'
import { AskLaboPanel } from './AskLaboPanel'
import type { AgentGraphAction } from '../core/agentic-graph'
import type { CustomCardCreatorHandle, CustomCardDestination, CustomCardCreateResult } from './CustomCardCreator'
import type { CustomPyTorchCard } from './custom-card'
import { ExportMenu } from './ExportMenu'
import { exportArchitectureDiagram, exportPyTorchCode } from './export-actions'
import { previewModelAtom } from '../core/browser-atomic-preview'
import { StudioEditor, StudioInspector, StudioLibrary, StudioStatusbar, StudioToolbar, StudioViewSwitcher, StudioWorkspace } from '../studio/StudioShell'
import { StudioLibrarySection } from '../studio/StudioLibraryParts'
import { InspectorMetric, InspectorSection, InspectorSelection } from '../studio/StudioInspectorParts'
import { StudioCodePanel, StudioPanelTab } from '../studio/StudioPanels'
import { ModelInteractionSwitcher, ModelPanelControls, ModelPlayerControls } from './ModelToolbarControls'
import { ModelPresetMenu, ModelPromptMenu } from './ModelPresetControls'
import { setLibraryDragPreview } from '../studio/libraryDragPreview'
import { WorkspaceSettingsContent } from '../studio/WorkspaceSettingsContent'

const CustomCardCreator = lazy(() => import('./CustomCardCreator').then((module) => ({ default: module.CustomCardCreator })))

type ViewMode = 'blocks' | 'pytorch' | 'split'
type InteractionMode = 'add' | 'edit'
export type ModelEditorContext = 'architecture-add' | 'architecture-edit' | 'reusable-card'

interface CardEditDraft {
  label: string
  attributes?: Record<string, number | string | boolean>
  code?: string
}

const CUSTOM_CARDS_STORAGE_KEY = 'labo.custom-pytorch-cards.v1'

function isCustomCard(value: unknown): value is CustomPyTorchCard {
  if (!value || typeof value !== 'object') return false
  const card = value as Partial<CustomPyTorchCard>
  const legacyModule = typeof card.code === 'string' && validCustomPyTorchModule(card.code)
  const compositeGraph = card.graph && typeof card.graph === 'object' && validateCustomCardGraph(card.graph).length === 0
  return typeof card.id === 'string' && typeof card.label === 'string' && (legacyModule || Boolean(compositeGraph))
}

const builtInModelPresets = [blankStarterPreset, gptLikeStarterPreset, trBasicPreset, tokenMoePreset, complexityDeepPreset, visionTransformerPreset, multimodalImageEditorPreset, videoTransformerPreset, audioEncoderPreset]
const presetMenuLabels: Record<string, string> = {
  [blankStarterPreset.id]: 'Blank starter',
  [gptLikeStarterPreset.id]: 'GPT-like',
  [trBasicPreset.id]: 'TR Basic',
  [tokenMoePreset.id]: 'Learned MoE',
  [complexityDeepPreset.id]: 'TR 300M',
  [visionTransformerPreset.id]: 'Vision',
  [multimodalImageEditorPreset.id]: 'Image edit',
  [videoTransformerPreset.id]: 'Video',
  [audioEncoderPreset.id]: 'Audio',
}

function loadCustomCards(): CustomPyTorchCard[] {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(CUSTOM_CARDS_STORAGE_KEY) ?? '[]')
    if (!Array.isArray(parsed)) return []
    return parsed.filter(isCustomCard)
  } catch {
    return []
  }
}

const graphInputDefinitions: Array<{ role: TensorRole; label: string }> = [
  { role: 'token-ids', label: 'Token IDs' },
  { role: 'image', label: 'Image Tensor' },
  { role: 'video', label: 'Video Tensor' },
  { role: 'audio', label: 'Audio Waveform' },
  { role: 'hidden', label: 'Hidden State' },
  { role: 'labels', label: 'Training Labels' },
]



export function ModelStudio({ askOpen = false, onCloseAsk = () => undefined, onEditorContextChange = () => undefined, requestedCard, onRequestedCardHandled = () => undefined }: { askOpen?: boolean; onCloseAsk?: () => void; onEditorContextChange?: (context: ModelEditorContext) => void; requestedCard?: { atomId: string; requestId: number }; onRequestedCardHandled?: () => void }) {
  const webRuntime = window.labo?.runtime === 'web'
  const desktopRuntime = window.labo?.runtime === 'electron'
  const [initialWorkspace] = useState(() => webRuntime || desktopRuntime ? emptyModelWorkspace() : loadModelWorkspace())
  const initialPreset = initialWorkspace.userPresets.find((preset) => preset.id === initialWorkspace.activePresetId)
    ?? builtInModelPresets.find((preset) => preset.id === initialWorkspace.activePresetId)
    ?? complexityDeepPreset
  const initialDraft = initialWorkspace.drafts[initialPreset.id]
  const initialGraph = initialDraft?.graph ?? initialPreset
  const [graph, setGraph] = useState(() => cloneArchitectureGraph(initialGraph))
  const [selectedNodeId, setSelectedNodeId] = useState(initialDraft?.selectedNodeId ?? initialGraph.nodes[0]?.id ?? '')
  const [agentSelectedNodeIds, setAgentSelectedNodeIds] = useState<string[]>([])
  const [view, setView] = useState<ViewMode>('blocks')
  const [selectedArchitectureId, setSelectedArchitectureId] = useState('')
  const [interactionMode, setInteractionMode] = useState<InteractionMode>('add')
  const [createCardOpen, setCreateCardOpen] = useState(false)
  const [libraryOpen, setLibraryOpen] = useState(true)
  const [inspectorOpen, setInspectorOpen] = useState(false)
  const [modelPlayerSnapshot, setModelPlayerSnapshot] = useState<AtomicPlayerSnapshot>({ status: 'idle', currentAtomId: initialGraph.nodes[0]?.id, results: initialGraph.nodes.map((node) => ({ atomId: node.id, status: 'pending' })) })
  const modelPlayerRef = useRef<AtomicPlayer | null>(null)
  const cardPlayerRef = useRef<CustomCardCreatorHandle | null>(null)
  const [cardPlayerSnapshot, setCardPlayerSnapshot] = useState<AtomicPlayerSnapshot>({ status: 'idle', results: [] })
  const pendingAgentRunRef = useRef<'play' | 'step' | undefined>(undefined)
  const presetDraftsRef = useRef(new Map<string, ModelPresetDraft>(Object.entries(initialWorkspace.drafts)))
  const [userPresets, setUserPresets] = useState(() => initialWorkspace.userPresets.map(cloneArchitectureGraph))
  const [presetName, setPresetName] = useState('My model')
  const [presetError, setPresetError] = useState('')
  const [confirmPresetReset, setConfirmPresetReset] = useState(false)
  const [databaseReady, setDatabaseReady] = useState(false)
  const [webAuthenticated, setWebAuthenticated] = useState(false)
  const webSaveTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const webPendingSaveRef = useRef<{ workspace: unknown; customCards: CustomPyTorchCard[] } | undefined>(undefined)
  const webAuthenticatedRef = useRef(false)
  const startupGraphRef = useRef(graph)
  const startupSelectionRef = useRef(selectedNodeId)
  const startupUserPresetsRef = useRef(userPresets)
  const latestGraphRef = useRef(graph)
  const latestSelectionRef = useRef(selectedNodeId)

  useEffect(() => onEditorContextChange(createCardOpen ? 'reusable-card' : interactionMode === 'edit' ? 'architecture-edit' : 'architecture-add'), [createCardOpen, interactionMode, onEditorContextChange])
  const latestUserPresetsRef = useRef(userPresets)
  latestGraphRef.current = graph
  latestSelectionRef.current = selectedNodeId
  latestUserPresetsRef.current = userPresets
  webAuthenticatedRef.current = webAuthenticated
  const graphArchitectures = useMemo(() => architectureComponents(graph, [...builtInModelPresets, ...userPresets]), [graph, userPresets])
  const selectedArchitecture = graphArchitectures.find((component) => component.id === selectedArchitectureId) ?? graphArchitectures[0]
  const codeGraph = selectedArchitecture?.graph ?? graph
  const code = useMemo(() => compileToPyTorch(codeGraph), [codeGraph])
  const [codeDraft, setCodeDraft] = useState(code)
  const [parseDiagnostics, setParseDiagnostics] = useState<PyTorchDialectDiagnostic[]>([])
  const [sampleText, setSampleText] = useState('Hello LABO AI')
  const [promptTokenCount, setPromptTokenCount] = useState<number>()
  const [modelOutput, setModelOutput] = useState<LaboRuntimeTrace['modelOutput']>()
  const [customCards, setCustomCards] = useState<CustomPyTorchCard[]>(() => webRuntime || desktopRuntime ? [] : loadCustomCards())
  const [editingNodeId, setEditingNodeId] = useState<string>()
  const [cardEditDraft, setCardEditDraft] = useState<CardEditDraft>()
  const [cardEditError, setCardEditError] = useState('')
  const customCardSequenceRef = useRef(1)
  const stats = useMemo(() => deriveGraphStats(graph), [graph])
  const validation = useMemo(() => validateGraph(graph), [graph])
  const acceptsTokenIds = useMemo(() => graph.nodes.some((node) => node.role === 'token-ids' || graph.edges.some((edge) => edge.source === node.id && (edge.sourcePort === 'tokenIds' || edge.targetPort === 'tokenIds'))), [graph])
  const blankGraph = graph.nodes.length === 0
  const pytorchMappingComplete = validation.valid && code.includes('class GeneratedModel(nn.Module):')
  const pytorchDraftAvailable = code.includes('class GeneratedModel(nn.Module):')
  const nativePyTorchRuntime = typeof window.labo?.runAtomic === 'function'
  const runtimeAvailable = !blankGraph
  const selectedNode = graph.nodes.find((node) => node.id === selectedNodeId)
  const selectedGroup = graph.groups?.find((group) => group.id === selectedNodeId)
  const editingNode = editingNodeId ? graph.nodes.find((node) => node.id === editingNodeId) : undefined


  useLayoutEffect(() => setCodeDraft(code), [code])

  useEffect(() => {
    if (selectedArchitecture && selectedArchitecture.id !== selectedArchitectureId) setSelectedArchitectureId(selectedArchitecture.id)
  }, [selectedArchitecture, selectedArchitectureId])

  useEffect(() => {
    const containingArchitecture = graphArchitectures.find((architecture) => architecture.nodeIds.includes(selectedNodeId))
    if (containingArchitecture && containingArchitecture.id !== selectedArchitectureId) setSelectedArchitectureId(containingArchitecture.id)
  }, [graphArchitectures, selectedArchitectureId, selectedNodeId])

  useEffect(() => setConfirmPresetReset(false), [graph.id])

  useEffect(() => {
    if (!webRuntime && !desktopRuntime) window.localStorage.setItem(CUSTOM_CARDS_STORAGE_KEY, JSON.stringify(customCards))
  }, [customCards, desktopRuntime, webRuntime])

  useEffect(() => {
    let cancelled = false
    if (webRuntime) {
      const load = window.labo?.loadWebWorkspace
      if (!load) {
        setDatabaseReady(true)
        return
      }
      void load().then((result) => {
        if (cancelled) return
        webAuthenticatedRef.current = result.authenticated
        setWebAuthenticated(result.authenticated)
        const serverWorkspace = parseModelWorkspace(result.workspace)
        const untouchedSinceStartup = latestGraphRef.current === startupGraphRef.current
          && latestSelectionRef.current === startupSelectionRef.current
          && latestUserPresetsRef.current === startupUserPresetsRef.current
        if (result.authenticated && serverWorkspace.updatedAt > 0) {
          if (untouchedSinceStartup) {
            const storedPreset = serverWorkspace.userPresets.find((preset) => preset.id === serverWorkspace.activePresetId)
              ?? builtInModelPresets.find((preset) => preset.id === serverWorkspace.activePresetId)
              ?? complexityDeepPreset
            const storedDraft = serverWorkspace.drafts[storedPreset.id]
            const storedGraph = cloneArchitectureGraph(storedDraft?.graph ?? storedPreset)
            presetDraftsRef.current = new Map(Object.entries(serverWorkspace.drafts))
            setUserPresets(serverWorkspace.userPresets.map(cloneArchitectureGraph))
            setGraph(storedGraph)
            setSelectedNodeId(storedDraft?.selectedNodeId ?? storedGraph.nodes[0]?.id ?? '')
          } else {
            presetDraftsRef.current = new Map([...Object.entries(serverWorkspace.drafts), ...presetDraftsRef.current])
            setUserPresets((current) => {
              const merged = new Map(serverWorkspace.userPresets.map((preset) => [preset.id, cloneArchitectureGraph(preset)]))
              for (const preset of current) merged.set(preset.id, preset)
              return [...merged.values()]
            })
          }
        }
        if (result.authenticated && Array.isArray(result.customCards)) {
          const storedCards = result.customCards.filter(isCustomCard)
          setCustomCards((current) => {
            if (untouchedSinceStartup) return storedCards
            const merged = new Map(storedCards.map((card) => [card.id, card]))
            for (const card of current) merged.set(card.id, card)
            return [...merged.values()]
          })
        }
        setDatabaseReady(true)
      }).catch(() => {
        if (!cancelled) setDatabaseReady(true)
      })
      return () => { cancelled = true }
    }
    if (desktopRuntime && window.labo?.loadDesktopState) {
      const loadDesktopState = window.labo.loadDesktopState
      void (async () => {
        const nativePayload = await loadDesktopState('model')
        const payload = nativePayload && typeof nativePayload === 'object' ? nativePayload as { workspace?: unknown; customCards?: unknown } : undefined
        const nativeWorkspace = parseModelWorkspace(payload?.workspace)
        const storedWorkspace = nativeWorkspace.updatedAt > 0 ? nativeWorkspace : undefined
        if (cancelled) return
        const untouchedSinceStartup = latestGraphRef.current === startupGraphRef.current
          && latestSelectionRef.current === startupSelectionRef.current
          && latestUserPresetsRef.current === startupUserPresetsRef.current
        if (storedWorkspace && untouchedSinceStartup) {
          const storedPreset = storedWorkspace.userPresets.find((preset) => preset.id === storedWorkspace.activePresetId)
            ?? builtInModelPresets.find((preset) => preset.id === storedWorkspace.activePresetId)
            ?? complexityDeepPreset
          const storedDraft = storedWorkspace.drafts[storedPreset.id]
          const storedGraph = cloneArchitectureGraph(storedDraft?.graph ?? storedPreset)
          presetDraftsRef.current = new Map(Object.entries(storedWorkspace.drafts))
          setUserPresets(storedWorkspace.userPresets.map(cloneArchitectureGraph))
          setGraph(storedGraph)
          setSelectedNodeId(storedDraft?.selectedNodeId ?? storedGraph.nodes[0]?.id ?? '')
        }
        if (Array.isArray(payload?.customCards) && untouchedSinceStartup) setCustomCards(payload.customCards.filter(isCustomCard))
        setDatabaseReady(true)
      })().catch(() => { if (!cancelled) setDatabaseReady(true) })
      return () => { cancelled = true }
    }
    setDatabaseReady(true)
    return () => { cancelled = true }
  }, [desktopRuntime, initialWorkspace, webRuntime])

  useEffect(() => {
    const selected = graph.nodes.some((node) => node.id === selectedNodeId) || graph.groups?.some((group) => group.id === selectedNodeId)
      ? selectedNodeId
      : ''
    presetDraftsRef.current.set(graph.id, { graph, selectedNodeId: selected })
    const workspace = {
      activePresetId: graph.id,
      drafts: Object.fromEntries(presetDraftsRef.current),
      userPresets,
      updatedAt: Date.now(),
    }
    if (databaseReady) {
      if (webRuntime) {
        if (webAuthenticated && window.labo?.saveWebWorkspace) {
          const payload = { workspace, customCards }
          webPendingSaveRef.current = payload
          if (webSaveTimerRef.current) clearTimeout(webSaveTimerRef.current)
          webSaveTimerRef.current = setTimeout(() => {
            void window.labo?.saveWebWorkspace?.(payload).then(() => {
              if (webPendingSaveRef.current === payload) webPendingSaveRef.current = undefined
            })
          }, 700)
        }
      } else if (desktopRuntime && window.labo?.saveDesktopState) {
        void window.labo.saveDesktopState('model', { workspace, customCards })
      } else saveModelWorkspace(workspace)
    } else if (!webRuntime && !desktopRuntime && (graph !== startupGraphRef.current || selectedNodeId !== startupSelectionRef.current || userPresets !== startupUserPresetsRef.current)) {
      saveModelWorkspaceCache(workspace)
    }
    return () => {
      if (webSaveTimerRef.current) clearTimeout(webSaveTimerRef.current)
    }
  }, [customCards, databaseReady, desktopRuntime, graph, selectedNodeId, userPresets, webAuthenticated, webRuntime])

  useEffect(() => () => {
    if (webSaveTimerRef.current) clearTimeout(webSaveTimerRef.current)
    const pending = webPendingSaveRef.current
    if (webRuntime && webAuthenticatedRef.current && pending && window.labo?.saveWebWorkspace) void window.labo.saveWebWorkspace(pending)
  }, [webRuntime])

  useEffect(() => {
    let tracePromise: Promise<LaboRuntimeTrace> | undefined
    setModelOutput(undefined)
    setPromptTokenCount(undefined)
    const executionPlan = validation.valid ? executionLayers(graph) : graph.nodes.map((node) => [node.id])
    const player = new AtomicPlayer(executionPlan, async (atomId) => {
      const runAtomic = window.labo?.runAtomic
      if (!runAtomic) return previewModelAtom(graph, atomId)
      const runArchitectures = async (tokenIds?: number[]) => {
        const traces = await Promise.all(graphArchitectures.map((architecture) => runAtomic({ kind: 'model', graph: architecture.graph, ...(tokenIds ? { tokenIds } : {}) })))
        const failed = traces.find((trace) => trace.status === 'failed')
        return {
          engine: 'pytorch' as const,
          status: failed ? 'failed' as const : 'completed' as const,
          ...(failed?.currentAtomId ? { currentAtomId: failed.currentAtomId } : {}),
          ...(failed?.error ? { error: failed.error } : {}),
          modelOutput: traces.findLast((trace) => trace.modelOutput)?.modelOutput,
          results: traces.flatMap((trace) => trace.results),
        }
      }
      tracePromise ??= (acceptsTokenIds
        ? (async () => {
            const tokenTrace = await runAtomic({ kind: 'tokenizer', pipeline: researchBpePreset, sample: sampleText })
            if (tokenTrace.status === 'failed') throw new Error(tokenTrace.error ?? 'Tokenizer failed')
            if (!tokenTrace.tokenIds?.length) throw new Error('Tokenizer returned no Token IDs')
            setPromptTokenCount(tokenTrace.tokenIds.length)
            return runArchitectures(tokenTrace.tokenIds)
          })()
        : runArchitectures()).then((trace) => {
          setModelOutput(trace.modelOutput)
          return trace
        })
      const trace = await tracePromise
      const result = trace.results.find((candidate) => candidate.atomId === atomId)
      if (!result) throw new Error(trace.error ?? `PyTorch stopped before ${atomId}`)
      if (result.status === 'failed') throw new Error(result.error ?? `PyTorch failed at ${atomId}`)
      return { summary: result.summary }
    }, { onRestart: () => { tracePromise = undefined; setModelOutput(undefined) }, continueAfterFailure: true })
    modelPlayerRef.current = player
    return player.subscribe(setModelPlayerSnapshot)
  }, [acceptsTokenIds, graph, graphArchitectures, sampleText, validation.valid])

  useEffect(() => {
    const mode = pendingAgentRunRef.current
    if (!mode) return
    pendingAgentRunRef.current = undefined
    if (mode === 'play') void modelPlayerRef.current?.play()
    else void modelPlayerRef.current?.step()
  }, [graph])

  const applyPyTorch = () => {
    const parsed = parsePyTorchDialect(codeDraft, codeGraph)
    setParseDiagnostics(parsed.diagnostics)
    const selectedIds = new Set(codeGraph.nodes.map((node) => node.id))
    setGraph((current) => ({
      ...current,
      config: parsed.graph.config,
      contracts: parsed.graph.contracts,
      nodes: [...current.nodes.filter((node) => !selectedIds.has(node.id)), ...parsed.graph.nodes],
      edges: [...current.edges.filter((edge) => !selectedIds.has(edge.source) && !selectedIds.has(edge.target)), ...parsed.graph.edges],
    }))
  }

  const addModelAtom = (definition: ModelAtomDefinition, desiredPosition?: { x: number; y: number }, variant?: { label: string; attributes: Record<string, number | string | boolean> }) => {
    const sequence = graph.nodes.filter((node) => node.id.startsWith(`${definition.id}-`)).length + 1
    const outputTensor = definition.outputs[0]?.tensor
    const role: TensorRole = outputTensor === 'query' || outputTensor === 'key' || outputTensor === 'value' || outputTensor === 'image' || outputTensor === 'video' || outputTensor === 'audio'
      ? outputTensor
      : outputTensor === 'logits' || outputTensor === 'scalar' ? 'output' : 'hidden'
    const id = `${definition.id}-${sequence}`
    setGraph((current) => {
      const position = desiredPosition ?? findOpenGraphPosition(current)
      return addNode(current, {
        id,
        kind: 'semantic',
        atomId: definition.id,
        label: variant?.label ?? definition.label,
        role,
        position,
        attributes: { ...Object.fromEntries(definition.settings.map((setting) => [setting.id, setting.default])), ...variant?.attributes },
      })
    })
    setSelectedNodeId(id)
  }

  const dropModelAtom = (atomId: string, position: { x: number; y: number }) => {
    const [definitionId, variantId] = atomId.split(':')
    const definition = modelAtomRegistry[definitionId]
    if (!definition) return
    if (definitionId === 'lm-head' && variantId === 'tied') {
      addModelAtom(definition, position, { label: 'Tied language-model head', attributes: { tieEmbeddingWeights: true, bias: false } })
      return
    }
    addModelAtom(definition, position)
  }

  const addGraphInput = (role: TensorRole, desiredPosition?: { x: number; y: number }) => {
    const definition = graphInputDefinitions.find((candidate) => candidate.role === role)
    if (!definition) return
    const baseId = role === 'token-ids' ? 'token-ids' : role === 'labels' ? 'labels' : role === 'image' ? 'image-tensor' : role === 'video' ? 'video-tensor' : role === 'audio' ? 'audio-waveform' : 'hidden-state'
    let sequence = 1
    let id = baseId
    while (graph.nodes.some((node) => node.id === id)) id = `${baseId}-${++sequence}`
    setGraph((current) => {
      const position = desiredPosition ?? findOpenGraphPosition(current)
      return addNode(current, { id, kind: 'input', label: definition.label, role, position })
    })
    setSelectedNodeId(id)
  }

  const addCustomCard = (card: CustomPyTorchCard, desiredPosition?: { x: number; y: number }) => {
    const id = `custom-${card.id}-${customCardSequenceRef.current++}`
    setGraph((current) => {
      const position = desiredPosition ?? findOpenGraphPosition(current)
      const firstOutput = card.graph ? customCardOutputPorts(card.graph)[0]?.tensor : undefined
      return addNode(current, { id, kind: 'custom-pytorch', label: card.label, role: firstOutput ?? card.outputRole ?? 'hidden', position, code: card.code, attributes: { inputRole: card.inputRole ?? 'hidden', customCardId: card.id }, ...(card.graph ? { customCardGraph: structuredClone(card.graph) } : {}) })
    })
    setSelectedNodeId(id)
  }

  const createCustomCard = ({ label, code, inputRole, outputRole, graph: cardGraph }: Omit<CustomPyTorchCard, 'id'>, destination: CustomCardDestination): CustomCardCreateResult => {
    const baseId = label.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'pytorch'
    let id = baseId
    let sequence = 2
    while (customCards.some((card) => card.id === id)) id = `${baseId}-${sequence++}`
    const card: CustomPyTorchCard = { id, label, code, ...(inputRole ? { inputRole } : {}), ...(outputRole ? { outputRole } : {}), ...(cardGraph ? { graph: structuredClone(cardGraph) } : {}) }
    const cardInputs = cardGraph ? customCardInputPorts(cardGraph) : []
    const cardOutputs = cardGraph ? customCardOutputPorts(cardGraph) : []
    const inputTensor = cardInputs[0]?.tensor ?? inputRole ?? 'hidden'
    const outputTensor = cardOutputs[0]?.tensor ?? outputRole ?? 'hidden'
    const current = latestGraphRef.current
    const graphNodeId = (() => {
      const base = `custom-${card.id}`
      let candidate = base
      let suffix = 2
      while (current.nodes.some((node) => node.id === candidate)) candidate = `${base}-${suffix++}`
      return candidate
    })()
    const customNode: ArchitectureNode = { id: graphNodeId, kind: 'custom-pytorch', label: card.label, role: outputTensor, position: findOpenGraphPosition(current), code: card.code, attributes: { inputRole: inputTensor, customCardId: card.id }, ...(cardGraph ? { customCardGraph: structuredClone(cardGraph) } : {}) }

    if (destination === 'selected') {
      const source = current.nodes.find((node) => node.id === selectedNodeId)
      const definition = source?.atomId ? modelAtomRegistry[source.atomId] : undefined
      const output = source?.kind === 'input'
        ? { id: source.role === 'token-ids' ? 'tokenIds' : source.role, tensor: source.role }
        : source?.kind === 'custom-pytorch'
          ? source.customCardGraph ? customCardOutputPorts(source.customCardGraph).find((port) => port.tensor === inputTensor) : { id: 'output', tensor: source.role }
          : definition?.outputs.find((port) => port.tensor === inputTensor)
      if (!source || !output || output.tensor !== inputTensor) return { ok: false, message: `${source?.label ?? 'The selected card'} has no ${inputTensor} output for this card.` }
      const withNode = addNode(current, customNode)
      const connected = connectCable(withNode, { sourceId: source.id, sourcePort: inputTensor, sourcePortId: output.id, targetId: graphNodeId, targetPort: inputTensor, targetPortId: cardInputs[0]?.id ?? 'input' })
      if (!connected.ok) return { ok: false, message: connected.message }
      setGraph(layoutArchitectureGraph(connected.graph, [graphNodeId]))
      setSelectedNodeId(graphNodeId)
    } else if (destination === 'new-architecture') {
      const metadata = {
        laboArchitectureName: `Custom · ${label}`,
        laboArchitectureHiddenSize: current.config.hiddenSize,
        laboArchitectureQueryHeads: current.config.queryHeads,
        laboArchitectureKeyValueHeads: current.config.keyValueHeads,
        laboArchitectureHeadDim: current.config.headDim,
      }
      const externalInputs = cardInputs.length > 0 ? cardInputs : [{ id: 'input', label: `${label} input`, tensor: inputTensor }]
      let withNodes = addNode(current, { ...customNode, attributes: { ...customNode.attributes, ...metadata } })
      const inputIds: string[] = []
      for (const [index, port] of externalInputs.entries()) {
        let inputId = `${graphNodeId}-${port.id}-input`
        let suffix = 2
        while (withNodes.nodes.some((node) => node.id === inputId)) inputId = `${graphNodeId}-${port.id}-input-${suffix++}`
        inputIds.push(inputId)
        withNodes = addNode(withNodes, { id: inputId, kind: 'input', label: port.label, role: port.tensor, position: { ...findOpenGraphPosition(withNodes), x: findOpenGraphPosition(withNodes).x + index * 180 }, attributes: metadata })
        const connected = connectCable(withNodes, { sourceId: inputId, sourcePort: port.tensor, sourcePortId: port.tensor === 'token-ids' ? 'tokenIds' : port.tensor, targetId: graphNodeId, targetPort: port.tensor, targetPortId: port.id })
        if (!connected.ok) return { ok: false, message: connected.message }
        withNodes = connected.graph
      }
      setGraph(layoutParallelArchitecture(withNodes, [...inputIds, graphNodeId]))
      setSelectedNodeId(graphNodeId)
    }
    setCustomCards((current) => [...current, card])
    setCreateCardOpen(false)
    return { ok: true }
  }

  const openCardCreator = () => {
    setInteractionMode('add')
    setCreateCardOpen(true)
  }

  const openCardEditor = (nodeId: string) => {
    const node = graph.nodes.find((candidate) => candidate.id === nodeId)
    if (!node) return
    setSelectedNodeId(node.id)
    setEditingNodeId(node.id)
    setCardEditDraft({ label: node.label, attributes: node.attributes ? { ...node.attributes } : undefined, code: node.code })
    setCardEditError('')
  }

  const closeCardEditor = () => {
    setEditingNodeId(undefined)
    setCardEditDraft(undefined)
    setCardEditError('')
  }

  const saveCardEditor = () => {
    if (!editingNode || !cardEditDraft) return
    const label = cardEditDraft.label.trim()
    if (!label) return setCardEditError('Give the card a name.')
    if (editingNode.kind === 'custom-pytorch' && !editingNode.customCardGraph && !validCustomPyTorchModule(cardEditDraft.code ?? '')) {
      return setCardEditError('Use one safe nn.Module constructor with literal arguments only.')
    }
    setGraph((current) => ({
      ...current,
      nodes: current.nodes.map((node) => node.id === editingNode.id ? {
        ...node,
        label,
        attributes: cardEditDraft.attributes,
        code: cardEditDraft.code?.trim(),
      } : node),
    }))
    closeCardEditor()
  }

  const deleteEditingCard = () => {
    if (!editingNode) return
    const next = removeNode(graph, editingNode.id)
    setGraph(next)
    setSelectedNodeId(next.nodes[0]?.id ?? '')
    closeCardEditor()
    setInteractionMode('edit')
  }

  const deleteGraphCard = (nodeId: string) => {
    const next = removeNode(graph, nodeId)
    setGraph(next)
    setSelectedNodeId(next.nodes[0]?.id ?? '')
    if (editingNodeId === nodeId) closeCardEditor()
  }

  const deleteGraphCards = (nodeIds: string[]) => {
    const removed = new Set(nodeIds)
    let next = graph
    for (const nodeId of removed) next = removeNode(next, nodeId)
    setGraph(next)
    setSelectedNodeId(next.nodes[0]?.id ?? '')
    if (editingNodeId && removed.has(editingNodeId)) closeCardEditor()
  }

  const deleteCustomCardDefinition = (cardId: string) => {
    const legacyPrefix = `custom-${cardId}`
    const removedNodeIds = new Set(graph.nodes.filter((node) => {
      if (node.kind !== 'custom-pytorch') return false
      if (node.attributes?.customCardId === cardId) return true
      if (node.id === legacyPrefix) return true
      const legacySuffix = node.id.startsWith(`${legacyPrefix}-`) ? node.id.slice(legacyPrefix.length + 1) : ''
      return /^\d+$/.test(legacySuffix)
    }).map((node) => node.id))

    setCustomCards((current) => current.filter((card) => card.id !== cardId))
    if (removedNodeIds.size === 0) return
    let next = graph
    for (const nodeId of removedNodeIds) next = removeNode(next, nodeId)
    setGraph(next)
    if (removedNodeIds.has(selectedNodeId)) setSelectedNodeId(next.nodes[0]?.id ?? '')
    if (editingNodeId && removedNodeIds.has(editingNodeId)) closeCardEditor()
  }

  useEffect(() => {
    if (!requestedCard) return
    if (createCardOpen) return
    setInteractionMode('add')
    if (requestedCard.atomId === 'token-ids-input') addGraphInput('token-ids')
    else if (requestedCard.atomId === 'hidden-state-input') addGraphInput('hidden')
    else if (requestedCard.atomId === 'training-labels-input') addGraphInput('labels')
    else {
      const definition = modelAtomRegistry[requestedCard.atomId]
      if (definition && !definition.composite) addModelAtom(definition)
    }
    onRequestedCardHandled()
  // The request id is the deliberate one-shot trigger; graph changes must not replay it.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [requestedCard?.requestId, createCardOpen])

  const loadPreset = (preset: ArchitectureGraph, selectedNodeId: string) => {
    presetDraftsRef.current.set(graph.id, { graph, selectedNodeId: selectedNode?.id ?? selectedGroup?.id ?? '' })
    const draft = presetDraftsRef.current.get(preset.id)
    setGraph(cloneArchitectureGraph(draft?.graph ?? preset))
    setSelectedNodeId(draft?.selectedNodeId ?? selectedNodeId)
    setParseDiagnostics([])
  }

  const saveGraphAsPreset = (requestedName: string, sourceGraph: ArchitectureGraph = graph) => {
    const name = requestedName.trim()
    if (!name) {
      setPresetError('Give the preset a name.')
      return
    }
    const baseId = `user-${name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'model'}`
    const usedIds = new Set([...builtInModelPresets, ...userPresets].map((preset) => preset.id))
    let id = baseId
    let sequence = 2
    while (usedIds.has(id)) id = `${baseId}-${sequence++}`
    const preset = cloneArchitectureGraph({ ...sourceGraph, id, name })
    if (sourceGraph.id === blankStarterPreset.id) {
      presetDraftsRef.current.set(blankStarterPreset.id, { graph: cloneArchitectureGraph(blankStarterPreset), selectedNodeId: '' })
    }
    presetDraftsRef.current.set(id, { graph: preset, selectedNodeId })
    setUserPresets((current) => [...current, preset])
    setGraph(preset)
    setPresetError('')
    return preset
  }


  const createUserPreset = () => { saveGraphAsPreset(presetName) }

  const createBlankWorkspace = () => {
    let sequence = 1
    const usedIds = new Set([...builtInModelPresets, ...userPresets].map((preset) => preset.id))
    while (usedIds.has(`user-blank-${sequence}`)) sequence += 1
    const blank = cloneArchitectureGraph({ ...blankStarterPreset, id: `user-blank-${sequence}`, name: `Blank canvas ${sequence}` })
    presetDraftsRef.current.set(blank.id, { graph: blank, selectedNodeId: '' })
    setUserPresets((current) => [...current, blank])
    setGraph(blank)
    setSelectedNodeId('')
    setInteractionMode('add')
  }

  const resetCurrentPreset = () => {
    const original = builtInModelPresets.find((preset) => preset.id === graph.id) ?? userPresets.find((preset) => preset.id === graph.id)
    if (!original) return
    if (!confirmPresetReset) {
      setConfirmPresetReset(true)
      return
    }
    const reset = cloneArchitectureGraph(original)
    presetDraftsRef.current.set(reset.id, { graph: reset, selectedNodeId: reset.nodes[0]?.id ?? '' })
    setGraph(reset)
    setSelectedNodeId(reset.nodes[0]?.id ?? '')
    setParseDiagnostics([])
    setConfirmPresetReset(false)
  }

  const deleteUserPreset = (preset: ArchitectureGraph) => {
    presetDraftsRef.current.delete(preset.id)
    setUserPresets((current) => current.filter((candidate) => candidate.id !== preset.id))
    if (graph.id === preset.id) {
      const blankDraft = presetDraftsRef.current.get(blankStarterPreset.id)
      setGraph(cloneArchitectureGraph(blankDraft?.graph ?? blankStarterPreset))
      setSelectedNodeId(blankDraft?.selectedNodeId ?? '')
      setParseDiagnostics([])
    }
  }

  const selectPreset = (presetId: string) => {
    const preset = [...builtInModelPresets, ...userPresets].find((candidate) => candidate.id === presetId)
    if (preset) loadPreset(preset, preset.nodes[0]?.id ?? '')
  }

  const addPresetForComparison = (preset: ArchitectureGraph) => {
    const current = latestGraphRef.current
    const usedIds = new Set(current.nodes.map((node) => node.id))
    const sourceArchitectures = architectureComponents(preset)
    const addedNodeIds: string[] = []
    let nextGraph = current
    for (const [componentIndex, architecture] of sourceArchitectures.entries()) {
      let sequence = 1
      let prefix = `${preset.id}-${componentIndex + 1}`
      while (architecture.nodeIds.some((nodeId) => usedIds.has(`${prefix}-${nodeId}`))) prefix = `${preset.id}-${componentIndex + 1}-${++sequence}`
      const remap = new Map(architecture.nodeIds.map((nodeId) => [nodeId, `${prefix}-${nodeId}`]))
      const metadata = {
        laboArchitectureName: architecture.label,
        laboArchitectureHiddenSize: architecture.graph.config.hiddenSize,
        laboArchitectureQueryHeads: architecture.graph.config.queryHeads,
        laboArchitectureKeyValueHeads: architecture.graph.config.keyValueHeads,
        laboArchitectureHeadDim: architecture.graph.config.headDim,
      }
      const nodes = architecture.graph.nodes.map((node) => ({ ...node, id: remap.get(node.id)!, position: { ...node.position }, attributes: { ...node.attributes, ...metadata } }))
      const edges = architecture.graph.edges.map((edge) => ({ ...edge, id: `${prefix}-${edge.id}`, source: remap.get(edge.source)!, target: remap.get(edge.target)! }))
      for (const node of nodes) { usedIds.add(node.id); addedNodeIds.push(node.id) }
      nextGraph = { ...nextGraph, nodes: [...nextGraph.nodes, ...nodes], edges: [...nextGraph.edges, ...edges] }
    }
    nextGraph = layoutParallelArchitecture(nextGraph, addedNodeIds)
    setGraph(nextGraph)
    setSelectedNodeId(addedNodeIds[0] ?? '')
    setInteractionMode('edit')
  }

  const applyAgentGraph = (nextGraph: ArchitectureGraph, actions: AgentGraphAction[]) => {
    const addedNode = nextGraph.nodes.find((node) => !graph.nodes.some((current) => current.id === node.id))
    const generatedCards = nextGraph.nodes.filter((node) => !graph.nodes.some((current) => current.id === node.id) && node.kind === 'custom-pytorch' && node.code && validCustomPyTorchModule(node.code))
    if (generatedCards.length > 0) {
      setCustomCards((current) => {
        const next = [...current]
        for (const node of generatedCards) {
          const inputRole = (node.attributes?.inputRole as TensorRole | undefined) ?? 'hidden'
          const outputRole = node.role
          if (next.some((card) => card.label === node.label && card.code === node.code && (card.inputRole ?? 'hidden') === inputRole && (card.outputRole ?? 'hidden') === outputRole)) continue
          const baseId = node.id.replace(/[^A-Za-z0-9-]+/g, '-').replace(/^-+|-+$/g, '') || 'agent-card'
          let id = baseId
          let sequence = 2
          while (next.some((card) => card.id === id)) id = `${baseId}-${sequence++}`
          next.push({ id, label: node.label, code: node.code!, inputRole, outputRole })
        }
        return next
      })
    }
    const run = actions.find((action): action is Extract<AgentGraphAction, { type: 'run' }> => action.type === 'run')
    if (run) pendingAgentRunRef.current = run.mode
    const preset = actions.find((action): action is Extract<AgentGraphAction, { type: 'save-preset' }> => action.type === 'save-preset')
    const appliedGraph = preset ? saveGraphAsPreset(preset.name, nextGraph) ?? nextGraph : nextGraph
    if (!preset) setGraph(appliedGraph)
    if (addedNode) setSelectedNodeId(addedNode.id)
    for (const action of actions) {
      if (action.type !== 'export') continue
      if (action.kind === 'svg' || action.kind === 'both') void exportArchitectureDiagram(appliedGraph)
      if (action.kind === 'python' || action.kind === 'both') {
        const architectures = architectureComponents(appliedGraph, [...builtInModelPresets, ...userPresets])
        for (const architecture of architectures) void exportPyTorchCode(architecture.graph, compileToPyTorch(architecture.graph))
      }
    }
  }

  const modelAtoms = Object.values(modelAtomRegistry)
  const routingAtom = (definition: ModelAtomDefinition) => definition.category === 'routing' || definition.id === 'load-balancing-loss'
  const activationAtoms = modelAtoms.filter((definition) => definition.category === 'activation')
  const objectiveAtoms = modelAtoms.filter((definition) => definition.category === 'objective' && definition.id !== 'load-balancing-loss' && definition.id !== 'router-entropy-loss')
  const genericAtomIds = new Set(['token-embedding', 'qkv-projection', 'attention-head-layout', 'causal-sdpa', 'merge-attention-heads', 'attention-output-projection', 'residual-add', 'linear-projection', 'dropout', 'scale', 'hadamard-product', 'identity', 'lm-head'])
  const genericAtoms = modelAtoms.filter((definition) => genericAtomIds.has(definition.id))
  const specializedAtom = (definition: ModelAtomDefinition) => !genericAtomIds.has(definition.id) && definition.category !== 'routing' && definition.category !== 'objective' && definition.category !== 'activation' && !definition.composite
  const mediaAtoms = modelAtoms.filter((definition) => specializedAtom(definition) && definition.category === 'media')
  const mediaFamilyRules: Array<{ label: string; matches: (id: string) => boolean }> = [
    {
      label: 'Image inputs & tokenization',
      matches: (id) => /^(image-(channel-normalization|resize|patch-embedding|vq-tokenizer|codebook-embedding)|global-image-embedding)$/.test(id),
    },
    {
      label: 'Video inputs & tokenization',
      matches: (id) => /^(video-(channel-normalization|spatial-resize|tubelet-embedding|vq-tokenizer|codebook-embedding))$/.test(id),
    },
    {
      label: 'Audio inputs & tokenization',
      matches: (id) => /^audio-(waveform-normalization|preemphasis|resample|frame-embedding|vq-tokenizer|codebook-embedding)$/.test(id),
    },
    {
      label: 'Audio processing & outputs',
      matches: (id) => /^audio-/.test(id),
    },
    {
      label: 'Media generation & outputs',
      matches: (id) => /(decoder|reconstruction|denoiser|diffusion|classifier-free-guidance)/.test(id),
    },
    {
      label: 'Vision & spatial processing',
      matches: (id) => /^(vision-|image-|patch-|spatial-|global-patch-|masked-patch-)/.test(id),
    },
    {
      label: 'Video & temporal processing',
      matches: (id) => /^(video-|tubelet-|temporal-|frame-)/.test(id),
    },
    { label: 'Multimodal fusion & conditioning', matches: () => true },
  ]
  const assignedMediaAtoms = new Set<string>()
  const mediaFamilies = mediaFamilyRules.map((family) => ({
    label: family.label,
    atoms: mediaAtoms.filter((definition) => {
      if (assignedMediaAtoms.has(definition.id) || !family.matches(definition.id)) return false
      assignedMediaAtoms.add(definition.id)
      return true
    }),
  }))
  const libraryFamilies = [
    { label: 'Embeddings', atoms: modelAtoms.filter((definition) => specializedAtom(definition) && definition.category === 'embedding') },
    { label: 'Normalization', atoms: modelAtoms.filter((definition) => specializedAtom(definition) && definition.category === 'normalization') },
    { label: 'Attention', atoms: modelAtoms.filter((definition) => specializedAtom(definition) && definition.category === 'attention') },
    { label: 'Position & sequence', atoms: modelAtoms.filter((definition) => specializedAtom(definition) && definition.category === 'position') },
    { label: 'Tensor composition', atoms: modelAtoms.filter((definition) => specializedAtom(definition) && definition.category === 'composition') },
    { label: 'MLP blocks', atoms: modelAtoms.filter((definition) => specializedAtom(definition) && definition.category === 'mlp') },
    { label: 'Output heads', atoms: modelAtoms.filter((definition) => specializedAtom(definition) && definition.category === 'output') },
    ...mediaFamilies,
  ].filter((family) => family.atoms.length > 0)
  const trBasicIds = new Set(['deterministic-token-routing', 'routed-expert-bank', 'shared-expert-bank', 'branch-gated-merge'])
  const learnedRouterIds = new Set(['moe-router', 'top-k-routing', 'load-balancing-loss', 'router-entropy-loss'])
  const trBasicAtoms = modelAtoms.filter((definition) => trBasicIds.has(definition.id))
  const learnedRouterAtoms = modelAtoms.filter((definition) => learnedRouterIds.has(definition.id))
  const routingRecipeAtoms = modelAtoms.filter((definition) => routingAtom(definition) && !trBasicIds.has(definition.id) && !learnedRouterIds.has(definition.id))
  const atomButton = (definition: ModelAtomDefinition) => <button
    aria-label={`Add ${definition.label}`}
    className="library-block"
    disabled={interactionMode === 'edit'}
    draggable={interactionMode === 'add'}
    key={definition.id}
    onDragEnd={(event) => event.currentTarget.blur()}
    onDragStart={(event) => {
      event.dataTransfer.effectAllowed = 'copy'
      event.dataTransfer.setData('application/x-labo-model-atom', definition.id)
      event.dataTransfer.setData('text/plain', definition.label)
      setLibraryDragPreview(event.dataTransfer, definition.label, `glyph-${definition.category}`)
    }}
    onClick={() => addModelAtom(definition)}
    title="Click to add automatically, or drag onto the graph"
  >
    <span className={`block-glyph glyph-${definition.category}`} />
    {definition.label}
  </button>
  const tiedLmHeadButton = <button
    aria-label="Add Tied language-model head"
    className="library-block"
    disabled={interactionMode === 'edit'}
    draggable={interactionMode === 'add'}
    onClick={() => addModelAtom(modelAtomRegistry['lm-head'], undefined, { label: 'Tied language-model head', attributes: { tieEmbeddingWeights: true, bias: false } })}
    onDragEnd={(event) => event.currentTarget.blur()}
    onDragStart={(event) => {
      event.dataTransfer.effectAllowed = 'copy'
      event.dataTransfer.setData('application/x-labo-model-atom', 'lm-head:tied')
      event.dataTransfer.setData('text/plain', 'Tied language-model head')
      setLibraryDragPreview(event.dataTransfer, 'Tied language-model head', 'glyph-output')
    }}
    title="Click to add automatically, or drag onto the graph"
  >
    <span className="block-glyph glyph-output" />
    Tied language-model head
  </button>
  const activeAgentSelection = agentSelectedNodeIds.length > 0 ? agentSelectedNodeIds : selectedNodeId ? [selectedNodeId] : []
  const askLaboPanel = <AskLaboPanel customCards={customCards} dockClassName={`view-${view} ${libraryOpen ? 'library-visible' : ''} ${inspectorOpen ? 'inspector-visible' : ''}`} graph={graph} interactionMode={interactionMode} selectedNodeIds={activeAgentSelection} onApply={applyAgentGraph} onClose={onCloseAsk} open={askOpen} workspaceSettings={<WorkspaceSettingsContent comparisonPresets={[...builtInModelPresets.filter((preset) => preset.nodes.length > 0), ...userPresets.filter((preset) => preset.nodes.length > 0)]} currentGraph={graph} currentLabel={presetMenuLabels[graph.id] ?? graph.name} error={presetError} name={presetName} onAddComparison={addPresetForComparison} onCreateBlank={createBlankWorkspace} onDeleteWorkspace={deleteUserPreset} onLoadWorkspace={(preset) => loadPreset(preset, preset.nodes[0]?.id ?? '')} onNameChange={setPresetName} onReset={resetCurrentPreset} onSave={createUserPreset} presetLabel={(preset) => presetMenuLabels[preset.id] ?? preset.name} resetConfirming={confirmPresetReset} resetDisabled={!builtInModelPresets.some((preset) => preset.id === graph.id) && !userPresets.some((preset) => preset.id === graph.id)} savedWorkspaces={userPresets} />} />

  return (
    <>
      <StudioToolbar meta={<>
          <span><span className={`status-dot ${createCardOpen || pytorchDraftAvailable || blankGraph ? '' : 'invalid'}`} /> {createCardOpen ? 'Reusable card workspace' : blankGraph ? 'Blank canvas ready' : pytorchMappingComplete ? 'PyTorch graph executable' : pytorchDraftAvailable ? 'Atomic PyTorch draft' : 'PyTorch compile error'}</span>
          <span>{createCardOpen ? cardPlayerSnapshot.results.length : stats.nodeCount} atoms</span>
          <ModelPlayerControls blankGraph={createCardOpen ? cardPlayerSnapshot.results.length === 0 : blankGraph} nativePyTorchRuntime={nativePyTorchRuntime} onArrange={() => createCardOpen ? cardPlayerRef.current?.arrange() : setGraph((current) => layoutArchitectureGraph(current))} onPause={() => createCardOpen ? cardPlayerRef.current?.pause() : modelPlayerRef.current?.pause()} onPlay={() => createCardOpen ? void cardPlayerRef.current?.play() : void modelPlayerRef.current?.play()} onStep={() => createCardOpen ? void cardPlayerRef.current?.step() : void modelPlayerRef.current?.step()} onStop={() => createCardOpen ? cardPlayerRef.current?.stop() : modelPlayerRef.current?.stop()} runtimeAvailable={createCardOpen ? cardPlayerSnapshot.results.length > 0 : runtimeAvailable} scope={createCardOpen ? 'reusable card' : 'model'} snapshot={createCardOpen ? cardPlayerSnapshot : modelPlayerSnapshot} />
          <ModelPanelControls inspectorOpen={inspectorOpen} libraryOpen={libraryOpen} onInspectorToggle={() => setInspectorOpen((current) => !current)} onLibraryToggle={() => setLibraryOpen((current) => !current)} />
          <ExportMenu code={code} codeGraph={codeGraph} graph={graph} onDiagram={createCardOpen ? () => cardPlayerRef.current?.exportDiagram() : undefined} onPyTorch={createCardOpen ? () => cardPlayerRef.current?.exportPyTorch() : undefined} />
        </>}>
          <StudioViewSwitcher<ViewMode> ariaLabel="Editor view" onChange={setView} options={[{ id: 'blocks', label: 'Blocks', icon: <Blocks size={14} /> }, { id: 'pytorch', label: 'PyTorch', icon: <Braces size={14} /> }, { id: 'split', label: 'Split', icon: <SplitSquareHorizontal size={14} /> }]} value={view} />
          <ModelInteractionSwitcher createCardOpen={createCardOpen} interactionMode={interactionMode} onCreateCard={() => { if (!createCardOpen) openCardCreator() }} onInteractionMode={(mode) => { setInteractionMode(mode); setCreateCardOpen(false) }} />
          {!createCardOpen && <ModelPresetMenu activeId={graph.id} activeName={graph.name} builtIns={builtInModelPresets} labels={presetMenuLabels} onSelect={selectPreset} userPresets={userPresets} />}
          {!createCardOpen && <ModelPromptMenu acceptsTokenIds={acceptsTokenIds} onChange={(value) => { setSampleText(value); setPromptTokenCount(undefined); setModelOutput(undefined) }} promptTokenCount={promptTokenCount} value={sampleText} />}
      </StudioToolbar>

      {createCardOpen ? <Suspense fallback={null}><CustomCardCreator editMode={interactionMode === 'edit'} inspectorOpen={inspectorOpen} libraryOpen={libraryOpen} onClose={() => setCreateCardOpen(false)} onCreate={createCustomCard} onPlayerSnapshotChange={setCardPlayerSnapshot} onRequestedCardHandled={onRequestedCardHandled} ref={cardPlayerRef} requestedCard={requestedCard} selectedTarget={selectedNode?.label} view={view} /></Suspense> : <>
      <StudioWorkspace inspectorOpen={inspectorOpen} libraryOpen={libraryOpen}>
        {libraryOpen && <StudioLibrary className={`mode-${interactionMode}`} heading="BLOCK LIBRARY" icon={<PanelLeft size={14} />}>
          <section className="block-group">
            <StudioLibrarySection className="graph-input-family" count={graphInputDefinitions.length} label="Graph inputs">
              {graphInputDefinitions.map((definition) => <button
                aria-label={`Add ${definition.label}`}
                className="library-block"
                disabled={interactionMode === 'edit'}
                draggable={interactionMode === 'add'}
                key={definition.role}
                onClick={() => addGraphInput(definition.role)}
                onDragEnd={(event) => event.currentTarget.blur()}
                onDragStart={(event) => {
                  event.dataTransfer.effectAllowed = 'copy'
                  event.dataTransfer.setData('application/x-labo-graph-input', definition.role)
                  event.dataTransfer.setData('text/plain', definition.label)
                  setLibraryDragPreview(event.dataTransfer, definition.label, 'glyph-input')
                }}
                title="Click to add automatically, or drag onto the graph"
              >
                <span className="block-glyph glyph-input" />
                {definition.label}
              </button>)}
            </StudioLibrarySection>
            <StudioLibrarySection className="custom-card-family" count={customCards.length} label="My cards">
              {customCards.length > 0 && <div className="custom-card-list">
                {customCards.map((card) => <div className="custom-card-list-row" key={card.id}><button
                  aria-label={`Add ${card.label}`}
                  className="library-block custom-library-block"
                  disabled={interactionMode === 'edit'}
                  draggable={interactionMode === 'add'}
                  onClick={() => addCustomCard(card)}
                  onDragEnd={(event) => event.currentTarget.blur()}
                  onDragStart={(event) => {
                    event.dataTransfer.effectAllowed = 'copy'
                    event.dataTransfer.setData('application/x-labo-custom-card', card.id)
                    event.dataTransfer.setData('text/plain', card.label)
                    setLibraryDragPreview(event.dataTransfer, card.label, 'glyph-custom')
                  }}
                  title={`${card.code} · Click or drag onto the graph`}
                >
                  <span className="block-glyph glyph-custom" />
                  <span><strong>{card.label}</strong><code>{card.code}</code></span>
                </button><button aria-label={`Delete custom card ${card.label}`} className="delete-custom-library-card" onClick={() => deleteCustomCardDefinition(card.id)} title="Delete card from library"><Trash2 size={12} /></button></div>)}
              </div>}
            </StudioLibrarySection>
            <StudioLibrarySection className="generic-atomics-family" count={genericAtoms.length} label="Generic atomics">
              {genericAtoms.map(atomButton)}
            </StudioLibrarySection>
            {libraryFamilies.map((family) => <StudioLibrarySection className="catalog-family" count={family.atoms.length} key={family.label} label={family.label}>
              {family.atoms.map(atomButton)}
            </StudioLibrarySection>)}
            <StudioLibrarySection className="catalog-family" count={1} label="Tied output head">
              {tiedLmHeadButton}
            </StudioLibrarySection>
            <StudioLibrarySection count={objectiveAtoms.length} label="Training objectives">
              {objectiveAtoms.map(atomButton)}
            </StudioLibrarySection>
            <StudioLibrarySection count={trBasicAtoms.length} label="Token-Routed MLP">
              {trBasicAtoms.map(atomButton)}
            </StudioLibrarySection>
            <StudioLibrarySection count={learnedRouterAtoms.length} label="Learned Router Controls">
              {learnedRouterAtoms.map(atomButton)}
            </StudioLibrarySection>
            <StudioLibrarySection count={routingRecipeAtoms.length} label="Routing Recipes">
              {routingRecipeAtoms.map(atomButton)}
            </StudioLibrarySection>
            <StudioLibrarySection count={activationAtoms.length} label="Activations">
              {activationAtoms.map(atomButton)}
            </StudioLibrarySection>
          </section>
        </StudioLibrary>}

        <StudioEditor className={`view-${view}`}>
          {view !== 'pytorch' && <GraphCanvas editMode={interactionMode === 'edit'} graph={graph} onDeleteNode={deleteGraphCard} onDeleteNodes={deleteGraphCards} onEditNode={openCardEditor} onSelectionChange={setAgentSelectedNodeIds} onDropAtom={dropModelAtom} onDropCustom={(cardId, position) => {
            const card = customCards.find((candidate) => candidate.id === cardId)
            if (card) addCustomCard(card, position)
          }} onDropInput={(role, position) => addGraphInput(role, position)} playerSnapshot={modelPlayerSnapshot} selectedNodeId={selectedNodeId} setGraph={setGraph} setSelectedNodeId={setSelectedNodeId} />}

          {view !== 'blocks' && (
            <StudioCodePanel tab={<StudioPanelTab actions={<button aria-label="Apply PyTorch to blocks" onClick={applyPyTorch}>Apply to blocks</button>} icon={<Code2 size={13} />} status="LABO DIALECT">generated_attention.py {graphArchitectures.length > 1 && <select aria-label="PyTorch architecture" onChange={(event) => { const architecture = graphArchitectures.find((candidate) => candidate.id === event.target.value); setSelectedArchitectureId(event.target.value); if (architecture?.nodeIds[0]) setSelectedNodeId(architecture.nodeIds[0]) }} value={selectedArchitecture?.id ?? ''}>{graphArchitectures.map((architecture) => <option key={architecture.id} value={architecture.id}>{architecture.label}</option>)}</select>}</StudioPanelTab>}>
              {blankGraph ? <div className="code-empty-state">
                <span><Code2 size={20} /></span>
                <strong>PyTorch appears with your graph</strong>
                <p>Add the first card to generate an inspectable module. Nothing invalid is emitted for an empty workspace.</p>
                <code>graph → typed IR → PyTorch</code>
              </div> : <PythonCodeEditor onChange={setCodeDraft} value={codeDraft} />}
              {!blankGraph && parseDiagnostics.length > 0 && <div className="code-diagnostics">{parseDiagnostics.map((diagnostic) => <p key={`${diagnostic.nodeId}-${diagnostic.code}`}>{diagnostic.message}</p>)}</div>}
            </StudioCodePanel>
          )}
        </StudioEditor>

        <StudioInspector heading="INSPECTOR" hidden={!inspectorOpen} icon={<Cpu size={14} />}>
          <InspectorSection title="Selection">
            <InspectorSelection detail={selectedNode?.id ?? selectedGroup?.id ?? '—'} icon={<Zap size={15} />} title={selectedNode?.label ?? selectedGroup?.label ?? 'No selection'} />
            {blankGraph && <p className="blank-graph-hint">Add an atomic block from the library or ask LABO to build a starter graph.</p>}
            {!blankGraph && !validation.valid && <p className="graph-incomplete-hint" title={validation.errors.join('\n')}>Graph incomplete · {validation.errors.length} wiring issue{validation.errors.length === 1 ? '' : 's'}. Connect the open ports before running.</p>}
          </InspectorSection>
          <section className="equivalence-card">
            <div className="equivalence-title"><Play size={14} /> {nativePyTorchRuntime ? 'Atomic PyTorch execution' : 'Atomic graph preview'}</div>
            <InspectorMetric label="Player" value={modelPlayerSnapshot.status} />
            <InspectorMetric label="Current level" value={modelPlayerSnapshot.currentAtomIds?.join(' + ') ?? '—'} />
            {selectedNode && <InspectorMetric label="Selected result" value={modelPlayerSnapshot.results.find((result) => result.atomId === selectedNode.id)?.status ?? 'pending'} />}
            {modelPlayerSnapshot.error && <p className="execution-error">{modelPlayerSnapshot.error}</p>}
            <div aria-label="Model generation output" className="model-runtime-output">
              <div><span>Output</span><b>{modelOutput ? modelOutput.kind : blankGraph ? 'waiting for blocks' : !nativePyTorchRuntime && modelPlayerSnapshot.status === 'completed' ? 'graph trace' : 'pending'}</b></div>
              {modelOutput && <div><span>Tensor</span><b>[{modelOutput.tensorShape.join(', ')}]</b></div>}
              {modelOutput?.predictedTokenId !== undefined && <div><span>Predicted Token ID</span><b>{modelOutput.predictedTokenId}</b></div>}
              {modelOutput?.topTokenIds && <div><span>Top 5</span><code>{modelOutput.topTokenIds.map((tokenId, index) => `${tokenId} (${(((modelOutput.topProbabilities?.[index]) ?? 0) * 100).toFixed(2)}%)`).join(' · ')}</code></div>}
            </div>
          </section>
        </StudioInspector>
      </StudioWorkspace>

      {editingNode && cardEditDraft && <div className="model-card-modal-backdrop" onPointerDown={(event) => { if (event.target === event.currentTarget) closeCardEditor() }}>
        <section aria-label="Edit model card" aria-modal="true" className={`model-card-modal ${editingNode.kind === 'custom-pytorch' ? 'code-card-modal' : ''}`} onPointerDown={(event) => event.stopPropagation()} role="dialog">
          <header><div><span>EDIT MODE</span><strong>{editingNode.label}</strong></div><button aria-label="Close model card editor" onClick={closeCardEditor}><X size={14} /></button></header>
          <p className="model-card-modal-hint">Edit this existing graph card. No new Blockly card is added in this mode.</p>
          <label><span>Name</span><input aria-label="Model card name" onChange={(event) => setCardEditDraft((current) => current ? { ...current, label: event.target.value } : current)} value={cardEditDraft.label} /></label>
          <label><span>Block ID</span><input aria-label="Model card ID" disabled value={editingNode.id} /></label>
          {editingNode.kind === 'custom-pytorch' ? editingNode.customCardGraph
            ? <div className="model-card-code-field"><span>Generated composite PyTorch</span><PythonCodePreview className="compact-python-editor model-card-python-editor" value={cardEditDraft.code ?? ''} /><small>Composite topology is preserved as a reusable internal graph.</small></div>
            : <label className="model-card-code-field"><span>PyTorch module</span><PythonCodeEditor ariaLabel="Model card PyTorch module" className="compact-python-editor model-card-python-editor" onChange={(value) => setCardEditDraft((current) => current ? { ...current, code: value } : current)} value={cardEditDraft.code ?? ''} /></label> : <div className="model-card-modal-settings">
            {modelAtomRegistry[editingNode.atomId ?? '']?.settings.map((setting) => {
              const value = cardEditDraft.attributes?.[setting.id] ?? setting.default
              return <label key={setting.id}><span>{setting.id}</span>{setting.type === 'boolean'
                ? <input aria-label={`Model card setting ${setting.id}`} checked={Boolean(value)} onChange={(event) => setCardEditDraft((current) => current ? { ...current, attributes: { ...current.attributes, [setting.id]: event.target.checked } } : current)} type="checkbox" />
                : setting.type === 'select'
                  ? <select aria-label={`Model card setting ${setting.id}`} onChange={(event) => setCardEditDraft((current) => current ? { ...current, attributes: { ...current.attributes, [setting.id]: event.target.value } } : current)} value={String(value)}>{setting.options?.map((option) => <option key={option}>{option}</option>)}</select>
                  : <input aria-label={`Model card setting ${setting.id}`} onChange={(event) => setCardEditDraft((current) => current ? { ...current, attributes: { ...current.attributes, [setting.id]: setting.type === 'number' ? Number(event.target.value) : event.target.value } } : current)} type={setting.type === 'number' ? 'number' : 'text'} value={String(value)} />}</label>
            })}
          </div>}
          {editingNode.kind === 'custom-pytorch' && <small className={editingNode.customCardGraph || validCustomPyTorchModule(cardEditDraft.code ?? '') ? 'custom-code-valid' : 'custom-code-invalid'}>{editingNode.customCardGraph ? 'Valid reusable composite graph' : validCustomPyTorchModule(cardEditDraft.code ?? '') ? 'Valid safe nn.Module constructor' : 'Invalid or unsupported nn.Module constructor'}</small>}
          {cardEditError && <p className="model-card-modal-error" role="alert">{cardEditError}</p>}
          <footer><button className="model-card-delete" onClick={deleteEditingCard}><Trash2 size={12} />Delete card</button><span /><button onClick={closeCardEditor}>Cancel</button><button className="model-card-save" onClick={saveCardEditor}>Save changes</button></footer>
        </section>
      </div>}

      <StudioStatusbar className="model-statusbar">
      {askLaboPanel}

        <span><span className={`status-dot ${validation.valid || blankGraph ? '' : 'invalid'}`} /> Neural IR {blankGraph ? 'blank' : validation.valid ? 'valid' : 'invalid'}</span>
        <span>{stats.nodeCount} nodes · {stats.edgeCount} links</span>
        <span className="status-spacer" />
        <span>PyTorch 2.7</span>
        <span>LABO Runtime · local</span>
      </StudioStatusbar>
      </>}
      {createCardOpen && askOpen && askLaboPanel}
    </>
  )
}
