import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react'
import { Code2, Cpu, PanelLeft, Plus, Sparkles, Trash2, X, Zap } from 'lucide-react'
import type { ArchitectureGraph, TensorRole } from '../core/ir'
import { addNode, removeNode, updateNodeAttributes, validateGraph } from '../core/ir'
import { compileRegistryGraph } from '../core/pytorch-compiler'
import { customCardInputPorts, customCardOutputPorts, validateCustomCardGraph } from '../core/custom-card-graph'
import { modelAtomRegistry, type ModelAtomCategory, type ModelAtomDefinition } from '../core/model-atoms'
import { createAgentGraphContext, previewAgentGraphPlan, type AgentGraphPlan, type AgentGraphPreview } from '../core/agentic-graph'
import { findOpenGraphPosition, layoutArchitectureGraph } from '../core/graph-placement'
import { GraphCanvas } from './GraphCanvas'
import { PythonCodePreview } from './PythonCodeEditor'
import type { CustomPyTorchCard } from './custom-card'
import { StudioEditor, StudioInspector, StudioLibrary, StudioStatusbar, StudioWorkspace } from '../studio/StudioShell'
import { StudioChoiceMenu } from '../studio/StudioChoiceMenu'
import { AgentPrompt } from '../studio/AgentPrompt'
import { AgentActivityPanel, type AgentActivityItem } from '../studio/AgentActivityPanel'
import { setLibraryDragPreview } from '../studio/libraryDragPreview'
import { AtomicPlayer, type AtomicPlayerSnapshot } from '../core/atomic-player'
import { executionLayers } from '../core/execution-plan'
import { previewModelAtom } from '../core/browser-atomic-preview'
import { exportArchitectureDiagram, exportPyTorchCode } from './export-actions'
import { useLaboLanguage } from '../studio/application-language'
import { AgentPlanReview } from './AgentPlanReview'
import { agentPlanReviewText } from './agent-plan-review-text'

type CardDraft = Omit<CustomPyTorchCard, 'id'>
export type CustomCardDestination = 'library' | 'selected' | 'new-architecture'
export interface CustomCardCreateResult { ok: boolean; message?: string }

export interface CustomCardCreatorHandle {
  arrange(): void
  exportDiagram(): unknown
  exportPyTorch(): unknown
  pause(): void
  play(): Promise<void>
  step(): Promise<void>
  stop(): void
}
const inputChoices: Array<{ role: TensorRole; label: string }> = [
  { role: 'hidden', label: 'Hidden state' }, { role: 'token-ids', label: 'Token IDs' },
  { role: 'image', label: 'Image tensor' }, { role: 'video', label: 'Video tensor' },
  { role: 'audio', label: 'Audio tensor' }, { role: 'labels', label: 'Training labels' },
]
const paletteCategories: Array<{ id: ModelAtomCategory | 'all'; label: string }> = [
  { id: 'all', label: 'All' }, { id: 'composition', label: 'Composition' },
  { id: 'normalization', label: 'Normalization' }, { id: 'activation', label: 'Activation' },
  { id: 'mlp', label: 'MLP' }, { id: 'attention', label: 'Attention' },
  { id: 'embedding', label: 'Embedding' }, { id: 'media', label: 'Media' },
  { id: 'routing', label: 'Routing' }, { id: 'output', label: 'Output' },
]

function roleFor(definition: ModelAtomDefinition): TensorRole {
  const tensor = definition.outputs[0]?.tensor
  if (tensor === 'query' || tensor === 'key' || tensor === 'value' || tensor === 'image' || tensor === 'video' || tensor === 'audio') return tensor
  if (tensor === 'logits' || tensor === 'scalar') return 'output'
  return 'hidden'
}

function uniqueNodeId(graph: ArchitectureGraph, base: string): string {
  let id = base
  let sequence = 2
  while (graph.nodes.some((node) => node.id === id)) id = `${base}-${sequence++}`
  return id
}

function draftGraph(definitionId = 'linear-projection'): ArchitectureGraph {
  const definition = modelAtomRegistry[definitionId] ?? modelAtomRegistry['linear-projection']
  const inputRole = definition.inputs[0]?.tensor ?? 'hidden'
  const inputId = inputRole === 'token-ids' ? 'token-ids' : `${inputRole}-input`
  const atomId = definition.id
  return {
    id: 'custom-card-draft', name: 'Reusable card', architecture: 'custom',
    config: { hiddenSize: 768, queryHeads: 12, keyValueHeads: 4, headDim: 64 },
    contracts: { causal: false, preservesGqaAtZeroGate: false, sdpaCompatible: false, contextualValue: true },
    nodes: [
      { id: inputId, kind: 'input', label: inputChoices.find((item) => item.role === inputRole)?.label ?? `${inputRole} input`, role: inputRole, position: { x: 90, y: 150 } },
      { id: atomId, kind: 'semantic', atomId, label: definition.label, role: roleFor(definition), position: { x: 330, y: 150 }, attributes: Object.fromEntries(definition.settings.map((setting) => [setting.id, setting.default])) },
    ],
    edges: [{ id: `${inputId}-${atomId}`, source: inputId, sourcePort: inputRole === 'token-ids' ? 'tokenIds' : inputRole, target: atomId, targetPort: definition.inputs[0]?.id ?? inputRole, label: inputRole.toUpperCase() }],
  }
}

export type CardBuilderView = 'blocks' | 'pytorch' | 'split'

interface PendingCardAgentPlan {
  composedGraph: ArchitectureGraph
  plan: AgentGraphPlan
  preview: AgentGraphPreview
  suggestedName: string
}

function blankDraftGraph(): ArchitectureGraph {
  const seed = draftGraph()
  return { ...seed, nodes: [], edges: [] }
}

export const CustomCardCreator = forwardRef<CustomCardCreatorHandle, { editMode: boolean; inspectorOpen: boolean; libraryOpen: boolean; onClose(): void; onCreate(card: CardDraft, destination: CustomCardDestination): CustomCardCreateResult; onPlayerSnapshotChange?(snapshot: AtomicPlayerSnapshot): void; onRequestedCardHandled?(): void; requestedCard?: { atomId: string; requestId: number }; selectedTarget?: string; view: CardBuilderView }>(function CustomCardCreator({ editMode, inspectorOpen, libraryOpen, onClose, onCreate, onPlayerSnapshotChange, onRequestedCardHandled = () => undefined, requestedCard, selectedTarget, view }, ref) {
  const language = useLaboLanguage()
  const [name, setName] = useState('My reusable card')
  const [need, setNeed] = useState('')
  const [graph, setGraph] = useState<ArchitectureGraph>(() => blankDraftGraph())
  const [selectedNodeId, setSelectedNodeId] = useState('')
  const [paletteCategory, setPaletteCategory] = useState<ModelAtomCategory | 'all'>('composition')
  const [error, setError] = useState('')
  const [agentBusy, setAgentBusy] = useState(false)
  const [activityOpen, setActivityOpen] = useState(false)
  const [activities, setActivities] = useState<AgentActivityItem[]>([])
  const [pendingAgentPlan, setPendingAgentPlan] = useState<PendingCardAgentPlan>()
  const [destination, setDestination] = useState<CustomCardDestination>('new-architecture')
  const [playerSnapshot, setPlayerSnapshot] = useState<AtomicPlayerSnapshot>({ status: 'idle', results: [] })
  const playerRef = useRef<AtomicPlayer | null>(null)
  const activeActivityIdRef = useRef<string | undefined>(undefined)

  const selectedNode = graph.nodes.find((node) => node.id === selectedNodeId)
  const generated = useMemo(() => {
    try { return { code: compileRegistryGraph(graph), error: '' } }
    catch (reason) { return { code: '', error: reason instanceof Error ? reason.message : String(reason) } }
  }, [graph])
  const structuralErrors = validateCustomCardGraph(graph)
  const graphValidation = validateGraph(graph)
  const validationErrors = [...new Set([...structuralErrors, ...graphValidation.errors, ...(generated.error ? [generated.error] : [])])]
  const cardInputs = customCardInputPorts(graph)
  const cardOutputs = customCardOutputPorts(graph)
  const palette = Object.values(modelAtomRegistry).filter((definition) => !definition.composite && (paletteCategory === 'all' || definition.category === paletteCategory))
  const destinationChoices = [
    { value: 'new-architecture' as const, label: 'New architecture' },
    { value: 'library' as const, label: 'Reusable library card' },
    ...(selectedTarget && cardInputs.length === 1 ? [{ value: 'selected' as const, label: `Place after ${selectedTarget}` }] : []),
  ]

  const addAtom = (definition: ModelAtomDefinition, position?: { x: number; y: number }) => {
    setGraph((current) => {
      const id = uniqueNodeId(current, definition.id)
      return addNode(current, { id, kind: 'semantic', atomId: definition.id, label: definition.label, role: roleFor(definition), position: position ?? findOpenGraphPosition(current), attributes: Object.fromEntries(definition.settings.map((setting) => [setting.id, setting.default])) })
    })
  }
  const addInput = (role: TensorRole, position?: { x: number; y: number }) => {
    setGraph((current) => {
      const base = role === 'token-ids' ? 'token-ids' : `${role}-input`
      const id = uniqueNodeId(current, base)
      return addNode(current, { id, kind: 'input', label: inputChoices.find((item) => item.role === role)?.label ?? `${role} input`, role, position: position ?? findOpenGraphPosition(current) })
    })
  }
  const deleteNode = (nodeId: string) => {
    setGraph((current) => removeNode(current, nodeId))
    setSelectedNodeId('')
  }

  useEffect(() => {
    const valid = validateGraph(graph).valid
    let tracePromise: Promise<LaboRuntimeTrace> | undefined
    const player = new AtomicPlayer(valid ? executionLayers(graph) : graph.nodes.map((node) => [node.id]), async (atomId) => {
      const runAtomic = window.labo?.runAtomic
      if (!runAtomic) return previewModelAtom(graph, atomId)
      tracePromise ??= runAtomic({ kind: 'model', graph })
      const trace = await tracePromise
      const result = trace.results.find((candidate) => candidate.atomId === atomId)
      if (!result) throw new Error(trace.error ?? `PyTorch stopped before ${atomId}`)
      if (result.status === 'failed') throw new Error(result.error ?? `PyTorch failed at ${atomId}`)
      return { summary: result.summary }
    }, { onRestart: () => { tracePromise = undefined }, continueAfterFailure: true })
    playerRef.current = player
    return player.subscribe(setPlayerSnapshot)
  }, [graph])

  useEffect(() => onPlayerSnapshotChange?.(playerSnapshot), [onPlayerSnapshotChange, playerSnapshot])

  useImperativeHandle(ref, () => ({
    arrange: () => setGraph((current) => layoutArchitectureGraph(current)),
    exportDiagram: () => exportArchitectureDiagram(graph),
    exportPyTorch: () => exportPyTorchCode(graph, generated.code),
    pause: () => playerRef.current?.pause(),
    play: async () => { await playerRef.current?.play() },
    step: async () => { await playerRef.current?.step() },
    stop: () => playerRef.current?.stop(),
  }), [generated.code, graph])

  useEffect(() => {
    if (!requestedCard) return
    const inputRoles: Record<string, TensorRole> = {
      'token-ids-input': 'token-ids',
      'hidden-state-input': 'hidden',
      'training-labels-input': 'labels',
      'image-input': 'image',
      'video-input': 'video',
      'audio-input': 'audio',
    }
    const role = inputRoles[requestedCard.atomId]
    if (role) addInput(role)
    else {
      const definition = modelAtomRegistry[requestedCard.atomId]
      if (definition && !definition.composite) addAtom(definition)
    }
    onRequestedCardHandled()
  // The request id deliberately triggers a single card-builder insertion.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [requestedCard?.requestId])

  const updateActivity = (id: string, patch: Partial<AgentActivityItem>) => {
    setActivities((current) => current.map((activity) => activity.id === id ? { ...activity, ...patch } : activity))
  }

  const composePrompt = async (requestedNeed = need) => {
    const prompt = requestedNeed.trim()
    if (!prompt || agentBusy) return
    const activityId = `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`
    activeActivityIdRef.current = activityId
    setActivities((current) => [{ id: activityId, prompt, status: 'running' as const, createdAt: Date.now() }, ...current].slice(0, 20))
    setActivityOpen(true)
    if (!window.labo?.askLabo) {
      setError('Connect Ask LABO before composing a reusable card. No graph was changed.')
      updateActivity(activityId, { status: 'failed', error: 'Connect Ask LABO before composing a reusable card.' })
      return
    }
    setAgentBusy(true)
    setError('')
    try {
      const plan = await window.labo.askLabo({
        request: `Compose a reusable typed card graph for this need: ${prompt}`,
        context: { ...createAgentGraphContext(graph), cardBuilderMode: true, responseLocale: language },
      })
      const preview = previewAgentGraphPlan(graph, plan)
      if (preview.acceptedBlocks.length + preview.acceptedCreatedBlocks.length + preview.accepted.length + (plan.movedBlocks?.length ?? 0) === 0) throw new Error('The agent did not change the reusable graph.')
      const composedGraph = layoutArchitectureGraph(preview.graph)
      const errors = validateCustomCardGraph(composedGraph)
      if (errors.length > 0) throw new Error(errors[0])
      const suggestedName = plan.createdBlocks[0]?.label ?? plan.addedBlocks.findLast((block) => block.atomId !== 'hidden-state-input' && !block.atomId.endsWith('-input'))?.reason.split(/[.!?]/)[0]?.slice(0, 52) ?? plan.summary.split(/[.!?]/)[0]?.slice(0, 52) ?? 'Agent composed card'
      setPendingAgentPlan({ composedGraph, plan, preview, suggestedName })
      updateActivity(activityId, {
        status: 'review',
        summary: plan.summary,
        accepted: preview.acceptedBlocks.length + preview.acceptedCreatedBlocks.length + preview.accepted.length + (plan.movedBlocks?.length ?? 0),
        rejected: preview.rejectedBlocks.length + preview.rejected.length + preview.rejectedMutations.length + plan.warnings.length,
        missing: plan.missingBlocks.length,
        tools: plan.toolTrace?.map((item) => item.tool) ?? [],
        plan,
      })
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : String(reason)
      setError(`LABO agent did not change the card. ${message}`)
      updateActivity(activityId, { status: 'failed', error: message })
    } finally { setAgentBusy(false) }
  }

  const create = () => {
    const label = name.trim()
    if (!label) return setError('Give the card a name.')
    if (validationErrors.length > 0 || !generated.code) return setError(validationErrors[0] ?? 'Complete the internal graph first.')
    const result = onCreate({ label, code: generated.code, graph: structuredClone(graph), inputRole: cardInputs[0]?.tensor, outputRole: cardOutputs[0]?.tensor }, destination)
    if (!result.ok) setError(result.message ?? 'The selected destination is not compatible with this card.')
  }

  const applyPendingAgentPlan = () => {
    if (!pendingAgentPlan) return
    setGraph(pendingAgentPlan.composedGraph)
    setName(pendingAgentPlan.suggestedName)
    setSelectedNodeId(pendingAgentPlan.composedGraph.nodes.find((node) => node.kind !== 'input')?.id ?? '')
    setPendingAgentPlan(undefined)
    if (activeActivityIdRef.current) updateActivity(activeActivityIdRef.current, { status: 'applied' })
    setActivityOpen(true)
  }

  const discardPendingAgentPlan = () => {
    if (activeActivityIdRef.current) updateActivity(activeActivityIdRef.current, { status: 'discarded' })
    setPendingAgentPlan(undefined)
    setActivityOpen(true)
  }

  const reviewActivity = (activity: AgentActivityItem) => {
    if (!activity.plan) return
    const preview = previewAgentGraphPlan(graph, activity.plan)
    const composedGraph = layoutArchitectureGraph(preview.graph)
    const suggestedName = activity.plan.createdBlocks[0]?.label ?? activity.plan.summary.split(/[.!?]/)[0]?.slice(0, 52) ?? 'Agent composed card'
    activeActivityIdRef.current = activity.id
    setPendingAgentPlan({ composedGraph, plan: activity.plan, preview, suggestedName })
    setActivityOpen(false)
  }

  useEffect(() => {
    if (destination === 'selected' && (!selectedTarget || cardInputs.length !== 1)) setDestination('new-architecture')
  }, [cardInputs.length, destination, selectedTarget])

  return <section aria-label="Create model card" className="card-builder-shell">
    <StudioWorkspace aria-label="Card construction blocks" className="card-builder-workspace" inspectorOpen={inspectorOpen} libraryOpen={libraryOpen}>
      {libraryOpen && <StudioLibrary className={`mode-${editMode ? 'edit' : 'add'}`} heading="BLOCK LIBRARY" icon={<PanelLeft size={14} />}>
        <section className="block-group card-builder-library-content">
          <details className="library-family graph-input-family" open><summary>Card inputs <span>{inputChoices.length}</span></summary>{inputChoices.map((item) => <button aria-label={`Add ${item.label}`} className="library-block" disabled={editMode} draggable={!editMode} key={item.role} onClick={() => addInput(item.role)} onDragStart={(event) => { event.dataTransfer.setData('application/x-labo-graph-input', item.role); setLibraryDragPreview(event.dataTransfer, item.label, 'glyph-input') }}><span className="block-glyph glyph-input" /><Plus size={10} />{item.label}</button>)}</details>
          <details className="library-family catalog-family" open><summary>Card atoms <span>{palette.length}</span></summary><StudioChoiceMenu<ModelAtomCategory | 'all'> ariaLabel="Custom card category" className="card-builder-family-select" label="Family" onChange={setPaletteCategory} options={paletteCategories.map((item) => ({ label: item.label, value: item.id }))} value={paletteCategory} /><div aria-label="Atomic card palette">{palette.map((definition) => <button aria-label={`Add ${definition.label}`} className="library-block" disabled={editMode} draggable={!editMode} key={definition.id} onClick={() => addAtom(definition)} onDragStart={(event) => { event.dataTransfer.setData('application/x-labo-model-atom', definition.id); setLibraryDragPreview(event.dataTransfer, definition.label, `glyph-${definition.category}`) }}><span className={`block-glyph glyph-${definition.category}`} />{definition.label}</button>)}</div></details>
          <details className="library-family card-builder-save-family" open><summary>Reusable card <span>{graph.nodes.length}</span></summary><label><span>Name</span><input aria-label="Custom card name" onChange={(event) => setName(event.target.value)} value={name} /></label><StudioChoiceMenu<CustomCardDestination> ariaLabel="Card destination" label="Save as" onChange={setDestination} options={destinationChoices} value={destination} />{(error || validationErrors.length > 0) && <p className="model-card-modal-error" role="alert">{error || validationErrors[0]}</p>}<div className="card-builder-save-actions"><button className="create-custom-card-button" disabled={validationErrors.length > 0} onClick={create}>{destination === 'library' ? 'Save to My cards' : destination === 'selected' ? `Create after ${selectedTarget ?? 'selection'}` : 'Create reusable architecture'}</button><button className="card-builder-cancel" onClick={onClose}>Cancel</button></div></details>
        </section>
      </StudioLibrary>}
      <StudioEditor className={`view-${view}`}>
        {view !== 'pytorch' && <GraphCanvas editMode={editMode} graph={graph} onDeleteNode={deleteNode} onDeleteNodes={(ids) => { setGraph((current) => ids.reduce((next, id) => next.nodes.some((node) => node.id === id) ? removeNode(next, id) : next, current)); setSelectedNodeId('') }} onDropAtom={(atomId, position) => { const definition = modelAtomRegistry[atomId.split(':')[0]]; if (definition) addAtom(definition, position) }} onDropCustom={() => undefined} onDropInput={addInput} playerSnapshot={playerSnapshot} selectedNodeId={selectedNodeId} setGraph={setGraph} setSelectedNodeId={setSelectedNodeId} />}
        {view !== 'blocks' && <div className="code-panel"><div className="panel-tab"><Code2 size={13} /> generated_card.py <span>{validationErrors.length === 0 ? 'VALID COMPOSITE' : `${validationErrors.length} ISSUES`}</span></div><PythonCodePreview className="compact-python-editor" value={generated.code || `# ${validationErrors.join('\n# ')}`} /></div>}
      </StudioEditor>
      <StudioInspector heading="INSPECTOR" hidden={!inspectorOpen} icon={<Cpu size={14} />}><section className="inspector-section"><div className="section-title">Selection</div><div className="selection-card"><span className="selection-icon"><Zap size={15} /></span><div><strong>{selectedNode?.label ?? 'No selection'}</strong><small>{selectedNode?.id ?? '—'}</small></div></div><div className="card-builder-port-summary"><span>EXPOSED PLUGS</span><strong>{cardInputs.length} in · {cardOutputs.length} out</strong></div>{selectedNode ? <div className="card-builder-inspector-fields"><label><span>Card label</span><input aria-label="Selected internal card label" onChange={(event) => setGraph((current) => ({ ...current, nodes: current.nodes.map((node) => node.id === selectedNode.id ? { ...node, label: event.target.value } : node) }))} value={selectedNode.label} /></label>{selectedNode.kind === 'semantic' && selectedNode.atomId && modelAtomRegistry[selectedNode.atomId]?.settings.map((setting) => <label key={setting.id}><span>{setting.id}</span>{setting.type === 'boolean' ? <input checked={Boolean(selectedNode.attributes?.[setting.id])} onChange={(event) => setGraph((current) => updateNodeAttributes(current, selectedNode.id, { [setting.id]: event.target.checked }))} type="checkbox" /> : <input onChange={(event) => setGraph((current) => updateNodeAttributes(current, selectedNode.id, { [setting.id]: setting.type === 'number' ? Number(event.target.value) : event.target.value }))} type={setting.type === 'number' ? 'number' : 'text'} value={String(selectedNode.attributes?.[setting.id] ?? setting.default)} />}</label>)}<button className="card-builder-delete" onClick={() => deleteNode(selectedNode.id)}><Trash2 size={12} />Delete internal card</button></div> : <p className="blank-graph-hint">Choose a card atom from the library.</p>}</section></StudioInspector>
    </StudioWorkspace>
    {activityOpen && !pendingAgentPlan && <AgentActivityPanel activities={activities} busy={agentBusy} onClear={() => { setActivities([]); activeActivityIdRef.current = undefined }} onClose={() => setActivityOpen(false)} onRetry={(activity) => { setNeed(activity.prompt); void composePrompt(activity.prompt) }} onReview={reviewActivity} />}
    <StudioStatusbar className="model-statusbar card-builder-statusbar"><AgentPrompt busy={agentBusy} details={{ active: activityOpen, count: activities.length > 0 ? (activities.some((activity) => activity.status === 'running') ? '…' : activities.length) : undefined, label: 'Open agent activity', onToggle: () => setActivityOpen((current) => !current) }} mode="reusable-card" onChange={setNeed} onSubmit={() => void composePrompt()} value={need} /><span><span className={`status-dot ${validationErrors.length === 0 ? '' : 'invalid'}`} /> Card IR {graph.nodes.length === 0 ? 'blank' : validationErrors.length === 0 ? 'valid' : 'incomplete'}</span><span>{graph.nodes.length} nodes · {graph.edges.length} links</span><span className="status-spacer" /><span>PyTorch 2.7</span><span>LABO Runtime · local</span></StudioStatusbar>
    {pendingAgentPlan && <div className="ask-labo-backdrop review-open" onPointerDown={(event) => { if (event.target === event.currentTarget) discardPendingAgentPlan() }}>
      <aside aria-label="Ask LABO" aria-modal="true" className="ask-labo-panel has-plan" role="dialog">
        <header className="ask-labo-header"><span><Sparkles size={15} />{agentPlanReviewText[language].title}</span><button aria-label={agentPlanReviewText[language].close} onClick={discardPendingAgentPlan} type="button"><X size={15} /></button></header>
        <AgentPlanReview language={language} onApply={applyPendingAgentPlan} onDiscard={discardPendingAgentPlan} plan={pendingAgentPlan.plan} preview={pendingAgentPlan.preview} />
      </aside>
    </div>}
  </section>
})
