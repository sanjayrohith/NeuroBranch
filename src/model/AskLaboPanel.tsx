import { AlertTriangle, FolderKanban, Lightbulb, Palette, Settings2, Sparkles, X } from 'lucide-react'
import { useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from 'react'
import { createAgentGraphContext, previewAgentGraphPlan, repairAgentGraphPlan, type AgentGraphMode, type AgentGraphPlan } from '../core/agentic-graph'
import type { ArchitectureGraph, ArchitectureNode } from '../core/ir'
import { modelAtomRegistry } from '../core/model-atoms'
import { validCustomPyTorchModule } from '../core/pytorch-compiler'
import { PythonCodeEditor } from './PythonCodeEditor'
import { StudioSettingsModal } from '../StudioSettingsModal'
import type { CustomPyTorchCard } from './custom-card'
import { AgentPrompt } from '../studio/AgentPrompt'
import { AgentActivityPanel, type AgentActivityItem } from '../studio/AgentActivityPanel'
import { AgentSettingsContent, StudioEditingTips } from '../studio/AgentSettingsContent'
import { ApplicationAppearanceSettings } from '../studio/ApplicationAppearanceSettings'
import { useLaboLanguage } from '../studio/application-language'
import { AgentPlanReview } from './AgentPlanReview'
import { agentPlanReviewText } from './agent-plan-review-text'

interface AskLaboPanelProps {
  graph: ArchitectureGraph
  customCards: CustomPyTorchCard[]
  dockClassName?: string
  interactionMode: 'add' | 'edit'
  selectedNodeIds?: string[]
  open: boolean
  workspaceSettings: ReactNode
  onApply(graph: ArchitectureGraph, actions: NonNullable<AgentGraphPlan['actions']>): void
  onClose(): void
}

const AGENT_AUTO_APPLY_STORAGE_KEY = 'labo.ask.auto-apply.v1'

function loadAutoApply(): boolean {
  if (window.labo?.runtime === 'web') return false
  try {
    return window.localStorage.getItem(AGENT_AUTO_APPLY_STORAGE_KEY) === 'true'
  } catch {
    return false
  }
}

interface AgentCardOverride {
  label: string
  attributes?: Record<string, number | string | boolean>
  code?: string
}

export function AskLaboPanel({ graph, customCards, dockClassName = '', interactionMode, selectedNodeIds = [], open, workspaceSettings, onApply, onClose }: AskLaboPanelProps) {
  const language = useLaboLanguage()
  const copy = agentPlanReviewText[language]
  const [request, setRequest] = useState('')
  const [plan, setPlan] = useState<AgentGraphPlan>()
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [settings, setSettings] = useState<OpenAISettingsStatus>()
  const [chatGPT, setChatGPT] = useState<ChatGPTSessionStatus>()
  const [apiKey, setApiKey] = useState('')
  const [showApiKey, setShowApiKey] = useState(false)
  const [credentialBusy, setCredentialBusy] = useState(false)
  const [credentialMessage, setCredentialMessage] = useState('')
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [autoApply, setAutoApply] = useState(loadAutoApply)
  const [graphMode, setGraphMode] = useState<AgentGraphMode>('extend')
  const [cardOverrides, setCardOverrides] = useState<Record<string, AgentCardOverride>>({})
  const [editingCard, setEditingCard] = useState<ArchitectureNode>()
  const [editorDraft, setEditorDraft] = useState<AgentCardOverride>()
  const [editorError, setEditorError] = useState('')
  const [activityOpen, setActivityOpen] = useState(false)
  const [activities, setActivities] = useState<AgentActivityItem[]>([])
  const activeActivityIdRef = useRef<string | undefined>(undefined)
  const activePlan = useMemo(() => plan ? repairAgentGraphPlan(graph, plan) : undefined, [graph, plan])
  const preview = useMemo(() => {
    if (!activePlan) return undefined
    const base = previewAgentGraphPlan(graph, activePlan, graphMode)
    return {
      ...base,
      graph: {
        ...base.graph,
        nodes: base.graph.nodes.map((node) => {
          const override = cardOverrides[node.id]
          return override ? { ...node, label: override.label, attributes: override.attributes, code: override.code } : node
        }),
      },
    }
  }, [activePlan, cardOverrides, graph, graphMode])

  useEffect(() => {
    setConfirmDelete(false)
    if (!window.labo?.getOpenAISettings) {
      setSettings({ configured: false, source: 'none', encryptionAvailable: false })
      return
    }
    void window.labo.getOpenAISettings()
      .then(setSettings)
      .catch((reason) => setCredentialMessage(reason instanceof Error ? reason.message : String(reason)))
    if (window.labo.runtime === 'electron' && window.labo.getChatGPTSession) {
      void window.labo.getChatGPTSession()
        .then(setChatGPT)
        .catch((reason) => setChatGPT({ available: false, connected: false, error: reason instanceof Error ? reason.message : String(reason) }))
    }
  }, [open])

  useEffect(() => {
    if (!error) return
    const timeout = window.setTimeout(() => setError(''), 7_000)
    return () => window.clearTimeout(timeout)
  }, [error])

  useEffect(() => {
    if (window.labo?.runtime !== 'web') window.localStorage.setItem(AGENT_AUTO_APPLY_STORAGE_KEY, String(autoApply))
  }, [autoApply])

  useEffect(() => {
    if (!open && !plan && !activityOpen) return
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      if (activityOpen && !open && !plan) {
        setActivityOpen(false)
        return
      }
      setPlan(undefined)
      setCardOverrides({})
      setError('')
      onClose()
    }
    document.addEventListener('keydown', closeOnEscape)
    return () => document.removeEventListener('keydown', closeOnEscape)
  }, [activityOpen, onClose, open, plan])

  useEffect(() => {
    if (open) setActivityOpen(false)
  }, [open])

  const updateActivity = (id: string, patch: Partial<AgentActivityItem>) => {
    setActivities((current) => current.map((activity) => activity.id === id ? { ...activity, ...patch } : activity))
  }

  const runAgentRequest = async (prompt: string) => {
    if (!prompt || loading) return
    if (!window.labo?.askLabo) {
      setError('Connect an agent from Settings before using Ask LABO.')
      return
    }
    if (settings?.configured !== true && chatGPT?.connected !== true) {
      const currentSettings = settings ?? await window.labo.getOpenAISettings?.()
      if (currentSettings) setSettings(currentSettings)
      const currentChatGPT = chatGPT ?? await window.labo.getChatGPTSession?.()
      if (currentChatGPT) setChatGPT(currentChatGPT)
      if (currentSettings?.configured !== true && currentChatGPT?.connected !== true) {
        setError('Open Settings → Agent to connect an agent first.')
        return
      }
    }
    const activityId = `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`
    activeActivityIdRef.current = activityId
    setActivities((current) => [{ id: activityId, prompt, status: 'running' as const, createdAt: Date.now() }, ...current].slice(0, 20))
    setActivityOpen(true)
    setLoading(true)
    setError('')
    setPlan(undefined)
    setCardOverrides({})
    try {
      const rawResponse = await window.labo.askLabo({
        request: prompt,
        context: {
          ...createAgentGraphContext(graph, graphMode, customCards, { active: interactionMode === 'edit', nodeIds: selectedNodeIds }),
          responseLocale: language,
        },
      })
      const response = repairAgentGraphPlan(graph, rawResponse)
      const responsePreview = previewAgentGraphPlan(graph, response, graphMode, interactionMode === 'edit' ? { editingNodeIds: selectedNodeIds } : undefined)
      const hasAcceptedChanges = responsePreview.acceptedBlocks.length > 0 || responsePreview.acceptedCreatedBlocks.length > 0 || responsePreview.accepted.length > 0 || (response.updatedBlocks?.length ?? 0) > 0 || (response.replacedBlocks?.length ?? 0) > 0 || (response.deletedBlocks?.length ?? 0) > 0 || (response.movedBlocks?.length ?? 0) > 0 || responsePreview.acceptedActions.some((action) => action.type !== 'layout')
      const hasPlan = hasAcceptedChanges || responsePreview.rejectedBlocks.length > 0 || responsePreview.rejected.length > 0 || responsePreview.rejectedMutations.length > 0 || response.missingBlocks.length > 0 || response.warnings.length > 0
      const accepted = responsePreview.acceptedBlocks.length + responsePreview.acceptedCreatedBlocks.length + responsePreview.accepted.length + (response.updatedBlocks?.length ?? 0) + (response.replacedBlocks?.length ?? 0) + (response.deletedBlocks?.length ?? 0) + (response.movedBlocks?.length ?? 0)
      const rejected = responsePreview.rejectedBlocks.length + responsePreview.rejected.length + responsePreview.rejectedMutations.length + response.warnings.length
      const activityResult = {
        summary: response.summary,
        accepted,
        rejected,
        missing: response.missingBlocks.length,
        tools: response.toolTrace?.map((item) => item.tool) ?? [],
        plan: response,
      }
      if (!hasPlan) {
        updateActivity(activityId, { ...activityResult, status: 'answered', plan: undefined })
        setRequest('')
        setActivityOpen(true)
      } else if (autoApply && hasAcceptedChanges) {
        updateActivity(activityId, { ...activityResult, status: 'applied' })
        onApply(responsePreview.graph, responsePreview.acceptedActions)
        setActivityOpen(true)
        onClose()
      } else {
        updateActivity(activityId, { ...activityResult, status: 'review' })
        setActivityOpen(false)
        setPlan(response)
      }
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : String(reason)
      setError(message)
      updateActivity(activityId, { status: 'failed', error: message })
      setActivityOpen(true)
      window.setTimeout(() => setActivityOpen(false), 7_000)
    } finally {
      setLoading(false)
    }
  }

  const submit = () => {
    void runAgentRequest(request.trim())
  }

  const apply = () => {
    if (!preview) return
    if (activeActivityIdRef.current) updateActivity(activeActivityIdRef.current, { status: 'applied' })
    onApply(preview.graph, preview.acceptedActions)
    setPlan(undefined)
    setCardOverrides({})
    setRequest('')
    setActivityOpen(true)
    onClose()
  }

  const reviewFullPlan = (activity: AgentActivityItem) => {
    if (!activity.plan) return
    const repairedPlan = repairAgentGraphPlan(graph, activity.plan)
    activeActivityIdRef.current = activity.id
    updateActivity(activity.id, { plan: repairedPlan })
    setCardOverrides({})
    setPlan(repairedPlan)
    setActivityOpen(false)
  }

  const clearActivities = () => {
    activeActivityIdRef.current = undefined
    setActivities([])
    setActivityOpen(false)
  }

  const openCardEditor = (nodeId: string) => {
    const node = preview?.graph.nodes.find((candidate) => candidate.id === nodeId)
    if (!node) return
    setEditingCard(node)
    setEditorDraft({ label: node.label, attributes: node.attributes ? { ...node.attributes } : undefined, code: node.code })
    setEditorError('')
  }

  const saveCardEditor = () => {
    if (!editingCard || !editorDraft) return
    const label = editorDraft.label.trim()
    if (!label) {
      setEditorError('Give the card a name.')
      return
    }
    if (editingCard.kind === 'custom-pytorch' && !validCustomPyTorchModule(editorDraft.code ?? '')) {
      setEditorError('Use one safe nn.Module constructor with literal arguments only.')
      return
    }
    setCardOverrides((current) => ({ ...current, [editingCard.id]: { ...editorDraft, label, code: editorDraft.code?.trim() } }))
    setEditingCard(undefined)
    setEditorDraft(undefined)
    setEditorError('')
  }

  const saveApiKey = async (event: FormEvent) => {
    event.preventDefault()
    if (!window.labo?.saveOpenAIKey || !apiKey.trim() || credentialBusy) return
    setCredentialBusy(true)
    setCredentialMessage('')
    try {
      const status = await window.labo.saveOpenAIKey(apiKey)
      setSettings(status)
      setApiKey('')
      if (window.labo.testOpenAIKey) {
        await window.labo.testOpenAIKey()
        setCredentialMessage('Key saved and verified with OpenAI.')
      } else setCredentialMessage('Key saved securely.')
    } catch (reason) {
      setCredentialMessage(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setCredentialBusy(false)
    }
  }

  const testApiKey = async () => {
    if (!window.labo?.testOpenAIKey || credentialBusy) return
    setCredentialBusy(true)
    setCredentialMessage('')
    try {
      await window.labo.testOpenAIKey()
      setCredentialMessage('OpenAI connection successful.')
    } catch (reason) {
      setCredentialMessage(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setCredentialBusy(false)
    }
  }

  const deleteApiKey = async () => {
    if (!confirmDelete) {
      setConfirmDelete(true)
      return
    }
    if (!window.labo?.deleteOpenAIKey || credentialBusy) return
    setCredentialBusy(true)
    setCredentialMessage('')
    try {
      setSettings(await window.labo.deleteOpenAIKey())
      setConfirmDelete(false)
      setCredentialMessage('API key removed from this user account.')
    } catch (reason) {
      setCredentialMessage(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setCredentialBusy(false)
    }
  }

  const connectChatGPT = async () => {
    if (!window.labo?.connectChatGPT || credentialBusy) return
    setCredentialBusy(true)
    setCredentialMessage('Complete ChatGPT sign-in in your browser.')
    try {
      const status = await window.labo.connectChatGPT()
      setChatGPT(status)
      setCredentialMessage(status.connected ? 'ChatGPT account connected. Ask LABO will use it by default.' : 'ChatGPT sign-in was not completed.')
    } catch (reason) {
      setCredentialMessage(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setCredentialBusy(false)
    }
  }

  const disconnectChatGPT = async () => {
    if (!window.labo?.disconnectChatGPT || credentialBusy) return
    setCredentialBusy(true)
    setCredentialMessage('')
    try {
      setChatGPT(await window.labo.disconnectChatGPT())
      setCredentialMessage('ChatGPT account disconnected from LABO AI.')
    } catch (reason) {
      setCredentialMessage(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setCredentialBusy(false)
    }
  }

  const configureChatGPT = async (configuration: { model: string; effort: string }) => {
    if (!window.labo?.configureChatGPT || credentialBusy) return
    setCredentialBusy(true)
    setCredentialMessage('')
    try {
      setChatGPT(await window.labo.configureChatGPT(configuration))
      setCredentialMessage('ChatGPT model settings saved for this desktop profile.')
    } catch (reason) {
      setCredentialMessage(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setCredentialBusy(false)
    }
  }

  const closeOverlay = () => {
    if (plan) {
      if (activeActivityIdRef.current) updateActivity(activeActivityIdRef.current, { status: 'discarded' })
      setPlan(undefined)
      setCardOverrides({})
      setActivityOpen(true)
    }
    setError('')
    onClose()
  }

  return <div className={`ask-labo-backdrop ${dockClassName} ${plan ? 'review-open' : ''}`} onPointerDown={(event) => { if (event.target === event.currentTarget) closeOverlay() }}>
  <aside aria-label="Ask LABO" aria-modal={Boolean(plan)} className={`ask-labo-panel ${plan ? 'has-plan' : ''}`} role={plan ? 'dialog' : 'region'}>
    <header className="ask-labo-header">
      <span>{plan ? <Sparkles size={15} /> : <Settings2 size={15} />}{plan ? copy.title : 'LABO settings'}</span>
      <button aria-label={copy.close} onClick={closeOverlay}><X size={15} /></button>
    </header>

    {activityOpen && !open && !plan && <AgentActivityPanel activities={activities} busy={loading} onClear={clearActivities} onClose={() => setActivityOpen(false)} onRetry={(activity) => { setRequest(activity.prompt); setActivityOpen(false); void runAgentRequest(activity.prompt) }} onReview={reviewFullPlan} />}

    <AgentPrompt busy={loading} context={settings?.configured === false && chatGPT?.connected !== true ? 'Connect an agent in Settings' : `${autoApply ? 'Auto apply' : 'Review'} · ${graphMode === 'parallel' ? 'New parallel' : 'Extend current'}`} details={{ active: activityOpen, count: activities.length > 0 ? (activities.find((activity) => activity.status === 'running') ? '…' : activities.length) : undefined, label: 'Open agent activity', onToggle: () => setActivityOpen((current) => !current) }} disabled={settings?.configured === false && chatGPT?.connected !== true} mode={interactionMode === 'edit' ? 'card-editing' : 'architecture'} onChange={(value) => { setRequest(value); if (error) setError('') }} onSubmit={submit} value={request} />

    {open && <StudioSettingsModal onClose={closeOverlay} sections={[
      { id: 'workspaces', label: 'Workspaces', icon: <FolderKanban size={13} />, content: <div className="ask-labo-workspace-settings">{workspaceSettings}</div> },
      { id: 'agent', label: 'Agent', icon: <Sparkles size={13} />, content: <AgentSettingsContent apiKey={apiKey} autoApply={autoApply} chatGPT={chatGPT} confirmDelete={confirmDelete} credentialBusy={credentialBusy} credentialMessage={credentialMessage} graphMode={graphMode} loading={loading} onApiKeyChange={setApiKey} onAutoApplyChange={setAutoApply} onChatGPTConfigurationChange={(configuration) => void configureChatGPT(configuration)} onConnectChatGPT={() => void connectChatGPT()} onDeleteApiKey={() => void deleteApiKey()} onDisconnectChatGPT={() => void disconnectChatGPT()} onGraphModeChange={(mode) => { setGraphMode(mode); setPlan(undefined) }} onSaveApiKey={(event) => void saveApiKey(event)} onShowApiKeyChange={setShowApiKey} onTestApiKey={() => void testApiKey()} settings={settings} showApiKey={showApiKey} /> },
      { id: 'studio', label: 'Application', icon: <Palette size={13} />, content: <ApplicationAppearanceSettings /> },
      { id: 'tips', label: 'Tips', icon: <Lightbulb size={13} />, content: <StudioEditingTips /> },
    ]} />}

    {error && <div className="ask-labo-error"><AlertTriangle size={14} /><span>{error}</span><button aria-label="Dismiss agent error" onClick={() => setError('')} type="button"><X size={12} /></button></div>}

    {activePlan && preview && <AgentPlanReview
      language={language}
      onApply={apply}
      onDiscard={() => {
        if (activeActivityIdRef.current) updateActivity(activeActivityIdRef.current, { status: 'discarded' })
        setPlan(undefined)
        setCardOverrides({})
        setActivityOpen(true)
      }}
      onEditCard={openCardEditor}
      plan={activePlan}
      preview={preview}
    />}

    {editingCard && editorDraft && <div className="ask-labo-card-modal-backdrop" onPointerDown={(event) => { if (event.target === event.currentTarget) { setEditingCard(undefined); setEditorDraft(undefined); setEditorError('') } }}>
      <section aria-label="Edit agent card" aria-modal="true" className="ask-labo-card-modal" onPointerDown={(event) => event.stopPropagation()} role="dialog">
        <header><strong>Edit card</strong><button aria-label="Close card editor" onClick={() => setEditingCard(undefined)} type="button"><X size={13} /></button></header>
        <label><span>Name</span><input aria-label="Agent card name" onChange={(event) => setEditorDraft((current) => current ? { ...current, label: event.target.value } : current)} value={editorDraft.label} /></label>
        <label><span>Block ID</span><input aria-label="Agent card ID" disabled value={editingCard.id} /></label>
        {editingCard.kind === 'custom-pytorch' ? <label><span>PyTorch module</span><PythonCodeEditor ariaLabel="Agent card PyTorch module" className="compact-python-editor" onChange={(value) => setEditorDraft((current) => current ? { ...current, code: value } : current)} value={editorDraft.code ?? ''} /></label> : <div className="ask-labo-card-settings">
          {modelAtomRegistry[editingCard.atomId ?? '']?.settings.map((setting) => {
            const value = editorDraft.attributes?.[setting.id] ?? setting.default
            return <label key={setting.id}><span>{setting.id}</span>{setting.type === 'boolean'
              ? <input aria-label={`Agent card setting ${setting.id}`} checked={Boolean(value)} onChange={(event) => setEditorDraft((current) => current ? { ...current, attributes: { ...current.attributes, [setting.id]: event.target.checked } } : current)} type="checkbox" />
              : setting.type === 'select'
                ? <select aria-label={`Agent card setting ${setting.id}`} onChange={(event) => setEditorDraft((current) => current ? { ...current, attributes: { ...current.attributes, [setting.id]: event.target.value } } : current)} value={String(value)}>{setting.options?.map((option) => <option key={option}>{option}</option>)}</select>
                : <input aria-label={`Agent card setting ${setting.id}`} onChange={(event) => setEditorDraft((current) => current ? { ...current, attributes: { ...current.attributes, [setting.id]: setting.type === 'number' ? Number(event.target.value) : event.target.value } } : current)} type={setting.type === 'number' ? 'number' : 'text'} value={String(value)} />}</label>
          })}
        </div>}
        {editorError && <p role="alert">{editorError}</p>}
        <footer><button onClick={() => setEditingCard(undefined)} type="button">Cancel</button><button onClick={saveCardEditor} type="button">Save card</button></footer>
      </section>
    </div>}
  </aside>
  </div>
}
