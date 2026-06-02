import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Blocks, Check, Code2, Cpu, PackageCheck, Pause, Pencil, Play, Plus, Square, StepForward, Trash2 } from 'lucide-react'
import { AtomicPlayer, type AtomicPlayerSnapshot, type AtomExecutionResult } from './core/atomic-player'
import { PythonCodePreview } from './model/PythonCodeEditor'
import {
  addTokenizerStep,
  compileTokenizer,
  removeTokenizerStep,
  tokenizerAtomDefinitions,
  tokenizerAtomMetadata,
  updateTokenizerStepSettings,
  type TokenizerPipeline,
  type TokenizerStep,
} from './core/tokenizer-ir'
import { builtInTokenizerPresets, researchBpePreset } from './core/tokenizer-presets'
import { TokenizerCardCreator } from './tokenizer/TokenizerCardCreator'
import type { CustomTokenizerCard } from './tokenizer/custom-tokenizer-card'
import { StudioEditor, StudioInspector, StudioLibrary, StudioStatusbar, StudioToolbar, StudioViewSwitcher, StudioWorkspace } from './studio/StudioShell'
import { StudioLibraryItem } from './studio/StudioLibraryParts'
import { InspectorMetric } from './studio/StudioInspectorParts'
import { StudioCanvasPanel, StudioCodePanel, StudioPanelTab } from './studio/StudioPanels'
import { StudioContextMenu, StudioContextMenuItem } from './studio/StudioContextMenu'

type TokenizerView = 'blocks' | 'split'

function isTokenizerPipeline(value: unknown): value is TokenizerPipeline {
  if (!value || typeof value !== 'object') return false
  const pipeline = value as Partial<TokenizerPipeline>
  return typeof pipeline.id === 'string' && typeof pipeline.name === 'string' && Array.isArray(pipeline.steps) && Array.isArray(pipeline.links)
}

function isCustomTokenizerCard(value: unknown): value is CustomTokenizerCard {
  if (!value || typeof value !== 'object') return false
  const card = value as Partial<CustomTokenizerCard>
  return typeof card.id === 'string' && typeof card.label === 'string' && typeof card.category === 'string' && typeof card.pythonCode === 'string'
}

function formatSetting(value: string | number | boolean | string[]): string {
  return Array.isArray(value) ? value.join(', ') : String(value)
}

async function executeTokenizerIrAtom(step: TokenizerStep): Promise<{ summary: string }> {
  if (step.atom === 'unicode-normalize') {
    const form = String(step.settings.form) as 'NFC' | 'NFD' | 'NFKC' | 'NFKD'
    return { summary: `Café normalized with ${form}: ${'Cafe\u0301'.normalize(form)}` }
  }
  if (step.atom === 'byte-level-pretokenize') {
    return { summary: `UTF-8 bytes: ${new TextEncoder().encode('LABO AI').length}` }
  }
  if (step.atom === 'bpe-model') {
    if (!step.settings.unkToken) throw new Error('BPE model requires unkToken')
    return { summary: `BPE model contract: unk=${String(step.settings.unkToken)}` }
  }
  if (step.atom === 'bpe-trainer') {
    const vocabSize = Number(step.settings.vocabSize)
    if (!Number.isInteger(vocabSize) || vocabSize <= 0) throw new Error('BPE trainer requires a positive integer vocabSize')
    return { summary: `BPE trainer contract: ${vocabSize} entries` }
  }
  if (step.atom === 'tiktoken-encoding') {
    return { summary: `${String(step.settings.encoding)} pretrained contract: ${Number(step.settings.vocabSize).toLocaleString('en-US')} tokens` }
  }
  if (step.atom.includes('image')) return { summary: `${tokenizerAtomMetadata[step.atom].label} contract ready for [B, C, H, W] images` }
  if (step.atom.includes('video')) return { summary: `${tokenizerAtomMetadata[step.atom].label} contract ready for [B, C, T, H, W] videos` }
  if (step.atom.includes('audio')) return { summary: `${tokenizerAtomMetadata[step.atom].label} contract ready for [B, C, samples] waveforms` }
  if (step.atom === 'custom-tokenizer') {
    return { summary: `Custom tokenizer card ready: ${String(step.settings.label)}` }
  }
  return { summary: `Byte-level round-trip: ${new TextDecoder().decode(new TextEncoder().encode('LABO AI'))}` }
}

export function TokenizerStudio({ onCatalogChange = () => undefined, onRequestedCardHandled = () => undefined, requestedCard }: { onCatalogChange?: (cards: CustomTokenizerCard[]) => void; onRequestedCardHandled?: () => void; requestedCard?: { cardId: string; kind: 'atom' | 'custom'; requestId: number } }) {
  const [pipeline, setPipeline] = useState(researchBpePreset)
  const [view, setView] = useState<TokenizerView>('blocks')
  const [selectedId, setSelectedId] = useState(pipeline.steps[0]?.id ?? '')
  const [customCards, setCustomCards] = useState<CustomTokenizerCard[]>([])
  const [customCardsReady, setCustomCardsReady] = useState(false)
  const [webAuthenticated, setWebAuthenticated] = useState(false)
  const [cardCreatorOpen, setCardCreatorOpen] = useState(false)
  const [editingCard, setEditingCard] = useState<CustomTokenizerCard>()
  const [cardMenu, setCardMenu] = useState<{ cardId: string; x: number; y: number; confirmDelete?: boolean }>()
  const presetMenuRef = useRef<HTMLDetailsElement>(null)
  const [playerSnapshot, setPlayerSnapshot] = useState<AtomicPlayerSnapshot>({
    status: 'idle', currentAtomId: pipeline.steps[0]?.id,
    results: pipeline.steps.map((step) => ({ atomId: step.id, status: 'pending' })),
  })
  const playerRef = useRef<AtomicPlayer | null>(null)
  const code = useMemo(() => compileTokenizer(pipeline), [pipeline])
  const selected = pipeline.steps.find((step) => step.id === selectedId) ?? pipeline.steps[0]
  const vocabularyStep = pipeline.steps.find((step) => step.atom === 'bpe-trainer' || step.atom === 'tiktoken-encoding' || step.atom === 'image-vq-encode' || step.atom === 'video-vq-encode' || step.atom === 'audio-vq-encode' || step.atom === 'image-codebook-embedding' || step.atom === 'video-codebook-embedding' || step.atom === 'audio-codebook-embedding')
  const vocabSize = Number(vocabularyStep?.settings.vocabSize ?? vocabularyStep?.settings.codebookSize ?? 0)

  useEffect(() => {
    const player = new AtomicPlayer(
      pipeline.steps.map((step) => step.id),
      async (atomId) => executeTokenizerIrAtom(pipeline.steps.find((step) => step.id === atomId)!),
    )
    playerRef.current = player
    return player.subscribe(setPlayerSnapshot)
  }, [pipeline])

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      if (window.labo?.runtime === 'electron' && window.labo.loadDesktopState) {
        const value = await window.labo.loadDesktopState('tokenizer')
        const stored = value && typeof value === 'object' ? value as { pipeline?: unknown; customCards?: unknown } : undefined
        if (!cancelled && isTokenizerPipeline(stored?.pipeline)) {
          setPipeline(stored.pipeline)
          setSelectedId(stored.pipeline.steps[0]?.id ?? '')
        }
        if (!cancelled && Array.isArray(stored?.customCards)) setCustomCards(stored.customCards.filter(isCustomTokenizerCard))
      } else if (window.labo?.runtime === 'web' && window.labo.loadWebWorkspace) {
        const result = await window.labo.loadWebWorkspace()
        const authenticated = Boolean(result && typeof result === 'object' && result.authenticated)
        if (!cancelled) setWebAuthenticated(authenticated)
        const stored = result && typeof result === 'object' && result.tokenizer && typeof result.tokenizer === 'object' ? result.tokenizer as { pipeline?: unknown; customCards?: unknown } : undefined
        if (!cancelled && authenticated && isTokenizerPipeline(stored?.pipeline)) {
          setPipeline(stored.pipeline)
          setSelectedId(stored.pipeline.steps[0]?.id ?? '')
        }
        if (!cancelled && authenticated && Array.isArray(stored?.customCards)) setCustomCards(stored.customCards.filter(isCustomTokenizerCard))
      }
      if (!cancelled) setCustomCardsReady(true)
    }
    void load().catch(() => { if (!cancelled) setCustomCardsReady(true) })
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    if (!customCardsReady) return
    if (window.labo?.runtime === 'electron' && window.labo.saveDesktopState) {
      void window.labo.saveDesktopState('tokenizer', { pipeline, customCards, updatedAt: Date.now() })
    } else if (window.labo?.runtime === 'web' && webAuthenticated && window.labo.saveWebWorkspace) {
      void window.labo.saveWebWorkspace({ tokenizer: { pipeline, customCards, updatedAt: Date.now() } })
    }
  }, [customCards, customCardsReady, pipeline, webAuthenticated])

  useEffect(() => {
    if (!cardMenu) return
    const dismiss = (event: PointerEvent) => {
      if (!(event.target as HTMLElement | null)?.closest('.tokenizer-library-context-menu')) setCardMenu(undefined)
    }
    const dismissOnEscape = (event: KeyboardEvent) => { if (event.key === 'Escape') setCardMenu(undefined) }
    window.addEventListener('pointerdown', dismiss)
    window.addEventListener('keydown', dismissOnEscape)
    return () => {
      window.removeEventListener('pointerdown', dismiss)
      window.removeEventListener('keydown', dismissOnEscape)
    }
  }, [cardMenu])

  const deleteSelected = () => {
    if (!selected) return
    const next = removeTokenizerStep(pipeline, selected.id)
    setPipeline(next)
    setSelectedId(next.steps[0]?.id ?? '')
  }

  const addAtom = useCallback((atom: TokenizerStep['atom']) => {
    setPipeline((current) => {
      const next = addTokenizerStep(current, atom)
      setSelectedId(next.steps.at(-1)!.id)
      return next
    })
  }, [])

  const addCustomTokenizerCard = useCallback((card: CustomTokenizerCard) => {
    setPipeline((current) => {
      const sequence = current.steps.filter((step) => step.id.startsWith(`${card.id}-`)).length + 1
      const step: TokenizerStep = {
        id: `${card.id}-${sequence}`,
        atom: 'custom-tokenizer',
        settings: { label: card.label, category: card.category, pythonCode: card.pythonCode },
      }
      setSelectedId(step.id)
      return { ...current, steps: [...current.steps, step] }
    })
  }, [])

  useEffect(() => onCatalogChange(customCards), [customCards, onCatalogChange])

  useEffect(() => {
    if (!requestedCard || !customCardsReady) return
    if (requestedCard.kind === 'atom' && requestedCard.cardId in tokenizerAtomDefinitions) addAtom(requestedCard.cardId as TokenizerStep['atom'])
    if (requestedCard.kind === 'custom') {
      const card = customCards.find((candidate) => candidate.id === requestedCard.cardId)
      if (card) addCustomTokenizerCard(card)
    }
    onRequestedCardHandled()
  }, [addAtom, addCustomTokenizerCard, customCards, customCardsReady, onRequestedCardHandled, requestedCard])

  return (
    <>
      <StudioToolbar>
        <StudioViewSwitcher<TokenizerView> ariaLabel="Tokenizer editor view" onChange={setView} options={[{ id: 'blocks', label: 'Blocks', icon: <Blocks size={14} /> }, { id: 'split', label: 'Split', icon: <Code2 size={14} /> }]} value={view} />
        <details className="preset-menu tokenizer-preset-menu" ref={presetMenuRef}>
          <summary aria-label="Tokenizer preset">{pipeline.name}</summary>
          <div>{builtInTokenizerPresets.map((preset) => <button aria-pressed={pipeline.id === preset.id} key={preset.id} onClick={() => {
            const next = structuredClone(preset)
            setPipeline(next)
            setSelectedId(next.steps[0]?.id ?? '')
            presetMenuRef.current?.removeAttribute('open')
          }} type="button">{preset.name}</button>)}</div>
        </details>
        <div className="atomic-player-controls" aria-label="Atomic pipeline player">
          <button aria-label="Play atomic pipeline" onClick={() => void playerRef.current?.play()}><Play size={13} /></button>
          <button aria-label="Pause atomic pipeline" onClick={() => playerRef.current?.pause()}><Pause size={13} /></button>
          <button aria-label="Step one atom" onClick={() => void playerRef.current?.step()}><StepForward size={13} /></button>
          <button aria-label="Stop atomic pipeline" onClick={() => playerRef.current?.stop()}><Square size={12} /></button>
          <span className={`player-status status-${playerSnapshot.status}`}>{playerSnapshot.status}</span>
        </div>
      </StudioToolbar>

      <StudioWorkspace className="tokenizer-workspace">
        <StudioLibrary heading="TOKENIZER ATOMS" icon={<Blocks size={14} />}>
          {Object.entries(tokenizerAtomDefinitions).filter(([atom]) => atom !== 'custom-tokenizer').map(([atom, metadata]) => {
            return (
              <StudioLibraryItem aria-label={`Add ${metadata.label}`} className="tokenizer-library-block" glyph={<span className="block-glyph glyph-transforms" />} key={atom} meta={metadata.category} onClick={() => addAtom(atom as TokenizerStep['atom'])}>{metadata.label}</StudioLibraryItem>
            )
          })}
          <StudioLibraryItem className="tokenizer-library-block" glyph={<Plus size={14} />} meta="Custom Python lowering" onClick={() => setCardCreatorOpen(true)}>New reusable card</StudioLibraryItem>
          {customCards.map((card) => <StudioLibraryItem aria-label={`Add ${card.label}`} className="tokenizer-library-block" glyph={<span className="block-glyph glyph-transforms" />} key={card.id} meta={`${card.category} · My cards`} onClick={() => addCustomTokenizerCard(card)} onContextMenu={(event) => {
            event.preventDefault()
            setCardMenu({ cardId: card.id, x: event.clientX, y: event.clientY })
          }} title="Right-click to edit or delete">{card.label}</StudioLibraryItem>)}
        </StudioLibrary>

        <StudioEditor className={`tokenizer-editor view-${view}`}>
          <StudioCanvasPanel tab={<StudioPanelTab icon={<Blocks size={13} />}>tokenizer.pipeline</StudioPanelTab>}>
            <div className="tokenizer-canvas">
              {pipeline.steps.map((step, index) => {
                const metadata = step.atom === 'custom-tokenizer'
                  ? { label: String(step.settings.label), category: String(step.settings.category) }
                  : tokenizerAtomMetadata[step.atom]
                return (
                  <button
                    aria-label={`Select ${metadata.label}`}
                    className={`tokenizer-atom ${selectedId === step.id ? 'selected' : ''} status-${playerSnapshot.results.find((result) => result.atomId === step.id)?.status ?? 'pending'}`}
                    key={step.id}
                    onClick={() => setSelectedId(step.id)}
                  >
                    <span className="atom-order">{String(index + 1).padStart(2, '0')}</span>
                    <span className="node-type">{metadata.category}</span>
                    <strong>{metadata.label}</strong>
                    <small>{Object.entries(step.settings).map(([key, value]) => `${key}: ${formatSetting(value)}`).join(' · ') || 'no settings'}</small>
                  </button>
                )
              })}
            </div>
          </StudioCanvasPanel>

          {view === 'split' && (
            <StudioCodePanel tab={<StudioPanelTab icon={<Code2 size={13} />} status="GENERATED">tokenizer.py</StudioPanelTab>}>
              <PythonCodePreview value={code} />
            </StudioCodePanel>
          )}
        </StudioEditor>

        <StudioInspector heading="ATOM INSPECTOR" icon={<Cpu size={14} />}>
          {selected && <TokenizerAtomInspector
            onDelete={deleteSelected}
            onSettingChange={(key, value) => setPipeline((current) => updateTokenizerStepSettings(current, selected.id, { [key]: value }))}
            result={playerSnapshot.results.find((result) => result.atomId === selected.id)}
            step={selected}
          />}
          <section className="equivalence-card tokenizer-artifact-card">
            <div className="equivalence-title"><PackageCheck size={14} /> Artifact contract</div>
            <InspectorMetric label="Vocabulary size" value={vocabSize.toLocaleString('en-US')} />
            <InspectorMetric label="Steps" value={pipeline.steps.length} />
            <InspectorMetric label="Typed links" value={pipeline.links.length} />
            <InspectorMetric label="Python lowering" value={<span className="passed">READY</span>} />
          </section>
        </StudioInspector>
      </StudioWorkspace>

      <StudioStatusbar>
        <span><span className="status-dot" /> Tokenizer IR valid</span>
        <span>{pipeline.steps.length} atoms · {pipeline.links.length} typed links</span>
        <span className="status-spacer" />
        <span>Python backend</span>
        <span>LABO Runtime · local</span>
      </StudioStatusbar>
      {cardMenu && (() => {
        const card = customCards.find((candidate) => candidate.id === cardMenu.cardId)
        if (!card) return null
        return <StudioContextMenu className="tokenizer-library-context-menu" position={cardMenu}>
          <div><span>MY TOKENIZER CARD</span><strong>{card.label}</strong></div>
          <StudioContextMenuItem onClick={() => { setEditingCard(card); setCardMenu(undefined) }}><Pencil size={13} />Edit card</StudioContextMenuItem>
          <StudioContextMenuItem className={cardMenu.confirmDelete ? 'confirm-delete' : ''} onClick={() => {
            if (!cardMenu.confirmDelete) return setCardMenu((current) => current ? { ...current, confirmDelete: true } : current)
            setCustomCards((current) => current.filter((candidate) => candidate.id !== card.id))
            setCardMenu(undefined)
          }}><Trash2 size={13} />{cardMenu.confirmDelete ? 'Confirm delete' : 'Delete card'}</StudioContextMenuItem>
        </StudioContextMenu>
      })()}
      {(cardCreatorOpen || editingCard) && <TokenizerCardCreator
        initialCard={editingCard}
        onCancel={() => { setCardCreatorOpen(false); setEditingCard(undefined) }}
        onCreate={(card) => {
          if (editingCard) {
            setCustomCards((current) => current.map((candidate) => candidate.id === editingCard.id ? card : candidate))
            setPipeline((current) => ({ ...current, steps: current.steps.map((step) => step.id.startsWith(`${editingCard.id}-`) ? { ...step, settings: { label: card.label, category: card.category, pythonCode: card.pythonCode } } : step) }))
            setEditingCard(undefined)
            return
          }
          setCustomCards((current) => [...current.filter((candidate) => candidate.id !== card.id), card])
          setCardCreatorOpen(false)
          addCustomTokenizerCard(card)
        }}
      />}
    </>
  )
}

function TokenizerAtomInspector({
  step,
  result,
  onDelete,
  onSettingChange,
}: {
  step: TokenizerStep
  result?: AtomExecutionResult
  onDelete(): void
  onSettingChange(key: string, value: string | number | boolean | string[]): void
}) {
  const metadata = step.atom === 'custom-tokenizer'
    ? { label: String(step.settings.label), category: String(step.settings.category) }
    : tokenizerAtomMetadata[step.atom]
  return (
    <>
      <section className="inspector-section">
        <div className="section-title">Selection</div>
        <div className="selection-card">
          <span className="selection-icon"><Blocks size={15} /></span>
          <div><strong>{metadata.label}</strong><small>{step.atom}</small></div>
        </div>
      </section>
      <section className="inspector-section">
        <div className="section-title">Atomic settings</div>
        <div className="atomic-settings">
          {Object.entries(step.settings).filter(([key]) => !['label', 'category', 'pythonCode'].includes(key)).map(([key, value]) => (
            <label key={key}>
              <span>{key}</span>
              {typeof value === 'number' ? (
                <input aria-label={key} onChange={(event) => onSettingChange(key, Number(event.target.value))} type="number" value={value} />
              ) : typeof value === 'boolean' ? (
                <input aria-label={key} checked={value} onChange={(event) => onSettingChange(key, event.target.checked)} type="checkbox" />
              ) : (
                <input
                  aria-label={key}
                  onChange={(event) => onSettingChange(key, Array.isArray(value) ? event.target.value.split(',').map((item) => item.trim()) : event.target.value)}
                  type="text"
                  value={formatSetting(value)}
                />
              )}
            </label>
          ))}
        </div>
        <button aria-label="Delete selected tokenizer atom" className="delete-atom-button" onClick={onDelete}><Trash2 size={13} />Delete atom</button>
      </section>
      <section className="inspector-section contract-section">
        <div className="contract-row"><span><Check size={12} />Registry lowering</span><strong>Defined</strong></div>
        <div className="contract-row"><span>Execution</span><strong>{result?.status ?? 'pending'}</strong></div>
        {result?.summary && <p className="execution-summary">{result.summary}</p>}
        {result?.error && <p className="execution-error">{result.error}</p>}
      </section>
    </>
  )
}
