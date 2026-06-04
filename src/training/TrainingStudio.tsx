import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Blocks, Braces, Cpu, Pencil, Play, Plus, Settings2, SplitSquareHorizontal, Trash2 } from 'lucide-react'
import { compileOptimizer, createOptimizerConfig, optimizerRegistry, type OptimizerConfig, type OptimizerDefinition, type OptimizerValue } from '../core/optimizer-ir'
import { OptimizerCreator } from './OptimizerCreator'
import { parseTrainingWorkspace } from './training-workspace'
import { PythonCodePreview } from '../model/PythonCodeEditor'
import { StudioEditor, StudioInspector, StudioLibrary, StudioStatusbar, StudioToolbar, StudioViewSwitcher, StudioWorkspace } from '../studio/StudioShell'
import { StudioLibraryItem } from '../studio/StudioLibraryParts'
import { InspectorSection, InspectorSelection } from '../studio/StudioInspectorParts'
import { StudioCanvasPanel, StudioCodePanel, StudioPanelTab } from '../studio/StudioPanels'
import { StudioContextMenu, StudioContextMenuItem } from '../studio/StudioContextMenu'

function formatValue(value: OptimizerValue): string {
  if (Array.isArray(value)) return value.map((item) => item === null ? 'None' : String(item)).join(', ')
  if (value === null) return 'None'
  return String(value)
}

function optimizerReference(definition: OptimizerDefinition): string {
  return definition.composition ? definition.torchClass : `torch.optim.${definition.torchClass}`
}

export function TrainingStudio({ onCatalogChange = () => undefined, onRequestedOptimizerHandled = () => undefined, requestedOptimizer }: { onCatalogChange?: (optimizers: OptimizerDefinition[]) => void; onRequestedOptimizerHandled?: () => void; requestedOptimizer?: { optimizerId: string; requestId: number } }) {
  const [config, setConfig] = useState<OptimizerConfig>(() => createOptimizerConfig('adamw'))
  const [view, setView] = useState<'graph' | 'pytorch' | 'split'>('graph')
  const [customOptimizers, setCustomOptimizers] = useState<OptimizerDefinition[]>([])
  const [creatorOpen, setCreatorOpen] = useState(false)
  const [editingOptimizer, setEditingOptimizer] = useState<OptimizerDefinition>()
  const [storageReady, setStorageReady] = useState(false)
  const [webAuthenticated, setWebAuthenticated] = useState(false)
  const [optimizerMenu, setOptimizerMenu] = useState<{ optimizerId: string; x: number; y: number }>()
  const interactedRef = useRef(false)
  const definitions = useMemo(() => ({ ...optimizerRegistry, ...Object.fromEntries(customOptimizers.map((optimizer) => [optimizer.id, optimizer])) }), [customOptimizers])
  const definition = definitions[config.kind]
  const code = useMemo(() => ['import torch', '', compileOptimizer(config, 'model.parameters()', definitions), ''].join('\n'), [config, definitions])

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      let stored
      if (window.labo?.runtime === 'electron' && window.labo.loadDesktopState) {
        stored = parseTrainingWorkspace(await window.labo.loadDesktopState('training'))
      } else if (window.labo?.runtime === 'web' && window.labo.loadWebWorkspace) {
        const result = await window.labo.loadWebWorkspace()
        const authenticated = Boolean(result && typeof result === 'object' && result.authenticated)
        if (!cancelled) setWebAuthenticated(authenticated)
        if (authenticated) stored = parseTrainingWorkspace(result.training)
      }
      if (!cancelled && stored && !interactedRef.current) {
        setCustomOptimizers(stored.customOptimizers)
        setConfig(stored.config)
      }
      if (!cancelled) setStorageReady(true)
    }
    void load().catch(() => { if (!cancelled) setStorageReady(true) })
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    if (!storageReady) return
    const workspace = { config, customOptimizers, updatedAt: Date.now() }
    if (window.labo?.runtime === 'electron' && window.labo.saveDesktopState) {
      void window.labo.saveDesktopState('training', workspace)
    } else if (window.labo?.runtime === 'web' && webAuthenticated && window.labo.saveWebWorkspace) {
      void window.labo.saveWebWorkspace({ training: workspace })
    }
  }, [config, customOptimizers, storageReady, webAuthenticated])

  useEffect(() => {
    if (!optimizerMenu) return
    const dismiss = (event: PointerEvent) => { if (!(event.target as HTMLElement | null)?.closest('.optimizer-context-menu')) setOptimizerMenu(undefined) }
    const dismissOnEscape = (event: KeyboardEvent) => { if (event.key === 'Escape') setOptimizerMenu(undefined) }
    window.addEventListener('pointerdown', dismiss)
    window.addEventListener('keydown', dismissOnEscape)
    return () => { window.removeEventListener('pointerdown', dismiss); window.removeEventListener('keydown', dismissOnEscape) }
  }, [optimizerMenu])

  const selectOptimizer = useCallback((kind: string) => {
    interactedRef.current = true
    setConfig(createOptimizerConfig(kind, {}, definitions))
  }, [definitions])
  const updateSetting = (key: string, value: OptimizerValue) => {
    interactedRef.current = true
    setConfig((current) => ({ ...current, settings: { ...current.settings, [key]: value } }))
  }
  const saveOptimizer = (optimizer: OptimizerDefinition) => {
    interactedRef.current = true
    const nextDefinitions = { ...definitions, [optimizer.id]: optimizer }
    setCustomOptimizers((current) => current.some((candidate) => candidate.id === optimizer.id)
      ? current.map((candidate) => candidate.id === optimizer.id ? optimizer : candidate)
      : [...current, optimizer])
    setConfig((current) => current.kind === optimizer.id
      ? createOptimizerConfig(optimizer.id, optimizer.defaults, nextDefinitions)
      : createOptimizerConfig(optimizer.id, {}, nextDefinitions))
    setCreatorOpen(false)
    setEditingOptimizer(undefined)
  }
  const openOptimizerEditor = (optimizerId: string) => {
    const optimizer = customOptimizers.find((candidate) => candidate.id === optimizerId)
    if (!optimizer) return
    setEditingOptimizer(optimizer)
    setCreatorOpen(true)
    setOptimizerMenu(undefined)
  }
  const deleteOptimizer = (optimizerId: string) => {
    interactedRef.current = true
    setCustomOptimizers((current) => current.filter((optimizer) => optimizer.id !== optimizerId))
    if (config.kind === optimizerId) setConfig(createOptimizerConfig('adamw'))
    setOptimizerMenu(undefined)
  }

  useEffect(() => onCatalogChange(customOptimizers), [customOptimizers, onCatalogChange])

  useEffect(() => {
    if (!requestedOptimizer || !storageReady || !definitions[requestedOptimizer.optimizerId]) return
    selectOptimizer(requestedOptimizer.optimizerId)
    onRequestedOptimizerHandled()
  }, [definitions, onRequestedOptimizerHandled, requestedOptimizer, selectOptimizer, storageReady])

  return <>
    <StudioToolbar meta={<span><span className="status-dot" /> Training IR synchronized</span>}>
      <StudioViewSwitcher<'graph' | 'pytorch' | 'split'> ariaLabel="Training editor view" onChange={setView} options={[{ id: 'graph', label: 'Training graph', icon: <Blocks size={14} /> }, { id: 'pytorch', label: 'PyTorch', icon: <Braces size={14} /> }, { id: 'split', label: 'Split', icon: <SplitSquareHorizontal size={14} /> }]} value={view} />
    </StudioToolbar>

    {creatorOpen ? <OptimizerCreator definition={editingOptimizer} onCancel={() => { setCreatorOpen(false); setEditingOptimizer(undefined) }} onSave={saveOptimizer} view={view} /> : <StudioWorkspace className="training-workspace">
      <StudioLibrary heading="OPTIMIZERS" icon={<Settings2 size={14} />}>
        <section className="block-group optimizer-library">
          <h3>PyTorch 2.13</h3>
          {Object.values(optimizerRegistry).map((optimizer) => <StudioLibraryItem aria-label={`Use ${optimizer.label}`} glyph={<span className="block-glyph glyph-objective" />} key={optimizer.id} onClick={() => selectOptimizer(optimizer.id)}>{optimizer.label}</StudioLibraryItem>)}
          <StudioLibraryItem aria-label="Create optimizer" className="optimizer-create-button" glyph={<Plus size={13} />} onClick={() => { setEditingOptimizer(undefined); setCreatorOpen(true); setView('graph') }}>Create optimizer</StudioLibraryItem>
          {customOptimizers.length > 0 && <h3>Created</h3>}
          {customOptimizers.map((optimizer) => <StudioLibraryItem aria-label={`Use ${optimizer.label}`} glyph={<span className="block-glyph glyph-objective" />} key={optimizer.id} onClick={() => selectOptimizer(optimizer.id)} onContextMenu={(event) => { event.preventDefault(); setOptimizerMenu({ optimizerId: optimizer.id, x: event.clientX, y: event.clientY }) }} title="Right-click to edit or delete">{optimizer.label}</StudioLibraryItem>)}
        </section>
      </StudioLibrary>

      <StudioEditor className={`view-${view === 'graph' ? 'blocks' : view}`}>
        {view !== 'pytorch' && <StudioCanvasPanel tab={<StudioPanelTab icon={<Blocks size={13} />}>training.optimizer</StudioPanelTab>}>
          <div className="training-canvas">
            <article aria-label={`Optimizer card ${definition.label}`} className="optimizer-block selected" onContextMenu={(event) => { if (!config.kind.startsWith('custom-')) return; event.preventDefault(); setOptimizerMenu({ optimizerId: config.kind, x: event.clientX, y: event.clientY }) }} onDoubleClick={() => openOptimizerEditor(config.kind)}>
              <header><span className="node-type">OPTIMIZER</span><strong>{definition.label}</strong><small>{optimizerReference(definition)}</small></header>
              <div className="optimizer-inline-editor">
                {Object.entries(config.settings).map(([key, value]) => <label key={key}>
                  <span>{key}</span>
                  {typeof value === 'boolean'
                    ? <input aria-label={`${definition.label} ${key}`} checked={value} onChange={(event) => updateSetting(key, event.target.checked)} type="checkbox" />
                    : <input aria-label={`${definition.label} ${key}`} onChange={(event) => {
                        if (Array.isArray(value)) updateSetting(key, event.target.value.split(',').map((part) => part.trim() === 'None' ? null : Number(part)))
                        else if (value === null || typeof value === 'string') updateSetting(key, event.target.value === 'None' ? null : event.target.value)
                        else updateSetting(key, Number(event.target.value))
                      }} type={typeof value === 'number' ? 'number' : 'text'} value={formatValue(value)} />}
                </label>)}
              </div>
            </article>
          </div>
        </StudioCanvasPanel>}
        {view !== 'graph' && <StudioCodePanel tab={<StudioPanelTab icon={<Braces size={13} />} status="GENERATED">optimizer.py</StudioPanelTab>}>
          <PythonCodePreview value={code} />
        </StudioCodePanel>}
      </StudioEditor>

      <StudioInspector heading="TRAINING INSPECTOR" icon={<Cpu size={14} />}>
        <InspectorSection title="Selection"><InspectorSelection detail={definition.notes ?? optimizerReference(definition)} icon={<Play size={15} />} title={definition.label} /></InspectorSection>
      </StudioInspector>
    </StudioWorkspace>}
    <StudioStatusbar><span><span className="status-dot" /> Training IR valid</span><span>optimizer · {Object.keys(config.settings).length} settings</span><span className="status-spacer" /><span>PyTorch 2.13</span></StudioStatusbar>
    {optimizerMenu && (() => {
      const optimizer = customOptimizers.find((candidate) => candidate.id === optimizerMenu.optimizerId)
      if (!optimizer) return null
      return <StudioContextMenu className="optimizer-context-menu" position={optimizerMenu}>
        <StudioContextMenuItem onClick={() => openOptimizerEditor(optimizer.id)}><Pencil size={12} />Edit {optimizer.label}</StudioContextMenuItem>
        <StudioContextMenuItem onClick={() => deleteOptimizer(optimizer.id)}><Trash2 size={12} />Delete {optimizer.label}</StudioContextMenuItem>
      </StudioContextMenu>
    })()}
  </>
}
