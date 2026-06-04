import { useEffect, useMemo, useState } from 'react'
import { Blocks, Braces, Check, Settings2, X } from 'lucide-react'
import type { AtomicPlayerSnapshot } from '../core/atomic-player'
import type { ArchitectureGraph } from '../core/ir'
import { compileOptimizer, optimizerRegistry, type OptimizerComposition, type OptimizerDefinition, type OptimizerValue } from '../core/optimizer-ir'
import { GraphCanvas } from '../model/GraphCanvas'
import { PythonCodePreview } from '../model/PythonCodeEditor'

const optimizerRecipes = ['adamw', 'adam', 'sgd', 'muon'] as const
type OptimizerRecipe = typeof optimizerRecipes[number]
type OptimizerAtom = 'gradient' | 'momentum' | 'adaptive' | 'normalize' | 'orthogonalize' | 'decay' | 'update' | 'parameters'

const defaultComposition: OptimizerComposition = { kind: 'composed', momentum: true, adaptiveScale: true, normalizeGradient: false, weightDecay: true, decoupledWeightDecay: true }
const recipeAtoms: Record<OptimizerRecipe, OptimizerAtom[]> = {
  adamw: ['gradient', 'momentum', 'adaptive', 'decay', 'update', 'parameters'],
  adam: ['gradient', 'momentum', 'adaptive', 'decay', 'update', 'parameters'],
  sgd: ['gradient', 'momentum', 'decay', 'update', 'parameters'],
  muon: ['gradient', 'momentum', 'orthogonalize', 'decay', 'update', 'parameters'],
}

function atomsForOptimizer(definition: OptimizerDefinition): OptimizerAtom[] {
  const atoms: OptimizerAtom[] = ['gradient']
  if ('betas' in definition.defaults || 'momentum' in definition.defaults || 'rho' in definition.defaults) atoms.push('momentum')
  if ('betas' in definition.defaults || 'alpha' in definition.defaults || 'beta2_decay' in definition.defaults || definition.torchClass === 'Adagrad' || definition.torchClass === 'Adadelta') atoms.push('adaptive')
  if (definition.torchClass === 'Muon') atoms.push('orthogonalize')
  if ('weight_decay' in definition.defaults) atoms.push('decay')
  atoms.push('update', 'parameters')
  return atoms
}
const atomLabels: Record<OptimizerAtom, { title: string; detail: string }> = {
  gradient: { title: 'Gradient input', detail: 'parameter.grad' },
  momentum: { title: 'Momentum', detail: 'first-moment state' },
  adaptive: { title: 'Adaptive scale', detail: 'squared-gradient state' },
  normalize: { title: 'Normalize update', detail: 'unit-norm direction' },
  orthogonalize: { title: 'Orthogonalize', detail: 'Newton–Schulz update' },
  decay: { title: 'Weight decay', detail: 'parameter regularization' },
  update: { title: 'Parameter update', detail: 'apply learning rate' },
  parameters: { title: 'Updated parameters', detail: 'in-place output' },
}

const optimizerGraphBase: Omit<ArchitectureGraph, 'nodes' | 'edges'> = {
  id: 'optimizer-builder',
  name: 'Optimizer builder',
  architecture: 'custom',
  config: { hiddenSize: 1, queryHeads: 1, keyValueHeads: 1, headDim: 1 },
  contracts: { causal: false, preservesGqaAtZeroGate: false, sdpaCompatible: false, contextualValue: false },
}

function optimizerNodeId(atom: OptimizerAtom): string {
  return `optimizer-${atom}`
}

function graphForAtoms(atoms: OptimizerAtom[], current?: ArchitectureGraph): ArchitectureGraph {
  const previous = new Map(current?.nodes.map((node) => [node.id, node.position]))
  const nodes = atoms.map((atom, index) => ({
    id: optimizerNodeId(atom),
    kind: 'custom-pytorch' as const,
    label: atomLabels[atom].title,
    role: 'hidden' as const,
    position: previous.get(optimizerNodeId(atom)) ?? { x: 360, y: 110 + index * 120 },
    attributes: { inputRole: 'hidden', optimizerAtom: atom, detail: atomLabels[atom].detail },
    code: 'nn.Identity()',
  }))
  return {
    ...optimizerGraphBase,
    nodes,
    edges: atoms.slice(0, -1).map((atom, index) => ({
      id: `${optimizerNodeId(atom)}-${optimizerNodeId(atoms[index + 1])}`,
      source: optimizerNodeId(atom),
      sourcePort: 'output',
      target: optimizerNodeId(atoms[index + 1]),
      targetPort: 'input',
    })),
  }
}

function atomFromNodeId(nodeId: string): OptimizerAtom | undefined {
  const atom = nodeId.replace(/^optimizer-/, '') as OptimizerAtom
  return atom in atomLabels ? atom : undefined
}

function recipeFor(definition?: OptimizerDefinition): string {
  return Object.values(optimizerRegistry).find((candidate) => candidate.torchClass === definition?.torchClass)?.id ?? 'adamw'
}

function settingsForComposition(composition: OptimizerComposition, current: Record<string, OptimizerValue> = {}): Record<string, OptimizerValue> {
  const numeric = (key: string, fallback: number) => typeof current[key] === 'number' ? current[key] as number : fallback
  return {
    lr: numeric('lr', 0.001),
    ...(composition.momentum ? { beta1: numeric('beta1', 0.9) } : {}),
    ...(composition.adaptiveScale ? { beta2: numeric('beta2', 0.999) } : {}),
    ...((composition.adaptiveScale || composition.normalizeGradient) ? { eps: numeric('eps', 1e-8) } : {}),
    ...(composition.weightDecay === false ? {} : { weight_decay: numeric('weight_decay', 0.01) }),
  }
}

export function OptimizerCreator({ definition, onCancel, onSave, view }: { definition?: OptimizerDefinition; onCancel(): void; onSave(definition: OptimizerDefinition): void; view: 'graph' | 'pytorch' | 'split' }) {
  const initialRecipe = recipeFor(definition)
  const [mode, setMode] = useState<'recipe' | 'custom'>(definition?.composition ? 'custom' : 'recipe')
  const [recipe, setRecipe] = useState(initialRecipe)
  const [composition, setComposition] = useState<OptimizerComposition>(definition?.composition ?? defaultComposition)
  const [label, setLabel] = useState(definition?.label ?? (definition?.composition ? 'My optimizer' : 'My AdamW optimizer'))
  const [torchClass, setTorchClass] = useState(definition?.torchClass ?? (definition?.composition ? 'LaboOptimizer' : optimizerRegistry[initialRecipe].torchClass))
  const [settings, setSettings] = useState<Record<string, OptimizerValue>>(() => ({ ...(definition?.defaults ?? optimizerRegistry[initialRecipe].defaults) }))
  const [selectedAtom, setSelectedAtom] = useState<OptimizerAtom>('momentum')
  const [error, setError] = useState('')

  const atoms = mode === 'recipe' ? (recipeAtoms[recipe as OptimizerRecipe] ?? atomsForOptimizer(optimizerRegistry[recipe])) : [
    'gradient' as const,
    ...(composition.momentum ? ['momentum' as const] : []),
    ...(composition.adaptiveScale ? ['adaptive' as const] : []),
    ...(composition.normalizeGradient ? ['normalize' as const] : []),
    ...(composition.weightDecay === false ? [] : ['decay' as const]),
    'update' as const,
    'parameters' as const,
  ]
  const atomsKey = atoms.join(':')
  const [graph, setGraph] = useState<ArchitectureGraph>(() => graphForAtoms(atoms))
  const playerSnapshot = useMemo<AtomicPlayerSnapshot>(() => ({
    status: 'idle',
    currentAtomId: graph.nodes[0]?.id,
    results: graph.nodes.map((node) => ({ atomId: node.id, status: 'pending' })),
  }), [graph.nodes])

  useEffect(() => {
    setGraph((current) => graphForAtoms(atoms, current))
  // The serialized atom sequence is the optimizer IR boundary for the visual graph.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [atomsKey])
  const preview = useMemo(() => {
    const previewDefinition: OptimizerDefinition = { id: 'preview', label, torchClass, defaults: settings, composition: mode === 'custom' ? composition : undefined }
    return compileOptimizer({ id: 'preview-1', kind: 'preview', settings }, 'model.parameters()', { preview: previewDefinition })
  }, [composition, label, mode, settings, torchClass])

  const updateSetting = (key: string, value: OptimizerValue) => setSettings((current) => ({ ...current, [key]: value }))
  const numberSetting = (key: string, fallback = 0) => typeof settings[key] === 'number' ? settings[key] as number : fallback
  const advancedSettings = Object.entries(settings).filter(([key]) => !['lr', 'betas', 'beta1', 'beta2', 'momentum', 'weight_decay', 'eps'].includes(key))
  const selectRecipe = (nextRecipe: string) => {
    const next = optimizerRegistry[nextRecipe]
    const nextAtoms = recipeAtoms[nextRecipe as OptimizerRecipe] ?? atomsForOptimizer(next)
    setMode('recipe')
    setRecipe(nextRecipe)
    setTorchClass(next.torchClass)
    setSettings({ ...next.defaults })
    setLabel((current) => current === 'My optimizer' || current.startsWith('My ') ? `My ${next.label} optimizer` : current)
    setSelectedAtom(nextAtoms[1] ?? 'update')
  }
  const toggleAtom = (atom: 'momentum' | 'adaptive' | 'normalize' | 'decay') => {
    const next = {
      ...composition,
      momentum: atom === 'momentum' ? !composition.momentum : composition.momentum,
      adaptiveScale: atom === 'adaptive' ? !composition.adaptiveScale : composition.adaptiveScale,
      normalizeGradient: atom === 'normalize' ? !composition.normalizeGradient : composition.normalizeGradient,
      weightDecay: atom === 'decay' ? composition.weightDecay === false : composition.weightDecay,
    }
    setMode('custom')
    setTorchClass('LaboOptimizer')
    setLabel((current) => current.startsWith('My ') ? 'My optimizer' : current)
    setComposition(next)
    setSettings((current) => settingsForComposition(next, current))
    setSelectedAtom(atom)
  }
  const updateBeta = (index: 0 | 1, value: number) => {
    const current = Array.isArray(settings.betas) ? settings.betas : [0.9, 0.999]
    updateSetting('betas', index === 0 ? [value, Number(current[1])] : [Number(current[0]), value])
  }
  const submit = () => {
    const name = label.trim()
    if (!name) return setError('Give the optimizer a name.')
    const idBase = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'optimizer'
    onSave({
      id: definition?.id ?? `custom-${idBase}-${Date.now().toString(36)}`,
      label: name,
      torchClass,
      defaults: settings,
      notes: mode === 'custom' ? 'Composed from LABO optimizer atoms.' : `Custom torch.optim.${torchClass} preset.`,
      composition: mode === 'custom' ? composition : undefined,
    })
  }

  const inspector = (() => {
    if (selectedAtom === 'momentum') {
      if (Array.isArray(settings.betas)) return <label><span>β1 coefficient</span><input aria-label="Optimizer beta 1" max="1" min="0" onChange={(event) => updateBeta(0, Number(event.target.value))} step="any" type="number" value={Number(settings.betas[0])} /></label>
      if ('beta1' in settings) return <label><span>β1 coefficient</span><input aria-label="Optimizer beta 1" max="1" min="0" onChange={(event) => updateSetting('beta1', Number(event.target.value))} step="any" type="number" value={numberSetting('beta1', 0.9)} /></label>
      return <label><span>Momentum</span><input aria-label="Optimizer momentum" max="1" min="0" onChange={(event) => updateSetting('momentum', Number(event.target.value))} step="any" type="number" value={numberSetting('momentum')} /></label>
    }
    if (selectedAtom === 'adaptive') {
      const beta2 = Array.isArray(settings.betas) ? Number(settings.betas[1]) : numberSetting('beta2', 0.999)
      return <><label><span>β2 coefficient</span><input aria-label="Optimizer beta 2" max="1" min="0" onChange={(event) => Array.isArray(settings.betas) ? updateBeta(1, Number(event.target.value)) : updateSetting('beta2', Number(event.target.value))} step="any" type="number" value={beta2} /></label><label><span>Epsilon</span><input aria-label="Optimizer Epsilon" onChange={(event) => updateSetting('eps', Number(event.target.value))} step="any" type="number" value={numberSetting('eps', 1e-8)} /></label></>
    }
    if (selectedAtom === 'decay') return <><label><span>Weight decay</span><input aria-label="Optimizer weight decay" onChange={(event) => updateSetting('weight_decay', Number(event.target.value))} step="any" type="number" value={numberSetting('weight_decay')} /></label>{mode === 'custom' && <label className="optimizer-inspector-toggle"><input aria-label="Decoupled weight decay" checked={composition.decoupledWeightDecay} onChange={(event) => setComposition((current) => ({ ...current, decoupledWeightDecay: event.target.checked }))} type="checkbox" /><span>Decoupled update</span></label>}</>
    if (selectedAtom === 'update') return <label><span>Learning rate</span><input aria-label="Optimizer learning rate" onChange={(event) => updateSetting('lr', Number(event.target.value))} step="any" type="number" value={numberSetting('lr', 0.001)} /></label>
    return <p>This atom has no tunable setting.</p>
  })()

  return <div aria-label={definition ? 'Edit optimizer' : 'Create optimizer'} className="workspace-grid optimizer-builder-workspace" role="form">
    <aside className="block-library">
      <div className="panel-heading"><Blocks size={14} /><span>OPTIMIZER ATOMS</span></div>
      <section className="block-group optimizer-builder-library">
        <h3>PyTorch 2.13</h3>
        {Object.values(optimizerRegistry).map((candidate) => <button aria-pressed={mode === 'recipe' && recipe === candidate.id} className="library-block" key={candidate.id} onClick={() => selectRecipe(candidate.id)} type="button"><span className="block-glyph glyph-objective" />{candidate.label}</button>)}
        <h3>Update atoms</h3>
        {(['momentum', 'adaptive', 'normalize', 'decay'] as const).map((atom) => {
          const active = atoms.includes(atom)
          return <button aria-label={`${active ? 'Remove' : 'Add'} ${atomLabels[atom].title}`} aria-pressed={active} className="library-block" key={atom} onClick={() => toggleAtom(atom)} type="button"><span className="block-glyph glyph-transforms" />{atomLabels[atom].title}</button>
        })}
      </section>
    </aside>

    <section className={`editor-grid optimizer-builder-editor view-${view === 'graph' ? 'blocks' : view}`}>
      {view !== 'pytorch' && <GraphCanvas
        graph={graph}
        onDeleteNode={(nodeId) => {
          const atom = atomFromNodeId(nodeId)
          if (atom === 'momentum' || atom === 'adaptive' || atom === 'normalize' || atom === 'decay') toggleAtom(atom)
        }}
        onDropAtom={() => undefined}
        onDropCustom={() => undefined}
        onDropInput={() => undefined}
        playerSnapshot={playerSnapshot}
        selectedNodeId={optimizerNodeId(selectedAtom)}
        setGraph={setGraph}
        setSelectedNodeId={(nodeId) => {
          const atom = atomFromNodeId(nodeId)
          if (atom) setSelectedAtom(atom)
        }}
      />}
      {view !== 'graph' && <div className="code-panel optimizer-builder-code-panel">
        <div className="panel-tab"><Braces size={13} /> optimizer.py <span>GENERATED</span></div>
        <PythonCodePreview className="optimizer-builder-code" value={preview} />
      </div>}
    </section>

    <aside className="inspector optimizer-builder-inspector">
      <div className="panel-heading"><Settings2 size={14} /><span>ATOM INSPECTOR</span></div>
      <section className="inspector-section optimizer-selection-section">
        <div className="section-title">Selection</div>
        <div className="selection-card optimizer-selection-card"><span className="selection-icon"><Settings2 size={14} /></span><div><strong>{atomLabels[selectedAtom].title}</strong><small>{atomLabels[selectedAtom].detail}</small></div></div>
      </section>
      <section className="inspector-section optimizer-identity-settings">
        <div className="section-title">Optimizer</div>
        <label><span>Name</span><input aria-label="Optimizer name" onChange={(event) => setLabel(event.target.value)} value={label} /></label>
        <div className="optimizer-reference"><span>Runtime</span><code>{mode === 'custom' ? 'LaboOptimizer' : `torch.optim.${torchClass}`}</code></div>
      </section>
      <section className="inspector-section optimizer-atom-settings"><div className="section-title">Parameters</div>{inspector}{advancedSettings.length > 0 && <details className="optimizer-advanced-settings"><summary>More parameters</summary>{advancedSettings.map(([key, value]) => <label key={key}><span>{key.replaceAll('_', ' ')}</span>{typeof value === 'boolean'
        ? <input aria-label={`Optimizer ${key === 'fused' ? 'Fused implementation' : key}`} checked={value} onChange={(event) => updateSetting(key, event.target.checked)} type="checkbox" />
        : <input aria-label={`Optimizer ${key}`} onChange={(event) => updateSetting(key, Array.isArray(value) ? event.target.value.split(',').map((part) => part.trim() === 'None' ? null : Number(part)) : value === null ? event.target.value === 'None' ? null : event.target.value : typeof value === 'number' ? Number(event.target.value) : event.target.value)} type={typeof value === 'number' ? 'number' : 'text'} value={Array.isArray(value) ? value.map((item) => item ?? 'None').join(', ') : value ?? 'None'} />}</label>)}</details>}</section>
      {error && <p className="optimizer-creator-error" role="alert">{error}</p>}
      <div className="optimizer-builder-actions"><button onClick={onCancel} type="button"><X size={12} />Cancel</button><button onClick={submit} type="button"><Check size={12} />{definition ? 'Save' : 'Create'}</button></div>
    </aside>
  </div>
}
