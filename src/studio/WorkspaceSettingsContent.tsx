import { ChevronDown, FolderPlus, GitCompareArrows, RotateCcw, Save, Search, Trash2 } from 'lucide-react'
import { useMemo, useState } from 'react'
import type { ArchitectureGraph } from '../core/ir'

interface WorkspaceSettingsContentProps {
  comparisonPresets: ArchitectureGraph[]
  currentGraph: ArchitectureGraph
  currentLabel: string
  error?: string
  name: string
  onAddComparison(preset: ArchitectureGraph): void
  onCreateBlank(): void
  onDeleteWorkspace(preset: ArchitectureGraph): void
  onLoadWorkspace(preset: ArchitectureGraph): void
  onNameChange(value: string): void
  onReset(): void
  onSave(): void
  presetLabel(preset: ArchitectureGraph): string
  resetConfirming: boolean
  resetDisabled: boolean
  savedWorkspaces: ArchitectureGraph[]
}

export function WorkspaceSettingsContent({
  comparisonPresets,
  currentGraph,
  currentLabel,
  error,
  name,
  onAddComparison,
  onCreateBlank,
  onDeleteWorkspace,
  onLoadWorkspace,
  onNameChange,
  onReset,
  onSave,
  presetLabel,
  resetConfirming,
  resetDisabled,
  savedWorkspaces,
}: WorkspaceSettingsContentProps) {
  const [comparisonQuery, setComparisonQuery] = useState('')
  const visiblePresets = useMemo(() => {
    const query = comparisonQuery.trim().toLocaleLowerCase()
    return query ? comparisonPresets.filter((preset) => presetLabel(preset).toLocaleLowerCase().includes(query)) : comparisonPresets
  }, [comparisonPresets, comparisonQuery, presetLabel])

  return <div className="workspace-settings-content">
    <section className="workspace-settings-primary">
      <div className="workspace-current-card">
        <div className="workspace-current-copy">
          <small>Current workspace</small>
          <strong>{currentLabel}</strong>
          <span>{currentGraph.nodes.length} cards · saved automatically</span>
        </div>
        <button aria-label={`Restore ${currentLabel}`} className={`workspace-reset-button${resetConfirming ? ' confirm-reset' : ''}`} disabled={resetDisabled} onClick={onReset} title="Restore the original workspace" type="button"><RotateCcw size={13} />{resetConfirming ? 'Confirm restore' : 'Restore'}</button>
      </div>

      <div className="workspace-save-card">
        <label><span>Save a reusable copy</span><input aria-label="New model preset name" onChange={(event) => onNameChange(event.target.value)} placeholder="Workspace name" value={name} /></label>
        <button aria-label={`Save a named copy of ${currentLabel}`} className="workspace-primary-action" onClick={onSave} type="button"><Save size={14} />Save copy</button>
        {error && <p role="alert">{error}</p>}
      </div>

      <button aria-label="Create and open a blank workspace" className="workspace-new-button" onClick={onCreateBlank} type="button"><FolderPlus size={14} /><span><strong>Create and open a blank workspace</strong><small>Your current draft stays saved.</small></span></button>
    </section>

    <section className="workspace-saved-section">
      <header><div><small>Your library</small><strong>Saved workspaces</strong></div><span>{savedWorkspaces.length}</span></header>
      {savedWorkspaces.length === 0 ? <div className="workspace-empty-library"><strong>No saved copy yet</strong><p>Name the current graph above when you want to reuse it later.</p></div> : <div className="workspace-saved-list">
        {savedWorkspaces.map((preset) => <div className={currentGraph.id === preset.id ? 'active' : ''} key={preset.id}>
          <button aria-label={`Load preset ${preset.name}`} aria-pressed={currentGraph.id === preset.id} onClick={() => onLoadWorkspace(preset)} type="button"><strong>{preset.name}</strong><small>{preset.nodes.length} cards</small></button>
          <button aria-label={`Delete preset ${preset.name}`} onClick={() => onDeleteWorkspace(preset)} title="Delete workspace" type="button"><Trash2 size={13} /></button>
        </div>)}
      </div>}
    </section>

    <details className="workspace-comparison-picker">
      <summary><span><GitCompareArrows size={15} /><span><strong>Add an architecture for comparison</strong><small>Place another complete graph beside this workspace.</small></span></span><ChevronDown size={15} /></summary>
      <div className="workspace-comparison-content">
        <label><Search size={13} /><input aria-label="Search comparison presets" onChange={(event) => setComparisonQuery(event.target.value)} placeholder="Search architectures" value={comparisonQuery} /></label>
        <div className="workspace-comparison-grid">
          {visiblePresets.map((preset) => <button aria-label={`Add ${presetLabel(preset)} beside current graph`} key={`compare-${preset.id}`} onClick={() => onAddComparison(preset)} type="button"><span><strong>{presetLabel(preset)}</strong><small>{preset.nodes.length} cards</small></span><b>+</b></button>)}
          {visiblePresets.length === 0 && <p>No architecture matches “{comparisonQuery}”.</p>}
        </div>
      </div>
    </details>
  </div>
}
