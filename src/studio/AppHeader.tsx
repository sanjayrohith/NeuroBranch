import { Search, Settings2 } from 'lucide-react'
import { LaboMark } from '../LaboMark'
import { StudioButton } from './StudioControls'

export type StudioWorkspaceId = 'model' | 'training' | 'tokenizer'

const studios: Array<{ id: StudioWorkspaceId; label: string }> = [
  { id: 'model', label: 'Model Studio' },
  { id: 'training', label: 'Training Studio' },
  { id: 'tokenizer', label: 'Tokenizer Studio' },
]

export function AppHeader({ onOpenSearch, onOpenSettings, onWorkspaceChange, searchDisabled = false, searchLabel, searchShortcut, settingsOpen, workspace }: { onOpenSearch(): void; onOpenSettings(): void; onWorkspaceChange(workspace: StudioWorkspaceId): void; searchDisabled?: boolean; searchLabel: string; searchShortcut: string; settingsOpen: boolean; workspace: StudioWorkspaceId }) {
  return <header className="topbar">
    <div className="brand"><span className="brand-mark"><LaboMark /></span><strong>LABO AI</strong><span className="alpha-pill">BETA</span></div>
    <nav aria-label="LABO studios" className="studio-navigation">{studios.map((studio) => <button aria-pressed={workspace === studio.id} key={studio.id} onClick={() => onWorkspaceChange(studio.id)} type="button">{studio.label}</button>)}</nav>
    <div className="header-actions">
      <StudioButton aria-label={searchLabel} className="ghost-button" disabled={searchDisabled} onClick={onOpenSearch} title={searchDisabled ? 'Switch to Add blocks to search the card library' : undefined}><Search size={14} />Search<kbd>{searchShortcut}</kbd></StudioButton>
      <StudioButton aria-label="Open LABO settings" aria-pressed={settingsOpen} className="codex-button" onClick={onOpenSettings} variant="accent"><Settings2 size={14} />Settings</StudioButton>
    </div>
  </header>
}
