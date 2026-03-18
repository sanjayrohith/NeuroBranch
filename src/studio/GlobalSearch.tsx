import { Search, X } from 'lucide-react'
import { StudioDialog } from './StudioDialog'
import { StudioIconButton } from './StudioControls'

export interface GlobalSearchResult {
  id: string
  kind: string
  label: string
  description: string
}

export function GlobalSearch({ onClose, onQueryChange, onSelect, placeholder, query, results }: { onClose(): void; onQueryChange(query: string): void; onSelect(result: GlobalSearchResult): void; placeholder: string; query: string; results: GlobalSearchResult[] }) {
  return <StudioDialog ariaLabel="Search cards" backdropClassName="card-search-backdrop" className="card-search-modal" onClose={onClose}>
    <header><span><Search size={14} />Find a card</span><StudioIconButton label="Close card search" onClick={onClose}><X size={14} /></StudioIconButton></header>
    <input autoFocus aria-label="Natural language card search" onChange={(event) => onQueryChange(event.target.value)} placeholder={placeholder} value={query} />
    <div className="card-search-results">
      {query && results.length === 0 && <p>No matching native card.</p>}
      {results.map((result) => <button key={`${result.kind}-${result.id}`} onClick={() => onSelect(result)} type="button"><strong>{result.label}</strong><small>{result.description}</small></button>)}
    </div>
  </StudioDialog>
}
