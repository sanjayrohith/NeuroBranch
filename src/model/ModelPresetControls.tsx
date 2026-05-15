interface ModelPresetOption {
  id: string
  name: string
}

export function ModelPresetMenu({ activeId, activeName, builtIns, labels, onSelect, userPresets }: { activeId: string; activeName: string; builtIns: ModelPresetOption[]; labels: Record<string, string>; onSelect(id: string): void; userPresets: ModelPresetOption[] }) {
  const option = (preset: ModelPresetOption) => <button aria-pressed={activeId === preset.id} key={preset.id} onClick={(event) => {
    onSelect(preset.id)
    event.currentTarget.closest('details')?.removeAttribute('open')
  }} type="button">{labels[preset.id] ?? preset.name}</button>

  return <details className="preset-menu">
    <summary>{labels[activeId] ?? activeName}</summary>
    <div aria-label="Model preset">{builtIns.map(option)}{userPresets.map(option)}</div>
  </details>
}

export function ModelPromptMenu({ acceptsTokenIds, onChange, promptTokenCount, value }: { acceptsTokenIds: boolean; onChange(value: string): void; promptTokenCount?: number; value: string }) {
  return <details className="model-prompt-menu">
    <summary>Prompt</summary>
    <label className="model-prompt-control">
      <span>Generation prompt</span>
      <input aria-label="Model generation prompt" onChange={(event) => onChange(event.target.value)} value={value} />
      <small>{acceptsTokenIds ? (promptTokenCount === undefined ? 'Research BPE' : `${promptTokenCount} Token IDs`) : 'Add a Token IDs input'}</small>
    </label>
  </details>
}
