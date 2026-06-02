import { useEffect, useState } from 'react'
import { X } from 'lucide-react'
import type { CustomTokenizerCard } from './custom-tokenizer-card'

export function TokenizerCardCreator({ initialCard, onCancel, onCreate }: { initialCard?: CustomTokenizerCard; onCancel(): void; onCreate(card: CustomTokenizerCard): void }) {
  const [label, setLabel] = useState(initialCard?.label ?? 'My tokenizer card')
  const [category, setCategory] = useState(initialCard?.category ?? 'Transform')
  const [pythonCode, setPythonCode] = useState(initialCard?.pythonCode ?? 'text = text.lower()')
  const valid = label.trim().length > 0 && pythonCode.trim().length > 0

  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => { if (event.key === 'Escape') onCancel() }
    window.addEventListener('keydown', closeOnEscape)
    return () => window.removeEventListener('keydown', closeOnEscape)
  }, [onCancel])

  return <div className="tokenizer-card-overlay" onPointerDown={(event) => { if (event.target === event.currentTarget) onCancel() }}>
    <section aria-label="Tokenizer card builder" aria-modal="true" className="tokenizer-card-modal" onPointerDown={(event) => event.stopPropagation()} role="dialog">
      <header><div><span className="eyebrow">TOKENIZER CARD BUILDER</span><h2>{initialCard ? 'Edit reusable tokenizer card' : 'Compose a reusable tokenizer card'}</h2></div><button aria-label="Close tokenizer card builder" onClick={onCancel} type="button"><X size={18} /></button></header>
      <div className="tokenizer-card-form">
        <label><span>Name</span><input autoFocus onChange={(event) => setLabel(event.target.value)} value={label} /></label>
        <label><span>Category</span><input onChange={(event) => setCategory(event.target.value)} value={category} /></label>
        <label><span>Python lowering</span><textarea onChange={(event) => setPythonCode(event.target.value)} spellCheck={false} value={pythonCode} /></label>
      </div>
      <footer><button onClick={onCancel} type="button">Cancel</button><button disabled={!valid} onClick={() => onCreate({ id: initialCard?.id ?? (label.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'custom-tokenizer'), label: label.trim(), category: category.trim() || 'Custom', pythonCode })} type="button">{initialCard ? 'Save changes' : 'Create and add card'}</button></footer>
    </section>
  </div>
}
