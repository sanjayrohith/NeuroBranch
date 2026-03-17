export interface StudioChoice<Option extends string> {
  label: string
  value: Option
}

export function StudioChoiceMenu<Option extends string>({ ariaLabel, className = '', label, onChange, options, value }: { ariaLabel: string; className?: string; label: string; onChange(value: Option): void; options: Array<StudioChoice<Option>>; value: Option }) {
  const selected = options.find((option) => option.value === value) ?? options[0]
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const closeOutside = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false)
    }
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('pointerdown', closeOutside)
    document.addEventListener('keydown', closeOnEscape)
    return () => {
      document.removeEventListener('pointerdown', closeOutside)
      document.removeEventListener('keydown', closeOnEscape)
    }
  }, [open])

  return <div className={`studio-choice-field ${className}`.trim()}>
    <span>{label}</span>
    <div className={`studio-choice-menu ${open ? 'is-open' : ''}`} ref={rootRef}>
      <button aria-expanded={open} aria-label={ariaLabel} className="studio-choice-trigger" onClick={() => setOpen((current) => !current)} type="button">{selected?.label}</button>
      {open && <div className="studio-choice-options">
        {options.map((option) => <button aria-label={`Choose ${option.label}`} aria-pressed={option.value === value} key={option.value} onClick={(event) => {
          onChange(option.value)
          setOpen(false)
          event.currentTarget.blur()
        }} type="button">{option.label}</button>)}
      </div>}
    </div>
  </div>
}
import { useEffect, useRef, useState } from 'react'
