import { useEffect, type PointerEvent, type ReactNode } from 'react'
import { createPortal } from 'react-dom'

export function StudioDialog({ ariaLabel, backdropClassName = '', children, className = '', onClose }: { ariaLabel: string; backdropClassName?: string; children: ReactNode; className?: string; onClose(): void }) {
  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose() }
    window.addEventListener('keydown', closeOnEscape)
    return () => window.removeEventListener('keydown', closeOnEscape)
  }, [onClose])

  const closeFromBackdrop = (event: PointerEvent<HTMLDivElement>) => {
    if (event.target === event.currentTarget) onClose()
  }

  return createPortal(<div className={`studio-dialog-backdrop ${backdropClassName}`.trim()} onPointerDown={closeFromBackdrop}>
    <section aria-label={ariaLabel} aria-modal="true" className={`studio-dialog ${className}`.trim()} onPointerDown={(event) => event.stopPropagation()} role="dialog">{children}</section>
  </div>, document.body)
}
