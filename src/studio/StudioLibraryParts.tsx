import type { ButtonHTMLAttributes, DetailsHTMLAttributes, ReactNode } from 'react'
import { StudioCountBadge } from './StudioControls'

export function StudioLibrarySection({ children, className = '', count, label, ...props }: DetailsHTMLAttributes<HTMLDetailsElement> & { children: ReactNode; count: ReactNode; label: ReactNode }) {
  return <details className={`library-family ${className}`.trim()} {...props}><summary>{label}<StudioCountBadge>{count}</StudioCountBadge></summary>{children}</details>
}

export function StudioLibraryItem({ children, className = '', glyph, meta, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { children: ReactNode; glyph?: ReactNode; meta?: ReactNode }) {
  return <button className={`library-block ${className}`.trim()} type="button" {...props}>{glyph}{meta ? <span><strong>{children}</strong><small>{meta}</small></span> : children}</button>
}
