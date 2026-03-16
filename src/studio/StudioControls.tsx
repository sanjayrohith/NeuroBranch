import type { ButtonHTMLAttributes, ReactNode } from 'react'

export type StudioButtonVariant = 'ghost' | 'accent' | 'danger' | 'quiet'

export function StudioButton({ children, className = '', variant = 'ghost', ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { children: ReactNode; variant?: StudioButtonVariant }) {
  return <button className={`studio-button studio-button-${variant} ${className}`.trim()} type="button" {...props}>{children}</button>
}

export function StudioIconButton({ children, className = '', label, ...props }: Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'aria-label'> & { children: ReactNode; label: string }) {
  return <StudioButton aria-label={label} className={`studio-icon-button ${className}`.trim()} {...props}>{children}</StudioButton>
}

export interface SegmentedOption<Value extends string> {
  id: Value
  label: ReactNode
  icon?: ReactNode
  disabled?: boolean
}

export function StudioSegmentedControl<Value extends string>({ ariaLabel, className = '', onChange, options, value }: { ariaLabel: string; className?: string; onChange(value: Value): void; options: Array<SegmentedOption<Value>>; value: Value }) {
  return <div aria-label={ariaLabel} className={`studio-segmented-control ${className}`.trim()}>{options.map((option) => <button aria-pressed={value === option.id} disabled={option.disabled} key={option.id} onClick={() => onChange(option.id)} type="button">{option.icon}{option.label}</button>)}</div>
}

export function StudioCountBadge({ children }: { children: ReactNode }) {
  return <span className="studio-count-badge">{children}</span>
}
