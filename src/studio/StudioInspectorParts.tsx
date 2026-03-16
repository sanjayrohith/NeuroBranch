import type { ReactNode } from 'react'

export function InspectorSection({ children, className = '', title }: { children: ReactNode; className?: string; title?: ReactNode }) {
  return <section className={`inspector-section ${className}`.trim()}>{title && <div className="section-title">{title}</div>}{children}</section>
}

export function InspectorSelection({ detail, icon, title }: { detail?: ReactNode; icon: ReactNode; title: ReactNode }) {
  return <div className="selection-card"><span className="selection-icon">{icon}</span><div><strong>{title}</strong>{detail && <small>{detail}</small>}</div></div>
}

export function InspectorMetric({ label, value }: { label: ReactNode; value: ReactNode }) {
  return <div className="check-row"><span>{label}</span><b>{value}</b></div>
}
