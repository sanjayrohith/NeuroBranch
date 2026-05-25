import type { ReactNode } from 'react'

export function AgentChoiceCard({ ariaLabel, description, label, children }: {
  ariaLabel: string
  description: string
  label: string
  children: ReactNode
}) {
  return <section aria-label={ariaLabel} className="agent-choice-card">
    <header><strong>{label}</strong><small>{description}</small></header>
    <div className="agent-choice-card-options">{children}</div>
  </section>
}
