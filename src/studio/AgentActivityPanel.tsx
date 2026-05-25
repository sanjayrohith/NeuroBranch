import { AlertTriangle, Check, Clock3, RotateCcw, X } from 'lucide-react'
import type { AgentGraphPlan } from '../core/agentic-graph'

export type AgentActivityStatus = 'running' | 'answered' | 'review' | 'applied' | 'failed' | 'discarded'

export interface AgentActivityItem {
  id: string
  prompt: string
  status: AgentActivityStatus
  createdAt: number
  summary?: string
  accepted?: number
  rejected?: number
  missing?: number
  tools?: string[]
  error?: string
  plan?: AgentGraphPlan
}

export function AgentActivityPanel({ activities, busy, onClear, onClose, onRetry, onReview }: {
  activities: AgentActivityItem[]
  busy: boolean
  onClear(): void
  onClose(): void
  onRetry(activity: AgentActivityItem): void
  onReview(activity: AgentActivityItem): void
}) {
  return <section aria-label="Agent activity" className="agent-activity-panel">
    <header>
      <span><Check size={14} /><strong>Agent activity</strong><small>{activities.length} task{activities.length === 1 ? '' : 's'}</small></span>
      <div><button disabled={busy || activities.length === 0} onClick={onClear} type="button">Clear</button><button aria-label="Close agent activity" onClick={onClose} type="button"><X size={13} /></button></div>
    </header>
    {activities.length === 0 ? <p className="agent-activity-empty">Your agent runs, validation results and errors will appear here.</p> : <ol>
      {activities.map((activity) => <li data-status={activity.status} key={activity.id}>
        <div className="agent-activity-status"><span>{activity.status === 'running' ? <Clock3 size={12} /> : activity.status === 'failed' ? <AlertTriangle size={12} /> : <Check size={12} />}{activity.status === 'review' ? 'Awaiting full-plan review' : activity.status === 'discarded' ? 'Closed — plan saved' : activity.status}</span><time>{new Date(activity.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</time></div>
        <strong>{activity.prompt}</strong>
        {activity.status === 'running' ? <p>Inspecting cards and validating graph changes…</p> : activity.error ? <p>{activity.error}</p> : activity.summary && <p>{activity.summary}</p>}
        {activity.status !== 'running' && activity.status !== 'answered' && !activity.error && <div className="agent-activity-metrics"><span>{activity.accepted ?? 0} accepted</span><span>{activity.rejected ?? 0} rejected</span><span>{activity.missing ?? 0} missing</span>{Boolean(activity.tools?.length) && <span>{activity.tools!.length} tools</span>}</div>}
        {Boolean(activity.tools?.length) && <code className="agent-activity-tools">{activity.tools!.join(' → ')}</code>}
        <div className="agent-activity-actions">
          {activity.plan && activity.status !== 'applied' && <button aria-label={`Review full agent plan: ${activity.prompt}`} onClick={() => onReview(activity)} type="button"><Check size={11} />Review full plan</button>}
          <button aria-label={`Retry agent task: ${activity.prompt}`} disabled={busy} onClick={() => onRetry(activity)} type="button"><RotateCcw size={11} />Retry</button>
        </div>
      </li>)}
    </ol>}
  </section>
}
