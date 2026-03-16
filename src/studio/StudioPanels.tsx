import type { ReactNode } from 'react'

export function StudioPanelTab({ actions, children, icon, status }: { actions?: ReactNode; children: ReactNode; icon?: ReactNode; status?: ReactNode }) {
  return <div className="panel-tab">{icon}{children}{status && <span>{status}</span>}{actions}</div>
}

export function StudioCanvasPanel({ children, className = '', tab }: { children: ReactNode; className?: string; tab: ReactNode }) {
  return <div className={`canvas-panel ${className}`.trim()}>{tab}{children}</div>
}

export function StudioCodePanel({ children, className = '', tab }: { children: ReactNode; className?: string; tab: ReactNode }) {
  return <div className={`code-panel ${className}`.trim()}>{tab}{children}</div>
}
