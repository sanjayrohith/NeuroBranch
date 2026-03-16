import type { ReactNode } from 'react'
import { StudioSegmentedControl } from './StudioControls'

export interface StudioViewOption<View extends string> {
  id: View
  label: string
  icon: ReactNode
}

export function StudioToolbar({ children, className = '', meta }: { children: ReactNode; className?: string; meta?: ReactNode }) {
  return <section className={`workspace-toolbar ${className}`.trim()}><div className="toolbar-controls">{children}</div>{meta && <div className="toolbar-meta">{meta}</div>}</section>
}

export function StudioViewSwitcher<View extends string>({ ariaLabel, onChange, options, value }: { ariaLabel: string; onChange(view: View): void; options: Array<StudioViewOption<View>>; value: View }) {
  return <StudioSegmentedControl ariaLabel={ariaLabel} className="view-switcher" onChange={onChange} options={options} value={value} />
}

export function StudioWorkspace({ children, className = '', inspectorOpen = true, libraryOpen = true, ...props }: { children: ReactNode; className?: string; inspectorOpen?: boolean; libraryOpen?: boolean } & React.HTMLAttributes<HTMLDivElement>) {
  return <div className={`workspace-grid ${className} ${libraryOpen ? '' : 'library-hidden'} ${inspectorOpen ? '' : 'inspector-hidden'}`.trim()} {...props}>{children}</div>
}

export function StudioPanelHeading({ children, icon }: { children: ReactNode; icon: ReactNode }) {
  return <div className="panel-heading">{icon}<span>{children}</span></div>
}

export function StudioLibrary({ children, className = '', heading, icon }: { children: ReactNode; className?: string; heading: ReactNode; icon: ReactNode }) {
  return <aside className={`block-library ${className}`.trim()}>
    <StudioPanelHeading icon={icon}>{heading}</StudioPanelHeading>
    <div className="studio-library-scroll">{children}</div>
  </aside>
}

export function StudioEditor({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <section className={`editor-grid ${className}`.trim()}>{children}</section>
}

export function StudioInspector({ children, className = '', heading, hidden = false, icon }: { children: ReactNode; className?: string; heading: ReactNode; hidden?: boolean; icon: ReactNode }) {
  return <aside className={`inspector ${className}`.trim()} hidden={hidden}><StudioPanelHeading icon={icon}>{heading}</StudioPanelHeading>{children}</aside>
}

export function StudioStatusbar({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <footer className={`statusbar ${className}`.trim()}>{children}</footer>
}
