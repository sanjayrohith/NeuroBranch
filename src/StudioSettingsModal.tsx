import { Database, FolderKanban, Lightbulb, Palette, Settings2, Sparkles, X } from 'lucide-react'
import { useState, type ReactNode } from 'react'
import { DesktopUpdateSettings } from './DesktopUpdateSettings'
import { StudioDialog } from './studio/StudioDialog'
import { StudioButton, StudioIconButton } from './studio/StudioControls'

export interface StudioSettingsSection {
  id: string
  label: string
  icon: ReactNode
  content: ReactNode
}

function SettingsNavigation({ activeId, onChange, sections }: { activeId: string; onChange(id: string): void; sections: Array<Pick<StudioSettingsSection, 'id' | 'label' | 'icon'>> }) {
  return <nav aria-label="Settings sections" className="studio-settings-navigation"><small>Configuration</small>{sections.map((section) => <button aria-label={section.label} aria-pressed={activeId === section.id} key={section.id} onClick={() => onChange(section.id)} type="button"><span>{section.icon}</span><strong>{section.label}</strong></button>)}</nav>
}

function SettingsSectionHeading({ description, icon, label }: { description?: string; icon: ReactNode; label: ReactNode }) {
  return <div className="studio-settings-section-heading"><span>{icon}</span><div><small>Settings</small><h2>{label}</h2>{description && <p>{description}</p>}</div></div>
}

function SettingsCard({ children, icon, tone = 'sage' }: { children: ReactNode; icon: ReactNode; tone?: 'sage' | 'violet' }) {
  return <article className={`studio-settings-card tone-${tone}`}>{icon}<div>{children}</div></article>
}

export function StudioSettingsModal({ onClose, sections }: { onClose(): void; sections: StudioSettingsSection[] }) {
  const [sectionId, setSectionId] = useState('general')
  const runtime = window.labo?.runtime
  const sectionContent = Object.fromEntries(sections.map((section) => [section.id, section.content]))
  const sectionDescriptions: Record<string, string> = {
    general: 'Storage, privacy and desktop updates.',
    workspaces: 'Save, restore and compare your private graphs.',
    agent: 'Choose how Ask LABO plans and applies changes.',
    studio: 'Appearance shared across every LABO AI studio.',
    tips: 'Shortcuts and gestures for faster graph editing.',
  }
  const defaultNavigation = [
    { id: 'general', label: 'General', icon: <Settings2 size={13} /> },
    { id: 'workspaces', label: 'Workspaces', icon: <FolderKanban size={13} /> },
    { id: 'agent', label: 'Agent', icon: <Sparkles size={13} /> },
    { id: 'studio', label: 'Application', icon: <Palette size={13} /> },
    { id: 'tips', label: 'Tips', icon: <Lightbulb size={13} /> },
  ]
  const customSections = new Map(sections.map((section) => [section.id, section]))
  const navigation = defaultNavigation.map((section) => customSections.get(section.id) ?? section)
  const activeSection = navigation.find((section) => section.id === sectionId) ?? navigation[0]
  const fallbackCopy: Record<string, { title: string; body: string }> = {
    workspaces: { title: 'Private workspace', body: 'The active studio saves its current draft automatically for this profile.' },
    agent: { title: 'Shared agent preferences', body: 'Ask LABO uses the same private credentials and review defaults throughout the application.' },
    studio: { title: 'Application appearance', body: 'Choose one visual identity shared by the complete LABO AI workspace.' },
    tips: { title: 'Studio shortcuts', body: 'Use the same selection, editing and keyboard conventions throughout LABO AI.' },
  }

  return <StudioDialog ariaLabel="LABO AI settings" backdropClassName="model-card-modal-backdrop studio-settings-backdrop" className="model-card-modal studio-settings-modal" onClose={onClose}>
      <header className="studio-settings-header"><div><span>LABO AI / PREFERENCES</span><strong>Settings</strong><p>Configure this private workspace without leaving the studio.</p></div><StudioIconButton label="Close LABO AI settings" onClick={onClose}><X size={15} /></StudioIconButton></header>
      <div className="studio-settings-layout">
        <SettingsNavigation activeId={sectionId} onChange={setSectionId} sections={navigation} />
        <main className="studio-settings-panel">
          <SettingsSectionHeading description={sectionDescriptions[activeSection.id]} icon={activeSection.icon} label={activeSection.label} />
          <div className="studio-settings-body">
            {sectionId === 'general' ? <div className="studio-settings-general">
              <SettingsCard icon={<Database size={16} />}><strong>Private automatic save</strong><p>{runtime === 'web' ? 'Signed-in workspaces are stored in the account-scoped server database. Guest work is temporary.' : runtime === 'electron' ? 'Workspaces, cards and optimizers are stored in the persistent local SQLite profile.' : 'This development preview keeps only temporary browser state.'}</p></SettingsCard>
              <SettingsCard icon={<Settings2 size={16} />} tone="violet"><strong>Shared defaults, private creations</strong><p>Built-in cards and presets are read-only defaults. Everything you create belongs only to the current user profile.</p></SettingsCard>
              <DesktopUpdateSettings />
              {runtime === 'web' && <a className="studio-settings-account-link" href="/dashboard/settings" target="_top">Manage account and private data</a>}
            </div> : <div className={`studio-settings-context studio-settings-section-${sectionId}`}>{sectionContent[sectionId] ?? <div className="studio-settings-empty"><strong>{fallbackCopy[sectionId]?.title ?? activeSection.label}</strong><p>{fallbackCopy[sectionId]?.body ?? 'These settings use the shared LABO AI defaults in this studio.'}</p></div>}</div>}
          </div>
        </main>
      </div>
      <footer><span>Changes are saved automatically for this profile.</span><StudioButton onClick={onClose} variant="accent">Done</StudioButton></footer>
  </StudioDialog>
}
