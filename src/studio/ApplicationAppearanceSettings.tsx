import { Check, Globe2, MonitorCog, Palette } from 'lucide-react'
import { useEffect, useState } from 'react'
import { LABO_THEMES, loadLaboTheme, readLaboTheme, saveLaboTheme, type LaboTheme } from './application-appearance'
import { loadLaboLanguage, readLaboLanguage, saveLaboLanguage, type LaboLanguage } from './application-language'

export function ApplicationAppearanceSettings() {
  const [theme, setTheme] = useState<LaboTheme>(readLaboTheme)
  const [language, setLanguage] = useState<LaboLanguage>(readLaboLanguage)

  useEffect(() => {
    let active = true
    void loadLaboTheme().then((storedTheme) => { if (active) setTheme(storedTheme) })
    void loadLaboLanguage().then((storedLanguage) => { if (active) setLanguage(storedLanguage) })
    return () => { active = false }
  }, [])

  const selectTheme = (nextTheme: LaboTheme) => {
    setTheme(nextTheme)
    void saveLaboTheme(nextTheme)
  }

  const selectLanguage = (nextLanguage: LaboLanguage) => {
    setLanguage(nextLanguage)
    void saveLaboLanguage(nextLanguage)
  }

  return <section aria-label="Application appearance" className="application-appearance-settings">
    <header>
      <span><Palette size={15} /></span>
      <div><strong>Workspace colour</strong><p>Choose one palette for the complete LABO AI application.</p></div>
    </header>
    <div className="application-theme-grid">
      {LABO_THEMES.map((candidate) => <button aria-label={`Use ${candidate.name} theme`} aria-pressed={theme === candidate.id} key={candidate.id} onClick={() => selectTheme(candidate.id)} type="button">
        <span className="application-theme-preview" data-theme-preview={candidate.id}>{candidate.colors.map((color) => <i key={color} style={{ background: color }} />)}</span>
        <span className="application-theme-copy"><strong>{candidate.name}</strong><small>{candidate.description}</small></span>
        <span className="application-theme-check">{theme === candidate.id ? <Check size={14} /> : null}</span>
      </button>)}
    </div>
    <section className="application-language-settings">
      <header><span><Globe2 size={14} /></span><div><strong>Interface language</strong><p>Also controls the language used by Ask LABO plans.</p></div></header>
      <div>
        <button aria-pressed={language === 'en'} onClick={() => selectLanguage('en')} type="button"><strong>English</strong><small>Interface and plans in English</small>{language === 'en' && <Check size={13} />}</button>
        <button aria-pressed={language === 'fr'} onClick={() => selectLanguage('fr')} type="button"><strong>Français</strong><small>Interface et plans en français</small>{language === 'fr' && <Check size={13} />}</button>
      </div>
    </section>
    <footer><MonitorCog size={13} /><span>Applied across Model, Training and Tokenizer Studio.</span></footer>
  </section>
}
