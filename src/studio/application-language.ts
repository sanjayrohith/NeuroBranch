import { useEffect, useState } from 'react'
import { applicationSettingsRecord, readApplicationSettings, readLocalApplicationSettings, saveApplicationSettings } from './application-appearance'

export type NeuroBranchLanguage = 'en' | 'fr'

const languageChangeEvent = 'neurobranch-language-change'

function validLanguage(value: unknown): value is NeuroBranchLanguage {
  return value === 'en' || value === 'fr'
}

export function readNeuroBranchLanguage(): NeuroBranchLanguage {
  const activeLanguage = document.documentElement.dataset.neurobranchLanguage
  if (validLanguage(activeLanguage)) return activeLanguage
  const localLanguage = readLocalApplicationSettings().appearance?.language
  if (validLanguage(localLanguage)) return localLanguage
  return document.documentElement.lang.toLowerCase().startsWith('fr') ? 'fr' : 'en'
}

export function applyNeuroBranchLanguage(language: NeuroBranchLanguage): void {
  document.documentElement.lang = language
  document.documentElement.dataset.neurobranchLanguage = language
  window.dispatchEvent(new CustomEvent<NeuroBranchLanguage>(languageChangeEvent, { detail: language }))
}

export async function loadNeuroBranchLanguage(): Promise<NeuroBranchLanguage> {
  try {
    const { settings } = await readApplicationSettings()
    return validLanguage(settings.appearance?.language) ? settings.appearance.language : 'en'
  } catch {
    return readNeuroBranchLanguage()
  }
}

export async function saveNeuroBranchLanguage(language: NeuroBranchLanguage): Promise<void> {
  applyNeuroBranchLanguage(language)
  try {
    const { authenticated, settings } = await readApplicationSettings()
    const nextSettings = {
      ...settings,
      appearance: { ...applicationSettingsRecord(settings.appearance), language },
    }
    await saveApplicationSettings(nextSettings, authenticated, { appearance: { language } })
  } catch {
    // A language preference must never prevent the workspace from opening.
  }
}

export async function initializeNeuroBranchLanguage(): Promise<NeuroBranchLanguage> {
  applyNeuroBranchLanguage(readNeuroBranchLanguage())
  const language = await loadNeuroBranchLanguage()
  applyNeuroBranchLanguage(language)
  return language
}

export function useNeuroBranchLanguage(): NeuroBranchLanguage {
  const [language, setLanguage] = useState<NeuroBranchLanguage>(readNeuroBranchLanguage)

  useEffect(() => {
    const update = (event: Event) => setLanguage((event as CustomEvent<NeuroBranchLanguage>).detail)
    window.addEventListener(languageChangeEvent, update)
    return () => window.removeEventListener(languageChangeEvent, update)
  }, [])

  return language
}
