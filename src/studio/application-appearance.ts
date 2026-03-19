export type LaboTheme = 'labo-dark' | 'complexity-spectrum'

export const LABO_THEMES: Array<{
  id: LaboTheme
  name: string
  description: string
  colors: string[]
}> = [
  {
    id: 'labo-dark',
    name: 'LABO Dark',
    description: 'The restrained pastel workspace used by default.',
    colors: ['#91c7ad', '#91c3cc', '#aaa4d6'],
  },
  {
    id: 'complexity-spectrum',
    name: 'Complexity Spectrum',
    description: 'The green, cyan and violet identity of complexity-ai.fr.',
    colors: ['#6ee7b7', '#7dd3fc', '#c4b5fd', '#fcd34d', '#f9a8d4'],
  },
]

export type ApplicationSettingsRecord = Record<string, unknown> & {
  appearance?: Record<string, unknown> & { theme?: unknown; language?: unknown }
}

const WEB_PREFERENCES_STORAGE_KEY = 'labo.web.preferences.v1'

export function applicationSettingsRecord(value: unknown): ApplicationSettingsRecord {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as ApplicationSettingsRecord : {}
}

function validTheme(value: unknown): value is LaboTheme {
  return value === 'labo-dark' || value === 'complexity-spectrum'
}

export function readLocalApplicationSettings(): ApplicationSettingsRecord {
  try {
    return applicationSettingsRecord(JSON.parse(window.localStorage.getItem(WEB_PREFERENCES_STORAGE_KEY) ?? '{}'))
  } catch {
    return {}
  }
}

function saveLocalApplicationSettings(settings: ApplicationSettingsRecord): void {
  try {
    window.localStorage.setItem(WEB_PREFERENCES_STORAGE_KEY, JSON.stringify(settings))
  } catch {
    // Preferences still apply to the current document when storage is unavailable.
  }
}

function mergeApplicationSettings(
  base: ApplicationSettingsRecord,
  override: ApplicationSettingsRecord,
): ApplicationSettingsRecord {
  return {
    ...base,
    ...override,
    appearance: {
      ...applicationSettingsRecord(base.appearance),
      ...applicationSettingsRecord(override.appearance),
    },
  }
}

export function readLaboTheme(): LaboTheme {
  const activeTheme = document.documentElement.dataset.laboTheme
  if (validTheme(activeTheme)) return activeTheme
  const localTheme = readLocalApplicationSettings().appearance?.theme
  return validTheme(localTheme) ? localTheme : 'labo-dark'
}

export function applyLaboTheme(theme: LaboTheme): void {
  document.documentElement.dataset.laboTheme = theme
}

export async function readApplicationSettings(): Promise<{ authenticated: boolean; settings: ApplicationSettingsRecord }> {
  if (window.labo?.runtime === 'electron' && window.labo.loadDesktopState) {
    return { authenticated: true, settings: applicationSettingsRecord(await window.labo.loadDesktopState('settings')) }
  }
  if (window.labo?.runtime === 'web' && window.labo.loadWebWorkspace) {
    const localSettings = readLocalApplicationSettings()
    try {
      const workspace = await window.labo.loadWebWorkspace()
      const settings = workspace.authenticated
        ? mergeApplicationSettings(localSettings, applicationSettingsRecord(workspace.settings))
        : localSettings
      saveLocalApplicationSettings(settings)
      return { authenticated: workspace.authenticated, settings }
    } catch {
      return { authenticated: false, settings: localSettings }
    }
  }
  return { authenticated: false, settings: {} }
}

export async function saveApplicationSettings(
  settings: ApplicationSettingsRecord,
  authenticated: boolean,
  remotePatch: ApplicationSettingsRecord = settings,
): Promise<void> {
  if (window.labo?.runtime === 'electron' && window.labo.saveDesktopState) {
    await window.labo.saveDesktopState('settings', remotePatch)
  } else if (window.labo?.runtime === 'web') {
    saveLocalApplicationSettings(settings)
    if (authenticated && window.labo.saveWebWorkspace) {
      await window.labo.saveWebWorkspace({ settings: remotePatch })
    }
  }
}

export async function loadLaboTheme(): Promise<LaboTheme> {
  try {
    const { settings } = await readApplicationSettings()
    return validTheme(settings.appearance?.theme) ? settings.appearance.theme : 'labo-dark'
  } catch {
    return readLaboTheme()
  }
}

export async function saveLaboTheme(theme: LaboTheme): Promise<void> {
  applyLaboTheme(theme)
  try {
    const { authenticated, settings } = await readApplicationSettings()
    const nextSettings: ApplicationSettingsRecord = {
      ...settings,
      appearance: { ...applicationSettingsRecord(settings.appearance), theme },
    }
    await saveApplicationSettings(nextSettings, authenticated, { appearance: { theme } })
  } catch {
    // A visual preference must never prevent the workspace from opening.
  }
}

export async function initializeLaboTheme(): Promise<LaboTheme> {
  applyLaboTheme(readLaboTheme())
  const theme = await loadLaboTheme()
  applyLaboTheme(theme)
  return theme
}
