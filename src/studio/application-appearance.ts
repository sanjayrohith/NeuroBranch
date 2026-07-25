export type NeuroBranchTheme = 'neurobranch-dark' | 'complexity-spectrum'

export const NEUROBRANCH_THEMES: Array<{
  id: NeuroBranchTheme
  name: string
  description: string
  colors: string[]
}> = [
  {
    id: 'neurobranch-dark',
    name: 'NeuroBranch Dark',
    description: 'The restrained pastel workspace used by default.',
    colors: ['#91c7ad', '#91c3cc', '#aaa4d6'],
  },
  {
    id: 'complexity-spectrum',
    name: 'Complexity Spectrum',
    description: 'The green, cyan and violet identity of NeuroBranch.',
    colors: ['#6ee7b7', '#7dd3fc', '#c4b5fd', '#fcd34d', '#f9a8d4'],
  },
]

export type ApplicationSettingsRecord = Record<string, unknown> & {
  appearance?: Record<string, unknown> & { theme?: unknown; language?: unknown }
}

const WEB_PREFERENCES_STORAGE_KEY = 'neurobranch.web.preferences.v1'

export function applicationSettingsRecord(value: unknown): ApplicationSettingsRecord {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as ApplicationSettingsRecord : {}
}

function validTheme(value: unknown): value is NeuroBranchTheme {
  return value === 'neurobranch-dark' || value === 'complexity-spectrum'
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

export function readNeuroBranchTheme(): NeuroBranchTheme {
  const activeTheme = document.documentElement.dataset.neurobranchTheme
  if (validTheme(activeTheme)) return activeTheme
  const localTheme = readLocalApplicationSettings().appearance?.theme
  return validTheme(localTheme) ? localTheme : 'neurobranch-dark'
}

export function applyNeuroBranchTheme(theme: NeuroBranchTheme): void {
  document.documentElement.dataset.neurobranchTheme = theme
}

export async function readApplicationSettings(): Promise<{ authenticated: boolean; settings: ApplicationSettingsRecord }> {
  if (window.neurobranch?.runtime === 'electron' && window.neurobranch.loadDesktopState) {
    return { authenticated: true, settings: applicationSettingsRecord(await window.neurobranch.loadDesktopState('settings')) }
  }
  if (window.neurobranch?.runtime === 'web' && window.neurobranch.loadWebWorkspace) {
    const localSettings = readLocalApplicationSettings()
    try {
      const workspace = await window.neurobranch.loadWebWorkspace()
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
  if (window.neurobranch?.runtime === 'electron' && window.neurobranch.saveDesktopState) {
    await window.neurobranch.saveDesktopState('settings', remotePatch)
  } else if (window.neurobranch?.runtime === 'web') {
    saveLocalApplicationSettings(settings)
    if (authenticated && window.neurobranch.saveWebWorkspace) {
      await window.neurobranch.saveWebWorkspace({ settings: remotePatch })
    }
  }
}

export async function loadNeuroBranchTheme(): Promise<NeuroBranchTheme> {
  try {
    const { settings } = await readApplicationSettings()
    return validTheme(settings.appearance?.theme) ? settings.appearance.theme : 'neurobranch-dark'
  } catch {
    return readNeuroBranchTheme()
  }
}

export async function saveNeuroBranchTheme(theme: NeuroBranchTheme): Promise<void> {
  applyNeuroBranchTheme(theme)
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

export async function initializeNeuroBranchTheme(): Promise<NeuroBranchTheme> {
  applyNeuroBranchTheme(readNeuroBranchTheme())
  const theme = await loadNeuroBranchTheme()
  applyNeuroBranchTheme(theme)
  return theme
}
