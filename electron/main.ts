import { app, BrowserWindow, dialog, ipcMain, shell, type BrowserWindowConstructorOptions } from 'electron'
import { writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { rendererWebPreferences } from './security.js'
import { runAtomicRuntime, type AtomicRuntimePayload } from './atomic-runtime.js'
import { askLaboChannel, atomicRuntimeChannel, chatGPTSessionChannel, configureChatGPTChannel, connectChatGPTChannel, deleteOpenAIKeyChannel, desktopUpdateStatusChannel, disconnectChatGPTChannel, exportFileChannel, launchDesktopUpdateChannel, loadDesktopStateChannel, openAISettingsChannel, openDesktopSetupChannel, saveDesktopStateChannel, saveOpenAIKeyChannel, testOpenAIKeyChannel, windowStateChannel } from './ipc-contract.js'
import { askLabo } from './ask-labo.js'
import { loadDesktopState, saveDesktopState } from './desktop-state.js'
import { deleteOpenAIApiKey, getOpenAISettingsStatus, saveOpenAIApiKey, testOpenAIConnection } from './openai-credentials.js'
import { cacheDesktopUpdateStatus, desktopSetupReleaseUrl, getDesktopUpdateStatus, launchDesktopUpdate, parseDesktopUpdateCache, restoreDesktopUpdateStatus, validDesktopUpdateChannel } from './desktop-updates.js'
import { CodexAppServer } from './chatgpt-session.js'

const currentDirectory = dirname(fileURLToPath(import.meta.url))

function allowedChatGPTSignInUrl(candidate: string): string {
  const url = new URL(candidate)
  const allowedHost = url.hostname === 'openai.com'
    || url.hostname.endsWith('.openai.com')
    || url.hostname === 'chatgpt.com'
    || url.hostname.endsWith('.chatgpt.com')
  if (url.protocol !== 'https:' || !allowedHost) throw new Error('ChatGPT sign-in returned an untrusted URL')
  return url.toString()
}

const chatGPT = new CodexAppServer(
  (url) => shell.openExternal(allowedChatGPTSignInUrl(url)),
  app.getVersion(),
  process.env.LABO_CODEX_HOME?.trim() || join(app.getPath('userData'), 'codex'),
)

interface ChatGPTPreferences { model?: string; effort?: string }

async function chatGPTPreferences(): Promise<ChatGPTPreferences> {
  const value = await loadDesktopState(app.getPath('userData'), 'settings')
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {}
  const preferences = (value as { chatGPT?: unknown }).chatGPT
  if (!preferences || typeof preferences !== 'object' || Array.isArray(preferences)) return {}
  const record = preferences as Record<string, unknown>
  return {
    model: typeof record.model === 'string' ? record.model : undefined,
    effort: typeof record.effort === 'string' ? record.effort : undefined,
  }
}

async function desktopUpdateChannelPreference() {
  const value = await loadDesktopState(app.getPath('userData'), 'settings')
  return validDesktopUpdateChannel(value && typeof value === 'object' && !Array.isArray(value) ? (value as { desktopUpdateChannel?: unknown }).desktopUpdateChannel : undefined)
}

async function saveDesktopUpdateChannelPreference(channel: 'stable' | 'main') {
  await saveDesktopState(app.getPath('userData'), 'settings', { desktopUpdateChannel: channel })
}

async function configuredChatGPTStatus() {
  const [status, preferences] = await Promise.all([chatGPT.status(), chatGPTPreferences()])
  const selectedModel = status.models?.find((model) => model.id === preferences.model) ?? status.models?.find((model) => model.isDefault) ?? status.models?.[0]
  const selectedEffort = selectedModel?.efforts.includes(preferences.effort ?? '') ? preferences.effort : selectedModel?.defaultEffort ?? selectedModel?.efforts[0]
  return { ...status, selectedModel: selectedModel?.id, selectedEffort }
}

function createMainWindow(): BrowserWindow {
  const platformFrame: BrowserWindowConstructorOptions = process.platform === 'darwin'
    ? { titleBarStyle: 'hiddenInset', trafficLightPosition: { x: 15, y: 17 } }
    : { titleBarStyle: 'default', autoHideMenuBar: true }
  const window = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1024,
    minHeight: 680,
    backgroundColor: '#08090b',
    title: 'LABO AI',
    ...platformFrame,
    webPreferences: {
      ...rendererWebPreferences,
      preload: join(currentDirectory, 'preload.cjs'),
    },
  })

  const developmentUrl = process.env.VITE_DEV_SERVER_URL
  if (developmentUrl) {
    void window.loadURL(developmentUrl)
  } else {
    void window.loadFile(join(currentDirectory, '..', 'dist', 'index.html'))
  }

  const publishWindowState = () => window.webContents.send(windowStateChannel, { fullScreen: window.isFullScreen() })
  window.on('enter-full-screen', publishWindowState)
  window.on('leave-full-screen', publishWindowState)

  return window
}

app.whenReady().then(() => {
  ipcMain.handle(atomicRuntimeChannel, (_event, payload: AtomicRuntimePayload) => runAtomicRuntime(payload, app.isPackaged ? { userDataDirectory: app.getPath('userData') } : {}))
  ipcMain.handle(askLaboChannel, async (_event, payload) => {
    const session = await chatGPT.status()
    return session.connected ? chatGPT.ask(payload, await chatGPTPreferences()) : askLabo(payload)
  })
  ipcMain.handle(openAISettingsChannel, () => getOpenAISettingsStatus())
  ipcMain.handle(saveOpenAIKeyChannel, (_event, payload) => saveOpenAIApiKey(payload))
  ipcMain.handle(deleteOpenAIKeyChannel, () => deleteOpenAIApiKey())
  ipcMain.handle(testOpenAIKeyChannel, () => testOpenAIConnection())
  ipcMain.handle(chatGPTSessionChannel, () => configuredChatGPTStatus())
  ipcMain.handle(connectChatGPTChannel, async () => { await chatGPT.connect(); return configuredChatGPTStatus() })
  ipcMain.handle(disconnectChatGPTChannel, () => chatGPT.disconnect())
  ipcMain.handle(configureChatGPTChannel, async (_event, payload: { model?: unknown; effort?: unknown }) => {
    const status = await chatGPT.status()
    if (!status.connected) throw new Error('Connect ChatGPT before choosing a model')
    const model = status.models?.find((candidate) => candidate.id === payload?.model)
    if (!model) throw new Error('Choose a model available to this ChatGPT account')
    const effort = typeof payload?.effort === 'string' && model.efforts.includes(payload.effort) ? payload.effort : model.defaultEffort ?? model.efforts[0]
    await saveDesktopState(app.getPath('userData'), 'settings', { chatGPT: { model: model.id, effort } })
    return configuredChatGPTStatus()
  })
  ipcMain.handle(windowStateChannel, (event) => ({ fullScreen: BrowserWindow.fromWebContents(event.sender)?.isFullScreen() ?? false }))
  ipcMain.handle(loadDesktopStateChannel, (_event, payload: { scope?: unknown }) => loadDesktopState(app.getPath('userData'), payload?.scope))
  ipcMain.handle(saveDesktopStateChannel, (_event, payload: { scope?: unknown; data?: unknown }) => saveDesktopState(app.getPath('userData'), payload?.scope, payload?.data))
  ipcMain.handle(desktopUpdateStatusChannel, async (_event, payload: { channel?: unknown } | undefined) => {
    const channel = payload?.channel === undefined ? await desktopUpdateChannelPreference() : validDesktopUpdateChannel(payload.channel)
    const current = await loadDesktopState(app.getPath('userData'), 'settings')
    const settings = current && typeof current === 'object' && !Array.isArray(current) ? current as Record<string, unknown> : {}
    const previousCache = parseDesktopUpdateCache(settings.desktopUpdateCache)
    const freshStatus = await getDesktopUpdateStatus(app.getPath('userData'), app.getVersion(), channel)
    const nextCache = cacheDesktopUpdateStatus(previousCache, freshStatus)
    await saveDesktopState(app.getPath('userData'), 'settings', { desktopUpdateChannel: channel, desktopUpdateCache: nextCache })
    return restoreDesktopUpdateStatus(freshStatus, nextCache)
  })
  ipcMain.handle(launchDesktopUpdateChannel, async (_event, payload: { channel?: unknown } | undefined) => {
    const channel = payload?.channel === undefined ? await desktopUpdateChannelPreference() : validDesktopUpdateChannel(payload.channel)
    await saveDesktopUpdateChannelPreference(channel)
    const result = await launchDesktopUpdate(app.getPath('userData'), channel)
    setTimeout(() => app.quit(), 350)
    return result
  })
  ipcMain.handle(openDesktopSetupChannel, () => shell.openExternal(desktopSetupReleaseUrl))
  ipcMain.handle(exportFileChannel, async (event, payload: { filename?: unknown; content?: unknown; kind?: unknown }) => {
    if (typeof payload?.filename !== 'string' || typeof payload.content !== 'string' || !['svg', 'python'].includes(String(payload.kind)) || payload.content.length > 10_000_000) throw new Error('Invalid LABO export payload')
    const extension = payload.kind === 'svg' ? 'svg' : 'py'
    const filename = payload.filename.replace(/[^A-Za-z0-9._-]+/g, '-').replace(new RegExp(`\\.${extension}$`, 'i'), '') + `.${extension}`
    const owner = BrowserWindow.fromWebContents(event.sender) ?? undefined
    const options = { defaultPath: filename, filters: [{ name: payload.kind === 'svg' ? 'SVG diagram' : 'Python source', extensions: [extension] }] }
    const result = owner ? await dialog.showSaveDialog(owner, options) : await dialog.showSaveDialog(options)
    if (result.canceled || !result.filePath) return { saved: false }
    await writeFile(result.filePath, payload.content, 'utf8')
    return { saved: true, path: result.filePath }
  })
  createMainWindow()
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createMainWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', () => chatGPT.stop())
