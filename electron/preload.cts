import { contextBridge, ipcRenderer } from 'electron'
import type { AtomicRuntimePayload } from './atomic-runtime.js'
import type { AskNeuroBranchPayload } from './ask-neurobranch.js'

const atomicRuntimeChannel = 'neurobranch:atomic-runtime'
const askNeuroBranchChannel = 'neurobranch:ask'
const openAISettingsChannel = 'neurobranch:openai-settings'
const saveOpenAIKeyChannel = 'neurobranch:openai-key-save'
const deleteOpenAIKeyChannel = 'neurobranch:openai-key-delete'
const testOpenAIKeyChannel = 'neurobranch:openai-key-test'
const chatGPTSessionChannel = 'neurobranch:chatgpt-session'
const connectChatGPTChannel = 'neurobranch:chatgpt-connect'
const disconnectChatGPTChannel = 'neurobranch:chatgpt-disconnect'
const configureChatGPTChannel = 'neurobranch:chatgpt-configure'
const exportFileChannel = 'neurobranch:export-file'
const windowStateChannel = 'neurobranch:window-state'
const loadDesktopStateChannel = 'neurobranch:desktop-state-load'
const saveDesktopStateChannel = 'neurobranch:desktop-state-save'
const desktopUpdateStatusChannel = 'neurobranch:desktop-update-status'
const launchDesktopUpdateChannel = 'neurobranch:desktop-update-launch'
const openDesktopSetupChannel = 'neurobranch:desktop-setup-open'

contextBridge.exposeInMainWorld('neurobranch', {
  platform: process.platform,
  runtime: 'electron',
  runAtomic: (payload: AtomicRuntimePayload) => ipcRenderer.invoke(atomicRuntimeChannel, payload),
  askNeuroBranch: (payload: AskNeuroBranchPayload) => ipcRenderer.invoke(askNeuroBranchChannel, payload),
  getOpenAISettings: () => ipcRenderer.invoke(openAISettingsChannel),
  saveOpenAIKey: (apiKey: string) => ipcRenderer.invoke(saveOpenAIKeyChannel, { apiKey }),
  deleteOpenAIKey: () => ipcRenderer.invoke(deleteOpenAIKeyChannel),
  testOpenAIKey: () => ipcRenderer.invoke(testOpenAIKeyChannel),
  getChatGPTSession: () => ipcRenderer.invoke(chatGPTSessionChannel),
  connectChatGPT: () => ipcRenderer.invoke(connectChatGPTChannel),
  disconnectChatGPT: () => ipcRenderer.invoke(disconnectChatGPTChannel),
  configureChatGPT: (configuration: { model: string; effort: string }) => ipcRenderer.invoke(configureChatGPTChannel, configuration),
  exportFile: (payload: { filename: string; content: string; kind: 'svg' | 'python' }) => ipcRenderer.invoke(exportFileChannel, payload),
  loadDesktopState: (scope: 'model' | 'training' | 'tokenizer' | 'settings') => ipcRenderer.invoke(loadDesktopStateChannel, { scope }),
  saveDesktopState: (scope: 'model' | 'training' | 'tokenizer' | 'settings', data: unknown) => ipcRenderer.invoke(saveDesktopStateChannel, { scope, data }),
  getDesktopUpdateStatus: (channel?: 'stable' | 'main') => ipcRenderer.invoke(desktopUpdateStatusChannel, { channel }),
  launchDesktopUpdate: (channel?: 'stable' | 'main') => ipcRenderer.invoke(launchDesktopUpdateChannel, { channel }),
  openDesktopSetup: () => ipcRenderer.invoke(openDesktopSetupChannel),
  getWindowState: () => ipcRenderer.invoke(windowStateChannel),
  onWindowStateChange: (callback: (state: { fullScreen: boolean }) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, state: { fullScreen: boolean }) => callback(state)
    ipcRenderer.on(windowStateChannel, listener)
    return () => ipcRenderer.removeListener(windowStateChannel, listener)
  },
})
