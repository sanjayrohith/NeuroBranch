import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { rendererWebPreferences } from './security'
import { askNeuroBranchChannel, atomicRuntimeChannel, chatGPTSessionChannel, connectChatGPTChannel, deleteOpenAIKeyChannel, desktopUpdateStatusChannel, disconnectChatGPTChannel, exportFileChannel, launchDesktopUpdateChannel, openAISettingsChannel, openDesktopSetupChannel, saveOpenAIKeyChannel, testOpenAIKeyChannel, windowStateChannel } from './ipc-contract'

describe('Electron renderer boundary', () => {
  it('isolates the NeuroBranch renderer from Node and the main process', () => {
    expect(rendererWebPreferences.contextIsolation).toBe(true)
    expect(rendererWebPreferences.nodeIntegration).toBe(false)
    expect(rendererWebPreferences.sandbox).toBe(true)
  })

  it('exposes only the named NeuroBranch IPC channels', () => {
    expect(atomicRuntimeChannel).toBe('neurobranch:atomic-runtime')
    expect(askNeuroBranchChannel).toBe('neurobranch:ask')
    expect(openAISettingsChannel).toBe('neurobranch:openai-settings')
    expect(saveOpenAIKeyChannel).toBe('neurobranch:openai-key-save')
    expect(deleteOpenAIKeyChannel).toBe('neurobranch:openai-key-delete')
    expect(testOpenAIKeyChannel).toBe('neurobranch:openai-key-test')
    expect(chatGPTSessionChannel).toBe('neurobranch:chatgpt-session')
    expect(connectChatGPTChannel).toBe('neurobranch:chatgpt-connect')
    expect(disconnectChatGPTChannel).toBe('neurobranch:chatgpt-disconnect')
    expect(exportFileChannel).toBe('neurobranch:export-file')
    expect(windowStateChannel).toBe('neurobranch:window-state')
    expect(desktopUpdateStatusChannel).toBe('neurobranch:desktop-update-status')
    expect(launchDesktopUpdateChannel).toBe('neurobranch:desktop-update-launch')
    expect(openDesktopSetupChannel).toBe('neurobranch:desktop-setup-open')
  })

  it('loads a sandbox-compatible CommonJS preload', () => {
    const mainSource = readFileSync(new URL('./main.ts', import.meta.url), 'utf8')
    expect(mainSource).toContain("preload: join(currentDirectory, 'preload.cjs')")
  })
})
