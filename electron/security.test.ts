import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { rendererWebPreferences } from './security'
import { askLaboChannel, atomicRuntimeChannel, chatGPTSessionChannel, connectChatGPTChannel, deleteOpenAIKeyChannel, desktopUpdateStatusChannel, disconnectChatGPTChannel, exportFileChannel, launchDesktopUpdateChannel, openAISettingsChannel, openDesktopSetupChannel, saveOpenAIKeyChannel, testOpenAIKeyChannel, windowStateChannel } from './ipc-contract'

describe('Electron renderer boundary', () => {
  it('isolates the LABO AI renderer from Node and the main process', () => {
    expect(rendererWebPreferences.contextIsolation).toBe(true)
    expect(rendererWebPreferences.nodeIntegration).toBe(false)
    expect(rendererWebPreferences.sandbox).toBe(true)
  })

  it('exposes only the named LABO IPC channels', () => {
    expect(atomicRuntimeChannel).toBe('labo:atomic-runtime')
    expect(askLaboChannel).toBe('labo:ask')
    expect(openAISettingsChannel).toBe('labo:openai-settings')
    expect(saveOpenAIKeyChannel).toBe('labo:openai-key-save')
    expect(deleteOpenAIKeyChannel).toBe('labo:openai-key-delete')
    expect(testOpenAIKeyChannel).toBe('labo:openai-key-test')
    expect(chatGPTSessionChannel).toBe('labo:chatgpt-session')
    expect(connectChatGPTChannel).toBe('labo:chatgpt-connect')
    expect(disconnectChatGPTChannel).toBe('labo:chatgpt-disconnect')
    expect(exportFileChannel).toBe('labo:export-file')
    expect(windowStateChannel).toBe('labo:window-state')
    expect(desktopUpdateStatusChannel).toBe('labo:desktop-update-status')
    expect(launchDesktopUpdateChannel).toBe('labo:desktop-update-launch')
    expect(openDesktopSetupChannel).toBe('labo:desktop-setup-open')
  })

  it('loads a sandbox-compatible CommonJS preload', () => {
    const mainSource = readFileSync(new URL('./main.ts', import.meta.url), 'utf8')
    expect(mainSource).toContain("preload: join(currentDirectory, 'preload.cjs')")
  })
})
