import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { loadDesktopState, saveDesktopState } from './desktop-state'

let directory = ''

afterEach(async () => {
  if (directory) await rm(directory, { recursive: true, force: true })
  directory = ''
})

describe('desktop SQLite workspace state', () => {
  it('keeps independent studio and settings records across database reopen', async () => {
    directory = await mkdtemp(join(tmpdir(), 'labo-ai-state-'))
    await saveDesktopState(directory, 'model', { preset: 'my-model', nodes: 12 })
    await saveDesktopState(directory, 'training', { optimizer: 'my-muon' })
    await saveDesktopState(directory, 'settings', { chatGPT: { model: 'gpt-5.6-sol', effort: 'high' } })

    expect(await loadDesktopState(directory, 'model')).toEqual({ preset: 'my-model', nodes: 12 })
    expect(await loadDesktopState(directory, 'training')).toEqual({ optimizer: 'my-muon' })
    expect(await loadDesktopState(directory, 'settings')).toEqual({ chatGPT: { model: 'gpt-5.6-sol', effort: 'high' } })
    expect(await loadDesktopState(directory, 'tokenizer')).toBeUndefined()
  })

  it('rejects unknown state scopes', async () => {
    directory = await mkdtemp(join(tmpdir(), 'labo-ai-state-'))
    await expect(saveDesktopState(directory, 'secret', {})).rejects.toThrow('Invalid LABO desktop state scope')
  })

  it('atomically merges independent settings patches', async () => {
    directory = await mkdtemp(join(tmpdir(), 'labo-ai-state-'))
    await saveDesktopState(directory, 'settings', {
      appearance: { theme: 'labo-dark', language: 'en' },
      chatGPT: { model: 'gpt-5.6-terra', effort: 'medium' },
    })

    await Promise.all([
      saveDesktopState(directory, 'settings', { appearance: { theme: 'complexity-spectrum' } }),
      saveDesktopState(directory, 'settings', { chatGPT: { effort: 'high' } }),
      saveDesktopState(directory, 'settings', { desktopUpdateChannel: 'stable' }),
    ])

    expect(await loadDesktopState(directory, 'settings')).toEqual({
      appearance: { theme: 'complexity-spectrum', language: 'en' },
      chatGPT: { model: 'gpt-5.6-terra', effort: 'high' },
      desktopUpdateChannel: 'stable',
    })
  })
})
