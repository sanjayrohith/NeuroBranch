import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const electronMocks = vi.hoisted(() => ({
  getPath: vi.fn(),
  isAsyncEncryptionAvailable: vi.fn(async () => true),
  encryptStringAsync: vi.fn(async (value: string) => Buffer.from(`encrypted:${value}`)),
  decryptStringAsync: vi.fn(async (value: Buffer) => ({ result: value.toString().replace(/^encrypted:/, ''), shouldReEncrypt: false })),
}))

vi.mock('electron', () => ({
  app: { getPath: electronMocks.getPath },
  safeStorage: {
    isAsyncEncryptionAvailable: electronMocks.isAsyncEncryptionAvailable,
    encryptStringAsync: electronMocks.encryptStringAsync,
    decryptStringAsync: electronMocks.decryptStringAsync,
  },
}))

import { deleteOpenAIApiKey, getOpenAISettingsStatus, resolveOpenAIConfig, saveOpenAIApiKey, testOpenAIConnection } from './openai-credentials'

let userDataDirectory = ''
const previousApiKey = process.env.OPENAI_API_KEY

beforeEach(async () => {
  userDataDirectory = await mkdtemp(join(tmpdir(), 'labo-ai-credentials-'))
  electronMocks.getPath.mockReturnValue(userDataDirectory)
  electronMocks.isAsyncEncryptionAvailable.mockClear()
  electronMocks.encryptStringAsync.mockClear()
  electronMocks.decryptStringAsync.mockClear()
  delete process.env.OPENAI_API_KEY
})

afterEach(async () => {
  vi.unstubAllGlobals()
  await rm(userDataDirectory, { recursive: true, force: true })
  if (previousApiKey === undefined) delete process.env.OPENAI_API_KEY
  else process.env.OPENAI_API_KEY = previousApiKey
})

describe('per-user OpenAI credentials', () => {
  it('checks configuration without touching the system keychain', async () => {
    await expect(getOpenAISettingsStatus()).resolves.toMatchObject({ configured: false, source: 'none' })
    expect(electronMocks.isAsyncEncryptionAvailable).not.toHaveBeenCalled()
    expect(electronMocks.decryptStringAsync).not.toHaveBeenCalled()

    await saveOpenAIApiKey({ apiKey: 'sk-project-user-secret-123456789' })
    electronMocks.isAsyncEncryptionAvailable.mockClear()
    electronMocks.decryptStringAsync.mockClear()
    await expect(getOpenAISettingsStatus()).resolves.toMatchObject({ configured: true, source: 'secure-storage' })
    expect(electronMocks.isAsyncEncryptionAvailable).not.toHaveBeenCalled()
    expect(electronMocks.decryptStringAsync).not.toHaveBeenCalled()
  })

  it('encrypts, resolves, and removes a user key without writing plaintext', async () => {
    const apiKey = 'sk-project-user-secret-123456789'
    await expect(getOpenAISettingsStatus()).resolves.toMatchObject({ configured: false, source: 'none' })

    await expect(saveOpenAIApiKey({ apiKey })).resolves.toMatchObject({ configured: true, source: 'secure-storage' })
    await expect(resolveOpenAIConfig()).resolves.toMatchObject({ apiKey })
    const stored = await readFile(join(userDataDirectory, 'labo-ai', 'openai-credentials.json'), 'utf8')
    expect(stored).not.toContain(apiKey)

    await expect(deleteOpenAIApiKey()).resolves.toMatchObject({ configured: false, source: 'none' })
    await expect(resolveOpenAIConfig()).resolves.toBeUndefined()
  })

  it('tests the stored key only from the Electron main process', async () => {
    await saveOpenAIApiKey({ apiKey: 'sk-project-user-secret-123456789' })
    const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      expect(init?.headers).toMatchObject({ Authorization: 'Bearer sk-project-user-secret-123456789' })
      return new Response('{}', { status: 200 })
    })
    vi.stubGlobal('fetch', fetchMock)

    await expect(testOpenAIConnection()).resolves.toEqual({ ok: true })
    expect(fetchMock).toHaveBeenCalledOnce()
  })
})
