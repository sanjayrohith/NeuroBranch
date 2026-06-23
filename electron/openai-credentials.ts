import { app, safeStorage } from 'electron'
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

export type OpenAIKeySource = 'environment' | 'secure-storage' | 'none'

export interface OpenAISettingsStatus {
  configured: boolean
  source: OpenAIKeySource
  encryptionAvailable?: boolean
}

interface StoredCredentials {
  version: 1
  encryptedApiKey: string
}

function credentialsDirectory(): string {
  return join(app.getPath('userData'), 'labo-ai')
}

function credentialsPath(): string {
  return join(credentialsDirectory(), 'openai-credentials.json')
}

async function encryptionAvailable(): Promise<boolean> {
  try {
    return await safeStorage.isAsyncEncryptionAvailable()
  } catch {
    return false
  }
}

async function readStoredCredentials(): Promise<StoredCredentials | undefined> {
  try {
    const parsed = JSON.parse(await readFile(credentialsPath(), 'utf8')) as Partial<StoredCredentials>
    if (parsed.version !== 1 || typeof parsed.encryptedApiKey !== 'string' || !parsed.encryptedApiKey) return undefined
    return { version: 1, encryptedApiKey: parsed.encryptedApiKey }
  } catch {
    return undefined
  }
}

async function storedApiKey(): Promise<string | undefined> {
  const stored = await readStoredCredentials()
  if (!stored || !await encryptionAvailable()) return undefined
  try {
    const decrypted = await safeStorage.decryptStringAsync(Buffer.from(stored.encryptedApiKey, 'base64'))
    return decrypted.result.trim() || undefined
  } catch {
    return undefined
  }
}

export async function getOpenAISettingsStatus(): Promise<OpenAISettingsStatus> {
  // Status checks must be passive. Probing or decrypting Electron safeStorage
  // can wake the macOS keychain, so only touch it after an explicit API action.
  if (await readStoredCredentials()) {
    return { configured: true, source: 'secure-storage' }
  }
  if (process.env.OPENAI_API_KEY?.trim()) {
    return { configured: true, source: 'environment' }
  }
  return { configured: false, source: 'none' }
}

export async function saveOpenAIApiKey(payload: { apiKey: string }): Promise<OpenAISettingsStatus> {
  const apiKey = payload?.apiKey?.trim()
  if (!apiKey || apiKey.length < 20) throw new Error('OpenAI API key is too short')
  if (!await encryptionAvailable()) throw new Error('Secure credential storage is unavailable on this system')

  const encrypted = await safeStorage.encryptStringAsync(apiKey)
  const directory = credentialsDirectory()
  const target = credentialsPath()
  const temporary = `${target}.tmp`
  await mkdir(directory, { recursive: true, mode: 0o700 })
  await writeFile(temporary, JSON.stringify({ version: 1, encryptedApiKey: encrypted.toString('base64') }), { encoding: 'utf8', mode: 0o600 })
  await rename(temporary, target)
  return getOpenAISettingsStatus()
}

export async function deleteOpenAIApiKey(): Promise<OpenAISettingsStatus> {
  await rm(credentialsPath(), { force: true })
  return getOpenAISettingsStatus()
}

export async function resolveOpenAIConfig(): Promise<{ apiKey: string; model: string } | undefined> {
  const stored = await storedApiKey()
  const apiKey = stored ?? process.env.OPENAI_API_KEY?.trim()
  if (!apiKey) return undefined
  return { apiKey, model: process.env.OPENAI_MODEL?.trim() || 'gpt-5.6-terra' }
}

export async function testOpenAIConnection(): Promise<{ ok: true }> {
  const config = await resolveOpenAIConfig()
  if (!config) throw new Error('No OpenAI API key is configured')
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 15_000)
  try {
    const response = await fetch('https://api.openai.com/v1/models', {
      signal: controller.signal,
      headers: { Authorization: `Bearer ${config.apiKey}` },
    })
    if (!response.ok) {
      let message = `OpenAI rejected the key with status ${response.status}`
      try {
        const body = await response.json() as { error?: { message?: string } }
        if (body.error?.message) message = body.error.message
      } catch {
        // Keep the status-only error when OpenAI does not return JSON.
      }
      throw new Error(message)
    }
    return { ok: true }
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') throw new Error('OpenAI connection test timed out')
    throw error
  } finally {
    clearTimeout(timeout)
  }
}
