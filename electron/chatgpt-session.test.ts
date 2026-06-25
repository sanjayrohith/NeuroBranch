import { existsSync } from 'node:fs'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { CodexAppServer, chatGPTPlanSchema, dedicatedCodexEnvironment } from './chatgpt-session'

describe('ChatGPT structured graph plan schema', () => {
  it('declares a JSON type for every constant discriminator', () => {
    const missingTypes: string[] = []
    const visit = (value: unknown, path = '$') => {
      if (!value || typeof value !== 'object') return
      if (Array.isArray(value)) {
        value.forEach((item, index) => visit(item, `${path}[${index}]`))
        return
      }
      const record = value as Record<string, unknown>
      if ('const' in record && typeof record.type !== 'string') missingTypes.push(path)
      for (const [key, child] of Object.entries(record)) visit(child, `${path}.${key}`)
    }

    visit(chatGPTPlanSchema)
    expect(missingTypes).toEqual([])
  })

  it('does not start or reuse a shared account before an explicit LABO sign-in', async () => {
    const codexHome = await mkdtemp(join(tmpdir(), 'labo-codex-test-'))
    const session = new CodexAppServer(async () => undefined, 'test', codexHome)
    try {
      await expect(session.status()).resolves.toEqual({ available: true, connected: false })
      expect(existsSync(join(codexHome, 'auth.json'))).toBe(false)
      expect(existsSync(join(codexHome, 'config.toml'))).toBe(false)
    } finally {
      session.stop()
      await rm(codexHome, { recursive: true, force: true })
    }
  })

  it('forces ChatGPT file authentication and removes inherited API credentials', async () => {
    const codexHome = await mkdtemp(join(tmpdir(), 'labo-codex-env-test-'))
    try {
      const environment = dedicatedCodexEnvironment(codexHome, {
        OPENAI_API_KEY: 'not-forwarded',
        CODEX_API_KEY: 'not-forwarded',
        CODEX_ACCESS_TOKEN: 'not-forwarded',
      })
      expect(environment.CODEX_HOME).toBe(codexHome)
      expect(environment.OPENAI_API_KEY).toBeUndefined()
      expect(environment.CODEX_API_KEY).toBeUndefined()
      expect(environment.CODEX_ACCESS_TOKEN).toBeUndefined()
      await expect(readFile(join(codexHome, 'config.toml'), 'utf8')).resolves.toBe(
        'cli_auth_credentials_store = "file"\nforced_login_method = "chatgpt"\n',
      )
    } finally {
      await rm(codexHome, { recursive: true, force: true })
    }
  })
})
