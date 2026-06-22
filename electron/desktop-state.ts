import { mkdir } from 'node:fs/promises'
import { join } from 'node:path'
import { DatabaseSync } from 'node:sqlite'

export type DesktopStateScope = 'model' | 'training' | 'tokenizer' | 'settings'

const stateDirectoryName = 'labo-ai'
const databaseFilename = 'workspaces.sqlite'
let writeQueue: Promise<void> = Promise.resolve()

function validScope(value: unknown): value is DesktopStateScope {
  return value === 'model' || value === 'training' || value === 'tokenizer' || value === 'settings'
}

export function desktopStateDatabasePath(userDataDirectory: string): string {
  return join(userDataDirectory, stateDirectoryName, databaseFilename)
}

async function openDatabase(userDataDirectory: string): Promise<DatabaseSync> {
  await mkdir(join(userDataDirectory, stateDirectoryName), { recursive: true })
  const database = new DatabaseSync(desktopStateDatabasePath(userDataDirectory))
  database.exec(`
    PRAGMA journal_mode = WAL;
    PRAGMA synchronous = FULL;
    CREATE TABLE IF NOT EXISTS workspace_state (
      scope TEXT PRIMARY KEY NOT NULL,
      payload TEXT NOT NULL,
      updated_at INTEGER NOT NULL
    );
  `)
  return database
}

export async function loadDesktopState(userDataDirectory: string, scope: unknown): Promise<unknown> {
  if (!validScope(scope)) throw new Error('Invalid LABO desktop state scope')
  await writeQueue
  const database = await openDatabase(userDataDirectory)
  try {
    const row = database.prepare('SELECT payload FROM workspace_state WHERE scope = ?').get(scope) as { payload?: unknown } | undefined
    return typeof row?.payload === 'string' ? JSON.parse(row.payload) as unknown : undefined
  } catch {
    return undefined
  } finally {
    database.close()
  }
}

export function saveDesktopState(userDataDirectory: string, scope: unknown, data: unknown): Promise<{ saved: true }> {
  if (!validScope(scope)) return Promise.reject(new Error('Invalid LABO desktop state scope'))
  if (scope === 'settings' && (!data || typeof data !== 'object' || Array.isArray(data))) {
    return Promise.reject(new Error('Invalid LABO desktop settings patch'))
  }
  const serialized = JSON.stringify(data)
  if (typeof serialized !== 'string') return Promise.reject(new Error('Invalid LABO desktop state payload'))
  if (serialized.length > 10_000_000) return Promise.reject(new Error('LABO desktop state is too large'))
  writeQueue = writeQueue.then(async () => {
    const database = await openDatabase(userDataDirectory)
    try {
      const conflictUpdate = scope === 'settings'
        ? 'payload = json_patch(workspace_state.payload, excluded.payload), updated_at = excluded.updated_at'
        : 'payload = excluded.payload, updated_at = excluded.updated_at'
      database.prepare(`
        INSERT INTO workspace_state(scope, payload, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(scope) DO UPDATE SET ${conflictUpdate}
      `).run(scope, serialized, Date.now())
    } finally {
      database.close()
    }
  })
  return writeQueue.then(() => ({ saved: true }))
}
