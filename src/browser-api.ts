function apiError(body: unknown, fallback: string): Error {
  if (body && typeof body === 'object' && 'error' in body) {
    const value = (body as { error?: unknown }).error
    if (typeof value === 'string') return new Error(value)
    if (value && typeof value === 'object' && 'message' in value && typeof (value as { message?: unknown }).message === 'string') {
      return new Error((value as { message: string }).message)
    }
  }
  return new Error(fallback)
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    credentials: 'same-origin',
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  const body = await response.json().catch(() => undefined) as unknown
  if (!response.ok) throw apiError(body, `LABO web request failed (${response.status})`)
  return body as T
}

type WebWorkspaceSave = {
  workspace?: unknown
  customCards?: unknown[]
  training?: unknown
  tokenizer?: unknown
  settings?: unknown
}

type WebWorkspaceSaveResult = { saved: true; updatedAt: number }

let pendingWorkspaceSave: WebWorkspaceSave | undefined
let pendingWorkspaceWaiters: Array<{
  resolve: (value: WebWorkspaceSaveResult) => void
  reject: (error: unknown) => void
}> = []
let workspaceSaveTimer: ReturnType<typeof setTimeout> | undefined
let workspaceSaveInFlight = false

function scheduleWorkspaceSave(): void {
  if (workspaceSaveTimer || workspaceSaveInFlight) return
  workspaceSaveTimer = setTimeout(() => {
    workspaceSaveTimer = undefined
    void flushWorkspaceSave()
  }, 120)
}

async function flushWorkspaceSave(): Promise<void> {
  if (workspaceSaveInFlight || !pendingWorkspaceSave) return
  workspaceSaveInFlight = true
  const payload = pendingWorkspaceSave
  const waiters = pendingWorkspaceWaiters
  pendingWorkspaceSave = undefined
  pendingWorkspaceWaiters = []
  try {
    const result = await requestJson<WebWorkspaceSaveResult>('/api/labo/workspace', {
      method: 'PUT',
      body: JSON.stringify(payload),
    })
    for (const waiter of waiters) waiter.resolve(result)
  } catch (error) {
    for (const waiter of waiters) waiter.reject(error)
  } finally {
    workspaceSaveInFlight = false
    if (pendingWorkspaceSave) scheduleWorkspaceSave()
  }
}

function queueWorkspaceSave(payload: WebWorkspaceSave): Promise<WebWorkspaceSaveResult> {
  const previousSettings = pendingWorkspaceSave?.settings && typeof pendingWorkspaceSave.settings === 'object'
    ? pendingWorkspaceSave.settings as Record<string, unknown>
    : undefined
  const nextSettings = payload.settings && typeof payload.settings === 'object'
    ? payload.settings as Record<string, unknown>
    : undefined
  const previousAppearance = previousSettings?.appearance && typeof previousSettings.appearance === 'object'
    ? previousSettings.appearance as Record<string, unknown>
    : undefined
  const nextAppearance = nextSettings?.appearance && typeof nextSettings.appearance === 'object'
    ? nextSettings.appearance as Record<string, unknown>
    : undefined
  pendingWorkspaceSave = {
    ...pendingWorkspaceSave,
    ...payload,
    ...(nextSettings ? {
      settings: {
        ...previousSettings,
        ...nextSettings,
        appearance: { ...previousAppearance, ...nextAppearance },
      },
    } : {}),
  }
  const promise = new Promise<WebWorkspaceSaveResult>((resolve, reject) => {
    pendingWorkspaceWaiters.push({ resolve, reject })
  })
  scheduleWorkspaceSave()
  return promise
}

let knownWebAuthentication: boolean | undefined

function recordWebAuthentication(authenticated: boolean, reloadOnChange = false): void {
  const changed = knownWebAuthentication !== undefined && knownWebAuthentication !== authenticated
  knownWebAuthentication = authenticated
  if (changed && reloadOnChange) window.location.reload()
}

async function loadWebWorkspace() {
  type Result = Awaited<ReturnType<NonNullable<NonNullable<Window['labo']>['loadWebWorkspace']>>>
  const workspace = await requestJson<Result>('/api/labo/workspace')
  recordWebAuthentication(workspace.authenticated)
  return workspace
}

async function refreshWebAuthentication(): Promise<void> {
  try {
    const workspace = await requestJson<{ authenticated: boolean }>('/api/labo/workspace')
    recordWebAuthentication(workspace.authenticated, true)
  } catch {
    // A temporary network failure must not tear down an active studio.
  }
}

async function openAISettings(): Promise<OpenAISettingsStatus> {
  const response = await fetch('/api/labo/key', { credentials: 'same-origin' })
  if (response.status === 401) return { configured: false, source: 'none', encryptionAvailable: true, authRequired: true }
  const body = await response.json().catch(() => undefined) as unknown
  if (!response.ok) throw apiError(body, `LABO web request failed (${response.status})`)
  return body as OpenAISettingsStatus
}

export function installBrowserApi(): void {
  if (window.labo || window.location.protocol === 'file:') return

  window.labo = {
    platform: 'web',
    runtime: 'web',
    askLabo: (payload) => requestJson('/api/labo/ask', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
    getOpenAISettings: openAISettings,
    saveOpenAIKey: (apiKey) => requestJson('/api/labo/key', {
      method: 'POST',
      body: JSON.stringify({ apiKey }),
    }),
    deleteOpenAIKey: () => requestJson('/api/labo/key', { method: 'DELETE' }),
    testOpenAIKey: () => requestJson('/api/labo/key/test', { method: 'POST' }),
    loadWebWorkspace,
    saveWebWorkspace: queueWorkspaceSave,
  }
  window.addEventListener('focus', refreshWebAuthentication)
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') void refreshWebAuthentication()
  })
}
