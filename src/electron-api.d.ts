interface LaboRuntimeResult {
  atomId: string
  status: 'passed' | 'failed'
  summary?: string
  error?: string
}

interface LaboRuntimeTrace {
  engine: 'pytorch' | 'tokenizers'
  status: 'completed' | 'failed'
  currentAtomId?: string
  error?: string
  tokenIds?: number[]
  modelOutput?: {
    kind: 'logits' | 'tensor'
    tensorShape: number[]
    logitsShape?: number[]
    predictedTokenId?: number
    topTokenIds?: number[]
    topProbabilities?: number[]
  }
  results: LaboRuntimeResult[]
}

interface Window {
  labo?: {
    platform: string
    runtime: 'electron' | 'web'
    runAtomic?(payload: { kind: 'model'; graph: unknown; tokenIds?: number[] } | { kind: 'tokenizer'; pipeline: unknown; sample?: string }): Promise<LaboRuntimeTrace>
    askLabo?(payload: { request: string; context: Record<string, unknown> }): Promise<import('./core/agentic-graph').AgentGraphPlan>
    getOpenAISettings?(): Promise<OpenAISettingsStatus>
    saveOpenAIKey?(apiKey: string): Promise<OpenAISettingsStatus>
    deleteOpenAIKey?(): Promise<OpenAISettingsStatus>
    testOpenAIKey?(): Promise<{ ok: true }>
    getChatGPTSession?(): Promise<ChatGPTSessionStatus>
    connectChatGPT?(): Promise<ChatGPTSessionStatus>
    disconnectChatGPT?(): Promise<ChatGPTSessionStatus>
    configureChatGPT?(configuration: { model: string; effort: string }): Promise<ChatGPTSessionStatus>
    loadWebWorkspace?(): Promise<{ authenticated: boolean; workspace: unknown; customCards: unknown[]; training?: unknown; tokenizer?: unknown; settings?: unknown; updatedAt?: number; warning?: string }>
    saveWebWorkspace?(payload: { workspace?: unknown; customCards?: unknown[]; training?: unknown; tokenizer?: unknown; settings?: unknown }): Promise<{ saved: true; updatedAt: number }>
    exportFile?(payload: { filename: string; content: string; kind: 'svg' | 'python' }): Promise<{ saved: boolean; path?: string }>
    loadDesktopState?(scope: 'model' | 'training' | 'tokenizer' | 'settings'): Promise<unknown>
    saveDesktopState?(scope: 'model' | 'training' | 'tokenizer' | 'settings', data: unknown): Promise<{ saved: true }>
    getDesktopUpdateStatus?(channel?: DesktopUpdateChannel): Promise<DesktopUpdateStatus>
    launchDesktopUpdate?(channel?: DesktopUpdateChannel): Promise<{ launched: true }>
    openDesktopSetup?(): Promise<void>
    getWindowState?(): Promise<{ fullScreen: boolean }>
    onWindowStateChange?(callback: (state: { fullScreen: boolean }) => void): () => void
  }
}

interface OpenAISettingsStatus {
  configured: boolean
  source: 'environment' | 'secure-storage' | 'none'
  encryptionAvailable?: boolean
  authRequired?: boolean
  prefix?: string
}

interface ChatGPTSessionStatus {
  available: boolean
  connected: boolean
  email?: string
  planType?: string
  models?: Array<{ id: string; label: string; description?: string; efforts: string[]; defaultEffort?: string; isDefault: boolean }>
  selectedModel?: string
  selectedEffort?: string
  error?: string
}

interface DesktopUpdateStatus {
  currentVersion: string
  channel: DesktopUpdateChannel
  installedTag?: string
  installedChannel?: DesktopUpdateChannel
  installedRevision?: string
  latestTag?: string
  latestRevision?: string
  latestCheckedAt?: number
  cachedLatest?: boolean
  helperInstalled: boolean
  updateAvailable: boolean
  setupUrl: string
  error?: string
}

type DesktopUpdateChannel = 'stable' | 'main'
