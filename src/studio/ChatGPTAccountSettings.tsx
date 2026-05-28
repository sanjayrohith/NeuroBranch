import { LogIn, LogOut, ShieldCheck, UserRound } from 'lucide-react'

export function ChatGPTAccountSettings({ busy, session, onConfigurationChange, onConnect, onDisconnect }: {
  busy: boolean
  session?: ChatGPTSessionStatus
  onConfigurationChange(configuration: { model: string; effort: string }): void
  onConnect(): void
  onDisconnect(): void
}) {
  const selectedModel = session?.models?.find((model) => model.id === session.selectedModel) ?? session?.models?.find((model) => model.isDefault) ?? session?.models?.[0]
  const selectedEffort = selectedModel?.efforts.includes(session?.selectedEffort ?? '') ? session?.selectedEffort ?? '' : selectedModel?.defaultEffort ?? selectedModel?.efforts[0] ?? ''
  const configure = (modelId: string, effort?: string) => {
    const model = session?.models?.find((candidate) => candidate.id === modelId)
    if (!model) return
    const nextEffort = effort && model.efforts.includes(effort) ? effort : model.defaultEffort ?? model.efforts[0] ?? ''
    onConfigurationChange({ model: model.id, effort: nextEffort })
  }

  return <section className="ask-labo-key-settings ask-labo-chatgpt-settings">
    <div className="ask-labo-key-heading"><span><UserRound size={13} />ChatGPT account</span>{session?.connected && <b><span className="status-dot" />Connected</b>}</div>
    {session?.connected ? <>
      <div className="ask-labo-key-status"><ShieldCheck size={16} /><span><strong>{session.email || 'ChatGPT account connected'}</strong><small>{session.planType ? `${session.planType} plan · ` : ''}Ask LABO uses this session by default. No API key is required.</small></span></div>
      {selectedModel && <div className="chatgpt-agent-controls">
        <label><span>Model</span><select aria-label="ChatGPT model" disabled={busy} onChange={(event) => configure(event.target.value)} value={selectedModel.id}>{session.models?.map((model) => <option key={model.id} value={model.id}>{model.label}</option>)}</select></label>
        <label><span>Reasoning</span><select aria-label="ChatGPT reasoning effort" disabled={busy || selectedModel.efforts.length === 0} onChange={(event) => configure(selectedModel.id, event.target.value)} value={selectedEffort}>{selectedModel.efforts.map((effort) => <option key={effort} value={effort}>{effort.replaceAll('-', ' ')}</option>)}</select></label>
        {selectedModel.description && <small>{selectedModel.description}</small>}
      </div>}
      <div className="ask-labo-key-actions"><button disabled={busy} onClick={onDisconnect} type="button"><LogOut size={12} />Disconnect ChatGPT</button></div>
    </> : <div className="ask-labo-key-form">
      <p>Choose the ChatGPT account used only by LABO AI. Sign-in opens in your browser and never reuses the Codex CLI or VS Code session.</p>
      <button disabled={busy} onClick={onConnect} type="button"><LogIn size={12} />{busy ? 'Waiting for sign-in…' : session?.available === false ? 'Retry ChatGPT connection' : 'Continue with ChatGPT'}</button>
      {session?.error && <small>{session.error}</small>}
    </div>}
  </section>
}
