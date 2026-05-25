import { ListChecks, Send, Sparkles } from 'lucide-react'
import { useLayoutEffect, useRef } from 'react'

export type AgentPromptMode = 'architecture' | 'card-editing' | 'reusable-card'

const MODE_COPY: Record<AgentPromptMode, { ariaLabel: string; placeholder: string; context: string; submitLabel: string }> = {
  architecture: {
    ariaLabel: 'What should these blocks build?',
    placeholder: 'Ask LABO to build or extend your architecture…',
    context: 'Build architecture',
    submitLabel: 'Propose graph changes',
  },
  'card-editing': {
    ariaLabel: 'What should these blocks build?',
    placeholder: 'Ask LABO to edit selected cards or connections…',
    context: 'Edit selected cards',
    submitLabel: 'Propose graph changes',
  },
  'reusable-card': {
    ariaLabel: 'Custom card need',
    placeholder: 'Ask LABO to compose this reusable card…',
    context: 'Reusable card · current card only',
    submitLabel: 'Compose card graph',
  },
}

interface AgentPromptDetails {
  active: boolean
  count?: number | string
  disabled?: boolean
  label: string
  onToggle(): void
}

export function AgentPrompt({ busy = false, context, details, disabled = false, mode, onChange, onSubmit, value }: {
  busy?: boolean
  context?: string
  details?: AgentPromptDetails
  disabled?: boolean
  mode: AgentPromptMode
  onChange(value: string): void
  onSubmit(): void
  value: string
}) {
  const copy = MODE_COPY[mode]
  const formRef = useRef<HTMLFormElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useLayoutEffect(() => {
    const textarea = textareaRef.current
    if (!textarea) return
    textarea.style.height = 'auto'
    const height = Math.min(Math.max(textarea.scrollHeight, 27), 96)
    textarea.style.height = `${height}px`
    textarea.style.overflowY = textarea.scrollHeight > 96 ? 'auto' : 'hidden'
  }, [value])

  useLayoutEffect(() => {
    const form = formRef.current
    if (!form) return
    const publishHeight = () => document.documentElement.style.setProperty('--labo-agent-prompt-height', `${form.offsetHeight}px`)
    publishHeight()
    if (typeof ResizeObserver === 'undefined') {
      return () => document.documentElement.style.removeProperty('--labo-agent-prompt-height')
    }
    const observer = new ResizeObserver(publishHeight)
    observer.observe(form)
    return () => {
      observer.disconnect()
      document.documentElement.style.removeProperty('--labo-agent-prompt-height')
    }
  }, [])

  return <div className="agent-prompt-dock">
    <form className="agent-prompt" data-mode={mode} onSubmit={(event) => { event.preventDefault(); onSubmit() }} ref={formRef}>
      <textarea
        aria-label={copy.ariaLabel}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => {
          if (event.key !== 'Enter' || event.shiftKey || event.nativeEvent.isComposing) return
          event.preventDefault()
          event.currentTarget.form?.requestSubmit()
        }}
        placeholder={copy.placeholder}
        ref={textareaRef}
        rows={1}
        value={value}
      />
      <div className="agent-prompt-context">
        <span><Sparkles size={12} />LABO agent</span>
        <small>{context ? `${copy.context} · ${context}` : copy.context}</small>
      </div>
      <button className="agent-prompt-send" aria-label={copy.submitLabel} disabled={disabled || busy || !value.trim()} title={busy ? 'Inspecting graph' : 'Send to LABO'} type="submit">
        <Send size={15} />
      </button>
      {details && <button aria-expanded={details.active} aria-label={details.label} className="agent-prompt-details" disabled={details.disabled} onClick={details.onToggle} title={details.label} type="button">
        <ListChecks size={14} />
        {details.count !== undefined && <b>{details.count}</b>}
      </button>}
    </form>
  </div>
}
