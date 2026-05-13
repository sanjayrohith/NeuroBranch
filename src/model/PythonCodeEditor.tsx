import { useRef, type ChangeEvent, type ReactNode, type UIEvent } from 'react'

interface PythonCodeEditorProps {
  value: string
  onChange: (value: string) => void
  ariaLabel?: string
  className?: string
}

const keywords = new Set([
  'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del',
  'elif', 'else', 'except', 'False', 'finally', 'for', 'from', 'global', 'if', 'import',
  'in', 'is', 'lambda', 'None', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return',
  'True', 'try', 'while', 'with', 'yield',
])

const builtins = new Set([
  'bool', 'dict', 'enumerate', 'float', 'int', 'len', 'list', 'max', 'min', 'print',
  'range', 'set', 'str', 'sum', 'super', 'tuple', 'zip',
])

const tokenPattern = /#[^\n]*|"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|@[A-Za-z_]\w*|\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b|\b[A-Za-z_]\w*\b/g

function tokenClass(token: string, source: string, start: number, previousWord?: string): string | undefined {
  if (token.startsWith('#')) return 'comment'
  if (token.startsWith('"') || token.startsWith("'")) return 'string'
  if (token.startsWith('@')) return 'decorator'
  if (/^\d/.test(token)) return 'number'
  if (keywords.has(token)) return 'keyword'
  if (previousWord === 'class') return 'class-name'
  if (previousWord === 'def') return 'function-name'
  if (token === 'self') return 'self'
  if (token === 'torch' || token === 'nn' || token === 'F') return 'namespace'
  if (builtins.has(token)) return 'builtin'
  if (/^\s*\(/.test(source.slice(start + token.length))) return 'function-call'
  return undefined
}

function highlightPython(source: string): ReactNode[] {
  const nodes: ReactNode[] = []
  let cursor = 0
  let previousWord: string | undefined
  let match: RegExpExecArray | null
  tokenPattern.lastIndex = 0

  while ((match = tokenPattern.exec(source)) !== null) {
    if (match.index > cursor) nodes.push(source.slice(cursor, match.index))
    const token = match[0]
    const className = tokenClass(token, source, match.index, previousWord)
    nodes.push(className
      ? <span className={`python-token ${className}`} key={`${match.index}-${token}`}>{token}</span>
      : token)
    if (/^[A-Za-z_]\w*$/.test(token)) previousWord = token
    cursor = match.index + token.length
  }
  if (cursor < source.length) nodes.push(source.slice(cursor))
  return nodes
}

export function PythonCodeEditor({ ariaLabel = 'PyTorch editor', className = '', value, onChange }: PythonCodeEditorProps) {
  const highlightRef = useRef<HTMLPreElement>(null)

  const syncScroll = (event: UIEvent<HTMLTextAreaElement>) => {
    if (!highlightRef.current) return
    highlightRef.current.scrollTop = event.currentTarget.scrollTop
    highlightRef.current.scrollLeft = event.currentTarget.scrollLeft
  }

  const update = (event: ChangeEvent<HTMLTextAreaElement>) => onChange(event.target.value)

  return (
    <div className={`python-editor-shell ${className}`}>
      <pre aria-hidden="true" className="code-editor python-syntax-layer" ref={highlightRef}><code>{highlightPython(value)}{'\n'}</code></pre>
      <textarea
        aria-label={ariaLabel}
        className="code-editor pytorch-textarea"
        onChange={update}
        onScroll={syncScroll}
        spellCheck={false}
        value={value}
      />
    </div>
  )
}

export function PythonCodePreview({ className = '', value }: { className?: string; value: string }) {
  return <pre aria-label="Python code preview" className={`code-editor python-code-preview ${className}`}><code>{highlightPython(value)}{`\n`}</code></pre>
}
