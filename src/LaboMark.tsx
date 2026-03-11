type LaboMarkProps = {
  className?: string
}

export function LaboMark({ className }: LaboMarkProps) {
  return (
    <svg aria-hidden="true" className={className} viewBox="0 0 64 64" fill="none">
      <path d="M17 17 32 32 47 17M17 47l15-15 15 15" stroke="currentColor" strokeWidth="3" strokeLinecap="round" opacity=".72" />
      <path d="M17 17h30M17 47h30" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" opacity=".16" />
      <rect x="26" y="26" width="12" height="12" rx="3" transform="rotate(45 32 32)" fill="#c4b5fd" />
      <circle cx="17" cy="17" r="5" fill="#6ee7b7" />
      <circle cx="47" cy="17" r="5" fill="#7dd3fc" />
      <circle cx="17" cy="47" r="5" fill="#fcd34d" />
      <circle cx="47" cy="47" r="5" fill="#f9a8d4" />
      <circle cx="32" cy="32" r="2.2" fill="#fff" opacity=".82" />
    </svg>
  )
}
