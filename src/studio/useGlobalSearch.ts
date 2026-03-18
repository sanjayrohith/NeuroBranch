import { useCallback, useEffect, useState } from 'react'

export function useGlobalSearch(enabled = true) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const close = useCallback(() => { setOpen(false); setQuery('') }, [])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (enabled && (event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        setOpen(true)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [enabled])

  useEffect(() => {
    if (!enabled) close()
  }, [close, enabled])

  return { close, open, openSearch: () => { if (enabled) setOpen(true) }, query, setQuery }
}
