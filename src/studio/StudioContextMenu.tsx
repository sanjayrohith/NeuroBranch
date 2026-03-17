import { useLayoutEffect, useRef, useState, type CSSProperties, type PointerEvent, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { fitContextMenuToViewport } from './contextMenuPosition'

export function StudioContextMenu({ children, className = '', position }: { children: ReactNode; className?: string; position: { x: number; y: number } }) {
  const { x, y } = position
  const menuRef = useRef<HTMLDivElement | null>(null)
  const [layout, setLayout] = useState({ sourceX: Number.NaN, sourceY: Number.NaN, x, y })

  useLayoutEffect(() => {
    const updatePosition = () => {
      const menu = menuRef.current
      if (!menu) return
      const bounds = menu.getBoundingClientRect()
      const fitted = fitContextMenuToViewport({ x, y }, bounds, { width: window.innerWidth, height: window.innerHeight })
      setLayout((current) => current.sourceX === x && current.sourceY === y && current.x === fitted.x && current.y === fitted.y
        ? current
        : { sourceX: x, sourceY: y, ...fitted })
    }

    updatePosition()
    window.addEventListener('resize', updatePosition)
    const observer = typeof ResizeObserver === 'undefined' ? undefined : new ResizeObserver(updatePosition)
    if (menuRef.current) observer?.observe(menuRef.current)
    return () => {
      observer?.disconnect()
      window.removeEventListener('resize', updatePosition)
    }
  }, [x, y])

  const positioned = layout.sourceX === x && layout.sourceY === y
  return createPortal(<div
    className={`card-context-menu ${className}`.trim()}
    ref={menuRef}
    role="menu"
    style={{ left: positioned ? layout.x : x, top: positioned ? layout.y : y, visibility: positioned ? undefined : 'hidden' } as CSSProperties}
    onPointerDown={(event: PointerEvent<HTMLDivElement>) => event.stopPropagation()}
  >{children}</div>, document.body)
}

export function StudioContextMenuItem({ children, className = '', onClick }: { children: ReactNode; className?: string; onClick(): void }) {
  return <button className={className} onClick={onClick} role="menuitem" type="button">{children}</button>
}
