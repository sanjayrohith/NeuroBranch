const VIEWPORT_PADDING = 8

interface ContextMenuBounds {
  height: number
  width: number
}

export function fitContextMenuToViewport(position: { x: number; y: number }, menu: ContextMenuBounds, viewport: ContextMenuBounds, padding = VIEWPORT_PADDING) {
  const maxX = Math.max(padding, viewport.width - menu.width - padding)
  const maxY = Math.max(padding, viewport.height - menu.height - padding)
  return {
    x: Math.min(Math.max(position.x, padding), maxX),
    y: Math.min(Math.max(position.y, padding), maxY),
  }
}
