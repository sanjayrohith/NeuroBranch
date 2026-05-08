export interface Point2D {
  x: number
  y: number
}

export interface GraphViewport extends Point2D {
  zoom: number
}

export const MIN_GRAPH_ZOOM = 0.2
export const MAX_GRAPH_ZOOM = 2.4

export function clampZoom(zoom: number): number {
  return Math.min(MAX_GRAPH_ZOOM, Math.max(MIN_GRAPH_ZOOM, zoom))
}

export function screenToWorld(point: Point2D, viewport: GraphViewport): Point2D {
  return {
    x: (point.x - viewport.x) / viewport.zoom,
    y: (point.y - viewport.y) / viewport.zoom,
  }
}

export function panViewport(viewport: GraphViewport, deltaX: number, deltaY: number): GraphViewport {
  return { ...viewport, x: viewport.x + deltaX, y: viewport.y + deltaY }
}

export function graphGridStyle(viewport: GraphViewport, spacing = 24): { backgroundPosition: string; backgroundSize: string } {
  const scaledSpacing = spacing * viewport.zoom
  const wrap = (value: number) => ((value % scaledSpacing) + scaledSpacing) % scaledSpacing
  return {
    backgroundPosition: `${wrap(viewport.x)}px ${wrap(viewport.y)}px`,
    backgroundSize: `${scaledSpacing}px ${scaledSpacing}px`,
  }
}

export function zoomViewportAt(viewport: GraphViewport, requestedZoom: number, pointer: Point2D): GraphViewport {
  const zoom = clampZoom(requestedZoom)
  const world = screenToWorld(pointer, viewport)
  return {
    x: pointer.x - world.x * zoom,
    y: pointer.y - world.y * zoom,
    zoom,
  }
}
