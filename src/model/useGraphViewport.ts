import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent, type RefObject } from 'react'
import { clampZoom, graphGridStyle, panViewport, screenToWorld, zoomViewportAt, type GraphViewport } from './viewport'

const DEFAULT_VIEWPORT: GraphViewport = { x: 0, y: 0, zoom: 1 }

type PanGesture = { pointerId: number; clientX: number; clientY: number; viewport: GraphViewport }

export function useGraphViewport(canvasRef: RefObject<HTMLDivElement | null>) {
  const [viewport, setViewport] = useState<GraphViewport>(DEFAULT_VIEWPORT)
  const [isPanning, setIsPanning] = useState(false)
  const panGesture = useRef<PanGesture | null>(null)
  const spacePressed = useRef(false)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || typeof ResizeObserver === 'undefined') return
    let previous = canvas.getBoundingClientRect()
    const observer = new ResizeObserver(([entry]) => {
      if (!entry) return
      const next = entry.contentRect
      const deltaX = (next.width - previous.width) / 2
      const deltaY = (next.height - previous.height) / 2
      previous = next
      if (deltaX || deltaY) setViewport((current) => ({ ...current, x: current.x + deltaX, y: current.y + deltaY }))
    })
    observer.observe(canvas)
    return () => observer.disconnect()
  }, [canvasRef])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const handleWheel = (event: globalThis.WheelEvent) => {
      event.preventDefault()
      if (event.ctrlKey || event.metaKey) {
        const bounds = canvas.getBoundingClientRect()
        const pointer = { x: event.clientX - bounds.left, y: event.clientY - bounds.top }
        setViewport((current) => zoomViewportAt(current, current.zoom * Math.exp(-event.deltaY * 0.002), pointer))
        return
      }
      setViewport((current) => panViewport(current, -event.deltaX, -event.deltaY))
    }
    canvas.addEventListener('wheel', handleWheel, { passive: false })
    return () => canvas.removeEventListener('wheel', handleWheel)
  }, [canvasRef])

  const canvasCenter = () => {
    const rect = canvasRef.current?.getBoundingClientRect()
    return rect ? { x: rect.width / 2, y: rect.height / 2 } : { x: 0, y: 0 }
  }

  const zoomBy = (delta: number) => setViewport((current) => zoomViewportAt(current, clampZoom(Number((current.zoom + delta).toFixed(2))), canvasCenter()))
  const resetZoom = () => setViewport((current) => zoomViewportAt(current, 1, canvasCenter()))

  const fitElements = (elements: HTMLElement[], maximumZoom = 1.4) => {
    const canvas = canvasRef.current
    if (!canvas) return
    if (elements.length === 0) return setViewport(DEFAULT_VIEWPORT)
    const canvasRect = canvas.getBoundingClientRect()
    if (canvasRect.width <= 0 || canvasRect.height <= 0) return
    const boxes = elements.map((element) => {
      const rect = element.getBoundingClientRect()
      const topLeft = screenToWorld({ x: rect.left - canvasRect.left, y: rect.top - canvasRect.top }, viewport)
      return { left: topLeft.x, top: topLeft.y, right: topLeft.x + rect.width / viewport.zoom, bottom: topLeft.y + rect.height / viewport.zoom }
    })
    const left = Math.min(...boxes.map((box) => box.left))
    const top = Math.min(...boxes.map((box) => box.top))
    const right = Math.max(...boxes.map((box) => box.right))
    const bottom = Math.max(...boxes.map((box) => box.bottom))
    const width = Math.max(1, right - left)
    const height = Math.max(1, bottom - top)
    const padding = 70
    const zoom = clampZoom(Math.min((canvasRect.width - padding * 2) / width, (canvasRect.height - padding * 2) / height, maximumZoom))
    setViewport({ x: (canvasRect.width - width * zoom) / 2 - left * zoom, y: (canvasRect.height - height * zoom) / 2 - top * zoom, zoom })
  }

  const fitGraph = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    fitElements([...canvas.querySelectorAll<HTMLElement>('[data-graph-node="true"]')])
  }

  const focusNodes = (nodeIds: ReadonlySet<string>) => {
    const canvas = canvasRef.current
    if (!canvas || nodeIds.size === 0) return
    const elements = [...canvas.querySelectorAll<HTMLElement>('[data-graph-node="true"][data-node-id]')]
      .filter((element) => nodeIds.has(element.dataset.nodeId ?? ''))
    if (elements.length > 0) fitElements(elements, 1.05)
  }

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    const target = event.target as HTMLElement
    const interactive = target.closest('button,input,select,textarea,.architecture-node,.qkv-composite,.qkv-expanded-group')
    const canPan = event.button === 1 || (event.button === 0 && (spacePressed.current || !interactive))
    if (!canPan) return
    event.preventDefault()
    event.currentTarget.setPointerCapture?.(event.pointerId)
    panGesture.current = { pointerId: event.pointerId, clientX: event.clientX, clientY: event.clientY, viewport }
    setIsPanning(true)
  }

  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    const gesture = panGesture.current
    if (!gesture || gesture.pointerId !== event.pointerId) return
    setViewport(panViewport(gesture.viewport, event.clientX - gesture.clientX, event.clientY - gesture.clientY))
  }

  const endPan = (event: PointerEvent<HTMLDivElement>) => {
    if (panGesture.current?.pointerId !== event.pointerId) return
    panGesture.current = null
    setIsPanning(false)
    event.currentTarget.releasePointerCapture?.(event.pointerId)
  }

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.code === 'Space') {
      spacePressed.current = true
      event.preventDefault()
    }
  }
  const onKeyUp = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.code === 'Space') spacePressed.current = false
  }

  return {
    viewport,
    isPanning,
    gridStyle: graphGridStyle(viewport),
    worldStyle: { transform: `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.zoom})` },
    onPointerDown,
    onPointerMove,
    onPointerUp: endPan,
    onPointerCancel: endPan,
    onKeyDown,
    onKeyUp,
    zoomIn: () => zoomBy(0.1),
    zoomOut: () => zoomBy(-0.1),
    resetZoom,
    fitGraph,
    focusNodes,
  }
}
