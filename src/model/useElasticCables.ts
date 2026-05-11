import { useCallback, useEffect, useLayoutEffect, useMemo, useState, type Dispatch, type PointerEvent as ReactPointerEvent, type RefObject, type SetStateAction } from 'react'
import { connectCable, disconnectCable, inferredEdgeTargetPort } from '../core/cables'
import type { ArchitectureEdge, ArchitectureGraph, TensorRole } from '../core/ir'
import { modelAtomRegistry } from '../core/model-atoms'
import { parallelEdgeColors } from '../core/parallel-edge-colors'
import { screenToWorld, type GraphViewport } from './viewport'

export type PortDirection = 'input' | 'output'
export type CablePath = { id: string; path: string; role: TensorRole; color?: string }

type CableDraft = {
  sourceId: string
  sourcePortId: string
  sourceRole: TensorRole
  x: number
  y: number
  detachedEdgeIds: string[]
}

function elasticPath(sourceX: number, sourceY: number, targetX: number, targetY: number): string {
  const lead = 18
  const bend = Math.max(34, Math.abs(targetY - sourceY) * 0.42)
  return `M${sourceX} ${sourceY} L${sourceX} ${sourceY + lead} C${sourceX} ${sourceY + bend} ${targetX} ${targetY - bend} ${targetX} ${targetY - lead} L${targetX} ${targetY}`
}

function sourceRole(graph: ArchitectureGraph, edge: ArchitectureEdge): TensorRole {
  const node = graph.nodes.find((candidate) => candidate.id === edge.source)
  if (node?.kind === 'semantic' && node.atomId) {
    const port = modelAtomRegistry[node.atomId]?.outputs.find((candidate) => candidate.id === edge.sourcePort)
    if (port) return port.tensor
  }
  if (node?.kind === 'input' && node.id.toLowerCase().includes('token')) return 'token-ids'
  if (edge.sourcePort && ['hidden', 'query', 'key', 'value', 'attention', 'output'].includes(edge.sourcePort)) return edge.sourcePort as TensorRole
  return node?.role === 'attention' ? 'hidden' : (node?.role ?? 'hidden')
}

export function useElasticCables(graph: ArchitectureGraph, setGraph: Dispatch<SetStateAction<ArchitectureGraph>>, canvasRef: RefObject<HTMLDivElement | null>, viewport: GraphViewport, layoutKey: string) {
  const [draft, setDraft] = useState<CableDraft | null>(null)
  const [paths, setPaths] = useState<CablePath[]>([])
  const [draftPath, setDraftPath] = useState<{ path: string; role: TensorRole }>()
  const [message, setMessage] = useState('Drag a plug to rewire · outputs support multi-plug')
  const qkvGroup = useMemo(() => graph.groups?.find((group) => group.kind === 'qkv-projection'), [graph.groups])
  const qkvIds = useMemo(() => new Set(qkvGroup?.nodeIds ?? []), [qkvGroup])
  const branchColors = useMemo(() => parallelEdgeColors(graph), [graph])

  const portIdForSource = useCallback((edge: ArchitectureEdge) => {
    const role = sourceRole(graph, edge)
    if (!qkvGroup?.expanded && qkvIds.has(edge.source)) return `qkv-${role}-output`
    return `${edge.source}-${edge.sourcePort ?? role}-output`
  }, [graph, qkvGroup?.expanded, qkvIds])

  const portIdForTarget = useCallback((edge: ArchitectureEdge) => {
    if (!qkvGroup?.expanded && qkvIds.has(edge.target)) return 'qkv-hidden-input'
    return `${edge.target}-${edge.targetPort ?? inferredEdgeTargetPort(graph, edge)}-input`
  }, [graph, qkvGroup?.expanded, qkvIds])

  const center = useCallback((element: Element, bounds: DOMRect) => {
    const rect = element.getBoundingClientRect()
    return screenToWorld({ x: rect.left + rect.width / 2 - bounds.left, y: rect.top + rect.height / 2 - bounds.top }, viewport)
  }, [viewport])

  useLayoutEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const update = () => {
      const bounds = canvas.getBoundingClientRect()
      const visibleConnections = new Set<string>()
      const next = graph.edges.flatMap((edge) => {
        const sourcePortId = portIdForSource(edge)
        const targetPortId = portIdForTarget(edge)
        const visibleConnection = `${sourcePortId}->${targetPortId}`
        if (visibleConnections.has(visibleConnection)) return []
        visibleConnections.add(visibleConnection)
        const source = canvas.querySelector(`[data-port-id="${sourcePortId}"]`)
        const target = canvas.querySelector(`[data-port-id="${targetPortId}"]`)
        if (!source || !target) return []
        const start = center(source, bounds)
        const end = center(target, bounds)
        const role = sourceRole(graph, edge)
        const color = ['query', 'key', 'value'].includes(role) ? undefined : branchColors.get(edge.id)
        return [{ id: edge.id, role, color, path: elasticPath(start.x, start.y, end.x, end.y) }]
      })
      setPaths(next)
      if (draft) {
        const source = canvas.querySelector(`[data-port-id="${draft.sourceId}-${draft.sourcePortId}-output"], [data-port-id="qkv-${draft.sourceRole}-output"]`)
        if (source) {
          const start = center(source, bounds)
          setDraftPath({ role: draft.sourceRole, path: elasticPath(start.x, start.y, draft.x, draft.y) })
        }
      } else setDraftPath(undefined)
    }
    update()
    const resizeObserver = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(update)
    canvas.querySelectorAll('[data-graph-node="true"], [data-port-id]').forEach((element) => resizeObserver?.observe(element))
    window.addEventListener('resize', update)
    return () => {
      resizeObserver?.disconnect()
      window.removeEventListener('resize', update)
    }
  }, [branchColors, canvasRef, center, draft, graph, layoutKey, portIdForSource, portIdForTarget])

  const beginCable = (event: ReactPointerEvent<HTMLButtonElement>, nodeId: string, portId: string, role: TensorRole, direction: PortDirection) => {
    event.preventDefault()
    event.stopPropagation()
    const canvas = canvasRef.current
    if (!canvas) return
    const bounds = canvas.getBoundingClientRect()
    const pointer = screenToWorld({ x: event.clientX - bounds.left, y: event.clientY - bounds.top }, viewport)
    if (direction === 'output') {
      setDraft({ sourceId: nodeId, sourcePortId: portId, sourceRole: role, ...pointer, detachedEdgeIds: [] })
      return
    }

    const targetIds = nodeId === 'qkv-projections' ? qkvGroup?.nodeIds ?? [] : [nodeId]
    const incoming = graph.edges.filter((edge) => targetIds.includes(edge.target) && (edge.targetPort ? edge.targetPort === portId : inferredEdgeTargetPort(graph, edge) === role))
    const first = incoming[0]
    if (!first) return
    setDraft({ sourceId: first.source, sourcePortId: first.sourcePort ?? sourceRole(graph, first), sourceRole: sourceRole(graph, first), ...pointer, detachedEdgeIds: incoming.map((edge) => edge.id) })
  }

  useEffect(() => {
    if (!draft) return
    const canvas = canvasRef.current
    const move = (event: PointerEvent) => {
      const bounds = canvas?.getBoundingClientRect()
      if (!bounds) return
      const pointer = screenToWorld({ x: event.clientX - bounds.left, y: event.clientY - bounds.top }, viewport)
      setDraft((current) => current ? { ...current, ...pointer } : null)
    }
    const finish = (event: PointerEvent) => {
      const target = document.elementFromPoint(event.clientX, event.clientY)?.closest<HTMLElement>('[data-port-direction="input"]')
      if (!target) {
        if (draft.detachedEdgeIds.length > 0) {
          setGraph((current) => draft.detachedEdgeIds.reduce((next, edgeId) => disconnectCable(next, edgeId), current))
          setMessage('Cable disconnected')
        }
        setDraft(null)
        return
      }
      const targetId = target.dataset.nodeId!
      const targetPortId = target.dataset.portKey!
      const targetRole = target.dataset.portRole as TensorRole
      if (targetId === 'qkv-projections') {
        if (draft.sourceRole !== 'hidden' || !qkvGroup) {
          setMessage(`${draft.sourceRole} cannot plug into hidden`)
        } else {
          setGraph((current) => qkvGroup.nodeIds.reduce((next, childId) => connectCable(next, { sourceId: draft.sourceId, sourcePort: 'hidden', sourcePortId: draft.sourcePortId, targetId: childId, targetPort: 'hidden' }).graph, current))
          setMessage('Hidden output multi-plugged into Q, K and V')
        }
      } else {
        setGraph((current) => {
          const outcome = connectCable(current, { sourceId: draft.sourceId, sourcePort: draft.sourceRole, sourcePortId: draft.sourcePortId, targetId, targetPort: targetRole, targetPortId })
          setMessage(outcome.message)
          return outcome.graph
        })
      }
      setDraft(null)
    }
    const cancel = (event: KeyboardEvent) => { if (event.key === 'Escape') setDraft(null) }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', finish, { once: true })
    window.addEventListener('keydown', cancel)
    return () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', finish)
      window.removeEventListener('keydown', cancel)
    }
  }, [canvasRef, draft, qkvGroup, setGraph, viewport])

  return { paths, draftPath, beginCable, message }
}
