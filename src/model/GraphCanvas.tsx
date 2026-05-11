import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type Dispatch, type DragEvent, type MouseEvent, type PointerEvent, type SetStateAction } from 'react'
import { Blocks, Cable, Minus, MousePointer2, Pencil, Plus, Scan, Sparkles, Trash2, Zap } from 'lucide-react'
import type { AtomicPlayerSnapshot } from '../core/atomic-player'
import { moveGroup, moveNode, type ArchitectureGraph, type ArchitectureNode, type TensorRole } from '../core/ir'
import { modelAtomRegistry } from '../core/model-atoms'
import { customCardInputPorts, customCardOutputPorts } from '../core/custom-card-graph'
import { useElasticCables, type CablePath, type PortDirection } from './useElasticCables'
import { useGraphViewport } from './useGraphViewport'
import { screenToWorld } from './viewport'
import { MODEL_CARD_HEIGHT, MODEL_CARD_WIDTH } from './card-layout'
import { orderedNodeInputPorts } from './port-layout'
import { StudioContextMenu, StudioContextMenuItem } from '../studio/StudioContextMenu'

function describeNode(node: ArchitectureNode): string {
  if (typeof node.attributes?.detail === 'string') return node.attributes.detail
  if (node.attributes?.inFeatures && node.attributes?.outFeatures) return `${node.attributes.inFeatures} → ${node.attributes.outFeatures}`
  if (node.kind === 'sdpa') return 'causal · SDPA'
  return node.role
}


function CableLayer({ paths, draftPath }: { paths: CablePath[]; draftPath?: { path: string; role: TensorRole } }) {
  return <svg aria-hidden="true" className="graph-connections" viewBox="0 0 4000 4000">
    <defs><marker id="edge-arrow" markerHeight="5" markerWidth="5" orient="auto" refX="4" refY="2.5"><path d="M0,0 L5,2.5 L0,5 Z" /></marker></defs>
    {paths.map((cable) => <path className={`graph-edge edge-${cable.role}`} d={cable.path} data-edge-id={cable.id} key={cable.id} markerEnd="url(#edge-arrow)" style={cable.color ? { stroke: cable.color } : undefined} />)}
    {draftPath && <path className={`graph-edge cable-draft edge-${draftPath.role}`} d={draftPath.path} />}
  </svg>
}

type PortHandler = (event: PointerEvent<HTMLButtonElement>, nodeId: string, portId: string, role: TensorRole, direction: PortDirection) => void
type NodeDragHandler = (event: PointerEvent<HTMLButtonElement>, node: ArchitectureNode) => void

function Port({ direction, id, portId, nodeId, role, label, className = '', style, onPointerDown }: { direction: PortDirection; id: string; portId?: string; nodeId: string; role: TensorRole; label: string; className?: string; style?: CSSProperties; onPointerDown: PortHandler }) {
  return <button aria-label={`${nodeId} ${direction} ${label}`} className={`block-port port-${direction === 'input' ? 'top' : 'bottom'} port-${role} ${className}`} data-node-id={nodeId} data-port-direction={direction} data-port-id={id} data-port-key={portId ?? role} data-port-role={role} onPointerDown={(event) => onPointerDown(event, nodeId, portId ?? role, role, direction)} style={style} type="button">{label}</button>
}

function NodePorts({ graph, node, onPointerDown }: { graph: ArchitectureGraph; node: ArchitectureNode; onPointerDown: PortHandler }) {
  if (node.kind === 'input') {
    const role = node.role === 'token-ids' || node.id.toLowerCase().includes('token') ? 'token-ids' : node.role
    const portId = role === 'token-ids' ? 'tokenIds' : role
    const label = ({ 'token-ids': 'IDs', image: 'IMG', video: 'VID', audio: 'AUD', hidden: 'H', labels: 'Y', query: 'Q', key: 'K', value: 'V', attention: 'A', output: 'O', logits: 'L', scalar: 'S', 'routing-logits': 'R', 'expert-indices': 'I', 'routing-weights': 'W' } as Record<TensorRole, string>)[role]
    return <Port direction="output" id={`${node.id}-${portId}-output`} label={label} nodeId={node.id} onPointerDown={onPointerDown} portId={portId} role={role} />
  }
  if (node.kind === 'semantic' && node.atomId) {
    const definition = modelAtomRegistry[node.atomId]
    if (!definition) return null
    const label = (tensor: TensorRole) => ({ 'token-ids': 'IDs', image: 'IMG', video: 'VID', audio: 'AUD', hidden: 'H', query: 'Q', key: 'K', value: 'V', logits: 'L', labels: 'Y', scalar: 'S', 'routing-logits': 'R', 'expert-indices': 'I', 'routing-weights': 'W', attention: 'A', output: 'O' }[tensor])
    return <>
      {orderedNodeInputPorts(graph, node).map((port, index) => <Port direction="input" id={`${node.id}-${port.id}-input`} key={`in-${port.id}`} label={label(port.tensor)} nodeId={node.id} onPointerDown={onPointerDown} portId={port.id} role={port.tensor} style={{ left: `${((index + 1) / (definition.inputs.length + 1)) * 100}%` }} />)}
      {definition.outputs.map((port, index) => <Port direction="output" id={`${node.id}-${port.id}-output`} key={`out-${port.id}`} label={label(port.tensor)} nodeId={node.id} onPointerDown={onPointerDown} portId={port.id} role={port.tensor} style={{ left: `${((index + 1) / (definition.outputs.length + 1)) * 100}%` }} />)}
    </>
  }
  if (node.kind === 'custom-pytorch') {
    if (node.customCardGraph) {
      const inputs = customCardInputPorts(node.customCardGraph)
      const outputs = customCardOutputPorts(node.customCardGraph)
      const label = (role: TensorRole) => ({ 'token-ids': 'IDs', image: 'IMG', video: 'VID', audio: 'AUD', hidden: 'H', query: 'Q', key: 'K', value: 'V', logits: 'L', labels: 'Y', scalar: 'S', 'routing-logits': 'R', 'expert-indices': 'I', 'routing-weights': 'W', attention: 'A', output: 'O' }[role])
      return <>
        {inputs.map((port, index) => <Port direction="input" id={`${node.id}-${port.id}-input`} key={`in-${port.id}`} label={label(port.tensor)} nodeId={node.id} onPointerDown={onPointerDown} portId={port.id} role={port.tensor} style={{ left: `${((index + 1) / (inputs.length + 1)) * 100}%` }} />)}
        {outputs.map((port, index) => <Port direction="output" id={`${node.id}-${port.id}-output`} key={`out-${port.id}`} label={label(port.tensor)} nodeId={node.id} onPointerDown={onPointerDown} portId={port.id} role={port.tensor} style={{ left: `${((index + 1) / (outputs.length + 1)) * 100}%` }} />)}
      </>
    }
    const inputRole = (node.attributes?.inputRole as TensorRole | undefined) ?? 'hidden'
    const label = (role: TensorRole) => ({ 'token-ids': 'IDs', image: 'IMG', video: 'VID', audio: 'AUD', hidden: 'H', query: 'Q', key: 'K', value: 'V', logits: 'L', labels: 'Y', scalar: 'S', 'routing-logits': 'R', 'expert-indices': 'I', 'routing-weights': 'W', attention: 'A', output: 'O' }[role])
    return <>
      <Port direction="input" id={`${node.id}-input-input`} label={label(inputRole)} nodeId={node.id} onPointerDown={onPointerDown} portId="input" role={inputRole} />
      <Port direction="output" id={`${node.id}-output-output`} label={label(node.role)} nodeId={node.id} onPointerDown={onPointerDown} portId="output" role={node.role} />
    </>
  }
  if (node.kind === 'sdpa') return <>
    <Port className="port-third-1" direction="input" id={`${node.id}-query-input`} label="Q" nodeId={node.id} onPointerDown={onPointerDown} role="query" />
    <Port className="port-third-2" direction="input" id={`${node.id}-key-input`} label="K" nodeId={node.id} onPointerDown={onPointerDown} role="key" />
    <Port className="port-third-3" direction="input" id={`${node.id}-value-input`} label="V" nodeId={node.id} onPointerDown={onPointerDown} role="value" />
    <Port direction="output" id={`${node.id}-hidden-output`} label="H" nodeId={node.id} onPointerDown={onPointerDown} role="hidden" />
  </>
  const output = node.role === 'query' ? 'Q' : node.role === 'key' ? 'K' : node.role === 'value' ? 'V' : 'H'
  return <>
    <Port direction="input" id={`${node.id}-hidden-input`} label="H" nodeId={node.id} onPointerDown={onPointerDown} role="hidden" />
    <Port direction="output" id={`${node.id}-${node.role}-output`} label={output} nodeId={node.id} onPointerDown={onPointerDown} role={node.role === 'attention' ? 'hidden' : node.role} />
  </>
}

function ArchitectureNodeCard({ editMode = false, graph, node, selected, highlighted = false, status, grouped = false, dragging = false, onContextMenu, onEdit, onSelect, onPortPointerDown, onDragPointerDown }: { editMode?: boolean; graph: ArchitectureGraph; node: ArchitectureNode; selected: boolean; highlighted?: boolean; status: string; grouped?: boolean; dragging?: boolean; onContextMenu?(event: MouseEvent<HTMLDivElement>, node: ArchitectureNode): void; onEdit?(): void; onSelect(event: MouseEvent<HTMLButtonElement>): void; onPortPointerDown: PortHandler; onDragPointerDown?: NodeDragHandler }) {
  const editability = node.kind === 'custom-pytorch' ? 'CODE' : node.kind === 'input' ? 'LABEL' : 'SETTINGS'
  return <div className={`architecture-node node-${node.role} ${selected ? 'selected' : ''} ${highlighted ? 'architecture-target' : ''} status-${status} ${grouped ? 'grouped-node' : ''} ${dragging ? 'dragging' : ''}`} data-graph-node="true" data-node-id={node.id} data-atom-id={node.atomId} onContextMenu={(event) => onContextMenu?.(event, node)} style={grouped ? { overflow: 'visible' } : { left: node.position.x, top: node.position.y, overflow: 'visible' }}>
    <NodePorts graph={graph} node={node} onPointerDown={onPortPointerDown} />
    <button aria-label={`Select ${node.label}`} className="node-select" onClick={onSelect} onDoubleClick={(event) => { event.preventDefault(); event.stopPropagation(); onEdit?.() }} onPointerDown={(event) => {
      if (editMode || event.detail > 1) {
        event.stopPropagation()
        return
      }
      onDragPointerDown?.(event, node)
    }}>
      <span className="node-type">{node.kind}</span><strong>{node.label}</strong><small>{describeNode(node)}</small>
    </button>
    {editMode && <span aria-label={`Editable card: ${editability.toLowerCase()}`} className="card-editability-badge"><Pencil size={9} />{editability}</span>}
  </div>
}

export function GraphCanvas({ editMode = false, graph, setGraph, selectedNodeId, setSelectedNodeId, highlightedNodeIds, playerSnapshot, onDropAtom, onDropCustom, onDropInput, onDeleteNode, onDeleteNodes, onEditNode, onSelectionChange }: { editMode?: boolean; graph: ArchitectureGraph; setGraph: Dispatch<SetStateAction<ArchitectureGraph>>; selectedNodeId: string; setSelectedNodeId(id: string): void; highlightedNodeIds?: ReadonlySet<string>; playerSnapshot: AtomicPlayerSnapshot; onDropAtom(atomId: string, position: { x: number; y: number }): void; onDropCustom(cardId: string, position: { x: number; y: number }): void; onDropInput(inputRole: TensorRole, position: { x: number; y: number }): void; onDeleteNode?(nodeId: string): void; onDeleteNodes?(nodeIds: string[]): void; onEditNode?(nodeId: string): void; onSelectionChange?(nodeIds: string[]): void }) {
  const qkvGroup = graph.groups?.find((group) => group.kind === 'qkv-projection')
  const qkvNodeIds = new Set(qkvGroup?.nodeIds ?? [])
  const canvasRef = useRef<HTMLDivElement | null>(null)
  const camera = useGraphViewport(canvasRef)
  const focusNodesRef = useRef(camera.focusNodes)
  focusNodesRef.current = camera.focusNodes
  const previousGraphRef = useRef<{ id: string; nodeIds: Set<string>; positions: Map<string, string> } | undefined>(undefined)
  const manualNodePositionsRef = useRef(new Map<string, string>())
  const manualDropPositionRef = useRef<string | undefined>(undefined)
  const nodeDrag = useRef<{ pointerId: number; nodeId: string; offsetX: number; offsetY: number; position: { x: number; y: number } } | null>(null)
  const groupDrag = useRef<{ pointerId: number; groupId: string; offsetX: number; offsetY: number; position: { x: number; y: number } } | null>(null)
  const selectionDrag = useRef<{ pointerId: number; startX: number; startY: number; base: Set<string> } | null>(null)
  const [dragPreview, setDragPreview] = useState<{ nodeId: string; position: { x: number; y: number } } | null>(null)
  const [groupPreview, setGroupPreview] = useState<{ groupId: string; position: { x: number; y: number } } | null>(null)
  const [acceptsLibraryDrop, setAcceptsLibraryDrop] = useState(false)
  const [cardMenu, setCardMenu] = useState<{ nodeId: string; label: string; x: number; y: number; confirmDelete?: boolean }>()
  const [selectedNodeIds, setSelectedNodeIds] = useState<Set<string>>(() => new Set())
  const [selectionBox, setSelectionBox] = useState<{ left: number; top: number; width: number; height: number }>()
  const [confirmSelectionDelete, setConfirmSelectionDelete] = useState(false)
  const cables = useElasticCables(graph, setGraph, canvasRef, camera.viewport, `${selectedNodeId}:${dragPreview?.nodeId ?? ''}:${dragPreview?.position.x ?? ''}:${dragPreview?.position.y ?? ''}:${groupPreview?.groupId ?? ''}:${groupPreview?.position.x ?? ''}:${groupPreview?.position.y ?? ''}`)

  useLayoutEffect(() => {
    const nodeIds = new Set(graph.nodes.map((node) => node.id))
    const positions = new Map(graph.nodes.map((node) => [node.id, `${node.position.x}:${node.position.y}`]))
    const previous = previousGraphRef.current
    if (!previous || previous.id !== graph.id) {
      focusNodesRef.current(nodeIds)
    } else {
      const added = new Set([...nodeIds].filter((nodeId) => !previous.nodeIds.has(nodeId)))
      if (added.size > 0) {
        const manualDrop = added.size === 1 && positions.get([...added][0]) === manualDropPositionRef.current
        if (!manualDrop) focusNodesRef.current(added)
        manualDropPositionRef.current = undefined
      }
      else {
        const movedNodeIds = graph.nodes.filter((node) => previous.positions.get(node.id) !== positions.get(node.id)).map((node) => node.id)
        const manualMoveOnly = movedNodeIds.length > 0 && movedNodeIds.every((nodeId) => manualNodePositionsRef.current.get(nodeId) === positions.get(nodeId))
        if (movedNodeIds.length > 0 && !manualMoveOnly) focusNodesRef.current(nodeIds)
        for (const nodeId of movedNodeIds) manualNodePositionsRef.current.delete(nodeId)
      }
    }
    previousGraphRef.current = { id: graph.id, nodeIds, positions }
  }, [graph.edges, graph.id, graph.nodes])

  useEffect(() => {
    if (!cardMenu) return
    const closeMenu = (event: globalThis.PointerEvent) => {
      if (!(event.target as HTMLElement | null)?.closest('.card-context-menu')) setCardMenu(undefined)
    }
    document.addEventListener('pointerdown', closeMenu)
    return () => document.removeEventListener('pointerdown', closeMenu)
  }, [cardMenu])

  useEffect(() => {
    if (!editMode) {
      setSelectedNodeIds(new Set())
      setSelectionBox(undefined)
      setConfirmSelectionDelete(false)
      return
    }
    const available = new Set(graph.nodes.map((node) => node.id))
    setSelectedNodeIds((current) => new Set([...current].filter((nodeId) => available.has(nodeId))))
  }, [editMode, graph.nodes])

  useEffect(() => {
    onSelectionChange?.([...selectedNodeIds])
  }, [onSelectionChange, selectedNodeIds])

  const selectNode = (event: MouseEvent<HTMLButtonElement>, nodeId: string) => {
    event.stopPropagation()
    if (!editMode) {
      setSelectedNodeId(nodeId)
      return
    }
    const additive = event.shiftKey || event.metaKey || event.ctrlKey
    if (!additive) {
      setSelectedNodeIds(new Set())
      setSelectedNodeId(nodeId)
      setConfirmSelectionDelete(false)
      onEditNode?.(nodeId)
      return
    }
    setSelectedNodeIds((current) => {
      const next = new Set(current)
      if (next.has(nodeId)) next.delete(nodeId)
      else next.add(nodeId)
      return next
    })
    setSelectedNodeId(nodeId)
    setConfirmSelectionDelete(false)
  }

  const selectQkvGroup = (event: MouseEvent<HTMLButtonElement>) => {
    if (!editMode || !qkvGroup) {
      setSelectedNodeId(qkvGroup?.id ?? '')
      return
    }
    event.stopPropagation()
    const additive = event.shiftKey || event.metaKey || event.ctrlKey
    setSelectedNodeIds((current) => {
      const next = additive ? new Set(current) : new Set<string>()
      const allSelected = qkvGroup.nodeIds.every((nodeId) => next.has(nodeId))
      for (const nodeId of qkvGroup.nodeIds) {
        if (additive && allSelected) next.delete(nodeId)
        else next.add(nodeId)
      }
      return next
    })
    setSelectedNodeId(qkvGroup.nodeIds[0] ?? qkvGroup.id)
    setConfirmSelectionDelete(false)
  }

  const nodesInsideSelection = (box: { left: number; top: number; width: number; height: number }) => {
    const start = screenToWorld({ x: box.left, y: box.top }, camera.viewport)
    const end = screenToWorld({ x: box.left + box.width, y: box.top + box.height }, camera.viewport)
    const left = Math.min(start.x, end.x)
    const right = Math.max(start.x, end.x)
    const top = Math.min(start.y, end.y)
    const bottom = Math.max(start.y, end.y)
    const intersects = (x: number, y: number, width: number, height: number) => x < right && x + width > left && y < bottom && y + height > top
    const matched = new Set<string>()
    const shift = qkvGroup ? (qkvGroup.expanded ? 140 : 55) : 0
    for (const node of graph.nodes) {
      if (qkvNodeIds.has(node.id)) continue
      const y = node.position.y >= 300 ? node.position.y + shift : node.position.y
      if (intersects(node.position.x, y, MODEL_CARD_WIDTH, MODEL_CARD_HEIGHT)) matched.add(node.id)
    }
    if (qkvGroup && intersects(qkvGroup.position.x, qkvGroup.position.y, qkvGroup.expanded ? 390 : 340, qkvGroup.expanded ? 130 : 92)) {
      for (const nodeId of qkvGroup.nodeIds) matched.add(nodeId)
    }
    return matched
  }

  const beginCanvasSelection = (event: PointerEvent<HTMLDivElement>) => {
    const target = event.target as HTMLElement
    const interactive = target.closest('button,input,select,textarea,[data-graph-node],.card-context-menu,.graph-selection-toolbar,.graph-viewport-controls')
    const additive = event.shiftKey || event.metaKey || event.ctrlKey
    if (event.button === 0 && !interactive && !additive) {
      setSelectedNodeId('')
      setSelectedNodeIds(new Set())
      setConfirmSelectionDelete(false)
    }
    if (!editMode || event.button !== 0 || interactive) {
      camera.onPointerDown(event)
      return
    }
    event.preventDefault()
    const bounds = event.currentTarget.getBoundingClientRect()
    const x = event.clientX - bounds.left
    const y = event.clientY - bounds.top
    selectionDrag.current = { pointerId: event.pointerId, startX: x, startY: y, base: additive ? new Set(selectedNodeIds) : new Set() }
    if (!additive) setSelectedNodeIds(new Set())
    setConfirmSelectionDelete(false)
    setSelectionBox({ left: x, top: y, width: 0, height: 0 })
    event.currentTarget.setPointerCapture?.(event.pointerId)
  }

  const beginNodeDrag: NodeDragHandler = (event, node) => {
    if (event.button !== 0) return
    event.preventDefault()
    event.stopPropagation()
    const canvas = canvasRef.current
    if (!canvas) return
    const bounds = canvas.getBoundingClientRect()
    const pointer = screenToWorld({ x: event.clientX - bounds.left, y: event.clientY - bounds.top }, camera.viewport)
    const position = { ...node.position }
    nodeDrag.current = { pointerId: event.pointerId, nodeId: node.id, offsetX: pointer.x - position.x, offsetY: pointer.y - position.y, position }
    setDragPreview({ nodeId: node.id, position })
    setSelectedNodeId(node.id)
    canvas.setPointerCapture?.(event.pointerId)
  }

  const openCardMenu = (event: MouseEvent<HTMLDivElement>, node: ArchitectureNode) => {
    event.preventDefault()
    event.stopPropagation()
    setSelectedNodeId(node.id)
    setCardMenu({
      nodeId: node.id,
      label: node.label,
      x: event.clientX,
      y: event.clientY,
    })
  }

  const beginGroupDrag = (event: PointerEvent<HTMLElement>, groupId: string, position: { x: number; y: number }) => {
    if (event.button !== 0 || (event.currentTarget.tagName !== 'BUTTON' && event.currentTarget !== event.target && (event.target as HTMLElement).closest('button'))) return
    const canvas = canvasRef.current
    if (!canvas) return
    event.preventDefault()
    event.stopPropagation()
    const bounds = canvas.getBoundingClientRect()
    const pointer = screenToWorld({ x: event.clientX - bounds.left, y: event.clientY - bounds.top }, camera.viewport)
    const initial = { ...position }
    groupDrag.current = { pointerId: event.pointerId, groupId, offsetX: pointer.x - initial.x, offsetY: pointer.y - initial.y, position: initial }
    setGroupPreview({ groupId, position: initial })
    setSelectedNodeId(groupId)
    canvas.setPointerCapture?.(event.pointerId)
  }

  const movePointer = (event: PointerEvent<HTMLDivElement>) => {
    const selecting = selectionDrag.current
    if (selecting?.pointerId === event.pointerId) {
      event.preventDefault()
      const bounds = event.currentTarget.getBoundingClientRect()
      const x = event.clientX - bounds.left
      const y = event.clientY - bounds.top
      const box = { left: Math.min(selecting.startX, x), top: Math.min(selecting.startY, y), width: Math.abs(x - selecting.startX), height: Math.abs(y - selecting.startY) }
      setSelectionBox(box)
      const next = new Set(selecting.base)
      for (const nodeId of nodesInsideSelection(box)) next.add(nodeId)
      setSelectedNodeIds(next)
      return
    }
    const drag = nodeDrag.current
    const movingGroup = groupDrag.current
    if ((!drag || drag.pointerId !== event.pointerId) && (!movingGroup || movingGroup.pointerId !== event.pointerId)) return camera.onPointerMove(event)
    event.preventDefault()
    const bounds = event.currentTarget.getBoundingClientRect()
    const pointer = screenToWorld({ x: event.clientX - bounds.left, y: event.clientY - bounds.top }, camera.viewport)
    if (movingGroup && movingGroup.pointerId === event.pointerId) {
      movingGroup.position = { x: pointer.x - movingGroup.offsetX, y: pointer.y - movingGroup.offsetY }
      setGroupPreview({ groupId: movingGroup.groupId, position: movingGroup.position })
      return
    }
    if (!drag) return
    drag.position = { x: pointer.x - drag.offsetX, y: pointer.y - drag.offsetY }
    setDragPreview({ nodeId: drag.nodeId, position: drag.position })
  }

  const endPointer = (event: PointerEvent<HTMLDivElement>) => {
    if (selectionDrag.current?.pointerId === event.pointerId) {
      selectionDrag.current = null
      setSelectionBox(undefined)
      event.currentTarget.releasePointerCapture?.(event.pointerId)
      return
    }
    const drag = nodeDrag.current
    const movingGroup = groupDrag.current
    if (movingGroup?.pointerId === event.pointerId) {
      groupDrag.current = null
      setGraph((current) => moveGroup(current, movingGroup.groupId, movingGroup.position))
      setGroupPreview(null)
      event.currentTarget.releasePointerCapture?.(event.pointerId)
      return
    }
    if (!drag || drag.pointerId !== event.pointerId) return camera.onPointerUp(event)
    nodeDrag.current = null
    manualNodePositionsRef.current.set(drag.nodeId, `${drag.position.x}:${drag.position.y}`)
    setGraph((current) => moveNode(current, drag.nodeId, drag.position))
    setDragPreview(null)
    event.currentTarget.releasePointerCapture?.(event.pointerId)
  }

  const cancelPointer = (event: PointerEvent<HTMLDivElement>) => {
    if (selectionDrag.current?.pointerId === event.pointerId) {
      selectionDrag.current = null
      setSelectionBox(undefined)
      event.currentTarget.releasePointerCapture?.(event.pointerId)
      return
    }
    if (nodeDrag.current?.pointerId === event.pointerId) {
      nodeDrag.current = null
      setDragPreview(null)
      event.currentTarget.releasePointerCapture?.(event.pointerId)
      return
    }
    if (groupDrag.current?.pointerId === event.pointerId) {
      groupDrag.current = null
      setGroupPreview(null)
      event.currentTarget.releasePointerCapture?.(event.pointerId)
      return
    }
    camera.onPointerCancel(event)
  }

  const setQkvExpanded = (expanded: boolean) => {
    setGraph((current) => ({ ...current, groups: current.groups?.map((group) => group.kind === 'qkv-projection' ? { ...group, expanded } : group) }))
    setSelectedNodeId(expanded ? 'q-proj' : 'qkv-projections')
  }


  const status = (nodeId: string) => playerSnapshot.results.find((result) => result.atomId === nodeId)?.status ?? 'pending'

  const acceptsAtomDrag = (event: DragEvent<HTMLDivElement>) => event.dataTransfer.types.includes('application/x-labo-model-atom') || event.dataTransfer.types.includes('application/x-labo-graph-input') || event.dataTransfer.types.includes('application/x-labo-custom-card')

  const dragLibraryAtomOver = (event: DragEvent<HTMLDivElement>) => {
    if (!acceptsAtomDrag(event)) return
    event.preventDefault()
    event.dataTransfer.dropEffect = 'copy'
    setAcceptsLibraryDrop(true)
  }

  const dropLibraryAtom = (event: DragEvent<HTMLDivElement>) => {
    const atomId = event.dataTransfer.getData('application/x-labo-model-atom')
    const inputRole = event.dataTransfer.getData('application/x-labo-graph-input') as TensorRole
    const customCardId = event.dataTransfer.getData('application/x-labo-custom-card')
    setAcceptsLibraryDrop(false)
    if (!atomId && !inputRole && !customCardId) return
    event.preventDefault()
    const bounds = event.currentTarget.getBoundingClientRect()
    const pointer = screenToWorld({ x: event.clientX - bounds.left, y: event.clientY - bounds.top }, camera.viewport)
    const position = { x: pointer.x - MODEL_CARD_WIDTH / 2, y: pointer.y - MODEL_CARD_HEIGHT / 2 }
    manualDropPositionRef.current = `${position.x}:${position.y}`
    if (inputRole) onDropInput(inputRole, position)
    else if (customCardId) onDropCustom(customCardId, position)
    else onDropAtom(atomId, position)
  }

  return <div className="canvas-panel">
    <div className="panel-tab"><Blocks size={13} /> Architecture.graph <span className="cable-help">{editMode ? <><MousePointer2 size={11} />Drag empty space to select · Shift adds</> : <><Cable size={11} />{cables.message}</>}</span></div>
    <div
      aria-label="Architecture graph canvas"
      className={`architecture-canvas ${editMode ? 'edit-mode' : ''} ${highlightedNodeIds?.size ? 'has-architecture-target' : ''} ${camera.isPanning ? 'is-panning' : ''} ${acceptsLibraryDrop ? 'accepts-library-drop' : ''}`}

      onDragLeave={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setAcceptsLibraryDrop(false)
      }}
      onDragOver={dragLibraryAtomOver}
      onDrop={dropLibraryAtom}
      onKeyDown={(event) => {
        if (event.key === 'Escape') {
          setCardMenu(undefined)
          setSelectedNodeId('')
          setSelectedNodeIds(new Set())
          setConfirmSelectionDelete(false)
        }
        camera.onKeyDown(event)
      }}
      onKeyUp={camera.onKeyUp}
      onPointerCancel={cancelPointer}
      onPointerDown={(event) => {
        if (!(event.target as HTMLElement).closest('.card-context-menu')) setCardMenu(undefined)
        beginCanvasSelection(event)
      }}
      onPointerMove={movePointer}
      onPointerUp={endPointer}
      ref={canvasRef}
      tabIndex={0}
    >
      <div className="canvas-grid" style={camera.gridStyle} />
      {selectionBox && <div aria-hidden="true" className="graph-selection-box" style={selectionBox} />}
      {editMode && selectedNodeIds.size > 0 && <div aria-label="Selected graph cards" className="graph-selection-toolbar">
        <strong>{selectedNodeIds.size} card{selectedNodeIds.size === 1 ? '' : 's'} selected</strong>
        <button onClick={() => { setSelectedNodeId(''); setSelectedNodeIds(new Set()); setConfirmSelectionDelete(false) }} type="button">Clear</button>
        <button className={confirmSelectionDelete ? 'confirm-delete' : ''} onClick={() => {
          if (!confirmSelectionDelete) {
            setConfirmSelectionDelete(true)
            return
          }
          onDeleteNodes?.([...selectedNodeIds])
          setSelectedNodeIds(new Set())
          setConfirmSelectionDelete(false)
        }} type="button"><Trash2 size={12} />{confirmSelectionDelete ? `Confirm delete ${selectedNodeIds.size}` : 'Delete selection'}</button>
      </div>}
      {graph.nodes.length === 0 && <div className="graph-empty-state" aria-hidden="true">
        <span className="graph-empty-mark"><Sparkles size={19} /></span>
        <div><strong>Start with an atomic idea</strong><p>Drag a card from the library, or ask LABO to compose the first architecture.</p></div>
        <small><MousePointer2 size={12} /> The canvas stays infinite as your graph grows.</small>
      </div>}
      <div className="graph-world" data-testid="graph-world" style={camera.worldStyle}>
      <CableLayer draftPath={cables.draftPath} paths={cables.paths} />
      {qkvGroup && !qkvGroup.expanded && <div className={`qkv-composite ${selectedNodeId === qkvGroup.id ? 'selected' : ''} ${groupPreview?.groupId === qkvGroup.id ? 'dragging' : ''}`} data-graph-node="true" style={{ left: groupPreview?.groupId === qkvGroup.id ? groupPreview.position.x : qkvGroup.position.x, top: groupPreview?.groupId === qkvGroup.id ? groupPreview.position.y : qkvGroup.position.y }}>
        <Port direction="input" id="qkv-hidden-input" label="H" nodeId="qkv-projections" onPointerDown={cables.beginCable} role="hidden" />
        <Port className="port-third-1" direction="output" id="qkv-query-output" label="Q" nodeId="q-proj" onPointerDown={cables.beginCable} role="query" />
        <Port className="port-third-2" direction="output" id="qkv-key-output" label="K" nodeId="k-proj" onPointerDown={cables.beginCable} role="key" />
        <Port className="port-third-3" direction="output" id="qkv-value-output" label="V" nodeId="v-proj" onPointerDown={cables.beginCable} role="value" />
        <button aria-label="Select QKV projection" className="qkv-select" onClick={selectQkvGroup} onPointerDown={(event) => { if (editMode) event.stopPropagation(); else beginGroupDrag(event, qkvGroup.id, qkvGroup.position) }}><span className="node-type">COMPOSITE · GQA</span><strong>QKV projection</strong><small>Q{graph.config.queryHeads} · KV{graph.config.keyValueHeads} · head {graph.config.headDim}</small></button>
        {editMode && <span className="card-editability-badge card-editability-expand">EXPAND TO EDIT</span>}

        <button aria-label="Expand QKV projections" className="group-transform-button" onClick={() => setQkvExpanded(true)}>Décomposer en Q / K / V</button>
      </div>}
      {qkvGroup?.expanded && <section className={`qkv-expanded-group ${groupPreview?.groupId === qkvGroup.id ? 'dragging' : ''}`} aria-label="QKV projection group" data-graph-node="true" style={{ left: groupPreview?.groupId === qkvGroup.id ? groupPreview.position.x : qkvGroup.position.x, top: groupPreview?.groupId === qkvGroup.id ? groupPreview.position.y : qkvGroup.position.y }}><div className="qkv-group-header" onPointerDown={(event) => beginGroupDrag(event, qkvGroup.id, qkvGroup.position)}><div><span className="node-type">COMPOSITE · EXPANDED</span><strong>QKV projections</strong></div><button aria-label="Collapse QKV projections" onClick={() => setQkvExpanded(false)}>Regrouper en QKV</button></div><div className="qkv-child-grid">{graph.nodes.filter((node) => qkvNodeIds.has(node.id)).map((node) => <ArchitectureNodeCard editMode={editMode} graph={graph} grouped highlighted={highlightedNodeIds?.has(node.id)} key={node.id} node={node} onContextMenu={openCardMenu} onEdit={() => onEditNode?.(node.id)} onPortPointerDown={cables.beginCable} onSelect={(event) => selectNode(event, node.id)} selected={selectedNodeIds.has(node.id) || selectedNodeId === node.id} status={status(node.id)} />)}</div></section>}
      {graph.nodes.filter((node) => !qkvNodeIds.has(node.id)).map((node) => {
        const shift = qkvGroup ? (qkvGroup.expanded ? 140 : 55) : 0
        const previewPosition = dragPreview?.nodeId === node.id ? dragPreview.position : node.position
        const displayed = previewPosition.y >= 300 ? { ...node, position: { ...previewPosition, y: previewPosition.y + shift } } : { ...node, position: previewPosition }
        return <ArchitectureNodeCard dragging={dragPreview?.nodeId === node.id} editMode={editMode} graph={graph} highlighted={highlightedNodeIds?.has(node.id)} key={node.id} node={displayed} onContextMenu={openCardMenu} onDragPointerDown={beginNodeDrag} onEdit={() => onEditNode?.(node.id)} onPortPointerDown={cables.beginCable} onSelect={(event) => selectNode(event, node.id)} selected={selectedNodeIds.has(node.id) || selectedNodeId === node.id} status={status(node.id)} />
      })}
      </div>
      {cardMenu && <StudioContextMenu position={cardMenu}>
        <div><span>CARD</span><strong>{cardMenu.label}</strong></div>
        <StudioContextMenuItem onClick={() => { onEditNode?.(cardMenu.nodeId); setCardMenu(undefined) }}><Pencil size={13} />Edit card</StudioContextMenuItem>
        <StudioContextMenuItem className={cardMenu.confirmDelete ? 'confirm-delete' : ''} onClick={() => {
          if (!cardMenu.confirmDelete) return setCardMenu((current) => current ? { ...current, confirmDelete: true } : current)
          onDeleteNode?.(cardMenu.nodeId)
          setCardMenu(undefined)
        }}><Trash2 size={13} />{cardMenu.confirmDelete ? 'Confirm delete' : 'Delete card'}</StudioContextMenuItem>
      </StudioContextMenu>}
      <div className="graph-viewport-controls" aria-label="Graph viewport controls">
        <button aria-label="Zoom out" onClick={camera.zoomOut}><Minus size={13} /></button>
        <button aria-label="Reset zoom" className="zoom-value" onClick={camera.resetZoom}>{Math.round(camera.viewport.zoom * 100)}%</button>
        <button aria-label="Zoom in" onClick={camera.zoomIn}><Plus size={13} /></button>
        <button aria-label="Fit graph" onClick={camera.fitGraph} title="Center and fit all blocks"><Scan size={14} /></button>
      </div>
      <div className="canvas-badge"><Zap size={12} /> {graph.architecture} · {graph.contracts.causal ? 'causal' : 'non-causal'}</div>
    </div>
  </div>
}
