import { removeNode, updateNodeAttributes, type ArchitectureGraph } from './ir'

export interface GraphEditorState {
  past: ArchitectureGraph[]
  present: ArchitectureGraph
  future: ArchitectureGraph[]
  version: number
}

export type GraphAction =
  | { type: 'delete-node'; nodeId: string }
  | { type: 'connect-ports'; edgeId: string; sourceId: string; sourcePort: string; targetId: string; targetPort: string }
  | { type: 'update-attributes'; nodeId: string; attributes: Record<string, number | string | boolean> }
  | { type: 'undo' }
  | { type: 'redo' }

export interface GraphActionEvent {
  action: GraphAction['type']
  code: string
  nodeId?: string
  portId?: string
  settingId?: string
  message: string
}

export interface GraphActionOutcome {
  ok: boolean
  state: GraphEditorState
  event: GraphActionEvent
}

function cloneGraph(graph: ArchitectureGraph): ArchitectureGraph {
  return structuredClone(graph)
}

function hasPath(graph: ArchitectureGraph, startId: string, destinationId: string): boolean {
  const visited = new Set<string>()
  const pending = [startId]
  while (pending.length > 0) {
    const current = pending.pop()!
    if (current === destinationId) return true
    if (visited.has(current)) continue
    visited.add(current)
    for (const edge of graph.edges) {
      if (edge.source === current) pending.push(edge.target)
    }
  }
  return false
}

export function createGraphEditorState(graph: ArchitectureGraph): GraphEditorState {
  return { past: [], present: cloneGraph(graph), future: [], version: 0 }
}

export function dispatchGraphAction(state: GraphEditorState, action: GraphAction): GraphActionOutcome {
  if (action.type === 'delete-node') {
    const next = removeNode(state.present, action.nodeId)
    return {
      ok: true,
      state: {
        past: [...state.past, cloneGraph(state.present)],
        present: next,
        future: [],
        version: state.version + 1,
      },
      event: {
        action: action.type,
        code: 'NODE_DELETED',
        nodeId: action.nodeId,
        message: `Deleted atom ${action.nodeId}`,
      },
    }
  }

  if (action.type === 'connect-ports') {
    const source = state.present.nodes.find((node) => node.id === action.sourceId)
    const target = state.present.nodes.find((node) => node.id === action.targetId)
    if (!source || !target) {
      return {
        ok: false,
        state,
        event: {
          action: action.type,
          code: 'UNKNOWN_PORT_NODE',
          nodeId: !source ? action.sourceId : action.targetId,
          message: 'Cannot connect a port on an unknown atom',
        },
      }
    }
    const expectedRole = target.kind === 'sdpa' && ['query', 'key', 'value'].includes(action.targetPort)
      ? action.targetPort
      : undefined
    if (expectedRole && source.role !== expectedRole) {
      return {
        ok: false,
        state,
        event: {
          action: action.type,
          code: 'PORT_TYPE_MISMATCH',
          nodeId: target.id,
          portId: action.targetPort,
          message: `${source.role} cannot connect to ${target.id}.${action.targetPort}`,
        },
      }
    }
    if (hasPath(state.present, target.id, source.id)) {
      return {
        ok: false,
        state,
        event: {
          action: action.type,
          code: 'GRAPH_CYCLE',
          nodeId: target.id,
          portId: action.targetPort,
          message: `Connecting ${source.id} to ${target.id} would create a cycle`,
        },
      }
    }
    const present = {
      ...state.present,
      edges: [...state.present.edges, {
        id: action.edgeId,
        source: source.id,
        sourcePort: action.sourcePort,
        target: target.id,
        targetPort: action.targetPort,
        label: action.targetPort,
      }],
    }
    return {
      ok: true,
      state: {
        past: [...state.past, cloneGraph(state.present)],
        present,
        future: [],
        version: state.version + 1,
      },
      event: { action: action.type, code: 'PORTS_CONNECTED', nodeId: target.id, portId: action.targetPort, message: 'Connected typed ports' },
    }
  }

  if (action.type === 'update-attributes') {
    const node = state.present.nodes.find((candidate) => candidate.id === action.nodeId)
    if (!node) {
      return { ok: false, state, event: { action: action.type, code: 'UNKNOWN_NODE', nodeId: action.nodeId, message: 'Cannot edit an unknown atom' } }
    }
    const positiveSetting = Object.entries(action.attributes).find(([key, value]) =>
      typeof value === 'number'
      && (!Number.isFinite(value) || (/(features|size|heads|dim)$/i.test(key) && value <= 0)),
    )
    if (positiveSetting) {
      return {
        ok: false,
        state,
        event: {
          action: action.type,
          code: 'INVALID_SETTING',
          nodeId: action.nodeId,
          settingId: positiveSetting[0],
          message: `${positiveSetting[0]} must be a positive finite number`,
        },
      }
    }
    const present = updateNodeAttributes(state.present, action.nodeId, action.attributes)
    return {
      ok: true,
      state: {
        past: [...state.past, cloneGraph(state.present)],
        present,
        future: [],
        version: state.version + 1,
      },
      event: { action: action.type, code: 'ATTRIBUTES_UPDATED', nodeId: action.nodeId, message: `Updated atom ${action.nodeId}` },
    }
  }

  if (action.type === 'undo') {
    const previous = state.past.at(-1)
    if (!previous) {
      return { ok: false, state, event: { action: 'undo', code: 'NOTHING_TO_UNDO', message: 'No graph action to undo' } }
    }
    return {
      ok: true,
      state: {
        past: state.past.slice(0, -1),
        present: cloneGraph(previous),
        future: [cloneGraph(state.present), ...state.future],
        version: state.version + 1,
      },
      event: { action: 'undo', code: 'ACTION_UNDONE', message: 'Restored previous graph version' },
    }
  }

  const next = state.future[0]
  if (!next) {
    return { ok: false, state, event: { action: 'redo', code: 'NOTHING_TO_REDO', message: 'No graph action to redo' } }
  }
  return {
    ok: true,
    state: {
      past: [...state.past, cloneGraph(state.present)],
      present: cloneGraph(next),
      future: state.future.slice(1),
      version: state.version + 1,
    },
    event: { action: 'redo', code: 'ACTION_REDONE', message: 'Reapplied graph action' },
  }
}
