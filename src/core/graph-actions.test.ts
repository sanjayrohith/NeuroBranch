import { describe, expect, it } from 'vitest'
import { createGraphEditorState, dispatchGraphAction } from './graph-actions'
import { gqaPreset } from './presets'

describe('transactional graph actions', () => {
  it('deletes an atom and supports exact undo and redo', () => {
    const initial = createGraphEditorState(gqaPreset)
    const deleted = dispatchGraphAction(initial, { type: 'delete-node', nodeId: 'q-proj' })

    expect(deleted.ok).toBe(true)
    expect(deleted.event).toMatchObject({ action: 'delete-node', code: 'NODE_DELETED', nodeId: 'q-proj' })
    expect(deleted.state.present.nodes.some((node) => node.id === 'q-proj')).toBe(false)
    expect(deleted.state.version).toBe(1)

    const undone = dispatchGraphAction(deleted.state, { type: 'undo' })
    expect(undone.ok).toBe(true)
    expect(undone.state.present).toEqual(gqaPreset)
    expect(undone.state.version).toBe(2)

    const redone = dispatchGraphAction(undone.state, { type: 'redo' })
    expect(redone.ok).toBe(true)
    expect(redone.state.present.nodes.some((node) => node.id === 'q-proj')).toBe(false)
    expect(redone.state.version).toBe(3)
  })

  it('rejects an incompatible typed port without mutating the graph', () => {
    const initial = createGraphEditorState(gqaPreset)
    const outcome = dispatchGraphAction(initial, {
      type: 'connect-ports',
      edgeId: 'wrong-v-query',
      sourceId: 'v-proj',
      sourcePort: 'output',
      targetId: 'sdpa',
      targetPort: 'query',
    })

    expect(outcome.ok).toBe(false)
    expect(outcome.state).toBe(initial)
    expect(outcome.event).toMatchObject({
      action: 'connect-ports',
      code: 'PORT_TYPE_MISMATCH',
      nodeId: 'sdpa',
      portId: 'query',
    })
    expect(outcome.state.version).toBe(0)
  })

  it('rejects a connection that creates a graph cycle', () => {
    const initial = createGraphEditorState(gqaPreset)
    const outcome = dispatchGraphAction(initial, {
      type: 'connect-ports',
      edgeId: 'output-hidden-cycle',
      sourceId: 'output',
      sourcePort: 'output',
      targetId: 'hidden',
      targetPort: 'input',
    })

    expect(outcome.ok).toBe(false)
    expect(outcome.state).toBe(initial)
    expect(outcome.event).toMatchObject({ code: 'GRAPH_CYCLE', nodeId: 'hidden', portId: 'input' })
  })

  it('rejects invalid numeric settings without changing the atom', () => {
    const initial = createGraphEditorState(gqaPreset)
    const outcome = dispatchGraphAction(initial, {
      type: 'update-attributes',
      nodeId: 'q-proj',
      attributes: { outFeatures: 0 },
    })

    expect(outcome.ok).toBe(false)
    expect(outcome.state).toBe(initial)
    expect(outcome.event).toMatchObject({ code: 'INVALID_SETTING', nodeId: 'q-proj', settingId: 'outFeatures' })
    expect(outcome.state.present.nodes.find((node) => node.id === 'q-proj')?.attributes?.outFeatures).toBe(384)
  })
})
