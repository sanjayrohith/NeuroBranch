import { describe, expect, it } from 'vitest'
import { cloneArchitectureGraph, parseModelWorkspace } from './model-workspace'
import { gptLikeStarterPreset } from './presets'

describe('model workspace persistence', () => {
  it('removes a legacy Blank starter draft duplicated into a saved user preset', () => {
    const blankDraft = cloneArchitectureGraph({ ...gptLikeStarterPreset, id: 'blank-starter', name: 'Blank starter' })
    const savedPreset = cloneArchitectureGraph({ ...blankDraft, id: 'user-my-model', name: 'My model' })

    const workspace = parseModelWorkspace({
      activePresetId: savedPreset.id,
      drafts: {
        'blank-starter': { graph: blankDraft, selectedNodeId: blankDraft.nodes[0]?.id ?? '' },
        [savedPreset.id]: { graph: savedPreset, selectedNodeId: savedPreset.nodes[0]?.id ?? '' },
      },
      userPresets: [savedPreset],
      updatedAt: 1,
    })

    expect(workspace.drafts['blank-starter']).toBeUndefined()
    expect(workspace.drafts[savedPreset.id]?.graph.nodes).toHaveLength(savedPreset.nodes.length)
  })
})
