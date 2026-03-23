import { describe, expect, it } from 'vitest'
import { activationAtomRegistry, gatedActivationRegistry } from './activation-atoms'

describe('activation atom catalog', () => {
  it('covers the torch.nn activation families as editable atoms', () => {
    const required = [
      'relu', 'relu6', 'gelu', 'silu', 'sigmoid', 'tanh', 'softplus', 'softsign',
      'elu', 'celu', 'selu', 'leaky-relu', 'prelu', 'rrelu', 'mish',
      'hardsigmoid', 'hardswish', 'hardtanh', 'hardshrink', 'softshrink',
      'tanhshrink', 'threshold', 'logsigmoid', 'glu',
    ]
    for (const id of required) {
      expect(activationAtomRegistry[id], `missing ${id}`).toBeDefined()
      expect(activationAtomRegistry[id].lowerings.pytorch).toMatchObject({
        executable: true,
        declarations: [expect.stringMatching(/^self\.\{\{module\}\} = nn\./)],
      })
      expect(activationAtomRegistry[id].inputs[0]?.tensor).toBe('hidden')
      expect(activationAtomRegistry[id].outputs[0]?.tensor).toBe('hidden')
    }
  })

  it('defines expandable gated activation composites', () => {
    expect(Object.keys(gatedActivationRegistry)).toEqual(['swiglu', 'geglu', 'reglu'])
    expect(gatedActivationRegistry.swiglu).toMatchObject({ gateActivation: 'silu' })
    expect(gatedActivationRegistry.geglu).toMatchObject({ gateActivation: 'gelu' })
    expect(gatedActivationRegistry.reglu).toMatchObject({ gateActivation: 'relu' })
  })
})
