import { describe, expect, it } from 'vitest'
import { compileOptimizer, createOptimizerConfig, optimizerRegistry } from './optimizer-ir'

describe('training optimizer IR', () => {
  it('catalogs the optimizers available in the pinned PyTorch runtime', () => {
    const required = ['adamw', 'adam', 'muon', 'sgd', 'adafactor', 'rmsprop', 'nadam', 'radam', 'sparse-adam', 'adagrad', 'adadelta', 'asgd', 'lbfgs']
    for (const id of required) expect(optimizerRegistry[id], `missing ${id}`).toBeDefined()
  })

  it('compiles editable AdamW settings to exact PyTorch', () => {
    const config = createOptimizerConfig('adamw', { lr: 0.0003, betas: [0.9, 0.95], weight_decay: 0.1, fused: true })
    expect(compileOptimizer(config)).toBe('optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, amsgrad=False, fused=True)')
  })

  it('compiles Muon with its PyTorch 2.13 Newton-Schulz controls', () => {
    const config = createOptimizerConfig('muon', { lr: 0.02, momentum: 0.95, weight_decay: 0.1, nesterov: true, ns_steps: 5 })
    expect(compileOptimizer(config)).toContain('torch.optim.Muon(model.parameters()')
    expect(compileOptimizer(config)).toContain('momentum=0.95')
    expect(compileOptimizer(config)).toContain('nesterov=True')
    expect(compileOptimizer(config)).toContain('ns_steps=5')
  })

  it('compiles a genuinely composed optimizer class from update modules', () => {
    const custom = {
      id: 'custom-labo',
      label: 'LABO optimizer',
      torchClass: 'LaboOptimizer',
      defaults: { lr: 0.001, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.01 },
      composition: { kind: 'composed' as const, momentum: true, adaptiveScale: true, normalizeGradient: false, decoupledWeightDecay: true },
    }
    const code = compileOptimizer(createOptimizerConfig(custom.id, {}, { [custom.id]: custom }), 'model.parameters()', { [custom.id]: custom })
    expect(code).toContain('class LaboOptimizer(torch.optim.Optimizer):')
    expect(code).toContain("momentum = state.setdefault('momentum'")
    expect(code).toContain("variance = state.setdefault('variance'")
    expect(code).toContain("parameter.mul_(1 - group['lr'] * group.get('weight_decay', 0.0))")
    expect(code).toContain('optimizer = LaboOptimizer(model.parameters()')
    expect(code).not.toContain('torch.optim.LaboOptimizer')
  })
})
