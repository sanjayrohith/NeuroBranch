export type OptimizerValue = number | boolean | string | null | Array<number | null>

export interface OptimizerComposition {
  kind: 'composed'
  momentum: boolean
  adaptiveScale: boolean
  normalizeGradient: boolean
  weightDecay?: boolean
  decoupledWeightDecay: boolean
}

export interface OptimizerDefinition {
  id: string
  label: string
  torchClass: string
  defaults: Record<string, OptimizerValue>
  notes?: string
  composition?: OptimizerComposition
}

export interface OptimizerConfig {
  id: string
  kind: string
  settings: Record<string, OptimizerValue>
}

export const optimizerRegistry: Record<string, OptimizerDefinition> = {
  adamw: { id: 'adamw', label: 'AdamW', torchClass: 'AdamW', defaults: { lr: 0.001, betas: [0.9, 0.999], eps: 1e-8, weight_decay: 0.01, amsgrad: false, fused: false } },
  adam: { id: 'adam', label: 'Adam', torchClass: 'Adam', defaults: { lr: 0.001, betas: [0.9, 0.999], eps: 1e-8, weight_decay: 0, amsgrad: false, fused: false } },
  muon: { id: 'muon', label: 'Muon', torchClass: 'Muon', defaults: { lr: 0.001, weight_decay: 0.1, momentum: 0.95, nesterov: true, ns_coefficients: [3.4445, -4.775, 2.0315], eps: 1e-7, ns_steps: 5, adjust_lr_fn: null }, notes: 'PyTorch 2.13 Muon with Newton–Schulz orthogonalization controls.' },
  sgd: { id: 'sgd', label: 'SGD', torchClass: 'SGD', defaults: { lr: 0.001, momentum: 0, dampening: 0, weight_decay: 0, nesterov: false, fused: false } },
  adafactor: { id: 'adafactor', label: 'Adafactor', torchClass: 'Adafactor', defaults: { lr: 0.01, beta2_decay: -0.8, eps: [null, 0.001], d: 1, weight_decay: 0 } },
  rmsprop: { id: 'rmsprop', label: 'RMSprop', torchClass: 'RMSprop', defaults: { lr: 0.01, alpha: 0.99, eps: 1e-8, weight_decay: 0, momentum: 0, centered: false } },
  nadam: { id: 'nadam', label: 'NAdam', torchClass: 'NAdam', defaults: { lr: 0.002, betas: [0.9, 0.999], eps: 1e-8, weight_decay: 0, momentum_decay: 0.004, decoupled_weight_decay: false } },
  radam: { id: 'radam', label: 'RAdam', torchClass: 'RAdam', defaults: { lr: 0.001, betas: [0.9, 0.999], eps: 1e-8, weight_decay: 0, decoupled_weight_decay: false } },
  'sparse-adam': { id: 'sparse-adam', label: 'SparseAdam', torchClass: 'SparseAdam', defaults: { lr: 0.001, betas: [0.9, 0.999], eps: 1e-8 } },
  adagrad: { id: 'adagrad', label: 'Adagrad', torchClass: 'Adagrad', defaults: { lr: 0.01, lr_decay: 0, weight_decay: 0, initial_accumulator_value: 0, eps: 1e-10 } },
  adadelta: { id: 'adadelta', label: 'Adadelta', torchClass: 'Adadelta', defaults: { lr: 1, rho: 0.9, eps: 1e-6, weight_decay: 0 } },
  asgd: { id: 'asgd', label: 'ASGD', torchClass: 'ASGD', defaults: { lr: 0.01, lambd: 0.0001, alpha: 0.75, t0: 1000000, weight_decay: 0 } },
  lbfgs: { id: 'lbfgs', label: 'LBFGS', torchClass: 'LBFGS', defaults: { lr: 1, max_iter: 20, max_eval: 25, tolerance_grad: 1e-7, tolerance_change: 1e-9, history_size: 100, line_search_fn: null } },
}

function cloneValue(value: OptimizerValue): OptimizerValue {
  return Array.isArray(value) ? [...value] : value
}

export function createOptimizerConfig(kind: string, overrides: Record<string, OptimizerValue> = {}, definitions: Record<string, OptimizerDefinition> = optimizerRegistry): OptimizerConfig {
  const definition = definitions[kind]
  if (!definition) throw new Error(`Unknown optimizer: ${kind}`)
  for (const key of Object.keys(overrides)) {
    if (!(key in definition.defaults)) throw new Error(`Unknown ${kind} setting: ${key}`)
  }
  return {
    id: `${kind}-1`,
    kind,
    settings: Object.fromEntries(Object.entries({ ...definition.defaults, ...overrides }).map(([key, value]) => [key, cloneValue(value)])),
  }
}

function pythonValue(value: OptimizerValue): string {
  if (value === null) return 'None'
  if (typeof value === 'boolean') return value ? 'True' : 'False'
  if (typeof value === 'string') return JSON.stringify(value)
  if (Array.isArray(value)) return `(${value.map(pythonValue).join(', ')})`
  return String(value)
}

function compileComposedOptimizer(definition: OptimizerDefinition): string {
  const composition = definition.composition
  if (!composition) return ''
  const className = definition.torchClass
  const initArguments = Object.entries(definition.defaults).map(([key, value]) => `${key}=${pythonValue(value)}`).join(', ')
  const defaultKeys = Object.keys(definition.defaults).map((key) => `${key}=${key}`).join(', ')
  const stateLines = [
    composition.momentum ? "                momentum = state.setdefault('momentum', torch.zeros_like(parameter))\n                momentum.mul_(group['beta1']).add_(gradient, alpha=1 - group['beta1'])\n                update = momentum" : '                update = gradient',
    composition.adaptiveScale ? "                variance = state.setdefault('variance', torch.zeros_like(parameter))\n                variance.mul_(group['beta2']).addcmul_(gradient, gradient, value=1 - group['beta2'])\n                update = update / (variance.sqrt() + group['eps'])" : '',
    composition.normalizeGradient ? "                update = update / (update.norm().clamp_min(group.get('eps', 1e-8)))" : '',
    composition.weightDecay === false ? '' : composition.decoupledWeightDecay ? "                parameter.mul_(1 - group['lr'] * group.get('weight_decay', 0.0))" : "                update = update.add(parameter, alpha=group.get('weight_decay', 0.0))",
  ].filter(Boolean).join('\n')
  return `class ${className}(torch.optim.Optimizer):
    def __init__(self, params, ${initArguments}):
        defaults = dict(${defaultKeys})
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
${stateLines}
                parameter.add_(update, alpha=-group['lr'])
        return loss`
}

export function compileOptimizer(config: OptimizerConfig, parameters = 'model.parameters()', definitions: Record<string, OptimizerDefinition> = optimizerRegistry): string {
  const definition = definitions[config.kind]
  if (!definition) throw new Error(`Unknown optimizer: ${config.kind}`)
  const settings = Object.entries(config.settings).map(([key, value]) => `${key}=${pythonValue(value)}`).join(', ')
  if (definition.composition) return `${compileComposedOptimizer(definition)}\n\noptimizer = ${definition.torchClass}(${parameters}, ${settings})`
  return `optimizer = torch.optim.${definition.torchClass}(${parameters}, ${settings})`
}
