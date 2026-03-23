import { activationAtomRegistry } from './activation-atoms'
import { mediaAtomRegistry } from './media-atoms'

export type ModelAtomCategory =
  | 'embedding'
  | 'normalization'
  | 'attention'
  | 'position'
  | 'composition'
  | 'mlp'
  | 'output'
  | 'objective'
  | 'activation'
  | 'routing'
  | 'media'

export interface AtomPort {
  id: string
  tensor: 'token-ids' | 'image' | 'video' | 'audio' | 'hidden' | 'query' | 'key' | 'value' | 'attention' | 'logits' | 'labels' | 'scalar' | 'routing-logits' | 'expert-indices' | 'routing-weights'
  rank?: number
}

export interface AtomSetting {
  id: string
  type: 'number' | 'boolean' | 'string' | 'select'
  default: number | boolean | string
  options?: string[]
}

export interface PyTorchAtomLowering {
  executable: true
  declarations: string[]
  forward: string[]
  helpers?: string[]
}

export interface ModelAtomDefinition {
  id: string
  label: string
  category: ModelAtomCategory
  inputs: AtomPort[]
  outputs: AtomPort[]
  settings: AtomSetting[]
  lowerings: { pytorch: PyTorchAtomLowering }
  composite?: { atomIds: string[] }
}

const hiddenInput: AtomPort = { id: 'hidden', tensor: 'hidden', rank: 3 }
const hiddenOutput: AtomPort = { id: 'output', tensor: 'hidden', rank: 3 }

function lowering(declarations: string[], forward: string[], helpers: string[] = []): ModelAtomDefinition['lowerings'] {
  return { pytorch: { executable: true, declarations, forward, ...(helpers.length > 0 ? { helpers } : {}) } }
}

function moduleLowering(constructor: string, input = 'hidden', output = 'output'): ModelAtomDefinition['lowerings'] {
  return lowering([`self.{{module}} = ${constructor}`], [`{{out:${output}}} = self.{{module}}({{in:${input}}})`])
}

function unaryHiddenAtom(id: string, label: string, category: ModelAtomCategory, expression: string, settings: AtomSetting[] = [], declarations: string[] = []): ModelAtomDefinition {
  return { id, label, category, inputs: [hiddenInput], outputs: [hiddenOutput], settings, lowerings: lowering(declarations, [`{{out:output}} = ${expression}`]) }
}

function binaryHiddenAtom(id: string, label: string, expression: string): ModelAtomDefinition {
  return {
    id, label, category: 'composition',
    inputs: [{ id: 'left', tensor: 'hidden', rank: 3 }, { id: 'right', tensor: 'hidden', rank: 3 }],
    outputs: [hiddenOutput], settings: [], lowerings: lowering([], [`{{out:output}} = ${expression}`]),
  }
}

function denseMlpAtom(id: string, label: string, activation: string, residual = false): ModelAtomDefinition {
  const branch = `self.{{module}}_down(${activation.replaceAll('$up', 'self.{{module}}_up({{in:hidden}})')})`
  return {
    id, label, category: 'mlp', inputs: [hiddenInput], outputs: [hiddenOutput],
    settings: [{ id: 'intermediateSize', type: 'number', default: 3072 }, { id: 'bias', type: 'boolean', default: true }],
    lowerings: lowering([
      'self.{{module}}_up = nn.Linear({{hiddenSize}}, {{intermediateSize}}, bias={{bias}})',
      'self.{{module}}_down = nn.Linear({{intermediateSize}}, {{hiddenSize}}, bias={{bias}})',
    ], [`{{out:output}} = ${residual ? `{{in:hidden}} + ${branch}` : branch}`]),
  }
}

const ropeHelper = 'def _labo_apply_rope(x, base):\n    sequence, dim = x.shape[-2], x.shape[-1]\n    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim))\n    angles = torch.outer(torch.arange(sequence, device=x.device, dtype=torch.float32), inv_freq)\n    cos = torch.repeat_interleave(angles.cos(), 2, dim=-1).to(dtype=x.dtype)\n    sin = torch.repeat_interleave(angles.sin(), 2, dim=-1).to(dtype=x.dtype)\n    rotated = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)\n    return x * cos + rotated * sin'

const additionalAtomRegistry: Record<string, ModelAtomDefinition> = {
  'scale-norm': unaryHiddenAtom('scale-norm', 'ScaleNorm', 'normalization', '{{in:hidden}} * ({{scale}} / {{in:hidden}}.norm(dim=-1, keepdim=True).clamp_min({{epsilon}}))', [
    { id: 'scale', type: 'number', default: 1 }, { id: 'epsilon', type: 'number', default: 1e-6 },
  ]),
  'l2-normalization': unaryHiddenAtom('l2-normalization', 'L2 normalization', 'normalization', 'F.normalize({{in:hidden}}, p=2, dim=-1, eps={{epsilon}})', [{ id: 'epsilon', type: 'number', default: 1e-12 }]),
  'group-norm': {
    id: 'group-norm', label: 'GroupNorm 1D', category: 'normalization', inputs: [hiddenInput], outputs: [hiddenOutput],
    settings: [{ id: 'numGroups', type: 'number', default: 8 }, { id: 'epsilon', type: 'number', default: 1e-5 }],
    lowerings: lowering(['self.{{module}} = nn.GroupNorm({{numGroups}}, {{hiddenSize}}, eps={{epsilon}})'], ['{{out:output}} = self.{{module}}({{in:hidden}}.transpose(1, 2)).transpose(1, 2)']),
  },
  'batch-norm-1d': {
    id: 'batch-norm-1d', label: 'BatchNorm 1D', category: 'normalization', inputs: [hiddenInput], outputs: [hiddenOutput],
    settings: [{ id: 'epsilon', type: 'number', default: 1e-5 }, { id: 'momentum', type: 'number', default: 0.1 }],
    lowerings: lowering(['self.{{module}} = nn.BatchNorm1d({{hiddenSize}}, eps={{epsilon}}, momentum={{momentum}})'], ['{{out:output}} = self.{{module}}({{in:hidden}}.transpose(1, 2)).transpose(1, 2)']),
  },
  'instance-norm-1d': {
    id: 'instance-norm-1d', label: 'InstanceNorm 1D', category: 'normalization', inputs: [hiddenInput], outputs: [hiddenOutput],
    settings: [{ id: 'epsilon', type: 'number', default: 1e-5 }, { id: 'affine', type: 'boolean', default: true }],
    lowerings: lowering(['self.{{module}} = nn.InstanceNorm1d({{hiddenSize}}, eps={{epsilon}}, affine={{affine}})'], ['{{out:output}} = self.{{module}}({{in:hidden}}.transpose(1, 2)).transpose(1, 2)']),
  },
  'mean-centering': unaryHiddenAtom('mean-centering', 'Mean centering', 'normalization', '{{in:hidden}} - {{in:hidden}}.mean(dim=-1, keepdim=True)'),
  standardization: unaryHiddenAtom('standardization', 'Feature standardization', 'normalization', '({{in:hidden}} - {{in:hidden}}.mean(dim=-1, keepdim=True)) / {{in:hidden}}.std(dim=-1, keepdim=True, unbiased=False).clamp_min({{epsilon}})', [{ id: 'epsilon', type: 'number', default: 1e-6 }]),
  'unit-rms': unaryHiddenAtom('unit-rms', 'Unit RMS normalization', 'normalization', '{{in:hidden}} * torch.rsqrt({{in:hidden}}.pow(2).mean(dim=-1, keepdim=True) + {{epsilon}})', [{ id: 'epsilon', type: 'number', default: 1e-6 }]),

  'sinusoidal-position-encoding': {
    id: 'sinusoidal-position-encoding', label: 'Sinusoidal position encoding', category: 'position',
    inputs: [{ id: 'tokenIds', tensor: 'token-ids', rank: 2 }], outputs: [hiddenOutput], settings: [{ id: 'base', type: 'number', default: 10000 }],
    lowerings: lowering([], [
      '{{module}}_position = torch.arange({{in:tokenIds}}.shape[-1], device={{in:tokenIds}}.device, dtype=torch.float32).unsqueeze(1)',
      '{{module}}_frequency = torch.exp(torch.arange(0, {{hiddenSize}}, 2, device={{in:tokenIds}}.device, dtype=torch.float32) * (-torch.log(torch.tensor({{base}}, device={{in:tokenIds}}.device)) / {{hiddenSize}}))',
      '{{module}}_encoding = torch.zeros({{in:tokenIds}}.shape[-1], {{hiddenSize}}, device={{in:tokenIds}}.device)',
      '{{module}}_encoding[:, 0::2] = torch.sin({{module}}_position * {{module}}_frequency)',
      '{{module}}_encoding[:, 1::2] = torch.cos({{module}}_position * {{module}}_frequency)',
      '{{out:output}} = {{module}}_encoding.unsqueeze(0).expand({{in:tokenIds}}.shape[0], -1, -1)',
    ]),
  },
  'position-ramp': {
    id: 'position-ramp', label: 'Normalized position ramp', category: 'position',
    inputs: [{ id: 'tokenIds', tensor: 'token-ids', rank: 2 }], outputs: [hiddenOutput], settings: [],
    lowerings: lowering([], [
      '{{module}}_ramp = torch.linspace(0, 1, {{in:tokenIds}}.shape[-1], device={{in:tokenIds}}.device)',
      '{{out:output}} = {{module}}_ramp.view(1, -1, 1).expand({{in:tokenIds}}.shape[0], -1, {{hiddenSize}})',
    ]),
  },
  'query-rope': {
    id: 'query-rope', label: 'Query-only RoPE', category: 'position', inputs: [{ id: 'q', tensor: 'query', rank: 4 }], outputs: [{ id: 'q', tensor: 'query', rank: 4 }],
    settings: [{ id: 'base', type: 'number', default: 10000 }], lowerings: lowering([], ['{{out:q}} = _labo_apply_rope({{in:q}}, {{base}})'], [ropeHelper]),
  },
  'key-rope': {
    id: 'key-rope', label: 'Key-only RoPE', category: 'position', inputs: [{ id: 'k', tensor: 'key', rank: 4 }], outputs: [{ id: 'k', tensor: 'key', rank: 4 }],
    settings: [{ id: 'base', type: 'number', default: 10000 }], lowerings: lowering([], ['{{out:k}} = _labo_apply_rope({{in:k}}, {{base}})'], [ropeHelper]),
  },

  'noncausal-sdpa': {
    id: 'noncausal-sdpa', label: 'Bidirectional SDPA', category: 'attention',
    inputs: [{ id: 'q', tensor: 'query', rank: 4 }, { id: 'k', tensor: 'key', rank: 4 }, { id: 'v', tensor: 'value', rank: 4 }], outputs: [{ id: 'output', tensor: 'attention', rank: 4 }],
    settings: [{ id: 'dropout', type: 'number', default: 0 }], lowerings: lowering([], ['{{out:output}} = F.scaled_dot_product_attention({{in:q}}, {{in:k}}, {{in:v}}, dropout_p={{dropout}}, is_causal=False, enable_gqa={{in:q}}.size(1) != {{in:k}}.size(1))']),
  },
  'attention-scores': {
    id: 'attention-scores', label: 'Scaled attention scores', category: 'attention',
    inputs: [{ id: 'q', tensor: 'query', rank: 4 }, { id: 'k', tensor: 'key', rank: 4 }], outputs: [{ id: 'scores', tensor: 'attention', rank: 4 }], settings: [],
    lowerings: lowering([], ['{{out:scores}} = torch.matmul({{in:q}}, {{in:k}}.transpose(-2, -1)) * ({{in:q}}.shape[-1] ** -0.5)']),
  },
  'attention-softmax': {
    id: 'attention-softmax', label: 'Attention softmax', category: 'attention', inputs: [{ id: 'scores', tensor: 'attention', rank: 4 }], outputs: [{ id: 'output', tensor: 'attention', rank: 4 }],
    settings: [{ id: 'dimension', type: 'number', default: -1 }], lowerings: lowering([], ['{{out:output}} = torch.softmax({{in:scores}}, dim={{dimension}})']),
  },
  'attention-dropout': {
    id: 'attention-dropout', label: 'Attention dropout', category: 'attention', inputs: [{ id: 'attention', tensor: 'attention', rank: 4 }], outputs: [{ id: 'output', tensor: 'attention', rank: 4 }],
    settings: [{ id: 'probability', type: 'number', default: 0.1 }], lowerings: lowering([], ['{{out:output}} = F.dropout({{in:attention}}, p={{probability}}, training=self.training)']),
  },
  'attention-value-mix': {
    id: 'attention-value-mix', label: 'Attention × value mix', category: 'attention',
    inputs: [{ id: 'weights', tensor: 'attention', rank: 4 }, { id: 'v', tensor: 'value', rank: 4 }], outputs: [{ id: 'output', tensor: 'attention', rank: 4 }], settings: [],
    lowerings: lowering([], ['{{out:output}} = torch.matmul({{in:weights}}, {{in:v}})']),
  },

  subtract: binaryHiddenAtom('subtract', 'Tensor subtraction', '{{in:left}} - {{in:right}}'),
  average: binaryHiddenAtom('average', 'Tensor average', '({{in:left}} + {{in:right}}) * 0.5'),
  maximum: binaryHiddenAtom('maximum', 'Elementwise maximum', 'torch.maximum({{in:left}}, {{in:right}})'),
  minimum: binaryHiddenAtom('minimum', 'Elementwise minimum', 'torch.minimum({{in:left}}, {{in:right}})'),
  'gated-blend': {
    id: 'gated-blend', label: 'Learned gated blend', category: 'composition',
    inputs: [{ id: 'left', tensor: 'hidden', rank: 3 }, { id: 'right', tensor: 'hidden', rank: 3 }], outputs: [hiddenOutput], settings: [{ id: 'gateInit', type: 'number', default: 0 }],
    lowerings: lowering(['self.{{module}}_gate = nn.Parameter(torch.tensor({{gateInit}}, dtype=torch.float32))'], ['{{module}}_weight = torch.sigmoid(self.{{module}}_gate)', '{{out:output}} = {{module}}_weight * {{in:left}} + (1 - {{module}}_weight) * {{in:right}}']),
  },
  'concatenate-projection': {
    id: 'concatenate-projection', label: 'Concatenate + projection', category: 'composition',
    inputs: [{ id: 'left', tensor: 'hidden', rank: 3 }, { id: 'right', tensor: 'hidden', rank: 3 }], outputs: [hiddenOutput], settings: [{ id: 'bias', type: 'boolean', default: false }],
    lowerings: lowering(['self.{{module}} = nn.Linear({{hiddenSize}} * 2, {{hiddenSize}}, bias={{bias}})'], ['{{out:output}} = self.{{module}}(torch.cat(({{in:left}}, {{in:right}}), dim=-1))']),
  },
  clamp: unaryHiddenAtom('clamp', 'Tensor clamp', 'composition', 'torch.clamp({{in:hidden}}, min={{minimum}}, max={{maximum}})', [{ id: 'minimum', type: 'number', default: -1 }, { id: 'maximum', type: 'number', default: 1 }]),
  power: unaryHiddenAtom('power', 'Tensor power', 'composition', '{{in:hidden}}.pow({{exponent}})', [{ id: 'exponent', type: 'number', default: 2 }]),
  'absolute-value': unaryHiddenAtom('absolute-value', 'Absolute value', 'composition', '{{in:hidden}}.abs()'),
  negate: unaryHiddenAtom('negate', 'Tensor negation', 'composition', '-{{in:hidden}}'),
  'stop-gradient': unaryHiddenAtom('stop-gradient', 'Stop gradient / detach', 'composition', '{{in:hidden}}.detach()'),
  'stochastic-depth': unaryHiddenAtom('stochastic-depth', 'Stochastic depth', 'composition', 'F.dropout({{in:hidden}}, p={{probability}}, training=self.training) * (1 - {{probability}})', [{ id: 'probability', type: 'number', default: 0.1 }]),

  'tanh-mlp': denseMlpAtom('tanh-mlp', 'Tanh MLP', 'torch.tanh($up)'),
  'sigmoid-mlp': denseMlpAtom('sigmoid-mlp', 'Sigmoid MLP', 'torch.sigmoid($up)'),
  'mish-mlp': denseMlpAtom('mish-mlp', 'Mish MLP', 'F.mish($up)'),
  'squared-relu-mlp': denseMlpAtom('squared-relu-mlp', 'Squared-ReLU MLP', 'F.relu($up).pow(2)'),
  'leaky-relu-mlp': denseMlpAtom('leaky-relu-mlp', 'Leaky-ReLU MLP', 'F.leaky_relu($up, negative_slope=0.01)'),
  'residual-gelu-mlp': denseMlpAtom('residual-gelu-mlp', 'Residual GELU MLP', 'F.gelu($up)', true),
  'residual-swiglu-mlp': {
    id: 'residual-swiglu-mlp', label: 'Residual SwiGLU MLP', category: 'mlp', inputs: [hiddenInput], outputs: [hiddenOutput],
    settings: [{ id: 'intermediateSize', type: 'number', default: 2048 }, { id: 'bias', type: 'boolean', default: false }],
    lowerings: lowering([
      'self.{{module}}_gate = nn.Linear({{hiddenSize}}, {{intermediateSize}}, bias={{bias}})',
      'self.{{module}}_up = nn.Linear({{hiddenSize}}, {{intermediateSize}}, bias={{bias}})',
      'self.{{module}}_down = nn.Linear({{intermediateSize}}, {{hiddenSize}}, bias={{bias}})',
    ], ['{{out:output}} = {{in:hidden}} + self.{{module}}_down(F.silu(self.{{module}}_gate({{in:hidden}})) * self.{{module}}_up({{in:hidden}}))']),
  },

  'softmax-output': {
    id: 'softmax-output', label: 'Softmax output', category: 'output', inputs: [{ id: 'logits', tensor: 'logits', rank: 3 }], outputs: [{ id: 'logits', tensor: 'logits', rank: 3 }],
    settings: [{ id: 'dimension', type: 'number', default: -1 }], lowerings: lowering([], ['{{out:logits}} = torch.softmax({{in:logits}}, dim={{dimension}})']),
  },
  'temperature-scaling': {
    id: 'temperature-scaling', label: 'Logit temperature scaling', category: 'output', inputs: [{ id: 'logits', tensor: 'logits', rank: 3 }], outputs: [{ id: 'logits', tensor: 'logits', rank: 3 }],
    settings: [{ id: 'temperature', type: 'number', default: 1 }], lowerings: lowering([], ['{{out:logits}} = {{in:logits}} / max({{temperature}}, 1e-6)']),
  },
  'logits-clamp': {
    id: 'logits-clamp', label: 'Logits clamp', category: 'output', inputs: [{ id: 'logits', tensor: 'logits', rank: 3 }], outputs: [{ id: 'logits', tensor: 'logits', rank: 3 }],
    settings: [{ id: 'minimum', type: 'number', default: -30 }, { id: 'maximum', type: 'number', default: 30 }], lowerings: lowering([], ['{{out:logits}} = torch.clamp({{in:logits}}, min={{minimum}}, max={{maximum}})']),
  },

  'mean-squared-error': {
    id: 'mean-squared-error', label: 'Mean-squared error', category: 'objective',
    inputs: [{ id: 'prediction', tensor: 'hidden', rank: 3 }, { id: 'target', tensor: 'hidden', rank: 3 }], outputs: [{ id: 'loss', tensor: 'scalar', rank: 0 }], settings: [],
    lowerings: lowering([], ['{{out:loss}} = F.mse_loss({{in:prediction}}, {{in:target}})']),
  },
  'l1-loss': {
    id: 'l1-loss', label: 'L1 loss', category: 'objective',
    inputs: [{ id: 'prediction', tensor: 'hidden', rank: 3 }, { id: 'target', tensor: 'hidden', rank: 3 }], outputs: [{ id: 'loss', tensor: 'scalar', rank: 0 }], settings: [],
    lowerings: lowering([], ['{{out:loss}} = F.l1_loss({{in:prediction}}, {{in:target}})']),
  },
  'binary-cross-entropy': {
    id: 'binary-cross-entropy', label: 'Binary cross-entropy with logits', category: 'objective',
    inputs: [{ id: 'logits', tensor: 'hidden', rank: 3 }, { id: 'targets', tensor: 'hidden', rank: 3 }], outputs: [{ id: 'loss', tensor: 'scalar', rank: 0 }], settings: [],
    lowerings: lowering([], ['{{out:loss}} = F.binary_cross_entropy_with_logits({{in:logits}}, {{in:targets}})']),
  },
  'logits-z-loss': {
    id: 'logits-z-loss', label: 'Logits Z-loss', category: 'objective', inputs: [{ id: 'logits', tensor: 'logits', rank: 3 }], outputs: [{ id: 'loss', tensor: 'scalar', rank: 0 }],
    settings: [{ id: 'coefficient', type: 'number', default: 0.0001 }], lowerings: lowering([], ['{{out:loss}} = {{coefficient}} * torch.logsumexp({{in:logits}}, dim=-1).pow(2).mean()']),
  },
  'output-entropy-loss': {
    id: 'output-entropy-loss', label: 'Output entropy loss', category: 'objective', inputs: [{ id: 'logits', tensor: 'logits', rank: 3 }], outputs: [{ id: 'loss', tensor: 'scalar', rank: 0 }],
    settings: [{ id: 'coefficient', type: 'number', default: 0.001 }], lowerings: lowering([], ['{{module}}_log_probs = F.log_softmax({{in:logits}}, dim=-1)', '{{out:loss}} = -{{coefficient}} * ({{module}}_log_probs.exp() * {{module}}_log_probs).sum(dim=-1).mean()']),
  },
}

export const modelAtomRegistry: Record<string, ModelAtomDefinition> = {
  ...activationAtomRegistry,
  ...mediaAtomRegistry,
  ...additionalAtomRegistry,
  'token-embedding': {
    id: 'token-embedding', label: 'Token embedding', category: 'embedding',
    inputs: [{ id: 'tokenIds', tensor: 'token-ids', rank: 2 }], outputs: [hiddenOutput],
    settings: [
      { id: 'vocabSize', type: 'number', default: 32000 },
      { id: 'hiddenSize', type: 'number', default: 768 },
    ],
    lowerings: moduleLowering('nn.Embedding({{vocabSize}}, {{hiddenSize}})', 'tokenIds'),
  },
  'learned-position-embedding': {
    id: 'learned-position-embedding', label: 'Learned position embedding', category: 'position',
    inputs: [{ id: 'tokenIds', tensor: 'token-ids', rank: 2 }], outputs: [hiddenOutput],
    settings: [
      { id: 'maxPositions', type: 'number', default: 2048 },
      { id: 'hiddenSize', type: 'number', default: 768 },
    ],
    lowerings: lowering(
      ['self.{{module}} = nn.Embedding({{maxPositions}}, {{hiddenSize}})'],
      [
        '{{module}}_positions = torch.arange({{in:tokenIds}}.shape[-1], device={{in:tokenIds}}.device)',
        '{{out:output}} = self.{{module}}({{module}}_positions).unsqueeze(0).expand({{in:tokenIds}}.shape[0], -1, -1)',
      ],
    ),
  },
  'rms-norm': {
    id: 'rms-norm', label: 'RMSNorm', category: 'normalization',
    inputs: [hiddenInput], outputs: [hiddenOutput],
    settings: [{ id: 'epsilon', type: 'number', default: 1e-6 }],
    lowerings: moduleLowering('nn.RMSNorm({{hiddenSize}}, eps={{epsilon}})'),
  },
  'layer-norm': {
    id: 'layer-norm', label: 'LayerNorm', category: 'normalization',
    inputs: [hiddenInput], outputs: [hiddenOutput],
    settings: [{ id: 'epsilon', type: 'number', default: 1e-5 }],
    lowerings: moduleLowering('nn.LayerNorm({{hiddenSize}}, eps={{epsilon}})'),
  },
  'query-projection': {
    id: 'query-projection', label: 'Query projection', category: 'attention',
    inputs: [hiddenInput], outputs: [{ id: 'q', tensor: 'query', rank: 3 }],
    settings: [{ id: 'bias', type: 'boolean', default: false }],
    lowerings: lowering(
      ['self.{{module}} = nn.Linear({{hiddenSize}}, {{queryHeads}} * {{headDim}}, bias={{bias}})'],
      ['{{out:q}} = self.{{module}}({{in:hidden}})'],
    ),
  },
  'key-projection': {
    id: 'key-projection', label: 'Key projection', category: 'attention',
    inputs: [hiddenInput], outputs: [{ id: 'k', tensor: 'key', rank: 3 }],
    settings: [{ id: 'bias', type: 'boolean', default: false }],
    lowerings: lowering(
      ['self.{{module}} = nn.Linear({{hiddenSize}}, {{keyValueHeads}} * {{headDim}}, bias={{bias}})'],
      ['{{out:k}} = self.{{module}}({{in:hidden}})'],
    ),
  },
  'value-projection': {
    id: 'value-projection', label: 'Value projection', category: 'attention',
    inputs: [hiddenInput], outputs: [{ id: 'v', tensor: 'value', rank: 3 }],
    settings: [{ id: 'bias', type: 'boolean', default: false }],
    lowerings: lowering(
      ['self.{{module}} = nn.Linear({{hiddenSize}}, {{keyValueHeads}} * {{headDim}}, bias={{bias}})'],
      ['{{out:v}} = self.{{module}}({{in:hidden}})'],
    ),
  },
  'qkv-projection': {
    id: 'qkv-projection', label: 'QKV projection', category: 'attention',
    inputs: [hiddenInput],
    outputs: [
      { id: 'q', tensor: 'query', rank: 3 },
      { id: 'k', tensor: 'key', rank: 3 },
      { id: 'v', tensor: 'value', rank: 3 },
    ],
    settings: [
      { id: 'bias', type: 'boolean', default: false },
      { id: 'fused', type: 'boolean', default: false },
    ],
    lowerings: lowering([
      'self.{{module}}_q = nn.Linear({{hiddenSize}}, {{queryHeads}} * {{headDim}}, bias={{bias}})',
      'self.{{module}}_k = nn.Linear({{hiddenSize}}, {{keyValueHeads}} * {{headDim}}, bias={{bias}})',
      'self.{{module}}_v = nn.Linear({{hiddenSize}}, {{keyValueHeads}} * {{headDim}}, bias={{bias}})',
    ], [
      '{{out:q}} = self.{{module}}_q({{in:hidden}})',
      '{{out:k}} = self.{{module}}_k({{in:hidden}})',
      '{{out:v}} = self.{{module}}_v({{in:hidden}})',
    ]),
  },
  'attention-head-layout': {
    id: 'attention-head-layout', label: 'Attention head layout', category: 'attention',
    inputs: [
      { id: 'q', tensor: 'query', rank: 3 },
      { id: 'k', tensor: 'key', rank: 3 },
      { id: 'v', tensor: 'value', rank: 3 },
    ],
    outputs: [
      { id: 'qHeads', tensor: 'query', rank: 4 },
      { id: 'kHeads', tensor: 'key', rank: 4 },
      { id: 'vHeads', tensor: 'value', rank: 4 },
    ],
    settings: [
      { id: 'queryHeads', type: 'number', default: 12 },
      { id: 'keyValueHeads', type: 'number', default: 12 },
      { id: 'headDim', type: 'number', default: 64 },
    ],
    lowerings: lowering([], [
      '{{module}}_batch, {{module}}_sequence, _ = {{in:q}}.shape',
      '{{out:qHeads}} = {{in:q}}.view({{module}}_batch, {{module}}_sequence, {{queryHeads}}, {{headDim}}).transpose(1, 2)',
      '{{out:kHeads}} = {{in:k}}.view({{module}}_batch, {{module}}_sequence, {{keyValueHeads}}, {{headDim}}).transpose(1, 2)',
      '{{out:vHeads}} = {{in:v}}.view({{module}}_batch, {{module}}_sequence, {{keyValueHeads}}, {{headDim}}).transpose(1, 2)',
    ]),
  },
  'qk-normalization': {
    id: 'qk-normalization', label: 'QK normalization', category: 'normalization',
    inputs: [{ id: 'q', tensor: 'query', rank: 4 }, { id: 'k', tensor: 'key', rank: 4 }],
    outputs: [{ id: 'q', tensor: 'query', rank: 4 }, { id: 'k', tensor: 'key', rank: 4 }],
    settings: [{ id: 'epsilon', type: 'number', default: 1e-6 }],
    lowerings: lowering([
      'self.{{module}}_q = nn.RMSNorm({{headDim}}, eps={{epsilon}})',
      'self.{{module}}_k = nn.RMSNorm({{headDim}}, eps={{epsilon}})',
    ], [
      '{{out:q}} = self.{{module}}_q({{in:q}})',
      '{{out:k}} = self.{{module}}_k({{in:k}})',
    ]),
  },
  rope: {
    id: 'rope', label: 'Rotary position embedding', category: 'position',
    inputs: [{ id: 'q', tensor: 'query', rank: 4 }, { id: 'k', tensor: 'key', rank: 4 }],
    outputs: [{ id: 'q', tensor: 'query', rank: 4 }, { id: 'k', tensor: 'key', rank: 4 }],
    settings: [{ id: 'base', type: 'number', default: 10000 }],
    lowerings: lowering([], [
      '{{out:q}} = _labo_apply_rope({{in:q}}, {{base}})',
      '{{out:k}} = _labo_apply_rope({{in:k}}, {{base}})',
    ], [
      'def _labo_apply_rope(x, base):\n    sequence, dim = x.shape[-2], x.shape[-1]\n    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim))\n    angles = torch.outer(torch.arange(sequence, device=x.device, dtype=torch.float32), inv_freq)\n    cos = torch.repeat_interleave(angles.cos(), 2, dim=-1).to(dtype=x.dtype)\n    sin = torch.repeat_interleave(angles.sin(), 2, dim=-1).to(dtype=x.dtype)\n    rotated = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)\n    return x * cos + rotated * sin',
    ]),
  },
  'gqa-kv-expand': {
    id: 'gqa-kv-expand', label: 'GQA KV expansion', category: 'attention',
    inputs: [{ id: 'k', tensor: 'key', rank: 4 }, { id: 'v', tensor: 'value', rank: 4 }],
    outputs: [{ id: 'k', tensor: 'key', rank: 4 }, { id: 'v', tensor: 'value', rank: 4 }],
    settings: [],
    lowerings: lowering([], [
      '{{out:k}} = {{in:k}}.repeat_interleave({{queryHeads}} // {{keyValueHeads}}, dim=1)',
      '{{out:v}} = {{in:v}}.repeat_interleave({{queryHeads}} // {{keyValueHeads}}, dim=1)',
    ]),
  },
  'causal-sdpa': {
    id: 'causal-sdpa', label: 'Causal SDPA', category: 'attention',
    inputs: [
      { id: 'q', tensor: 'query', rank: 4 },
      { id: 'k', tensor: 'key', rank: 4 },
      { id: 'v', tensor: 'value', rank: 4 },
    ],
    outputs: [{ id: 'output', tensor: 'attention', rank: 4 }],
    settings: [{ id: 'dropout', type: 'number', default: 0 }],
    lowerings: lowering([], ['{{out:output}} = F.scaled_dot_product_attention({{in:q}}, {{in:k}}, {{in:v}}, dropout_p={{dropout}}, is_causal=True, enable_gqa={{in:q}}.size(1) != {{in:k}}.size(1))']),
  },
  'eager-causal-attention': {
    id: 'eager-causal-attention', label: 'Eager causal attention', category: 'attention',
    inputs: [
      { id: 'q', tensor: 'query', rank: 4 },
      { id: 'k', tensor: 'key', rank: 4 },
      { id: 'v', tensor: 'value', rank: 4 },
    ],
    outputs: [{ id: 'output', tensor: 'attention', rank: 4 }],
    settings: [{ id: 'dropout', type: 'number', default: 0 }],
    lowerings: lowering([], [
      '{{module}}_scores = torch.matmul({{in:q}}, {{in:k}}.transpose(-2, -1)) * ({{in:q}}.shape[-1] ** -0.5)',
      "{{module}}_mask = torch.ones({{module}}_scores.shape[-2:], device={{module}}_scores.device, dtype=torch.bool).triu(1)",
      "{{module}}_weights = torch.softmax({{module}}_scores.masked_fill({{module}}_mask, float('-inf')), dim=-1)",
      '{{module}}_weights = F.dropout({{module}}_weights, p={{dropout}}, training=self.training)',
      '{{out:output}} = torch.matmul({{module}}_weights, {{in:v}})',
    ]),
  },
  'merge-attention-heads': {
    id: 'merge-attention-heads', label: 'Merge attention heads', category: 'attention',
    inputs: [{ id: 'attention', tensor: 'attention', rank: 4 }], outputs: [hiddenOutput], settings: [],
    lowerings: lowering([], [
      '{{module}}_batch, _, {{module}}_sequence, _ = {{in:attention}}.shape',
      '{{out:output}} = {{in:attention}}.transpose(1, 2).contiguous().view({{module}}_batch, {{module}}_sequence, {{queryHeads}} * {{headDim}})',
    ]),
  },
  'attention-output-projection': {
    id: 'attention-output-projection', label: 'Attention output projection', category: 'attention',
    inputs: [hiddenInput], outputs: [hiddenOutput],
    settings: [{ id: 'bias', type: 'boolean', default: false }],
    lowerings: moduleLowering('nn.Linear({{queryHeads}} * {{headDim}}, {{hiddenSize}}, bias={{bias}})'),
  },
  'residual-add': {
    id: 'residual-add', label: 'Residual add', category: 'composition',
    inputs: [{ id: 'residual', tensor: 'hidden', rank: 3 }, { id: 'branch', tensor: 'hidden', rank: 3 }],
    outputs: [hiddenOutput], settings: [],
    lowerings: lowering([], ['{{out:output}} = {{in:residual}} + {{in:branch}}']),
  },
  'linear-projection': {
    id: 'linear-projection', label: 'Linear projection', category: 'composition',
    inputs: [hiddenInput], outputs: [hiddenOutput],
    settings: [{ id: 'bias', type: 'boolean', default: false }],
    lowerings: moduleLowering('nn.Linear({{hiddenSize}}, {{hiddenSize}}, bias={{bias}})'),
  },
  dropout: {
    id: 'dropout', label: 'Dropout', category: 'composition',
    inputs: [hiddenInput], outputs: [hiddenOutput],
    settings: [{ id: 'probability', type: 'number', default: 0.1 }],
    lowerings: moduleLowering('nn.Dropout(p={{probability}})'),
  },
  scale: {
    id: 'scale', label: 'Tensor scale', category: 'composition',
    inputs: [hiddenInput], outputs: [hiddenOutput],
    settings: [{ id: 'factor', type: 'number', default: 1 }],
    lowerings: lowering([], ['{{out:output}} = {{in:hidden}} * {{factor}}']),
  },
  'hadamard-product': {
    id: 'hadamard-product', label: 'Hadamard product', category: 'composition',
    inputs: [{ id: 'left', tensor: 'hidden', rank: 3 }, { id: 'right', tensor: 'hidden', rank: 3 }],
    outputs: [hiddenOutput], settings: [],
    lowerings: lowering([], ['{{out:output}} = {{in:left}} * {{in:right}}']),
  },
  identity: {
    id: 'identity', label: 'Identity / bypass', category: 'composition',
    inputs: [hiddenInput], outputs: [hiddenOutput], settings: [],
    lowerings: lowering([], ['{{out:output}} = {{in:hidden}}']),
  },
  'swiglu-mlp': {
    id: 'swiglu-mlp', label: 'SwiGLU MLP', category: 'mlp',
    inputs: [hiddenInput], outputs: [hiddenOutput],
    settings: [{ id: 'intermediateSize', type: 'number', default: 2048 }, { id: 'bias', type: 'boolean', default: false }],
    lowerings: lowering([
      'self.{{module}}_gate = nn.Linear({{hiddenSize}}, {{intermediateSize}}, bias={{bias}})',
      'self.{{module}}_up = nn.Linear({{hiddenSize}}, {{intermediateSize}}, bias={{bias}})',
      'self.{{module}}_down = nn.Linear({{intermediateSize}}, {{hiddenSize}}, bias={{bias}})',
    ], ['{{out:output}} = self.{{module}}_down(F.silu(self.{{module}}_gate({{in:hidden}})) * self.{{module}}_up({{in:hidden}}))']),
  },
  'gelu-mlp': {
    id: 'gelu-mlp', label: 'GELU MLP', category: 'mlp',
    inputs: [hiddenInput], outputs: [hiddenOutput],
    settings: [{ id: 'intermediateSize', type: 'number', default: 3072 }, { id: 'bias', type: 'boolean', default: true }],
    lowerings: lowering([
      'self.{{module}}_up = nn.Linear({{hiddenSize}}, {{intermediateSize}}, bias={{bias}})',
      'self.{{module}}_down = nn.Linear({{intermediateSize}}, {{hiddenSize}}, bias={{bias}})',
    ], ['{{out:output}} = self.{{module}}_down(F.gelu(self.{{module}}_up({{in:hidden}})))']),
  },
  'geglu-mlp': {
    id: 'geglu-mlp', label: 'GEGLU MLP', category: 'mlp',
    inputs: [hiddenInput], outputs: [hiddenOutput],
    settings: [{ id: 'intermediateSize', type: 'number', default: 2048 }, { id: 'bias', type: 'boolean', default: false }],
    lowerings: lowering([
      'self.{{module}}_gate = nn.Linear({{hiddenSize}}, {{intermediateSize}}, bias={{bias}})',
      'self.{{module}}_up = nn.Linear({{hiddenSize}}, {{intermediateSize}}, bias={{bias}})',
      'self.{{module}}_down = nn.Linear({{intermediateSize}}, {{hiddenSize}}, bias={{bias}})',
    ], ['{{out:output}} = self.{{module}}_down(F.gelu(self.{{module}}_gate({{in:hidden}})) * self.{{module}}_up({{in:hidden}}))']),
  },
  'reglu-mlp': {
    id: 'reglu-mlp', label: 'ReGLU MLP', category: 'mlp',
    inputs: [hiddenInput], outputs: [hiddenOutput],
    settings: [{ id: 'intermediateSize', type: 'number', default: 2048 }, { id: 'bias', type: 'boolean', default: false }],
    lowerings: lowering([
      'self.{{module}}_gate = nn.Linear({{hiddenSize}}, {{intermediateSize}}, bias={{bias}})',
      'self.{{module}}_up = nn.Linear({{hiddenSize}}, {{intermediateSize}}, bias={{bias}})',
      'self.{{module}}_down = nn.Linear({{intermediateSize}}, {{hiddenSize}}, bias={{bias}})',
    ], ['{{out:output}} = self.{{module}}_down(F.relu(self.{{module}}_gate({{in:hidden}})) * self.{{module}}_up({{in:hidden}}))']),
  },
  'relu-mlp': {
    id: 'relu-mlp', label: 'ReLU feed-forward MLP', category: 'mlp',
    inputs: [hiddenInput], outputs: [hiddenOutput],
    settings: [{ id: 'intermediateSize', type: 'number', default: 3072 }, { id: 'bias', type: 'boolean', default: true }],
    lowerings: lowering([
      'self.{{module}}_up = nn.Linear({{hiddenSize}}, {{intermediateSize}}, bias={{bias}})',
      'self.{{module}}_down = nn.Linear({{intermediateSize}}, {{hiddenSize}}, bias={{bias}})',
    ], ['{{out:output}} = self.{{module}}_down(F.relu(self.{{module}}_up({{in:hidden}})))']),
  },
  'moe-router': {
    id: 'moe-router', label: 'Learned Hidden-State Router', category: 'routing',
    inputs: [hiddenInput], outputs: [{ id: 'scores', tensor: 'routing-logits', rank: 3 }],
    settings: [
      { id: 'nExperts', type: 'number', default: 64 },
      { id: 'scoringFunction', type: 'select', default: 'sigmoid', options: ['softmax', 'sigmoid'] },
      { id: 'routerBias', type: 'boolean', default: true },
      { id: 'routerDtype', type: 'select', default: 'float32', options: ['float32', 'bfloat16'] },
    ],
    lowerings: lowering(
      ['self.{{module}} = nn.Linear({{hiddenSize}}, {{nExperts}}, bias={{routerBias}})'],
      [
        '{{module}}_logits = self.{{module}}({{in:hidden}}).float()',
        "{{out:scores}} = torch.softmax({{module}}_logits, dim=-1) if {{scoringFunction}} == 'softmax' else torch.sigmoid({{module}}_logits)",
      ],
    ),
  },
  'top-k-routing': {
    id: 'top-k-routing', label: 'Top-K Routing', category: 'routing',
    inputs: [{ id: 'scores', tensor: 'routing-logits', rank: 3 }],
    outputs: [
      { id: 'expertIndices', tensor: 'expert-indices', rank: 3 },
      { id: 'expertWeights', tensor: 'routing-weights', rank: 3 },
    ],
    settings: [
      { id: 'topK', type: 'number', default: 6 },
      { id: 'selectionMethod', type: 'select', default: 'group-limited-greedy', options: ['greedy', 'group-limited-greedy', 'aux-free'] },
      { id: 'nExpertGroups', type: 'number', default: 8 },
      { id: 'topkGroups', type: 'number', default: 4 },
      { id: 'normalizeWeights', type: 'boolean', default: true },
      { id: 'routedScalingFactor', type: 'number', default: 1 },
    ],
    lowerings: lowering([], [
      '{{out:expertWeights}}, {{out:expertIndices}} = _labo_topk_route({{in:scores}}, {{topK}}, {{normalizeWeights}}, {{routedScalingFactor}}, {{selectionMethod}}, {{nExpertGroups}}, {{topkGroups}})',
    ], [
      "def _labo_topk_route(scores, top_k, normalize, scale, method, n_groups, top_groups):\n    filtered = scores\n    if method == 'group-limited-greedy' and scores.shape[-1] % n_groups == 0:\n        grouped = scores.view(*scores.shape[:-1], n_groups, scores.shape[-1] // n_groups)\n        group_scores = grouped.max(dim=-1).values\n        keep_groups = group_scores.topk(min(top_groups, n_groups), dim=-1).indices\n        group_mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(-1, keep_groups, True)\n        filtered = grouped.masked_fill(~group_mask.unsqueeze(-1), float('-inf')).flatten(-2)\n    weights, indices = filtered.topk(min(top_k, filtered.shape[-1]), dim=-1)\n    if normalize:\n        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)\n    return weights * scale, indices",
    ]),
  },
  'deterministic-token-routing': {
    id: 'deterministic-token-routing', label: 'Fixed Token-ID Router', category: 'routing',
    inputs: [{ id: 'tokenIds', tensor: 'token-ids', rank: 2 }],
    outputs: [
      { id: 'expertIndices', tensor: 'expert-indices', rank: 3 },
      { id: 'expertWeights', tensor: 'routing-weights', rank: 3 },
    ],
    settings: [
      { id: 'vocabSize', type: 'number', default: 32000 },
      { id: 'nExperts', type: 'number', default: 4 },
      { id: 'topK', type: 'number', default: 2 },
      { id: 'layerIndex', type: 'number', default: 0 },
      { id: 'primaryWeight', type: 'number', default: 0.5 },
    ],
    lowerings: lowering([
      "self.register_buffer('{{module}}', _labo_fixed_token_routes({{vocabSize}}, {{nExperts}}, {{topK}}, {{layerIndex}}))",
    ], [
      '{{out:expertIndices}} = self.{{module}}[:, {{in:tokenIds}}].permute(1, 2, 0)',
      '{{out:expertWeights}} = _labo_fixed_route_weights({{out:expertIndices}}, {{primaryWeight}})',
    ], [
      "def _labo_fixed_token_routes(vocab_size, n_experts, top_k, layer_index):\n    generator = torch.Generator().manual_seed(0xC0DE + layer_index)\n    permutation = torch.randperm(n_experts, generator=generator)\n    primary = permutation[torch.arange(vocab_size) % n_experts]\n    routes = torch.empty(top_k, vocab_size, dtype=torch.long)\n    routes[0] = primary\n    for route_idx in range(1, top_k):\n        counts = torch.zeros(n_experts, dtype=torch.long)\n        for token_id in range(vocab_size):\n            blocked = set(int(routes[previous, token_id]) for previous in range(route_idx))\n            candidates = [expert for expert in range(n_experts) if expert not in blocked]\n            selected = min(candidates, key=lambda expert: (int(counts[expert]), expert))\n            routes[route_idx, token_id] = selected\n            counts[selected] += 1\n    return routes\n\ndef _labo_fixed_route_weights(indices, primary_weight):\n    top_k = indices.shape[-1]\n    if top_k == 1:\n        return torch.ones_like(indices, dtype=torch.float32)\n    secondary = (1.0 - primary_weight) / (top_k - 1)\n    weights = torch.full_like(indices, secondary, dtype=torch.float32)\n    weights[..., 0] = primary_weight\n    return weights",
    ]),
  },
  'routed-expert-bank': {
    id: 'routed-expert-bank', label: 'Routed Residual Experts', category: 'routing',
    inputs: [hiddenInput, { id: 'expertIndices', tensor: 'expert-indices', rank: 3 }, { id: 'expertWeights', tensor: 'routing-weights', rank: 3 }],
    outputs: [hiddenOutput],
    settings: [
      { id: 'nExperts', type: 'number', default: 64 },
      { id: 'intermediateSize', type: 'number', default: 2048 },
      { id: 'activation', type: 'select', default: 'swiglu', options: ['swiglu', 'geglu', 'reglu'] },
      { id: 'expertParallelSize', type: 'number', default: 1 },
    ],
    lowerings: lowering([
      "self.{{module}} = nn.ModuleList([nn.ModuleDict({'gate': nn.Linear({{hiddenSize}}, {{intermediateSize}}, bias=False), 'up': nn.Linear({{hiddenSize}}, {{intermediateSize}}, bias=False), 'down': nn.Linear({{intermediateSize}}, {{hiddenSize}}, bias=False)}) for _ in range({{nExperts}})])",
    ], [
      '{{out:output}} = _labo_routed_experts({{in:hidden}}, {{in:expertIndices}}, {{in:expertWeights}}, self.{{module}})',
    ], [
      "def _labo_routed_experts(hidden, indices, weights, experts):\n    output = torch.zeros_like(hidden)\n    for expert_id, expert in enumerate(experts):\n        mask = indices == expert_id\n        token_mask = mask.any(dim=-1)\n        if token_mask.any():\n            selected = hidden[token_mask]\n            selected_weights = (weights[token_mask] * mask[token_mask]).sum(dim=-1, keepdim=True).to(hidden.dtype)\n            expert_output = expert['down'](F.silu(expert['gate'](selected)) * expert['up'](selected))\n            output[token_mask] += expert_output * selected_weights\n    return output",
    ]),
  },
  'shared-expert-bank': {
    id: 'shared-expert-bank', label: 'Shared Dense Expert', category: 'routing',
    inputs: [hiddenInput], outputs: [hiddenOutput],
    settings: [
      { id: 'nSharedExperts', type: 'number', default: 2 },
      { id: 'intermediateSize', type: 'number', default: 2048 },
      { id: 'activation', type: 'select', default: 'swiglu', options: ['swiglu', 'geglu', 'reglu'] },
    ],
    lowerings: lowering([
      "self.{{module}} = nn.ModuleList([nn.ModuleDict({'gate': nn.Linear({{hiddenSize}}, {{intermediateSize}}, bias=False), 'up': nn.Linear({{hiddenSize}}, {{intermediateSize}}, bias=False), 'down': nn.Linear({{intermediateSize}}, {{hiddenSize}}, bias=False)}) for _ in range({{nSharedExperts}})])",
    ], ["{{out:output}} = sum(expert['down'](F.silu(expert['gate']({{in:hidden}})) * expert['up']({{in:hidden}})) for expert in self.{{module}})"]),
  },
  'expert-merge': {
    id: 'expert-merge', label: 'Expert Merge', category: 'routing',
    inputs: [{ id: 'routed', tensor: 'hidden', rank: 3 }, { id: 'shared', tensor: 'hidden', rank: 3 }],
    outputs: [hiddenOutput],
    settings: [{ id: 'merge', type: 'select', default: 'sum', options: ['sum', 'gated-sum'] }],
    lowerings: lowering(
      ['self.{{module}}_gate = nn.Parameter(torch.zeros(()))'],
      ["{{out:output}} = torch.sigmoid(self.{{module}}_gate) * {{in:routed}} + (1.0 - torch.sigmoid(self.{{module}}_gate)) * {{in:shared}} if {{merge}} == 'gated-sum' else {{in:routed}} + {{in:shared}}"],
    ),
  },
  'branch-gated-merge': {
    id: 'branch-gated-merge', label: 'Shared + Routed Merge', category: 'routing',
    inputs: [{ id: 'shared', tensor: 'hidden', rank: 3 }, { id: 'routed', tensor: 'hidden', rank: 3 }],
    outputs: [hiddenOutput],
    settings: [
      { id: 'sharedGateInit', type: 'number', default: 1 },
      { id: 'routedGateInit', type: 'number', default: 0.1 },
    ],
    lowerings: lowering([
      'self.{{module}}_shared_gate = nn.Parameter(torch.tensor({{sharedGateInit}}, dtype=torch.float32))',
      'self.{{module}}_routed_gate = nn.Parameter(torch.tensor({{routedGateInit}}, dtype=torch.float32))',
    ], [
      '{{out:output}} = self.{{module}}_shared_gate * {{in:shared}} + self.{{module}}_routed_gate * {{in:routed}}',
    ]),
  },
  'load-balancing-loss': {
    id: 'load-balancing-loss', label: 'Load-Balancing Loss', category: 'objective',
    inputs: [{ id: 'scores', tensor: 'routing-logits', rank: 3 }],
    outputs: [{ id: 'loss', tensor: 'scalar', rank: 0 }],
    settings: [{ id: 'coefficient', type: 'number', default: 0.001 }],
    lowerings: lowering([], ['{{out:loss}} = {{coefficient}} * ({{in:scores}}.mean(dim=(0, 1)) * {{in:scores}}.shape[-1]).pow(2).mean()']),
  },
  'deepseek-moe': {
    id: 'deepseek-moe', label: 'DeepSeek-style MoE', category: 'routing',
    inputs: [hiddenInput], outputs: [hiddenOutput, { id: 'routerScores', tensor: 'routing-logits', rank: 3 }],
    settings: [
      { id: 'nRoutedExperts', type: 'number', default: 64 },
      { id: 'nSharedExperts', type: 'number', default: 2 },
      { id: 'topK', type: 'number', default: 6 },
      { id: 'intermediateSize', type: 'number', default: 2048 },
      { id: 'scoringFunction', type: 'select', default: 'sigmoid', options: ['softmax', 'sigmoid'] },
      { id: 'selectionMethod', type: 'select', default: 'group-limited-greedy', options: ['greedy', 'group-limited-greedy', 'aux-free'] },
      { id: 'nExpertGroups', type: 'number', default: 8 },
      { id: 'topkGroups', type: 'number', default: 4 },
      { id: 'normalizeWeights', type: 'boolean', default: true },
      { id: 'routedScalingFactor', type: 'number', default: 1 },
      { id: 'expertParallelSize', type: 'number', default: 1 },
    ],
    lowerings: lowering([], [
      "raise RuntimeError('DeepSeek-style MoE is a composite recipe; expand it into Router, Top-K, Expert Banks and Merge before compilation')",
    ]),
    composite: { atomIds: ['moe-router', 'top-k-routing', 'routed-expert-bank', 'shared-expert-bank', 'expert-merge'] },
  },
  'lm-head': {
    id: 'lm-head', label: 'Language-model head', category: 'output',
    inputs: [hiddenInput], outputs: [{ id: 'logits', tensor: 'logits', rank: 3 }],
    settings: [
      { id: 'vocabSize', type: 'number', default: 32000 },
      { id: 'tieEmbeddingWeights', type: 'boolean', default: true },
      { id: 'bias', type: 'boolean', default: false },
    ],
    lowerings: lowering(
      ['self.{{module}} = nn.Linear({{hiddenSize}}, {{vocabSize}}, bias={{bias}})'],
      ['{{out:logits}} = self.{{module}}({{in:hidden}})'],
    ),
  },
  'greedy-token-decoder': {
    id: 'greedy-token-decoder', label: 'Greedy token decoder', category: 'output',
    inputs: [{ id: 'logits', tensor: 'logits', rank: 3 }],
    outputs: [{ id: 'tokenIds', tensor: 'token-ids', rank: 2 }],
    settings: [],
    lowerings: lowering([], ['{{out:tokenIds}} = torch.argmax({{in:logits}}, dim=-1)']),
  },
  'top-k-token-sampler': {
    id: 'top-k-token-sampler', label: 'Top-k token sampler', category: 'output',
    inputs: [{ id: 'logits', tensor: 'logits', rank: 3 }],
    outputs: [{ id: 'tokenIds', tensor: 'token-ids', rank: 2 }],
    settings: [{ id: 'topK', type: 'number', default: 50 }, { id: 'temperature', type: 'number', default: 1 }],
    lowerings: lowering([], ['{{out:tokenIds}} = torch.topk({{in:logits}} / max({{temperature}}, 1e-6), k=min({{topK}}, {{in:logits}}.shape[-1]), dim=-1).indices[..., 0]']),
  },
  'multinomial-token-sampler': {
    id: 'multinomial-token-sampler', label: 'Multinomial token sampler', category: 'output',
    inputs: [{ id: 'logits', tensor: 'logits', rank: 3 }],
    outputs: [{ id: 'tokenIds', tensor: 'token-ids', rank: 2 }],
    settings: [{ id: 'temperature', type: 'number', default: 1 }],
    lowerings: lowering([], [
      '{{module}}_probabilities = F.softmax({{in:logits}} / max({{temperature}}, 1e-6), dim=-1)',
      '{{out:tokenIds}} = torch.multinomial({{module}}_probabilities.reshape(-1, {{module}}_probabilities.shape[-1]), num_samples=1).reshape({{module}}_probabilities.shape[:-1])',
    ]),
  },
  'log-softmax': {
    id: 'log-softmax', label: 'LogSoftmax output', category: 'output',
    inputs: [{ id: 'logits', tensor: 'logits', rank: 3 }],
    outputs: [{ id: 'logits', tensor: 'logits', rank: 3 }],
    settings: [{ id: 'dimension', type: 'number', default: -1 }],
    lowerings: lowering([], ['{{out:logits}} = F.log_softmax({{in:logits}}, dim={{dimension}})']),
  },
  'cross-entropy-loss': {
    id: 'cross-entropy-loss', label: 'Cross-entropy loss', category: 'objective',
    inputs: [{ id: 'logits', tensor: 'logits', rank: 3 }, { id: 'labels', tensor: 'labels', rank: 2 }],
    outputs: [{ id: 'loss', tensor: 'scalar', rank: 0 }],
    settings: [{ id: 'ignoreIndex', type: 'number', default: -100 }],
    lowerings: lowering([], ['{{out:loss}} = F.cross_entropy({{in:logits}}.flatten(0, 1), {{in:labels}}.flatten(), ignore_index={{ignoreIndex}})']),
  },
  'label-smoothed-cross-entropy': {
    id: 'label-smoothed-cross-entropy', label: 'Label-smoothed cross-entropy', category: 'objective',
    inputs: [{ id: 'logits', tensor: 'logits', rank: 3 }, { id: 'labels', tensor: 'labels', rank: 2 }],
    outputs: [{ id: 'loss', tensor: 'scalar', rank: 0 }],
    settings: [
      { id: 'smoothing', type: 'number', default: 0.1 },
      { id: 'ignoreIndex', type: 'number', default: -100 },
    ],
    lowerings: lowering([], ['{{out:loss}} = F.cross_entropy({{in:logits}}.flatten(0, 1), {{in:labels}}.flatten(), ignore_index={{ignoreIndex}}, label_smoothing={{smoothing}})']),
  },
  'router-entropy-loss': {
    id: 'router-entropy-loss', label: 'Router entropy loss', category: 'objective',
    inputs: [{ id: 'scores', tensor: 'routing-logits', rank: 3 }],
    outputs: [{ id: 'loss', tensor: 'scalar', rank: 0 }],
    settings: [{ id: 'coefficient', type: 'number', default: 0.001 }],
    lowerings: lowering([], [
      '{{module}}_probabilities = {{in:scores}} / {{in:scores}}.sum(dim=-1, keepdim=True).clamp_min(1e-9)',
      '{{out:loss}} = -{{coefficient}} * ({{module}}_probabilities * {{module}}_probabilities.clamp_min(1e-9).log()).sum(dim=-1).mean()',
    ]),
  },
}
