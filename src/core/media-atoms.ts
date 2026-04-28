import type { AtomPort, AtomSetting, ModelAtomDefinition } from './model-atoms'

const hiddenInput: AtomPort = { id: 'hidden', tensor: 'hidden', rank: 3 }
const hiddenOutput: AtomPort = { id: 'output', tensor: 'hidden', rank: 3 }

function lowering(
  declarations: string[],
  forward: string[],
  helpers: string[] = [],
): ModelAtomDefinition['lowerings'] {
  return { pytorch: { executable: true, declarations, forward, ...(helpers.length > 0 ? { helpers } : {}) } }
}

function unaryMediaAtom(
  id: string,
  label: string,
  expression: string,
  settings: AtomSetting[] = [],
  declarations: string[] = [],
): ModelAtomDefinition {
  return {
    id,
    label,
    category: 'media',
    inputs: [hiddenInput],
    outputs: [hiddenOutput],
    settings,
    lowerings: lowering(declarations, [`{{out:output}} = ${expression}`]),
  }
}

function linearMediaAtom(id: string, label: string, bias = true): ModelAtomDefinition {
  return {
    id,
    label,
    category: 'media',
    inputs: [hiddenInput],
    outputs: [hiddenOutput],
    settings: [{ id: 'bias', type: 'boolean', default: bias }],
    lowerings: lowering(
      ['self.{{module}} = nn.Linear({{hiddenSize}}, {{hiddenSize}}, bias={{bias}})'],
      ['{{out:output}} = self.{{module}}({{in:hidden}})'],
    ),
  }
}

const twoHiddenInputs: AtomPort[] = [
  { id: 'left', tensor: 'hidden', rank: 3 },
  { id: 'right', tensor: 'hidden', rank: 3 },
]

/**
 * Executable image, video, audio and multimodal atoms.
 *
 * LABO represents media as rank-3 token sequences [batch, tokens, channels].
 * Raw pixels/frames are patchified before entering the graph; every atom below
 * therefore composes with the same typed elastic system as language tokens.
 */
export const mediaAtomRegistry: Record<string, ModelAtomDefinition> = {
  // Raw media preprocessing and tokenization.
  'image-channel-normalization': {
    id: 'image-channel-normalization',
    label: 'Image channel normalization',
    category: 'media',
    inputs: [{ id: 'image', tensor: 'image', rank: 4 }],
    outputs: [{ id: 'image', tensor: 'image', rank: 4 }],
    settings: [{ id: 'mean', type: 'number', default: 0.5 }, { id: 'standardDeviation', type: 'number', default: 0.5 }],
    lowerings: lowering([], ['{{out:image}} = ({{in:image}} - {{mean}}) / max({{standardDeviation}}, 1e-6)']),
  },
  'image-resize': {
    id: 'image-resize',
    label: 'Image bilinear resize',
    category: 'media',
    inputs: [{ id: 'image', tensor: 'image', rank: 4 }],
    outputs: [{ id: 'image', tensor: 'image', rank: 4 }],
    settings: [{ id: 'height', type: 'number', default: 224 }, { id: 'width', type: 'number', default: 224 }],
    lowerings: lowering([], ["{{out:image}} = F.interpolate({{in:image}}, size=(int({{height}}), int({{width}})), mode='bilinear', align_corners=False)"]),
  },
  'image-patch-embedding': {
    id: 'image-patch-embedding',
    label: 'Image patch embedding',
    category: 'media',
    inputs: [{ id: 'image', tensor: 'image', rank: 4 }],
    outputs: [hiddenOutput],
    settings: [{ id: 'inputChannels', type: 'number', default: 3 }, { id: 'patchSize', type: 'number', default: 16 }, { id: 'bias', type: 'boolean', default: true }],
    lowerings: lowering(
      ['self.{{module}} = nn.Conv2d({{inputChannels}}, {{hiddenSize}}, kernel_size={{patchSize}}, stride={{patchSize}}, bias={{bias}})'],
      ['{{module}}_patches = self.{{module}}({{in:image}})', '{{out:output}} = {{module}}_patches.flatten(2).transpose(1, 2)'],
    ),
  },
  'global-image-embedding': {
    id: 'global-image-embedding',
    label: 'Global image embedding',
    category: 'media',
    inputs: [{ id: 'image', tensor: 'image', rank: 4 }],
    outputs: [hiddenOutput],
    settings: [{ id: 'inputChannels', type: 'number', default: 3 }, { id: 'patchSize', type: 'number', default: 16 }, { id: 'bias', type: 'boolean', default: true }],
    lowerings: lowering(
      ['self.{{module}} = nn.Conv2d({{inputChannels}}, {{hiddenSize}}, kernel_size={{patchSize}}, stride={{patchSize}}, bias={{bias}})'],
      ['{{out:output}} = self.{{module}}({{in:image}}).flatten(2).mean(dim=-1).unsqueeze(1)'],
    ),
  },
  'image-vq-tokenizer': {
    id: 'image-vq-tokenizer',
    label: 'Image VQ tokenizer',
    category: 'media',
    inputs: [{ id: 'image', tensor: 'image', rank: 4 }],
    outputs: [{ id: 'tokenIds', tensor: 'token-ids', rank: 2 }],
    settings: [{ id: 'inputChannels', type: 'number', default: 3 }, { id: 'patchSize', type: 'number', default: 16 }, { id: 'codebookSize', type: 'number', default: 1024 }, { id: 'initialScale', type: 'number', default: 0.02 }],
    lowerings: lowering(
      ['self.{{module}}_encoder = nn.Conv2d({{inputChannels}}, {{hiddenSize}}, kernel_size={{patchSize}}, stride={{patchSize}})', 'self.{{module}}_codebook = nn.Parameter(torch.randn({{codebookSize}}, {{hiddenSize}}) * {{initialScale}})'],
      [
        '{{module}}_latents = self.{{module}}_encoder({{in:image}}).flatten(2).transpose(1, 2)',
        '{{module}}_distances = {{module}}_latents.pow(2).sum(dim=-1, keepdim=True) + self.{{module}}_codebook.pow(2).sum(dim=-1) - 2 * torch.matmul({{module}}_latents, self.{{module}}_codebook.t())',
        '{{out:tokenIds}} = {{module}}_distances.argmin(dim=-1)',
      ],
    ),
  },
  'image-codebook-embedding': {
    id: 'image-codebook-embedding',
    label: 'Image codebook embedding',
    category: 'media',
    inputs: [{ id: 'tokenIds', tensor: 'token-ids', rank: 2 }],
    outputs: [hiddenOutput],
    settings: [{ id: 'codebookSize', type: 'number', default: 1024 }],
    lowerings: lowering(
      ['self.{{module}} = nn.Embedding({{codebookSize}}, {{hiddenSize}})'],
      ['{{out:output}} = self.{{module}}({{in:tokenIds}})'],
    ),
  },
  'image-token-decoder': {
    id: 'image-token-decoder',
    label: 'Image token decoder',
    category: 'media',
    inputs: [hiddenInput],
    outputs: [{ id: 'image', tensor: 'image', rank: 4 }],
    settings: [{ id: 'outputChannels', type: 'number', default: 3 }, { id: 'patchSize', type: 'number', default: 16 }, { id: 'bias', type: 'boolean', default: true }],
    lowerings: lowering(
      ['self.{{module}} = nn.ConvTranspose2d({{hiddenSize}}, {{outputChannels}}, kernel_size={{patchSize}}, stride={{patchSize}}, bias={{bias}})'],
      [
        '{{module}}_grid = max(1, int({{in:hidden}}.shape[1] ** 0.5))',
        '{{module}}_tokens = {{in:hidden}}[:, :{{module}}_grid * {{module}}_grid].transpose(1, 2).reshape({{in:hidden}}.shape[0], {{hiddenSize}}, {{module}}_grid, {{module}}_grid)',
        '{{out:image}} = self.{{module}}({{module}}_tokens)',
      ],
    ),
  },
  'video-channel-normalization': {
    id: 'video-channel-normalization',
    label: 'Video channel normalization',
    category: 'media',
    inputs: [{ id: 'video', tensor: 'video', rank: 5 }],
    outputs: [{ id: 'video', tensor: 'video', rank: 5 }],
    settings: [{ id: 'mean', type: 'number', default: 0.5 }, { id: 'standardDeviation', type: 'number', default: 0.5 }],
    lowerings: lowering([], ['{{out:video}} = ({{in:video}} - {{mean}}) / max({{standardDeviation}}, 1e-6)']),
  },
  'video-spatial-resize': {
    id: 'video-spatial-resize',
    label: 'Video spatial resize',
    category: 'media',
    inputs: [{ id: 'video', tensor: 'video', rank: 5 }],
    outputs: [{ id: 'video', tensor: 'video', rank: 5 }],
    settings: [{ id: 'height', type: 'number', default: 224 }, { id: 'width', type: 'number', default: 224 }],
    lowerings: lowering([], [
      '{{module}}_batch, {{module}}_channels, {{module}}_frames, {{module}}_height, {{module}}_width = {{in:video}}.shape',
      '{{module}}_flat = {{in:video}}.permute(0, 2, 1, 3, 4).reshape({{module}}_batch * {{module}}_frames, {{module}}_channels, {{module}}_height, {{module}}_width)',
      "{{module}}_resized = F.interpolate({{module}}_flat, size=(int({{height}}), int({{width}})), mode='bilinear', align_corners=False)",
      '{{out:video}} = {{module}}_resized.reshape({{module}}_batch, {{module}}_frames, {{module}}_channels, int({{height}}), int({{width}})).permute(0, 2, 1, 3, 4)',
    ]),
  },
  'video-tubelet-embedding': {
    id: 'video-tubelet-embedding',
    label: 'Video tubelet embedding',
    category: 'media',
    inputs: [{ id: 'video', tensor: 'video', rank: 5 }],
    outputs: [hiddenOutput],
    settings: [{ id: 'inputChannels', type: 'number', default: 3 }, { id: 'tubeletSize', type: 'number', default: 2 }, { id: 'patchSize', type: 'number', default: 16 }, { id: 'bias', type: 'boolean', default: true }],
    lowerings: lowering(
      ['self.{{module}} = nn.Conv3d({{inputChannels}}, {{hiddenSize}}, kernel_size=({{tubeletSize}}, {{patchSize}}, {{patchSize}}), stride=({{tubeletSize}}, {{patchSize}}, {{patchSize}}), bias={{bias}})'],
      ['{{module}}_tubelets = self.{{module}}({{in:video}})', '{{out:output}} = {{module}}_tubelets.flatten(2).transpose(1, 2)'],
    ),
  },
  'video-vq-tokenizer': {
    id: 'video-vq-tokenizer',
    label: 'Video VQ tokenizer',
    category: 'media',
    inputs: [{ id: 'video', tensor: 'video', rank: 5 }],
    outputs: [{ id: 'tokenIds', tensor: 'token-ids', rank: 2 }],
    settings: [{ id: 'inputChannels', type: 'number', default: 3 }, { id: 'tubeletSize', type: 'number', default: 2 }, { id: 'patchSize', type: 'number', default: 16 }, { id: 'codebookSize', type: 'number', default: 1024 }, { id: 'initialScale', type: 'number', default: 0.02 }],
    lowerings: lowering(
      ['self.{{module}}_encoder = nn.Conv3d({{inputChannels}}, {{hiddenSize}}, kernel_size=({{tubeletSize}}, {{patchSize}}, {{patchSize}}), stride=({{tubeletSize}}, {{patchSize}}, {{patchSize}}))', 'self.{{module}}_codebook = nn.Parameter(torch.randn({{codebookSize}}, {{hiddenSize}}) * {{initialScale}})'],
      [
        '{{module}}_latents = self.{{module}}_encoder({{in:video}}).flatten(2).transpose(1, 2)',
        '{{module}}_distances = {{module}}_latents.pow(2).sum(dim=-1, keepdim=True) + self.{{module}}_codebook.pow(2).sum(dim=-1) - 2 * torch.matmul({{module}}_latents, self.{{module}}_codebook.t())',
        '{{out:tokenIds}} = {{module}}_distances.argmin(dim=-1)',
      ],
    ),
  },
  'video-codebook-embedding': {
    id: 'video-codebook-embedding',
    label: 'Video codebook embedding',
    category: 'media',
    inputs: [{ id: 'tokenIds', tensor: 'token-ids', rank: 2 }],
    outputs: [hiddenOutput],
    settings: [{ id: 'codebookSize', type: 'number', default: 1024 }],
    lowerings: lowering(
      ['self.{{module}} = nn.Embedding({{codebookSize}}, {{hiddenSize}})'],
      ['{{out:output}} = self.{{module}}({{in:tokenIds}})'],
    ),
  },
  'video-token-decoder': {
    id: 'video-token-decoder',
    label: 'Video token decoder',
    category: 'media',
    inputs: [hiddenInput],
    outputs: [{ id: 'video', tensor: 'video', rank: 5 }],
    settings: [{ id: 'outputChannels', type: 'number', default: 3 }, { id: 'tubeletSize', type: 'number', default: 2 }, { id: 'patchSize', type: 'number', default: 16 }, { id: 'temporalTokens', type: 'number', default: 1 }, { id: 'bias', type: 'boolean', default: true }],
    lowerings: lowering(
      ['self.{{module}} = nn.ConvTranspose3d({{hiddenSize}}, {{outputChannels}}, kernel_size=({{tubeletSize}}, {{patchSize}}, {{patchSize}}), stride=({{tubeletSize}}, {{patchSize}}, {{patchSize}}), bias={{bias}})'],
      [
        '{{module}}_time = max(1, min(int({{temporalTokens}}), {{in:hidden}}.shape[1]))',
        '{{module}}_grid = max(1, int(({{in:hidden}}.shape[1] / {{module}}_time) ** 0.5))',
        '{{module}}_count = {{module}}_time * {{module}}_grid * {{module}}_grid',
        '{{module}}_tokens = {{in:hidden}}[:, :{{module}}_count].transpose(1, 2).reshape({{in:hidden}}.shape[0], {{hiddenSize}}, {{module}}_time, {{module}}_grid, {{module}}_grid)',
        '{{out:video}} = self.{{module}}({{module}}_tokens)',
      ],
    ),
  },
  'audio-waveform-normalization': {
    id: 'audio-waveform-normalization',
    label: 'Audio waveform normalization',
    category: 'media',
    inputs: [{ id: 'audio', tensor: 'audio', rank: 3 }],
    outputs: [{ id: 'audio', tensor: 'audio', rank: 3 }],
    settings: [{ id: 'epsilon', type: 'number', default: 1e-6 }],
    lowerings: lowering([], [
      '{{module}}_centered = {{in:audio}} - {{in:audio}}.mean(dim=-1, keepdim=True)',
      '{{out:audio}} = {{module}}_centered / {{module}}_centered.abs().amax(dim=-1, keepdim=True).clamp_min({{epsilon}})',
    ]),
  },
  'audio-preemphasis': {
    id: 'audio-preemphasis',
    label: 'Audio pre-emphasis',
    category: 'media',
    inputs: [{ id: 'audio', tensor: 'audio', rank: 3 }],
    outputs: [{ id: 'audio', tensor: 'audio', rank: 3 }],
    settings: [{ id: 'coefficient', type: 'number', default: 0.97 }],
    lowerings: lowering([], [
      '{{out:audio}} = torch.cat(({{in:audio}}[..., :1], {{in:audio}}[..., 1:] - {{coefficient}} * {{in:audio}}[..., :-1]), dim=-1)',
    ]),
  },
  'audio-resample': {
    id: 'audio-resample',
    label: 'Audio linear resample',
    category: 'media',
    inputs: [{ id: 'audio', tensor: 'audio', rank: 3 }],
    outputs: [{ id: 'audio', tensor: 'audio', rank: 3 }],
    settings: [{ id: 'sourceRate', type: 'number', default: 48000 }, { id: 'targetRate', type: 'number', default: 16000 }],
    lowerings: lowering([], [
      '{{module}}_length = max(1, int({{in:audio}}.shape[-1] * {{targetRate}} / max({{sourceRate}}, 1)))',
      "{{out:audio}} = F.interpolate({{in:audio}}, size={{module}}_length, mode='linear', align_corners=False)",
    ]),
  },
  'audio-frame-embedding': {
    id: 'audio-frame-embedding',
    label: 'Audio frame embedding',
    category: 'media',
    inputs: [{ id: 'audio', tensor: 'audio', rank: 3 }],
    outputs: [hiddenOutput],
    settings: [{ id: 'inputChannels', type: 'number', default: 1 }, { id: 'frameSize', type: 'number', default: 400 }, { id: 'hopSize', type: 'number', default: 160 }, { id: 'bias', type: 'boolean', default: true }],
    lowerings: lowering(
      ['self.{{module}} = nn.Conv1d({{inputChannels}}, {{hiddenSize}}, kernel_size={{frameSize}}, stride={{hopSize}}, bias={{bias}})'],
      ['{{out:output}} = self.{{module}}({{in:audio}}).transpose(1, 2)'],
    ),
  },
  'audio-vq-tokenizer': {
    id: 'audio-vq-tokenizer',
    label: 'Audio VQ tokenizer',
    category: 'media',
    inputs: [{ id: 'audio', tensor: 'audio', rank: 3 }],
    outputs: [{ id: 'tokenIds', tensor: 'token-ids', rank: 2 }],
    settings: [{ id: 'inputChannels', type: 'number', default: 1 }, { id: 'frameSize', type: 'number', default: 400 }, { id: 'hopSize', type: 'number', default: 160 }, { id: 'codebookSize', type: 'number', default: 1024 }, { id: 'initialScale', type: 'number', default: 0.02 }],
    lowerings: lowering(
      ['self.{{module}}_encoder = nn.Conv1d({{inputChannels}}, {{hiddenSize}}, kernel_size={{frameSize}}, stride={{hopSize}})', 'self.{{module}}_codebook = nn.Parameter(torch.randn({{codebookSize}}, {{hiddenSize}}) * {{initialScale}})'],
      [
        '{{module}}_latents = self.{{module}}_encoder({{in:audio}}).transpose(1, 2)',
        '{{module}}_distances = {{module}}_latents.pow(2).sum(dim=-1, keepdim=True) + self.{{module}}_codebook.pow(2).sum(dim=-1) - 2 * torch.matmul({{module}}_latents, self.{{module}}_codebook.t())',
        '{{out:tokenIds}} = {{module}}_distances.argmin(dim=-1)',
      ],
    ),
  },
  'audio-codebook-embedding': {
    id: 'audio-codebook-embedding',
    label: 'Audio codebook embedding',
    category: 'media',
    inputs: [{ id: 'tokenIds', tensor: 'token-ids', rank: 2 }],
    outputs: [hiddenOutput],
    settings: [{ id: 'codebookSize', type: 'number', default: 1024 }],
    lowerings: lowering(
      ['self.{{module}} = nn.Embedding({{codebookSize}}, {{hiddenSize}})'],
      ['{{out:output}} = self.{{module}}({{in:tokenIds}})'],
    ),
  },
  'audio-token-decoder': {
    id: 'audio-token-decoder',
    label: 'Audio token decoder',
    category: 'media',
    inputs: [hiddenInput],
    outputs: [{ id: 'audio', tensor: 'audio', rank: 3 }],
    settings: [{ id: 'outputChannels', type: 'number', default: 1 }, { id: 'frameSize', type: 'number', default: 400 }, { id: 'hopSize', type: 'number', default: 160 }, { id: 'bias', type: 'boolean', default: true }],
    lowerings: lowering(
      ['self.{{module}} = nn.ConvTranspose1d({{hiddenSize}}, {{outputChannels}}, kernel_size={{frameSize}}, stride={{hopSize}}, bias={{bias}})'],
      ['{{out:audio}} = torch.tanh(self.{{module}}({{in:hidden}}.transpose(1, 2)))'],
    ),
  },
  'audio-feature-projector': linearMediaAtom('audio-feature-projector', 'Audio feature projector'),
  'audio-temporal-convolution': {
    id: 'audio-temporal-convolution',
    label: 'Audio temporal convolution',
    category: 'media',
    inputs: [hiddenInput],
    outputs: [hiddenOutput],
    settings: [{ id: 'kernelSize', type: 'number', default: 5 }],
    lowerings: lowering(
      ['self.{{module}} = nn.Conv1d({{hiddenSize}}, {{hiddenSize}}, kernel_size={{kernelSize}}, padding={{kernelSize}} // 2, groups={{hiddenSize}})'],
      ['{{out:output}} = self.{{module}}({{in:hidden}}.transpose(1, 2)).transpose(1, 2)'],
    ),
  },
  'audio-position-embedding': {
    id: 'audio-position-embedding',
    label: 'Learned audio positions',
    category: 'media',
    inputs: [hiddenInput],
    outputs: [hiddenOutput],
    settings: [{ id: 'maxFrames', type: 'number', default: 2048 }, { id: 'initialScale', type: 'number', default: 0.02 }],
    lowerings: lowering(
      ['self.{{module}} = nn.Parameter(torch.randn(1, {{maxFrames}}, {{hiddenSize}}) * {{initialScale}})'],
      ['{{out:output}} = {{in:hidden}} + self.{{module}}[:, :{{in:hidden}}.shape[1]]'],
    ),
  },
  'audio-mean-pooling': unaryMediaAtom('audio-mean-pooling', 'Audio mean pooling', '{{in:hidden}}.mean(dim=1, keepdim=True)'),
  'audio-ctc-head': {
    id: 'audio-ctc-head',
    label: 'Audio CTC vocabulary head',
    category: 'media',
    inputs: [hiddenInput],
    outputs: [{ id: 'logits', tensor: 'logits', rank: 3 }],
    settings: [{ id: 'vocabSize', type: 'number', default: 1024 }, { id: 'bias', type: 'boolean', default: true }],
    lowerings: lowering(
      ['self.{{module}} = nn.Linear({{hiddenSize}}, {{vocabSize}}, bias={{bias}})'],
      ['{{out:logits}} = self.{{module}}({{in:hidden}})'],
    ),
  },

  // Image / vision tokenization and spatial processing.
  'vision-patch-projection': linearMediaAtom('vision-patch-projection', 'Vision patch projection'),
  'vision-feature-projector': linearMediaAtom('vision-feature-projector', 'Vision feature projector', false),
  'image-latent-encoder': linearMediaAtom('image-latent-encoder', 'Image latent encoder'),
  'image-latent-decoder': linearMediaAtom('image-latent-decoder', 'Image latent decoder'),
  'image-reconstruction-head': {
    id: 'image-reconstruction-head',
    label: 'Image patch reconstruction head',
    category: 'media',
    inputs: [hiddenInput],
    outputs: [hiddenOutput],
    settings: [{ id: 'patchFeatures', type: 'number', default: 768 }, { id: 'bias', type: 'boolean', default: true }],
    lowerings: lowering(
      ['self.{{module}} = nn.Linear({{hiddenSize}}, {{patchFeatures}}, bias={{bias}})'],
      ['{{out:output}} = self.{{module}}({{in:hidden}})'],
    ),
  },
  'patch-layer-normalization': {
    id: 'patch-layer-normalization',
    label: 'Patch LayerNorm',
    category: 'media',
    inputs: [hiddenInput],
    outputs: [hiddenOutput],
    settings: [{ id: 'epsilon', type: 'number', default: 1e-5 }],
    lowerings: lowering(
      ['self.{{module}} = nn.LayerNorm({{hiddenSize}}, eps={{epsilon}})'],
      ['{{out:output}} = self.{{module}}({{in:hidden}})'],
    ),
  },
  'spatial-position-embedding': {
    id: 'spatial-position-embedding',
    label: 'Learned spatial positions',
    category: 'media',
    inputs: [hiddenInput],
    outputs: [hiddenOutput],
    settings: [{ id: 'maxPatches', type: 'number', default: 1024 }, { id: 'initialScale', type: 'number', default: 0.02 }],
    lowerings: lowering(
      ['self.{{module}} = nn.Parameter(torch.randn(1, {{maxPatches}}, {{hiddenSize}}) * {{initialScale}})'],
      ['{{out:output}} = {{in:hidden}} + self.{{module}}[:, :{{in:hidden}}.shape[1]]'],
    ),
  },
  'vision-class-token': {
    id: 'vision-class-token',
    label: 'Vision class token',
    category: 'media',
    inputs: [hiddenInput],
    outputs: [hiddenOutput],
    settings: [{ id: 'initialScale', type: 'number', default: 0.02 }],
    lowerings: lowering(
      ['self.{{module}} = nn.Parameter(torch.randn(1, 1, {{hiddenSize}}) * {{initialScale}})'],
      ['{{out:output}} = torch.cat((self.{{module}}.expand({{in:hidden}}.shape[0], -1, -1), {{in:hidden}}), dim=1)'],
    ),
  },
  'masked-patch-token': {
    id: 'masked-patch-token',
    label: 'Masked patch token',
    category: 'media',
    inputs: [hiddenInput, { id: 'mask', tensor: 'token-ids', rank: 2 }],
    outputs: [hiddenOutput],
    settings: [{ id: 'initialScale', type: 'number', default: 0.02 }],
    lowerings: lowering(
      ['self.{{module}} = nn.Parameter(torch.randn(1, 1, {{hiddenSize}}) * {{initialScale}})'],
      ['{{out:output}} = torch.where({{in:mask}}.bool().unsqueeze(-1), self.{{module}}.to({{in:hidden}}.dtype), {{in:hidden}})'],
    ),
  },
  'spatial-token-mixer': {
    id: 'spatial-token-mixer',
    label: 'Spatial depthwise token mixer',
    category: 'media',
    inputs: [hiddenInput],
    outputs: [hiddenOutput],
    settings: [{ id: 'kernelSize', type: 'number', default: 3 }],
    lowerings: lowering(
      ['self.{{module}} = nn.Conv1d({{hiddenSize}}, {{hiddenSize}}, kernel_size={{kernelSize}}, padding={{kernelSize}} // 2, groups={{hiddenSize}})'],
      ['{{out:output}} = self.{{module}}({{in:hidden}}.transpose(1, 2)).transpose(1, 2)'],
    ),
  },
  'global-patch-pooling': unaryMediaAtom('global-patch-pooling', 'Global patch pooling', '{{in:hidden}}.mean(dim=1, keepdim=True)'),
  'patch-token-downsample': unaryMediaAtom(
    'patch-token-downsample',
    'Patch token downsample',
    '{{in:hidden}}[:, ::max(1, int({{stride}}))]',
    [{ id: 'stride', type: 'number', default: 2 }],
  ),
  'patch-token-expansion': unaryMediaAtom(
    'patch-token-expansion',
    'Patch token expansion',
    '{{in:hidden}}.repeat_interleave(max(1, int({{factor}})), dim=1)',
    [{ id: 'factor', type: 'number', default: 2 }],
  ),
  'patch-pair-merger': {
    id: 'patch-pair-merger',
    label: 'Patch pair merger',
    category: 'media',
    inputs: [hiddenInput],
    outputs: [hiddenOutput],
    settings: [{ id: 'bias', type: 'boolean', default: true }],
    lowerings: lowering(
      ['self.{{module}} = nn.Linear({{hiddenSize}} * 2, {{hiddenSize}}, bias={{bias}})'],
      [
        '{{module}}_length = ({{in:hidden}}.shape[1] // 2) * 2',
        '{{module}}_pairs = {{in:hidden}}[:, :{{module}}_length].reshape({{in:hidden}}.shape[0], -1, {{hiddenSize}} * 2)',
        '{{out:output}} = self.{{module}}({{module}}_pairs)',
      ],
    ),
  },
  'spatial-relative-bias': {
    id: 'spatial-relative-bias',
    label: 'Spatial relative-position bias',
    category: 'media',
    inputs: [hiddenInput],
    outputs: [{ id: 'bias', tensor: 'attention', rank: 3 }],
    settings: [{ id: 'maxPatches', type: 'number', default: 1024 }],
    lowerings: lowering(
      ['self.{{module}} = nn.Parameter(torch.zeros({{maxPatches}}, {{maxPatches}}))'],
      ['{{out:bias}} = self.{{module}}[:{{in:hidden}}.shape[1], :{{in:hidden}}.shape[1]].unsqueeze(0).expand({{in:hidden}}.shape[0], -1, -1)'],
    ),
  },

  // Video / temporal token processing.
  'tubelet-projection': linearMediaAtom('tubelet-projection', 'Video tubelet projection'),
  'video-latent-encoder': linearMediaAtom('video-latent-encoder', 'Video latent encoder'),
  'video-latent-decoder': linearMediaAtom('video-latent-decoder', 'Video frame decoder'),
  'video-frame-reconstruction': {
    id: 'video-frame-reconstruction',
    label: 'Video frame reconstruction head',
    category: 'media',
    inputs: [hiddenInput],
    outputs: [hiddenOutput],
    settings: [{ id: 'tubeletFeatures', type: 'number', default: 768 }, { id: 'bias', type: 'boolean', default: true }],
    lowerings: lowering(
      ['self.{{module}} = nn.Linear({{hiddenSize}}, {{tubeletFeatures}}, bias={{bias}})'],
      ['{{out:output}} = self.{{module}}({{in:hidden}})'],
    ),
  },
  'temporal-depthwise-convolution': {
    id: 'temporal-depthwise-convolution',
    label: 'Temporal depthwise convolution',
    category: 'media',
    inputs: [hiddenInput],
    outputs: [hiddenOutput],
    settings: [{ id: 'kernelSize', type: 'number', default: 3 }],
    lowerings: lowering(
      ['self.{{module}} = nn.Conv1d({{hiddenSize}}, {{hiddenSize}}, kernel_size={{kernelSize}}, padding={{kernelSize}} // 2, groups={{hiddenSize}})'],
      ['{{out:output}} = self.{{module}}({{in:hidden}}.transpose(1, 2)).transpose(1, 2)'],
    ),
  },
  'temporal-position-embedding': {
    id: 'temporal-position-embedding',
    label: 'Learned temporal positions',
    category: 'media',
    inputs: [hiddenInput],
    outputs: [hiddenOutput],
    settings: [{ id: 'maxFrames', type: 'number', default: 256 }, { id: 'initialScale', type: 'number', default: 0.02 }],
    lowerings: lowering(
      ['self.{{module}} = nn.Parameter(torch.randn(1, {{maxFrames}}, {{hiddenSize}}) * {{initialScale}})'],
      ['{{out:output}} = {{in:hidden}} + self.{{module}}[:, :{{in:hidden}}.shape[1]]'],
    ),
  },
  'temporal-average-pooling': unaryMediaAtom(
    'temporal-average-pooling',
    'Temporal average pooling',
    'F.avg_pool1d({{in:hidden}}.transpose(1, 2), kernel_size=max(1, int({{kernelSize}})), stride=max(1, int({{stride}}))).transpose(1, 2)',
    [{ id: 'kernelSize', type: 'number', default: 2 }, { id: 'stride', type: 'number', default: 2 }],
  ),
  'temporal-token-downsample': unaryMediaAtom(
    'temporal-token-downsample',
    'Temporal token downsample',
    '{{in:hidden}}[:, ::max(1, int({{stride}}))]',
    [{ id: 'stride', type: 'number', default: 2 }],
  ),
  'temporal-token-interpolation': unaryMediaAtom(
    'temporal-token-interpolation',
    'Temporal token interpolation',
    "F.interpolate({{in:hidden}}.transpose(1, 2), scale_factor={{scaleFactor}}, mode='linear', align_corners=False).transpose(1, 2)",
    [{ id: 'scaleFactor', type: 'number', default: 2 }],
  ),
  'temporal-difference': unaryMediaAtom(
    'temporal-difference',
    'Temporal frame difference',
    'torch.cat((torch.zeros_like({{in:hidden}}[:, :1]), {{in:hidden}}[:, 1:] - {{in:hidden}}[:, :-1]), dim=1)',
  ),
  'temporal-smoothing': unaryMediaAtom(
    'temporal-smoothing',
    'Temporal smoothing',
    'F.avg_pool1d({{in:hidden}}.transpose(1, 2), kernel_size={{kernelSize}}, stride=1, padding={{kernelSize}} // 2).transpose(1, 2)',
    [{ id: 'kernelSize', type: 'number', default: 3 }],
  ),
  'temporal-gated-unit': {
    id: 'temporal-gated-unit',
    label: 'Temporal gated unit',
    category: 'media',
    inputs: [hiddenInput],
    outputs: [hiddenOutput],
    settings: [{ id: 'bias', type: 'boolean', default: true }],
    lowerings: lowering(
      ['self.{{module}}_value = nn.Linear({{hiddenSize}}, {{hiddenSize}}, bias={{bias}})', 'self.{{module}}_gate = nn.Linear({{hiddenSize}}, {{hiddenSize}}, bias={{bias}})'],
      ['{{out:output}} = self.{{module}}_value({{in:hidden}}) * torch.sigmoid(self.{{module}}_gate({{in:hidden}}))'],
    ),
  },
  'frame-type-embedding': {
    id: 'frame-type-embedding',
    label: 'Video frame type embedding',
    category: 'media',
    inputs: [hiddenInput],
    outputs: [hiddenOutput],
    settings: [{ id: 'initialScale', type: 'number', default: 0.02 }],
    lowerings: lowering(
      ['self.{{module}} = nn.Parameter(torch.randn(1, 1, {{hiddenSize}}) * {{initialScale}})'],
      ['{{out:output}} = {{in:hidden}} + self.{{module}}'],
    ),
  },

  // Multimodal alignment, fusion and generation.
  'conditioning-projection': linearMediaAtom('conditioning-projection', 'Text conditioning projection', false),
  'modality-type-embedding': {
    id: 'modality-type-embedding',
    label: 'Modality type embedding',
    category: 'media',
    inputs: [hiddenInput],
    outputs: [hiddenOutput],
    settings: [{ id: 'initialScale', type: 'number', default: 0.02 }],
    lowerings: lowering(
      ['self.{{module}} = nn.Parameter(torch.randn(1, 1, {{hiddenSize}}) * {{initialScale}})'],
      ['{{out:output}} = {{in:hidden}} + self.{{module}}'],
    ),
  },
  'adaptive-conditioning': {
    id: 'adaptive-conditioning',
    label: 'Adaptive multimodal conditioning',
    category: 'media',
    inputs: [{ id: 'content', tensor: 'hidden', rank: 3 }, { id: 'conditioning', tensor: 'hidden', rank: 3 }],
    outputs: [hiddenOutput],
    settings: [{ id: 'bias', type: 'boolean', default: true }],
    lowerings: lowering(
      ['self.{{module}} = nn.Linear({{hiddenSize}} * 2, {{hiddenSize}}, bias={{bias}})'],
      ['{{out:output}} = self.{{module}}(torch.cat(({{in:content}}, {{in:conditioning}}), dim=-1))'],
    ),
  },
  'multimodal-token-concatenation': {
    id: 'multimodal-token-concatenation',
    label: 'Multimodal token concatenation',
    category: 'media',
    inputs: twoHiddenInputs,
    outputs: [hiddenOutput],
    settings: [],
    lowerings: lowering([], ['{{out:output}} = torch.cat(({{in:left}}, {{in:right}}), dim=1)']),
  },
  'gated-multimodal-fusion': {
    id: 'gated-multimodal-fusion',
    label: 'Gated multimodal fusion',
    category: 'media',
    inputs: twoHiddenInputs,
    outputs: [hiddenOutput],
    settings: [{ id: 'bias', type: 'boolean', default: true }],
    lowerings: lowering(
      ['self.{{module}}_gate = nn.Linear({{hiddenSize}} * 2, {{hiddenSize}}, bias={{bias}})'],
      [
        '{{module}}_gate = torch.sigmoid(self.{{module}}_gate(torch.cat(({{in:left}}, {{in:right}}), dim=-1)))',
        '{{out:output}} = {{module}}_gate * {{in:left}} + (1 - {{module}}_gate) * {{in:right}}',
      ],
    ),
  },
  'film-conditioning': {
    id: 'film-conditioning',
    label: 'FiLM multimodal conditioning',
    category: 'media',
    inputs: [{ id: 'content', tensor: 'hidden', rank: 3 }, { id: 'conditioning', tensor: 'hidden', rank: 3 }],
    outputs: [hiddenOutput],
    settings: [{ id: 'bias', type: 'boolean', default: true }],
    lowerings: lowering(
      ['self.{{module}} = nn.Linear({{hiddenSize}}, {{hiddenSize}} * 2, bias={{bias}})'],
      [
        '{{module}}_scale, {{module}}_shift = self.{{module}}({{in:conditioning}}).chunk(2, dim=-1)',
        '{{out:output}} = {{in:content}} * (1 + {{module}}_scale) + {{module}}_shift',
      ],
    ),
  },
  'cross-modal-attention': {
    id: 'cross-modal-attention',
    label: 'Cross-modal attention',
    category: 'media',
    inputs: [{ id: 'query', tensor: 'hidden', rank: 3 }, { id: 'context', tensor: 'hidden', rank: 3 }],
    outputs: [hiddenOutput],
    settings: [{ id: 'numHeads', type: 'number', default: 8 }, { id: 'dropout', type: 'number', default: 0 }, { id: 'bias', type: 'boolean', default: true }],
    lowerings: lowering(
      ['self.{{module}} = nn.MultiheadAttention({{hiddenSize}}, {{numHeads}}, dropout={{dropout}}, bias={{bias}}, batch_first=True)'],
      ['{{out:output}}, _ = self.{{module}}({{in:query}}, {{in:context}}, {{in:context}}, need_weights=False)'],
    ),
  },
  'perceiver-resampler': {
    id: 'perceiver-resampler',
    label: 'Perceiver media resampler',
    category: 'media',
    inputs: [hiddenInput],
    outputs: [hiddenOutput],
    settings: [{ id: 'numLatents', type: 'number', default: 32 }, { id: 'numHeads', type: 'number', default: 8 }, { id: 'initialScale', type: 'number', default: 0.02 }],
    lowerings: lowering(
      ['self.{{module}}_latents = nn.Parameter(torch.randn(1, {{numLatents}}, {{hiddenSize}}) * {{initialScale}})', 'self.{{module}}_attention = nn.MultiheadAttention({{hiddenSize}}, {{numHeads}}, batch_first=True)'],
      [
        '{{module}}_queries = self.{{module}}_latents.expand({{in:hidden}}.shape[0], -1, -1)',
        '{{out:output}}, _ = self.{{module}}_attention({{module}}_queries, {{in:hidden}}, {{in:hidden}}, need_weights=False)',
      ],
    ),
  },
  'contrastive-media-projection': linearMediaAtom('contrastive-media-projection', 'Contrastive media projection', false),
  'media-embedding-normalization': unaryMediaAtom('media-embedding-normalization', 'Media embedding normalization', 'F.normalize({{in:hidden}}, p=2, dim=-1, eps={{epsilon}})', [{ id: 'epsilon', type: 'number', default: 1e-12 }]),
  'cross-modal-cosine-similarity': {
    id: 'cross-modal-cosine-similarity',
    label: 'Cross-modal cosine similarity',
    category: 'media',
    inputs: twoHiddenInputs,
    outputs: [{ id: 'scores', tensor: 'attention', rank: 3 }],
    settings: [{ id: 'temperature', type: 'number', default: 0.07 }],
    lowerings: lowering([], [
      '{{module}}_left = F.normalize({{in:left}}, p=2, dim=-1)',
      '{{module}}_right = F.normalize({{in:right}}, p=2, dim=-1)',
      '{{out:scores}} = torch.matmul({{module}}_left, {{module}}_right.transpose(-2, -1)) / max({{temperature}}, 1e-6)',
    ]),
  },
  'media-token-selector': {
    id: 'media-token-selector',
    label: 'Media token selector',
    category: 'media',
    inputs: [hiddenInput, { id: 'indices', tensor: 'token-ids', rank: 2 }],
    outputs: [hiddenOutput],
    settings: [],
    lowerings: lowering([], ['{{out:output}} = torch.gather({{in:hidden}}, 1, {{in:indices}}.long().unsqueeze(-1).expand(-1, -1, {{in:hidden}}.shape[-1]))']),
  },
  'classifier-free-guidance': {
    id: 'classifier-free-guidance',
    label: 'Classifier-free guidance merge',
    category: 'media',
    inputs: [{ id: 'unconditioned', tensor: 'hidden', rank: 3 }, { id: 'conditioned', tensor: 'hidden', rank: 3 }],
    outputs: [hiddenOutput],
    settings: [{ id: 'guidanceScale', type: 'number', default: 7.5 }],
    lowerings: lowering([], ['{{out:output}} = {{in:unconditioned}} + {{guidanceScale}} * ({{in:conditioned}} - {{in:unconditioned}})']),
  },
  'latent-denoiser': {
    id: 'latent-denoiser',
    label: 'Residual latent denoiser',
    category: 'media',
    inputs: [hiddenInput],
    outputs: [hiddenOutput],
    settings: [{ id: 'intermediateSize', type: 'number', default: 3072 }, { id: 'bias', type: 'boolean', default: true }],
    lowerings: lowering(
      ['self.{{module}}_up = nn.Linear({{hiddenSize}}, {{intermediateSize}}, bias={{bias}})', 'self.{{module}}_down = nn.Linear({{intermediateSize}}, {{hiddenSize}}, bias={{bias}})'],
      ['{{out:output}} = {{in:hidden}} + self.{{module}}_down(F.gelu(self.{{module}}_up({{in:hidden}})))'],
    ),
  },
  'diffusion-noise-injection': unaryMediaAtom(
    'diffusion-noise-injection',
    'Diffusion noise injection',
    '{{in:hidden}} + torch.randn_like({{in:hidden}}) * {{noiseScale}}',
    [{ id: 'noiseScale', type: 'number', default: 0.1 }],
  ),
  'diffusion-velocity-blend': {
    id: 'diffusion-velocity-blend',
    label: 'Diffusion velocity blend',
    category: 'media',
    inputs: twoHiddenInputs,
    outputs: [hiddenOutput],
    settings: [{ id: 'mix', type: 'number', default: 0.5 }],
    lowerings: lowering([], ['{{out:output}} = (1 - {{mix}}) * {{in:left}} + {{mix}} * {{in:right}}']),
  },
}
