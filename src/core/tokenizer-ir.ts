export type TokenizerAtomKind =
  | 'unicode-normalize'
  | 'byte-level-pretokenize'
  | 'bpe-model'
  | 'bpe-trainer'
  | 'byte-level-decode'
  | 'tiktoken-encoding'
  | 'image-normalize'
  | 'image-vq-encode'
  | 'image-codebook-embedding'
  | 'image-vq-decode'
  | 'video-normalize'
  | 'video-vq-encode'
  | 'video-codebook-embedding'
  | 'video-vq-decode'
  | 'audio-normalize'
  | 'audio-vq-encode'
  | 'audio-codebook-embedding'
  | 'audio-vq-decode'
  | 'custom-tokenizer'

export type TokenizerLinkKind = 'text' | 'pieces' | 'vocabulary' | 'token-ids' | 'image' | 'video' | 'audio' | 'hidden'
export type TokenizerSetting = string | number | boolean | string[]

export interface TokenizerStep {
  id: string
  atom: TokenizerAtomKind
  settings: Record<string, TokenizerSetting>
}

export interface TokenizerLink {
  id: string
  kind: TokenizerLinkKind
  source: string
  target: string
}

export interface TokenizerPipeline {
  id: string
  name: string
  steps: TokenizerStep[]
  links: TokenizerLink[]
}

interface AtomLowering {
  python(step: TokenizerStep): string[]
}

export const tokenizerAtomDefinitions: Record<TokenizerAtomKind, { label: string; category: string; defaultSettings: Record<string, TokenizerSetting> }> = {
  'unicode-normalize': { label: 'Unicode normalization', category: 'Normalization', defaultSettings: { form: 'NFKC' } },
  'byte-level-pretokenize': { label: 'Byte-level pre-tokenizer', category: 'Pre-tokenization', defaultSettings: { addPrefixSpace: false } },
  'bpe-model': { label: 'BPE model', category: 'Model', defaultSettings: { unkToken: '<unk>' } },
  'bpe-trainer': { label: 'BPE trainer', category: 'Training', defaultSettings: { vocabSize: 32768, specialTokens: ['<unk>', '<eos>'] } },
  'byte-level-decode': { label: 'Byte-level decoder', category: 'Decoding', defaultSettings: {} },
  'tiktoken-encoding': { label: 'tiktoken pretrained encoding', category: 'Model', defaultSettings: { encoding: 'o200k_base', vocabSize: 200019 } },
  'image-normalize': { label: 'Image normalization', category: 'Image input', defaultSettings: { mean: 0.5, standardDeviation: 0.5 } },
  'image-vq-encode': { label: 'Image VQ encoder', category: 'Image tokenization', defaultSettings: { inputChannels: 3, hiddenSize: 256, patchSize: 16, codebookSize: 1024 } },
  'image-codebook-embedding': { label: 'Image codebook embedding', category: 'Image embedding', defaultSettings: { codebookSize: 1024, hiddenSize: 256 } },
  'image-vq-decode': { label: 'Image token decoder', category: 'Image decoding', defaultSettings: { hiddenSize: 256, outputChannels: 3, patchSize: 16 } },
  'video-normalize': { label: 'Video normalization', category: 'Video input', defaultSettings: { mean: 0.5, standardDeviation: 0.5 } },
  'video-vq-encode': { label: 'Video VQ encoder', category: 'Video tokenization', defaultSettings: { inputChannels: 3, hiddenSize: 256, tubeletSize: 2, patchSize: 16, codebookSize: 1024 } },
  'video-codebook-embedding': { label: 'Video codebook embedding', category: 'Video embedding', defaultSettings: { codebookSize: 1024, hiddenSize: 256 } },
  'video-vq-decode': { label: 'Video token decoder', category: 'Video decoding', defaultSettings: { hiddenSize: 256, outputChannels: 3, tubeletSize: 2, patchSize: 16 } },
  'audio-normalize': { label: 'Audio waveform normalization', category: 'Audio input', defaultSettings: { epsilon: 1e-6 } },
  'audio-vq-encode': { label: 'Audio VQ encoder', category: 'Audio tokenization', defaultSettings: { inputChannels: 1, hiddenSize: 256, frameSize: 400, hopSize: 160, codebookSize: 1024 } },
  'audio-codebook-embedding': { label: 'Audio codebook embedding', category: 'Audio embedding', defaultSettings: { codebookSize: 1024, hiddenSize: 256 } },
  'audio-vq-decode': { label: 'Audio token decoder', category: 'Audio decoding', defaultSettings: { hiddenSize: 256, outputChannels: 1, frameSize: 400, hopSize: 160 } },
  'custom-tokenizer': { label: 'Custom tokenizer card', category: 'Custom', defaultSettings: { label: 'Custom tokenizer card', category: 'Custom', pythonCode: '# Python tokenizer transform' } },
}

export const tokenizerAtomMetadata = tokenizerAtomDefinitions

function pythonValue(value: TokenizerSetting): string {
  if (typeof value === 'boolean') return value ? 'True' : 'False'
  return JSON.stringify(value)
}

const atomLowerings: Record<TokenizerAtomKind, AtomLowering> = {
  'unicode-normalize': {
    python: (step) => [`tokenizer.normalizer = normalizers.${String(step.settings.form)}()`],
  },
  'byte-level-pretokenize': {
    python: (step) => [
      `tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=${pythonValue(step.settings.addPrefixSpace)})`,
    ],
  },
  'bpe-model': {
    python: (step) => [`tokenizer = Tokenizer(models.BPE(unk_token=${pythonValue(step.settings.unkToken)}))`],
  },
  'bpe-trainer': {
    python: (step) => [
      `trainer = trainers.BpeTrainer(vocab_size=${String(step.settings.vocabSize)}, special_tokens=${pythonValue(step.settings.specialTokens)})`,
    ],
  },
  'byte-level-decode': {
    python: () => ['tokenizer.decoder = decoders.ByteLevel()'],
  },
  'tiktoken-encoding': {
    python: (step) => [`tokenizer = tiktoken.get_encoding(${pythonValue(step.settings.encoding)})`],
  },
  'image-normalize': { python: (step) => [`image = (image - ${step.settings.mean}) / ${step.settings.standardDeviation}`] },
  'image-vq-encode': { python: () => ['token_ids, spatial_shape = tokenizer.encode(image)'] },
  'image-codebook-embedding': { python: () => ['image_embeddings = tokenizer.embed(token_ids)'] },
  'image-vq-decode': { python: () => ['reconstruction = tokenizer.decode(token_ids, spatial_shape)'] },
  'video-normalize': { python: (step) => [`video = (video - ${step.settings.mean}) / ${step.settings.standardDeviation}`] },
  'video-vq-encode': { python: () => ['token_ids, spatiotemporal_shape = tokenizer.encode(video)'] },
  'video-codebook-embedding': { python: () => ['video_embeddings = tokenizer.embed(token_ids)'] },
  'video-vq-decode': { python: () => ['reconstruction = tokenizer.decode(token_ids, spatiotemporal_shape)'] },
  'audio-normalize': { python: () => ['audio = audio - audio.mean(dim=-1, keepdim=True)', 'audio = audio / audio.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)'] },
  'audio-vq-encode': { python: () => ['token_ids, frame_count = tokenizer.encode(audio)'] },
  'audio-codebook-embedding': { python: () => ['audio_embeddings = tokenizer.embed(token_ids)'] },
  'audio-vq-decode': { python: () => ['reconstruction = tokenizer.decode(token_ids, frame_count)'] },
  'custom-tokenizer': {
    python: (step) => String(step.settings.pythonCode).split('\n'),
  },
}

const imageTokenizerAtoms: TokenizerAtomKind[] = ['image-normalize', 'image-vq-encode', 'image-codebook-embedding', 'image-vq-decode']
const videoTokenizerAtoms: TokenizerAtomKind[] = ['video-normalize', 'video-vq-encode', 'video-codebook-embedding', 'video-vq-decode']
const audioTokenizerAtoms: TokenizerAtomKind[] = ['audio-normalize', 'audio-vq-encode', 'audio-codebook-embedding', 'audio-vq-decode']

function compileImageTokenizer(pipeline: TokenizerPipeline): string {
  const normalization = pipeline.steps.find((step) => step.atom === 'image-normalize')?.settings ?? tokenizerAtomDefinitions['image-normalize'].defaultSettings
  const encoder = pipeline.steps.find((step) => step.atom === 'image-vq-encode')?.settings ?? tokenizerAtomDefinitions['image-vq-encode'].defaultSettings
  const embedding = pipeline.steps.find((step) => step.atom === 'image-codebook-embedding')?.settings ?? tokenizerAtomDefinitions['image-codebook-embedding'].defaultSettings
  const decoder = pipeline.steps.find((step) => step.atom === 'image-vq-decode')?.settings ?? tokenizerAtomDefinitions['image-vq-decode'].defaultSettings
  return [
    'import torch',
    'import torch.nn as nn',
    '',
    'class ImageVQTokenizer(nn.Module):',
    '    def __init__(self):',
    `        super().__init__()`,
    `        self.mean = ${normalization.mean}`,
    `        self.standard_deviation = ${normalization.standardDeviation}`,
    `        self.encoder = nn.Conv2d(${encoder.inputChannels}, ${encoder.hiddenSize}, kernel_size=${encoder.patchSize}, stride=${encoder.patchSize})`,
    `        self.codebook = nn.Embedding(${embedding.codebookSize}, ${embedding.hiddenSize})`,
    `        self.decoder = nn.ConvTranspose2d(${decoder.hiddenSize}, ${decoder.outputChannels}, kernel_size=${decoder.patchSize}, stride=${decoder.patchSize})`,
    '',
    '    def encode(self, image):',
    '        latents = self.encoder((image - self.mean) / self.standard_deviation)',
    '        height, width = latents.shape[-2:]',
    '        flat = latents.flatten(2).transpose(1, 2)',
    '        codebook = self.codebook.weight',
    '        distances = flat.pow(2).sum(-1, keepdim=True) + codebook.pow(2).sum(-1) - 2 * flat @ codebook.t()',
    '        return distances.argmin(-1), (height, width)',
    '',
    '    def embed(self, token_ids):',
    '        return self.codebook(token_ids)',
    '',
    '    def decode(self, token_ids, spatial_shape):',
    '        height, width = spatial_shape',
    '        latents = self.embed(token_ids).transpose(1, 2).reshape(token_ids.shape[0], -1, height, width)',
    '        return self.decoder(latents).sigmoid()',
    '',
    '    def forward(self, image):',
    '        token_ids, spatial_shape = self.encode(image)',
    '        return self.decode(token_ids, spatial_shape), token_ids',
    '',
    'tokenizer = ImageVQTokenizer()',
    '',
  ].join('\n')
}

function compileVideoTokenizer(pipeline: TokenizerPipeline): string {
  const normalization = pipeline.steps.find((step) => step.atom === 'video-normalize')?.settings ?? tokenizerAtomDefinitions['video-normalize'].defaultSettings
  const encoder = pipeline.steps.find((step) => step.atom === 'video-vq-encode')?.settings ?? tokenizerAtomDefinitions['video-vq-encode'].defaultSettings
  const embedding = pipeline.steps.find((step) => step.atom === 'video-codebook-embedding')?.settings ?? tokenizerAtomDefinitions['video-codebook-embedding'].defaultSettings
  const decoder = pipeline.steps.find((step) => step.atom === 'video-vq-decode')?.settings ?? tokenizerAtomDefinitions['video-vq-decode'].defaultSettings
  return [
    'import torch',
    'import torch.nn as nn',
    '',
    'class VideoVQTokenizer(nn.Module):',
    '    def __init__(self):',
    '        super().__init__()',
    `        self.mean = ${normalization.mean}`,
    `        self.standard_deviation = ${normalization.standardDeviation}`,
    `        self.encoder = nn.Conv3d(${encoder.inputChannels}, ${encoder.hiddenSize}, kernel_size=(${encoder.tubeletSize}, ${encoder.patchSize}, ${encoder.patchSize}), stride=(${encoder.tubeletSize}, ${encoder.patchSize}, ${encoder.patchSize}))`,
    `        self.codebook = nn.Embedding(${embedding.codebookSize}, ${embedding.hiddenSize})`,
    `        self.decoder = nn.ConvTranspose3d(${decoder.hiddenSize}, ${decoder.outputChannels}, kernel_size=(${decoder.tubeletSize}, ${decoder.patchSize}, ${decoder.patchSize}), stride=(${decoder.tubeletSize}, ${decoder.patchSize}, ${decoder.patchSize}))`,
    '',
    '    def encode(self, video):',
    '        latents = self.encoder((video - self.mean) / self.standard_deviation)',
    '        time, height, width = latents.shape[-3:]',
    '        flat = latents.flatten(2).transpose(1, 2)',
    '        codebook = self.codebook.weight',
    '        distances = flat.pow(2).sum(-1, keepdim=True) + codebook.pow(2).sum(-1) - 2 * flat @ codebook.t()',
    '        return distances.argmin(-1), (time, height, width)',
    '',
    '    def embed(self, token_ids):',
    '        return self.codebook(token_ids)',
    '',
    '    def decode(self, token_ids, spatiotemporal_shape):',
    '        time, height, width = spatiotemporal_shape',
    '        latents = self.embed(token_ids).transpose(1, 2).reshape(token_ids.shape[0], -1, time, height, width)',
    '        return self.decoder(latents).sigmoid()',
    '',
    '    def forward(self, video):',
    '        token_ids, shape = self.encode(video)',
    '        return self.decode(token_ids, shape), token_ids',
    '',
    'tokenizer = VideoVQTokenizer()',
    '',
  ].join('\n')
}

function compileAudioTokenizer(pipeline: TokenizerPipeline): string {
  const normalization = pipeline.steps.find((step) => step.atom === 'audio-normalize')?.settings ?? tokenizerAtomDefinitions['audio-normalize'].defaultSettings
  const encoder = pipeline.steps.find((step) => step.atom === 'audio-vq-encode')?.settings ?? tokenizerAtomDefinitions['audio-vq-encode'].defaultSettings
  const embedding = pipeline.steps.find((step) => step.atom === 'audio-codebook-embedding')?.settings ?? tokenizerAtomDefinitions['audio-codebook-embedding'].defaultSettings
  const decoder = pipeline.steps.find((step) => step.atom === 'audio-vq-decode')?.settings ?? tokenizerAtomDefinitions['audio-vq-decode'].defaultSettings
  return [
    'import torch',
    'import torch.nn as nn',
    '',
    'class AudioVQTokenizer(nn.Module):',
    '    def __init__(self):',
    '        super().__init__()',
    `        self.epsilon = ${normalization.epsilon}`,
    `        self.encoder = nn.Conv1d(${encoder.inputChannels}, ${encoder.hiddenSize}, kernel_size=${encoder.frameSize}, stride=${encoder.hopSize})`,
    `        self.codebook = nn.Embedding(${embedding.codebookSize}, ${embedding.hiddenSize})`,
    `        self.decoder = nn.ConvTranspose1d(${decoder.hiddenSize}, ${decoder.outputChannels}, kernel_size=${decoder.frameSize}, stride=${decoder.hopSize})`,
    '',
    '    def normalize(self, audio):',
    '        centered = audio - audio.mean(dim=-1, keepdim=True)',
    '        return centered / centered.abs().amax(dim=-1, keepdim=True).clamp_min(self.epsilon)',
    '',
    '    def encode(self, audio):',
    '        latents = self.encoder(self.normalize(audio)).transpose(1, 2)',
    '        codebook = self.codebook.weight',
    '        distances = latents.pow(2).sum(-1, keepdim=True) + codebook.pow(2).sum(-1) - 2 * latents @ codebook.t()',
    '        return distances.argmin(-1), latents.shape[1]',
    '',
    '    def embed(self, token_ids):',
    '        return self.codebook(token_ids)',
    '',
    '    def decode(self, token_ids, frame_count):',
    '        latents = self.embed(token_ids[:, :frame_count]).transpose(1, 2)',
    '        return torch.tanh(self.decoder(latents))',
    '',
    '    def forward(self, audio):',
    '        token_ids, frame_count = self.encode(audio)',
    '        return self.decode(token_ids, frame_count), token_ids',
    '',
    'tokenizer = AudioVQTokenizer()',
    '',
  ].join('\n')
}

export function createTokenizerPipeline(input: TokenizerPipeline): TokenizerPipeline {
  return {
    ...input,
    steps: input.steps.map((step) => ({ ...step, settings: { ...step.settings } })),
    links: input.links.map((link) => ({ ...link })),
  }
}

export function addTokenizerStep(pipeline: TokenizerPipeline, atom: TokenizerAtomKind): TokenizerPipeline {
  const definition = tokenizerAtomDefinitions[atom]
  const usedNumbers = pipeline.steps
    .filter((step) => step.id.startsWith(`${atom}-`))
    .map((step) => Number(step.id.slice(atom.length + 1)))
    .filter(Number.isFinite)
  const sequence = Math.max(0, ...usedNumbers) + 1
  const settings = Object.fromEntries(Object.entries(definition.defaultSettings).map(([key, value]) => [key, Array.isArray(value) ? [...value] : value]))
  return {
    ...pipeline,
    steps: [...pipeline.steps.map((step) => ({ ...step, settings: { ...step.settings } })), { id: `${atom}-${sequence}`, atom, settings }],
    links: pipeline.links.map((link) => ({ ...link })),
  }
}

export function updateTokenizerStepSettings(
  pipeline: TokenizerPipeline,
  stepId: string,
  settings: Record<string, TokenizerSetting>,
): TokenizerPipeline {
  if (!pipeline.steps.some((step) => step.id === stepId)) throw new Error(`Unknown tokenizer step: ${stepId}`)
  return {
    ...pipeline,
    steps: pipeline.steps.map((step) => step.id === stepId
      ? { ...step, settings: { ...step.settings, ...settings } }
      : { ...step, settings: { ...step.settings } }),
    links: pipeline.links.map((link) => ({ ...link })),
  }
}

export function removeTokenizerStep(pipeline: TokenizerPipeline, stepId: string): TokenizerPipeline {
  if (!pipeline.steps.some((step) => step.id === stepId)) throw new Error(`Unknown tokenizer step: ${stepId}`)
  return {
    ...pipeline,
    steps: pipeline.steps.filter((step) => step.id !== stepId).map((step) => ({ ...step, settings: { ...step.settings } })),
    links: pipeline.links
      .filter((link) => link.source !== stepId && link.target !== stepId)
      .map((link) => ({ ...link })),
  }
}

export function validateTokenizerPipeline(pipeline: TokenizerPipeline) {
  const errors: string[] = []
  const ids = new Set<string>()
  for (const step of pipeline.steps) {
    if (ids.has(step.id)) errors.push(`Duplicate tokenizer step: ${step.id}`)
    ids.add(step.id)
  }
  for (const link of pipeline.links) {
    if (!ids.has(link.source)) errors.push(`Unknown tokenizer link source: ${link.source}`)
    if (!ids.has(link.target)) errors.push(`Unknown tokenizer link target: ${link.target}`)
  }
  return { valid: errors.length === 0, errors }
}

export function compileTokenizer(pipeline: TokenizerPipeline): string {
  const validation = validateTokenizerPipeline(pipeline)
  if (!validation.valid) throw new Error(validation.errors.join('\n'))

  if (pipeline.steps.some((step) => imageTokenizerAtoms.includes(step.atom))) return compileImageTokenizer(pipeline)
  if (pipeline.steps.some((step) => videoTokenizerAtoms.includes(step.atom))) return compileVideoTokenizer(pipeline)
  if (pipeline.steps.some((step) => audioTokenizerAtoms.includes(step.atom))) return compileAudioTokenizer(pipeline)

  const tiktokenStep = pipeline.steps.find((step) => step.atom === 'tiktoken-encoding')
  if (tiktokenStep) {
    const customSteps = pipeline.steps.filter((step) => step.atom === 'custom-tokenizer')
    return [
      'import tiktoken',
      '',
      ...atomLowerings['tiktoken-encoding'].python(tiktokenStep),
      ...customSteps.flatMap((step) => atomLowerings['custom-tokenizer'].python(step)),
      '',
      'def encode(text: str) -> list[int]:',
      '    return tokenizer.encode(text)',
      '',
      'def decode(token_ids: list[int]) -> str:',
      '    return tokenizer.decode(token_ids)',
      '',
    ].join('\n')
  }

  const body = pipeline.steps.flatMap((step) => atomLowerings[step.atom].python(step))
  return [
    'from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers',
    '',
    ...body,
    '',
    '# Provide corpus paths from the experiment manifest.',
    'tokenizer.train(files=corpus_files, trainer=trainer)',
    'tokenizer.save(output_path)',
    '',
  ].join('\n')
}
