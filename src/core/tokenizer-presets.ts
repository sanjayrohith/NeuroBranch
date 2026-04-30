import { createTokenizerPipeline } from './tokenizer-ir'

export const researchBpePreset = createTokenizerPipeline({
  id: 'research-bpe',
  name: 'Research BPE',
  steps: [
    { id: 'normalize', atom: 'unicode-normalize', settings: { form: 'NFKC' } },
    { id: 'pretokenize', atom: 'byte-level-pretokenize', settings: { addPrefixSpace: false } },
    { id: 'model', atom: 'bpe-model', settings: { unkToken: '<unk>' } },
    { id: 'train', atom: 'bpe-trainer', settings: { vocabSize: 32768, specialTokens: ['<unk>', '<eos>'] } },
    { id: 'decode', atom: 'byte-level-decode', settings: {} },
  ],
  links: [
    { id: 'text-normalized', kind: 'text', source: 'normalize', target: 'pretokenize' },
    { id: 'pieces-model', kind: 'pieces', source: 'pretokenize', target: 'model' },
    { id: 'model-training', kind: 'vocabulary', source: 'model', target: 'train' },
    { id: 'model-decoder', kind: 'token-ids', source: 'model', target: 'decode' },
  ],
})

export const o200kBasePreset = createTokenizerPipeline({
  id: 'o200k-base',
  name: 'o200k_base · OpenAI tiktoken',
  steps: [
    { id: 'o200k', atom: 'tiktoken-encoding', settings: { encoding: 'o200k_base', vocabSize: 200019 } },
  ],
  links: [],
})

export const imageVqPreset = createTokenizerPipeline({
  id: 'image-vq',
  name: 'Image VQ · tokenizer + embedding',
  steps: [
    { id: 'image-normalize', atom: 'image-normalize', settings: { mean: 0.5, standardDeviation: 0.5 } },
    { id: 'image-encode', atom: 'image-vq-encode', settings: { inputChannels: 3, hiddenSize: 256, patchSize: 16, codebookSize: 1024 } },
    { id: 'image-embedding', atom: 'image-codebook-embedding', settings: { codebookSize: 1024, hiddenSize: 256 } },
    { id: 'image-decode', atom: 'image-vq-decode', settings: { hiddenSize: 256, outputChannels: 3, patchSize: 16 } },
  ],
  links: [
    { id: 'image-normalized', kind: 'image', source: 'image-normalize', target: 'image-encode' },
    { id: 'image-token-ids', kind: 'token-ids', source: 'image-encode', target: 'image-embedding' },
    { id: 'image-latents', kind: 'hidden', source: 'image-embedding', target: 'image-decode' },
  ],
})

export const videoVqPreset = createTokenizerPipeline({
  id: 'video-vq',
  name: 'Video VQ · tokenizer + embedding',
  steps: [
    { id: 'video-normalize', atom: 'video-normalize', settings: { mean: 0.5, standardDeviation: 0.5 } },
    { id: 'video-encode', atom: 'video-vq-encode', settings: { inputChannels: 3, hiddenSize: 256, tubeletSize: 2, patchSize: 16, codebookSize: 1024 } },
    { id: 'video-embedding', atom: 'video-codebook-embedding', settings: { codebookSize: 1024, hiddenSize: 256 } },
    { id: 'video-decode', atom: 'video-vq-decode', settings: { hiddenSize: 256, outputChannels: 3, tubeletSize: 2, patchSize: 16 } },
  ],
  links: [
    { id: 'video-normalized', kind: 'video', source: 'video-normalize', target: 'video-encode' },
    { id: 'video-token-ids', kind: 'token-ids', source: 'video-encode', target: 'video-embedding' },
    { id: 'video-latents', kind: 'hidden', source: 'video-embedding', target: 'video-decode' },
  ],
})

export const audioVqPreset = createTokenizerPipeline({
  id: 'audio-vq',
  name: 'Audio VQ · tokenizer + embedding',
  steps: [
    { id: 'audio-normalize', atom: 'audio-normalize', settings: { epsilon: 1e-6 } },
    { id: 'audio-encode', atom: 'audio-vq-encode', settings: { inputChannels: 1, hiddenSize: 256, frameSize: 400, hopSize: 160, codebookSize: 1024 } },
    { id: 'audio-embedding', atom: 'audio-codebook-embedding', settings: { codebookSize: 1024, hiddenSize: 256 } },
    { id: 'audio-decode', atom: 'audio-vq-decode', settings: { hiddenSize: 256, outputChannels: 1, frameSize: 400, hopSize: 160 } },
  ],
  links: [
    { id: 'audio-normalized', kind: 'audio', source: 'audio-normalize', target: 'audio-encode' },
    { id: 'audio-token-ids', kind: 'token-ids', source: 'audio-encode', target: 'audio-embedding' },
    { id: 'audio-latents', kind: 'hidden', source: 'audio-embedding', target: 'audio-decode' },
  ],
})

export const builtInTokenizerPresets = [researchBpePreset, o200kBasePreset, imageVqPreset, videoVqPreset, audioVqPreset]
