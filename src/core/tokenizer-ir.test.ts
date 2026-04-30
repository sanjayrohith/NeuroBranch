import { describe, expect, it } from 'vitest'
import { addTokenizerStep, compileTokenizer, createTokenizerPipeline, removeTokenizerStep, tokenizerAtomDefinitions, updateTokenizerStepSettings, validateTokenizerPipeline } from './tokenizer-ir'
import { audioVqPreset, o200kBasePreset } from './tokenizer-presets'

const pipeline = createTokenizerPipeline({
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

describe('atomic Tokenizer IR', () => {
  it('compiles one block composition to Python', () => {
    expect(validateTokenizerPipeline(pipeline)).toEqual({ valid: true, errors: [] })

    const python = compileTokenizer(pipeline)

    expect(python).toContain('normalizers.NFKC()')
    expect(python).toContain('pre_tokenizers.ByteLevel(add_prefix_space=False)')
    expect(python).toContain('vocab_size=32768')
  })

  it('recompiles Python from tokenizer mutations', () => {
    const resized = updateTokenizerStepSettings(pipeline, 'train', { vocabSize: 4096 })
    const withoutNormalization = removeTokenizerStep(resized, 'normalize')
    const python = compileTokenizer(withoutNormalization)

    expect(python).toContain('vocab_size=4096')
    expect(python).not.toContain('tokenizer.normalizer')
    expect(withoutNormalization.links.every((link) => link.source !== 'normalize' && link.target !== 'normalize')).toBe(true)
    expect(pipeline.steps.find((step) => step.id === 'train')?.settings.vocabSize).toBe(32768)
  })

  it('rebuilds an empty pipeline from a permanent atom library', () => {
    let empty = pipeline
    for (const step of pipeline.steps) empty = removeTokenizerStep(empty, step.id)

    expect(empty.steps).toEqual([])
    expect(Object.keys(tokenizerAtomDefinitions)).toHaveLength(19)
    const rebuilt = addTokenizerStep(empty, 'bpe-model')
    expect(rebuilt.steps).toEqual([
      { id: 'bpe-model-1', atom: 'bpe-model', settings: { unkToken: '<unk>' } },
    ])
    expect(empty.steps).toEqual([])
  })

  it('compiles the real o200k_base preset without pretending to retrain its vocabulary', () => {
    const python = compileTokenizer(o200kBasePreset)

    expect(validateTokenizerPipeline(o200kBasePreset)).toEqual({ valid: true, errors: [] })
    expect(o200kBasePreset.steps[0].settings.vocabSize).toBe(200019)
    expect(python).toContain('tiktoken.get_encoding("o200k_base")')
    expect(python).not.toContain('tokenizer.train')
  })

  it('compiles the audio tokenizer preset to an executable waveform codec', () => {
    const python = compileTokenizer(audioVqPreset)

    expect(validateTokenizerPipeline(audioVqPreset)).toEqual({ valid: true, errors: [] })
    expect(python).toContain('class AudioVQTokenizer(nn.Module):')
    expect(python).toContain('nn.Conv1d(')
    expect(python).toContain('nn.Embedding(')
    expect(python).toContain('nn.ConvTranspose1d(')
  })

  it('lowers a user-created tokenizer card to Python', () => {
    const custom = createTokenizerPipeline({
      ...pipeline,
      id: 'custom-tokenizer-card',
      steps: [...pipeline.steps, { id: 'lowercase', atom: 'custom-tokenizer', settings: { label: 'Lowercase', category: 'Normalization', pythonCode: 'text = text.lower()' } }],
    })
    expect(compileTokenizer(custom)).toContain('text = text.lower()')
  })
})
