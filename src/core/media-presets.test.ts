import { describe, expect, it } from 'vitest'
import { compileToPyTorch, validateGraph } from './ir'
import { audioEncoderPreset, audioVqTokenizerPreset, imageVqTokenizerPreset, multimodalImageEditorPreset, videoTransformerPreset, videoVqTokenizerPreset, visionTransformerPreset } from './media-presets'
import { modelAtomRegistry } from './model-atoms'

const mediaPresets = [visionTransformerPreset, multimodalImageEditorPreset, videoTransformerPreset, audioEncoderPreset, imageVqTokenizerPreset, videoVqTokenizerPreset, audioVqTokenizerPreset]

describe('executable media presets', () => {
  it.each(mediaPresets.map((preset) => [preset.name, preset] as const))('%s validates and compiles to PyTorch', (_name, preset) => {
    expect(validateGraph(preset)).toEqual({ valid: true, errors: [] })
    const code = compileToPyTorch(preset)
    expect(code).toContain('class GeneratedModel(nn.Module):')
    expect(code).toContain('# labo:node=')
  })

  it('uses explicit image, video, audio and multimodal atomic capabilities', () => {
    for (const atomId of ['vision-patch-projection', 'modality-type-embedding', 'adaptive-conditioning', 'temporal-depthwise-convolution', 'audio-temporal-convolution', 'audio-ctc-head', 'latent-denoiser', 'image-latent-decoder', 'video-latent-decoder']) {
      expect(modelAtomRegistry[atomId]).toMatchObject({ id: atomId, category: 'media', lowerings: { pytorch: { executable: true } } })
    }
  })

  it('ships complete executable media families instead of preset-only placeholders', () => {
    const mediaAtoms = Object.values(modelAtomRegistry).filter((atom) => atom.category === 'media')
    expect(mediaAtoms.length).toBeGreaterThanOrEqual(40)

    const required = [
      // Raw image/video inputs can become hidden token sequences.
      'image-channel-normalization',
      'image-resize',
      'image-patch-embedding',
      'global-image-embedding',
      'image-vq-tokenizer',
      'image-codebook-embedding',
      'image-token-decoder',
      'video-channel-normalization',
      'video-spatial-resize',
      'video-tubelet-embedding',
      'video-vq-tokenizer',
      'video-codebook-embedding',
      'video-token-decoder',
      'audio-waveform-normalization',
      'audio-preemphasis',
      'audio-resample',
      'audio-frame-embedding',
      'audio-vq-tokenizer',
      'audio-codebook-embedding',
      'audio-token-decoder',
      // Image and video token processing.
      'spatial-position-embedding',
      'vision-class-token',
      'masked-patch-token',
      'patch-pair-merger',
      'temporal-position-embedding',
      'temporal-difference',
      'temporal-token-interpolation',
      'video-frame-reconstruction',
      // Multimodal fusion and generative media.
      'multimodal-token-concatenation',
      'gated-multimodal-fusion',
      'film-conditioning',
      'cross-modal-attention',
      'perceiver-resampler',
      'cross-modal-cosine-similarity',
      'classifier-free-guidance',
      'diffusion-noise-injection',
    ]

    for (const atomId of required) {
      const atom = modelAtomRegistry[atomId]
      expect(atom, `missing ${atomId}`).toBeDefined()
      expect(atom.category, atomId).toBe('media')
      expect(atom.lowerings.pytorch.executable, atomId).toBe(true)
      expect(atom.lowerings.pytorch.forward.length, atomId).toBeGreaterThan(0)
    }

    expect(modelAtomRegistry['image-patch-embedding']).toMatchObject({
      inputs: [{ id: 'image', tensor: 'image', rank: 4 }],
      outputs: [{ id: 'output', tensor: 'hidden', rank: 3 }],
    })
    expect(modelAtomRegistry['video-tubelet-embedding']).toMatchObject({
      inputs: [{ id: 'video', tensor: 'video', rank: 5 }],
      outputs: [{ id: 'output', tensor: 'hidden', rank: 3 }],
    })
  })
})
