import { describe, expect, it } from 'vitest'
import { previewModelAtom } from './browser-atomic-preview'
import { audioEncoderPreset, videoTransformerPreset, visionTransformerPreset } from './media-presets'

describe('browser atomic graph preview', () => {
  it('previews raw image inputs and image patch embeddings', async () => {
    await expect(previewModelAtom(visionTransformerPreset, 'image-input')).resolves.toEqual({
      summary: 'Graph preview · Image Tensor → image · rank 4',
    })
    await expect(previewModelAtom(visionTransformerPreset, 'patch-projection')).resolves.toEqual({
      summary: 'Graph preview · Image patch embedding → hidden · rank 3',
    })
  })

  it('previews raw video inputs and video tubelet embeddings', async () => {
    await expect(previewModelAtom(videoTransformerPreset, 'video-input')).resolves.toEqual({
      summary: 'Graph preview · Video Tensor → video · rank 5',
    })
    await expect(previewModelAtom(videoTransformerPreset, 'video-spatial')).resolves.toEqual({
      summary: 'Graph preview · Video tubelet embedding → hidden · rank 3',
    })
  })

  it('previews raw audio inputs and frame embeddings', async () => {
    await expect(previewModelAtom(audioEncoderPreset, 'audio-input')).resolves.toEqual({
      summary: 'Graph preview · Audio Waveform → audio · rank 3',
    })
    await expect(previewModelAtom(audioEncoderPreset, 'audio-frames')).resolves.toEqual({
      summary: 'Graph preview · Audio frame embedding → hidden · rank 3',
    })
  })

  it('reports a missing typed input', async () => {
    const disconnected = {
      ...visionTransformerPreset,
      edges: visionTransformerPreset.edges.filter((edge) => edge.target !== 'patch-projection'),
    }
    await expect(previewModelAtom(disconnected, 'patch-projection')).rejects.toThrow('Missing image input on Image patch embedding')
  })
})
