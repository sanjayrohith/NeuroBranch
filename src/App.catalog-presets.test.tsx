// @vitest-environment jsdom

import './test/app-test-setup'
import { fireEvent, render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import App from './App'

describe('LABO AI catalog presets', () => {
  it('shows the canonical TR 300M GQA-attention and deterministic routed-MLP workspace', () => {
    render(<App />)
  
    expect(screen.getByText('LABO AI')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Blocks' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'PyTorch' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Blocks' })).toHaveAttribute('aria-pressed', 'true')
    for (const label of ['Tied token embedding', 'GQA QKV projection', '16 Q / 4 KV heads', 'QK-Norm', 'Causal SDPA / Flash', 'Fixed lexical top-2 lookup', 'Shared dense SwiGLU', '4 × routed SwiGLU']) {
      expect(screen.getByRole('button', { name: `Select ${label}` })).not.toHaveAttribute('draggable')
    }
    expect(document.querySelectorAll('[data-edge-id]')).toHaveLength(31)
    expect(document.querySelector('[data-port-id="qkv-q-output"]')).toHaveAttribute('data-port-role', 'query')
    expect(document.querySelector('[data-port-id="sdpa-output-output"]')).toHaveAttribute('data-port-role', 'attention')
    expect(document.querySelector('[data-port-id="fixed-routes-expertIndices-output"]')).toHaveAttribute('data-port-role', 'expert-indices')
    expect(document.querySelector('[data-port-id="routed-expertWeights-input"]')).toHaveAttribute('data-port-role', 'routing-weights')
    fireEvent.click(screen.getByRole('button', { name: 'PyTorch' }))
    const pytorch = (screen.getByRole('textbox', { name: 'PyTorch editor' }) as HTMLTextAreaElement).value
    expect(pytorch).toContain('class GeneratedModel')
    expect(pytorch).toContain('self.embedding = nn.Embedding(32000, 1024)')
    expect(pytorch).toContain('# labo:edge=rope-q-sdpa source=rope target=sdpa')
    expect(pytorch).toContain('self.head.weight = self.embedding.weight')
    expect(document.querySelector('.python-syntax-layer')).toHaveAttribute('aria-hidden', 'true')
    expect(document.querySelector('.python-token.keyword')).toHaveTextContent('import')
    expect(document.querySelector('.python-token.class-name')).toHaveTextContent('GeneratedModel')
    expect(screen.queryByRole('button', { name: 'Run checks' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Play model atoms' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Step one model atom' })).toBeInTheDocument()
    expect(screen.getByText('20 atoms')).toBeInTheDocument()
    expect(screen.getByText('PyTorch graph executable')).toBeInTheDocument()
    expect(screen.queryByText('368.64K params')).not.toBeInTheDocument()
    expect(screen.queryByText('98.18M params')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Add Learned Hidden-State Router' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Add Top-K Routing' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Add Routed Residual Experts' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Add Shared Dense Expert' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Add DeepSeek-style MoE' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /dispatch|gather|scatter/i })).not.toBeInTheDocument()
    fireEvent.click(screen.getByText(/Activations/))
    expect(screen.getByRole('button', { name: 'Add ReLU' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Add GELU' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Add SiLU / Swish' })).toBeInTheDocument()
  })
  
  it('exposes the 100+ native card catalog by useful atomic families', () => {
    render(<App />)
  
    for (const family of ['Normalization', 'Attention', 'Position & sequence', 'Tensor composition', 'MLP blocks', 'Output heads']) {
      fireEvent.click(screen.getByText(family, { selector: 'summary' }))
    }
    for (const card of ['ScaleNorm', 'Bidirectional SDPA', 'Sinusoidal position encoding', 'Learned gated blend', 'Squared-ReLU MLP', 'Softmax output']) {
      expect(screen.getByRole('button', { name: `Add ${card}` })).toBeInTheDocument()
    }
  })
  
  it('offers executable vision, image-editing, and video starters with their own card family', () => {
    render(<App />)
  
    fireEvent.click(screen.getByText('Graph inputs'))
    expect(screen.getByRole('button', { name: 'Add Image Tensor' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Add Video Tensor' })).toBeInTheDocument()
    for (const family of ['Image inputs & tokenization', 'Vision & spatial processing', 'Video inputs & tokenization', 'Video & temporal processing', 'Multimodal fusion & conditioning']) {
      fireEvent.click(screen.getByText(family))
    }
    expect(screen.getByRole('button', { name: 'Add Image VQ tokenizer' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Add Image codebook embedding' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Add Video VQ tokenizer' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Add Video token decoder' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Add Vision patch projection' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Add Adaptive multimodal conditioning' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Add Temporal depthwise convolution' })).toBeInTheDocument()
    expect(screen.queryByText('Specialized variants')).not.toBeInTheDocument()
    expect(screen.queryByText('Image, video & multimodal')).not.toBeInTheDocument()
  
    fireEvent.click(document.querySelector('.preset-menu > summary')!)
    for (const preset of ['Vision', 'Image edit', 'Video']) expect(screen.getByRole('button', { name: preset })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Image tokenizer' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Video tokenizer' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Image edit' }))
    expect(screen.getByRole('button', { name: 'Select Adaptive text-image conditioning' })).toBeInTheDocument()
    expect(screen.getByText('17 atoms')).toBeInTheDocument()
  })
  
  it('switches to a focused TR Basic module graph without changing the full TR 300M preset', () => {
    render(<App />)
  
    fireEvent.click(screen.getByRole('button', { name: 'TR Basic' }))
  
    for (const label of ['Token IDs', 'Contextual hidden state', 'Fixed Token-ID Router', 'Shared Dense Expert', '4 × Routed Residual Experts', 'Shared + Routed Merge']) {
      expect(screen.getByRole('button', { name: `Select ${label}` })).toBeInTheDocument()
    }
    expect(screen.getByText('6 atoms')).toBeInTheDocument()
    expect(screen.getByText(/6 nodes · 7 links/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'PyTorch' }))
    const trCode = (screen.getByRole('textbox', { name: 'PyTorch editor' }) as HTMLTextAreaElement).value
    expect(trCode).toContain('def forward(self, token_ids: torch.Tensor, hidden_states: torch.Tensor):')
    expect(trCode).not.toContain('atom=moe-router')
  
    fireEvent.click(screen.getByRole('button', { name: 'TR 300M' }))
    fireEvent.click(screen.getByRole('button', { name: 'Blocks' }))
    expect(screen.getByRole('button', { name: 'Select GQA QKV projection' })).toBeInTheDocument()
    expect(screen.getByText('20 atoms')).toBeInTheDocument()
  })
  
  it('offers a blank starter and focused architecture presets without a static contracts panel', () => {
    render(<App />)
  
    expect(screen.queryByText('Architecture contracts')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Blank starter' }))
    expect(screen.getByText('Blank canvas ready')).toBeInTheDocument()
    expect(screen.getByText('0 atoms')).toBeInTheDocument()
    expect(screen.getByText(/0 nodes · 0 links/)).toBeInTheDocument()
    expect(screen.getByText('Neural IR blank')).toBeInTheDocument()
    expect(screen.getByText(/Add an atomic block from the library/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Play model atoms' })).toBeDisabled()
    expect(screen.getByRole('textbox', { name: 'Model generation prompt' })).toBeInTheDocument()
    expect(screen.getByLabelText('Model generation output')).toHaveTextContent('waiting for blocks')
  
    fireEvent.click(screen.getByRole('button', { name: 'Learned MoE' }))
    expect(screen.getByRole('button', { name: 'Select Learned Hidden-State Router' })).toBeInTheDocument()
    expect(screen.getByText('9 atoms')).toBeInTheDocument()
  })
  
  it('loads an executable dense GPT-like starter', () => {
    render(<App />)
  
    fireEvent.click(screen.getByRole('button', { name: 'GPT-like' }))
  
    for (const label of ['Tied token embedding', '12-head QKV projection', 'Causal self-attention', 'Dense SwiGLU MLP', 'Tied language-model head']) {
      expect(screen.getByRole('button', { name: `Select ${label}` })).toBeInTheDocument()
    }
    expect(screen.getByText('15 atoms')).toBeInTheDocument()
    expect(screen.getByText('PyTorch graph executable')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'PyTorch' }))
    const code = (screen.getByRole('textbox', { name: 'PyTorch editor' }) as HTMLTextAreaElement).value
    expect(code).toContain('F.scaled_dot_product_attention')
    expect(code).toContain('self.mlp_gate = nn.Linear(768, 3072, bias=False)')
    expect(code).toContain('self.head.weight = self.embedding.weight')
  })
  
  it('keeps the prompt and output module on every built-in model starter', () => {
    render(<App />)
  
    for (const preset of ['Blank starter', 'GPT-like', 'TR Basic', 'Learned MoE', 'TR 300M']) {
      fireEvent.click(screen.getByRole('button', { name: preset }))
      expect(screen.getByRole('textbox', { name: 'Model generation prompt' })).toBeInTheDocument()
      expect(screen.getByLabelText('Model generation output')).toBeInTheDocument()
    }
  })
  
  it('adds graph input cards from the Blockly library to a blank model', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Blank starter' }))
  
    for (const label of ['Token IDs', 'Hidden State', 'Training Labels']) {
      expect(screen.getByRole('button', { name: `Add ${label}` })).toHaveAttribute('draggable', 'true')
    }
    fireEvent.click(screen.getByRole('button', { name: 'Add Token IDs' }))
  
    expect(screen.getByRole('button', { name: 'Select Token IDs' })).toBeInTheDocument()
    expect(document.querySelector('[data-port-id="token-ids-tokenIds-output"]')).toHaveAttribute('data-port-role', 'token-ids')
    expect(screen.getByText('Atomic PyTorch draft')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'PyTorch' }))
    expect((screen.getByRole('textbox', { name: 'PyTorch editor' }) as HTMLTextAreaElement).value).toContain('return token_ids')
  })
  
  it('searches native cards with natural language and adds the selected result', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Blank starter' }))
    fireEvent.click(screen.getByRole('button', { name: 'Search model cards' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'Natural language card search' }), { target: { value: 'convertir logits en token généré' } })
  
    fireEvent.click(within(screen.getByRole('dialog', { name: 'Search cards' })).getByRole('button', { name: /Greedy token decoder/ }))
    expect(screen.queryByRole('dialog', { name: 'Search cards' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Select Greedy token decoder' })).toBeInTheDocument()
  })
})
