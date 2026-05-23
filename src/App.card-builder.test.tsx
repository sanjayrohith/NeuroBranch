// @vitest-environment jsdom

import './test/app-test-setup'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import App from './App'

function connectCardAgent(atomId: string, label: string) {
  window.labo = { platform: 'web', runtime: 'web', askLabo: async () => ({
    summary: `Compose ${label}.`,
    addedBlocks: [
      { atomId: 'hidden-state-input', nodeId: 'builder-input', reason: 'Card input.' },
      { atomId, nodeId: `builder-${atomId}`, reason: label },
    ],
    createdBlocks: [],
    connections: [{ sourceId: 'builder-input', sourcePortId: 'hidden', targetId: `builder-${atomId}`, targetPortId: 'hidden', reason: 'Connect card input.' }],
    missingBlocks: [], warnings: [],
  }) }
}

async function applyCardAgentPlan() {
  expect(await screen.findByRole('region', { name: 'Review graph plan' })).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: 'Apply full graph plan' }))
}

describe('LABO AI card builder', () => {
  it('keeps card editing distinct from Blockly adding and uses the central modal', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Edit cards' }))

    expect(screen.getByRole('button', { name: 'Add Token IDs' })).toBeDisabled()
    expect(screen.getAllByLabelText(/Editable card:/).length).toBeGreaterThan(10)
    expect(screen.getAllByLabelText('Editable card: settings').length).toBeGreaterThan(5)
    fireEvent.click(screen.getByRole('button', { name: 'Select Tied token embedding' }))
    expect(screen.getByRole('dialog', { name: 'Edit model card' })).toBeInTheDocument()
    expect(screen.getByText(/No new Blockly card is added in this mode/)).toBeInTheDocument()
    expect(screen.queryByRole('textbox', { name: 'Tied token embedding vocabSize' })).not.toBeInTheDocument()
  })
  
  it('opens the central card editor on double-click without switching modes first', () => {
    render(<App />)
    const card = screen.getByRole('button', { name: 'Select Tied token embedding' })
  
    fireEvent.pointerDown(card, { detail: 2, pointerId: 1 })
    fireEvent.doubleClick(card)
  
    expect(screen.getByRole('dialog', { name: 'Edit model card' })).toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: 'Model card name' })).toHaveValue('Tied token embedding')
  })
  
  it('creates a reusable composite graph and keeps its generated PyTorch inspectable', async () => {
    connectCardAgent('linear-projection', 'My projection')
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Split' }))
    fireEvent.click(screen.getByRole('button', { name: 'Blank starter' }))
    expect(screen.queryByRole('textbox', { name: 'Custom card name' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Reusable card' }))
    expect(await screen.findByRole('region', { name: 'Create model card' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Reusable card' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: 'Add blocks' })).toHaveAttribute('aria-pressed', 'false')
    expect(screen.getByRole('button', { name: 'Edit cards' })).toHaveAttribute('aria-pressed', 'false')
    expect(screen.getByRole('button', { name: 'Split' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByText('Card inputs')).toBeInTheDocument()
    expect(screen.getByText('Card atoms')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Card destination' })).toHaveTextContent('New architecture')
    expect(screen.queryByRole('button', { name: /Choose Place after/ })).not.toBeInTheDocument()
    fireEvent.change(screen.getByRole('textbox', { name: 'Custom card need' }), { target: { value: 'Create a linear projection card' } })
    fireEvent.click(screen.getByRole('button', { name: 'Compose card graph' }))
    await applyCardAgentPlan()
    await waitFor(() => expect(screen.getByLabelText('Card construction blocks')).toHaveTextContent('Linear projection'))
    fireEvent.change(screen.getByRole('textbox', { name: 'Custom card name' }), { target: { value: 'My projection' } })
    fireEvent.click(screen.getByRole('button', { name: 'Create reusable architecture' }))

    expect(screen.getByRole('button', { name: 'Add My projection' })).toHaveAttribute('draggable', 'true')
    expect(screen.getByRole('button', { name: 'Select My projection' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Edit cards' }))
    fireEvent.doubleClick(screen.getByRole('button', { name: 'Select My projection' }))
    expect(screen.getByRole('dialog', { name: 'Edit model card' })).toBeInTheDocument()
    expect(screen.getByText('Valid reusable composite graph')).toBeInTheDocument()
    const code = (screen.getByRole('textbox', { name: 'PyTorch editor' }) as HTMLTextAreaElement).value
    expect(code).toContain('kind=custom-card-graph')
    expect(screen.getByRole('button', { name: 'Select My projection' })).toBeInTheDocument()
  })

  it('removes only custom-card graph instances when their library definition is deleted', async () => {
    connectCardAgent('linear-projection', 'Disposable projection')
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Split' }))
    fireEvent.click(screen.getByRole('button', { name: 'Blank starter' }))
    fireEvent.click(screen.getByRole('button', { name: 'Reusable card' }))
    await screen.findByRole('region', { name: 'Create model card' })
    fireEvent.change(screen.getByRole('textbox', { name: 'Custom card need' }), { target: { value: 'Create a disposable projection' } })
    fireEvent.click(screen.getByRole('button', { name: 'Compose card graph' }))
    await applyCardAgentPlan()
    await waitFor(() => expect(screen.getByLabelText('Card construction blocks')).toHaveTextContent('Linear projection'))
    fireEvent.change(screen.getByRole('textbox', { name: 'Custom card name' }), { target: { value: 'Disposable projection' } })
    fireEvent.click(screen.getByRole('button', { name: 'Create reusable architecture' }))

    expect(screen.getByRole('button', { name: 'Select Disposable projection' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Select Hidden State' })).toBeInTheDocument()
    expect((screen.getByRole('textbox', { name: 'PyTorch editor' }) as HTMLTextAreaElement).value).toContain('kind=custom-card-graph')

    fireEvent.click(screen.getByRole('button', { name: 'Delete custom card Disposable projection' }))

    expect(screen.queryByRole('button', { name: 'Add Disposable projection' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Select Disposable projection' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Select Hidden State' })).toBeInTheDocument()
    await waitFor(() => expect((screen.getByRole('textbox', { name: 'PyTorch editor' }) as HTMLTextAreaElement).value).not.toContain('kind=custom-card-graph'))
  })
  
  it('returns from the dedicated Card Builder workspace to Model Studio', async () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Reusable card' }))
    await screen.findByRole('region', { name: 'Create model card' })

    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))

    expect(screen.queryByRole('region', { name: 'Create model card' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Reusable card' })).toHaveAttribute('aria-pressed', 'false')
  })

  it('keeps the shared application settings reachable from Reusable Card', async () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Reusable card' }))
    await screen.findByRole('region', { name: 'Create model card' })
    fireEvent.click(screen.getByRole('button', { name: 'Open LABO settings' }))

    expect(screen.getByRole('dialog', { name: 'LABO AI settings' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Application' }))
    expect(screen.getByRole('button', { name: 'Use Complexity Spectrum theme' })).toBeInTheDocument()
  })

  it('switches cleanly between model construction, card editing, and reusable-card creation', async () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Reusable card' }))
    await screen.findByRole('region', { name: 'Create model card' })

    fireEvent.click(screen.getByRole('button', { name: 'Add blocks' }))
    expect(screen.queryByRole('region', { name: 'Create model card' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Add blocks' })).toHaveAttribute('aria-pressed', 'true')

    fireEvent.click(screen.getByRole('button', { name: 'Reusable card' }))
    await screen.findByRole('region', { name: 'Create model card' })
    fireEvent.click(screen.getByRole('button', { name: 'Edit cards' }))
    expect(screen.queryByRole('region', { name: 'Create model card' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Edit cards' })).toHaveAttribute('aria-pressed', 'true')
  })
  
  it('can save a reusable card to the library without mutating the graph', async () => {
    connectCardAgent('rms-norm', 'Library RMSNorm')
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Blank starter' }))
    fireEvent.click(screen.getByRole('button', { name: 'Reusable card' }))
    await screen.findByRole('region', { name: 'Create model card' })
    fireEvent.change(screen.getByRole('textbox', { name: 'Custom card need' }), { target: { value: 'Create an RMSNorm card' } })
    fireEvent.click(screen.getByRole('button', { name: 'Compose card graph' }))
    await applyCardAgentPlan()
    await waitFor(() => expect(screen.getByLabelText('Card construction blocks')).toHaveTextContent('RMSNorm'))
    fireEvent.click(screen.getByRole('button', { name: 'Card destination' }))
    fireEvent.click(screen.getByRole('button', { name: 'Choose Reusable library card' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'Custom card name' }), { target: { value: 'Library RMSNorm' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save to My cards' }))
  
    expect(screen.getByRole('button', { name: 'Add Library RMSNorm' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Select Library RMSNorm' })).not.toBeInTheDocument()
    expect(screen.getByText('0 atoms')).toBeInTheDocument()
  })
  
  it('auto-composes category-specific Blockly card construction blocks', async () => {
    connectCardAgent('silu', 'Use a SiLU activation for the expert branch')
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Reusable card' }))
    await screen.findByRole('region', { name: 'Create model card' })
    fireEvent.change(screen.getByRole('textbox', { name: 'Custom card need' }), { target: { value: 'Use a SiLU activation for the expert branch' } })
    fireEvent.click(screen.getByRole('button', { name: 'Compose card graph' }))
    await applyCardAgentPlan()

    await waitFor(() => expect(screen.getByLabelText('Card construction blocks')).toHaveTextContent('SiLU'))
    fireEvent.click(screen.getByRole('button', { name: 'PyTorch' }))
    expect(screen.getByRole('button', { name: 'PyTorch' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByLabelText('Python code preview')).toHaveTextContent('atom=silu')
    expect((screen.getByRole('textbox', { name: 'Custom card name' }) as HTMLInputElement).value).toMatch(/^Use a SiLU activation/)
    expect(screen.queryByText('Create PyTorch card')).not.toBeInTheDocument()
  })
  
  it('prompts Ask LABO directly from the Card Builder without mutating the graph', async () => {
    const askLabo = vi.fn(async () => ({
      summary: 'One reusable card.', addedBlocks: [{ atomId: 'hidden-state-input', nodeId: 'builder-input', reason: 'input' }],
      createdBlocks: [{ nodeId: 'builder-gelu', label: 'Expert GELU', pytorchModule: 'nn.GELU()', inputRole: 'hidden' as const, outputRole: 'hidden' as const, reason: 'Requested in Card Builder.' }],
      connections: [{ sourceId: 'builder-input', sourcePortId: 'hidden', targetId: 'builder-gelu', targetPortId: 'hidden', reason: 'connect' }], missingBlocks: [], warnings: [],
    }))
    window.labo = { platform: 'web', runtime: 'web', askLabo }
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Reusable card' }))
    await screen.findByRole('region', { name: 'Create model card' })
    fireEvent.change(screen.getByRole('textbox', { name: 'Custom card need' }), { target: { value: 'Create a GELU expert activation' } })
    fireEvent.click(screen.getByRole('button', { name: 'Compose card graph' }))
    await applyCardAgentPlan()

    await waitFor(() => expect(screen.getByLabelText('Card construction blocks')).toHaveTextContent('GELU'))
    expect(screen.getByRole('textbox', { name: 'Custom card name' })).toHaveValue('Expert GELU')
    expect(askLabo).toHaveBeenCalledWith(expect.objectContaining({ context: expect.objectContaining({ cardBuilderMode: true }) }))
    expect(within(screen.getByRole('region', { name: 'Create model card' })).getByRole('button', { name: 'Select Expert GELU' })).toBeInTheDocument()
  })

  it('keeps reusable-card agent activity visible across graph and view rerenders', async () => {
    connectCardAgent('linear-projection', 'Persistent projection')
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Reusable card' }))
    await screen.findByRole('region', { name: 'Create model card' })

    fireEvent.change(screen.getByRole('textbox', { name: 'Custom card need' }), { target: { value: 'Create a persistent projection' } })
    fireEvent.click(screen.getByRole('button', { name: 'Compose card graph' }))
    await applyCardAgentPlan()

    const activity = await screen.findByLabelText('Agent activity')
    expect(activity).toHaveTextContent('Create a persistent projection')
    expect(activity).toHaveTextContent('applied')
    expect(screen.getByRole('button', { name: 'Open agent activity' })).toHaveAttribute('aria-expanded', 'true')

    fireEvent.click(screen.getByRole('button', { name: 'PyTorch' }))
    expect(screen.getByLabelText('Agent activity')).toHaveTextContent('Create a persistent projection')
    fireEvent.click(screen.getByRole('button', { name: 'Blocks' }))
    expect(screen.getByLabelText('Agent activity')).toHaveTextContent('applied')
  })
  
  it('changes the real atomic palette with the selected family', async () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Reusable card' }))
    await screen.findByRole('region', { name: 'Create model card' })
    const palette = screen.getByLabelText('Atomic card palette')
    expect(within(palette).getByRole('button', { name: /Linear projection/ })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Custom card category' }))
    fireEvent.click(screen.getByRole('button', { name: 'Choose Normalization' }))
    expect(within(palette).getByRole('button', { name: /RMSNorm/ })).toBeInTheDocument()
    expect(within(palette).getByRole('button', { name: /LayerNorm/ })).toBeInTheDocument()
    expect(within(palette).queryByRole('button', { name: /Dropout/ })).not.toBeInTheDocument()
  })
  
  it('exports the Blockly diagram or generated PyTorch through the desktop save bridge', async () => {
    const exportFile = vi.fn(async (_payload: { filename: string; content: string; kind: 'svg' | 'python' }) => ({ saved: true }))
    window.labo = { platform: 'darwin', runtime: 'electron', runAtomic: async () => ({ engine: 'pytorch', status: 'completed', results: [] }), exportFile }
    render(<App />)
  
    fireEvent.click(screen.getByLabelText('Export architecture'))
    fireEvent.click(screen.getByRole('button', { name: /Diagram SVG/ }))
    await waitFor(() => expect(exportFile).toHaveBeenCalledOnce())
    expect(exportFile.mock.calls[0]![0]).toMatchObject({ filename: 'tr-300m.svg', kind: 'svg' })
    expect(exportFile.mock.calls[0]![0].content).toContain('<svg')
    delete window.labo
  })

  it('exports the active reusable-card graph instead of the underlying model', async () => {
    const exportFile = vi.fn(async (_payload: { filename: string; content: string; kind: 'svg' | 'python' }) => ({ saved: true }))
    window.labo = {
      platform: 'darwin', runtime: 'electron', exportFile,
      askLabo: async () => ({
        summary: 'Compose card.',
        addedBlocks: [
          { atomId: 'hidden-state-input', nodeId: 'builder-input', reason: 'Card input.' },
          { atomId: 'silu', nodeId: 'builder-silu', reason: 'Reusable activation.' },
        ],
        createdBlocks: [],
        connections: [{ sourceId: 'builder-input', sourcePortId: 'hidden', targetId: 'builder-silu', targetPortId: 'hidden', reason: 'Connect card input.' }],
        missingBlocks: [], warnings: [],
      }),
    }
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Reusable card' }))
    await screen.findByRole('region', { name: 'Create model card' })
    fireEvent.change(screen.getByRole('textbox', { name: 'Custom card need' }), { target: { value: 'Create a SiLU card' } })
    fireEvent.click(screen.getByRole('button', { name: 'Compose card graph' }))
    await applyCardAgentPlan()
    await waitFor(() => expect(screen.getByLabelText('Card construction blocks')).toHaveTextContent('SiLU'))

    fireEvent.click(screen.getByLabelText('Export architecture'))
    fireEvent.click(screen.getByRole('button', { name: /Diagram SVG/ }))
    await waitFor(() => expect(exportFile).toHaveBeenCalledOnce())
    expect(exportFile.mock.calls[0]![0]).toMatchObject({ filename: 'reusable-card.svg', kind: 'svg' })
    expect(exportFile.mock.calls[0]![0].content).toContain('SiLU')
  })
  
  it('exposes the supervised objective that consumes the Training Labels Y plug', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Blank starter' }))
    fireEvent.click(screen.getByRole('button', { name: 'Add Training Labels' }))
    fireEvent.click(screen.getByRole('button', { name: 'Add Cross-entropy loss' }))
  
    expect(document.querySelector('[data-port-id="labels-labels-output"]')).toHaveAttribute('data-port-role', 'labels')
    expect(document.querySelector('[data-port-id="cross-entropy-loss-1-labels-input"]')).toHaveAttribute('data-port-role', 'labels')
    expect(document.querySelector('[data-port-id="cross-entropy-loss-1-logits-input"]')).toHaveAttribute('data-port-role', 'logits')
  })
  
  it('offers an explicit tied language-model head Blockly card', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Split' }))
    fireEvent.click(screen.getByRole('button', { name: 'Blank starter' }))
    fireEvent.click(screen.getByText(/Tied output head/))
    const card = screen.getByRole('button', { name: 'Add Tied language-model head' })
    expect(card).toHaveAttribute('draggable', 'true')
    fireEvent.click(card)
  
    expect(screen.getByRole('button', { name: 'Select Tied language-model head' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Edit cards' }))
    fireEvent.doubleClick(screen.getByRole('button', { name: 'Select Tied language-model head' }))
    expect(screen.getByRole('checkbox', { name: 'Model card setting tieEmbeddingWeights' })).toBeChecked()
    expect((screen.getByRole('textbox', { name: 'PyTorch editor' }) as HTMLTextAreaElement).value).toContain('# labo:node=lm-head-1 atom=lm-head')
    expect(screen.getByText('Atomic PyTorch draft')).toBeInTheDocument()
  })
})
