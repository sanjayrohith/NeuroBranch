// @vitest-environment jsdom

import './test/app-test-setup'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import App from './App'

describe('LABO AI workspaces player', () => {
  it('preserves each preset draft while switching between model starters', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Blank starter' }))
    fireEvent.click(screen.getByRole('button', { name: 'Add Token IDs' }))
    expect(screen.getByText('1 atoms')).toBeInTheDocument()
  
    fireEvent.click(screen.getByRole('button', { name: 'GPT-like' }))
    expect(screen.getByText('15 atoms')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Blank starter' }))
  
    expect(screen.getByText('1 atoms')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Select Token IDs' })).toBeInTheDocument()
    expect(document.querySelector('[data-port-id="token-ids-tokenIds-output"]')).toBeInTheDocument()
  })
  
  it('restores the active graph after the editor is closed and reopened', () => {
    const first = render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Blank starter' }))
    fireEvent.click(screen.getByRole('button', { name: 'Add Token IDs' }))
    expect(screen.getByText('1 atoms')).toBeInTheDocument()
  
    first.unmount()
    render(<App />)
  
    expect(screen.getByText('1 atoms')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Select Token IDs' })).toBeInTheDocument()
  })
  
  it('creates a named user preset and restores its evolving draft', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Blank starter' }))
    fireEvent.click(screen.getByRole('button', { name: 'Add Token IDs' }))
    fireEvent.click(screen.getByRole('button', { name: 'Open LABO settings' }))
    fireEvent.click(screen.getByRole('button', { name: 'Workspaces' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'New model preset name' }), { target: { value: 'My routed model' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save a named copy of Blank starter' }))
    fireEvent.click(screen.getByText(/Activations/))
    fireEvent.click(screen.getByRole('button', { name: 'Add ReLU' }))
    expect(screen.getByText('2 atoms')).toBeInTheDocument()
  
    fireEvent.click(screen.getByRole('button', { name: 'Blank starter' }))
    expect(screen.getByText('0 atoms')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Select Token IDs' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Load preset My routed model' }))
  
    expect(screen.getByText('2 atoms')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Select ReLU' })).toBeInTheDocument()
  })
  
  it('creates independent blank workspaces, mixes presets, and deletes a multi-selection in edit mode', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Open LABO settings' }))
    fireEvent.click(screen.getByRole('button', { name: 'Workspaces' }))
    fireEvent.click(screen.getByRole('button', { name: 'Create and open a blank workspace' }))
    expect(screen.getByText('0 atoms')).toBeInTheDocument()
    expect(screen.getAllByText('Blank canvas 1').length).toBeGreaterThan(0)

    fireEvent.click(screen.getByText('Add an architecture for comparison'))
    fireEvent.click(screen.getByRole('button', { name: 'Add GPT-like beside current graph' }))
    fireEvent.click(screen.getByRole('button', { name: 'Add TR Basic beside current graph' }))
    expect(screen.getByText('21 atoms')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Done' }))
    fireEvent.click(screen.getByRole('button', { name: 'Split' }))
    expect(screen.getByRole('combobox', { name: 'PyTorch architecture' })).toHaveTextContent('GPT-like Starter')
    expect(screen.getByRole('combobox', { name: 'PyTorch architecture' })).toHaveTextContent('TR Basic · Shared + Residual Top-2')
  
    expect(screen.queryByRole('button', { name: /Clear architecture/ })).not.toBeInTheDocument()
    const trCards = [...document.querySelectorAll<HTMLButtonElement>('[data-node-id^="tr-basic-residual-mlp-1-"] .node-select')]
    expect(trCards).toHaveLength(6)
    for (const card of trCards) fireEvent.click(card, { shiftKey: true })
    expect(screen.getByLabelText('Selected graph cards')).toHaveTextContent('6 cards selected')
    fireEvent.click(screen.getByRole('button', { name: 'Delete selection' }))
    fireEvent.click(screen.getByRole('button', { name: 'Confirm delete 6' }))
    expect(screen.getByText('15 atoms')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Select Shared Dense Expert' })).not.toBeInTheDocument()
  }, 15_000)
  
  it('tokenizes a real prompt before running the GPT-like graph', async () => {
    const runAtomic = vi.fn(async (payload: { kind: 'model'; graph: unknown; tokenIds?: number[] } | { kind: 'tokenizer'; pipeline: unknown; sample?: string }) => {
      if (payload.kind === 'tokenizer') {
        return { engine: 'tokenizers' as const, status: 'completed' as const, tokenIds: [4, 8, 15, 16, 23, 42], results: [] }
      }
      const nodes = (payload.graph as { nodes: { id: string }[] }).nodes
      return {
        engine: 'pytorch' as const,
        status: 'completed' as const,
        modelOutput: { kind: 'logits' as const, tensorShape: [1, 6, 32000], logitsShape: [1, 6, 32000], predictedTokenId: 42, topTokenIds: [42, 23, 16, 15, 8], topProbabilities: [0.4, 0.25, 0.15, 0.12, 0.08] },
        results: nodes.map((node) => ({ atomId: node.id, status: 'passed' as const, summary: `${node.id} ok` })),
      }
    })
    window.labo = { platform: 'darwin', runtime: 'electron', runAtomic }
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'GPT-like' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'Model generation prompt' }), { target: { value: 'Salut GPT' } })
  
    fireEvent.click(screen.getByRole('button', { name: 'Play model atoms' }))
  
    await waitFor(() => expect(screen.getAllByText('completed')).toHaveLength(2))
    expect(runAtomic).toHaveBeenCalledTimes(2)
    expect(runAtomic.mock.calls[0][0]).toMatchObject({ kind: 'tokenizer', sample: 'Salut GPT' })
    expect(runAtomic.mock.calls[1][0]).toMatchObject({ kind: 'model', tokenIds: [4, 8, 15, 16, 23, 42] })
    expect(screen.getByText('6 Token IDs')).toBeInTheDocument()
    expect(screen.getByLabelText('Model generation output')).toHaveTextContent('Predicted Token ID42')
    expect(screen.getByLabelText('Model generation output')).toHaveTextContent('42 (40.00%)')
    delete window.labo
  })
  
  it('steps one elastic execution level without completing the whole graph', async () => {
    window.labo = {
      platform: 'darwin',
      runtime: 'electron',
      runAtomic: async (payload) => {
        if (payload.kind === 'tokenizer') return { engine: 'tokenizers', status: 'completed', tokenIds: [1, 2, 3], results: [] }
        const nodes = payload.kind === 'model' ? (payload.graph as { nodes: { id: string }[] }).nodes : []
        return { engine: 'pytorch', status: 'completed', results: nodes.map((node) => ({ atomId: node.id, status: 'passed', summary: `${node.id} ok` })) }
      },
    }
    render(<App />)
  
    fireEvent.click(screen.getByRole('button', { name: 'Step one model atom' }))
  
    await waitFor(() => expect(screen.getAllByText('paused')).toHaveLength(2))
    expect(screen.getByText('embedding + fixed-routes')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Select Token IDs' }).closest('.architecture-node')).toHaveClass('status-passed')
    expect(screen.getByRole('button', { name: 'Select Tied token embedding' }).closest('.architecture-node')).toHaveClass('status-pending')
    delete window.labo
  })
  
  it('starts a fresh PyTorch trace when replaying or stepping after completion', async () => {
    const runAtomic = vi.fn(async (payload: { kind: 'model'; graph: unknown; tokenIds?: number[] } | { kind: 'tokenizer'; pipeline: unknown; sample?: string }) => {
      if (payload.kind === 'tokenizer') return { engine: 'tokenizers' as const, status: 'completed' as const, tokenIds: [1, 2, 3], results: [] }
      const nodes = payload.kind === 'model' ? (payload.graph as { nodes: { id: string }[] }).nodes : []
      return { engine: 'pytorch' as const, status: 'completed' as const, results: nodes.map((node) => ({ atomId: node.id, status: 'passed' as const, summary: `${node.id} ok` })) }
    })
    window.labo = { platform: 'darwin', runtime: 'electron', runAtomic }
    render(<App />)
  
    fireEvent.click(screen.getByRole('button', { name: 'Play model atoms' }))
    await waitFor(() => expect(screen.getAllByText('completed')).toHaveLength(2))
    expect(runAtomic).toHaveBeenCalledTimes(2)
  
    fireEvent.click(screen.getByRole('button', { name: 'Step one model atom' }))
    await waitFor(() => expect(screen.getAllByText('paused')).toHaveLength(2))
    expect(runAtomic).toHaveBeenCalledTimes(4)
    expect(screen.getByText('embedding + fixed-routes')).toBeInTheDocument()
    delete window.labo
  })

  it('resets every green execution highlight when Stop is pressed', async () => {
    window.labo = {
      platform: 'darwin',
      runtime: 'electron',
      runAtomic: async (payload) => {
        if (payload.kind === 'tokenizer') return { engine: 'tokenizers', status: 'completed', tokenIds: [1, 2, 3], results: [] }
        const nodes = payload.kind === 'model' ? (payload.graph as { nodes: { id: string }[] }).nodes : []
        return { engine: 'pytorch', status: 'completed', results: nodes.map((node) => ({ atomId: node.id, status: 'passed', summary: `${node.id} ok` })) }
      },
    }
    render(<App />)

    fireEvent.click(screen.getByRole('button', { name: 'Play model atoms' }))
    await waitFor(() => expect(screen.getAllByText('completed')).toHaveLength(2))
    expect(document.querySelectorAll('.architecture-node.status-passed').length).toBeGreaterThan(0)

    fireEvent.click(screen.getByRole('button', { name: 'Stop model atoms' }))
    expect(document.querySelectorAll('.architecture-node.status-passed')).toHaveLength(0)
    expect(document.querySelectorAll('.architecture-node.status-pending')).toHaveLength(20)
    expect(screen.getAllByText('idle')).toHaveLength(2)
    delete window.labo
  })
})
