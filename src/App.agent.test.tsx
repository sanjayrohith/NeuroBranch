// @vitest-environment jsdom

import './test/app-test-setup'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import App from './App'

describe('LABO AI agent', () => {
  it('connects a desktop ChatGPT session from the shared Agent settings', async () => {
    const connectedSession = {
      available: true,
      connected: true,
      email: 'judge@example.com',
      planType: 'plus',
      models: [
        { id: 'gpt-5.6-sol', label: 'GPT-5.6 Sol', description: 'Frontier coding model.', efforts: ['medium', 'high'], defaultEffort: 'medium', isDefault: true },
        { id: 'gpt-5.6-terra', label: 'GPT-5.6 Terra', efforts: ['low', 'medium'], defaultEffort: 'medium', isDefault: false },
      ],
      selectedModel: 'gpt-5.6-sol',
      selectedEffort: 'medium',
    }
    const connectChatGPT = vi.fn(async () => connectedSession)
    const configureChatGPT = vi.fn(async ({ model, effort }: { model: string; effort: string }) => ({ ...connectedSession, selectedModel: model, selectedEffort: effort }))
    window.labo = {
      platform: 'darwin',
      runtime: 'electron',
      getOpenAISettings: async () => ({ configured: false, source: 'none', encryptionAvailable: true }),
      getChatGPTSession: async () => ({ available: true, connected: false }),
      connectChatGPT,
      configureChatGPT,
      disconnectChatGPT: async () => ({ available: true, connected: false }),
    }
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Open LABO settings' }))
    fireEvent.click(screen.getByRole('button', { name: 'Agent' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Continue with ChatGPT' }))

    expect(connectChatGPT).toHaveBeenCalledOnce()
    expect(await screen.findByText('judge@example.com')).toBeInTheDocument()
    expect(screen.getByText(/plus plan/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Disconnect ChatGPT/ })).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('ChatGPT model'), { target: { value: 'gpt-5.6-terra' } })
    await waitFor(() => expect(configureChatGPT).toHaveBeenCalledWith({ model: 'gpt-5.6-terra', effort: 'medium' }))
    delete window.labo
  })

  it('answers conversational prompts without opening an empty graph-plan review', async () => {
    window.labo = {
      platform: 'darwin',
      runtime: 'electron',
      getOpenAISettings: async () => ({ configured: true, source: 'secure-storage', encryptionAvailable: true }),
      askLabo: async () => ({
        summary: 'Hello! I can explain the current graph or help you edit it when you are ready.',
        addedBlocks: [],
        createdBlocks: [],
        connections: [],
        updatedBlocks: [],
        deletedBlocks: [],
        movedBlocks: [],
        actions: [],
        missingBlocks: [],
        warnings: [],
        toolTrace: [],
      }),
    }
    render(<App />)
    fireEvent.change(screen.getByLabelText('What should these blocks build?'), { target: { value: 'hello' } })
    fireEvent.keyDown(screen.getByLabelText('What should these blocks build?'), { key: 'Enter' })

    expect(await screen.findByLabelText('Agent activity')).toHaveTextContent('Hello! I can explain the current graph')
    expect(screen.getByLabelText('Agent activity')).toHaveTextContent('answered')
    expect(screen.queryByText('Review graph plan')).not.toBeInTheDocument()
    delete window.labo
  })

  it('tracks running, validated and applied agent tasks in the footer activity center', async () => {
    let resolveAsk: ((plan: {
      summary: string
      addedBlocks: { atomId: string; nodeId: string; reason: string }[]
      createdBlocks: never[]
      connections: never[]
      missingBlocks: never[]
      warnings: never[]
    }) => void) | undefined
    window.labo = {
      platform: 'darwin',
      runtime: 'electron',
      runAtomic: async () => ({ engine: 'pytorch', status: 'completed', results: [] }),
      getOpenAISettings: async () => ({ configured: true, source: 'secure-storage', encryptionAvailable: true }),
      askLabo: () => new Promise((resolve) => { resolveAsk = resolve }),
    }
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Open LABO settings' }))
    fireEvent.click(screen.getByRole('button', { name: 'Agent' }))
    fireEvent.click(screen.getByRole('button', { name: 'Auto apply' }))
    fireEvent.click(screen.getByRole('button', { name: 'Close Ask LABO' }))
    fireEvent.change(screen.getByLabelText('What should these blocks build?'), { target: { value: 'Add one ReLU card' } })
    fireEvent.click(screen.getByRole('button', { name: 'Propose graph changes' }))
  
    expect(await screen.findByLabelText('Agent activity')).toHaveTextContent('Inspecting cards and validating graph changes')
    resolveAsk?.({ summary: 'ReLU added and graph validated.', addedBlocks: [{ atomId: 'relu', nodeId: 'agent-relu-activity', reason: 'Requested activation.' }], createdBlocks: [], connections: [], missingBlocks: [], warnings: [] })
  
    expect(await screen.findByRole('button', { name: 'Select ReLU' })).toBeInTheDocument()
    expect(screen.getByLabelText('Agent activity')).toHaveTextContent('applied')
    expect(screen.getByLabelText('Agent activity')).toHaveTextContent('1 accepted')
    expect(screen.getByRole('button', { name: 'Retry agent task: Add one ReLU card' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Clear' }))
    expect(screen.queryByLabelText('Agent activity')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Open agent activity' }))
    expect(screen.getByLabelText('Agent activity')).toHaveTextContent('0 tasks')
    expect(screen.getByLabelText('Agent activity')).toHaveTextContent('Your agent runs, validation results and errors will appear here.')
    delete window.labo
  }, 10_000)
  
  it('repairs underscore block IDs before displaying and applying an agent plan', async () => {
    const askLabo = vi.fn(async () => ({
      summary: 'Add two cards with model-generated identifiers.',
      addedBlocks: [
        { atomId: 'hidden-state-input', nodeId: 'image_input', reason: 'Add an image branch input.' },
        { atomId: 'relu', nodeId: 'vision_projector', reason: 'Add a vision projection placeholder.' },
      ],
      createdBlocks: [],
      connections: [{ sourceId: 'image_input', sourcePortId: 'hidden', targetId: 'vision_projector', targetPortId: 'hidden', reason: 'Connect the branch.' }],
      missingBlocks: [],
      warnings: [],
    }))
    window.labo = {
      platform: 'darwin',
      runtime: 'electron',
      runAtomic: async () => ({ engine: 'pytorch', status: 'completed', results: [] }),
      getOpenAISettings: async () => ({ configured: true, source: 'secure-storage', encryptionAvailable: true }),
      askLabo,
    }
    render(<App />)
  
    const prompt = screen.getByLabelText('What should these blocks build?')
    fireEvent.change(prompt, { target: { value: 'Build a multimodal branch' } })
    fireEvent.keyDown(prompt, { key: 'Enter', shiftKey: true })
    expect(askLabo).not.toHaveBeenCalled()
    fireEvent.keyDown(prompt, { key: 'Enter' })
  
    expect(await screen.findByText('2 atomic blocks ready')).toBeInTheDocument()
    expect(askLabo).toHaveBeenCalledOnce()
    expect(screen.getByText('1 elastic ready')).toBeInTheDocument()
    expect(screen.getByText('Ready · 2 cards · 1 elastic')).toBeInTheDocument()
    expect(screen.queryByText(/Block id must start with a letter/)).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Apply full graph plan' }))
    expect(screen.getByRole('button', { name: 'Select Hidden State' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Select ReLU' })).toBeInTheDocument()
    delete window.labo
  })
  
  it('restores and applies an entire review plan from agent activity', async () => {
    window.labo = {
      platform: 'darwin',
      runtime: 'electron',
      runAtomic: async () => ({ engine: 'pytorch', status: 'completed', results: [] }),
      getOpenAISettings: async () => ({ configured: true, source: 'secure-storage', encryptionAvailable: true }),
      askLabo: async () => ({
        summary: 'One complete graph plan is ready.',
        addedBlocks: [{ atomId: 'relu', nodeId: 'agent-full-plan-relu', reason: 'Requested activation.' }],
        createdBlocks: [],
        connections: [],
        missingBlocks: [],
        warnings: [],
      }),
    }
    render(<App />)
    fireEvent.change(screen.getByLabelText('What should these blocks build?'), { target: { value: 'Prepare one full plan' } })
    fireEvent.click(screen.getByRole('button', { name: 'Propose graph changes' }))
  
    expect(await screen.findByText('1 atomic block ready')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Close Ask LABO' }))
    await waitFor(() => expect(screen.getByLabelText('Agent activity')).toHaveTextContent('Closed — plan saved'))
  
    fireEvent.click(screen.getByRole('button', { name: 'Review full agent plan: Prepare one full plan' }))
    expect(screen.getByText('1 atomic block ready')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Apply full graph plan' }))
  
    expect(screen.getByRole('button', { name: 'Select ReLU' })).toBeInTheDocument()
    await waitFor(() => expect(screen.getByLabelText('Agent activity')).toHaveTextContent('applied'))
    delete window.labo
  })
  
  it('previews and applies an agent-added atomic with its elastic', async () => {
    window.labo = {
      platform: 'darwin',
      runtime: 'electron',
      runAtomic: async () => ({ engine: 'pytorch', status: 'completed', results: [] }),
      getOpenAISettings: async () => ({ configured: true, source: 'secure-storage', encryptionAvailable: true }),
      askLabo: async () => ({
        summary: 'Add a ReLU branch after the final normalization.',
        addedBlocks: [{ atomId: 'relu', nodeId: 'agent-relu', reason: 'The activation is not on the canvas.' }],
        createdBlocks: [],
        connections: [{
          sourceId: 'final-norm', sourcePortId: 'output', targetId: 'agent-relu', targetPortId: 'hidden', reason: 'Feed normalized states into the new activation.',
        }],
        missingBlocks: [],
        warnings: [],
      }),
    }
    render(<App />)
  
    fireEvent.change(screen.getByLabelText('What should these blocks build?'), { target: { value: 'Add a ReLU after the final normalization' } })
    fireEvent.click(screen.getByRole('button', { name: 'Propose graph changes' }))
  
    expect(await screen.findByText('1 atomic block ready')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Edit ReLU' })).toBeInTheDocument()
    expect(screen.getByText('1 elastic ready')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Select ReLU' })).not.toBeInTheDocument()
  
    fireEvent.click(screen.getByRole('button', { name: 'Edit ReLU' }))
    expect(screen.getByRole('dialog', { name: 'Edit agent card' })).toBeInTheDocument()
    fireEvent.pointerDown(document.querySelector('.ask-labo-card-modal-backdrop')!)
    expect(screen.queryByRole('dialog', { name: 'Edit agent card' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Edit ReLU' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'Agent card name' }), { target: { value: 'Reviewed ReLU' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save card' }))
  
    fireEvent.click(screen.getByRole('button', { name: 'Apply full graph plan' }))
    expect(screen.getByRole('button', { name: 'Select Reviewed ReLU' })).toBeInTheDocument()
    expect(screen.getByText(/21 nodes · 32 links/)).toBeInTheDocument()
    delete window.labo
  })
  
  it('adds a complete parallel architecture without moving or wiring into existing cards', async () => {
    const askLabo = vi.fn(async () => ({
      summary: 'Add an independent branch.',
      addedBlocks: [
        { atomId: 'hidden-state-input', nodeId: 'parallel-input', reason: 'Independent input.' },
        { atomId: 'identity', nodeId: 'parallel-output', reason: 'Independent output.' },
      ],
      createdBlocks: [],
      connections: [{ sourceId: 'parallel-input', sourcePortId: 'hidden', targetId: 'parallel-output', targetPortId: 'hidden', reason: 'Internal branch connection.' }],
      missingBlocks: [],
      warnings: [],
    }))
    window.labo = {
      platform: 'darwin',
      runtime: 'electron',
      runAtomic: async () => ({ engine: 'pytorch', status: 'completed', results: [] }),
      getOpenAISettings: async () => ({ configured: true, source: 'secure-storage', encryptionAvailable: true }),
      askLabo,
    }
    render(<App />)
    const originalCard = screen.getByRole('button', { name: 'Select Token IDs' }).closest<HTMLElement>('.architecture-node')!
    const originalPosition = { left: originalCard.style.left, top: originalCard.style.top }
  
    fireEvent.click(screen.getByRole('button', { name: 'Open LABO settings' }))
    fireEvent.click(screen.getByRole('button', { name: 'Agent' }))
    fireEvent.click(screen.getByRole('button', { name: 'New parallel' }))
    fireEvent.click(screen.getByRole('button', { name: 'Close Ask LABO' }))
    fireEvent.change(screen.getByLabelText('What should these blocks build?'), { target: { value: 'Add another independent architecture' } })
    fireEvent.click(screen.getByRole('button', { name: 'Propose graph changes' }))
  
    expect(await screen.findByText('2 atomic blocks ready')).toBeInTheDocument()
    expect(askLabo).toHaveBeenCalledWith(expect.objectContaining({ context: expect.objectContaining({ operationMode: 'parallel' }) }))
    fireEvent.click(screen.getByRole('button', { name: 'Apply full graph plan' }))
    expect(screen.getByText(/22 nodes · 32 links/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Split' }))
    expect(screen.getByRole('combobox', { name: 'PyTorch architecture' })).toHaveTextContent('TR 300M')
    expect(screen.getByRole('combobox', { name: 'PyTorch architecture' })).toHaveTextContent('Architecture 2')
    expect({ left: originalCard.style.left, top: originalCard.style.top }).toEqual(originalPosition)
    expect(Number.parseFloat(screen.getByRole('button', { name: 'Select Hidden State' }).closest<HTMLElement>('.architecture-node')!.style.left)).toBeGreaterThan(Number.parseFloat(originalPosition.left))
    delete window.labo
  })
  
  it('replaces false agent missing claims with native Token IDs and decoder cards', async () => {
    window.labo = {
      platform: 'darwin',
      runtime: 'electron',
      runAtomic: async () => ({ engine: 'pytorch', status: 'completed', results: [] }),
      getOpenAISettings: async () => ({ configured: true, source: 'secure-storage', encryptionAvailable: true }),
      askLabo: async () => ({
        summary: 'Build a generation branch.',
        addedBlocks: [
          { atomId: 'token-embedding', nodeId: 'repair-embedding', reason: 'Embed tokens.' },
          { atomId: 'lm-head', nodeId: 'repair-head', reason: 'Produce logits.' },
        ],
        createdBlocks: [],
        connections: [{ sourceId: 'repair-embedding', sourcePortId: 'output', targetId: 'repair-head', targetPortId: 'hidden', reason: 'Project states.' }],
        missingBlocks: [
          { atomId: null, label: 'Source de Token IDs / tokenizer', reason: 'Aucun atomic disponible.' },
          { atomId: null, label: 'Échantillonneur ou décodeur de logits', reason: 'Convertir les logits en token généré.' },
        ],
        warnings: [],
      }),
    }
    render(<App />)
  
    fireEvent.change(screen.getByLabelText('What should these blocks build?'), { target: { value: 'Build a generation branch' } })
    fireEvent.click(screen.getByRole('button', { name: 'Propose graph changes' }))
  
    expect(await screen.findByText('4 atomic blocks ready')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Edit Greedy token decoder' })).toBeInTheDocument()
    expect(screen.queryByText('Missing blocks')).not.toBeInTheDocument()
    delete window.labo
  })
  
  it('previews an agent-generated safe PyTorch card before applying it', async () => {
    window.labo = {
      platform: 'darwin',
      runtime: 'electron',
      runAtomic: async () => ({ engine: 'pytorch', status: 'completed', results: [] }),
      getOpenAISettings: async () => ({ configured: true, source: 'secure-storage', encryptionAvailable: true }),
      askLabo: async () => ({
        summary: 'Create a missing projection card.',
        addedBlocks: [],
        createdBlocks: [{ nodeId: 'agent-projection', label: 'Agent projection', pytorchModule: 'nn.Linear(1024, 1024, bias=False)', reason: 'A dedicated projection is required.' }],
        connections: [{ sourceId: 'final-norm', sourcePortId: 'output', targetId: 'agent-projection', targetPortId: 'hidden', reason: 'Project final hidden states.' }],
        missingBlocks: [],
        warnings: [],
      }),
    }
    render(<App />)
  
    fireEvent.change(screen.getByLabelText('What should these blocks build?'), { target: { value: 'Create the missing projection' } })
    fireEvent.click(screen.getByRole('button', { name: 'Propose graph changes' }))
  
    expect(await screen.findByText('1 generated reusable card ready')).toBeInTheDocument()
    expect(screen.getByText('nn.Linear(1024, 1024, bias=False)')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Select Agent projection' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Edit Agent projection' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'Agent card name' }), { target: { value: 'Edited projection' } })
    fireEvent.change(screen.getByRole('textbox', { name: 'Agent card PyTorch module' }), { target: { value: 'nn.Linear(1024, 1024, bias=True)' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save card' }))
    fireEvent.click(screen.getByRole('button', { name: 'Apply full graph plan' }))
    expect(screen.getByRole('button', { name: 'Select Edited projection' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Add Edited projection' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Edit cards' }))
    fireEvent.doubleClick(screen.getByRole('button', { name: 'Select Edited projection' }))
    expect(screen.getByRole('textbox', { name: 'Model card PyTorch module' })).toHaveValue('nn.Linear(1024, 1024, bias=True)')
    delete window.labo
  })
  
  it('auto-applies a clean agent plan when Auto apply mode is enabled', async () => {
    window.labo = {
      platform: 'darwin',
      runtime: 'electron',
      runAtomic: async () => ({ engine: 'pytorch', status: 'completed', results: [] }),
      getOpenAISettings: async () => ({ configured: true, source: 'secure-storage', encryptionAvailable: true }),
      askLabo: async () => ({
        summary: 'Add a ReLU after final normalization.',
        addedBlocks: [{ atomId: 'relu', nodeId: 'auto-relu', reason: 'Requested activation.' }],
        createdBlocks: [],
        connections: [{ sourceId: 'final-norm', sourcePortId: 'output', targetId: 'auto-relu', targetPortId: 'hidden', reason: 'Feed final states.' }],
        missingBlocks: [],
        warnings: ['The model stopped after a usable partial plan.'],
      }),
    }
    render(<App />)
  
    fireEvent.click(screen.getByRole('button', { name: 'Open LABO settings' }))
    fireEvent.click(screen.getByRole('button', { name: 'Agent' }))
    fireEvent.click(screen.getByRole('button', { name: 'Auto apply' }))
    fireEvent.click(screen.getByRole('button', { name: 'Close Ask LABO' }))
    fireEvent.change(screen.getByLabelText('What should these blocks build?'), { target: { value: 'Add a ReLU automatically' } })
    fireEvent.click(screen.getByRole('button', { name: 'Propose graph changes' }))
  
    expect(await screen.findByRole('button', { name: 'Select ReLU' })).toBeInTheDocument()
    expect(screen.queryByRole('dialog', { name: 'Ask LABO' })).not.toBeInTheDocument()
    expect(window.localStorage.getItem('labo.ask.auto-apply.v1')).toBe('true')
    delete window.labo
  })
  
  it('lets each desktop user save, verify, and remove their own API key', async () => {
    const saveOpenAIKey = vi.fn(async () => ({ configured: true, source: 'secure-storage' as const, encryptionAvailable: true }))
    const deleteOpenAIKey = vi.fn(async () => ({ configured: false, source: 'none' as const, encryptionAvailable: true }))
    const testOpenAIKey = vi.fn(async () => ({ ok: true as const }))
    window.labo = {
      platform: 'darwin',
      runtime: 'electron',
      runAtomic: async () => ({ engine: 'pytorch', status: 'completed', results: [] }),
      getOpenAISettings: async () => ({ configured: false, source: 'none', encryptionAvailable: true }),
      saveOpenAIKey,
      deleteOpenAIKey,
      testOpenAIKey,
    }
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Open LABO settings' }))
    fireEvent.click(screen.getByRole('button', { name: 'Agent' }))
  
    expect(await screen.findByText('No API key configured for this user.')).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('OpenAI API key'), { target: { value: 'sk-project-user-secret-123456789' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save and verify key' }))
    expect(await screen.findByText('Key saved and verified with OpenAI.')).toBeInTheDocument()
    expect(saveOpenAIKey).toHaveBeenCalledWith('sk-project-user-secret-123456789')
    expect(testOpenAIKey).toHaveBeenCalledOnce()
  
    fireEvent.click(screen.getByRole('button', { name: 'Remove key' }))
    fireEvent.click(screen.getByRole('button', { name: 'Confirm removal' }))
    expect(await screen.findByText('API key removed from this user account.')).toBeInTheDocument()
    expect(deleteOpenAIKey).toHaveBeenCalledOnce()
    delete window.labo
  })
})
