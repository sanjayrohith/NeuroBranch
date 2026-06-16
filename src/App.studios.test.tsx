// @vitest-environment jsdom

import './test/app-test-setup'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import App from './App'

describe('LABO AI studios', () => {
  it('shows the source-first desktop updater in the shared settings', async () => {
    window.labo = {
      platform: 'darwin',
      runtime: 'electron',
      runAtomic: async () => ({ engine: 'pytorch', status: 'completed', results: [] }),
      getDesktopUpdateStatus: async () => ({
        currentVersion: '0.1.26',
        channel: 'stable',
        installedTag: 'v0.1.26',
        latestTag: 'v0.1.27',
        helperInstalled: true,
        updateAvailable: true,
        setupUrl: 'https://github.com/Complexity-ML/labo-ai/releases/latest',
      }),
    }
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Open LABO settings' }))

    expect(await screen.findByText('Desktop updates')).toBeInTheDocument()
    expect(screen.getByText('v0.1.26')).toBeInTheDocument()
    expect(screen.getByText('v0.1.27')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Check for updates' })).toBeEnabled()
    expect(screen.getByRole('button', { name: 'Install update' })).toBeEnabled()
    fireEvent.click(screen.getByRole('button', { name: 'Check for updates' }))
    expect(await screen.findByText('v0.1.27 is ready to install.')).toBeInTheDocument()
    delete window.labo
  })

  it('does not reinstall a main commit that is already installed', async () => {
    window.labo = {
      platform: 'darwin',
      runtime: 'electron',
      getDesktopUpdateStatus: async () => ({
        currentVersion: '0.1.45',
        channel: 'main',
        installedTag: 'main@abcdef1',
        latestTag: 'main@abcdef1234567890',
        helperInstalled: true,
        updateAvailable: false,
        setupUrl: 'https://github.com/Complexity-ML/labo-ai/releases/latest',
      }),
    }
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Open LABO settings' }))

    expect(await screen.findByRole('button', { name: 'Up to date' })).toBeDisabled()
    delete window.labo
  })

  it('persists the selected theme in the private SQLite settings scope', async () => {
    const settings = { desktopUpdateChannel: 'main', appearance: { theme: 'complexity-spectrum' } }
    const saveDesktopState = vi.fn(async () => ({ saved: true as const }))
    window.labo = {
      platform: 'darwin',
      runtime: 'electron',
      loadDesktopState: vi.fn(async (scope) => scope === 'settings' ? settings : undefined),
      saveDesktopState,
    }
    render(<App />)

    await waitFor(() => expect(document.documentElement).toHaveAttribute('data-labo-theme', 'complexity-spectrum'))
    fireEvent.click(screen.getByRole('button', { name: 'Open LABO settings' }))
    fireEvent.click(screen.getByRole('button', { name: 'Application' }))
    expect(screen.getByRole('button', { name: 'Use Complexity Spectrum theme' })).toHaveAttribute('aria-pressed', 'true')
    fireEvent.click(screen.getByRole('button', { name: 'Use LABO Dark theme' }))

    await waitFor(() => expect(saveDesktopState).toHaveBeenCalledWith('settings', {
      appearance: { theme: 'labo-dark' },
    }))
  })

  it('persists the interface and agent-plan language in the private SQLite settings scope', async () => {
    const settings = { appearance: { theme: 'labo-dark', language: 'fr' } }
    const saveDesktopState = vi.fn(async () => ({ saved: true as const }))
    window.labo = {
      platform: 'darwin',
      runtime: 'electron',
      loadDesktopState: vi.fn(async (scope) => scope === 'settings' ? settings : undefined),
      saveDesktopState,
    }
    render(<App />)

    await waitFor(() => expect(document.documentElement).toHaveAttribute('lang', 'fr'))
    fireEvent.click(screen.getByRole('button', { name: 'Open LABO settings' }))
    fireEvent.click(screen.getByRole('button', { name: 'Application' }))
    fireEvent.click(screen.getByRole('button', { name: 'EnglishInterface and plans in English' }))

    await waitFor(() => expect(saveDesktopState).toHaveBeenCalledWith('settings', {
      appearance: { language: 'en' },
    }))
    expect(document.documentElement).toHaveAttribute('lang', 'en')
    delete window.labo
  })

  it('documents desktop ChatGPT sign-in as the no-API-key agent path', () => {
    window.labo = { platform: 'darwin', runtime: 'electron' }
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Open LABO settings' }))
    fireEvent.click(screen.getByRole('button', { name: 'Tips' }))

    expect(screen.getByText('Use your ChatGPT account without an API key')).toBeInTheDocument()
    delete window.labo
  })

  it('keeps the desktop ChatGPT sign-in tip visible from the web app', () => {
    window.labo = { platform: 'web', runtime: 'web' }
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Open LABO settings' }))
    fireEvent.click(screen.getByRole('button', { name: 'Tips' }))

    expect(screen.getByText('Use your ChatGPT account without an API key')).toBeInTheDocument()
    expect(screen.getByText(/In the desktop app/)).toBeInTheDocument()
    delete window.labo
  })

  it('keeps stable and main update detection separate when switching channels', async () => {
    const getDesktopUpdateStatus = vi.fn(async (requestedChannel?: DesktopUpdateChannel) => {
      const channel = requestedChannel ?? 'stable'
      return channel === 'main' ? {
        currentVersion: '0.1.45',
        channel,
        installedTag: 'v0.1.45',
        installedChannel: 'stable' as const,
        latestTag: 'main@abcdef1',
        helperInstalled: true,
        updateAvailable: true,
        setupUrl: 'https://github.com/Complexity-ML/labo-ai/releases/latest',
      } : {
        currentVersion: '0.1.45',
        channel,
        installedTag: 'v0.1.45',
        installedChannel: 'stable' as const,
        latestTag: 'v0.1.45',
        helperInstalled: true,
        updateAvailable: false,
        setupUrl: 'https://github.com/Complexity-ML/labo-ai/releases/latest',
      }
    })
    window.labo = { platform: 'darwin', runtime: 'electron', getDesktopUpdateStatus }
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Open LABO settings' }))

    expect(await screen.findByRole('button', { name: 'Up to date' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: /MainExperimental/ }))
    expect(await screen.findByText('main@abcdef1')).toBeInTheDocument()
    expect(screen.getByText('Latest main commit')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /StableRecommended/ }))
    await waitFor(() => expect(screen.queryByText('main@abcdef1')).not.toBeInTheDocument())
    expect(screen.getByText('Latest stable release')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Up to date' })).toBeDisabled()
    delete window.labo
  })

  it('allows returning from Main to the verified Stable release at the same source revision', async () => {
    const revision = 'abcdef1234567890'
    const getDesktopUpdateStatus = vi.fn(async (requestedChannel?: DesktopUpdateChannel) => requestedChannel === 'stable' ? {
      currentVersion: '0.1.47',
      channel: 'stable' as const,
      installedTag: 'main@abcdef1',
      installedChannel: 'main' as const,
      installedRevision: revision,
      latestTag: 'v0.1.47',
      latestRevision: revision,
      helperInstalled: true,
      updateAvailable: true,
      setupUrl: 'https://github.com/Complexity-ML/labo-ai/releases/latest',
    } : {
      currentVersion: '0.1.47',
      channel: 'main' as const,
      installedTag: 'main@abcdef1',
      installedChannel: 'main' as const,
      installedRevision: revision,
      latestTag: 'main@abcdef1',
      latestRevision: revision,
      helperInstalled: true,
      updateAvailable: false,
      setupUrl: 'https://github.com/Complexity-ML/labo-ai/releases/latest',
    })
    window.labo = { platform: 'darwin', runtime: 'electron', getDesktopUpdateStatus }
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Open LABO settings' }))
    expect(await screen.findByRole('button', { name: 'Up to date' })).toBeDisabled()

    fireEvent.click(screen.getByRole('button', { name: /StableRecommended/ }))
    expect(await screen.findByRole('button', { name: 'Switch to Stable' })).toBeEnabled()
    expect(screen.getByText('The v0.1.47 source is already installed from Main. Switch only to return to the verified Stable channel.')).toBeInTheDocument()
  })

  it('labels a return from a newer Main commit as a Stable channel switch, not an update', async () => {
    window.labo = {
      platform: 'darwin',
      runtime: 'electron',
      getDesktopUpdateStatus: async (requestedChannel) => requestedChannel === 'stable' ? {
        currentVersion: '0.1.47', channel: 'stable', installedTag: 'main@a63f876', installedChannel: 'main', installedRevision: 'a63f8760000000',
        latestTag: 'v0.1.47', latestRevision: '761521d0000000', helperInstalled: true, updateAvailable: true, setupUrl: 'https://github.com/Complexity-ML/labo-ai/releases/latest',
      } : {
        currentVersion: '0.1.47', channel: 'main', installedTag: 'main@a63f876', installedChannel: 'main', installedRevision: 'a63f8760000000',
        latestTag: 'main@a63f876', latestRevision: 'a63f8760000000', helperInstalled: true, updateAvailable: false, setupUrl: 'https://github.com/Complexity-ML/labo-ai/releases/latest',
      },
    }
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Open LABO settings' }))
    expect(await screen.findByRole('button', { name: 'Up to date' })).toBeDisabled()

    fireEvent.click(screen.getByRole('button', { name: /StableRecommended/ }))
    expect(await screen.findByRole('button', { name: 'Switch to Stable' })).toBeEnabled()
    expect(screen.getByText('Switch from experimental main@a63f876 to verified Stable v0.1.47.')).toBeInTheDocument()
    expect(screen.queryByText('v0.1.47 is ready to install.')).not.toBeInTheDocument()
  })

  it('opens Training Studio with real AdamW and Muon settings and PyTorch', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Training Studio' }))

    expect(screen.getByRole('button', { name: 'Use AdamW' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Use Muon' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Use Muon' }))
    expect(screen.getByRole('spinbutton', { name: 'Muon momentum' })).toHaveValue(0.95)
    expect(screen.getByRole('button', { name: 'Training graph' })).toHaveAttribute('aria-pressed', 'true')

    fireEvent.click(screen.getByRole('button', { name: 'PyTorch' }))
    expect(screen.getByLabelText('Python code preview')).toHaveTextContent(/torch\.optim\.Muon\(model\.parameters\(\)/)
    expect(screen.queryByText('training.optimizer')).not.toBeInTheDocument()
    expect(screen.getByText('optimizer.py')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Training graph' }))
    expect(screen.getByText('training.optimizer')).toBeInTheDocument()
    expect(screen.queryByText('optimizer.py')).not.toBeInTheDocument()
  })
  
  it('creates a custom optimizer directly in Training Studio', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Training Studio' }))
    fireEvent.click(screen.getByRole('button', { name: 'Create optimizer' }))
  
    const dialog = screen.getByRole('form', { name: 'Create optimizer' })
    fireEvent.change(within(dialog).getByRole('textbox', { name: 'Optimizer name' }), { target: { value: 'Research AdamW' } })
    fireEvent.click(within(dialog).getByRole('button', { name: 'Select Parameter update' }))
    fireEvent.change(within(dialog).getByRole('spinbutton', { name: 'Optimizer learning rate' }), { target: { value: '0.0002' } })
    fireEvent.click(within(dialog).getByRole('button', { name: 'Select Weight decay' }))
    fireEvent.change(within(dialog).getByRole('spinbutton', { name: 'Optimizer weight decay' }), { target: { value: '0.05' } })
    fireEvent.click(within(dialog).getByText('More parameters'))
    fireEvent.click(within(dialog).getByRole('checkbox', { name: 'Optimizer Fused implementation' }))
    fireEvent.click(within(dialog).getByRole('button', { name: 'Create' }))

    expect(screen.queryByRole('form', { name: 'Create optimizer' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Use Research AdamW' })).toBeInTheDocument()
    expect(screen.getByRole('spinbutton', { name: 'Research AdamW lr' })).toHaveValue(0.0002)
    expect(screen.getAllByText('torch.optim.AdamW').length).toBeGreaterThan(0)
    fireEvent.click(screen.getByRole('button', { name: 'PyTorch' }))
    expect(screen.getByLabelText('Python code preview')).toHaveTextContent(/optimizer = torch\.optim\.AdamW\(model\.parameters\(\), lr=0\.0002/)
    expect(screen.getByLabelText('Python code preview')).toHaveTextContent(/weight_decay=0\.05/)
    expect(screen.getByLabelText('Python code preview')).toHaveTextContent(/fused=True/)
    fireEvent.click(screen.getByRole('button', { name: 'Training graph' }))
  
    fireEvent.doubleClick(screen.getByRole('article', { name: 'Optimizer card Research AdamW' }))
    const editor = screen.getByRole('form', { name: 'Edit optimizer' })
    fireEvent.change(within(editor).getByRole('textbox', { name: 'Optimizer name' }), { target: { value: 'Updated AdamW' } })
    fireEvent.click(within(editor).getByRole('button', { name: 'Save' }))
    expect(screen.getByRole('button', { name: 'Use Updated AdamW' })).toBeInTheDocument()
    fireEvent.contextMenu(screen.getByRole('article', { name: 'Optimizer card Updated AdamW' }))
    expect(screen.getByRole('menuitem', { name: 'Edit Updated AdamW' })).toBeInTheDocument()
    expect(screen.getByRole('menuitem', { name: 'Delete Updated AdamW' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('menuitem', { name: 'Edit Updated AdamW' }))
    fireEvent.click(within(screen.getByRole('form', { name: 'Edit optimizer' })).getByRole('button', { name: 'Cancel' }))
  
    fireEvent.click(screen.getByRole('button', { name: 'Open LABO settings' }))
    expect(screen.getByRole('button', { name: 'Training Studio' })).toHaveAttribute('aria-pressed', 'true')
    const settings = screen.getByRole('dialog', { name: 'LABO AI settings' })
    expect(within(settings).getByRole('button', { name: 'General' })).toHaveAttribute('aria-pressed', 'true')
    for (const section of ['General', 'Workspaces', 'Agent', 'Application', 'Tips']) expect(within(settings).getByRole('button', { name: section })).toBeInTheDocument()
    fireEvent.click(within(settings).getByRole('button', { name: 'Application' }))
    expect(within(settings).getByRole('button', { name: 'Use LABO Dark theme' })).toHaveAttribute('aria-pressed', 'true')
    fireEvent.click(within(settings).getByRole('button', { name: 'Use Complexity Spectrum theme' }))
    expect(document.documentElement).toHaveAttribute('data-labo-theme', 'complexity-spectrum')
    fireEvent.pointerDown(document.querySelector('.model-card-modal-backdrop')!)
    expect(screen.queryByRole('dialog', { name: 'LABO AI settings' })).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Tokenizer Studio' }))
    fireEvent.click(screen.getByRole('button', { name: 'Open LABO settings' }))
    const tokenizerSettings = screen.getByRole('dialog', { name: 'LABO AI settings' })
    for (const section of ['General', 'Workspaces', 'Agent', 'Application', 'Tips']) expect(within(tokenizerSettings).getByRole('button', { name: section })).toBeInTheDocument()
    fireEvent.click(within(tokenizerSettings).getByRole('button', { name: 'Application' }))
    expect(within(tokenizerSettings).getByRole('button', { name: 'Use Complexity Spectrum theme' })).toHaveAttribute('aria-pressed', 'true')
  })

  it('composes a real optimizer rule without a redundant edit mode', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Training Studio' }))

    expect(screen.queryByRole('button', { name: 'Edit optimizers' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Use optimizers' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Create optimizer' }))

    const dialog = screen.getByRole('form', { name: 'Create optimizer' })
    fireEvent.click(within(dialog).getByRole('button', { name: 'Add Normalize update' }))
    fireEvent.change(within(dialog).getByRole('textbox', { name: 'Optimizer name' }), { target: { value: 'My LABO rule' } })
    expect(within(dialog).getByRole('button', { name: 'Select Momentum' })).toBeInTheDocument()
    expect(within(dialog).getByRole('button', { name: 'Select Adaptive scale' })).toBeInTheDocument()
    expect(within(dialog).getByRole('button', { name: 'Select Normalize update' })).toBeInTheDocument()
    fireEvent.click(within(dialog).getByRole('button', { name: 'Create' }))

    expect(screen.getByRole('button', { name: 'Use My LABO rule' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'PyTorch' }))
    expect(screen.getByLabelText('Python code preview')).toHaveTextContent(/class LaboOptimizer\(torch\.optim\.Optimizer\):/)
    expect(screen.getByLabelText('Python code preview')).toHaveTextContent(/momentum = state\.setdefault/)
    expect(screen.getByLabelText('Python code preview')).toHaveTextContent(/variance = state\.setdefault/)
    expect(screen.getByLabelText('Python code preview')).toHaveTextContent(/optimizer = LaboOptimizer\(model\.parameters\(\)/)
    expect(screen.getByLabelText('Python code preview')).not.toHaveTextContent(/torch\.optim\.LaboOptimizer/)
  })
  
  it('keeps natural-language search inside the active studio', async () => {
    render(<App />)
  
    fireEvent.click(screen.getByRole('button', { name: 'Training Studio' }))
    fireEvent.click(screen.getByRole('button', { name: 'Search optimizers' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'Natural language card search' }), { target: { value: 'muon momentum' } })
    fireEvent.click(within(screen.getByRole('dialog', { name: 'Search cards' })).getByRole('button', { name: /Muon/ }))
    expect(screen.getByRole('button', { name: 'Training Studio' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('spinbutton', { name: 'Muon momentum' })).toBeInTheDocument()
  
    fireEvent.click(screen.getByRole('button', { name: 'Tokenizer Studio' }))
    fireEvent.click(screen.getByRole('button', { name: 'Search tokenizer cards' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'Natural language card search' }), { target: { value: 'video normalization' } })
    fireEvent.click(within(screen.getByRole('dialog', { name: 'Search cards' })).getByRole('button', { name: /Video normalization/ }))
    expect(screen.getByRole('button', { name: 'Tokenizer Studio' })).toHaveAttribute('aria-pressed', 'true')
    expect(await screen.findByRole('button', { name: 'Select Video normalization' })).toBeInTheDocument()
  })
  
  it('restores custom optimizers from the persistent desktop workspace database', async () => {
    const saveDesktopState = vi.fn(async () => ({ saved: true as const }))
    window.labo = {
      platform: 'darwin',
      runtime: 'electron',
      runAtomic: async () => ({ engine: 'pytorch', status: 'completed', results: [] }),
      loadDesktopState: async (scope) => scope === 'training' ? {
        config: { id: 'custom-persistent-muon-1', kind: 'custom-persistent-muon', settings: { lr: 0.002, momentum: 0.9 } },
        customOptimizers: [{ id: 'custom-persistent-muon', label: 'Persistent Muon', torchClass: 'Muon', defaults: { lr: 0.002, momentum: 0.9 } }],
        updatedAt: 1,
      } : undefined,
      saveDesktopState,
    }
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Training Studio' }))
  
    expect(await screen.findByRole('button', { name: 'Use Persistent Muon' })).toBeInTheDocument()
    expect(screen.getByRole('spinbutton', { name: 'Persistent Muon lr' })).toHaveValue(0.002)
    await waitFor(() => expect(saveDesktopState).toHaveBeenCalledWith('training', expect.objectContaining({ customOptimizers: expect.arrayContaining([expect.objectContaining({ label: 'Persistent Muon' })]) })))
  })
  
  it('opens the atomic Tokenizer Studio and compiles its IR to Python', () => {
    render(<App />)
  
    fireEvent.click(screen.getByRole('button', { name: 'Tokenizer Studio' }))
  
    expect(screen.getByLabelText('Tokenizer preset')).toHaveTextContent('Research BPE')
    expect(screen.getAllByText('Unicode normalization').length).toBeGreaterThanOrEqual(2)
    expect(screen.getAllByText('Byte-level pre-tokenizer').length).toBeGreaterThanOrEqual(2)
    expect(screen.getAllByText('BPE trainer').length).toBeGreaterThanOrEqual(2)
    expect(screen.queryByRole('button', { name: 'Rust' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Split' }))
    expect(screen.getByLabelText('Python code preview')).toHaveTextContent(/vocab_size=32768/)
  
    fireEvent.click(screen.getByRole('button', { name: 'Select BPE trainer' }))
    fireEvent.change(screen.getByRole('spinbutton', { name: 'vocabSize' }), { target: { value: '4096' } })
    expect(screen.getByLabelText('Python code preview')).toHaveTextContent(/vocab_size=4096/)
  
    fireEvent.click(screen.getByRole('button', { name: 'Select Unicode normalization' }))
    fireEvent.click(screen.getByRole('button', { name: 'Delete selected tokenizer atom' }))
    expect(screen.getByLabelText('Python code preview')).not.toHaveTextContent(/tokenizer\.normalizer/)
    expect(screen.getByText(/^4 atoms/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Play atomic pipeline' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Step one atom' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Stop atomic pipeline' })).toBeInTheDocument()
  
    fireEvent.click(screen.getByRole('button', { name: 'o200k_base · OpenAI tiktoken' }))
    expect(screen.getByLabelText('Python code preview')).toHaveTextContent(/tiktoken\.get_encoding\("o200k_base"\)/)
    expect(screen.getByText('200,019')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /New reusable card/ }))
    expect(screen.getByRole('dialog', { name: 'Tokenizer card builder' })).toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: 'Python lowering' })).toBeInTheDocument()
    expect(screen.queryByText(/Rust lowering/)).not.toBeInTheDocument()
  })
  
  it('can delete every tokenizer step and add a real atom from the permanent library', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Tokenizer Studio' }))
  
    for (const label of ['Unicode normalization', 'Byte-level pre-tokenizer', 'BPE model', 'BPE trainer', 'Byte-level decoder']) {
      fireEvent.click(screen.getByRole('button', { name: `Select ${label}` }))
      fireEvent.click(screen.getByRole('button', { name: 'Delete selected tokenizer atom' }))
    }
  
    expect(screen.getByText(/^0 atoms/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Add BPE model' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Add BPE model' }))
    expect(screen.getByRole('button', { name: 'Select BPE model' })).toBeInTheDocument()
    expect(screen.getByText(/^1 atoms/)).toBeInTheDocument()
  })
  
  it('edits and deletes a user tokenizer card from the library context menu', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Tokenizer Studio' }))
    fireEvent.click(screen.getByRole('button', { name: /New reusable card/ }))
    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'Lowercase QA text' } })
    fireEvent.click(screen.getByRole('button', { name: 'Create and add card' }))
  
    const libraryCard = screen.getByRole('button', { name: 'Add Lowercase QA text' })
    fireEvent.contextMenu(libraryCard)
    fireEvent.click(screen.getByRole('menuitem', { name: 'Edit card' }))
    expect(screen.getByRole('dialog', { name: 'Tokenizer card builder' })).toHaveTextContent('Edit reusable tokenizer card')
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  
    fireEvent.contextMenu(libraryCard)
    fireEvent.click(screen.getByRole('menuitem', { name: 'Delete card' }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Confirm delete' }))
    expect(screen.queryByRole('button', { name: 'Add Lowercase QA text' })).not.toBeInTheDocument()
  })
})
