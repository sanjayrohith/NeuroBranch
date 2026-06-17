// @vitest-environment jsdom

import './test/app-test-setup'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import App from './App'
import { blankStarterPreset } from './core/presets'
import { researchBpePreset } from './core/tokenizer-presets'

describe('LABO AI shell web', () => {
  it('reserves the native macOS titlebar area only inside Electron', () => {
    window.labo = { platform: 'darwin', runtime: 'electron', runAtomic: async () => ({ engine: 'pytorch', status: 'completed', results: [] }) }
  
    render(<App />)
  
    expect(document.querySelector('.app-shell')).toHaveClass('runtime-electron')
    delete window.labo
  })
  
  it('removes the macOS traffic-light offset in native fullscreen', async () => {
    let publish: ((state: { fullScreen: boolean }) => void) | undefined
    window.labo = {
      platform: 'darwin', runtime: 'electron',
      runAtomic: async () => ({ engine: 'pytorch', status: 'completed', results: [] }),
      getWindowState: async () => ({ fullScreen: false }),
      onWindowStateChange: (callback) => { publish = callback; return () => undefined },
    }
  
    render(<App />)
    await waitFor(() => expect(publish).toBeTypeOf('function'))
    publish?.({ fullScreen: true })
  
    await waitFor(() => expect(document.querySelector('.app-shell')).toHaveClass('native-fullscreen'))
    delete window.labo
  })
  
  it('uses native Windows chrome and Windows keyboard labels', () => {
    window.labo = { platform: 'win32', runtime: 'electron', runAtomic: async () => ({ engine: 'pytorch', status: 'completed', results: [] }) }
  
    render(<App />)
  
    expect(document.querySelector('.app-shell')).toHaveClass('runtime-electron', 'runtime-win32')
    expect(document.querySelector('.app-shell')).not.toHaveClass('runtime-darwin')
    expect(screen.getByRole('button', { name: 'Search model cards' })).toHaveTextContent('Ctrl+K')
    delete window.labo
  })
  
  it('offers a typed graph preview in a browser renderer', async () => {
    delete window.labo
    render(<App />)
  
    expect(screen.getByRole('button', { name: 'Play model atoms' })).toBeEnabled()
    expect(screen.getByRole('button', { name: 'Step one model atom' })).toBeEnabled()
    expect(screen.getByText('preview · idle')).toBeInTheDocument()
  
    fireEvent.click(screen.getByRole('button', { name: 'Step one model atom' }))
    await waitFor(() => expect(screen.getByText('preview · paused')).toBeInTheDocument())
  })

  it('closes card search when its backdrop is clicked', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Search model cards' }))

    const dialog = screen.getByRole('dialog', { name: 'Search cards' })
    expect(dialog).toBeInTheDocument()
    fireEvent.pointerDown(dialog.parentElement!)

    expect(screen.queryByRole('dialog', { name: 'Search cards' })).not.toBeInTheDocument()
  })
  
  it('keeps workspace data on the account and a local fallback for visual preferences', async () => {
    const saveWebWorkspace = vi.fn(async () => ({ saved: true as const, updatedAt: Date.now() }))
    const setItem = vi.spyOn(Storage.prototype, 'setItem')
    window.labo = {
      platform: 'web',
      runtime: 'web',
      loadWebWorkspace: async () => ({ authenticated: true, workspace: null, customCards: [], settings: { appearance: { theme: 'complexity-spectrum' } } }),
      saveWebWorkspace,
    }
  
    render(<App />)
    await waitFor(() => expect(document.documentElement).toHaveAttribute('data-labo-theme', 'complexity-spectrum'))
    fireEvent.click(screen.getByRole('button', { name: 'Open LABO settings' }))
    fireEvent.click(screen.getByRole('button', { name: 'Application' }))
    fireEvent.click(screen.getByRole('button', { name: 'Use LABO Dark theme' }))
    await waitFor(() => expect(saveWebWorkspace).toHaveBeenCalledWith(expect.objectContaining({ settings: { appearance: { theme: 'labo-dark' } } })))
    fireEvent.click(screen.getByRole('button', { name: 'Workspaces' }))
    fireEvent.click(screen.getByRole('button', { name: 'Create and open a blank workspace' }))
  
    await waitFor(() => expect(saveWebWorkspace).toHaveBeenCalled(), { timeout: 2_000 })
    expect(setItem).toHaveBeenCalledWith('labo.web.preferences.v1', expect.stringContaining('labo-dark'))
    delete window.labo
    setItem.mockRestore()
  })
  
  it('keeps a guest workspace ephemeral but preserves its visual preference locally', async () => {
    const saveWebWorkspace = vi.fn(async () => ({ saved: true as const, updatedAt: Date.now() }))
    window.labo = {
      platform: 'web',
      runtime: 'web',
      loadWebWorkspace: async () => ({ authenticated: false, workspace: null, customCards: [] }),
      saveWebWorkspace,
    }
  
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Open LABO settings' }))
    fireEvent.click(screen.getByRole('button', { name: 'Application' }))
    fireEvent.click(screen.getByRole('button', { name: 'Use Complexity Spectrum theme' }))
    await waitFor(() => expect(window.localStorage.getItem('labo.web.preferences.v1')).toContain('complexity-spectrum'))
    fireEvent.click(screen.getByRole('button', { name: 'Workspaces' }))
    fireEvent.click(screen.getByRole('button', { name: 'Create and open a blank workspace' }))
    await new Promise((resolve) => setTimeout(resolve, 850))
  
    expect(saveWebWorkspace).not.toHaveBeenCalled()
    delete window.labo
  })
  
  it('restores and saves account-scoped Training and Tokenizer workspaces on the web', async () => {
    const saveWebWorkspace = vi.fn(async () => ({ saved: true as const, updatedAt: Date.now() }))
    window.labo = {
      platform: 'web',
      runtime: 'web',
      loadWebWorkspace: async () => ({
        authenticated: true,
        workspace: null,
        customCards: [],
        training: {
          config: { id: 'web-muon-1', kind: 'web-muon', settings: { lr: 0.002 } },
          customOptimizers: [{ id: 'web-muon', label: 'Private Web Muon', torchClass: 'Muon', defaults: { lr: 0.002 } }],
          updatedAt: 1,
        },
        tokenizer: {
          pipeline: researchBpePreset,
          customCards: [{ id: 'private-web-tokenizer', label: 'Private Web Tokenizer', category: 'Transform', pythonCode: 'text = text.lower()' }],
          updatedAt: 1,
        },
      }),
      saveWebWorkspace,
    }
  
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Training Studio' }))
    expect(await screen.findByRole('button', { name: 'Use Private Web Muon' })).toBeInTheDocument()
    await waitFor(() => expect(saveWebWorkspace).toHaveBeenCalledWith(expect.objectContaining({ training: expect.any(Object) })))
  
    fireEvent.click(screen.getByRole('button', { name: 'Tokenizer Studio' }))
    expect(await screen.findByRole('button', { name: 'Add Private Web Tokenizer' })).toBeInTheDocument()
    await waitFor(() => expect(saveWebWorkspace).toHaveBeenCalledWith(expect.objectContaining({ tokenizer: expect.any(Object) })))
  })
  
  it('documents edit-mode lasso selection and full-graph deletion in Settings tips', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Open LABO settings' }))
    fireEvent.click(screen.getByRole('button', { name: 'Tips' }))
  
    const tip = screen.getByText('Delete several cards or a full graph').closest('article')
    expect(tip).toHaveTextContent('Switch to Edit cards, then drag on empty canvas around the cards')
    expect(tip).toHaveTextContent('Connected elastics are removed with the cards')
  })
  
  it('never lets a late web restore delete a card added from search', async () => {
    let resolveWorkspace: ((value: { authenticated: boolean; workspace: unknown; customCards: unknown[] }) => void) | undefined
    window.labo = {
      platform: 'web', runtime: 'web',
      loadWebWorkspace: () => new Promise((resolve) => { resolveWorkspace = resolve }),
      saveWebWorkspace: async () => ({ saved: true, updatedAt: Date.now() }),
    }
    render(<App />)
  
    fireEvent.click(screen.getByRole('button', { name: 'Search model cards' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'Natural language card search' }), { target: { value: 'decode generated logits token' } })
    fireEvent.click(within(screen.getByRole('dialog', { name: 'Search cards' })).getByRole('button', { name: /^Greedy token decoder/ }))
    expect(screen.getByRole('button', { name: 'Select Greedy token decoder' })).toBeInTheDocument()
  
    resolveWorkspace?.({
      authenticated: true,
      workspace: { activePresetId: blankStarterPreset.id, drafts: { [blankStarterPreset.id]: { graph: blankStarterPreset, selectedNodeId: '' } }, userPresets: [], updatedAt: 1 },
      customCards: [],
    })
    await waitFor(() => expect(screen.getByRole('button', { name: 'Select Greedy token decoder' })).toBeInTheDocument())
    delete window.labo
  })
})
