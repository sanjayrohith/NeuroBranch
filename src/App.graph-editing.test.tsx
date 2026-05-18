// @vitest-environment jsdom

import './test/app-test-setup'
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import App from './App'

describe('LABO AI graph editing', () => {
  it('clears the purple card selection when clicking empty canvas space', () => {
    render(<App />)
    const card = screen.getByRole('button', { name: 'Select Attention RMSNorm' }).closest('.architecture-node')

    fireEvent.click(screen.getByRole('button', { name: 'Select Attention RMSNorm' }))
    expect(card).toHaveClass('selected')

    fireEvent.pointerDown(screen.getByLabelText('Architecture graph canvas'), { button: 0, pointerId: 1, clientX: 900, clientY: 500 })
    expect(card).not.toHaveClass('selected')
  })

  it('selects, edits, and adds freely manipulable model atoms', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Split' }))
  
    fireEvent.click(screen.getByRole('button', { name: 'Select Attention RMSNorm' }))
    const selectedCard = screen.getByRole('button', { name: 'Select Attention RMSNorm' }).closest('.architecture-node')
    expect(selectedCard).not.toHaveClass('editing')
    expect(selectedCard?.querySelector('.block-inline-editor')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Edit cards' }))
    fireEvent.doubleClick(screen.getByRole('button', { name: 'Select Attention RMSNorm' }))
    expect(screen.getByRole('spinbutton', { name: 'Model card setting epsilon' })).toBeInTheDocument()
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Model card setting epsilon' }), { target: { value: '0.00001' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }))
    expect((screen.getByRole('textbox', { name: 'PyTorch editor' }) as HTMLTextAreaElement).value).toContain('self.attention_norm = nn.RMSNorm(1024, eps=1e-05)')
  
    fireEvent.click(screen.getByRole('button', { name: 'Add blocks' }))
    fireEvent.click(screen.getByText(/Activations/))
    fireEvent.click(screen.getByRole('button', { name: 'Add ReLU' }))
    expect(screen.getByText('21 atoms')).toBeInTheDocument()
    expect(screen.getByText('Atomic PyTorch draft')).toBeInTheDocument()
    const relu = screen.getByRole('button', { name: 'Select ReLU' })
    expect(relu.closest('.architecture-node')).toHaveAttribute('data-atom-id', 'relu')
    const updatedCode = (screen.getByRole('textbox', { name: 'PyTorch editor' }) as HTMLTextAreaElement).value
    expect(updatedCode).toContain('# labo:node=relu-1 atom=relu')
    expect(updatedCode).toContain('self.relu_1 = nn.ReLU(inplace=False)')
    expect(updatedCode).not.toContain('relu_1_output = self.relu_1(')
  }, 15_000)
  
  it('drags a library card onto the requested graph position', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Blank starter' }))
    fireEvent.click(screen.getByText(/Activations/))
  
    const values = new Map<string, string>()
    let dragImage: Element | undefined
    const dataTransfer = {
      dropEffect: 'none',
      effectAllowed: 'none',
      getData: (type: string) => values.get(type) ?? '',
      setData: (type: string, value: string) => values.set(type, value),
      setDragImage: (element: Element) => { dragImage = element },
      get types() { return [...values.keys()] },
    }
    const libraryCard = screen.getByRole('button', { name: 'Add ReLU' })
    const canvas = screen.getByLabelText('Architecture graph canvas')
  
    expect(libraryCard).toHaveAttribute('draggable', 'true')
    fireEvent.dragStart(libraryCard, { dataTransfer })
    expect(dragImage).toHaveClass('library-drag-preview')
    expect(dragImage).toHaveTextContent('ReLU')
    fireEvent.dragOver(canvas, { clientX: 740, clientY: 260, dataTransfer })
    expect(canvas).toHaveClass('accepts-library-drop')
    const dropEvent = new Event('drop', { bubbles: true, cancelable: true })
    Object.defineProperties(dropEvent, {
      clientX: { value: 740 },
      clientY: { value: 260 },
      dataTransfer: { value: dataTransfer },
    })
    fireEvent(canvas, dropEvent)
  
    const droppedCard = screen.getByRole('button', { name: 'Select ReLU' }).closest('.architecture-node')
    expect(droppedCard).toHaveStyle({ left: '666px', top: '222px' })
    expect(canvas).not.toHaveClass('accepts-library-drop')
    expect(screen.getByText('1 atoms')).toBeInTheDocument()
  })
  
  it('keeps an exact manual library drop without snapping or moving the camera', () => {
    render(<App />)
    fireEvent.click(screen.getByText(/Activations/))
    const target = screen.getByRole('button', { name: 'Select Attention RMSNorm' }).closest<HTMLElement>('.architecture-node')!
    const targetPosition = { x: Number.parseFloat(target.style.left), y: Number.parseFloat(target.style.top) }
    const canvas = screen.getByLabelText('Architecture graph canvas')
    const world = screen.getByTestId('graph-world')
    const viewportBefore = world.style.transform
    const values = new Map<string, string>([['application/x-labo-model-atom', 'relu']])
    const dataTransfer = {
      dropEffect: 'copy', effectAllowed: 'copy',
      getData: (type: string) => values.get(type) ?? '',
      setData: (type: string, value: string) => values.set(type, value),
      get types() { return [...values.keys()] },
    }
    const dropEvent = new Event('drop', { bubbles: true, cancelable: true })
    Object.defineProperties(dropEvent, {
      clientX: { value: targetPosition.x + 74 },
      clientY: { value: targetPosition.y + 38 },
      dataTransfer: { value: dataTransfer },
    })
    fireEvent(canvas, dropEvent)
  
    const dropped = screen.getByRole('button', { name: 'Select ReLU' }).closest<HTMLElement>('.architecture-node')!
    expect(dropped).toHaveStyle({ left: `${targetPosition.x}px`, top: `${targetPosition.y}px` })
    expect(world.style.transform).toBe(viewportBefore)
  })
  
  it('exposes distinct semantic ports for routing and expert merging', () => {
    render(<App />)
    expect(document.querySelector('[data-port-id="fixed-routes-expertIndices-output"]')).toHaveAttribute('data-port-key', 'expertIndices')
    expect(document.querySelector('[data-port-id="fixed-routes-expertWeights-output"]')).toHaveAttribute('data-port-key', 'expertWeights')
    expect(document.querySelector('[data-port-id="branch-gates-routed-input"]')).toHaveAttribute('data-port-key', 'routed')
    expect(document.querySelector('[data-port-id="branch-gates-shared-input"]')).toHaveAttribute('data-port-key', 'shared')
  })
  
  it('unplugs an elastic cable by dragging its connected input into empty canvas', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Split' }))
  
    const qkvInput = screen.getByRole('button', { name: 'qkv input H' })
    const elementFromPoint = document.elementFromPoint
    Object.defineProperty(document, 'elementFromPoint', { configurable: true, value: () => null })
    fireEvent.pointerDown(qkvInput, { clientX: 80, clientY: 160 })
    fireEvent.pointerMove(window, { clientX: 20, clientY: 300 })
    fireEvent.pointerUp(window, { clientX: 20, clientY: 300 })
    Object.defineProperty(document, 'elementFromPoint', { configurable: true, value: elementFromPoint })
  
    expect(screen.getByText(/20 nodes · 30 links/)).toBeInTheDocument()
    expect(screen.getByText('Cable disconnected')).toBeInTheDocument()
    expect(screen.getByText('Neural IR invalid')).toBeInTheDocument()
    expect(screen.getByText(/Graph incomplete · 1 wiring issue/)).toBeInTheDocument()
    const code = (screen.getByRole('textbox', { name: 'PyTorch editor' }) as HTMLTextAreaElement).value
    expect(code).toContain('class GeneratedModel(nn.Module):')
    expect(code).not.toContain('class GeneratedInvalidGraph')
    expect(code).toContain('self.attention_norm = nn.RMSNorm')
    expect(code).toContain('self.qkv_q =')
    expect(code).not.toContain('qkv_q = self.qkv_q(')
    expect(code).not.toContain('# labo:edge=attention-norm-qkv')
    expect(code).not.toContain('qkv_hidden: torch.Tensor')
  })
  
  it('previews and confirms Ask LABO elastics between existing blocks', async () => {
    window.labo = {
      platform: 'darwin',
      runtime: 'electron',
      runAtomic: async () => ({ engine: 'pytorch', status: 'completed', results: [] }),
      getOpenAISettings: async () => ({ configured: true, source: 'secure-storage', encryptionAvailable: true }),
      askLabo: async () => ({
        summary: 'Reconnect the attention path.',
        addedBlocks: [],
        createdBlocks: [],
        connections: [{
          sourceId: 'attention-norm', sourcePortId: 'output', targetId: 'qkv', targetPortId: 'hidden', reason: 'QKV needs normalized hidden states.',
        }],
        missingBlocks: [],
        warnings: [],
      }),
    }
    render(<App />)
  
    const elementFromPoint = document.elementFromPoint
    Object.defineProperty(document, 'elementFromPoint', { configurable: true, value: () => null })
    fireEvent.pointerDown(screen.getByRole('button', { name: 'qkv input H' }), { clientX: 80, clientY: 160 })
    fireEvent.pointerUp(window, { clientX: 20, clientY: 300 })
    Object.defineProperty(document, 'elementFromPoint', { configurable: true, value: elementFromPoint })
    expect(screen.getByText(/20 nodes · 30 links/)).toBeInTheDocument()
  
    fireEvent.change(screen.getByLabelText('What should these blocks build?'), { target: { value: 'Reconnect the attention blocks' } })
    fireEvent.click(screen.getByRole('button', { name: 'Propose graph changes' }))
  
    expect(await screen.findByText('1 elastic ready')).toBeInTheDocument()
    expect(screen.getByText('attention-norm.output')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Apply full graph plan' }))
    expect(screen.getByText(/20 nodes · 31 links/)).toBeInTheDocument()
    expect(screen.queryByRole('dialog', { name: 'Ask LABO' })).not.toBeInTheDocument()
    delete window.labo
  })
  
  it('pans and zooms the graph viewport without changing graph coordinates', () => {
    render(<App />)
    const canvas = screen.getByLabelText('Architecture graph canvas')
    const world = screen.getByTestId('graph-world')
  
    expect(world).toHaveStyle({ transform: 'translate(0px, 0px) scale(1)' })
    fireEvent.wheel(canvas, { deltaX: 45, deltaY: 30 })
    expect(world).toHaveStyle({ transform: 'translate(-45px, -30px) scale(1)' })
  
    fireEvent.click(screen.getByRole('button', { name: 'Zoom in' }))
    expect(screen.getByRole('button', { name: 'Reset zoom' })).toHaveTextContent('110%')
    fireEvent.click(screen.getByRole('button', { name: 'Zoom out' }))
    expect(screen.getByRole('button', { name: 'Reset zoom' })).toHaveTextContent('100%')
    expect(screen.getByRole('button', { name: 'Fit graph' })).toBeInTheDocument()
  })
  
  it('moves a stacked model card with a direct pointer drag', () => {
    render(<App />)
    const canvas = screen.getByLabelText('Architecture graph canvas')
    const handle = screen.getByRole('button', { name: 'Select Attention RMSNorm' })
    const card = handle.closest('.architecture-node') as HTMLElement
    const initialLeft = Number.parseFloat(card.style.left)
    const initialTop = Number.parseFloat(card.style.top)
  
    fireEvent.pointerDown(handle, { button: 0, pointerId: 7, clientX: 120, clientY: 180 })
    fireEvent.pointerMove(canvas, { pointerId: 7, clientX: 320, clientY: 275 })
  
    expect(card).toHaveClass('dragging')
    expect(Number.parseFloat(card.style.left)).toBe(initialLeft + 200)
    expect(Number.parseFloat(card.style.top)).toBe(initialTop + 95)
  
    fireEvent.pointerUp(canvas, { pointerId: 7, clientX: 320, clientY: 275 })
    expect(card).not.toHaveClass('dragging')
    expect(Number.parseFloat(card.style.left)).toBe(initialLeft + 200)
    expect(Number.parseFloat(card.style.top)).toBe(initialTop + 95)
  })
  
  it('keeps the viewport and exact card position under manual pointer placement', () => {
    render(<App />)
    const canvas = screen.getByLabelText('Architecture graph canvas')
    const world = screen.getByTestId('graph-world')
    const handle = screen.getByRole('button', { name: 'Select Attention RMSNorm' })
    const card = handle.closest('.architecture-node') as HTMLElement
    const initialLeft = Number.parseFloat(card.style.left)
    const initialTop = Number.parseFloat(card.style.top)
    const viewportBefore = world.style.transform
  
    fireEvent.pointerDown(handle, { button: 0, pointerId: 11, clientX: 120, clientY: 180 })
    fireEvent.pointerMove(canvas, { pointerId: 11, clientX: 120, clientY: 270 })
    fireEvent.pointerUp(canvas, { pointerId: 11, clientX: 120, clientY: 270 })
  
    expect(Number.parseFloat(card.style.left)).toBe(initialLeft)
    expect(Number.parseFloat(card.style.top)).toBe(initialTop + 90)
    expect(world.style.transform).toBe(viewportBefore)
  })
  
  it('keeps elastic endpoints stable when selecting a fixed-size card', () => {
    const originalRect = HTMLElement.prototype.getBoundingClientRect
    HTMLElement.prototype.getBoundingClientRect = function () {
      const portId = this.getAttribute?.('data-port-id') ?? ''
      const x = portId.includes('qkv-') ? 220 : 160
      const y = portId.includes('qkv-') ? 320 : 160
      return { x, y, left: x, top: y, right: x + 16, bottom: y + 16, width: 16, height: 16, toJSON: () => ({}) } as DOMRect
    }
    try {
      render(<App />)
      const qkvCable = document.querySelector('[data-edge-id="attention-norm-qkv"]')
      const before = qkvCable?.getAttribute('d')
      fireEvent.click(screen.getByRole('button', { name: 'Select GQA QKV projection' }))
      expect(screen.getByRole('button', { name: 'Select GQA QKV projection' }).closest('.architecture-node')).not.toHaveClass('editing')
      expect(qkvCable?.getAttribute('d')).toBe(before)
    } finally {
      HTMLElement.prototype.getBoundingClientRect = originalRect
    }
  })
  
  it('never clips circular plugs at atom boundaries', () => {
    render(<App />)
    const embeddingCard = screen.getByRole('button', { name: 'Select Tied token embedding' }).closest('.architecture-node')
    const routedCard = screen.getByRole('button', { name: 'Select 4 × routed SwiGLU' }).closest('.architecture-node')
    expect(embeddingCard).toHaveStyle({ overflow: 'visible' })
    expect(routedCard).toHaveStyle({ overflow: 'visible' })
  })
  
  it('deletes a model atom without recreating it in PyTorch', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Split' }))
  
    fireEvent.click(screen.getByRole('button', { name: 'Edit cards' }))
    fireEvent.doubleClick(screen.getByRole('button', { name: 'Select GQA QKV projection' }))
    expect(screen.getByRole('dialog', { name: 'Edit model card' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Delete card' }))
  
    const code = (screen.getByRole('textbox', { name: 'PyTorch editor' }) as HTMLTextAreaElement).value
    expect(code).not.toContain('self.qkv_q =')
    expect(code).toContain('class GeneratedModel(nn.Module):')
    expect(code).not.toContain('class GeneratedInvalidGraph')
    expect(code).not.toContain('self.head_layout')
    expect(code).not.toContain('head_layout_q: torch.Tensor')
    expect(screen.getByText('Neural IR invalid')).toBeInTheDocument()
    expect(screen.getByText(/Graph incomplete · \d+ wiring issues?/)).toBeInTheDocument()
  })
  
  it('opens card edit and guarded delete actions from the graph context menu', () => {
    render(<App />)
  
    const card = screen.getByRole('button', { name: 'Select Attention RMSNorm' })
    fireEvent.contextMenu(card.closest('[data-graph-node="true"]')!)
    expect(screen.getByRole('menu')).toHaveTextContent('Attention RMSNorm')
    fireEvent.pointerDown(screen.getByRole('button', { name: 'Blocks' }))
    expect(screen.queryByRole('menu')).not.toBeInTheDocument()
  
    fireEvent.contextMenu(card.closest('[data-graph-node="true"]')!)
    fireEvent.click(screen.getByRole('menuitem', { name: 'Edit card' }))
    expect(screen.getByRole('dialog', { name: 'Edit model card' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Close model card editor' }))
  
    fireEvent.contextMenu(card.closest('[data-graph-node="true"]')!)
    fireEvent.click(screen.getByRole('menuitem', { name: 'Delete card' }))
    expect(screen.getByRole('menuitem', { name: 'Confirm delete' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('menuitem', { name: 'Confirm delete' }))
    expect(screen.queryByRole('button', { name: 'Select Attention RMSNorm' })).not.toBeInTheDocument()
  })
  
  it('applies supported PyTorch edits back to the same model atom', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Split' }))
  
    const editor = screen.getByRole('textbox', { name: 'PyTorch editor' })
    fireEvent.change(editor, { target: { value: (editor as HTMLTextAreaElement).value.replace('self.attention_norm = nn.RMSNorm(1024, eps=1e-06)', 'self.attention_norm = nn.RMSNorm(1024, eps=1e-05)') } })
    fireEvent.click(screen.getByRole('button', { name: 'Apply PyTorch to blocks' }))
    fireEvent.click(screen.getByRole('button', { name: 'Select Attention RMSNorm' }))
    fireEvent.click(screen.getByRole('button', { name: 'Edit cards' }))
    fireEvent.doubleClick(screen.getByRole('button', { name: 'Select Attention RMSNorm' }))
  
    expect(screen.getByRole('spinbutton', { name: 'Model card setting epsilon' })).toHaveValue(0.00001)
  })
})
