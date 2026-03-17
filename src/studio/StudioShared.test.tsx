// @vitest-environment jsdom

import '../test/app-test-setup'
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { StudioContextMenu } from './StudioContextMenu'
import { fitContextMenuToViewport } from './contextMenuPosition'
import { StudioLibrary } from './StudioShell'

describe('shared studio surfaces', () => {
  it('keeps context menus inside every viewport edge', () => {
    expect(fitContextMenuToViewport({ x: 790, y: 590 }, { width: 206, height: 130 }, { width: 800, height: 600 })).toEqual({ x: 586, y: 462 })
    expect(fitContextMenuToViewport({ x: -20, y: -12 }, { width: 206, height: 130 }, { width: 800, height: 600 })).toEqual({ x: 8, y: 8 })
  })

  it('portals the shared context menu to the document body', () => {
    const host = document.createElement('div')
    document.body.append(host)
    render(<StudioContextMenu position={{ x: 40, y: 50 }}><button type="button">Edit</button></StudioContextMenu>, { container: host })

    expect(screen.getByRole('menu').parentElement).toBe(document.body)
  })

  it('gives every studio library the same internal scrolling surface', () => {
    const { container } = render(<StudioLibrary heading="BLOCK LIBRARY" icon={<span>icon</span>}><section>Cards</section></StudioLibrary>)

    expect(container.querySelector('.block-library > .panel-heading')).toBeInTheDocument()
    expect(container.querySelector('.block-library > .studio-library-scroll')).toHaveTextContent('Cards')
  })
})
