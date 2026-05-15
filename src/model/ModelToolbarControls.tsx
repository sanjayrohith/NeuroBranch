import { Blocks, Code2, Cpu, PanelLeft, Pause, Pencil, Play, Square, StepForward } from 'lucide-react'
import type { AtomicPlayerSnapshot } from '../core/atomic-player'

export function ModelInteractionSwitcher({ createCardOpen, interactionMode, onCreateCard, onInteractionMode }: { createCardOpen: boolean; interactionMode: 'add' | 'edit'; onCreateCard(): void; onInteractionMode(mode: 'add' | 'edit'): void }) {
  return <div aria-label="Canvas interaction mode" className="interaction-switcher">
    <button aria-pressed={!createCardOpen && interactionMode === 'add'} onClick={() => onInteractionMode('add')} type="button"><Blocks size={13} />Add blocks</button>
    <button aria-pressed={!createCardOpen && interactionMode === 'edit'} onClick={() => onInteractionMode('edit')} type="button"><Pencil size={13} />Edit cards</button>
    <button aria-pressed={createCardOpen} onClick={onCreateCard} title="Open the dedicated reusable card workspace" type="button"><Code2 size={13} />Reusable card</button>
  </div>
}

export function ModelPlayerControls({ blankGraph, nativePyTorchRuntime, onArrange, onPause, onPlay, onStep, onStop, runtimeAvailable, scope = 'model', snapshot }: { blankGraph: boolean; nativePyTorchRuntime: boolean; onArrange(): void; onPause(): void; onPlay(): void; onStep(): void; onStop(): void; runtimeAvailable: boolean; scope?: 'model' | 'reusable card'; snapshot: AtomicPlayerSnapshot }) {
  return <div aria-label={`${scope === 'model' ? 'Model' : 'Reusable card'} atomic player`} className="atomic-player-controls">
    <button aria-label="Auto-arrange graph" disabled={blankGraph} onClick={onArrange} title="Arrange execution levels and parallel branches" type="button"><span aria-hidden="true">XY</span></button>
    <button aria-label={`Play ${scope} atoms`} disabled={!runtimeAvailable} onClick={onPlay} title={blankGraph ? 'Add a card before running the graph' : nativePyTorchRuntime ? 'Run local PyTorch' : 'Preview typed graph execution'} type="button"><Play size={13} /></button>
    <button aria-label={`Pause ${scope} atoms`} disabled={!runtimeAvailable} onClick={onPause} type="button"><Pause size={13} /></button>
    <button aria-label={`Step one ${scope} atom`} disabled={!runtimeAvailable} onClick={onStep} title={blankGraph ? 'Add a card before stepping through the graph' : nativePyTorchRuntime ? 'Step through local PyTorch' : 'Step through the typed graph preview'} type="button"><StepForward size={13} /></button>
    <button aria-label={`Stop ${scope} atoms`} disabled={!runtimeAvailable} onClick={onStop} type="button"><Square size={12} /></button>
    <span className={`player-status status-${snapshot.status}`} title={nativePyTorchRuntime ? 'Local PyTorch runtime' : 'Browser graph preview'}>{blankGraph ? 'waiting' : nativePyTorchRuntime ? snapshot.status : `preview · ${snapshot.status}`}</span>
  </div>
}

export function ModelPanelControls({ inspectorOpen, libraryOpen, onInspectorToggle, onLibraryToggle }: { inspectorOpen: boolean; libraryOpen: boolean; onInspectorToggle(): void; onLibraryToggle(): void }) {
  return <>
    <button aria-pressed={libraryOpen} className="panel-visibility-button" onClick={onLibraryToggle} type="button"><PanelLeft size={13} />Library</button>
    <button aria-pressed={inspectorOpen} className="panel-visibility-button" onClick={onInspectorToggle} type="button"><Cpu size={13} />Inspector</button>
  </>
}
