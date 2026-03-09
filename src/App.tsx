import { useEffect, useLayoutEffect, useMemo, useState } from 'react'
import './App.scss'
import { ModelStudio } from './model/ModelStudio'
import { TokenizerStudio } from './TokenizerStudio'
import { TrainingStudio } from './training/TrainingStudio'
import { searchModelCards, searchReusableCardAtoms } from './model/card-search'
import { searchOptimizers, searchTokenizerCards } from './studio-search'
import type { OptimizerDefinition } from './core/optimizer-ir'
import type { CustomTokenizerCard } from './tokenizer/custom-tokenizer-card'
import { AppHeader, type StudioWorkspaceId } from './studio/AppHeader'
import { GlobalSearch, type GlobalSearchResult } from './studio/GlobalSearch'
import { useGlobalSearch } from './studio/useGlobalSearch'
import './styles/desktop.scss'
import { initializeLaboTheme } from './studio/application-appearance'
import { initializeLaboLanguage } from './studio/application-language'

function App() {
  const [workspace, setWorkspace] = useState<StudioWorkspaceId>('model')
  const [modelEditorContext, setModelEditorContext] = useState<'architecture-add' | 'architecture-edit' | 'reusable-card'>('architecture-add')
  const [askOpen, setAskOpen] = useState(false)
  const searchDisabled = workspace === 'model' && modelEditorContext === 'architecture-edit'
  const search = useGlobalSearch(!searchDisabled)
  const [requestedCard, setRequestedCard] = useState<{ atomId: string; requestId: number }>()
  const [requestedOptimizer, setRequestedOptimizer] = useState<{ optimizerId: string; requestId: number }>()
  const [requestedTokenizerCard, setRequestedTokenizerCard] = useState<{ cardId: string; kind: 'atom' | 'custom'; requestId: number }>()
  const [customOptimizers, setCustomOptimizers] = useState<OptimizerDefinition[]>([])
  const [customTokenizerCards, setCustomTokenizerCards] = useState<CustomTokenizerCard[]>([])
  const [nativeFullScreen, setNativeFullScreen] = useState(false)
  const searchResults = useMemo(() => workspace === 'model'
    ? (modelEditorContext === 'reusable-card' ? searchReusableCardAtoms(search.query) : searchModelCards(search.query)).map((result) => ({ ...result, id: result.atomId, kind: 'model' as const }))
    : workspace === 'training' ? searchOptimizers(search.query, customOptimizers) : searchTokenizerCards(search.query, customTokenizerCards), [customOptimizers, customTokenizerCards, modelEditorContext, search.query, workspace])
  const platform = window.labo?.platform
  const runtimeClass = window.labo?.runtime === 'electron' ? ` runtime-electron runtime-${platform ?? 'desktop'}` : ''
  const searchShortcut = platform === 'darwin' ? '⌘K' : 'Ctrl+K'

  useLayoutEffect(() => { void Promise.all([initializeLaboTheme(), initializeLaboLanguage()]) }, [])

  useEffect(() => {
    let active = true
    void window.labo?.getWindowState?.().then((state) => { if (active) setNativeFullScreen(state.fullScreen) })
    const unsubscribe = window.labo?.onWindowStateChange?.((state) => setNativeFullScreen(state.fullScreen))
    return () => { active = false; unsubscribe?.() }
  }, [])

  return <main className={`app-shell workspace-${workspace}${runtimeClass}${nativeFullScreen ? ' native-fullscreen' : ''}`}>
    <AppHeader onOpenSearch={search.openSearch} onOpenSettings={() => setAskOpen((current) => !current)} onWorkspaceChange={(next) => { setWorkspace(next); setAskOpen(false) }} searchDisabled={searchDisabled} searchLabel={`Search ${workspace === 'model' ? modelEditorContext === 'reusable-card' ? 'reusable card atoms' : 'model cards' : workspace === 'training' ? 'optimizers' : 'tokenizer cards'}`} searchShortcut={searchShortcut} settingsOpen={askOpen} workspace={workspace} />

    <div className="studio-host" hidden={workspace !== 'model'}><ModelStudio askOpen={askOpen} onCloseAsk={() => setAskOpen(false)} onEditorContextChange={setModelEditorContext} onRequestedCardHandled={() => setRequestedCard(undefined)} requestedCard={requestedCard} /></div>
    {workspace === 'training' && <TrainingStudio onCatalogChange={setCustomOptimizers} onRequestedOptimizerHandled={() => setRequestedOptimizer(undefined)} requestedOptimizer={requestedOptimizer} />}
    {workspace === 'tokenizer' && <TokenizerStudio onCatalogChange={setCustomTokenizerCards} onRequestedCardHandled={() => setRequestedTokenizerCard(undefined)} requestedCard={requestedTokenizerCard} />}

    {search.open && <GlobalSearch onClose={search.close} onQueryChange={search.setQuery} onSelect={(result: GlobalSearchResult) => {
      if (result.kind === 'model') setRequestedCard({ atomId: result.id, requestId: Date.now() })
      else if (result.kind === 'optimizer') setRequestedOptimizer({ optimizerId: result.id, requestId: Date.now() })
      else setRequestedTokenizerCard({ cardId: result.id, kind: result.kind === 'tokenizer-atom' ? 'atom' : 'custom', requestId: Date.now() })
      search.close()
    }} placeholder={workspace === 'model' ? modelEditorContext === 'reusable-card' ? 'Search card atoms: Linear projection, GELU, Image encoder…' : 'Search cards: Token embedding, Causal SDPA, LM head…' : workspace === 'training' ? 'Search optimizers: AdamW, Muon, SGD…' : 'Search tokenizers: o200k_base, Image VQ, Audio…'} query={search.query} results={searchResults} />}
  </main>
}

export default App
