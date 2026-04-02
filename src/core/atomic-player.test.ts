import { describe, expect, it } from 'vitest'
import { AtomicPlayer } from './atomic-player'

describe('AtomicPlayer', () => {
  it('continues atom by atom until completion when every atom passes', async () => {
    const executed: string[] = []
    const player = new AtomicPlayer(['normalize', 'pretokenize', 'bpe'], async (atomId) => {
      executed.push(atomId)
      return { summary: `${atomId} ok` }
    })

    await player.play()

    expect(executed).toEqual(['normalize', 'pretokenize', 'bpe'])
    expect(player.snapshot.status).toBe('completed')
    expect(player.snapshot.results.map((result) => result.status)).toEqual(['passed', 'passed', 'passed'])
  })

  it('stays on the failing atom and never executes following atoms', async () => {
    const executed: string[] = []
    const player = new AtomicPlayer(['q-proj', 'sdpa', 'output'], async (atomId) => {
      executed.push(atomId)
      if (atomId === 'sdpa') throw new Error('key shape mismatch')
      return { summary: `${atomId} ok` }
    })

    await player.play()

    expect(executed).toEqual(['q-proj', 'sdpa'])
    expect(player.snapshot.status).toBe('failed')
    expect(player.snapshot.currentAtomId).toBe('sdpa')
    expect(player.snapshot.error).toBe('key shape mismatch')
    expect(player.snapshot.results[2]?.status).toBe('pending')
  })

  it('steps exactly one atom and then remains paused', async () => {
    const executed: string[] = []
    const player = new AtomicPlayer(['q', 'k'], async (atomId) => {
      executed.push(atomId)
      return { summary: 'ok' }
    })

    await player.step()

    expect(executed).toEqual(['q'])
    expect(player.snapshot.status).toBe('paused')
    expect(player.snapshot.currentAtomId).toBe('k')
  })

  it('executes independent atoms in the same parallel level', async () => {
    const executed: string[] = []
    const player = new AtomicPlayer([['embedding', 'fixed-routes'], ['attention-norm']], async (atomId) => {
      executed.push(atomId)
      return { summary: `${atomId} ok` }
    })

    await player.step()

    expect(executed).toEqual(['embedding', 'fixed-routes'])
    expect(player.snapshot.results.slice(0, 2).map((result) => result.status)).toEqual(['passed', 'passed'])
    expect(player.snapshot.currentAtomId).toBe('attention-norm')
    expect(player.snapshot.status).toBe('paused')
  })

  it('finishes all levels in diagnostic mode even when one architecture fails', async () => {
    const executed: string[] = []
    const player = new AtomicPlayer([['broken-start', 'healthy-start'], ['broken-end', 'healthy-end']], async (atomId) => {
      executed.push(atomId)
      if (atomId.startsWith('broken')) throw new Error('broken architecture')
      return { summary: 'ok' }
    }, { continueAfterFailure: true })

    await player.play()

    expect(executed).toEqual(['broken-start', 'healthy-start', 'broken-end', 'healthy-end'])
    expect(player.snapshot.status).toBe('failed')
    expect(player.snapshot.results.find((result) => result.atomId === 'healthy-end')).toMatchObject({ status: 'passed' })
    expect(player.snapshot.results.some((result) => result.status === 'pending')).toBe(false)
  })

  it('reruns the complete plan after completion', async () => {
    const executed: string[] = []
    let restarts = 0
    const player = new AtomicPlayer(['q', 'k'], async (atomId) => {
      executed.push(atomId)
      return { summary: 'ok' }
    }, { onRestart: () => { restarts += 1 } })

    await player.play()
    await player.play()

    expect(executed).toEqual(['q', 'k', 'q', 'k'])
    expect(restarts).toBe(1)
    expect(player.snapshot.status).toBe('completed')
  })

  it('restarts from the first level when stepping after completion', async () => {
    const executed: string[] = []
    const player = new AtomicPlayer(['q', 'k'], async (atomId) => {
      executed.push(atomId)
      return { summary: 'ok' }
    })

    await player.play()
    await player.step()

    expect(executed).toEqual(['q', 'k', 'q'])
    expect(player.snapshot.status).toBe('paused')
    expect(player.snapshot.currentAtomId).toBe('k')
    expect(player.snapshot.results.map((result) => result.status)).toEqual(['passed', 'pending'])
  })

  it('restarts from the beginning when play is pressed after stop', async () => {
    const executed: string[] = []
    let restarts = 0
    const player = new AtomicPlayer(['q'], async (atomId) => {
      executed.push(atomId)
      return { summary: 'ok' }
    }, { onRestart: () => { restarts += 1 } })

    player.stop()
    await player.play()

    expect(executed).toEqual(['q'])
    expect(restarts).toBe(1)
    expect(player.snapshot.status).toBe('completed')
  })

  it('clears completed and in-flight execution state when stopped', async () => {
    let releaseExecution: (() => void) | undefined
    const player = new AtomicPlayer(['q', 'k'], (atomId) => atomId === 'q'
      ? new Promise<{ summary: string }>((resolve) => { releaseExecution = () => resolve({ summary: 'late result' }) })
      : Promise.resolve({ summary: 'ok' }))

    const playing = player.play()
    expect(player.snapshot.results[0].status).toBe('running')

    player.stop()
    expect(player.snapshot.status).toBe('idle')
    expect(player.snapshot.results.map((result) => result.status)).toEqual(['pending', 'pending'])

    releaseExecution?.()
    await playing
    expect(player.snapshot.status).toBe('idle')
    expect(player.snapshot.results.map((result) => result.status)).toEqual(['pending', 'pending'])
  })
})
