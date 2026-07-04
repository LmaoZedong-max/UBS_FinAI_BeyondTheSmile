/**
 * Unit tests for frontend/src/api.ts
 *
 * Covers:
 *   1. parseSSEChunks — the pure SSE frame parser extracted from streamChat.
 *      Tests the split-chunk reassembly, multiple-frames-per-chunk ordering,
 *      the full tool→delta→done sequence, error events, and JSON parse errors.
 *
 *   2. streamChat — integration tests via a mocked global fetch returning a
 *      ReadableStream. Verifies the non-event-stream / fetch-rejection paths
 *      that the public API must reject on.
 *
 *   3. request() — mocked fetch for 404 / 422 / 503 → ApiError, plus happy
 *      path returning typed JSON (tested through exported wrappers).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { parseSSEChunks, streamChat, ApiError, getHealth, getFactors } from '../api'
import type { StreamChatCallbacks } from '../api'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeCallbacks(): StreamChatCallbacks {
  return {
    onTool: vi.fn(),
    onDelta: vi.fn(),
    onDone: vi.fn(),
    onError: vi.fn(),
  }
}

/** Build a ReadableStream from an array of pre-encoded Uint8Array chunks. */
function makeStream(chunks: Uint8Array[]): ReadableStream<Uint8Array> {
  let i = 0
  return new ReadableStream<Uint8Array>({
    pull(controller) {
      if (i < chunks.length) {
        controller.enqueue(chunks[i++])
      } else {
        controller.close()
      }
    },
  })
}

const enc = new TextEncoder()

/** Encode a string to Uint8Array. */
function e(s: string): Uint8Array {
  return enc.encode(s)
}

// ---------------------------------------------------------------------------
// 1. parseSSEChunks — pure parser
// ---------------------------------------------------------------------------

describe('parseSSEChunks', () => {
  it('reassembles a frame split mid-line across two chunks', () => {
    const cb = makeCallbacks()
    // Frame arrives as two chunks: "event: del" then "ta\ndata: {...}\n\n"
    const leftover1 = parseSSEChunks(['event: del'], cb, '')
    expect(leftover1).toBe('event: del') // incomplete, held in buffer
    const leftover2 = parseSSEChunks(
      ['ta\ndata: {"text":"hi"}\n\n'],
      cb,
      leftover1,
    )
    expect(leftover2).toBe('')
    expect(cb.onDelta).toHaveBeenCalledOnce()
    expect(cb.onDelta).toHaveBeenCalledWith('hi')
  })

  it('dispatches multiple frames arriving in one chunk in order', () => {
    const cb = makeCallbacks()
    const chunk =
      'event: delta\ndata: {"text":"A"}\n\n' +
      'event: delta\ndata: {"text":"B"}\n\n' +
      'event: delta\ndata: {"text":"C"}\n\n'
    parseSSEChunks([chunk], cb)
    expect(cb.onDelta).toHaveBeenCalledTimes(3)
    const calls = (cb.onDelta as ReturnType<typeof vi.fn>).mock.calls
    expect(calls.map((c: unknown[]) => c[0])).toEqual(['A', 'B', 'C'])
  })

  it('fires onTool → onDelta → onDone in sequence with correct payloads', () => {
    const cb = makeCallbacks()
    const order: string[] = []
    ;(cb.onTool as ReturnType<typeof vi.fn>).mockImplementation(() => order.push('tool'))
    ;(cb.onDelta as ReturnType<typeof vi.fn>).mockImplementation(() => order.push('delta'))
    ;(cb.onDone as ReturnType<typeof vi.fn>).mockImplementation(() => order.push('done'))

    const stream =
      'event: tool\ndata: {"name":"search"}\n\n' +
      'event: delta\ndata: {"text":"result"}\n\n' +
      'event: done\ndata: {}\n\n'
    parseSSEChunks([stream], cb)

    expect(order).toEqual(['tool', 'delta', 'done'])
    expect(cb.onTool).toHaveBeenCalledWith('search')
    expect(cb.onDelta).toHaveBeenCalledWith('result')
    expect(cb.onDone).toHaveBeenCalledOnce()
  })

  it('fires onError with the detail string from an error event', () => {
    const cb = makeCallbacks()
    parseSSEChunks(
      ['event: error\ndata: {"detail":"something went wrong"}\n\n'],
      cb,
    )
    expect(cb.onError).toHaveBeenCalledOnce()
    expect(cb.onError).toHaveBeenCalledWith('something went wrong')
  })

  it('falls back to "Unknown error" when error event has no detail field', () => {
    const cb = makeCallbacks()
    parseSSEChunks(['event: error\ndata: {"code":500}\n\n'], cb)
    expect(cb.onError).toHaveBeenCalledWith('Unknown error')
  })

  it('silently skips frames whose data is invalid JSON', () => {
    const cb = makeCallbacks()
    parseSSEChunks(
      [
        'event: delta\ndata: not-json\n\n' +
          'event: delta\ndata: {"text":"ok"}\n\n',
      ],
      cb,
    )
    // Only the valid frame should produce a callback
    expect(cb.onDelta).toHaveBeenCalledOnce()
    expect(cb.onDelta).toHaveBeenCalledWith('ok')
  })

  it('ignores frames with no data line', () => {
    const cb = makeCallbacks()
    // comment-only frame (no data:) followed by a valid frame
    parseSSEChunks([': keep-alive\n\nevent: done\ndata: {}\n\n'], cb)
    expect(cb.onDone).toHaveBeenCalledOnce()
    expect(cb.onDelta).not.toHaveBeenCalled()
  })

  it('returns remaining incomplete buffer without dispatching', () => {
    const cb = makeCallbacks()
    const leftover = parseSSEChunks(
      ['event: delta\ndata: {"text":"partial'],
      cb,
    )
    expect(leftover).toBe('event: delta\ndata: {"text":"partial')
    expect(cb.onDelta).not.toHaveBeenCalled()
  })

  it('handles empty string chunks without throwing', () => {
    const cb = makeCallbacks()
    expect(() => parseSSEChunks(['', '', ''], cb)).not.toThrow()
    expect(cb.onDelta).not.toHaveBeenCalled()
  })
})

// ---------------------------------------------------------------------------
// 2. streamChat — fetch-level integration (mocked fetch)
// ---------------------------------------------------------------------------

describe('streamChat', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('rejects when fetch itself throws (network error)', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Network failure')))
    const cb = makeCallbacks()
    await expect(
      streamChat([{ role: 'user', content: 'hi' }], cb),
    ).rejects.toThrow('Network failure')
  })

  it('rejects when response is not an event-stream (non-ok status)', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response('Internal server error', {
          status: 503,
          headers: { 'content-type': 'text/plain' },
        }),
      ),
    )
    const cb = makeCallbacks()
    await expect(
      streamChat([{ role: 'user', content: 'hi' }], cb),
    ).rejects.toThrow('stream unavailable')
  })

  it('rejects when content-type is not text/event-stream even on 200', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ reply: 'hi' }), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        }),
      ),
    )
    const cb = makeCallbacks()
    await expect(
      streamChat([{ role: 'user', content: 'hi' }], cb),
    ).rejects.toThrow('stream unavailable')
  })

  it('dispatches callbacks for a complete SSE stream', async () => {
    const sseBody =
      'event: delta\ndata: {"text":"hello"}\n\n' +
      'event: done\ndata: {}\n\n'

    const stream = makeStream([e(sseBody)])
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(stream, {
          status: 200,
          headers: { 'content-type': 'text/event-stream' },
        }),
      ),
    )
    const cb = makeCallbacks()
    await streamChat([{ role: 'user', content: 'hi' }], cb)
    expect(cb.onDelta).toHaveBeenCalledWith('hello')
    expect(cb.onDone).toHaveBeenCalledOnce()
  })

  it('handles frames split across multiple stream chunks', async () => {
    // "event: delta\n" arrives in chunk 1, "data: {\"text\":\"x\"}\n\n" in chunk 2
    const stream = makeStream([
      e('event: delta\n'),
      e('data: {"text":"x"}\n\n'),
    ])
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(stream, {
          status: 200,
          headers: { 'content-type': 'text/event-stream' },
        }),
      ),
    )
    const cb = makeCallbacks()
    await streamChat([{ role: 'user', content: 'q' }], cb)
    expect(cb.onDelta).toHaveBeenCalledWith('x')
  })
})

// ---------------------------------------------------------------------------
// 3. request() — via exported convenience wrappers (mocked fetch)
// ---------------------------------------------------------------------------

describe('request() error handling', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('throws ApiError with status 404 and statusText as detail when body is not JSON', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response('Not Found', {
          status: 404,
          statusText: 'Not Found',
          headers: { 'content-type': 'text/plain' },
        }),
      ),
    )
    const err = await getHealth().catch((e: unknown) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect((err as ApiError).status).toBe(404)
    expect((err as ApiError).detail).toBe('Not Found')
  })

  it('throws ApiError with status 422 and JSON detail field', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: 'Validation failed' }), {
          status: 422,
          statusText: 'Unprocessable Entity',
          headers: { 'content-type': 'application/json' },
        }),
      ),
    )
    const err = await getHealth().catch((e: unknown) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect((err as ApiError).status).toBe(422)
    expect((err as ApiError).detail).toBe('Validation failed')
  })

  it('throws ApiError with status 503', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response('Service Unavailable', {
          status: 503,
          statusText: 'Service Unavailable',
          headers: { 'content-type': 'text/plain' },
        }),
      ),
    )
    const err = await getHealth().catch((e: unknown) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect((err as ApiError).status).toBe(503)
  })

  it('returns typed JSON on 200 — getHealth happy path', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ status: 'ok' }), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        }),
      ),
    )
    const result = await getHealth()
    expect(result).toEqual({ status: 'ok' })
  })

  it('returns typed JSON on 200 — getFactors unwraps .factors array', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({ factors: ['CNH_ATM_PC1', 'CNH_RR_PC1'] }),
          {
            status: 200,
            headers: { 'content-type': 'application/json' },
          },
        ),
      ),
    )
    const factors = await getFactors()
    expect(factors).toEqual(['CNH_ATM_PC1', 'CNH_RR_PC1'])
  })
})
