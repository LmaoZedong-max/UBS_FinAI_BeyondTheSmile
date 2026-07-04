// Typed API client for the Beyond the Smile backend (docs/API_CONTRACT.md).
// All endpoints are relative "/api/..." — proxied by Vite to :8000 in dev.

export interface ForecastPoint {
  date: string
  rv_realized: number | null
  rv_forecast_harx: number | null
  rv_forecast_gbm: number | null
}

export interface ForecastsResponse {
  factor: string
  series: ForecastPoint[]
}

export interface EvalRow {
  model: string
  n_oos: number
  QLIKE_daily: number | null
  Corr_daily: number | null
  QLIKE_week: number | null
  Corr_week: number | null
}

export interface EvalResponse {
  factor: string
  rows: EvalRow[]
}

export type ShapModel = 'HAR-X' | 'GBM'

export interface ShapDriver {
  feature: string
  shap_value: number
}

export interface ShapResponse {
  date: string
  factor: string
  model: ShapModel
  drivers: ShapDriver[]
}

// --- SHAP timeseries (v2) ---
export interface ShapTimeseriesPoint {
  period: string
  values: Record<string, number>
}

export interface ShapTimeseriesResponse {
  factor: string
  model: ShapModel
  freq: string
  features: string[]
  series: ShapTimeseriesPoint[]
}

// --- Sentiment (v2) ---
export interface SentimentRow {
  date: string
  doc_id: string
  label: string
  sent_score: number
  confidence: number
}

export interface SentimentResponse {
  rows: SentimentRow[]
}

// --- Alerts ---
export interface AlertMeta {
  date: string
  filename: string
}

export interface AlertDetail {
  date: string
  text: string
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export class ApiError extends Error {
  status: number
  detail: string

  constructor(status: number, detail: string) {
    super(`API ${status}: ${detail}`)
    this.status = status
    this.detail = detail
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init)
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      if (body && typeof body.detail === 'string') detail = body.detail
    } catch {
      // non-JSON error body — keep statusText
    }
    throw new ApiError(res.status, detail)
  }
  return (await res.json()) as T
}

function qs(params: Record<string, string | number | undefined>): string {
  const sp = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== '') sp.set(k, String(v))
  }
  const s = sp.toString()
  return s ? `?${s}` : ''
}

export function getHealth(): Promise<{ status: string }> {
  return request('/api/health')
}

export async function getFactors(): Promise<string[]> {
  const data = await request<{ factors: string[] }>('/api/factors')
  return data.factors
}

export function getForecasts(
  factor: string,
  start?: string,
  end?: string,
): Promise<ForecastsResponse> {
  return request(`/api/forecasts${qs({ factor, start, end })}`)
}

export function getEval(factor: string): Promise<EvalResponse> {
  return request(`/api/eval${qs({ factor })}`)
}

export function getShap(
  factor: string,
  date: string,
  model: ShapModel,
  k = 8,
): Promise<ShapResponse> {
  return request(`/api/shap${qs({ factor, date, model, k })}`)
}

export async function getShapDates(factor: string): Promise<string[]> {
  const data = await request<{ dates: string[] }>(
    `/api/shap/dates${qs({ factor })}`,
  )
  return data.dates
}

export async function getAlerts(): Promise<AlertMeta[]> {
  const data = await request<{ alerts: AlertMeta[] }>('/api/alerts')
  return data.alerts
}

export function getAlert(date: string): Promise<AlertDetail> {
  return request(`/api/alerts/${date}`)
}

export function getShapTimeseries(
  factor: string,
  model: ShapModel,
  top_k = 6,
  freq = 'M',
): Promise<ShapTimeseriesResponse> {
  return request(`/api/shap/timeseries${qs({ factor, model, top_k, freq })}`)
}

export async function getSentiment(): Promise<SentimentRow[]> {
  const data = await request<SentimentResponse>('/api/sentiment')
  return data.rows
}

export async function postChat(messages: ChatMessage[]): Promise<string> {
  const data = await request<{ reply: string }>('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages }),
  })
  return data.reply
}

// --- Streaming chat (v3) ---

export interface StreamChatCallbacks {
  /** Called for each tool invocation event before the final answer. */
  onTool: (name: string) => void
  /** Called with each incremental text chunk from the assistant. */
  onDelta: (text: string) => void
  /** Called when the stream finishes cleanly. */
  onDone: () => void
  /** Called when an error event is received from the server. */
  onError: (detail: string) => void
}

/**
 * Pure SSE frame parser — extracted for testability.
 *
 * Accepts an iterable of raw text chunks (as would come from a decoded
 * ReadableStream) and dispatches the appropriate callback for each complete
 * SSE frame. The public behaviour of streamChat is unchanged — it simply
 * delegates to this helper.
 *
 * @returns the leftover (incomplete) buffer after all complete frames have
 *   been processed, so callers can carry it forward across chunk boundaries.
 */
export function parseSSEChunks(
  chunks: string[],
  callbacks: StreamChatCallbacks,
  initialBuffer = '',
): string {
  let buffer = initialBuffer
  for (const chunk of chunks) {
    buffer += chunk

    // SSE frames are separated by double newline
    const frames = buffer.split('\n\n')
    // Last element may be an incomplete frame — keep it in the buffer
    buffer = frames.pop() ?? ''

    for (const frame of frames) {
      if (!frame.trim()) continue

      let eventName = 'message'
      let dataLine = ''

      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) {
          eventName = line.slice('event:'.length).trim()
        } else if (line.startsWith('data:')) {
          dataLine = line.slice('data:'.length).trim()
        }
      }

      if (!dataLine) continue

      let payload: Record<string, unknown>
      try {
        payload = JSON.parse(dataLine) as Record<string, unknown>
      } catch {
        continue
      }

      if (eventName === 'tool') {
        callbacks.onTool(typeof payload.name === 'string' ? payload.name : '')
      } else if (eventName === 'delta') {
        callbacks.onDelta(typeof payload.text === 'string' ? payload.text : '')
      } else if (eventName === 'done') {
        callbacks.onDone()
      } else if (eventName === 'error') {
        callbacks.onError(typeof payload.detail === 'string' ? payload.detail : 'Unknown error')
      }
    }
  }
  return buffer
}

/**
 * Stream a chat turn via POST /api/chat/stream.
 * Parses SSE frames from the ReadableStream manually.
 * Throws (rejects) only if the fetch itself fails or the response is not an
 * event-stream — callers should fall back to postChat in that case.
 */
export async function streamChat(
  messages: ChatMessage[],
  callbacks: StreamChatCallbacks,
): Promise<void> {
  const res = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages }),
  })

  if (!res.ok || !res.body || !res.headers.get('content-type')?.includes('text/event-stream')) {
    // Signal to caller that streaming is unavailable
    throw new Error(`stream unavailable: ${res.status} ${res.statusText}`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    const chunk = decoder.decode(value, { stream: true })
    buffer = parseSSEChunks([chunk], callbacks, buffer)
  }
}
