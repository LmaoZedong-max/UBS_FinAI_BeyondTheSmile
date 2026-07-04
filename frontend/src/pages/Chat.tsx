import { useEffect, useRef, useState } from 'react'
import { ApiError, postChat, streamChat, type ChatMessage } from '../api'
import { Spinner } from '../components/ui'

export default function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  // Streaming state
  const [streamingContent, setStreamingContent] = useState<string | null>(null)
  const [toolStatus, setToolStatus] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
  }, [messages, pending, streamingContent, toolStatus])

  async function send() {
    const content = input.trim()
    if (!content || pending) return
    const next: ChatMessage[] = [...messages, { role: 'user', content }]
    setMessages(next)
    setInput('')
    setPending(true)
    setError('')
    setStreamingContent('')
    setToolStatus(null)

    // Attempt streaming path first
    let streamSucceeded = false
    try {
      let accumulated = ''
      await streamChat(next, {
        onTool: (name) => {
          setToolStatus(`consulting data store… (${name})`)
        },
        onDelta: (text) => {
          setToolStatus(null)
          accumulated += text
          setStreamingContent(accumulated)
        },
        onDone: () => {
          // Commit the streamed bubble to messages
          setMessages([...next, { role: 'assistant', content: accumulated }])
          setStreamingContent(null)
          setToolStatus(null)
        },
        onError: (detail) => {
          setStreamingContent(null)
          setToolStatus(null)
          if (detail === 'DEEPSEEK_API_KEY not configured') {
            setError('Chat backend not configured - set DEEPSEEK_API_KEY')
          } else {
            setError(detail)
          }
        },
      })
      streamSucceeded = true
    } catch {
      // Streaming unavailable - fall back to non-streaming path
      setStreamingContent(null)
      setToolStatus(null)
    }

    if (!streamSucceeded) {
      // Fallback: existing non-streaming sendChat
      try {
        const reply = await postChat(next)
        setMessages([...next, { role: 'assistant', content: reply }])
      } catch (e) {
        if (e instanceof ApiError && e.status === 503) {
          setError('Chat backend not configured - set DEEPSEEK_API_KEY')
        } else {
          setError(e instanceof Error ? e.message : String(e))
        }
      }
    }

    setPending(false)
  }

  // The assistant bubble currently being streamed
  const isStreaming = streamingContent !== null

  return (
    <div className="flex h-full flex-col">
      {/* Message list */}
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto p-5">
        {messages.length === 0 && !pending && (
          <div className="mt-16 text-center">
            <div className="text-sm text-muted">
              Ask about USD/CNY–CNH volatility, model forecasts, SHAP drivers or
              risk alerts.
            </div>
            <div className="mt-2 font-mono text-xs text-muted">
              e.g. &quot;What drove CNH ATM vol on 2025-12-16?&quot;
            </div>
          </div>
        )}
        <div className="mx-auto max-w-3xl space-y-3">
          {messages.map((m, i) => (
            <div
              key={i}
              className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] whitespace-pre-wrap rounded-sm border px-3.5 py-2.5 text-sm leading-relaxed ${
                  m.role === 'user'
                    ? 'border-ubs/40 bg-ubs/15 text-ink'
                    : 'border-edge bg-panel text-ink'
                }`}
              >
                {m.content}
              </div>
            </div>
          ))}

          {/* Live streaming bubble */}
          {pending && (
            <div className="flex justify-start">
              {toolStatus && !isStreaming && (
                <div className="mb-1 text-xs text-muted italic">{toolStatus}</div>
              )}
              {isStreaming ? (
                <div className="flex max-w-[80%] flex-col gap-1">
                  {toolStatus && (
                    <div className="text-xs text-muted italic">{toolStatus}</div>
                  )}
                  <div className="whitespace-pre-wrap rounded-sm border border-edge bg-panel px-3.5 py-2.5 text-sm leading-relaxed text-ink">
                    {streamingContent}
                    <span className="ml-0.5 inline-block h-3.5 w-px animate-pulse bg-ink align-middle" />
                  </div>
                </div>
              ) : (
                <div className="rounded-sm border border-edge bg-panel px-3.5">
                  <Spinner label="Thinking…" />
                </div>
              )}
            </div>
          )}

          {error && (
            <div className="flex justify-start">
              <div className="max-w-[80%] rounded-sm border border-edge bg-page px-3.5 py-2.5 font-mono text-xs text-muted">
                <span className="mr-2 font-semibold text-ubs">ERR</span>
                {error}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Input pinned to bottom */}
      <div className="shrink-0 border-t border-edge bg-panel p-4">
        <div className="mx-auto flex max-w-3xl gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') send()
            }}
            placeholder="Message the research assistant…"
            className="flex-1 rounded-sm border border-edge bg-page px-3.5 py-2.5 text-sm text-ink outline-none placeholder:text-muted focus:border-ubs"
          />
          <button
            onClick={send}
            disabled={pending || !input.trim()}
            className="rounded-sm bg-ubs px-5 py-2.5 text-sm font-semibold text-white transition-opacity disabled:opacity-40"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  )
}
