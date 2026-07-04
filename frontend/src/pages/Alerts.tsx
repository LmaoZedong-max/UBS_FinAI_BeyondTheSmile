import { useEffect, useState } from 'react'
import { getAlert, getAlerts, getSentiment, type AlertMeta, type SentimentRow } from '../api'
import { EmptyNote, ErrorNote, Panel, Spinner } from '../components/ui'

function sentimentBadgeClass(label: string): string {
  const l = label.toLowerCase()
  if (l === 'negative') return 'bg-red-900/40 text-red-300 border border-red-800/50'
  if (l === 'positive') return 'bg-green-900/40 text-green-300 border border-green-800/50'
  return 'bg-panel text-muted border border-edge'
}

export default function Alerts() {
  const [alerts, setAlerts] = useState<AlertMeta[]>([])
  const [listError, setListError] = useState('')
  const [listLoading, setListLoading] = useState(true)

  const [selected, setSelected] = useState('')
  const [text, setText] = useState('')
  const [textLoading, setTextLoading] = useState(false)
  const [textError, setTextError] = useState('')

  // Sentiment state
  const [sentRows, setSentRows] = useState<SentimentRow[]>([])
  const [sentLoading, setSentLoading] = useState(true)
  const [sentError, setSentError] = useState('')

  useEffect(() => {
    getSentiment()
      .then((rows) => setSentRows(rows))
      .catch((e) => setSentError(e instanceof Error ? e.message : String(e)))
      .finally(() => setSentLoading(false))
  }, [])

  useEffect(() => {
    getAlerts()
      .then((a) => {
        setAlerts(a)
        if (a.length > 0) setSelected(a[a.length - 1].date)
      })
      .catch((e) => setListError(e instanceof Error ? e.message : String(e)))
      .finally(() => setListLoading(false))
  }, [])

  useEffect(() => {
    if (!selected) return
    setTextLoading(true)
    setTextError('')
    getAlert(selected)
      .then((r) => setText(r.text))
      .catch((e) => {
        setText('')
        setTextError(e instanceof Error ? e.message : String(e))
      })
      .finally(() => setTextLoading(false))
  }, [selected])

  return (
    <div className="flex flex-col gap-4 p-5">
      {/* News Sentiment strip */}
      <Panel title="News Sentiment">
        {sentLoading ? (
          <Spinner label="Loading sentiment…" />
        ) : sentError ? (
          <ErrorNote message={sentError} />
        ) : sentRows.length === 0 ? (
          <EmptyNote message="No sentiment data available." />
        ) : (
          <div className="flex flex-wrap gap-2">
            {sentRows.map((row, i) => (
              <div
                key={`${row.doc_id}-${i}`}
                className="flex min-w-[180px] flex-col gap-1 rounded-sm border border-edge bg-page px-3 py-2.5"
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="font-mono text-[10px] text-muted">{row.date}</span>
                  <span
                    className={`rounded-sm px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-wide ${sentimentBadgeClass(row.label)}`}
                  >
                    {row.label}
                  </span>
                </div>
                <div className="truncate font-mono text-xs text-ink" title={row.doc_id}>
                  {row.doc_id}
                </div>
                <div className="flex gap-3 font-mono text-[10px] text-muted">
                  <span>score <span className="text-ink">{row.sent_score.toFixed(2)}</span></span>
                  <span>conf <span className="text-ink">{row.confidence.toFixed(2)}</span></span>
                </div>
              </div>
            ))}
          </div>
        )}
      </Panel>

      {/* Alert list + reader */}
      <div className="grid h-full grid-cols-1 gap-4 lg:grid-cols-[280px_1fr]">
      {/* Alert list */}
      <Panel title="Risk Alerts" className="self-start">
        {listLoading ? (
          <Spinner label="Loading alerts…" />
        ) : listError ? (
          <ErrorNote message={listError} />
        ) : alerts.length === 0 ? (
          <EmptyNote message="No alerts found." />
        ) : (
          <ul className="space-y-2">
            {[...alerts].reverse().map((a) => (
              <li key={a.date}>
                <button
                  onClick={() => setSelected(a.date)}
                  className={`w-full rounded-sm border border-l-2 px-3 py-2.5 text-left transition-colors ${
                    selected === a.date
                      ? 'border-edge border-l-ubs bg-page'
                      : 'border-transparent border-l-edge bg-page/40 hover:bg-page'
                  }`}
                >
                  <div className="font-mono text-sm text-ink">{a.date}</div>
                  <div className="mt-0.5 truncate font-mono text-[10px] text-muted">
                    {a.filename}
                  </div>
                </button>
              </li>
            ))}
          </ul>
        )}
      </Panel>

      {/* Reader pane */}
      <Panel
        title={selected ? `Alert — ${selected}` : 'Alert Reader'}
        className="min-h-[400px]"
      >
        {textLoading ? (
          <Spinner label="Loading alert…" />
        ) : textError ? (
          <ErrorNote message={textError} />
        ) : !selected ? (
          <EmptyNote message="Select an alert to read it." />
        ) : (
          <pre className="whitespace-pre-wrap font-mono text-[13px] leading-relaxed text-ink">
            {text}
          </pre>
        )}
      </Panel>
      </div>
    </div>
  )
}
