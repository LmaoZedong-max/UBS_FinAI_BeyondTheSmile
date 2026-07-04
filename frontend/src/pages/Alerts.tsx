import { useEffect, useState } from 'react'
import { getAlert, getAlerts, type AlertMeta } from '../api'
import { EmptyNote, ErrorNote, Panel, Spinner } from '../components/ui'

export default function Alerts() {
  const [alerts, setAlerts] = useState<AlertMeta[]>([])
  const [listError, setListError] = useState('')
  const [listLoading, setListLoading] = useState(true)

  const [selected, setSelected] = useState('')
  const [text, setText] = useState('')
  const [textLoading, setTextLoading] = useState(false)
  const [textError, setTextError] = useState('')

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
    <div className="grid h-full grid-cols-1 gap-4 p-5 lg:grid-cols-[280px_1fr]">
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
  )
}
