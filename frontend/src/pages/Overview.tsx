import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getFactors, getEval, type EvalRow } from '../api'
import { Panel, Spinner, ErrorNote } from '../components/ui'

const METHODS = [
  {
    title: 'PCA Factors',
    body: 'Decomposes the full USD/CNY and USD/CNH implied-volatility surface into orthogonal level, skew, and curvature factors.',
  },
  {
    title: 'HAR-X + GBM',
    body: 'Rolling, no-look-ahead realized-volatility forecasts from a HAR-X linear model and a Gradient Boosting Machine, evaluated OOS.',
  },
  {
    title: 'SHAP',
    body: 'Shapley-value attribution assigns each macro driver a signed contribution to each forecast, updated daily.',
  },
  {
    title: 'FinBERT',
    body: 'Financial BERT scores news headlines for sentiment (positive / neutral / negative) and feeds the signal into the driver set.',
  },
  {
    title: 'Risk Alerts',
    body: 'LLM-generated daily risk reports grounded in model output and SHAP attribution, delivered via the Alerts page.',
  },
]

function fmt3(v: number | null | undefined): string {
  return v === null || v === undefined || Number.isNaN(v) ? '—' : v.toFixed(3)
}

interface MetricCardProps {
  label: string
  value: string
  sub?: string
}

function MetricCard({ label, value, sub }: MetricCardProps) {
  return (
    <div className="flex flex-col gap-1 rounded-sm border border-edge bg-panel px-4 py-3">
      <span className="font-mono text-[10px] uppercase tracking-widest text-muted">{label}</span>
      <span className="font-mono text-2xl font-semibold text-ink">{value}</span>
      {sub && <span className="font-mono text-[10px] text-muted">{sub}</span>}
    </div>
  )
}

export default function Overview() {
  const [factors, setFactors] = useState<string[]>([])
  const [evalRows, setEvalRows] = useState<EvalRow[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError('')

    getFactors()
      .then((fs) => {
        if (cancelled) return
        setFactors(fs)
        if (fs.length === 0) {
          setLoading(false)
          return
        }
        // fetch eval for the first CNH factor (prefer CNH, fall back to first)
        const cnh = fs.find((f) => f.includes('CNH')) ?? fs[0]
        return getEval(cnh).then((r) => {
          if (!cancelled) setEvalRows(r.rows)
        })
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [])

  const harx = evalRows.find((r) => r.model === 'HAR-X')
  const gbm = evalRows.find((r) => r.model === 'GBM')
  const nOos = harx?.n_oos ?? gbm?.n_oos

  return (
    <div className="space-y-10 p-6 pb-16">
      {/* ── Hero ─────────────────────────────────────────────────────── */}
      <section className="space-y-3 pt-4">
        <h1 className="text-4xl font-bold tracking-tight leading-tight">
          <span className="text-ubs">Beyond</span> the Smile
        </h1>
        <p className="max-w-2xl text-sm leading-relaxed text-muted">
          Decomposing the USD/CNY &amp; USD/CNH volatility surface — beyond ATM, beyond the smile
        </p>
        <p className="font-mono text-xs uppercase tracking-widest text-muted/60">
          UBS Fin AI Bootcamp
        </p>
      </section>

      {/* ── Methodology strip ────────────────────────────────────────── */}
      <section className="space-y-3">
        <h2 className="text-[10px] font-semibold uppercase tracking-widest text-muted">
          Methodology
        </h2>
        <div className="flex flex-wrap gap-3">
          {METHODS.map((m) => (
            <div
              key={m.title}
              className="min-w-[160px] flex-1 rounded-sm border border-edge border-l-2 border-l-ubs bg-panel p-4"
            >
              <p className="mb-1.5 text-xs font-semibold uppercase tracking-widest text-ink">
                {m.title}
              </p>
              <p className="text-xs leading-relaxed text-muted">{m.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Live metrics ─────────────────────────────────────────────── */}
      <section className="space-y-3">
        <h2 className="text-[10px] font-semibold uppercase tracking-widest text-muted">
          Live Metrics
          {factors.length > 0 && (
            <span className="ml-2 normal-case tracking-normal text-muted/60">
              — {factors.find((f) => f.includes('CNH')) ?? factors[0]}
            </span>
          )}
        </h2>

        {loading ? (
          <Panel title="Loading metrics…">
            <Spinner label="Fetching model evaluation…" />
          </Panel>
        ) : error ? (
          <Panel title="Metrics">
            <ErrorNote message={error} />
          </Panel>
        ) : evalRows.length === 0 ? (
          <Panel title="Metrics">
            <span className="font-mono text-xs text-muted">No evaluation data available.</span>
          </Panel>
        ) : (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
            {nOos !== undefined && (
              <MetricCard
                label="N (OOS)"
                value={String(nOos)}
                sub="out-of-sample observations"
              />
            )}
            <MetricCard
              label="Factors"
              value={String(factors.length)}
              sub="PCA surface factors"
            />
            {harx && (
              <>
                <MetricCard
                  label="HAR-X Corr (w)"
                  value={fmt3(harx.Corr_week)}
                  sub="weekly correlation"
                />
                <MetricCard
                  label="HAR-X QLIKE (d)"
                  value={fmt3(harx.QLIKE_daily)}
                  sub="daily QLIKE loss"
                />
              </>
            )}
            {gbm && (
              <>
                <MetricCard
                  label="GBM Corr (w)"
                  value={fmt3(gbm.Corr_week)}
                  sub="weekly correlation"
                />
                <MetricCard
                  label="GBM QLIKE (d)"
                  value={fmt3(gbm.QLIKE_daily)}
                  sub="daily QLIKE loss"
                />
              </>
            )}
          </div>
        )}
      </section>

      {/* ── CTA links ────────────────────────────────────────────────── */}
      <section className="space-y-3">
        <h2 className="text-[10px] font-semibold uppercase tracking-widest text-muted">
          Explore
        </h2>
        <div className="flex flex-wrap gap-3">
          <Link
            to="/terminal"
            className="rounded-sm border border-ubs px-5 py-2.5 font-mono text-xs uppercase tracking-widest text-ubs transition-colors hover:bg-ubs hover:text-white"
          >
            Terminal
          </Link>
          <Link
            to="/alerts"
            className="rounded-sm border border-ubs px-5 py-2.5 font-mono text-xs uppercase tracking-widest text-ubs transition-colors hover:bg-ubs hover:text-white"
          >
            Risk Alerts
          </Link>
          <Link
            to="/chat"
            className="rounded-sm border border-ubs px-5 py-2.5 font-mono text-xs uppercase tracking-widest text-ubs transition-colors hover:bg-ubs hover:text-white"
          >
            Chat
          </Link>
        </div>
      </section>
    </div>
  )
}
