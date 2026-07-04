import { useEffect, useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  getEval,
  getFactors,
  getForecasts,
  getShap,
  getShapDates,
  getShapTimeseries,
  ApiError,
  type EvalRow,
  type ForecastPoint,
  type ShapDriver,
  type ShapModel,
  type ShapTimeseriesPoint,
} from '../api'
import { EmptyNote, ErrorNote, Panel, Spinner, selectClass } from '../components/ui'

const COLORS = {
  realized: '#ECECEC',
  harx: '#E60000',
  gbm: '#8A8D90',
  grid: '#2A2D30',
  muted: '#6A6A6A',
}

// Attribution panel — stable color palette per feature slot.
// Slot 0 = UBS red (first driver), then muted grey tones, "Other" darkest.
const ATTRIB_PALETTE = [
  '#E60000', // UBS red — most impactful driver
  '#8A8D90',
  '#6A6A6A',
  '#AEAEAE',
  '#4A4A4A',
  '#C8C8C8',
  '#3A3A3A', // "Other" — darkest grey
]

function attribColor(index: number, total: number): string {
  // Always assign the last palette entry to the last feature ("Other" if present)
  if (index === total - 1 && total > 1) return ATTRIB_PALETTE[ATTRIB_PALETTE.length - 1]
  return ATTRIB_PALETTE[Math.min(index, ATTRIB_PALETTE.length - 2)]
}

function fmt(v: number | null | undefined): string {
  return v === null || v === undefined || Number.isNaN(v) ? '—' : v.toFixed(3)
}

const tooltipStyle = {
  backgroundColor: '#1B1E21',
  border: '1px solid #2A2D30',
  borderRadius: 2,
  fontSize: 12,
  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
}

export default function Terminal() {
  const [factors, setFactors] = useState<string[]>([])
  const [factor, setFactor] = useState('')
  const [factorsError, setFactorsError] = useState('')

  // Forecast chart state
  const [series, setSeries] = useState<ForecastPoint[]>([])
  const [seriesLoading, setSeriesLoading] = useState(false)
  const [seriesError, setSeriesError] = useState('')

  // Eval table state
  const [evalRows, setEvalRows] = useState<EvalRow[]>([])
  const [evalError, setEvalError] = useState('')

  // SHAP state
  const [shapModel, setShapModel] = useState<ShapModel>('HAR-X')
  const [shapDates, setShapDates] = useState<string[]>([])
  const [shapDate, setShapDate] = useState('')
  const [drivers, setDrivers] = useState<ShapDriver[]>([])
  const [shapLoading, setShapLoading] = useState(false)
  const [shapError, setShapError] = useState('')

  // Attribution timeseries state
  const [attribFeatures, setAttribFeatures] = useState<string[]>([])
  const [attribSeries, setAttribSeries] = useState<ShapTimeseriesPoint[]>([])
  const [attribLoading, setAttribLoading] = useState(false)
  const [attribError, setAttribError] = useState('')

  useEffect(() => {
    getFactors()
      .then((f) => {
        setFactors(f)
        if (f.length > 0) setFactor(f[0])
      })
      .catch((e) => setFactorsError(e instanceof Error ? e.message : String(e)))
  }, [])

  useEffect(() => {
    if (!factor) return
    setSeriesLoading(true)
    setSeriesError('')
    getForecasts(factor)
      .then((r) => setSeries(r.series))
      .catch((e) => {
        setSeries([])
        setSeriesError(e instanceof Error ? e.message : String(e))
      })
      .finally(() => setSeriesLoading(false))

    setEvalError('')
    getEval(factor)
      .then((r) => setEvalRows(r.rows))
      .catch((e) => {
        setEvalRows([])
        setEvalError(e instanceof Error ? e.message : String(e))
      })

    getShapDates(factor)
      .then((dates) => {
        setShapDates(dates)
        setShapDate((prev) =>
          dates.includes(prev) ? prev : dates[dates.length - 1] ?? '',
        )
      })
      .catch(() => setShapDates([]))
  }, [factor])

  useEffect(() => {
    if (!factor || !shapDate) {
      setDrivers([])
      return
    }
    setShapLoading(true)
    setShapError('')
    getShap(factor, shapDate, shapModel)
      .then((r) => setDrivers(r.drivers))
      .catch((e) => {
        setDrivers([])
        setShapError(
          e instanceof ApiError && e.status === 404
            ? `No SHAP data for ${shapModel} on ${shapDate}`
            : e instanceof Error
              ? e.message
              : String(e),
        )
      })
      .finally(() => setShapLoading(false))
  }, [factor, shapDate, shapModel])

  // Fetch attribution timeseries whenever factor or model changes
  useEffect(() => {
    if (!factor) return
    setAttribLoading(true)
    setAttribError('')
    getShapTimeseries(factor, shapModel)
      .then((r) => {
        setAttribFeatures(r.features)
        setAttribSeries(r.series)
      })
      .catch((e) => {
        setAttribFeatures([])
        setAttribSeries([])
        setAttribError(
          e instanceof ApiError && e.status === 404
            ? 'Attribution endpoint not yet available.'
            : e instanceof Error
              ? e.message
              : String(e),
        )
      })
      .finally(() => setAttribLoading(false))
  }, [factor, shapModel])

  const shapChartData = useMemo(
    () => [...drivers].reverse(), // largest |shap| at top of horizontal chart
    [drivers],
  )

  // Flatten attrib series into recharts row objects: { period, feature1: v, feature2: v, ... }
  const attribChartData = useMemo(
    () =>
      attribSeries.map((pt) => ({
        period: pt.period,
        ...pt.values,
      })),
    [attribSeries],
  )

  return (
    <div className="space-y-4 p-5">
      {/* Factor selector row */}
      <div className="flex items-center gap-3">
        <label className="text-xs font-semibold uppercase tracking-widest text-muted">
          Vol Factor
        </label>
        <select
          className={selectClass}
          value={factor}
          onChange={(e) => setFactor(e.target.value)}
          disabled={factors.length === 0}
        >
          {factors.length === 0 && <option value="">— no factors —</option>}
          {factors.map((f) => (
            <option key={f} value={f}>
              {f}
            </option>
          ))}
        </select>
        {factorsError && (
          <span className="font-mono text-xs text-muted">
            backend unreachable — {factorsError}
          </span>
        )}
      </div>

      {/* Forecast chart */}
      <Panel title="Realized vs Forecast Volatility">
        {seriesLoading ? (
          <Spinner label="Loading forecasts…" />
        ) : seriesError ? (
          <ErrorNote message={seriesError} />
        ) : series.length === 0 ? (
          <EmptyNote message="No forecast data available." />
        ) : (
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={series} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid stroke={COLORS.grid} strokeDasharray="3 3" vertical={false} />
                <XAxis
                  dataKey="date"
                  tick={{ fill: COLORS.muted, fontSize: 11, fontFamily: 'monospace' }}
                  stroke={COLORS.grid}
                  minTickGap={60}
                />
                <YAxis
                  tick={{ fill: COLORS.muted, fontSize: 11, fontFamily: 'monospace' }}
                  stroke={COLORS.grid}
                  width={55}
                  tickFormatter={(v: number) => v.toFixed(2)}
                />
                <Tooltip
                  contentStyle={tooltipStyle}
                  labelStyle={{ color: '#ECECEC' }}
                  formatter={(value) =>
                    typeof value === 'number' ? value.toFixed(4) : '—'
                  }
                />
                <Legend wrapperStyle={{ fontSize: 12, fontFamily: 'monospace' }} />
                <Line
                  type="monotone"
                  dataKey="rv_realized"
                  name="Realized"
                  stroke={COLORS.realized}
                  strokeWidth={1.4}
                  dot={false}
                  connectNulls
                />
                <Line
                  type="monotone"
                  dataKey="rv_forecast_harx"
                  name="HAR-X"
                  stroke={COLORS.harx}
                  strokeWidth={1.4}
                  strokeDasharray="6 4"
                  dot={false}
                  connectNulls
                />
                <Line
                  type="monotone"
                  dataKey="rv_forecast_gbm"
                  name="GBM"
                  stroke={COLORS.gbm}
                  strokeWidth={1.4}
                  strokeDasharray="2 4"
                  dot={false}
                  connectNulls
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </Panel>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        {/* Model evaluation table */}
        <Panel title="Model Evaluation (OOS)">
          {evalError ? (
            <ErrorNote message={evalError} />
          ) : evalRows.length === 0 ? (
            <EmptyNote message="No evaluation data available." />
          ) : (
            <table className="w-full font-mono text-xs">
              <thead>
                <tr className="border-b border-edge text-left text-muted">
                  <th className="py-2 pr-3 font-medium">Model</th>
                  <th className="py-2 pr-3 text-right font-medium">N (OOS)</th>
                  <th className="py-2 pr-3 text-right font-medium">QLIKE (d)</th>
                  <th className="py-2 pr-3 text-right font-medium">Corr (d)</th>
                  <th className="py-2 pr-3 text-right font-medium">QLIKE (w)</th>
                  <th className="py-2 text-right font-medium">Corr (w)</th>
                </tr>
              </thead>
              <tbody>
                {evalRows.map((r) => (
                  <tr key={r.model} className="border-b border-edge/50 last:border-0">
                    <td className="py-2 pr-3 font-semibold text-ink">{r.model}</td>
                    <td className="py-2 pr-3 text-right">{r.n_oos}</td>
                    <td className="py-2 pr-3 text-right">{fmt(r.QLIKE_daily)}</td>
                    <td className="py-2 pr-3 text-right">{fmt(r.Corr_daily)}</td>
                    <td className="py-2 pr-3 text-right">{fmt(r.QLIKE_week)}</td>
                    <td className="py-2 text-right">{fmt(r.Corr_week)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Panel>

        {/* SHAP drivers */}
        <Panel
          title="SHAP Drivers"
          right={
            <div className="flex items-center gap-2">
              <div className="flex overflow-hidden rounded-sm border border-edge">
                {(['HAR-X', 'GBM'] as ShapModel[]).map((m) => (
                  <button
                    key={m}
                    onClick={() => setShapModel(m)}
                    className={`px-3 py-1 font-mono text-xs transition-colors ${
                      shapModel === m
                        ? 'bg-ubs text-white'
                        : 'bg-page text-muted hover:text-ink'
                    }`}
                  >
                    {m}
                  </button>
                ))}
              </div>
              <select
                className={selectClass}
                value={shapDate}
                onChange={(e) => setShapDate(e.target.value)}
                disabled={shapDates.length === 0}
              >
                {shapDates.length === 0 && <option value="">— no dates —</option>}
                {shapDates.map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
            </div>
          }
        >
          {shapLoading ? (
            <Spinner label="Loading SHAP…" />
          ) : shapError ? (
            <ErrorNote message={shapError} />
          ) : shapChartData.length === 0 ? (
            <EmptyNote message="No SHAP data for this selection." />
          ) : (
            <div style={{ height: Math.max(220, shapChartData.length * 36) }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={shapChartData}
                  layout="vertical"
                  margin={{ top: 5, right: 20, bottom: 5, left: 10 }}
                >
                  <CartesianGrid stroke={COLORS.grid} strokeDasharray="3 3" horizontal={false} />
                  <XAxis
                    type="number"
                    tick={{ fill: COLORS.muted, fontSize: 11, fontFamily: 'monospace' }}
                    stroke={COLORS.grid}
                    tickFormatter={(v: number) => v.toFixed(3)}
                  />
                  <YAxis
                    type="category"
                    dataKey="feature"
                    width={140}
                    tick={{ fill: '#ECECEC', fontSize: 11, fontFamily: 'monospace' }}
                    stroke={COLORS.grid}
                  />
                  <Tooltip
                    cursor={{ fill: '#2A2D3055' }}
                    contentStyle={tooltipStyle}
                    labelStyle={{ color: '#ECECEC' }}
                    formatter={(value) =>
                      typeof value === 'number' ? value.toFixed(4) : '—'
                    }
                  />
                  <ReferenceLine x={0} stroke={COLORS.muted} />
                  <Bar dataKey="shap_value" name="SHAP" barSize={16}>
                    {shapChartData.map((d) => (
                      <Cell
                        key={d.feature}
                        fill={d.shap_value >= 0 ? COLORS.harx : COLORS.gbm}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </Panel>
      </div>

      {/* Attribution panel — monthly mean SHAP (macro drivers) */}
      <Panel title="Attribution — monthly mean SHAP (macro drivers)">
        {attribLoading ? (
          <Spinner label="Loading attribution…" />
        ) : attribError ? (
          <ErrorNote message={attribError} />
        ) : attribChartData.length === 0 ? (
          <EmptyNote message="No attribution data for this selection." />
        ) : (
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={attribChartData}
                margin={{ top: 5, right: 20, bottom: 5, left: 10 }}
                stackOffset="sign"
              >
                <CartesianGrid stroke={COLORS.grid} strokeDasharray="3 3" vertical={false} />
                <XAxis
                  dataKey="period"
                  tick={{ fill: COLORS.muted, fontSize: 11, fontFamily: 'monospace' }}
                  stroke={COLORS.grid}
                  minTickGap={40}
                />
                <YAxis
                  tick={{ fill: COLORS.muted, fontSize: 11, fontFamily: 'monospace' }}
                  stroke={COLORS.grid}
                  width={60}
                  tickFormatter={(v: number) => v.toFixed(3)}
                />
                <Tooltip
                  cursor={{ fill: '#2A2D3033' }}
                  contentStyle={tooltipStyle}
                  labelStyle={{ color: '#ECECEC' }}
                  formatter={(value) =>
                    typeof value === 'number' ? value.toFixed(4) : '—'
                  }
                />
                <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'monospace' }} />
                <ReferenceLine y={0} stroke={COLORS.muted} />
                {attribFeatures.map((feat, i) => (
                  <Bar
                    key={feat}
                    dataKey={feat}
                    stackId="attrib"
                    fill={attribColor(i, attribFeatures.length)}
                    name={feat}
                    isAnimationActive={false}
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </Panel>
    </div>
  )
}
