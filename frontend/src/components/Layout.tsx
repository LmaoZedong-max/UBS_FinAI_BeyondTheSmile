import { NavLink, Outlet } from 'react-router-dom'

const NAV = [
  {
    to: '/terminal',
    label: 'Terminal',
    icon: (
      <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.4">
        <path d="M1 12l4-5 3 3 4-6 3 4" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M1 14.5h14" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    to: '/alerts',
    label: 'Risk Alerts',
    icon: (
      <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.4">
        <path d="M8 1.5L15 14H1L8 1.5z" strokeLinejoin="round" />
        <path d="M8 6v4" strokeLinecap="round" />
        <circle cx="8" cy="12" r="0.5" fill="currentColor" />
      </svg>
    ),
  },
  {
    to: '/chat',
    label: 'Chat',
    icon: (
      <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.4">
        <path d="M2 3h12v8H6l-3 3v-3H2V3z" strokeLinejoin="round" />
      </svg>
    ),
  },
]

export default function Layout() {
  return (
    <div className="flex h-full flex-col">
      {/* Top bar */}
      <header className="flex h-14 shrink-0 items-center gap-3 border-b-2 border-ubs bg-panel px-5">
        <div className="flex items-baseline gap-3">
          <span className="text-lg font-bold tracking-tight">
            <span className="text-ubs">Beyond</span> the Smile
          </span>
          <span className="text-xs uppercase tracking-widest text-muted">
            UBS Fin AI Bootcamp
          </span>
        </div>
        <div className="ml-auto flex items-center gap-2 font-mono text-xs text-muted">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-ubs" />
          USD/CNY–CNH FX VOLATILITY RESEARCH
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        {/* Sidebar */}
        <nav className="flex w-48 shrink-0 flex-col gap-1 border-r border-edge bg-panel/60 p-3">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `flex items-center gap-2.5 rounded-sm border-l-2 px-3 py-2 text-sm transition-colors ${
                  isActive
                    ? 'border-ubs bg-page text-ink'
                    : 'border-transparent text-muted hover:bg-page/60 hover:text-ink'
                }`
              }
            >
              {item.icon}
              {item.label}
            </NavLink>
          ))}
          <div className="mt-auto px-3 pb-1 font-mono text-[10px] leading-relaxed text-muted">
            HAR-X / GBM RV FORECASTS
            <br />
            SHAP ATTRIBUTION · NLP ALERTS
          </div>
        </nav>

        {/* Main content */}
        <main className="min-w-0 flex-1 overflow-y-auto">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
