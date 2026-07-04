import type { ReactNode } from 'react'

export function Panel({
  title,
  right,
  children,
  className = '',
}: {
  title: string
  right?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <section
      className={`rounded-sm border border-edge border-l-2 border-l-ubs bg-panel ${className}`}
    >
      <header className="flex items-center justify-between gap-3 border-b border-edge px-4 py-2.5">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-muted">
          {title}
        </h2>
        {right}
      </header>
      <div className="p-4">{children}</div>
    </section>
  )
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2.5 py-6 text-sm text-muted">
      <svg className="h-4 w-4 animate-spin text-ubs" viewBox="0 0 24 24" fill="none">
        <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" opacity="0.25" />
        <path d="M22 12a10 10 0 0 0-10-10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
      </svg>
      {label ?? 'Loading…'}
    </div>
  )
}

export function ErrorNote({ message }: { message: string }) {
  return (
    <div className="rounded-sm border border-edge bg-page px-3 py-2.5 font-mono text-xs text-muted">
      <span className="mr-2 font-semibold text-ubs">ERR</span>
      {message}
    </div>
  )
}

export function EmptyNote({ message }: { message: string }) {
  return (
    <div className="py-6 text-center font-mono text-xs text-muted">{message}</div>
  )
}

export const selectClass =
  'rounded-sm border border-edge bg-page px-2.5 py-1.5 font-mono text-xs text-ink outline-none focus:border-ubs'
