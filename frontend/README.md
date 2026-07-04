# Beyond the Smile — Frontend

USD/CNY–CNH FX volatility research terminal for the UBS Fin AI Bootcamp.

Vite + React + TypeScript + Tailwind CSS + Recharts.

## Run

```bash
npm install
npm run dev
```

Dev server runs on http://localhost:5173 and proxies `/api` to the backend
at http://localhost:8000 (see `docs/API_CONTRACT.md` in the repo root).

## Build

```bash
npm run build   # type-checks then bundles to dist/
```

## Pages

- `/terminal` — factor picker, realized vs HAR-X vs GBM volatility chart,
  model evaluation table (QLIKE / Corr), SHAP driver bar chart with model
  toggle and date picker.
- `/alerts` — NLP risk alert list with monospace reader pane.
- `/chat` — research assistant chat (requires `DEEPSEEK_API_KEY` on the backend).
