# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Afya Jamii is a maternal health app for mothers in Kenya. A mother submits vitals, an XGBoost model scores risk, and a Groq-hosted LLM turns the score into plain-language guidance (English or Kiswahili). Two independent apps live here: `afya_jamii_backend/` (FastAPI, Python 3.12) and `frontend/` (React + TypeScript + Vite + Tailwind/shadcn).

## Commands

### Backend (run from `afya_jamii_backend/`)

```bash
pip install -r requirements.txt          # requirements.lock.txt is what Render installs
cp .env.example .env                      # SECRET_KEY (32+ chars) and GROQ_API_KEY are required
uvicorn app.main:app --reload --reload-dir app   # http://localhost:8000, docs at /docs
pytest tests/ -v
pytest tests/test_risk_model.py -v        # single file
pytest tests/test_risk_model.py::test_name -v   # single test
```

Use `--reload-dir app`, not plain `--reload`: a `.venv` inside this directory makes the watcher pin a CPU core and can starve start-up.

`tests/conftest.py` sets env vars (SQLite `.pytest_afya.db`, rate limiting off) before importing `app`, because settings are validated at import time. Any new test that imports `app` relies on this.

### Frontend (run from `frontend/`)

```bash
npm install
npm run dev          # http://localhost:8080 (strictPort)
npm run typecheck
npm run lint
npm run build        # tsc -b && vite build
npm run icons:fetch  # re-vendor Icons8 PNGs into src/assets/icons/
```

`VITE_API_BASE_URL` selects the backend (defaults to `http://localhost:8000` in dev). The `@/` alias maps to `frontend/src`.

## Backend architecture

- **`app/main.py`** holds every route, middleware (request ID, request logging, security headers), exception handlers, and the lifespan that loads DB, model, and LLM. Routes are under `settings.API_V1_STR` (`/api/v1`). Rate limits use slowapi via `@limiter.limit(settings.RATE_LIMIT_*)` decorators; these need a `request: Request` parameter to work.
- **`app/config.py`**: pydantic-settings, everything from the environment. In `ENVIRONMENT=production` it rejects `DEBUG`/`RELOAD` on, wildcard `ALLOWED_HOSTS`, or wildcard `CORS_ORIGINS`, and the process exits rather than starting. Keep that behaviour when adding settings.
- **`app/models.py`**: SQLModel tables (users, vitals records, conversations) and the API request/response schemas, all in one file.
- **`app/database.py`**: no migrations. `create_db_and_tables()` runs `SQLModel.metadata.create_all` at startup, plus MySQL-only `ALTER ... LONGTEXT` for long text columns. Schema changes to existing columns need manual handling. Production is MySQL on Aiven (TLS via `DB_SSL_MODE`); SQLite works locally.
- **`app/ml_model.py`**: `data/risk_model_v1.pkl` is a **3-class** XGBoost `multi:softprob` model; `data/risk_label_encoder.pkl` gives the labels `['high risk', 'low risk', 'mid risk']`. Predictions must go through `argmax` plus the encoder. A past bug read `predict_proba[0, 1]` as if the model were binary and inverted results; `tests/test_risk_model.py` guards against that. Loading fails if the encoder and model class counts disagree. scikit-learn is pinned to 1.6.x to match the pickles.
- **`app/llm_groq.py` + `app/prompt_loader.py`**: prompts are data, not code. `app/prompts/clinical_assistant.json` declares `version`, `input_variables`, `includes` (e.g. `emergency_contacts.md`), and `template` as a list of lines. The loader fails on undeclared placeholders, missing includes, or missing values at render time. Model, temperature, prompt name/dir, and history length (`CHAT_HISTORY_TURNS`, `CHAT_HISTORY_CHAR_BUDGET`) are all settings.
- **Degradation rule:** DB and risk model are required (`/health` returns 503 without them). The LLM is optional: if advice generation fails, vitals, scoring, auth, and history still work. The response marks the guidance as a fallback, and failed exchanges are **not** written to conversation history, so they never get replayed as context.
- **Auth** (`app/auth.py`): bcrypt via passlib (bcrypt pinned `<4.1` for passlib compatibility) and JWT signed with `SECRET_KEY`. Usernames are immutable because they are the token subject. Account deletion needs the password plus the exact phrase `"DELETE MY ACCOUNT"`, and removes child rows first in a single transaction.
- `scripts/rescore_vitals.py` re-scores stored vitals written before the risk-label fix (`--dry-run`, `--before DATE`).

When changing dependencies, keep `requirements.txt` and `pyproject.toml` in sync; FastAPI Cloud reads `pyproject.toml`. Regenerate `requirements.lock.txt` too.

## Frontend architecture

- `src/lib/api.ts` is the single fetch client (base URL, timeout, token, typed request/response shapes). It must not throw at module scope when `VITE_API_BASE_URL` is missing, because the landing page has to render without the API.
- `src/lib/queries.ts` wraps the API in TanStack Query hooks (`useVitalsHistory`, `useSubmitVitals`, `useAskQuestion`, ...) with shared `queryKeys`.
- `src/contexts/AuthContext.tsx` (session/token) and `ThemeContext.tsx` (theme, contrast, and text size, persisted).
- `src/pages/Dashboard.tsx` hosts the feature components in `src/components/`. `src/components/ui/` holds shadcn primitives.
- Icons are vendored Icons8 PNGs rendered through `src/components/Icon.tsx` + `src/lib/icons.ts` as CSS masks, so they follow `currentColor`. Don't hot-link third-party assets: the app makes no third-party requests. The footer's Icons8 attribution is a licence requirement.
- Design intent: warm, papery, flat surfaces with hairline rules, high-contrast focus rings. It targets cheap phones in bright sunlight.

## Deployment

- Backend: Render via `render.yaml` (rootDir `afya_jamii_backend`, gunicorn with 1 Uvicorn worker on the free tier, health check `/health`). `.github/workflows/keep-api-awake.yml` pings it every 10 minutes to avoid cold starts.
- Frontend: Vercel (`frontend/vercel.json` has the SPA rewrite, caching, and security headers).
- `UPGRADE.md` covers the v1.0.0 breaking config changes and the risk-label re-scoring.
