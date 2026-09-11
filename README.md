# Afya Jamii

Maternal health monitoring and guidance for mothers in Kenya.

A mother enters her vitals — blood pressure, heart rate, blood sugar,
temperature. A risk model scores them, and a language model turns that score
into plain-language guidance grounded in Kenyan clinical practice, in English
or Kiswahili. Every reading and conversation is kept so she can show a health
worker how things have changed.

The service supports the care a mother receives from a health worker. It does
not replace a clinic visit, and it is not for emergencies.

---

## Layout

```
AfyaJamii/
├── afya_jamii_backend/      FastAPI service, risk model, prompts
│   ├── app/
│   │   ├── main.py            HTTP API and middleware
│   │   ├── config.py          Environment-driven settings
│   │   ├── models.py          API schemas and database tables
│   │   ├── auth.py            Password hashing and JWT
│   │   ├── database.py        Engine, sessions, schema
│   │   ├── ml_model.py        Risk classification
│   │   ├── llm_groq.py        Groq client
│   │   ├── prompt_loader.py   Loads and validates prompt files
│   │   └── prompts/           Prompt templates — data, not code
│   │       ├── clinical_assistant.json
│   │       └── emergency_contacts.md
│   ├── data/                  Model artefacts
│   └── tests/
└── frontend/                React + TypeScript + Tailwind
    ├── src/
    │   ├── pages/             Landing, auth, dashboard
    │   ├── components/        Feature components and UI primitives
    │   ├── contexts/          Auth and theme
    │   ├── lib/               API client, icon registry, formatting
    │   └── assets/icons/      Vendored Icons8 PNGs
    └── scripts/fetch-icons.mjs
```

---

## Running it locally

### Backend

Requires Python 3.12 and a MySQL database (SQLite works for development).

```bash
cd afya_jamii_backend

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

Then edit `.env`. Two values are required and the service will refuse to start
without them:

- `SECRET_KEY` — generate with
  `python -c "import secrets; print(secrets.token_urlsafe(48))"`
- `GROQ_API_KEY` — from <https://console.groq.com>

```bash
uvicorn app.main:app --reload --reload-dir app
```

The API listens on <http://localhost:8000>, with interactive documentation at
`/docs` (disabled when `ENVIRONMENT=production`).

`--reload-dir app` matters if your virtualenv lives inside this directory.
Plain `--reload` watches the whole tree, and stat-polling a 270 MB `.venv`
pins a CPU core and can starve the worker before start-up finishes.

### Frontend

Requires Node 18.18 or newer.

```bash
cd frontend

npm install
cp .env.example .env.local    # defaults point at http://localhost:8000
npm run dev
```

The app runs at <http://localhost:8080>.

---

## Configuration

Everything is read from the environment. Nothing that varies between
deployments is compiled in — model name, prompt directory, database, CORS
origins, and rate limits are all settings. See `afya_jamii_backend/.env.example`
for the annotated list.

The values you are most likely to change:

| Variable | Purpose |
| --- | --- |
| `LLM_MODEL_NAME` | Which model serves advice. Swap it without touching code. |
| `LLM_TEMPERATURE` | `0.0` is deterministic; a little warmth reads less robotic. |
| `PROMPT_NAME` | Which prompt file to use. |
| `PROMPT_DIR` | Override the bundled prompts per environment. |
| `CHAT_HISTORY_TURNS` | How many past turns are replayed to the model. |
| `VITE_API_BASE_URL` | Which API the frontend talks to (frontend build). |

In production the configuration is validated more strictly: `DEBUG` and
`RELOAD` must be off, `ALLOWED_HOSTS` must name real hostnames, and
`CORS_ORIGINS` may not be a wildcard. A deployment that violates any of these
fails at start-up with an explanatory message rather than running unsafely.

---

## Prompts

Prompt text lives in `afya_jamii_backend/app/prompts/` as data files, so it can
be reviewed and edited — by clinical staff, ideally — without touching Python.

`clinical_assistant.json` declares its version, the variables it expects, and
the template itself as a list of lines (which keeps diffs readable). Bulk
reference material is kept in its own Markdown file and pulled in through
`includes`:

```json
{
  "id": "clinical_assistant",
  "version": "2.0.0",
  "input_variables": ["context", "history", "question"],
  "includes": { "emergency_contacts": "emergency_contacts.md" },
  "template": ["You are Afya Jamii AI…", "{context}", "…"]
}
```

The loader validates on read: a placeholder the template uses but does not
declare is an error, a missing include is an error, and a variable with no
value at render time is an error. Set `PROMPT_RELOAD_ON_CHANGE=true` to re-read
the file when it changes, which is convenient while editing wording.

To revise the emergency numbers, edit `emergency_contacts.md`. No redeploy of
application code is involved.

---

## Account settings

Endpoints for a user managing their own account. All require authentication
and act only on the caller — there is no way to reach another user's record.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/users/me` | Read the profile |
| `PATCH` | `/api/v1/users/me` | Update email, full name, or account type |
| `POST` | `/api/v1/users/me/password` | Change the password |
| `POST` | `/api/v1/users/me/deactivate` | Disable sign-in, keep the records |
| `DELETE` | `/api/v1/users/me` | Delete the account and all its records |

The username is not editable: it identifies the account and is the subject of
every issued token, so changing it would invalidate the caller's own session
mid-request.

### Deactivating vs deleting

Deactivation is offered first because it is usually what people actually mean.
Sign-in stops working, but the health records survive, so a mother who comes
back still has her history. Reactivation is a manual administrative step.

Deletion is permanent. It removes the user together with every vitals record
and conversation, child rows first, in a single transaction — a failure
part-way through leaves the account intact rather than orphaned.

Because it destroys health records, it asks for two things a mis-click cannot
supply:

```http
DELETE /api/v1/users/me
Authorization: Bearer <token>

{
  "password": "<the account password>",
  "confirmation": "DELETE MY ACCOUNT"
}
```

The phrase must match exactly. The response reports what was removed:

```json
{
  "detail": "Your account and all associated health records have been permanently deleted.",
  "username": "amina.w",
  "vitals_records_deleted": 3,
  "conversations_deleted": 3,
  "deleted_at": "2026-09-11T17:31:04Z"
}
```

A deleted account frees its username and email for reuse.

One limitation worth knowing: changing a password does not revoke tokens
already issued. They are signed with the application secret rather than the
password, so they stay valid until they expire (`ACCESS_TOKEN_EXPIRE_MINUTES`,
30 by default). Revoking them immediately would need a token version column or
a deny-list, which this service does not yet have.

---

## The risk model

`data/risk_model_v1.pkl` is a three-class XGBoost classifier (`multi:softprob`)
over six vitals. `data/risk_label_encoder.pkl` holds the class labels, in
order: `['high risk', 'low risk', 'mid risk']`.

Both artefacts are required. At load time the encoder's class count is checked
against the model's; if they disagree the service refuses to start, because
loading them anyway would mislabel every prediction.

> **A note on a fixed defect.** An earlier revision read
> `predict_proba(x)[0, 1]` and compared it to `0.5`, treating the model as
> binary. Column 1 is *low risk*, so the code reported "high risk" precisely
> when the model was confident a patient was low risk — and reported "low risk"
> for a patient at 170/110 with a blood sugar of 15.0. Predictions now go
> through `argmax` and the label encoder. `tests/test_risk_model.py` covers
> this case directly.

---

## Tests

```bash
cd afya_jamii_backend
pytest tests/ -v
```

```bash
cd frontend
npm run typecheck
npm run lint
npm run build
```

---

## Deployment

### Backend

```bash
cd afya_jamii_backend
docker build -t afya-jamii-api .
docker run -p 8000:8000 --env-file .env afya-jamii-api
```

The image is built in two stages so the runtime carries no compilers, runs as
an unprivileged user, and reports health through `/health`. That endpoint
returns 503 when a required component is down, so an orchestrator can pull the
instance out of rotation.

Without Docker, run under Gunicorn:

```bash
gunicorn app.main:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers 4 --bind 0.0.0.0:8000
```

`SECRET_KEY` must be identical across workers and stable across restarts —
otherwise every issued token is invalidated on deploy.

### Frontend

```bash
cd frontend
VITE_API_BASE_URL=https://api.example.com npm run build
```

Serve `dist/` as a static site with a SPA fallback to `index.html`. A
`vercel.json` is included with that rewrite, immutable caching for hashed
assets, and security headers.

---

## Design notes

**Degradation.** The database and risk model are required; the language model
is not. When advice generation fails, vitals capture, risk scoring,
authentication, and history all keep working, and the response says the
guidance is a fallback rather than presenting it as clinical advice. Failed
exchanges are not written to history, so a failure does not contaminate the
context replayed on the next turn.

**Interface.** Warm and papery rather than clinical blue, with flat surfaces
and hairline rules. The service is used on inexpensive phones, often in bright
sunlight, so focus rings are high-contrast and text size, contrast, and theme
are adjustable and persist across sessions.

**Icons.** From [Icons8](https://icons8.com), vendored into the repository by
`npm run icons:fetch` rather than hot-linked, so a page carrying a mother's
health data makes no third-party request and the interface renders offline.
They are painted through a CSS mask so they follow `currentColor` and the
active theme. The free licence requires attribution, which the footer carries.

---

## Licence

MIT. See [LICENSE](LICENSE).

Icons by [Icons8](https://icons8.com).
