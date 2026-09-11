# Upgrading to v1.0.0

The short answer to "is this a drop-in replacement?" is **no**. The API is
almost entirely backward compatible, but **the service will refuse to start on
the previous release's environment variables**, and the risk labels it returns
have changed meaning because the previous ones were wrong.

Work through this page before deploying. It takes about ten minutes.

---

## 1. The one that will stop your deploy

The previous release shipped these defaults:

```python
ENVIRONMENT: str = "production"
DEBUG: bool = True
RELOAD: bool = True
```

v1.0.0 rejects exactly that combination. Running with `DEBUG=true` in
production leaks stack traces and disables the security headers, so the
configuration is now validated at start-up and the process exits rather than
serving traffic in that state.

**If your deployment relies on the old defaults, it will fail to boot** with:

```
Afya Jamii failed to start: configuration is invalid.

Invalid production configuration:
  - DEBUG must be false in production
  - ALLOWED_HOSTS must name the API's real hostnames in production, not '*'
```

### What to set before deploying

| Variable | Required value | Why |
| --- | --- | --- |
| `DEBUG` | `false` | Rejected in production |
| `RELOAD` | `false` | Rejected in production |
| `ALLOWED_HOSTS` | `afyajamii.onrender.com` | `*` is rejected; comma-separate multiple hosts |
| `CORS_ORIGINS` | your frontend's origin | A wildcard is rejected, and cannot be combined with credentials |
| `SECRET_KEY` | 32+ characters | Now required and length-checked |

`SECRET_KEY` deserves particular attention. The previous release generated a
random key at start-up whenever the placeholder was left in place, which meant
every deploy silently signed all users out — and under Gunicorn, each worker
held a *different* key, so a token issued by one worker was rejected by the
next. Set a real value:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Set it once, keep it stable, and share it across workers. Changing it signs
everyone out.

### A format change that fails silently

`CORS_ORIGINS` was typed `list[str]`, which pydantic-settings required to be
JSON: `CORS_ORIGINS=["https://example.com"]`. It is now a comma-separated
string.

Both forms are accepted — the parser detects a leading `[` and falls back to
JSON — specifically so an existing deployment does not end up with a single
"origin" that still has its brackets and quotes attached, which would fail at
request time rather than at start-up. New deployments should use the plain
form:

```
CORS_ORIGINS=https://afyajamii.vercel.app,http://localhost:8080
```

### New variables (all optional, sensible defaults)

```
LOG_FORMAT=json                  # `text` locally, `json` for log aggregators
LABEL_ENCODER_PATH=./data/risk_label_encoder.pkl
LLM_MAX_TOKENS=1500
LLM_TIMEOUT_SECONDS=45
LLM_MAX_RETRIES=2
PROMPT_DIR=                      # empty uses the bundled app/prompts
PROMPT_NAME=clinical_assistant
PROMPT_RELOAD_ON_CHANGE=false
CHAT_HISTORY_CHAR_BUDGET=6000
CHAT_HISTORY_REPLY_CHARS=700
RATE_LIMIT_ENABLED=true
```

---

## 2. Risk labels have changed meaning

This is a behaviour change, not a bug: the previous behaviour was the bug.

The model is a **three-class** classifier whose encoder orders the classes
`['high risk', 'low risk', 'mid risk']`. The previous code read
`predict_proba(x)[0, 1]` — the *low risk* column — and compared it to 0.5.

The practical effect: a patient at **170/110 with blood sugar 15.0 and a fever
was reported as "low risk"**, while the model itself scored her high risk at
100% confidence.

| | Before | After |
| --- | --- | --- |
| `risk_label` values | `"low risk"`, `"high risk"` | `"low risk"`, `"mid risk"`, `"high risk"` |
| `probability` means | P(low risk), regardless of the label | Confidence in the label actually returned |
| Correctness | Inverted | Correct |

**Consumers must handle `"mid risk"`.** The previous frontend matched on
`"medium"` and `"moderate"` but not `"mid"`, so a moderate-risk result would
have fallen through to the green "low risk" styling. The bundled frontend
handles all three.

`class_probabilities` is new and gives the full distribution, so a client no
longer has to infer anything from a single number.

### Existing rows hold wrong labels

Stored labels from before the fix are unreliable. The original vitals are
intact, so they can be re-scored:

```bash
cd afya_jamii_backend
python scripts/rescore_vitals.py --dry-run     # report what would change
python scripts/rescore_vitals.py               # apply
python scripts/rescore_vitals.py --before 2026-09-11   # only pre-upgrade rows
```

It rewrites `ml_risk_label`, `ml_probability` and `ml_feature_importances` in
place, never the vitals, and is safe to run more than once. Back up first.

---

## 3. Database

`SQLModel.metadata.create_all()` creates missing tables but never alters
existing ones, so an existing database will not pick up the new column types
or indexes on its own.

```bash
mysqldump -u USER -p afya_jamii > afya_jamii_backup.sql
mysql -u USER -p afya_jamii < afya_jamii_backend/migrations/001_upgrade_to_v1.sql
```

The migration widens the generated-text columns to `LONGTEXT` and adds indexes
on `user_id` and `created_at` for both history tables — every history query
filters and orders on those, and without the indexes each request is a full
table scan.

No data is deleted and no column is dropped, so the previous release can still
run against the migrated schema if you need to roll back.

---

## 4. API changes

Additive unless marked otherwise.

### Responses gaining fields

| Endpoint | New field |
| --- | --- |
| `POST /auth/login` | `username`, `account_type` — so a client stops guessing. The previous frontend hard-coded every session to `"general"`. |
| `POST /vitals/submit` | `ml_output.class_probabilities`, `llm_advice.generated` |
| `POST /chat/advice` | `generated` — `false` means the text is an offline fallback, not clinical advice |
| `GET /health` | `environment`, per-service detail, pool statistics |

### New endpoint

`GET /api/v1/auth/me` returns the signed-in user, so a client can restore a
session from a stored token instead of trusting it blindly.

### Changed behaviour

| What | Before | After | Impact |
| --- | --- | --- | --- |
| `POST /vitals/submit` status | `200` | `201` | None for clients checking `response.ok` |
| Duplicate signup | `400` | `409` | Cosmetic; `detail` still carries the message |
| `POST /chat/advice` when the model is down | `200` with the error as `advice` | `503` | **Check this.** A client that renders any 200 body would previously have shown an error string as medical advice. |
| `ml_feature_importances` in history | JSON **string** | parsed **object** | **Breaking** for anything calling `JSON.parse()` on it |
| `account_type` in vitals submission | required | optional, defaults to the user's own | Old payloads still work |
| Timestamps | naive, no offset | timezone-aware, `+00:00` | Previously parsed as *local* time by JavaScript, so displayed times were wrong by your UTC offset |

`user_id` is still returned by both history endpoints — the response models
were split away from the table models, and it was deliberately kept.

### Stricter validation

Requests that previously succeeded may now return `422`:

- `diastolic_bp` must be below `systolic_bp`
- `body_temp` is range-checked **per unit** (35–42 °C, 95–107.6 °F). Previously
  a Fahrenheit reading was checked against the Celsius range and the value was
  then scored as if it were Celsius.
- `email` is properly validated
- passwords must mix letters with digits or symbols
- `username` is restricted to letters, digits, `_`, `.`, `-`

### CORS

Allowed methods narrowed from `*` to `GET, POST, OPTIONS`, and headers from
`*` to `Authorization, Content-Type, X-Request-ID`. If a client sends anything
else, add it in `app/main.py`.

---

## 5. Dependencies

`requirements.txt` is now the curated list of direct dependencies;
`requirements.lock.txt` is the full resolved environment. **Build from the
lock file**:

```bash
pip install -r requirements.lock.txt
```

The file was previously UTF-16 encoded, which makes `pip install -r` fail
outright; it is now UTF-8.

`langchain` and `langchain-community` are gone — the deprecated `LLMChain` was
replaced with the LCEL pipeline from `langchain-core`. `pandas` was listed but
never imported. `scikit-learn` moved to `1.6.1` to match the version the
bundled artefacts were pickled with.

---

## 6. Recommended order

```bash
# 1. Back up
mysqldump -u USER -p afya_jamii > afya_jamii_backup.sql

# 2. Update environment variables (section 1) — the deploy fails without this

# 3. Migrate the schema
mysql -u USER -p afya_jamii < afya_jamii_backend/migrations/001_upgrade_to_v1.sql

# 4. Deploy, then confirm
curl https://your-api/health
```

A healthy response reports `"status": "healthy"` with `database`, `risk_model`
and `advice` all ready. `"degraded"` means the core service is up but advice
generation is not — check `GROQ_API_KEY` and `LLM_MODEL_NAME`.

```bash
# 5. Correct the historic labels
python scripts/rescore_vitals.py --dry-run
python scripts/rescore_vitals.py
```

### Rolling back

The schema migration is additive, so the previous release runs against the
migrated database unchanged. Restore the old environment variables and
redeploy. Note that re-scored rows keep their corrected labels — which is what
you want.
