#!/usr/bin/env bash
#
# Deploy the Afya Jamii API to FastAPI Cloud.
#
# Run `fastapi login` first — that step opens a browser and cannot be
# scripted. Everything after it is here.
#
#   cd afya_jamii_backend
#   .venv/bin/fastapi login
#   ./scripts/deploy_fastapi_cloud.sh
#
# Secrets are read from the local .env, which is git-ignored, and set with
# --value-stdin so they never appear in the process list or shell history.
# Every variable is saved with --no-redeploy, then a single deploy picks them
# all up at once, rather than triggering a redeploy per variable.

set -euo pipefail

cd "$(dirname "$0")/.."

FASTAPI="${FASTAPI:-.venv/bin/fastapi}"
FRONTEND_ORIGIN="${FRONTEND_ORIGIN:-https://afya-jamii.vercel.app}"

if [ ! -f .env ]; then
  echo "error: .env not found. Copy .env.example and fill it in." >&2
  exit 1
fi

# Pull the secrets out of .env without exporting the whole file.
value_of() { grep -E "^$1=" .env | head -1 | cut -d= -f2- ; }

# The production database, kept separate from the local SQLite one so that
# development can never write to patient records.
DATABASE_URL="$(value_of PROD_DATABASE_URL)"
GROQ_API_KEY="$(value_of GROQ_API_KEY)"

if [ -z "$DATABASE_URL" ] || [ -z "$GROQ_API_KEY" ]; then
  echo "error: PROD_DATABASE_URL and GROQ_API_KEY must both be set in .env" >&2
  exit 1
fi

if ! "$FASTAPI" cloud whoami >/dev/null 2>&1; then
  echo "error: not logged in. Run:  $FASTAPI login" >&2
  exit 1
fi

echo "==> Creating the app (first run) and uploading the project"
# The first deploy registers the app and writes .fastapicloud/cloud.json.
# It is expected to fail its health check until the variables below exist.
"$FASTAPI" deploy --no-wait || true

echo
echo "==> Configuring environment"

set_var()    { echo "  $1"; "$FASTAPI" cloud env set "$1" "$2" --no-redeploy >/dev/null; }
set_secret() { echo "  $1 (secret)"; printf '%s' "$2" | "$FASTAPI" cloud env set "$1" --value-stdin --secret --no-redeploy >/dev/null; }

# A fresh signing key for this deployment. Keep it stable afterwards —
# changing it signs every user out.
SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"

set_secret SECRET_KEY   "$SECRET_KEY"
set_secret DATABASE_URL "$DATABASE_URL"
set_secret GROQ_API_KEY "$GROQ_API_KEY"

# v1.0.0 refuses to start in production with debug on or a wildcard host.
set_var ENVIRONMENT  production
set_var DEBUG        false
set_var RELOAD       false
set_var CORS_ORIGINS "$FRONTEND_ORIGIN"

# Aiven accepts unencrypted connections unless the client asks for TLS, so
# this must not be left at `disable`.
set_var DB_SSL_MODE require

set_var LOG_LEVEL                INFO
set_var LOG_FORMAT               json
set_var LLM_MODEL_NAME           openai/gpt-oss-120b
set_var LLM_TEMPERATURE          0.2
set_var PROMPT_NAME              clinical_assistant
set_var CHAT_HISTORY_TURNS       10
set_var CHAT_HISTORY_CHAR_BUDGET 6000
set_var RATE_LIMIT_ENABLED       true

# ALLOWED_HOSTS has to name the hostname FastAPI Cloud assigns, which is only
# known after the first deploy. Left permissive here and tightened below.
set_var ALLOWED_HOSTS "*"

echo
echo "==> Deploying with the configuration applied"
"$FASTAPI" deploy

echo
echo "Done. Next:"
echo "  1. Note the URL printed above."
echo "  2. Tighten the host allow-list to it:"
echo "       $FASTAPI cloud env set ALLOWED_HOSTS <your-host> "
echo "  3. Point the frontend at it:"
echo "       cd ../frontend"
echo "       vercel env rm VITE_API_BASE_URL production --yes"
echo "       echo https://<your-host> | vercel env add VITE_API_BASE_URL production"
echo "       vercel deploy --prod"
echo "  4. Check it:  curl https://<your-host>/health"
