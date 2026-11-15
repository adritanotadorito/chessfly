#!/usr/bin/env bash
# Run a frontier agent (OpenCode harness) on ChessFly: Counterplay via Harbor.
#   bash scripts/eval_frontier.sh list                      # models your key can use
#   bash scripts/eval_frontier.sh gemini/gemini-2.5-pro 3   # 3 trials
# Loads API keys from ./.env if present (never printed), runs N trials, then
# refuses to keep logs that contain the key verbatim, and refreshes the report.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:?usage: eval_frontier.sh <provider/model> [n_trials]}"
N="${2:-1}"
eval "$(python3 scripts/_env_exports.py)"
KEY="${GEMINI_API_KEY:-${GOOGLE_API_KEY:-${GOOGLE_GENERATIVE_AI_API_KEY:-}}}"
if [[ "$MODEL" == google/* || "$MODEL" == gemini/* || "$MODEL" == list ]]; then
  [ -n "$KEY" ] || { echo "No GEMINI_API_KEY/GOOGLE_API_KEY in env or .env" >&2; exit 1; }
  export GEMINI_API_KEY="$KEY" GOOGLE_API_KEY="$KEY" GOOGLE_GENERATIVE_AI_API_KEY="$KEY"
fi
if [ "$MODEL" = "list" ]; then  # list Gemini models this key can call
  curl -s -H "x-goog-api-key: $KEY" "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('error',{}).get('message',''));[print('google/'+m['name'].split('/')[1]) for m in d.get('models',[]) if 'generateContent' in m.get('supportedGenerationMethods',[])]"
  exit 0
fi
# Harbor/LiteLLM name Google's provider "gemini/..." (google/... is not resolved and no key is forwarded).
MODEL="${MODEL/#google\//gemini/}"
JOB="frontier-$(echo "$MODEL" | tr '/:' '--')-$(date +%s)"
harbor run -p ./harbor_task -a opencode -m "$MODEL" --api-key-name GEMINI_API_KEY -k "$N" -n 1 -o harbor_runs --job-name "$JOB" || true
if [ -n "$KEY" ] && grep -rqF -- "$KEY" "harbor_runs/$JOB" 2>/dev/null; then
  echo "!! API key found verbatim in harbor_runs/$JOB; redacting." >&2
  grep -rlF -- "$KEY" "harbor_runs/$JOB" | while read -r f; do
    python3 - "$f" <<'PY'
import os,sys; p=sys.argv[1]; k=os.environ["GEMINI_API_KEY"]
d=open(p,'rb').read().replace(k.encode(),b"<REDACTED>"); open(p,'wb').write(d)
PY
  done
fi
python3 scripts/compare_runs.py
