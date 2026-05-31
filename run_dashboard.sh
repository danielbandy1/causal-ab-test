#!/usr/bin/env bash
# Launch the Streamlit dashboard locally.
# Usage: ./run_dashboard.sh [port]
set -uo pipefail
PORT=${1:-8502}
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/.venv/bin/streamlit" run "$DIR/app.py" \
    --server.port "$PORT" \
    --server.headless true \
    --browser.gatherUsageStats false
