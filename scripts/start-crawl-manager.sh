#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/.kompile/state/logs"
PID_DIR="$ROOT/.kompile/state/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"
COMMAND="${KOMPILE_CRAWL_MANAGER_COMMAND:-}"
if [ -z "$COMMAND" ]; then
  # No explicit command set — fall back to the kompile CLI (discovers the installed component).
  if command -v kompile >/dev/null 2>&1; then
    cd "$ROOT"
    COMMAND="kompile project serve --crawl-manager-only"
  else
    echo "Set KOMPILE_CRAWL_MANAGER_COMMAND to start the Kompile crawl manager for this project." >&2
    echo "  (or install the kompile CLI so this script can auto-discover and start it)" >&2
    exit 2
  fi
fi
nohup bash -lc "$COMMAND" > "$LOG_DIR/crawl-manager.log" 2>&1 &
echo $! > "$PID_DIR/crawl-manager.pid"
echo "Started crawl-manager with PID $(cat "$PID_DIR/crawl-manager.pid")"
