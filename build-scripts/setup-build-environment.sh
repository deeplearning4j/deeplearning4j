#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=build-common.sh
source "$SCRIPT_DIR/build-common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") [PLATFORM | --all]

Install build dependencies for one or more platforms.

Arguments:
  PLATFORM   Install deps for this specific platform (see --list)
  --all      Install deps for everything this host can build
  (none)     Auto-detect host type and install deps for it

Options:
  --list     Print available platforms and exit
  --help     Show this message and exit

Examples:
  $(basename "$0")
  $(basename "$0") cpu
  $(basename "$0") cuda
  $(basename "$0") --all
EOF
}

# ── arg parsing ───────────────────────────────────────────────────────────────
PLATFORM=""
SETUP_ALL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            SETUP_ALL=1; shift ;;
        --list)
            list_platforms
            exit 0 ;;
        --help|-h)
            usage; exit 0 ;;
        -*)
            log_error "Unknown option: $1"
            usage
            exit 1 ;;
        *)
            if [[ -z "$PLATFORM" ]]; then
                PLATFORM="$1"; shift
            else
                log_error "Unexpected argument: $1"
                usage
                exit 1
            fi ;;
    esac
done

# ── dispatch ──────────────────────────────────────────────────────────────────
if [[ $SETUP_ALL -eq 1 ]]; then
    log_info "Setting up dependencies for all buildable platforms on this host..."
    auto_setup_host
elif [[ -n "$PLATFORM" ]]; then
    log_info "Setting up dependencies for platform: $PLATFORM"
    setup_environment "$PLATFORM"
else
    log_info "Auto-detecting host type and setting up dependencies..."
    auto_setup_host
fi

log_info "Environment setup complete."
