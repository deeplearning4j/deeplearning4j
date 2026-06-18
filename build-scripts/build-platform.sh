#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=build-common.sh
source "$SCRIPT_DIR/build-common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") PLATFORM [OPTIONS]

Build a single platform target.

Arguments:
  PLATFORM           Name of the platform to build (see --list)

Options:
  --setup            Run setup_environment for this platform before building
  --deploy           Add -DperformRelease to Maven (sets DEPLOY=1)
  --sdx-output DIR   Override SDX_OUTPUT_DIR
  --list             Print available platforms and exit
  --help             Show this message and exit

Examples:
  $(basename "$0") cpu
  $(basename "$0") cuda --deploy
  $(basename "$0") cpu-onednn --setup
  $(basename "$0") --list
EOF
}

# ── defaults ──────────────────────────────────────────────────────────────────
PLATFORM=""
RUN_SETUP=0
export DEPLOY=${DEPLOY:-0}

# ── arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --setup)
            RUN_SETUP=1; shift ;;
        --deploy)
            export DEPLOY=1; shift ;;
        --sdx-output)
            SDX_OUTPUT_DIR="$2"; export SDX_OUTPUT_DIR; shift 2 ;;
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

if [[ -z "$PLATFORM" ]]; then
    log_error "PLATFORM argument is required."
    usage
    exit 1
fi

# ── optional platform setup ───────────────────────────────────────────────────
if [[ $RUN_SETUP -eq 1 ]]; then
    log_info "Running setup_environment for platform: $PLATFORM"
    setup_environment "$PLATFORM"
else
    preflight_check
fi

# ── host capability check ─────────────────────────────────────────────────────
if ! can_build_on_host "$PLATFORM"; then
    log_error "Cannot build '$PLATFORM' on this host. Missing required tools, GPU, or SDK."
    log_error "Re-run with --setup to attempt automatic dependency installation."
    exit 1
fi

# ── build ─────────────────────────────────────────────────────────────────────
log_info "Building platform: $PLATFORM"
build_platform "$PLATFORM"

# ── report ────────────────────────────────────────────────────────────────────
log_info "Build complete for platform: $PLATFORM"
if [[ -n "${SDX_OUTPUT_DIR:-}" && -d "${SDX_OUTPUT_DIR}" ]]; then
    log_info "SDX bindings location: $SDX_OUTPUT_DIR"
    ls -lh "$SDX_OUTPUT_DIR" 2>/dev/null || true
fi
