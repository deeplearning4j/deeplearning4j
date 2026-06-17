#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=build-common.sh
source "$SCRIPT_DIR/build-common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Orchestration wrapper: build one or more platform targets.

Options:
  --platforms LIST   Comma-separated platform names or group name.
                     Groups: cpu, cuda, android, host, all
                     (default: auto-detect from get_host_platforms())
  --threads N        Override BUILD_THREADS (default: ${BUILD_THREADS:-12})
  --deploy           Add -DperformRelease to Maven (sets DEPLOY=1)
  --setup            Run auto_setup_host() before building
  --list             Print available platforms and exit
  --sdx-output DIR   Override SDX_OUTPUT_DIR
  --help             Show this message and exit

Examples:
  $(basename "$0") --platforms cpu
  $(basename "$0") --platforms cuda,cpu --threads 16
  $(basename "$0") --platforms all --deploy
  $(basename "$0") --setup --platforms host
EOF
}

# ── defaults ──────────────────────────────────────────────────────────────────
PLATFORMS_ARG=""
RUN_SETUP=0
export DEPLOY=${DEPLOY:-0}

# ── arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --platforms)
            PLATFORMS_ARG="$2"; shift 2 ;;
        --threads)
            BUILD_THREADS="$2"; export BUILD_THREADS; shift 2 ;;
        --deploy)
            export DEPLOY=1; shift ;;
        --setup)
            RUN_SETUP=1; shift ;;
        --list)
            list_platforms
            exit 0 ;;
        --sdx-output)
            SDX_OUTPUT_DIR="$2"; export SDX_OUTPUT_DIR; shift 2 ;;
        --help|-h)
            usage; exit 0 ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1 ;;
    esac
done

# ── optional host setup ───────────────────────────────────────────────────────
if [[ $RUN_SETUP -eq 1 ]]; then
    log_info "Running auto_setup_host() before build..."
    auto_setup_host
fi

# ── resolve platform list ─────────────────────────────────────────────────────
if [[ -z "$PLATFORMS_ARG" ]]; then
    log_info "No --platforms specified; auto-detecting buildable platforms for this host..."
    mapfile -t PLATFORM_LIST < <(get_host_platforms)
else
    mapfile -t PLATFORM_LIST < <(resolve_platform_group "$PLATFORMS_ARG")
fi

if [[ ${#PLATFORM_LIST[@]} -eq 0 ]]; then
    log_error "No platforms resolved. Use --list to see available platforms."
    exit 1
fi

log_info "Platforms to build: ${PLATFORM_LIST[*]}"

# ── preflight ─────────────────────────────────────────────────────────────────
preflight_check

# ── build loop ────────────────────────────────────────────────────────────────
SUCCEEDED=()
FAILED=()
SKIPPED=()

for platform in "${PLATFORM_LIST[@]}"; do
    if ! can_build_on_host "$platform"; then
        log_warn "Skipping '$platform': cannot build on this host (missing tools/GPU/SDK)."
        SKIPPED+=("$platform")
        continue
    fi

    log_info "═══ Building platform: $platform ═══"
    if build_platform "$platform"; then
        log_info "✓ $platform succeeded"
        SUCCEEDED+=("$platform")
    else
        log_error "✗ $platform FAILED"
        FAILED+=("$platform")
    fi
done

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo " Build summary"
echo "════════════════════════════════════════"
echo "  Succeeded : ${#SUCCEEDED[@]}  (${SUCCEEDED[*]:-none})"
echo "  Failed    : ${#FAILED[@]}  (${FAILED[*]:-none})"
echo "  Skipped   : ${#SKIPPED[@]}  (${SKIPPED[*]:-none})"
echo ""

if [[ -n "${SDX_OUTPUT_DIR:-}" && -d "${SDX_OUTPUT_DIR}" ]]; then
    echo "SDX artifacts collected in: $SDX_OUTPUT_DIR"
    ls -lh "$SDX_OUTPUT_DIR" 2>/dev/null || true
    echo ""
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
    log_error "One or more platforms failed. See output above for details."
    exit 1
fi

log_info "All requested platforms built successfully."
