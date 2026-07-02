#!/usr/bin/env bash
#
# fetch-pjrt-plugin.sh — obtain a PJRT plugin shared library with NO python/pip.
#
# PJRT plugins are published as PyPI wheels, but a wheel is just a ZIP archive and
# PyPI exposes a JSON metadata API, so the plugin .so can be fetched with only
# curl + jq + unzip — no interpreter, no pip, no virtualenv.
#
# The extracted .so exports the standard GetPjrtApi() symbol and is loaded at
# runtime by libnd4j's PjrtClientManager (dlopen). Point the manager at it via
#   export PJRT_PLUGIN_LIBRARY_PATH=<printed path>
#
# Usage:
#   fetch-pjrt-plugin.sh <rocm|tpu|cpu> [dest-dir]
#
# Prints the absolute path of the extracted .so to STDOUT (last line); all
# progress logging goes to STDERR so `PJRT_SO=$(fetch-pjrt-plugin.sh rocm)` works.
#
# Plugin → PyPI package → .so inside the wheel:
#   rocm : jax-rocm7-pjrt          -> xla_rocm_plugin.so       (AMD RDNA + CDNA/MI300X)
#   tpu  : libtpu                  -> libtpu.so                (Google Cloud TPU)
#   cpu  : jaxlib                  -> libpjrt_c_api_cpu_dynamic.so (hardware-free smoke)
#
set -euo pipefail

log() { echo "[fetch-pjrt-plugin] $*" >&2; }

PLUGIN="${1:-}"
DEST="${2:-${RUNNER_TEMP:-/tmp}/pjrt-plugin-${PLUGIN}}"

if [ -z "$PLUGIN" ]; then
  log "ERROR: plugin type required (rocm|tpu|cpu)"
  exit 2
fi
for tool in curl jq unzip; do
  command -v "$tool" >/dev/null 2>&1 || { log "ERROR: '$tool' not found (need curl, jq, unzip)"; exit 3; }
done

case "$PLUGIN" in
  rocm) PKGS="jax-rocm7-pjrt jax-rocm60-pjrt"; SO_GLOB="xla_rocm*plugin*.so"; SO_NAME="xla_rocm_plugin.so" ;;
  tpu)  PKGS="libtpu libtpu-nightly";          SO_GLOB="libtpu.so";           SO_NAME="libtpu.so" ;;
  cpu)  PKGS="jaxlib";                          SO_GLOB="*pjrt_c_api_cpu*.so"; SO_NAME="libpjrt_c_api_cpu_dynamic.so" ;;
  *)    log "ERROR: unknown plugin '$PLUGIN' (expected rocm|tpu|cpu)"; exit 2 ;;
esac

mkdir -p "$DEST"

# Resolve a manylinux/linux x86_64 wheel URL from the PyPI JSON API.
wheel_url_for() {
  local pkg="$1"
  local meta
  meta=$(curl -sfL "https://pypi.org/pypi/${pkg}/json" 2>/dev/null) || return 1
  # Prefer a platform-specific manylinux/linux x86_64 wheel; fall back to any wheel.
  echo "$meta" | jq -r '
    ([.urls[] | select(.packagetype=="bdist_wheel")
               | select(.filename | test("manylinux.*x86_64|linux_x86_64"))
               | .url] | first)
    // ([.urls[] | select(.packagetype=="bdist_wheel") | .url] | first)
    // empty'
}

WHEEL_URL=""
for pkg in $PKGS; do
  log "querying PyPI for ${pkg} ..."
  WHEEL_URL=$(wheel_url_for "$pkg" || true)
  if [ -n "$WHEEL_URL" ]; then
    log "found wheel: $WHEEL_URL"
    break
  fi
  log "no usable wheel for ${pkg}, trying next"
done

if [ -z "$WHEEL_URL" ]; then
  log "ERROR: could not locate a $PLUGIN PJRT wheel on PyPI (tried: $PKGS)"
  exit 4
fi

WHEEL="$DEST/plugin.whl"
log "downloading wheel ..."
curl -fL "$WHEEL_URL" -o "$WHEEL"

log "extracting .so from wheel (zip) ..."
# Try a targeted extract first, then fall back to extracting everything.
unzip -o -q "$WHEEL" "*.so*" -d "$DEST/unpacked" 2>/dev/null \
  || unzip -o -q "$WHEEL" -d "$DEST/unpacked"

# Locate the plugin .so: exact name first, then the glob.
SO=$(find "$DEST/unpacked" -name "$SO_NAME" 2>/dev/null | head -1)
if [ -z "$SO" ]; then
  SO=$(find "$DEST/unpacked" -name "$SO_GLOB" 2>/dev/null | head -1)
fi
# Last resort for cpu/rocm: any .so exporting GetPjrtApi.
if [ -z "$SO" ] && command -v nm >/dev/null 2>&1; then
  while IFS= read -r cand; do
    if nm -D "$cand" 2>/dev/null | grep -q "GetPjrtApi"; then SO="$cand"; break; fi
  done < <(find "$DEST/unpacked" -name "*.so*" 2>/dev/null)
fi

if [ -z "$SO" ]; then
  log "ERROR: no PJRT plugin .so found inside the wheel"
  find "$DEST/unpacked" -name "*.so*" 2>/dev/null | head -20 >&2
  exit 5
fi

SO_ABS=$(readlink -f "$SO")
log "plugin .so: $SO_ABS"

# Static sanity: confirm the GetPjrtApi entry point is exported (no execution).
# Plugins export it as a VERSIONED symbol (e.g. GetPjrtApi@@VERS_1.0); readelf's
# dynamic-symbol table is the reliable way to see that (nm -D can miss it).
sym_check() {
  if command -v readelf >/dev/null 2>&1; then
    readelf -sW --dyn-syms "$SO_ABS" 2>/dev/null | grep -q "GetPjrtApi" && return 0
  fi
  if command -v nm >/dev/null 2>&1; then
    nm -D "$SO_ABS" 2>/dev/null | grep -q "GetPjrtApi" && return 0
  fi
  return 1
}
if sym_check; then
  log "verified: exports GetPjrtApi (dlsym-resolvable entry point)"
else
  log "WARNING: GetPjrtApi symbol not found (plugin may still be valid; readelf/nm unavailable?)"
fi

# Report unmet shared-object deps (e.g. ROCm runtime on a non-AMD host). Informational:
# the plugin dlopens only where its backend runtime is installed (ROCm 7 for the AMD plugin).
if command -v ldd >/dev/null 2>&1; then
  MISSING=$(ldd "$SO_ABS" 2>/dev/null | grep -c "not found" || true)
  if [ "${MISSING:-0}" -gt 0 ]; then
    log "note: $MISSING shared-object dependencies unresolved on this host — the plugin will"
    log "      dlopen only where its backend runtime is present (ROCm 7 for xla_rocm_plugin.so)."
  fi
fi

# The path is the deliverable — stdout only.
echo "$SO_ABS"
