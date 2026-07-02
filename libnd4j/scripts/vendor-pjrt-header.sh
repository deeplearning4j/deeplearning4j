#!/usr/bin/env bash
# vendor-pjrt-header.sh — download a pinned copy of pjrt_c_api.h from openxla/xla.
# Run this to update the vendored copy when the upstream header changes.
# Usage: ./libnd4j/scripts/vendor-pjrt-header.sh [COMMIT_SHA]
set -euo pipefail

REPO="openxla/xla"
FILE="xla/pjrt/c/pjrt_c_api.h"
DEST="$(dirname "$0")/../include/external/pjrt/pjrt_c_api.h"

if [ -n "${1:-}" ]; then
  SHA="$1"
else
  SHA=$(curl -fsSL "https://api.github.com/repos/${REPO}/commits?path=${FILE}&per_page=1" \
        | python3 -c "import json,sys; print(json.load(sys.stdin)[0]['sha'])")
fi

URL="https://raw.githubusercontent.com/${REPO}/${SHA}/${FILE}"
echo "Vendoring ${FILE} at commit ${SHA}"
echo "Source: ${URL}"
mkdir -p "$(dirname "$DEST")"
curl -fsSL "$URL" -o "$DEST"

PROV_COMMENT="/* Vendored: ${URL} | commit: ${SHA} | date: $(date +%Y-%m-%d) */"
# Prepend after the Apache license block
python3 - <<PY
with open('$DEST', 'r') as f:
    content = f.read()
prov = '$PROV_COMMENT\n\n'
# Insert after the closing */ of the top-level Apache license comment
idx = content.find('limitations under the License.\n')
if idx >= 0:
    idx = content.index('\n', idx) + 1
    content = content[:idx] + '\n' + prov + content[idx:]
with open('$DEST', 'w') as f:
    f.write(content)
PY

echo "Done. Placed at: $DEST"
echo "Update TpuConfiguration.cmake pinned SHA comment when you use a new commit."
