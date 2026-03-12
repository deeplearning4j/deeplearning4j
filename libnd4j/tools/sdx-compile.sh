#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sdx-compile.sh --input <model.sdz|model.sdnb> --output <bundle.dspb-dir> [options]

Options:
  --model-id <id>          Optional model ID (default: input filename without extension)
  --targets <csv>          Optional target list to store in manifest
  --backends <csv>         Optional backend list to store in manifest
  --gpu-target <value>     Optional GPU target hint: auto|cuda|amd
  --overwrite              Remove output directory if it already exists
  -h, --help               Show this help

Notes:
  - This is an M1 bundler utility that emits an unpacked .dspb directory.
  - The runtime resolves modelPath from manifest.json.
EOF
}

INPUT=""
OUTPUT=""
MODEL_ID=""
TARGETS=""
BACKENDS=""
GPU_TARGET="auto"
OVERWRITE="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT="${2:-}"
      shift 2
      ;;
    --output)
      OUTPUT="${2:-}"
      shift 2
      ;;
    --model-id)
      MODEL_ID="${2:-}"
      shift 2
      ;;
    --targets)
      TARGETS="${2:-}"
      shift 2
      ;;
    --backends)
      BACKENDS="${2:-}"
      shift 2
      ;;
    --gpu-target)
      GPU_TARGET="${2:-}"
      shift 2
      ;;
    --overwrite)
      OVERWRITE="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
  echo "Both --input and --output are required." >&2
  usage >&2
  exit 2
fi

if [[ ! -f "$INPUT" ]]; then
  echo "Input model file not found: $INPUT" >&2
  exit 1
fi

if [[ -z "$MODEL_ID" ]]; then
  base_name="$(basename "$INPUT")"
  MODEL_ID="${base_name%.*}"
fi

if [[ -e "$OUTPUT" ]]; then
  if [[ "$OVERWRITE" != "1" ]]; then
    echo "Output path already exists: $OUTPUT (use --overwrite to replace)" >&2
    exit 1
  fi
  rm -rf "$OUTPUT"
fi

mkdir -p "$OUTPUT/graph" "$OUTPUT/weights" "$OUTPUT/segments"

input_name="$(basename "$INPUT")"
cp "$INPUT" "$OUTPUT/graph/$input_name"

checksum=""
if command -v sha256sum >/dev/null 2>&1; then
  checksum="$(sha256sum "$OUTPUT/graph/$input_name" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  checksum="$(shasum -a 256 "$OUTPUT/graph/$input_name" | awk '{print $1}')"
fi

json_array_from_csv() {
  local csv="$1"
  local out=""
  IFS=',' read -r -a arr <<< "$csv"
  for item in "${arr[@]}"; do
    trimmed="$(echo "$item" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    if [[ -z "$trimmed" ]]; then
      continue
    fi
    if [[ -n "$out" ]]; then
      out+=","
    fi
    out+="\"$trimmed\""
  done
  echo "[$out]"
}

targets_json="$(json_array_from_csv "$TARGETS")"
backends_json="$(json_array_from_csv "$BACKENDS")"

gpu_target_norm="$(echo "$GPU_TARGET" | tr '[:lower:]' '[:upper:]')"
case "$gpu_target_norm" in
  AUTO|CUDA|AMD)
    ;;
  *)
    echo "Invalid --gpu-target value: $GPU_TARGET (expected auto|cuda|amd)" >&2
    exit 2
    ;;
esac

cat > "$OUTPUT/manifest.json" <<EOF
{
  "formatVersion": 1,
  "modelId": "$MODEL_ID",
  "producer": {
    "tool": "sdx-compile.sh",
    "version": "m1"
  },
  "modelPath": "graph/$input_name",
  "targets": $targets_json,
  "runtimeRequirements": [],
  "preferredBackends": $backends_json,
  "gpuTarget": "$gpu_target_norm",
  "weights": {
    "path": "",
    "sha256": "$checksum"
  },
  "compatibility": {
    "minRuntimeAbi": 1,
    "maxRuntimeAbi": 1
  }
}
EOF

echo "Bundle created: $OUTPUT"
echo "Manifest: $OUTPUT/manifest.json"
