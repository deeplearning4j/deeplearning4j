#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  sdx-compile.sh --input <model.sdz|model.sdnb> --output <bundle.dspb-dir> [options]

Options:
  --model-id <id>          Optional model ID (default: input filename without extension)
  --targets <csv>          Optional target list to store in manifest
  --backends <csv>         Optional backend list to store in manifest
  --gpu-target <value>     Optional GPU target hint: auto|cuda|amd|vulkan|metal
  --vulkan-spirv-dir <dir>
                            Copy validated spv_<key>.spv/.meta pairs into the bundle
  --metal-library <file>    Copy a precompiled Apple .metallib into the bundle
  --hexagon-kernel-dir <dir>
                            Copy shape-keyed Qualcomm Hexagon/HTP AOT kernels
  --tensor-g5-model <file>
                            Copy a checksummed LiteRT-LM Tensor G5 derivative
  --tokenizer <file>        Copy an offline tokenizer.json into the bundle
  --llm-config <file>       Copy text-generation IO/KV/sampling metadata JSON
  --quantization-config <file>
                            Validate and copy fail-closed INT8 metadata JSON
  --packed-output <file>    Also emit a deterministic, ZIP_STORED .dspb archive
  --overwrite              Remove output directory if it already exists
  -h, --help               Show this help

Notes:
  - This is an M3 bundler utility that always emits an unpacked .dspb directory.
  - --packed-output creates the mobile-importable archive plus a SHA-256 sidecar.
  - The runtime resolves modelPath from manifest.json.
EOF
}

INPUT=""
OUTPUT=""
MODEL_ID=""
TARGETS=""
BACKENDS=""
GPU_TARGET="auto"
VULKAN_SPIRV_DIR=""
METAL_LIBRARY=""
METAL_LIBRARY_NAME=""
HEXAGON_KERNEL_DIR=""
HEXAGON_MANIFEST_NAME=""
TENSOR_G5_MODEL=""
TENSOR_G5_MODEL_SHA256=""
TOKENIZER=""
TOKENIZER_NAME=""
LLM_CONFIG=""
LLM_CONFIG_NAME=""
QUANTIZATION_CONFIG=""
QUANTIZATION_CONFIG_NAME=""
PACKED_OUTPUT=""
OVERWRITE="0"

validate_vulkan_spirv_dir() (
  local artifact_dir="$1"
  local spv_file meta_file base_name byte_count word_count magic
  local cache_abi descriptor_bindings declared_words
  local -a spv_files meta_files bindings

  shopt -s nullglob
  spv_files=("$artifact_dir"/spv_*.spv)
  meta_files=("$artifact_dir"/spv_*.meta)

  if [[ ${#spv_files[@]} -eq 0 ]]; then
    echo "Vulkan SPIR-V artifact directory contains no spv_<16hex>.spv files: $artifact_dir" >&2
    return 1
  fi
  if [[ ${#spv_files[@]} -ne ${#meta_files[@]} ]]; then
    echo "Vulkan SPIR-V artifacts must contain one .meta sidecar per .spv file: $artifact_dir" >&2
    return 1
  fi

  for spv_file in "${spv_files[@]}"; do
    base_name="$(basename "$spv_file" .spv)"
    if [[ ! "$base_name" =~ ^spv_[0-9a-f]{16}$ ]]; then
      echo "Invalid Vulkan SPIR-V artifact name: $spv_file (expected spv_<16 lowercase hex>.spv)" >&2
      return 1
    fi
    meta_file="$artifact_dir/$base_name.meta"
    if [[ ! -f "$meta_file" ]]; then
      echo "Missing Vulkan SPIR-V metadata sidecar: $meta_file" >&2
      return 1
    fi

    byte_count="$(wc -c < "$spv_file")"
    if [[ ! "$byte_count" =~ ^[0-9]+$ ]] || (( byte_count < 20 || byte_count % 4 != 0 )); then
      echo "Invalid Vulkan SPIR-V byte length in $spv_file: $byte_count" >&2
      return 1
    fi
    magic="$(od -An -N4 -tx4 "$spv_file" | tr -d '[:space:]')"
    if [[ "$magic" != "07230203" ]]; then
      echo "Invalid Vulkan SPIR-V magic in $spv_file: $magic" >&2
      return 1
    fi

    cache_abi=""
    descriptor_bindings=""
    declared_words=""
    while IFS='=' read -r key value; do
      case "$key" in
        cacheAbi) cache_abi="$value" ;;
        descriptorBindings) descriptor_bindings="$value" ;;
        spirvWords) declared_words="$value" ;;
      esac
    done < "$meta_file"

    if [[ "$cache_abi" != "vulkan-spirv-disk-cache-v2" ]]; then
      echo "Unsupported Vulkan SPIR-V cache ABI in $meta_file: ${cache_abi:-missing}" >&2
      return 1
    fi
    if [[ ! "$descriptor_bindings" =~ ^[0-9]+(;[0-9]+)*$ ]]; then
      echo "Invalid or missing descriptorBindings in $meta_file" >&2
      return 1
    fi
    IFS=';' read -r -a bindings <<< "$descriptor_bindings"
    local previous_binding=-1
    local binding
    for binding in "${bindings[@]}"; do
      if (( binding <= previous_binding )); then
        echo "descriptorBindings must be sorted and duplicate-free in $meta_file" >&2
        return 1
      fi
      previous_binding="$binding"
    done

    word_count=$((byte_count / 4))
    if [[ ! "$declared_words" =~ ^[0-9]+$ ]] || (( declared_words != word_count )); then
      echo "spirvWords does not match the SPIR-V payload in $meta_file" >&2
      return 1
    fi
  done

  for meta_file in "${meta_files[@]}"; do
    base_name="$(basename "$meta_file" .meta)"
    if [[ ! "$base_name" =~ ^spv_[0-9a-f]{16}$ ]] ||
       [[ ! -f "$artifact_dir/$base_name.spv" ]]; then
      echo "Orphan or invalid Vulkan SPIR-V metadata sidecar: $meta_file" >&2
      return 1
    fi
  done
)

validate_hexagon_kernel_dir() (
  local artifact_dir="$1"
  local kernel_file meta_file base_name cache_abi adapter_abi soc
  local range_semantics meta_start meta_end meta_shape declared_bytes declared_sha
  local file_start file_end file_shape actual_bytes actual_sha
  local -a kernel_files meta_files

  shopt -s nullglob
  kernel_files=("$artifact_dir"/hexagon_*_*_*.bin)
  meta_files=("$artifact_dir"/hexagon_*_*_*.meta)
  if [[ ${#kernel_files[@]} -eq 0 ]]; then
    echo "Hexagon artifact directory contains no hexagon_<start>_<end>_<16hex>.bin files: $artifact_dir" >&2
    return 1
  fi
  if [[ ${#kernel_files[@]} -ne ${#meta_files[@]} ]]; then
    echo "Hexagon artifacts require one .meta sidecar per .bin file: $artifact_dir" >&2
    return 1
  fi

  for kernel_file in "${kernel_files[@]}"; do
    base_name="$(basename "$kernel_file" .bin)"
    if [[ "$base_name" =~ ^hexagon_([0-9]+)_([0-9]+)_([0-9a-f]{16})$ ]]; then
      file_start="${BASH_REMATCH[1]}"
      file_end="${BASH_REMATCH[2]}"
      file_shape="${BASH_REMATCH[3]}"
    else
      echo "Invalid Hexagon AOT artifact name: $kernel_file" >&2
      return 1
    fi
    if (( file_start > file_end )); then
      echo "Hexagon AOT range must use inclusive start <= end: $kernel_file" >&2
      return 1
    fi
    if [[ ! -s "$kernel_file" ]]; then
      echo "Hexagon AOT artifact is empty: $kernel_file" >&2
      return 1
    fi
    meta_file="$artifact_dir/$base_name.meta"
    if [[ ! -f "$meta_file" ]]; then
      echo "Missing Hexagon AOT metadata sidecar: $meta_file" >&2
      return 1
    fi

    cache_abi=""
    adapter_abi=""
    soc=""
    range_semantics=""
    meta_start=""
    meta_end=""
    meta_shape=""
    declared_bytes=""
    declared_sha=""
    while IFS='=' read -r key value; do
      case "$key" in
        cacheAbi) cache_abi="$value" ;;
        adapterAbi) adapter_abi="$value" ;;
        soc) soc="$value" ;;
        rangeSemantics) range_semantics="$value" ;;
        startSlot) meta_start="$value" ;;
        endSlot) meta_end="$value" ;;
        shapeKey) meta_shape="$value" ;;
        byteSize) declared_bytes="$value" ;;
        sha256) declared_sha="$value" ;;
      esac
    done < "$meta_file"
    if [[ "$cache_abi" != "sdx-hexagon-aot-v1" ]]; then
      echo "Unsupported Hexagon cache ABI in $meta_file: ${cache_abi:-missing}" >&2
      return 1
    fi
    if [[ "$adapter_abi" != "1" ]]; then
      echo "Unsupported Hexagon adapter ABI in $meta_file: ${adapter_abi:-missing}" >&2
      return 1
    fi
    if [[ ! "$soc" =~ ^[A-Za-z0-9._-]+$ ]]; then
      echo "Missing or invalid Qualcomm SoC identifier in $meta_file" >&2
      return 1
    fi
    if [[ "$range_semantics" != "inclusive" ||
          "$meta_start" != "$file_start" ||
          "$meta_end" != "$file_end" ||
          "$meta_shape" != "$file_shape" ]]; then
      echo "Hexagon AOT identity metadata does not match $base_name" >&2
      return 1
    fi

    actual_bytes="$(wc -c < "$kernel_file" | tr -d '[:space:]')"
    if [[ ! "$declared_bytes" =~ ^[0-9]+$ ||
          "$declared_bytes" != "$actual_bytes" ]]; then
      echo "Hexagon AOT byteSize mismatch in $meta_file" >&2
      return 1
    fi
    if command -v sha256sum >/dev/null 2>&1; then
      actual_sha="$(sha256sum "$kernel_file" | awk '{print $1}')"
    elif command -v shasum >/dev/null 2>&1; then
      actual_sha="$(shasum -a 256 "$kernel_file" | awk '{print $1}')"
    else
      echo "sha256sum or shasum is required for Hexagon AOT validation" >&2
      return 1
    fi
    if [[ ! "$declared_sha" =~ ^[0-9a-f]{64}$ ||
          "$declared_sha" != "$actual_sha" ]]; then
      echo "Hexagon AOT SHA-256 mismatch in $meta_file" >&2
      return 1
    fi
  done

  for meta_file in "${meta_files[@]}"; do
    base_name="$(basename "$meta_file" .meta)"
    if [[ ! "$base_name" =~ ^hexagon_[0-9]+_[0-9]+_[0-9a-f]{16}$ ]] ||
       [[ ! -f "$artifact_dir/$base_name.bin" ]]; then
      echo "Orphan or invalid Hexagon AOT metadata sidecar: $meta_file" >&2
      return 1
    fi
  done
)

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
    --vulkan-spirv-dir)
      VULKAN_SPIRV_DIR="${2:-}"
      shift 2
      ;;
    --metal-library)
      METAL_LIBRARY="${2:-}"
      shift 2
      ;;
    --hexagon-kernel-dir)
      HEXAGON_KERNEL_DIR="${2:-}"
      shift 2
      ;;
    --tensor-g5-model)
      TENSOR_G5_MODEL="${2:-}"
      shift 2
      ;;
    --tokenizer)
      TOKENIZER="${2:-}"
      shift 2
      ;;
    --llm-config)
      LLM_CONFIG="${2:-}"
      shift 2
      ;;
    --quantization-config)
      QUANTIZATION_CONFIG="${2:-}"
      shift 2
      ;;
    --packed-output)
      PACKED_OUTPUT="${2:-}"
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
if [[ -n "$PACKED_OUTPUT" && "$PACKED_OUTPUT" != *.dspb ]]; then
  echo "Packed output must use the .dspb extension: $PACKED_OUTPUT" >&2
  exit 1
fi
if [[ -n "$PACKED_OUTPUT" && "$PACKED_OUTPUT" == "$OUTPUT" ]]; then
  echo "--packed-output must differ from the unpacked --output path" >&2
  exit 1
fi
if [[ -n "$VULKAN_SPIRV_DIR" && ! -d "$VULKAN_SPIRV_DIR" ]]; then
  echo "Vulkan SPIR-V artifact directory not found: $VULKAN_SPIRV_DIR" >&2
  exit 1
fi
if [[ -n "$VULKAN_SPIRV_DIR" ]]; then
  validate_vulkan_spirv_dir "$VULKAN_SPIRV_DIR"
fi
if [[ -n "$METAL_LIBRARY" && ! -f "$METAL_LIBRARY" ]]; then
  echo "Metal library not found: $METAL_LIBRARY" >&2
  exit 1
fi
if [[ -n "$HEXAGON_KERNEL_DIR" && ! -d "$HEXAGON_KERNEL_DIR" ]]; then
  echo "Hexagon AOT artifact directory not found: $HEXAGON_KERNEL_DIR" >&2
  exit 1
fi
if [[ -n "$HEXAGON_KERNEL_DIR" ]]; then
  validate_hexagon_kernel_dir "$HEXAGON_KERNEL_DIR"
fi
if [[ -n "$TENSOR_G5_MODEL" ]]; then
  if [[ ! -s "$TENSOR_G5_MODEL" || "$TENSOR_G5_MODEL" != *.litertlm ]]; then
    echo "Tensor G5 derivative must be a non-empty .litertlm file: $TENSOR_G5_MODEL" >&2
    exit 1
  fi
fi
if [[ -n "$TOKENIZER" && ! -f "$TOKENIZER" ]]; then
  echo "Tokenizer file not found: $TOKENIZER" >&2
  exit 1
fi
if [[ -n "$LLM_CONFIG" && ! -f "$LLM_CONFIG" ]]; then
  echo "LLM config file not found: $LLM_CONFIG" >&2
  exit 1
fi
if [[ -n "$QUANTIZATION_CONFIG" && ! -f "$QUANTIZATION_CONFIG" ]]; then
  echo "Quantization config file not found: $QUANTIZATION_CONFIG" >&2
  exit 1
fi
if [[ -n "$QUANTIZATION_CONFIG" ]]; then
  python3 "$SCRIPT_DIR/mobile/validate-int8-quantization.py" "$QUANTIZATION_CONFIG"
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
if [[ -n "$PACKED_OUTPUT" && -e "$PACKED_OUTPUT" && "$OVERWRITE" != "1" ]]; then
  echo "Packed output already exists: $PACKED_OUTPUT (use --overwrite to replace)" >&2
  exit 1
fi

mkdir -p "$OUTPUT/graph" "$OUTPUT/weights" "$OUTPUT/segments"

if [[ -n "$VULKAN_SPIRV_DIR" ]]; then
  mkdir -p "$OUTPUT/artifacts/vulkan/spirv"
  cp "$VULKAN_SPIRV_DIR"/spv_*.spv "$VULKAN_SPIRV_DIR"/spv_*.meta \
    "$OUTPUT/artifacts/vulkan/spirv/"
fi
if [[ -n "$METAL_LIBRARY" ]]; then
  mkdir -p "$OUTPUT/artifacts/metal"
  METAL_LIBRARY_NAME="$(basename "$METAL_LIBRARY")"
  cp "$METAL_LIBRARY" "$OUTPUT/artifacts/metal/$METAL_LIBRARY_NAME"
fi
if [[ -n "$HEXAGON_KERNEL_DIR" ]]; then
  mkdir -p "$OUTPUT/artifacts/hexagon/kernels"
  cp "$HEXAGON_KERNEL_DIR"/hexagon_*_*_*.bin \
    "$HEXAGON_KERNEL_DIR"/hexagon_*_*_*.meta \
    "$OUTPUT/artifacts/hexagon/kernels/"
  if [[ -f "$HEXAGON_KERNEL_DIR/hexagon-aot-manifest.json" ]]; then
    HEXAGON_MANIFEST_NAME="hexagon-aot-manifest.json"
    cp "$HEXAGON_KERNEL_DIR/$HEXAGON_MANIFEST_NAME" \
      "$OUTPUT/artifacts/hexagon/kernels/$HEXAGON_MANIFEST_NAME"
  fi
fi
if [[ -n "$TENSOR_G5_MODEL" ]]; then
  mkdir -p "$OUTPUT/artifacts/tensor-g5"
  cp "$TENSOR_G5_MODEL" "$OUTPUT/artifacts/tensor-g5/model.litertlm"
  if command -v sha256sum >/dev/null 2>&1; then
    TENSOR_G5_MODEL_SHA256="$(sha256sum "$OUTPUT/artifacts/tensor-g5/model.litertlm" | awk '{print $1}')"
  elif command -v shasum >/dev/null 2>&1; then
    TENSOR_G5_MODEL_SHA256="$(shasum -a 256 "$OUTPUT/artifacts/tensor-g5/model.litertlm" | awk '{print $1}')"
  else
    echo "sha256sum or shasum is required for Tensor G5 artifact validation" >&2
    exit 1
  fi
fi
if [[ -n "$TOKENIZER" ]]; then
  mkdir -p "$OUTPUT/assets/tokenizer"
  TOKENIZER_NAME="$(basename "$TOKENIZER")"
  cp "$TOKENIZER" "$OUTPUT/assets/tokenizer/$TOKENIZER_NAME"
fi
if [[ -n "$LLM_CONFIG" ]]; then
  mkdir -p "$OUTPUT/metadata"
  LLM_CONFIG_NAME="$(basename "$LLM_CONFIG")"
  cp "$LLM_CONFIG" "$OUTPUT/metadata/$LLM_CONFIG_NAME"
fi
if [[ -n "$QUANTIZATION_CONFIG" ]]; then
  mkdir -p "$OUTPUT/metadata"
  QUANTIZATION_CONFIG_NAME="$(basename "$QUANTIZATION_CONFIG")"
  cp "$QUANTIZATION_CONFIG" "$OUTPUT/metadata/$QUANTIZATION_CONFIG_NAME"
fi

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
  AUTO|CUDA|AMD|VULKAN|METAL)
    ;;
  *)
    echo "Invalid --gpu-target value: $GPU_TARGET (expected auto|cuda|amd|vulkan|metal)" >&2
    exit 2
    ;;
esac

compiled_artifact_entries=""
artifact_separator=""
if [[ -n "$VULKAN_SPIRV_DIR" ]]; then
  compiled_artifact_entries+="${artifact_separator}\"vulkanSpirv\":\"artifacts/vulkan/spirv\""
  artifact_separator=","
fi
if [[ -n "$METAL_LIBRARY" ]]; then
  compiled_artifact_entries+="${artifact_separator}\"metalLibrary\":\"artifacts/metal/$METAL_LIBRARY_NAME\""
  artifact_separator=","
fi
if [[ -n "$HEXAGON_KERNEL_DIR" ]]; then
  compiled_artifact_entries+="${artifact_separator}\"hexagonKernels\":\"artifacts/hexagon/kernels\""
  artifact_separator=","
  if [[ -n "$HEXAGON_MANIFEST_NAME" ]]; then
    compiled_artifact_entries+="${artifact_separator}\"hexagonManifest\":\"artifacts/hexagon/kernels/$HEXAGON_MANIFEST_NAME\""
  fi
fi
if [[ -n "$TENSOR_G5_MODEL" ]]; then
  compiled_artifact_entries+="${artifact_separator}\"tensorG5LiteRtLm\":{\"path\":\"artifacts/tensor-g5/model.litertlm\",\"sha256\":\"$TENSOR_G5_MODEL_SHA256\"}"
  artifact_separator=","
fi
compiled_artifacts_json="{$compiled_artifact_entries}"

text_generation_json="{}"
if [[ -n "$TOKENIZER" && -n "$LLM_CONFIG" ]]; then
  text_generation_json="{\"tokenizerPath\":\"assets/tokenizer/$TOKENIZER_NAME\",\"configPath\":\"metadata/$LLM_CONFIG_NAME\"}"
elif [[ -n "$TOKENIZER" ]]; then
  text_generation_json="{\"tokenizerPath\":\"assets/tokenizer/$TOKENIZER_NAME\"}"
elif [[ -n "$LLM_CONFIG" ]]; then
  text_generation_json="{\"configPath\":\"metadata/$LLM_CONFIG_NAME\"}"
fi

quantization_json="{}"
if [[ -n "$QUANTIZATION_CONFIG" ]]; then
  quantization_json="{\"configPath\":\"metadata/$QUANTIZATION_CONFIG_NAME\"}"
fi

cat > "$OUTPUT/manifest.json" <<EOF
{
  "formatVersion": 1,
  "modelId": "$MODEL_ID",
  "producer": {
    "tool": "sdx-compile.sh",
    "version": "m3"
  },
  "modelPath": "graph/$input_name",
  "targets": $targets_json,
  "runtimeRequirements": [],
  "preferredBackends": $backends_json,
  "gpuTarget": "$gpu_target_norm",
  "compiledArtifacts": $compiled_artifacts_json,
  "textGeneration": $text_generation_json,
  "quantization": $quantization_json,
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

if [[ -n "$PACKED_OUTPUT" ]]; then
  python3 - "$OUTPUT" "$PACKED_OUTPUT" <<'PY'
import hashlib
import os
import stat
import sys
import zipfile
from pathlib import Path

root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).expanduser().resolve()
try:
    output.relative_to(root)
except ValueError:
    pass
else:
    raise SystemExit("packed output must be outside the unpacked bundle directory")

output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_name(output.name + ".tmp")
temporary.unlink(missing_ok=True)
try:
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise SystemExit(f"bundle contains a symlink: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative, (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_STORED
            with path.open("rb") as source, archive.open(
                info, "w", force_zip64=True
            ) as destination:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    destination.write(chunk)
    os.replace(temporary, output)
except BaseException:
    temporary.unlink(missing_ok=True)
    raise

digest = hashlib.sha256()
with output.open("rb") as source:
    while True:
        chunk = source.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
output.with_suffix(output.suffix + ".sha256").write_text(
    f"{digest.hexdigest()}  {output.name}\n",
    encoding="utf-8",
)
PY
fi

echo "Bundle created: $OUTPUT"
echo "Manifest: $OUTPUT/manifest.json"
if [[ -n "$PACKED_OUTPUT" ]]; then
  echo "Packed bundle: $PACKED_OUTPUT"
  echo "Packed SHA-256: $PACKED_OUTPUT.sha256"
fi
