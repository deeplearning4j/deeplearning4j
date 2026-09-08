#!/usr/bin/env bash
# Safely bound immutable Android SDX SDK generations without touching build caches.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: prune-android-sdx-build-cache.sh [options]

Options:
  --build-root DIR          Android SDX build root
                            (default: $SDX_ANDROID_BUILD_ROOT or $TMPDIR/sdx-android-build)
  --retain-generations N    Generations retained per SDK family, including current
                            (default: $SDX_ANDROID_GENERATION_RETENTION or 1)
  --retain-managed-stages N Additional unreferenced CPU managed stages to retain
                            (default: $SDX_ANDROID_MANAGED_STAGE_RETENTION or 0)
  --retain-object-stages N  Additional unreferenced Native Image object stages
                            to retain (default: $SDX_ANDROID_OBJECT_STAGE_RETENTION or 1)
  --keep-cpu-sdk DIR         Preserve an explicitly selected CPU base for an AOT build
  --dry-run                 Validate and report without deleting
  -h, --help                Show this help

The active CPU/AOT symlink targets and CPU bases of retained AOT generations are
preserved. Every stage referenced by a retained generation is also preserved,
plus one unreferenced AOT object for retries after packaging failure. Increase
--retain-generations explicitly when rollback publications are needed.
Cleanup holds the pipeline lock and validates the complete plan before deletion.
Dry-run uses that same plan (only lock files may be created). Superseded
immutable publications, orphan managed/object stages beyond that bound, and
explicitly named disposable state from interrupted builds are removed. The stable
native CMake workspace, shared checksum-verified native caches, accelerator
native build/dist artifacts, ccache, and final APK candidates are outside this
script's deletion scope.
USAGE
}

fail() {
  printf 'prune-android-sdx-build-cache: %s\n' "$*" >&2
  exit 3
}

BUILD_ROOT="${SDX_ANDROID_BUILD_ROOT:-${TMPDIR:-/tmp}/sdx-android-build}"
RETAIN_GENERATIONS="${SDX_ANDROID_GENERATION_RETENTION:-1}"
RETAIN_MANAGED_STAGES="${SDX_ANDROID_MANAGED_STAGE_RETENTION:-0}"
RETAIN_OBJECT_STAGES="${SDX_ANDROID_OBJECT_STAGE_RETENTION:-1}"
DRY_RUN=0
KEEP_CPU_SDK=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-root) BUILD_ROOT="${2:?missing value for --build-root}"; shift 2 ;;
    --retain-generations) RETAIN_GENERATIONS="${2:?missing value for --retain-generations}"; shift 2 ;;
    --retain-managed-stages) RETAIN_MANAGED_STAGES="${2:?missing value for --retain-managed-stages}"; shift 2 ;;
    --retain-object-stages) RETAIN_OBJECT_STAGES="${2:?missing value for --retain-object-stages}"; shift 2 ;;
    --keep-cpu-sdk) KEEP_CPU_SDK="${2:?missing value for --keep-cpu-sdk}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[[ "$RETAIN_GENERATIONS" =~ ^[1-9][0-9]*$ ]] ||
  fail "--retain-generations must be a positive integer"
[[ "$RETAIN_MANAGED_STAGES" =~ ^[0-9]+$ ]] ||
  fail "--retain-managed-stages must be a non-negative integer"
[[ "$RETAIN_OBJECT_STAGES" =~ ^[0-9]+$ ]] ||
  fail "--retain-object-stages must be a non-negative integer"
[[ ! -L "$BUILD_ROOT" ]] || fail "build root must not be a symlink"
BUILD_ROOT="$(realpath -m -- "$BUILD_ROOT")"
[[ "$BUILD_ROOT" != / ]] || fail "refusing to use the filesystem root"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || fail "invalid dry-run mode"
if [[ -e "$BUILD_ROOT" ]]; then
  [[ -d "$BUILD_ROOT" && ! -L "$BUILD_ROOT" ]] ||
    fail "build root must be a real directory: $BUILD_ROOT"
else
  printf 'Android SDX build root does not exist; nothing to prune: %s\n' "$BUILD_ROOT"
  exit 0
fi

# The Boolean alone does not prove ownership: nested callers must pass the
# inherited flock descriptor for this exact root. Standalone cleanup never waits
# behind a build and then unexpectedly removes its diagnostic state.
command -v flock >/dev/null 2>&1 || fail "flock is required"
[[ ! -L "$BUILD_ROOT/.locks" ]] || fail "pipeline lock root must not be a symlink"
mkdir -p -- "$BUILD_ROOT/.locks"
PIPELINE_LOCK="$BUILD_ROOT/.locks/tensor-g3-offline-apk.lock"
[[ ! -L "$PIPELINE_LOCK" ]] || fail "pipeline lock must not be a symlink"
case "${SDX_ANDROID_PIPELINE_LOCK_HELD:-0}" in
  0) exec {PRUNE_LOCK_FD}>"$PIPELINE_LOCK" ;;
  1)
    PRUNE_LOCK_FD="${SDX_ANDROID_PIPELINE_LOCK_FD:-}"
    [[ "$PRUNE_LOCK_FD" =~ ^[0-9]+$ &&
       "/proc/self/fd/$PRUNE_LOCK_FD" -ef "$PIPELINE_LOCK" ]] ||
      fail "inherited pipeline lock descriptor is missing or belongs to another root"
    ;;
  *) fail "SDX_ANDROID_PIPELINE_LOCK_HELD must be 0 or 1" ;;
esac
flock -n "$PRUNE_LOCK_FD" || fail "Android SDK build is active; refusing cleanup"

# Reject redirected ownership roots before traversing or planning any removal.
for owned in cpu-sdk aot-sdk cpu-sdk/work aot-sdk/work accelerator; do
  owner="$BUILD_ROOT/$owned"
  if [[ -e "$owner" || -L "$owner" ]]; then
    [[ -d "$owner" && ! -L "$owner" && "$(realpath -e -- "$owner")" == "$owner" ]] ||
      fail "unsafe cleanup owner: $owner"
  fi
done

# Plan everything first. In particular, stage references must exclude planned
# generation removals even during --dry-run; no real deletion is needed to do so.
declare -A RETAINED_GENERATIONS=() PINNED_GENERATIONS=()
declare -a REMOVE_PATHS=() REMOVE_LABELS=()

measure_directory_kib() {
  local path="$1"
  local size_kib
  size_kib="$(du -sk -- "$path" | cut -f 1)"
  [[ "$size_kib" =~ ^[0-9]+$ ]] ||
    fail "could not measure disposable build state: $path"
  printf '%s\n' "$size_kib"
}

remove_owned_directory() {
  local candidate="$1"
  local owner="$2"
  local label="$3"
  local candidate_real owner_real size_kib

  [[ -e "$candidate" || -L "$candidate" ]] || return 0
  [[ -d "$owner" && ! -L "$owner" ]] ||
    fail "$label owner must be a real directory: $owner"
  [[ -d "$candidate" && ! -L "$candidate" ]] ||
    fail "unsafe $label candidate: $candidate"
  owner_real="$(realpath -e -- "$owner")"
  candidate_real="$(realpath -e -- "$candidate")"
  case "$candidate_real/" in
    "$owner_real"/*/) ;;
    *) fail "$label candidate escapes its owner: $candidate_real" ;;
  esac
  [[ "$candidate_real" == "$candidate" ]] || fail "$label traverses a symlink: $candidate"
  [[ -z "$(find "$candidate_real" -type l -print -quit)" ]] ||
    fail "$label contains a symlink: $candidate"
  REMOVE_PATHS+=("$candidate_real")
  REMOVE_LABELS+=("$label")
}

remove_owned_file() {
  local candidate="$1"
  local owner="$2"
  local label="$3"
  local candidate_real owner_real size_kib

  [[ -e "$candidate" || -L "$candidate" ]] || return 0
  [[ -d "$owner" && ! -L "$owner" ]] ||
    fail "$label owner must be a real directory: $owner"
  [[ -f "$candidate" && ! -L "$candidate" ]] ||
    fail "unsafe $label candidate: $candidate"
  owner_real="$(realpath -e -- "$owner")"
  candidate_real="$(realpath -e -- "$candidate")"
  [[ "$(dirname -- "$candidate_real")" == "$owner_real" ]] ||
    fail "$label candidate escapes its owner: $candidate_real"
  [[ "$candidate_real" == "$candidate" ]] || fail "$label traverses a symlink: $candidate"
  REMOVE_PATHS+=("$candidate_real")
  REMOVE_LABELS+=("$label")
}

prune_directory_pattern() {
  local parent="$1"
  local name_glob="$2"
  local label="$3"
  local candidate

  [[ -e "$parent" ]] || return 0
  [[ -d "$parent" && ! -L "$parent" ]] ||
    fail "$label parent must be a real directory: $parent"
  while IFS= read -r -d '' candidate; do
    remove_owned_directory "$candidate" "$parent" "$label"
  done < <(find "$parent" -mindepth 1 -maxdepth 1 -name "$name_glob" -print0)
}

prune_file_pattern() {
  local parent="$1"
  local name_glob="$2"
  local label="$3"
  local candidate

  [[ -e "$parent" ]] || return 0
  [[ -d "$parent" && ! -L "$parent" ]] ||
    fail "$label parent must be a real directory: $parent"
  while IFS= read -r -d '' candidate; do
    remove_owned_file "$candidate" "$parent" "$label"
  done < <(find "$parent" -mindepth 1 -maxdepth 1 -name "$name_glob" -print0)
}

prune_disposable_state() {
  local provider_root provider_lock_fd

  prune_directory_pattern "$BUILD_ROOT/cpu-sdk/work" 'generation.*' 'CPU publication work'
  prune_directory_pattern "$BUILD_ROOT/cpu-sdk/work" 'managed-stage.*' 'CPU managed-stage temporary work'
  prune_file_pattern "$BUILD_ROOT/cpu-sdk/work" 'published-native-manifest.*' 'CPU manifest temporary file'
  prune_directory_pattern "$BUILD_ROOT/aot-sdk/work" 'generation.*' 'AOT publication work'
  prune_directory_pattern "$BUILD_ROOT/aot-sdk/work" 'staging-copy-test.*' 'AOT copy-test work'
  prune_directory_pattern "$BUILD_ROOT/aot-sdk/work" 'staging-regression.*' 'AOT regression work'
  prune_directory_pattern "$BUILD_ROOT/aot-sdk/work/native-image-object-stages" \
    '.native-image-object.*' 'incomplete Native Image object publication'

  if [[ -d "$BUILD_ROOT/accelerator" && ! -L "$BUILD_ROOT/accelerator" ]]; then
    while IFS= read -r -d '' provider_root; do
      [[ -d "$provider_root" && ! -L "$provider_root" ]] ||
        fail "unsafe accelerator provider root: $provider_root"
      command -v flock >/dev/null 2>&1 ||
        fail "flock is required to prune accelerator temporary state"
      [[ ! -L "$provider_root/.build.lock" ]] || fail "provider lock must not be a symlink"
      exec {provider_lock_fd}>"$provider_root/.build.lock"
      flock -n "$provider_lock_fd" ||
        fail "accelerator provider is active; refusing cleanup: $provider_root"
      prune_directory_pattern "$provider_root" 'quarantined-maven-targets.*' \
        'accelerator Maven quarantine'
      prune_file_pattern "$provider_root/dist" 'fresh-java-builds.tmp.*' \
        'accelerator manifest temporary file'
      # Keep every provider descriptor open through application of the plan.
    done < <(find "$BUILD_ROOT/accelerator" -mindepth 1 -maxdepth 1 -print0)
  elif [[ -e "$BUILD_ROOT/accelerator" ]]; then
    fail "accelerator root must be a real directory: $BUILD_ROOT/accelerator"
  fi
}

prune_family() {
  local label="$1"
  local current_link="$2"
  local generations_dir="$3"
  local name_pattern="$4"
  local current_target=""
  local entry name path size_kib
  local retained=0
  local removed=0
  local reclaimed_kib=0
  local -a candidates=()

  [[ -e "$generations_dir" ]] || {
    printf '%s generations do not exist; nothing to prune.\n' "$label"
    return
  }
  [[ -d "$generations_dir" && ! -L "$generations_dir" ]] ||
    fail "$label generations path must be a real directory: $generations_dir"

  if [[ -e "$current_link" || -L "$current_link" ]]; then
    [[ -L "$current_link" ]] ||
      fail "$label current path is not a symlink: $current_link"
    current_target="$(realpath -e -- "$current_link")" ||
      fail "$label current symlink is broken: $current_link"
    case "$current_target/" in
      "$generations_dir"/*/) ;;
      *) fail "$label current symlink escapes its generations directory" ;;
    esac
    [[ -d "$current_target" && ! -L "$current_target" ]] ||
      fail "$label current target is not an immutable generation directory"
    [[ "$(dirname -- "$current_target")" == "$generations_dir" &&
       "$(basename -- "$current_target")" =~ $name_pattern ]] || fail "unexpected $label current generation name"
    RETAINED_GENERATIONS["$current_target"]=1
    retained=1
  fi

  while IFS= read -r -d '' entry; do
    name="${entry#* }"
    [[ "$name" =~ $name_pattern ]] || continue
    path="$generations_dir/$name"
    [[ -d "$path" && ! -L "$path" ]] ||
      fail "unsafe $label generation candidate: $path"
    candidates+=("$path")
  done < <(
    find "$generations_dir" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %f\0' |
      LC_ALL=C sort -z -nr
  )

  for path in "${candidates[@]}"; do
    if [[ -n "${PINNED_GENERATIONS[$path]:-}" && -z "${RETAINED_GENERATIONS[$path]:-}" ]]; then
      RETAINED_GENERATIONS["$path"]=1
      retained=$((retained + 1))
    fi
  done
  for path in "${candidates[@]}"; do
    [[ -z "${RETAINED_GENERATIONS[$path]:-}" ]] || continue
    if (( retained < RETAIN_GENERATIONS )); then
      RETAINED_GENERATIONS["$path"]=1
      retained=$((retained + 1))
      continue
    fi
    size_kib="$(measure_directory_kib "$path")"
    removed=$((removed + 1))
    reclaimed_kib=$((reclaimed_kib + size_kib))
    remove_owned_directory "$path" "$generations_dir" "superseded $label generation"
  done

  printf '%s retention complete: retained=%s removable=%s reclaimable_kib=%s dry_run=%s\n' \
    "$label" "$retained" "$removed" "$reclaimed_kib" "$DRY_RUN"
}

required_receipt_value() {
  local generation="$1" field="$2" key value found="" count=0
  local receipt="$generation/metadata/build-receipt"
  [[ -d "$generation/metadata" && ! -L "$generation/metadata" &&
     -f "$receipt" && ! -L "$receipt" && "$(realpath -e -- "$receipt")" == "$receipt" ]] ||
    fail "missing or unsafe retained generation receipt: $receipt"
  while IFS='=' read -r key value || [[ -n "$key" ]]; do
    [[ "$key" == "$field" ]] || continue
    found="$value"
    count=$((count + 1))
  done <"$receipt"
  [[ "$count" == 1 && -n "$found" ]] || fail "expected one $field in retained receipt: $receipt"
  printf '%s\n' "$found"
}

pin_cpu_base() {
  local base="$1" parent="$BUILD_ROOT/cpu-sdk/.android-cpu-importer-generations"
  case "$base" in
    "$parent/"*)
      [[ "$(dirname -- "$base")" == "$parent" &&
         "$(basename -- "$base")" =~ ^[0-9a-f]{64}-[0-9a-f]{64}$ &&
         -d "$base" && ! -L "$base" && "$(realpath -e -- "$base")" == "$base" ]] ||
        fail "retained AOT CPU base is missing or unsafe: $base"
      PINNED_GENERATIONS["$base"]=1
      ;;
  esac
}

prune_content_addressed_stages() {
  local label="$1"
  local stages_dir="$2"
  local generations_dir="$3"
  local receipt_key="$4"
  local retain_unreferenced="$5"
  local generation line_value entry name path size_kib
  local retained_referenced=0
  local retained_unreferenced=0
  local removed=0
  local reclaimed_kib=0
  local -a candidates=()
  declare -A referenced=()

  [[ -e "$stages_dir" ]] || {
    printf '%s stages do not exist; nothing to prune.\n' "$label"
    return
  }
  [[ -d "$stages_dir" && ! -L "$stages_dir" ]] ||
    fail "$label stages path must be a real directory: $stages_dir"
  [[ ! -e "$generations_dir" || ( -d "$generations_dir" && ! -L "$generations_dir" ) ]] ||
    fail "$label generations path must be a real directory: $generations_dir"

  for generation in "${!RETAINED_GENERATIONS[@]}"; do
    [[ "$(dirname -- "$generation")" == "$generations_dir" ]] || continue
    line_value="$(required_receipt_value "$generation" "$receipt_key")" || exit 3
    [[ "$line_value" =~ ^[0-9a-f]{64}$ ]] || fail "invalid $receipt_key in $generation"
    referenced["$line_value"]=1
  done

  while IFS= read -r -d '' entry; do
    name="${entry#* }"
    [[ "$name" =~ ^[0-9a-f]{64}$ ]] || continue
    path="$stages_dir/$name"
    [[ -d "$path" && ! -L "$path" ]] ||
      fail "unsafe $label stage candidate: $path"
    candidates+=("$path")
  done < <(
    find "$stages_dir" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %f\0' |
      LC_ALL=C sort -z -nr
  )

  for path in "${candidates[@]}"; do
    name="$(basename -- "$path")"
    if [[ -n "${referenced[$name]:-}" ]]; then
      retained_referenced=$((retained_referenced + 1))
      continue
    fi
    if (( retained_unreferenced < retain_unreferenced )); then
      retained_unreferenced=$((retained_unreferenced + 1))
      continue
    fi
    size_kib="$(measure_directory_kib "$path")"
    removed=$((removed + 1))
    reclaimed_kib=$((reclaimed_kib + size_kib))
    remove_owned_directory "$path" "$stages_dir" "unreferenced $label stage"
  done

  printf '%s stage retention complete: referenced=%s fallback=%s removable=%s reclaimable_kib=%s dry_run=%s\n' \
    "$label" "$retained_referenced" "$retained_unreferenced" "$removed" "$reclaimed_kib" "$DRY_RUN"
}

prune_disposable_state

prune_family \
  AOT \
  "$BUILD_ROOT/aot-sdk/current" \
  "$BUILD_ROOT/aot-sdk/.android-aot-generations" \
  '^[0-9a-f]{64}-[0-9a-f]{16}$'

# A retained AOT receipt can refer to a CPU SDK that is no longer current.
# Never break that chain just to meet the nominal generation count.
aot_generations="$BUILD_ROOT/aot-sdk/.android-aot-generations"
for generation in "${!RETAINED_GENERATIONS[@]}"; do
  [[ "$(dirname -- "$generation")" == "$aot_generations" ]] || continue
  base="$(required_receipt_value "$generation" base_sdk)" || exit 3
  pin_cpu_base "$base"
done
if [[ -n "$KEEP_CPU_SDK" ]]; then
  base="$(realpath -e -- "$KEEP_CPU_SDK")" || fail "explicit CPU base is missing: $KEEP_CPU_SDK"
  pin_cpu_base "$base"
fi
prune_family \
  CPU \
  "$BUILD_ROOT/cpu-sdk/current" \
  "$BUILD_ROOT/cpu-sdk/.android-cpu-importer-generations" \
  '^[0-9a-f]{64}-[0-9a-f]{64}$'

prune_content_addressed_stages \
  'CPU managed' \
  "$BUILD_ROOT/cpu-sdk/work/managed-stages" \
  "$BUILD_ROOT/cpu-sdk/.android-cpu-importer-generations" \
  managed_stage_key \
  "$RETAIN_MANAGED_STAGES"
prune_content_addressed_stages \
  'Native Image object' \
  "$BUILD_ROOT/aot-sdk/work/native-image-object-stages" \
  "$BUILD_ROOT/aot-sdk/.android-aot-generations" \
  object_stage_inputs_sha256 \
  "$RETAIN_OBJECT_STAGES"

# All validation and lock acquisition succeeded. Report/apply the exact same
# paths; in dry-run no generations were removed to compute stage reachability.
total_kib=0
REMOVE_SIZES=()
for path in "${REMOVE_PATHS[@]}"; do
  size_kib="$(measure_directory_kib "$path")"
  REMOVE_SIZES+=("$size_kib")
  total_kib=$((total_kib + size_kib))
done
for index in "${!REMOVE_PATHS[@]}"; do
  path="${REMOVE_PATHS[$index]}"
  size_kib="${REMOVE_SIZES[$index]}"
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'Would remove %s: %s (%s KiB)\n' "${REMOVE_LABELS[$index]}" "$path" "$size_kib"
  else
    chmod -R u+w -- "$path" || fail "could not make disposable state removable: $path"
    rm -rf -- "$path"
    [[ ! -e "$path" && ! -L "$path" ]] || fail "could not remove disposable state: $path"
    printf 'Removed %s: %s (%s KiB)\n' "${REMOVE_LABELS[$index]}" "$path" "$size_kib"
  fi
done
printf 'Cleanup plan: paths=%s reclaimable_kib=%s dry_run=%s\n' "${#REMOVE_PATHS[@]}" "$total_kib" "$DRY_RUN"
