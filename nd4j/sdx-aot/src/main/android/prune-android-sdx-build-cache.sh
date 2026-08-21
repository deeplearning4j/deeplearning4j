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
                            (default: $SDX_ANDROID_GENERATION_RETENTION or 2)
  --retain-managed-stages N Additional unreferenced CPU managed stages to retain
                            (default: $SDX_ANDROID_MANAGED_STAGE_RETENTION or 1)
  --retain-object-stages N  Additional unreferenced Native Image object stages
                            to retain (default: $SDX_ANDROID_OBJECT_STAGE_RETENTION or 1)
  --dry-run                 Validate and report without deleting
  -h, --help                Show this help

The active CPU/AOT symlink targets and one rollback generation are preserved by
default. Every content-addressed stage referenced by a retained generation is
also preserved, plus one unreferenced fallback stage by default. Superseded
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
RETAIN_GENERATIONS="${SDX_ANDROID_GENERATION_RETENTION:-2}"
RETAIN_MANAGED_STAGES="${SDX_ANDROID_MANAGED_STAGE_RETENTION:-1}"
RETAIN_OBJECT_STAGES="${SDX_ANDROID_OBJECT_STAGE_RETENTION:-1}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-root) BUILD_ROOT="${2:?missing value for --build-root}"; shift 2 ;;
    --retain-generations) RETAIN_GENERATIONS="${2:?missing value for --retain-generations}"; shift 2 ;;
    --retain-managed-stages) RETAIN_MANAGED_STAGES="${2:?missing value for --retain-managed-stages}"; shift 2 ;;
    --retain-object-stages) RETAIN_OBJECT_STAGES="${2:?missing value for --retain-object-stages}"; shift 2 ;;
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
  size_kib="$(measure_directory_kib "$candidate_real")"
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'Would remove disposable %s: %s (%s KiB)\n' "$label" "$candidate_real" "$size_kib"
    return 0
  fi
  chmod -R u+w -- "$candidate_real" ||
    fail "could not make disposable $label removable: $candidate_real"
  rm -rf -- "$candidate_real"
  [[ ! -e "$candidate_real" && ! -L "$candidate_real" ]] ||
    fail "could not remove disposable $label: $candidate_real"
  printf 'Removed disposable %s: %s (%s KiB)\n' "$label" "$candidate_real" "$size_kib"
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
  size_kib="$(du -k -- "$candidate_real" | cut -f 1)"
  [[ "$size_kib" =~ ^[0-9]+$ ]] ||
    fail "could not measure disposable $label: $candidate_real"
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'Would remove disposable %s: %s (%s KiB)\n' "$label" "$candidate_real" "$size_kib"
    return 0
  fi
  chmod u+w -- "$candidate_real" ||
    fail "could not make disposable $label removable: $candidate_real"
  rm -f -- "$candidate_real"
  [[ ! -e "$candidate_real" && ! -L "$candidate_real" ]] ||
    fail "could not remove disposable $label: $candidate_real"
  printf 'Removed disposable %s: %s (%s KiB)\n' "$label" "$candidate_real" "$size_kib"
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
  prune_directory_pattern "$BUILD_ROOT/aot-sdk/work/native-image-object-stages" \
    '.native-image-object.*' 'incomplete Native Image object publication'

  if [[ -d "$BUILD_ROOT/accelerator" && ! -L "$BUILD_ROOT/accelerator" ]]; then
    while IFS= read -r -d '' provider_root; do
      [[ -d "$provider_root" && ! -L "$provider_root" ]] ||
        fail "unsafe accelerator provider root: $provider_root"
      command -v flock >/dev/null 2>&1 ||
        fail "flock is required to prune accelerator temporary state"
      exec {provider_lock_fd}>"$provider_root/.build.lock"
      flock -n "$provider_lock_fd" ||
        fail "accelerator provider is active; refusing cleanup: $provider_root"
      prune_directory_pattern "$provider_root" 'quarantined-maven-targets.*' \
        'accelerator Maven quarantine'
      prune_file_pattern "$provider_root/dist" 'fresh-java-builds.tmp.*' \
        'accelerator manifest temporary file'
      exec {provider_lock_fd}>&-
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
    if [[ -n "$current_target" && "$path" -ef "$current_target" ]]; then
      continue
    fi
    if (( retained < RETAIN_GENERATIONS )); then
      retained=$((retained + 1))
      continue
    fi
    size_kib="$(measure_directory_kib "$path")"
    removed=$((removed + 1))
    reclaimed_kib=$((reclaimed_kib + size_kib))
    if [[ "$DRY_RUN" == 1 ]]; then
      printf 'Would remove superseded %s generation: %s (%s KiB)\n' "$label" "$path" "$size_kib"
      continue
    fi
    chmod -R u+w -- "$path" ||
      fail "could not make superseded $label generation removable: $path"
    rm -rf -- "$path"
    [[ ! -e "$path" && ! -L "$path" ]] ||
      fail "could not remove superseded $label generation: $path"
    printf 'Removed superseded %s generation: %s (%s KiB)\n' "$label" "$path" "$size_kib"
  done

  printf '%s retention complete: retained=%s removable=%s reclaimable_kib=%s dry_run=%s\n' \
    "$label" "$retained" "$removed" "$reclaimed_kib" "$DRY_RUN"
}

prune_content_addressed_stages() {
  local label="$1"
  local stages_dir="$2"
  local generations_dir="$3"
  local receipt_key="$4"
  local retain_unreferenced="$5"
  local receipt line_key line_value entry name path size_kib
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

  if [[ -d "$generations_dir" ]]; then
    while IFS= read -r -d '' receipt; do
      [[ -f "$receipt" && ! -L "$receipt" ]] ||
        fail "unsafe retained $label generation receipt: $receipt"
      while IFS='=' read -r line_key line_value; do
        [[ "$line_key" == "$receipt_key" ]] || continue
        [[ "$line_value" =~ ^[0-9a-f]{64}$ ]] ||
          fail "invalid $receipt_key in retained generation receipt: $receipt"
        referenced["$line_value"]=1
      done <"$receipt"
    done < <(
      find "$generations_dir" -mindepth 3 -maxdepth 3 -path '*/metadata/build-receipt' -type f -print0
    )
  fi

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
    if [[ "$DRY_RUN" == 1 ]]; then
      printf 'Would remove unreferenced %s stage: %s (%s KiB)\n' "$label" "$path" "$size_kib"
      continue
    fi
    chmod -R u+w -- "$path" ||
      fail "could not make unreferenced $label stage removable: $path"
    rm -rf -- "$path"
    [[ ! -e "$path" && ! -L "$path" ]] ||
      fail "could not remove unreferenced $label stage: $path"
    printf 'Removed unreferenced %s stage: %s (%s KiB)\n' "$label" "$path" "$size_kib"
  done

  printf '%s stage retention complete: referenced=%s fallback=%s removable=%s reclaimable_kib=%s dry_run=%s\n' \
    "$label" "$retained_referenced" "$retained_unreferenced" "$removed" "$reclaimed_kib" "$DRY_RUN"
}

prune_disposable_state

prune_family \
  CPU \
  "$BUILD_ROOT/cpu-sdk/current" \
  "$BUILD_ROOT/cpu-sdk/.android-cpu-importer-generations" \
  '^[0-9a-f]{64}-[0-9a-f]{64}$'
prune_family \
  AOT \
  "$BUILD_ROOT/aot-sdk/current" \
  "$BUILD_ROOT/aot-sdk/.android-aot-generations" \
  '^[0-9a-f]{64}-[0-9a-f]{16}$'

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
