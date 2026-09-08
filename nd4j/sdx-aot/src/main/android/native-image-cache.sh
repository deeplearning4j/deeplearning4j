#!/usr/bin/env bash

# Shared checksum-verified artifact cache for GraalVM Native Image outputs.
#
# Call sdx_native_cache_configure once, then use restore/publish with a target,
# SHA-256 fingerprint, stable artifact name, and normal target path. The caller
# owns the fingerprint and any target-specific format/ABI validation.

SDX_NATIVE_CACHE_SCHEMA=sdx-native-artifact-cache-v1

sdx_native_cache_configure() {
  SDX_NATIVE_CACHE="${SDX_NATIVE_CACHE:-1}"
  # Number of verified generations per existing target bucket, including current.
  SDX_NATIVE_CACHE_RETENTION="${SDX_NATIVE_CACHE_RETENTION:-2}"
  [[ "$SDX_NATIVE_CACHE_RETENTION" =~ ^[1-9][0-9]{0,8}$ ]] || {
    printf 'SDX_NATIVE_CACHE_RETENTION must be a positive integer (at most 9 digits)\n' >&2
    return 1
  }
  SDX_NATIVE_FORCE_REBUILD="${SDX_NATIVE_FORCE_REBUILD:-0}"
  SDX_NATIVE_CACHE_DIR="${SDX_NATIVE_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME:?HOME is required}/.cache}/sdx/native-images}"

  case "$SDX_NATIVE_CACHE" in
    0|1) ;;
    *) printf 'SDX_NATIVE_CACHE must be 0 or 1 (got %s)\n' "$SDX_NATIVE_CACHE" >&2; return 1 ;;
  esac
  case "$SDX_NATIVE_FORCE_REBUILD" in
    0|1) ;;
    *) printf 'SDX_NATIVE_FORCE_REBUILD must be 0 or 1 (got %s)\n' "$SDX_NATIVE_FORCE_REBUILD" >&2; return 1 ;;
  esac
}

sdx_native_cache_sha256_file() {
  sha256sum -- "$1" | cut -d ' ' -f 1
}

sdx_native_cache_artifact_path() {
  local target="${1:?cache target is required}"
  local fingerprint="${2:?cache fingerprint is required}"
  local artifact_name="${3:?cache artifact name is required}"
  [[ "$target" =~ ^[A-Za-z0-9._-]+$ ]] || {
    printf 'Unsafe native cache target: %s\n' "$target" >&2
    return 1
  }
  [[ "$fingerprint" =~ ^[0-9a-f]{64}$ ]] || {
    printf 'Unsafe native cache fingerprint: %s\n' "$fingerprint" >&2
    return 1
  }
  [[ "$artifact_name" =~ ^[A-Za-z0-9._+-]+$ ]] || {
    printf 'Unsafe native cache artifact name: %s\n' "$artifact_name" >&2
    return 1
  }
  [[ "$target" != . && "$target" != .. && "$artifact_name" != . &&
     "$artifact_name" != .. && "$artifact_name" != .last-used ]] || return 1
  printf '%s/%s/%s/%s' "$SDX_NATIVE_CACHE_DIR" "$target" "$fingerprint" "$artifact_name"
}

sdx_native_cache_validate_artifact() {
  local artifact="${1:?cached artifact is required}"
  local checksum_file="$artifact.sha256"
  local recorded_checksum actual_checksum
  [[ -f "$artifact" && ! -L "$artifact" && -s "$artifact" ]] || return 1
  [[ -f "$checksum_file" && ! -L "$checksum_file" && -s "$checksum_file" ]] || return 1
  [[ "$(wc -l <"$checksum_file")" -eq 1 ]] || return 1
  IFS= read -r recorded_checksum <"$checksum_file" || return 1
  [[ "$recorded_checksum" =~ ^[0-9a-f]{64}$ ]] || return 1
  actual_checksum="$(sdx_native_cache_sha256_file "$artifact")" || return 1
  [[ "$actual_checksum" == "$recorded_checksum" ]]
}

sdx_native_cache_write_sidecar() {
  local target_path="${1:?target path is required}"
  local fingerprint="${2:?cache fingerprint is required}"
  local checksum="${3:?artifact checksum is required}"
  local sidecar="$target_path.native-cache"
  local temporary_sidecar
  temporary_sidecar="$(mktemp "$(dirname -- "$sidecar")/.$(basename -- "$sidecar").XXXXXXXX")" || return 1
  printf '%s %s\n' "$fingerprint" "$checksum" >"$temporary_sidecar" || return 1
  mv -fT -- "$temporary_sidecar" "$sidecar"
}

# Reject symlinks in the whole path, including ancestors of a configured root.
sdx_native_cache_safe_path() {
  local path="$1"
  [[ "/$path/" != */../* ]] || return 1
  while [[ -n "$path" && "$path" != / && "$path" != . ]]; do
    [[ ! -L "$path" ]] || return 1
    path="$(dirname -- "$path")"
  done
}

sdx_native_cache_restore() {
  [[ "$SDX_NATIVE_CACHE" == 1 && "$SDX_NATIVE_FORCE_REBUILD" != 1 ]] || return 1
  sdx_native_cache_locked restore "$@"
}

sdx_native_cache_publish() {
  [[ "$SDX_NATIVE_CACHE" == 1 ]] || return 0
  sdx_native_cache_locked publish "$@"
}

# Subshell owns the descriptor. Internals never reacquire the lock.
sdx_native_cache_locked() (
  local operation="$1"; shift
  local artifact bucket lock_fd skipped=1
  [[ "$operation" != publish ]] || skipped=0
  command -v flock >/dev/null 2>&1 || return "$skipped"
  artifact="$(sdx_native_cache_artifact_path "$1" "$2" "$3")" || return 1
  bucket="$SDX_NATIVE_CACHE_DIR/$1"
  sdx_native_cache_safe_path "$artifact" || return 1
  mkdir -p -- "$bucket" || return "$skipped"
  [[ ! -L "$bucket/.lock" && ( ! -e "$bucket/.lock" || -f "$bucket/.lock" ) ]] || return 1
  exec {lock_fd}>>"$bucket/.lock" || return "$skipped"
  flock -n "$lock_fd" || return "$skipped"
  sdx_native_cache_safe_path "$artifact" || return 1
  sdx_native_cache_${operation}_unlocked "$@" || return $?
  # Local-target hits may have no shared entry yet.
  if sdx_native_cache_validate_artifact "$artifact"; then
    [[ ! -L "${artifact%/*}/.last-used" &&
       ( ! -e "${artifact%/*}/.last-used" || -f "${artifact%/*}/.last-used" ) ]] || return 1
    if [[ -e "${artifact%/*}/.last-used" ]]; then
      [[ -O "${artifact%/*}/.last-used" &&
         "$(stat -c %h -- "${artifact%/*}/.last-used")" == 1 ]] || return 1
    fi
    touch -- "${artifact%/*}/.last-used" || return 1
    sdx_native_cache_prune_unlocked "$bucket" "$2" "$3" ||
      printf 'WARNING: native cache retention skipped for unsafe/incomplete bucket %s\n' "$bucket" >&2
  fi
)

# Validate the complete plan first. Unknown/corrupt entries fail closed; never
# recursively purge a directory. Legacy entries use the artifact's mtime.
sdx_native_cache_prune_unlocked() (
  local bucket="$1" protected="$2" name="$3" entry child marker oldest oldest_marker i excess
  local -a entries=() victims=()
  shopt -s nullglob dotglob
  for entry in "$bucket"/*; do
    [[ "${entry##*/}" =~ ^[0-9a-f]{64}$ ]] || continue
    [[ -d "$entry" && ! -L "$entry" && -O "$entry" ]] || return 1
    for child in "$entry"/*; do
      [[ -f "$child" && ! -L "$child" && -O "$child" ]] || return 1
      case "${child##*/}" in
        "$name"|"$name.sha256"|.last-used) ;;
        *) return 1 ;;
      esac
    done
    sdx_native_cache_validate_artifact "$entry/$name" || return 1
    entries+=("$entry")
  done
  excess=$((${#entries[@]} - ${SDX_NATIVE_CACHE_RETENTION:-2}))
  while (( excess > 0 )); do
    oldest=-1
    for i in "${!entries[@]}"; do
      entry="${entries[i]}"
      [[ "${entry##*/}" != "$protected" ]] || continue
      marker="$entry/.last-used"
      [[ -e "$marker" ]] || marker="$entry/$name"
      if (( oldest < 0 )) || [[ "$marker" -ot "$oldest_marker" ]]; then
        oldest="$i"
        oldest_marker="$marker"
      fi
    done
    (( oldest >= 0 )) || return 1
    victims+=("${entries[oldest]}")
    unset 'entries[oldest]'
    excess=$((excess - 1))
  done
  for entry in "${victims[@]}"; do
    rm -f -- "$entry/$name" "$entry/$name.sha256" "$entry/.last-used" || return 1
    rmdir -- "$entry" || return 1
  done
)

sdx_native_cache_restore_unlocked() {
  local target="${1:?cache target is required}"
  local fingerprint="${2:?cache fingerprint is required}"
  local artifact_name="${3:?cache artifact name is required}"
  local target_path="${4:?target path is required}"
  local target_sidecar="$target_path.native-cache"
  local cached_artifact temporary_artifact checksum sidecar_fingerprint sidecar_checksum
  [[ "$SDX_NATIVE_CACHE" == 1 && "$SDX_NATIVE_FORCE_REBUILD" != 1 ]] || return 1

  # The normal target is the cheapest cache tier. Accept it only when both
  # sidecar fields match this invocation and the current bytes verify.
  if [[ -f "$target_path" && ! -L "$target_path" && -s "$target_path" &&
        -f "$target_sidecar" && ! -L "$target_sidecar" && -s "$target_sidecar" &&
        "$(wc -l <"$target_sidecar")" -eq 1 ]]; then
    read -r sidecar_fingerprint sidecar_checksum <"$target_sidecar" || return 1
    if [[ "$sidecar_fingerprint" == "$fingerprint" &&
          "$sidecar_checksum" =~ ^[0-9a-f]{64}$ &&
          "$(sdx_native_cache_sha256_file "$target_path")" == "$sidecar_checksum" ]]; then
      printf 'CACHE HIT: reusing verified native target %s at %s\n' "$target" "$target_path"
      return 0
    fi
  fi

  cached_artifact="$(sdx_native_cache_artifact_path "$target" "$fingerprint" "$artifact_name")" || return 1
  sdx_native_cache_validate_artifact "$cached_artifact" || return 1
  checksum="$(sdx_native_cache_sha256_file "$cached_artifact")" || return 1

  mkdir -p -- "$(dirname -- "$target_path")" || return 1
  temporary_artifact="$(mktemp "$(dirname -- "$target_path")/.$(basename -- "$target_path").native-cache.XXXXXXXX")" || return 1
  if ! cp -p --reflink=auto -- "$cached_artifact" "$temporary_artifact"; then
    rm -f -- "$temporary_artifact"
    return 1
  fi
  if [[ "$(stat -c '%d:%i' "$cached_artifact")" == "$(stat -c '%d:%i' "$temporary_artifact")" ]] ||
     [[ "$(sdx_native_cache_sha256_file "$temporary_artifact")" != "$checksum" ]]; then
    rm -f -- "$temporary_artifact"
    return 1
  fi
  mv -fT -- "$temporary_artifact" "$target_path" || return 1
  sdx_native_cache_write_sidecar "$target_path" "$fingerprint" "$checksum" || return 1
  printf 'CACHE HIT: restored native artifact %s from %s\n' "$target" "$(dirname -- "$cached_artifact")"
}

sdx_native_cache_publish_unlocked() {
  local target="${1:?cache target is required}"
  local fingerprint="${2:?cache fingerprint is required}"
  local artifact_name="${3:?cache artifact name is required}"
  local source_path="${4:?source path is required}"
  local cached_artifact cache_dir checksum cached_checksum temporary_artifact temporary_checksum
  [[ "$SDX_NATIVE_CACHE" == 1 ]] || return 0
  [[ -f "$source_path" && ! -L "$source_path" && -s "$source_path" ]] || return 1

  cached_artifact="$(sdx_native_cache_artifact_path "$target" "$fingerprint" "$artifact_name")" || return 1
  cache_dir="$(dirname -- "$cached_artifact")"
  checksum="$(sdx_native_cache_sha256_file "$source_path")" || return 1
  if sdx_native_cache_validate_artifact "$cached_artifact"; then
    cached_checksum="$(sdx_native_cache_sha256_file "$cached_artifact")" || return 1
    if [[ "$cached_checksum" == "$checksum" ]]; then
      sdx_native_cache_write_sidecar "$source_path" "$fingerprint" "$checksum"
      return
    fi
  fi

  mkdir -p -- "$cache_dir" || return 1
  [[ -d "$cache_dir" && ! -L "$cache_dir" ]] || return 1
  temporary_artifact="$(mktemp "$cache_dir/.$artifact_name.XXXXXXXX")" || return 1
  temporary_checksum="$(mktemp "$cache_dir/.$artifact_name.sha256.XXXXXXXX")" || {
    rm -f -- "$temporary_artifact"
    return 1
  }
  if ! cp -p --reflink=auto -- "$source_path" "$temporary_artifact" ||
     [[ "$(stat -c '%d:%i' "$source_path")" == "$(stat -c '%d:%i' "$temporary_artifact")" ]] ||
     [[ "$(sdx_native_cache_sha256_file "$temporary_artifact")" != "$checksum" ]]; then
    rm -f -- "$temporary_artifact" "$temporary_checksum"
    return 1
  fi
  chmod a-w -- "$temporary_artifact" || return 1
  printf '%s\n' "$checksum" >"$temporary_checksum" || return 1
  mv -fT -- "$temporary_artifact" "$cached_artifact" || return 1
  mv -fT -- "$temporary_checksum" "$cached_artifact.sha256" || return 1
  sdx_native_cache_write_sidecar "$source_path" "$fingerprint" "$checksum" || return 1
  printf 'Cached native artifact %s: %s\n' "$target" "$cache_dir"
}
