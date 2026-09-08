#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRUNE_SCRIPT="$(realpath -e -- "$SCRIPT_DIR/../../main/android/prune-android-sdx-build-cache.sh")"
TEST_ROOT="$(mktemp -d)"
trap 'chmod -R u+w -- "$TEST_ROOT"; rm -rf -- "$TEST_ROOT"' EXIT
unset SDX_ANDROID_PIPELINE_LOCK_HELD SDX_ANDROID_PIPELINE_LOCK_FD
unset SDX_ANDROID_GENERATION_RETENTION SDX_ANDROID_MANAGED_STAGE_RETENTION SDX_ANDROID_OBJECT_STAGE_RETENTION
# Exercise explicit rollback retention first, then the shared minimal defaults.
ROLLBACK_ARGS=(--retain-generations 2 --retain-managed-stages 1)

fail() {
  printf 'prune-android-sdx-build-cache-test: %s\n' "$*" >&2
  exit 1
}

put_marker() {
  local path="$1"
  mkdir -p -- "$(dirname -- "$path")"
  printf 'keep\n' >"$path"
}

put_receipt() {
  local path="$1"
  local field="$2"
  local value="$3"
  mkdir -p -- "$(dirname -- "$path")"
  printf '%s=%s\n' "$field" "$value" >"$path"
}

cpu_current="$(printf 'a%.0s' {1..64})-$(printf 'b%.0s' {1..64})"
cpu_rollback="$(printf 'c%.0s' {1..64})-$(printf 'd%.0s' {1..64})"
cpu_old="$(printf 'e%.0s' {1..64})-$(printf 'f%.0s' {1..64})"
aot_current="$(printf '1%.0s' {1..64})-$(printf '2%.0s' {1..16})"
aot_rollback="$(printf '3%.0s' {1..64})-$(printf '4%.0s' {1..16})"
aot_old="$(printf '5%.0s' {1..64})-$(printf '6%.0s' {1..16})"
cpu_current_stage="$(printf '0%.0s' {1..64})"
cpu_rollback_stage="$(printf '1%.0s' {1..64})"
cpu_fallback_stage="$(printf '2%.0s' {1..64})"
cpu_orphan_stage="$(printf '3%.0s' {1..64})"
aot_current_stage="$(printf '4%.0s' {1..64})"
aot_rollback_stage="$(printf '5%.0s' {1..64})"
aot_fallback_stage="$(printf '6%.0s' {1..64})"
aot_orphan_stage="$(printf '7%.0s' {1..64})"

cpu_generations="$TEST_ROOT/cpu-sdk/.android-cpu-importer-generations"
aot_generations="$TEST_ROOT/aot-sdk/.android-aot-generations"
mkdir -p --   "$cpu_generations/$cpu_current"   "$cpu_generations/$cpu_rollback"   "$cpu_generations/$cpu_old"   "$aot_generations/$aot_current"   "$aot_generations/$aot_rollback"   "$aot_generations/$aot_old"
put_marker "$cpu_generations/$cpu_current/current"
put_marker "$cpu_generations/$cpu_rollback/rollback"
put_marker "$cpu_generations/$cpu_old/old"
put_marker "$aot_generations/$aot_current/current"
put_marker "$aot_generations/$aot_rollback/rollback"
put_marker "$aot_generations/$aot_old/old"
put_receipt "$cpu_generations/$cpu_current/metadata/build-receipt" managed_stage_key "$cpu_current_stage"
put_receipt "$cpu_generations/$cpu_rollback/metadata/build-receipt" managed_stage_key "$cpu_rollback_stage"
put_receipt "$cpu_generations/$cpu_old/metadata/build-receipt" managed_stage_key "$cpu_orphan_stage"
put_receipt "$aot_generations/$aot_current/metadata/build-receipt" object_stage_inputs_sha256 "$aot_current_stage"
put_receipt "$aot_generations/$aot_rollback/metadata/build-receipt" object_stage_inputs_sha256 "$aot_rollback_stage"
put_receipt "$aot_generations/$aot_old/metadata/build-receipt" object_stage_inputs_sha256 "$aot_orphan_stage"
printf 'base_sdk=%s\n' "$cpu_generations/$cpu_current" >>"$aot_generations/$aot_current/metadata/build-receipt"
printf 'base_sdk=%s\n' "$cpu_generations/$cpu_rollback" >>"$aot_generations/$aot_rollback/metadata/build-receipt"
printf 'base_sdk=%s\n' "$cpu_generations/$cpu_old" >>"$aot_generations/$aot_old/metadata/build-receipt"
ln -s ".android-cpu-importer-generations/$cpu_current" "$TEST_ROOT/cpu-sdk/current"
ln -s ".android-aot-generations/$aot_current" "$TEST_ROOT/aot-sdk/current"

touch -t 202001010000 "$cpu_generations/$cpu_current" "$aot_generations/$aot_current"
touch -t 202301010000 "$cpu_generations/$cpu_rollback" "$aot_generations/$aot_rollback"
touch -t 202201010000 "$cpu_generations/$cpu_old" "$aot_generations/$aot_old"

put_marker "$TEST_ROOT/cpu-sdk/work/generation.interrupted/remove"
put_marker "$TEST_ROOT/cpu-sdk/work/managed-stage.interrupted/remove"
put_marker "$TEST_ROOT/cpu-sdk/work/published-native-manifest.interrupted"
put_marker "$TEST_ROOT/aot-sdk/work/generation.interrupted/remove"
put_marker "$TEST_ROOT/aot-sdk/work/staging-copy-test.interrupted/remove"
put_marker "$TEST_ROOT/aot-sdk/work/staging-regression.interrupted/remove"
put_marker "$TEST_ROOT/aot-sdk/work/native-image-object-stages.invalid/evidence/keep"
put_marker "$TEST_ROOT/aot-sdk/work/native-image-object-stages/.native-image-object.interrupted/remove"
put_marker "$TEST_ROOT/accelerator/tensor-g3/quarantined-maven-targets.interrupted/remove"
put_marker "$TEST_ROOT/accelerator/tensor-g3/dist/fresh-java-builds.tmp.interrupted"

put_marker "$TEST_ROOT/cpu-sdk/work/native-builds/cache/keep"
put_marker "$TEST_ROOT/cpu-sdk/work/managed-stages/$cpu_current_stage/keep"
put_marker "$TEST_ROOT/cpu-sdk/work/managed-stages/$cpu_rollback_stage/keep"
put_marker "$TEST_ROOT/cpu-sdk/work/managed-stages/$cpu_fallback_stage/keep"
put_marker "$TEST_ROOT/cpu-sdk/work/managed-stages/$cpu_orphan_stage/remove"
put_marker "$TEST_ROOT/aot-sdk/work/native-image-object-stages/$aot_current_stage/keep"
put_marker "$TEST_ROOT/aot-sdk/work/native-image-object-stages/$aot_rollback_stage/keep"
put_marker "$TEST_ROOT/aot-sdk/work/native-image-object-stages/$aot_fallback_stage/keep"
put_marker "$TEST_ROOT/aot-sdk/work/native-image-object-stages/$aot_orphan_stage/remove"
touch -t 202401010000 \
  "$TEST_ROOT/cpu-sdk/work/managed-stages/$cpu_fallback_stage" \
  "$TEST_ROOT/aot-sdk/work/native-image-object-stages/$aot_fallback_stage"
touch -t 202101010000 \
  "$TEST_ROOT/cpu-sdk/work/managed-stages/$cpu_orphan_stage" \
  "$TEST_ROOT/aot-sdk/work/native-image-object-stages/$aot_orphan_stage"
put_marker "$TEST_ROOT/accelerator/tensor-g3/native/keep"
put_marker "$TEST_ROOT/accelerator/tensor-g3/dist/current.aar"
put_marker "$TEST_ROOT/ccache/keep"
put_marker "$TEST_ROOT/native-images-cache/keep"
put_marker "$TEST_ROOT/apk-output/current.apk"

# A standalone prune refuses an active pipeline before planning/removing work.
mkdir -p "$TEST_ROOT/.locks"
exec {pipeline_fd}>"$TEST_ROOT/.locks/tensor-g3-offline-apk.lock"
flock "$pipeline_fd"
if "$PRUNE_SCRIPT" --build-root "$TEST_ROOT" >/dev/null 2>&1; then
  fail "standalone cleanup ran while pipeline lock was held"
fi
[[ -f "$TEST_ROOT/cpu-sdk/work/generation.interrupted/remove" ]] || fail "lock rejection deleted work"
# Nested cleanup can reuse the actual inherited descriptor, but not a bare flag.
SDX_ANDROID_PIPELINE_LOCK_HELD=1 SDX_ANDROID_PIPELINE_LOCK_FD="$pipeline_fd" \
  "$PRUNE_SCRIPT" --build-root "$TEST_ROOT" "${ROLLBACK_ARGS[@]}" --dry-run >"$TEST_ROOT/inherited.log"
if SDX_ANDROID_PIPELINE_LOCK_HELD=1 "$PRUNE_SCRIPT" --build-root "$TEST_ROOT" >/dev/null 2>&1; then
  fail "cleanup trusted an unverified lock flag"
fi
exec {pipeline_fd}>&-
"$PRUNE_SCRIPT" --build-root "$TEST_ROOT" "${ROLLBACK_ARGS[@]}" --dry-run >"$TEST_ROOT/dry.log"
grep -Fq "Would remove unreferenced CPU managed stage: $TEST_ROOT/cpu-sdk/work/managed-stages/$cpu_orphan_stage" "$TEST_ROOT/dry.log" ||
  fail "dry-run retained references from a superseded CPU generation"
grep -Fq "Would remove unreferenced Native Image object stage: $TEST_ROOT/aot-sdk/work/native-image-object-stages/$aot_orphan_stage" "$TEST_ROOT/dry.log" ||
  fail "dry-run retained references from a superseded AOT generation"

[[ -d "$TEST_ROOT/cpu-sdk/work/generation.interrupted" ]] ||
  fail "dry-run removed CPU publication work"
[[ -d "$cpu_generations/$cpu_old" ]] ||
  fail "dry-run removed a CPU generation"
[[ -d "$aot_generations/$aot_old" ]] ||
  fail "dry-run removed an AOT generation"
[[ -d "$TEST_ROOT/cpu-sdk/work/managed-stages/$cpu_orphan_stage" ]] ||
  fail "dry-run removed a CPU managed stage"
[[ -d "$TEST_ROOT/aot-sdk/work/native-image-object-stages/$aot_orphan_stage" ]] ||
  fail "dry-run removed a Native Image object stage"

"$PRUNE_SCRIPT" --build-root "$TEST_ROOT" "${ROLLBACK_ARGS[@]}" >"$TEST_ROOT/applied.log"
diff -u <(grep '^Would remove ' "$TEST_ROOT/dry.log" | sed 's/^Would remove /Removed /') \
  <(grep '^Removed ' "$TEST_ROOT/applied.log") || fail "dry-run and application plans differed"
for leftover in staging-copy-test.interrupted staging-regression.interrupted; do
  [[ ! -e "$TEST_ROOT/aot-sdk/work/$leftover" ]] || fail "leftover retained: $leftover"
done
[[ -f "$TEST_ROOT/aot-sdk/work/native-image-object-stages.invalid/evidence/keep" ]] || fail "corruption evidence deleted"

for removed in   "$TEST_ROOT/cpu-sdk/work/generation.interrupted"   "$TEST_ROOT/cpu-sdk/work/managed-stage.interrupted"   "$TEST_ROOT/cpu-sdk/work/published-native-manifest.interrupted"   "$TEST_ROOT/aot-sdk/work/generation.interrupted"   "$TEST_ROOT/aot-sdk/work/native-image-object-stages/.native-image-object.interrupted"   "$TEST_ROOT/accelerator/tensor-g3/quarantined-maven-targets.interrupted"   "$TEST_ROOT/accelerator/tensor-g3/dist/fresh-java-builds.tmp.interrupted"   "$cpu_generations/$cpu_old"   "$aot_generations/$aot_old"   "$TEST_ROOT/cpu-sdk/work/managed-stages/$cpu_orphan_stage"   "$TEST_ROOT/aot-sdk/work/native-image-object-stages/$aot_orphan_stage"; do
  [[ ! -e "$removed" && ! -L "$removed" ]] ||
    fail "expected cleanup to remove $removed"
done

for preserved in   "$cpu_generations/$cpu_current/current"   "$cpu_generations/$cpu_rollback/rollback"   "$aot_generations/$aot_current/current"   "$aot_generations/$aot_rollback/rollback"   "$TEST_ROOT/cpu-sdk/work/native-builds/cache/keep"   "$TEST_ROOT/cpu-sdk/work/managed-stages/$cpu_current_stage/keep"   "$TEST_ROOT/cpu-sdk/work/managed-stages/$cpu_rollback_stage/keep"   "$TEST_ROOT/cpu-sdk/work/managed-stages/$cpu_fallback_stage/keep"   "$TEST_ROOT/aot-sdk/work/native-image-object-stages/$aot_current_stage/keep"   "$TEST_ROOT/aot-sdk/work/native-image-object-stages/$aot_rollback_stage/keep"   "$TEST_ROOT/aot-sdk/work/native-image-object-stages/$aot_fallback_stage/keep"   "$TEST_ROOT/accelerator/tensor-g3/native/keep"   "$TEST_ROOT/accelerator/tensor-g3/dist/current.aar"   "$TEST_ROOT/ccache/keep"   "$TEST_ROOT/native-images-cache/keep"   "$TEST_ROOT/apk-output/current.apk"; do
  [[ -e "$preserved" ]] || fail "cleanup removed preserved state: $preserved"
done

[[ "$(realpath -e -- "$TEST_ROOT/cpu-sdk/current")" == "$cpu_generations/$cpu_current" ]] ||
  fail "CPU current symlink changed"
[[ "$(realpath -e -- "$TEST_ROOT/aot-sdk/current")" == "$aot_generations/$aot_current" ]] ||
  fail "AOT current symlink changed"

# A completed object whose subsequent SDK packaging failed has no generation
# reference yet. The APK wrapper's bounded policy must retain it on the next run.
fallback="$TEST_ROOT/aot-sdk/work/native-image-object-stages/$aot_fallback_stage"
put_marker "$fallback/libsdx_llm.o"
put_receipt "$fallback/build-receipt" object_stage_inputs_sha256 "$aot_fallback_stage"
for attempt in 1 2; do
  "$PRUNE_SCRIPT" --build-root "$TEST_ROOT" >/dev/null
  [[ -f "$fallback/libsdx_llm.o" && -f "$fallback/build-receipt" ]] ||
    fail "retry $attempt pruned a completed AOT object after SDK packaging failure"
  [[ -f "$TEST_ROOT/aot-sdk/work/native-image-object-stages/$aot_current_stage/keep" ]] ||
    fail "retry $attempt pruned the active generation's object"
done
[[ ! -d "$aot_generations/$aot_rollback" ]] || fail "obsolete rollback generation retained"
[[ ! -d "$TEST_ROOT/aot-sdk/work/native-image-object-stages/$aot_rollback_stage" ]] ||
  fail "more than one unreferenced AOT fallback retained"

# Pin an explicitly selected base even before any AOT publication refers to it.
put_marker "$cpu_generations/$cpu_old/selected"
put_receipt "$cpu_generations/$cpu_old/metadata/build-receipt" managed_stage_key "$cpu_orphan_stage"
put_marker "$TEST_ROOT/cpu-sdk/work/managed-stages/$cpu_orphan_stage/selected"
"$PRUNE_SCRIPT" --build-root "$TEST_ROOT" --keep-cpu-sdk "$cpu_generations/$cpu_old" >/dev/null
[[ -f "$cpu_generations/$cpu_old/selected" &&
   -f "$TEST_ROOT/cpu-sdk/work/managed-stages/$cpu_orphan_stage/selected" ]] || fail "explicit CPU base pruned"

# Pin an older CPU generation through the retained AOT receipt, including its stage.
put_marker "$cpu_generations/$cpu_rollback/rollback"
put_receipt "$cpu_generations/$cpu_rollback/metadata/build-receipt" managed_stage_key "$cpu_rollback_stage"
put_marker "$TEST_ROOT/cpu-sdk/work/managed-stages/$cpu_rollback_stage/keep"
put_receipt "$aot_generations/$aot_current/metadata/build-receipt" object_stage_inputs_sha256 "$aot_current_stage"
printf 'base_sdk=%s\n' "$cpu_generations/$cpu_rollback" >>"$aot_generations/$aot_current/metadata/build-receipt"
"$PRUNE_SCRIPT" --build-root "$TEST_ROOT" >/dev/null
[[ -f "$cpu_generations/$cpu_rollback/rollback" &&
   -f "$TEST_ROOT/cpu-sdk/work/managed-stages/$cpu_rollback_stage/keep" ]] || fail "retained AOT lost its CPU base"

# Retained receipts must fail closed: neither symlinks nor absent fields may
# hide reference edges and turn live stages into apparent orphans.
put_marker "$TEST_ROOT/cpu-sdk/work/generation.interrupted/remove"
receipt="$aot_generations/$aot_current/metadata/build-receipt"
cp "$receipt" "$TEST_ROOT/receipt.backup"
for defect in missing symlink metadata-symlink missing-key duplicate-key; do
  case "$defect" in
    missing) rm "$receipt" ;;
    symlink) rm "$receipt"; ln -s "$TEST_ROOT/receipt.backup" "$receipt" ;;
    metadata-symlink)
      mv "${receipt%/build-receipt}" "$TEST_ROOT/metadata.backup"
      ln -s "$TEST_ROOT/metadata.backup" "${receipt%/build-receipt}" ;;
    missing-key) put_receipt "$receipt" base_sdk "$cpu_generations/$cpu_rollback" ;;
    duplicate-key) printf 'base_sdk=%s\n' "$cpu_generations/$cpu_rollback" >>"$receipt" ;;
  esac
  if "$PRUNE_SCRIPT" --build-root "$TEST_ROOT" >/dev/null 2>&1; then
    fail "cleanup accepted retained receipt defect: $defect"
  fi
  [[ -f "$TEST_ROOT/cpu-sdk/work/generation.interrupted/remove" ]] || fail "receipt failure partially applied cleanup"
  if [[ "$defect" == metadata-symlink ]]; then
    rm "${receipt%/build-receipt}"
    mv "$TEST_ROOT/metadata.backup" "${receipt%/build-receipt}"
  fi
  rm -f "$receipt"
  cp "$TEST_ROOT/receipt.backup" "$receipt"
done

# Later validation failures must not partially apply an earlier removal plan.
put_marker "$TEST_ROOT/cpu-sdk/work/generation.interrupted/remove"
exec {held_provider_lock_fd}>"$TEST_ROOT/accelerator/tensor-g3/.build.lock"
flock "$held_provider_lock_fd"
if "$PRUNE_SCRIPT" --build-root "$TEST_ROOT" >/dev/null 2>&1; then
  fail "cleanup ran while an accelerator provider lock was held"
fi
[[ -f "$TEST_ROOT/cpu-sdk/work/generation.interrupted/remove" ]] || fail "provider lock failure partially applied cleanup"
exec {held_provider_lock_fd}>&-

mkdir -p -- "$TEST_ROOT/outside-provider"
ln -s "$TEST_ROOT/outside-provider" "$TEST_ROOT/accelerator/symlink-provider"
if "$PRUNE_SCRIPT" --build-root "$TEST_ROOT" >/dev/null 2>&1; then
  fail "cleanup accepted a symlink accelerator provider"
fi
rm -- "$TEST_ROOT/accelerator/symlink-provider"

mv "$TEST_ROOT/cpu-sdk/work/managed-stages" "$TEST_ROOT/cpu-sdk/work/managed-stages.real"
ln -s managed-stages.real "$TEST_ROOT/cpu-sdk/work/managed-stages"
if "$PRUNE_SCRIPT" --build-root "$TEST_ROOT" >/dev/null 2>&1; then
  fail "cleanup accepted a symlink managed-stage root"
fi

[[ -f "$TEST_ROOT/cpu-sdk/work/generation.interrupted/remove" ]] || fail "unsafe stage root partially applied cleanup"
rm "$TEST_ROOT/cpu-sdk/work/managed-stages"
mv "$TEST_ROOT/cpu-sdk/work/managed-stages.real" "$TEST_ROOT/cpu-sdk/work/managed-stages"
ln -s "$TEST_ROOT/ccache/keep" "$TEST_ROOT/cpu-sdk/work/generation.interrupted/outside"
if "$PRUNE_SCRIPT" --build-root "$TEST_ROOT" >/dev/null 2>&1; then
  fail "cleanup accepted a disposable directory containing a symlink"
fi
[[ -f "$TEST_ROOT/ccache/keep" ]] || fail "symlink target deleted"
printf 'prune-android-sdx-build-cache-test: PASS\n'
