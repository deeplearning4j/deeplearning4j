#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../main/android/native-image-cache.sh"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf -- "$WORK_DIR"' EXIT
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }
command -v flock >/dev/null || fail 'fixture requires flock'

# Never inherit a real cache root or cache controls from the caller.
SDX_NATIVE_CACHE_DIR="$WORK_DIR/cache"
SDX_NATIVE_CACHE=1
SDX_NATIVE_FORCE_REBUILD=0
unset SDX_NATIVE_CACHE_RETENTION
sdx_native_cache_configure
export SDX_NATIVE_CACHE_DIR SDX_NATIVE_CACHE SDX_NATIVE_FORCE_REBUILD SDX_NATIVE_CACHE_RETENTION
[[ "$SDX_NATIVE_CACHE_RETENTION" == 2 ]] || fail 'default retention'
for invalid in 0 -1 abc 01 9999999999; do
  if (SDX_NATIVE_CACHE_RETENTION="$invalid"; sdx_native_cache_configure); then
    fail "accepted invalid retention $invalid"
  fi
done
printf 'native fixture bytes\n' >"$WORK_DIR/source"
a="$(printf 'a%.0s' {1..64})"
b="$(printf 'b%.0s' {1..64})"
c="$(printf 'c%.0s' {1..64})"
d="$(printf 'd%.0s' {1..64})"
artifact=libfixture.o
publish() { sdx_native_cache_publish "$1" "$2" "$artifact" "$WORK_DIR/source"; }
restore() { sdx_native_cache_restore "$1" "$2" "$artifact" "$WORK_DIR/restored-$1"; }
age() { touch -t "$3" "$SDX_NATIVE_CACHE_DIR/$1/$2/.last-used"; }

publish bounded "$a"
age bounded "$a" 202001010000
publish bounded "$b"
age bounded "$b" 202101010000
publish bounded "$c"
[[ ! -e "$SDX_NATIVE_CACHE_DIR/bounded/$a" && -d "$SDX_NATIVE_CACHE_DIR/bounded/$b" &&
   -d "$SDX_NATIVE_CACHE_DIR/bounded/$c" ]] || fail 'publish three retain two'

publish lru "$a"
publish lru "$b"
age lru "$a" 202001010000
age lru "$b" 202101010000
restore lru "$a"
[[ "$SDX_NATIVE_CACHE_DIR/lru/$a/.last-used" -nt "$SDX_NATIVE_CACHE_DIR/lru/$b/.last-used" ]] || fail 'shared hit not touched'
publish lru "$c"
[[ -d "$SDX_NATIVE_CACHE_DIR/lru/$a" && ! -e "$SDX_NATIVE_CACHE_DIR/lru/$b" ]] || fail 'hit did not affect eviction'
# A verified local-target hit also updates usage; repeated publish cannot nest locks.
age lru "$a" 202001010000
restore lru "$a"
[[ "$SDX_NATIVE_CACHE_DIR/lru/$a/.last-used" -nt "$WORK_DIR/source" ]] || fail 'local hit not touched'
timeout 5 bash -c 'source "$1"; sdx_native_cache_configure; sdx_native_cache_publish lru "$2" libfixture.o "$3"' \
  test "$SCRIPT_DIR/../../main/android/native-image-cache.sh" "$a" "$WORK_DIR/source" \
  || fail 'repeated publish failed or deadlocked'

# Restore bytes must remain independent even when the consumer modifies them.
[[ "$(stat -c '%d:%i' "$WORK_DIR/restored-lru")" != "$(stat -c '%d:%i' "$SDX_NATIVE_CACHE_DIR/lru/$a/$artifact")" ]] || fail 'hardlinked restore'
chmod u+w "$WORK_DIR/restored-lru"
printf 'consumer mutation\n' >"$WORK_DIR/restored-lru"
sdx_native_cache_validate_artifact "$SDX_NATIVE_CACHE_DIR/lru/$a/$artifact" || fail 'mutable restore damaged cache'

# Lowering the count on a hit protects that key even with future-dated peers.
SDX_NATIVE_CACHE_RETENTION=1
age lru "$c" 203001010000
restore lru "$a"
[[ -d "$SDX_NATIVE_CACHE_DIR/lru/$a" && ! -e "$SDX_NATIVE_CACHE_DIR/lru/$c" ]] || fail 'protected hit evicted'
SDX_NATIVE_CACHE_RETENTION=2
[[ -d "$SDX_NATIVE_CACHE_DIR/bounded/$b" ]] || fail 'cross-target eviction'

# Check the entire bucket before deleting anything: the last entry is unsafe.
for kind in corrupt unknown symlink nested marker-link; do
  publish "$kind" "$a"
  publish "$kind" "$b"
  age "$kind" "$a" 202001010000
  age "$kind" "$b" 202101010000
  bucket="$SDX_NATIVE_CACHE_DIR/$kind"
  if [[ "$kind" == symlink ]]; then
    ln -s "$bucket/$a" "$bucket/$d"
  else
    mkdir -p "$bucket/$d"
    cp "$bucket/$a/$artifact" "$bucket/$d/$artifact"
    cp "$bucket/$a/$artifact.sha256" "$bucket/$d/$artifact.sha256"
  fi
  case "$kind" in
    corrupt) chmod u+w "$bucket/$d/$artifact"; printf 'broken\n' >"$bucket/$d/$artifact" ;;
    unknown) printf 'unknown\n' >"$bucket/$d/.keep" ;;
    symlink) ;;
    nested) mkdir "$bucket/$d/nested" ;;
    marker-link) ln -s "$WORK_DIR/source" "$bucket/$d/.last-used" ;;
  esac
  publish "$kind" "$c"
  [[ -d "$bucket/$a" && -d "$bucket/$b" && -e "$bucket/$d" ]] || fail "$kind not fail-closed"
done
# Legacy entries without a usage marker remain eligible, using artifact mtime.
publish legacy "$a"
publish legacy "$b"
rm -- "$SDX_NATIVE_CACHE_DIR/legacy/$a/.last-used" "$SDX_NATIVE_CACHE_DIR/legacy/$b/.last-used"
touch -t 202001010000 "$SDX_NATIVE_CACHE_DIR/legacy/$a/$artifact"
touch -t 202101010000 "$SDX_NATIVE_CACHE_DIR/legacy/$b/$artifact"
publish legacy "$c"
[[ ! -e "$SDX_NATIVE_CACHE_DIR/legacy/$a" && -d "$SDX_NATIVE_CACHE_DIR/legacy/$b" ]] || fail 'legacy retention'

# Non-key children (including a stage-like directory) are never purge candidates.
publish names "$a"
publish names "$b"
mkdir -p "$SDX_NATIVE_CACHE_DIR/names/stages/keep" "$SDX_NATIVE_CACHE_DIR/names/${a}extra"
publish names "$c"
[[ -d "$SDX_NATIVE_CACHE_DIR/names/stages/keep" && -d "$SDX_NATIVE_CACHE_DIR/names/${a}extra" ]] || fail 'unknown names removed'

# Symlinked roots, ancestors, buckets and entries cannot be read or published.
mkdir "$WORK_DIR/outside"
ln -s "$WORK_DIR/outside" "$WORK_DIR/link"
for root in "$WORK_DIR/link" "$WORK_DIR/link/subdir"; do
  if (SDX_NATIVE_CACHE_DIR="$root"; publish unsafe "$a"); then fail 'symlink root accepted'; fi
done
ln -s "$WORK_DIR/outside" "$SDX_NATIVE_CACHE_DIR/link-target"
if publish link-target "$a"; then fail 'symlink bucket accepted'; fi
if restore symlink "$d"; then fail 'symlink entry restored'; fi
if publish symlink "$d"; then fail 'symlink entry published'; fi
if publish .. "$a"; then fail 'dot-dot target accepted'; fi

# A separate process holds the target lock: neither restore, publish nor pruning
# may proceed. Nonblocking lock policy avoids hangs and lets the caller build.
exec {held_fd}>>"$SDX_NATIVE_CACHE_DIR/bounded/.lock"
flock "$held_fd"
export SDX_NATIVE_CACHE_DIR SDX_NATIVE_CACHE SDX_NATIVE_FORCE_REBUILD SDX_NATIVE_CACHE_RETENTION
if timeout 5 bash -c 'source "$1"; sdx_native_cache_configure; sdx_native_cache_restore bounded "$2" libfixture.o "$3"' \
    test "$SCRIPT_DIR/../../main/android/native-image-cache.sh" "$c" "$WORK_DIR/locked-restore"; then
  fail 'restore ignored target lock'
else
  [[ "$?" != 124 ]] || fail 'restore blocked on lock'
fi
timeout 5 bash -c 'source "$1"; sdx_native_cache_configure; sdx_native_cache_publish bounded "$2" libfixture.o "$3"' \
  test "$SCRIPT_DIR/../../main/android/native-image-cache.sh" "$d" "$WORK_DIR/source" || fail 'busy publish did not skip safely'
[[ ! -e "$SDX_NATIVE_CACHE_DIR/bounded/$d" && ! -e "$WORK_DIR/locked-restore" &&
   -d "$SDX_NATIVE_CACHE_DIR/bounded/$b" ]] || fail 'locked cache changed'
flock -u "$held_fd"
exec {held_fd}>&-
publish bounded "$d"
[[ -d "$SDX_NATIVE_CACHE_DIR/bounded/$d" ]] || fail 'lock not released'

# Missing flock must bypass both operations before any filesystem mutation.
(
  command() { if [[ "$*" == '-v flock' ]]; then return 1; else builtin command "$@"; fi; }
  if restore bounded "$d"; then fail 'restore accepted without flock'; fi
  publish unavailable "$a"
)
[[ ! -e "$SDX_NATIVE_CACHE_DIR/unavailable" ]] || fail 'publish wrote without flock'
printf 'PASS: bounded native cache retention, hit LRU, safety, locking and copy independence\n'
