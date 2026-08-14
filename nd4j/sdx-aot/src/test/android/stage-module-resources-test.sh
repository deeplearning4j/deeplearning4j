#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGER="$(realpath -e -- "$SCRIPT_DIR/../../main/android/stage-module-resources.sh")"
# shellcheck source=../../main/android/stage-module-resources.sh
source "$STAGER"

fail() {
  printf 'stage-module-resources-test: %s\n' "$*" >&2
  exit 1
}

TEST_ROOT="$(mktemp -d)"
trap 'rm -rf -- "$TEST_ROOT"' EXIT
source_a="$TEST_ROOT/source-a"
source_b="$TEST_ROOT/source-b"
target="$TEST_ROOT/target"
mkdir -p -- "$source_a/META-INF/native-image/a" "$source_b/META-INF/native-image/b"
printf 'feature-a\n' >"$source_a/META-INF/native-image/a/native-image.properties"
printf 'reflect-b\n' >"$source_b/META-INF/native-image/b/reflect-config.json"
mkdir -p -- "$source_a/META-INF/services" "$source_b/META-INF/services"
printf 'example.ProviderA\n' >"$source_a/META-INF/services/example.Service"
printf 'example.ProviderB\nexample.ProviderA\n' >"$source_b/META-INF/services/example.Service"

sdx_stage_module_resources "$source_a" "$target" module-a
sdx_stage_module_resources "$source_b" "$target" module-b
cmp -s -- \
  "$source_a/META-INF/native-image/a/native-image.properties" \
  "$target/META-INF/native-image/a/native-image.properties" ||
  fail "first module metadata was not staged"
cmp -s -- \
  "$source_b/META-INF/native-image/b/reflect-config.json" \
  "$target/META-INF/native-image/b/reflect-config.json" ||
  fail "second module metadata was not staged"
[[ "$(grep -c '^example[.]Provider' "$target/META-INF/services/example.Service")" == 2 ]] ||
  fail "service providers were not merged and deduplicated"
grep -q '^example[.]ProviderA$' "$target/META-INF/services/example.Service" ||
  fail "first service provider was lost"
grep -q '^example[.]ProviderB$' "$target/META-INF/services/example.Service" ||
  fail "second service provider was lost"

mkdir -p -- "$source_b/META-INF/native-image/a"
cp -p -- \
  "$source_a/META-INF/native-image/a/native-image.properties" \
  "$source_b/META-INF/native-image/a/native-image.properties"
sdx_stage_module_resources "$source_b" "$target" module-b ||
  fail "identical resource collision was rejected"

printf 'different\n' >"$source_b/META-INF/native-image/a/native-image.properties"
if sdx_stage_module_resources "$source_b" "$target" module-b >/dev/null 2>&1; then
  fail "different resource collision was accepted"
fi
grep -q '^feature-a$' "$target/META-INF/native-image/a/native-image.properties" ||
  fail "failed collision modified the original resource"

symlink_source="$TEST_ROOT/symlink-source"
mkdir -p -- "$symlink_source"
ln -s "$source_a/META-INF" "$symlink_source/META-INF"
if sdx_stage_module_resources "$symlink_source" "$TEST_ROOT/symlink-target" symlink-module >/dev/null 2>&1; then
  fail "symlink resource tree was accepted"
fi

printf 'stage-module-resources-test: PASS\n'
