#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_MODULE="$REPO_ROOT/nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx-model"
CLI_CLASS="$MODEL_MODULE/target/classes/org/nd4j/dsp/model/SdxCompilerCli.class"
MVN_CMD="${MVN_CMD:-mvn}"
JAVA_CMD="${JAVA_CMD:-java}"

if ! command -v "$MVN_CMD" >/dev/null 2>&1 && [[ ! -x "$MVN_CMD" ]]; then
  echo "Maven executable not found: $MVN_CMD" >&2
  exit 1
fi
if ! command -v "$JAVA_CMD" >/dev/null 2>&1 && [[ ! -x "$JAVA_CMD" ]]; then
  echo "Java executable not found: $JAVA_CMD" >&2
  exit 1
fi

needs_build=0
if [[ ! -f "$CLI_CLASS" || "$MODEL_MODULE/pom.xml" -nt "$CLI_CLASS" ]]; then
  needs_build=1
else
  for source in "$MODEL_MODULE"/src/main/java/org/nd4j/dsp/model/*.java; do
    if [[ "$source" -nt "$CLI_CLASS" ]]; then
      needs_build=1
      break
    fi
  done
fi

if [[ "$needs_build" == "1" ]]; then
  maven_flags=()
  if [[ "${SDX_OFFLINE:-0}" == "1" || "${SDX_OFFLINE:-}" == "ON" ]]; then
    maven_flags+=("-o")
  fi
  "$MVN_CMD" "${maven_flags[@]}" -f "$MODEL_MODULE/pom.xml" -DskipTests package >&2
fi

exec "$JAVA_CMD" -cp "$MODEL_MODULE/target/classes" \
  org.nd4j.dsp.model.SdxCompilerCli "$@"
