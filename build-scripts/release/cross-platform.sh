#!/usr/bin/env bash
set -Eeuo pipefail

: "${DL4J_PLATFORM:?DL4J_PLATFORM is required}"
: "${DL4J_OS:?DL4J_OS is required}"
: "${DL4J_MAVEN_GOAL:=deploy}"
: "${DL4J_MAVEN_REPOSITORY:=}"

repository=()
if [ -n "${DL4J_MAVEN_REPOSITORY}" ]; then
  repository=("-Dmaven.repo.local=${DL4J_MAVEN_REPOSITORY}")
fi
mingw=()
if [ "${DL4J_OS}" = windows ]; then
  mingw=(-Djavacpp.platform.build=windows-x86_64-mingw -Djavacpp.platform.compiler=g++)
fi
protoc_profile=()
if [ "${DL4J_PLATFORM}" = linux-arm64 ] || [ "${DL4J_PLATFORM}" = macosx-arm64 ]; then
  protoc_profile=(-Posx-aarch64-protoc)
fi
platform_profiles=()
if [ "${DL4J_PLATFORM}" = linux-x86_64 ]; then
  platform_profiles=(-Pzluda,tpu,hexagon)
fi

tokenizers=(mvn -pl :libtokenizers,:tokenizers-native-preset,:tokenizers-native --also-make "-Djavacpp.platform=${DL4J_PLATFORM}" "${mingw[@]}" "${repository[@]}" -DskipTestResourceEnforcement=true -Dmaven.javadoc.failOnError=false --no-transfer-progress --batch-mode "${DL4J_MAVEN_GOAL}" -DskipTests)
java=(mvn -pl '!:blas-lapack-generator,!:libnd4j-gen,!:libnd4j,!:libtokenizers,!:tokenizers-native-preset,!:tokenizers-native,!:platform-tests' "${protoc_profile[@]}" "${platform_profiles[@]}" "${repository[@]}" -DskipTestResourceEnforcement=true "-Djavacpp.platform=${DL4J_PLATFORM}" -Dmaven.javadoc.failOnError=false -Dmaven.test.skip=true --no-transfer-progress --batch-mode "${DL4J_MAVEN_GOAL}")

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

case "${1:-}" in
  --print-tokenizers) print_command "${tokenizers[@]}" ;;
  --print-java) print_command "${java[@]}" ;;
  --run-tokenizers)
    printf '+ '; print_command "${tokenizers[@]}"
    exec "${tokenizers[@]}"
    ;;
  --run-java)
    printf '+ '; print_command "${java[@]}"
    exec "${java[@]}"
    ;;
  *)
    printf 'Usage: %s --print-tokenizers|--print-java|--run-tokenizers|--run-java\n' "$0" >&2
    exit 2
    ;;
esac
