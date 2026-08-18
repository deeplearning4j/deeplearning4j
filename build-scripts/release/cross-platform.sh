#!/usr/bin/env bash
set -Eeuo pipefail

: "${DL4J_PLATFORM:?DL4J_PLATFORM is required}"
: "${DL4J_OS:?DL4J_OS is required}"
: "${DL4J_MAVEN_GOAL:=deploy}"
: "${DL4J_MAVEN_REPOSITORY:=}"
: "${DL4J_BUILD_SDX:=0}"

case "${DL4J_BUILD_SDX}" in
  0|1) ;;
  *) printf 'DL4J_BUILD_SDX must be 0 or 1: %s\n' "${DL4J_BUILD_SDX}" >&2; exit 2 ;;
esac

repository=()
if [ -n "${DL4J_MAVEN_REPOSITORY}" ]; then
  repository=("-Dmaven.repo.local=${DL4J_MAVEN_REPOSITORY}")
fi
mingw=()
if [ "${DL4J_OS}" = windows ]; then
  # The Maven JavaCPP configurations pass javacpp.platform.build to the plugin's
  # properties selector. Keep platform.properties too for direct JavaCPP callers.
  mingw=(-Djavacpp.platform.build=windows-x86_64-mingw -Djavacpp.platform.properties=windows-x86_64-mingw -Djavacpp.platform.compiler=g++)
fi
protoc_profile=()
if [ "${DL4J_PLATFORM}" = linux-arm64 ] || [ "${DL4J_PLATFORM}" = macosx-arm64 ]; then
  protoc_profile=(-Posx-aarch64-protoc)
fi
sdx_profile=()
if [ "${DL4J_BUILD_SDX}" = 1 ]; then
  sdx_profile=(-Psdx)
fi
# Architecture selects only cross-platform toolchain behavior. Accelerator profiles
# belong exclusively to their explicit CUDA, Metal, TPU, Hexagon, Vulkan, and ZLUDA
# matrix lanes; inferring them here would contaminate CPU builds.
tokenizers=(mvn -pl :libtokenizers,:tokenizers-native-preset,:tokenizers-native --also-make "-Djavacpp.platform=${DL4J_PLATFORM}" ${mingw[@]+"${mingw[@]}"} ${repository[@]+"${repository[@]}"} -DskipTestResourceEnforcement=true -Dmaven.javadoc.failOnError=false --no-transfer-progress --batch-mode "${DL4J_MAVEN_GOAL}" -DskipTests)
java=(mvn -pl '!:blas-lapack-generator,!:libnd4j-gen,!:libnd4j,!:libtokenizers,!:tokenizers-native-preset,!:tokenizers-native,!:platform-tests' ${protoc_profile[@]+"${protoc_profile[@]}"} "${sdx_profile[@]}" ${repository[@]+"${repository[@]}"} -DskipTestResourceEnforcement=true "-Djavacpp.platform=${DL4J_PLATFORM}" ${mingw[@]+"${mingw[@]}"} -Dmaven.javadoc.failOnError=false -Dmaven.test.skip=true --no-transfer-progress --batch-mode "${DL4J_MAVEN_GOAL}")

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
