#!/usr/bin/env bash
set -Eeuo pipefail

: "${DL4J_HELPER:=}"
: "${DL4J_EXTENSION:=}"
: "${DL4J_LIBND4J_FILE_DOWNLOAD:=}"
: "${DL4J_BUILD_THREADS:?DL4J_BUILD_THREADS is required}"
: "${DL4J_MATRIX_MVN_EXT:=}"
: "${DL4J_MAVEN_GOAL:=deploy}"
: "${DL4J_MAVEN_REPOSITORY:=}"

if [ -n "${DL4J_LIBND4J_FILE_DOWNLOAD}" ]; then
  modules=':nd4j-native,:nd4j-native-preset'
else
  modules=':nd4j-native,:nd4j-native-preset,:libnd4j'
fi

mvn_ext=()
if [ "${DL4J_HELPER}" = compile ]; then
  compile_suffix=compile
  if [ -n "${DL4J_EXTENSION}" ]; then
    compile_suffix="compile-${DL4J_EXTENSION}"
  fi
  mvn_ext=(-Dlibnd4j.triton=ON "-Djavacpp.platform.extension=-${compile_suffix}" "-Dlibnd4j.classifier=linux-x86_64-${compile_suffix}" -Dlibnd4j.helpers=mlir -Dlibnd4j.mlir=ON)
  if [ -n "${DL4J_EXTENSION}" ]; then
    mvn_ext+=("-Dlibnd4j.extension=${DL4J_EXTENSION}")
  fi
elif [ -n "${DL4J_HELPER}" ] && [ -n "${DL4J_EXTENSION}" ]; then
  mvn_ext=("-Dlibnd4j.helper=${DL4J_HELPER}" "-Dlibnd4j.extension=${DL4J_EXTENSION}" "-Djavacpp.platform.extension=-${DL4J_HELPER}-${DL4J_EXTENSION}" "-Dlibnd4j.classifier=linux-x86_64-${DL4J_HELPER}-${DL4J_EXTENSION}")
elif [ -n "${DL4J_HELPER}" ]; then
  mvn_ext=("-Dlibnd4j.helper=${DL4J_HELPER}" "-Dlibnd4j.extension=${DL4J_HELPER}" "-Djavacpp.platform.extension=-${DL4J_HELPER}" "-Dlibnd4j.classifier=linux-x86_64-${DL4J_HELPER}")
elif [ -n "${DL4J_EXTENSION}" ]; then
  mvn_ext=("-Dlibnd4j.extension=${DL4J_EXTENSION}" "-Djavacpp.platform.extension=-${DL4J_EXTENSION}" "-Dlibnd4j.classifier=linux-x86_64-${DL4J_EXTENSION}")
fi

matrix_ext=()
if [ -n "${DL4J_MATRIX_MVN_EXT}" ]; then
  read -r -a matrix_ext <<<"${DL4J_MATRIX_MVN_EXT}"
fi
repository=()
if [ -n "${DL4J_MAVEN_REPOSITORY}" ]; then
  repository=("-Dmaven.repo.local=${DL4J_MAVEN_REPOSITORY}")
fi

command=(mvn -X "${matrix_ext[@]}" "${repository[@]}" -Dlibnd4j.generate.flatc=ON -Dlibnd4j.sdx.standalone=ON --no-transfer-progress -pl "${modules}" -Pcpu "-Dlibnd4j.buildthreads=${DL4J_BUILD_THREADS}" -Dhttp.keepAlive=false -Dmaven.wagon.http.pool=false -Dmaven.wagon.http.retryHandler.count=3 -DskipTestResourceEnforcement=true -Dmaven.javadoc.failOnError=false -Djavacpp.platform=linux-x86_64 -Pcpu --also-make --batch-mode "${DL4J_MAVEN_GOAL}" -DskipTests "${mvn_ext[@]}")

case "${1:---run}" in
  --print)
    printf '%q ' "${command[@]}"
    printf '\n'
    ;;
  --run)
    printf '+ '
    printf '%q ' "${command[@]}"
    printf '\n'
    exec "${command[@]}"
    ;;
  *)
    printf 'Usage: %s [--print|--run]\n' "$0" >&2
    exit 2
    ;;
esac
