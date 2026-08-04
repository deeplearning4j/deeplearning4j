#!/usr/bin/env bash
set -Eeuo pipefail

: "${DL4J_FAMILY:?DL4J_FAMILY is required}"
: "${DL4J_BUILD_THREADS:?DL4J_BUILD_THREADS is required}"
: "${DL4J_HELPER:=}"
: "${DL4J_EXTENSION:=}"
: "${DL4J_MVN_FLAGS:=}"
: "${DL4J_MAVEN_GOAL:=deploy}"
: "${DL4J_MAVEN_REPOSITORY:=}"
: "${DL4J_LIBND4J_URL:=}"
: "${DL4J_CMAKE_ARGS:=}"
: "${DL4J_ANDROID_API:=24}"

split_flags=()
[ -z "${DL4J_MVN_FLAGS}" ] || read -r -a split_flags <<<"${DL4J_MVN_FLAGS}"
repo=()
[ -z "${DL4J_MAVEN_REPOSITORY}" ] || repo=("-Dmaven.repo.local=${DL4J_MAVEN_REPOSITORY}")

variant_cpu() {
  local classifier=$1
  VARIANT=()
  if [ "${DL4J_FAMILY}" = macos-arm64 ] && [ "${DL4J_HELPER}" = mps-compile ]; then
    VARIANT=(-Dlibnd4j.triton=ON -Posx-aarch64-protoc -Djavacpp.platform.extension=-mps-compile "-Dlibnd4j.classifier=${classifier}-mps-compile" -Dlibnd4j.helper=mps -Dlibnd4j.helpers=mlir -Dlibnd4j.mlir=ON)
  elif [ "${DL4J_FAMILY}" = macos-arm64 ] && [ "${DL4J_HELPER}" = mps ]; then
    VARIANT=(-Dlibnd4j.triton=ON -Posx-aarch64-protoc -Djavacpp.platform.extension=-mps "-Dlibnd4j.classifier=${classifier}-mps" -Dlibnd4j.helper=mps)
  elif [ "${DL4J_HELPER}" = compile ]; then
    VARIANT=(-Dlibnd4j.triton=ON -Djavacpp.platform.extension=-compile "-Dlibnd4j.classifier=${classifier}-compile" -Dlibnd4j.helpers=mlir -Dlibnd4j.mlir=ON)
  elif [ "${DL4J_HELPER}" = compile-nnapi ]; then
    VARIANT=(-Djavacpp.platform.extension=-compile-nnapi "-Dlibnd4j.classifier=${classifier}-compile-nnapi" -Dlibnd4j.helpers=mlir,nnapi -Dlibnd4j.mlir=ON)
  elif [ -n "${DL4J_HELPER}" ] && [ -n "${DL4J_EXTENSION}" ]; then
    VARIANT=("-Dlibnd4j.helper=${DL4J_HELPER}" "-Dlibnd4j.extension=${DL4J_EXTENSION}" "-Djavacpp.platform.extension=-${DL4J_HELPER}-${DL4J_EXTENSION}" "-Dlibnd4j.classifier=${classifier}-${DL4J_HELPER}-${DL4J_EXTENSION}")
  elif [ -n "${DL4J_HELPER}" ]; then
    VARIANT=("-Dlibnd4j.helper=${DL4J_HELPER}" "-Djavacpp.platform.extension=-${DL4J_HELPER}" "-Dlibnd4j.classifier=${classifier}-${DL4J_HELPER}")
    [ "${DL4J_FAMILY}" != windows-cpu ] || VARIANT+=("-Dlibnd4j.extension=${DL4J_HELPER}")
  elif [ -n "${DL4J_EXTENSION}" ]; then
    VARIANT=("-Dlibnd4j.extension=${DL4J_EXTENSION}" "-Djavacpp.platform.extension=-${DL4J_EXTENSION}" "-Dlibnd4j.classifier=${classifier}-${DL4J_EXTENSION}")
  else
    VARIANT=("-Dlibnd4j.classifier=${classifier}")
  fi
}

case "${DL4J_FAMILY}" in
  linux-arm64|macos-arm64|android-x86_64|android-arm64)
    case "${DL4J_FAMILY}" in
      linux-arm64) platform=linux-arm64; profiles=(-Posx-aarch64-protoc -Pcpu); extra=();;
      macos-arm64) platform=macosx-arm64; profiles=(-Pcpu -Pmetal -Posx-aarch64-protoc); extra=(-Dlibnd4j.arch=armv8-a -Dlibnd4j.platform=macosx-arm64);;
      android-x86_64) platform=android-x86_64; profiles=(-Pcpu); extra=("-Dlibnd4j.cmake=${DL4J_CMAKE_ARGS}" "-Dlibnd4j.android.api=${DL4J_ANDROID_API}" -Dlibnd4j.build.with.java=OFF);;
      android-arm64) platform=android-arm64; profiles=(-Posx-aarch64-protoc -Pcpu); extra=("-Djavacpp.platform.compiler=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++" "-Dlibnd4j.cmake=${DL4J_CMAKE_ARGS}" "-Dlibnd4j.android.api=${DL4J_ANDROID_API}" -Dlibnd4j.build.with.java=OFF);;
    esac
    modules=:nd4j-native,:nd4j-native-preset
    [ -n "${DL4J_LIBND4J_URL}" ] || modules+=,:libnd4j
    variant_cpu "${platform}"
    command=(mvn "${split_flags[@]}" "${repo[@]}" -Dlibnd4j.generate.flatc=ON -Dlibnd4j.sdx.standalone=ON --no-transfer-progress "${profiles[@]}" "-Dlibnd4j.buildthreads=${DL4J_BUILD_THREADS}" -Dhttp.keepAlive=false -Dmaven.wagon.http.pool=false -Dmaven.wagon.http.retryHandler.count=3 -DskipTestResourceEnforcement=true -Dmaven.javadoc.failOnError=false "-Djavacpp.platform=${platform}" --batch-mode -DskipTests "${extra[@]}" "${VARIANT[@]}" -pl "${modules}" --also-make "${DL4J_MAVEN_GOAL}")
    ;;
  linux-cuda|windows-cuda)
    : "${DL4J_CUDA_VERSION:?DL4J_CUDA_VERSION is required}"
    platform=linux-x86_64
    [ "${DL4J_FAMILY}" = linux-cuda ] || platform=windows-x86_64
    modules=":nd4j-cuda-${DL4J_CUDA_VERSION},:nd4j-cuda-${DL4J_CUDA_VERSION}-preset"
    [ -n "${DL4J_LIBND4J_URL}" ] || modules+=,:libnd4j
    if [ "${DL4J_HELPER}" = compile ]; then
      variant=(-Dlibnd4j.triton=ON -Djavacpp.platform.extension=-compile "-Dlibnd4j.classifier=${platform}-cuda-${DL4J_CUDA_VERSION}-compile")
    elif [ -n "${DL4J_HELPER}" ] && [ -n "${DL4J_EXTENSION}" ]; then
      variant=("-Djavacpp.platform.extension=-${DL4J_HELPER}-${DL4J_EXTENSION}" "-Dlibnd4j.helper=${DL4J_HELPER}" "-Dlibnd4j.extension=${DL4J_EXTENSION}" "-Dlibnd4j.classifier=${platform}-cuda-${DL4J_CUDA_VERSION}-${DL4J_HELPER}-${DL4J_EXTENSION}")
    elif [ -n "${DL4J_HELPER}" ]; then
      variant=("-Djavacpp.platform.extension=-${DL4J_HELPER}" "-Dlibnd4j.helper=${DL4J_HELPER}" "-Dlibnd4j.classifier=${platform}-cuda-${DL4J_CUDA_VERSION}-${DL4J_HELPER}")
      [ "${DL4J_FAMILY}" != windows-cuda ] || variant+=("-Dlibnd4j.extension=${DL4J_HELPER}")
    else
      variant=("-Dlibnd4j.classifier=${platform}-cuda-${DL4J_CUDA_VERSION}")
    fi
    win=(); [ "${DL4J_FAMILY}" = linux-cuda ] || win=(-Dlibnd4j.platform=windows-x86_64 -Dlibnd4j.oom.killer=OFF)
    command=(mvn "${split_flags[@]}" "${repo[@]}" -Pcuda -Dlibnd4j.generate.flatc=ON -Dlibnd4j.sdx.standalone=ON -Dlibnd4j.oom.memory.threshold=95 -Dlibnd4j.oom.velocity.threshold=40 --no-transfer-progress -Dlibnd4j.cuda.compile.skip=false -Dlibnd4j.chip=cuda -Pcuda '-Dlibnd4j.compute=8.6 9.0' -Dlibnd4j.cpu.compile.skip=true "-Dlibnd4j.buildthreads=${DL4J_BUILD_THREADS}" -Dhttp.keepAlive=false -Dmaven.wagon.http.pool=false -Dmaven.wagon.http.retryHandler.count=3 "-Djavacpp.platform=${platform}" "${win[@]}" -DskipTests "${variant[@]}" -pl "${modules}" --also-make "${DL4J_MAVEN_GOAL}")
    ;;
  windows-cpu)
    modules=:nd4j-native-preset,:nd4j-native
    [ -n "${DL4J_LIBND4J_URL}" ] || modules+=,:libnd4j
    variant_cpu windows-x86_64
    command=(mvn "${split_flags[@]}" "${repo[@]}" -Dlibnd4j.generate.flatc=ON -Dlibnd4j.sdx.standalone=ON -Dlibnd4j.oom.memory.threshold=95 -Dlibnd4j.oom.velocity.threshold=40 --no-transfer-progress -Pcpu "-Dlibnd4j.buildthreads=${DL4J_BUILD_THREADS}" -Dhttp.keepAlive=false -Dmaven.wagon.http.pool=false -Dmaven.wagon.http.retryHandler.count=3 -Djavacpp.platform=windows-x86_64 -Dlibnd4j.platform=windows-x86_64 -DskipTests "${VARIANT[@]}" -pl "${modules}" --also-make "${DL4J_MAVEN_GOAL}")
    ;;
  vulkan|vulkan-mlir|hexagon|tpu)
    backend=${DL4J_FAMILY%%-*}; [ "${DL4J_FAMILY}" != vulkan-mlir ] || backend=vulkan
    classifier=linux-x86_64
    [ "${DL4J_FAMILY}" != vulkan-mlir ] || classifier=linux-x86_64-compile
    modules=":nd4j-${backend},:nd4j-${backend}-preset"
    [ -n "${DL4J_LIBND4J_URL}" ] || modules+=,:libnd4j
    flags=("-Dlibnd4j.${backend}")
    if [ "${DL4J_FAMILY}" = vulkan ] || [ "${DL4J_FAMILY}" = vulkan-mlir ]; then flags+=(-Dlibnd4j.triton=ON); fi
    [ "${DL4J_FAMILY}" != vulkan-mlir ] || flags+=(-Dlibnd4j.mlir=ON)
    command=(mvn "${split_flags[@]}" "${repo[@]}" --no-transfer-progress "-P${backend}" -pl "${modules}" "${flags[@]}" "-Dlibnd4j.buildthreads=${DL4J_BUILD_THREADS}" -Dhttp.keepAlive=false -Dmaven.wagon.http.pool=false -Dmaven.wagon.http.retryHandler.count=3 -DskipTestResourceEnforcement=true -Dmaven.javadoc.failOnError=false -Djavacpp.platform=linux-x86_64 "-Dplatform.classifier=${classifier}" --also-make --batch-mode "${DL4J_MAVEN_GOAL}" -DskipTests)
    ;;
  compat)
    command=(mvn -pl :nd4j-native-preset,:libnd4j,:nd4j-native "${split_flags[@]}" "${repo[@]}" -Pcpu "-Dlibnd4j.buildthreads=${DL4J_BUILD_THREADS}" -Dhttp.keepAlive=false -Dmaven.wagon.http.pool=false -Dmaven.wagon.http.retryHandler.count=3 -DskipTestResourceEnforcement=true -Dmaven.javadoc.failOnError=false -Djavacpp.platform=linux-x86_64 -Pcpu --also-make --batch-mode -DskipTests -Dlibnd4j.extension=compat -Djavacpp.platform.extension=-compat -Dlibnd4j.classifier=linux-x86_64-compat "${DL4J_MAVEN_GOAL}")
    ;;
  zluda|windows-zluda)
    : "${DL4J_ZLUDA_TARGET:=AMD}"
    platform=linux-x86_64
    zluda_win=()
    if [ "${DL4J_FAMILY}" = windows-zluda ]; then
      platform=windows-x86_64
      zluda_win=(-Dlibnd4j.platform=windows-x86_64 -Dlibnd4j.oom.killer=OFF)
    fi
    command=(mvn "${split_flags[@]}" "${repo[@]}" -Pcuda -Pzluda -Dlibnd4j.generate.flatc=ON -Dlibnd4j.oom.memory.threshold=95 -Dlibnd4j.oom.velocity.threshold=40 --no-transfer-progress -Dlibnd4j.cuda.compile.skip=false -Dlibnd4j.chip=cuda '-Dlibnd4j.compute=8.6 9.0' -Dlibnd4j.cpu.compile.skip=true "-Dlibnd4j.zluda=${DL4J_ZLUDA_TARGET}" -Djavacpp.platform.extension=-zluda "-Dlibnd4j.classifier=${platform}-cuda-12.9-zluda" -Dhttp.keepAlive=false -Dmaven.wagon.http.pool=false -Dmaven.wagon.http.retryHandler.count=3 "-Dlibnd4j.buildthreads=${DL4J_BUILD_THREADS}" "-Djavacpp.platform=${platform}" "${zluda_win[@]}" --batch-mode -DskipTests -pl :nd4j-cuda-12.9,:nd4j-cuda-12.9-preset,:nd4j-zluda,:libnd4j --also-make install)
    ;;
  *) printf 'Unsupported DL4J_FAMILY=%s\n' "${DL4J_FAMILY}" >&2; exit 2;;
esac

case "${1:---run}" in
  --print) printf '%q ' "${command[@]}"; printf '\n';;
  --run) printf '+ '; printf '%q ' "${command[@]}"; printf '\n'; exec "${command[@]}";;
  *) printf 'Usage: %s [--print|--run]\n' "$0" >&2; exit 2;;
esac
