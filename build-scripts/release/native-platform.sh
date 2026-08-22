#!/usr/bin/env bash
set -Eeuo pipefail

: "${DL4J_FAMILY:?DL4J_FAMILY is required}"
: "${DL4J_BUILD_THREADS:?DL4J_BUILD_THREADS is required}"
: "${DL4J_HELPER:=}"
: "${DL4J_EXTENSION:=}"
: "${DL4J_PLATFORM_EXTENSION:=}"
: "${DL4J_CLASSIFIER:=}"
: "${DL4J_MVN_FLAGS:=}"
: "${DL4J_MAVEN_GOAL:=deploy}"
: "${DL4J_MAVEN_REPOSITORY:=}"
: "${DL4J_ROCM_VERSION:=}"
: "${DL4J_ZLUDA_VERSION:=v7-preview.8}"
: "${DL4J_LIBND4J_URL:=}"
: "${DL4J_CMAKE_ARGS:=}"
: "${DL4J_ANDROID_API:=24}"
: "${DL4J_NATIVE_ONLY:=0}"
: "${DL4J_MAVEN_ALSO_MAKE:=1}"
: "${DL4J_BUILD_SDX:=0}"
: "${DL4J_SDX_CLASSIFIER:=${DL4J_CLASSIFIER}}"
: "${DL4J_PROTOC_COMMAND:=}"
case "$DL4J_NATIVE_ONLY" in
  0|1) ;;
  *) printf 'DL4J_NATIVE_ONLY must be 0 or 1: %s\n' "$DL4J_NATIVE_ONLY" >&2; exit 2 ;;
esac
case "$DL4J_BUILD_SDX" in
  0|1) ;;
  *) printf 'DL4J_BUILD_SDX must be 0 or 1: %s\n' "$DL4J_BUILD_SDX" >&2; exit 2 ;;
esac
case "$DL4J_MAVEN_ALSO_MAKE" in
  0|1) ;;
  *) printf 'DL4J_MAVEN_ALSO_MAKE must be 0 or 1: %s\n' "$DL4J_MAVEN_ALSO_MAKE" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
: "${DL4J_NATIVE_OUTPUT_ROOT:=${REPO_ROOT}/libnd4j/blasbuild}"
: "${DL4J_SDX_NATIVE_LIBRARY:=nd4jcpu}"
: "${DL4J_SDX_PLATFORM_LINKS:=${DL4J_SDX_NATIVE_LIBRARY}}"
: "${DL4J_SDX_OUTPUT_PATH:=${REPO_ROOT}/libnd4j/blasbuild/cpu}"
case "$DL4J_NATIVE_OUTPUT_ROOT" in
  /*) ;;
  *) printf 'DL4J_NATIVE_OUTPUT_ROOT must be absolute: %s\n' "$DL4J_NATIVE_OUTPUT_ROOT" >&2; exit 2 ;;
esac

split_flags=()
[ -z "${DL4J_MVN_FLAGS}" ] || read -r -a split_flags <<<"${DL4J_MVN_FLAGS}"
protoc_command=${DL4J_PROTOC_COMMAND}
if [ -z "${protoc_command}" ]; then
  for candidate in /opt/protoc-21.7/bin/protoc /usr/local/bin/protoc; do
    if [ -x "${candidate}" ]; then
      protoc_command=${candidate}
      break
    fi
  done
fi
if [ -n "${protoc_command}" ]; then
  split_flags+=("-DprotocCommand=${protoc_command}" "-DprotocExecutable=${protoc_command}")
fi
repo=()
[ -z "${DL4J_MAVEN_REPOSITORY}" ] || repo=("-Dmaven.repo.local=${DL4J_MAVEN_REPOSITORY}")
also_make=()
[ "$DL4J_MAVEN_ALSO_MAKE" != 1 ] || also_make=(--also-make)
sdx_profile=()
sdx_maven_flags=()
if [ "$DL4J_BUILD_SDX" = 1 ]; then
  sdx_profile=(-Psdx)
  sdx_maven_flags=(
    "-Dsdx.native.library=${DL4J_SDX_NATIVE_LIBRARY}"
    "-Dsdx.platform.links=${DL4J_SDX_PLATFORM_LINKS}"
    "-Dlibnd4j.outputPath=${DL4J_SDX_OUTPUT_PATH}"
    "-Dsdx.platform.classifier=${DL4J_SDX_CLASSIFIER}"
  )
  if [ "$DL4J_FAMILY" = windows-cuda ] ||
     [ "$DL4J_FAMILY" = windows-zluda ]; then
    # MSVC's import library is generated beside nd4jcuda.dll. Pass its absolute
    # stem so JavaCPP emits the exact file path instead of relying on a later
    # /LIBPATH lookup, which can fail under the Git Bash/MSVC boundary.
    sdx_maven_flags+=(
      "-Dsdx.platform.links=${DL4J_SDX_OUTPUT_PATH}/nd4jcuda"
    )
  fi
fi
append_sdx_modules() {
  [ "$DL4J_BUILD_SDX" != 1 ] || modules+=,:nd4j-sdx-preset,:nd4j-sdx-model,:nd4j-sdx,:nd4j-sdx-litertlm
}

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
  elif [ -n "${DL4J_EXTENSION}" ]; then
    VARIANT=("-Dlibnd4j.extension=${DL4J_EXTENSION}" "-Djavacpp.platform.extension=-${DL4J_EXTENSION}" "-Dlibnd4j.classifier=${classifier}-${DL4J_EXTENSION}")
  else
    VARIANT=("-Dlibnd4j.classifier=${classifier}")
  fi
}

case "${DL4J_FAMILY}" in
  linux-arm64|macos-arm64|android-x86_64|android-arm64)
    case "${DL4J_FAMILY}" in
      # This shard runs natively on ARM64; use the VM compiler and its matching GCC runtime.
      linux-arm64) platform=linux-arm64; profiles=(-Posx-aarch64-protoc -Pcpu); extra=(-Djavacpp.platform.compiler=g++);;
      macos-arm64) platform=macosx-arm64; profiles=(-Pcpu -Pmetal -Posx-aarch64-protoc); extra=(-Dlibnd4j.arch=armv8-a -Dlibnd4j.platform=macosx-arm64);;
      android-x86_64) platform=android-x86_64; profiles=(-Pcpu); extra=("-Djavacpp.platform.compiler=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/x86_64-linux-android${DL4J_ANDROID_API}-clang++" "-Dlibnd4j.cmake=${DL4J_CMAKE_ARGS}" "-Dlibnd4j.android.api=${DL4J_ANDROID_API}" "-Dandroid.api=${DL4J_ANDROID_API}" "-Djavacpp.platform.sysroot=--sysroot=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot" "-Djavacpp.compiler.options=--sysroot=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot" "-Dlibnd4j.outputPath=${DL4J_NATIVE_OUTPUT_ROOT}/android-x86_64-api${DL4J_ANDROID_API}-cpu" -Dlibnd4j.build.with.java=OFF);;
      android-arm64) platform=android-arm64; profiles=(-Posx-aarch64-protoc -Pcpu); extra=("-Djavacpp.platform.compiler=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android${DL4J_ANDROID_API}-clang++" "-Djavacpp.platform.sysroot=--sysroot=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot" "-Djavacpp.compiler.options=--sysroot=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot" "-Dlibnd4j.cmake=${DL4J_CMAKE_ARGS}" "-Dlibnd4j.android.api=${DL4J_ANDROID_API}" "-Dandroid.api=${DL4J_ANDROID_API}" "-Dlibnd4j.outputPath=${DL4J_NATIVE_OUTPUT_ROOT}/android-arm64-api${DL4J_ANDROID_API}-cpu" -Dlibnd4j.build.with.java=OFF);;
    esac
    modules=:nd4j-native,:nd4j-native-preset
    [ -n "${DL4J_LIBND4J_URL}" ] || modules+=,:libnd4j
    # Native compilation must be independently receiptable. Callers that only need
    # the CMake product should not fail on unrelated Java modules in the full reactor.
    [ "$DL4J_NATIVE_ONLY" != 1 ] || modules=:libnd4j
    [ "$DL4J_NATIVE_ONLY" = 1 ] || append_sdx_modules
    variant_cpu "${platform}"
    command=(mvn ${split_flags[@]+"${split_flags[@]}"} ${repo[@]+"${repo[@]}"} -Dlibnd4j.generate.flatc=ON -Dlibnd4j.sdx.standalone=ON -Dlibnd4j.oom.memory.threshold=95 -Dlibnd4j.oom.velocity.threshold=40 --no-transfer-progress "${profiles[@]}" "${sdx_profile[@]}" "-Dlibnd4j.buildthreads=${DL4J_BUILD_THREADS}" -Dhttp.keepAlive=false -Dmaven.wagon.http.pool=false -Dmaven.wagon.http.retryHandler.count=3 -DskipTestResourceEnforcement=true -Dmaven.javadoc.failOnError=false "-Djavacpp.platform=${platform}" --batch-mode -DskipTests "${extra[@]}" "${VARIANT[@]}" "${sdx_maven_flags[@]}" -pl "${modules}" ${also_make[@]+"${also_make[@]}"} "${DL4J_MAVEN_GOAL}")
    ;;
  linux-cuda|windows-cuda)
    : "${DL4J_CUDA_VERSION:?DL4J_CUDA_VERSION is required}"
    platform=linux-x86_64
    [ "${DL4J_FAMILY}" = linux-cuda ] || platform=windows-x86_64
    modules=":nd4j-cuda-${DL4J_CUDA_VERSION},:nd4j-cuda-${DL4J_CUDA_VERSION}-preset"
    [ -n "${DL4J_LIBND4J_URL}" ] || modules+=,:libnd4j
    append_sdx_modules
    if [ "${DL4J_HELPER}" = compile ]; then
      # Linux uses the managed Triton/MLIR compile-only classifier. Windows
      # native compile variants leave DL4J_HELPER empty and use the extension
      # branch below so MSVC/MinGW does not request managed LLVM.
      variant=(-Dlibnd4j.triton=ON -Djavacpp.platform.extension=-compile "-Dlibnd4j.classifier=${platform}-cuda-${DL4J_CUDA_VERSION}-compile")
    elif [ -n "${DL4J_HELPER}" ] && [ -n "${DL4J_EXTENSION}" ]; then
      variant=("-Djavacpp.platform.extension=-${DL4J_HELPER}-${DL4J_EXTENSION}" "-Dlibnd4j.helper=${DL4J_HELPER}" "-Dlibnd4j.extension=${DL4J_EXTENSION}" "-Dlibnd4j.classifier=${platform}-cuda-${DL4J_CUDA_VERSION}-${DL4J_HELPER}-${DL4J_EXTENSION}")
    elif [ -n "${DL4J_HELPER}" ]; then
      variant=("-Djavacpp.platform.extension=-${DL4J_HELPER}" "-Dlibnd4j.helper=${DL4J_HELPER}" "-Dlibnd4j.classifier=${platform}-cuda-${DL4J_CUDA_VERSION}-${DL4J_HELPER}")
    elif [ -n "${DL4J_EXTENSION}" ]; then
      variant=("-Djavacpp.platform.extension=-${DL4J_EXTENSION}" "-Dlibnd4j.extension=${DL4J_EXTENSION}" "-Dlibnd4j.classifier=${platform}-cuda-${DL4J_CUDA_VERSION}-${DL4J_EXTENSION}")
    else
      variant=("-Dlibnd4j.classifier=${platform}-cuda-${DL4J_CUDA_VERSION}")
    fi
    win=(); [ "${DL4J_FAMILY}" = linux-cuda ] || win=(-Dlibnd4j.platform=windows-x86_64 -Dlibnd4j.oom.killer=OFF)
    command=(mvn ${split_flags[@]+"${split_flags[@]}"} ${repo[@]+"${repo[@]}"} -Pcuda "${sdx_profile[@]}" -Dlibnd4j.generate.flatc=ON -Dlibnd4j.sdx.standalone=ON -Dlibnd4j.oom.memory.threshold=95 -Dlibnd4j.oom.velocity.threshold=40 --no-transfer-progress -Dlibnd4j.cuda.compile.skip=false -Dlibnd4j.chip=cuda -Pcuda '-Dlibnd4j.compute=8.6 9.0' -Dlibnd4j.cpu.compile.skip=true "-Dlibnd4j.buildthreads=${DL4J_BUILD_THREADS}" -Dhttp.keepAlive=false -Dmaven.wagon.http.pool=false -Dmaven.wagon.http.retryHandler.count=3 "-Djavacpp.platform=${platform}" ${win[@]+"${win[@]}"} -DskipTests "${variant[@]}" "${sdx_maven_flags[@]}" -pl "${modules}" ${also_make[@]+"${also_make[@]}"} "${DL4J_MAVEN_GOAL}")
    ;;
  windows-cpu)
    modules=:nd4j-native-preset,:nd4j-native
    [ -n "${DL4J_LIBND4J_URL}" ] || modules+=,:libnd4j
    [ "$DL4J_NATIVE_ONLY" = 1 ] || append_sdx_modules
    variant_cpu windows-x86_64
    command=(mvn ${split_flags[@]+"${split_flags[@]}"} ${repo[@]+"${repo[@]}"} -Dlibnd4j.generate.flatc=ON -Dlibnd4j.sdx.standalone=ON -Dlibnd4j.oom.memory.threshold=95 -Dlibnd4j.oom.velocity.threshold=40 --no-transfer-progress -Pcpu "${sdx_profile[@]}" "-Dlibnd4j.buildthreads=${DL4J_BUILD_THREADS}" -Dhttp.keepAlive=false -Dmaven.wagon.http.pool=false -Dmaven.wagon.http.retryHandler.count=3 -Djavacpp.platform=windows-x86_64 -Djavacpp.platform.build=windows-x86_64-mingw -Djavacpp.platform.properties=windows-x86_64-mingw -Djavacpp.platform.compiler=g++ -Dlibnd4j.platform=windows-x86_64 -DskipTests "${VARIANT[@]}" "${sdx_maven_flags[@]}" -pl "${modules}" ${also_make[@]+"${also_make[@]}"} "${DL4J_MAVEN_GOAL}")
    ;;
  android-arm64-vulkan|android-x86_64-vulkan)
    case "${DL4J_FAMILY}" in
      android-arm64-vulkan)
        platform=android-arm64
        profiles=(-Posx-aarch64-protoc -Pvulkan)
        extra=("-Djavacpp.platform.compiler=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android${DL4J_ANDROID_API}-clang++" "-Djavacpp.platform.sysroot=--sysroot=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot" "-Djavacpp.compiler.options=--sysroot=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot" "-Dlibnd4j.cmake=${DL4J_CMAKE_ARGS}" "-Dlibnd4j.android.api=${DL4J_ANDROID_API}" "-Dandroid.api=${DL4J_ANDROID_API}" -Dlibnd4j.build.with.java=OFF)
        ;;
      android-x86_64-vulkan)
        platform=android-x86_64
        profiles=(-Pvulkan)
        extra=("-Djavacpp.platform.compiler=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/x86_64-linux-android${DL4J_ANDROID_API}-clang++" "-Djavacpp.platform.sysroot=--sysroot=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot" "-Djavacpp.compiler.options=--sysroot=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot" "-Dlibnd4j.cmake=${DL4J_CMAKE_ARGS}" "-Dlibnd4j.android.api=${DL4J_ANDROID_API}" "-Dandroid.api=${DL4J_ANDROID_API}" -Dlibnd4j.build.with.java=OFF)
        ;;
    esac
    modules=:nd4j-vulkan,:nd4j-vulkan-preset,:nd4j-vulkan-platform
    [ -n "${DL4J_LIBND4J_URL}" ] || modules+=,:libnd4j
    append_sdx_modules
    variant_cpu "${platform}"
    command=(mvn ${split_flags[@]+"${split_flags[@]}"} ${repo[@]+"${repo[@]}"} --no-transfer-progress "${profiles[@]}" -pl "${modules}" -Dlibnd4j.vulkan -Dlibnd4j.triton=ON "-Dlibnd4j.buildthreads=${DL4J_BUILD_THREADS}" -Dhttp.keepAlive=false -Dmaven.wagon.http.pool=false -Dmaven.wagon.http.retryHandler.count=3 -DskipTestResourceEnforcement=true -Dmaven.javadoc.failOnError=true "-Djavacpp.platform=${platform}" "-Dplatform.classifier=${platform}" ${also_make[@]+"${also_make[@]}"} --batch-mode "${extra[@]}" "${VARIANT[@]}" "${DL4J_MAVEN_GOAL}" -DskipTests)
    ;;
  windows-vulkan)
    platform=windows-x86_64
    modules=:nd4j-vulkan,:nd4j-vulkan-preset,:nd4j-vulkan-platform
    [ -n "${DL4J_LIBND4J_URL}" ] || modules+=,:libnd4j
    append_sdx_modules
    variant_cpu "${platform}"
    command=(mvn ${split_flags[@]+"${split_flags[@]}"} ${repo[@]+"${repo[@]}"} --no-transfer-progress -Pvulkan "${sdx_profile[@]}" -pl "${modules}" -Dlibnd4j.vulkan -Dlibnd4j.mlir=ON -Dlibnd4j.triton=ON "-Dlibnd4j.buildthreads=${DL4J_BUILD_THREADS}" -Dhttp.keepAlive=false -Dmaven.wagon.http.pool=false -Dmaven.wagon.http.retryHandler.count=3 -DskipTestResourceEnforcement=true -Dmaven.javadoc.failOnError=false "-Djavacpp.platform=${platform}" -Djavacpp.platform.build=windows-x86_64-mingw -Djavacpp.platform.properties=windows-x86_64-mingw -Djavacpp.platform.compiler=g++ "-Dlibnd4j.platform=${platform}" "-Dplatform.classifier=${platform}" -Dlibnd4j.oom.killer=OFF ${also_make[@]+"${also_make[@]}"} --batch-mode "${VARIANT[@]}" "${sdx_maven_flags[@]}" "${DL4J_MAVEN_GOAL}" -DskipTests)
    ;;
  vulkan|vulkan-mlir|hexagon|tpu)
    backend=${DL4J_FAMILY%%-*}; [ "${DL4J_FAMILY}" != vulkan-mlir ] || backend=vulkan
    classifier=linux-x86_64
    [ "${DL4J_FAMILY}" != vulkan-mlir ] || classifier=linux-x86_64-compile
    modules=":nd4j-${backend},:nd4j-${backend}-preset"
    [ "${backend}" != tpu ] || modules+=,:nd4j-cpu-backend-common
    [ "${backend}" != vulkan ] || modules+=,:nd4j-vulkan-platform
    [ -n "${DL4J_LIBND4J_URL}" ] || modules+=,:libnd4j
    [ "${backend}" != vulkan ] || append_sdx_modules
    flags=("-Dlibnd4j.${backend}")
    if [ "${DL4J_FAMILY}" = vulkan ] || [ "${DL4J_FAMILY}" = vulkan-mlir ]; then flags+=(-Dlibnd4j.triton=ON -Dlibnd4j.mlir=ON); fi
    [ "${DL4J_FAMILY}" != vulkan-mlir ] || flags+=(-Dlibnd4j.mlir=ON)
    command=(mvn ${split_flags[@]+"${split_flags[@]}"} ${repo[@]+"${repo[@]}"} --no-transfer-progress "-P${backend}" "${sdx_profile[@]}" -pl "${modules}" "${flags[@]}" "-Dlibnd4j.buildthreads=${DL4J_BUILD_THREADS}" -Dhttp.keepAlive=false -Dmaven.wagon.http.pool=false -Dmaven.wagon.http.retryHandler.count=3 -DskipTestResourceEnforcement=true -Dmaven.javadoc.failOnError=false -Djavacpp.platform=linux-x86_64 "-Dplatform.classifier=${classifier}" ${also_make[@]+"${also_make[@]}"} --batch-mode "${sdx_maven_flags[@]}" "${DL4J_MAVEN_GOAL}" -DskipTests)
    ;;
  compat)
    command=(mvn -pl :nd4j-native-preset,:libnd4j,:nd4j-native ${split_flags[@]+"${split_flags[@]}"} ${repo[@]+"${repo[@]}"} -Pcpu "-Dlibnd4j.buildthreads=${DL4J_BUILD_THREADS}" -Dhttp.keepAlive=false -Dmaven.wagon.http.pool=false -Dmaven.wagon.http.retryHandler.count=3 -DskipTestResourceEnforcement=true -Dmaven.javadoc.failOnError=false -Djavacpp.platform=linux-x86_64 -Pcpu ${also_make[@]+"${also_make[@]}"} --batch-mode -DskipTests -Dlibnd4j.extension=compat -Djavacpp.platform.extension=-compat -Dlibnd4j.classifier=linux-x86_64-compat "${DL4J_MAVEN_GOAL}")
    ;;
  zluda|windows-zluda)
    : "${DL4J_CUDA_VERSION:?DL4J_CUDA_VERSION is required}"
    : "${DL4J_PLATFORM_EXTENSION:?DL4J_PLATFORM_EXTENSION is required}"
    : "${DL4J_CLASSIFIER:?DL4J_CLASSIFIER is required}"
    : "${DL4J_ZLUDA_TARGET:=AMD}"
    rocm_version=()
    [ -z "${DL4J_ROCM_VERSION}" ] || rocm_version=("-Drocm.version=${DL4J_ROCM_VERSION}")
    platform=linux-x86_64
    zluda_profiles=(-Pcuda -Pzluda -Pzluda-platform)
    zluda_modules=:nd4j-cuda-backend-common,:nd4j-cuda-${DL4J_CUDA_VERSION}-preset,:nd4j-zluda-${DL4J_CUDA_VERSION},:nd4j-zluda-${DL4J_CUDA_VERSION}-platform
    zluda_win=()
    if [ "${DL4J_FAMILY}" = windows-zluda ]; then
      platform=windows-x86_64
      zluda_win=(-Dlibnd4j.platform=windows-x86_64 -Dlibnd4j.oom.killer=OFF)
    fi
    zluda_modules+=,:libnd4j
    modules=${zluda_modules}
    append_sdx_modules
    zluda_modules=${modules}
    # Let CMake resolve the ZLUDA virtual architecture from the selected ROCm
    # SDK. Passing a CUDA-only list here used to inject compute_90 into ROCm 7
    # builds, which is outside ZLUDA's supported virtual-architecture contract.
    command=(mvn ${split_flags[@]+"${split_flags[@]}"} ${repo[@]+"${repo[@]}"} "${zluda_profiles[@]}" "${sdx_profile[@]}" -Dlibnd4j.generate.flatc=ON -Dlibnd4j.oom.memory.threshold=95 -Dlibnd4j.oom.velocity.threshold=40 --no-transfer-progress -Dlibnd4j.cuda.compile.skip=false -Dlibnd4j.chip=cuda -Dlibnd4j.cpu.compile.skip=true "-Dlibnd4j.zluda=${DL4J_ZLUDA_TARGET}" "-Dlibnd4j.zluda.version=${DL4J_ZLUDA_VERSION}" "-Djavacpp.platform.extension=${DL4J_PLATFORM_EXTENSION}" "-Dlibnd4j.classifier=${DL4J_CLASSIFIER}" -Dhttp.keepAlive=false -Dmaven.wagon.http.pool=false -Dmaven.wagon.http.retryHandler.count=3 "-Dlibnd4j.buildthreads=${DL4J_BUILD_THREADS}" "-Djavacpp.platform=${platform}" ${rocm_version[@]+"${rocm_version[@]}"} ${zluda_win[@]+"${zluda_win[@]}"} --batch-mode -DskipTests "${sdx_maven_flags[@]}" -pl "${zluda_modules}" ${also_make[@]+"${also_make[@]}"} "${DL4J_MAVEN_GOAL}")
    ;;
  *) printf 'Unsupported DL4J_FAMILY=%s\n' "${DL4J_FAMILY}" >&2; exit 2;;
esac

case "${1:---run}" in
  --print) printf '%q ' "${command[@]}"; printf '\n';;
  --run) printf '+ '; printf '%q ' "${command[@]}"; printf '\n'; exec "${command[@]}";;
  *) printf 'Usage: %s [--print|--run]\n' "$0" >&2; exit 2;;
esac
