#!/usr/bin/env bash
set -Eeuo pipefail

shard=${1:?shard is required}
ndk_version=${2:-}
variant=${3:-}
toolchain_root=${DL4J_TOOLCHAIN_ROOT:-/opt}

as_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

install_linux_packages() {
  if command -v apt-get >/dev/null 2>&1; then
    as_root env DEBIAN_FRONTEND=noninteractive apt-get update
    as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      autoconf automake build-essential ca-certificates ccache cmake curl gfortran git \
      gnupg jq libdwarf-dev libdw-dev libelf-dev libgomp1 libomp-dev libopenblas-dev \
      libssl-dev libtool libusb-1.0-0-dev libvulkan-dev libvulkan1 maven mesa-vulkan-drivers \
      nasm ninja-build openjdk-11-jdk pinentry-curses pkg-config python3 python3-pip \
      swig tar unzip vulkan-tools wget xz-utils zip zlib1g-dev
    as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y llvm-18-dev mlir-18-tools ||
      as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y llvm-dev libmlir-dev mlir-tools ||
      true
  elif command -v dnf >/dev/null 2>&1; then
    as_root dnf install -y dnf-plugins-core epel-release
    as_root dnf config-manager --set-enabled powertools ||
      as_root dnf config-manager --set-enabled crb
    as_root dnf groupinstall -y "Development Tools"
    as_root dnf install -y \
      autoconf automake ca-certificates ccache cmake curl findutils gcc-gfortran git \
      java-11-openjdk-devel jq libtool libusb-devel libusbx-devel make maven nasm \
      ninja-build openblas-devel openssl-devel patch pkgconfig python3 python3.11 swig tar unzip wget which xz \
      zip zlib-devel
  else
    printf 'Unsupported Linux package manager\n' >&2
    exit 2
  fi
}

install_macos_packages() {
  brew install \
    autoconf automake autoconf-archive cmake gcc gnu-sed libomp libtool libusb \
    maven nasm ninja openblas pkg-config rust swig wget xz
  printf '%s\n' "$(brew --prefix)/opt/gnu-sed/libexec/gnubin" >>"${GITHUB_PATH}"
}

ensure_modern_maven() {
  maven_version=3.9.9
  target="${toolchain_root}/apache-maven-${maven_version}"
  if [ ! -x "${target}/bin/mvn" ]; then
    work=$(mktemp -d)
    trap 'rm -rf "${work}"' RETURN
    maven_archive="apache-maven-${maven_version}-bin.tar.gz"
    maven_sha512=a555254d6b53d267965a3404ecb14e53c3827c09c3b94b5678835887ab404556bfaf78dcfe03ba76fa2508649dca8531c74bca4d5846513522404d48e8c4ac8b
    maven_downloaded=0
    for maven_url in \
      "https://repo.maven.apache.org/maven2/org/apache/maven/apache-maven/${maven_version}/${maven_archive}" \
      "https://dlcdn.apache.org/maven/maven-3/${maven_version}/binaries/${maven_archive}" \
      "https://archive.apache.org/dist/maven/maven-3/${maven_version}/binaries/${maven_archive}"; do
      if curl --fail --location --retry 5 --retry-all-errors \
          --connect-timeout 20 --max-time 300 \
          "${maven_url}" -o "${work}/maven.tar.gz"; then
        maven_downloaded=1
        break
      fi
    done
    [ "${maven_downloaded}" -eq 1 ] || {
      printf 'Failed to download Maven %s from all configured mirrors\n' "${maven_version}" >&2
      return 1
    }
    if command -v sha512sum >/dev/null 2>&1; then
      printf '%s  %s\n' "${maven_sha512}" "${work}/maven.tar.gz" | sha512sum --check --status
    else
      printf '%s  %s\n' "${maven_sha512}" "${work}/maven.tar.gz" | shasum -a 512 --check --status
    fi
    as_root tar -xzf "${work}/maven.tar.gz" -C "${toolchain_root}"
    trap - RETURN
    rm -rf "${work}"
  fi
  "${target}/bin/mvn" --version
  printf '%s\n' "${target}/bin" >>"${GITHUB_PATH}"
}

ensure_protobuf() {
  if [ "$(uname -s)" != Linux ] || [ -x "${toolchain_root}/protobuf/bin/protoc" ]; then
    return
  fi
  work=$(mktemp -d)
  trap 'rm -rf "${work}"' RETURN
  curl --fail --location --retry 5 \
    https://github.com/google/protobuf/releases/download/v3.8.0/protobuf-cpp-3.8.0.tar.gz \
    -o "${work}/protobuf.tar.gz"
  tar -xzf "${work}/protobuf.tar.gz" -C "${work}"
  (
    cd "${work}/protobuf-3.8.0"
    ./configure --prefix="${toolchain_root}/protobuf"
    make -j2
    as_root make install
  )
  trap - RETURN
  rm -rf "${work}"
}

ensure_protoc_21() {
  if [ -x "${toolchain_root}/protoc-21.7/bin/protoc" ]; then
    return
  fi
  case "$(uname -s):$(uname -m)" in
    Linux:aarch64|Linux:arm64) protoc_target=linux-aarch_64 ;;
    Linux:*) protoc_target=linux-x86_64 ;;
    Darwin:arm64) protoc_target=osx-aarch_64 ;;
    Darwin:*) protoc_target=osx-x86_64 ;;
    *) return ;;
  esac
  work=$(mktemp -d)
  trap 'rm -rf "${work}"' RETURN
  curl --fail --location --retry 5 \
    "https://github.com/protocolbuffers/protobuf/releases/download/v21.7/protoc-21.7-${protoc_target}.zip" \
    -o "${work}/protoc.zip"
  as_root mkdir -p "${toolchain_root}/protoc-21.7"
  as_root unzip -qo "${work}/protoc.zip" bin/protoc -d "${toolchain_root}/protoc-21.7"
  as_root chmod +x "${toolchain_root}/protoc-21.7/bin/protoc"
  "${toolchain_root}/protoc-21.7/bin/protoc" --version
  trap - RETURN
  rm -rf "${work}"
}

ensure_android_ndk() {
  [ -n "${ndk_version}" ] || return 0
  target="${toolchain_root}/android/android-ndk-${ndk_version}"
  if [ ! -d "${target}" ]; then
    work=$(mktemp -d)
    trap 'rm -rf "${work}"' RETURN
    curl --fail --location --retry 5 \
      "https://dl.google.com/android/repository/android-ndk-${ndk_version}-linux.zip" \
      -o "${work}/android-ndk.zip"
    as_root mkdir -p "${toolchain_root}/android"
    as_root unzip -q "${work}/android-ndk.zip" -d "${toolchain_root}/android"
    trap - RETURN
    rm -rf "${work}"
  fi
  printf 'ANDROID_NDK=%s\n' "${target}" >>"${GITHUB_ENV}"
  printf 'ANDROID_NDK_HOME=%s\n' "${target}" >>"${GITHUB_ENV}"
}

ensure_rust_toolchain() {
  command -v cargo >/dev/null 2>&1 && return 0

  work=$(mktemp -d)
  trap 'rm -rf "${work}"' RETURN
  curl --proto '=https' --tlsv1.2 --fail --silent --show-error --location \
    --retry 5 --retry-all-errors --connect-timeout 20 --max-time 300 \
    https://sh.rustup.rs -o "${work}/rustup-init.sh"
  sh "${work}/rustup-init.sh" -y --default-toolchain stable
  export PATH="${HOME}/.cargo/bin:${PATH}"
  printf '%s\n' "${HOME}/.cargo/bin" >>"${GITHUB_PATH}"
  trap - RETURN
  rm -rf "${work}"
}

ensure_cuda_sbsa_cross() {
  [ "${shard}" = linux-arm64-cuda-13-1 ] || return 0
  case "$(uname -m)" in
    x86_64|amd64) ;;
    *) return 0 ;;
  esac

  as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

  work=$(mktemp -d)
  trap 'rm -rf "${work}"' RETURN
  curl --fail --location --retry 5 --retry-all-errors \
    --connect-timeout 20 --max-time 300 \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/cross-linux-sbsa/cuda-keyring_1.1-1_all.deb \
    -o "${work}/cuda-keyring.deb"
  as_root dpkg -i "${work}/cuda-keyring.deb"
  as_root env DEBIAN_FRONTEND=noninteractive apt-get update
  as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    cuda-cross-sbsa-13-1
  if [ "${variant}" = cudnn ]; then
    as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      cudnn9-cross-sbsa=9.19.1-1 libcudnn9-cross-sbsa-cuda-13=9.19.1.2-1
  fi
  trap - RETURN
  rm -rf "${work}"

  sbsa_root=/usr/local/cuda/targets/sbsa-linux
  [ -d "${sbsa_root}/include" ] && [ -d "${sbsa_root}/lib" ] || {
    printf 'CUDA 13.1 SBSA cross target is incomplete: %s\n' "${sbsa_root}" >&2
    return 1
  }
  printf 'CUDA_SBSA_TARGET_ROOT=%s\n' "${sbsa_root}" >>"${GITHUB_ENV}"
  printf 'LIBRARY_PATH=%s:%s\n' "${sbsa_root}/lib" "${LIBRARY_PATH:-}" >>"${GITHUB_ENV}"
  printf 'CUDNN_ROOT=%s\n' "${sbsa_root}" >>"${GITHUB_ENV}"

  target_jdk="${toolchain_root}/temurin-11-aarch64"
  if [ ! -f "${target_jdk}/include/jni.h" ] || [ ! -f "${target_jdk}/lib/server/libjvm.so" ]; then
    jdk_archive=OpenJDK11U-jdk_aarch64_linux_hotspot_11.0.32.1_1.tar.gz
    jdk_sha256=f27033e6f7523c1b0b2565a78e9c0e0abe5596a854ce00ca04ec1b06ece7a935
    work=$(mktemp -d)
    trap 'rm -rf "${work}"' RETURN
    curl --fail --location --retry 5 --retry-all-errors \
      --connect-timeout 20 --max-time 300 \
      "https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.32.1%2B1/${jdk_archive}" \
      -o "${work}/${jdk_archive}"
    printf '%s  %s\n' "${jdk_sha256}" "${work}/${jdk_archive}" | sha256sum --check --status
    as_root mkdir -p "${target_jdk}"
    as_root tar -xzf "${work}/${jdk_archive}" -C "${target_jdk}" --strip-components=1
    trap - RETURN
    rm -rf "${work}"
  fi
  printf 'JAVA_AARCH64_HOME=%s\n' "${target_jdk}" >>"${GITHUB_ENV}"
}

case "$(uname -s)" in
  Linux)
    install_linux_packages
    ensure_modern_maven
    ensure_protobuf
    ensure_protoc_21
    ensure_android_ndk
    ensure_cuda_sbsa_cross
    printf '%s\n' "${toolchain_root}/protobuf/bin" "${toolchain_root}/protoc-21.7/bin" >>"${GITHUB_PATH}"
    ;;
  Darwin)
    install_macos_packages
    ensure_protoc_21
    printf '%s\n' "${toolchain_root}/protoc-21.7/bin" >>"${GITHUB_PATH}"
    ;;
  *)
    printf 'Unsupported Unix host: %s\n' "$(uname -s)" >&2
    exit 2
    ;;
esac

ensure_rust_toolchain
if ! command -v cbindgen >/dev/null 2>&1; then
  cargo install --locked cbindgen
fi

printf '[dl4j-bootstrap] shard=%s ndk=%s status=complete\n' "${shard}" "${ndk_version:-none}"
