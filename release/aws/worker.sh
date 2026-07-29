#!/usr/bin/env bash
set -Eeuo pipefail

CONFIG_B64='__DL4J_WORKER_CONFIG_B64__'
BUILD_DRIVER_B64='__DL4J_BUILD_DRIVER_B64__'
LOG_FORWARDER_B64='__DL4J_LOG_FORWARDER_B64__'
CONFIG_FILE=/tmp/dl4j-release-worker.json
BUILD_DRIVER=/tmp/dl4j-build-platform.py
LOG_FORWARDER=/tmp/dl4j-log-forwarder.py
LOG_FORWARDER_STOP=/tmp/dl4j-log-forwarder.stop
LOG_FORWARDER_ERROR=/tmp/dl4j-log-forwarder.err
WORK_ROOT=${DL4J_WORK_ROOT:-/opt/dl4j-release}
SOURCE_DIR=${WORK_ROOT}/source
OUTPUT_DIR=${WORK_ROOT}/output
MAVEN_REPO=${WORK_ROOT}/m2
BUILD_LOG=${OUTPUT_DIR}/build.log
BUILD_PID_FILE=/tmp/dl4j-release-build.pid
WATCHDOG_PID=""
LOG_FORWARDER_PID=""
mkdir -p "${OUTPUT_DIR}" "${MAVEN_REPO}"
decode_b64() {
  if [ "$(uname -s)" = Darwin ]; then
    printf '%s' "$1" | base64 -D > "$2"
  else
    printf '%s' "$1" | base64 --decode > "$2"
  fi
}
decode_b64 "${CONFIG_B64}" "${CONFIG_FILE}"
decode_b64 "${BUILD_DRIVER_B64}" "${BUILD_DRIVER}"
decode_b64 "${LOG_FORWARDER_B64}" "${LOG_FORWARDER}"
exec > >(tee -a "${BUILD_LOG}") 2>&1

phase() {
  printf '[dl4j-phase] timestamp=%s phase=%s status=%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "${3:-}"
}
trap 'phase bootstrap failed "line=${LINENO} command=${BASH_COMMAND}"' ERR
phase worker started "pid=$$"

config() {
  python3 -c 'import json,sys; value=json.load(open(sys.argv[1]));
for part in sys.argv[2].split("."): value=value[part]
print(json.dumps(value) if isinstance(value,(dict,list)) else value)' "${CONFIG_FILE}" "$1"
}

REGION=$(config region)
BUCKET=$(config bucket)
ARTIFACT_PREFIX=$(config artifactPrefix)
RUN_ID=$(config runId)
SHARD_ID=$(config shard.id)
RELEASE_VERSION=$(config releaseVersion)
COMMIT=$(config commit)
REPOSITORY=$(config repository)
KILL_SWITCH_PARAMETER=$(config killSwitchParameter)
LOG_GROUP=$(config logGroupName)
LOG_STREAM=$(config logStreamName)
OS_NAME=$(config shard.os)
PLATFORM=$(config shard.build.javacppPlatform)
BACKEND=$(config shard.build.backend)
CONTAINER_IMAGE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["shard"].get("containerImage", ""))' "${CONFIG_FILE}")
CONTAINER_FAMILY=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["shard"].get("containerFamily", "debian"))' "${CONFIG_FILE}")
S3_PREFIX="s3://${BUCKET}/${ARTIFACT_PREFIX}/${RUN_ID}/${SHARD_ID}"

upload_if_present() {
  [ -e "$1" ] && aws --region "${REGION}" s3 cp "$1" "${S3_PREFIX}/$2" --only-show-errors || true
}

start_log_forwarder() {
  [ -z "${LOG_FORWARDER_PID}" ] || return 0
  phase cloudwatch-forwarder started
  python3 "${LOG_FORWARDER}" --file "${BUILD_LOG}" --stop-file "${LOG_FORWARDER_STOP}" \
    --region "${REGION}" --group "${LOG_GROUP}" --stream "${LOG_STREAM}" \
    >"${LOG_FORWARDER_ERROR}" 2>&1 &
  LOG_FORWARDER_PID=$!
  phase cloudwatch-forwarder complete "group=${LOG_GROUP} stream=${LOG_STREAM} pid=${LOG_FORWARDER_PID}"
}

finish() {
  local exit_code=$?
  set +e
  phase finalize started "exitCode=${exit_code}"
  [ -n "${WATCHDOG_PID}" ] && kill "${WATCHDOG_PID}" 2>/dev/null
  touch "${LOG_FORWARDER_STOP}"
  if [ -n "${LOG_FORWARDER_PID}" ]; then
    for _ in $(seq 1 20); do
      kill -0 "${LOG_FORWARDER_PID}" 2>/dev/null || break
      sleep 2
    done
    kill "${LOG_FORWARDER_PID}" 2>/dev/null || true
    wait "${LOG_FORWARDER_PID}" 2>/dev/null || true
  fi
  [ -s "${LOG_FORWARDER_ERROR}" ] && { printf '\nCloudWatch forwarder diagnostics:\n'; sed 's/^/[cloudwatch] /' "${LOG_FORWARDER_ERROR}"; }
  python3 -c 'import json,sys,time; json.dump({"shard":sys.argv[1],"exitCode":int(sys.argv[2]),"completedAt":int(time.time())},open(sys.argv[3],"w"),sort_keys=True)' "${SHARD_ID}" "${exit_code}" "${OUTPUT_DIR}/status.json"
  upload_if_present "${BUILD_LOG}" build.log
  upload_if_present "${OUTPUT_DIR}/status.json" status.json
  upload_if_present "${OUTPUT_DIR}/maven-repository.tar.gz" maven-repository.tar.gz
  upload_if_present "${OUTPUT_DIR}/sdk-assets.tar.gz" sdk-assets.tar.gz
  upload_if_present "${OUTPUT_DIR}/shard-manifest.json" shard-manifest.json
  sync
  shutdown -h now || true
  exit "${exit_code}"
}
trap finish EXIT

if [ "${OS_NAME}" = "linux" ]; then
  export DEBIAN_FRONTEND=noninteractive
  phase package-index started
  apt-get update
  phase package-index complete
  phase logging-prerequisites started
  apt-get install -y --no-install-recommends awscli ca-certificates python3
  phase logging-prerequisites complete
  start_log_forwarder
  phase toolchain-packages started
  apt-get install -y --no-install-recommends autoconf automake build-essential ccache cmake curl docker.io gfortran git gnupg jq libdwarf-dev libdw-dev libelf-dev libgomp1 libomp-dev libopenblas-dev libtool libusb-1.0-0-dev libvulkan-dev libvulkan1 maven mesa-vulkan-drivers nasm ninja-build openjdk-11-jdk pinentry-curses pkg-config swig tar unzip vulkan-tools wget zip zlib1g-dev
  apt-get install -y llvm-18-dev mlir-18-tools || apt-get install -y llvm-dev libmlir-dev mlir-tools || true
  phase toolchain-packages complete
  export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-$(dpkg --print-architecture)
  phase protobuf-toolchain started
  curl --fail --location --retry 5 https://github.com/google/protobuf/releases/download/v3.8.0/protobuf-cpp-3.8.0.tar.gz -o /tmp/protobuf-3.8.0.tar.gz
  tar -xzf /tmp/protobuf-3.8.0.tar.gz -C /tmp
  (cd /tmp/protobuf-3.8.0 && ./configure --prefix=/opt/protobuf && make -j2 && make install)
  export PATH="/opt/protobuf/bin:${PATH}"
  PROTOC_ARCH=$([ "$(uname -m)" = aarch64 ] && printf aarch_64 || printf x86_64)
  curl --fail --location --retry 5 "https://github.com/protocolbuffers/protobuf/releases/download/v21.7/protoc-21.7-linux-${PROTOC_ARCH}.zip" -o /tmp/protoc.zip
  unzip -qo /tmp/protoc.zip -d /opt/protoc-21.7 bin/protoc
  chmod +x /opt/protoc-21.7/bin/protoc
  phase protobuf-toolchain complete
  if [ "$(uname -m)" = x86_64 ]; then
    curl --fail --location --retry 5 https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.tar.gz -o /tmp/cmake.tar.gz
    mkdir -p /opt/cmake && tar -xzf /tmp/cmake.tar.gz -C /opt/cmake --strip-components=1
    export PATH="/opt/cmake/bin:${PATH}"
  fi
  phase rust-toolchain started
  curl --proto '=https' --tlsv1.2 --fail --silent --show-error https://sh.rustup.rs | sh -s -- -y --profile minimal
  export PATH="${HOME}/.cargo/bin:${PATH}"
  cargo install --locked cbindgen
  phase rust-toolchain complete
  if [[ "${PLATFORM}" == android-* ]]; then
    NDK_VERSION=$(config shard.build.ndkVersion)
    curl --fail --location --retry 5 "https://dl.google.com/android/repository/android-ndk-${NDK_VERSION}-linux.zip" -o /tmp/android-ndk.zip
    unzip -q /tmp/android-ndk.zip -d /opt/android
    export ANDROID_NDK="/opt/android/android-ndk-${NDK_VERSION}" ANDROID_NDK_HOME="/opt/android/android-ndk-${NDK_VERSION}"
  fi
  if python3 -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["shard"]["build"].get("buildAot") else 1)' "${CONFIG_FILE}"; then
    curl --fail --location --retry 5 https://download.oracle.com/graalvm/21/latest/graalvm-jdk-21_linux-x64_bin.tar.gz -o /tmp/graalvm.tar.gz
    mkdir -p /opt/graalvm
    tar -xzf /tmp/graalvm.tar.gz -C /opt/graalvm --strip-components=1
    export GRAALVM_HOME=/opt/graalvm
  fi
else
  phase macos-toolchain started
  export PATH="/opt/homebrew/bin:/usr/local/bin:${PATH}"
  if ! command -v brew >/dev/null 2>&1; then
    NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    export PATH="/opt/homebrew/bin:${PATH}"
  fi
  phase logging-prerequisites started
  brew install awscli python@3.12
  export PATH="$(brew --prefix python@3.12)/libexec/bin:${PATH}"
  phase logging-prerequisites complete
  start_log_forwarder
  brew install ant autoconf autoconf-archive automake binutils bison ccache cmake flex gcc git gmp gnu-sed gradle isl jq libmpc libomp libtool libusb llvm maven mpfr nasm ninja openblas openjdk@11 perl pkg-config ragel rust sdl swig unzip wget xz || true
  export JAVA_HOME="$(brew --prefix openjdk@11)/libexec/openjdk.jdk/Contents/Home"
  export PATH="$(brew --prefix gnu-sed)/libexec/gnubin:${JAVA_HOME}/bin:${PATH}"
  curl --fail --location --retry 5 https://github.com/protocolbuffers/protobuf/releases/download/v21.7/protoc-21.7-osx-aarch_64.zip -o /tmp/protoc.zip
  unzip -qo /tmp/protoc.zip -d /usr/local bin/protoc
  chmod +x /usr/local/bin/protoc
  cargo install --locked cbindgen
  phase macos-toolchain complete
fi

watch_kill_switch() {
  while true; do
    value=$(aws --region "${REGION}" ssm get-parameter --name "${KILL_SWITCH_PARAMETER}" --query 'Parameter.Value' --output text 2>/dev/null || printf true)
    if [ "${value}" = true ]; then
      printf 'Global release kill switch enabled; stopping %s.\n' "${SHARD_ID}"
      current_pid=$([ -f "${BUILD_PID_FILE}" ] && tr -dc '0-9' < "${BUILD_PID_FILE}" || true)
      [ -n "${current_pid}" ] && kill -TERM -- "-${current_pid}" 2>/dev/null
      sleep 5
      [ -n "${current_pid}" ] && kill -KILL -- "-${current_pid}" 2>/dev/null
      shutdown -h now || true
      return
    fi
    sleep 15
  done
}
watch_kill_switch &
WATCHDOG_PID=$!

kill_value=$(aws --region "${REGION}" ssm get-parameter --name "${KILL_SWITCH_PARAMETER}" --query 'Parameter.Value' --output text)
[ "${kill_value}" != true ] || exit 130

phase source-checkout started "commit=${COMMIT}"
git clone --filter=blob:none "${REPOSITORY}" "${SOURCE_DIR}"
git -C "${SOURCE_DIR}" fetch --depth=1 origin "${COMMIT}"
git -C "${SOURCE_DIR}" checkout --detach "${COMMIT}"
[ "$(git -C "${SOURCE_DIR}" rev-parse HEAD)" = "${COMMIT}" ] || exit 2
phase source-checkout complete "commit=${COMMIT}"
mkdir -p "${OUTPUT_DIR}/maven-repository" "${OUTPUT_DIR}/sdk-assets"

build=(python3 "${BUILD_DRIVER}" --config "${CONFIG_FILE}" --source "${SOURCE_DIR}" --repository "${MAVEN_REPO}" --maven-output "${OUTPUT_DIR}/maven-repository" --sdk-output "${OUTPUT_DIR}/sdk-assets")
if [ -n "${CONTAINER_IMAGE}" ]; then
  if [ "${CONTAINER_FAMILY}" = almalinux ]; then
    docker build --tag dl4j-release-compat --file "${SOURCE_DIR}/.github/actions/build-centos/Dockerfile" "${SOURCE_DIR}/.github/actions/build-centos"
    build=(docker run --rm --network host -e GITHUB_WORKSPACE=/github/workspace -e LIBND4J_FILE_NAME= -e LIBND4J_URL= -e OPENBLAS_PATH=/usr -e "INSTALL_COMMAND=python3 /dl4j-build-platform.py --config /dl4j-config.json --source /github/workspace --repository /dl4j-m2 --maven-output /dl4j-output/maven-repository --sdk-output /dl4j-output/sdk-assets" -v "${SOURCE_DIR}:/github/workspace" -v "${MAVEN_REPO}:/dl4j-m2" -v "${OUTPUT_DIR}:/dl4j-output" -v "${CONFIG_FILE}:/dl4j-config.json:ro" -v "${BUILD_DRIVER}:/dl4j-build-platform.py:ro" dl4j-release-compat)
  else
    build=(docker run --rm --network host -v "${SOURCE_DIR}:/workspace" -v "${MAVEN_REPO}:/dl4j-m2" -v "${OUTPUT_DIR}:/dl4j-output" -v "${CONFIG_FILE}:/dl4j-config.json:ro" -v "${BUILD_DRIVER}:/dl4j-build-platform.py:ro" -v /opt/protobuf:/opt/protobuf:ro -v /opt/cmake:/opt/cmake:ro -w /workspace "${CONTAINER_IMAGE}" bash -lc "apt-get update && apt-get install -y --no-install-recommends autoconf automake build-essential ca-certificates cmake gfortran git libomp-dev libopenblas-dev libtool maven nasm ninja-build openjdk-11-jdk pkg-config python3 swig unzip xz-utils zip && export PATH=/opt/protobuf/bin:/opt/cmake/bin:\$PATH && python3 /dl4j-build-platform.py --config /dl4j-config.json --source /workspace --repository /dl4j-m2 --maven-output /dl4j-output/maven-repository --sdk-output /dl4j-output/sdk-assets")
  fi
fi
phase matrix-build started
setsid "${build[@]}" &
BUILD_PID=$!
printf '%s\n' "${BUILD_PID}" > "${BUILD_PID_FILE}"
wait "${BUILD_PID}"
rm -f "${BUILD_PID_FILE}"
phase matrix-build complete
phase artifact-packaging started

python3 -c 'import hashlib,json,pathlib,sys; root=pathlib.Path(sys.argv[1]); files=[]
for p in sorted(x for x in root.rglob("*") if x.is_file()):
 h=hashlib.sha256();
 with p.open("rb") as f:
  for chunk in iter(lambda:f.read(1048576),b""): h.update(chunk)
 files.append({"path":p.relative_to(root).as_posix(),"sha256":h.hexdigest(),"size":p.stat().st_size})
c=json.load(open(sys.argv[2])); json.dump({"schemaVersion":1,"runId":c["runId"],"shard":c["shard"]["id"],"commit":c["commit"],"releaseVersion":c["releaseVersion"],"workloads":c["shard"]["workloads"],"os":c["shard"]["os"],"platform":c["shard"]["build"]["javacppPlatform"],"backend":c["shard"]["build"]["backend"],"files":files},open(root/"shard-manifest.json","w"),indent=2,sort_keys=True)' "${OUTPUT_DIR}" "${CONFIG_FILE}"
tar -C "${OUTPUT_DIR}/maven-repository" -czf "${OUTPUT_DIR}/maven-repository.tar.gz" .
tar -C "${OUTPUT_DIR}/sdk-assets" -czf "${OUTPUT_DIR}/sdk-assets.tar.gz" .
phase artifact-packaging complete
