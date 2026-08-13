#!/usr/bin/env bash
set -Eeuo pipefail

# Azure Linux custom-data workers run as root/non-login systemd services.
export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
export HOME="${HOME:-/root}"
export USER="${USER:-$(id -un)}"
export LOGNAME="${LOGNAME:-${USER}}"

CONFIG_B64='__DL4J_WORKER_CONFIG_B64__'
BUILD_DRIVER_B64='__DL4J_BUILD_DRIVER_B64__'
CLOUD_IO_B64='__DL4J_CLOUD_IO_B64__'
DEPENDENCY_CACHE_B64='__DL4J_DEPENDENCY_CACHE_B64__'
WORK_ROOT=${DL4J_WORK_ROOT:-/opt/dl4j-release}
BOOTSTRAP_ROOT=${WORK_ROOT}/bootstrap
CONFIG_FILE=${BOOTSTRAP_ROOT}/worker.json
BUILD_DRIVER=${BOOTSTRAP_ROOT}/build-platform.py
CLOUD_IO=${BOOTSTRAP_ROOT}/cloud-io.py
DEPENDENCY_CACHE=${BOOTSTRAP_ROOT}/dependency-cache.py
SOURCE_ROOT=${WORK_ROOT}/sources
SCCACHE_ROOT=${SOURCE_ROOT}/sccache
OUTPUT_ROOT=${WORK_ROOT}/outputs
MAVEN_REPO_ROOT=${WORK_ROOT}/m2
TOOLCHAIN_ROOT=${WORK_ROOT}/toolchains
LANE_LOG=${WORK_ROOT}/lane.log
LANE_FORWARDER_STOP=${WORK_ROOT}/lane-forwarder.stop
LANE_FORWARDER_ERROR=${WORK_ROOT}/lane-forwarder.err
BUILD_PID_FILE=${WORK_ROOT}/build.pid
MAIN_PID=$$
WATCHDOG_PID=""
LANE_FORWARDER_PID=""
LOG_FORWARDER_PID=""
CURRENT_ACTIVE=0
CURRENT_FINALIZED=1
CURRENT_SHARD_ID=""
CURRENT_CONFIG_FILE=""
CURRENT_OUTPUT_DIR=""
CURRENT_BUILD_LOG=""
CURRENT_OBJECT_PREFIX=""
CURRENT_LOG_STOP=""
CURRENT_LOG_ERROR=""

export CARGO_HOME="${CARGO_HOME:-${TOOLCHAIN_ROOT}/cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-${TOOLCHAIN_ROOT}/rustup}"
export PATH="${CARGO_HOME}/bin:${PATH}"

mkdir -p "${BOOTSTRAP_ROOT}" "${SOURCE_ROOT}" "${SCCACHE_ROOT}" "${OUTPUT_ROOT}" "${MAVEN_REPO_ROOT}"   "${CARGO_HOME}" "${RUSTUP_HOME}" "${TOOLCHAIN_ROOT}"
decode_b64() { printf '%s' "$1" | base64 --decode > "$2"; }
decode_b64 "${CONFIG_B64}" "${CONFIG_FILE}"
decode_b64 "${BUILD_DRIVER_B64}" "${BUILD_DRIVER}"
decode_b64 "${CLOUD_IO_B64}" "${CLOUD_IO}"
decode_b64 "${DEPENDENCY_CACHE_B64}" "${DEPENDENCY_CACHE}"
exec > >(tee -a "${LANE_LOG}") 2>&1

phase() {
  printf '[dl4j-phase] timestamp=%s phase=%s status=%s %s\n'     "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "${3:-}"
}

config() {
  python3 -c 'import json,sys; value=json.load(open(sys.argv[1]));
for part in sys.argv[2].split("."): value=value[part]
print(json.dumps(value) if isinstance(value,(dict,list)) else value)' "${CONFIG_FILE}" "$1"
}

BUCKET=$(config bucket)
KILL_SWITCH_BUCKET=$(config killSwitchBucket)
ARTIFACT_PREFIX=$(config artifactPrefix)
RUN_ID=$(config runId)
LANE_ID=$(config laneId)
COMMIT=$(config commit)
REPOSITORY=$(config repository)
MAVEN_REPOSITORY_PREFIX=$(config mavenRepositoryPrefix)
KILL_SWITCH_OBJECT=$(config killSwitchObject)
RUN_KILL_SWITCH_OBJECT=$(config runKillSwitchObject)
AZURE_CLIENT_ID=$(config managedIdentityClientId)
CONTROLLER_EPOCH=$(config controllerEpoch)
TOOLCHAIN_CACHE_BUCKET=$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); v=c.get("compilerCache",{}); print((str(v.get("account","")) + "/" + str(v.get("container",""))).strip("/"))' "${CONFIG_FILE}")
TOOLCHAIN_CACHE_PREFIX=$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c.get("compilerCache",{}).get("toolchainCache",{}).get("keyPrefix",""))' "${CONFIG_FILE}")
export AZURE_CLIENT_ID
export DL4J_CLOUD_IO="${CLOUD_IO}"
export DL4J_DEPENDENCY_CACHE_HELPER="${DEPENDENCY_CACHE}"
export DL4J_TOOLCHAIN_CACHE_BUCKET="${TOOLCHAIN_CACHE_BUCKET}"
export DL4J_TOOLCHAIN_CACHE_PREFIX="${TOOLCHAIN_CACHE_PREFIX}"

cache_identity() {
  python3 -c 'import hashlib,json,sys; print(hashlib.sha256(json.dumps(sys.argv[1:],separators=(",",":")).encode()).hexdigest())' "$@"
}

restore_dependency() {
  local name="$1" identity="$2" destination="$3" code
  if [ -z "${TOOLCHAIN_CACHE_BUCKET}" ] || [ -z "${TOOLCHAIN_CACHE_PREFIX}" ]; then
    return 1
  fi
  phase dependency-cache started "dependency=${name} identity=${identity} operation=restore"
  set +e
  python3 "${DEPENDENCY_CACHE}" restore     --cloud-io "${CLOUD_IO}"     --bucket "${TOOLCHAIN_CACHE_BUCKET}"     --prefix "${TOOLCHAIN_CACHE_PREFIX}"     --name "${name}"     --identity "${identity}"     --destination "${destination}"     --client-id "${AZURE_CLIENT_ID}"
  code=$?
  set -e
  if [ "${code}" -eq 0 ]; then
    phase dependency-cache hit "dependency=${name} identity=${identity}"
    return 0
  fi
  if [ "${code}" -eq 3 ]; then
    phase dependency-cache miss "dependency=${name} identity=${identity}"
    return 1
  fi
  phase dependency-cache failed "dependency=${name} identity=${identity} exitCode=${code}"
  exit "${code}"
}

publish_dependency() {
  local name="$1" identity="$2" source="$3"
  if [ -z "${TOOLCHAIN_CACHE_BUCKET}" ] || [ -z "${TOOLCHAIN_CACHE_PREFIX}" ]; then
    return 0
  fi
  phase dependency-cache started "dependency=${name} identity=${identity} operation=publish"
  python3 "${DEPENDENCY_CACHE}" publish     --cloud-io "${CLOUD_IO}"     --bucket "${TOOLCHAIN_CACHE_BUCKET}"     --prefix "${TOOLCHAIN_CACHE_PREFIX}"     --name "${name}"     --identity "${identity}"     --source "${source}"     --client-id "${AZURE_CLIENT_ID}"
  phase dependency-cache complete "dependency=${name} identity=${identity} operation=publish"
}

mapfile -t SHARD_IDS < <(
  python3 -c 'import json,sys; print(*[s["id"] for s in json.load(open(sys.argv[1]))["shards"]], sep="\n")' "${CONFIG_FILE}"
)
if [ "${#SHARD_IDS[@]}" -eq 0 ]; then
  phase worker failed "lane=${LANE_ID} reason=no-shards"
  exit 2
fi

upload_object() {
  python3 "${CLOUD_IO}" upload --bucket "${BUCKET}" --object "$1"     --file "$2" --client-id "${AZURE_CLIENT_ID}"
}

start_lane_forwarder() {
  rm -f "${LANE_FORWARDER_STOP}" "${LANE_FORWARDER_ERROR}"
  python3 "${CLOUD_IO}" forward --bucket "${BUCKET}"     --object "${ARTIFACT_PREFIX}/${RUN_ID}/lanes/${LANE_ID}/live.log"     --file "${LANE_LOG}" --stop-file "${LANE_FORWARDER_STOP}"     --client-id "${AZURE_CLIENT_ID}" >"${LANE_FORWARDER_ERROR}" 2>&1 &
  LANE_FORWARDER_PID=$!
}

stop_forwarder() {
  local pid="$1"
  local stop_file="$2"
  [ -n "${pid}" ] || return 0
  touch "${stop_file}"
  for _ in $(seq 1 20); do
    kill -0 "${pid}" 2>/dev/null || break
    sleep 2
  done
  kill "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
}

prepare_shard_context() {
  CURRENT_SHARD_ID="$1"
  local safe_id
  safe_id=$(printf '%s' "${CURRENT_SHARD_ID}" | tr -c 'A-Za-z0-9._-' '-')
  CURRENT_CONFIG_FILE="${BOOTSTRAP_ROOT}/${safe_id}.json"
  CURRENT_OUTPUT_DIR="${OUTPUT_ROOT}/${safe_id}"
  CURRENT_BUILD_LOG="${CURRENT_OUTPUT_DIR}/build.log"
  CURRENT_OBJECT_PREFIX="${ARTIFACT_PREFIX}/${RUN_ID}/${CURRENT_SHARD_ID}"
  CURRENT_LOG_STOP="${CURRENT_OUTPUT_DIR}/log-forwarder.stop"
  CURRENT_LOG_ERROR="${CURRENT_OUTPUT_DIR}/log-forwarder.err"
  mkdir -p "${CURRENT_OUTPUT_DIR}"
  local path
  shopt -s dotglob nullglob
  for path in "${CURRENT_OUTPUT_DIR}"/*; do
    [ "$(basename "${path}")" = build.log ] || rm -rf "${path}"
  done
  shopt -u dotglob nullglob
  if [ -s "${CURRENT_BUILD_LOG}" ]; then
    phase worker-restart started "attempt-log-preserved shard=${CURRENT_SHARD_ID}"
  fi
  python3 - "${CONFIG_FILE}" "${CURRENT_CONFIG_FILE}" "${CURRENT_SHARD_ID}" <<'PY'
import json
import sys
source, destination, shard_id = sys.argv[1:]
config = json.load(open(source, encoding="utf-8"))
shard = next(item for item in config["shards"] if item["id"] == shard_id)
config["shard"] = shard
config.pop("shards", None)
with open(destination, "w", encoding="utf-8") as stream:
    json.dump(config, stream, indent=2, sort_keys=True)
PY
}

start_shard_forwarder() {
  rm -f "${CURRENT_LOG_STOP}" "${CURRENT_LOG_ERROR}"
  touch "${CURRENT_BUILD_LOG}"
  python3 "${CLOUD_IO}" forward --bucket "${BUCKET}"     --object "${CURRENT_OBJECT_PREFIX}/live.log"     --file "${CURRENT_BUILD_LOG}" --stop-file "${CURRENT_LOG_STOP}"     --client-id "${AZURE_CLIENT_ID}" >"${CURRENT_LOG_ERROR}" 2>&1 &
  LOG_FORWARDER_PID=$!
}

remote_shard_succeeded() {
  local shard_id="$1"
  local safe_id status_file
  safe_id=$(printf '%s' "${shard_id}" | tr -c 'A-Za-z0-9._-' '-')
  status_file="${BOOTSTRAP_ROOT}/remote-${safe_id}.json"
  rm -f "${status_file}"
  if ! python3 "${CLOUD_IO}" download --bucket "${BUCKET}"       --object "${ARTIFACT_PREFIX}/${RUN_ID}/${shard_id}/status.json"       --file "${status_file}" --client-id "${AZURE_CLIENT_ID}" >/dev/null 2>&1; then
    return 1
  fi
  python3 - "${status_file}" "${CONFIG_FILE}" "${shard_id}" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
config = json.load(open(sys.argv[2], encoding="utf-8"))
shard = next(item for item in config["shards"] if item["id"] == sys.argv[3])
valid = (
    value.get("shard") == shard["id"]
    and value.get("runId") == config["runId"]
    and value.get("controllerEpoch") == config["controllerEpoch"]
    and value.get("repository") == config["repository"]
    and value.get("commit") == config["commit"]
    and value.get("releaseVersion") == config["releaseVersion"]
    and value.get("snapshotVersion") == config["snapshotVersion"]
    and value.get("contractDigest") == shard["contractDigest"]
    and value.get("variants") == [item["name"] for item in shard["build"]["variants"]]
    and int(value.get("exitCode", 1)) == 0
)
raise SystemExit(0 if valid else 1)
PY
}

finalize_shard() {
  local requested_code="$1"
  local final_code="${requested_code}"
  set +e
  stop_forwarder "${LOG_FORWARDER_PID}" "${CURRENT_LOG_STOP}"
  LOG_FORWARDER_PID=""
  if [ -s "${CURRENT_LOG_ERROR}" ]; then
    {
      printf '\nAzure Blob log forwarder diagnostics:\n'
      sed 's/^/[azure-blob-log] /' "${CURRENT_LOG_ERROR}"
    } >>"${CURRENT_BUILD_LOG}"
  fi
  local artifact
  for artifact in build.log build-benchmark.json maven-publish.json sdk-assets.tar.gz shard-manifest.json; do
    if [ -e "${CURRENT_OUTPUT_DIR}/${artifact}" ]; then
      upload_object "${CURRENT_OBJECT_PREFIX}/${artifact}"         "${CURRENT_OUTPUT_DIR}/${artifact}" || final_code=1
    fi
  done
  python3 - "${CURRENT_CONFIG_FILE}" "${final_code}"     "${CURRENT_OUTPUT_DIR}/status.json" <<'PY'
import json
import sys
import time
config = json.load(open(sys.argv[1], encoding="utf-8"))
shard = config["shard"]
json.dump(
    {
        "runId": config["runId"],
        "shard": shard["id"],
        "controllerEpoch": config["controllerEpoch"],
        "repository": config["repository"],
        "commit": config["commit"],
        "releaseVersion": config["releaseVersion"],
        "snapshotVersion": config["snapshotVersion"],
        "contractDigest": shard["contractDigest"],
        "variants": [item["name"] for item in shard["build"]["variants"]],
        "exitCode": int(sys.argv[2]),
        "completedAt": int(time.time()),
    },
    open(sys.argv[3], "w", encoding="utf-8"),
    sort_keys=True,
)
PY
  upload_object "${CURRENT_OBJECT_PREFIX}/status.json"     "${CURRENT_OUTPUT_DIR}/status.json" || \
    phase status-upload failed "shard=${CURRENT_SHARD_ID}"
  CURRENT_FINALIZED=1
  CURRENT_ACTIVE=0
  set -e
  return "${final_code}"
}

# Invoked indirectly by the EXIT trap.
# shellcheck disable=SC2329
finish_lane() {
  local exit_code=$?
  trap - EXIT
  set +e
  phase finalize started "lane=${LANE_ID} exitCode=${exit_code}"
  [ -n "${WATCHDOG_PID}" ] && kill "${WATCHDOG_PID}" 2>/dev/null
  if [ "${CURRENT_ACTIVE}" -eq 1 ] && [ "${CURRENT_FINALIZED}" -eq 0 ]; then
    finalize_shard "${exit_code}" || exit_code=1
  fi
  stop_forwarder "${LANE_FORWARDER_PID}" "${LANE_FORWARDER_STOP}"
  [ -s "${LANE_FORWARDER_ERROR}" ] &&     sed 's/^/[azure-lane-log] /' "${LANE_FORWARDER_ERROR}"
  python3 - "${LANE_ID}" "${exit_code}" "${WORK_ROOT}/lane-status.json" <<'PY'
import json
import sys
import time
json.dump(
    {"lane": sys.argv[1], "exitCode": int(sys.argv[2]), "completedAt": int(time.time())},
    open(sys.argv[3], "w", encoding="utf-8"),
    sort_keys=True,
)
PY
  upload_object "${ARTIFACT_PREFIX}/${RUN_ID}/lanes/${LANE_ID}/status.json"     "${WORK_ROOT}/lane-status.json" || true
  sync
  shutdown -h now || true
  exit "${exit_code}"
}
trap finish_lane EXIT
trap 'phase worker failed "line=${LINENO} command=${BASH_COMMAND}"' ERR

watch_kill_switch() {
  while true; do
    if python3 "${CLOUD_IO}" kill-enabled --bucket "${KILL_SWITCH_BUCKET}"         --object "${RUN_KILL_SWITCH_OBJECT}" --emergency-object "${KILL_SWITCH_OBJECT}"         --controller-epoch "${CONTROLLER_EPOCH}" --client-id "${AZURE_CLIENT_ID}" >/dev/null 2>&1; then
      reason=enabled
    else
      state=$?
      if [ "${state}" -eq 1 ]; then
        sleep 15
        continue
      fi
      reason=unreadable
    fi
    phase kill-switch failed "state=${reason} lane=${LANE_ID}"
    current_pid=$([ -f "${BUILD_PID_FILE}" ] && tr -dc '0-9' < "${BUILD_PID_FILE}" || true)
    [ -n "${current_pid}" ] && kill -TERM -- "-${current_pid}" 2>/dev/null || true
    sleep 5
    [ -n "${current_pid}" ] && kill -KILL -- "-${current_pid}" 2>/dev/null || true
    kill -TERM "${MAIN_PID}" 2>/dev/null || true
    # The main EXIT trap owns checkpoint publication and normal shutdown. This
    # watchdog only forces power-off if finalization itself remains stuck.
    for _ in $(seq 1 90); do
      kill -0 "${MAIN_PID}" 2>/dev/null || return
      sleep 2
    done
    shutdown -h now || true
    return
  done
}

ensure_common_toolchains() {
  local java_arch machine protobuf_identity protoc_arch protoc_identity
  local cmake_identity rust_identity
  export DEBIAN_FRONTEND=noninteractive
  machine=$(uname -m)
  phase package-index started "lane=${LANE_ID}"
  apt-get update
  phase package-index complete
  phase toolchain-packages started
  apt-get install -y --no-install-recommends autoconf automake build-essential     ca-certificates ccache cmake curl docker.io gfortran git gnupg jq     libdwarf-dev libdw-dev libelf-dev libgomp1 libomp-dev libopenblas-dev     libtool libusb-1.0-0-dev libvulkan-dev libvulkan1 maven     mesa-vulkan-drivers nasm ninja-build openjdk-11-jdk pinentry-curses     pkg-config python3 swig tar unzip vulkan-tools wget zip zlib1g-dev
  apt-get install -y llvm-18-dev mlir-18-tools ||     apt-get install -y llvm-dev libmlir-dev mlir-tools || true
  systemctl start docker || true
  phase toolchain-packages complete
  java_arch=$(dpkg --print-architecture)
  export JAVA_HOME="/usr/lib/jvm/java-11-openjdk-${java_arch}"

  protobuf_identity=$(cache_identity protobuf 3.8.0 linux "${machine}" /opt/protobuf)
  if [ ! -x /opt/protobuf/bin/protoc ] && ! restore_dependency protobuf "${protobuf_identity}" /opt/protobuf; then
    phase protobuf-toolchain started
    curl --fail --location --retry 5       https://github.com/google/protobuf/releases/download/v3.8.0/protobuf-cpp-3.8.0.tar.gz       -o /tmp/protobuf-3.8.0.tar.gz
    rm -rf /tmp/protobuf-3.8.0
    tar -xzf /tmp/protobuf-3.8.0.tar.gz -C /tmp
    (cd /tmp/protobuf-3.8.0 && ./configure --prefix=/opt/protobuf && make -j2 && make install)
    publish_dependency protobuf "${protobuf_identity}" /opt/protobuf
    phase protobuf-toolchain complete
  fi
  [ -x /opt/protobuf/bin/protoc ] || { phase protobuf-toolchain failed "reason=missing-protoc"; return 1; }
  export PATH="/opt/protobuf/bin:${PATH}"

  protoc_arch=$([ "${machine}" = aarch64 ] && printf aarch_64 || printf x86_64)
  protoc_identity=$(cache_identity protoc 21.7 linux "${protoc_arch}" /opt/protoc-21.7)
  if [ ! -x /opt/protoc-21.7/bin/protoc ] && ! restore_dependency protoc "${protoc_identity}" /opt/protoc-21.7; then
    curl --fail --location --retry 5       "https://github.com/protocolbuffers/protobuf/releases/download/v21.7/protoc-21.7-linux-${protoc_arch}.zip"       -o /tmp/protoc.zip
    mkdir -p /opt/protoc-21.7
    unzip -qo /tmp/protoc.zip -d /opt/protoc-21.7 bin/protoc
    chmod +x /opt/protoc-21.7/bin/protoc
    publish_dependency protoc "${protoc_identity}" /opt/protoc-21.7
  fi
  [ -x /opt/protoc-21.7/bin/protoc ] || { phase protoc-toolchain failed "reason=missing-protoc"; return 1; }

  if [ "${machine}" = x86_64 ]; then
    cmake_identity=$(cache_identity cmake 3.28.3 linux x86_64 /opt/cmake)
    if [ ! -x /opt/cmake/bin/cmake ] && ! restore_dependency cmake "${cmake_identity}" /opt/cmake; then
      curl --fail --location --retry 5       https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.tar.gz       -o /tmp/cmake.tar.gz
      mkdir -p /opt/cmake
      tar -xzf /tmp/cmake.tar.gz -C /opt/cmake --strip-components=1
      publish_dependency cmake "${cmake_identity}" /opt/cmake
    fi
  fi
  [ ! -x /opt/cmake/bin/cmake ] || export PATH="/opt/cmake/bin:${PATH}"

  rust_identity=$(cache_identity rust-cbindgen rustc-1.97.1 cbindgen-0.29.4 linux "${machine}" "${TOOLCHAIN_ROOT}")
  if [ ! -x "${CARGO_HOME}/bin/cbindgen" ] && ! restore_dependency rust-cbindgen "${rust_identity}" "${TOOLCHAIN_ROOT}"; then
    phase rust-toolchain started
    curl --proto '=https' --tlsv1.2 --fail --silent --show-error       https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.97.1
    cargo install --locked --version 0.29.4 cbindgen
    publish_dependency rust-cbindgen "${rust_identity}" "${TOOLCHAIN_ROOT}"
    phase rust-toolchain complete
  fi
  [ -x "${CARGO_HOME}/bin/cbindgen" ] || { phase rust-toolchain failed "reason=missing-cbindgen"; return 1; }
}

ensure_shard_toolchains() {
  local platform ndk_version ndk_identity graal_arch
  platform=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["shard"]["build"]["javacppPlatform"])' "${CURRENT_CONFIG_FILE}")
  if [[ "${platform}" == android-* ]]; then
    ndk_version=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["shard"]["build"]["ndkVersion"])' "${CURRENT_CONFIG_FILE}")
    ndk_identity=$(cache_identity android-ndk "${ndk_version}" linux "$(uname -m)" "/opt/android/android-ndk-${ndk_version}")
    if [ ! -d "/opt/android/android-ndk-${ndk_version}" ] && ! restore_dependency android-ndk "${ndk_identity}" "/opt/android/android-ndk-${ndk_version}"; then
      curl --fail --location --retry 5         "https://dl.google.com/android/repository/android-ndk-${ndk_version}-linux.zip"         -o "/tmp/android-ndk-${ndk_version}.zip"
      mkdir -p /opt/android
      unzip -q "/tmp/android-ndk-${ndk_version}.zip" -d /opt/android
      publish_dependency android-ndk "${ndk_identity}" "/opt/android/android-ndk-${ndk_version}"
    fi
    export ANDROID_NDK="/opt/android/android-ndk-${ndk_version}"
    export ANDROID_NDK_HOME="${ANDROID_NDK}"
  else
    unset ANDROID_NDK ANDROID_NDK_HOME || true
  fi
  if python3 -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["shard"]["build"].get("buildAot") else 1)' "${CURRENT_CONFIG_FILE}"; then
    graal_arch=$([ "$(uname -m)" = aarch64 ] && printf aarch64 || printf x64)
    if [ ! -x /opt/graalvm/bin/java ]; then
      # The upstream URL is intentionally not cached until the plan pins an
      # immutable GraalVM build instead of the moving "latest" alias.
      curl --fail --location --retry 5         "https://download.oracle.com/graalvm/21/latest/graalvm-jdk-21_linux-${graal_arch}_bin.tar.gz"         -o /tmp/graalvm.tar.gz
      mkdir -p /opt/graalvm
      tar -xzf /tmp/graalvm.tar.gz -C /opt/graalvm --strip-components=1
    fi
    export GRAALVM_HOME=/opt/graalvm
  else
    unset GRAALVM_HOME || true
  fi
}

ensure_container_image() {
  local image="$1" manifest_digest identity cache_dir
  [ -n "${image}" ] || return 0
  if docker image inspect "${image}" >/dev/null 2>&1; then
    return 0
  fi
  manifest_digest=$(docker manifest inspect "${image}" | python3 -c 'import hashlib,sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())')
  identity=$(cache_identity container-image "${image}" "$(uname -m)" "${manifest_digest}")
  cache_dir="${TOOLCHAIN_ROOT}/container-images/${identity}"
  if restore_dependency container-image "${identity}" "${cache_dir}"; then
    docker load --input "${cache_dir}/image.tar"
    docker image inspect "${image}" >/dev/null
    return 0
  fi
  docker pull "${image}"
  mkdir -p "${cache_dir}"
  docker save --output "${cache_dir}/image.tar" "${image}"
  publish_dependency container-image "${identity}" "${cache_dir}"
}

run_shard() (
  set -Eeuo pipefail
  local safe_id source_dir output_dir maven_repo container_image container_family
  safe_id=$(printf '%s' "${CURRENT_SHARD_ID}" | tr -c 'A-Za-z0-9._-' '-')
  source_dir="${SOURCE_ROOT}/${safe_id}"
  output_dir="${CURRENT_OUTPUT_DIR}"
  maven_repo="${MAVEN_REPO_ROOT}/${safe_id}"
  rm -rf "${source_dir}"
  mkdir -p "${maven_repo}" "${output_dir}/maven-repository" "${output_dir}/sdk-assets"
  ensure_shard_toolchains
  phase source-checkout started "shard=${CURRENT_SHARD_ID} commit=${COMMIT}"
  git clone --filter=blob:none "${REPOSITORY}" "${source_dir}"
  git -C "${source_dir}" fetch --depth=1 origin "${COMMIT}"
  git -C "${source_dir}" checkout --detach "${COMMIT}"
  [ "$(git -C "${source_dir}" rev-parse HEAD)" = "${COMMIT}" ] || exit 2
  phase source-checkout complete "shard=${CURRENT_SHARD_ID} commit=${COMMIT}"

  container_image=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["shard"].get("containerImage", ""))' "${CURRENT_CONFIG_FILE}")
  container_family=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["shard"].get("containerFamily", "debian"))' "${CURRENT_CONFIG_FILE}")
  ensure_container_image "${container_image}"
  build=(python3 "${BUILD_DRIVER}" --config "${CURRENT_CONFIG_FILE}"     --source "${source_dir}" --repository "${maven_repo}"     --maven-output "${output_dir}/maven-repository"     --sdk-output "${output_dir}/sdk-assets")
  if [ -n "${container_image}" ]; then
    if [ "${container_family}" = almalinux ]; then
      docker build --tag dl4j-release-compat         --file "${source_dir}/.github/actions/build-centos/Dockerfile"         "${source_dir}/.github/actions/build-centos"
      build=(docker run --rm --network host -e GITHUB_WORKSPACE=/github/workspace         -e LIBND4J_FILE_NAME= -e LIBND4J_URL= -e OPENBLAS_PATH=/usr         -e "INSTALL_COMMAND=python3 /dl4j-build-platform.py --config /dl4j-config.json --source /github/workspace --repository /dl4j-m2 --maven-output /dl4j-output/maven-repository --sdk-output /dl4j-output/sdk-assets"         -e DL4J_CLOUD_IO=/dl4j-cloud-io.py -e DL4J_DEPENDENCY_CACHE_HELPER=/dl4j-dependency-cache.py -e "AZURE_CLIENT_ID=${AZURE_CLIENT_ID}"         -v "${source_dir}:/github/workspace" -v "${SCCACHE_ROOT}:/github/sccache" -v "${maven_repo}:/dl4j-m2"         -v "${output_dir}:/dl4j-output" -v "${CURRENT_CONFIG_FILE}:/dl4j-config.json:ro"         -v "${BUILD_DRIVER}:/dl4j-build-platform.py:ro" -v "${CLOUD_IO}:/dl4j-cloud-io.py:ro"         -v "${DEPENDENCY_CACHE}:/dl4j-dependency-cache.py:ro" dl4j-release-compat)
    else
      build=(docker run --rm --network host -v "${source_dir}:/workspace"         -v "${SCCACHE_ROOT}:/sccache" -v "${maven_repo}:/dl4j-m2" -v "${output_dir}:/dl4j-output"         -v "${CURRENT_CONFIG_FILE}:/dl4j-config.json:ro"         -v "${BUILD_DRIVER}:/dl4j-build-platform.py:ro" -v "${CLOUD_IO}:/dl4j-cloud-io.py:ro"         -v "${DEPENDENCY_CACHE}:/dl4j-dependency-cache.py:ro" -e DL4J_CLOUD_IO=/dl4j-cloud-io.py         -e DL4J_DEPENDENCY_CACHE_HELPER=/dl4j-dependency-cache.py -e "AZURE_CLIENT_ID=${AZURE_CLIENT_ID}"         -v /opt/protobuf:/opt/protobuf:ro -v /opt/cmake:/opt/cmake:ro         -w /workspace "${container_image}" bash -lc         "apt-get update && apt-get install -y --no-install-recommends autoconf automake build-essential ca-certificates cmake gfortran git libomp-dev libopenblas-dev libtool libzstd-dev maven nasm ninja-build openjdk-11-jdk pkg-config python3 swig unzip xz-utils zip zlib1g-dev && export JAVA_HOME=\$(dirname \$(dirname \$(readlink -f \$(command -v javac)))) && export PATH=\${JAVA_HOME}/bin:/opt/protobuf/bin:/opt/cmake/bin:\$PATH && python3 /dl4j-build-platform.py --config /dl4j-config.json --source /workspace --repository /dl4j-m2 --maven-output /dl4j-output/maven-repository --sdk-output /dl4j-output/sdk-assets")
    fi
  fi
  phase matrix-build started "shard=${CURRENT_SHARD_ID}"
  trap 'rm -f "${BUILD_PID_FILE}"' EXIT
  setsid "${build[@]}" &
  build_pid=$!
  printf '%s\n' "${build_pid}" >"${BUILD_PID_FILE}"
  set +e
  wait "${build_pid}"
  build_code=$?
  set -e
  rm -f "${BUILD_PID_FILE}"
  trap - EXIT
  if [ "${build_code}" -eq 0 ]; then
    phase matrix-build complete "shard=${CURRENT_SHARD_ID}"
  else
    phase matrix-build failed "shard=${CURRENT_SHARD_ID} exitCode=${build_code} packaging=partial"
  fi

  if python3 - "${CURRENT_CONFIG_FILE}" <<'PY'
import json
import sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if "maven" in config["shard"].get("workloads", []) else 1)
PY
  then
    phase maven-publish started "shard=${CURRENT_SHARD_ID} repository=${MAVEN_REPOSITORY_PREFIX}"
    publisher=(python3 "${source_dir}/release/azure/maven-publish.py" \
      --repository "${output_dir}/maven-repository" \
      --sdk-assets "${output_dir}/sdk-assets" \
      --config "${CURRENT_CONFIG_FILE}" \
      --central-repository "${source_dir}/release/central/repository.py" \
      --cloud-io "${CLOUD_IO}" \
      --bucket "${BUCKET}" \
      --repository-prefix "${MAVEN_REPOSITORY_PREFIX}" \
      --client-id "${AZURE_CLIENT_ID}" \
      --run-id "${RUN_ID}" \
      --shard "${CURRENT_SHARD_ID}" \
      --release-version "$(config releaseVersion)" \
      --commit "${COMMIT}" \
      --accounting "${output_dir}/maven-publish.json")
    if [ "${build_code}" -ne 0 ]; then
      publisher+=(--allow-missing-unclassified)
    fi
    "${publisher[@]}"
    phase maven-publish complete "shard=${CURRENT_SHARD_ID} accounting=${output_dir}/maven-publish.json"
  fi

  phase artifact-packaging started "shard=${CURRENT_SHARD_ID} buildExitCode=${build_code}"
  tar -C "${output_dir}/sdk-assets" -czf "${output_dir}/sdk-assets.tar.gz" .
  python3 - "${output_dir}" "${CURRENT_CONFIG_FILE}" "${build_code}" <<'PY'
import hashlib
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
files = []
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    files.append({
        "path": path.relative_to(root).as_posix(),
        "sha256": digest.hexdigest(),
        "size": path.stat().st_size,
    })
config = json.load(open(sys.argv[2], encoding="utf-8"))
shard = config["shard"]
build_exit_code = int(sys.argv[3])
progress_path = root / "build-result.json"
if progress_path.is_file():
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    variants = progress.get("completedVariants", [])
else:
    variants = (
        [variant["name"] for variant in shard["build"]["variants"]]
        if build_exit_code == 0 else []
    )
json.dump({
    "schemaVersion": 1,
    "provider": "azure",
    "runId": config["runId"],
    "shard": shard["id"],
    "commit": config["commit"],
    "releaseVersion": config["releaseVersion"],
    "workloads": shard["workloads"],
    "os": shard["os"],
    "platform": shard["build"]["javacppPlatform"],
    "backend": shard["build"]["backend"],
    "variants": variants,
    "partial": build_exit_code != 0,
    "buildExitCode": build_exit_code,
    "files": files,
}, open(root / "shard-manifest.json", "w", encoding="utf-8"), indent=2, sort_keys=True)
PY
  phase artifact-packaging complete "shard=${CURRENT_SHARD_ID} buildExitCode=${build_code}"
  return "${build_code}"
)

phase worker started "lane=${LANE_ID} pid=$$"
start_lane_forwarder
if python3 "${CLOUD_IO}" kill-enabled --bucket "${KILL_SWITCH_BUCKET}"     --object "${RUN_KILL_SWITCH_OBJECT}" --emergency-object "${KILL_SWITCH_OBJECT}"     --controller-epoch "${CONTROLLER_EPOCH}" --client-id "${AZURE_CLIENT_ID}" >/dev/null 2>&1; then
  exit 130
else
  KILL_STATE=$?
  [ "${KILL_STATE}" -eq 1 ] || exit 131
fi
watch_kill_switch &
WATCHDOG_PID=$!

PENDING_SHARDS=()
for shard_id in "${SHARD_IDS[@]}"; do
  if remote_shard_succeeded "${shard_id}"; then
    phase shard skipped "shard=${shard_id} reason=remote-success-checkpoint"
  else
    PENDING_SHARDS+=("${shard_id}")
  fi
done
if [ "${#PENDING_SHARDS[@]}" -eq 0 ]; then
  phase worker complete "lane=${LANE_ID} reason=all-shards-checkpointed"
  exit 0
fi

prepare_shard_context "${PENDING_SHARDS[0]}"
CURRENT_ACTIVE=1
CURRENT_FINALIZED=0
start_shard_forwarder
ensure_common_toolchains

for shard_id in "${PENDING_SHARDS[@]}"; do
  if [ "${CURRENT_SHARD_ID}" != "${shard_id}" ]; then
    prepare_shard_context "${shard_id}"
    CURRENT_ACTIVE=1
    CURRENT_FINALIZED=0
    start_shard_forwarder
  fi
  phase shard started "lane=${LANE_ID} shard=${CURRENT_SHARD_ID}"
  set +e
  run_shard 2>&1 | tee -a "${CURRENT_BUILD_LOG}"
  shard_code=${PIPESTATUS[0]}
  set -e
  set +e
  finalize_shard "${shard_code}"
  final_code=$?
  set -e
  if [ "${final_code}" -ne 0 ]; then
    phase shard failed "lane=${LANE_ID} shard=${CURRENT_SHARD_ID} exitCode=${final_code}"
    exit "${final_code}"
  fi
  phase shard complete "lane=${LANE_ID} shard=${CURRENT_SHARD_ID}"
done
phase worker complete "lane=${LANE_ID} shards=${#PENDING_SHARDS[@]}"
exit 0
