#!/usr/bin/env bash
set -Eeuo pipefail
export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
export HOME="${HOME:-/root}"
export USER="${USER:-$(id -un)}"
unset LD_PRELOAD

CONFIG_B64='__DL4J_WORKER_CONFIG_B64__'
CLOUD_IO_B64='__DL4J_CLOUD_IO_B64__'
CONFIG_FILE=/tmp/dl4j-tpu-smoke.json
CLOUD_IO=/tmp/dl4j-cloud-io.py
WORK_ROOT=/opt/dl4j-tpu-smoke
SOURCE_DIR=${WORK_ROOT}/source
OUTPUT_DIR=${WORK_ROOT}/output
BUILD_LOG=${OUTPUT_DIR}/tpu-smoke.log
STOP_FILE=/tmp/dl4j-tpu-log-forwarder.stop
BUILD_PID_FILE=/tmp/dl4j-tpu-smoke.pid
FORWARDER_PID=""
WATCHDOG_PID=""
mkdir -p "${OUTPUT_DIR}" "${HOME}/.m2"
printf '%s' "${CONFIG_B64}" | base64 --decode > "${CONFIG_FILE}"
printf '%s' "${CLOUD_IO_B64}" | base64 --decode > "${CLOUD_IO}"
exec > >(tee -a "${BUILD_LOG}") 2>&1

config() {
  python3 -c 'import json,sys; value=json.load(open(sys.argv[1]));
for part in sys.argv[2].split("."): value=value[part]
print(value)' "${CONFIG_FILE}" "$1"
}
phase() { printf '[dl4j-phase] timestamp=%s phase=%s status=%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "${3:-}"; }

PROJECT=$(config project)
BUCKET=$(config bucket)
KILL_SWITCH_BUCKET=$(config killSwitchBucket)
PREFIX=$(config artifactPrefix)/$(config runId)/tpu-smoke
RUN_ID=$(config runId)
COMMIT=$(config commit)
REPOSITORY=$(config repository)
LOG_ID=$(config logId)
KILL_SWITCH_OBJECT=$(config killSwitchObject)
EXIT_CODE=1

upload_if_present() {
  [ -e "$1" ] && python3 "${CLOUD_IO}" upload --bucket "${BUCKET}" --object "${PREFIX}/$2" --file "$1" || true
}

finish() {
  EXIT_CODE=$?
  set +e
  phase finalize started "exitCode=${EXIT_CODE}"
  [ -n "${WATCHDOG_PID}" ] && kill "${WATCHDOG_PID}" 2>/dev/null || true
  touch "${STOP_FILE}"
  [ -n "${FORWARDER_PID}" ] && wait "${FORWARDER_PID}" 2>/dev/null || true
  if [ -d "${SOURCE_DIR}/platform-tests/target/surefire-reports" ]; then
    tar -C "${SOURCE_DIR}/platform-tests/target" -czf "${OUTPUT_DIR}/surefire-reports.tar.gz" surefire-reports
  fi
  python3 -c 'import json,sys,time; json.dump({"resource":"cloud-tpu-vm","exitCode":int(sys.argv[1]),"completedAt":int(time.time())},open(sys.argv[2],"w"),sort_keys=True)' "${EXIT_CODE}" "${OUTPUT_DIR}/status.json"
  upload_if_present "${BUILD_LOG}" tpu-smoke.log
  upload_if_present "${OUTPUT_DIR}/surefire-reports.tar.gz" surefire-reports.tar.gz
  upload_if_present "${OUTPUT_DIR}/status.json" status.json
  sync
  shutdown -h now || true
  exit "${EXIT_CODE}"
}
trap finish EXIT
trap 'phase tpu-smoke failed "line=${LINENO} command=${BASH_COMMAND}"' ERR

watch_kill_switch() {
  while true; do
    if python3 "${CLOUD_IO}" kill-enabled --bucket "${KILL_SWITCH_BUCKET}" --object "${KILL_SWITCH_OBJECT}" >/dev/null 2>&1; then
      reason=enabled
    else
      state=$?
      if [ "${state}" -eq 1 ]; then
        sleep 15
        continue
      fi
      reason=unreadable
    fi
    phase kill-switch "${reason}"
    current_pid=$([ -f "${BUILD_PID_FILE}" ] && tr -dc '0-9' < "${BUILD_PID_FILE}" || true)
    [ -n "${current_pid}" ] && kill -TERM -- "-${current_pid}" 2>/dev/null || true
    sleep 5
    [ -n "${current_pid}" ] && kill -KILL -- "-${current_pid}" 2>/dev/null || true
    shutdown -h now || true
    return
  done
}
if python3 "${CLOUD_IO}" kill-enabled --bucket "${KILL_SWITCH_BUCKET}" --object "${KILL_SWITCH_OBJECT}" >/dev/null 2>&1; then
  exit 130
else
  KILL_STATE=$?
  [ "${KILL_STATE}" -eq 1 ] || exit 131
fi
watch_kill_switch &
WATCHDOG_PID=$!

phase worker started "pid=$$"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl git jq maven openjdk-11-jdk python3 tar unzip
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-$(dpkg --print-architecture)
export MAVEN_OPTS="-Xmx8g"
python3 "${CLOUD_IO}" forward --project "${PROJECT}" --file "${BUILD_LOG}" --stop-file "${STOP_FILE}" --log-id "${LOG_ID}" --run-id "${RUN_ID}" --shard tpu-smoke >/tmp/dl4j-tpu-log-forwarder.err 2>&1 &
FORWARDER_PID=$!

phase source-checkout started "commit=${COMMIT}"
git clone --filter=blob:none "${REPOSITORY}" "${SOURCE_DIR}"
git -C "${SOURCE_DIR}" fetch --depth=1 origin "${COMMIT}"
git -C "${SOURCE_DIR}" checkout --detach "${COMMIT}"
[ "$(git -C "${SOURCE_DIR}" rev-parse HEAD)" = "${COMMIT}" ] || exit 2
phase source-checkout complete "commit=${COMMIT}"

cat > "${HOME}/.m2/settings.xml" <<'SETTINGS'
<settings>
  <profiles>
    <profile>
      <id>sonatype-snapshots</id>
      <repositories>
        <repository>
          <id>central-portal-snapshots</id>
          <url>https://central.sonatype.com/repository/maven-snapshots/</url>
          <releases><enabled>false</enabled></releases>
          <snapshots><enabled>true</enabled><updatePolicy>always</updatePolicy></snapshots>
        </repository>
      </repositories>
    </profile>
  </profiles>
  <activeProfiles><activeProfile>sonatype-snapshots</activeProfile></activeProfiles>
</settings>
SETTINGS

phase libtpu-discovery started
PJRT_PATH="${TPU_LIBRARY_PATH:-}"
if [ -z "${PJRT_PATH}" ]; then
  PJRT_PATH=$(find /lib /usr/lib /usr/local/lib /lib64 /usr/lib64 -name libtpu.so 2>/dev/null | head -1 || true)
fi
if [ -z "${PJRT_PATH}" ]; then
  LIBTPU_WHL=$(curl -sfL https://pypi.org/pypi/libtpu/json | jq -r '.urls[] | select(.filename | test("linux.*x86_64|manylinux.*x86_64")) | .url' | head -1)
  [ -n "${LIBTPU_WHL}" ]
  curl -fL "${LIBTPU_WHL}" -o /tmp/libtpu.whl
  mkdir -p /tmp/libtpu-extracted
  unzip -o /tmp/libtpu.whl '*.so*' -d /tmp/libtpu-extracted/ || true
  PJRT_PATH=$(find /tmp/libtpu-extracted -name libtpu.so 2>/dev/null | head -1 || true)
fi
[ -n "${PJRT_PATH}" ] || { printf 'libtpu.so not found on this TPU VM\n'; exit 1; }
export PJRT_PATH TPU_LIBRARY_PATH="${PJRT_PATH}"
phase libtpu-discovery complete "path=${PJRT_PATH}"

phase tpu-smoke-build started
setsid bash -c 'cd "$1" && mvn install -DskipTests -DskipTestResourceEnforcement=true -Ptpu,test-tpu -Dbackend.artifactId=nd4j-native -Djavacpp.platform=linux-x86_64 -Djavacpp.platform.extension= -Dplatform.classifier=linux-x86_64 -Dmaven.javadoc.skip=true -pl :platform-tests --also-make -pl !:libnd4j --no-transfer-progress --batch-mode' _ "${SOURCE_DIR}" &
BUILD_PID=$!
printf '%s\n' "${BUILD_PID}" > "${BUILD_PID_FILE}"
wait "${BUILD_PID}"
phase tpu-smoke-build complete

phase tpu-smoke-test started
setsid bash -c 'cd "$1/platform-tests" && mvn test -Ptest-tpu -Dbackend.artifactId=nd4j-native -Djavacpp.platform=linux-x86_64 -Djavacpp.platform.extension= -Dplatform.classifier=linux-x86_64 -Dtest.heap.size=4g -Dtest.offheap.size=6g -Domp.num.threads=2 -DskipTestResourceEnforcement=true -Dtest=TpuBackendSmokeTest --no-transfer-progress --batch-mode' _ "${SOURCE_DIR}" &
BUILD_PID=$!
printf '%s\n' "${BUILD_PID}" > "${BUILD_PID_FILE}"
wait "${BUILD_PID}"
rm -f "${BUILD_PID_FILE}"
phase tpu-smoke-test complete
