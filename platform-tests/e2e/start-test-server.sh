#!/bin/bash
# Start the DSP UI test server for Playwright tests.
# This script is invoked by playwright.config.ts via the webServer option.
#
# Uses the CPU backend (nd4j-native) — the test server only trains a tiny
# Iris MLP for UI chart data; CUDA would waste GPU memory and crash when
# another process already holds the device.
set -e

cd "$(dirname "$0")/.."

MVN=/home/agibsonccc/dev-apps/mvn/bin/mvn

# Force CPU backend for the UI test server
BACKEND=nd4j-native

# Ensure test classes are compiled
if [ ! -d target/test-classes ]; then
  $MVN test-compile -DskipTests -Dbackend.artifactId=$BACKEND -q
fi

# Build classpath with CPU backend
$MVN -q dependency:build-classpath -DincludeScope=test \
  -Dbackend.artifactId=$BACKEND \
  -Dmdep.outputFile=/tmp/pw-test-cp.txt 2>/dev/null

CP="target/test-classes:target/classes:$(cat /tmp/pw-test-cp.txt)"

exec java -cp "$CP" \
  -Dorg.nd4j.linalg.factory.Nd4jBackend=org.nd4j.linalg.cpu.nativecpu.CpuBackend \
  org.deeplearning4j.ui.playwright.DspUiTestServer
