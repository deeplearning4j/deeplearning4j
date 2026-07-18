#!/bin/bash
# Vulkan Backend Build Script — follows CUDA/CPU conventions
# 
# Usage: ./build-vulkan.sh [OPTIONS]
# 
# Common options:
#   --rebuild      Full clean rebuild (default: incremental)
#   --threads N    Build threads (default: 12, max 16 if RAM available)
#   --skip-tests   Skip test compilation
#   --llvm-root P  Use the shared LLVM/MLIR install prefix selected for coexisting backends
#   --android      Cross-compile for Android arm64 (requires Android NDK + cmake)
#   --desktop      Build for desktop (Linux/Windows/macOS)

set -euo pipefail

# CRITICAL RULES (from AGENTS.md):
# 1. NEVER run: ccache -C / ccache --clear (destroys 2+ hours of built cache)
# 2. NEVER change -Dlibnd4j.compute=... (invalidates entire CUDA ccache)
# 3. Use Maven from MVN when provided, otherwise PATH
# 4. Pipe build output through tee to a log file
# 5. Always include -Dlibnd4j.buildthreads and -Dlibnd4j.log

# Configuration
BUILD_TYPE="incremental"
BUILD_THREADS=12
SKIP_TESTS=""
TARGET="vulkan"
LOG_FILE="vulkan-build-output.log"
LIBND4J_LOG="libnd4j-vulkan-build.log"
LLVM_ROOT=""
MVN="${MVN:-mvn}"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --rebuild)
      BUILD_TYPE="clean"
      shift
      ;;
    --threads)
      BUILD_THREADS="$2"
      shift 2
      ;;
    --skip-tests)
      SKIP_TESTS="-DskipTests"
      shift
      ;;
    --llvm-root)
      LLVM_ROOT="$2"
      shift 2
      ;;
    --android)
      TARGET="vulkan-android"
      LOG_FILE="vulkan-android-build.log"
      shift
      ;;
    --desktop)
      TARGET="vulkan"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: ./build-vulkan.sh [--rebuild] [--threads N] [--skip-tests] [--llvm-root PATH] [--android|--desktop]"
      exit 1
      ;;
  esac
done

# Verify working directory
if [ ! -f "pom.xml" ]; then
  echo "ERROR: Must run from project root (where pom.xml exists)"
  exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Vulkan Backend Build — $BUILD_TYPE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Target:       $TARGET"
echo "Threads:      $BUILD_THREADS"
echo "Build type:   $BUILD_TYPE"
echo "Log file:     $LOG_FILE"
echo "Skip tests:   ${SKIP_TESTS:-no}"
echo "LLVM root:    ${LLVM_ROOT:-automatic discovery}"
echo ""
echo "CRITICAL: Ensure ccache is warm before running (do NOT run: ccache -C)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Build command for Vulkan
# Vulkan backend can build on desktop for testing/development
if [ "$BUILD_TYPE" = "clean" ]; then
  CLEAN_ARG="clean"
else
  CLEAN_ARG=""
fi

$MVN \
  -Pvulkan \
  -Dlibnd4j.buildthreads=$BUILD_THREADS \
  -Dlibnd4j.log=$LIBND4J_LOG \
  -Dlibnd4j.llvm.root="$LLVM_ROOT" \
  -pl libnd4j,:nd4j-vulkan \
  $CLEAN_ARG install $SKIP_TESTS 2>&1 | tee "$LOG_FILE"

# Check build status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "✓ Vulkan build SUCCEEDED"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  echo "Next steps:"
  echo "  1. Run tests:      cd platform-tests && mvn test -Dtest=TestVulkan"
  echo "  2. Benchmark:      ./run-benchmark.sh --backend vulkan --tokens 250"
  echo "  3. DSP validation: ./run-validation.sh --backend vulkan --test ALL"
  echo ""
  exit 0
else
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "✗ Vulkan build FAILED (see $LOG_FILE)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  echo "Debug steps:"
  echo "  1. Check $LOG_FILE for Maven errors"
  echo "  2. Check $LIBND4J_LOG for C++ compilation errors"
  echo "  3. Verify CMake configuration: libnd4j/blasbuild/vulkan/CMakeCache.txt"
  echo "  4. For header errors: check -DMLIR_ENABLE_VULKAN=ON in CMake"
  echo ""
  exit 1
fi
