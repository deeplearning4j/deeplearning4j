/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <system/config/CoreConfig.h>
#include <system/config/EnvHelper.h>
#include <helpers/BlasHelper.h>
#include <helpers/logger.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

// Portable setenv: MinGW/MSVC lack POSIX setenv(); use _putenv instead.
#ifdef _WIN32
static inline void sd_setenv(const char* name, const char* value, int /*overwrite*/) {
  std::string s = std::string(name) + "=" + value;
  _putenv(s.c_str());
}
#else
static inline void sd_setenv(const char* name, const char* value, int overwrite) {
  setenv(name, value, overwrite);
}
#endif

namespace sd {
namespace config {

CoreConfig::CoreConfig() {
  _maxThreads.store(std::thread::hardware_concurrency());
  _maxMasterThreads.store(_maxThreads.load());
}

void CoreConfig::setMaxMasterThreads(int max) {
  if (max > maxThreads()) max = maxThreads();
  if (max < 1) return;
  _maxMasterThreads.store(max);
}

void CoreConfig::setDefaultFloatDataType(sd::DataType dtype) {
  if (dtype != FLOAT32 && dtype != DOUBLE && dtype != FLOAT8 && dtype != HALF)
    THROW_EXCEPTION("Default Float data type must be one of [FLOAT8, FLOAT16, FLOAT32, DOUBLE]");
  _dataType.store(dtype);
}

bool CoreConfig::isSerializeBlasCalls() {
  return BlasHelper::getInstance().isSerializeBlasCalls();
}

void CoreConfig::setSerializeBlasCalls(bool serialize) {
  _serializeBlasCallsSet.store(true);
  BlasHelper::getInstance().setSerializeBlasCalls(serialize);
}

int CoreConfig::getOpenBlasThreads() {
  return BlasHelper::getInstance().getOpenblasThreads();
}

void CoreConfig::setOpenBlasThreads(int threads) {
  _openBlasThreads.store(threads);
  BlasHelper::getInstance().setOpenblasThreads(threads);
}

std::string CoreConfig::homeDirectory() const {
#ifdef _WIN32
  const char* homeDrive = std::getenv("HOMEDRIVE");
  const char* homePath = std::getenv("HOMEPATH");
  if (homeDrive != nullptr && homePath != nullptr && homeDrive[0] != '\0' &&
      homePath[0] != '\0') {
    return std::string(homeDrive) + std::string(homePath);
  }
#endif
  const char* home = std::getenv("HOME");
  if (home != nullptr && home[0] != '\0') return std::string(home);
  return "";
}

void CoreConfig::setCpuSoftLimitPercent(int percent) {
  if (percent < 0) percent = 0;
  if (percent > 100) percent = 100;
  _cpuSoftLimitPercent.store(percent);
}

std::string CoreConfig::cudaToolkitPath() const {
  const char* cudaPath = std::getenv("CUDA_PATH");
  if (cudaPath != nullptr && cudaPath[0] != '\0') return std::string(cudaPath);
  return "";
}

void CoreConfig::initFromEnvironment() {
#ifndef ANDROID
  {
    int v = readIntEnv("OMP_NUM_THREADS", -1);
    if (v > 0) {
      _maxThreads.store(v);
      _maxMasterThreads.store(v);
    }
  }
#endif

  {
    int v = readIntEnv("SD_MAX_THREADS", -1);
    if (v > 0) _maxThreads.store(v);
  }

  {
    int v = readIntEnv("SD_MASTER_THREADS", -1);
    if (v > 0) _maxMasterThreads.store(v);
  }

  if (_maxMasterThreads.load() > _maxThreads.load()) {
    sd_printf("Warning! MAX_MASTER_THREADS > MAX_THREADS, tuning them down to match each other\n", "");
    _maxMasterThreads.store(_maxThreads.load());
  }

  // ── OMP inference tuning ─────────────────────────────────────────────────
  // These settings apply to ALL OpenMP-threaded code (libnd4j ops, OneDNN,
  // OpenBLAS if OMP-built, etc.). Tuned for inference (low-latency,
  // single-request, sequential execution).
#ifdef _OPENMP
  // Set thread count to match our configured value.
  omp_set_num_threads(_maxThreads.load());

  // KMP_BLOCKTIME: Intel OpenMP spin-wait time (ms) after parallel region.
  // Default is 200ms — threads burn CPU waiting for work between ops.
  // For inference, each op is a brief burst followed by host-side work;
  // sleeping immediately (blocktime=0) frees cores for the host thread
  // and reduces power/heat on high-core-count systems.
  // Only set if user hasn't explicitly configured it.
  if (!std::getenv("KMP_BLOCKTIME")) {
    sd_setenv("KMP_BLOCKTIME", "0", 0);
  }

  // KMP_AFFINITY: Thread-to-core binding (Intel OpenMP).
  // compact,1,0,granularity=fine: pack threads onto physical cores first
  // before using HT siblings. Maximizes per-thread memory bandwidth for
  // memory-bound matmul/attention ops.
  // Only set if user hasn't explicitly configured it.
  if (!std::getenv("KMP_AFFINITY")) {
    sd_setenv("KMP_AFFINITY", "compact,1,0,granularity=fine", 0);
  }

  // GOMP_SPINCOUNT: GCC OpenMP spin-wait count (equivalent to KMP_BLOCKTIME).
  // 0 = sleep immediately after parallel region.
  if (!std::getenv("GOMP_SPINCOUNT")) {
    sd_setenv("GOMP_SPINCOUNT", "0", 0);
  }
#endif

  if (std::getenv("SD_FORBID_HELPERS") != nullptr) {
    _allowHelpers.store(false);
  }

  {
    int64_t v = readInt64Env("SD_MAX_PRIMARY_BYTES", -1);
    if (v >= 0) _maxTotalPrimaryMemory.store(v);
  }
  {
    int64_t v = readInt64Env("SD_MAX_SPECIAL_BYTES", -1);
    if (v >= 0) _maxTotalSpecialMemory.store(v);
  }
  {
    int64_t v = readInt64Env("SD_MAX_DEVICE_BYTES", -1);
    if (v >= 0) _maxDeviceMemory.store(v);
  }

  if (std::getenv("SD_BLAS_FALLBACK") != nullptr) {
    _blasFallback = true;
  }

  // Debug / verbose flags — settable via ND4J_DEBUG=1 and ND4J_VERBOSE=1 env vars.
  // When debug is on, DataBuffer::validateIntegrity() checks magic bytes on every
  // NDArray construction, catching use-after-free and heap corruption early.
  if (readBoolEnv("ND4J_DEBUG", false)) {
    _debug.store(true);
  }
  if (readBoolEnv("ND4J_VERBOSE", false)) {
    _verbose.store(true);
  }

  {
    int v = readIntEnv("SD_CPU_SOFT_LIMIT_PERCENT", -1);
    if (v >= 0 && v <= 100) _cpuSoftLimitPercent.store(v);
  }
}

}  // namespace config
}  // namespace sd
