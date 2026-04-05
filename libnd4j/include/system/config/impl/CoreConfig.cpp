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
#include <thread>

#ifdef _OPENMP
#include <omp.h>
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
}

}  // namespace config
}  // namespace sd
