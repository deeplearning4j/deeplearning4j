/* ******************************************************************************
 *
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

//
//  @author raver119@gmail.com
//
#include <helpers/BlasHelper.h>
#include <cstdlib>
#include <string>

// OpenBLAS thread control function declaration
#if HAVE_OPENBLAS
extern "C" void openblas_set_num_threads(int num_threads);
extern "C" int openblas_get_num_threads(void);
#endif

namespace sd {

BlasHelper::BlasHelper() {
  // Initialize BLAS threading configuration from environment
  initializeBlasThreading();
}

BlasHelper &BlasHelper::getInstance() {
  static BlasHelper instance;
  return instance;
}

void BlasHelper::initializeFunctions(Pointer *functions) {
  sd_debug("Initializing BLAS\n", "");

  _hasSgemv = functions[0] != nullptr;
  _hasSgemm = functions[2] != nullptr;

  _hasDgemv = functions[1] != nullptr;
  _hasDgemm = functions[3] != nullptr;

  _hasSgemmBatch = functions[4] != nullptr;
  _hasDgemmBatch = functions[5] != nullptr;
#if !defined(SD_CUDA)
  this->cblasSgemv = (CblasSgemv)functions[0];
  this->cblasDgemv = (CblasDgemv)functions[1];
  this->cblasSgemm = (CblasSgemm)functions[2];
  this->cblasDgemm = (CblasDgemm)functions[3];
  this->cblasSgemmBatch = (CblasSgemmBatch)functions[4];
  this->cblasDgemmBatch = (CblasDgemmBatch)functions[5];
  this->lapackeSgesvd = (LapackeSgesvd)functions[6];
  this->lapackeDgesvd = (LapackeDgesvd)functions[7];
  this->lapackeSgesdd = (LapackeSgesdd)functions[8];
  this->lapackeDgesdd = (LapackeDgesdd)functions[9];
#endif
}

void BlasHelper::initializeDeviceFunctions(Pointer *functions) {
  sd_debug("Initializing device BLAS\n", "");

}

#if defined(HAS_FLOAT32)
template <>
bool BlasHelper::hasGEMV<float>() {
  if (Environment::getInstance().blasFallback()) return false;

#if __EXTERNAL_BLAS__ || HAVE_OPENBLAS
  return true;
#else
  return _hasSgemv;
#endif
}
#endif

#if defined(HAS_DOUBLE)
template <>
bool BlasHelper::hasGEMV<double>() {
  if (Environment::getInstance().blasFallback()) return false;

#if __EXTERNAL_BLAS__ || HAVE_OPENBLAS
  return true;
#else
  return _hasDgemv;
#endif
}
#endif

#if defined(HAS_FLOAT16)
template <>
bool BlasHelper::hasGEMV<float16>() {
  return false;
}
#endif

#if defined(HAS_BFLOAT16)
template <>
bool BlasHelper::hasGEMV<bfloat16>() {
  return false;
}
#endif

#if defined(HAS_BOOL)
template <>
bool BlasHelper::hasGEMV<bool>() {
  return false;
}
#endif

#if defined(HAS_INT32)
template <>
bool BlasHelper::hasGEMV<int>() {
  return false;
}
#endif

#if defined(HAS_INT8)
template <>
bool BlasHelper::hasGEMV<int8_t>() {
  return false;
}
#endif

#if defined(HAS_UINT8)
template <>
bool BlasHelper::hasGEMV<uint8_t>() {
  return false;
}
#endif

#if defined(HAS_INT16)
template <>
bool BlasHelper::hasGEMV<int16_t>() {
  return false;
}
#endif

#if defined(HAS_LONG)
template <>
bool BlasHelper::hasGEMV<LongType>() {
  return false;
}
#endif

bool BlasHelper::hasGEMV(const DataType dtype) {
#if defined(HAS_FLOAT32)
  if (dtype == FLOAT32) {
    if (Environment::getInstance().blasFallback()) return false;

#if __EXTERNAL_BLAS__ || HAVE_OPENBLAS
    return true;
#else
    return _hasSgemv;
#endif
  }
#endif
#if defined(HAS_DOUBLE)
  if (dtype == DOUBLE) {
    if (Environment::getInstance().blasFallback()) return false;

#if __EXTERNAL_BLAS__ || HAVE_OPENBLAS
    return true;
#else
    return _hasDgemv;
#endif
  }
#endif
  return false;
}

#if defined(HAS_FLOAT32)
template <>
bool BlasHelper::hasGEMM<float>() {
  if (Environment::getInstance().blasFallback()) return false;

#if __EXTERNAL_BLAS__ || HAVE_OPENBLAS
  return true;
#else
  return _hasSgemm;
#endif
}
#endif

#if defined(HAS_DOUBLE)
template <>
bool BlasHelper::hasGEMM<double>() {
  if (Environment::getInstance().blasFallback()) return false;

#if __EXTERNAL_BLAS__ || HAVE_OPENBLAS
  return true;
#else
  return _hasDgemm;
#endif
}
#endif

#if defined(HAS_FLOAT16)
template <>
bool BlasHelper::hasGEMM<float16>() {
  return false;
}
#endif

#if defined(HAS_BFLOAT16)
template <>
bool BlasHelper::hasGEMM<bfloat16>() {
  return false;
}
#endif

#if defined(HAS_INT32)
template <>
bool BlasHelper::hasGEMM<int>() {
  return false;
}
#endif

#if defined(HAS_UINT8)
template <>
bool BlasHelper::hasGEMM<uint8_t>() {
  return false;
}
#endif

#if defined(HAS_INT8)
template <>
bool BlasHelper::hasGEMM<int8_t>() {
  return false;
}
#endif

#if defined(HAS_INT16)
template <>
bool BlasHelper::hasGEMM<int16_t>() {
  return false;
}
#endif

#if defined(HAS_BOOL)
template <>
bool BlasHelper::hasGEMM<bool>() {
  return false;
}
#endif

#if defined(HAS_LONG)
template <>
bool BlasHelper::hasGEMM<LongType>() {
  return false;
}
#endif

bool BlasHelper::hasGEMM(const DataType dtype) {
#if defined(HAS_FLOAT32)
  if (dtype == FLOAT32) {
    if (Environment::getInstance().blasFallback()) return false;

#if __EXTERNAL_BLAS__ || HAVE_OPENBLAS
    return true;
#else
    return _hasSgemm;
#endif
  }
#endif
#if defined(HAS_DOUBLE)
  if (dtype == DOUBLE) {
    if (Environment::getInstance().blasFallback()) return false;

#if __EXTERNAL_BLAS__ || HAVE_OPENBLAS
    return true;
#else
    return _hasDgemm;
#endif
  }
#endif
  return false;
}

#if defined(HAS_FLOAT32)
template <>
bool BlasHelper::hasBatchedGEMM<float>() {
  if (Environment::getInstance().blasFallback()) return false;

  return _hasSgemmBatch;
}
#endif

#if defined(HAS_DOUBLE)
template <>
bool BlasHelper::hasBatchedGEMM<double>() {
  if (Environment::getInstance().blasFallback()) return false;

  return _hasDgemmBatch;
}
#endif

#if defined(HAS_FLOAT16)
template <>
bool BlasHelper::hasBatchedGEMM<float16>() {
  return false;
}
#endif

#if defined(HAS_BFLOAT16)
template <>
bool BlasHelper::hasBatchedGEMM<bfloat16>() {
  return false;
}
#endif

#if defined(HAS_LONG)
template <>
bool BlasHelper::hasBatchedGEMM<LongType>() {
  return false;
}
#endif

#if defined(HAS_INT32)
template <>
bool BlasHelper::hasBatchedGEMM<int>() {
  return false;
}
#endif

#if defined(HAS_INT8)
template <>
bool BlasHelper::hasBatchedGEMM<int8_t>() {
  return false;
}
#endif

#if defined(HAS_UINT8)
template <>
bool BlasHelper::hasBatchedGEMM<uint8_t>() {
  return false;
}
#endif

#if defined(HAS_INT16)
template <>
bool BlasHelper::hasBatchedGEMM<int16_t>() {
  return false;
}
#endif

#if defined(HAS_BOOL)
template <>
bool BlasHelper::hasBatchedGEMM<bool>() {
  return false;
}
#endif

#if !defined(SD_CUDA)
#if defined(HAS_FLOAT32)
CblasSgemv BlasHelper::sgemv() {
#if __EXTERNAL_BLAS__ || HAVE_OPENBLAS
  return (CblasSgemv)&cblas_sgemv;
#else
  return this->cblasSgemv;
#endif
}

CblasSgemm BlasHelper::sgemm() {
#if __EXTERNAL_BLAS__ || HAVE_OPENBLAS
  return (CblasSgemm)&cblas_sgemm;
#else
  return this->cblasSgemm;
#endif
}

CblasSgemmBatch BlasHelper::sgemmBatched() { return this->cblasSgemmBatch; }

LapackeSgesvd BlasHelper::sgesvd() { return this->lapackeSgesvd; }

LapackeSgesdd BlasHelper::sgesdd() { return this->lapackeSgesdd; }
#endif

#if defined(HAS_DOUBLE)
CblasDgemv BlasHelper::dgemv() {
#if __EXTERNAL_BLAS__ || HAVE_OPENBLAS
  return (CblasDgemv)&cblas_dgemv;
#else
  return this->cblasDgemv;
#endif
}

CblasDgemm BlasHelper::dgemm() {
#if __EXTERNAL_BLAS__ || HAVE_OPENBLAS
  return (CblasDgemm)&cblas_dgemm;
#else
  return this->cblasDgemm;
#endif
}

CblasDgemmBatch BlasHelper::dgemmBatched() { return this->cblasDgemmBatch; }

LapackeDgesvd BlasHelper::dgesvd() { return this->lapackeDgesvd; }

LapackeDgesdd BlasHelper::dgesdd() { return this->lapackeDgesdd; }
#endif
#endif

// BLAS call serialization implementation

std::unique_lock<std::mutex> BlasHelper::lockBlas() const {
  if (_serializeBlasCalls.load()) {
    return std::unique_lock<std::mutex>(_blasMutex);
  }
  // Return an unlocked lock if serialization is disabled
  return std::unique_lock<std::mutex>(_blasMutex, std::defer_lock);
}

bool BlasHelper::isSerializeBlasCalls() const {
  return _serializeBlasCalls.load();
}

void BlasHelper::setSerializeBlasCalls(bool serialize) {
  _serializeBlasCalls.store(serialize);
}

int BlasHelper::getOpenblasThreads() const {
  return _openblasThreads.load();
}

void BlasHelper::setOpenblasThreads(int threads) {
  _openblasThreads.store(threads);
#if HAVE_OPENBLAS
  if (threads > 0) {
    openblas_set_num_threads(threads);
    sd_debug("OpenBLAS threads set to %d\n", threads);
  }
#endif
}

void BlasHelper::initializeBlasThreading() {
  // Check SD_BLAS_SERIALIZE environment variable
  // Default is true (serialization enabled) for OpenBLAS safety
  const char* serializeEnv = std::getenv("SD_BLAS_SERIALIZE");
  if (serializeEnv != nullptr) {
    std::string val(serializeEnv);
    if (val == "0" || val == "false" || val == "FALSE" || val == "no" || val == "NO") {
      _serializeBlasCalls.store(false);
      sd_debug("BLAS call serialization DISABLED via SD_BLAS_SERIALIZE=%s\n", serializeEnv);
    } else {
      _serializeBlasCalls.store(true);
      sd_debug("BLAS call serialization ENABLED via SD_BLAS_SERIALIZE=%s\n", serializeEnv);
    }
  } else {
    // Default: enable serialization for OpenBLAS safety
    _serializeBlasCalls.store(true);
    sd_debug("BLAS call serialization ENABLED by default (set SD_BLAS_SERIALIZE=0 to disable)\n", "");
  }

  // Check SD_OPENBLAS_THREADS environment variable for OpenBLAS thread count
  // This is separate from the serialization - you can have both:
  // - Serialization ON + multi-threaded OpenBLAS = safe concurrent BLAS with internal parallelism
  // - Serialization OFF + single-threaded OpenBLAS = original behavior
  const char* threadsEnv = std::getenv("SD_OPENBLAS_THREADS");
  if (threadsEnv != nullptr) {
#ifdef __cpp_exceptions
    try {
      int threads = std::stoi(std::string(threadsEnv));
      if (threads > 0) {
        _openblasThreads.store(threads);
#if HAVE_OPENBLAS
        openblas_set_num_threads(threads);
        sd_debug("OpenBLAS threads set to %d via SD_OPENBLAS_THREADS\n", threads);
#endif
      }
    } catch (...) {
      // Invalid value, ignore
    }
#else
    int threads = std::atoi(threadsEnv);
    if (threads > 0) {
      _openblasThreads.store(threads);
#if HAVE_OPENBLAS
      openblas_set_num_threads(threads);
      sd_debug("OpenBLAS threads set to %d via SD_OPENBLAS_THREADS\n", threads);
#endif
    }
#endif
  }

  // Also check OPENBLAS_NUM_THREADS (standard OpenBLAS env var) if SD_OPENBLAS_THREADS not set
  if (_openblasThreads.load() == 0) {
    const char* openblasEnv = std::getenv("OPENBLAS_NUM_THREADS");
    if (openblasEnv != nullptr) {
#ifdef __cpp_exceptions
      try {
        int threads = std::stoi(std::string(openblasEnv));
        if (threads > 0) {
          _openblasThreads.store(threads);
          sd_debug("OpenBLAS threads detected from OPENBLAS_NUM_THREADS=%d\n", threads);
        }
      } catch (...) {
        // Invalid value, ignore
      }
#else
      int threads = std::atoi(openblasEnv);
      if (threads > 0) {
        _openblasThreads.store(threads);
        sd_debug("OpenBLAS threads detected from OPENBLAS_NUM_THREADS=%d\n", threads);
      }
#endif
    }
  }
}

// destructor
BlasHelper::~BlasHelper() noexcept {}
}  // namespace sd