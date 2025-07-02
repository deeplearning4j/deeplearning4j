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
namespace sd {
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

template <>
bool BlasHelper::hasGEMV<float>() {
  if (Environment::getInstance().blasFallback()) return false;

#if __EXTERNAL_BLAS__ || HAVE_OPENBLAS
  return true;
#else
  return _hasSgemv;
#endif
}

template <>
bool BlasHelper::hasGEMV<double>() {
  if (Environment::getInstance().blasFallback()) return false;

#if __EXTERNAL_BLAS__ || HAVE_OPENBLAS
  return true;
#else
  return _hasDgemv;
#endif
}

template <>
bool BlasHelper::hasGEMV<float16>() {
  return false;
}

template <>
bool BlasHelper::hasGEMV<bfloat16>() {
  return false;
}

template <>
bool BlasHelper::hasGEMV<bool>() {
  return false;
}

template <>
bool BlasHelper::hasGEMV<int>() {
  return false;
}

template <>
bool BlasHelper::hasGEMV<int8_t>() {
  return false;
}

template <>
bool BlasHelper::hasGEMV<uint8_t>() {
  return false;
}

template <>
bool BlasHelper::hasGEMV<int16_t>() {
  return false;
}

template <>
bool BlasHelper::hasGEMV<LongType>() {
  return false;
}

bool BlasHelper::hasGEMV(const DataType dtype) {
  if (dtype == FLOAT32) {
    if (Environment::getInstance().blasFallback()) return false;

#if __EXTERNAL_BLAS__ || HAVE_OPENBLAS
    return true;
#else
    return _hasSgemv;
#endif
  }
  if (dtype == DOUBLE) {
    if (Environment::getInstance().blasFallback()) return false;

#if __EXTERNAL_BLAS__ || HAVE_OPENBLAS
    return true;
#else
    return _hasDgemv;
#endif
  }
  return false;
}

template <>
bool BlasHelper::hasGEMM<float>() {
  if (Environment::getInstance().blasFallback()) return false;

#if __EXTERNAL_BLAS__ || HAVE_OPENBLAS
  return true;
#else
  return _hasSgemm;
#endif
}

template <>
bool BlasHelper::hasGEMM<double>() {
  if (Environment::getInstance().blasFallback()) return false;

#if __EXTERNAL_BLAS__ || HAVE_OPENBLAS
  return true;
#else
  return _hasDgemm;
#endif
}

template <>
bool BlasHelper::hasGEMM<float16>() {
  return false;
}

template <>
bool BlasHelper::hasGEMM<bfloat16>() {
  return false;
}

template <>
bool BlasHelper::hasGEMM<int>() {
  return false;
}

template <>
bool BlasHelper::hasGEMM<uint8_t>() {
  return false;
}

template <>
bool BlasHelper::hasGEMM<int8_t>() {
  return false;
}

template <>
bool BlasHelper::hasGEMM<int16_t>() {
  return false;
}

template <>
bool BlasHelper::hasGEMM<bool>() {
  return false;
}

template <>
bool BlasHelper::hasGEMM<LongType>() {
  return false;
}

bool BlasHelper::hasGEMM(const DataType dtype) {
  if (dtype == FLOAT32) {
    if (Environment::getInstance().blasFallback()) return false;

#if __EXTERNAL_BLAS__ || HAVE_OPENBLAS
    return true;
#else
    return _hasSgemm;
#endif
  }
  if (dtype == DOUBLE) {
    if (Environment::getInstance().blasFallback()) return false;

#if __EXTERNAL_BLAS__ || HAVE_OPENBLAS
    return true;
#else
    return _hasDgemm;
#endif
  }
  return false;
}

template <>
bool BlasHelper::hasBatchedGEMM<float>() {
  if (Environment::getInstance().blasFallback()) return false;

  return _hasSgemmBatch;
}

template <>
bool BlasHelper::hasBatchedGEMM<double>() {
  if (Environment::getInstance().blasFallback()) return false;

  return _hasDgemmBatch;
}

template <>
bool BlasHelper::hasBatchedGEMM<float16>() {
  return false;
}

template <>
bool BlasHelper::hasBatchedGEMM<bfloat16>() {
  return false;
}

template <>
bool BlasHelper::hasBatchedGEMM<LongType>() {
  return false;
}

template <>
bool BlasHelper::hasBatchedGEMM<int>() {
  return false;
}

template <>
bool BlasHelper::hasBatchedGEMM<int8_t>() {
  return false;
}

template <>
bool BlasHelper::hasBatchedGEMM<uint8_t>() {
  return false;
}

template <>
bool BlasHelper::hasBatchedGEMM<int16_t>() {
  return false;
}

template <>
bool BlasHelper::hasBatchedGEMM<bool>() {
  return false;
}
#if !defined(SD_CUDA)
CblasSgemv BlasHelper::sgemv() {
#if __EXTERNAL_BLAS__ || HAVE_OPENBLAS
  return (CblasSgemv)&cblas_sgemv;
#else
  return this->cblasSgemv;
#endif
}
CblasDgemv BlasHelper::dgemv() {
#if __EXTERNAL_BLAS__ || HAVE_OPENBLAS
  return (CblasDgemv)&cblas_dgemv;
#else
  return this->cblasDgemv;
#endif
}

CblasSgemm BlasHelper::sgemm() {
#if __EXTERNAL_BLAS__ || HAVE_OPENBLAS
  return (CblasSgemm)&cblas_sgemm;
#else
  return this->cblasSgemm;
#endif
}

CblasDgemm BlasHelper::dgemm() {
#if __EXTERNAL_BLAS__ || HAVE_OPENBLAS
  return (CblasDgemm)&cblas_dgemm;
#else
  return this->cblasDgemm;
#endif
}

CblasSgemmBatch BlasHelper::sgemmBatched() { return this->cblasSgemmBatch; }

CblasDgemmBatch BlasHelper::dgemmBatched() { return this->cblasDgemmBatch; }

LapackeSgesvd BlasHelper::sgesvd() { return this->lapackeSgesvd; }

LapackeDgesvd BlasHelper::dgesvd() { return this->lapackeDgesvd; }

LapackeSgesdd BlasHelper::sgesdd() { return this->lapackeSgesdd; }

LapackeDgesdd BlasHelper::dgesdd() { return this->lapackeDgesdd; }
#endif
// destructor
BlasHelper::~BlasHelper() noexcept {}
}  // namespace sd
