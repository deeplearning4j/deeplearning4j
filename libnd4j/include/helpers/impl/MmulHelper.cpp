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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 05.06.2018
//

#ifndef LIBND4J_MMULHELPER_CPP
#define LIBND4J_MMULHELPER_CPP
#include "../MmulHelper.h"

#include <array/DataBuffer.h>
#include <array/NDArrayFactory.h>
#include <execution/Threads.h>
#include <helpers/BlasHelper.h>
#include <helpers/ShapeUtils.h>
#include <system/Environment.h>
#include <ops/declarable/headers/shape.h>
#include <ops/declarable/helpers/batched_gemm.h>
#include <system/openmp_pragmas.h>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>

#include "ops/declarable/headers/blas.h"

#if !defined(__STANDALONE_BUILD__)
#include "config.h"
#endif

#if HAVE_ONEDNN
#include <dnnl.hpp>
#include <ops/declarable/platform/mkldnn/mkldnnUtils.h>
#endif

#if defined(HAVE_MKL) && !defined(SD_CUDA)
#include <helpers/MklBlasHelper.h>
#endif

namespace sd {

// ============================================================================
// BATCHED MATMUL BACKEND DISPATCH
// ============================================================================

#if HAVE_ONEDNN
//////////////////////////////////////////////////////////////////////////
bool MmulHelper::tryOneDnnBatched(NDArray* A, NDArray* B, NDArray* C,
                                   double alpha, double beta) {
  const auto aRank = A->rankOf();
  const auto bRank = B->rankOf();
  const auto cRank = C->rankOf();

  // Handle 3D and 4D batched cases
  if ((aRank != 3 && aRank != 4) || aRank != bRank || bRank != cRank) {
    return false;
  }

  if (!Environment::getInstance().helpersAllowed()) {
    return false;
  }

  const auto xType = A->dataType();
  const auto yType = B->dataType();
  const auto zType = C->dataType();

  // Check supported types
  if (xType != yType || yType != zType) {
    return false;
  }
  if (xType != DataType::FLOAT32 && xType != DataType::BFLOAT16 && xType != DataType::HALF) {
    return false;
  }

  // Calculate batch size (flatten all leading dimensions)
  LongType bS = 1;
  for (int i = 0; i < aRank - 2; ++i) {
    bS *= A->sizeAt(i);
  }
  if (bS < 1) {
    return false;
  }

  const LongType M = A->sizeAt(-2);
  const LongType K = A->sizeAt(-1);
  const LongType N = B->sizeAt(-1);

  // Verify shape compatibility
  if (B->sizeAt(-2) != K) {
    return false;
  }
  if (C->sizeAt(-2) != M || C->sizeAt(-1) != N) {
    return false;
  }

  // Check if arrays are contiguous enough for OneDNN
  // For 3D: stride[0] should be >= M*K, stride[1] >= K, stride[2] == 1
  // For 4D: we need contiguous batch dims
  bool aContiguous, bContiguous, cContiguous;

  if (aRank == 3) {
    aContiguous = (A->strideAt(2) == 1) && (A->strideAt(1) >= K) && (A->strideAt(0) >= M * K);
    bContiguous = (B->strideAt(2) == 1) && (B->strideAt(1) >= N) && (B->strideAt(0) >= K * N);
    cContiguous = (C->strideAt(2) == 1) && (C->strideAt(1) >= N) && (C->strideAt(0) >= M * N);
  } else {
    // For 4D, check if it's effectively 3D (contiguous batch)
    LongType bS0 = A->sizeAt(0);
    LongType bS1 = A->sizeAt(1);
    aContiguous = (A->strideAt(3) == 1) && (A->strideAt(2) >= K) &&
                  (A->strideAt(1) >= M * K) && (A->strideAt(0) >= bS1 * M * K);
    bContiguous = (B->strideAt(3) == 1) && (B->strideAt(2) >= N) &&
                  (B->strideAt(1) >= K * N) && (B->strideAt(0) >= bS1 * K * N);
    cContiguous = (C->strideAt(3) == 1) && (C->strideAt(2) >= N) &&
                  (C->strideAt(1) >= M * N) && (C->strideAt(0) >= bS1 * M * N);
  }

  if (!aContiguous || !bContiguous || !cContiguous) {
    return false;
  }

  try {
    auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
    dnnl::stream stream(engine);

    dnnl::memory::data_type dtype;
    if (xType == DataType::FLOAT32)
      dtype = dnnl::memory::data_type::f32;
    else if (xType == DataType::BFLOAT16)
      dtype = dnnl::memory::data_type::bf16;
    else if (xType == DataType::HALF)
      dtype = dnnl::memory::data_type::f16;
    else
      return false;

    // Use 3D view regardless of original rank (flattened batch)
    dnnl::memory::dims aShape = {bS, M, K};
    dnnl::memory::dims bShape = {bS, K, N};
    dnnl::memory::dims cShape = {bS, M, N};

    // Calculate effective strides for 3D view
    dnnl::memory::dims aStrides, bStrides, cStrides;
    if (aRank == 3) {
      aStrides = {A->strideAt(0), A->strideAt(1), A->strideAt(2)};
      bStrides = {B->strideAt(0), B->strideAt(1), B->strideAt(2)};
      cStrides = {C->strideAt(0), C->strideAt(1), C->strideAt(2)};
    } else {
      // For 4D, calculate batch stride as product of inner batch and matrix strides
      aStrides = {M * K, A->strideAt(2), A->strideAt(3)};
      bStrides = {K * N, B->strideAt(2), B->strideAt(3)};
      cStrides = {M * N, C->strideAt(2), C->strideAt(3)};
    }

    auto a_md = dnnl::memory::desc(aShape, dtype, aStrides);
    auto b_md = dnnl::memory::desc(bShape, dtype, bStrides);
    auto c_md = dnnl::memory::desc(cShape, dtype, cStrides);

    // Configure alpha/beta via post_ops
    dnnl::primitive_attr attr;
    dnnl::post_ops po;

    if (alpha != 1.0) {
      po.append_eltwise(dnnl::algorithm::eltwise_linear, static_cast<float>(alpha), 0.f);
    }
    if (beta != 0.0) {
      po.append_sum(static_cast<float>(beta));
    }
    if (alpha != 1.0 || beta != 0.0) {
      attr.set_post_ops(po);
    }

    dnnl::matmul::primitive_desc matmul_pd(engine, a_md, b_md, c_md, attr);

    dnnl::memory a_mem(a_md, engine, A->buffer());
    dnnl::memory b_mem(b_md, engine, B->buffer());
    dnnl::memory c_mem(c_md, engine, C->buffer());

    std::unordered_map<int, dnnl::memory> args;
    args[DNNL_ARG_SRC] = a_mem;
    args[DNNL_ARG_WEIGHTS] = b_mem;
    args[DNNL_ARG_DST] = c_mem;

    dnnl::matmul(matmul_pd).execute(stream, args);
    stream.wait();

    return true;
  } catch (const dnnl::error& e) {
    return false;
  }
}
#else
bool MmulHelper::tryOneDnnBatched(NDArray* A, NDArray* B, NDArray* C,
                                   double alpha, double beta) {
  return false;  // OneDNN not available
}
#endif

//////////////////////////////////////////////////////////////////////////
// Strided batch GEMM - most efficient for contiguous batched data
// Uses cblas_sgemm_batch_strided/cblas_dgemm_batch_strided (MKL on CPU)
// Only handles simple row-major contiguous case for reliability
// Non-standard layouts fall through to tryBlasBatched/tryBlasPerBatch
//////////////////////////////////////////////////////////////////////////
#if !defined(SD_CUDA)
bool MmulHelper::tryBlasStridedBatched(NDArray* A, NDArray* B, NDArray* C,
                                        double alpha, double beta) {
#if defined(HAVE_MKL)
  if (!Environment::getInstance().isEnableBlas()) {
    return false;
  }

  // Check if strided batch GEMM is available
  if (!BlasHelper::getInstance().hasStridedBatchGEMM()) {
    return false;
  }

  const int aRank = A->rankOf();
  const int bRank = B->rankOf();
  const int cRank = C->rankOf();

  // Only handle 3D tensors for simplicity (most common case)
  if (aRank != 3 || bRank != 3 || cRank != 3) {
    return false;
  }

  const auto xType = A->dataType();
  const auto yType = B->dataType();
  const auto zType = C->dataType();

  // Types must match
  if (xType != yType || yType != zType) {
    return false;
  }

  // Only float and double supported for strided batch
  if (xType != DataType::FLOAT32 && xType != DataType::DOUBLE) {
    return false;
  }

  // Validate buffers
  if (A->buffer() == nullptr || B->buffer() == nullptr || C->buffer() == nullptr) {
    return false;
  }

  // Get dimensions: A[bS,M,K] @ B[bS,K,N] = C[bS,M,N]
  const int batchSize = static_cast<int>(A->sizeAt(0));
  const int M = static_cast<int>(A->sizeAt(1));
  const int K = static_cast<int>(A->sizeAt(2));
  const int K2 = static_cast<int>(B->sizeAt(1));
  const int N = static_cast<int>(B->sizeAt(2));

  // Validate dimensions
  if (M <= 0 || K <= 0 || N <= 0 || batchSize <= 0 || K != K2) {
    return false;
  }

  // Require strict row-major contiguous layout for all matrices
  // A: stride[2]=1, stride[1]=K, stride[0]=M*K
  // B: stride[2]=1, stride[1]=N, stride[0]=K*N
  // C: stride[2]=1, stride[1]=N, stride[0]=M*N
  const LongType expectedStrideA = static_cast<LongType>(M) * K;
  const LongType expectedStrideB = static_cast<LongType>(K) * N;
  const LongType expectedStrideC = static_cast<LongType>(M) * N;

  const bool aValid = (A->strideAt(2) == 1) && (A->strideAt(1) == K) && (A->strideAt(0) == expectedStrideA);
  const bool bValid = (B->strideAt(2) == 1) && (B->strideAt(1) == N) && (B->strideAt(0) == expectedStrideB);
  const bool cValid = (C->strideAt(2) == 1) && (C->strideAt(1) == N) && (C->strideAt(0) == expectedStrideC);

  if (!aValid || !bValid || !cValid) {
    return false;  // Non-contiguous or non-row-major - let other backends handle it
  }

  // Use MKL's strided batch GEMM
  auto blasLock = BlasHelper::getInstance().lockBlas();

  if (xType == DataType::FLOAT32) {
    sd::mkl::sgemmBatchStrided(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K,
        static_cast<float>(alpha),
        A->bufferAsT<float>(), K, static_cast<MKL_INT>(expectedStrideA),
        B->bufferAsT<float>(), N, static_cast<MKL_INT>(expectedStrideB),
        static_cast<float>(beta),
        C->bufferAsT<float>(), N, static_cast<MKL_INT>(expectedStrideC),
        batchSize);
  } else {
    sd::mkl::dgemmBatchStrided(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K,
        alpha,
        A->bufferAsT<double>(), K, static_cast<MKL_INT>(expectedStrideA),
        B->bufferAsT<double>(), N, static_cast<MKL_INT>(expectedStrideB),
        beta,
        C->bufferAsT<double>(), N, static_cast<MKL_INT>(expectedStrideC),
        batchSize);
  }

  return true;
#else
  // MKL not available - strided batch not supported here
  return false;
#endif
}
#endif  // !SD_CUDA

//////////////////////////////////////////////////////////////////////////
#if !defined(SD_CUDA)
bool MmulHelper::tryBlasBatched(NDArray* A, NDArray* B, NDArray* C,
                                 double alpha, double beta) {
  if (!Environment::getInstance().isEnableBlas()) {
    return false;
  }

  const int aRank = A->rankOf();
  const int bRank = B->rankOf();
  const int cRank = C->rankOf();

  // Only handle 3D and 4D
  if (aRank < 3 || aRank > 4 || aRank != bRank || bRank != cRank) {
    return false;
  }

  const auto xType = A->dataType();
  const auto yType = B->dataType();
  const auto zType = C->dataType();

  // Types must match
  if (xType != yType || yType != zType) {
    return false;
  }

  // Only float and double supported
  if (xType != DataType::FLOAT32 && xType != DataType::DOUBLE) {
    return false;
  }

  // Check if batched GEMM is available
  const bool hasFloat = (xType == DataType::FLOAT32) && BlasHelper::getInstance().hasBatchedGEMM<float>();
  const bool hasDouble = (xType == DataType::DOUBLE) && BlasHelper::getInstance().hasBatchedGEMM<double>();

  if (!hasFloat && !hasDouble) {
    return false;
  }

  // Check function pointers
  if (hasFloat && BlasHelper::getInstance().sgemmBatched() == nullptr) {
    return false;
  }
  if (hasDouble && BlasHelper::getInstance().dgemmBatched() == nullptr) {
    return false;
  }

  // Compute dimensions
  LongType batchSize = 1;
  for (int i = 0; i < aRank - 2; ++i) {
    batchSize *= A->sizeAt(i);
  }

  const int M = static_cast<int>(A->sizeAt(-2));
  const int K = static_cast<int>(A->sizeAt(-1));
  const int N = static_cast<int>(B->sizeAt(-1));

  // Validate buffers
  if (A->buffer() == nullptr || B->buffer() == nullptr || C->buffer() == nullptr) {
    return false;
  }
  if (M <= 0 || K <= 0 || N <= 0) {
    return false;
  }

  // Check for row-major contiguous layout
  const bool aRowMajor = (A->strideAt(-1) == 1) && (A->strideAt(-2) == K);
  const bool bRowMajor = (B->strideAt(-1) == 1) && (B->strideAt(-2) == N);
  const bool cRowMajor = (C->strideAt(-1) == 1) && (C->strideAt(-2) == N);

  if (!aRowMajor || !bRowMajor || !cRowMajor) {
    return false;
  }

  const int batchCount = static_cast<int>(batchSize);
  const LongType strideA = static_cast<LongType>(M) * K;
  const LongType strideB = static_cast<LongType>(K) * N;
  const LongType strideC = static_cast<LongType>(M) * N;

  // Prepare batch arrays
  std::vector<CBLAS_TRANSPOSE> transA_arr(batchCount, CblasNoTrans);
  std::vector<CBLAS_TRANSPOSE> transB_arr(batchCount, CblasNoTrans);
  std::vector<int> M_arr(batchCount, M);
  std::vector<int> N_arr(batchCount, N);
  std::vector<int> K_arr(batchCount, K);
  std::vector<int> lda_arr(batchCount, K);
  std::vector<int> ldb_arr(batchCount, N);
  std::vector<int> ldc_arr(batchCount, N);

  int groupSize = batchCount;
  auto blasLock = BlasHelper::getInstance().lockBlas();

  if (hasFloat) {
    std::vector<float*> A_arr(batchCount);
    std::vector<float*> B_arr(batchCount);
    std::vector<float*> C_arr(batchCount);
    std::vector<float> alpha_arr(batchCount, static_cast<float>(alpha));
    std::vector<float> beta_arr(batchCount, static_cast<float>(beta));

    float* aPtr = A->bufferAsT<float>();
    float* bPtr = B->bufferAsT<float>();
    float* cPtr = C->bufferAsT<float>();

    for (int b = 0; b < batchCount; ++b) {
      A_arr[b] = aPtr + b * strideA;
      B_arr[b] = bPtr + b * strideB;
      C_arr[b] = cPtr + b * strideC;
    }

    BlasHelper::getInstance().sgemmBatched()(
        CblasRowMajor, transA_arr.data(), transB_arr.data(),
        M_arr.data(), N_arr.data(), K_arr.data(),
        alpha_arr.data(), A_arr.data(), lda_arr.data(),
        B_arr.data(), ldb_arr.data(),
        beta_arr.data(), C_arr.data(), ldc_arr.data(),
        1, &groupSize);
  } else {
    std::vector<double*> A_arr(batchCount);
    std::vector<double*> B_arr(batchCount);
    std::vector<double*> C_arr(batchCount);
    std::vector<double> alpha_arr(batchCount, alpha);
    std::vector<double> beta_arr(batchCount, beta);

    double* aPtr = A->bufferAsT<double>();
    double* bPtr = B->bufferAsT<double>();
    double* cPtr = C->bufferAsT<double>();

    for (int b = 0; b < batchCount; ++b) {
      A_arr[b] = aPtr + b * strideA;
      B_arr[b] = bPtr + b * strideB;
      C_arr[b] = cPtr + b * strideC;
    }

    BlasHelper::getInstance().dgemmBatched()(
        CblasRowMajor, transA_arr.data(), transB_arr.data(),
        M_arr.data(), N_arr.data(), K_arr.data(),
        alpha_arr.data(), A_arr.data(), lda_arr.data(),
        B_arr.data(), ldb_arr.data(),
        beta_arr.data(), C_arr.data(), ldc_arr.data(),
        1, &groupSize);
  }

  return true;
}

//////////////////////////////////////////////////////////////////////////
bool MmulHelper::tryBlasPerBatch(NDArray* A, NDArray* B, NDArray* C,
                                  double alpha, double beta) {
  if (!Environment::getInstance().isEnableBlas()) {
    return false;
  }

  const int aRank = A->rankOf();
  const int bRank = B->rankOf();
  const int cRank = C->rankOf();

  // Handle 3D and 4D
  if ((aRank != 3 && aRank != 4) || aRank != bRank || bRank != cRank) {
    return false;
  }

  const auto xType = A->dataType();
  const auto yType = B->dataType();
  const auto zType = C->dataType();

  if (xType != yType || yType != zType) {
    return false;
  }
  if (xType != DataType::FLOAT32 && xType != DataType::DOUBLE) {
    return false;
  }
  if (!BlasHelper::getInstance().hasGEMM(xType)) {
    return false;
  }

  const int M = static_cast<int>(A->sizeAt(-2));
  const int K = static_cast<int>(A->sizeAt(-1));
  const int N = static_cast<int>(B->sizeAt(-1));

  // Check row-major contiguous for last 2 dimensions
  const bool aRowMajor = (A->strideAt(-1) == 1) && (A->strideAt(-2) == K);
  const bool bRowMajor = (B->strideAt(-1) == 1) && (B->strideAt(-2) == N);
  const bool cRowMajor = (C->strideAt(-1) == 1) && (C->strideAt(-2) == N);

  if (!aRowMajor || !bRowMajor || !cRowMajor) {
    return false;
  }

  if (A->buffer() == nullptr || B->buffer() == nullptr || C->buffer() == nullptr) {
    return false;
  }

  // Calculate total batch size (flatten leading dimensions)
  LongType batchSize = 1;
  for (int i = 0; i < aRank - 2; ++i) {
    batchSize *= A->sizeAt(i);
  }

  const LongType strideA = static_cast<LongType>(M) * K;
  const LongType strideB = static_cast<LongType>(K) * N;
  const LongType strideC = static_cast<LongType>(M) * N;

  auto& blasHelper = BlasHelper::getInstance();
  auto blasLock = blasHelper.lockBlas();

  if (xType == DataType::FLOAT32) {
    float* aPtr = A->bufferAsT<float>();
    float* bPtr = B->bufferAsT<float>();
    float* cPtr = C->bufferAsT<float>();
    const float alphaF = static_cast<float>(alpha);
    const float betaF = static_cast<float>(beta);

    for (LongType b = 0; b < batchSize; ++b) {
      blasHelper.sgemm()(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                         M, N, K, alphaF,
                         aPtr + b * strideA, K,
                         bPtr + b * strideB, N,
                         betaF,
                         cPtr + b * strideC, N);
    }
  } else {
    double* aPtr = A->bufferAsT<double>();
    double* bPtr = B->bufferAsT<double>();
    double* cPtr = C->bufferAsT<double>();

    for (LongType b = 0; b < batchSize; ++b) {
      blasHelper.dgemm()(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                         M, N, K, alpha,
                         aPtr + b * strideA, K,
                         bPtr + b * strideB, N,
                         beta,
                         cPtr + b * strideC, N);
    }
  }

  return true;
}
#else
// CUDA stubs - batched BLAS not available on CUDA through this path
bool MmulHelper::tryBlasBatched(NDArray* A, NDArray* B, NDArray* C,
                                 double alpha, double beta) {
  return false;
}

bool MmulHelper::tryBlasPerBatch(NDArray* A, NDArray* B, NDArray* C,
                                  double alpha, double beta) {
  return false;
}
#endif

//////////////////////////////////////////////////////////////////////////
void MmulHelper::manualBatchedGemm(NDArray* A, NDArray* B, NDArray* C,
                                    double alpha, double beta) {
  const int aRank = A->rankOf();
  const auto xType = A->dataType();

  // Calculate total batch size and matrix dimensions
  LongType totalBatchSize = 1;
  for (int i = 0; i < aRank - 2; ++i) {
    totalBatchSize *= A->sizeAt(i);
  }

  const int M = static_cast<int>(A->sizeAt(-2));
  const int K = static_cast<int>(A->sizeAt(-1));
  const int N = static_cast<int>(B->sizeAt(-1));

  // Check row-major for fast path (last 2 dims)
  const bool aRowMajor = (A->strideAt(-1) == 1) && (A->strideAt(-2) == K);
  const bool bRowMajor = (B->strideAt(-1) == 1) && (B->strideAt(-2) == N);
  const bool cRowMajor = (C->strideAt(-1) == 1) && (C->strideAt(-2) == N);

  // Check if batch dimensions are also contiguous
  bool batchContiguous = true;
  if (aRank == 3) {
    batchContiguous = (A->strideAt(0) >= M * K) && (B->strideAt(0) >= K * N) && (C->strideAt(0) >= M * N);
  } else if (aRank == 4) {
    LongType bS1 = A->sizeAt(1);
    batchContiguous = (A->strideAt(0) >= bS1 * M * K) && (A->strideAt(1) >= M * K) &&
                      (B->strideAt(0) >= bS1 * K * N) && (B->strideAt(1) >= K * N) &&
                      (C->strideAt(0) >= bS1 * M * N) && (C->strideAt(1) >= M * N);
  } else {
    batchContiguous = false;  // For ranks > 4, use slice-based approach
  }

  if (aRowMajor && bRowMajor && cRowMajor && batchContiguous &&
      (xType == DataType::FLOAT32 || xType == DataType::DOUBLE)) {
    // Optimized parallel implementation
    const LongType totalRows = totalBatchSize * M;
    const LongType strideA = static_cast<LongType>(M) * K;
    const LongType strideB = static_cast<LongType>(K) * N;
    const LongType strideC = static_cast<LongType>(M) * N;

    if (xType == DataType::FLOAT32) {
      float* aPtr = A->bufferAsT<float>();
      float* bPtr = B->bufferAsT<float>();
      float* cPtr = C->bufferAsT<float>();
      const float alphaF = static_cast<float>(alpha);
      const float betaF = static_cast<float>(beta);
      const bool betaPresent = (beta != 0.0);

      auto func = PRAGMA_THREADS_FOR {
        for (auto rowIdx = start; rowIdx < stop; ++rowIdx) {
          const LongType b = rowIdx / M;
          const LongType m = rowIdx % M;
          float* aRow = aPtr + b * strideA + m * K;
          float* bBase = bPtr + b * strideB;
          float* cRow = cPtr + b * strideC + m * N;

          if (betaPresent) {
            PRAGMA_OMP_SIMD
            for (int n = 0; n < N; ++n) {
              cRow[n] *= betaF;
            }
          } else {
            PRAGMA_OMP_SIMD
            for (int n = 0; n < N; ++n) {
              cRow[n] = 0.0f;
            }
          }

          for (int k = 0; k < K; ++k) {
            const float aVal = alphaF * aRow[k];
            const float* bRow = bBase + k * N;
            PRAGMA_OMP_SIMD
            for (int n = 0; n < N; ++n) {
              cRow[n] += aVal * bRow[n];
            }
          }
        }
      };
      samediff::Threads::parallel_tad(func, 0, totalRows);
    } else {
      double* aPtr = A->bufferAsT<double>();
      double* bPtr = B->bufferAsT<double>();
      double* cPtr = C->bufferAsT<double>();
      const bool betaPresent = (beta != 0.0);

      auto func = PRAGMA_THREADS_FOR {
        for (auto rowIdx = start; rowIdx < stop; ++rowIdx) {
          const LongType b = rowIdx / M;
          const LongType m = rowIdx % M;
          double* aRow = aPtr + b * strideA + m * K;
          double* bBase = bPtr + b * strideB;
          double* cRow = cPtr + b * strideC + m * N;

          if (betaPresent) {
            PRAGMA_OMP_SIMD
            for (int n = 0; n < N; ++n) {
              cRow[n] *= beta;
            }
          } else {
            PRAGMA_OMP_SIMD
            for (int n = 0; n < N; ++n) {
              cRow[n] = 0.0;
            }
          }

          for (int k = 0; k < K; ++k) {
            const double aVal = alpha * aRow[k];
            const double* bRow = bBase + k * N;
            PRAGMA_OMP_SIMD
            for (int n = 0; n < N; ++n) {
              cRow[n] += aVal * bRow[n];
            }
          }
        }
      };
      samediff::Threads::parallel_tad(func, 0, totalRows);
    }
  } else {
    // Generic slice-based fallback for non-contiguous or unsupported types
    const LongType outerBatchSize = A->sizeAt(0);
    for (LongType b = 0; b < outerBatchSize; ++b) {
      auto aSlice = (*A)(b, {0});
      auto bSlice = (*B)(b, {0});
      auto cSlice = (*C)(b, {0});
      mmul(aSlice, bSlice, cSlice, alpha, beta);
      delete aSlice;
      delete bSlice;
      delete cSlice;
    }
  }
}

//////////////////////////////////////////////////////////////////////////
bool MmulHelper::mmulBatched(NDArray* A, NDArray* B, NDArray* C,
                              double alpha, double beta) {
  // Try backends in order of preference:
  // 1. OneDNN (single batched primitive - most efficient for bf16/f16)
  if (tryOneDnnBatched(A, B, C, alpha, beta)) {
    return true;
  }

  // 2. MKL strided batch GEMM (cblas_sgemm_batch_strided) - most efficient for f32/f64
  //    This is faster than pointer-array batch GEMM as it avoids building pointer arrays
  if (tryBlasStridedBatched(A, B, C, alpha, beta)) {
    return true;
  }

  // 3. BLAS pointer-array batched GEMM (cblas_sgemm_batch)
  if (tryBlasBatched(A, B, C, alpha, beta)) {
    return true;
  }

  // 4. Per-batch BLAS calls
  if (tryBlasPerBatch(A, B, C, alpha, beta)) {
    return true;
  }

  // 5. Manual parallel implementation (always works)
  manualBatchedGemm(A, B, C, alpha, beta);
  return true;
}

// ============================================================================
// TENSOR DOT OPERATIONS
// ============================================================================

//////////////////////////////////////////////////////////////////////////
NDArray* MmulHelper::tensorDot(NDArray* A, NDArray* B,
                               const std::initializer_list<LongType>& axesA,
                               const std::initializer_list<LongType>& axesB) {
  std::vector<LongType> aA(axesA);
  std::vector<LongType> aB(axesB);
  return tensorDot(A, B, aA, aB);
}

//////////////////////////////////////////////////////////////////////////
NDArray* MmulHelper::tensorDot(NDArray* A, NDArray* B, const std::vector<LongType>& axesA,
                               const std::vector<LongType>& axesB) {
  std::vector<LongType> permutAt, permutBt;
  std::vector<LongType> shapeAt, shapeBt;

  auto outShape = ShapeUtils::evalShapeForTensorDot(A, B, axesA, axesB, permutAt, permutBt, shapeAt, shapeBt);

  NDArray* aP = permutAt.empty() ? A : A->permute(permutAt, false, false);
  NDArray* bP = permutBt.empty() ? B : B->permute(permutBt, false, false);

  NDArray* aPR = aP->isSameShape(shapeAt) ? aP : aP->reshape(aP->ordering(), shapeAt);
  NDArray* bPR = bP->isSameShape(shapeAt) ? bP : bP->reshape(bP->ordering(), shapeBt);

  NDArray* c = mmul(aPR, bPR, nullptr, 1.0, 0.0);
  c->reshapei(outShape);

  if (aPR != A && aPR != aP) delete aPR;
  if (bPR != B && bPR != bP) delete bPR;
  if (aP != A) delete aP;
  if (bP != B) delete bP;

  return c;
}

//////////////////////////////////////////////////////////////////////////
void MmulHelper::computeNewShapesAndAxes(
    NDArray& as_, const std::vector<LongType>& axes_a,
    NDArray& bs, const std::vector<LongType>& axes_b,
    std::vector<LongType>& newshape_a, std::vector<LongType>& newaxes_a,
    std::vector<LongType>& newshape_b, std::vector<LongType>& newaxes_b) {

  const int aRank = as_.rankOf();
  const int bRank = bs.rankOf();

  std::vector<LongType> notin_a;
  for (int k = 0; k < aRank; ++k) {
    if (std::find(axes_a.begin(), axes_a.end(), k) == axes_a.end())
      notin_a.push_back(k);
  }

  newaxes_a.clear();
  std::copy(notin_a.begin(), notin_a.end(), std::back_inserter(newaxes_a));
  std::copy(axes_a.begin(), axes_a.end(), std::back_inserter(newaxes_a));

  LongType N2_a = std::accumulate(axes_a.begin(), axes_a.end(), 1L, [&](LongType product, LongType i) {
    return product * as_.sizeAt(i);
  });

  newshape_a.clear();
  newshape_a.push_back(std::accumulate(notin_a.begin(), notin_a.end(), 1L, [&](LongType product, LongType i) {
    return product * as_.sizeAt(i);
  }));
  newshape_a.push_back(N2_a);

  std::vector<LongType> notin_b;
  for (int k = 0; k < bRank; ++k) {
    if (std::find(axes_b.begin(), axes_b.end(), k) == axes_b.end())
      notin_b.push_back(k);
  }

  newaxes_b.clear();
  std::copy(axes_b.begin(), axes_b.end(), std::back_inserter(newaxes_b));
  std::copy(notin_b.begin(), notin_b.end(), std::back_inserter(newaxes_b));

  LongType N2_b = std::accumulate(axes_b.begin(), axes_b.end(), 1L, [&](LongType product, LongType i) {
    return product * bs.sizeAt(i);
  });

  newshape_b.clear();
  newshape_b.push_back(N2_b);
  newshape_b.push_back(std::accumulate(notin_b.begin(), notin_b.end(), 1L, [&](LongType product, LongType i) {
    return product * bs.sizeAt(i);
  }));
}

//////////////////////////////////////////////////////////////////////////
void MmulHelper::tensorDot2(NDArray* a, NDArray* b, NDArray* c, const std::vector<LongType>& axes_a,
                            const std::vector<LongType>& axes_b, std::vector<LongType>& permutAt,
                            std::vector<LongType>& permuteBt, std::vector<LongType>& permuteCt,
                            NDArray* realFinalResult) {
  NDArray* cP = permuteCt.empty() ? c : c->permute(permuteCt, false, false);

  std::vector<LongType> newshape_a, newaxes_a, newshape_b, newaxes_b;
  computeNewShapesAndAxes(*a, axes_a, *b, axes_b, newshape_a, newaxes_a, newshape_b, newaxes_b);

  NDArray* aP = permutAt.empty() ? a : a->permute(permutAt, false, false);
  NDArray* bP = permuteBt.empty() ? b : b->permute(permuteBt, false, false);

  NDArray* aPermuted = aP->permute(newaxes_a, false, false);
  NDArray* aPR = aPermuted->reshape('c', newshape_a, false);
  if (aPR == nullptr || (!aPR->isView() && aPR->buffer() != aPermuted->buffer())) {
    if (aPR != nullptr && aPR != aPermuted) delete aPR;
    aPR = aPermuted->reshape('c', newshape_a, true);
  }

  NDArray* bPermuted = bP->permute(newaxes_b, false, false);
  NDArray* bPR = bPermuted->reshape('c', newshape_b, false);
  if (bPR == nullptr || (!bPR->isView() && bPR->buffer() != bPermuted->buffer())) {
    if (bPR != nullptr && bPR != bPermuted) delete bPR;
    bPR = bPermuted->reshape('c', newshape_b, true);
  }

  std::vector<LongType> requiredCshape = {aPR->sizeAt(0), bPR->sizeAt(1)};
  NDArray* cPR = cP->reshape('f', requiredCshape, false);

  mmul(aPR, bPR, cPR, 1.0, 0.0);

  if (cPR->buffer() != c->buffer()) {
    c->assign(cPR);
  }

  if (realFinalResult != nullptr && realFinalResult != c) {
    realFinalResult->assign(c);
  }

  if (aPR != aPermuted && !aPR->isView()) delete aPR;
  if (aPermuted != aP && !aPermuted->isView()) delete aPermuted;
  if (aP != a && !aP->isView()) delete aP;

  if (bPR != bPermuted && !bPR->isView()) delete bPR;
  if (bPermuted != bP && !bPermuted->isView()) delete bPermuted;
  if (bP != b && !bP->isView()) delete bP;

  if (cPR != cP && !cPR->isView()) delete cPR;
  if (cP != c && !cP->isView()) delete cP;
}

//////////////////////////////////////////////////////////////////////////
void MmulHelper::tensorDot(NDArray* a, NDArray* b, NDArray* c,
                           std::vector<LongType>& axes_a, std::vector<LongType>& axes_b,
                           std::vector<LongType>& permutForC) {
  std::vector<LongType> permutAt, permutBt;
  std::vector<LongType> shapeAt, shapeBt;
  ShapeUtils::evalShapeForTensorDot(a, b, axes_a, axes_b, permutAt, permutBt, shapeAt, shapeBt);

  NDArray* cP = permutForC.empty() ? c : c->permute(permutForC, false, false);
  NDArray* aP = permutAt.empty() ? a : a->permute(permutAt, false, false);
  NDArray* bP = permutBt.empty() ? b : b->permute(permutBt, false, false);

  NDArray* aPR = aP->isSameShape(shapeAt) ? aP : aP->reshape(aP->ordering(), shapeAt, false);
  NDArray* bPR = bP->isSameShape(shapeBt) ? bP : bP->reshape(bP->ordering(), shapeBt, false);

  std::vector<LongType> requiredCshape = {aPR->sizeAt(0), bPR->sizeAt(1)};
  NDArray* cPR = cP->isSameShape(requiredCshape) ? cP : cP->reshape(cP->ordering(), requiredCshape, false);

  mmul(aPR, bPR, cPR, 1.0, 0.0);

  if (cPR->buffer() != c->buffer()) {
    c->assign(cPR);
  }

  if (aPR != aP && !aPR->isView()) delete aPR;
  if (bPR != bP && !bPR->isView()) delete bPR;
  if (cPR != cP && !cPR->isView()) delete cPR;
  if (aP != a && !aP->isView()) delete aP;
  if (bP != b && !bP->isView()) delete bP;
  if (cP != c && !cP->isView()) delete cP;
}

#ifndef __JAVACPP_HACK__
//////////////////////////////////////////////////////////////////////////
void MmulHelper::tensorDot(NDArray* a, NDArray* b, NDArray* c,
                           std::vector<std::vector<LongType>>& modifA,
                           std::vector<std::vector<LongType>>& modifB,
                           std::vector<std::vector<LongType>>& modifC) {
  NDArray *aPR(const_cast<NDArray*>(a)), *bPR(const_cast<NDArray*>(b));
  std::string whatToDoWithA, whatToDoWithB, whatToDoWithC;

  for (const auto& arr : modifA)
    whatToDoWithA = (std::find(arr.begin(), arr.end(), 0) != arr.end())
                    ? whatToDoWithA + "p" : whatToDoWithA + "r";
  for (const auto& arr : modifB)
    whatToDoWithB = (std::find(arr.begin(), arr.end(), 0) != arr.end())
                    ? whatToDoWithB + "p" : whatToDoWithB + "r";
  for (const auto& arr : modifC)
    whatToDoWithC = (std::find(arr.begin(), arr.end(), 0) != arr.end())
                    ? whatToDoWithC + "p" : whatToDoWithC + "r";

  if (!whatToDoWithA.empty())
    aPR = (whatToDoWithA[0] == 'p') ? a->permute(modifA[0], false, false)
                                    : a->reshape(a->ordering(), modifA[0], false);
  if (!whatToDoWithB.empty())
    bPR = (whatToDoWithB[0] == 'p') ? b->permute(modifB[0], false, false)
                                    : b->reshape(b->ordering(), modifB[0], false);

  for (size_t i = 1; i < whatToDoWithA.size(); ++i)
    if (whatToDoWithA[i] == 'p')
      aPR->permutei(modifA[i], false, false);
    else
      aPR->reshapei(modifA[i]);

  for (size_t i = 1; i < whatToDoWithB.size(); ++i)
    if (whatToDoWithB[i] == 'p')
      bPR->permutei(modifB[i], false, false);
    else
      bPR->reshapei(modifB[i]);

  std::vector<NDArray*> cArrs = {c};
  if (!whatToDoWithC.empty()) {
    cArrs = std::vector<NDArray*>(whatToDoWithC.size() + 1, c);
    for (size_t i = 0; i < cArrs.size() - 1; ++i)
      cArrs[i + 1] = (whatToDoWithC[i] == 'p')
                     ? cArrs[i]->permute(modifC[i], false, false)
                     : cArrs[i]->reshape(c->ordering(), modifC[i], false);
  }

  mmul(aPR, bPR, cArrs[cArrs.size() - 1], 1.0, 0.0);

  if (!whatToDoWithC.empty()) {
    for (int i = cArrs.size() - 1; i > 0; --i) {
      if (cArrs[i]->buffer() != cArrs[i - 1]->buffer() ||
          cArrs[i]->specialBuffer() != cArrs[i - 1]->specialBuffer())
        cArrs[i - 1]->assign(cArrs[i]);
      delete cArrs[i];
    }
  }

  if (aPR != a) delete aPR;
  if (bPR != b) delete bPR;
}

//////////////////////////////////////////////////////////////////////////
NDArray* MmulHelper::tensorDot(NDArray* a, NDArray* b,
                               std::vector<std::vector<LongType>>& modifA,
                               std::vector<std::vector<LongType>>& modifB) {
  NDArray *aPR(const_cast<NDArray*>(a)), *bPR(const_cast<NDArray*>(b));
  std::string whatToDoWithA, whatToDoWithB;

  for (const auto& arr : modifA)
    whatToDoWithA = (std::find(arr.begin(), arr.end(), 0) != arr.end())
                    ? whatToDoWithA + "p" : whatToDoWithA + "r";
  for (const auto& arr : modifB)
    whatToDoWithB = (std::find(arr.begin(), arr.end(), 0) != arr.end())
                    ? whatToDoWithB + "p" : whatToDoWithB + "r";

  if (!whatToDoWithA.empty())
    aPR = (whatToDoWithA[0] == 'p') ? a->permute(modifA[0], false, false)
                                    : a->reshape(a->ordering(), modifA[0], false);
  if (!whatToDoWithB.empty())
    bPR = (whatToDoWithB[0] == 'p') ? b->permute(modifB[0], false, false)
                                    : b->reshape(b->ordering(), modifB[0], false);

  for (size_t i = 1; i < whatToDoWithA.size(); ++i)
    if (whatToDoWithA[i] == 'p')
      aPR->permutei(modifA[i], false, false);
    else
      aPR->reshapei(modifA[i]);

  for (size_t i = 1; i < whatToDoWithB.size(); ++i)
    if (whatToDoWithB[i] == 'p')
      bPR->permutei(modifB[i], false, false);
    else
      bPR->reshapei(modifB[i]);

  NDArray* result = mmul(aPR, bPR, nullptr, 1.0, 0.0);

  return result;
}
#endif

// ============================================================================
// CORE MMUL DISPATCH
// ============================================================================

//////////////////////////////////////////////////////////////////////////
NDArray* MmulHelper::mmul(NDArray* A, NDArray* B, NDArray* C, const double alpha,
                          const double beta, const char outOrder) {
  LongType lenDim;
  const LongType aRank = A->rankOf();
  const LongType bRank = B->rankOf();
  const bool isAVector = shape::isCommonVector(A->shapeInfo(), lenDim);
  const bool isBVector = shape::isCommonVector(B->shapeInfo(), lenDim);

  // Dot product of 2 vectors
  if (A->lengthOf() == B->lengthOf() && isAVector && isBVector &&
      (aRank != 2 || (aRank == 2 && (A->isSameShape(B) || (bRank == 1 && A->sizeAt(1) == 1))))) {
    return dot(A, B, C, alpha, beta);
  }

  // Matrix x matrix
  if (aRank == 2 && bRank == 2) {
    return mmulMxM(A, B, C, alpha, beta, outOrder);
  }

  // Matrix x vector
  if (aRank == 2 && isBVector) {
    return mmulMxV(A, B, C, alpha, beta, outOrder);
  }

  // Vector x matrix
  if (isAVector && bRank == 2) {
    std::vector<sd::LongType> aShape = {1, A->lengthOf()};
    std::vector<sd::LongType> cShape = {1, C->lengthOf()};

    NDArray* A2 = A->reshape(A->ordering(), aShape);
    NDArray* C2 = C ? C->reshape(C->ordering(), cShape, false) : nullptr;
    auto result = mmulMxM(A2, B, C2, alpha, beta, outOrder);

    if (A2 != A) delete A2;
    if (C2 != nullptr && C2 != C) delete C2;

    if (!C) {
      result->reshapei({result->lengthOf()});
      return result;
    }
    return C;
  }

  // Batched matrix multiplication - try optimized BLAS/OneDNN path first
  // For 3D/4D tensors with matching ranks, use mmulBatched for OneDNN/BLAS acceleration
  if ((aRank == 3 || aRank == 4) && aRank == bRank) {
    // Create output array if not provided
    if (C == nullptr) {
      std::vector<LongType> cShape;
      if (aRank == 3) {
        cShape = {A->sizeAt(0), A->sizeAt(1), B->sizeAt(2)};
      } else {
        cShape = {A->sizeAt(0), A->sizeAt(1), A->sizeAt(2), B->sizeAt(3)};
      }
      C = new NDArray(outOrder, cShape, DataTypeUtils::pickPairwiseResultType(A->dataType(), B->dataType()),
                      A->getContext());
    }
    if (C->isEmpty()) return C;

    // Try optimized batched path (OneDNN -> BLAS batched -> BLAS per-batch -> manual)
    if (mmulBatched(A, B, C, alpha, beta)) {
      return C;
    }
    // If mmulBatched returned false (shouldn't happen), fall through to mmulNxN
  }

  // Fallback: general N-dimensional batched multiplication
  return mmulNxN(A, B, C, alpha, beta, outOrder);
}

//////////////////////////////////////////////////////////////////////////
bool MmulHelper::resolveTranspose(sd::NDArray& a, sd::NDArray& b, bool& transA, bool& transB) {
  int rowsA = a.sizeAt(-2);
  int colsA = a.sizeAt(-1);
  int rowsB = b.sizeAt(-2);
  int colsB = b.sizeAt(-1);

  transA = false;
  transB = false;

  if (colsA == rowsB) {
    return true;
  } else if (rowsA == rowsB) {
    transA = true;
    return true;
  } else if (colsA == colsB) {
    transB = true;
    return true;
  }
  return false;
}

// ============================================================================
// HIGH-LEVEL MATMUL WITH TRANSPOSE SUPPORT
// ============================================================================

//////////////////////////////////////////////////////////////////////////
void MmulHelper::matmul(NDArray* x, NDArray* y, NDArray* z, const bool transX, const bool transY,
                        double alpha, double beta, NDArray* realFinalResult) {
  int xRank = x->rankOf();
  int yRank = y->rankOf();

  auto outShape = ShapeUtils::evalShapeForMatmul(x->shapeInfo(), y->shapeInfo(), transX, transY);
  if (!z->isSameShape(outShape)) {
    std::string errorMessage = "NDArrayFactory::matmul: output shape mismatch, actual is ";
    errorMessage += ShapeUtils::shapeAsString(z) + " expected " + ShapeUtils::shapeAsString(outShape);
    THROW_EXCEPTION(errorMessage.c_str());
  }

  if (z->isEmpty()) return;

#if !defined(SD_CUDA)
  // Fast path for 2D with BLAS transpose flags (CPU only)
  if (xRank == 2 && yRank == 2 && Environment::getInstance().isEnableBlas()) {
    const auto xType = x->dataType();
    const bool sameTypes = (xType == y->dataType()) && (xType == z->dataType());
    const bool isFloat = (xType == DataType::FLOAT32);
    const bool isDouble = (xType == DataType::DOUBLE);

    // Compute dimensions first so we can validate leading dimensions
    const LongType M = transX ? x->sizeAt(1) : x->sizeAt(0);
    const LongType K = transX ? x->sizeAt(0) : x->sizeAt(1);
    const LongType N = transY ? y->sizeAt(0) : y->sizeAt(1);

    const int ldA = static_cast<int>(x->strideAt(0));
    const int ldB = static_cast<int>(y->strideAt(0));
    const int ldC = static_cast<int>(z->strideAt(0));

    // Check inner stride == 1 AND outer stride meets BLAS lda/ldb/ldc constraints.
    // For CblasRowMajor: lda/ldb must be >= the number of columns of the stored matrix.
    // NoTrans A is M×K stored → lda >= K;  Trans A is K×M stored → lda >= M
    // NoTrans B is K×N stored → ldb >= N;  Trans B is N×K stored → ldb >= K
    const LongType xLdMin = transX ? M : K;
    const LongType yLdMin = transY ? K : N;
    const bool xRowMajor = (x->strideAt(1) == 1) && (ldA >= xLdMin);
    const bool yRowMajor = (y->strideAt(1) == 1) && (ldB >= yLdMin);
    const bool zRowMajor = (z->strideAt(1) == 1) && (ldC >= N);

    if (sameTypes && (isFloat || isDouble) && xRowMajor && yRowMajor && zRowMajor &&
        BlasHelper::getInstance().hasGEMM(xType)) {

      const CBLAS_TRANSPOSE transAblas = transX ? CblasTrans : CblasNoTrans;
      const CBLAS_TRANSPOSE transBblas = transY ? CblasTrans : CblasNoTrans;

      auto& blasHelper = BlasHelper::getInstance();
      auto blasLock = blasHelper.lockBlas();

      if (isFloat) {
        blasHelper.sgemm()(CblasRowMajor, transAblas, transBblas,
                           static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                           static_cast<float>(alpha),
                           x->bufferAsT<float>(), ldA,
                           y->bufferAsT<float>(), ldB,
                           static_cast<float>(beta),
                           z->bufferAsT<float>(), ldC);
      } else {
        blasHelper.dgemm()(CblasRowMajor, transAblas, transBblas,
                           static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                           alpha,
                           x->bufferAsT<double>(), ldA,
                           y->bufferAsT<double>(), ldB,
                           beta,
                           z->bufferAsT<double>(), ldC);
      }

      if (realFinalResult != nullptr && realFinalResult != z) {
        realFinalResult->assign(z);
      }
      return;
    }
  }
#endif

  // Handle transpose via permute + dup
  NDArray *xT = const_cast<NDArray*>(x);
  NDArray *yT = const_cast<NDArray*>(y);
  NDArray *zT = z;

  if ((transX && xRank > 1) || (transY && yRank > 1)) {
    const int rank = xRank >= yRank ? xRank : yRank;
    std::vector<LongType> permut(rank);
    for (int i = 0; i < rank - 2; ++i) permut[i] = i;
    permut[rank - 2] = rank - 1;
    permut[rank - 1] = rank - 2;

    if (transX) {
      NDArray* permutedView = x->permute(permut, false, false);
      xT = permutedView->dup();
      delete permutedView;
    }
    if (transY) {
      NDArray* permutedView = y->permute(permut, false, false);
      yT = permutedView->dup();
      delete permutedView;
    }
  }

  if (xRank <= 2 && yRank <= 2) {
    // 2D or lower
    NDArray* xReshaped = nullptr;
    NDArray* zReshaped = nullptr;

    if (xRank == 1 && yRank == 2) {
      std::vector<sd::LongType> xShape = {1, xT->lengthOf()};
      std::vector<sd::LongType> zShape = {1, z->lengthOf()};

      NDArray* xPermuted = (xT != x) ? xT : nullptr;
      NDArray* zPermuted = (zT != z) ? zT : nullptr;

      xReshaped = xT->reshape(xT->ordering(), xShape, false);
      xT = xReshaped;
      zReshaped = z->reshape(z->ordering(), zShape, false);
      zT = zReshaped;

      if (xPermuted != nullptr && !xPermuted->isView()) delete xPermuted;
      if (zPermuted != nullptr && !zPermuted->isView()) delete zPermuted;
    }

    mmul(xT, yT, zT, alpha, beta);

    if (zT != z) {
      z->assign(zT);
      delete zT;
      zT = z;
    }
    if (xReshaped != nullptr && xReshaped != x) {
      delete xReshaped;
      xT = x;
    }
  } else {
    // Batched matmul - use unified dispatch
    const int xRankT = xT->rankOf();
    const int yRankT = yT->rankOf();
    const int zRankT = zT->rankOf();

#if !defined(SD_CUDA)
    // CPU: use mmulBatched for 3D/4D with matching ranks (uses OneDNN/BLAS)
    if ((xRankT == 3 || xRankT == 4) && xRankT == yRankT && yRankT == zRankT) {
      mmulBatched(xT, yT, zT, alpha, beta);
    } else {
      // Fall back to mmulNxN for other cases
      mmulNxN(xT, yT, zT, alpha, beta, z->ordering());
    }
#else
    // CUDA: try cuBLAS strided batched GEMM first (most efficient for 3D/4D)
    if ((xRankT == 3 || xRankT == 4) && xRankT == yRankT && yRankT == zRankT) {
      if (!tryBlasStridedBatched(xT, yT, zT, alpha, beta)) {
        // cuBLAS couldn't handle it (non-contiguous strides, etc.)
        // Create contiguous copies and retry cuBLAS before falling back to custom kernel.
        NDArray* xDup = xT->dup();
        NDArray* yDup = yT->dup();
        if (!tryBlasStridedBatched(xDup, yDup, zT, alpha, beta)) {
          // Still failed (unsupported type, etc.) — fall back to custom CUDA kernel
          mmulNxN(xDup, yDup, zT, alpha, beta, z->ordering());
        }
        // Sync stream before freeing temporary arrays to ensure async GEMM completes.
        // During CUDA graph capture, stream sync is illegal (poisons the capture).
        // The dup'd arrays live until end of capture; graph replay uses fixed addresses.
        if (!tl_graphExecutionActive) {
          auto* lc = LaunchContext::defaultContext();
          if (lc->getCudaStream() != nullptr) {
            cudaStreamSynchronize(*lc->getCudaStream());
          }
        }
        delete xDup;
        delete yDup;
      }
    } else {
      mmulNxN(xT, yT, zT, alpha, beta, z->ordering());
    }
#endif
  }

  // Cleanup
  if (xT != x && xT != nullptr) delete xT;
  if (yT != y && yT != nullptr) delete yT;

  if (realFinalResult != nullptr && realFinalResult != z) {
    // Skip redundant copy when the result already shares the same buffer.
    if (realFinalResult->getDataBuffer() != z->getDataBuffer() ||
        realFinalResult->offset() != z->offset() ||
        realFinalResult->lengthOf() != z->lengthOf()) {
      realFinalResult->assign(z);
    }
  }
}

}  // namespace sd

#endif
