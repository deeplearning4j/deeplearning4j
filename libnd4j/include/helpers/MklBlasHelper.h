/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// MKL BLAS Helper
//
// Provides MKL-specific BLAS operations that go beyond standard OpenBLAS:
// - Half precision (float16) GEMM
// - BFloat16 GEMM
// - Optimized batch GEMM with uniform parameters
// - Integration with VML for vectorized math
//
// Usage:
//   if (sd::mkl::isAvailable()) {
//       sd::mkl::hgemm(...);  // Half precision GEMM
//       sd::mkl::bf16gemm(...);  // BFloat16 GEMM
//       sd::mkl::sgemmBatchStrided(...);  // Strided batch GEMM
//   }
//

#ifndef LIBND4J_MKL_BLAS_HELPER_H
#define LIBND4J_MKL_BLAS_HELPER_H

#include <system/common.h>
#include <types/float16.h>
#include <types/bfloat16.h>
#include <memory>

// Include CBLAS headers - use MKL or OpenBLAS, but not both
// MKL's cblas and OpenBLAS's cblas have conflicting declarations
#if !defined(SD_CUDA)
  #ifdef HAVE_MKL
    // MKL-specific headers (avoid mkl.h which pulls in everything)
    #include <mkl_types.h>    // MKL_INT, MKL_F16, MKL_BF16
    #include <mkl_cblas.h>    // CBLAS functions including extended precision
    #include <mkl_service.h>  // mkl_set_num_threads, mkl_get_max_threads, mkl_set_dynamic
  #else
    // Fall back to OpenBLAS
    #include <blas/cblas.h>
  #endif
#endif

namespace sd {
namespace mkl {

// ============================================================================
// MKL Availability Check
// ============================================================================

/**
 * Check if MKL BLAS is available at runtime
 */
SD_INLINE bool isAvailable() {
#ifdef HAVE_MKL
    return true;
#else
    return false;
#endif
}

/**
 * Check if MKL supports half precision (float16) GEMM
 * Note: MKL supports this through cblas_gemm_f16f16f32 or oneDNN
 */
SD_INLINE bool hasHalfGemm() {
#ifdef HAVE_MKL
    // MKL 2019+ has half precision support via oneDNN integration
    return true;
#else
    return false;
#endif
}

/**
 * Check if MKL supports BFloat16 GEMM
 * Note: MKL 2020+ has BFloat16 support
 */
SD_INLINE bool hasBFloat16Gemm() {
#ifdef HAVE_MKL
    // MKL 2020+ has BFloat16 support
    return true;
#else
    return false;
#endif
}

// ============================================================================
// Standard GEMM Operations (with MKL optimizations)
// ============================================================================

#ifdef HAVE_MKL

/**
 * Single precision GEMM: C = alpha * op(A) * op(B) + beta * C
 */
SD_INLINE void sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                     MKL_INT m, MKL_INT n, MKL_INT k,
                     float alpha, const float* A, MKL_INT lda,
                     const float* B, MKL_INT ldb,
                     float beta, float* C, MKL_INT ldc) {
    cblas_sgemm(layout, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

/**
 * Double precision GEMM: C = alpha * op(A) * op(B) + beta * C
 */
SD_INLINE void dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                     MKL_INT m, MKL_INT n, MKL_INT k,
                     double alpha, const double* A, MKL_INT lda,
                     const double* B, MKL_INT ldb,
                     double beta, double* C, MKL_INT ldc) {
    cblas_dgemm(layout, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

#endif  // HAVE_MKL

// ============================================================================
// Half Precision (Float16) GEMM
// ============================================================================

#ifdef HAVE_MKL

/**
 * Half precision GEMM with float32 accumulation
 * C (float32) = alpha * op(A_fp16) * op(B_fp16) + beta * C
 *
 * This uses MKL's mixed precision capabilities for better accuracy
 * while maintaining the speed benefits of half precision inputs.
 */
SD_INLINE void hgemm_f32accum(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                               MKL_INT m, MKL_INT n, MKL_INT k,
                               float alpha, const MKL_F16* A, MKL_INT lda,
                               const MKL_F16* B, MKL_INT ldb,
                               float beta, float* C, MKL_INT ldc) {
    // Use MKL's mixed precision GEMM: f16 * f16 -> f32
    cblas_gemm_f16f16f32(layout, transA, transB, m, n, k,
                          alpha, A, lda, B, ldb, beta, C, ldc);
}

/**
 * Half precision GEMM wrapper for float16 types
 * Converts sd::float16 pointers to MKL_F16 for MKL compatibility
 */
SD_INLINE void hgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                     MKL_INT m, MKL_INT n, MKL_INT k,
                     float alpha, const float16* A, MKL_INT lda,
                     const float16* B, MKL_INT ldb,
                     float beta, float* C, MKL_INT ldc) {
    // sd::float16 and MKL_F16 should have the same binary representation
    static_assert(sizeof(float16) == sizeof(MKL_F16), "float16 size mismatch with MKL_F16");
    hgemm_f32accum(layout, transA, transB, m, n, k, alpha,
                   reinterpret_cast<const MKL_F16*>(A), lda,
                   reinterpret_cast<const MKL_F16*>(B), ldb,
                   beta, C, ldc);
}

#endif  // HAVE_MKL

// ============================================================================
// BFloat16 GEMM
// ============================================================================

#ifdef HAVE_MKL

/**
 * BFloat16 GEMM with float32 accumulation
 * C (float32) = alpha * op(A_bf16) * op(B_bf16) + beta * C
 */
SD_INLINE void bf16gemm_f32accum(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                                  MKL_INT m, MKL_INT n, MKL_INT k,
                                  float alpha, const MKL_BF16* A, MKL_INT lda,
                                  const MKL_BF16* B, MKL_INT ldb,
                                  float beta, float* C, MKL_INT ldc) {
    // Use MKL's BFloat16 GEMM: bf16 * bf16 -> f32
    cblas_gemm_bf16bf16f32(layout, transA, transB, m, n, k,
                            alpha, A, lda, B, ldb, beta, C, ldc);
}

/**
 * BFloat16 GEMM wrapper for bfloat16 types
 * Converts sd::bfloat16 pointers to MKL_BF16 for MKL compatibility
 */
SD_INLINE void bf16gemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                        MKL_INT m, MKL_INT n, MKL_INT k,
                        float alpha, const bfloat16* A, MKL_INT lda,
                        const bfloat16* B, MKL_INT ldb,
                        float beta, float* C, MKL_INT ldc) {
    // sd::bfloat16 and MKL_BF16 should have the same binary representation
    static_assert(sizeof(bfloat16) == sizeof(MKL_BF16), "bfloat16 size mismatch with MKL_BF16");
    bf16gemm_f32accum(layout, transA, transB, m, n, k, alpha,
                      reinterpret_cast<const MKL_BF16*>(A), lda,
                      reinterpret_cast<const MKL_BF16*>(B), ldb,
                      beta, C, ldc);
}

#endif  // HAVE_MKL

// ============================================================================
// Batch GEMM Operations
// ============================================================================

#ifdef HAVE_MKL

/**
 * Single precision strided batch GEMM
 * Performs multiple GEMM operations where matrices are laid out with fixed strides
 *
 * This is more efficient than the pointer-array batch GEMM when matrices
 * are stored contiguously with regular strides.
 */
SD_INLINE void sgemmBatchStrided(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                                  MKL_INT m, MKL_INT n, MKL_INT k,
                                  float alpha,
                                  const float* A, MKL_INT lda, MKL_INT strideA,
                                  const float* B, MKL_INT ldb, MKL_INT strideB,
                                  float beta,
                                  float* C, MKL_INT ldc, MKL_INT strideC,
                                  MKL_INT batchCount) {
    cblas_sgemm_batch_strided(layout, transA, transB, m, n, k,
                               alpha, A, lda, strideA, B, ldb, strideB,
                               beta, C, ldc, strideC, batchCount);
}

/**
 * Double precision strided batch GEMM
 */
SD_INLINE void dgemmBatchStrided(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                                  MKL_INT m, MKL_INT n, MKL_INT k,
                                  double alpha,
                                  const double* A, MKL_INT lda, MKL_INT strideA,
                                  const double* B, MKL_INT ldb, MKL_INT strideB,
                                  double beta,
                                  double* C, MKL_INT ldc, MKL_INT strideC,
                                  MKL_INT batchCount) {
    cblas_dgemm_batch_strided(layout, transA, transB, m, n, k,
                               alpha, A, lda, strideA, B, ldb, strideB,
                               beta, C, ldc, strideC, batchCount);
}

/**
 * Single precision batch GEMM with uniform parameters
 * All batches use the same M, N, K, alpha, beta values
 */
SD_INLINE void sgemmBatch(CBLAS_LAYOUT layout,
                           const CBLAS_TRANSPOSE* transA_array,
                           const CBLAS_TRANSPOSE* transB_array,
                           const MKL_INT* m_array, const MKL_INT* n_array, const MKL_INT* k_array,
                           const float* alpha_array,
                           const float** A_array, const MKL_INT* lda_array,
                           const float** B_array, const MKL_INT* ldb_array,
                           const float* beta_array,
                           float** C_array, const MKL_INT* ldc_array,
                           MKL_INT group_count, const MKL_INT* group_size) {
    cblas_sgemm_batch(layout, transA_array, transB_array,
                       m_array, n_array, k_array,
                       alpha_array, A_array, lda_array,
                       B_array, ldb_array, beta_array,
                       C_array, ldc_array, group_count, group_size);
}

/**
 * Double precision batch GEMM with uniform parameters
 */
SD_INLINE void dgemmBatch(CBLAS_LAYOUT layout,
                           const CBLAS_TRANSPOSE* transA_array,
                           const CBLAS_TRANSPOSE* transB_array,
                           const MKL_INT* m_array, const MKL_INT* n_array, const MKL_INT* k_array,
                           const double* alpha_array,
                           const double** A_array, const MKL_INT* lda_array,
                           const double** B_array, const MKL_INT* ldb_array,
                           const double* beta_array,
                           double** C_array, const MKL_INT* ldc_array,
                           MKL_INT group_count, const MKL_INT* group_size) {
    cblas_dgemm_batch(layout, transA_array, transB_array,
                       m_array, n_array, k_array,
                       alpha_array, A_array, lda_array,
                       B_array, ldb_array, beta_array,
                       C_array, ldc_array, group_count, group_size);
}

#endif  // HAVE_MKL

// ============================================================================
// GEMV Operations (Matrix-Vector Multiply)
// ============================================================================

#ifdef HAVE_MKL

/**
 * Single precision GEMV: y = alpha * op(A) * x + beta * y
 */
SD_INLINE void sgemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
                     MKL_INT m, MKL_INT n,
                     float alpha, const float* A, MKL_INT lda,
                     const float* x, MKL_INT incx,
                     float beta, float* y, MKL_INT incy) {
    cblas_sgemv(layout, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

/**
 * Double precision GEMV: y = alpha * op(A) * x + beta * y
 */
SD_INLINE void dgemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
                     MKL_INT m, MKL_INT n,
                     double alpha, const double* A, MKL_INT lda,
                     const double* x, MKL_INT incx,
                     double beta, double* y, MKL_INT incy) {
    cblas_dgemv(layout, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

#endif  // HAVE_MKL

// ============================================================================
// Packed GEMM (for optimized repeated operations)
// ============================================================================

#ifdef HAVE_MKL

/**
 * Pack matrix A for repeated GEMM operations
 * Returns a handle to the packed matrix that can be used in packed GEMM calls
 */
SD_INLINE void* sgemm_pack_get_size(CBLAS_IDENTIFIER identifier,
                                     MKL_INT m, MKL_INT n, MKL_INT k) {
    size_t size = cblas_sgemm_pack_get_size(identifier, m, n, k);
    return malloc(size);
}

SD_INLINE void sgemm_pack(CBLAS_LAYOUT layout, CBLAS_IDENTIFIER identifier,
                          CBLAS_TRANSPOSE trans, MKL_INT m, MKL_INT n, MKL_INT k,
                          float alpha, const float* src, MKL_INT ld, float* dest) {
    cblas_sgemm_pack(layout, identifier, trans, m, n, k, alpha, src, ld, dest);
}

SD_INLINE void sgemm_compute(CBLAS_LAYOUT layout, MKL_INT transA, MKL_INT transB,
                              MKL_INT m, MKL_INT n, MKL_INT k,
                              const float* A, MKL_INT lda,
                              const float* B, MKL_INT ldb,
                              float beta, float* C, MKL_INT ldc) {
    cblas_sgemm_compute(layout, transA, transB, m, n, k, A, lda, B, ldb, beta, C, ldc);
}

#endif  // HAVE_MKL

// ============================================================================
// MKL Thread Control
// ============================================================================

#ifdef HAVE_MKL

/**
 * Set the number of threads MKL should use
 */
SD_INLINE void setNumThreads(int numThreads) {
    mkl_set_num_threads(numThreads);
}

/**
 * Get the current number of threads MKL is using
 */
SD_INLINE int getMaxThreads() {
    return mkl_get_max_threads();
}

/**
 * Set MKL to use sequential (single-threaded) execution
 */
SD_INLINE void setSequential() {
    mkl_set_num_threads(1);
}

/**
 * Enable MKL's dynamic thread adjustment
 */
SD_INLINE void setDynamicThreading(bool enable) {
    mkl_set_dynamic(enable ? 1 : 0);
}

#endif  // HAVE_MKL

// ============================================================================
// Fallback Implementations (when MKL is not available)
// Delegates to OpenBLAS - which MUST be available as primary BLAS backend
// ============================================================================

#ifndef HAVE_MKL

// Build configuration check - we MUST have a BLAS backend
#if !HAVE_OPENBLAS && !__EXTERNAL_BLAS__
#error "No BLAS backend available. Either MKL, OpenBLAS, or external BLAS must be configured."
#endif

// OpenBLAS thread control - these are always available when HAVE_OPENBLAS is defined
#if HAVE_OPENBLAS
extern "C" {
    void openblas_set_num_threads(int num_threads);
    int openblas_get_num_threads(void);
}
#endif

// ============================================================================
// Thread Control - delegates to OpenBLAS
// ============================================================================

/**
 * Set the number of threads for BLAS operations
 * Delegates to OpenBLAS openblas_set_num_threads
 */
SD_INLINE void setNumThreads(int numThreads) {
#if HAVE_OPENBLAS
    openblas_set_num_threads(numThreads);
#else
    // External BLAS may not have thread control - this is acceptable
    // as MKL or OpenBLAS are the expected backends
    (void)numThreads;
#endif
}

/**
 * Get the current number of threads for BLAS operations
 * Delegates to OpenBLAS openblas_get_num_threads
 */
SD_INLINE int getMaxThreads() {
#if HAVE_OPENBLAS
    return openblas_get_num_threads();
#else
    // External BLAS - assume single-threaded if no query available
    return 1;
#endif
}

/**
 * Set BLAS to use sequential (single-threaded) execution
 */
SD_INLINE void setSequential() {
#if HAVE_OPENBLAS
    openblas_set_num_threads(1);
#endif
}

/**
 * Dynamic threading control
 * OpenBLAS doesn't support MKL-style dynamic threading,
 * but we can still set thread count dynamically via setNumThreads
 */
SD_INLINE void setDynamicThreading(bool enable) {
    // OpenBLAS doesn't have mkl_set_dynamic equivalent
    // Thread count is controlled via openblas_set_num_threads
    (void)enable;
}

// ============================================================================
// Standard GEMM/GEMV - delegate directly to cblas
// ============================================================================

/**
 * Single precision GEMM - delegates to cblas_sgemm
 */
SD_INLINE void sgemm(CBLAS_ORDER layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                     int m, int n, int k,
                     float alpha, const float* A, int lda,
                     const float* B, int ldb,
                     float beta, float* C, int ldc) {
    cblas_sgemm(layout, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

/**
 * Double precision GEMM - delegates to cblas_dgemm
 */
SD_INLINE void dgemm(CBLAS_ORDER layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                     int m, int n, int k,
                     double alpha, const double* A, int lda,
                     const double* B, int ldb,
                     double beta, double* C, int ldc) {
    cblas_dgemm(layout, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

/**
 * Single precision GEMV - delegates to cblas_sgemv
 */
SD_INLINE void sgemv(CBLAS_ORDER layout, CBLAS_TRANSPOSE transA,
                     int m, int n,
                     float alpha, const float* A, int lda,
                     const float* x, int incx,
                     float beta, float* y, int incy) {
    cblas_sgemv(layout, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

/**
 * Double precision GEMV - delegates to cblas_dgemv
 */
SD_INLINE void dgemv(CBLAS_ORDER layout, CBLAS_TRANSPOSE transA,
                     int m, int n,
                     double alpha, const double* A, int lda,
                     const double* x, int incx,
                     double beta, double* y, int incy) {
    cblas_dgemv(layout, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

// ============================================================================
// Extended Precision GEMM - convert to float32 and use cblas_sgemm
// ============================================================================

/**
 * Half precision GEMM with float32 accumulation
 * Converts float16 inputs to float32, calls cblas_sgemm
 */
SD_INLINE void hgemm(CBLAS_ORDER layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                     int m, int n, int k,
                     float alpha, const float16* A, int lda,
                     const float16* B, int ldb,
                     float beta, float* C, int ldc) {
    // Calculate actual sizes based on transpose
    const int rowsA = (transA == CblasNoTrans) ? m : k;
    const int colsA = (transA == CblasNoTrans) ? k : m;
    const int rowsB = (transB == CblasNoTrans) ? k : n;
    const int colsB = (transB == CblasNoTrans) ? n : k;

    const int sizeA = rowsA * colsA;
    const int sizeB = rowsB * colsB;

    // Allocate and convert float16 -> float32
    std::unique_ptr<float[]> A_f32(new float[sizeA]);
    std::unique_ptr<float[]> B_f32(new float[sizeB]);

    // Convert with proper striding for row-major layout
    if (layout == CblasRowMajor) {
        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < colsA; ++j) {
                A_f32[i * colsA + j] = static_cast<float>(A[i * lda + j]);
            }
        }
        for (int i = 0; i < rowsB; ++i) {
            for (int j = 0; j < colsB; ++j) {
                B_f32[i * colsB + j] = static_cast<float>(B[i * ldb + j]);
            }
        }
        // Use actual dimensions as lda for converted arrays
        cblas_sgemm(layout, transA, transB, m, n, k, alpha,
                    A_f32.get(), colsA, B_f32.get(), colsB, beta, C, ldc);
    } else {
        // Column-major
        for (int j = 0; j < colsA; ++j) {
            for (int i = 0; i < rowsA; ++i) {
                A_f32[j * rowsA + i] = static_cast<float>(A[j * lda + i]);
            }
        }
        for (int j = 0; j < colsB; ++j) {
            for (int i = 0; i < rowsB; ++i) {
                B_f32[j * rowsB + i] = static_cast<float>(B[j * ldb + i]);
            }
        }
        cblas_sgemm(layout, transA, transB, m, n, k, alpha,
                    A_f32.get(), rowsA, B_f32.get(), rowsB, beta, C, ldc);
    }
}

/**
 * BFloat16 GEMM with float32 accumulation
 * Converts bfloat16 inputs to float32, calls cblas_sgemm
 */
SD_INLINE void bf16gemm(CBLAS_ORDER layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                        int m, int n, int k,
                        float alpha, const bfloat16* A, int lda,
                        const bfloat16* B, int ldb,
                        float beta, float* C, int ldc) {
    // Calculate actual sizes based on transpose
    const int rowsA = (transA == CblasNoTrans) ? m : k;
    const int colsA = (transA == CblasNoTrans) ? k : m;
    const int rowsB = (transB == CblasNoTrans) ? k : n;
    const int colsB = (transB == CblasNoTrans) ? n : k;

    const int sizeA = rowsA * colsA;
    const int sizeB = rowsB * colsB;

    // Allocate and convert bfloat16 -> float32
    std::unique_ptr<float[]> A_f32(new float[sizeA]);
    std::unique_ptr<float[]> B_f32(new float[sizeB]);

    // Convert with proper striding
    if (layout == CblasRowMajor) {
        for (int i = 0; i < rowsA; ++i) {
            for (int j = 0; j < colsA; ++j) {
                A_f32[i * colsA + j] = static_cast<float>(A[i * lda + j]);
            }
        }
        for (int i = 0; i < rowsB; ++i) {
            for (int j = 0; j < colsB; ++j) {
                B_f32[i * colsB + j] = static_cast<float>(B[i * ldb + j]);
            }
        }
        cblas_sgemm(layout, transA, transB, m, n, k, alpha,
                    A_f32.get(), colsA, B_f32.get(), colsB, beta, C, ldc);
    } else {
        for (int j = 0; j < colsA; ++j) {
            for (int i = 0; i < rowsA; ++i) {
                A_f32[j * rowsA + i] = static_cast<float>(A[j * lda + i]);
            }
        }
        for (int j = 0; j < colsB; ++j) {
            for (int i = 0; i < rowsB; ++i) {
                B_f32[j * rowsB + i] = static_cast<float>(B[j * ldb + i]);
            }
        }
        cblas_sgemm(layout, transA, transB, m, n, k, alpha,
                    A_f32.get(), rowsA, B_f32.get(), rowsB, beta, C, ldc);
    }
}

// ============================================================================
// Batch GEMM - loop over individual cblas_sgemm/dgemm calls
// ============================================================================

/**
 * Strided batch SGEMM - loops over cblas_sgemm
 */
SD_INLINE void sgemmBatchStrided(CBLAS_ORDER layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                                  int m, int n, int k,
                                  float alpha,
                                  const float* A, int lda, int strideA,
                                  const float* B, int ldb, int strideB,
                                  float beta,
                                  float* C, int ldc, int strideC,
                                  int batchCount) {
    for (int batch = 0; batch < batchCount; ++batch) {
        cblas_sgemm(layout, transA, transB, m, n, k, alpha,
                    A + batch * strideA, lda,
                    B + batch * strideB, ldb,
                    beta, C + batch * strideC, ldc);
    }
}

/**
 * Strided batch DGEMM - loops over cblas_dgemm
 */
SD_INLINE void dgemmBatchStrided(CBLAS_ORDER layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                                  int m, int n, int k,
                                  double alpha,
                                  const double* A, int lda, int strideA,
                                  const double* B, int ldb, int strideB,
                                  double beta,
                                  double* C, int ldc, int strideC,
                                  int batchCount) {
    for (int batch = 0; batch < batchCount; ++batch) {
        cblas_dgemm(layout, transA, transB, m, n, k, alpha,
                    A + batch * strideA, lda,
                    B + batch * strideB, ldb,
                    beta, C + batch * strideC, ldc);
    }
}

/**
 * Pointer-array batch SGEMM - loops over cblas_sgemm
 */
SD_INLINE void sgemmBatch(CBLAS_ORDER layout,
                           const CBLAS_TRANSPOSE* transA_array,
                           const CBLAS_TRANSPOSE* transB_array,
                           const int* m_array, const int* n_array, const int* k_array,
                           const float* alpha_array,
                           const float** A_array, const int* lda_array,
                           const float** B_array, const int* ldb_array,
                           const float* beta_array,
                           float** C_array, const int* ldc_array,
                           int group_count, const int* group_size) {
    int idx = 0;
    for (int g = 0; g < group_count; ++g) {
        for (int i = 0; i < group_size[g]; ++i) {
            cblas_sgemm(layout, transA_array[g], transB_array[g],
                        m_array[g], n_array[g], k_array[g],
                        alpha_array[g], A_array[idx], lda_array[g],
                        B_array[idx], ldb_array[g],
                        beta_array[g], C_array[idx], ldc_array[g]);
            ++idx;
        }
    }
}

/**
 * Pointer-array batch DGEMM - loops over cblas_dgemm
 */
SD_INLINE void dgemmBatch(CBLAS_ORDER layout,
                           const CBLAS_TRANSPOSE* transA_array,
                           const CBLAS_TRANSPOSE* transB_array,
                           const int* m_array, const int* n_array, const int* k_array,
                           const double* alpha_array,
                           const double** A_array, const int* lda_array,
                           const double** B_array, const int* ldb_array,
                           const double* beta_array,
                           double** C_array, const int* ldc_array,
                           int group_count, const int* group_size) {
    int idx = 0;
    for (int g = 0; g < group_count; ++g) {
        for (int i = 0; i < group_size[g]; ++i) {
            cblas_dgemm(layout, transA_array[g], transB_array[g],
                        m_array[g], n_array[g], k_array[g],
                        alpha_array[g], A_array[idx], lda_array[g],
                        B_array[idx], ldb_array[g],
                        beta_array[g], C_array[idx], ldc_array[g]);
            ++idx;
        }
    }
}

#endif  // !HAVE_MKL

}  // namespace mkl
}  // namespace sd

#endif  // LIBND4J_MKL_BLAS_HELPER_H
