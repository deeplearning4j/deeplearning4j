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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 05.06.2018
//

#ifndef LIBND4J_MMULHELPER_H
#define LIBND4J_MMULHELPER_H
#include "array/NDArray.h"
#include <utility>

namespace sd {

/**
 * MmulHelper - Matrix multiplication helper with automatic backend dispatch
 *
 * Provides unified interface for matrix operations with automatic selection of:
 * - OneDNN (MKL) batched matmul for 3D tensors when available
 * - BLAS batched GEMM (cblas_sgemm_batch) when available
 * - BLAS per-batch GEMM calls as fallback
 * - Optimized manual implementation for non-standard types
 *
 * Key methods:
 * - mmul(): General matrix multiply with automatic dispatch
 * - matmul(): High-level matmul with transpose support
 * - tensorDot(): Tensor contraction along specified axes
 */
class SD_LIB_EXPORT MmulHelper {
 private:
  // ============== Core Matrix Operations ==============

  // Matrix x Matrix: [M,K] x [K,N] = [M,N]
  static NDArray* mmulMxM(NDArray* A, NDArray* B, NDArray* C, double alpha = 1.0,
                          double beta = 0.0, const char outOrder = 'f');

  // Matrix x Vector: [M,N] x [N] = [M]
  static NDArray* mmulMxV(NDArray* A, NDArray* B, NDArray* C, double alpha = 1.0,
                          double beta = 0.0, const char outOrder = 'f');

  // Dot product: [N] x [N] = scalar
  static NDArray* dot(NDArray* X, NDArray* Y, NDArray* Z, const double alpha = 1.0,
                      const double beta = 0.0);

  // N-dimensional batched matmul: [...,M,K] x [...,K,N] = [...,M,N]
  static NDArray* mmulNxN(NDArray* A, NDArray* B, NDArray* C, const double alpha = 1.0,
                          const double beta = 0.0, const char outOrder = 'f');

  // ============== Batched Matmul with Backend Dispatch ==============

  /**
   * Batched matrix multiplication with automatic backend selection
   *
   * Tries backends in order of preference:
   * 1. OneDNN batched matmul (single primitive for entire batch)
   * 2. BLAS batched GEMM (cblas_sgemm_batch)
   * 3. Per-batch BLAS GEMM calls
   * 4. Manual parallel implementation
   *
   * @param A Input tensor [batch..., M, K]
   * @param B Input tensor [batch..., K, N]
   * @param C Output tensor [batch..., M, N]
   * @param alpha Scalar multiplier for A*B
   * @param beta Scalar multiplier for C (accumulation)
   * @return true if computation succeeded, false on error
   */
  static bool mmulBatched(NDArray* A, NDArray* B, NDArray* C,
                          double alpha = 1.0, double beta = 0.0);

  /**
   * Try OneDNN batched matmul for 3D tensors
   * @return true if OneDNN handled the operation
   */
  static bool tryOneDnnBatched(NDArray* A, NDArray* B, NDArray* C,
                               double alpha, double beta);

  /**
   * Try strided batch GEMM for contiguous batched data
   *
   * On CPU (MKL): Uses cblas_sgemm_batch_strided/cblas_dgemm_batch_strided
   * On GPU (cuBLAS): Uses cublasSgemmStridedBatched/cublasDgemmStridedBatched/cublasHgemmStridedBatched
   *
   * More efficient than pointer-array batch GEMM as it avoids building pointer arrays
   * and allows the library to optimize memory access patterns.
   *
   * @return true if strided batch GEMM handled the operation
   */
  static bool tryBlasStridedBatched(NDArray* A, NDArray* B, NDArray* C,
                                     double alpha, double beta,
                                     bool transA = false, bool transB = false);

  /**
   * Try BLAS batched GEMM (cblas_sgemm_batch/cblas_dgemm_batch)
   * @return true if batched BLAS handled the operation
   */
  static bool tryBlasBatched(NDArray* A, NDArray* B, NDArray* C,
                             double alpha, double beta);

  /**
   * Try per-batch BLAS GEMM calls
   * @return true if per-batch BLAS handled the operation
   */
  static bool tryBlasPerBatch(NDArray* A, NDArray* B, NDArray* C,
                              double alpha, double beta);

  /**
   * Manual parallel batched matmul fallback
   * Always succeeds for supported types
   */
  static void manualBatchedGemm(NDArray* A, NDArray* B, NDArray* C,
                                double alpha, double beta);

 public:
  // ============== Public Interface ==============

  /**
   * General matrix multiplication with automatic dispatch
   *
   * Handles:
   * - Dot product (vector x vector)
   * - Matrix x vector
   * - Vector x matrix
   * - Matrix x matrix
   * - Batched matmul (3D+)
   *
   * @param A First input array
   * @param B Second input array
   * @param C Output array (optional, created if null)
   * @param alpha Scalar for A*B
   * @param beta Scalar for C accumulation
   * @param outOrder Memory order for output ('c' or 'f')
   * @return Result array (C if provided, new array otherwise)
   */
  static NDArray* mmul(NDArray* A, NDArray* B, NDArray* C = nullptr,
                       const double alpha = 1.0, const double beta = 0.0,
                       const char outOrder = 'f');

  /**
   * High-level matmul with transpose support
   *
   * Supports arbitrary ranks with broadcasting.
   * Transpose flags apply to the last two dimensions.
   *
   * @param x First input
   * @param y Second input
   * @param z Output (must be pre-allocated with correct shape)
   * @param transX Transpose x's last two dims
   * @param transY Transpose y's last two dims
   * @param alpha Scalar for x*y
   * @param beta Scalar for z accumulation
   * @param realFinalResult Optional final destination (for chained ops)
   */
  static void matmul(NDArray* x, NDArray* y, NDArray* z,
                     const bool transX, const bool transY,
                     double alpha, double beta,
                     NDArray* realFinalResult = nullptr);

  /**
   * Set cublasLt epilogue state for the next matmul call on this thread.
   * Used by DSP executor to fuse bias+activation into the matmul kernel.
   * @param type 0=none, 1=bias, 2=bias+relu, 3=bias+gelu
   * @param biasPtr Device pointer to bias data
   * @param biasSize Size in bytes of the bias buffer
   */
  static void setLtEpilogue(int type, const void* biasPtr, int64_t biasSize);

  /**
   * Clear cublasLt epilogue state after matmul execution.
   */
  static void clearLtEpilogue();

  /**
   * Reset the mixed-precision cast cache indices to 0.
   * Must be called before CUDA graph capture to ensure cached buffers are
   * reused in the same order as the non-capture warmup execution.
   */
  static void resetCastCacheIndices();

  /**
   * Reset the mixed-precision cast cache indices to the given high-water-mark
   * values instead of 0.  Used during composite replay when a merged CUDA
   * graph already "owns" cast-cache slots [0, hwmA) / [0, hwmB): unmerged
   * gap matmuls must start from the high-water mark so they don't alias the
   * baked-in device pointers inside the merged graph.
   *
   * @param hwmA  First A-cache index not owned by any merged graph
   * @param hwmB  First B-cache index not owned by any merged graph
   */
  static void resetCastCacheIndicesTo(size_t hwmA, size_t hwmB);

  /**
   * Return the current cast-cache indices (A and B) as a pair.
   * Called after merged-capture completes to record the high-water mark:
   * the merged graph owns cache slots [0, idxA) and [0, idxB).
   */
  static std::pair<size_t, size_t> getCastCacheHighWaterMark();

  /**
   * Clear the entire cast cache (delete buffers and reset indices).
   * Must be called when shapes change (e.g. prefill → decode transition)
   * so the cache is repopulated with correctly-sized buffers during warmup.
   */
  static void clearCastCache();

  /**
   * Resolve transpose flags based on dimension compatibility
   * @return true if dimensions are compatible
   */
  static bool resolveTranspose(NDArray& a, NDArray& b, bool& transA, bool& transB);

  // ============== Tensor Dot Operations ==============

  static NDArray* tensorDot(NDArray* A, NDArray* B,
                            const std::initializer_list<LongType>& axesA,
                            const std::initializer_list<LongType>& axesB = {});

  static NDArray* tensorDot(NDArray* A, NDArray* B,
                            const std::vector<LongType>& axesA,
                            const std::vector<LongType>& axesB);

  static void tensorDot(NDArray* a, NDArray* b, NDArray* c,
                        std::vector<LongType>& axes_a,
                        std::vector<LongType>& axes_b,
                        std::vector<LongType>& permutForC);

  static void computeNewShapesAndAxes(
      NDArray& as_, const std::vector<LongType>& axes_a,
      NDArray& bs, const std::vector<LongType>& axes_b,
      std::vector<LongType>& newshape_a, std::vector<LongType>& newaxes_a,
      std::vector<LongType>& newshape_b, std::vector<LongType>& newaxes_b);

#ifndef __JAVACPP_HACK__
  static void tensorDot(NDArray* a, NDArray* b, NDArray* c,
                        std::vector<std::vector<LongType>>& modifA,
                        std::vector<std::vector<LongType>>& modifB,
                        std::vector<std::vector<LongType>>& modifC);

  static NDArray* tensorDot(NDArray* a, NDArray* b,
                            std::vector<std::vector<LongType>>& modifA,
                            std::vector<std::vector<LongType>>& modifB);

  static void tensorDot2(NDArray* a, NDArray* b, NDArray* c,
                         const std::vector<LongType>& axes_a,
                         const std::vector<LongType>& axes_b,
                         std::vector<LongType>& permutAt,
                         std::vector<LongType>& permuteBt,
                         std::vector<LongType>& permuteCt,
                         NDArray* realFinalResult = nullptr);
#endif
};

}  // namespace sd

#endif  // LIBND4J_MMULHELPER_H
