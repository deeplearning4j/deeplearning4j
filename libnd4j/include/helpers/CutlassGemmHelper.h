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

#ifndef LIBND4J_CUTLASS_GEMM_HELPER_H
#define LIBND4J_CUTLASS_GEMM_HELPER_H

#include <array/NDArray.h>
#include <system/common.h>

namespace sd {

/**
 * CUTLASS-based GEMM helper.
 *
 * Provides optimized matrix multiplication via NVIDIA CUTLASS templates.
 * Falls back gracefully when CUTLASS is unavailable or the operation
 * is not suited for CUTLASS (e.g., GEMV with M=1 where cuBLAS excels).
 *
 * Supported configurations:
 *   - FP16 x FP16 -> FP16  (SM80+, Ampere)
 *   - BF16 x BF16 -> BF16  (SM80+, Ampere)
 *   - FP32 x FP32 -> FP32  (SM80+, Ampere)
 *   - FP8_E4M3 x FP8_E4M3 -> FP16  (SM89+, Ada Lovelace)
 */
class SD_LIB_EXPORT CutlassGemmHelper {
 public:
  /**
   * Determines whether CUTLASS should be used for a given GEMM operation.
   *
   * @param M      Number of rows of A (and C)
   * @param N      Number of columns of B (and C)
   * @param K      Number of columns of A / rows of B
   * @param aType  Data type of matrix A
   * @param bType  Data type of matrix B
   * @param smVersion  SM version of the target device (e.g. 89 for RTX 4090)
   * @return true if CUTLASS should be preferred over cuBLAS
   */
  static bool shouldUseCutlass(LongType M, LongType N, LongType K,
                               DataType aType, DataType bType, int smVersion);

  /**
   * Performs C = alpha * A * B + beta * C using CUTLASS.
   *
   * @param A      Left input matrix [M, K]
   * @param B      Right input matrix [K, N]
   * @param C      Output matrix [M, N] (pre-allocated)
   * @param alpha  Scalar multiplier for A*B
   * @param beta   Scalar multiplier for existing C
   * @return true if CUTLASS executed successfully, false if fallback to cuBLAS is needed
   */
  static bool gemm(NDArray* A, NDArray* B, NDArray* C,
                    double alpha = 1.0, double beta = 0.0);
};

}  // namespace sd

#endif  // LIBND4J_CUTLASS_GEMM_HELPER_H
