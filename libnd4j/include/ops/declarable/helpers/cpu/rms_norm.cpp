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

#include <execution/Threads.h>
#include <math/templatemath.h>
#include <array/NDArrayFactory.h>
#include <helpers/MmulHelper.h>
#include <ops/declarable/helpers/rms_norm.h>
#if NOT_EXCLUDED(OP_rms_norm)
namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void rmsNorm_(NDArray* input, NDArray* gamma, NDArray* output, float epsilon) {
    const LongType numRows = input->lengthOf() / input->sizeAt(-1);
    const LongType rowLen = input->sizeAt(-1);

    const T* x = input->bufferAsT<T>();
    T* z = output->bufferAsT<T>();
    const T* g = gamma != nullptr ? gamma->bufferAsT<T>() : nullptr;
    const T eps = static_cast<T>(epsilon);

    auto func = PRAGMA_THREADS_FOR {
        for (auto row = start; row < stop; ++row) {
            const T* xRow = x + row * rowLen;
            T* zRow = z + row * rowLen;

            T sumSq = static_cast<T>(0);
            PRAGMA_OMP_SIMD_ARGS(reduction(+:sumSq))
            for (LongType i = 0; i < rowLen; ++i) {
                sumSq += xRow[i] * xRow[i];
            }
            const T invRms = static_cast<T>(1) / sd::math::sd_sqrt<T, T>(sumSq / static_cast<T>(rowLen) + eps);

            if (g != nullptr) {
                PRAGMA_OMP_SIMD
                for (LongType i = 0; i < rowLen; ++i) {
                    zRow[i] = xRow[i] * invRms * g[i];
                }
            } else {
                PRAGMA_OMP_SIMD
                for (LongType i = 0; i < rowLen; ++i) {
                    zRow[i] = xRow[i] * invRms;
                }
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, numRows);
}

void rmsNorm(LaunchContext* context, NDArray* input, NDArray* gamma, NDArray* output, float epsilon) {
    NDArray::preparePrimaryUse({output}, {input, gamma});

    BUILD_SINGLE_SELECTOR(input->dataType(), rmsNorm_, (input, gamma, output, epsilon), SD_FLOAT_TYPES);

    NDArray::registerPrimaryUse({output}, {input, gamma});
}

///////////////////////////////////////////////////////////////////////////////
// Fused RMSNorm + Linear: output = matmul(rmsNorm(input, gamma, eps), weight)
//
// Strategy: normalize rows in-place into a temp buffer using parallel_tad
// with SIMD vectorization, then delegate matmul to MmulHelper::mmul which
// dispatches to MKL/OneDNN/BLAS for optimal GEMM/GEMV performance.
///////////////////////////////////////////////////////////////////////////////
template <typename T>
static void rmsNormLinear_(NDArray* input, NDArray* gamma, NDArray* weight,
                            NDArray* output, float epsilon) {
    const LongType M = input->sizeAt(0);
    const LongType K = input->sizeAt(1);

    const T* x = input->bufferAsT<T>();
    const T* g = gamma != nullptr ? gamma->bufferAsT<T>() : nullptr;
    const T eps = static_cast<T>(epsilon);

    // Allocate temp buffer for normalized input [M, K]
    auto shapeVec = *input->getShapeAsVector();
    NDArray normalized(input->ordering(), shapeVec, input->dataType());
    T* normBuf = normalized.bufferAsT<T>();

    // Normalize rows with parallel_tad + SIMD
    auto func = PRAGMA_THREADS_FOR {
        for (auto row = start; row < stop; ++row) {
            const T* xRow = x + row * K;
            T* nRow = normBuf + row * K;

            // Reduction: sum of squares with SIMD
            T sumSq = static_cast<T>(0);
            PRAGMA_OMP_SIMD_ARGS(reduction(+:sumSq))
            for (LongType i = 0; i < K; ++i) {
                sumSq += xRow[i] * xRow[i];
            }
            const T invRms = static_cast<T>(1) / sd::math::sd_sqrt<T, T>(sumSq / static_cast<T>(K) + eps);

            // Normalize + scale with SIMD
            if (g != nullptr) {
                PRAGMA_OMP_SIMD
                for (LongType i = 0; i < K; ++i) {
                    nRow[i] = xRow[i] * invRms * g[i];
                }
            } else {
                PRAGMA_OMP_SIMD
                for (LongType i = 0; i < K; ++i) {
                    nRow[i] = xRow[i] * invRms;
                }
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, M);

    // Matmul: normalized [M, K] @ weight [K, N] -> output [M, N]
    // Delegates to MKL/OneDNN/BLAS via MmulHelper
    MmulHelper::mmul(&normalized, weight, output, 1.0, 0.0);
}

void rmsNormLinear(LaunchContext* context, NDArray* input, NDArray* gamma,
                    NDArray* weight, NDArray* output, float epsilon) {
    NDArray::preparePrimaryUse({output}, {input, gamma, weight});

    BUILD_SINGLE_SELECTOR(input->dataType(), rmsNormLinear_,
                           (input, gamma, weight, output, epsilon), SD_FLOAT_TYPES);

    NDArray::registerPrimaryUse({output}, {input, gamma, weight});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
