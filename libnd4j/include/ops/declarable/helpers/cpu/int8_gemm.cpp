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
// INT8 and FP8 scaled GEMM — CPU implementation.
// Reference implementation for correctness verification.
//

#include <execution/Threads.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/int8_gemm.h>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////////
// INT8 scaled GEMM (CPU)
// C_float = (A_int8 * B_int8) * scaleA * scaleB + bias
//////////////////////////////////////////////////////////////////////////////
void int8ScaledGemm(LaunchContext* context,
                     NDArray* A,
                     NDArray* B,
                     NDArray* scaleA,
                     NDArray* scaleB,
                     NDArray* output,
                     NDArray* bias) {
    const LongType M = A->sizeAt(0);
    const LongType K = A->sizeAt(1);
    const LongType N = B->sizeAt(1);

    const int8_t* aPtr = reinterpret_cast<const int8_t*>(A->buffer());
    const int8_t* bPtr = reinterpret_cast<const int8_t*>(B->buffer());
    const float* sAPtr = reinterpret_cast<const float*>(scaleA->buffer());
    const float* sBPtr = reinterpret_cast<const float*>(scaleB->buffer());
    const float* biasPtr = (bias != nullptr) ?
        reinterpret_cast<const float*>(bias->buffer()) : nullptr;

    bool perTokenA = (scaleA->lengthOf() > 1);
    bool perChannelB = (scaleB->lengthOf() > 1);

    auto dtype = output->dataType();

    if (dtype == DataType::FLOAT32) {
        float* outPtr = output->bufferAsT<float>();

        auto func = PRAGMA_THREADS_FOR {
            for (auto m = start; m < stop; ++m) {
                float sA = perTokenA ? sAPtr[m] : sAPtr[0];

                for (LongType n = 0; n < N; ++n) {
                    int32_t acc = 0;
                    for (LongType k = 0; k < K; ++k) {
                        acc += static_cast<int32_t>(aPtr[m * K + k]) *
                               static_cast<int32_t>(bPtr[k * N + n]);
                    }

                    float sB = perChannelB ? sBPtr[n] : sBPtr[0];
                    float val = static_cast<float>(acc) * sA * sB;
                    if (biasPtr != nullptr) val += biasPtr[n];
                    outPtr[m * N + n] = val;
                }
            }
        };
        samediff::Threads::parallel_tad(func, 0, M);
    } else {
        // Double or other types
        auto func = PRAGMA_THREADS_FOR {
            for (auto m = start; m < stop; ++m) {
                float sA = perTokenA ? sAPtr[m] : sAPtr[0];

                for (LongType n = 0; n < N; ++n) {
                    int32_t acc = 0;
                    for (LongType k = 0; k < K; ++k) {
                        acc += static_cast<int32_t>(aPtr[m * K + k]) *
                               static_cast<int32_t>(bPtr[k * N + n]);
                    }

                    float sB = perChannelB ? sBPtr[n] : sBPtr[0];
                    float val = static_cast<float>(acc) * sA * sB;
                    if (biasPtr != nullptr) val += biasPtr[n];

                    // Write via generic accessor
                    output->p(m * N + n, val);
                }
            }
        };
        samediff::Threads::parallel_tad(func, 0, M);
    }

    output->tickWriteHost();
}

//////////////////////////////////////////////////////////////////////////////
// FP8 scaled GEMM (CPU)
// Treats FP8 as INT8 bytes, dequantizes, does FP32 GEMM.
//////////////////////////////////////////////////////////////////////////////
void fp8ScaledGemm(LaunchContext* context,
                    NDArray* A,
                    NDArray* B,
                    NDArray* scaleA,
                    NDArray* scaleB,
                    NDArray* output) {
    const LongType M = A->sizeAt(0);
    const LongType K = A->sizeAt(1);
    const LongType N = B->sizeAt(1);

    // On CPU, FP8 is stored as int8 with scale factors.
    // Dequantize then do FP32 matmul.
    const int8_t* aPtr = reinterpret_cast<const int8_t*>(A->buffer());
    const int8_t* bPtr = reinterpret_cast<const int8_t*>(B->buffer());
    float sA = reinterpret_cast<const float*>(scaleA->buffer())[0];
    float sB = reinterpret_cast<const float*>(scaleB->buffer())[0];

    float* outPtr = output->bufferAsT<float>();

    auto func = PRAGMA_THREADS_FOR {
        for (auto m = start; m < stop; ++m) {
            for (LongType n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (LongType k = 0; k < K; ++k) {
                    float aVal = static_cast<float>(aPtr[m * K + k]) * sA;
                    float bVal = static_cast<float>(bPtr[k * N + n]) * sB;
                    acc += aVal * bVal;
                }
                outPtr[m * N + n] = acc;
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, M);

    output->tickWriteHost();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
