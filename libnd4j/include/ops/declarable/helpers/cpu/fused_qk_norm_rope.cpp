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
// Fused QK RMSNorm + RoPE — CPU implementation.
// Combines RMSNorm per head + Rotary Position Embedding on Q and K.
//

#include <execution/Threads.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/fused_qk_norm_rope.h>

#include <cmath>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////////
// Process one head tensor: RMSNorm + RoPE
// input:  [B, S, H, D]
// output: [B, S, H, D]
// gamma:  [D]
//////////////////////////////////////////////////////////////////////////////
template <typename T>
static void fusedNormRopeCpu_(const T* input, const T* gamma,
                               const T* cosCache, const T* sinCache,
                               T* output,
                               LongType batch, LongType seqLen,
                               LongType numHeads, LongType headDim,
                               float epsilon, double freqBase, bool isNeox) {
    const LongType totalHeads = batch * seqLen * numHeads;
    const LongType halfDim = headDim / 2;

    auto func = PRAGMA_THREADS_FOR {
        for (auto idx = start; idx < stop; ++idx) {
            LongType b = idx / (seqLen * numHeads);
            LongType rem = idx % (seqLen * numHeads);
            LongType s = rem / numHeads;
            LongType h = rem % numHeads;

            LongType offset = ((b * seqLen + s) * numHeads + h) * headDim;
            const T* inPtr = input + offset;
            T* outPtr = output + offset;

            // Step 1: compute RMS
            float sumSq = 0.0f;
            for (LongType d = 0; d < headDim; ++d) {
                float val = static_cast<float>(inPtr[d]);
                sumSq += val * val;
            }
            float rms = std::sqrt(sumSq / static_cast<float>(headDim) + epsilon);
            float rmsInv = 1.0f / rms;

            // Step 2: normalize with gamma, then apply RoPE
            // First, compute normalized values into a temp buffer
            // (needed because RoPE pairs reference each other)
            std::vector<float> normed(headDim);
            for (LongType d = 0; d < headDim; ++d) {
                normed[d] = static_cast<float>(inPtr[d]) * rmsInv * static_cast<float>(gamma[d]);
            }

            // Step 3: apply RoPE rotation
            for (LongType d = 0; d < headDim; ++d) {
                float cosVal, sinVal;

                if (cosCache != nullptr && sinCache != nullptr) {
                    LongType pairIdx = d / 2;
                    if (pairIdx < halfDim) {
                        cosVal = static_cast<float>(cosCache[s * halfDim + pairIdx]);
                        sinVal = static_cast<float>(sinCache[s * halfDim + pairIdx]);
                    } else {
                        cosVal = 1.0f;
                        sinVal = 0.0f;
                    }
                } else {
                    // Compute on-the-fly
                    LongType pairIdx;
                    if (isNeox) {
                        pairIdx = d % halfDim;
                    } else {
                        pairIdx = d / 2;
                    }
                    float theta = static_cast<float>(s) * std::pow(static_cast<float>(freqBase),
                                   -2.0f * static_cast<float>(pairIdx) / static_cast<float>(headDim));
                    cosVal = std::cos(theta);
                    sinVal = std::sin(theta);
                }

                float rotatedVal;
                if (isNeox) {
                    // NeoX style: split-half
                    if (d < halfDim) {
                        rotatedVal = normed[d] * cosVal - normed[d + halfDim] * sinVal;
                    } else {
                        rotatedVal = normed[d - halfDim] * sinVal + normed[d] * cosVal;
                    }
                } else {
                    // GPT-J style: interleaved pairs
                    if (d % 2 == 0) {
                        rotatedVal = normed[d] * cosVal - normed[d + 1] * sinVal;
                    } else {
                        rotatedVal = normed[d - 1] * sinVal + normed[d] * cosVal;
                    }
                }

                outPtr[d] = static_cast<T>(rotatedVal);
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, totalHeads);
}

//////////////////////////////////////////////////////////////////////////////
// Public interface
//////////////////////////////////////////////////////////////////////////////
void fusedQkNormRope(LaunchContext* context,
                      NDArray* query,
                      NDArray* key,
                      NDArray* gammaQ,
                      NDArray* gammaK,
                      NDArray* cosCache,
                      NDArray* sinCache,
                      NDArray* queryOut,
                      NDArray* keyOut,
                      float epsilon,
                      double freqBase,
                      bool isNeox) {
    const LongType batch = query->sizeAt(0);
    const LongType seqLen = query->sizeAt(1);
    const LongType numQHeads = query->sizeAt(2);
    const LongType headDim = query->sizeAt(3);
    const LongType numKVHeads = key->sizeAt(2);

    auto dtype = query->dataType();

    const void* cosPtr = (cosCache != nullptr) ? cosCache->buffer() : nullptr;
    const void* sinPtr = (sinCache != nullptr) ? sinCache->buffer() : nullptr;

    // Process Q heads
    if (dtype == DataType::FLOAT32) {
        fusedNormRopeCpu_<float>(
            query->bufferAsT<float>(), gammaQ->bufferAsT<float>(),
            reinterpret_cast<const float*>(cosPtr), reinterpret_cast<const float*>(sinPtr),
            queryOut->bufferAsT<float>(),
            batch, seqLen, numQHeads, headDim, epsilon, freqBase, isNeox);
    } else if (dtype == DataType::DOUBLE) {
        fusedNormRopeCpu_<double>(
            query->bufferAsT<double>(), gammaQ->bufferAsT<double>(),
            reinterpret_cast<const double*>(cosPtr), reinterpret_cast<const double*>(sinPtr),
            queryOut->bufferAsT<double>(),
            batch, seqLen, numQHeads, headDim, epsilon, freqBase, isNeox);
    } else {
        THROW_EXCEPTION("fusedQkNormRope CPU: unsupported data type (use FLOAT32 or DOUBLE)");
    }

    // Process K heads
    if (dtype == DataType::FLOAT32) {
        fusedNormRopeCpu_<float>(
            key->bufferAsT<float>(), gammaK->bufferAsT<float>(),
            reinterpret_cast<const float*>(cosPtr), reinterpret_cast<const float*>(sinPtr),
            keyOut->bufferAsT<float>(),
            batch, seqLen, numKVHeads, headDim, epsilon, freqBase, isNeox);
    } else if (dtype == DataType::DOUBLE) {
        fusedNormRopeCpu_<double>(
            key->bufferAsT<double>(), gammaK->bufferAsT<double>(),
            reinterpret_cast<const double*>(cosPtr), reinterpret_cast<const double*>(sinPtr),
            keyOut->bufferAsT<double>(),
            batch, seqLen, numKVHeads, headDim, epsilon, freqBase, isNeox);
    }

    queryOut->tickWriteHost();
    keyOut->tickWriteHost();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
