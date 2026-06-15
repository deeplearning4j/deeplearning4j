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

//
// Fused Decoder Masked Multi-Head Attention CPU implementation
// Single-token decode: QKV projection + RoPE + KV cache write + masked attention + output projection
//

#include <execution/Threads.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/decoder_masked_mha.h>
#if NOT_EXCLUDED(OP_decoder_masked_mha)
namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void decoderMaskedMhaCpu_(
    NDArray* hiddenStates,
    NDArray* qkvWeight,
    NDArray* oWeight,
    NDArray* keyCache,
    NDArray* valueCache,
    NDArray* mask,
    NDArray* cosCache,
    NDArray* sinCache,
    NDArray* output,
    NDArray* updatedKeyCache,
    NDArray* updatedValueCache,
    const DecoderMhaConfig& config) {

    const int numHeads = config.numHeads;
    const int numKvHeads = config.numKvHeads;
    const int headDim = config.headDim;
    const int cachePosition = config.cachePosition;
    const int ropeType = config.ropeType;
    const float ropeFreqBase = config.ropeFreqBase;

    const LongType batch = hiddenStates->sizeAt(0);
    const LongType hiddenDim = hiddenStates->sizeAt(2);
    const LongType seqLen = cachePosition + 1;

    const T* hiddenBuf = hiddenStates->bufferAsT<T>();
    const T* qkvW = qkvWeight->bufferAsT<T>();
    const T* oW = oWeight->bufferAsT<T>();
    const T* keyCacheBuf = keyCache->bufferAsT<T>();
    const T* valueCacheBuf = valueCache->bufferAsT<T>();
    const T* maskBuf = mask != nullptr ? mask->bufferAsT<T>() : nullptr;
    const T* cosBuf = cosCache != nullptr ? cosCache->bufferAsT<T>() : nullptr;
    const T* sinBuf = sinCache != nullptr ? sinCache->bufferAsT<T>() : nullptr;

    T* outputBuf = output->bufferAsT<T>();
    T* updatedKBuf = updatedKeyCache->bufferAsT<T>();
    T* updatedVBuf = updatedValueCache->bufferAsT<T>();

    // Copy existing cache to output cache
    const LongType kvCacheMaxSeq = keyCache->sizeAt(2);
    const LongType kvCacheSize = batch * numKvHeads * kvCacheMaxSeq * headDim;
    for (LongType i = 0; i < kvCacheSize; ++i) {
        updatedKBuf[i] = keyCacheBuf[i];
        updatedVBuf[i] = valueCacheBuf[i];
    }

    // Compute attention scale
    float attScale = config.attScale;
    if (attScale == 0.0f) {
        attScale = static_cast<float>(1.0) / sd::math::sd_sqrt<float, float>(static_cast<float>(headDim));
    }

    const LongType totalQkvDim = 3 * hiddenDim;

    auto func = PRAGMA_THREADS_FOR {
        for (auto b = start; b < stop; ++b) {
            // Temporary storage for QKV projection result: [3 * hiddenDim]
            std::vector<T> qkv(totalQkvDim, static_cast<T>(0));

            // Step 1: QKV projection: hidden [1, H] @ qkvWeight [H, 3H] -> qkv [3H]
            const T* hRow = hiddenBuf + b * hiddenDim;
            for (LongType j = 0; j < totalQkvDim; ++j) {
                T sum = static_cast<T>(0);
                for (LongType k = 0; k < hiddenDim; ++k) {
                    sum += hRow[k] * qkvW[k * totalQkvDim + j];
                }
                qkv[j] = sum;
            }

            // Step 2: Split into Q, K, V
            // Q: [numHeads, headDim], K: [numKvHeads, headDim], V: [numKvHeads, headDim]
            const LongType qSize = numHeads * headDim;
            const LongType kvSize = numKvHeads * headDim;
            T* qPtr = qkv.data();
            T* kPtr = qkv.data() + qSize;
            T* vPtr = qkv.data() + qSize + kvSize;

            // Step 3: Apply RoPE if enabled
            if (ropeType > 0 && cosBuf != nullptr && sinBuf != nullptr) {
                const int halfDim = headDim / 2;
                const T* cosRow = cosBuf + cachePosition * halfDim;
                const T* sinRow = sinBuf + cachePosition * halfDim;

                // Apply RoPE to Q heads
                for (int h = 0; h < numHeads; ++h) {
                    T* qHead = qPtr + h * headDim;
                    for (int i = 0; i < halfDim; ++i) {
                        T c = static_cast<T>(cosRow[i]);
                        T s = static_cast<T>(sinRow[i]);
                        if (ropeType == 1) {
                            // Standard RoPE: pair (i, i + halfDim)
                            T x0 = qHead[i];
                            T x1 = qHead[i + halfDim];
                            qHead[i] = x0 * c - x1 * s;
                            qHead[i + halfDim] = x0 * s + x1 * c;
                        } else {
                            // NeoX RoPE: pair (2i, 2i+1)
                            T x0 = qHead[2 * i];
                            T x1 = qHead[2 * i + 1];
                            qHead[2 * i] = x0 * c - x1 * s;
                            qHead[2 * i + 1] = x0 * s + x1 * c;
                        }
                    }
                }

                // Apply RoPE to K heads
                for (int h = 0; h < numKvHeads; ++h) {
                    T* kHead = kPtr + h * headDim;
                    for (int i = 0; i < halfDim; ++i) {
                        T c = static_cast<T>(cosRow[i]);
                        T s = static_cast<T>(sinRow[i]);
                        if (ropeType == 1) {
                            T x0 = kHead[i];
                            T x1 = kHead[i + halfDim];
                            kHead[i] = x0 * c - x1 * s;
                            kHead[i + halfDim] = x0 * s + x1 * c;
                        } else {
                            T x0 = kHead[2 * i];
                            T x1 = kHead[2 * i + 1];
                            kHead[2 * i] = x0 * c - x1 * s;
                            kHead[2 * i + 1] = x0 * s + x1 * c;
                        }
                    }
                }
            }

            // Step 4: Write K, V to cache at cachePosition
            // Cache layout: [B, numKvHeads, maxSeq, headDim]
            for (int h = 0; h < numKvHeads; ++h) {
                LongType cacheOffset = ((b * numKvHeads + h) * kvCacheMaxSeq + cachePosition) * headDim;
                const T* kHead = kPtr + h * headDim;
                const T* vHead = vPtr + h * headDim;
                for (int d = 0; d < headDim; ++d) {
                    updatedKBuf[cacheOffset + d] = kHead[d];
                    updatedVBuf[cacheOffset + d] = vHead[d];
                }
            }

            // Step 5: Compute attention for each query head
            // attnOut: [numHeads, headDim]
            std::vector<T> attnOut(numHeads * headDim, static_cast<T>(0));

            for (int qh = 0; qh < numHeads; ++qh) {
                // GQA mapping: which KV head does this query head use
                int kvh = qh * numKvHeads / numHeads;

                const T* qHead = qPtr + qh * headDim;

                // Compute attention scores: Q @ K^T for all positions up to seqLen
                std::vector<T> scores(seqLen);
                T maxScore = static_cast<T>(-1e30);

                for (LongType pos = 0; pos < seqLen; ++pos) {
                    LongType kOffset = ((b * numKvHeads + kvh) * kvCacheMaxSeq + pos) * headDim;
                    T dot = static_cast<T>(0);
                    for (int d = 0; d < headDim; ++d) {
                        dot += static_cast<T>(qHead[d]) * static_cast<T>(updatedKBuf[kOffset + d]);
                    }
                    dot *= static_cast<T>(attScale);

                    // Apply mask if present
                    if (maskBuf != nullptr) {
                        // mask layout: [B, 1, 1, seqLen]
                        dot += static_cast<T>(maskBuf[b * seqLen + pos]);
                    }

                    scores[pos] = dot;
                    if (dot > maxScore) maxScore = dot;
                }

                // Softmax
                T sumExp = static_cast<T>(0);
                for (LongType pos = 0; pos < seqLen; ++pos) {
                    scores[pos] = sd::math::sd_exp<T, T>(scores[pos] - maxScore);
                    sumExp += scores[pos];
                }
                T invSumExp = static_cast<T>(1) / sumExp;
                for (LongType pos = 0; pos < seqLen; ++pos) {
                    scores[pos] *= invSumExp;
                }

                // Weighted sum of values: attn_scores @ V
                T* attnHead = attnOut.data() + qh * headDim;
                for (LongType pos = 0; pos < seqLen; ++pos) {
                    LongType vOffset = ((b * numKvHeads + kvh) * kvCacheMaxSeq + pos) * headDim;
                    T weight = static_cast<T>(scores[pos]);
                    for (int d = 0; d < headDim; ++d) {
                        attnHead[d] += weight * updatedVBuf[vOffset + d];
                    }
                }
            }

            // Step 6: Concatenate heads and output projection
            // attnOut is [numHeads * headDim] = [hiddenDim]
            // output = attnOut @ oWeight [H, H]
            T* outRow = outputBuf + b * hiddenDim;
            for (LongType j = 0; j < hiddenDim; ++j) {
                T sum = static_cast<T>(0);
                for (LongType k = 0; k < hiddenDim; ++k) {
                    sum += attnOut[k] * oW[k * hiddenDim + j];
                }
                outRow[j] = sum;
            }
        }
    };

    samediff::Threads::parallel_tad(func, 0, batch);
}

void decoderMaskedMhaCpu(
    NDArray* hiddenStates,
    NDArray* qkvWeight,
    NDArray* oWeight,
    NDArray* keyCache,
    NDArray* valueCache,
    NDArray* mask,
    NDArray* cosCache,
    NDArray* sinCache,
    NDArray* output,
    NDArray* updatedKeyCache,
    NDArray* updatedValueCache,
    const DecoderMhaConfig& config) {

    hiddenStates->syncToHost();
    qkvWeight->syncToHost();
    oWeight->syncToHost();
    keyCache->syncToHost();
    valueCache->syncToHost();
    if (mask != nullptr) mask->syncToHost();
    if (cosCache != nullptr) cosCache->syncToHost();
    if (sinCache != nullptr) sinCache->syncToHost();

    BUILD_SINGLE_SELECTOR(hiddenStates->dataType(), decoderMaskedMhaCpu_,
                          (hiddenStates, qkvWeight, oWeight, keyCache, valueCache,
                           mask, cosCache, sinCache, output, updatedKeyCache, updatedValueCache, config),
                          SD_FLOAT_TYPES);

    output->tickWriteHost();
    output->syncToDevice();
    updatedKeyCache->tickWriteHost();
    updatedKeyCache->syncToDevice();
    updatedValueCache->tickWriteHost();
    updatedValueCache->syncToDevice();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
