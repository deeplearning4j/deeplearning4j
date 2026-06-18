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
// Multi-head Latent Attention (MLA) CPU implementation
//
// Decompresses a shared latent KV cache per-layer via kvDownProj,
// then performs standard scaled dot-product attention with GQA support.
//

#include <execution/Threads.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/mla_attention.h>

#if NOT_EXCLUDED(OP_mla_attention)
namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void mlaAttentionCpu_(NDArray* query, NDArray* latentKVCache, NDArray* kvDownProj,
                              NDArray* ropeCache, NDArray* output, int seqLen,
                              const MLAConfig& config) {
    const int batchSize = static_cast<int>(query->sizeAt(0));
    const int numHeads = config.numHeads;
    const int numKvHeads = config.numKvHeads;
    const int headDim = config.headDim;
    const int latentDim = config.latentDim;
    const int ropeHeadDim = config.ropeHeadDim;
    const int headsPerGroup = numHeads / numKvHeads;

    // Compute attention scale
    const T scale = config.attScale > 0.0f
        ? static_cast<T>(config.attScale)
        : static_cast<T>(1.0) / sd::math::sd_sqrt<T, T>(static_cast<T>(headDim));

    const T* qBuf = query->bufferAsT<T>();
    const T* latentBuf = latentKVCache->bufferAsT<T>();
    const T* projBuf = kvDownProj->bufferAsT<T>();
    T* outBuf = output->bufferAsT<T>();

    // kvDownProj layout: [latentDim, 2 * numKvHeads * headDim]
    // First half columns are K projection, second half are V projection
    const int kvProjectedDim = numKvHeads * headDim;

    // Process each (batch, head) pair in parallel
    const int totalWork = batchSize * numHeads;

    auto func = PRAGMA_THREADS_FOR {
        // Allocate thread-local scratch buffers
        // Decompressed K: [seqLen, headDim]  (for the relevant KV head)
        // Decompressed V: [seqLen, headDim]
        // Attention scores: [seqLen]
        std::vector<T> kBuf(seqLen * headDim);
        std::vector<T> vBuf(seqLen * headDim);
        std::vector<T> scores(seqLen);

        for (auto idx = start; idx < stop; ++idx) {
            const int b = idx / numHeads;
            const int h = idx % numHeads;
            const int kvHead = h / headsPerGroup;  // GQA mapping

            // Step 1: Decompress latent KV cache for this batch and KV head
            // latentKVCache[b, s, :] @ kvDownProj[:, kvHead*headDim : (kvHead+1)*headDim] -> K[s, headDim]
            // latentKVCache[b, s, :] @ kvDownProj[:, kvProjectedDim + kvHead*headDim : ...] -> V[s, headDim]
            for (int s = 0; s < seqLen; ++s) {
                const T* latentRow = latentBuf + (b * latentKVCache->sizeAt(1) + s) * latentDim;

                for (int d = 0; d < headDim; ++d) {
                    // K decompression: dot product of latent row with K projection column
                    T kVal = static_cast<T>(0);
                    const int kColIdx = kvHead * headDim + d;
                    for (int l = 0; l < latentDim; ++l) {
                        kVal += latentRow[l] * projBuf[l * (2 * kvProjectedDim) + kColIdx];
                    }
                    kBuf[s * headDim + d] = kVal;

                    // V decompression: dot product of latent row with V projection column
                    T vVal = static_cast<T>(0);
                    const int vColIdx = kvProjectedDim + kvHead * headDim + d;
                    for (int l = 0; l < latentDim; ++l) {
                        vVal += latentRow[l] * projBuf[l * (2 * kvProjectedDim) + vColIdx];
                    }
                    vBuf[s * headDim + d] = vVal;
                }
            }

            // Step 2: Compute attention scores: Q[b, 0, h, :] @ K[s, :]^T * scale
            // Query pointer for this batch and head
            const T* qRow = qBuf + (b * numHeads + h) * headDim;

            T maxScore = -std::numeric_limits<T>::max();
            for (int s = 0; s < seqLen; ++s) {
                T dot = static_cast<T>(0);
                for (int d = 0; d < headDim; ++d) {
                    dot += qRow[d] * kBuf[s * headDim + d];
                }
                scores[s] = dot * scale;
                if (scores[s] > maxScore) {
                    maxScore = scores[s];
                }
            }

            // Step 3: Softmax
            T sumExp = static_cast<T>(0);
            for (int s = 0; s < seqLen; ++s) {
                scores[s] = sd::math::sd_exp<T, T>(scores[s] - maxScore);
                sumExp += scores[s];
            }
            const T invSum = static_cast<T>(1) / sumExp;
            for (int s = 0; s < seqLen; ++s) {
                scores[s] *= invSum;
            }

            // Step 4: Weighted sum of V: attention_weights @ V -> output[b, 0, h, :]
            T* outRow = outBuf + (b * numHeads + h) * headDim;
            for (int d = 0; d < headDim; ++d) {
                T val = static_cast<T>(0);
                for (int s = 0; s < seqLen; ++s) {
                    val += scores[s] * vBuf[s * headDim + d];
                }
                outRow[d] = val;
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, totalWork);
}

void mlaAttention(NDArray* query, NDArray* latentKVCache, NDArray* kvDownProj,
                   NDArray* ropeCache, NDArray* output, int seqLen,
                   const MLAConfig& config,
                   LaunchContext* /*context*/) {
    BUILD_SINGLE_SELECTOR(query->dataType(), mlaAttentionCpu_,
                           (query, latentKVCache, kvDownProj, ropeCache, output, seqLen, config),
                           SD_FLOAT_TYPES);

    output->tickWriteHost();
    output->syncToDevice();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
