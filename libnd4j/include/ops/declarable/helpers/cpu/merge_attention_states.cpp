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
// Merge attention states — CPU implementation.
// Merges partial attention outputs from chunked prefill using
// numerically-stable log-sum-exp weighting.
//

#include <array/NDArrayFactory.h>
#include <execution/Threads.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/merge_attention_states.h>

#include <cmath>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////////
// Two-chunk merge (CPU)
// For each position: merge using LSE weights
//   m = max(lseA, lseB)
//   wA = exp(lseA - m),  wB = exp(lseB - m)
//   output = (wA * attnA + wB * attnB) / (wA + wB)
//   lseOut = m + log(wA + wB)
//////////////////////////////////////////////////////////////////////////////
template <typename T>
static void mergeAttentionStatesCpu_(NDArray* attnA, NDArray* lseA,
                                      NDArray* attnB, NDArray* lseB,
                                      NDArray* output, NDArray* lseOutput) {
    const LongType batch = attnA->sizeAt(0);
    const LongType numHeads = attnA->sizeAt(1);
    const LongType seqLen = attnA->sizeAt(2);
    const LongType headDim = attnA->sizeAt(3);
    const LongType totalPositions = batch * numHeads * seqLen;

    const T* aPtr = attnA->bufferAsT<T>();
    const float* lseAPtr = reinterpret_cast<const float*>(lseA->buffer());
    const T* bPtr = attnB->bufferAsT<T>();
    const float* lseBPtr = reinterpret_cast<const float*>(lseB->buffer());
    T* outPtr = output->bufferAsT<T>();
    float* lseOutPtr = (lseOutput != nullptr) ?
        reinterpret_cast<float*>(lseOutput->buffer()) : nullptr;

    auto func = PRAGMA_THREADS_FOR {
        for (auto pos = start; pos < stop; ++pos) {
            float lA = lseAPtr[pos];
            float lB = lseBPtr[pos];

            // Numerically stable merge
            float m = sd::math::sd_max<float>(lA, lB);
            float wA = sd::math::sd_exp<float, float>(lA - m);
            float wB = sd::math::sd_exp<float, float>(lB - m);
            float wSum = wA + wB;
            float invWSum = 1.0f / wSum;

            const T* aSrc = aPtr + pos * headDim;
            const T* bSrc = bPtr + pos * headDim;
            T* dst = outPtr + pos * headDim;

            for (LongType d = 0; d < headDim; ++d) {
                float valA = static_cast<float>(aSrc[d]);
                float valB = static_cast<float>(bSrc[d]);
                float merged = (wA * valA + wB * valB) * invWSum;
                dst[d] = static_cast<T>(merged);
            }

            if (lseOutPtr != nullptr) {
                lseOutPtr[pos] = m + sd::math::sd_log<float, float>(wSum);
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, totalPositions);
}

//////////////////////////////////////////////////////////////////////////////
// Public: mergeAttentionStates (two chunks)
//////////////////////////////////////////////////////////////////////////////
void mergeAttentionStates(LaunchContext* context,
                           NDArray* attnPrefix,
                           NDArray* lsePrefix,
                           NDArray* attnSuffix,
                           NDArray* lseSuffix,
                           NDArray* output,
                           NDArray* lseOutput) {
    BUILD_SINGLE_SELECTOR(attnPrefix->dataType(), mergeAttentionStatesCpu_,
                          (attnPrefix, lsePrefix, attnSuffix, lseSuffix, output, lseOutput),
                          SD_FLOAT_TYPES);

    output->tickWriteHost();
    if (lseOutput != nullptr) lseOutput->tickWriteHost();
}

//////////////////////////////////////////////////////////////////////////////
// Public: mergeAttentionStatesMulti (N chunks)
// Iteratively merges chunks using two-chunk merge.
//////////////////////////////////////////////////////////////////////////////
void mergeAttentionStatesMulti(LaunchContext* context,
                                const std::vector<NDArray*>& attnChunks,
                                const std::vector<NDArray*>& lseChunks,
                                NDArray* output,
                                NDArray* lseOutput) {
    if (attnChunks.empty()) {
        THROW_EXCEPTION("mergeAttentionStatesMulti: empty chunk list");
    }

    if (attnChunks.size() == 1) {
        output->assign(attnChunks[0]);
        if (lseOutput != nullptr) {
            lseOutput->assign(lseChunks[0]);
        }
        return;
    }

    if (attnChunks.size() == 2) {
        mergeAttentionStates(context, attnChunks[0], lseChunks[0],
                             attnChunks[1], lseChunks[1], output, lseOutput);
        return;
    }

    // N > 2: iterative pairwise merge
    const auto shape4d = attnChunks[0]->shapeOf();
    const auto shape3d = lseChunks[0]->shapeOf();

    std::vector<sd::LongType> attnShape = {shape4d[0], shape4d[1], shape4d[2], shape4d[3]};
    std::vector<sd::LongType> lseShape = {shape3d[0], shape3d[1], shape3d[2]};
    auto tempAttn = NDArrayFactory::create_(attnChunks[0]->ordering(),
        attnShape, attnChunks[0]->dataType(), context);
    auto tempLse = NDArrayFactory::create_(lseChunks[0]->ordering(),
        lseShape, DataType::FLOAT32, context);

    mergeAttentionStates(context, attnChunks[0], lseChunks[0],
                         attnChunks[1], lseChunks[1], tempAttn, tempLse);

    for (size_t i = 2; i < attnChunks.size(); i++) {
        if (i == attnChunks.size() - 1) {
            mergeAttentionStates(context, tempAttn, tempLse,
                                 attnChunks[i], lseChunks[i], output, lseOutput);
        } else {
            auto tempAttn2 = NDArrayFactory::create_(tempAttn->ordering(),
                attnShape, tempAttn->dataType(), context);
            auto tempLse2 = NDArrayFactory::create_(tempLse->ordering(),
                lseShape, DataType::FLOAT32, context);
            mergeAttentionStates(context, tempAttn, tempLse,
                                 attnChunks[i], lseChunks[i], tempAttn2, tempLse2);
            delete tempAttn;
            delete tempLse;
            tempAttn = tempAttn2;
            tempLse = tempLse2;
        }
    }

    delete tempAttn;
    delete tempLse;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
