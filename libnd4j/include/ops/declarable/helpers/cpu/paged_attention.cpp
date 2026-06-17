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
// @author Adam Gibson
//
// CPU fallback implementations of paged attention ops.
// Primary target is CUDA; these provide correctness on CPU for testing.
//

#include <ops/declarable/helpers/paged_attention.h>
#include <array/NDArrayFactory.h>
#include <math/templatemath.h>
#include <system/op_boilerplate.h>
#include <cfloat>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void pagedAttentionForward_(
    NDArray* query,
    NDArray* keyBlockPool,
    NDArray* valueBlockPool,
    NDArray* pageTables,
    NDArray* contextLens,
    NDArray* output,
    const PagedAttentionConfig& config,
    LaunchContext* context) {

    int batch = static_cast<int>(query->sizeAt(0));
    int numHeads = config.numHeads > 0 ? config.numHeads : static_cast<int>(query->sizeAt(2));
    int headDim = config.headDim > 0 ? config.headDim : static_cast<int>(query->sizeAt(3));
    int numKvHeads = config.numKvHeads > 0 ? config.numKvHeads : numHeads;
    int blockSize = config.blockSize;
    int maxBlocksPerSeq = static_cast<int>(pageTables->sizeAt(1));

    T scale = static_cast<T>(config.scale);
    if (config.scale <= 0.0f) {
        scale = static_cast<T>(1.0f / std::sqrt(static_cast<float>(headDim)));
    }

    // Simple CPU implementation: iterate over batch, heads, compute attention
    PRAGMA_OMP_PARALLEL_FOR
    for (int b = 0; b < batch; b++) {
        int ctxLen = contextLens->e<int>(b);
        if (ctxLen <= 0) continue;

        for (int h = 0; h < numHeads; h++) {
            int kvHead = (numKvHeads > 0 && numKvHeads < numHeads)
                ? (h * numKvHeads / numHeads) : h;

            // Compute attention scores
            std::vector<T> scores(ctxLen);
            T maxScore = static_cast<T>(-FLT_MAX);

            for (int pos = 0; pos < ctxLen; pos++) {
                int logicalBlock = pos / blockSize;
                int offsetInBlock = pos % blockSize;
                int physicalBlock = pageTables->e<int>(b, logicalBlock);

                if (physicalBlock < 0) {
                    scores[pos] = static_cast<T>(-FLT_MAX);
                    continue;
                }

                T score = static_cast<T>(0);
                for (int d = 0; d < headDim; d++) {
                    T q = query->e<T>(b, 0, h, d);
                    T k = keyBlockPool->e<T>(physicalBlock, offsetInBlock, kvHead, d);
                    score += q * k;
                }
                scores[pos] = score * scale;
                if (scores[pos] > maxScore) maxScore = scores[pos];
            }

            // Softmax
            T sumExp = static_cast<T>(0);
            for (int pos = 0; pos < ctxLen; pos++) {
                scores[pos] = sd::math::sd_exp<T, T>(scores[pos] - maxScore);
                sumExp += scores[pos];
            }
            T invSum = static_cast<T>(1) / (sumExp + static_cast<T>(1e-8f));
            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for (int pos = 0; pos < ctxLen; pos++) {
                scores[pos] *= invSum;
            }

            // Weighted sum of values
            PRAGMA_OMP_PARALLEL_FOR
            for (int d = 0; d < headDim; d++) {
                T acc = static_cast<T>(0);
                for (int pos = 0; pos < ctxLen; pos++) {
                    int logicalBlock = pos / blockSize;
                    int offsetInBlock = pos % blockSize;
                    int physicalBlock = pageTables->e<int>(b, logicalBlock);
                    if (physicalBlock < 0) continue;

                    T v = valueBlockPool->e<T>(physicalBlock, offsetInBlock, kvHead, d);
                    acc += scores[pos] * v;
                }
                output->p<T>(b, 0, h, d, acc);
            }
        }
    }
}

void pagedAttentionForward(
    NDArray* query,
    NDArray* keyBlockPool,
    NDArray* valueBlockPool,
    NDArray* pageTables,
    NDArray* contextLens,
    NDArray* output,
    const PagedAttentionConfig& config,
    LaunchContext* context) {

    BUILD_SINGLE_SELECTOR(query->dataType(), pagedAttentionForward_,
                          (query, keyBlockPool, valueBlockPool, pageTables, contextLens, output, config, context),
                          SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void pagedAttentionForward_,
                      (NDArray*, NDArray*, NDArray*, NDArray*, NDArray*, NDArray*,
                       const PagedAttentionConfig&, LaunchContext*),
                      SD_FLOAT_TYPES);

template <typename T>
static void pagedKvCacheAppend_(
    NDArray* keyBlockPool,
    NDArray* valueBlockPool,
    NDArray* newKeys,
    NDArray* newValues,
    NDArray* pageTables,
    NDArray* contextLens,
    int blockSize,
    LaunchContext* context) {

    int batch = static_cast<int>(newKeys->sizeAt(0));
    int newLen = static_cast<int>(newKeys->sizeAt(1));
    int numKvHeads = static_cast<int>(newKeys->sizeAt(2));
    int headDim = static_cast<int>(newKeys->sizeAt(3));
    int maxBlocksPerSeq = static_cast<int>(pageTables->sizeAt(1));

    PRAGMA_OMP_PARALLEL_FOR
    for (int b = 0; b < batch; b++) {
        int startPos = contextLens->e<int>(b);

        for (int t = 0; t < newLen; t++) {
            int pos = startPos + t;
            int logicalBlock = pos / blockSize;
            int offsetInBlock = pos % blockSize;

            if (logicalBlock >= maxBlocksPerSeq) continue;

            int physicalBlock = pageTables->e<int>(b, logicalBlock);
            if (physicalBlock < 0) continue;

            for (int h = 0; h < numKvHeads; h++) {
                PRAGMA_OMP_PARALLEL_FOR_SIMD
                for (int d = 0; d < headDim; d++) {
                    T kVal = newKeys->e<T>(b, t, h, d);
                    T vVal = newValues->e<T>(b, t, h, d);
                    keyBlockPool->p<T>(physicalBlock, offsetInBlock, h, d, kVal);
                    valueBlockPool->p<T>(physicalBlock, offsetInBlock, h, d, vVal);
                }
            }
        }
    }
}

void pagedKvCacheAppend(
    NDArray* keyBlockPool,
    NDArray* valueBlockPool,
    NDArray* newKeys,
    NDArray* newValues,
    NDArray* pageTables,
    NDArray* contextLens,
    int blockSize,
    LaunchContext* context) {

    BUILD_SINGLE_SELECTOR(newKeys->dataType(), pagedKvCacheAppend_,
                          (keyBlockPool, valueBlockPool, newKeys, newValues, pageTables, contextLens, blockSize, context),
                          SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void pagedKvCacheAppend_,
                      (NDArray*, NDArray*, NDArray*, NDArray*, NDArray*, NDArray*, int, LaunchContext*),
                      SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
