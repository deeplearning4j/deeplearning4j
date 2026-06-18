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
#include <cmath>
#include <cfloat>

namespace sd {
namespace ops {
namespace helpers {

void pagedAttentionForward(
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

    float scale = config.scale;
    if (scale <= 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(headDim));
    }

    // Simple CPU implementation: iterate over batch, heads, compute attention
    for (int b = 0; b < batch; b++) {
        int ctxLen = contextLens->e<int>(b);
        if (ctxLen <= 0) continue;

        for (int h = 0; h < numHeads; h++) {
            int kvHead = (numKvHeads > 0 && numKvHeads < numHeads)
                ? (h * numKvHeads / numHeads) : h;

            // Compute attention scores
            std::vector<float> scores(ctxLen);
            float maxScore = -FLT_MAX;

            for (int pos = 0; pos < ctxLen; pos++) {
                int logicalBlock = pos / blockSize;
                int offsetInBlock = pos % blockSize;
                int physicalBlock = pageTables->e<int>(b, logicalBlock);

                if (physicalBlock < 0) {
                    scores[pos] = -FLT_MAX;
                    continue;
                }

                float score = 0.0f;
                for (int d = 0; d < headDim; d++) {
                    float q = query->e<float>(b, 0, h, d);
                    float k = keyBlockPool->e<float>(physicalBlock, offsetInBlock, kvHead, d);
                    score += q * k;
                }
                scores[pos] = score * scale;
                if (scores[pos] > maxScore) maxScore = scores[pos];
            }

            // Softmax
            float sumExp = 0.0f;
            for (int pos = 0; pos < ctxLen; pos++) {
                scores[pos] = std::exp(scores[pos] - maxScore);
                sumExp += scores[pos];
            }
            float invSum = 1.0f / (sumExp + 1e-8f);
            for (int pos = 0; pos < ctxLen; pos++) {
                scores[pos] *= invSum;
            }

            // Weighted sum of values
            for (int d = 0; d < headDim; d++) {
                float acc = 0.0f;
                for (int pos = 0; pos < ctxLen; pos++) {
                    int logicalBlock = pos / blockSize;
                    int offsetInBlock = pos % blockSize;
                    int physicalBlock = pageTables->e<int>(b, logicalBlock);
                    if (physicalBlock < 0) continue;

                    float v = valueBlockPool->e<float>(physicalBlock, offsetInBlock, kvHead, d);
                    acc += scores[pos] * v;
                }
                output->p(b, 0, h, d, acc);
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

    int batch = static_cast<int>(newKeys->sizeAt(0));
    int newLen = static_cast<int>(newKeys->sizeAt(1));
    int numKvHeads = static_cast<int>(newKeys->sizeAt(2));
    int headDim = static_cast<int>(newKeys->sizeAt(3));
    int maxBlocksPerSeq = static_cast<int>(pageTables->sizeAt(1));

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
                for (int d = 0; d < headDim; d++) {
                    float kVal = newKeys->e<float>(b, t, h, d);
                    float vVal = newValues->e<float>(b, t, h, d);
                    keyBlockPool->p(physicalBlock, offsetInBlock, h, d, kVal);
                    valueBlockPool->p(physicalBlock, offsetInBlock, h, d, vVal);
                }
            }
        }
    }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
