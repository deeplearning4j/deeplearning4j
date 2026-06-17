/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#include <ops/declarable/helpers/top_k_renorm.h>
#include <math/templatemath.h>
#include <system/op_boilerplate.h>
#include <algorithm>
#include <numeric>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void topKRenorm_(LaunchContext* context, NDArray* logits, NDArray* output, int k) {
    auto rank = logits->rankOf();
    LongType batch = 1;
    LongType vocabSize;

    if (rank == 1) {
        vocabSize = logits->sizeAt(0);
    } else {
        batch = logits->sizeAt(0);
        vocabSize = logits->sizeAt(1);
    }

    if (k <= 0 || k >= vocabSize) {
        // No filtering needed — just softmax
        PRAGMA_OMP_PARALLEL_FOR
        for (LongType b = 0; b < batch; b++) {
            float maxLogit = -std::numeric_limits<float>::infinity();
            for (LongType v = 0; v < vocabSize; v++) {
                float val = static_cast<float>((rank == 1) ? logits->e<T>(v) : logits->e<T>(b, v));
                if (val > maxLogit) maxLogit = val;
            }
            float sumExp = 0.0f;
            for (LongType v = 0; v < vocabSize; v++) {
                float val = static_cast<float>((rank == 1) ? logits->e<T>(v) : logits->e<T>(b, v));
                float prob = sd::math::sd_exp<float, float>(val - maxLogit);
                sumExp += prob;
            }
            for (LongType v = 0; v < vocabSize; v++) {
                float val = static_cast<float>((rank == 1) ? logits->e<T>(v) : logits->e<T>(b, v));
                float prob = sd::math::sd_exp<float, float>(val - maxLogit) / sumExp;
                if (rank == 1) {
                    output->p(v, prob);
                } else {
                    output->p(b, v, prob);
                }
            }
        }
        return;
    }

    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < batch; b++) {
        // Step 1: Compute softmax probabilities
        std::vector<float> probs(vocabSize);
        float maxLogit = -std::numeric_limits<float>::infinity();
        for (LongType v = 0; v < vocabSize; v++) {
            float val = static_cast<float>((rank == 1) ? logits->e<T>(v) : logits->e<T>(b, v));
            if (val > maxLogit) maxLogit = val;
        }
        float sumExp = 0.0f;
        for (LongType v = 0; v < vocabSize; v++) {
            float val = static_cast<float>((rank == 1) ? logits->e<T>(v) : logits->e<T>(b, v));
            probs[v] = sd::math::sd_exp<float, float>(val - maxLogit);
            sumExp += probs[v];
        }
        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (LongType v = 0; v < vocabSize; v++) {
            probs[v] /= sumExp;
        }

        // Step 2: Find K-th largest probability via partial sort
        std::vector<int> indices(vocabSize);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                         [&](int a, int b2) { return probs[a] > probs[b2]; });
        float threshold = probs[indices[k - 1]];

        // Step 3: Zero probabilities below threshold, renormalize
        float renormSum = 0.0f;
        for (LongType v = 0; v < vocabSize; v++) {
            if (probs[v] < threshold) {
                probs[v] = 0.0f;
            } else {
                renormSum += probs[v];
            }
        }

        if (renormSum > 0.0f) {
            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for (LongType v = 0; v < vocabSize; v++) {
                probs[v] /= renormSum;
            }
        }

        // Step 4: Write output
        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (LongType v = 0; v < vocabSize; v++) {
            if (rank == 1) {
                output->p(v, probs[v]);
            } else {
                output->p(b, v, probs[v]);
            }
        }
    }
}

BUILD_SINGLE_TEMPLATE(static void topKRenorm_, (LaunchContext* context, NDArray* logits, NDArray* output, int k),
                      SD_FLOAT_TYPES);

void topKRenorm(LaunchContext* context, NDArray* logits, NDArray* output, int k) {
    BUILD_SINGLE_SELECTOR(logits->dataType(), topKRenorm_, (context, logits, output, k), SD_FLOAT_TYPES);
}

template <typename T>
static void topPRenorm_(LaunchContext* context, NDArray* logits, NDArray* output, double p) {
    auto rank = logits->rankOf();
    LongType batch = 1;
    LongType vocabSize;

    if (rank == 1) {
        vocabSize = logits->sizeAt(0);
    } else {
        batch = logits->sizeAt(0);
        vocabSize = logits->sizeAt(1);
    }

    PRAGMA_OMP_PARALLEL_FOR
    for (LongType b = 0; b < batch; b++) {
        // Step 1: Compute softmax probabilities
        std::vector<float> probs(vocabSize);
        float maxLogit = -std::numeric_limits<float>::infinity();
        for (LongType v = 0; v < vocabSize; v++) {
            float val = static_cast<float>((rank == 1) ? logits->e<T>(v) : logits->e<T>(b, v));
            if (val > maxLogit) maxLogit = val;
        }
        float sumExp = 0.0f;
        for (LongType v = 0; v < vocabSize; v++) {
            float val = static_cast<float>((rank == 1) ? logits->e<T>(v) : logits->e<T>(b, v));
            probs[v] = sd::math::sd_exp<float, float>(val - maxLogit);
            sumExp += probs[v];
        }
        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (LongType v = 0; v < vocabSize; v++) {
            probs[v] /= sumExp;
        }

        // Step 2: Sort by descending probability
        std::vector<int> indices(vocabSize);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                 [&](int a, int b2) { return probs[a] > probs[b2]; });

        // Step 3: Accumulate until cumulative probability >= p
        float cumProb = 0.0f;
        int cutoff = vocabSize;
        for (int k = 0; k < vocabSize; k++) {
            cumProb += probs[indices[k]];
            if (cumProb >= static_cast<float>(p)) {
                cutoff = k + 1;
                break;
            }
        }

        // Step 4: Zero probabilities beyond cutoff
        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (int k = cutoff; k < vocabSize; k++) {
            probs[indices[k]] = 0.0f;
        }

        // Step 5: Renormalize
        float renormSum = 0.0f;
        for (LongType v = 0; v < vocabSize; v++) {
            renormSum += probs[v];
        }
        if (renormSum > 0.0f) {
            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for (LongType v = 0; v < vocabSize; v++) {
                probs[v] /= renormSum;
            }
        }

        // Step 6: Write output
        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (LongType v = 0; v < vocabSize; v++) {
            if (rank == 1) {
                output->p(v, probs[v]);
            } else {
                output->p(b, v, probs[v]);
            }
        }
    }
}

BUILD_SINGLE_TEMPLATE(static void topPRenorm_, (LaunchContext* context, NDArray* logits, NDArray* output, double p),
                      SD_FLOAT_TYPES);

void topPRenorm(LaunchContext* context, NDArray* logits, NDArray* output, double p) {
    BUILD_SINGLE_SELECTOR(logits->dataType(), topPRenorm_, (context, logits, output, p), SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
