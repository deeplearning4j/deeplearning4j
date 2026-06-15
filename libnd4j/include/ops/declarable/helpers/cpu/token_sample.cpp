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

#include <ops/declarable/helpers/token_sample.h>
#include <ops/declarable/helpers/sampling_penalties.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

void tokenSampleCpu(NDArray* logits, NDArray* output,
                    double temperature, int topK, double topP,
                    LongType seed, LaunchContext* context) {
  auto rank = logits->rankOf();
  LongType batch = 1;
  LongType vocabSize;
  LongType seqLen = 1;

  if (rank == 1) {
    vocabSize = logits->sizeAt(0);
  } else if (rank == 2) {
    batch = logits->sizeAt(0);
    vocabSize = logits->sizeAt(1);
  } else {
    // rank 3: [batch, seqLen, vocabSize]
    batch = logits->sizeAt(0);
    seqLen = logits->sizeAt(1);
    vocabSize = logits->sizeAt(2);
  }

  bool greedy = (temperature <= 0.0 && topK <= 0 && topP <= 0.0);

  for (LongType b = 0; b < batch; b++) {
    // Find the start of the logits row for this batch
    // For rank 3, use the last sequence position
    LongType seqPos = seqLen - 1;

    if (greedy) {
      // Argmax
      float maxVal = -std::numeric_limits<float>::infinity();
      LongType maxIdx = 0;
      for (LongType v = 0; v < vocabSize; v++) {
        float val;
        if (rank == 1) {
          val = logits->e<float>(v);
        } else if (rank == 2) {
          val = logits->e<float>(b, v);
        } else {
          val = logits->e<float>(b, seqPos, v);
        }
        if (val > maxVal) {
          maxVal = val;
          maxIdx = v;
        }
      }
      if (rank == 1) {
        output->p(0, maxIdx);
      } else {
        output->p(b, maxIdx);
      }
    } else {
      // Full sampling pipeline: temperature -> topK -> softmax -> topP -> sample
      std::vector<float> logitsVec(vocabSize);
      for (LongType v = 0; v < vocabSize; v++) {
        if (rank == 1) {
          logitsVec[v] = logits->e<float>(v);
        } else if (rank == 2) {
          logitsVec[v] = logits->e<float>(b, v);
        } else {
          logitsVec[v] = logits->e<float>(b, seqPos, v);
        }
      }

      // Temperature scaling
      if (temperature > 0.0) {
        for (auto& l : logitsVec) l /= static_cast<float>(temperature);
      }

      // TopK filtering
      std::vector<int> indices(vocabSize);
      std::iota(indices.begin(), indices.end(), 0);

      if (topK > 0 && topK < vocabSize) {
        std::partial_sort(indices.begin(), indices.begin() + topK, indices.end(),
                         [&](int a, int b2) { return logitsVec[a] > logitsVec[b2]; });
        float threshold = logitsVec[indices[topK - 1]];
        for (LongType v = 0; v < vocabSize; v++) {
          if (logitsVec[v] < threshold) logitsVec[v] = -std::numeric_limits<float>::infinity();
        }
      }

      // Softmax
      float maxLogit = *std::max_element(logitsVec.begin(), logitsVec.end());
      float sumExp = 0.0f;
      for (auto& l : logitsVec) {
        l = std::exp(l - maxLogit);
        sumExp += l;
      }
      for (auto& l : logitsVec) l /= sumExp;

      // TopP (nucleus) filtering
      if (topP > 0.0 && topP < 1.0) {
        std::sort(indices.begin(), indices.end(),
                 [&](int a, int b2) { return logitsVec[a] > logitsVec[b2]; });
        float cumProb = 0.0f;
        int cutoff = vocabSize;
        for (int k = 0; k < vocabSize; k++) {
          cumProb += logitsVec[indices[k]];
          if (cumProb >= topP) {
            cutoff = k + 1;
            break;
          }
        }
        for (int k = cutoff; k < vocabSize; k++) {
          logitsVec[indices[k]] = 0.0f;
        }
        // Re-normalize
        sumExp = 0.0f;
        for (auto& l : logitsVec) sumExp += l;
        for (auto& l : logitsVec) l /= sumExp;
      }

      // Sample from distribution
      std::mt19937 rng(seed > 0 ? static_cast<unsigned>(seed + b) : std::random_device{}());
      std::discrete_distribution<LongType> dist(logitsVec.begin(), logitsVec.end());
      LongType sampled = dist(rng);

      if (rank == 1) {
        output->p(0, sampled);
      } else {
        output->p(b, sampled);
      }
    }
  }
}

void tokenSampleWithPenaltiesCpu(NDArray* logits, NDArray* output,
                                 NDArray* inputIds,
                                 double temperature, int topK,
                                 double topP, double minP,
                                 double repPenalty, double freqPenalty,
                                 double presPenalty,
                                 LongType seed, LaunchContext* context) {
    // Step 1: Apply penalties to logits (in-place)
    if (inputIds != nullptr && (repPenalty != 1.0 || freqPenalty != 0.0 || presPenalty != 0.0)) {
        applyLogitPenaltiesCpu(logits, inputIds, repPenalty, freqPenalty, presPenalty, context);
    }

    // Step 2: Apply min-p filtering (in-place)
    if (minP > 0.0) {
        applyMinPFilterCpu(logits, minP, context);
    }

    // Step 3: Standard sampling (temperature, topK, topP)
    tokenSampleCpu(logits, output, temperature, topK, topP, seed, context);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
