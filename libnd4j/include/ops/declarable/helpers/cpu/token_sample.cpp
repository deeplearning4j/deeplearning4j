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
#include <math/templatemath.h>
#include <system/op_boilerplate.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void tokenSample_(NDArray* logits, NDArray* output,
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

  PRAGMA_OMP_PARALLEL_FOR
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
          val = static_cast<float>(logits->e<T>(v));
        } else if (rank == 2) {
          val = static_cast<float>(logits->e<T>(b, v));
        } else {
          val = static_cast<float>(logits->e<T>(b, seqPos, v));
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
      PRAGMA_OMP_PARALLEL_FOR_SIMD
      for (LongType v = 0; v < vocabSize; v++) {
        if (rank == 1) {
          logitsVec[v] = static_cast<float>(logits->e<T>(v));
        } else if (rank == 2) {
          logitsVec[v] = static_cast<float>(logits->e<T>(b, v));
        } else {
          logitsVec[v] = static_cast<float>(logits->e<T>(b, seqPos, v));
        }
      }

      // Temperature scaling
      if (temperature > 0.0) {
        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (LongType v = 0; v < vocabSize; v++) logitsVec[v] /= static_cast<float>(temperature);
      }

      // TopK filtering
      std::vector<int> indices(vocabSize);
      std::iota(indices.begin(), indices.end(), 0);

      if (topK > 0 && topK < vocabSize) {
        std::partial_sort(indices.begin(), indices.begin() + topK, indices.end(),
                         [&](int a, int b2) { return logitsVec[a] > logitsVec[b2]; });
        float threshold = logitsVec[indices[topK - 1]];
        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (LongType v = 0; v < vocabSize; v++) {
          if (logitsVec[v] < threshold) logitsVec[v] = -std::numeric_limits<float>::infinity();
        }
      }

      // Softmax
      float maxLogit = *std::max_element(logitsVec.begin(), logitsVec.end());
      float sumExp = 0.0f;
      for (LongType v = 0; v < vocabSize; v++) {
        logitsVec[v] = sd::math::sd_exp<float, float>(logitsVec[v] - maxLogit);
        sumExp += logitsVec[v];
      }
      PRAGMA_OMP_PARALLEL_FOR_SIMD
      for (LongType v = 0; v < vocabSize; v++) logitsVec[v] /= sumExp;

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
        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (int k = cutoff; k < vocabSize; k++) {
          logitsVec[indices[k]] = 0.0f;
        }
        // Re-normalize
        sumExp = 0.0f;
        for (LongType v = 0; v < vocabSize; v++) sumExp += logitsVec[v];
        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (LongType v = 0; v < vocabSize; v++) logitsVec[v] /= sumExp;
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

void tokenSample(NDArray* logits, NDArray* output,
                    double temperature, int topK, double topP,
                    LongType seed, LaunchContext* context) {
  BUILD_SINGLE_SELECTOR(logits->dataType(), tokenSample_,
                        (logits, output, temperature, topK, topP, seed, context),
                        SD_FLOAT_TYPES);
}

void tokenSampleWithPenalties(NDArray* logits, NDArray* output,
                                 NDArray* inputIds,
                                 double temperature, int topK,
                                 double topP, double minP,
                                 double repPenalty, double freqPenalty,
                                 double presPenalty,
                                 double typicalP,
                                 double xtcProbability, double xtcThreshold,
                                 LongType seed, LaunchContext* context) {
    // Sampling pipeline order (see token_sample.h for canonical documentation):
    // 1. Penalties (repetition / frequency / presence)
    // 2. Temperature + top-k + min-p + top-p → handled by tokenSample_
    // 3. Typical-p filter (entropy-deviation, applied after softmax stage inside tokenSample_
    //    cannot be injected there cleanly — so we pre-filter on logits before tokenSample_)
    // 4. XTC filter
    // 5. multinomial / greedy in tokenSample_
    //
    // NOTE: typical-p and XTC are both applied to pre-softmax logits (in-place).
    // typical-p: operates on the post-temperature logits; since tokenSample_ applies temperature
    // internally, we must apply typical-p *after* we scale the logits ourselves here and
    // pass temperature=1.0 to tokenSample_. However, to stay compatible with the tokenSample_
    // flow (which also does topK, topP), we apply typical-p to the raw logits before calling
    // tokenSample_. The semantic difference is minor: typical-p is designed to operate on
    // the softmax probability distribution, which is shift-invariant w.r.t. temperature
    // scaling only if typicalP acts on the temperature-scaled logits. Conservative decision:
    // apply typical-p to logits after penalties but before tokenSample_ (which applies
    // temperature). This matches llama.cpp's pipeline placement (typical sampling runs
    // before temperature in llama.cpp 2024+, after penalties).

    // Step 1: Apply penalties to logits (in-place)
    if (inputIds != nullptr && (repPenalty != 1.0 || freqPenalty != 0.0 || presPenalty != 0.0)) {
        applyLogitPenalties(logits, inputIds, repPenalty, freqPenalty, presPenalty, context);
    }

    // Step 2: Apply min-p filtering (in-place, uses softmax internally)
    if (minP > 0.0) {
        applyMinPFilter(logits, minP, context);
    }

    // Step 3: Apply typical-p filtering (in-place; computes its own softmax)
    if (typicalP > 0.0 && typicalP < 1.0) {
        applyTypicalPFilter(logits, typicalP, context);
    }

    // Step 4: Apply XTC filter (in-place; stochastic, uses seed)
    if (xtcProbability > 0.0) {
        applyXtcFilter(logits, xtcProbability, xtcThreshold, seed, context);
    }

    // Step 5: Standard sampling (temperature, topK, topP)
    tokenSample(logits, output, temperature, topK, topP, seed, context);
}

static int resolveScalarStrategy(const TokenSampleConfig& config) {
    if (config.strategy == TOKEN_SAMPLE_AUTO) {
        return (config.temperature <= 0.0 || (config.topK <= 1 && config.topP <= 0.0))
               ? TOKEN_SAMPLE_GREEDY : TOKEN_SAMPLE_SAMPLE;
    }
    return config.strategy;
}

static bool shouldSuppressStopTokens(const TokenSampleConfig& config) {
    return config.minNewTokens > 0
        && config.generatedTokenOffset < config.minNewTokens
        && config.stopTokenIds != nullptr
        && config.stopTokenCount > 0;
}

template <typename T>
static void suppressStopTokens_(NDArray* logits, const TokenSampleConfig& config) {
    if (!shouldSuppressStopTokens(config)) return;

    const int rank = logits->rankOf();
    LongType batch = 1;
    LongType vocabSize;
    LongType seqLen = 1;
    if (rank == 1) {
        vocabSize = logits->sizeAt(0);
    } else if (rank == 2) {
        batch = logits->sizeAt(0);
        vocabSize = logits->sizeAt(1);
    } else {
        batch = logits->sizeAt(0);
        seqLen = logits->sizeAt(1);
        vocabSize = logits->sizeAt(2);
    }

    T* buf = logits->bufferAsT<T>();
    auto strides = logits->stridesOf();
    LongType batchStride = 0;
    LongType elemStride;
    LongType rowOffset = 0;
    if (rank == 1) {
        elemStride = strides[0];
    } else if (rank == 2) {
        batchStride = strides[0];
        elemStride = strides[1];
    } else {
        batchStride = strides[0];
        elemStride = strides[2];
        rowOffset = (seqLen - 1) * strides[1];
    }

    for (LongType b = 0; b < batch; b++) {
        LongType base = b * batchStride + rowOffset;
        for (int i = 0; i < config.stopTokenCount; i++) {
            int stopId = config.stopTokenIds[i];
            if (stopId >= 0 && stopId < vocabSize) {
                buf[base + static_cast<LongType>(stopId) * elemStride] =
                    static_cast<T>(-std::numeric_limits<float>::infinity());
            }
        }
    }
}

static void suppressStopTokens(NDArray* logits, const TokenSampleConfig& config, LaunchContext* context) {
    BUILD_SINGLE_SELECTOR(logits->dataType(), suppressStopTokens_,
                          (logits, config), SD_FLOAT_TYPES);
}

void tokenSamplePolicy(NDArray* logits, NDArray* output,
                       NDArray* inputIds,
                       const TokenSampleConfig& config,
                       TokenSampleResult* result,
                       LaunchContext* context) {
    const int strategy = resolveScalarStrategy(config);
    const bool scalar = config.batchMax == 1 && config.windowMax == 1
                        && config.activeBatch == 1 && config.activeWindow == 1;
    if (!scalar || (strategy != TOKEN_SAMPLE_GREEDY && strategy != TOKEN_SAMPLE_SAMPLE)) {
        THROW_EXCEPTION("tokenSamplePolicy: only scalar GREEDY/SAMPLE is implemented until "
                        "autoregressive_decode exposes the ADR 0106 B/W substrate");
    }

    if (result != nullptr) {
        result->selectedToken = -1;
        result->selectedBatch = 0;
        result->selectedWindow = 0;
        result->acceptedCount = 1;
        result->parentBeam = 0;
        result->score = 0.0;
    }

    suppressStopTokens(logits, config, context);

    if (strategy == TOKEN_SAMPLE_GREEDY) {
        tokenSample(logits, output, 0.0, 0, 0.0, config.seed, context);
    } else {
        const bool hasPenalties = inputIds != nullptr
            && (config.repPenalty != 1.0 || config.freqPenalty != 0.0 || config.presPenalty != 0.0);
        const bool hasExtendedSamplers = config.minP > 0.0
            || (config.typicalP > 0.0 && config.typicalP < 1.0)
            || config.xtcProbability > 0.0;
        if (hasPenalties || hasExtendedSamplers) {
            tokenSampleWithPenalties(logits, output, inputIds,
                                     config.temperature, config.topK, config.topP, config.minP,
                                     config.repPenalty, config.freqPenalty, config.presPenalty,
                                     config.typicalP, config.xtcProbability, config.xtcThreshold,
                                     config.seed, context);
        } else {
            tokenSample(logits, output, config.temperature, config.topK, config.topP,
                        config.seed, context);
        }
    }

    if (result != nullptr && output != nullptr && output->lengthOf() > 0) {
        result->selectedToken = output->e<LongType>(0);
    }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
