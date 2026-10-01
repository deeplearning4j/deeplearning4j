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
#include <ops/declarable/helpers/reproducible_math.h>
#include <ops/declarable/helpers/sampling_penalties.h>
#include <array/DataTypeUtils.h>
#include <math/templatemath.h>
#include <ops/op_types.h>
#include <system/op_boilerplate.h>
#include <algorithm>
#include <functional>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

// Selects the token of one row at the draw u (see tokenSampleDraw) and the probability of that
// selection. row addresses token v at row[v * elemStride].
template <typename T, typename AccT>
static void tokenSampleDrawRow(const T* row, LongType vocabSize, LongType elemStride, bool greedy, AccT invTemp,
                               int topK, AccT topP, AccT u, LongType& token, AccT& probability) {
  if (greedy) {
    // Strict comparison keeps the lowest index of a tie and never takes NaN or -inf.
    AccT best = -DataTypeUtils::infOrMax<AccT>();
    LongType bestIndex = -1;
    for (LongType v = 0; v < vocabSize; v++) {
      const AccT value = static_cast<AccT>(row[v * elemStride]);
      if (value > best) {
        best = value;
        bestIndex = v;
      }
    }
    token = bestIndex >= 0 ? bestIndex : 0;
    probability = static_cast<AccT>(bestIndex >= 0 ? 1 : 0);
    return;
  }

  // The maximum starts at the lowest finite value, so a row of -inf weighs 0 throughout.
  std::vector<AccT> weights(vocabSize);
  AccT rowMax = -DataTypeUtils::max<AccT>();
  for (LongType v = 0; v < vocabSize; v++) {
    weights[v] = tokenSampleScaled<AccT>(static_cast<AccT>(row[v * elemStride]), invTemp);
    if (weights[v] > rowMax) rowMax = weights[v];
  }
  for (LongType v = 0; v < vocabSize; v++) weights[v] = tokenSampleWeight<AccT>(weights[v], rowMax);

  const AccT zero = static_cast<AccT>(0);
  AccT threshold = zero;
  if (topK > 0 && topK < vocabSize) {
    std::vector<AccT> positive;
    for (LongType v = 0; v < vocabSize; v++) {
      if (weights[v] > zero) positive.push_back(weights[v]);
    }
    if (static_cast<LongType>(positive.size()) >= topK) {
      std::nth_element(positive.begin(), positive.begin() + (topK - 1), positive.end(), std::greater<AccT>());
      threshold = positive[topK - 1];
    }
  }

  if (topP > zero && topP < static_cast<AccT>(1)) {
    // Walking the kept weights from the largest, the first prefix reaching topP of the kept mass
    // ends at the largest threshold that still holds it: every larger threshold keeps a shorter
    // prefix. The mass sums in the walk's order, so the full prefix equals it.
    std::vector<AccT> kept;
    for (LongType v = 0; v < vocabSize; v++) {
      if (weights[v] > zero && weights[v] >= threshold) kept.push_back(weights[v]);
    }
    std::sort(kept.begin(), kept.end(), std::greater<AccT>());
    AccT mass = zero;
    for (const AccT weight : kept) mass = reproducible::add<AccT>(mass, weight);
    const AccT need = reproducible::multiply<AccT>(topP, mass);
    AccT prefix = zero;
    for (const AccT weight : kept) {
      prefix = reproducible::add<AccT>(prefix, weight);
      if (prefix >= need) {
        threshold = weight;
        break;
      }
    }
  }

  AccT total = zero;
  LongType last = -1;
  for (LongType v = 0; v < vocabSize; v++) {
    if (weights[v] > zero && weights[v] >= threshold) {
      total = reproducible::add<AccT>(total, weights[v]);
      last = v;
    }
  }
  if (last < 0) {
    token = 0;
    probability = zero;
    return;
  }
  const AccT target = reproducible::multiply<AccT>(u, total);
  AccT cumulative = zero;
  token = last;
  for (LongType v = 0; v < last; v++) {
    if (weights[v] > zero && weights[v] >= threshold) {
      cumulative = reproducible::add<AccT>(cumulative, weights[v]);
      if (cumulative > target) {
        token = v;
        break;
      }
    }
  }
  probability = reproducible::divide<AccT>(weights[token], total);
}

// Rows are independent: each draws at its own u_b and writes its own token and probability.
// Per-row operands of other data types (FP8 included) convert through the central assign.
template <typename T>
static void tokenSampleDraw_(NDArray* logits, NDArray* output, NDArray* probabilities, NDArray* uniforms,
                             graph::RandomGenerator rng, double temperature, int topK, double topP,
                             LaunchContext* context) {
  using AccT = typename simdOps::AggregateType<T>::type;
  const DataType accType = DataTypeUtils::fromT<AccT>();
  const TokenSampleRows rows = tokenSampleRows(logits);
  const bool greedy = tokenSampleIsGreedy(temperature, topK, topP);
  const AccT invTemp = static_cast<AccT>(temperature > 0.0 ? 1.0 / temperature : 1.0);
  const AccT keepMass = static_cast<AccT>(topP);
  std::vector<LongType> batchShape = {rows.batch};

  std::vector<AccT> draws(rows.batch, static_cast<AccT>(0));
  if (!greedy) {
    if (uniforms != nullptr) {
      NDArray staged('c', batchShape, accType, context);
      staged.assign(uniforms);
      const AccT* values = staged.bufferAsT<AccT>();
      for (LongType b = 0; b < rows.batch; b++) draws[b] = values[b];
    } else {
      for (LongType b = 0; b < rows.batch; b++) draws[b] = rng.relativeT<AccT>(b);
    }
  }

  const T* x = logits->bufferAsT<T>();
  std::vector<LongType> tokens(rows.batch, 0);
  NDArray chosen('c', batchShape, accType, context);
  AccT* chosenProbability = chosen.bufferAsT<AccT>();
  PRAGMA_OMP_PARALLEL_FOR
  for (LongType b = 0; b < rows.batch; b++) {
    tokenSampleDrawRow<T, AccT>(x + b * rows.rowStride + rows.rowOffset, rows.vocabSize, rows.elemStride, greedy,
                                invTemp, topK, keepMass, draws[b], tokens[b], chosenProbability[b]);
  }

  for (LongType b = 0; b < rows.batch; b++) output->p(b, tokens[b]);
  if (probabilities != nullptr) probabilities->assign(&chosen);
}

void tokenSampleDraw(NDArray* logits, NDArray* output, NDArray* probabilities, NDArray* uniforms,
                     graph::RandomGenerator rng, double temperature, int topK, double topP,
                     LaunchContext* context) {
  tokenSampleDrawCheck(logits, output, probabilities, uniforms);
  if (logits->lengthOf() == 0) return;
  BUILD_SINGLE_SELECTOR(logits->dataType(), tokenSampleDraw_,
                        (logits, output, probabilities, uniforms, rng, temperature, topK, topP, context),
                        SD_FLOAT_TYPES);
}

void tokenSample(NDArray* logits, NDArray* output,
                    double temperature, int topK, double topP,
                    LongType seed, LaunchContext* context) {
  // A greedy selection draws nothing, so it takes no entropy.
  const bool greedy = tokenSampleIsGreedy(temperature, topK, topP);
  tokenSampleDraw(logits, output, nullptr, nullptr,
                  greedy ? graph::RandomGenerator(1, 1) : tokenSampleGenerator(seed),
                  temperature, topK, topP, context);
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
                buf[base + static_cast<LongType>(stopId) * elemStride] = static_cast<T>(-DataTypeUtils::infOrMax<T>());
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
    const int strategy = tokenSampleScalarStrategy(config);
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
        const bool hasPenalties = inputIds != nullptr && tokenSamplePolicyReadsHistory(config);
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
