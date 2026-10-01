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

#ifndef LIBND4J_HELPERS_TOKEN_SAMPLE_H
#define LIBND4J_HELPERS_TOKEN_SAMPLE_H

#include <system/op_boilerplate.h>
#include <array/DataTypeUtils.h>
#include <array/NDArray.h>
#include <graph/RandomGenerator.h>
#include <ops/declarable/helpers/reproducible_math.h>

#include <cstdint>
#include <random>
#include <string>

namespace sd {
namespace ops {
namespace helpers {

/** Strategy IDs mirrored by SamplingConfig.DecodeStrategy ordinal values. */
enum TokenSampleStrategy {
    TOKEN_SAMPLE_AUTO = 0,
    TOKEN_SAMPLE_GREEDY = 1,
    TOKEN_SAMPLE_SAMPLE = 2,
    TOKEN_SAMPLE_SPECULATIVE = 3,
    TOKEN_SAMPLE_CONTRASTIVE = 4,
    TOKEN_SAMPLE_BEAM = 5
};

/**
 * Unified token-selection policy config used by autoregressive_decode.
 *
 * The existing scalar row samplers remain the primitive; this struct is the shared policy envelope that
 * lets greedy, stochastic sampling, speculative, contrastive, and beam decoding enter one selector path.
 * Today only the scalar B=1,W=1 GREEDY/SAMPLE policies execute; wider policies are validated here and
 * rejected until the ADR 0106 masked multi-position forward lands.
 */
struct TokenSampleConfig {
    int strategy = TOKEN_SAMPLE_AUTO;
    int batchMax = 1;
    int windowMax = 1;
    int activeBatch = 1;
    int activeWindow = 1;

    double temperature = 0.0;
    int topK = 0;
    double topP = 0.0;
    double minP = 0.0;
    double repPenalty = 1.0;
    double freqPenalty = 0.0;
    double presPenalty = 0.0;

    int numBeams = 1;
    double lengthPenalty = 1.0;
    double penaltyAlpha = 0.0;
    int contrastiveTopK = 0;
    int hiddenOutputIdx = -1;

    // Stop-token floor: while generatedTokenOffset < minNewTokens, stop-token logits are masked.
    int minNewTokens = 0;
    int generatedTokenOffset = 0;
    const int* stopTokenIds = nullptr;
    int stopTokenCount = 0;
    LongType seed = 0;

    // Typical-p (entropy-based) filtering — 0 < typicalP < 1 enables; 1.0 = off.
    // Keeps tokens whose |−log p_i − H| deviation from entropy is smallest,
    // accumulating until their cumulative mass >= typicalP.
    // Applied after temperature scaling, before standard truncation samplers.
    double typicalP = 1.0;

    // XTC (Exclude Top Choices) sampling — with probability xtcProbability,
    // among tokens with softmax prob >= xtcThreshold, mask all EXCEPT the
    // lowest-probability surviving one. xtcProbability=0 = off.
    // Applied after typical-p and other truncation stages, before final sample.
    double xtcProbability = 0.0;
    double xtcThreshold = 0.1;
};

struct TokenSampleResult {
    LongType selectedToken = -1;
    int selectedBatch = 0;
    int selectedWindow = 0;
    int acceptedCount = 1;
    int parentBeam = 0;
    double score = 0.0;
};

/**
 * Samples one token per row of logits (see tokenSampleDraw), drawing from
 * tokenSampleGenerator(seed). logits is read only.
 */
SD_LIB_HIDDEN void tokenSample(NDArray* logits, NDArray* output,
                                double temperature, int topK, double topP,
                                LongType seed, LaunchContext* context);

/**
 * The generator of tokenSample's draws: a positive seed fixes them (seeded as the random ops seed
 * theirs), any other seed takes fresh entropy. The generator is counter based: draw b is a pure
 * function of the generator's state and b, so a seed yields the same u_b on every backend.
 */
SD_INLINE graph::RandomGenerator tokenSampleGenerator(LongType seed) {
  if (seed > 0) return graph::RandomGenerator(seed, seed ^ 0xdeadbeef);
  std::random_device entropy;
  const auto state = [&entropy]() {
    return static_cast<LongType>((static_cast<uint64_t>(entropy()) << 32) | static_cast<uint64_t>(entropy()));
  };
  const LongType root = state();
  return graph::RandomGenerator(root, state());
}

/**
 * Selects one token per row of logits and reports the probability of each selected token.
 *
 * Greedy (temperature <= 0, topK <= 0 and topP <= 0): the argmax of the row, lowest index on
 * ties, never NaN; its probability is 1, the selection being a draw from a point mass. A row with
 * no value above -inf selects token 0 with probability 0.
 *
 * Otherwise the row becomes softmax weights (tokenSampleWeight) of the logits scaled by
 * 1 / temperature (by 1 when temperature <= 0): the largest weight is 1. Truncation keeps the
 * tokens whose weight reaches a threshold, so equal weights are kept or dropped together:
 *  - top-k (0 < topK < vocab): the topK-th largest positive weight (0 when fewer are positive);
 *  - top-p (0 < topP < 1): the largest threshold, at least the top-k threshold, at which the
 *    kept weights still hold topP of the weight kept by top-k.
 * Row b is drawn at u_b by inverse CDF over the kept tokens of positive weight: the first, in
 * vocabulary order, whose cumulative weight exceeds u_b times the kept total (the last such token
 * when rounding leaves none). u_b is uniforms[b] when uniforms is given, else rng.relativeT(b).
 * A row without positive weight selects token 0 with probability 0. NaN weights are never kept.
 *
 * @param logits        floating [vocab], [batch, vocab] or [batch, seqLen, vocab] (the last
 *                      position is sampled); read only
 * @param output        INT64 token ids, batch elements (one for rank-1 logits)
 * @param probabilities floating, batch elements: the probability of each selected token under the
 *                      kept, renormalized distribution (may be nullptr)
 * @param uniforms      floating, batch values in [0, 1) (may be nullptr)
 * @param rng           generator of the draws when uniforms is nullptr
 */
SD_LIB_HIDDEN void tokenSampleDraw(NDArray* logits, NDArray* output, NDArray* probabilities,
                                    NDArray* uniforms, graph::RandomGenerator rng,
                                    double temperature, int topK, double topP,
                                    LaunchContext* context);

SD_INLINE bool tokenSampleIsGreedy(double temperature, int topK, double topP) {
  return temperature <= 0.0 && topK <= 0 && topP <= 0.0;
}

// The strategy tokenSamplePolicy runs for a config: AUTO selects GREEDY at temperature <= 0 or when
// nothing past the top token can be kept (topK <= 1 and topP <= 0), SAMPLE otherwise.
SD_INLINE int tokenSampleScalarStrategy(const TokenSampleConfig& config) {
  if (config.strategy != TOKEN_SAMPLE_AUTO) return config.strategy;
  return config.temperature <= 0.0 || (config.topK <= 1 && config.topP <= 0.0) ? TOKEN_SAMPLE_GREEDY
                                                                               : TOKEN_SAMPLE_SAMPLE;
}

// Whether tokenSamplePolicy reads its token history (inputIds): only the repetition, frequency and
// presence penalties of a SAMPLE do. A caller may pass no history when this is false.
SD_INLINE bool tokenSamplePolicyReadsHistory(const TokenSampleConfig& config) {
  return tokenSampleScalarStrategy(config) == TOKEN_SAMPLE_SAMPLE &&
         (config.repPenalty != 1.0 || config.freqPenalty != 0.0 || config.presPenalty != 0.0);
}

// A logit scaled by the inverse temperature. The product and the difference of tokenSampleWeight
// round on their own (never fused), so a weight recomputed in any pass or on any backend compares
// equal to itself.
template <typename AccT>
SD_HOST_DEVICE SD_INLINE AccT tokenSampleScaled(AccT logit, AccT invTemp) {
  return reproducible::multiply<AccT>(logit, invTemp);
}

// Softmax weight of a scaled logit against the row maximum of the scaled logits: 1 at the maximum
// (an infinite one included), exp(scaled - rowMax) below it, NaN for NaN.
template <typename AccT>
SD_HOST_DEVICE SD_INLINE AccT tokenSampleWeight(AccT scaled, AccT rowMax) {
  return scaled == rowMax ? static_cast<AccT>(1)
                          : math::sd_exp<AccT, AccT>(reproducible::subtract<AccT>(scaled, rowMax));
}

// Distance between consecutive elements of a per-row array: 0 for a single element, the stride of
// the one non-unit axis of a vector (of any rank), -1 when the elements span several axes.
SD_INLINE LongType tokenSampleRowStride(NDArray* perRow) {
  if (perRow->lengthOf() <= 1) return 0;
  LongType axis = -1;
  return perRow->isCommonVector(axis) ? perRow->stridesOf()[axis] : -1;
}

// Row geometry of a logits array [vocab], [batch, vocab] or [batch, seqLen, vocab]: row b starts
// at b * rowStride + rowOffset (the last position for rank 3) and steps elemStride per token.
struct TokenSampleRows {
  LongType batch = 1;
  LongType vocabSize = 0;
  LongType rowStride = 0;
  LongType elemStride = 1;
  LongType rowOffset = 0;
};

SD_INLINE TokenSampleRows tokenSampleRows(NDArray* logits) {
  TokenSampleRows rows;
  const int rank = logits->rankOf();
  const LongType* strides = logits->stridesOf();
  rows.vocabSize = logits->sizeAt(rank - 1);
  rows.elemStride = strides[rank - 1];
  if (rank > 1) {
    rows.batch = logits->sizeAt(0);
    rows.rowStride = strides[0];
  }
  if (rank == 3) rows.rowOffset = (logits->sizeAt(1) - 1) * strides[1];
  return rows;
}

// Checks the operands of tokenSampleDraw against its contract.
SD_INLINE void tokenSampleDrawCheck(NDArray* logits, NDArray* output, NDArray* probabilities, NDArray* uniforms) {
  const int rank = logits->rankOf();
  if (!DataTypeUtils::isR(logits->dataType()) || rank < 1 || rank > 3) {
    const std::string message = "tokenSampleDraw: logits must be floating of rank 1 to 3, got " +
                                DataTypeUtils::asString(logits->dataType()) + " of rank " + std::to_string(rank);
    THROW_EXCEPTION(message.c_str());
  }
  const LongType batch = rank == 1 ? 1 : logits->sizeAt(0);
  if (batch > 0 && (logits->sizeAt(rank - 1) == 0 || (rank == 3 && logits->sizeAt(1) == 0))) {
    THROW_EXCEPTION("tokenSampleDraw: logits hold no token to sample");
  }
  if (output->dataType() != DataTypeUtils::fromT<LongType>() || output->lengthOf() != batch) {
    const std::string message = "tokenSampleDraw: output must hold " + std::to_string(batch) +
                                " INT64 token ids, got " + std::to_string(output->lengthOf()) + " of " +
                                DataTypeUtils::asString(output->dataType());
    THROW_EXCEPTION(message.c_str());
  }
  for (NDArray* perRow : {probabilities, uniforms}) {
    if (perRow != nullptr && (!DataTypeUtils::isR(perRow->dataType()) || perRow->lengthOf() != batch)) {
      const std::string message = std::string("tokenSampleDraw: ") +
                                  (perRow == probabilities ? "probabilities" : "uniforms") + " must hold " +
                                  std::to_string(batch) + " floating values, got " +
                                  std::to_string(perRow->lengthOf()) + " of " +
                                  DataTypeUtils::asString(perRow->dataType());
      THROW_EXCEPTION(message.c_str());
    }
  }
}

/**
 * Extended token sampling with penalty and min-p support.
 *
 * Sampling pipeline order (documented here and enforced in token_sample.cpp / token_sample.cu):
 *   1. Penalties (repetition / frequency / presence)
 *   2. Temperature scaling
 *   3. Top-k truncation
 *   4. Softmax → min-p filter (adaptive probability threshold)
 *   5. Top-p (nucleus) filter
 *   6. Typical-p (entropy-deviation) filter
 *   7. XTC (exclude top choices) — stochastic, skipped when xtcProbability=0
 *   8. Multinomial sample (or greedy argmax when temperature<=0)
 *
 * Pipeline: penalties → temperature → top-k → min-p → top-p → typical-p → xtc → sample
 *
 * @param logits       [batch, vocabSize] — will be modified in-place by penalties
 * @param output       [batch] INT64 — sampled token IDs
 * @param inputIds     [batch, seqLen] INT64 — prior tokens for penalty computation (nullable)
 * @param temperature  Temperature for logit scaling (<=0 = greedy)
 * @param topK         Top-K filtering (<=0 = off)
 * @param topP         Top-P nucleus filtering (<=0 or >=1 = off)
 * @param minP         Min-P adaptive filtering (<=0 = off)
 * @param repPenalty   Repetition penalty (1.0 = off)
 * @param freqPenalty  Frequency penalty (0.0 = off)
 * @param presPenalty  Presence penalty (0.0 = off)
 * @param typicalP     Typical-p entropy-deviation filter (1.0 = off)
 * @param xtcProbability XTC probability (0.0 = off)
 * @param xtcThreshold   XTC per-token probability threshold (default 0.1)
 * @param seed         RNG seed (0 = random)
 */
SD_LIB_HIDDEN void tokenSampleWithPenalties(NDArray* logits, NDArray* output,
                                             NDArray* inputIds,
                                             double temperature, int topK,
                                             double topP, double minP,
                                             double repPenalty, double freqPenalty,
                                             double presPenalty,
                                             double typicalP,
                                             double xtcProbability, double xtcThreshold,
                                             LongType seed, LaunchContext* context);

/**
 * Central policy selector for autoregressive decode.
 *
 * Output is strategy-dependent but starts with scalar token IDs in {@code output}. The current
 * implementation executes scalar GREEDY/SAMPLE by delegating to {@link tokenSample} /
 * {@link tokenSampleWithPenalties}; non-scalar policies fail fast until autoregressive_decode exposes
 * the fixed B/W substrate and policy result buffers.
 */
SD_LIB_HIDDEN void tokenSamplePolicy(NDArray* logits, NDArray* output,
                                     NDArray* inputIds,
                                     const TokenSampleConfig& config,
                                     TokenSampleResult* result,
                                     LaunchContext* context);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
