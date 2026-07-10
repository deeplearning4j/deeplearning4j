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
#include <array/NDArray.h>

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

SD_LIB_HIDDEN void tokenSample(NDArray* logits, NDArray* output,
                                double temperature, int topK, double topP,
                                LongType seed, LaunchContext* context);

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
