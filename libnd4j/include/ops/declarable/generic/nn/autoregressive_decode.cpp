/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_autoregressive_decode)

#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/autoregressive_decode.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/Context.h>

#include <limits>

namespace sd {
namespace ops {

/**
 * autoregressive_decode — full native decode loop.
 *
 * Inputs:
 *   0: prefillEmbeddings [1, seqLen, hidden]
 *   1: embeddingTable [vocabSize, hidden]
 *   2: inputIds [1, seqLen] INT64
 *   [optional, based on optionalMask bitmask:]
 *   3+: attentionMask [1, 1, 1, maxKvLen] (bit 0)
 *   3+: positionIds [1, 1] INT64 (bit 1)
 *   3+: staticKvBuffers [2*numKvPairs] (bit 2)
 *
 * Outputs:
 *   0: generatedTokenIds [maxNewTokens] INT64
 *   1: tokenCount [1] INT64
 *   2: timingInfo [10] FLOAT32 (negative [6] means explicit repetition finish)
 *
 * iArgs:
 *   0: maxNewTokens
 *   1: eosTokenId
 *   2: numKvPairs
 *   3: prefillSeqLen
 *   4: optionalInputMask (bitmask: bit0=mask, bit1=posIds, bit2=KV)
 *   5: planHandleLow (lower 32 bits of plan pointer)
 *   6: planHandleHigh (upper 32 bits of plan pointer)
 *   7: contextHandleLow (lower 32 bits of OpaqueContext pointer)
 *   8: contextHandleHigh (upper 32 bits of OpaqueContext pointer)
 *   9: numPlanExternalInputs
 *  10: numPlanOutputs
 *  11: embeddingsExtIdx
 *  12: maskExtIdx
 *  13: causalMaskExtIdx (-1 if unused)
 *  14: posIdsExtIdx
 *  15: inputIdsExtIdx
 *  16: logitsOutputIdx
 *  17..17+2*numKvPairs-1: kvInputExtIndices
 *  17+2*numKvPairs..17+4*numKvPairs-1: kvOutputIndices
 *  [if bit 5 set in optionalMask: GDN/conv recurrent state]
 *  next: numGdnStatePairs, numConvStatePairs
 *  next+2..next+2+numGdnStatePairs-1: gdnStateExtIndices
 *  ..next+2+2*numGdnStatePairs-1: gdnStateOutputIndices
 *  ..+numConvStatePairs: convStateExtIndices
 *  ..+numConvStatePairs: convStateOutputIndices
 *  after state indices: additional stop token IDs
 *
 * tArgs:
 *   0: temperature
 *   1: topP
 *   2: topK (as double)
 *   3: repetitionPenalty
 *   4: decodeStrategy
 *   5: batchMax
 *   6: windowMax
 *   7: activeBatch
 *   8: activeWindow
 *   9: hiddenOutputIdx
 *  10: numBeams
 *  11: lengthPenalty
 *  12: penaltyAlpha
 *  13: contrastiveTopK
 *  14: minP
 *  15: frequencyPenalty
 *  16: presencePenalty
 *  17: minNewTokens
 *  18: generatedTokenOffset
 *  19: seed low 32 bits
 *  20: seed high 32 bits
 *  21: typicalP (1.0 = off)
 *  22: xtcProbability (0.0 = off)
 *  23: xtcThreshold (default 0.1)
 *
 * Plan external inputs and outputs are passed as additional input arrays after
 * the KV buffers. The indices in iArgs tell the C++ code which entries in the
 * plan's external input array correspond to mutable decode-step inputs.
 */
CUSTOM_OP_IMPL(autoregressive_decode, 3, 3, false, 3, 5) {
  auto prefillEmbeddings = INPUT_VARIABLE(0);  // [1, seqLen, hidden]
  auto embeddingTable = INPUT_VARIABLE(1);      // [vocabSize, hidden]
  auto inputIds = INPUT_VARIABLE(2);            // [1, seqLen] INT64

  auto generatedTokenIds = OUTPUT_VARIABLE(0);  // [maxNewTokens] INT64
  auto tokenCount = OUTPUT_VARIABLE(1);         // [1] INT64
  auto timingInfo = OUTPUT_VARIABLE(2);         // [10] FLOAT

  // Integer args: fixed layout
  int maxNewTokens = INT_ARG(0);
  int eosTokenId = INT_ARG(1);
  int numKvPairs = INT_ARG(2);
  int prefillSeqLen = INT_ARG(3);
  int optionalMask = INT_ARG(4);

  // Parse optional inputs based on the mask
  int nextInput = 3;
  NDArray* attentionMask = nullptr;
  NDArray* positionIds = nullptr;
  if (optionalMask & 1) {
    attentionMask = INPUT_VARIABLE(nextInput++);
  }
  if (optionalMask & 2) {
    positionIds = INPUT_VARIABLE(nextInput++);
  }

  // Collect static KV buffers (may be INT8 in V2 quantized mode)
  std::vector<NDArray*> staticKvBuffers;
  if (optionalMask & 4) {
    for (int i = 0; i < 2 * numKvPairs; i++) {
      staticKvBuffers.push_back(INPUT_VARIABLE(nextInput + i));
    }
    nextInput += 2 * numKvPairs;
  }

  // ADR 0107 V2: collect KV scale buffers (float32, [batch, maxKvLen, kvHeads]).
  // Bit 7 (128) in optionalMask signals that 2*numKvPairs scale arrays follow the KV buffers.
  // Layout: scaleBuffers[0..numKvPairs-1] = key scales per layer
  //         scaleBuffers[numKvPairs..2*numKvPairs-1] = value scales per layer
  std::vector<NDArray*> kvScaleBuffersVec;
  bool hasQuantisedKvScales = (optionalMask & 128) != 0;
  if (hasQuantisedKvScales) {
    for (int i = 0; i < 2 * numKvPairs; i++) {
      kvScaleBuffersVec.push_back(INPUT_VARIABLE(nextInput + i));
    }
    nextInput += 2 * numKvPairs;
  }

  // Qwen3.5 bundled MTP inputs (bit 8 / 256). These seven stable-address arrays follow
  // target KV and optional quantised-scale inputs in a fixed order.
  bool hasMtpPlan = (optionalMask & 256) != 0;
  NDArray* mtpInputIds = nullptr;
  NDArray* mtpTargetHidden = nullptr;
  NDArray* mtpCausalMask = nullptr;
  NDArray* mtpPositionOffset = nullptr;
  NDArray* mtpCachePosition = nullptr;
  NDArray* mtpKeyCache = nullptr;
  NDArray* mtpValueCache = nullptr;
  if (hasMtpPlan) {
    REQUIRE_TRUE(block.width() >= nextInput + 7, 0,
                 "autoregressive_decode: MTP bit is set but only %d inputs remain (need 7)",
                 block.width() - nextInput);
    mtpInputIds = INPUT_VARIABLE(nextInput++);
    mtpTargetHidden = INPUT_VARIABLE(nextInput++);
    mtpCausalMask = INPUT_VARIABLE(nextInput++);
    mtpPositionOffset = INPUT_VARIABLE(nextInput++);
    mtpCachePosition = INPUT_VARIABLE(nextInput++);
    mtpKeyCache = INPUT_VARIABLE(nextInput++);
    mtpValueCache = INPUT_VARIABLE(nextInput++);
  }

  // Collect additional stop token IDs (always includes eosTokenId)
  std::vector<int> stopTokenIds;
  stopTokenIds.push_back(eosTokenId);

  // Float args: sampling config. Args 0..3 are the legacy scalar sampler contract;
  // args 4+ carry the ADR 0106 policy envelope without disturbing variable iArgs
  // (KV indices + stop token IDs).
  double temperature = T_ARG(0);
  double topP = T_ARG(1);
  int topK = static_cast<int>(T_ARG(2));
  double repPenalty = block.getTArguments()->size() > 3 ? T_ARG(3) : 1.0;

  helpers::TokenSampleConfig sampleConfig;
  sampleConfig.temperature = temperature;
  sampleConfig.topP = topP;
  sampleConfig.topK = topK;
  sampleConfig.repPenalty = repPenalty;
  sampleConfig.strategy = (temperature <= 0.0 || (topK <= 1 && topP <= 0.0))
                          ? helpers::TOKEN_SAMPLE_GREEDY : helpers::TOKEN_SAMPLE_SAMPLE;
  if (block.getTArguments()->size() > 4) sampleConfig.strategy = static_cast<int>(T_ARG(4));
  if (block.getTArguments()->size() > 5) sampleConfig.batchMax = static_cast<int>(T_ARG(5));
  if (block.getTArguments()->size() > 6) sampleConfig.windowMax = static_cast<int>(T_ARG(6));
  if (block.getTArguments()->size() > 7) sampleConfig.activeBatch = static_cast<int>(T_ARG(7));
  if (block.getTArguments()->size() > 8) sampleConfig.activeWindow = static_cast<int>(T_ARG(8));
  if (block.getTArguments()->size() > 9) sampleConfig.hiddenOutputIdx = static_cast<int>(T_ARG(9));
  if (block.getTArguments()->size() > 10) sampleConfig.numBeams = static_cast<int>(T_ARG(10));
  if (block.getTArguments()->size() > 11) sampleConfig.lengthPenalty = T_ARG(11);
  if (block.getTArguments()->size() > 12) sampleConfig.penaltyAlpha = T_ARG(12);
  if (block.getTArguments()->size() > 13) sampleConfig.contrastiveTopK = static_cast<int>(T_ARG(13));
  if (block.getTArguments()->size() > 14) sampleConfig.minP = T_ARG(14);
  if (block.getTArguments()->size() > 15) sampleConfig.freqPenalty = T_ARG(15);
  if (block.getTArguments()->size() > 16) sampleConfig.presPenalty = T_ARG(16);
  if (block.getTArguments()->size() > 17) sampleConfig.minNewTokens = static_cast<int>(T_ARG(17));
  if (block.getTArguments()->size() > 18) sampleConfig.generatedTokenOffset = static_cast<int>(T_ARG(18));
  if (block.getTArguments()->size() > 20) {
    uint64_t seedLow = static_cast<uint64_t>(static_cast<uint32_t>(T_ARG(19)));
    uint64_t seedHigh = static_cast<uint64_t>(static_cast<uint32_t>(T_ARG(20)));
    sampleConfig.seed = static_cast<LongType>((seedHigh << 32) | seedLow);
  }
  // tArgs 21-23 (appended, never renumber above):
  //   21: typicalP    (1.0 = off)
  //   22: xtcProbability (0.0 = off)
  //   23: xtcThreshold   (default 0.1)
  if (block.getTArguments()->size() > 21) sampleConfig.typicalP = T_ARG(21);
  if (block.getTArguments()->size() > 22) sampleConfig.xtcProbability = T_ARG(22);
  if (block.getTArguments()->size() > 23) sampleConfig.xtcThreshold = T_ARG(23);

  // ADR 0106 Phase 2: optional speculative decode parameters (tArgs 24 and 25).
  // Read early so they can be propagated into decodeConfig after the plan config block.
  int speculativeK_arg = 0;
  int speculatorType_arg = 0;
  if (block.getTArguments()->size() > 24) speculativeK_arg = static_cast<int>(T_ARG(24));
  if (block.getTArguments()->size() > 25) speculatorType_arg = static_cast<int>(T_ARG(25));
  int actualSequenceLengthExtIdx_arg = -1;
  if (block.getTArguments()->size() > 26) {
    actualSequenceLengthExtIdx_arg = static_cast<int>(T_ARG(26));
  }
  int nativeRepetitionLoopMaxPeriod = 0;
  int nativeRepetitionLoopMaxRepeats = 0;
  if (block.getTArguments()->size() > 43) {
    nativeRepetitionLoopMaxPeriod = static_cast<int>(T_ARG(43));
  }
  if (block.getTArguments()->size() > 44) {
    nativeRepetitionLoopMaxRepeats = static_cast<int>(T_ARG(44));
  }
  REQUIRE_TRUE(nativeRepetitionLoopMaxPeriod >= 0 && nativeRepetitionLoopMaxPeriod <= 1024, 0,
               "autoregressive_decode: native repetition max period must be in [0,1024], got %d",
               nativeRepetitionLoopMaxPeriod);
  REQUIRE_TRUE(nativeRepetitionLoopMaxRepeats >= 0 && nativeRepetitionLoopMaxRepeats <= 1024, 0,
               "autoregressive_decode: native repetition max repeats must be in [0,1024], got %d",
               nativeRepetitionLoopMaxRepeats);
  REQUIRE_TRUE((nativeRepetitionLoopMaxPeriod == 0) == (nativeRepetitionLoopMaxRepeats == 0)
                   && (nativeRepetitionLoopMaxRepeats == 0 || nativeRepetitionLoopMaxRepeats >= 2),
               0, "autoregressive_decode: native repetition termination requires 0/0 or period>=1,repeats>=2");
  REQUIRE_TRUE(!(nativeRepetitionLoopMaxPeriod > 0 && speculativeK_arg > 0), 0,
               "autoregressive_decode: native repetition termination is not supported with speculative decode");

  // Validate inputs
  REQUIRE_TRUE(prefillEmbeddings->rankOf() == 3, 0,
               "autoregressive_decode: prefillEmbeddings must be rank 3 [batch, seqLen, hidden], got %d",
               prefillEmbeddings->rankOf());
  REQUIRE_TRUE(embeddingTable->rankOf() == 2, 0,
               "autoregressive_decode: embeddingTable must be rank 2 [vocabSize, hidden], got %d",
               embeddingTable->rankOf());
  REQUIRE_TRUE(inputIds->rankOf() == 2, 0,
               "autoregressive_decode: inputIds must be rank 2 [1, seqLen], got %d",
               inputIds->rankOf());
  REQUIRE_TRUE(maxNewTokens > 0, 0,
               "autoregressive_decode: maxNewTokens must be > 0, got %d", maxNewTokens);
  REQUIRE_TRUE(numKvPairs >= 0, 0,
               "autoregressive_decode: numKvPairs must be >= 0, got %d", numKvPairs);

  auto hidden = prefillEmbeddings->sizeAt(2);
  auto tableHidden = embeddingTable->sizeAt(1);
  REQUIRE_TRUE(hidden == tableHidden, 0,
               "autoregressive_decode: embedding hidden size mismatch: prefill=%lld, table=%lld",
               hidden, tableHidden);

  // ── Build AutoregressiveDecodeConfig from iArgs ──
  helpers::AutoregressiveDecodeConfig decodeConfig;
  decodeConfig.sampleConfig = sampleConfig;
  decodeConfig.planHandle = nullptr;
  decodeConfig.extInputContext = nullptr;
  decodeConfig.planExternalInputs = nullptr;
  decodeConfig.numPlanExternalInputs = 0;
  decodeConfig.planOutputs = nullptr;
  decodeConfig.numPlanOutputs = 0;
  decodeConfig.embeddingsExtIdx = -1;
  decodeConfig.maskExtIdx = -1;
  decodeConfig.causalMaskExtIdx = -1;
  decodeConfig.posIdsExtIdx = -1;
  decodeConfig.inputIdsExtIdx = -1;
  decodeConfig.logitsOutputIdx = -1;
  decodeConfig.attnMaskReformatExtIdx = -1;
  decodeConfig.kvInputExtIndices = nullptr;
  decodeConfig.kvOutputIndices = nullptr;
  // ADR 0106 Phase 1: window substrate — propagate from tArgs into config.
  // windowGridMask and windowPositionGrid are owned by the Java caller and passed
  // via extInputContext / additional inputs; the C++ helpers update them in-place.
  // They remain nullptr when activeWindow == 1 (W=1 path unchanged).
  decodeConfig.windowMax = sampleConfig.windowMax;
  decodeConfig.activeWindow = sampleConfig.activeWindow;
  // windowGridMask / windowPositionGrid pointers are wired by the helper after
  // extracting them from the ext input context when activeWindow > 1.
  decodeConfig.windowGridMask = nullptr;
  decodeConfig.windowPositionGrid = nullptr;
  // ADR 0106 Phase 2: n-gram speculative decoding parameters.
  // speculativeK=0 (default) leaves Phase 1 path completely unchanged.
  decodeConfig.speculativeK = speculativeK_arg;
  decodeConfig.speculatorType = speculatorType_arg;
  decodeConfig.actualSequenceLengthExtIdx = actualSequenceLengthExtIdx_arg;
  decodeConfig.nativeRepetitionLoopMaxPeriod = nativeRepetitionLoopMaxPeriod;
  decodeConfig.nativeRepetitionLoopMaxRepeats = nativeRepetitionLoopMaxRepeats;

  if (hasMtpPlan) {
    REQUIRE_TRUE(speculatorType_arg == 2, 0,
                 "autoregressive_decode: MTP inputs require speculatorType=2, got %d",
                 speculatorType_arg);
    REQUIRE_TRUE(block.getTArguments()->size() > 42, 0,
                 "autoregressive_decode: MTP metadata requires tArgs[27..42], got %d tArgs",
                 static_cast<int>(block.getTArguments()->size()));

    uint64_t mtpPlanAddr =
        (static_cast<uint64_t>(static_cast<uint32_t>(T_ARG(28))) << 32)
        | static_cast<uint64_t>(static_cast<uint32_t>(T_ARG(27)));
    uint64_t mtpContextAddr =
        (static_cast<uint64_t>(static_cast<uint32_t>(T_ARG(30))) << 32)
        | static_cast<uint64_t>(static_cast<uint32_t>(T_ARG(29)));

    decodeConfig.mtpPlanHandle =
        reinterpret_cast<graph::NativeDynamicShapePlan*>(mtpPlanAddr);
    decodeConfig.mtpExtInputContext = reinterpret_cast<void*>(mtpContextAddr);
    decodeConfig.mtpNumPlanExternalInputs = static_cast<int>(T_ARG(31));
    decodeConfig.mtpNumPlanOutputs = static_cast<int>(T_ARG(32));
    decodeConfig.mtpInputIdsExtIdx = static_cast<int>(T_ARG(33));
    decodeConfig.mtpTargetHiddenExtIdx = static_cast<int>(T_ARG(34));
    decodeConfig.mtpCausalMaskExtIdx = static_cast<int>(T_ARG(35));
    decodeConfig.mtpPositionOffsetExtIdx = static_cast<int>(T_ARG(36));
    decodeConfig.mtpCachePositionExtIdx = static_cast<int>(T_ARG(37));
    decodeConfig.mtpKvInputExtIndices[0] = static_cast<int>(T_ARG(38));
    decodeConfig.mtpKvInputExtIndices[1] = static_cast<int>(T_ARG(39));
    decodeConfig.mtpLogitsOutputIdx = static_cast<int>(T_ARG(40));
    decodeConfig.mtpHiddenOutputIdx = static_cast<int>(T_ARG(41));
    decodeConfig.targetHiddenOutputIdx = static_cast<int>(T_ARG(42));

    decodeConfig.mtpInputIds = mtpInputIds;
    decodeConfig.mtpTargetHidden = mtpTargetHidden;
    decodeConfig.mtpCausalMask = mtpCausalMask;
    decodeConfig.mtpPositionOffset = mtpPositionOffset;
    decodeConfig.mtpCachePosition = mtpCachePosition;
    decodeConfig.mtpKvBuffers[0] = mtpKeyCache;
    decodeConfig.mtpKvBuffers[1] = mtpValueCache;

    REQUIRE_TRUE(decodeConfig.mtpPlanHandle != nullptr
                     && decodeConfig.mtpExtInputContext != nullptr,
                 0, "autoregressive_decode: MTP plan/context handles must be non-null");
    REQUIRE_TRUE(decodeConfig.mtpNumPlanExternalInputs > 0
                     && decodeConfig.mtpNumPlanOutputs > 0,
                 0, "autoregressive_decode: invalid MTP plan dimensions: inputs=%d outputs=%d",
                 decodeConfig.mtpNumPlanExternalInputs, decodeConfig.mtpNumPlanOutputs);
    REQUIRE_TRUE(mtpInputIds->rankOf() == 2 && mtpInputIds->lengthOf() == 1, 0,
                 "autoregressive_decode: mtpInputIds must be scalar-shaped [1,1], got rank=%d length=%lld",
                 mtpInputIds->rankOf(), mtpInputIds->lengthOf());
    REQUIRE_TRUE(mtpTargetHidden->rankOf() == 3 && mtpTargetHidden->sizeAt(0) == 1
                     && mtpTargetHidden->sizeAt(1) == 1,
                 0, "autoregressive_decode: mtpTargetHidden must be [1,1,hidden]");
    REQUIRE_TRUE(mtpCausalMask->rankOf() == 4, 0,
                 "autoregressive_decode: mtpCausalMask must be rank 4, got %d",
                 mtpCausalMask->rankOf());
    REQUIRE_TRUE(mtpPositionOffset->lengthOf() == 1 && mtpCachePosition->lengthOf() == 1,
                 0, "autoregressive_decode: MTP position/cache inputs must be scalar arrays");
    REQUIRE_TRUE(mtpKeyCache != nullptr && mtpValueCache != nullptr, 0,
                 "autoregressive_decode: MTP key/value cache arrays must be non-null");
    REQUIRE_TRUE(decodeConfig.mtpInputIdsExtIdx >= 0
                     && decodeConfig.mtpInputIdsExtIdx < decodeConfig.mtpNumPlanExternalInputs
                     && decodeConfig.mtpTargetHiddenExtIdx >= 0
                     && decodeConfig.mtpTargetHiddenExtIdx < decodeConfig.mtpNumPlanExternalInputs
                     && decodeConfig.mtpCausalMaskExtIdx >= 0
                     && decodeConfig.mtpCausalMaskExtIdx < decodeConfig.mtpNumPlanExternalInputs
                     && decodeConfig.mtpPositionOffsetExtIdx >= 0
                     && decodeConfig.mtpPositionOffsetExtIdx < decodeConfig.mtpNumPlanExternalInputs
                     && decodeConfig.mtpCachePositionExtIdx >= 0
                     && decodeConfig.mtpCachePositionExtIdx < decodeConfig.mtpNumPlanExternalInputs
                     && decodeConfig.mtpKvInputExtIndices[0] >= 0
                     && decodeConfig.mtpKvInputExtIndices[0] < decodeConfig.mtpNumPlanExternalInputs
                     && decodeConfig.mtpKvInputExtIndices[1] >= 0
                     && decodeConfig.mtpKvInputExtIndices[1] < decodeConfig.mtpNumPlanExternalInputs
                     && decodeConfig.mtpLogitsOutputIdx >= 0
                     && decodeConfig.mtpLogitsOutputIdx < decodeConfig.mtpNumPlanOutputs
                     && decodeConfig.mtpHiddenOutputIdx >= 0
                     && decodeConfig.mtpHiddenOutputIdx < decodeConfig.mtpNumPlanOutputs
                     && decodeConfig.targetHiddenOutputIdx >= 0,
                 0, "autoregressive_decode: unresolved or out-of-range MTP plan index");
  }

  int iArgCount = block.getIArguments()->size();
  bool hasStopSequenceTrailer = (optionalMask & 512) != 0;
  int declaredTrailerStart = -1;
  if (hasStopSequenceTrailer) {
    REQUIRE_TRUE(iArgCount > 5, 0, "autoregressive_decode: missing stop trailer length");
    LongType trailerLengthArg = INT_ARG(iArgCount - 1);
    REQUIRE_TRUE(trailerLengthArg >= 6 && trailerLengthArg <= iArgCount - 5, 0,
                 "autoregressive_decode: invalid stop trailer length %lld",
                 static_cast<long long>(trailerLengthArg));
    declaredTrailerStart = iArgCount - static_cast<int>(trailerLengthArg);
    REQUIRE_TRUE(INT_ARG(declaredTrailerStart) == -1398034256LL, 0,
                 "autoregressive_decode: stop trailer magic mismatch");
  }
  bool hasPlanConfig = hasStopSequenceTrailer
      ? (declaredTrailerStart >= 0 && declaredTrailerStart + 3 < iArgCount
          && INT_ARG(declaredTrailerStart + 2) != 0)
      : (iArgCount > 8);

  std::vector<int> kvInputExtIndicesVec;
  std::vector<int> kvOutputIndicesVec;
  std::vector<int> gdnStateExtIndicesVec;
  std::vector<int> gdnStateOutputIndicesVec;
  std::vector<int> convStateExtIndicesVec;
  std::vector<int> convStateOutputIndicesVec;

  int stopTokenStartIdx = 5;  // default: additional stop IDs start at iArg[5]

  if (hasPlanConfig) {
    // Reconstruct plan pointer from two 32-bit halves (iArgs 5-6)
    LongType planLow = INT_ARG(5);
    LongType planHigh = INT_ARG(6);
    uint64_t planAddr = (static_cast<uint64_t>(static_cast<uint32_t>(planHigh)) << 32)
                      | static_cast<uint64_t>(static_cast<uint32_t>(planLow));
    decodeConfig.planHandle = reinterpret_cast<graph::NativeDynamicShapePlan*>(planAddr);

    // Reconstruct OpaqueContext pointer from two 32-bit halves (iArgs 7-8)
    LongType ctxLow = INT_ARG(7);
    LongType ctxHigh = INT_ARG(8);
    uint64_t ctxAddr = (static_cast<uint64_t>(static_cast<uint32_t>(ctxHigh)) << 32)
                     | static_cast<uint64_t>(static_cast<uint32_t>(ctxLow));
    decodeConfig.extInputContext = reinterpret_cast<void*>(ctxAddr);

    if (iArgCount > 10) {
      decodeConfig.numPlanExternalInputs = INT_ARG(9);
      decodeConfig.numPlanOutputs = INT_ARG(10);
    }
    if (iArgCount > 16) {
      decodeConfig.embeddingsExtIdx = INT_ARG(11);
      decodeConfig.maskExtIdx = INT_ARG(12);
      decodeConfig.causalMaskExtIdx = INT_ARG(13);
      decodeConfig.posIdsExtIdx = INT_ARG(14);
      decodeConfig.inputIdsExtIdx = INT_ARG(15);
      decodeConfig.logitsOutputIdx = INT_ARG(16);
    }

    // optionalMask bit 3 indicates attnMaskReformatExtIdx is present.
    // When present, iArg 17 is attnMaskReformatExtIdx and KV indices start at 18.
    // Otherwise KV indices start at 17 for backward compatibility.
    bool hasAttnMaskReformat = (optionalMask & 8) != 0;
    int kvStart = 17;
    if (hasAttnMaskReformat && iArgCount > 17) {
      decodeConfig.attnMaskReformatExtIdx = INT_ARG(17);
      kvStart = 18;
    }

    // optionalMask bit 4 indicates in-graph KV cache mode (GGUF pattern).
    // Two additional iArgs at kvStart: positionOffsetExtIdx, cachePositionExtIdx.
    // planOwnsKvScatter is set true since the attention op writes KV in-place.
    bool hasInGraphKvCache = (optionalMask & 16) != 0;
    if (hasInGraphKvCache && iArgCount > kvStart + 1) {
      decodeConfig.positionOffsetExtIdx = INT_ARG(kvStart);
      decodeConfig.cachePositionExtIdx = INT_ARG(kvStart + 1);
      decodeConfig.planOwnsKvScatter = true;
      kvStart += 2;
    }

    // KV indices: kvStart..kvStart+2*numKvPairs-1 are kvInputExtIndices
    //             For non-GGUF: kvStart+2*numKvPairs..kvStart+4*numKvPairs-1 are kvOutputIndices
    //             For in-graph KV (GGUF): no kvOutputIndices sent (attention writes KV in-place)
    if (numKvPairs > 0 && iArgCount > kvStart) {
      for (int i = 0; i < 2 * numKvPairs && (kvStart + i) < iArgCount; i++) {
        kvInputExtIndicesVec.push_back(INT_ARG(kvStart + i));
      }
      decodeConfig.kvInputExtIndices = kvInputExtIndicesVec.data();

      if (!hasInGraphKvCache) {
        // Non-GGUF path: kvOutputIndices follow kvInputExtIndices
        int kvOutStart = kvStart + 2 * numKvPairs;
        for (int i = 0; i < 2 * numKvPairs && (kvOutStart + i) < iArgCount; i++) {
          kvOutputIndicesVec.push_back(INT_ARG(kvOutStart + i));
        }
        decodeConfig.kvOutputIndices = kvOutputIndicesVec.data();
      }
    }

    // In-graph KV: only kvInput indices present (2*numKvPairs)
    // Non-GGUF: both kvInput and kvOutput present (4*numKvPairs)
    int nextIdx = kvStart + 2 * numKvPairs + (hasInGraphKvCache ? 0 : 2 * numKvPairs);

    // GDN/conv state indices (optionalMask bit 5).
    // Layout: numGdnStatePairs, numConvStatePairs,
    //         gdnStateExtIndices[numGdnStatePairs],
    //         gdnStateOutputIndices[numGdnStatePairs],
    //         convStateExtIndices[numConvStatePairs],
    //         convStateOutputIndices[numConvStatePairs]
    bool hasRecurrentState = (optionalMask & 32) != 0;
    if (hasRecurrentState && nextIdx + 1 < iArgCount) {
      decodeConfig.numGdnStatePairs = INT_ARG(nextIdx);
      decodeConfig.numConvStatePairs = INT_ARG(nextIdx + 1);
      nextIdx += 2;

      int numGdn = decodeConfig.numGdnStatePairs;
      int numConv = decodeConfig.numConvStatePairs;

      if (numGdn > 0 && nextIdx + 2 * numGdn <= iArgCount) {
        gdnStateExtIndicesVec.resize(numGdn);
        gdnStateOutputIndicesVec.resize(numGdn);
        for (int i = 0; i < numGdn; i++) {
          gdnStateExtIndicesVec[i] = INT_ARG(nextIdx + i);
        }
        nextIdx += numGdn;
        for (int i = 0; i < numGdn; i++) {
          gdnStateOutputIndicesVec[i] = INT_ARG(nextIdx + i);
        }
        nextIdx += numGdn;
        decodeConfig.gdnStateExtIndices = gdnStateExtIndicesVec.data();
        decodeConfig.gdnStateOutputIndices = gdnStateOutputIndicesVec.data();
      }

      if (numConv > 0 && nextIdx + 2 * numConv <= iArgCount) {
        convStateExtIndicesVec.resize(numConv);
        convStateOutputIndicesVec.resize(numConv);
        for (int i = 0; i < numConv; i++) {
          convStateExtIndicesVec[i] = INT_ARG(nextIdx + i);
        }
        nextIdx += numConv;
        for (int i = 0; i < numConv; i++) {
          convStateOutputIndicesVec[i] = INT_ARG(nextIdx + i);
        }
        nextIdx += numConv;
        decodeConfig.convStateExtIndices = convStateExtIndicesVec.data();
        decodeConfig.convStateOutputIndices = convStateOutputIndicesVec.data();
      }
    }

    // ADR 0106 Phase 1: window substrate ext input indices (optionalMask bit 6).
    // Layout: windowGridMaskExtIdx (INT), windowPosGridExtIdx (INT)
    // These point into the plan's external input array at the pre-allocated
    // [1,1,W_max,past+W_max] mask and [1,W_max] position grid respectively.
    // When optionalMask bit 6 is NOT set, windowGridMaskExtIdx / windowPosGridExtIdx
    // are -1 and the W=1 path runs unchanged.
    bool hasWindowSubstrate = (optionalMask & 64) != 0;
    if (hasWindowSubstrate && nextIdx + 1 < iArgCount) {
      int windowGridMaskExtIdx = INT_ARG(nextIdx);
      int windowPosGridExtIdx = INT_ARG(nextIdx + 1);
      nextIdx += 2;

      // Wire the pre-allocated window tensors from the ext input context.
      // The Java caller pre-allocates these at [1,1,W_max,past+W_max] and [1,W_max]
      // and registers them in the ext input context before calling the op.
      auto* extCtx = reinterpret_cast<graph::Context*>(decodeConfig.extInputContext);
      int numExtInputs = decodeConfig.numPlanExternalInputs;
      if (extCtx != nullptr) {
        if (windowGridMaskExtIdx >= 0 && windowGridMaskExtIdx < numExtInputs) {
          decodeConfig.windowGridMask = extCtx->array(windowGridMaskExtIdx);
        }
        if (windowPosGridExtIdx >= 0 && windowPosGridExtIdx < numExtInputs) {
          decodeConfig.windowPositionGrid = extCtx->array(windowPosGridExtIdx);
        }
      }
    }

    // GGUF in-graph KV decode reuses the model's fixed W-wide causal-mask input as
    // the speculative window grid. Position IDs are derived internally from the
    // scalar position_offset, so no separate position grid is required.
    if (decodeConfig.planOwnsKvScatter && decodeConfig.windowMax > 1
        && decodeConfig.windowGridMask == nullptr) {
      auto* extCtx = reinterpret_cast<graph::Context*>(decodeConfig.extInputContext);
      if (extCtx != nullptr && decodeConfig.causalMaskExtIdx >= 0
          && decodeConfig.causalMaskExtIdx < decodeConfig.numPlanExternalInputs) {
        decodeConfig.windowGridMask = extCtx->array(decodeConfig.causalMaskExtIdx);
      }
    }

    stopTokenStartIdx = nextIdx;

    // ADR 0107 V2: wire scale buffers into decodeConfig when bit 7 was set.
    // kvScaleBuffersVec[0..numKvPairs-1] = key scales, [numKvPairs..2N-1] = value scales.
    if (hasQuantisedKvScales && !kvScaleBuffersVec.empty()) {
      decodeConfig.kvScaleBuffers = kvScaleBuffersVec.data();
      decodeConfig.kvQuantFormat = 1;  // INT8_KV
    }

    // Plan external inputs are read from the extInputContext (OpaqueContext).
    // Plan outputs are allocated locally in the decode helper.
    decodeConfig.planExternalInputs = nullptr;
    decodeConfig.planOutputs = nullptr;
  }

  std::vector<std::vector<int>> stopTokenSequences;
  std::vector<int> stopTokenHistory;
  int scalarStopEnd = iArgCount;
  if (hasStopSequenceTrailer) {
    constexpr LongType kStopMagic = -1398034256LL;
    int trailerStart = declaredTrailerStart;
    REQUIRE_TRUE(trailerStart >= 0 && trailerStart + 3 < iArgCount, 0,
                 "autoregressive_decode: malformed stop-sequence trailer");
    scalarStopEnd = trailerStart;
    REQUIRE_TRUE(INT_ARG(trailerStart + 1) == 1, 0,
                 "autoregressive_decode: unsupported stop-sequence trailer version %lld",
                 static_cast<long long>(INT_ARG(trailerStart + 1)));
    LongType planLayoutFlag = INT_ARG(trailerStart + 2);
    REQUIRE_TRUE(planLayoutFlag == 0 || planLayoutFlag == 1, 0,
                 "autoregressive_decode: invalid stop-sequence plan-layout flag");
    LongType countArg = INT_ARG(trailerStart + 3);
    REQUIRE_TRUE(countArg >= 0 && countArg <= std::numeric_limits<int>::max(), 0,
                 "autoregressive_decode: invalid stop sequence count");
    int count = static_cast<int>(countArg);
    int cursor = trailerStart + 4;
    for (int sequenceIndex = 0; sequenceIndex < count; sequenceIndex++) {
      REQUIRE_TRUE(cursor < iArgCount, 0,
                   "autoregressive_decode: truncated stop sequence length");
      LongType lengthArg = INT_ARG(cursor++);
      REQUIRE_TRUE(lengthArg > 0 && lengthArg <= std::numeric_limits<int>::max(), 0,
                   "autoregressive_decode: invalid stop sequence length");
      int length = static_cast<int>(lengthArg);
      REQUIRE_TRUE(cursor <= iArgCount - 1 && length <= (iArgCount - 1) - cursor, 0,
                   "autoregressive_decode: truncated stop sequence");
      std::vector<int> sequence;
      sequence.reserve(length);
      for (int j = 0; j < length; j++) {
        LongType token = INT_ARG(cursor++);
        REQUIRE_TRUE(token >= 0 && token <= std::numeric_limits<int>::max(), 0,
                     "autoregressive_decode: invalid stop sequence token %lld",
                     static_cast<long long>(token));
        sequence.push_back(static_cast<int>(token));
      }
      stopTokenSequences.push_back(std::move(sequence));
    }
    REQUIRE_TRUE(cursor < iArgCount, 0,
                 "autoregressive_decode: missing stop-sequence history length");
    LongType historyLengthArg = INT_ARG(cursor++);
    REQUIRE_TRUE(historyLengthArg >= 0 && historyLengthArg <= std::numeric_limits<int>::max(), 0,
                 "autoregressive_decode: invalid stop-sequence history length");
    int historyLength = static_cast<int>(historyLengthArg);
    REQUIRE_TRUE(cursor <= iArgCount - 1 && historyLength == (iArgCount - 1) - cursor, 0,
                 "autoregressive_decode: truncated or trailing stop history");
    for (int i = 0; i < historyLength; i++) {
      LongType token = INT_ARG(cursor++);
      REQUIRE_TRUE(token >= 0 && token <= std::numeric_limits<int>::max(), 0,
                   "autoregressive_decode: invalid stop history token %lld",
                   static_cast<long long>(token));
      stopTokenHistory.push_back(static_cast<int>(token));
    }
    cursor++;  // consume the validated final trailer-length word
    REQUIRE_TRUE(cursor == iArgCount, 0, "autoregressive_decode: malformed stop trailer");
  }

  // Collect legacy additional scalar stop token IDs before the optional trailer.
  for (int i = stopTokenStartIdx; i < scalarStopEnd; i++) {
    stopTokenIds.push_back(INT_ARG(i));
  }

  helpers::AutoregressiveDecodeConfig* configPtr = hasPlanConfig ? &decodeConfig : nullptr;

  helpers::autoregressiveDecode(
      prefillEmbeddings, embeddingTable, inputIds, attentionMask, positionIds,
      staticKvBuffers.empty() ? nullptr : staticKvBuffers.data(), numKvPairs,
      generatedTokenIds, tokenCount, timingInfo,
      maxNewTokens, prefillSeqLen, stopTokenIds, stopTokenSequences, stopTokenHistory,
      temperature, topK, topP, repPenalty,
      block.launchContext(), configPtr);

  return sd::Status::OK;
}

DECLARE_TYPES(autoregressive_decode) {
  getOpDescriptor()->addTraits(OP_TRAIT_DATA_DEPENDENT);
  getOpDescriptor()->addTraits(OP_TRAIT_FULLY_WRITING | (OP_TRAIT_DATA_DEPENDENT));
  // Allow INT8 for V2 quantized KV buffers (ADR 0107) and INT64 for token IDs.
  // ALL_FLOATS covers the embedding table, attention mask, and KV scale buffers.
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS, INT64, INT8});
  getOpDescriptor()->setAllowedOutputTypes(0, {INT64});
  getOpDescriptor()->setAllowedOutputTypes(1, {INT64});
  getOpDescriptor()->setAllowedOutputTypes(2, {FLOAT32});
}

DECLARE_SHAPE_FN(autoregressive_decode) {
  int maxNewTokens = INT_ARG(0);

  // Output 0: generatedTokenIds [maxNewTokens] INT64
  auto tokenIdsShape = ConstantShapeHelper::getInstance().vectorShapeInfo(
      static_cast<LongType>(maxNewTokens), INT64);

  // Output 1: tokenCount [1] INT64 scalar
  auto tokenCountShape = ConstantShapeHelper::getInstance().vectorShapeInfo(1, INT64);

  // Output 2: timingInfo [10] FLOAT32
  // [0]=totalMs [1]=avgDecodeMs [2]=tokPerSec [3]=p50Ms [4]=p99Ms
  // [5]=lateSteadyTokPerSec (steps 60+) [6]=lateSteadyAvgMs
  // [6]=lateSteadyAvgMs, or -1 when explicit native repetition termination fired
  // [7]=speculativeProposed [8]=speculativeAccepted [9]=speculativeSteps
  auto timingShape = ConstantShapeHelper::getInstance().vectorShapeInfo(10, FLOAT32);

  return SHAPELIST(tokenIdsShape, tokenCountShape, timingShape);
}

}  // namespace ops
}  // namespace sd

#endif
