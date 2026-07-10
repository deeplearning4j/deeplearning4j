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
 *   2: timingInfo [5] FLOAT32
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
  auto timingInfo = OUTPUT_VARIABLE(2);         // [5] FLOAT

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

  int iArgCount = block.getIArguments()->size();
  bool hasPlanConfig = (iArgCount > 8);  // need at least plan + context pointers

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

  // Collect additional stop token IDs (after the plan config block)
  for (int i = stopTokenStartIdx; i < iArgCount; i++) {
    stopTokenIds.push_back(INT_ARG(i));
  }

  helpers::AutoregressiveDecodeConfig* configPtr = hasPlanConfig ? &decodeConfig : nullptr;

  helpers::autoregressiveDecode(
      prefillEmbeddings, embeddingTable, inputIds, attentionMask, positionIds,
      staticKvBuffers.empty() ? nullptr : staticKvBuffers.data(), numKvPairs,
      generatedTokenIds, tokenCount, timingInfo,
      maxNewTokens, prefillSeqLen, stopTokenIds,
      temperature, topK, topP, repPenalty,
      block.launchContext(), configPtr);

  return sd::Status::OK;
}

DECLARE_TYPES(autoregressive_decode) {
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

  // Output 2: timingInfo [7] FLOAT32
  // [0]=totalMs [1]=avgDecodeMs [2]=tokPerSec [3]=p50Ms [4]=p99Ms
  // [5]=lateSteadyTokPerSec (steps 60+) [6]=lateSteadyAvgMs
  auto timingShape = ConstantShapeHelper::getInstance().vectorShapeInfo(7, FLOAT32);

  return SHAPELIST(tokenIdsShape, tokenCountShape, timingShape);
}

}  // namespace ops
}  // namespace sd

#endif
