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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.Getter;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.bytedeco.javacpp.Pointer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * Autoregressive decode loop as a single native op.
 *
 * Runs the full autoregressive decode loop in C++, eliminating per-step
 * Java to C++ round-trips. Handles: plan execute, KV scatter, token sample,
 * embedding lookup, input buffer advance, repeat.
 *
 * Inputs:
 *   0: prefillEmbeddings [1, seqLen, hidden]
 *   1: embeddingTable [vocabSize, hidden]
 *   2: inputIds [1, seqLen] INT64
 *   3: attentionMask [1, 1, 1, maxKvLen] (optional via mask)
 *   4: positionIds [1, 1] INT64 (optional via mask)
 *   5..5+2N-1: staticKvBuffers (N key + N value) (optional via mask)
 *   after KV: planExternalInputs[numPlanExternalInputs]
 *
 * Outputs:
 *   0: generatedTokenIds [maxNewTokens] INT64
 *   1: tokenCount [1] INT64
 *   2: timingInfo [10] FLOAT (timing metrics plus speculative proposed/accepted/step counts)
 *
 * iArgs layout (when plan config is present):
 *   0: maxNewTokens
 *   1: eosTokenId
 *   2: numKvPairs
 *   3: prefillSeqLen
 *   4: optionalInputMask
 *   5: planHandleLow (lower 32 bits)
 *   6: planHandleHigh (upper 32 bits)
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
 *  after kv indices: additional stop token IDs
 *
 * tArgs:
 *   0: temperature
 *   1: topP
 *   2: topK (as double)
 *   3: repetitionPenalty (1.0 = disabled)
 *   4: decodeStrategy (SamplingConfig.DecodeStrategy ordinal: AUTO/GREEDY/SAMPLE/SPECULATIVE/CONTRASTIVE/BEAM)
 *   5: batchMax (ADR 0106 B dimension)
 *   6: windowMax (ADR 0106 W dimension)
 *   7: activeBatch
 *   8: activeWindow
 *   9: hiddenOutputIdx (-1 when unused)
 *   10: numBeams
 *   11: lengthPenalty
 *   12: penaltyAlpha
 *   13: contrastiveTopK
 *   14: minP
 *   15: frequencyPenalty
 *   16: presencePenalty
 *   17: minNewTokens
 *   18: generatedTokenOffset
 *   19: seed low 32 bits
 *   20: seed high 32 bits
 *   21: typicalP (1.0 = off)
 *   22: xtcProbability (0.0 = off)
 *   23: xtcThreshold (default 0.1)
 *   24: speculativeK (0 = off; ADR 0106 Phase 2 n-gram speculation)
 *   25: speculatorType (0=none, 1=NGRAM; ADR 0106 Phase 2)
 *   26: actualSequenceLengthExtIdx (-1 when the target graph has no recurrent-length control)
 */
@NoArgsConstructor
public class AutoregressiveDecode extends DynamicCustomOp {

    public static final int DECODE_STRATEGY_AUTO = 0;
    public static final int DECODE_STRATEGY_GREEDY = 1;
    public static final int DECODE_STRATEGY_SAMPLE = 2;
    public static final int DECODE_STRATEGY_SPECULATIVE = 3;
    public static final int DECODE_STRATEGY_CONTRASTIVE = 4;
    public static final int DECODE_STRATEGY_BEAM = 5;
    public static final int SPECULATOR_TYPE_NONE = 0;
    public static final int SPECULATOR_TYPE_NGRAM = 1;
    public static final int SPECULATOR_TYPE_MTP = 2;

    @Getter private int maxNewTokens;
    @Getter private int eosTokenId;
    @Getter private int numKvPairs;
    @Getter private int prefillSeqLen;
    @Getter private double temperature;
    @Getter private double topP;
    @Getter private int topK;
    @Getter private double repetitionPenalty;
    @Getter private int optionalInputMask;
    @Getter private int decodeStrategy = DECODE_STRATEGY_AUTO;
    @Getter private int batchMax = 1;
    @Getter private int windowMax = 1;
    @Getter private int activeBatch = 1;
    @Getter private int activeWindow = 1;
    @Getter private int hiddenOutputIdx = -1;
    @Getter private int numBeams = 1;
    @Getter private double lengthPenalty = 1.0;
    @Getter private double penaltyAlpha = 0.0;
    @Getter private int contrastiveTopK = 0;
    @Getter private double minP = 0.0;
    @Getter private double frequencyPenalty = 0.0;
    @Getter private double presencePenalty = 0.0;
    @Getter private int minNewTokens = 0;
    @Getter private int generatedTokenOffset = 0;
    @Getter private long seed = 0L;
    @Getter private double typicalP = 1.0;
    @Getter private double xtcProbability = 0.0;
    @Getter private double xtcThreshold = 0.1;
    // ADR 0106 Phase 2: n-gram speculative decoding parameters (tArgs[24/25]).
    // speculativeK=0 (default) leaves the W=1 path completely unchanged.
    @Getter private int speculativeK = 0;     // 0=off; >0 enables n-gram speculation
    @Getter private int speculatorType = 0;   // 0=none, 1=NGRAM
    @Getter private int actualSequenceLengthExtIdx = -1;

    private static int resolveScalarDecodeStrategy(double temperature, int topK, double topP) {
        return (temperature <= 0.0 || (topK <= 1 && topP <= 0.0))
                ? DECODE_STRATEGY_GREEDY : DECODE_STRATEGY_SAMPLE;
    }

    private static double seedLowBits(long seed) {
        return (double) (seed & 0xFFFFFFFFL);
    }

    private static double seedHighBits(long seed) {
        return (double) ((seed >>> 32) & 0xFFFFFFFFL);
    }

    private static long combineSeed(double lowBits, double highBits) {
        long low = ((long) lowBits) & 0xFFFFFFFFL;
        long high = ((long) highBits) & 0xFFFFFFFFL;
        return (high << 32) | low;
    }

    private void addSamplingPolicyArguments(double temperature, double topP, int topK, double repetitionPenalty) {
        this.decodeStrategy = resolveScalarDecodeStrategy(temperature, topK, topP);
        addSamplingPolicyArguments(temperature, topP, topK, repetitionPenalty,
                this.decodeStrategy, 1, 1, 1, 1, -1, 1, 1.0, 0.0, 0,
                0.0, 0.0, 0.0, 0, 0, 0L,
                1.0, 0.0, 0.1);
    }

    private void addSamplingPolicyArguments(double temperature, double topP, int topK, double repetitionPenalty,
                                            int decodeStrategy, int batchMax, int windowMax,
                                            int activeBatch, int activeWindow, int hiddenOutputIdx,
                                            int numBeams, double lengthPenalty,
                                            double penaltyAlpha, int contrastiveTopK,
                                            double minP, double frequencyPenalty,
                                            double presencePenalty, int minNewTokens,
                                            int generatedTokenOffset, long seed) {
        addSamplingPolicyArguments(temperature, topP, topK, repetitionPenalty,
                decodeStrategy, batchMax, windowMax, activeBatch, activeWindow, hiddenOutputIdx,
                numBeams, lengthPenalty, penaltyAlpha, contrastiveTopK,
                minP, frequencyPenalty, presencePenalty, minNewTokens, generatedTokenOffset, seed,
                this.typicalP, this.xtcProbability, this.xtcThreshold);
    }

    private void addSamplingPolicyArguments(double temperature, double topP, int topK, double repetitionPenalty,
                                            int decodeStrategy, int batchMax, int windowMax,
                                            int activeBatch, int activeWindow, int hiddenOutputIdx,
                                            int numBeams, double lengthPenalty,
                                            double penaltyAlpha, int contrastiveTopK,
                                            double minP, double frequencyPenalty,
                                            double presencePenalty, int minNewTokens,
                                            int generatedTokenOffset, long seed,
                                            double typicalP, double xtcProbability,
                                            double xtcThreshold) {
        this.temperature = temperature;
        this.topP = topP;
        this.topK = topK;
        this.repetitionPenalty = repetitionPenalty;
        this.decodeStrategy = decodeStrategy;
        this.batchMax = batchMax;
        this.windowMax = windowMax;
        this.activeBatch = activeBatch;
        this.activeWindow = activeWindow;
        this.hiddenOutputIdx = hiddenOutputIdx;
        this.numBeams = numBeams;
        this.lengthPenalty = lengthPenalty;
        this.penaltyAlpha = penaltyAlpha;
        this.contrastiveTopK = contrastiveTopK;
        this.minP = minP;
        this.frequencyPenalty = frequencyPenalty;
        this.presencePenalty = presencePenalty;
        this.minNewTokens = minNewTokens;
        this.generatedTokenOffset = generatedTokenOffset;
        this.seed = seed;
        this.typicalP = typicalP;
        this.xtcProbability = xtcProbability;
        this.xtcThreshold = xtcThreshold;

        // Args 0..3 are the legacy scalar sampler contract. Args 4+ are the ADR 0106 policy envelope.
        // Args 21-23 are typical-p / XTC (appended; existing indices never renumbered).
        addTArgument(temperature, topP, (double) topK, repetitionPenalty,
                (double) decodeStrategy,
                (double) batchMax,
                (double) windowMax,
                (double) activeBatch,
                (double) activeWindow,
                (double) hiddenOutputIdx,
                (double) numBeams,
                lengthPenalty,
                penaltyAlpha,
                (double) contrastiveTopK,
                minP,
                frequencyPenalty,
                presencePenalty,
                (double) minNewTokens,
                (double) generatedTokenOffset,
                seedLowBits(seed),
                seedHighBits(seed),
                typicalP,
                xtcProbability,
                xtcThreshold);
    }

    /**
     * Full constructor for native decode loop with plan handle.
     *
     * @param prefillEmbeddings  merged embeddings for step 0 [1, seqLen, hidden]
     * @param embeddingTable     token embedding table [vocabSize, hidden]
     * @param inputIds           prompt token IDs [1, seqLen] INT64
     * @param attentionMask      initial attention mask [1, 1, 1, maxKvLen] (nullable)
     * @param positionIds        initial position IDs [1, 1] INT64 (nullable)
     * @param staticKvBuffers    pre-allocated static KV cache tensors (nullable)
     * @param planHandle         native plan pointer (from DynamicShapePlanExecutor)
     * @param contextHandle      OpaqueContext pointer with ext inputs registered (persistent)
     * @param numPlanExternalInputs number of external inputs in the plan
     * @param numPlanOutputs     number of plan outputs
     * @param embeddingsExtIdx   index of inputs_embeds in plan ext inputs
     * @param maskExtIdx         index of attention_mask in plan ext inputs
     * @param causalMaskExtIdx   index of causal_mask (-1 if unused)
     * @param posIdsExtIdx       index of position_ids in plan ext inputs
     * @param inputIdsExtIdx     index of input_ids in plan ext inputs
     * @param logitsOutputIdx    which plan output is the logits tensor
     * @param attnMaskReformatExtIdx index of attn_mask_reformat (-1 if unused)
     * @param cachePositionExtIdx ext input index for cache_position / seqlens_k (-1 if unused).
     *                            When >= 0, enables in-place KV write: the onnx_mha op writes
     *                            new K/V at this position into pastKey/pastValue, skipping the
     *                            bulk past→present copy. The decode loop updates the device-side
     *                            value each step. Sets planOwnsKvScatter to skip external KV scatter.
     * @param kvInputExtIndices  ext input indices for past_key_values (2*numKvPairs)
     * @param kvOutputIndices    plan output indices for present KVs (2*numKvPairs)
     * @param maxNewTokens       maximum decode steps
     * @param eosTokenId         end-of-sequence token
     * @param numKvPairs         number of KV layer pairs
     * @param prefillSeqLen      length of the prefill sequence
     * @param temperature        sampling temperature (0 = greedy)
     * @param topK               top-K sampling (0 = disabled)
     * @param topP               nucleus sampling threshold (0 = disabled)
     * @param repetitionPenalty  repetition penalty (1.0 = disabled)
     * @param additionalStopIds  additional stop token IDs (may be null)
     */
    public AutoregressiveDecode(INDArray prefillEmbeddings,
                                 INDArray embeddingTable,
                                 INDArray inputIds,
                                 INDArray attentionMask,
                                 INDArray positionIds,
                                 INDArray[] staticKvBuffers,
                                 Pointer planHandle,
                                 Pointer contextHandle,
                                 int numPlanExternalInputs,
                                 int numPlanOutputs,
                                 int embeddingsExtIdx,
                                 int maskExtIdx,
                                 int causalMaskExtIdx,
                                 int posIdsExtIdx,
                                 int inputIdsExtIdx,
                                 int logitsOutputIdx,
                                 int attnMaskReformatExtIdx,
                                 int cachePositionExtIdx,
                                 int[] kvInputExtIndices,
                                 int[] kvOutputIndices,
                                 int maxNewTokens,
                                 int eosTokenId,
                                 int numKvPairs,
                                 int prefillSeqLen,
                                 double temperature,
                                 int topK,
                                 double topP,
                                 double repetitionPenalty,
                                 Set<Integer> additionalStopIds) {
        // Build input array
        int optionalMask = 0;
        List<INDArray> inputList = new ArrayList<>();
        inputList.add(prefillEmbeddings);
        inputList.add(embeddingTable);
        inputList.add(inputIds);
        if (attentionMask != null) {
            inputList.add(attentionMask);
            optionalMask |= 1;
        }
        if (positionIds != null) {
            inputList.add(positionIds);
            optionalMask |= 2;
        }
        if (staticKvBuffers != null) {
            optionalMask |= 4;
            for (INDArray kv : staticKvBuffers) {
                if (kv != null) inputList.add(kv);
            }
        }
        // optionalMask bit 3 signals attnMaskReformatExtIdx is present in iArgs.
        if (attnMaskReformatExtIdx >= 0) {
            optionalMask |= 8;
        }
        // optionalMask bit 4: in-place KV write mode.
        // When cachePositionExtIdx >= 0, the onnx_mha op writes K/V directly into
        // pastKey/pastValue at cache_position, eliminating the bulk copy.
        // planOwnsKvScatter is set in C++ so the decode loop skips external KV scatter.
        if (cachePositionExtIdx >= 0) {
            optionalMask |= 16;
        }
        // Plan ext inputs read from the persistent OpaqueContext in C++.
        this.optionalInputMask = optionalMask;

        addInputArgument(inputList.toArray(new INDArray[0]));

        this.maxNewTokens = maxNewTokens;
        this.eosTokenId = eosTokenId;
        this.numKvPairs = numKvPairs;
        this.prefillSeqLen = prefillSeqLen;
        this.temperature = temperature;
        this.topK = topK;
        this.topP = topP;
        this.repetitionPenalty = repetitionPenalty;

        // Encode plan handle as two 32-bit integers
        long planAddr = planHandle != null ? planHandle.address() : 0L;
        long planLow = planAddr & 0xFFFFFFFFL;
        long planHigh = (planAddr >>> 32) & 0xFFFFFFFFL;

        // Encode context handle as two 32-bit integers
        long ctxAddr = contextHandle != null ? contextHandle.address() : 0L;
        long ctxLow = ctxAddr & 0xFFFFFFFFL;
        long ctxHigh = (ctxAddr >>> 32) & 0xFFFFFFFFL;

        // iArgs: fixed layout
        // 0: maxNewTokens, 1: eosTokenId, 2: numKvPairs, 3: prefillSeqLen, 4: optionalMask
        // 5: planHandleLow, 6: planHandleHigh
        // 7: contextHandleLow, 8: contextHandleHigh
        // 9: numPlanExternalInputs, 10: numPlanOutputs
        // 11: embeddingsExtIdx, 12: maskExtIdx, 13: causalMaskExtIdx
        // 14: posIdsExtIdx, 15: inputIdsExtIdx, 16: logitsOutputIdx
        // 17: attnMaskReformatExtIdx (when optionalMask bit 3 is set)
        // 18..18+2N-1: kvInputExtIndices
        // 18+2N..18+4N-1: kvOutputIndices
        // after kv indices: additional stop token IDs
        List<Long> iArgs = new ArrayList<>();
        iArgs.add((long) maxNewTokens);          // 0
        iArgs.add((long) eosTokenId);            // 1
        iArgs.add((long) numKvPairs);            // 2
        iArgs.add((long) prefillSeqLen);         // 3
        iArgs.add((long) optionalMask);          // 4
        iArgs.add(planLow);                      // 5
        iArgs.add(planHigh);                     // 6
        iArgs.add(ctxLow);                       // 7
        iArgs.add(ctxHigh);                      // 8
        iArgs.add((long) numPlanExternalInputs); // 9
        iArgs.add((long) numPlanOutputs);        // 10
        iArgs.add((long) embeddingsExtIdx);      // 11
        iArgs.add((long) maskExtIdx);            // 12
        iArgs.add((long) causalMaskExtIdx);      // 13
        iArgs.add((long) posIdsExtIdx);          // 14
        iArgs.add((long) inputIdsExtIdx);        // 15
        iArgs.add((long) logitsOutputIdx);       // 16
        if ((optionalMask & 8) != 0) {
            iArgs.add((long) attnMaskReformatExtIdx); // 17 when present
        }
        // In-place KV: positionOffsetExtIdx (-1 for ONNX, unused) + cachePositionExtIdx
        if ((optionalMask & 16) != 0) {
            iArgs.add(-1L);  // positionOffsetExtIdx — not used by ONNX models
            iArgs.add((long) cachePositionExtIdx);
        }

        // KV input ext indices
        if (kvInputExtIndices != null) {
            for (int idx : kvInputExtIndices) {
                iArgs.add((long) idx);
            }
        }
        // KV output indices — only for non-in-graph KV mode.
        // When bit 4 is set (planOwnsKvScatter), attention writes KV in-place;
        // the decode loop skips external KV scatter and doesn't need output indices.
        // Matching the GGUF constructor which sends 0 kvOutputIndices.
        if ((optionalMask & 16) == 0 && kvOutputIndices != null) {
            for (int idx : kvOutputIndices) {
                iArgs.add((long) idx);
            }
        }
        // Additional stop IDs
        if (additionalStopIds != null) {
            for (int stopId : additionalStopIds) {
                iArgs.add((long) stopId);
            }
        }

        addIArgument(iArgs.stream().mapToLong(Long::longValue).toArray());

        // Float args: legacy scalar sampler config + ADR 0106 policy envelope.
        addSamplingPolicyArguments(temperature, topP, topK, repetitionPenalty);
    }

    /**
     * Constructor for in-graph KV cache mode (GGUF pattern).
     *
     * <p>The attention op writes K/V in-place at cachePosition. No present outputs,
     * no external KV scatter. positionOffsetExtIdx and cachePositionExtIdx point to
     * scalar ext inputs updated per step by the C++ decode loop.</p>
     *
     * @param positionOffsetExtIdx ext input index for position_offset scalar (-1 if unused)
     * @param cachePositionExtIdx  ext input index for cache_position scalar (-1 if unused)
     */
    public AutoregressiveDecode(INDArray prefillEmbeddings,
                                 INDArray embeddingTable,
                                 INDArray inputIds,
                                 INDArray attentionMask,
                                 INDArray positionIds,
                                 INDArray[] staticKvBuffers,
                                 Pointer planHandle,
                                 Pointer contextHandle,
                                 int numPlanExternalInputs,
                                 int numPlanOutputs,
                                 int embeddingsExtIdx,
                                 int maskExtIdx,
                                 int causalMaskExtIdx,
                                 int posIdsExtIdx,
                                 int inputIdsExtIdx,
                                 int logitsOutputIdx,
                                 int attnMaskReformatExtIdx,
                                 int positionOffsetExtIdx,
                                 int cachePositionExtIdx,
                                 int[] kvInputExtIndices,
                                 int[] kvOutputIndices,
                                 int[] gdnStateExtIndices,
                                 int[] gdnStateOutputIndices,
                                 int[] convStateExtIndices,
                                 int[] convStateOutputIndices,
                                 int maxNewTokens,
                                 int eosTokenId,
                                 int numKvPairs,
                                 int prefillSeqLen,
                                 double temperature,
                                 int topK,
                                 double topP,
                                 double repetitionPenalty,
                                 Set<Integer> additionalStopIds) {
        // Build input array
        int optionalMask = 0;
        List<INDArray> inputList = new ArrayList<>();
        inputList.add(prefillEmbeddings);
        inputList.add(embeddingTable);
        inputList.add(inputIds);
        if (attentionMask != null) {
            inputList.add(attentionMask);
            optionalMask |= 1;
        }
        if (positionIds != null) {
            inputList.add(positionIds);
            optionalMask |= 2;
        }
        if (staticKvBuffers != null) {
            optionalMask |= 4;
            for (INDArray kv : staticKvBuffers) {
                if (kv != null) inputList.add(kv);
            }
        }
        if (attnMaskReformatExtIdx >= 0) {
            optionalMask |= 8;
        }
        // Bit 4: in-graph KV cache mode (GGUF pattern)
        optionalMask |= 16;
        // Bit 5: GDN/conv recurrent state feedback
        boolean hasRecurrentState = (gdnStateExtIndices != null && gdnStateExtIndices.length > 0)
                || (convStateExtIndices != null && convStateExtIndices.length > 0);
        if (hasRecurrentState) {
            optionalMask |= 32;
        }
        this.optionalInputMask = optionalMask;

        addInputArgument(inputList.toArray(new INDArray[0]));

        this.maxNewTokens = maxNewTokens;
        this.eosTokenId = eosTokenId;
        this.numKvPairs = numKvPairs;
        this.prefillSeqLen = prefillSeqLen;
        this.temperature = temperature;
        this.topK = topK;
        this.topP = topP;
        this.repetitionPenalty = repetitionPenalty;

        long planAddr = planHandle != null ? planHandle.address() : 0L;
        long planLow = planAddr & 0xFFFFFFFFL;
        long planHigh = (planAddr >>> 32) & 0xFFFFFFFFL;

        long ctxAddr = contextHandle != null ? contextHandle.address() : 0L;
        long ctxLow = ctxAddr & 0xFFFFFFFFL;
        long ctxHigh = (ctxAddr >>> 32) & 0xFFFFFFFFL;

        List<Long> iArgs = new ArrayList<>();
        iArgs.add((long) maxNewTokens);          // 0
        iArgs.add((long) eosTokenId);            // 1
        iArgs.add((long) numKvPairs);            // 2
        iArgs.add((long) prefillSeqLen);         // 3
        iArgs.add((long) optionalMask);          // 4
        iArgs.add(planLow);                      // 5
        iArgs.add(planHigh);                     // 6
        iArgs.add(ctxLow);                       // 7
        iArgs.add(ctxHigh);                      // 8
        iArgs.add((long) numPlanExternalInputs); // 9
        iArgs.add((long) numPlanOutputs);        // 10
        iArgs.add((long) embeddingsExtIdx);      // 11
        iArgs.add((long) maskExtIdx);            // 12
        iArgs.add((long) causalMaskExtIdx);      // 13
        iArgs.add((long) posIdsExtIdx);          // 14
        iArgs.add((long) inputIdsExtIdx);        // 15
        iArgs.add((long) logitsOutputIdx);       // 16
        if ((optionalMask & 8) != 0) {
            iArgs.add((long) attnMaskReformatExtIdx); // 17 when present
        }
        // In-graph KV cache: positionOffsetExtIdx + cachePositionExtIdx
        iArgs.add((long) positionOffsetExtIdx);
        iArgs.add((long) cachePositionExtIdx);

        // KV input ext indices
        if (kvInputExtIndices != null) {
            for (int idx : kvInputExtIndices) {
                iArgs.add((long) idx);
            }
        }
        // KV output indices (empty for in-graph KV — no present outputs)
        if (kvOutputIndices != null) {
            for (int idx : kvOutputIndices) {
                iArgs.add((long) idx);
            }
        }
        // GDN/conv recurrent state indices (bit 5)
        if (hasRecurrentState) {
            int numGdn = gdnStateExtIndices != null ? gdnStateExtIndices.length : 0;
            int numConv = convStateExtIndices != null ? convStateExtIndices.length : 0;
            iArgs.add((long) numGdn);
            iArgs.add((long) numConv);
            if (gdnStateExtIndices != null) {
                for (int idx : gdnStateExtIndices) iArgs.add((long) idx);
            }
            if (gdnStateOutputIndices != null) {
                for (int idx : gdnStateOutputIndices) iArgs.add((long) idx);
            }
            if (convStateExtIndices != null) {
                for (int idx : convStateExtIndices) iArgs.add((long) idx);
            }
            if (convStateOutputIndices != null) {
                for (int idx : convStateOutputIndices) iArgs.add((long) idx);
            }
        }
        // Additional stop IDs
        if (additionalStopIds != null) {
            for (int stopId : additionalStopIds) {
                iArgs.add((long) stopId);
            }
        }

        addIArgument(iArgs.stream().mapToLong(Long::longValue).toArray());
        addSamplingPolicyArguments(temperature, topP, topK, repetitionPenalty);
    }

    /**
     * Simplified constructor without plan handle (legacy/testing).
     */
    public AutoregressiveDecode(INDArray prefillEmbeddings,
                                 INDArray embeddingTable,
                                 INDArray inputIds,
                                 INDArray attentionMask,
                                 INDArray positionIds,
                                 INDArray[] staticKvBuffers,
                                 int maxNewTokens,
                                 int eosTokenId,
                                 int numKvPairs,
                                 int prefillSeqLen,
                                 double temperature,
                                 int topK,
                                 double topP,
                                 Set<Integer> additionalStopIds) {
        // Build input array
        int optionalMask = 0;
        List<INDArray> inputList = new ArrayList<>();
        inputList.add(prefillEmbeddings);
        inputList.add(embeddingTable);
        inputList.add(inputIds);
        if (attentionMask != null) {
            inputList.add(attentionMask);
            optionalMask |= 1;
        }
        if (positionIds != null) {
            inputList.add(positionIds);
            optionalMask |= 2;
        }
        if (staticKvBuffers != null) {
            optionalMask |= 4;
            for (INDArray kv : staticKvBuffers) {
                if (kv != null) inputList.add(kv);
            }
        }
        this.optionalInputMask = optionalMask;

        addInputArgument(inputList.toArray(new INDArray[0]));

        this.maxNewTokens = maxNewTokens;
        this.eosTokenId = eosTokenId;
        this.numKvPairs = numKvPairs;
        this.prefillSeqLen = prefillSeqLen;
        this.temperature = temperature;
        this.topK = topK;
        this.topP = topP;

        // iArgs: only the first 5 (no plan config)
        addIArgument((long) maxNewTokens, (long) eosTokenId, (long) numKvPairs,
                     (long) prefillSeqLen, (long) optionalMask);
        if (additionalStopIds != null) {
            for (int stopId : additionalStopIds) {
                addIArgument((long) stopId);
            }
        }

        // Float args: legacy scalar sampler config + ADR 0106 policy envelope.
        addSamplingPolicyArguments(temperature, topP, topK, 1.0);
    }

    /**
     * Wire ADR 0107 V2 quantized-KV scale buffers into the op after construction.
     *
     * <p>The caller has already constructed this op with {@code staticKvBuffers} containing the INT8
     * KV arrays (bit 2 in optionalMask). This method appends {@code 2*numKvPairs} float32 scale
     * arrays after the KV buffers and sets bit 7 (128) in {@code iArgs[4]} (optionalInputMask) so
     * the C++ decode op knows to read them.</p>
     *
     * <p>Scale layout: {@code kvScaleBuffers[0..numKvPairs-1]} = key scales per layer,
     * {@code [numKvPairs..2*numKvPairs-1]} = value scales per layer.
     * Each array shape: {@code [batch, maxKvLen, kvHeads]} float32.</p>
     *
     * @param kvScaleBuffers the 2*numKvPairs float32 scale arrays (key scales then value scales)
     * @return this op (for chaining)
     * @throws IllegalArgumentException if the array length != 2*numKvPairs
     */
    public AutoregressiveDecode withQuantisedKvScales(INDArray[] kvScaleBuffers) {
        if (kvScaleBuffers == null || kvScaleBuffers.length == 0) return this;
        if (kvScaleBuffers.length != 2 * numKvPairs) {
            throw new IllegalArgumentException(
                    "withQuantisedKvScales: expected " + (2 * numKvPairs) + " scale arrays, got " + kvScaleBuffers.length);
        }
        // Append scale buffers to the input list
        for (INDArray sc : kvScaleBuffers) {
            if (sc != null) inputArguments.add(sc);
        }
        // Set bit 7 (128) in iArgs[4] (optionalInputMask)
        if (iArguments.size() > 4) {
            long prevMask = iArguments.get(4);
            iArguments.set(4, prevMask | 128L);
            this.optionalInputMask = (int)(prevMask | 128L);
        }
        return this;
    }

    /**
     * Override the decode-policy envelope carried in {@code tArgs[4..13]} while preserving the scalar
     * sampler args and extended sampler policy. This is the Java-side handoff point for ADR 0106 once
     * GenerationPipeline starts constructing B/W substrate calls.
     */
    public AutoregressiveDecode withDecodePolicy(int decodeStrategy, int batchMax, int windowMax,
                                                 int activeBatch, int activeWindow, int hiddenOutputIdx,
                                                 int numBeams, double lengthPenalty,
                                                 double penaltyAlpha, int contrastiveTopK) {
        tArguments.clear();
        addSamplingPolicyArguments(temperature, topP, topK, repetitionPenalty,
                decodeStrategy, batchMax, windowMax, activeBatch, activeWindow,
                hiddenOutputIdx, numBeams, lengthPenalty, penaltyAlpha, contrastiveTopK,
                minP, frequencyPenalty, presencePenalty, minNewTokens, generatedTokenOffset, seed,
                typicalP, xtcProbability, xtcThreshold);
        return this;
    }

    /**
     * Override scalar sampler policy metadata carried in {@code tArgs[14..20]} while preserving iArgs
     * and the ADR 0106 decode policy envelope.
     */
    public AutoregressiveDecode withSamplingPolicy(double minP, double frequencyPenalty,
                                                   double presencePenalty, int minNewTokens,
                                                   int generatedTokenOffset, long seed) {
        tArguments.clear();
        addSamplingPolicyArguments(temperature, topP, topK, repetitionPenalty,
                decodeStrategy, batchMax, windowMax, activeBatch, activeWindow,
                hiddenOutputIdx, numBeams, lengthPenalty, penaltyAlpha, contrastiveTopK,
                minP, frequencyPenalty, presencePenalty, minNewTokens, generatedTokenOffset, seed,
                typicalP, xtcProbability, xtcThreshold);
        return this;
    }

    /**
     * Override typical-p and XTC parameters while preserving all other policy arguments.
     */
    public AutoregressiveDecode withTypicalPAndXtc(double typicalP, double xtcProbability,
                                                   double xtcThreshold) {
        tArguments.clear();
        addSamplingPolicyArguments(temperature, topP, topK, repetitionPenalty,
                decodeStrategy, batchMax, windowMax, activeBatch, activeWindow,
                hiddenOutputIdx, numBeams, lengthPenalty, penaltyAlpha, contrastiveTopK,
                minP, frequencyPenalty, presencePenalty, minNewTokens, generatedTokenOffset, seed,
                typicalP, xtcProbability, xtcThreshold);
        return this;
    }

    /**
     * Wire ADR 0106 Phase 2 n-gram speculative decoding parameters into tArgs[24/25].
     *
     * <p>Must be called AFTER all other tArg-building methods (withDecodePolicy,
     * withSamplingPolicy, withTypicalPAndXtc) since it appends at fixed offsets 24 and 25.
     * speculativeK=0 disables speculation and leaves the W=1 path completely unchanged.</p>
     *
     * <p>The C++ decode loop reads these optional args with a size() guard so older
     * callers that do not call this method see speculativeK=0 (off) automatically.</p>
     *
     * @param k              max draft tokens per step (0 to disable)
     * @param speculatorType 0=none, 1=NGRAM (bigram proposer, host-side)
     * @return this op (for chaining)
     */
    public AutoregressiveDecode withSpeculativeDecoding(int k, int speculatorType) {
        this.speculativeK = k;
        this.speculatorType = speculatorType;
        // Ensure tArguments has exactly 24 entries before appending [24] and [25].
        // The standard sampling path writes 24 args (indices 0..23); we extend to 26.
        while (tArguments.size() < 24) {
            tArguments.add(0.0);
        }
        if (tArguments.size() == 24) {
            tArguments.add((double) k);
            tArguments.add((double) speculatorType);
        } else if (tArguments.size() == 25) {
            tArguments.set(24, (double) k);
            tArguments.add((double) speculatorType);
        } else {
            // tArguments.size() >= 26: overwrite in-place
            tArguments.set(24, (double) k);
            tArguments.set(25, (double) speculatorType);
        }
        return this;
    }

    /**
     * Identify the target plan's {@code actual_sequence_length} scalar so the native
     * speculative loop can set it to the live verification width before each replay.
     * This argument is appended after the Phase 2 parameters and is ignored by callers
     * whose target graph has no recurrent sequence-length control.
     */
    public AutoregressiveDecode withActualSequenceLengthExtIdx(int extIdx) {
        this.actualSequenceLengthExtIdx = extIdx;
        while (tArguments.size() < 26) {
            tArguments.add(0.0);
        }
        if (tArguments.size() == 26) {
            tArguments.add((double) extIdx);
        } else {
            tArguments.set(26, (double) extIdx);
        }
        return this;
    }

    /**
     * Attach the isolated Qwen3.5 MTP predictor plan to this target decode invocation.
     *
     * <p>The seven arrays are appended after the target KV (and optional quantised-scale)
     * inputs. MTP plan metadata is encoded in tArgs[27..42], preserving every existing
     * iArg and stop-token offset. All pointer halves are unsigned 32-bit values and are
     * therefore exactly representable as doubles.</p>
     */
    public AutoregressiveDecode withMtpPlan(
            INDArray mtpInputIds,
            INDArray mtpTargetHiddenStates,
            INDArray mtpCausalMask,
            INDArray mtpPositionOffset,
            INDArray mtpCachePosition,
            INDArray[] mtpKvBuffers,
            Pointer mtpPlanHandle,
            Pointer mtpContextHandle,
            int mtpNumPlanExternalInputs,
            int mtpNumPlanOutputs,
            int mtpInputIdsExtIdx,
            int mtpTargetHiddenExtIdx,
            int mtpCausalMaskExtIdx,
            int mtpPositionOffsetExtIdx,
            int mtpCachePositionExtIdx,
            int[] mtpKvInputExtIndices,
            int mtpLogitsOutputIdx,
            int mtpHiddenOutputIdx,
            int targetHiddenOutputIdx) {

        if (mtpInputIds == null || mtpTargetHiddenStates == null || mtpCausalMask == null
                || mtpPositionOffset == null || mtpCachePosition == null) {
            throw new IllegalArgumentException("withMtpPlan requires all five mutable MTP inputs");
        }
        if (mtpKvBuffers == null || mtpKvBuffers.length != 2
                || mtpKvBuffers[0] == null || mtpKvBuffers[1] == null) {
            throw new IllegalArgumentException("withMtpPlan requires exactly one MTP key/value cache pair");
        }
        if (mtpKvInputExtIndices == null || mtpKvInputExtIndices.length != 2) {
            throw new IllegalArgumentException("withMtpPlan requires exactly two MTP KV external-input indices");
        }
        if (mtpPlanHandle == null || mtpPlanHandle.isNull()
                || mtpContextHandle == null || mtpContextHandle.isNull()) {
            throw new IllegalArgumentException("withMtpPlan requires non-null native plan and context handles");
        }
        if (targetHiddenOutputIdx < 0 || mtpLogitsOutputIdx < 0 || mtpHiddenOutputIdx < 0) {
            throw new IllegalArgumentException("withMtpPlan requires resolved target/MTP output indices");
        }

        inputArguments.add(mtpInputIds);
        inputArguments.add(mtpTargetHiddenStates);
        inputArguments.add(mtpCausalMask);
        inputArguments.add(mtpPositionOffset);
        inputArguments.add(mtpCachePosition);
        inputArguments.add(mtpKvBuffers[0]);
        inputArguments.add(mtpKvBuffers[1]);

        long prevMask = iArguments.get(4);
        iArguments.set(4, prevMask | 256L);
        this.optionalInputMask = (int) (prevMask | 256L);
        this.speculatorType = SPECULATOR_TYPE_MTP;

        while (tArguments.size() < 27) {
            tArguments.add(0.0);
        }
        tArguments.set(25, (double) SPECULATOR_TYPE_MTP);

        long planAddr = mtpPlanHandle.address();
        long contextAddr = mtpContextHandle.address();
        double[] metadata = {
                (double) (planAddr & 0xFFFFFFFFL),
                (double) ((planAddr >>> 32) & 0xFFFFFFFFL),
                (double) (contextAddr & 0xFFFFFFFFL),
                (double) ((contextAddr >>> 32) & 0xFFFFFFFFL),
                (double) mtpNumPlanExternalInputs,
                (double) mtpNumPlanOutputs,
                (double) mtpInputIdsExtIdx,
                (double) mtpTargetHiddenExtIdx,
                (double) mtpCausalMaskExtIdx,
                (double) mtpPositionOffsetExtIdx,
                (double) mtpCachePositionExtIdx,
                (double) mtpKvInputExtIndices[0],
                (double) mtpKvInputExtIndices[1],
                (double) mtpLogitsOutputIdx,
                (double) mtpHiddenOutputIdx,
                (double) targetHiddenOutputIdx
        };
        for (int i = 0; i < metadata.length; i++) {
            int index = 27 + i;
            if (tArguments.size() == index) tArguments.add(metadata[i]);
            else tArguments.set(index, metadata[i]);
        }
        return this;
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) this.maxNewTokens = iArguments.get(0).intValue();
        if (iArguments.size() > 1) this.eosTokenId = iArguments.get(1).intValue();
        if (iArguments.size() > 2) this.numKvPairs = iArguments.get(2).intValue();
        if (iArguments.size() > 3) this.prefillSeqLen = iArguments.get(3).intValue();
        if (tArguments.size() > 0) this.temperature = tArguments.get(0);
        if (tArguments.size() > 1) this.topP = tArguments.get(1);
        if (tArguments.size() > 2) this.topK = tArguments.get(2).intValue();
        if (tArguments.size() > 3) this.repetitionPenalty = tArguments.get(3);
        this.decodeStrategy = resolveScalarDecodeStrategy(this.temperature, this.topK, this.topP);
        if (tArguments.size() > 4) this.decodeStrategy = tArguments.get(4).intValue();
        if (tArguments.size() > 5) this.batchMax = tArguments.get(5).intValue();
        if (tArguments.size() > 6) this.windowMax = tArguments.get(6).intValue();
        if (tArguments.size() > 7) this.activeBatch = tArguments.get(7).intValue();
        if (tArguments.size() > 8) this.activeWindow = tArguments.get(8).intValue();
        if (tArguments.size() > 9) this.hiddenOutputIdx = tArguments.get(9).intValue();
        if (tArguments.size() > 10) this.numBeams = tArguments.get(10).intValue();
        if (tArguments.size() > 11) this.lengthPenalty = tArguments.get(11);
        if (tArguments.size() > 12) this.penaltyAlpha = tArguments.get(12);
        if (tArguments.size() > 13) this.contrastiveTopK = tArguments.get(13).intValue();
        if (tArguments.size() > 14) this.minP = tArguments.get(14);
        if (tArguments.size() > 15) this.frequencyPenalty = tArguments.get(15);
        if (tArguments.size() > 16) this.presencePenalty = tArguments.get(16);
        if (tArguments.size() > 17) this.minNewTokens = tArguments.get(17).intValue();
        if (tArguments.size() > 18) this.generatedTokenOffset = tArguments.get(18).intValue();
        if (tArguments.size() > 20) this.seed = combineSeed(tArguments.get(19), tArguments.get(20));
        if (tArguments.size() > 21) this.typicalP = tArguments.get(21);
        if (tArguments.size() > 22) this.xtcProbability = tArguments.get(22);
        if (tArguments.size() > 23) this.xtcThreshold = tArguments.get(23);
        // ADR 0106 Phase 2: n-gram speculative decoding (tArgs 24/25, optional)
        if (tArguments.size() > 24) this.speculativeK = tArguments.get(24).intValue();
        if (tArguments.size() > 25) this.speculatorType = tArguments.get(25).intValue();
        if (tArguments.size() > 26) this.actualSequenceLengthExtIdx = tArguments.get(26).intValue();
    }

    @Override
    public String opName() {
        return "autoregressive_decode";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Arrays.asList(DataType.INT64, DataType.INT64, DataType.FLOAT);
    }

    @Override
    public int getNumOutputs() {
        return 3;
    }
}
