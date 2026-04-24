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
 *   2: timingInfo [5] FLOAT (totalMs, avgDecodeMs, tokPerSec, p50Ms, p99Ms)
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
 */
@NoArgsConstructor
public class AutoregressiveDecode extends DynamicCustomOp {

    @Getter private int maxNewTokens;
    @Getter private int eosTokenId;
    @Getter private int numKvPairs;
    @Getter private int prefillSeqLen;
    @Getter private double temperature;
    @Getter private double topP;
    @Getter private int topK;
    @Getter private int optionalInputMask;

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
     * @param kvInputExtIndices  ext input indices for past_key_values (2*numKvPairs)
     * @param kvOutputIndices    plan output indices for present KVs (2*numKvPairs)
     * @param maxNewTokens       maximum decode steps
     * @param eosTokenId         end-of-sequence token
     * @param numKvPairs         number of KV layer pairs
     * @param prefillSeqLen      length of the prefill sequence
     * @param temperature        sampling temperature (0 = greedy)
     * @param topK               top-K sampling (0 = disabled)
     * @param topP               nucleus sampling threshold (0 = disabled)
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
                                 int[] kvInputExtIndices,
                                 int[] kvOutputIndices,
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
        // optionalMask bit 3 signals attnMaskReformatExtIdx is present in iArgs.
        if (attnMaskReformatExtIdx >= 0) {
            optionalMask |= 8;
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

        // KV input ext indices
        if (kvInputExtIndices != null) {
            for (int idx : kvInputExtIndices) {
                iArgs.add((long) idx);
            }
        }
        // KV output indices
        if (kvOutputIndices != null) {
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

        // Float args: temperature, topP, topK (as float)
        addTArgument(temperature, topP, (double) topK);
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

        // Float args: temperature, topP, topK (as float)
        addTArgument(temperature, topP, (double) topK);
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
