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

package org.eclipse.deeplearning4j.llm.generation;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.factory.Nd4j;

import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Utilities for decoder model execution: causal masks, KV cache, and output name resolution.
 */
@Slf4j
public class DecoderUtils {

    /** Mask fill value matching torch.finfo(torch.float32).min */
    public static final float MASK_FILL = -3.4028235e+38f;

    public static class KVCacheNames {
        public final List<String> keyNames;
        public final List<String> valueNames;

        public KVCacheNames(List<String> keyNames, List<String> valueNames) {
            this.keyNames = keyNames;
            this.valueNames = valueNames;
        }
    }

    /**
     * Build a causal attention mask for the decoder (FLOAT dtype).
     *
     * For prefill (currentSeqLen &gt; 1): upper-triangular mask filled with MASK_FILL
     *   mask[q][k] = 0 if k &lt;= (pastSeqLen + q), else MASK_FILL
     *
     * For decode (currentSeqLen == 1): all zeros (single token attends to all past)
     *
     * Uses FLOAT (not LONG) because AddOp doesn't promote LONG-&gt;FLOAT correctly.
     *
     * @param currentSeqLen number of query tokens in this step
     * @param totalSeqLen total KV length (past + current)
     * @return INDArray of shape [1, 1, currentSeqLen, totalSeqLen] with FLOAT dtype
     */
    public static INDArray buildCausalMask(long currentSeqLen, long totalSeqLen) {
        return buildCausalMask(1, currentSeqLen, totalSeqLen);
    }

    /**
     * Build a batched causal attention mask for the decoder (FLOAT dtype).
     *
     * <p>For batch processing, all sequences share the same causal mask pattern
     * (they all have the same sequence length at each step).</p>
     *
     * @param batchSize number of sequences in the batch
     * @param currentSeqLen number of query tokens in this step
     * @param totalSeqLen total KV length (past + current)
     * @return INDArray of shape [batchSize, 1, currentSeqLen, totalSeqLen] with FLOAT dtype
     */
    public static INDArray buildCausalMask(long batchSize, long currentSeqLen, long totalSeqLen) {
        if (currentSeqLen == 1) {
            return Nd4j.zeros(DataType.FLOAT, batchSize, 1, 1, totalSeqLen);
        }

        long pastSeqLen = totalSeqLen - currentSeqLen;
        int Q = (int) currentSeqLen;
        int K = (int) totalSeqLen;
        float[] data = new float[Q * K];

        for (int q = 0; q < Q; q++) {
            int lastVisibleK = (int) pastSeqLen + q;
            int rowOffset = q * K;
            for (int k = lastVisibleK + 1; k < K; k++) {
                data[rowOffset + k] = MASK_FILL;
            }
        }

        // Create single mask pattern
        INDArray singleMask = Nd4j.createFromArray(data).reshape(1, 1, currentSeqLen, totalSeqLen);

        if (batchSize == 1) {
            log.info("Built causal mask: shape=[1, 1, {}, {}], pastSeqLen={}, dtype=FLOAT",
                    currentSeqLen, totalSeqLen, pastSeqLen);
            return singleMask;
        }

        // Tile for batch dimension
        INDArray mask = Nd4j.tile(singleMask, (int) batchSize, 1, 1, 1);
        log.info("Built batched causal mask: shape=[{}, 1, {}, {}], pastSeqLen={}, dtype=FLOAT",
                batchSize, currentSeqLen, totalSeqLen, pastSeqLen);
        return mask;
    }

    /**
     * Create an empty KV cache tensor for the first decoder step.
     * Shape is [batchSize, numHeads, 0, headDim].
     *
     * Infers numHeads and headDim from the decoder graph's variable shapes.
     *
     * @param decoder the SameDiff decoder model
     * @param inputName the past_key_values input name
     * @param batchSize batch size
     * @param hiddenSize model hidden size (used for fallback inference)
     * @return empty KV cache INDArray
     */
    public static INDArray createEmptyKvCache(SameDiff decoder, String inputName, long batchSize, long hiddenSize) {
        long numHeads = -1;
        long headDim = -1;
        DataType kvType = DataType.FLOAT;

        SDVariable inputVar = decoder.getVariable(inputName);
        if (inputVar != null && inputVar.getShape() != null && inputVar.getShape().length >= 4) {
            long[] shape = inputVar.getShape();
            if (inputVar.dataType() != null) {
                kvType = inputVar.dataType();
            }
            if (shape[1] > 0) {
                numHeads = shape[1];
            }
            if (shape[3] > 0) {
                headDim = shape[3];
            }
        }

        if (numHeads <= 0 || headDim <= 0) {
            String presentName = inputName.replace("past_key_values", "present");
            SDVariable presentVar = decoder.getVariable(presentName);
            if (presentVar != null && presentVar.getShape() != null && presentVar.getShape().length >= 4) {
                long[] shape = presentVar.getShape();
                if (numHeads <= 0 && shape[1] > 0) {
                    numHeads = shape[1];
                }
                if (headDim <= 0 && shape[3] > 0) {
                    headDim = shape[3];
                }
            }
        }

        if (headDim <= 0 && numHeads > 0 && hiddenSize > 0) {
            headDim = Math.max(1, hiddenSize / numHeads);
        }
        if (numHeads <= 0 && headDim > 0 && hiddenSize > 0) {
            numHeads = Math.max(1, hiddenSize / headDim);
        }
        if (headDim <= 0) {
            headDim = 64;
        }
        if (numHeads <= 0) {
            numHeads = Math.max(1, hiddenSize / headDim);
        }

        return Nd4j.zeros(kvType, batchSize, numHeads, 0, headDim);
    }

    /**
     * Create a KV cache tensor with a specific sequence length.
     * Used for probe testing multi-token decode support.
     */
    public static INDArray createEmptyKvCache(SameDiff decoder, String inputName, long batchSize,
                                              long hiddenSize, long seqLen) {
        if (seqLen == 0) {
            return createEmptyKvCache(decoder, inputName, batchSize, hiddenSize);
        }

        long numHeads = -1;
        long headDim = -1;
        DataType kvType = DataType.FLOAT;

        SDVariable inputVar = decoder.getVariable(inputName);
        if (inputVar != null && inputVar.getShape() != null && inputVar.getShape().length >= 4) {
            long[] shape = inputVar.getShape();
            if (inputVar.dataType() != null) {
                kvType = inputVar.dataType();
            }
            if (shape[1] > 0) numHeads = shape[1];
            if (shape[3] > 0) headDim = shape[3];
        }

        if (numHeads <= 0 || headDim <= 0) {
            String presentName = inputName.replace("past_key_values", "present");
            SDVariable presentVar = decoder.getVariable(presentName);
            if (presentVar != null && presentVar.getShape() != null && presentVar.getShape().length >= 4) {
                long[] shape = presentVar.getShape();
                if (numHeads <= 0 && shape[1] > 0) numHeads = shape[1];
                if (headDim <= 0 && shape[3] > 0) headDim = shape[3];
            }
        }

        if (headDim <= 0 && numHeads > 0 && hiddenSize > 0) headDim = Math.max(1, hiddenSize / numHeads);
        if (numHeads <= 0 && headDim > 0 && hiddenSize > 0) numHeads = Math.max(1, hiddenSize / headDim);
        if (headDim <= 0) headDim = 64;
        if (numHeads <= 0) numHeads = Math.max(1, hiddenSize / headDim);

        return Nd4j.zeros(kvType, batchSize, numHeads, seqLen, headDim);
    }

    /**
     * Find the logits output variable name from a decoder model.
     *
     * @param decoder the SameDiff decoder model
     * @return the logits output name, or null if not found
     */
    public static String findLogitsOutputName(SameDiff decoder) {
        for (String outputName : decoder.outputs()) {
            if (outputName.contains("logit") || outputName.equals("logits")) {
                return outputName;
            }
        }
        if (!decoder.outputs().isEmpty()) {
            return decoder.outputs().get(0);
        }
        return null;
    }

    /**
     * Find the KV cache output names (present key/value) from a decoder model.
     *
     * @param decoder the SameDiff decoder model
     * @return KVCacheNames with sorted key and value output names
     */
    public static KVCacheNames findKVCacheOutputNames(SameDiff decoder) {
        List<String> presentKeyNames = new ArrayList<>();
        List<String> presentValueNames = new ArrayList<>();

        for (String outputName : decoder.outputs()) {
            if (outputName.contains("present") && outputName.contains("key")) {
                presentKeyNames.add(outputName);
            } else if (outputName.contains("present") && outputName.contains("value")) {
                presentValueNames.add(outputName);
            }
        }

        Collections.sort(presentKeyNames);
        Collections.sort(presentValueNames);

        return new KVCacheNames(presentKeyNames, presentValueNames);
    }

    // ==================== Static KV Cache for CUDA Graph Replay ====================

    /**
     * Pad KV cache tensors from prefill to a fixed static size for CUDA graph replay.
     *
     * Takes the present_kv outputs from prefill (shape [batch, heads, prefillLen, dim])
     * and creates new tensors padded to [batch, heads, maxKvLen, dim] with zeros for
     * unfilled positions. Returns a map from past_key_values input names to the padded tensors.
     *
     * @param prefillOutputs map of present output names to their tensors from prefill
     * @param presentKeyNames list of present key output names
     * @param presentValueNames list of present value output names
     * @param maxKvLen maximum KV cache length (prefillLen + maxNewTokens)
     * @return map from past_key_values input names to static padded tensors
     */
    public static Map<String, INDArray> padKvCacheToStaticSize(
            Map<String, INDArray> prefillOutputs,
            List<String> presentKeyNames,
            List<String> presentValueNames,
            long maxKvLen) {

        Map<String, INDArray> staticKvBuffers = new HashMap<>();
        List<String> allPresentNames = new ArrayList<>();
        allPresentNames.addAll(presentKeyNames);
        allPresentNames.addAll(presentValueNames);

        for (String presentName : allPresentNames) {
            INDArray prefillKv = prefillOutputs.get(presentName);
            if (prefillKv == null) continue;

            // prefillKv shape: [batch, heads, prefillLen, dim]
            long[] shape = prefillKv.shape();
            long batch = shape[0];
            long heads = shape[1];
            long prefillLen = shape[2];
            long dim = shape[3];

            // Create static buffer: [batch, heads, maxKvLen, dim]
            INDArray staticBuf = Nd4j.zeros(prefillKv.dataType(), batch, heads, maxKvLen, dim);

            // Copy prefill data into positions 0..prefillLen-1
            if (prefillLen > 0) {
                INDArray destSlice = staticBuf.get(
                        NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(0, prefillLen), NDArrayIndex.all());
                destSlice.assign(prefillKv);
            }

            // Map present name -> past_key_values input name
            String pastInputName = presentName.replace("present", "past_key_values");
            staticKvBuffers.put(pastInputName, staticBuf);

            log.info("  Static KV buffer '{}': [{},{},{},{}] (prefill={} padded to {})",
                    pastInputName, batch, heads, maxKvLen, dim, prefillLen, maxKvLen);
        }

        return staticKvBuffers;
    }

    /**
     * Scatter the new KV entry from decoder output into the static KV buffer at cachePos.
     *
     * The decoder concatenates past_kv [batch,heads,maxKvLen,dim] with the new token's KV,
     * producing present_kv [batch,heads,maxKvLen+1,dim]. The new token's entry is at the
     * last position (index maxKvLen). We extract it and write it into the static buffer
     * at position cachePos.
     *
     * @param staticKvBuffers map of past_key_values input names to static buffers
     * @param decoderOutputs map of decoder output names to tensors
     * @param presentKeyNames list of present key output names
     * @param presentValueNames list of present value output names
     * @param maxKvLen the static KV length (without the +1 from concat)
     * @param cachePos the position in the static buffer to write the new entry
     */
    public static void scatterNewKvEntries(
            Map<String, INDArray> staticKvBuffers,
            Map<String, INDArray> decoderOutputs,
            List<String> presentKeyNames,
            List<String> presentValueNames,
            long maxKvLen,
            long cachePos) {

        List<String> allPresentNames = new ArrayList<>();
        allPresentNames.addAll(presentKeyNames);
        allPresentNames.addAll(presentValueNames);

        for (String presentName : allPresentNames) {
            INDArray presentKv = decoderOutputs.get(presentName);
            if (presentKv == null) continue;

            String pastInputName = presentName.replace("present", "past_key_values");
            INDArray staticBuf = staticKvBuffers.get(pastInputName);
            if (staticBuf != null) {
                // Java view+assign: lightweight NDArrayIndex operations, faster than native KvScatter
                // op dispatch overhead for 60 individual tensor copies
                long lastPos = presentKv.size(2) - 1;
                INDArray newSlice = presentKv.get(
                        NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(lastPos), NDArrayIndex.all());
                INDArray targetSlice = staticBuf.get(
                        NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(cachePos), NDArrayIndex.all());
                targetSlice.assign(newSlice);
            }
        }
    }

    /**
     * Build the complete decoder input map for one decode step.
     *
     * Handles both dynamic KV (prefill, step 0) and static KV (decode, steps 1+) modes.
     * Constructs inputs_embeds, attention_mask, _causal_mask, input_ids, position_ids,
     * and past_key_values entries based on the decoder's declared input names.
     *
     * @param decoderInputNames the decoder model's input names
     * @param decoder the SameDiff decoder model (for createEmptyKvCache fallback)
     * @param embeddings current step's embeddings [batch, seqLen, hidden]
     * @param inputIds current step's input IDs [batch, seqLen]
     * @param pastSeqLen logical past sequence length (for position_ids / RoPE)
     * @param currentSeqLen current step's sequence length
     * @param staticKvBuffers map of past_key_values input names to static buffers (null if not using static KV)
     * @param maxKvLen total static KV length (ignored if not using static KV)
     * @param cachePos next write position in static buffer (ignored if not using static KV)
     * @param usingStaticKv whether we are in static KV mode
     * @param hiddenSize model hidden size (for empty KV cache creation)
     * @return map of input name to INDArray ready to pass to decoder.output()
     */
    public static Map<String, INDArray> buildDecoderInputMap(
            List<String> decoderInputNames, SameDiff decoder,
            INDArray embeddings, INDArray inputIds,
            long pastSeqLen, long currentSeqLen,
            Map<String, INDArray> staticKvBuffers, long maxKvLen, long cachePos,
            boolean usingStaticKv, long hiddenSize) {
        return buildDecoderInputMap(decoderInputNames, decoder, embeddings, inputIds,
                pastSeqLen, currentSeqLen, staticKvBuffers, maxKvLen, cachePos,
                usingStaticKv, hiddenSize, null);
    }

    /**
     * Build the complete decoder input map for one decode step, with optional reusable input cache.
     *
     * When {@code reusableInputs} is non-null and in static KV mode with seqLen=1,
     * attention_mask, _causal_mask, and position_ids are allocated once and updated in-place
     * on subsequent calls, avoiding per-step allocations.
     *
     * @param reusableInputs optional cache map; populated on first use, updated in-place thereafter
     */
    public static Map<String, INDArray> buildDecoderInputMap(
            List<String> decoderInputNames, SameDiff decoder,
            INDArray embeddings, INDArray inputIds,
            long pastSeqLen, long currentSeqLen,
            Map<String, INDArray> staticKvBuffers, long maxKvLen, long cachePos,
            boolean usingStaticKv, long hiddenSize,
            Map<String, INDArray> reusableInputs) {

        Map<String, INDArray> decoderInputMap = new HashMap<>();
        boolean canReuse = reusableInputs != null && usingStaticKv && currentSeqLen == 1;

        for (String inputName : decoderInputNames) {
            if (inputName.equals("inputs_embeds")) {
                decoderInputMap.put(inputName, embeddings);
            } else if (inputName.equals("attention_mask")) {
                if (usingStaticKv) {
                    long totalSeqLen = maxKvLen + currentSeqLen;
                    if (canReuse && reusableInputs.containsKey("attention_mask")) {
                        // Reuse existing mask, just set the new cachePos bit
                        INDArray mask = reusableInputs.get("attention_mask");
                        if (cachePos > 0) {
                            mask.putScalar(0, cachePos - 1, 1);
                        }
                        // putScalar writes to host buffer via JavaCPP indexer but does NOT
                        // mark the host as "written" in the C++ DataBuffer tracking counters.
                        // Without this tag, syncToDevice() in the frozen fast path is a no-op
                        // (isSpecialActual() returns true), so the CUDA graph replays with
                        // stale attention_mask data — the model never sees new decode positions.
                        Nd4j.getAffinityManager().tagLocation(mask, AffinityManager.Location.HOST);
                        decoderInputMap.put(inputName, mask);
                    } else {
                        INDArray mask = Nd4j.zeros(DataType.LONG, 1, totalSeqLen);
                        if (cachePos > 0) {
                            mask.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, cachePos)).assign(1);
                        }
                        mask.putScalar(0, totalSeqLen - 1, 1);
                        // Ensure host-side putScalar is visible to C++ syncToDevice
                        Nd4j.getAffinityManager().tagLocation(mask, AffinityManager.Location.HOST);
                        decoderInputMap.put(inputName, mask);
                        if (canReuse) reusableInputs.put("attention_mask", mask);
                    }
                } else {
                    long totalSeqLen = pastSeqLen + currentSeqLen;
                    decoderInputMap.put(inputName, Nd4j.ones(DataType.LONG, 1, totalSeqLen));
                }
            } else if (inputName.equals("_causal_mask")) {
                if (canReuse && reusableInputs.containsKey("_causal_mask")) {
                    // For seqLen=1, causal mask is all zeros — reuse directly
                    decoderInputMap.put(inputName, reusableInputs.get("_causal_mask"));
                } else {
                    long totalSeqLen = usingStaticKv ? maxKvLen + currentSeqLen : pastSeqLen + currentSeqLen;
                    INDArray causalMask = buildCausalMask(currentSeqLen, totalSeqLen);
                    decoderInputMap.put(inputName, causalMask);
                    if (canReuse) reusableInputs.put("_causal_mask", causalMask);
                }
            } else if (inputName.equals("input_ids")) {
                decoderInputMap.put(inputName, inputIds);
            } else if (inputName.equals("position_ids")) {
                if (canReuse && reusableInputs.containsKey("position_ids")) {
                    INDArray posIds = reusableInputs.get("position_ids");
                    posIds.putScalar(0, 0, pastSeqLen);
                    // Ensure host-side putScalar is visible to C++ syncToDevice
                    Nd4j.getAffinityManager().tagLocation(posIds, AffinityManager.Location.HOST);
                    decoderInputMap.put(inputName, posIds);
                } else {
                    INDArray posIds = Nd4j.arange(pastSeqLen, pastSeqLen + currentSeqLen)
                            .reshape(1, currentSeqLen).castTo(DataType.LONG);
                    decoderInputMap.put(inputName, posIds);
                    if (canReuse) reusableInputs.put("position_ids", posIds);
                }
            } else if (inputName.startsWith("past_key_values.")) {
                if (usingStaticKv) {
                    INDArray staticBuf = staticKvBuffers.get(inputName);
                    if (staticBuf != null) {
                        decoderInputMap.put(inputName, staticBuf);
                    } else {
                        decoderInputMap.put(inputName, createEmptyKvCache(decoder, inputName, 1, hiddenSize));
                    }
                } else {
                    decoderInputMap.put(inputName, createEmptyKvCache(decoder, inputName, 1, hiddenSize));
                }
            }
        }

        if (!decoderInputMap.containsKey("inputs_embeds")) {
            decoderInputMap.put("inputs_embeds", embeddings);
        }

        return decoderInputMap;
    }

    /**
     * Build a static-size attention mask for decode steps with a padded KV cache.
     *
     * Returns [batchSize, maxKvLen+1] LONG mask: 1 for valid positions 0..cachePos-1,
     * 0 for padding positions cachePos..maxKvLen-1, and 1 for position maxKvLen
     * (the new token being decoded). Finished sequences get all zeros.
     *
     * @param batchSize number of sequences in the batch
     * @param cachePos number of valid KV entries (positions 0..cachePos-1 are filled)
     * @param maxKvLen maximum KV cache length (total static buffer size)
     * @param finished per-sequence finished flags (length = original batchSize)
     * @return INDArray of shape [batchSize, maxKvLen+1] with LONG dtype
     */
    public static INDArray buildStaticDecodeMask(int batchSize, long cachePos, long maxKvLen, boolean[] finished) {
        long totalLen = maxKvLen + 1; // past KV positions + new token position
        INDArray mask = Nd4j.zeros(DataType.LONG, batchSize, totalLen);

        // Set all valid past positions (columns 0..cachePos-1) to 1 for all rows
        if (cachePos > 0) {
            mask.get(NDArrayIndex.all(), NDArrayIndex.interval(0, cachePos)).assign(1);
        }
        // Set new token position (column maxKvLen) to 1 for all rows
        mask.get(NDArrayIndex.all(), NDArrayIndex.point(maxKvLen)).assign(1);

        // Zero out finished sequences: build [batchSize,1] active mask, broadcast-multiply
        long[] activeFlags = new long[batchSize];
        for (int b = 0; b < batchSize; b++) {
            activeFlags[b] = (b < finished.length && finished[b]) ? 0L : 1L;
        }
        INDArray activeMask = Nd4j.createFromArray(activeFlags).reshape(batchSize, 1);
        mask.muli(activeMask);

        return mask;
    }
}
