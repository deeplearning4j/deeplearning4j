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

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Draft-model speculator that runs a smaller SameDiff model to produce speculative tokens.
 *
 * <p>This implements the classic draft-target speculative decoding approach from
 * <em>Leviathan et al. "Fast Inference from Transformers via Speculative Decoding" (2023)</em>.
 * A smaller "draft" model (e.g., 135M params) generates K candidate tokens autoregressively,
 * then the larger "target" model verifies all K tokens in a single forward pass.</p>
 *
 * <p>The draft model runs K sequential steps (each cheap due to small model size),
 * and the target model runs 1 step with K+1 positions. When acceptance rate is high,
 * the effective throughput approaches {@code (K+1) / (draftTime + targetTime)} tokens per
 * unit time, versus {@code 1 / targetTime} without speculation.</p>
 *
 * <h3>KV Cache Management</h3>
 * <p>Uses pre-allocated static KV buffers with scatter-based writes instead of dup/close.
 * Checkpoint and rollback are zero-copy via {@link SpeculativeKVCacheManager} -- only the
 * cache position pointer is saved/restored, since positions beyond draftPastSeqLen are
 * masked out by the attention mask and will be overwritten on the next append.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see Speculator
 * @see SpeculativeKVCacheManager
 * @see TreeAttentionVerifier
 */
@Slf4j
public class DraftModelSpeculator implements Speculator, AutoCloseable {

    @Getter
    private final String name;

    @Getter
    private final int maxSpeculativeTokens;

    private final SameDiff draftModel;
    private final Function<int[], INDArray> embedFunction;
    private final Function<INDArray, Integer> decodeFunction;
    private final ModelIOConfig ioConfig;
    private final long hiddenSize;
    private final long vocabSize;

    /** Maximum draft KV cache sequence length. */
    private final int maxDraftKvLen;

    /** Pre-allocated static KV buffers, keyed by past_key_values.* input names. Reused across steps. */
    private Map<String, INDArray> staticDraftKvBuffers;
    private boolean staticKvInitialized;

    /** Draft model past sequence length (rolled back after each speculation round). */
    private long draftPastSeqLen;

    /** Checkpoint position saved at last verified state. */
    private long checkpointPastSeqLen;

    /** Zero-copy checkpoint/rollback manager for static KV caches. */
    private final SpeculativeKVCacheManager specKvManager = new SpeculativeKVCacheManager();
    private int activeCheckpointId = -1;

    // Cached decoder input names (resolved once from the model)
    private List<String> decoderInputNames;

    /** Device where the draft model's constants reside (detected lazily). */
    private int draftModelDevice = -1;

    // Statistics
    private long totalDraftSteps;
    private long totalDraftTimeMs;

    /**
     * Create a draft-model speculator.
     *
     * @param name               human-readable name (e.g., "draft-smollm2-135m")
     * @param draftModel         the draft SameDiff model
     * @param embedFunction      maps token ID array to embeddings [1, seqLen, hiddenSize]
     * @param decodeFunction     extracts greedy token from logits [1, 1, vocabSize]
     * @param ioConfig           model I/O configuration (input names, output names, KV cache names)
     * @param hiddenSize         hidden dimension of the draft model
     * @param vocabSize          vocabulary size of the draft model (for clamping out-of-range token IDs)
     * @param maxSpeculativeTokens maximum tokens to speculate per round (K)
     * @param maxDraftKvLen      maximum static KV cache length
     * @param modelDeviceId      CUDA device where the draft model's constants reside (-1 for auto-detect)
     */
    public DraftModelSpeculator(
            String name,
            SameDiff draftModel,
            Function<int[], INDArray> embedFunction,
            Function<INDArray, Integer> decodeFunction,
            ModelIOConfig ioConfig,
            long hiddenSize,
            long vocabSize,
            int maxSpeculativeTokens,
            int maxDraftKvLen,
            int modelDeviceId) {

        if (maxSpeculativeTokens < 1) {
            throw new IllegalArgumentException("maxSpeculativeTokens must be >= 1, got " + maxSpeculativeTokens);
        }
        if (maxDraftKvLen < 1) {
            throw new IllegalArgumentException("maxDraftKvLen must be >= 1, got " + maxDraftKvLen);
        }

        this.name = name;
        this.draftModel = draftModel;
        this.embedFunction = embedFunction;
        this.decodeFunction = decodeFunction;
        this.ioConfig = ioConfig;
        this.hiddenSize = hiddenSize;
        this.vocabSize = vocabSize;
        this.maxSpeculativeTokens = maxSpeculativeTokens;
        this.maxDraftKvLen = maxDraftKvLen;
        this.draftPastSeqLen = 0;
        this.checkpointPastSeqLen = 0;
        this.draftModelDevice = modelDeviceId;
    }

    /**
     * Convenience constructor that auto-detects the model's device.
     */
    public DraftModelSpeculator(
            String name,
            SameDiff draftModel,
            Function<int[], INDArray> embedFunction,
            Function<INDArray, Integer> decodeFunction,
            ModelIOConfig ioConfig,
            long hiddenSize,
            long vocabSize,
            int maxSpeculativeTokens,
            int maxDraftKvLen) {
        this(name, draftModel, embedFunction, decodeFunction, ioConfig,
                hiddenSize, vocabSize, maxSpeculativeTokens, maxDraftKvLen, -1);
    }

    @Override
    public int[] speculate(int[] generatedTokens) {
        if (generatedTokens == null || generatedTokens.length == 0) {
            return new int[0];
        }

        // Switch to the draft model's device so all arrays are created there.
        // Without this, inputs land on device 0 (caller's device) and
        // ensureDeviceCoherency migrates ALL draft model constants to device 0
        // (~1.3GB/step leak that exhausts device 0 memory).
        int callerDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        int targetDevice = getDraftModelDevice();
        if (targetDevice >= 0 && targetDevice != callerDevice) {
            DeviceMemoryManager.getInstance().switchDevice(targetDevice,
                    "DraftModelSpeculator", "speculate-colocate-with-model");
        }

        try {
            return speculateOnDevice(generatedTokens);
        } finally {
            // Restore caller's device
            if (targetDevice >= 0 && targetDevice != callerDevice) {
                DeviceMemoryManager.getInstance().switchDevice(callerDevice,
                        "DraftModelSpeculator", "speculate-restore-caller-device");
            }
        }
    }

    private int[] speculateOnDevice(int[] generatedTokens) {
        // Rollback to checkpoint (previous round's speculation may have been partial)
        rollbackToCheckpoint();

        // Clamp the seed token to draft vocab range
        int lastToken = clampToken(generatedTokens[generatedTokens.length - 1]);
        int[] speculated = new int[maxSpeculativeTokens];
        int count = 0;

        long roundStartMs = System.currentTimeMillis();

        int currentToken = lastToken;
        for (int k = 0; k < maxSpeculativeTokens; k++) {
            try {
                int nextToken = draftModelStep(currentToken);
                speculated[count++] = nextToken;
                currentToken = nextToken;
            } catch (Exception e) {
                log.warn("Draft model step {} failed: {}", k, e.getMessage());
                break;
            }
        }

        // Reset the draft model's InferenceSession to free all intermediate arrays.
        draftModel.resetSession();

        long elapsed = System.currentTimeMillis() - roundStartMs;
        totalDraftSteps += count;
        totalDraftTimeMs += elapsed;

        if (count == 0) {
            return new int[0];
        }

        if (count < maxSpeculativeTokens) {
            int[] trimmed = new int[count];
            System.arraycopy(speculated, 0, trimmed, 0, count);
            return trimmed;
        }

        return speculated;
    }

    @Override
    public SpeculationResult verify(int[] speculativeTokens, INDArray logitsPerPosition) {
        if (speculativeTokens == null || speculativeTokens.length == 0) {
            int token = argmax(logitsPerPosition, 0);
            return new SpeculationResult(0, token);
        }

        int accepted = 0;
        for (int i = 0; i < speculativeTokens.length; i++) {
            int modelToken = argmax(logitsPerPosition, i);
            if (modelToken == speculativeTokens[i]) {
                accepted++;
            } else {
                break;
            }
        }

        int correctionToken = argmax(logitsPerPosition, accepted);

        saveDraftCheckpoint(accepted);

        return new SpeculationResult(accepted, correctionToken);
    }

    /**
     * Synchronize the draft model's KV cache after the target model has verified tokens.
     * Called by the decode loop after verification to save a checkpoint
     * at the accepted position for rollback on the next speculation round.
     *
     * @param acceptedCount number of speculative tokens accepted by the target model
     */
    public void syncAfterVerification(int acceptedCount) {
        saveDraftCheckpoint(acceptedCount);
    }

    /**
     * Synchronize the draft model's KV cache with the target model's verified state.
     *
     * @param verifiedSeqLen the total sequence length accepted by the target model
     */
    public void syncToVerifiedPosition(long verifiedSeqLen) {
        this.checkpointPastSeqLen = verifiedSeqLen;
    }

    /**
     * Reset the draft model's KV cache state entirely.
     * Call when starting a new generation sequence.
     */
    public void reset() {
        if (activeCheckpointId >= 0) {
            specKvManager.discardCheckpoint(activeCheckpointId);
            activeCheckpointId = -1;
        }
        draftPastSeqLen = 0;
        checkpointPastSeqLen = 0;
    }

    public String getStats() {
        double avgStepMs = totalDraftSteps > 0 ? (double) totalDraftTimeMs / totalDraftSteps : 0;
        return String.format("DraftModelSpeculator[%s]: steps=%d, totalTime=%dms, avgStep=%.1fms, %s",
                name, totalDraftSteps, totalDraftTimeMs, avgStepMs, specKvManager.getStats());
    }

    @Override
    public void close() {
        if (staticDraftKvBuffers != null) {
            for (INDArray buf : staticDraftKvBuffers.values()) {
                if (buf != null && !buf.wasClosed()) {
                    buf.close();
                }
            }
            staticDraftKvBuffers = null;
        }
        staticKvInitialized = false;
        if (activeCheckpointId >= 0) {
            specKvManager.discardCheckpoint(activeCheckpointId);
            activeCheckpointId = -1;
        }
    }

    // ========== Internal Methods ==========

    private int clampToken(int tokenId) {
        if (vocabSize > 0 && tokenId >= vocabSize) {
            tokenId = (int) (vocabSize - 1);
        }
        if (tokenId < 0) {
            tokenId = 0;
        }
        return tokenId;
    }

    /** Resolve decoder input names from the model (cached). */
    private List<String> getDecoderInputNames() {
        if (decoderInputNames == null) {
            decoderInputNames = draftModel.inputs();
        }
        return decoderInputNames;
    }

    /** Initialize pre-allocated static KV buffers for all layers. */
    private void initStaticKvBuffers() {
        staticDraftKvBuffers = new HashMap<>();

        for (String inputName : getDecoderInputNames()) {
            if (inputName.startsWith(ioConfig.getKvCachePrefix())) {
                INDArray empty = createEmptyKvCache(draftModel, inputName);
                long numHeads = empty.size(1);
                long headDim = empty.size(3);
                DataType kvType = empty.dataType();
                empty.close();

                staticDraftKvBuffers.put(inputName,
                        Nd4j.zeros(kvType, 1, numHeads, maxDraftKvLen, headDim));
            }
        }

        staticKvInitialized = true;
        log.debug("Initialized {} static draft KV buffers with maxKvLen={}", staticDraftKvBuffers.size(), maxDraftKvLen);
    }

    /**
     * Run one autoregressive step of the draft model.
     */
    private int draftModelStep(int tokenId) {
        tokenId = clampToken(tokenId);

        if (!staticKvInitialized) {
            initStaticKvBuffers();
        }

        List<String> inputNames = getDecoderInputNames();
        boolean hasInputIds = ioConfig.hasInputIds(inputNames);
        INDArray embeddings = hasInputIds ? null : embedFunction.apply(new int[]{tokenId});

        long currentSeqLen = 1;
        long totalSeqLen = currentSeqLen + draftPastSeqLen;

        // Build decoder input map
        List<INDArray> kvViewDups = new ArrayList<>();
        Map<String, INDArray> inputMap = new HashMap<>();
        for (String inputName : inputNames) {
            if (ioConfig.isInputEmbeddings(inputName)) {
                inputMap.put(inputName, embeddings);
            } else if (ioConfig.isInputIds(inputName)) {
                inputMap.put(inputName, Nd4j.createFromArray(new long[]{tokenId}).reshape(1, 1));
            } else if (ioConfig.isAttentionMask(inputName)) {
                inputMap.put(inputName, Nd4j.ones(DataType.LONG, 1, totalSeqLen));
            } else if (ioConfig.isCausalMask(inputName)) {
                inputMap.put(inputName, ModelIOConfig.buildCausalMask(1, currentSeqLen, totalSeqLen));
            } else if (ioConfig.isPositionIds(inputName)) {
                inputMap.put(inputName, Nd4j.createFromArray(new long[]{draftPastSeqLen}).reshape(1, 1));
            } else if (inputName.startsWith(ioConfig.getKvCachePrefix())) {
                if (draftPastSeqLen > 0) {
                    // Extract contiguous [0:draftPastSeqLen] from static buffer.
                    // dup() required because the view's strides are non-contiguous
                    // and attention kernels expect contiguous KV input.
                    INDArray fullBuf = staticDraftKvBuffers.get(inputName);
                    INDArray viewDup = fullBuf.get(
                            NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.interval(0, draftPastSeqLen), NDArrayIndex.all()).dup();
                    kvViewDups.add(viewDup);
                    inputMap.put(inputName, viewDup);
                } else {
                    inputMap.put(inputName, createEmptyKvCache(draftModel, inputName));
                }
            }
        }

        // Build output names: logits + KV cache
        List<String> outputNames = new ArrayList<>();
        outputNames.add(ioConfig.getLogitsOutputName());
        ModelIOConfig.KVCacheNames kvNames = ioConfig.getKvCacheNames();
        outputNames.addAll(kvNames.keyNames);
        outputNames.addAll(kvNames.valueNames);

        // Run draft model (InferenceSession dups outputs)
        Map<String, INDArray> outputs = draftModel.output(inputMap, outputNames.toArray(new String[0]));

        // Extract logits and decode
        INDArray logits = outputs.get(ioConfig.getLogitsOutputName());
        int nextToken = decodeFunction.apply(logits);

        // Scatter the new KV entry from present outputs into static buffers at draftPastSeqLen.
        // Present KV shape: [batch, heads, draftPastSeqLen+1, dim] (past concat new).
        // The new entry is at the last position (index draftPastSeqLen).
        scatterKvEntryAtPosition(staticDraftKvBuffers, outputs, kvNames, draftPastSeqLen, (int) draftPastSeqLen);

        // Close duped output arrays from InferenceSession
        for (INDArray arr : outputs.values()) {
            SameDiffMemoryUtils.safeClose(arr);
        }

        // Close input arrays created for this step (not static KV buffers)
        for (Map.Entry<String, INDArray> entry : inputMap.entrySet()) {
            if (!entry.getKey().startsWith(ioConfig.getKvCachePrefix())) {
                SameDiffMemoryUtils.safeClose(entry.getValue());
            }
        }

        // Close contiguous KV view dups used as inputs
        for (INDArray dup : kvViewDups) {
            SameDiffMemoryUtils.safeClose(dup);
        }

        draftPastSeqLen++;

        draftModel.clearPlaceholders(false);

        return nextToken;
    }

    /**
     * Save a zero-copy checkpoint of the current draft KV cache position.
     * Always saves, even with 0 acceptance -- needed so rollback can reset
     * draftPastSeqLen to discard rejected speculation positions.
     */
    private void saveDraftCheckpoint(int acceptedCount) {
        if (activeCheckpointId >= 0) {
            specKvManager.discardCheckpoint(activeCheckpointId);
            activeCheckpointId = -1;
        }

        if (staticKvInitialized) {
            long verifiedPos = checkpointPastSeqLen + acceptedCount;
            activeCheckpointId = specKvManager.checkpoint(staticDraftKvBuffers, (int) verifiedPos);
            checkpointPastSeqLen = verifiedPos;
        }
    }

    /**
     * Roll back the draft model's state to the last checkpoint.
     * Zero-copy: only resets the position pointer.
     */
    private void rollbackToCheckpoint() {
        if (activeCheckpointId >= 0) {
            int pos = specKvManager.rollback(activeCheckpointId);
            draftPastSeqLen = pos;
            activeCheckpointId = -1;
        }
    }

    private static int argmax(INDArray logits, int position) {
        INDArray posLogits;
        if (logits.rank() == 3) {
            posLogits = logits.get(NDArrayIndex.point(0), NDArrayIndex.point(position), NDArrayIndex.all());
        } else if (logits.rank() == 2) {
            posLogits = logits.get(NDArrayIndex.point(position), NDArrayIndex.all());
        } else {
            posLogits = logits;
        }
        return SamplerUtils.argmax(posLogits);
    }

    /**
     * Detect which device the draft model's constants reside on.
     * Cached after first call.
     */
    private int getDraftModelDevice() {
        if (draftModelDevice >= 0) return draftModelDevice;
        for (var v : draftModel.variables()) {
            INDArray arr = draftModel.getArrForVarName(v.name());
            if (arr != null && arr.data() != null) {
                draftModelDevice = arr.data().targetDevice();
                return draftModelDevice;
            }
        }
        return -1;
    }

    /**
     * Create an empty KV cache for a given input name, inferring shape from the model.
     * Delegates to ModelIOConfig.createEmptyKvCache with batch=1 and this model's hiddenSize.
     */
    private INDArray createEmptyKvCache(SameDiff decoder, String inputName) {
        return ModelIOConfig.createEmptyKvCache(decoder, inputName, 1, hiddenSize);
    }

    /**
     * Scatter the new KV entry from present outputs into static KV buffers at targetPos.
     * The new entry is extracted from sourcePosition in the present KV output.
     */
    private void scatterKvEntryAtPosition(Map<String, INDArray> staticBufs,
                                           Map<String, INDArray> outputs,
                                           ModelIOConfig.KVCacheNames kvNames,
                                           long targetPos, int sourcePosition) {
        List<String> allPresentNames = new ArrayList<>();
        allPresentNames.addAll(kvNames.keyNames);
        allPresentNames.addAll(kvNames.valueNames);

        for (String presentName : allPresentNames) {
            INDArray presentKv = outputs.get(presentName);
            if (presentKv == null) continue;

            String pastInputName = ioConfig.presentToInputName(presentName);
            INDArray staticBuf = staticBufs.get(pastInputName);
            if (staticBuf == null) continue;

            INDArray sourceSlice = presentKv.get(
                    NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(sourcePosition), NDArrayIndex.all());
            INDArray destSlice = staticBuf.get(
                    NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(targetPos), NDArrayIndex.all());
            destSlice.assign(sourceSlice);
        }
    }
}
