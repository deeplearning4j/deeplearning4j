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
import org.eclipse.deeplearning4j.llm.generation.kvcache.KvCacheManager;
import org.eclipse.deeplearning4j.llm.generation.kvcache.UnifiedKvCacheManager;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspCompilationMode;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Frozen decode step for speculative decoding with CUDA graph replay.
 *
 * <p>Encapsulates a single merged forward pass with fixed seqLen = K+1,
 * where K is the maximum number of speculative tokens. Every step processes
 * K+1 tokens (1 greedy + K speculative) with static shapes, enabling
 * CUDA graph capture and replay.</p>
 *
 * <p>Pre-allocates fixed-address GPU buffers for all inputs. Data is copied
 * into these buffers each step via assign(), preserving the pointer addresses
 * that CUDA graphs require.</p>
 *
 * <p>Requires a {@link KvCacheManager} that supports CUDA graph replay (i.e.,
 * {@link UnifiedKvCacheManager} with STATIC strategy). Paged or quantized strategies with dynamic shapes
 * are not compatible with CUDA graph capture.</p>
 *
 * <p>Usage:</p>
 * <pre>{@code
 * ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
 * FrozenDecodeStep step = new FrozenDecodeStep(decoder, 6, maxKvLen, hiddenSize, ioConfig);
 * step.setKvCacheManager(kvCacheManager);  // must support CUDA graph replay
 * step.compile(DspCompilationMode.MAX_AUTOTUNE);
 * INDArray logits = step.execute(embeddings, inputIds, cachePos);
 * // logits shape: [1, 6, vocabSize]
 * step.close();
 * }</pre>
 */
@Slf4j
public class FrozenDecodeStep implements AutoCloseable {

    private final SameDiff decoder;
    private final int seqLen;           // K+1 (fixed)
    private final long maxKvLen;
    private final long hiddenSize;
    private final String logitsOutputName;
    private final ModelIOConfig.KVCacheNames kvNames;
    private final List<String> decoderInputNames;
    private final ModelIOConfig ioConfig;

    // Pre-allocated fixed-address buffers
    private INDArray reusableEmbeddings;    // [1, seqLen, hidden]
    private INDArray reusableInputIds;      // [1, seqLen]
    private INDArray reusableAttentionMask; // [1, maxKvLen + seqLen]
    private INDArray reusableAttnBias;      // [1, 1, seqLen, maxKvLen + seqLen] — direct 4D attention bias
    private INDArray reusablePositionIds;   // [1, seqLen]

    // KV cache manager (shared with caller, NOT owned)
    private KvCacheManager kvCacheManager;

    // DSP state
    private DynamicShapePlanExecutor dspExec;
    private boolean compiled;

    // Output names
    private String[] fullOutputNames;

    // Step counter for logging
    private int stepCount;

    /**
     * Create a frozen decode step with explicit I/O configuration.
     *
     * @param decoder the SameDiff decoder model
     * @param seqLen fixed sequence length (K+1)
     * @param maxKvLen maximum KV cache length
     * @param hiddenSize model hidden dimension
     * @param ioConfig model I/O configuration with variable name mappings
     */
    public FrozenDecodeStep(SameDiff decoder, int seqLen, long maxKvLen, long hiddenSize,
                            ModelIOConfig ioConfig) {
        this.decoder = decoder;
        this.seqLen = seqLen;
        this.maxKvLen = maxKvLen;
        this.hiddenSize = hiddenSize;
        this.ioConfig = ioConfig;
        this.logitsOutputName = ioConfig.getLogitsOutputName();
        this.kvNames = ioConfig.getKvCacheNames() != null
                ? ioConfig.getKvCacheNames()
                : ModelIOConfig.findKVCacheOutputNames(decoder);
        this.decoderInputNames = decoder.inputs();
        this.compiled = false;
        this.stepCount = 0;

        // Build output name lists — every step requests logits + present KV so the
        // KvScatter op can copy entries into the Java-owned static KV buffers.
        List<String> fullList = new ArrayList<>();
        fullList.add(logitsOutputName);
        fullList.addAll(kvNames.keyNames);
        fullList.addAll(kvNames.valueNames);
        this.fullOutputNames = fullList.toArray(new String[0]);

        // Pre-allocate fixed-address buffers
        allocateBuffers();

        log.info("FrozenDecodeStep created: seqLen={}, maxKvLen={}, hiddenSize={}", seqLen, maxKvLen, hiddenSize);
    }

    /**
     * Create a frozen decode step (legacy constructor with explicit names).
     *
     * @param decoder the SameDiff decoder model
     * @param seqLen fixed sequence length (K+1)
     * @param maxKvLen maximum KV cache length
     * @param hiddenSize model hidden dimension
     * @param logitsOutputName name of the logits output variable
     * @param kvNames KV cache output names
     */
    public FrozenDecodeStep(SameDiff decoder, int seqLen, long maxKvLen, long hiddenSize,
                            String logitsOutputName, ModelIOConfig.KVCacheNames kvNames) {
        this(decoder, seqLen, maxKvLen, hiddenSize,
                ModelIOConfig.builder()
                        .logitsOutputName(logitsOutputName)
                        .kvCacheNames(kvNames)
                        .build());
    }

    private void allocateBuffers() {
        long totalMaskLen = maxKvLen + seqLen;

        // Detect the target model's device from its first constant array.
        // This ensures our reusable buffers co-locate with the model weights,
        // avoiding cross-device migration in ensureDeviceCoherency.
        int modelDevice = detectModelDevice();
        int callerDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        try {
            if (modelDevice >= 0 && modelDevice != callerDevice) {
                DeviceMemoryManager.getInstance().switchDevice(modelDevice,
                        "FrozenDecodeStep", "allocateBuffers-colocate-with-model");
            }

            reusableEmbeddings = Nd4j.zeros(DataType.FLOAT, 1, seqLen, hiddenSize);
            reusableInputIds = Nd4j.zeros(DataType.LONG, 1, seqLen);
            reusableAttentionMask = Nd4j.zeros(DataType.LONG, 1, totalMaskLen);
            // Direct 4D attention bias: [1, 1, seqLen, maxKvLen + seqLen]
            // Bypasses the model's attn_mask_reformat subgraph via placeholder override.
            // The attn_mask_reformat subgraph can't handle seqLen>1 with past KV because
            // it creates [seqLen, seqLen] from input_ids and [seqLen, totalSeqLen] from
            // attention_mask, then tries to multiply them — dimensions don't broadcast.
            // We override the subgraph output and directly provide the correct 4D mask.
            reusableAttnBias = Nd4j.zeros(DataType.FLOAT, 1, 1, seqLen, totalMaskLen);
            reusablePositionIds = Nd4j.zeros(DataType.LONG, 1, seqLen);

            log.info("  Allocated fixed-address buffers on device {}: embeds=[1,{},{}] mask=[1,{}] attnBias=[1,1,{},{}] posIds=[1,{}]",
                    modelDevice, seqLen, hiddenSize, totalMaskLen, seqLen, totalMaskLen, seqLen);
        } finally {
            if (modelDevice >= 0 && modelDevice != callerDevice) {
                DeviceMemoryManager.getInstance().switchDevice(callerDevice,
                        "FrozenDecodeStep", "allocateBuffers-restore-caller-device");
            }
        }
    }

    /**
     * Detect the CUDA device where the target model's constants are stored.
     * Returns the device index, or -1 if unknown.
     */
    private int detectModelDevice() {
        for (var v : decoder.variables()) {
            INDArray arr = decoder.getArrForVarName(v.name());
            if (arr != null && arr.data() != null) {
                return arr.data().targetDevice();
            }
        }
        return -1;
    }

    /**
     * Set the KV cache manager that provides static KV buffers.
     *
     * <p>The KvCacheManager must support CUDA graph replay (i.e., return true from
     * {@link KvCacheManager#supportsCudaGraphReplay()}). FrozenDecodeStep requires
     * fixed-shape KV buffers for CUDA graph capture — paged or quantized strategies
     * with dynamic shapes are not compatible.</p>
     *
     * <p>The manager is NOT owned by this step — the caller manages its lifecycle.</p>
     *
     * @param kvCacheManager the KV cache manager (must support CUDA graph replay)
     * @throws IllegalArgumentException if the manager does not support CUDA graph replay
     */
    public void setKvCacheManager(KvCacheManager kvCacheManager) {
        if (!kvCacheManager.supportsCudaGraphReplay()) {
            throw new IllegalArgumentException(
                    "FrozenDecodeStep requires a KvCacheManager that supports CUDA graph replay. " +
                    "Got strategy: " + kvCacheManager.getStrategy());
        }
        this.kvCacheManager = kvCacheManager;
    }

    /**
     * Compile DSP, freeze shapes, and prepare for CUDA graph capture.
     *
     * Must be called after {@link #setKvCacheManager(KvCacheManager)} and before first execute().
     */
    public void compile(DspCompilationMode mode) {
        if (kvCacheManager == null) {
            throw new IllegalStateException("Must call setKvCacheManager() before compile()");
        }
        Map<String, INDArray> staticKvBuffers = kvCacheManager.getStaticKvBuffers();
        if (staticKvBuffers == null || staticKvBuffers.isEmpty()) {
            throw new IllegalStateException("KvCacheManager returned null/empty static KV buffers");
        }

        // Override attn_mask_reformat output — bypass the subgraph that can't
        // handle seqLen>1 with past KV. This converts the Tile output from
        // OP_OUTPUT to PLACEHOLDER, pruning the entire attn_mask_reformat subgraph.
        String attnReformatNode = ioConfig.getAttnMaskReformatOutput();
        if (seqLen > 1 && attnReformatNode != null && decoder.hasVariable(attnReformatNode)) {
            decoder.addPlaceholderOverride(attnReformatNode);
            // Set the placeholder shape so associateArrayWithVariable validation passes.
            // Use -1 for dynamic dims (batch, seqLen, totalSeqLen). Only numHeads=1 is fixed.
            decoder.getVariable(attnReformatNode).setShape(-1, -1, -1, -1);
            log.info("  Added placeholder override for {} (bypassing attn_mask_reformat subgraph)", attnReformatNode);
        }

        // Associate static KV buffers so DSP sees correct shapes
        for (Map.Entry<String, INDArray> e : staticKvBuffers.entrySet()) {
            String pastName = e.getKey();
            if (decoder.hasVariable(pastName)) {
                decoder.associateArrayWithVariable(e.getValue(), pastName);
            }
        }

        // Build a probe input map so DSP sees the correct seqLen shapes
        // This ensures Triton kernels compile with seqQ = seqLen (not 1)
        updateInputs(reusableEmbeddings, reusableInputIds, 0);
        Map<String, INDArray> probeInputMap = buildInputMap();
        for (Map.Entry<String, INDArray> entry : probeInputMap.entrySet()) {
            if (decoder.hasVariable(entry.getKey())) {
                decoder.associateArrayWithVariable(entry.getValue(), entry.getKey());
            }
        }

        // Clear caches and compile fresh. The merged step always requests logits +
        // present KV — KvScatter runs as an ordinary op against the static KV
        // buffers and there is no native KV retention configuration.
        String[] compileOutputs = fullOutputNames;
        decoder.clearDynamicShapePlanCache();
        InferenceSession session = decoder.getOrCreateSession();
        session.clearAllCaches();

        log.info("  Compiling DSP for merged step (seqLen={}, maxKvLen={}, outputs={})",
                seqLen, maxKvLen, Arrays.toString(compileOutputs));
        decoder.compileNativeDynamicShapePlan(mode, compileOutputs);

        // Re-fetch executor and freeze
        session = decoder.getOrCreateSession();
        dspExec = session.getDynamicShapePlanExecutor();

        boolean skipFreeze = "true".equalsIgnoreCase(System.getProperty("nd4j.dsp.nofreeze"));
        if (dspExec != null && !skipFreeze) {
            dspExec.setShapesFrozen(true);
            dspExec.setTraceEnabled(true);
            dspExec.setExecutionTimingEnabled(true);
            log.info("  Shapes frozen for merged decode step (seqLen={})", seqLen);
        }

        compiled = true;

        // Dump model summary to inspect graph structure for seqLen>1 edge cases
        if ("true".equalsIgnoreCase(System.getProperty("nd4j.frozen.summary"))) {
            log.info("=== FrozenDecodeStep Graph Summary (seqLen={}) ===\n{}", seqLen, decoder.summary());
        }

        log.info("  FrozenDecodeStep compiled and ready");
    }

    /**
     * Execute one merged decode step.
     *
     * @param embeddings [1, seqLen, hidden] — K+1 token embeddings
     * @param inputIds [1, seqLen] — K+1 token IDs
     * @param cachePos current write position in the static KV cache
     * @return logits [1, seqLen, vocabSize]
     */
    public INDArray execute(INDArray embeddings, INDArray inputIds, long cachePos) {
        if (!compiled) {
            throw new IllegalStateException("Must call compile() before execute()");
        }

        long stepStart = System.nanoTime();
        stepCount++;

        // 1. Copy data into fixed-address buffers
        updateInputs(embeddings, inputIds, cachePos);

        // 2. Build input map from fixed-address buffers
        Map<String, INDArray> inputMap = buildInputMap();

        long tAfterInput = System.nanoTime();

        // 3. Run decoder via direct path (preserves DSP plan / CUDA graph across steps)
        // MUST use outputDirect for ALL steps — decoder.output() goes through
        // directExecHelper which clears session caches, destroying the frozen plan.
        //
        // Enable debug+verbose for first execution (warmup) to trace all op shapes
        boolean enableWarmupDebug = stepCount == 1 &&
                "true".equalsIgnoreCase(System.getProperty("nd4j.frozen.debug"));
        if (enableWarmupDebug) {
            log.info("  [FrozenStep] Enabling debug+verbose for warmup execution (step 1)");
            Nd4j.getEnvironment().setDebug(true);
            Nd4j.getEnvironment().setVerbose(true);
        }
        Map<String, INDArray> outputs = decoder.outputDirect(inputMap, fullOutputNames);
        if (enableWarmupDebug) {
            Nd4j.getEnvironment().setDebug(false);
            Nd4j.getEnvironment().setVerbose(false);
            log.info("  [FrozenStep] Disabled debug+verbose after warmup");
        }

        long tAfterDecoder = System.nanoTime();

        // 4. Extract logits — DO NOT dup or close!
        // In DSP mode, logits are views sharing data buffers with the slot cache.
        // The caller must NOT close them — the slot cache manages buffer lifetime.
        // Data is valid until the next execute() call overwrites the slot.
        INDArray logitsRaw = outputs.get(logitsOutputName);
        if (logitsRaw != null && logitsRaw.dataType() == DataType.HALF) {
            logitsRaw = logitsRaw.castTo(DataType.FLOAT); // castTo creates new array (safe)
        }

        // 5. KV scatter is NOT run here. Present KV from the merged step has shape
        // [1, heads, maxKvLen+seqLen, dim] with new entries at positions
        // [maxKvLen..maxKvLen+seqLen-1]. The caller decides how many speculative
        // tokens were accepted and invokes scatterAcceptedKv() to run KvScatter
        // against the Java-owned static KV buffers at [cachePos..cachePos+numAccepted-1].

        long tAfterKv = System.nanoTime();

        // Log timing
        if (stepCount <= 5 || stepCount % 10 == 0) {
            long inputMs = (tAfterInput - stepStart) / 1_000_000;
            long decMs = (tAfterDecoder - tAfterInput) / 1_000_000;
            long totalMs = (tAfterKv - stepStart) / 1_000_000;
            log.info("  [FrozenStep] step={} total={}ms [input={}ms dec={}ms] cachePos={}",
                    stepCount, totalMs, inputMs, decMs, cachePos);
        }

        // Clean up per-step inputs (skip fixed-address buffers and static KV)
        decoder.clearPlaceholders(false);

        // Store outputs for KV scatter by caller
        lastOutputs = outputs;

        return logitsRaw;
    }

    // Last decoder outputs — held for caller to scatter accepted KV entries
    private Map<String, INDArray> lastOutputs;

    /**
     * Get the last decoder outputs (for KV scatter by caller).
     */
    public Map<String, INDArray> getLastOutputs() {
        return lastOutputs;
    }

    /**
     * Get the DSP executor for this frozen step.
     * Used by callers to sync KV cache position for plan recompilation safety.
     */
    public DynamicShapePlanExecutor getDspExec() {
        return dspExec;
    }

    /**
     * Scatter accepted KV entries from the last merged step into static buffers.
     *
     * After verification determines how many tokens were accepted, scatter those
     * positions from the present KV outputs into the static KV cache.
     *
     * @param baseCachePos starting target position in static KV buffer
     * @param numAccepted number of positions to scatter (accepted + 1 for correction)
     */
    public void scatterAcceptedKv(long baseCachePos, int numAccepted) {
        if (lastOutputs == null || kvCacheManager == null) return;

        // Set cache position so scatterMultipleEntries writes at the right offset
        kvCacheManager.setCachePosition(baseCachePos);
        kvCacheManager.scatterMultipleEntries(lastOutputs, kvNames, numAccepted);

        // Close present KV outputs after scatter
        closeLastKvOutputs();
    }

    /**
     * Release references to present KV outputs from the last step.
     *
     * <p>In DSP mode, present KV outputs are views that share data buffers with
     * the slot cache. Closing them would destroy the underlying buffers, causing
     * "DataBuffer released via close()" errors on the next execution step.
     * Instead, just release the references — the slot cache manages buffer lifetime.</p>
     */
    private void closeLastKvOutputs() {
        if (lastOutputs == null) return;

        // Don't close — these are views sharing buffers with the DSP slot cache.
        // Just clear the references so they can be GC'd if not held elsewhere.
        for (String pn : kvNames.keyNames) {
            lastOutputs.remove(pn);
        }
        for (String pn : kvNames.valueNames) {
            lastOutputs.remove(pn);
        }
    }

    /**
     * Update fixed-address input buffers with new data.
     */
    private void updateInputs(INDArray embeddings, INDArray inputIds, long cachePos) {
        // Embeddings: [1, seqLen, hidden]
        reusableEmbeddings.assign(embeddings);

        // Input IDs: [1, seqLen]
        reusableInputIds.assign(inputIds);

        // Attention mask: [1, maxKvLen + seqLen]
        // 1s at positions 0..cachePos-1 (past) + last seqLen positions (current tokens)
        reusableAttentionMask.assign(0);
        if (cachePos > 0) {
            reusableAttentionMask.get(
                    NDArrayIndex.point(0),
                    NDArrayIndex.interval(0, cachePos)).assign(1);
        }
        // Current token positions at the end (the concat'd positions)
        long maskLen = maxKvLen + seqLen;
        reusableAttentionMask.get(
                NDArrayIndex.point(0),
                NDArrayIndex.interval(maskLen - seqLen, maskLen)).assign(1);

        // Position IDs: [1, seqLen] — consecutive from cachePos
        if (seqLen == 1) {
            reusablePositionIds.putScalar(0, 0, cachePos);
        } else {
            // Bulk-set via arange to avoid O(seqLen) JNI calls during prefill
            INDArray posRange = Nd4j.arange(cachePos, cachePos + seqLen).castTo(DataType.LONG).reshape(1, seqLen);
            reusablePositionIds.assign(posRange);
            posRange.close();
        }

        // 4D attention bias: [1, 1, seqLen, maxKvLen + seqLen]
        // Directly provides the attention mask to onnx_multi_head_attention,
        // bypassing the attn_mask_reformat subgraph.
        // Layout: past positions [0..maxKvLen-1], current positions [maxKvLen..maxKvLen+seqLen-1]
        //   - Past [0..cachePos-1]: 0 (valid past, attend freely)
        //   - Past [cachePos..maxKvLen-1]: MASK_FILL (empty past padding, block)
        //   - Current [maxKvLen+0..maxKvLen+q] for query q: 0 (causal, attend)
        //   - Current [maxKvLen+q+1..maxKvLen+seqLen-1] for query q: MASK_FILL (future, block)
        int totalLen = (int)(maxKvLen + seqLen);
        float[] biasData = new float[seqLen * totalLen];
        for (int q = 0; q < seqLen; q++) {
            int rowOffset = q * totalLen;
            // Past positions: valid up to cachePos, masked after
            for (int k = (int)cachePos; k < (int)maxKvLen; k++) {
                biasData[rowOffset + k] = ModelIOConfig.MASK_FILL;
            }
            // Current positions: causal — future tokens masked
            for (int k = q + 1; k < seqLen; k++) {
                biasData[rowOffset + (int)maxKvLen + k] = ModelIOConfig.MASK_FILL;
            }
        }

        INDArray newBias = Nd4j.create(biasData, new long[]{1, 1, seqLen, totalLen}, 'c');
        reusableAttnBias.assign(newBias);
        newBias.close();
    }

    /**
     * Build the input map from fixed-address buffers.
     */
    private Map<String, INDArray> buildInputMap() {
        Map<String, INDArray> inputMap = new HashMap<>();

        String attnReformatNode = ioConfig.getAttnMaskReformatOutput();
        boolean overrideActive = seqLen > 1 && attnReformatNode != null
                && decoder.getPlaceholderOverrides().contains(attnReformatNode);

        for (String inputName : decoderInputNames) {
            if (ioConfig.isInputEmbeddings(inputName)) {
                inputMap.put(inputName, reusableEmbeddings);
            } else if (ioConfig.isAttentionMask(inputName)) {
                // Provide even when override is active — it's a registered placeholder
                // and the executor expects values for all placeholders in the arg table.
                inputMap.put(inputName, reusableAttentionMask);
            } else if (ioConfig.isCausalMask(inputName)) {
                // When override is active, causal mask is still a registered placeholder.
                // Provide the attn bias buffer (same shape [1,1,seqLen,totalLen]) to
                // keep the arg table populated. Nothing consumes it (subgraph pruned).
                inputMap.put(inputName, reusableAttnBias);
            } else if (ioConfig.isInputIds(inputName)) {
                inputMap.put(inputName, reusableInputIds);
            } else if (ioConfig.isPositionIds(inputName)) {
                inputMap.put(inputName, reusablePositionIds);
            } else if (ioConfig.isKvCacheInput(inputName)) {
                if (kvCacheManager != null) {
                    Map<String, INDArray> kvBuffers = kvCacheManager.getStaticKvBuffers();
                    if (kvBuffers != null) {
                        INDArray staticBuf = kvBuffers.get(inputName);
                        if (staticBuf != null) {
                            inputMap.put(inputName, staticBuf);
                        }
                    }
                }
            }
        }

        // When override is active, provide attn bias directly as the Tile output
        if (overrideActive) {
            inputMap.put(attnReformatNode, reusableAttnBias);
        }

        String embedsName = ioConfig.getInputEmbeddingsName();
        if (embedsName != null && !inputMap.containsKey(embedsName)) {
            inputMap.put(embedsName, reusableEmbeddings);
        }

        return inputMap;
    }

    /**
     * Get the fixed sequence length (K+1).
     */
    public int getSeqLen() {
        return seqLen;
    }

    /**
     * Get the maximum KV cache length.
     */
    public long getMaxKvLen() {
        return maxKvLen;
    }

    /**
     * Whether compilation and CUDA graph capture have been done.
     */
    public boolean isCompiled() {
        return compiled;
    }

    @Override
    public void close() {
        SameDiffMemoryUtils.safeClose(reusableEmbeddings);
        SameDiffMemoryUtils.safeClose(reusableInputIds);
        SameDiffMemoryUtils.safeClose(reusableAttentionMask);
        SameDiffMemoryUtils.safeClose(reusableAttnBias);
        SameDiffMemoryUtils.safeClose(reusablePositionIds);
        // Remove placeholder override so decoder can be reused normally
        String attnReformatNode = ioConfig.getAttnMaskReformatOutput();
        if (attnReformatNode != null && decoder.getPlaceholderOverrides().contains(attnReformatNode)) {
            decoder.removePlaceholderOverride(attnReformatNode);
        }
        // Note: kvCacheManager is NOT closed here — it's owned by the caller
        log.info("FrozenDecodeStep closed (seqLen={}, {} steps executed)", seqLen, stepCount);
    }
}
