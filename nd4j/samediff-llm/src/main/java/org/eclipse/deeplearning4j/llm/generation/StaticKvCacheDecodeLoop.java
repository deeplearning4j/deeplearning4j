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

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DspCompilationMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.TokenSample;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import org.bytedeco.javacpp.LongPointer;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueDataBuffer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Autoregressive decode loop with static KV cache for CUDA graph capture.
 *
 * <p>Encapsulates the full decode pipeline: prefill with dynamic KV, transition to
 * fixed-shape static KV buffers, frozen shapes for CUDA graph replay, GPU-side
 * token sampling via {@link TokenSample}, and per-step KV scatter.</p>
 *
 * <p>Usage:</p>
 * <pre>{@code
 * StaticKvCacheDecodeLoop loop = StaticKvCacheDecodeLoop.builder()
 *     .decoder(decoder)
 *     .embedTokens(embedTokens)
 *     .tokenizer(tokenizer)
 *     .maxNewTokens(256)
 *     .hiddenSize(hiddenSize)
 *     .build();
 * GenerationResult result = loop.decode(prefillEmbeddings, promptTokenIds);
 * }</pre>
 */
@Slf4j
@Builder
public class StaticKvCacheDecodeLoop {

    /**
     * Reuses the token_sample op and output buffer across decode steps.
     * The per-step host readback still exists, but this removes the repeated
     * DynamicCustomOp construction and output allocation from the hot path.
     */
    private static final class ReusableTokenSampler {
        private final SamplingConfig samplingConfig;
        private TokenSample tokenSampleOp;
        private INDArray tokenOutput;
        private long outputBatchSize = -1;

        private ReusableTokenSampler(SamplingConfig samplingConfig) {
            this.samplingConfig = samplingConfig;
        }

        int sample(INDArray logits) {
            long batchSize = logits.rank() == 3 ? logits.size(0)
                    : (logits.rank() == 2 ? logits.size(0) : 1);

            if (tokenOutput == null || outputBatchSize != batchSize || tokenOutput.wasClosed()) {
                if (tokenOutput != null && !tokenOutput.wasClosed()) {
                    tokenOutput.setCloseable(true);
                    tokenOutput.close();
                }
                tokenOutput = Nd4j.createUninitialized(DataType.INT64, batchSize);
                outputBatchSize = batchSize;
            }

            if (tokenSampleOp == null) {
                if (samplingConfig.isGreedy()) {
                    tokenSampleOp = new TokenSample(logits);
                } else {
                    tokenSampleOp = new TokenSample(logits,
                            samplingConfig.getTemperature(),
                            samplingConfig.getTopK(),
                            samplingConfig.getTopP(),
                            samplingConfig.getSeed() != null ? samplingConfig.getSeed() : 0L);
                }
            } else {
                tokenSampleOp.setInputArgument(0, logits);
            }
            tokenSampleOp.setOutputArgument(0, tokenOutput);

            // Clear stale native error state from prior DSP graph-capture attempts.
            Nd4j.getNativeOps().clearLastError();
            Nd4j.getExecutioner().exec(tokenSampleOp);
            return tokenOutput.getInt(0);
        }

        void close() {
            if (tokenOutput != null && !tokenOutput.wasClosed()) {
                tokenOutput.setCloseable(true);
                tokenOutput.close();
            }
        }
    }

    private final SameDiff decoder;
    private final SameDiff embedTokens;
    private final Tokenizer tokenizer;

    @Builder.Default
    private final SamplingConfig samplingConfig = SamplingConfig.greedy();
    @Builder.Default
    private final int maxNewTokens = 256;
    @Builder.Default
    private final long hiddenSize = 0;

    @Builder.Default
    private final int maxSpeculativeTokens = 0;

    /** KV cache strategy to use. Defaults to STATIC for backward compatibility. */
    @Builder.Default
    private final KvCacheStrategy kvCacheStrategy = KvCacheStrategy.STATIC;

    /** Quantization format for QUANTIZED KV cache strategy. Defaults to INT8. */
    @Builder.Default
    private final QuantizedPagedKVCache.QuantFormat quantFormat = QuantizedPagedKVCache.QuantFormat.INT8;

    /** TurboQuant bit budget per coordinate for TURBOQUANT strategy. Defaults to 3. */
    @Builder.Default
    private final int turboQuantBits = 3;

    /** Optional speculator for draft-model speculation. When null, uses NgramSpeculator. */
    private final Speculator speculator;

    private final String embedInputName;
    private final String[] embedOutputNames;
    private final Set<Integer> additionalStopTokenIds;

    /** Model I/O configuration for variable name resolution. Auto-discovered if not provided. */
    private final ModelIOConfig ioConfig;

    /** Optional encoder model for encoder-decoder architectures (e.g., Whisper). */
    private final SameDiff encoder;

    /** Pre-computed encoder outputs. If null and encoder is set, encoder runs once at decode start. */
    private final INDArray encoderOutputs;

    /** Whether this is an encoder-decoder model. When true, encoder outputs are fed to decoder at each step. */
    @Builder.Default
    private final boolean encoderDecoder = false;

    /**
     * Create the appropriate KvCacheManager based on the configured strategy.
     *
     * <p>Currently supports STATIC strategy (the only one with a complete KvCacheManager
     * implementation). PAGED and QUANTIZED strategies have the underlying cache implementations
     * ({@link PagedKVCache}, {@link QuantizedPagedKVCache}) but no KvCacheManager wrappers yet.</p>
     *
     * @return a new KvCacheManager instance
     */
    private KvCacheManager createKvCacheManager(ModelIOConfig resolvedIOConfig) {
        switch (kvCacheStrategy) {
            case STATIC:
                return new StaticKvCacheManager(resolvedIOConfig);
            case PAGED:
                // PagedKVCache exists but has no KvCacheManager wrapper yet.
                // Fall back to static for now and log the gap.
                log.warn("PAGED KV cache strategy requested but PagedKvCacheManager not yet implemented. "
                        + "Falling back to STATIC. PagedKVCache is available for direct use.");
                return new StaticKvCacheManager(resolvedIOConfig);
            case QUANTIZED:
                log.info("Creating QuantizedKvCacheManager with format={}", quantFormat);
                return new QuantizedKvCacheManager(quantFormat, DataType.FLOAT);
            case TURBOQUANT:
                log.info("Creating TurboQuantKvCacheManager with bits={}", turboQuantBits);
                return new TurboQuantKvCacheManager(turboQuantBits, resolvedIOConfig);
            default:
                throw new IllegalStateException("Unknown KV cache strategy: " + kvCacheStrategy);
        }
    }

    /**
     * Run the autoregressive decode loop.
     *
     * @param prefillEmbeddings merged embeddings for the full prompt [1, seqLen, hidden]
     * @param promptTokenIds the prompt token IDs (used for input_ids at step 0)
     * @return generation result with text, token IDs, and timing
     */
    public GenerationResult decode(INDArray prefillEmbeddings, int[] promptTokenIds) {
        long decodeStart = System.currentTimeMillis();

        // Resolve I/O names via ModelIOConfig (auto-discover if not provided)
        ModelIOConfig resolvedIOConfig = ioConfig != null ? ioConfig : ModelIOConfig.discover(decoder);
        String logitsOutputName = resolvedIOConfig.getLogitsOutputName();
        DecoderUtils.KVCacheNames kvNames = resolvedIOConfig.getKvCacheNames() != null
                ? resolvedIOConfig.getKvCacheNames()
                : DecoderUtils.findKVCacheOutputNames(decoder);
        List<String> decoderInputNames = decoder.inputs();
        log.info("  Decoder input names: {}", decoderInputNames);

        String resolvedEmbedInputName = embedInputName;
        if (resolvedEmbedInputName == null) {
            resolvedEmbedInputName = embedTokens.inputs().isEmpty()
                    ? resolvedIOConfig.getInputIdsName()
                    : embedTokens.inputs().get(0);
        }
        String[] resolvedEmbedOutputNames = embedOutputNames;
        if (resolvedEmbedOutputNames == null) {
            resolvedEmbedOutputNames = embedTokens.outputs().toArray(new String[0]);
        }

        // Resolve stop tokens
        int eosTokenId = tokenizer.getEosTokenId();
        Set<Integer> stopTokenIds = new HashSet<>();
        stopTokenIds.add(eosTokenId);
        Integer endOfUtteranceTokenId = tokenizer.getTokenId("<end_of_utterance>");
        if (endOfUtteranceTokenId != null) {
            stopTokenIds.add(endOfUtteranceTokenId);
        }
        if (additionalStopTokenIds != null) {
            stopTokenIds.addAll(additionalStopTokenIds);
        }

        // Resolve hidden size
        long resolvedHiddenSize = hiddenSize;
        if (resolvedHiddenSize <= 0) {
            resolvedHiddenSize = prefillEmbeddings.shape()[2];
        }

        // Extract embedding weight table for direct lookup (avoids full SameDiff.output() per token).
        // The weight may be stored as CONSTANT or VARIABLE depending on the ONNX import path.
        INDArray embeddingTable = null;
        for (SDVariable var : embedTokens.variables()) {
            if (var.getVariableType() == VariableType.CONSTANT || var.getVariableType() == VariableType.VARIABLE) {
                INDArray arr = var.getArr();
                if (arr != null && arr.rank() == 2) {
                    if (embeddingTable == null || arr.length() > embeddingTable.length()) {
                        embeddingTable = arr;
                    }
                }
            }
        }
        if (embeddingTable != null) {
            log.info("  Using direct embedding lookup: shape={} (bypasses SameDiff.output() per token)",
                    Arrays.toString(embeddingTable.shape()));
        } else {
            log.warn("  Could not extract embedding table, falling back to SameDiff.output() per token");
        }

        // Prefill needs logits + present KV outputs. Once native KV retention is enabled,
        // decode can request logits only while C++ scatters present KV internally.
        List<String> fullOutputNameList = new ArrayList<>();
        fullOutputNameList.add(logitsOutputName);
        fullOutputNameList.addAll(kvNames.keyNames);
        fullOutputNameList.addAll(kvNames.valueNames);
        String[] fullOutputNames = fullOutputNameList.toArray(new String[0]);
        String[] logitsOnlyOutputNames = new String[]{logitsOutputName};

        // State
        List<Integer> generatedTokens = new ArrayList<>();
        INDArray currentEmbeddings = prefillEmbeddings;
        INDArray currentInputIds = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, promptTokenIds.length).castTo(DataType.LONG);
        long pastSeqLen = 0;

        // Per-step timing
        List<Long> stepTimesMs = new ArrayList<>();
        long prefillTimeMs = 0;

        // Phase timing accumulators (steps 3+)
        long totalInputBuildNs = 0, totalDecoderNs = 0, totalLogitsDupNs = 0;
        long totalKvUpdateNs = 0, totalSamplingNs = 0;
        int detailSteps = 0;
        long lateSteadyTotalNs = 0;
        int lateSteadySteps = 0;

        // Encoder-decoder: run encoder once and store outputs for all decode steps
        INDArray resolvedEncoderOutputs = encoderOutputs;
        if (encoderDecoder && resolvedEncoderOutputs == null && encoder != null) {
            log.info("Running encoder for encoder-decoder model...");
            long encoderStart = System.currentTimeMillis();
            // Run encoder with prefill embeddings as input features
            Map<String, INDArray> encoderInputMap = new HashMap<>();
            // Auto-detect encoder input name
            List<String> encoderInputNames = encoder.inputs();
            if (!encoderInputNames.isEmpty()) {
                encoderInputMap.put(encoderInputNames.get(0), prefillEmbeddings);
            }
            // Run encoder and get the first output (hidden states)
            String[] encoderOutputNames = encoder.outputs().toArray(new String[0]);
            Map<String, INDArray> encoderResult = encoder.output(encoderInputMap, encoderOutputNames);
            resolvedEncoderOutputs = encoderResult.values().iterator().next();
            log.info("Encoder completed in {}ms, output shape: {}",
                    System.currentTimeMillis() - encoderStart,
                    java.util.Arrays.toString(resolvedEncoderOutputs.shape()));
        }
        final INDArray encoderOutputsForDecode = resolvedEncoderOutputs;

        // KV cache management — delegated to KvCacheManager
        KvCacheManager kvCacheManager = createKvCacheManager(resolvedIOConfig);
        boolean usingStaticKv = false;
        boolean kvScatterInCpp = false;  // When true, C++ handles KV scatter — skip Java side
        boolean skipFreeze = "true".equalsIgnoreCase(System.getProperty("nd4j.dsp.nofreeze"));
        DynamicShapePlanExecutor decoderDspExec = null;  // Tracked DSP executor for decode input updates



        // Reusable input arrays — avoids per-step allocation of masks/position_ids
        // Also used for CUDA graph replay: fixed-address buffers for inputs_embeds/input_ids
        // prevent address key mismatch that would invalidate captured graphs.
        Map<String, INDArray> reusableInputs = new HashMap<>();
        // Fixed-address decode buffers (allocated once, data copied each step)
        INDArray reusableEmbeddings = null;  // [1, 1, hiddenSize]
        INDArray reusableInputIds = null;    // [1, 1]
        ReusableTokenSampler reusableTokenSampler = new ReusableTokenSampler(samplingConfig);

        GenerationResult.FinishReason finishReason = GenerationResult.FinishReason.MAX_TOKENS;

        // Suppress cross-device routing for the ENTIRE decode loop.
        // Without this, ops that run outside InferenceSession (token_sample, embedding
        // lookup) can trigger cross-device migration when GPU memory is tight. Under
        // memory pressure (pool_used near GPU total), selectTargetDevice() sees <128MB
        // free via cudaMemGetInfo and routes to a second GPU. This creates replicas on
        // device 1, and the subsequent frozen DSP execution on device 0 encounters stale
        // CUDA context state → SIGSEGV. The CUDA async memory pool can reuse freed entries
        // that cudaMemGetInfo doesn't report as free, so the pressure is overstated.
        // Suppressing routing keeps all decode-loop ops on the model's home device.
        OpaqueDataBuffer.suppressCrossDeviceRouting(true);
        try {

        for (int step = 0; step < maxNewTokens; step++) {
            long stepStart = System.nanoTime();
            long currentSeqLen = currentEmbeddings.shape()[1];

            // Build input map (with reusable input cache for decode steps)
            // dspActive = padded mode with frozen shapes (enables native C++ input updates)
            boolean dspActive = usingStaticKv
                    && !"true".equalsIgnoreCase(System.getProperty("nd4j.dsp.noPadded"));
            boolean nativeDecodeInputs = decoderDspExec != null && decoderDspExec.isDecodeInputsConfigured()
                    && !"true".equalsIgnoreCase(System.getProperty("nd4j.dsp.noNativeDecodeInputs"));
            long maxKvLen = kvCacheManager.getMaxKvLen();
            long cachePos = kvCacheManager.getCachePosition();
            Map<String, INDArray> decoderInputMap = DecoderUtils.buildDecoderInputMap(
                    resolvedIOConfig, decoderInputNames, decoder,
                    currentEmbeddings, currentInputIds,
                    pastSeqLen, currentSeqLen,
                    kvCacheManager.getStaticKvBuffers(), maxKvLen, cachePos,
                    usingStaticKv, resolvedHiddenSize,
                    reusableInputs, dspActive, nativeDecodeInputs,
                    encoderOutputsForDecode, null);

            long tAfterInputBuild = System.nanoTime();

            // Tell C++ the next token and cache position for device-side input updates.
            // C++ will write input_ids, position_ids, and attention_mask directly on device
            // memory during execute() — no Java putScalar/assign host-device round-trips.
            if (nativeDecodeInputs && step >= 1) {
                long tokenId = currentInputIds.getLong(0, 0);
                decoderDspExec.setNextDecodeToken(tokenId, (int) cachePos);
            }

            // Diagnostic logging for steps 1-3
            if (step >= 1 && step <= 3) {
                logDiagnostics(step, decoderInputMap, resolvedIOConfig);
            }

            // Run decoder — use fast path when shapes are frozen (skips setCloseable overhead)
            Map<String, INDArray> decoderOutputs;
            // outputDirect skips dup() of output arrays — safe only when shapes are frozen
            // and DSP slot cache is stable (CUDA graph replay). In no-freeze mode, DSP
            // slot arrays may be invalidated by subsequent executions, so use output()
            // which dups all results into independent arrays.
            boolean useDirect = usingStaticKv && step >= 2 && !skipFreeze
                    && !"true".equalsIgnoreCase(System.getProperty("nd4j.dsp.noDirect"));
            // When C++ KV scatter is enabled (kvScatterInCpp), the native DSP plan
            // scatters present KV into static buffers on EVERY execution — regardless
            // of whether Java called output() or outputDirect(). So we MUST use
            // logitsOnlyOutputNames to avoid requesting KV outputs that would conflict
            // with C++ scatter. Java scatter is only used when C++ scatter is disabled.
            String[] requestedOutputNames = (usingStaticKv && kvScatterInCpp)
                    ? logitsOnlyOutputNames : fullOutputNames;
            if (useDirect) {
                decoderOutputs = decoder.outputDirect(
                        decoderInputMap, requestedOutputNames);
            } else {
                decoderOutputs = decoder.output(
                        decoderInputMap, requestedOutputNames);
            }

            long tAfterDecoder = System.nanoTime();

            // Diagnostic: present KV shapes at steps 1-3 (skip when C++ KV scatter manages outputs)
            if (step >= 1 && step <= 3 && !kvScatterInCpp) {
                logPresentKvDiagnostics(step, decoderOutputs, kvNames);
            }

            // Extract logits — keep raw reference, no dup needed
            // TokenSample now accepts rank-3 [batch, seqLen, vocabSize] directly
            INDArray logitsRaw = decoderOutputs.get(logitsOutputName);
            if (logitsRaw == null) {
                log.error("No logits output at step {}", step);
                break;
            }
            // Mixed precision decoder (HALF weights + FLOAT activations) keeps logits in FLOAT.
            // If a future full-HALF path produces HALF logits, cast to FLOAT for sampling precision.
            if (logitsRaw.dataType() == DataType.HALF) {
                INDArray floatLogits = logitsRaw.castTo(DataType.FLOAT);
                if (logitsRaw.closeable()) logitsRaw.close();
                logitsRaw = floatLogits;
            }
            // Note: DSP outputs may have isConstant flags. We clear the stale native
            // error state before token_sample exec instead of calling setCloseable(true),
            // because setCloseable makes the DSP's frozen output closeable, and closing
            // it destroys the DSP's internal output slot cache → fallback to Java path.

            // Diagnostic: top logit values at steps 0-3
            if (step == 0) {
                logLogitsDiagnostics(step, logitsRaw);
            }

            long tAfterLogitsDup = System.nanoTime();

            // KV cache update — delegated to KvCacheManager
            if (usingStaticKv) {
                if (kvScatterInCpp) {
                    // C++ DSP plan scatters present KV into static buffers AND increments
                    // kvCachePosition_ on every execution (in NativeDynamicShapePlan::execute()).
                    // Java only needs to keep its own position counter in sync.
                    kvCacheManager.setCachePosition(kvCacheManager.getCachePosition() + 1);
                } else {
                    // Java scatter — C++ KV scatter is disabled, so we must scatter manually
                    kvCacheManager.scatterNewEntries(decoderOutputs, kvNames);
                }
                // Close present KV outputs — scatter already copied data into static buffers.
                // Without this, 60 tensors × ~4.5MB = ~270MB leaked per step.
                // Skip when C++ KV scatter is active — C++ manages these buffer lifecycles.
                if (!kvScatterInCpp) {
                    int kvClosed = 0;
                    long kvClosedBytes = 0;
                    int kvSkippedClosed = 0, kvSkippedNull = 0;
                    for (String pn : kvNames.keyNames) {
                        INDArray pv = decoderOutputs.get(pn);
                        if (pv == null) { kvSkippedNull++; continue; }
                        if (pv.wasClosed() || pv.data() == null) { kvSkippedClosed++; continue; }
                        long bytes = pv.data().length() * pv.data().getElementSize();
                        pv.setCloseable(true);
                        pv.close();
                        kvClosed++;
                        kvClosedBytes += bytes;
                    }
                    for (String pn : kvNames.valueNames) {
                        INDArray pv = decoderOutputs.get(pn);
                        if (pv == null) { kvSkippedNull++; continue; }
                        if (pv.wasClosed() || pv.data() == null) { kvSkippedClosed++; continue; }
                        long bytes = pv.data().length() * pv.data().getElementSize();
                        pv.setCloseable(true);
                        pv.close();
                        kvClosed++;
                        kvClosedBytes += bytes;
                    }
                    if (step < 5 || step % 10 == 0) {
                        log.info("  [KV-CLOSE] step={} closed={} ({}MB) skippedClosed={} skippedNull={}",
                                step, kvClosed, kvClosedBytes / (1024 * 1024), kvSkippedClosed, kvSkippedNull);
                    }
                }
            } else {
                // Step 0 (prefill): initialize KV cache via KvCacheManager
                long prefillSeqLen = currentSeqLen;
                kvCacheManager.initializeFromPrefill(decoderOutputs, kvNames, maxNewTokens, prefillSeqLen);
                usingStaticKv = true;
                // Re-read maxKvLen and cachePos — they were captured at the top of the
                // loop BEFORE initializeFromPrefill, so the local variables are stale
                // (maxKvLen=-1, cachePos=0). The DSP recompile code below needs the
                // correct post-prefill values for KV retention and decode input config.
                maxKvLen = kvCacheManager.getMaxKvLen();
                cachePos = kvCacheManager.getCachePosition();

                // Close prefill KV outputs — but ONLY when DSP native executor is NOT active.
                // When DSP is active, the C++ slotArrayCache_ still holds raw NDArray* pointers
                // to these outputs. Java close() calls opaqueNDArray.close() which deletes the
                // C++ NDArray, leaving slotArrayCache_ with dangling pointers → use-after-free
                // at the next execution step. The DSP will evict stale arrays naturally when
                // shapes change (prefill [1,h,679,d] → decode [1,h,699,d]).
                InferenceSession prefillSession = decoder.getOrCreateSession();
                boolean prefillDspActive = prefillSession.getDynamicShapePlanExecutor() != null
                        && prefillSession.getDynamicShapePlanExecutor().getCurrentPlan() != null;
                if (!prefillDspActive) {
                    for (String pn : kvNames.keyNames) {
                        INDArray pv = decoderOutputs.get(pn);
                        if (pv != null) { pv.setCloseable(true); pv.close(); }
                    }
                    for (String pn : kvNames.valueNames) {
                        INDArray pv = decoderOutputs.get(pn);
                        if (pv != null) { pv.setCloseable(true); pv.close(); }
                    }
                }

                // Recompile DSP plan now that static KV shapes are known.
                // When speculation is enabled, FrozenDecodeStep.compile() handles this
                // with seqLen=K+1 instead of seqLen=1.
                if (maxSpeculativeTokens <= 0) {
                    InferenceSession decoderSession = decoder.getOrCreateSession();
                    DynamicShapePlanExecutor dspExec = decoderSession.getDynamicShapePlanExecutor();

                    if (dspExec != null && dspExec.getCurrentPlan() != null) {
                        // First page — full compile path.
                        // Associate static KV buffers as placeholder values so compilation sees their shapes
                        Map<String, INDArray> staticKvBuffers = kvCacheManager.getStaticKvBuffers();
                        for (Map.Entry<String, INDArray> e : staticKvBuffers.entrySet()) {
                            String pastName = e.getKey();
                            if (decoder.hasVariable(pastName)) {
                                decoder.associateArrayWithVariable(e.getValue(), pastName);
                            }
                        }
                        // Clear old plan and recompile with correct KV shapes.
                        boolean cppKvEnabled = !skipFreeze
                                && !"true".equals(System.getProperty("nd4j.dsp.kvscatter.java", "false"));
                        String[] recompileOutputs = cppKvEnabled ? logitsOnlyOutputNames : fullOutputNames;
                        decoder.clearDynamicShapePlanCache();
                        decoderSession.clearAllCaches();
                        dspExec = null;  // old executor invalidated
                        if (!skipFreeze) {
                            log.info("  [Perf] Recompiling DSP plan with static KV shapes (maxKvLen={}, outputs={})",
                                    maxKvLen, Arrays.toString(recompileOutputs));
                            decoder.compileNativeDynamicShapePlan(DspCompilationMode.MAX_AUTOTUNE, recompileOutputs);
                        } else {
                            log.info("  [Perf] No-freeze mode: disabling DSP, using op-by-op execution");
                            decoder.setDspAutoCompileEnabled(false);
                            decoder.setDspNativeAutoCompileEnabled(false);
                        }
                        // Re-fetch executor after recompilation
                        decoderSession = decoder.getOrCreateSession();
                        dspExec = decoderSession.getDynamicShapePlanExecutor();

                        // Reassign device placement with fresh memory budgets.
                        // Vision encoder may have been freed, releasing GB of GPU memory.
                        // Without this, all ops stay on the primary GPU even if a secondary
                        // device now has ample free memory for graph capture.
                        decoder.reassignDynamicShapePlanDevices();

                        // CRITICAL: Freeze shapes IMMEDIATELY after recompile.
                        if (dspExec != null && !skipFreeze) {
                            dspExec.setShapesFrozen(true);
                            dspExec.setTraceEnabled(true);
                            dspExec.setExecutionTimingEnabled(true);
                            decoderDspExec = dspExec;
                            log.info("  [Perf] Shapes frozen AFTER recompile (stable Triton cache keys)");

                            // Configure C++ KV scatter: present outputs → static input buffers
                            DynamicShapePlan plan = cppKvEnabled ? dspExec.getCurrentPlan() : null;
                            if (plan != null) {
                                List<String> presentNames = new ArrayList<>();
                                presentNames.addAll(kvNames.keyNames);
                                presentNames.addAll(kvNames.valueNames);
                                List<String> pastNames = new ArrayList<>();
                                for (String pn : presentNames) {
                                    pastNames.add(resolvedIOConfig.presentToInputName(pn));
                                }
                                boolean kvConfigured = dspExec.configureKvCacheRetention(
                                        plan, presentNames, pastNames, (int) maxKvLen, (int) cachePos);
                                if (kvConfigured) {
                                    kvScatterInCpp = true;
                                    log.info("  [Perf] C++ KV scatter enabled: {} mappings, pos={}, decodeOutputs=logits-only",
                                            presentNames.size(), cachePos);

                                    dspExec.configureDecodeInputs(plan, (int) maxKvLen);
                                    if (dspExec.isDecodeInputsConfigured()) {
                                        log.info("  [Perf] C++ decode input updates enabled (zero host-device round-trips)");
                                    }
                                } else {
                                    decoderDspExec = null;
                                    decoder.clearDynamicShapePlanCache();
                                    decoderSession.clearAllCaches();
                                    log.warn("  [Perf] C++ KV scatter setup failed, falling back to Java KV scatter");
                                }
                            }
                        }
                    }
                } // end if (maxSpeculativeTokens <= 0) — DSP recompile guard
            }

            long tAfterKvUpdate = System.nanoTime();

            // Sample next token via native GPU op — pass logits directly (rank-3 supported)
            long tSampStart = System.nanoTime();
            int nextTokenId = reusableTokenSampler.sample(logitsRaw);
            long tSampArgmax = System.nanoTime();
            generatedTokens.add(nextTokenId);

            long stepElapsedNs = System.nanoTime() - stepStart;
            long stepElapsedMs = stepElapsedNs / 1_000_000;

            // Log sampling sub-timings
            if (step < 6 || step % 10 == 0) {
                log.info("  [SAMP] step={} total={}ms (native token_sample)",
                        step,
                        (tSampArgmax - tSampStart) / 1_000_000);
            }

            // Accumulate detailed timing for steps 3+
            if (step >= 3) {
                totalInputBuildNs += tAfterInputBuild - stepStart;
                totalDecoderNs    += tAfterDecoder - tAfterInputBuild;
                totalLogitsDupNs  += tAfterLogitsDup - tAfterDecoder;
                totalKvUpdateNs   += tAfterKvUpdate - tAfterLogitsDup;
                totalSamplingNs   += stepElapsedNs - (tAfterKvUpdate - stepStart);
                detailSteps++;
            }
            if (step >= 20) {
                lateSteadyTotalNs += stepElapsedNs;
                lateSteadySteps++;
            }

            if (step == 0) {
                prefillTimeMs = stepElapsedMs;
                log.info("  Step 0 (prefill): {}ms (seq_len={})", stepElapsedMs, currentSeqLen);
            } else {
                stepTimesMs.add(stepElapsedMs);
            }

            String tokenText = tokenizer.decode(new int[]{nextTokenId}, false);

            // Log every 10 steps or first 6
            if (step < 6 || step % 10 == 0) {
                double currentTokPerSec = step > 0 && stepElapsedMs > 0 ? 1000.0 / stepElapsedMs : 0;
                if (step >= 2) {
                    long inputMs  = (tAfterInputBuild - stepStart) / 1_000_000;
                    long decMs    = (tAfterDecoder - tAfterInputBuild) / 1_000_000;
                    long dupMs    = (tAfterLogitsDup - tAfterDecoder) / 1_000_000;
                    long kvMs     = (tAfterKvUpdate - tAfterLogitsDup) / 1_000_000;
                    long sampMs   = stepElapsedMs - (tAfterKvUpdate - stepStart) / 1_000_000;
                    log.info("  Step {}: '{}' (id={}) {}ms ({} tok/s) [input={}ms dec={}ms dup={}ms kv={}ms samp={}ms cachePos={}]",
                            step, tokenText, nextTokenId, stepElapsedMs, String.format("%.1f", currentTokPerSec),
                            inputMs, decMs, dupMs, kvMs, sampMs, cachePos - 1);
                } else {
                    log.info("  Step {}: '{}' (id={}) {}ms ({} tok/s)",
                            step, tokenText, nextTokenId, stepElapsedMs, String.format("%.1f", currentTokPerSec));
                }
            }

            // Check stop tokens
            if (stopTokenIds.contains(nextTokenId)) {
                log.info("  Stop token at step {}", step);
                finishReason = GenerationResult.FinishReason.EOS;
                // Only close if closeable (zero-copy outputs are non-closeable, managed by DSP cache)
                if (logitsRaw.closeable()) {
                    logitsRaw.close();
                }
                break;
            }

            // Only close if closeable (zero-copy outputs are non-closeable, managed by DSP cache)
            if (logitsRaw.closeable()) {
                logitsRaw.close();
            }

            // Clean up per-step inputs (masks, position_ids — NOT embeddings, input_ids, static KV, or reusable)
            for (var entry : decoderInputMap.entrySet()) {
                String name = entry.getKey();
                INDArray arr = entry.getValue();
                if (resolvedIOConfig.isInputEmbeddings(name) || resolvedIOConfig.isInputIds(name)) continue;
                if (resolvedIOConfig.isKvCacheInput(name)) continue;
                // Skip arrays managed by the reusable inputs cache
                if (reusableInputs.containsValue(arr)) continue;
                if (arr != null && !arr.wasClosed()) {
                    arr.setCloseable(true);
                    arr.close();
                }
            }
            decoder.clearPlaceholders(false);

            // Get embedding for next token
            INDArray prevEmbeddings = currentEmbeddings;
            if (embeddingTable != null) {
                // Direct lookup: single row from embedding table, reshaped to [1, 1, hiddenSize]
                INDArray rowEmbed = embeddingTable.getRow(nextTokenId).reshape(1, 1, resolvedHiddenSize);
                // Use fixed-address buffer for CUDA graph replay stability
                if (usingStaticKv) {
                    if (reusableEmbeddings == null) {
                        reusableEmbeddings = rowEmbed.dup();
                    } else {
                        reusableEmbeddings.assign(rowEmbed);
                    }
                    currentEmbeddings = reusableEmbeddings;
                } else {
                    currentEmbeddings = rowEmbed;
                }
                if (usingStaticKv) {
                    if (reusableInputIds == null) {
                        reusableInputIds = Nd4j.createFromArray(new int[]{nextTokenId}).reshape(1, 1).castTo(DataType.LONG);
                    } else {
                        reusableInputIds.putScalar(0, 0, nextTokenId);
                    }
                    // Close old currentInputIds if it's a different allocation (e.g., the initial prompt tensor)
                    if (currentInputIds != null && currentInputIds != reusableInputIds && !currentInputIds.wasClosed()) {
                        currentInputIds.setCloseable(true);
                        currentInputIds.close();
                    }
                    currentInputIds = reusableInputIds;
                } else {
                    INDArray newTokenTensor = Nd4j.createFromArray(new int[]{nextTokenId}).reshape(1, 1).castTo(DataType.LONG);
                    if (currentInputIds != null && currentInputIds != newTokenTensor && !currentInputIds.wasClosed()) {
                        currentInputIds.setCloseable(true);
                        currentInputIds.close();
                    }
                    currentInputIds = newTokenTensor;
                }
            } else {
                // Fallback: full SameDiff execution
                INDArray newTokenTensor = Nd4j.createFromArray(new int[]{nextTokenId}).reshape(1, 1).castTo(DataType.LONG);
                Map<String, INDArray> newEmbedOutputs = embedTokens.output(
                        Map.of(resolvedEmbedInputName, newTokenTensor), resolvedEmbedOutputNames);
                for (var entry : newEmbedOutputs.entrySet()) {
                    if (usingStaticKv) {
                        if (reusableEmbeddings == null) {
                            reusableEmbeddings = entry.getValue().dup();
                        } else {
                            reusableEmbeddings.assign(entry.getValue());
                        }
                        currentEmbeddings = reusableEmbeddings;
                    } else {
                        currentEmbeddings = entry.getValue();
                    }
                }
                if (currentInputIds != null && currentInputIds != newTokenTensor && !currentInputIds.wasClosed()) {
                    currentInputIds.setCloseable(true);
                    currentInputIds.close();
                }
                currentInputIds = newTokenTensor;
                embedTokens.clearPlaceholders(false);
            }
            // Close previous embeddings — but NOT if it's the same object as
            // currentEmbeddings (reusableEmbeddings is updated in-place via assign(),
            // so prevEmbeddings == currentEmbeddings == reusableEmbeddings).
            // Also skip the original prefillEmbeddings — it's externally owned by the caller.
            if (prevEmbeddings != null && prevEmbeddings != currentEmbeddings
                    && prevEmbeddings != prefillEmbeddings
                    && !prevEmbeddings.wasClosed()) {
                prevEmbeddings.setCloseable(true);
                prevEmbeddings.close();
            }
            pastSeqLen += currentSeqLen;

            // After prefill (step 0), switch to speculative decode path if configured
            if (maxSpeculativeTokens > 0 && usingStaticKv && step == 0) {
                log.info("  Switching to speculative decode path (K={}, seqLen={})",
                        maxSpeculativeTokens, maxSpeculativeTokens + 1);
                // Clean up step 0's reusable inputs
                for (INDArray arr : reusableInputs.values()) {
                    if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
                }
                reusableInputs.clear();
                // Close fixed-address decode buffers (not in reusableInputs map)
                if (reusableEmbeddings != null && !reusableEmbeddings.wasClosed()) {
                    reusableEmbeddings.setCloseable(true);
                    reusableEmbeddings.close();
                }
                if (reusableInputIds != null && !reusableInputIds.wasClosed()) {
                    reusableInputIds.setCloseable(true);
                    reusableInputIds.close();
                }
                // Close currentInputIds if different from reusableInputIds
                if (currentInputIds != null && currentInputIds != reusableInputIds && !currentInputIds.wasClosed()) {
                    currentInputIds.setCloseable(true);
                    currentInputIds.close();
                }
                reusableTokenSampler.close();
                // Re-query maxKvLen — it was -1 at loop entry (before prefill initialized the cache)
                long specMaxKvLen = kvCacheManager.getMaxKvLen();
                // Lift cross-device suppression before entering speculative path
                // (it will manage its own suppression if needed)
                OpaqueDataBuffer.suppressCrossDeviceRouting(false);
                return decodeSpeculative(kvCacheManager, specMaxKvLen, cachePos,
                        resolvedIOConfig, embeddingTable, resolvedHiddenSize,
                        stopTokenIds, generatedTokens, prefillTimeMs, decodeStart,
                        promptTokenIds, resolvedEmbedInputName, resolvedEmbedOutputNames);
            }
        }

        } finally {
            // Always restore cross-device routing when leaving the decode loop
            OpaqueDataBuffer.suppressCrossDeviceRouting(false);
        }

        // Release reusable input arrays
        for (INDArray arr : reusableInputs.values()) {
            if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
        }
        reusableInputs.clear();
        // Close fixed-address decode buffers (not in reusableInputs map)
        if (reusableEmbeddings != null && !reusableEmbeddings.wasClosed()) {
            reusableEmbeddings.setCloseable(true);
            reusableEmbeddings.close();
        }
        if (reusableInputIds != null && !reusableInputIds.wasClosed()) {
            reusableInputIds.setCloseable(true);
            reusableInputIds.close();
        }
        // Close currentInputIds if it's a different object from reusableInputIds
        // (e.g., if decode never ran and it's still the initial prompt tensor)
        if (currentInputIds != null && currentInputIds != reusableInputIds && !currentInputIds.wasClosed()) {
            currentInputIds.setCloseable(true);
            currentInputIds.close();
        }
        reusableTokenSampler.close();

        // Release KV cache buffers via manager
        kvCacheManager.close();

        long totalDecodeMs = System.currentTimeMillis() - decodeStart;

        // Log phase breakdown
        if (detailSteps > 0) {
            log.info("=== PHASE BREAKDOWN (avg over {} decode steps, fast path steps 3+) ===", detailSteps);
            log.info("  Input build:  {}ms", totalInputBuildNs / detailSteps / 1_000_000);
            log.info("  Decoder exec: {}ms", totalDecoderNs    / detailSteps / 1_000_000);
            log.info("  Logits dup:   {}ms", totalLogitsDupNs  / detailSteps / 1_000_000);
            log.info("  KV update:    {}ms", totalKvUpdateNs   / detailSteps / 1_000_000);
            log.info("  Sampling:     {}ms", totalSamplingNs   / detailSteps / 1_000_000);
            log.info("  Sum:          {}ms", (totalInputBuildNs + totalDecoderNs + totalLogitsDupNs + totalKvUpdateNs + totalSamplingNs) / detailSteps / 1_000_000);
        }

        // Build result
        int[] tokenIds = generatedTokens.stream().mapToInt(Integer::intValue).toArray();
        String generatedText = tokenizer.decode(tokenIds, false);

        int decodeTokens = stepTimesMs.size();
        double avgDecodeMs = decodeTokens > 0
                ? stepTimesMs.stream().mapToLong(Long::longValue).average().orElse(0) : 0;
        double decodeTokensPerSec = avgDecodeMs > 0 ? 1000.0 / avgDecodeMs : 0;
        double steadyStateAvgMs = detailSteps > 0
                ? (totalInputBuildNs + totalDecoderNs + totalLogitsDupNs + totalKvUpdateNs + totalSamplingNs)
                / (double) detailSteps / 1_000_000.0
                : 0;
        double steadyStateTokensPerSec = steadyStateAvgMs > 0 ? 1000.0 / steadyStateAvgMs : 0;
        double lateSteadyAvgMs = lateSteadySteps > 0
                ? lateSteadyTotalNs / (double) lateSteadySteps / 1_000_000.0
                : 0;
        double lateSteadyTokensPerSec = lateSteadyAvgMs > 0 ? 1000.0 / lateSteadyAvgMs : 0;
        long p50Ms = decodeTokens > 0 ? percentile(stepTimesMs, 50) : 0;
        long p90Ms = decodeTokens > 0 ? percentile(stepTimesMs, 90) : 0;
        long p99Ms = decodeTokens > 0 ? percentile(stepTimesMs, 99) : 0;

        log.info("========================================");
        log.info("GENERATED TEXT ({} tokens):", generatedTokens.size());
        log.info("{}", generatedText);
        log.info("========================================");
        log.info("PERFORMANCE SUMMARY:");
        log.info("  Prefill (step 0):  {}ms", prefillTimeMs);
        log.info("  Decode tokens:     {} (excluding prefill)", decodeTokens);
        log.info("  Avg decode time:   {}ms/token", String.format("%.1f", avgDecodeMs));
        log.info("  Decode throughput: {} tok/s", String.format("%.2f", decodeTokensPerSec));
        if (steadyStateTokensPerSec > 0) {
            log.info("  Steady-state:      {} tok/s (steps 3+)", String.format("%.2f", steadyStateTokensPerSec));
        }
        if (lateSteadyTokensPerSec > 0) {
            log.info("  Late steady-state: {} tok/s (steps 20+)", String.format("%.2f", lateSteadyTokensPerSec));
        }
        log.info("  Latency P50/P90/P99: {}ms / {}ms / {}ms", p50Ms, p90Ms, p99Ms);
        log.info("  Total decode time: {}ms ({} tokens)", totalDecodeMs, generatedTokens.size());
        log.info("========================================");

        int generated = tokenIds.length;
        return GenerationResult.builder()
                .text(generatedText)
                .tokenIds(tokenIds)
                .generatedTokenCount(generated)
                .promptTokenCount(promptTokenIds.length)
                .totalTokenCount(promptTokenIds.length + generated)
                .finishReason(finishReason)
                .firstTokenLatencyMs(prefillTimeMs)
                .generationTimeMs(totalDecodeMs)
                .tokensPerSecond(totalDecodeMs > 0 ? (generated * 1000.0 / totalDecodeMs) : 0)
                .decodeTokensPerSecond(decodeTokensPerSec)
                .steadyStateTokensPerSecond(steadyStateTokensPerSec)
                .lateSteadyStateTokensPerSecond(lateSteadyTokensPerSec)
                .build();
    }

    /**
     * Speculative decode loop using a merged frozen graph with seqLen=K+1.
     *
     * Each step processes K+1 tokens (1 greedy + K speculative) in a single forward pass.
     * After verification, accepted tokens + correction are added to the output.
     */
    private GenerationResult decodeSpeculative(
            KvCacheManager kvCacheManager, long maxKvLen, long cachePos,
            ModelIOConfig specIOConfig,
            INDArray embeddingTable, long resolvedHiddenSize,
            Set<Integer> stopTokenIds, List<Integer> generatedTokens,
            long prefillTimeMs, long decodeStart, int[] promptTokenIds,
            String resolvedEmbedInputName, String[] resolvedEmbedOutputNames) {

        int K = maxSpeculativeTokens;
        int mergedSeqLen = K + 1;

        // Use provided speculator (e.g., DraftModelSpeculator) or fall back to NgramSpeculator
        NgramSpeculator ngramFallback = this.speculator == null ? new NgramSpeculator(3, K) : null;

        // Create and compile frozen decode step with seqLen=K+1, using ModelIOConfig
        String logitsOutputName = specIOConfig.getLogitsOutputName();
        DecoderUtils.KVCacheNames kvNames = specIOConfig.getKvCacheNames() != null
                ? specIOConfig.getKvCacheNames()
                : DecoderUtils.findKVCacheOutputNames(decoder);
        FrozenDecodeStep frozenStep = new FrozenDecodeStep(
                decoder, mergedSeqLen, maxKvLen, resolvedHiddenSize,
                specIOConfig);
        frozenStep.setKvCacheManager(kvCacheManager);
        frozenStep.compile(DspCompilationMode.MAX_AUTOTUNE);

        // Last greedy token is the most recently generated one (from prefill sampling)
        int lastGreedyToken = generatedTokens.get(generatedTokens.size() - 1);

        // Reusable buffers for merged inputs (avoid per-step allocation)
        INDArray mergedEmbeddings = Nd4j.zeros(DataType.FLOAT, 1, mergedSeqLen, resolvedHiddenSize);
        INDArray mergedInputIds = Nd4j.zeros(DataType.LONG, 1, mergedSeqLen);

        // Speculation stats
        int totalSpeculative = 0;
        int totalAccepted = 0;
        int specSteps = 0;

        // Timing
        List<Long> stepTimesMs = new ArrayList<>();
        long totalStepNs = 0;
        int steadySteps = 0;
        long lateSteadyTotalNs = 0;
        int lateSteadySteps = 0;

        GenerationResult.FinishReason finishReason = GenerationResult.FinishReason.MAX_TOKENS;

        while (generatedTokens.size() < maxNewTokens) {
            long stepStart = System.nanoTime();
            specSteps++;

            // Pool tracking for memory leak diagnosis
            long poolBefore = getPoolUsedMB();

            // 1. Build merged input: [lastGreedyToken, spec0, ..., specK-1]
            int[] specTokens;
            if (ngramFallback != null) {
                specTokens = ngramFallback.speculate(generatedTokens);
            } else {
                int[] historyArr = generatedTokens.stream().mapToInt(Integer::intValue).toArray();
                specTokens = this.speculator.speculate(historyArr);
            }
            int numSpec = Math.min(specTokens.length, K);

            long poolAfterDraft = getPoolUsedMB();

            // Build token array: padded with token 0 for empty speculation positions
            int[] tokenArray = new int[mergedSeqLen];
            tokenArray[0] = lastGreedyToken;
            for (int i = 0; i < K; i++) {
                tokenArray[i + 1] = i < numSpec ? specTokens[i] : 0;
            }

            // Look up embeddings for all K+1 tokens
            if (embeddingTable != null) {
                for (int i = 0; i < mergedSeqLen; i++) {
                    INDArray rowEmbed = embeddingTable.getRow(tokenArray[i]);
                    mergedEmbeddings.get(NDArrayIndex.point(0), NDArrayIndex.point(i),
                            NDArrayIndex.all()).assign(rowEmbed);
                }
            } else {
                // Fallback: full SameDiff execution for embeddings
                INDArray idsTensor = Nd4j.createFromArray(tokenArray).reshape(1, mergedSeqLen).castTo(DataType.LONG);
                Map<String, INDArray> embedOut = embedTokens.output(
                        Map.of(resolvedEmbedInputName, idsTensor), resolvedEmbedOutputNames);
                for (var entry : embedOut.entrySet()) {
                    mergedEmbeddings.assign(entry.getValue());
                }
                embedTokens.clearPlaceholders(false);
                idsTensor.close();
                // Close embed output arrays to prevent leaks
                for (INDArray arr : embedOut.values()) {
                    if (arr != null && !arr.wasClosed()) {
                        arr.setCloseable(true);
                        arr.close();
                    }
                }
            }

            long poolAfterEmbed = getPoolUsedMB();

            // Set merged input IDs
            for (int i = 0; i < mergedSeqLen; i++) {
                mergedInputIds.putScalar(0, i, tokenArray[i]);
            }

            long tAfterEmbed = System.nanoTime();

            // 2. Execute frozen step → logits [1, mergedSeqLen, vocabSize]
            INDArray logits = frozenStep.execute(mergedEmbeddings, mergedInputIds, cachePos);

            if (logits == null) {
                log.error("  [SPEC] Null logits at step {}", specSteps);
                break;
            }

            long tAfterExec = System.nanoTime();

            // 3. Verify speculation — bulk argmax (one D2H transfer for all positions)
            int accepted = 0;
            List<Integer> newTokens = new ArrayList<>();
            boolean hitStop = false;

            // Compute argmax for all positions we need: numSpec + 1 (for correction/bonus)
            int numArgmax = Math.min(numSpec + 1, (int) logits.size(1));
            int[] allArgmax = argmaxAllPositions(logits, numArgmax);

            // Position p logits: distribution after seeing tokens 0..p
            // Check if argmax(logits[p]) == specTokens[p]
            for (int p = 0; p < numSpec; p++) {
                int modelToken = allArgmax[p];
                if (modelToken == specTokens[p]) {
                    accepted++;
                    newTokens.add(specTokens[p]);
                    if (stopTokenIds.contains(specTokens[p])) {
                        hitStop = true;
                        break;
                    }
                } else {
                    break;
                }
            }

            // Correction/bonus token at position `accepted`
            if (!hitStop && accepted < numArgmax) {
                int correctionToken = allArgmax[accepted];
                newTokens.add(correctionToken);
                if (stopTokenIds.contains(correctionToken)) {
                    hitStop = true;
                }
            }

            totalSpeculative += numSpec;
            totalAccepted += accepted;

            long tAfterVerify = System.nanoTime();

            // 4. KV scatter for accepted + correction positions
            int numToScatter = newTokens.size();
            frozenStep.scatterAcceptedKv(cachePos, numToScatter);
            cachePos += numToScatter;

            // 5. Update state
            generatedTokens.addAll(newTokens);
            if (!newTokens.isEmpty()) {
                lastGreedyToken = newTokens.get(newTokens.size() - 1);
            }

            // Sync draft model KV cache — save checkpoint at accepted position for rollback
            if (this.speculator instanceof DraftModelSpeculator) {
                ((DraftModelSpeculator) this.speculator).syncAfterVerification(accepted);
            }

            // Do NOT close logits — in DSP mode they are views into the slot cache.
            // Closing them would destroy the cache buffer, breaking subsequent steps.
            // The slot cache manages buffer lifetime.

            long stepElapsedNs = System.nanoTime() - stepStart;
            long stepElapsedMs = stepElapsedNs / 1_000_000;
            stepTimesMs.add(stepElapsedMs);

            if (specSteps >= 3) {
                totalStepNs += stepElapsedNs;
                steadySteps++;
            }
            if (specSteps >= 10) {
                lateSteadyTotalNs += stepElapsedNs;
                lateSteadySteps++;
            }

            long tAfterScatter = System.nanoTime();

            long poolAfterStep = getPoolUsedMB();

            // Log
            if (specSteps <= 5 || specSteps % 5 == 0) {
                double effectiveTokPerSec = stepElapsedMs > 0
                        ? newTokens.size() * 1000.0 / stepElapsedMs : 0;
                int[] newTokenArray = newTokens.stream().mapToInt(Integer::intValue).toArray();
                String tokenTexts = tokenizer.decode(newTokenArray, false);
                long embedMs = (tAfterEmbed - stepStart) / 1_000_000;
                long execMs = (tAfterExec - tAfterEmbed) / 1_000_000;
                long verifyMs = (tAfterVerify - tAfterExec) / 1_000_000;
                long scatterMs = (tAfterScatter - tAfterVerify) / 1_000_000;
                log.info("  [SPEC] step={} accepted={}/{} +{} tokens ({}ms, {} eff.tok/s) cachePos={} '{}' [embed={}ms exec={}ms verify={}ms scatter={}ms]",
                        specSteps, accepted, numSpec, newTokens.size(), stepElapsedMs,
                        String.format("%.1f", effectiveTokPerSec), cachePos, tokenTexts,
                        embedMs, execMs, verifyMs, scatterMs);
                log.info("  [POOL] step={} before={}MB afterDraft={}MB(+{}) afterEmbed={}MB(+{}) afterStep={}MB(+{}) netDelta=+{}MB",
                        specSteps, poolBefore, poolAfterDraft, poolAfterDraft - poolBefore,
                        poolAfterEmbed, poolAfterEmbed - poolAfterDraft,
                        poolAfterStep, poolAfterStep - poolAfterEmbed,
                        poolAfterStep - poolBefore);
            }

            if (hitStop) {
                finishReason = GenerationResult.FinishReason.EOS;
                // Truncate at stop token: remove anything after the stop token
                for (int i = generatedTokens.size() - 1; i >= 0; i--) {
                    if (stopTokenIds.contains(generatedTokens.get(i))) {
                        while (generatedTokens.size() > i + 1) {
                            generatedTokens.remove(generatedTokens.size() - 1);
                        }
                        break;
                    }
                }
                log.info("  Stop token at spec step {}", specSteps);
                break;
            }
        }

        // Cleanup
        frozenStep.close();
        mergedEmbeddings.close();
        mergedInputIds.close();

        // Release KV cache buffers via manager
        kvCacheManager.close();

        long totalDecodeMs = System.currentTimeMillis() - decodeStart;

        // Build result
        int[] tokenIds = generatedTokens.stream().mapToInt(Integer::intValue).toArray();
        String generatedText = tokenizer.decode(tokenIds, false);
        int generated = tokenIds.length;

        double avgTokensPerStep = specSteps > 0 ? (double) generated / specSteps : 0;
        double effectiveTokPerSec = totalDecodeMs > 0 ? generated * 1000.0 / totalDecodeMs : 0;
        double avgStepMs = steadySteps > 0
                ? totalStepNs / (double) steadySteps / 1_000_000.0 : 0;
        double steadyStateTokPerSec = avgStepMs > 0
                ? avgTokensPerStep * 1000.0 / avgStepMs : 0;
        double lateSteadyAvgMs = lateSteadySteps > 0
                ? lateSteadyTotalNs / (double) lateSteadySteps / 1_000_000.0 : 0;
        double lateSteadyTokPerSec = lateSteadyAvgMs > 0
                ? avgTokensPerStep * 1000.0 / lateSteadyAvgMs : 0;
        double acceptanceRate = totalSpeculative > 0
                ? (double) totalAccepted / totalSpeculative : 0;

        log.info("========================================");
        log.info("GENERATED TEXT ({} tokens):", generated);
        log.info("{}", generatedText);
        log.info("========================================");
        log.info("SPECULATIVE DECODE SUMMARY:");
        log.info("  Prefill:           {}ms", prefillTimeMs);
        log.info("  Tokens generated:  {} in {} spec steps ({} avg/step)",
                generated, specSteps, String.format("%.1f", avgTokensPerStep));
        log.info("  Acceptance rate:   {} ({}/{} speculative tokens)",
                String.format("%.1f%%", acceptanceRate * 100), totalAccepted, totalSpeculative);
        log.info("  Effective tok/s:   {}", String.format("%.1f", effectiveTokPerSec));
        if (steadyStateTokPerSec > 0) {
            log.info("  Steady-state:      {} tok/s (steps 3+)", String.format("%.1f", steadyStateTokPerSec));
        }
        if (lateSteadyTokPerSec > 0) {
            log.info("  Late steady-state: {} tok/s (steps 10+)", String.format("%.1f", lateSteadyTokPerSec));
        }
        log.info("  Speculator:        {}", this.speculator != null ? this.speculator.getName() : "ngram-3");
        if (this.speculator instanceof DraftModelSpeculator) {
            log.info("  Draft model stats: {}", ((DraftModelSpeculator) this.speculator).getStats());
        }
        log.info("  Avg step time:     {}ms", String.format("%.1f", avgStepMs));
        log.info("  Total decode time: {}ms", totalDecodeMs);
        log.info("========================================");

        return GenerationResult.builder()
                .text(generatedText)
                .tokenIds(tokenIds)
                .generatedTokenCount(generated)
                .promptTokenCount(promptTokenIds.length)
                .totalTokenCount(promptTokenIds.length + generated)
                .finishReason(finishReason)
                .firstTokenLatencyMs(prefillTimeMs)
                .generationTimeMs(totalDecodeMs)
                .tokensPerSecond(effectiveTokPerSec)
                .decodeTokensPerSecond(effectiveTokPerSec)
                .steadyStateTokensPerSecond(steadyStateTokPerSec)
                .lateSteadyStateTokensPerSecond(lateSteadyTokPerSec)
                .totalSpeculativeTokens(totalSpeculative)
                .totalAcceptedTokens(totalAccepted)
                .speculativeSteps(specSteps)
                .averageAcceptanceRate(acceptanceRate)
                .effectiveTokensPerSecond(effectiveTokPerSec)
                .build();
    }

    /**
     * Compute argmax for all sequence positions in logits [1, seqLen, vocab] at once.
     * Returns int[] where result[p] = argmax over vocab dimension for position p.
     * Transfers the logits slice to host once (bulk D2H) then iterates locally.
     */
    private int[] argmaxAllPositions(INDArray logits, int numPositions) {
        int vocabSize = (int) logits.size(2);
        int[] result = new int[numPositions];
        // Get the [numPositions, vocab] slice and transfer to host once
        INDArray slice = logits.get(NDArrayIndex.point(0),
                NDArrayIndex.interval(0, numPositions),
                NDArrayIndex.all());
        // dup() creates contiguous copy of view, data().asFloat() does bulk D2H
        INDArray contiguous = slice.dup();
        float[] data = contiguous.data().asFloat();
        contiguous.close();
        for (int p = 0; p < numPositions; p++) {
            int offset = p * vocabSize;
            int topId = 0;
            float topVal = data[offset];
            for (int v = 1; v < vocabSize; v++) {
                float val = data[offset + v];
                if (val > topVal) {
                    topVal = val;
                    topId = v;
                }
            }
            result[p] = topId;
        }
        return result;
    }

    private void logDiagnostics(int step, Map<String, INDArray> decoderInputMap, ModelIOConfig diagConfig) {
        for (var entry : decoderInputMap.entrySet()) {
            INDArray v = entry.getValue();
            String name = entry.getKey();
            if (diagConfig.isCausalMask(name)) {
                log.info("  [DIAG] step={} {}: shape={} min={} max={} nonzero={}",
                        step, name, Arrays.toString(v.shape()),
                        v.minNumber().floatValue(), v.maxNumber().floatValue(),
                        v.neq(0).sumNumber().longValue());
                long len = v.length();
                INDArray flat = v.reshape(len);
                StringBuilder sb = new StringBuilder("  [DIAG]   values[0..4]=");
                for (int i = 0; i < Math.min(5, len); i++) sb.append(flat.getFloat(i)).append(",");
                sb.append(" ... values[").append(len - 5).append("..").append(len - 1).append("]=");
                for (long i = Math.max(0, len - 5); i < len; i++) sb.append(flat.getFloat(i)).append(",");
                log.info(sb.toString());
            } else if (diagConfig.isAttentionMask(name)) {
                log.info("  [DIAG] step={} {}: shape={} sum={}",
                        step, name, Arrays.toString(v.shape()), v.sumNumber().longValue());
            } else if (diagConfig.isKvCacheInput(name) && name.contains(".key.0")) {
                log.info("  [DIAG] step={} {}: shape={} absMax={}",
                        step, name, Arrays.toString(v.shape()), v.amaxNumber().floatValue());
            } else if (diagConfig.isPositionIds(name)) {
                log.info("  [DIAG] step={} {}: shape={} values={}",
                        step, name, Arrays.toString(v.shape()), v);
            }
        }
    }

    private void logPresentKvDiagnostics(int step, Map<String, INDArray> decoderOutputs,
                                         DecoderUtils.KVCacheNames kvNames) {
        for (String pn : kvNames.keyNames) {
            if (pn.contains(".0")) {
                INDArray pv = decoderOutputs.get(pn);
                if (pv != null) {
                    log.info("  [DIAG] step={} present {}: shape={} lastPosAbsMax={}",
                            step, pn, Arrays.toString(pv.shape()),
                            pv.get(NDArrayIndex.point(0), NDArrayIndex.all(),
                                    NDArrayIndex.point(pv.shape()[2] - 1), NDArrayIndex.all()).amaxNumber().floatValue());
                }
            }
        }
    }

    private void logLogitsDiagnostics(int step, INDArray logits) {
        try {
            INDArray lastLogitsDiag = logits.rank() == 3
                    ? logits.get(NDArrayIndex.point(0), NDArrayIndex.point(logits.size(1) - 1), NDArrayIndex.all())
                    : logits.getRow(0);
            // Use only getFloat() — CUDA reduction ops (min/max/mean/argmax) fail on
            // constant-flagged DSP output arrays. getFloat() works on any array.
            int len = (int) lastLogitsDiag.length();
            int topId = 0;
            float topVal = lastLogitsDiag.getFloat(0);
            float logitsMin = topVal, logitsMax = topVal;
            double logitsSum = topVal;
            boolean hasNaN = Float.isNaN(topVal);
            for (int i = 1; i < len; i++) {
                float v = lastLogitsDiag.getFloat(i);
                if (Float.isNaN(v)) hasNaN = true;
                if (v > topVal) { topVal = v; topId = i; }
                if (v < logitsMin) logitsMin = v;
                if (v > logitsMax) logitsMax = v;
                logitsSum += v;
            }
            float logitsMean = (float) (logitsSum / len);
            boolean allZero = logitsMax == 0.0f && logitsMin == 0.0f;
            StringBuilder first10 = new StringBuilder();
            int show = Math.min(10, len);
            for (int i = 0; i < show; i++) {
                if (i > 0) first10.append(", ");
                first10.append(String.format("%.4f", lastLogitsDiag.getFloat(i)));
            }
            log.info("  [DIAG] step={} logits: topId={} topVal={} min={} max={} mean={} hasNaN={} allZero={} shape={} first10=[{}]",
                    step, topId, topVal, logitsMin, logitsMax, logitsMean, hasNaN, allZero,
                    Arrays.toString(lastLogitsDiag.shape()), first10);
        } catch (Exception e) {
            log.warn("  [DIAG] step={} logits diagnostics failed: {}", step, e.getMessage());
        }
    }

    private static long percentile(List<Long> values, int percentile) {
        if (values.isEmpty()) return 0;
        List<Long> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        int idx = (int) Math.ceil(percentile / 100.0 * sorted.size()) - 1;
        return sorted.get(Math.max(0, Math.min(idx, sorted.size() - 1)));
    }

    private static long getPoolUsedMB() {
        try (LongPointer used = new LongPointer(1);
             LongPointer reserved = new LongPointer(1)) {
            NativeOpsHolder.getInstance().getDeviceNativeOps()
                    .getMemoryPoolStats(Nd4j.getAffinityManager().getDeviceForCurrentThread(), used, reserved);
            return used.get() / (1024 * 1024);
        } catch (Exception e) {
            return -1;
        }
    }
}
