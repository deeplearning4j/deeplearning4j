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
import org.eclipse.deeplearning4j.llm.generation.kvcache.KvCacheManager;
import org.eclipse.deeplearning4j.llm.generation.kvcache.KvCacheStrategy;
import org.eclipse.deeplearning4j.llm.generation.kvcache.PagedKVCache;
import org.eclipse.deeplearning4j.llm.generation.kvcache.QuantizedPagedKVCache;
import org.eclipse.deeplearning4j.llm.generation.kvcache.UnifiedKvCacheManager;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.generation.speculative.Speculator;
import org.eclipse.deeplearning4j.llm.generation.speculative.NgramSpeculator;
import org.eclipse.deeplearning4j.llm.generation.speculative.DraftModelSpeculator;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.execution.DspCompilationMode;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.TokenSample;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import org.bytedeco.javacpp.LongPointer;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.linalg.api.buffer.DataBuffer;

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

    private static final int DEFAULT_DIAGNOSTIC_STEPS = 3;

    private static boolean decodePhaseLoggingEnabled() {
        return Boolean.parseBoolean(System.getProperty(ND4JSystemProperties.VLM_BENCHMARK_DECODE_PHASE_LOGGING, "true"));
    }

    private static boolean detailedDecodeDiagnosticsEnabled() {
        return Boolean.parseBoolean(System.getProperty(ND4JSystemProperties.VLM_BENCHMARK_DECODE_DIAGNOSTICS, "false"))
                || Boolean.parseBoolean(System.getProperty("vlm.benchmark.logitFingerprints", "false"))
                || Boolean.parseBoolean(System.getProperty("vlm.benchmark.tensorFingerprints", "false"));
    }

    private static int diagnosticSteps() {
        return Integer.getInteger("vlm.benchmark.diagnosticSteps", DEFAULT_DIAGNOSTIC_STEPS);
    }

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
            if (logits.wasClosed()) {
                throw new IllegalStateException(
                    "BUFFER_LIFECYCLE: ReusableTokenSampler.sample() received a CLOSED logits array. " +
                    "shape=" + java.util.Arrays.toString(logits.shape()));
            }
            DataBuffer db = logits.data();
            if (db == null || db.wasClosed()) {
                throw new IllegalStateException(
                    "BUFFER_LIFECYCLE: ReusableTokenSampler.sample() received logits with " +
                    (db == null ? "null" : "CLOSED") + " DataBuffer. " +
                    "shape=" + java.util.Arrays.toString(logits.shape()) +
                    ". The DSP output buffer was freed before sampling.");
            }
            long batchSize = logits.rank() == 3 ? logits.size(0)
                    : (logits.rank() == 2 ? logits.size(0) : 1);

            if (tokenOutput == null || outputBatchSize != batchSize || tokenOutput.wasClosed()) {
                SameDiffMemoryUtils.safeClose(tokenOutput);
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
            SameDiffMemoryUtils.safeClose(tokenOutput);
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
     * Enable per-step argmax + top-K logit tracing. When true (or the system property
     * {@code vlm.benchmark.argmaxTrace=true} is set), each decode step logs the sampled
     * token ID alongside the top-K logit indices and values. Used to diagnose silent
     * sampling divergence (e.g., CUDA graph replay returning the wrong token while still
     * producing a plausibly-typed vector).
     */
    @Builder.Default
    private final boolean argmaxTraceEnabled = false;

    /** Number of top-K logit entries to emit when {@link #argmaxTraceEnabled} is on. */
    @Builder.Default
    private final int argmaxTraceTopK = 5;

    /**
     * Optional reference token stream. When non-null, each decode step asserts that
     * the sampled token matches {@code referenceTokenStream[step]}. On mismatch the
     * loop throws {@link TokenStreamDivergenceException} with the step index, the
     * expected/actual token IDs, and the top-K logit snapshot. Enables byte-for-byte
     * golden-token comparison across runs to catch regressions not visible in output
     * text. A shorter array than {@code maxNewTokens} only checks the prefix it covers.
     */
    private final int[] referenceTokenStream;

    /**
     * When true (default), uses fixed-address reusable buffers for embeddings and
     * input IDs across decode steps. This enables CUDA graph replay stability but
     * was the root cause of stale-data bugs (see f8e83ff4c9). When false, allocates
     * fresh buffers each step — address drift forces graph re-evaluation.
     */
    @Builder.Default
    private final boolean useReusableBuffers = true;

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
        return new UnifiedKvCacheManager(kvCacheStrategy, resolvedIOConfig,
                quantFormat, DataType.FLOAT, turboQuantBits, PagedKVCache.DEFAULT_BLOCK_SIZE);
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
        ModelIOConfig.KVCacheNames kvNames = resolvedIOConfig.getKvCacheNames() != null
                ? resolvedIOConfig.getKvCacheNames()
                : ModelIOConfig.findKVCacheOutputNames(decoder);

        // Reusable decode-step diagnostics — auto-enables on accuracy failure
        DecodeStepDiagnostics stepDiag = new DecodeStepDiagnostics();
        // Get ALL external inputs — placeholders, constants, variables, and array-type
        // variables that the graph reads. This matches what the DSP compiler discovers.
        List<String> decoderInputNames = decoder.externalInputs();
        log.info("  Decoder input names: {}", decoderInputNames);

        // Resolve stop tokens
        int eosTokenId = samplingConfig.getEosTokenId() >= 0
                ? samplingConfig.getEosTokenId() : tokenizer.getEosTokenId();
        Set<Integer> stopTokenIds = new HashSet<>();
        if (eosTokenId >= 0) {
            stopTokenIds.add(eosTokenId);
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
        if (embeddingTable == null) {
            throw new IllegalStateException(
                    "StaticKvCacheDecodeLoop requires a direct embedding table from embedTokens; "
                            + "per-token SameDiff.output() embedding fallback has been removed");
        }
        log.info("  Using direct embedding lookup: shape={} (native decode path)",
                Arrays.toString(embeddingTable.shape()));

        // Decode always requests logits + present KV outputs. KvScatter runs as an
        // ordinary Java-invoked op that copies the present KV tensors into the
        // Java-owned static KV buffers at each step.
        List<String> fullOutputNameList = new ArrayList<>();
        fullOutputNameList.add(logitsOutputName);
        fullOutputNameList.addAll(kvNames.keyNames);
        fullOutputNameList.addAll(kvNames.valueNames);
        String[] fullOutputNames = fullOutputNameList.toArray(new String[0]);

        // State
        List<Integer> generatedTokens = new ArrayList<>();
        INDArray currentEmbeddings = prefillEmbeddings;
        INDArray currentInputIds = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, promptTokenIds.length).castTo(DataType.LONG);
        // Reusable fixed-address buffers for decode steps.
        // Graph replay correctness: the C++ capture path marks all external input
        // DataBuffers as device-actual (writeSpecial()) before capture, preventing
        // stale H2D memcpy nodes from being recorded. On replay, the Triton arg
        // table points to this fixed address and reads the fresh .assign() data.
        INDArray reusableEmbeddings = null;  // allocated on first decode step
        INDArray reusableInputIds = null;
        INDArray pendingEmbedClose = null;  // deferred close: freed at start of next step
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

        // Warmup vs steady-state classification.
        // A step is "warmup" when the DSP plan phase has not yet reached REPLAYING,
        // meaning compilation or CUDA graph capture overhead is included in the step time.
        // These steps MUST NOT be included in steady-state tok/s averages.
        int warmupStepCount = 0;
        long maxWarmupStepMs = 0;
        long totalWarmupNs = 0;
        int firstSteadyStep = -1;  // step index of first steady-state step

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

        // KV cache management — delegated to UnifiedKvCacheManager.
        // KV scatter runs as a standard KvScatter op (in-graph or post-execution).
        // The DSP plan is a pure graph executor with no KV-specific lifecycle.
        KvCacheManager kvCacheManager = createKvCacheManager(resolvedIOConfig);
        boolean usingStaticKv = false;
        boolean skipFreeze = "true".equalsIgnoreCase(System.getProperty(ND4JSystemProperties.DSP_NO_FREEZE));
        DynamicShapePlanExecutor decoderDspExec = null;  // Tracked DSP executor for shape freezing



        // Reusable input arrays — avoids per-step allocation of masks/position_ids
        Map<String, INDArray> reusableInputs = new HashMap<>();
        ReusableTokenSampler reusableTokenSampler = new ReusableTokenSampler(samplingConfig);
        boolean loggedDirectPath = false;

        GenerationResult.FinishReason finishReason = GenerationResult.FinishReason.MAX_TOKENS;

        // Pin thread to the decoder's execution device before the loop.
        // Vision encoder or model loading may have left the thread on a different
        // device. All arrays created during input building (position_ids, masks)
        // must be on the execution device to avoid cross-device migration.
        {
            InferenceSession session = decoder.getOrCreateSession();
            DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();
            if (dspExec != null) {
                dspExec.ensureExecutionDevice();
            }
        }

        boolean memoryDiag = "true".equalsIgnoreCase(System.getProperty(ND4JSystemProperties.ND4J_DECODE_MEMORY_DIAG));
        long prevStepFree = 0;
        for (int step = 0; step < maxNewTokens; step++) {
            long stepStart = System.nanoTime();
            long currentSeqLen = currentEmbeddings.shape()[1];

            if (memoryDiag && step <= 10) {
                org.nd4j.nativeblas.NativeOps nops = org.nd4j.nativeblas.NativeOpsHolder.getInstance().getDeviceNativeOps();
                int dev = org.nd4j.linalg.factory.Nd4j.getAffinityManager()
                        .getDeviceForCurrentThread().intValue();
                long free = nops.getDeviceFreeMemory(dev);
                long delta = step > 0 ? (prevStepFree - free) : 0;
                log.info("[MEM-DIAG] step={} free={}MB delta={}MB useDirect={}",
                        step, free / (1024*1024), delta / (1024*1024),
                        usingStaticKv && step >= 2 && !skipFreeze);
                prevStepFree = free;
            }

            // Re-pin to execution device at the start of each step.
            // Token sampling / embedding ops between steps can change the thread's device.
            {
                InferenceSession session = decoder.getOrCreateSession();
                DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();
                if (dspExec != null) {
                    dspExec.ensureExecutionDevice();
                }
            }

            // Build input map (with reusable input cache for decode steps)
            // dspActive = padded mode with frozen shapes (enables native C++ input updates)
            boolean dspActive = usingStaticKv
                    && !"true".equalsIgnoreCase(System.getProperty(ND4JSystemProperties.DSP_NO_PADDED));
            long maxKvLen = kvCacheManager.getMaxKvLen();
            long cachePos = kvCacheManager.getCachePosition();

            // Per-phase memory tracking for leak diagnosis
            long memBeforeInputBuild = 0;
            boolean trackMem = step <= 5 || step % 10 == 0;
            if (trackMem) {
                memBeforeInputBuild = NativeOpsHolder.getInstance().getDeviceNativeOps().getDeviceFreeMemory(0) / (1024 * 1024);
            }

            Map<String, INDArray> decoderInputMap;
            try {
                decoderInputMap = DecoderInputBuilder.buildDecoderInputMap(
                        resolvedIOConfig, decoderInputNames, decoder,
                        currentEmbeddings, currentInputIds,
                        pastSeqLen, currentSeqLen,
                        kvCacheManager.getStaticKvBuffers(), maxKvLen, cachePos,
                        usingStaticKv, resolvedHiddenSize,
                        reusableInputs, dspActive,
                        encoderOutputsForDecode, null);
            } catch (Exception e) {
                throw decodeStageFailure("INPUT_BUILD", step, pastSeqLen, currentSeqLen,
                        usingStaticKv, false, cachePos, e);
            }

            long tAfterInputBuild = System.nanoTime();

            long memAfterInputBuild = 0;
            if (trackMem) {
                memAfterInputBuild = NativeOpsHolder.getInstance().getDeviceNativeOps().getDeviceFreeMemory(0) / (1024 * 1024);
                log.info("  [PHASE_MEM] step={} phase=INPUT_BUILD before={}MB after={}MB delta={}MB",
                        step, memBeforeInputBuild, memAfterInputBuild, memBeforeInputBuild - memAfterInputBuild);
            }

            // All input updates are handled by Java via buildDecoderInputMap + syncToSpecial.
            // No C++ decode input management needed.

            boolean detailedStepDiagnostics = detailedDecodeDiagnosticsEnabled() && step <= diagnosticSteps();

            // Diagnostic logging for early steps only when explicitly enabled
            if (detailedStepDiagnostics && step >= 1) {
                logDiagnostics(step, decoderInputMap, resolvedIOConfig);
            }

            // Run decoder — use fast path when shapes are frozen (skips setCloseable overhead)
            Map<String, INDArray> decoderOutputs;
            // outputDirect skips dup() of output arrays — safe only when shapes are frozen
            // and DSP slot cache is stable (CUDA graph replay). In no-freeze mode, DSP
            // slot arrays may be invalidated by subsequent executions, so use output()
            // which dups all results into independent arrays.
            boolean useDirect = usingStaticKv && step >= 2 && !skipFreeze
                    && !"true".equalsIgnoreCase(System.getProperty(ND4JSystemProperties.DSP_NO_DIRECT));
            // Decode always requests full outputs (logits + present KV). The KV scatter
            // runs as an ordinary op that writes into the Java-owned static KV buffers.
            String[] requestedOutputNames = fullOutputNames;
            if (useDirect && !loggedDirectPath) {
                logDecodePhase("OUTPUT_DIRECT_ACTIVE", step,
                        "requestedOutputs=" + Arrays.toString(requestedOutputNames));
                loggedDirectPath = true;
            }
            try {
                if (useDirect) {
                    decoderOutputs = decoder.outputDirect(
                            decoderInputMap, requestedOutputNames);
                } else {
                    decoderOutputs = decoder.output(
                            decoderInputMap, requestedOutputNames);
                }
            } catch (Exception e) {
                throw decodeStageFailure(useDirect ? "DSP_OUTPUT_DIRECT" : "DSP_OUTPUT",
                        step, pastSeqLen, currentSeqLen, usingStaticKv,
                        useDirect, cachePos, e);
            }

            long tAfterDecoder = System.nanoTime();

            if (trackMem) {
                long memAfterDecoder = NativeOpsHolder.getInstance().getDeviceNativeOps().getDeviceFreeMemory(0) / (1024 * 1024);
                log.info("  [PHASE_MEM] step={} phase=DECODER_EXEC before={}MB after={}MB delta={}MB outputs={} useDirect={}",
                        step, memAfterInputBuild, memAfterDecoder, memAfterInputBuild - memAfterDecoder,
                        decoderOutputs.size(), useDirect);
            }

            // Diagnostic: present KV shapes at steps 1-3
            if (detailedStepDiagnostics && step >= 1) {
                logPresentKvDiagnostics(step, decoderOutputs, kvNames);
            }

            // Extract logits — keep raw reference, no dup needed
            // TokenSample now accepts rank-3 [batch, seqLen, vocabSize] directly
            INDArray logitsRaw = decoderOutputs.get(logitsOutputName);
            if (logitsRaw == null) {
                throw decodeStageFailure("LOGITS_MISSING", step, pastSeqLen, currentSeqLen,
                        usingStaticKv, useDirect, cachePos,
                        new IllegalStateException("Missing logits output '" + logitsOutputName
                                + "' from requested outputs " + Arrays.toString(requestedOutputNames)));
            }
            // Validate logits buffer IMMEDIATELY — catch freed/null OpaqueDataBuffer
            // here (step + context) instead of deep in getDeviceId()/dup().
            {
                DataBuffer logitsBuf = logitsRaw.data();
                if (logitsBuf == null) {
                    throw decodeStageFailure("LOGITS_BUFFER_NULL", step, pastSeqLen, currentSeqLen,
                            usingStaticKv, useDirect, cachePos,
                            new IllegalStateException(
                                "BUFFER_LIFECYCLE: logits output '" + logitsOutputName +
                                "' has null DataBuffer immediately after DSP execution. " +
                                "shape=" + java.util.Arrays.toString(logitsRaw.shape()) +
                                ", wasClosed=" + logitsRaw.wasClosed()));
                }
                if (logitsBuf.wasClosed()) {
                    throw decodeStageFailure("LOGITS_BUFFER_CLOSED", step, pastSeqLen, currentSeqLen,
                            usingStaticKv, useDirect, cachePos,
                            new IllegalStateException(
                                "BUFFER_LIFECYCLE: logits output '" + logitsOutputName +
                                "' DataBuffer is CLOSED immediately after DSP execution. " +
                                "shape=" + java.util.Arrays.toString(logitsRaw.shape()) +
                                ". The buffer was freed between DSP return and logits extraction."));
                }
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

            // Diagnostic: top logit values at early steps when explicitly enabled
            if (detailedStepDiagnostics) {
                logLogitsDiagnostics(step, logitsRaw);
            }

            long tAfterLogitsDup = System.nanoTime();

            // KV cache update — delegated to KvCacheManager.
            // KvScatter runs as an ordinary graph/native op that writes present KV
            // entries into the Java-owned static KV buffers at cachePosition.
            if (usingStaticKv) {
                kvCacheManager.scatterNewEntries(decoderOutputs, kvNames);

                // Close present KV outputs — scatter already copied data into static buffers.
                // Without this, 60 tensors × ~4.5MB = ~270MB leaked per step.
                //
                // IMPORTANT: When useDirect=true, the KV output arrays are managed by the
                // DSP executor's zeroCopyOutputCache. Force-closing them (setCloseable(true) +
                // close()) destroys the cache entries, causing the cache to be rebuilt every
                // step — allocating 61 new output arrays (~122MB) that never get freed.
                // Skip close when direct mode is active; the cache handles their lifecycle.
                if (!useDirect) {
                    for (String pn : kvNames.keyNames) {
                        SameDiffMemoryUtils.safeClose(decoderOutputs.get(pn));
                    }
                    for (String pn : kvNames.valueNames) {
                        SameDiffMemoryUtils.safeClose(decoderOutputs.get(pn));
                    }
                }
            } else {
                // Step 0 (prefill): initialize KV cache via KvCacheManager
                long prefillSeqLen = currentSeqLen;
                try {
                    kvCacheManager.initializeFromPrefill(decoderOutputs, kvNames, maxNewTokens, prefillSeqLen);
                } catch (Exception e) {
                    throw decodeStageFailure("KV_PREFILL_INIT", step, pastSeqLen, currentSeqLen,
                            usingStaticKv, useDirect, cachePos, e);
                }
                usingStaticKv = true;
                // Re-read maxKvLen and cachePos — they were captured at the top of the
                // loop BEFORE initializeFromPrefill, so the local variables are stale
                // (maxKvLen=-1, cachePos=0). The DSP recompile code below needs the
                // correct post-prefill values for KV retention and decode input config.
                maxKvLen = kvCacheManager.getMaxKvLen();
                cachePos = kvCacheManager.getCachePosition();
                logDecodePhase("PREFILL_COMPLETE", step,
                        "prefillSeqLen=" + prefillSeqLen + " maxKvLen=" + maxKvLen + " cachePos=" + cachePos);

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
                        SameDiffMemoryUtils.safeClose(decoderOutputs.get(pn));
                    }
                    for (String pn : kvNames.valueNames) {
                        SameDiffMemoryUtils.safeClose(decoderOutputs.get(pn));
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
                        // Decode requests logits + present KV on every step — the KvScatter op
                        // runs as an ordinary operation that reads the present outputs and
                        // writes them into the Java-owned static KV buffers at cachePosition.
                        String[] recompileOutputs = fullOutputNames;
                        // Invalidate stale prefill node outputs so the recompiled DSP plan
                        // doesn't read wrong values. Do NOT call clearAllCaches() —
                        // that also flushes the array cache which progressively destroys
                        // model constant DataBuffers across recompile cycles.
                        // NOTE: Do NOT call clearDynamicShapePlanCache() here — that
                        // destroys the DSP plan mid-inference, preventing proper CUDA
                        // graph replay. Instead, let the plan recompile naturally when
                        // shapes change (the plan detects shape mismatches automatically).
                        decoderSession.clearNodeOutputsOnly();
                        dspExec = null;  // old executor invalidated
                        if (!skipFreeze) {
                            log.info("  [Perf] Recompiling DSP plan with static KV shapes (maxKvLen={}, outputs={})",
                                    maxKvLen, Arrays.toString(recompileOutputs));
                            try {
                                // Use the model's current graphExecutionMode (set by
                                // BenchmarkConfigApplier or explicit setGraphExecutionMode).
                                // Do NOT force MAX_AUTOTUNE here — that overrides the caller's
                                // requested mode (e.g., SLOT_BY_SLOT for validation tests).
                                decoder.compileNativeDynamicShapePlan(recompileOutputs);
                            } catch (Exception e) {
                                throw decodeStageFailure("DSP_STATIC_KV_RECOMPILE", step, pastSeqLen, currentSeqLen,
                                        usingStaticKv, useDirect, cachePos, e);
                            }
                        } else {
                            // No-freeze mode: let DSP stay enabled so it can still capture
                            // and replay graphs. The plan handles shape changes internally.
                            log.info("  [Perf] No-freeze mode: DSP remains enabled, shapes not frozen");
                        }
                        // Re-fetch executor after recompilation
                        decoderSession = decoder.getOrCreateSession();
                        dspExec = decoderSession.getDynamicShapePlanExecutor();

                        // Reassign device placement with fresh memory budgets.
                        // Vision encoder may have been freed, releasing GB of GPU memory.
                        // Without this, all ops stay on the primary GPU even if a secondary
                        // device now has ample free memory for graph capture.
                        decoder.reassignDynamicShapePlanDevices();

                        if (dspExec == null && !skipFreeze) {
                            throw new IllegalStateException(
                                    "Static KV decode requires a DSP executor after recompilation; "
                                            + "native-only benchmark path cannot continue without it");
                        }

                        // CRITICAL: Freeze shapes IMMEDIATELY after recompile.
                        if (dspExec != null && !skipFreeze) {
                            dspExec.setShapesFrozen(true);
                            boolean dspDiagnosticsEnabled =
                                    Boolean.parseBoolean(System.getProperty(
                                            ND4JSystemProperties.DSP_EXECUTION_TIMING, "false"))
                                    || Boolean.parseBoolean(System.getProperty(
                                            ND4JSystemProperties.VLM_BENCHMARK_DSP_EXECUTION_TIMING, "false"))
                                    || Boolean.parseBoolean(System.getProperty(
                                            ND4JSystemProperties.VLM_BENCHMARK_OP_TIMING, "false"));
                            dspExec.setExecutionTimingEnabled(dspDiagnosticsEnabled);
                            dspExec.setTraceEnabled(
                                    System.getProperty(ND4JSystemProperties.DSP_TRACE) != null);
                            decoderDspExec = dspExec;
                            log.info("  [Perf] Shapes frozen AFTER recompile (stable Triton cache keys)");
                            logDecodePhase("SHAPES_FROZEN", step,
                                    "maxKvLen=" + maxKvLen + " cachePos=" + cachePos
                                            + " planPhase=" + dspExec.getPlanPhase()
                                            + " pointersStable=" + dspExec.arePointersStable());
                            // KV scatter runs as an ordinary op via KvCacheManager.scatterNewEntries —
                            // no C++ KV retention configuration or decode input wiring needed.
                            // All decode inputs flow through buildDecoderInputMap → syncToSpecial →
                            // capture buffer D2D on each step.
                        }
                    }
                } // end if (maxSpeculativeTokens <= 0) — DSP recompile guard

                // Re-read input names: the attn_mask_reformat placeholder override
                // may have added a new external input to the graph.
                decoderInputNames = decoder.externalInputs();
            }

            long tAfterKvUpdate = System.nanoTime();

            if (trackMem) {
                long memAfterKv = NativeOpsHolder.getInstance().getDeviceNativeOps().getDeviceFreeMemory(0) / (1024 * 1024);
                log.info("  [PHASE_MEM] step={} phase=KV_UPDATE+SAMPLING after={}MB totalStepDelta={}MB",
                        step, memAfterKv, memBeforeInputBuild - memAfterKv);
            }

            // Sample next token via native GPU op — pass logits directly (rank-3 supported)
            long tSampStart = System.nanoTime();
            int nextTokenId;
            try {
                nextTokenId = reusableTokenSampler.sample(logitsRaw);
            } catch (Exception e) {
                throw decodeStageFailure("TOKEN_SAMPLE", step, pastSeqLen, currentSeqLen,
                        usingStaticKv, useDirect, cachePos, e);
            }
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

            // ── Warmup vs Steady-State classification ──────────────────────
            // A step is "warmup" when the DSP plan is still compiling or capturing
            // (phase < REPLAYING). These steps have compilation/capture overhead baked
            // into the wall-clock time and MUST NOT contaminate steady-state averages.
            boolean isWarmupStep = false;
            String phaseLabel = "STEADY";
            if (step > 0 && decoderDspExec != null) {
                PlanPhase phase = decoderDspExec.getPlanPhase();
                if (phase == null || !phase.isAtLeast(PlanPhase.REPLAYING)) {
                    isWarmupStep = true;
                    phaseLabel = "WARMUP";
                }
            } else if (step == 0) {
                // Step 0 is always prefill — never counted as warmup or steady
                phaseLabel = "PREFILL";
            }

            if (isWarmupStep) {
                warmupStepCount++;
                totalWarmupNs += stepElapsedNs;
                if (stepElapsedMs > maxWarmupStepMs) {
                    maxWarmupStepMs = stepElapsedMs;
                }
            } else if (step > 0 && firstSteadyStep < 0) {
                firstSteadyStep = step;
                log.info("  ======== WARMUP COMPLETE at step {} ========", step);
                log.info("  Warmup steps: {} (total {}ms, max single step {}ms)",
                        warmupStepCount, totalWarmupNs / 1_000_000, maxWarmupStepMs);
                log.info("  Steady-state execution begins (phase=REPLAYING)");
            }

            // Accumulate detailed timing for steady-state steps only (skip warmup)
            if (step >= 3 && !isWarmupStep) {
                totalInputBuildNs += tAfterInputBuild - stepStart;
                totalDecoderNs    += tAfterDecoder - tAfterInputBuild;
                totalLogitsDupNs  += tAfterLogitsDup - tAfterDecoder;
                totalKvUpdateNs   += tAfterKvUpdate - tAfterLogitsDup;
                totalSamplingNs   += stepElapsedNs - (tAfterKvUpdate - stepStart);
                detailSteps++;
            }
            if (step >= 20 && !isWarmupStep) {
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

            // Per-step argmax trace + reference-stream assertion (no-op when disabled).
            // Must run BEFORE the stop-token check so a divergent EOS is still caught.
            traceAndVerifyToken(step, nextTokenId, tokenText, logitsRaw, phaseLabel);

            // Reusable decode-step diagnostics: logit health, KV mutation, present KV health.
            // Auto-enables full diagnostics on accuracy failure (consecutive zero-logit or argmax=0).
            stepDiag.diagnoseStep(step, logitsRaw, decoderInputMap, decoderOutputs,
                    kvNames, resolvedIOConfig, nextTokenId);

            // Log every 10 steps or first 6, with [WARMUP] or [STEADY] label
            if (step < 6 || step % 10 == 0) {
                double currentTokPerSec = step > 0 && stepElapsedMs > 0 ? 1000.0 / stepElapsedMs : 0;
                if (step >= 2) {
                    long inputMs  = (tAfterInputBuild - stepStart) / 1_000_000;
                    long decMs    = (tAfterDecoder - tAfterInputBuild) / 1_000_000;
                    long dupMs    = (tAfterLogitsDup - tAfterDecoder) / 1_000_000;
                    long kvMs     = (tAfterKvUpdate - tAfterLogitsDup) / 1_000_000;
                    long sampMs   = stepElapsedMs - (tAfterKvUpdate - stepStart) / 1_000_000;
                    log.info("  [{}] Step {}: '{}' (id={}) {}ms ({} tok/s) [input={}ms dec={}ms dup={}ms kv={}ms samp={}ms cachePos={}]",
                            phaseLabel, step, tokenText, nextTokenId, stepElapsedMs,
                            String.format("%.1f", currentTokPerSec),
                            inputMs, decMs, dupMs, kvMs, sampMs, cachePos - 1);
                } else {
                    log.info("  [{}] Step {}: '{}' (id={}) {}ms ({} tok/s)",
                            phaseLabel, step, tokenText, nextTokenId, stepElapsedMs,
                            String.format("%.1f", currentTokPerSec));
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
                SameDiffMemoryUtils.safeClose(arr);
            }
            decoder.clearPlaceholders(false);

            // Get embedding for next token
            INDArray prevEmbeddings = currentEmbeddings;
            INDArray rowEmbed;
            try {
                rowEmbed = embeddingTable.getRow(nextTokenId).reshape(1, 1, resolvedHiddenSize);
            } catch (Exception e) {
                throw decodeStageFailure("TOKEN_EMBED_LOOKUP", step, pastSeqLen, currentSeqLen,
                        usingStaticKv, useDirect, cachePos, e);
            }
            if (useReusableBuffers) {
                // Reusable fixed-address embedding buffer: allocate once, update via .assign() each step.
                // Cross-stream CUDA event sync (b997c15894) ensures .assign() on the default stream
                // completes before graph replay launches on the DSP stream.
                if (reusableEmbeddings == null) {
                    reusableEmbeddings = rowEmbed.dup();
                } else {
                    if (pendingEmbedClose != null) {
                        SameDiffMemoryUtils.safeClose(pendingEmbedClose);
                        pendingEmbedClose = null;
                    }
                    INDArray contiguousEmbed = rowEmbed.dup();
                    reusableEmbeddings.assign(contiguousEmbed);
                    pendingEmbedClose = contiguousEmbed;
                }
                currentEmbeddings = reusableEmbeddings;
                if (reusableInputIds == null) {
                    reusableInputIds = Nd4j.createFromArray(new int[]{nextTokenId}).reshape(1, 1).castTo(DataType.LONG);
                } else {
                    reusableInputIds.putScalar(new int[]{0, 0}, nextTokenId);
                    Nd4j.getAffinityManager().ensureLocation(reusableInputIds,
                            org.nd4j.linalg.api.concurrency.AffinityManager.Location.DEVICE);
                }
                if (currentInputIds != null && currentInputIds != reusableInputIds) {
                    SameDiffMemoryUtils.safeClose(currentInputIds);
                }
                currentInputIds = reusableInputIds;
                if (prevEmbeddings != null && prevEmbeddings != reusableEmbeddings
                        && prevEmbeddings != prefillEmbeddings) {
                    SameDiffMemoryUtils.safeClose(prevEmbeddings);
                }
            } else {
                // Fresh buffer each step: address changes force CUDA graph replay to
                // detect drift and re-capture when needed. The reusable pattern above
                // caused graph replay to read stale data at step 3+ (f8e83ff4c9).
                currentEmbeddings = rowEmbed.dup();
                INDArray newTokenTensor = Nd4j.createFromArray(new int[]{nextTokenId}).reshape(1, 1).castTo(DataType.LONG);
                if (currentInputIds != null && currentInputIds != newTokenTensor && !currentInputIds.wasClosed()) {
                    currentInputIds.setCloseable(true);
                    currentInputIds.close();
                }
                currentInputIds = newTokenTensor;
                if (prevEmbeddings != null && prevEmbeddings != currentEmbeddings
                        && prevEmbeddings != prefillEmbeddings
                        && !prevEmbeddings.wasClosed()) {
                    prevEmbeddings.setCloseable(true);
                    prevEmbeddings.close();
                }
            }
            pastSeqLen += currentSeqLen;

            // After prefill (step 0), switch to speculative decode path if configured
            if (maxSpeculativeTokens > 0 && usingStaticKv && step == 0) {
                log.info("  Switching to speculative decode path (K={}, seqLen={})",
                        maxSpeculativeTokens, maxSpeculativeTokens + 1);
                // Clean up step 0's reusable inputs
                for (INDArray arr : reusableInputs.values()) {
                    SameDiffMemoryUtils.safeClose(arr);
                }
                reusableInputs.clear();
                // Close reusable fixed-address decode buffers
                SameDiffMemoryUtils.safeClose(pendingEmbedClose);
                SameDiffMemoryUtils.safeClose(reusableEmbeddings);
                SameDiffMemoryUtils.safeClose(reusableInputIds);
                // Close any residual non-reusable decode buffers
                if (currentEmbeddings != null && currentEmbeddings != reusableEmbeddings
                        && currentEmbeddings != prefillEmbeddings) {
                    SameDiffMemoryUtils.safeClose(currentEmbeddings);
                }
                if (currentInputIds != null && currentInputIds != reusableInputIds) {
                    SameDiffMemoryUtils.safeClose(currentInputIds);
                }
                reusableTokenSampler.close();
                // Re-query maxKvLen — it was -1 at loop entry (before prefill initialized the cache)
                long specMaxKvLen = kvCacheManager.getMaxKvLen();
                return decodeSpeculative(kvCacheManager, specMaxKvLen, cachePos,
                        resolvedIOConfig, embeddingTable, resolvedHiddenSize,
                        stopTokenIds, generatedTokens, prefillTimeMs, decodeStart,
                        promptTokenIds);
            }
        }

        // Release reusable input arrays
        for (INDArray arr : reusableInputs.values()) {
            SameDiffMemoryUtils.safeClose(arr);
        }
        reusableInputs.clear();
        // Close fixed-address reusable buffers (allocated once, reused across all decode steps)
        SameDiffMemoryUtils.safeClose(pendingEmbedClose);
        SameDiffMemoryUtils.safeClose(reusableEmbeddings);
        SameDiffMemoryUtils.safeClose(reusableInputIds);
        // currentEmbeddings and currentInputIds alias the reusable buffers after step 1;
        // close only if they still point to a different object (e.g. prefillEmbeddings, or
        // the old non-reusable buffers from before this fix).
        if (currentEmbeddings != null && currentEmbeddings != reusableEmbeddings
                && currentEmbeddings != prefillEmbeddings) {
            SameDiffMemoryUtils.safeClose(currentEmbeddings);
        }
        if (currentInputIds != null && currentInputIds != reusableInputIds) {
            SameDiffMemoryUtils.safeClose(currentInputIds);
        }
        reusableTokenSampler.close();

        // Release KV cache buffers via manager
        kvCacheManager.close();

        long totalDecodeMs = System.currentTimeMillis() - decodeStart;

        // Log phase breakdown (steady-state steps only — warmup excluded)
        if (detailSteps > 0) {
            log.info("=== PHASE BREAKDOWN (avg over {} steady-state steps, excludes {} warmup steps) ===",
                    detailSteps, warmupStepCount);
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
        log.info("COMPILATION / WARMUP SUMMARY:");
        log.info("  Warmup steps:      {} (steps 1-{})", warmupStepCount,
                firstSteadyStep > 0 ? (firstSteadyStep - 1) : warmupStepCount);
        log.info("  Total warmup time: {}ms", totalWarmupNs / 1_000_000);
        log.info("  Max warmup step:   {}ms (includes compilation/capture overhead)", maxWarmupStepMs);
        if (firstSteadyStep > 0) {
            log.info("  First steady step: {}", firstSteadyStep);
        } else if (warmupStepCount > 0) {
            log.info("  WARNING: all decode steps were warmup — no steady-state reached");
        }
        log.info("========================================");
        log.info("PERFORMANCE SUMMARY (steady-state only, {} steps, warmup excluded):", detailSteps);
        log.info("  Prefill (step 0):  {}ms", prefillTimeMs);
        log.info("  Decode tokens:     {} (excluding prefill)", decodeTokens);
        log.info("  Avg decode time:   {}ms/token (all steps incl. warmup)", String.format("%.1f", avgDecodeMs));
        log.info("  Decode throughput: {} tok/s (all steps incl. warmup)", String.format("%.2f", decodeTokensPerSec));
        if (steadyStateTokensPerSec > 0) {
            log.info("  Steady-state:      {} tok/s (warmup-excluded, {} steps)",
                    String.format("%.2f", steadyStateTokensPerSec), detailSteps);
        }
        if (lateSteadyTokensPerSec > 0) {
            log.info("  Late steady-state: {} tok/s (steps 20+, warmup-excluded)", String.format("%.2f", lateSteadyTokensPerSec));
        }
        log.info("  Latency P50/P90/P99: {}ms / {}ms / {}ms", p50Ms, p90Ms, p99Ms);
        log.info("  Total decode time: {}ms ({} tokens)", totalDecodeMs, generatedTokens.size());
        log.info("========================================");

        stepDiag.logSummary(generatedTokens.size());

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
                .warmupStepCount(warmupStepCount)
                .maxWarmupStepMs(maxWarmupStepMs)
                .totalWarmupMs(totalWarmupNs / 1_000_000)
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
            long prefillTimeMs, long decodeStart, int[] promptTokenIds) {

        int K = maxSpeculativeTokens;
        int mergedSeqLen = K + 1;

        // Use provided speculator (e.g., DraftModelSpeculator) or fall back to NgramSpeculator
        NgramSpeculator ngramFallback = this.speculator == null ? new NgramSpeculator(3, K) : null;

        // Create and compile frozen decode step with seqLen=K+1, using ModelIOConfig
        String logitsOutputName = specIOConfig.getLogitsOutputName();
        ModelIOConfig.KVCacheNames kvNames = specIOConfig.getKvCacheNames() != null
                ? specIOConfig.getKvCacheNames()
                : ModelIOConfig.findKVCacheOutputNames(decoder);
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
            for (int i = 0; i < mergedSeqLen; i++) {
                INDArray rowEmbed = embeddingTable.getRow(tokenArray[i]);
                mergedEmbeddings.get(NDArrayIndex.point(0), NDArrayIndex.point(i),
                        NDArrayIndex.all()).assign(rowEmbed);
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

            // 4. KV scatter for accepted + correction positions.
            // KvScatter runs as an ordinary op against the Java-owned static KV buffers.
            // The cache position lives in Java (KvCacheManager / local cachePos); the
            // DSP executor never sees or tracks it.
            int numToScatter = newTokens.size();
            frozenStep.scatterAcceptedKv(cachePos, numToScatter);
            cachePos += numToScatter;

            // 5. Update state
            int preBatchSize = generatedTokens.size();
            generatedTokens.addAll(newTokens);
            // Reference-stream assertion for the speculative path — verify each
            // newly-added token against the golden stream. No top-K trace (the
            // bulk-argmax path doesn't retain individual logit slices), but the
            // divergence still gets localized to a specific step.
            if (referenceTokenStream != null) {
                for (int i = 0; i < newTokens.size(); i++) {
                    int absStep = preBatchSize + i;
                    if (absStep >= referenceTokenStream.length) break;
                    int actual = newTokens.get(i);
                    int expected = referenceTokenStream[absStep];
                    if (expected != actual) {
                        String expectedText = "";
                        String actualText = "";
                        try {
                            expectedText = tokenizer.decode(new int[]{expected}, false);
                            actualText = tokenizer.decode(new int[]{actual}, false);
                        } catch (Exception ignore) {}
                        throw new TokenStreamDivergenceException(absStep, expected, expectedText,
                                actual, actualText, null, null);
                    }
                }
            }
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

    private void logDecodePhase(String phase, int step, String detail) {
        if (decodePhaseLoggingEnabled()) {
            log.info("  [PHASE] {} step={} {}", phase, step, detail);
        }
    }

    private IllegalStateException decodeStageFailure(String stage,
                                                     int step,
                                                     long pastSeqLen,
                                                     long currentSeqLen,
                                                     boolean usingStaticKv,
                                                     boolean useDirect,
                                                     long cachePos,
                                                     Throwable cause) {
        DynamicShapePlanExecutor dspExec = null;
        try {
            InferenceSession session = decoder.getOrCreateSession();
            dspExec = session.getDynamicShapePlanExecutor();
        } catch (Exception ignored) {
            // Preserve the original failure if session state is already broken.
        }

        String phase = dspExec != null ? String.valueOf(dspExec.getPlanPhase()) : "null";
        boolean shapesFrozen = dspExec != null && dspExec.isShapesFrozen();
        boolean pointersStable = dspExec != null && dspExec.arePointersStable();
        boolean hasPlan = dspExec != null && dspExec.getCurrentPlan() != null;

        return new IllegalStateException(
                "Decode stage " + stage + " failed: step=" + step
                        + " pastSeqLen=" + pastSeqLen
                        + " currentSeqLen=" + currentSeqLen
                        + " usingStaticKv=" + usingStaticKv
                        + " useDirect=" + useDirect
                        + " cachePos=" + cachePos
                        + " hasPlan=" + hasPlan
                        + " shapesFrozen=" + shapesFrozen
                        + " planPhase=" + phase
                        + " pointersStable=" + pointersStable,
                cause);
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
                                         ModelIOConfig.KVCacheNames kvNames) {
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

    // ═══════════════════════════════════════════════════════════════════════
    // Argmax tracing + reference token stream assertion
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Thrown by the decode loop when {@link #referenceTokenStream} is set and the
     * sampled token at a step does not match the reference. The exception carries
     * enough context (step, expected/actual token IDs, top-K logit snapshot) to
     * localize the divergence without re-running with extra logging enabled.
     */
    public static final class TokenStreamDivergenceException extends RuntimeException {
        public final int step;
        public final int expectedTokenId;
        public final String expectedText;
        public final int actualTokenId;
        public final String actualText;
        public final int[] topKIndices;
        public final float[] topKValues;

        TokenStreamDivergenceException(int step, int expectedTokenId, String expectedText,
                                       int actualTokenId, String actualText,
                                       int[] topKIndices, float[] topKValues) {
            super(buildMessage(step, expectedTokenId, expectedText, actualTokenId,
                    actualText, topKIndices, topKValues));
            this.step = step;
            this.expectedTokenId = expectedTokenId;
            this.expectedText = expectedText;
            this.actualTokenId = actualTokenId;
            this.actualText = actualText;
            this.topKIndices = topKIndices;
            this.topKValues = topKValues;
        }

        private static String buildMessage(int step, int expected, String expectedText,
                                           int actual, String actualText,
                                           int[] topKIndices, float[] topKValues) {
            StringBuilder sb = new StringBuilder();
            sb.append("Decode token stream diverged at step ").append(step)
                    .append(": expected id=").append(expected).append(" '").append(expectedText)
                    .append("', got id=").append(actual).append(" '").append(actualText).append("'");
            if (topKIndices != null && topKValues != null) {
                sb.append(" top-K=[");
                for (int i = 0; i < topKIndices.length; i++) {
                    if (i > 0) sb.append(", ");
                    sb.append(topKIndices[i]).append(":")
                            .append(String.format("%.4f", topKValues[i]));
                }
                sb.append("]");
            }
            return sb.toString();
        }
    }

    /**
     * Per-step tracer: logs the sampled token + top-K logit snapshot when tracing
     * is enabled, and asserts against {@link #referenceTokenStream} when set.
     *
     * <p>The top-K scan runs on a host-side copy of the logits row only — it is
     * skipped entirely (no dup, no transfer) when neither tracing nor reference
     * checking is active, so the hot path stays unaffected.</p>
     *
     * <p>Reads {@code vlm.benchmark.argmaxTrace=true} as a runtime override so a
     * test run can enable tracing without touching caller code.</p>
     */
    private void traceAndVerifyToken(int step, int nextTokenId, String tokenText,
                                     INDArray logitsRaw, String phaseLabel) {
        boolean traceEnabled = argmaxTraceEnabled
                || Boolean.parseBoolean(System.getProperty("vlm.benchmark.argmaxTrace", "false"));
        boolean refCheckEnabled = referenceTokenStream != null
                && step < referenceTokenStream.length;

        if (!traceEnabled && !refCheckEnabled) return;

        int topK = Math.max(1, argmaxTraceTopK);

        // Extract the logits row for the last position (rank-3 -> [B,S,V], rank-2 -> [B,V]).
        INDArray flat;
        if (logitsRaw.rank() == 3) {
            long lastSeq = logitsRaw.size(1) - 1;
            flat = logitsRaw.get(
                    NDArrayIndex.point(0),
                    NDArrayIndex.point(lastSeq),
                    NDArrayIndex.all()).dup();
        } else if (logitsRaw.rank() == 2) {
            long lastRow = logitsRaw.size(0) - 1;
            flat = logitsRaw.getRow(lastRow).dup();
        } else {
            flat = logitsRaw.dup();
        }

        // On-host top-K via a single pass + insertion sort (K is small, V ~50K).
        int vocab = (int) flat.length();
        float[] topVals = new float[topK];
        int[] topIdx = new int[topK];
        Arrays.fill(topVals, Float.NEGATIVE_INFINITY);
        Arrays.fill(topIdx, -1);
        // `flat` is already a dup() — single bulk D2H transfer, no per-element JNI overhead.
        float[] host = flat.data().asFloat();
        for (int v = 0; v < vocab; v++) {
            float val = host[v];
            if (val > topVals[topK - 1]) {
                int pos = topK - 1;
                while (pos > 0 && topVals[pos - 1] < val) {
                    topVals[pos] = topVals[pos - 1];
                    topIdx[pos] = topIdx[pos - 1];
                    pos--;
                }
                topVals[pos] = val;
                topIdx[pos] = v;
            }
        }

        if (flat.closeable()) {
            flat.close();
        }

        if (traceEnabled) {
            StringBuilder topKStr = new StringBuilder("[");
            for (int k = 0; k < topK; k++) {
                if (k > 0) topKStr.append(", ");
                topKStr.append("id=").append(topIdx[k])
                        .append(" v=").append(String.format("%.4f", topVals[k]));
            }
            topKStr.append("]");
            boolean argmaxMatchesTop1 = topIdx[0] == nextTokenId;
            log.info("  [{}|ARGMAX] step={} picked=id={} '{}' top1Match={} topK={}",
                    phaseLabel, step, nextTokenId, tokenText, argmaxMatchesTop1, topKStr);
        }

        if (refCheckEnabled) {
            int expected = referenceTokenStream[step];
            if (expected != nextTokenId) {
                String expectedText = "";
                try {
                    expectedText = tokenizer.decode(new int[]{expected}, false);
                } catch (Exception ignore) {
                    // decode failures should not mask the real mismatch
                }
                throw new TokenStreamDivergenceException(step, expected, expectedText,
                        nextTokenId, tokenText, topIdx, topVals);
            }
        }
    }
}
