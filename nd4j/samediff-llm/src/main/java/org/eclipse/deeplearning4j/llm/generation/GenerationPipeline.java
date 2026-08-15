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
import org.eclipse.deeplearning4j.llm.generation.constraint.ConstraintConfig;
import org.eclipse.deeplearning4j.llm.generation.constraint.ConstraintMasker;
import org.eclipse.deeplearning4j.llm.generation.sampling.Sampler;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplerUtils;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.generation.speculative.Speculator;
import org.eclipse.deeplearning4j.llm.generation.speculative.DraftModelSpeculator;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfig;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfigApplier;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.config.ExecutionResult;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.common.config.ND4JSystemProperties;

import org.nd4j.linalg.api.ops.impl.transforms.custom.AutoregressiveDecode;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRule;
import org.nd4j.linalg.api.ops.impl.transforms.custom.KVCacheQuantize;
import org.nd4j.linalg.api.ops.impl.transforms.custom.KVCacheDequantize;
import org.eclipse.deeplearning4j.llm.generation.kvcache.KvCacheStrategy;
import org.eclipse.deeplearning4j.llm.generation.kvcache.KvPrefixBlockPool;
import org.eclipse.deeplearning4j.llm.generation.kvcache.PrefixLookupResult;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.execution.DspCompilationMode;
import org.nd4j.autodiff.samediff.execution.DspDebugger;
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.DynamicShapeSlot;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.nativeblas.OpaqueDataBuffer;
import org.bytedeco.javacpp.Pointer;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;

/**
 * Unified LLM inference pipeline that eliminates all manual wiring boilerplate.
 *
 * <p>Handles model loading (from paths or pre-loaded instances), I/O auto-discovery,
 * embedding table extraction, tokenization, and decode loop construction. Users
 * interact with a simple text-in/text-out API:</p>
 *
 * <pre>{@code
 * // With pre-loaded models
 * GenerationPipeline pipeline = GenerationPipeline.create(
 *     GenerationPipelineConfig.builder()
 *         .decoder(decoderSd)
 *         .embedTokens(embedTokensSd)
 *         .tokenizer(myTokenizer)
 *         .build());
 *
 * GenerationResult result = pipeline.generate("Explain transformers.");
 * System.out.println(result.getText());
 * pipeline.close();
 *
 * // With paths (SDZ natively, ONNX via custom modelLoader)
 * GenerationPipeline pipeline = GenerationPipeline.create(
 *     GenerationPipelineConfig.builder()
 *         .decoderPath("/models/decoder.sdz")
 *         .embedTokensPath("/models/embed_tokens.sdz")
 *         .tokenizer(myTokenizer)
 *         .modelLoader(path -> myOnnxImporter.importModel(path))
 *         .build());
 * }</pre>
 *
 * <p>Decoding is always performed by the native {@code autoregressive_decode} C++ op
 * ({@link AutoregressiveDecode}), which runs the full decode loop in C++ and eliminates
 * per-step Java&lt;&rarr;C++ round-trips.</p>
 *
 * <p>For VLM (vision-language model) use cases, call
 * {@link #generate(INDArray, int[])} with pre-built embeddings that include
 * merged vision and text tokens.</p>
 *
 * <h2>Continue generation (resume / incremental decode)</h2>
 * <p>For single-model in-graph-KV (GGUF) decoders, {@link #startSession(String, int)} returns a
 * {@link GenerationSession} that <em>resumes</em> autoregressive decoding across multiple calls,
 * reusing the already-populated in-graph KV cache — no session reset, no re-prefill — bounded by the
 * pre-sized KV buffer (the model's context ceiling). This lets a caller self-heal a truncated result
 * (finishReason {@code MAX_TOKENS}) by continuing into the unused context budget instead of
 * re-prefilling {@code prompt + textSoFar}:</p>
 * <pre>{@code
 * try (GenerationSession s = pipeline.startSession(prompt, contextLen - promptLen)) {
 *     GenerationResult r = s.generate(64);
 *     while (r.isTruncated() && s.getRemainingCapacity() > 0) r = s.continueGeneration(64);
 *     String full = s.getFullText();
 * }
 * }</pre>
 * <p>For greedy decoding with the default repetition penalty, one logical generation spread over K
 * session calls is token-for-token identical to a single {@code generate()} of the summed budget.
 * A session is thread-confined and only one may be open per pipeline at a time. See
 * {@link #startSession(String, int)} for the full contract.</p>
 *
 * @see GenerationPipelineConfig
 * @see AutoregressiveDecode
 * @see GenerationResult
 * @see ModelIOConfig
 * @see GenerationSession
 */
@Slf4j
public class GenerationPipeline implements AutoCloseable {

    // ---- Core components ----

    @Getter
    private final SameDiff decoder;

    /** ADR 0107 V2: original KV placeholder dtypes/shapes recorded before the INT8 (row-inline)
     *  conversion in prefillWarmupAndFreeze — restored in close() because the decoder SameDiff
     *  may be shared with later pipelines using a non-quantised strategy. */
    private Map<String, DataType> kvPlaceholderOriginalDtypes;
    private Map<String, long[]> kvPlaceholderOriginalShapes;

    @Getter
    private final SameDiff embedTokens;

    private final Tokenizer tokenizer;

    /** Metadata retained by the model importer instead of being discarded with the source container. */
    private final ModelMetadata modelMetadata;

    /** Tokenizer/importer-owned control-token vocabulary used by constrained decoding. */
    private final Set<Integer> specialTokenIds;

    /** Decoded control-token lexemes, retained because added tokens may not support string-to-id lookup. */
    private final List<String> specialTokenPieces;

    /** Opt-in raw versus post-constraint candidate observer; disabled by default and non-mutating. */
    private final ConstraintCandidateDiagnostics constraintCandidateDiagnostics;

    @Getter
    private final ModelIOConfig ioConfig;

    private final GenerationPipelineConfig config;

    /**
     * The active sampling configuration, applied to every subsequent {@code generate} / {@code startSession}
     * call. Initialized from {@link GenerationPipelineConfig#getSamplingConfig()} and mutable at runtime via
     * {@link #setSamplingConfig(SamplingConfig)}, so one loaded pipeline (a single multi-GB model load) can be
     * driven with many sampling strategies without rebuilding. Sampling is pure logits post-processing and
     * never touches graph shapes or the frozen DSP plan, so swapping it between generations is free.
     *
     * <p>{@code volatile} for safe publication; generation on a pipeline is single-threaded (it binds one
     * frozen DSP plan + KV buffer set — see {@link GenerationSession}). A change takes effect on the next
     * generation; an already-open session keeps the sampler it captured at {@code startSession}
     * (see {@link InGraphKvState#sampling}).</p>
     */
    private volatile SamplingConfig activeSamplingConfig;

    /** Template-owned assistant turn terminators active only during generateChat(). */
    private volatile Set<Integer> activeChatStopTokenIds = Collections.emptySet();

    private enum DecodePolicyKind {
        GREEDY,
        SAMPLE,
        SPECULATIVE,
        CONTRASTIVE,
        BEAM
    }

    private static final class DecodePolicy {
        final DecodePolicyKind kind;
        final int batchMax;
        final int windowMax;
        final boolean needsHiddenState;

        DecodePolicy(DecodePolicyKind kind, int batchMax, int windowMax, boolean needsHiddenState) {
            this.kind = kind;
            this.batchMax = batchMax;
            this.windowMax = windowMax;
            this.needsHiddenState = needsHiddenState;
        }

        boolean isScalarNativePolicy() {
            return batchMax == 1 && windowMax == 1 && !needsHiddenState;
        }
    }

    // ---- Derived state ----

    /** Embedding weight table for direct token-to-embedding lookup (bypasses SameDiff.output()). */
    private final INDArray embeddingTable;

    /** Resolved hidden size of the model. */
    private final long hiddenSize;

    /** Input variable name for the embed_tokens model. */
    private final String embedInputName;

    /** Output variable names for the embed_tokens model. */
    private final String[] embedOutputNames;

    /** Pre-loaded draft decoder for speculative decoding (null if disabled). */
    private final SameDiff draftDecoder;

    /** Whether the pipeline owns the loaded models (and should close them). */
    private final boolean ownsDecoder;
    private final boolean ownsEmbedTokens;
    private final boolean ownsDraftDecoder;

    /**
     * The single active continuation {@link GenerationSession} on this pipeline, or {@code null}.
     * Lock-free coordination: {@code startSession} CAS-sets {@code null → session};
     * {@link GenerationSession#close()} CAS-clears it. Only one session may be open at a time because
     * it binds the decoder's single frozen DSP plan + KV buffers. See the concurrency contract on
     * {@link GenerationSession}.
     */
    private final AtomicReference<GenerationSession> activeSession = new AtomicReference<>(null);
    private static final AtomicLong SESSION_ID_COUNTER = new AtomicLong(0);
    private static final String ACTUAL_SEQUENCE_LENGTH_NAME = "actual_sequence_length";
    private static final String TARGET_HIDDEN_STATES_NAME = "target_hidden_states";
    private static final String MTP_INPUT_IDS_NAME = "mtp_input_ids";
    private static final String MTP_TARGET_HIDDEN_NAME = "mtp_target_hidden_states";
    private static final String MTP_POSITION_OFFSET_NAME = "mtp_position_offset";
    private static final String MTP_CACHE_POSITION_NAME = "mtp_cache_position";
    private static final String MTP_CAUSAL_MASK_NAME = "mtp_causal_mask";
    private static final String MTP_KEY_CACHE_NAME = "mtp_past_key_values.0.key";
    private static final String MTP_VALUE_CACHE_NAME = "mtp_past_key_values.0.value";
    private static final String MTP_KEY_STATES_NAME = "mtp_key_states";
    private static final String MTP_VALUE_STATES_NAME = "mtp_value_states";
    private static final String MTP_HIDDEN_STATES_NAME = "mtp_hidden_states";
    private static final String MTP_LOGITS_NAME = "mtp_logits";

    /**
     * Fixed-buffer decode state retained between independent one-shot generations and continuation
     * sessions. Ownership transfers back to the pipeline when a session closes, then to the next
     * compatible caller. Keeping the same state preserves the frozen DSP plan, captured CUDA graph,
     * and every external-input device address; the next prefill overwrites the retained buffers in
     * place. Only {@code maxPrefillLength > 0} populates this cache. Freed in {@link #close()}.
     * Thread-confined to the pipeline's decode thread.
     */
    private InGraphKvState cachedFixedBufferState;

    /**
     * Cross-request KV prefix block pool. Non-null when
     * {@link GenerationPipelineConfig#isPrefixCacheEnabled()} is {@code true}. Shared
     * across all generate() and startSession() calls on this pipeline instance. Freed in
     * {@link #close()}.
     */
    private final KvPrefixBlockPool prefixBlockPool;

    private GenerationPipeline(
            SameDiff decoder, boolean ownsDecoder,
            SameDiff embedTokens, boolean ownsEmbedTokens,
            Tokenizer tokenizer,
            ModelMetadata modelMetadata,
            ModelIOConfig ioConfig,
            INDArray embeddingTable,
            long hiddenSize,
            String embedInputName, String[] embedOutputNames,
            SameDiff draftDecoder, boolean ownsDraftDecoder,
            GenerationPipelineConfig config,
            KvPrefixBlockPool prefixBlockPool) {
        this.decoder = decoder;
        this.ownsDecoder = ownsDecoder;
        this.embedTokens = embedTokens;
        this.ownsEmbedTokens = ownsEmbedTokens;
        this.tokenizer = tokenizer;
        this.modelMetadata = modelMetadata == null ? ModelMetadata.empty() : modelMetadata;
        Set<Integer> protocolTokens = new HashSet<>(tokenizer.getSpecialTokenIds());
        protocolTokens.addAll(this.modelMetadata.getSpecialTokenIds());
        Map<String, Integer> addedTokens = tokenizer.getAddedTokens();
        Integer nativeStartId = addedTokens.get(
                org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.NATIVE_TOOL_CALL_START);
        Integer nativeEndId = addedTokens.get(
                org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.NATIVE_TOOL_CALL_END);
        if (nativeStartId != null && nativeStartId >= 0
                && nativeEndId != null && nativeEndId >= 0) {
            protocolTokens.add(nativeStartId);
            protocolTokens.add(nativeEndId);
        }
        this.specialTokenIds = Collections.unmodifiableSet(protocolTokens);
        this.specialTokenPieces = decodeSpecialTokenPieces(tokenizer, protocolTokens);
        this.constraintCandidateDiagnostics =
                ConstraintCandidateDiagnostics.fromSystemProperties();
        this.ioConfig = ioConfig;
        this.embeddingTable = embeddingTable;
        this.hiddenSize = hiddenSize;
        this.embedInputName = embedInputName;
        this.embedOutputNames = embedOutputNames;
        this.draftDecoder = draftDecoder;
        this.ownsDraftDecoder = ownsDraftDecoder;
        this.config = config;
        this.prefixBlockPool = prefixBlockPool;
        // Sampling is pure logits post-processing (never touches graph shapes / the DSP plan), so the
        // active sampling config is runtime-mutable — one model load can serve many sampling strategies.
        this.activeSamplingConfig = config.getSamplingConfig();

        // Enable DSP auto-compile on models if requested.
        if (config.isDspEnabled()) {
            enableDspIfConfigured(decoder, "decoder");
            if (embedTokens != null) {
                enableDspIfConfigured(embedTokens, "embedTokens");
            }
        } else {
            disableDsp(decoder, "decoder");
            if (embedTokens != null) {
                disableDsp(embedTokens, "embedTokens");
            }
        }
    }

    private static void disableDsp(SameDiff model, String label) {
        model.setDspAutoCompileEnabled(false);
        model.setDspNativeAutoCompileEnabled(false);
        log.info("DSP auto-compile disabled on {} by GenerationPipelineConfig", label);
    }

    private static void enableDspIfConfigured(SameDiff model, String label) {
        boolean noFreeze = Boolean.parseBoolean(
                System.getProperty(ND4JSystemProperties.DSP_NO_FREEZE, "false"));
        if (!noFreeze) {
            model.setDspAutoCompileEnabled(true);
            model.setDspNativeAutoCompileEnabled(true);
            log.info("DSP auto-compile enabled (native=true) on {} ({}={})",
                    label, ND4JSystemProperties.DSP_NO_FREEZE, noFreeze);
        } else {
            log.info("DSP auto-compile disabled on {} ({}=true)",
                    label, ND4JSystemProperties.DSP_NO_FREEZE);
        }
    }

    // ==================== Factory ====================

    /**
     * Create a GenerationPipeline from a configuration.
     *
     * <p>This method performs the following steps:</p>
     * <ol>
     *   <li>Load models from paths (if not pre-loaded) via ONNX import with SDZ caching</li>
     *   <li>Auto-discover I/O variable names via {@link ModelIOConfig#discover(SameDiff)}</li>
     *   <li>Auto-detect hidden size from the model if not specified</li>
     *   <li>Extract the embedding table from the embed_tokens model for direct lookup</li>
     *   <li>Load draft decoder for speculative decoding if configured</li>
     * </ol>
     *
     * @param config pipeline configuration
     * @return a fully initialized GenerationPipeline
     * @throws IOException if model loading fails
     * @throws IllegalArgumentException if required configuration is missing
     */
    public static GenerationPipeline create(GenerationPipelineConfig config) throws IOException {
        if (config.getTokenizer() == null) {
            throw new IllegalArgumentException("Tokenizer is required");
        }

        // 1. Load or use pre-loaded models (in parallel when both need loading from paths)
        ModelLoader modelLoader = config.getModelLoader();
        SameDiff decoder = config.getDecoder();
        boolean ownsDecoder = false;
        SameDiff embedTokens = config.getEmbedTokens();
        boolean ownsEmbedTokens = false;

        boolean needsDecoderLoad = decoder == null;
        boolean needsEmbedLoad = embedTokens == null && config.getEmbedTokensPath() != null;

        if (needsDecoderLoad) {
            if (config.getDecoderPath() == null) {
                throw new IllegalArgumentException("Either decoder or decoderPath must be provided");
            }
        }

        if (needsDecoderLoad && needsEmbedLoad) {
            // Load both models in parallel — they are independent I/O operations
            long loadStart = System.currentTimeMillis();
            CompletableFuture<SameDiff> decoderFuture =
                    CompletableFuture.supplyAsync(() -> {
                        try {
                            return loadModel(config.getDecoderPath(), modelLoader);
                        } catch (IOException e) {
                            throw new CompletionException(e);
                        }
                    });
            CompletableFuture<SameDiff> embedFuture =
                    CompletableFuture.supplyAsync(() -> {
                        try {
                            return loadModel(config.getEmbedTokensPath(), modelLoader);
                        } catch (IOException e) {
                            throw new CompletionException(e);
                        }
                    });
            try {
                decoder = decoderFuture.join();
                ownsDecoder = true;
                embedTokens = embedFuture.join();
                ownsEmbedTokens = true;
            } catch (CompletionException e) {
                Throwable cause = e.getCause();
                throw cause instanceof IOException ? (IOException) cause
                        : new IOException("Parallel model loading failed", cause);
            }
            log.info("Parallel model loading (decoder + embedTokens) completed in {}ms",
                    System.currentTimeMillis() - loadStart);
        } else {
            if (needsDecoderLoad) {
                decoder = loadModel(config.getDecoderPath(), modelLoader);
                ownsDecoder = true;
            }
            if (needsEmbedLoad) {
                embedTokens = loadModel(config.getEmbedTokensPath(), modelLoader);
                ownsEmbedTokens = true;
            }
        }
        // embedTokens may be null here — single-model mode uses decoder for embeddings

        ModelMetadata modelMetadata = ModelMetadata.empty();
        if (modelLoader != null && config.getDecoderPath() != null) {
            ModelMetadata imported = modelLoader.getModelMetadata(config.getDecoderPath());
            if (imported != null) {
                modelMetadata = imported;
            }
        }

        // 1b. Run the complete default graph optimizer pipeline. This includes automatic
        // FP16 weight pre-casting; nd4j.optimizer.fp16=false is the explicit opt-out.
        if (config.isGraphOptimizerEnabled()) {
            int opsBefore = decoder.getOps().size();
            long optStart = System.currentTimeMillis();
            List<String> outputs = decoder.outputs() != null
                    ? new ArrayList<>(decoder.outputs()) : new ArrayList<>();
            SameDiff originalDecoder = decoder;
            boolean ownsOriginalDecoder = ownsDecoder;
            SameDiff optimizedDecoder = GraphOptimizer.optimize(decoder, outputs);
            if (optimizedDecoder != originalDecoder) {
                if (ownsOriginalDecoder) {
                    try {
                        SameDiffMemoryUtils.freeModelArrays(originalDecoder);
                        originalDecoder.close();
                    } catch (Exception e) {
                        log.warn("Error closing pre-optimization decoder: {}", e.getMessage());
                    }
                }
                ownsDecoder = true;
            }
            decoder = optimizedDecoder;
            long optMs = System.currentTimeMillis() - optStart;
            log.info("GraphOptimizer: {} -> {} ops ({} removed) in {}ms",
                    opsBefore, decoder.getOps().size(), opsBefore - decoder.getOps().size(), optMs);
        }

        // 2. Auto-discover I/O names
        ModelIOConfig ioConfig = config.getIoConfig();
        if (ioConfig == null) {
            ioConfig = ModelIOConfig.discover(decoder);
        }

        // 2b. Validate logits output name against actual decoder outputs and fix up if wrong.
        // This guards against the @Builder.Default "lm_logits" being used for models that
        // export "logits" (e.g. Gemma, OLMo, LFM2, OpenELM via GGUF), and against any
        // explicit ioConfig that was built before the model's output names were known.
        {
            String currentLogits = ioConfig.getLogitsOutputName();
            List<String> decoderOutputs = decoder.outputs();
            if (currentLogits == null || (!decoderOutputs.contains(currentLogits) && !decoderOutputs.isEmpty())) {
                String discovered = ModelIOConfig.findLogitsOutputName(decoder);
                if (discovered != null && !discovered.equals(currentLogits)) {
                    log.warn("GenerationPipeline: logits output name '{}' not found in decoder outputs {}; "
                            + "using auto-discovered name '{}'", currentLogits, decoderOutputs, discovered);
                    ioConfig = ModelIOConfig.builder()
                            .inputEmbeddingsName(ioConfig.getInputEmbeddingsName())
                            .inputIdsName(ioConfig.getInputIdsName())
                            .attentionMaskName(ioConfig.getAttentionMaskName())
                            .causalMaskName(ioConfig.getCausalMaskName())
                            .positionIdsName(ioConfig.getPositionIdsName())
                            .positionOffsetName(ioConfig.getPositionOffsetName())
                            .cachePositionName(ioConfig.getCachePositionName())
                            .kvCachePrefix(ioConfig.getKvCachePrefix())
                            .kvPresentToInputReplace(ioConfig.getKvPresentToInputReplace())
                            .logitsOutputName(discovered)
                            .kvCacheNames(ioConfig.getKvCacheNames())
                            .encoderHiddenStatesName(ioConfig.getEncoderHiddenStatesName())
                            .encoderAttentionMaskName(ioConfig.getEncoderAttentionMaskName())
                            .encoderDecoder(ioConfig.isEncoderDecoder())
                            .attnMaskReformatOutput(ioConfig.getAttnMaskReformatOutput())
                            .build();
                }
            }
        }

        // 3. Resolve embed_tokens input/output names
        SameDiff embedSource = embedTokens != null ? embedTokens : decoder;
        String embedInputName = embedSource.inputs().isEmpty()
                ? (ioConfig.getInputIdsName() != null ? ioConfig.getInputIdsName() : "input_ids")
                : embedSource.inputs().get(0);
        String[] embedOutputNames = embedSource.outputs().toArray(new String[0]);

        // 4. Extract embedding table for direct lookup (from embedTokens or decoder)
        //    Done before hidden size detection so we can reuse the result (avoids scanning all variables twice).
        INDArray embeddingTable = extractEmbeddingTable(embedSource);

        // 5. Auto-detect hidden size (use embedTokens if available, else decoder)
        long resolvedHiddenSize = config.getHiddenSize();
        if (resolvedHiddenSize <= 0) {
            resolvedHiddenSize = detectHiddenSize(embedSource, embeddingTable);
        }

        // 6. Load draft decoder for speculative decoding
        SameDiff draftDecoder = config.getDraftDecoder();
        boolean ownsDraftDecoder = false;
        if (draftDecoder == null && config.getDraftModelPath() != null) {
            draftDecoder = loadModel(config.getDraftModelPath(), modelLoader);
            ownsDraftDecoder = true;
        }

        // Speculative decoding is represented in the unified SamplingConfig strategy surface, but the
        // native masked multi-position substrate is still the execution gate. A caller that selects
        // DecodeStrategy.SPECULATIVE will fail fast at decode-policy resolution instead of silently
        // running the scalar path with zeroed speculation metrics.
        if (config.getMaxSpeculativeTokens() > 0
                || config.getSpeculator() != null
                || draftDecoder != null) {
            log.warn("Speculative decoding configured (maxSpeculativeTokens={}, draft={}, speculator={}). "
                            + "Select SamplingConfig.speculative() once the ADR 0106 native masked "
                            + "multi-position substrate is available; until then the policy resolver "
                            + "will reject SPECULATIVE rather than fall back silently.",
                    config.getMaxSpeculativeTokens(),
                    draftDecoder != null ? "present" : "none",
                    config.getSpeculator() != null ? config.getSpeculator().getClass().getSimpleName() : "none");
        }

        log.info("GenerationPipeline created: decoder ops={}, embedTokens={}, hiddenSize={}, "
                        + "embeddingTable={}, draftDecoder={}, kvStrategy={}, dsp={}",
                decoder.getOps().size(),
                embedTokens != null ? embedTokens.getOps().size() + " ops" : "single-model mode (using decoder)",
                resolvedHiddenSize,
                embeddingTable != null ? Arrays.toString(embeddingTable.shape()) : "null (fallback to SameDiff.output())",
                draftDecoder != null ? draftDecoder.getOps().size() + " ops" : "disabled",
                config.getKvCacheStrategy(),
                config.isDspEnabled());

        // Build cross-request KV prefix block pool if enabled.
        KvPrefixBlockPool prefixBlockPool = null;
        if (config.isPrefixCacheEnabled()) {
            // Validate mutually-exclusive options
            if (config.isRotatingKvEnabled()) {
                throw new IllegalArgumentException(
                        "prefixCacheEnabled=true is not compatible with rotatingKvEnabled=true "
                        + "(rotating KV changes physical write positions per-step; prefix blocks are "
                        + "position-indexed and cannot be shared across rotation boundaries in v1). "
                        + "Disable one or the other.");
            }
            if (config.getKvCacheStrategy() == KvCacheStrategy.QUANTIZED
                    || config.getKvCacheStrategy() == KvCacheStrategy.TURBOQUANT) {
                throw new IllegalArgumentException(
                        "prefixCacheEnabled=true is not compatible with KvCacheStrategy."
                        + config.getKvCacheStrategy() + " in v1. Only STATIC strategy is supported. "
                        + "QUANTIZED/TURBOQUANT store data in a separate compressed format whose "
                        + "per-row scale indices are dependent on the full buffer layout, making "
                        + "block-level reuse incorrect without dequantize+requantize round-trips.");
            }
            int resolvedBlockSize = config.getPrefixCacheBlockSize() > 0
                    ? config.getPrefixCacheBlockSize()
                    : KvPrefixBlockPool.DEFAULT_BLOCK_SIZE;
            long resolvedMaxBytes = config.getPrefixCacheMaxBytes() > 0
                    ? config.getPrefixCacheMaxBytes()
                    : resolvePrefixCacheDefaultBytes();
            // Rough bytes-per-block estimate: blockSize * numKVLayers * kvHeads * headDim * 4 bytes.
            // We don't know the model shape yet, so use a conservatively large placeholder.
            // The trie uses this only for a saved-memory log message — it does not gate eviction.
            long bytesPerBlockEstimate = resolvedBlockSize * 4096L;
            prefixBlockPool = new KvPrefixBlockPool(
                    resolvedBlockSize, resolvedMaxBytes, bytesPerBlockEstimate,
                    KvPrefixBlockPool.DEFAULT_MAX_CACHE_ENTRIES);
            log.info("KvPrefixBlockPool created: blockSize={} maxBytes={}MB",
                    resolvedBlockSize, resolvedMaxBytes / (1024 * 1024));
        }

        GenerationPipeline pipeline = new GenerationPipeline(
                decoder, ownsDecoder,
                embedTokens, ownsEmbedTokens,
                config.getTokenizer(),
                modelMetadata,
                ioConfig,
                embeddingTable,
                resolvedHiddenSize,
                embedInputName, embedOutputNames,
                draftDecoder, ownsDraftDecoder,
                config,
                prefixBlockPool);

        BenchmarkConfig benchmarkConfig = config.getBenchmarkConfig();
        if (benchmarkConfig == null) {
            benchmarkConfig = BenchmarkConfig.optimal();
            log.info("No BenchmarkConfig provided — using default optimal config (Triton + CUDA graph capture)");
        }
        log.info("Applying BenchmarkConfig.{} to optimized GenerationPipeline models", benchmarkConfig.getName());
        BenchmarkConfigApplier.apply(benchmarkConfig);
        // Set execution mode to TRITON on models WITHOUT creating a native plan.
        // Calling compileModels() here would create plans with no input data,
        // causing the Triton JIT to produce zero compiled kernels. The decode-shape
        // plan (created later by auto-compile during the first execution with real
        // shapes) would then find no Triton cache entries for its shape key, leaving
        // all 2419 slots as gaps. Gap range 2419 > maxCapturableGapSlots (32) blocks
        // native-only capture, and composite capture finds no Triton islands → zero
        // CUDA graph nodes → ZERO_KERNEL_SBS terminal → permanent slot-by-slot at
        // ~19 tok/s instead of ~65 tok/s.
        //
        // Fix: set TRITON mode + environment flags via setDspCompilationMode() so
        // auto-compile creates the plan during the warmup decode with real decode
        // shapes. The segment lifecycle then proceeds: WARMUP → COMPILE (Triton JIT
        // with actual data) → CAPTURE (CUDA graph with Triton islands).
        if (benchmarkConfig.isTriton()) {
            decoder.setDspCompilationMode(DspCompilationMode.MAX_AUTOTUNE);
            if (embedTokens != null) {
                embedTokens.setDspCompilationMode(DspCompilationMode.MAX_AUTOTUNE);
            }
            log.info("  Triton mode configured on decoder{} — compilation deferred to first execution with real shapes",
                    embedTokens != null ? " and embed_tokens" : "");
        } else {
            // Non-Triton configs (CUDA_GRAPHS, SLOT_BY_SLOT, etc.) can compile
            // eagerly since they don't depend on Triton JIT with shape-keyed caches.
            if (embedTokens != null) {
                BenchmarkConfigApplier.compileModels(
                        decoder, "decoder",
                        embedTokens, "embed_tokens",
                        benchmarkConfig);
            } else {
                List<String> decoderOutputs = decoder.outputs() != null
                        ? new ArrayList<>(decoder.outputs()) : new ArrayList<>();
                BenchmarkConfigApplier.compileModel(decoder, "decoder", decoderOutputs, benchmarkConfig);
            }
        }

        return pipeline;
    }

    // ==================== Simple API ====================

    /**
     * Generate text from a text prompt using default max tokens.
     *
     * <p>Tokenizes the prompt, embeds tokens via the embedding table (or embed_tokens model),
     * and runs autoregressive decoding via the native {@code autoregressive_decode} C++ op.</p>
     *
     * @param prompt the input text prompt
     * @return generation result with text, token IDs, timing, and throughput metrics
     */
    public GenerationResult generate(String prompt) {
        return generate(prompt, config.getMaxNewTokens());
    }

    /**
     * Generate text from a text prompt with a specified max token count.
     *
     * <p>Tokenizes the prompt, embeds tokens, and runs the native
     * {@code autoregressive_decode} C++ op.</p>
     *
     * @param prompt the input text prompt
     * @param maxNewTokens maximum number of tokens to generate
     * @return generation result with text, token IDs, timing, and throughput metrics
     */
    public GenerationResult generate(String prompt, int maxNewTokens) {
        return generateTokenIds(encodePromptToIds(prompt), maxNewTokens);
    }

    private GenerationResult generateTokenIds(int[] promptTokenIds, int maxNewTokens) {
        int restoreDevice = switchToDecoderDevice("text-generation");
        // Suppress cross-device routing for the entire generation. Model weights,
        // prompt tensors, and KV caches must stay on the decoder's execution device.
        // The CUDA async memory pool reports pool-reserved memory as "used" to
        // cudaMemGetInfo, causing false OOM detection and unnecessary cross-device
        // routing. The pool handles actual OOM via trim+retry.
        OpaqueDataBuffer.suppressCrossDeviceRouting(true);
        try {
            return generateInternal(promptTokenIds, maxNewTokens);
        } finally {
            OpaqueDataBuffer.suppressCrossDeviceRouting(false);
            restoreDevice(restoreDevice, "text-generation");
        }
    }

    /**
     * Generate with a one-off sampling configuration, restoring the pipeline's active config afterward.
     * Equivalent to bracketing a single {@link #generate(String, int)} call with
     * {@link #setSamplingConfig(SamplingConfig)}, but scoped: the previous active config is restored even on
     * failure. Reuses the one already-loaded model — no reload, no recompile. Because generation on a pipeline
     * is single-threaded, this override is observable via {@link #getSamplingConfig()} only for the call's
     * duration.
     *
     * @param prompt        the input text prompt
     * @param maxNewTokens  maximum number of tokens to generate
     * @param sampling      sampling configuration for this call only ({@code null} means the default config)
     * @return generation result
     */
    public GenerationResult generate(String prompt, int maxNewTokens, SamplingConfig sampling) {
        SamplingConfig prev = this.activeSamplingConfig;
        this.activeSamplingConfig = sampling;
        try {
            return generate(prompt, maxNewTokens);
        } finally {
            this.activeSamplingConfig = prev;
        }
    }

    /**
     * Generate one structured chat turn using the pipeline-configured template,
     * falling back to the tokenizer-owned template when no override is configured.
     *
     * <p>The request is rendered once and the already-loaded pipeline is reused. Non-terminal
     * special tokens remain visible to the protocol parser, while the active template's atomic
     * assistant-turn terminator is handled as a stop token and excluded from generated text.</p>
     */
    public ChatGenerationResult generateChat(
            org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.Request request,
            int maxNewTokens,
            SamplingConfig sampling) {
        if (request == null) {
            throw new IllegalArgumentException("Chat request must not be null");
        }
        if (maxNewTokens <= 0) {
            throw new IllegalArgumentException("maxNewTokens must be positive");
        }
        org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.Request effective =
                resolveChatRequest(request);
        String activeTemplateText = effectiveChatTemplateText();
        String prompt = tokenizer.applyChatTemplate(effective, activeTemplateText);
        List<org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.OutputBlockDefinition>
                outputBlocks = prefilledOutputBlocks(effective, activeTemplateText, prompt);
        SamplingConfig chatSampling = samplingForChat(effective, sampling, outputBlocks);
        Set<Integer> chatStops = tokenizer.getChatTemplateStopTokenIds(activeTemplateText);
        SamplingConfig previousSampling = this.activeSamplingConfig;
        Set<Integer> previousChatStops = this.activeChatStopTokenIds;
        this.activeSamplingConfig = chatSampling;
        this.activeChatStopTokenIds = chatStops;
        GenerationResult generated;
        try {
            generated = generateTokenIds(encodeFormattedChatToIds(prompt), maxNewTokens);
        } finally {
            this.activeSamplingConfig = previousSampling;
            this.activeChatStopTokenIds = previousChatStops;
        }
        return parseEffectiveChatOutput(effective, generated.getText(), outputBlocks);
    }

    /**
     * Decode an already-generated assistant string with the same imported
     * template/protocol contract used by {@link #generateChat}.
     */
    public ChatGenerationResult parseChatOutput(
            org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.Request request,
            String rawText) {
        if (request == null) {
            throw new IllegalArgumentException("Chat request must not be null");
        }
        org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.Request effective =
                resolveChatRequest(request);
        String activeTemplateText = effectiveChatTemplateText();
        String prompt = tokenizer.applyChatTemplate(effective, activeTemplateText);
        List<org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.OutputBlockDefinition>
                outputBlocks = prefilledOutputBlocks(effective, activeTemplateText, prompt);
        return parseEffectiveChatOutput(effective, rawText, outputBlocks);
    }

    private ChatGenerationResult parseEffectiveChatOutput(
            org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.Request effective,
            String rawText,
            List<org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.OutputBlockDefinition>
                    outputBlocks) {
        org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate activeTemplate =
                activeChatTemplate();
        org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.AssistantOutput normalized =
                activeTemplate.parseAssistantOutput(rawText, outputBlocks);
        return new ChatGenerationResult(rawText, normalized, effective.getTools(),
                effective.getToolCallFormat(), effective.getToolChoice());
    }

    private List<org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.OutputBlockDefinition>
            prefilledOutputBlocks(
                    org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.Request request,
                    String activeTemplateText,
                    String promptWithGeneration) {
        if (!request.isAddGenerationPrompt()) {
            return List.of();
        }
        org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.Request withoutGeneration =
                org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.Request.builder()
                        .messages(request.getMessages())
                        .tools(request.getTools())
                        .addGenerationPrompt(false)
                        .toolDefinitionFormat(request.getToolDefinitionFormat())
                        .toolCallFormat(request.getToolCallFormat())
                        .toolChoice(request.getToolChoice())
                        .templateArguments(request.getTemplateArguments())
                        .build();
        String promptWithoutGeneration =
                tokenizer.applyChatTemplate(withoutGeneration, activeTemplateText);
        return activeChatTemplate().prefilledOutputBlocks(
                promptWithoutGeneration, promptWithGeneration);
    }

    private org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.Request resolveChatRequest(
            org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.Request request) {
        org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.ToolChoice toolChoice =
                request.getToolChoice() == null
                        ? org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.ToolChoice.AUTO
                        : request.getToolChoice();
        List<org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.Tool> effectiveTools =
                toolChoice == org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.ToolChoice.NONE
                        ? List.of() : request.getTools();
        org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.ToolCallFormat toolCallFormat =
                request.getToolCallFormat() != null ? request.getToolCallFormat()
                        : config.getToolCallFormat() != null ? config.getToolCallFormat()
                        : modelToolCallFormat();
        return org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.Request.builder()
                .messages(request.getMessages())
                .tools(effectiveTools)
                .addGenerationPrompt(request.isAddGenerationPrompt())
                .toolDefinitionFormat(request.getToolDefinitionFormat() == null
                        ? config.getToolDefinitionFormat() : request.getToolDefinitionFormat())
                .toolCallFormat(toolCallFormat)
                .toolChoice(toolChoice)
                .templateArguments(request.getTemplateArguments())
                .build();
    }

    private org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.ToolCallFormat
            modelToolCallFormat() {
        org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.ToolCallFormat templateFormat =
                activeChatTemplate().toolCallFormat();
        Integer nativeStartId = tokenizer.getTokenId(
                org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.NATIVE_TOOL_CALL_START);
        Integer nativeEndId = tokenizer.getTokenId(
                org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.NATIVE_TOOL_CALL_END);
        return selectModelToolCallFormat(
                templateFormat, nativeStartId, nativeEndId, specialTokenIds,
                specialTokenPieces);
    }

    static org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.ToolCallFormat
            selectModelToolCallFormat(
                    org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.ToolCallFormat
                            templateFormat,
                    Integer nativeStartId,
                    Integer nativeEndId,
                    Set<Integer> specialTokenIds) {
        return selectModelToolCallFormat(
                templateFormat, nativeStartId, nativeEndId, specialTokenIds,
                Collections.emptyList());
    }

    static org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.ToolCallFormat
            selectModelToolCallFormat(
                    org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.ToolCallFormat
                            templateFormat,
                    Integer nativeStartId,
                    Integer nativeEndId,
                    Set<Integer> specialTokenIds,
                    Collection<String> specialTokenPieces) {
        if (templateFormat
                != org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.ToolCallFormat.JSON) {
            return templateFormat;
        }
        Set<Integer> specials = specialTokenIds == null
                ? Collections.emptySet() : specialTokenIds;
        if (nativeStartId != null && nativeStartId >= 0
                && nativeEndId != null && nativeEndId >= 0
                && specials.contains(nativeStartId) && specials.contains(nativeEndId)) {
            return org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.ToolCallFormat.NATIVE;
        }
        Collection<String> pieces = specialTokenPieces == null
                ? Collections.emptyList() : specialTokenPieces;
        if (pieces.contains(org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.NATIVE_TOOL_CALL_START)
                && pieces.contains(org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.NATIVE_TOOL_CALL_END)) {
            return org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.ToolCallFormat.NATIVE;
        }
        return org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.ToolCallFormat.JSON;
    }

    private String effectiveChatTemplateText() {
        if (config.getChatTemplate() != null && !config.getChatTemplate().isBlank()) {
            return config.getChatTemplate();
        }
        if (modelMetadata.getChatTemplate() != null
                && !modelMetadata.getChatTemplate().isBlank()) {
            return modelMetadata.getChatTemplate();
        }
        return tokenizer.getChatTemplate();
    }

    private org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate activeChatTemplate() {
        String activeTemplateText = effectiveChatTemplateText();
        if (activeTemplateText == null || activeTemplateText.isBlank()) {
            throw new IllegalStateException(
                    "Neither the model import nor pipeline provides a chat template");
        }
        return new org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate(
                activeTemplateText, tokenizer.getBosToken(), tokenizer.getEosToken());
    }

    static SamplingConfig samplingForChat(
            org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.Request request,
            SamplingConfig sampling) {
        return samplingForChat(request, sampling, List.of());
    }

    static SamplingConfig samplingForChat(
            org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.Request request,
            SamplingConfig sampling,
            List<org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.OutputBlockDefinition>
                    outputBlocks) {
        SamplingConfig base = sampling == null
                ? SamplingConfig.defaultConfig() : sampling;
        List<org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.OutputBlockDefinition>
                blocks = outputBlocks == null ? List.of() : List.copyOf(outputBlocks);
        if (base.hasConstraint() && !blocks.isEmpty()) {
            base = base.toBuilder()
                    .constraintConfig(base.getConstraintConfig().toBuilder()
                            .outputBlocks(blocks)
                            .build())
                    .build();
        }
        org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.ToolChoice choice =
                request.getToolChoice() == null
                        ? org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.ToolChoice.AUTO
                        : request.getToolChoice();
        if (choice != org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.ToolChoice.REQUIRED) {
            return base;
        }
        if (request.getTools() == null || request.getTools().isEmpty()) {
            throw new IllegalArgumentException(
                    "ChatTemplate.ToolChoice.REQUIRED requires at least one declared tool");
        }
        if (base.hasConstraint()) {
            return base;
        }

        String[] toolNames = request.getTools().stream()
                .map(org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.Tool::getName)
                .toArray(String[]::new);
        org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.ToolCallFormat format =
                request.getToolCallFormat();
        boolean structuredFormat = format
                == org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.ToolCallFormat.NATIVE
                || format
                == org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.ToolCallFormat.XML;
        ConstraintConfig constraint;
        if (structuredFormat) {
            Map<String, List<String>> argumentNamesByTool = new LinkedHashMap<>();
            Map<String, Map<String, List<String>>> argumentValuesByTool =
                    new LinkedHashMap<>();
            Map<String, Map<String, Object>> parameterSchemasByTool =
                    new LinkedHashMap<>();
            for (org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.Tool tool
                    : request.getTools()) {
                argumentNamesByTool.put(
                        tool.getName(), requiredToolArgumentNames(tool));
                parameterSchemasByTool.put(tool.getName(), tool.getParameters());
                Map<String, List<String>> allowedValues =
                        toolArgumentValues(tool);
                if (!allowedValues.isEmpty()) {
                    argumentValuesByTool.put(tool.getName(), allowedValues);
                }
            }
            constraint = format
                    == org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.ToolCallFormat.XML
                    ? ConstraintConfig.xmlToolCall(
                            argumentNamesByTool, argumentValuesByTool,
                            parameterSchemasByTool)
                    : ConstraintConfig.nativeToolCall(
                            argumentNamesByTool, argumentValuesByTool,
                            parameterSchemasByTool);
        } else {
            constraint = ConstraintConfig.toolCall(toolNames);
        }
        if (!blocks.isEmpty()) {
            constraint = constraint.toBuilder().outputBlocks(blocks).build();
        }
        return base.toBuilder().constraintConfig(constraint).build();
    }

    private static List<String> requiredToolArgumentNames(
            org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.Tool tool) {
        Object required = tool.getParameters().get("required");
        if (!(required instanceof Collection<?>)) {
            return List.of();
        }
        List<String> names = new ArrayList<>();
        for (Object value : (Collection<?>) required) {
            if (value instanceof String && !((String) value).isBlank()) {
                names.add((String) value);
            }
        }
        return List.copyOf(names);
    }

    private static Map<String, List<String>> toolArgumentValues(
            org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.Tool tool) {
        Object propertiesObject = tool.getParameters().get("properties");
        if (!(propertiesObject instanceof Map<?, ?>)) {
            return Map.of();
        }
        Map<?, ?> properties = (Map<?, ?>) propertiesObject;
        Map<String, List<String>> allowedValues = new LinkedHashMap<>();
        for (String argumentName : requiredToolArgumentNames(tool)) {
            Object schemaObject = properties.get(argumentName);
            if (!(schemaObject instanceof Map<?, ?>)) {
                continue;
            }
            Map<?, ?> schema = (Map<?, ?>) schemaObject;
            List<String> values = new ArrayList<>();
            Object enumObject = schema.get("enum");
            if (enumObject instanceof Collection<?>) {
                for (Object value : (Collection<?>) enumObject) {
                    if (value instanceof String) {
                        values.add((String) value);
                    }
                }
            }
            Object constObject = schema.get("const");
            if (constObject instanceof String) {
                values.add((String) constObject);
            }
            if (!values.isEmpty()) {
                allowedValues.put(argumentName, List.copyOf(values));
            }
        }
        return Collections.unmodifiableMap(allowedValues);
    }

    public ChatGenerationResult generateChat(
            org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.Request request,
            int maxNewTokens) {
        return generateChat(request, maxNewTokens, getSamplingConfig());
    }

    /**
     * The sampling configuration currently applied to generation. Never {@code null} — if the active config
     * was cleared (set to {@code null}), {@link SamplingConfig#defaultConfig()} is returned.
     */
    public SamplingConfig getSamplingConfig() {
        return effectiveSampling();
    }

    /**
     * Change the sampling strategy on this already-loaded pipeline. The new config applies to every
     * subsequent {@link #generate(String)} / {@link #generate(String, int)} / {@link #generate(INDArray, int[])}
     * and {@link #startSession(String, int)} call — <b>no model reload, no graph re-optimization, no DSP plan
     * reset</b>. Scalar greedy/sample configs are pure logits post-processing (temperature, top-k, top-p,
     * repetition penalty, stop tokens) and are forwarded to the native {@code autoregressive_decode} op without
     * changing graph shapes. Multi-hypothesis strategies (speculative/contrastive/beam) resolve to ADR 0106
     * masked-substrate dimensions and fail fast until the native B/W substrate is available.
     *
     * <p>Passing {@code null} resets to {@link SamplingConfig#defaultConfig()}. A change does NOT affect a
     * {@link GenerationSession} that is already open — that session keeps the sampler it captured when it was
     * started. Generation on a pipeline is single-threaded; do not call this concurrently with an in-flight
     * {@code generate}.</p>
     *
     * @param sampling the new active sampling configuration, or {@code null} to reset to the default
     */
    public void setSamplingConfig(SamplingConfig sampling) {
        this.activeSamplingConfig = sampling;
        log.info("GenerationPipeline sampling config updated to {} — reusing loaded model (no reload/recompile)",
                sampling != null ? sampling : "default");
    }

    /** Resolve the sampling config for the current generation: the active override, or a default if unset. */
    private SamplingConfig effectiveSampling() {
        return activeSamplingConfig != null ? activeSamplingConfig : SamplingConfig.defaultConfig();
    }

    /** Resolve the sampling config for a decode call and validate the selected strategy. */
    private SamplingConfig activeDecodeSampling() {
        return effectiveSampling();
    }

    /** Resolve the ADR 0106 decode policy for the current call. */
    private DecodePolicy activeDecodePolicy() {
        return resolveDecodePolicy(effectiveSampling(), config);
    }

    private static DecodePolicy resolveDecodePolicy(SamplingConfig sampling, GenerationPipelineConfig pipelineConfig) {
        if (sampling == null) sampling = SamplingConfig.defaultConfig();
        SamplingConfig.DecodeStrategy strategy = sampling.getDecodeStrategy();
        if (strategy == null) strategy = SamplingConfig.DecodeStrategy.AUTO;

        switch (strategy) {
            case GREEDY:
                return new DecodePolicy(DecodePolicyKind.GREEDY, 1, 1, false);
            case SAMPLE:
                if (sampling.getTemperature() <= 0.0) {
                    throw new IllegalArgumentException("SamplingConfig.decodeStrategy=SAMPLE requires temperature > 0");
                }
                return new DecodePolicy(DecodePolicyKind.SAMPLE, 1, 1, false);
            case SPECULATIVE: {
                int width = pipelineConfig != null ? pipelineConfig.getMaxSpeculativeTokens() : 0;
                if (width <= 0) {
                    throw new IllegalArgumentException("SamplingConfig.decodeStrategy=SPECULATIVE requires "
                            + "GenerationPipelineConfig.maxSpeculativeTokens > 0");
                }
                return new DecodePolicy(DecodePolicyKind.SPECULATIVE, 1, width + 1, false);
            }
            case CONTRASTIVE:
                if (!sampling.isContrastive()) {
                    throw new IllegalArgumentException("SamplingConfig.decodeStrategy=CONTRASTIVE requires "
                            + "penaltyAlpha > 0 and contrastiveTopK > 1");
                }
                return new DecodePolicy(DecodePolicyKind.CONTRASTIVE, 1, sampling.getContrastiveTopK(), true);
            case BEAM:
                if (!sampling.isBeam()) {
                    throw new IllegalArgumentException("SamplingConfig.decodeStrategy=BEAM requires numBeams > 1");
                }
                if (sampling.getNumReturnSequences() > sampling.getNumBeams()) {
                    throw new IllegalArgumentException("SamplingConfig.numReturnSequences cannot exceed numBeams");
                }
                if (sampling.getNumBeamGroups() > 1) {
                    if (sampling.getNumBeams() % sampling.getNumBeamGroups() != 0) {
                        throw new IllegalArgumentException("SamplingConfig.numBeams must be divisible by numBeamGroups");
                    }
                    if (sampling.getDiversityPenalty() <= 0.0) {
                        throw new IllegalArgumentException("SamplingConfig.numBeamGroups > 1 requires diversityPenalty > 0");
                    }
                }
                return new DecodePolicy(DecodePolicyKind.BEAM, sampling.getNumBeams(), 1, false);
            case AUTO:
            default:
                return sampling.isGreedy()
                        ? new DecodePolicy(DecodePolicyKind.GREEDY, 1, 1, false)
                        : new DecodePolicy(DecodePolicyKind.SAMPLE, 1, 1, false);
        }
    }

    private static void requireNativeSubstrateAvailable(DecodePolicy policy, SamplingConfig sampling) {
        if (sampling != null && sampling.getNumReturnSequences() > 1 && policy.kind != DecodePolicyKind.BEAM) {
            throw new UnsupportedOperationException("SamplingConfig.numReturnSequences > 1 requires BEAM "
                    + "decode; scalar greedy/sample can only return one sequence. Config=" + sampling);
        }
        if (policy.isScalarNativePolicy()) {
            return;
        }
        // ADR 0106 Phase 2: SPECULATIVE uses the window substrate with W=speculativeK+1.
        // The n-gram proposer runs inside the C++ decode loop — no separate draft model needed.
        // windowMax > 1 confirms the substrate has been allocated for the wider window.
        if (policy.kind == DecodePolicyKind.SPECULATIVE && policy.windowMax > 1) {
            return;
        }
        throw new UnsupportedOperationException(
                "Decode strategy " + policy.kind + " resolves to masked substrate B=" + policy.batchMax
                + ", W=" + policy.windowMax + " (hidden=" + policy.needsHiddenState + "), but the current "
                + "native autoregressive_decode op still exposes only the scalar B=1,W=1 contract. "
                + "Do not route through StaticKvCacheDecodeLoop, SpeculativeDecodeLoop, or TextGenerator; "
                + "finish ADR 0106 by extending the native masked multi-position substrate first. Config="
                + sampling);
    }

    private static int nativeDecodeStrategy(DecodePolicyKind kind) {
        switch (kind) {
            case GREEDY:
                return AutoregressiveDecode.DECODE_STRATEGY_GREEDY;
            case SAMPLE:
                return AutoregressiveDecode.DECODE_STRATEGY_SAMPLE;
            case SPECULATIVE:
                return AutoregressiveDecode.DECODE_STRATEGY_SPECULATIVE;
            case CONTRASTIVE:
                return AutoregressiveDecode.DECODE_STRATEGY_CONTRASTIVE;
            case BEAM:
                return AutoregressiveDecode.DECODE_STRATEGY_BEAM;
            default:
                return AutoregressiveDecode.DECODE_STRATEGY_AUTO;
        }
    }

    private static AutoregressiveDecode applyNativePolicy(AutoregressiveDecode op, DecodePolicy policy,
                                                          SamplingConfig sampling,
                                                          int generatedTokenOffset) {
        return applyNativePolicy(op, policy, sampling, generatedTokenOffset, policy.windowMax);
    }

    private static AutoregressiveDecode applyNativePolicy(AutoregressiveDecode op, DecodePolicy policy,
                                                          SamplingConfig sampling,
                                                          int generatedTokenOffset,
                                                          int windowEnvelope) {
        if (sampling == null) sampling = SamplingConfig.defaultConfig();
        if (windowEnvelope < policy.windowMax) {
            throw new IllegalArgumentException("Frozen decode window W=" + windowEnvelope
                    + " is narrower than requested policy W=" + policy.windowMax);
        }
        long seed = sampling.getSeed() != null ? sampling.getSeed() : 0L;
        int activeWindow = policy.kind == DecodePolicyKind.SPECULATIVE ? 1 : policy.windowMax;
        return op.withDecodePolicy(nativeDecodeStrategy(policy.kind),
                        policy.batchMax, windowEnvelope, policy.batchMax, activeWindow,
                        -1, sampling.getNumBeams(), sampling.getLengthPenalty(),
                        sampling.getPenaltyAlpha(), sampling.getContrastiveTopK())
                .withSamplingPolicy(sampling.getMinP(), sampling.getFrequencyPenalty(),
                        sampling.getPresencePenalty(), sampling.getMinNewTokens(),
                        generatedTokenOffset, seed)
                .withTypicalPAndXtc(sampling.getTypicalP(), sampling.getXtcProbability(),
                        sampling.getXtcThreshold());
    }

    private GenerationResult generateInternal(int[] promptTokenIds, int maxNewTokens) {
        // Single-model mode: no separate embedTokens model was provided.
        // The decoder handles its own embedding lookup internally
        // (input_ids → gather → transformer → logits).
        // Use a simple autoregressive loop like TextGenerator.
        // Note: the decoder may still have an inputs_embeds variable, but without
        // a separate embedTokens model, input_ids is the intended entry point.
        if (embedTokens == null) {
            return generateSimple(promptTokenIds, maxNewTokens);
        }

        // Two-model mode: embed tokens externally, then run native decode
        INDArray embeddings = embedTokens(promptTokenIds);
        return generate(embeddings, promptTokenIds, maxNewTokens);
    }

    /**
     * Simple autoregressive generation for single-model GGUF models.
     *
     * <p>The decoder graph handles embedding lookup internally (input_ids → gather → transformer → logits).
     * Each step feeds input_ids and reads logits.</p>
     *
     * <p>Routes to the appropriate decode strategy:</p>
     * <ol>
     *   <li>In-graph KV cache (GGUF with dotProductAttentionV2 built-in scatter): fixed shapes, DSP replay</li>
     *   <li>External KV cache (ONNX with present outputs): static padding, DSP replay</li>
     *   <li>No KV cache: shapes grow each step, no replay</li>
     * </ol>
     */
    private GenerationResult generateSimple(int[] promptTokenIds, int maxNewTokens) {
        SamplingConfig sampling = activeDecodeSampling();
        DecodePolicy decodePolicy = activeDecodePolicy();
        requireNativeSubstrateAvailable(decodePolicy, sampling);

        // Check for in-graph KV cache (GGUF pattern: KV inputs, no present outputs)
        if (ModelIOConfig.isInGraphKvCache(decoder)) {
            ModelIOConfig.KVCacheNames kvInputNames = ModelIOConfig.findKVCacheInputNames(decoder);
            return generateSimpleWithInGraphKvCache(promptTokenIds, maxNewTokens, kvInputNames);
        }

        // All decode is handled by the native autoregressive_decode C++ op.
        // Models without in-graph KV cache must use the two-model (embed + decode) path
        // which also routes through the native op via generateNative().
        throw new UnsupportedOperationException(
                "Model does not have in-graph KV cache. Use a GGUF model or provide a separate " +
                "embedTokens model to use the native autoregressive_decode path.");
    }

    /**
     * In-graph KV cache autoregressive generation for GGUF models using the native
     * {@code autoregressive_decode} C++ op.
     *
     * <p>The decoder's dotProductAttentionV2 ops write K/V into static cache buffers
     * in-place at cachePosition (inputs 5-7). No present outputs, no external scatter.
     * All decode-step tensor shapes are fixed after the first decode step, enabling DSP replay.</p>
     *
     * <p>Flow (mirrors {@link #generateNative}):</p>
     * <ol>
     *   <li>Prefill: full prompt, empty KV cache inputs, extract K/V from per-layer outputs</li>
     *   <li>Initialize static KV buffers from prefill K/V</li>
     *   <li>Warmup decode step: compile DSP plan for decode shapes</li>
     *   <li>Get plan handle, freeze shapes, resolve ext input indices</li>
     *   <li>AutoregressiveDecode native op: full decode loop in C++</li>
     * </ol>
     */
    /**
     * Prefill the prompt, run one warmup decode step, and freeze the DSP plan — producing an
     * {@link InGraphKvState} that retains the static KV / recurrent buffers, decode-step tensors,
     * frozen plan handles, and resolved external-input indices so decoding can later be
     * <em>resumed</em> (see {@link #runInGraphNativeDecode}) without a reset or re-prefill.
     *
     * <p>Shared by the one-shot {@link #generateSimpleWithInGraphKvCache} path and by
     * {@link #startSession(String, int)}. {@code maxNewTokens} here sizes the KV buffer
     * ({@code maxKvLen = prefillLen + maxNewTokens}, capped by {@code maxKvCacheLength}); for a session
     * this is the total continuation capacity.</p>
     *
     * <p>On the terminal prefill/warmup outcomes (first token is EOS, or no native plan handle) the
     * returned state carries a {@link InGraphKvState#terminalResult} and is already closed — callers
     * return that result directly.</p>
     */
    private InGraphKvState prefillWarmupAndFreeze(int[] promptTokenIds, int maxNewTokens,
                                                  ModelIOConfig.KVCacheNames kvInputNames,
                                                  long startTime, InGraphKvState reuseState) {

        // ADR 0107 V2 INLINE-SCALE: when INT8 KV quantization is requested, declare the KV cache
        // placeholders as INT8 before any plan is built. A runtime INT8 buffer bound to a FLOAT
        // placeholder is dtype-invisible to the plan — shape/segment cache keys, Triton kernel
        // selection and dot_product_attention_v2's quantised-path detection all see FLOAT — so the
        // quantised decode would reuse the FLOAT attention kernel and skip the quantised write.
        // dot_product_attention_v2 accepts an INT8 keyCache/valueCache (inputs 5/6); with an INT8
        // placeholder the decode takes the quantised-on-write + inline-scale read path and gets a
        // distinct compiled kernel from the float KV plan.
        if (config.getKvCacheStrategy() == KvCacheStrategy.QUANTIZED && config.getKvQuantFormat() > 0) {
            Map<String, DataType> kvInt8 = new LinkedHashMap<>();
            for (String kn : kvInputNames.keyNames) {
                if (decoder.hasVariable(kn) && decoder.getVariable(kn).dataType() != DataType.INT8) {
                    kvInt8.put(kn, DataType.INT8);
                }
            }
            for (String vn : kvInputNames.valueNames) {
                if (decoder.hasVariable(vn) && decoder.getVariable(vn).dataType() != DataType.INT8) {
                    kvInt8.put(vn, DataType.INT8);
                }
            }
            if (!kvInt8.isEmpty()) {
                // The decoder SameDiff may be shared across pipelines (and with future STATIC
                // pipelines) — record the original dtype/shape ONCE so close() can restore them.
                // Without the restore, a later non-quantised pipeline on the same decoder would
                // allocate INT8 KV buffers against the mutated placeholders and take the
                // quantised attention path with a float-sized cache.
                if (kvPlaceholderOriginalDtypes == null) {
                    kvPlaceholderOriginalDtypes = new LinkedHashMap<>();
                    kvPlaceholderOriginalShapes = new LinkedHashMap<>();
                    for (String kvn : kvInt8.keySet()) {
                        kvPlaceholderOriginalDtypes.put(kvn, decoder.getVariable(kvn).dataType());
                        long[] declared = decoder.getVariable(kvn).getShape();
                        kvPlaceholderOriginalShapes.put(kvn, declared != null ? declared.clone() : null);
                    }
                }
                decoder.convertDataTypes(kvInt8);
                // ROW-INLINE: the fed INT8 caches are [batch, maxKvLen, kvHeads, headDim+4]
                // (per-row float32 scale inside the tensor) — relax the declared trailing dim
                // so placeholder validation accepts the widened rows.
                for (String kvn : kvInt8.keySet()) {
                    long[] declared = decoder.getVariable(kvn).getShape();
                    if (declared != null && declared.length == 4) {
                        decoder.getVariable(kvn).setShape(declared[0], declared[1], declared[2], -1);
                    }
                }
                log.info("[GGUF-KV] Declared {} KV cache placeholders as INT8 (row-inline scale) for quantised decode",
                        kvInt8.size());
            }
        }

        int maxPrefill = config.getMaxPrefillLength();
        boolean fixedBuffers = maxPrefill > 0;

        if (fixedBuffers) {
            if (reuseState != null) {
                // FORWARD-FIX reuse: keep the plan AND its node-output buffers. Do NOT call
                // clearNodeOutputsOnly() — it closes+nulls the session DynamicShapePlan
                // (InferenceSession.java:455), which frees the native plan handle → the next
                // dispatch is a NEW BORROWER → #52 external-view invalidation → segment reset →
                // full re-warm (~130s). The frozen plan + captured graphs are reused as-is; STEP 1
                // re-prefills in place into the retained (stable-address) buffers.
                log.info("[Lifecycle] Reusing cached fixed-buffer state — keep plan, in-place re-prefill (maxPrefill={})", maxPrefill);
            } else {
                // A frozen fixed-buffer plan is reusable only together with the state that owns its
                // captured external-input addresses. Reaching this branch without reuseState means
                // that ownership was lost (for example after an incompatible-capacity handoff).
                // Tear the stale session down completely; clearNodeOutputsOnly() is not a reset — it
                // closes the Java DynamicShapePlan while leaving native cached allocations behind.
                InferenceSession existSession = decoder.getOrCreateSession();
                if (existSession != null) {
                    DynamicShapePlanExecutor existExecutor = existSession.getDynamicShapePlanExecutor();
                    if (existExecutor != null && existExecutor.isShapesFrozen()) {
                        log.warn("[Lifecycle] Frozen fixed-buffer DSP plan has no retained state; resetting it before fresh prefill");
                        decoder.resetSession();
                        decoder.clearDynamicShapePlanCache();
                        // The cache reset releases plan-owned arrays with cudaFreeAsync.
                        // Drain those frees before allocating the replacement plan or each
                        // independent generation retains another request-sized pool footprint.
                        SameDiffMemoryUtils.trimAllDevicePools();
                    }
                }
            }
        } else {
            // Variable-size buffers: must reset frozen DSP executor from previous generation
            InferenceSession existSession = decoder.getOrCreateSession();
            if (existSession != null) {
                DynamicShapePlanExecutor existExecutor = existSession.getDynamicShapePlanExecutor();
                if (existExecutor != null && existExecutor.isShapesFrozen()) {
                    log.info("[Lifecycle] Resetting frozen DSP executor for new GGUF generation");
                    // resetSession() MUST come before clearDynamicShapePlanCache():
                    // the executor holds a nativePlanHandle and calls
                    // releaseGpuIntermediates() during destroySession(). Clearing the
                    // C++ cache first destroys the plan, then releaseGpuIntermediates()
                    // dereferences a dangling pointer → free(): invalid pointer.
                    decoder.resetSession();
                    decoder.clearDynamicShapePlanCache();
                    // resetSession()/cache clear enqueue plan-owned frees. Synchronize and
                    // return them before this independent generation allocates a new plan.
                    SameDiffMemoryUtils.trimAllDevicePools();
                }
            }
        }

        SamplingConfig sampling = activeDecodeSampling();
        int eosTokenId = resolveEosTokenId(sampling);
        Set<Integer> stopTokenIds = buildStopTokenIds(eosTokenId);

        DecodePolicy decodePolicy = activeDecodePolicy();
        requireNativeSubstrateAvailable(decodePolicy, sampling);
        Random rng = sampling.getSeed() != null ? new Random(sampling.getSeed()) : new Random();

        String inputIdsName = ioConfig.getInputIdsName() != null ? ioConfig.getInputIdsName() : "input_ids";
        String logitsName = ioConfig.getLogitsOutputName() != null ? ioConfig.getLogitsOutputName() : "lm_logits";
        String posOffsetName = ioConfig.getPositionOffsetName();
        String cachePosName = ioConfig.getCachePositionName();
        String causalMaskName = ioConfig.getCausalMaskName();
        boolean useNativeMtp = config.getMaxSpeculativeTokens() > 0 && hasBundledMtpGraph();
        if (useNativeMtp) {
            log.info("[MTP] Bundled Qwen3.5 predictor enabled (K={})", config.getMaxSpeculativeTokens());
        }

        int actualPrefillLen = promptTokenIds.length;

        // When fixedBuffers is enabled, pad/truncate prompt to maxPrefillLength
        // so all prefill shapes are identical across calls.
        int prefillSeqLen;
        int[] effectiveTokenIds;
        if (fixedBuffers) {
            prefillSeqLen = maxPrefill;
            if (actualPrefillLen > maxPrefill) {
                log.warn("[GGUF-KV] Prompt length {} exceeds maxPrefillLength {} — truncating",
                        actualPrefillLen, maxPrefill);
                effectiveTokenIds = new int[maxPrefill];
                System.arraycopy(promptTokenIds, actualPrefillLen - maxPrefill,
                        effectiveTokenIds, 0, maxPrefill);
                actualPrefillLen = maxPrefill;
            } else if (actualPrefillLen < maxPrefill) {
                // Right-pad with pad token (0). The causal mask will prevent
                // attention to padding positions.
                effectiveTokenIds = new int[maxPrefill];
                System.arraycopy(promptTokenIds, 0, effectiveTokenIds, 0, actualPrefillLen);
                // Remaining positions are 0 (pad token)
                log.info("[GGUF-KV] Padded prompt from {} to {} tokens", actualPrefillLen, maxPrefill);
            } else {
                effectiveTokenIds = promptTokenIds;
            }
        } else {
            prefillSeqLen = actualPrefillLen;
            effectiveTokenIds = promptTokenIds;
        }

        int kvCap = config.getMaxKvCacheLength();
        if (kvCap > 0 && prefillSeqLen >= kvCap) {
            throw new IllegalArgumentException(
                    "Prompt token count "
                            + prefillSeqLen
                            + " leaves no room for generation within "
                            + "maxKvCacheLength="
                            + kvCap
            );
        }

        long maxKvLen =
                prefillSeqLen + maxNewTokens;
        // Cap to configured KV cache length to keep buffer shapes stable across
        // calls with different maxNewTokens — avoids plan recompilation.
        if (kvCap > 0 && maxKvLen > kvCap) {
            maxNewTokens = kvCap - prefillSeqLen;
            maxKvLen = kvCap;
        }
        int numLayers = kvInputNames.keyNames.size();

        // Discover recurrent state input→output pairs from graph structure
        List<ModelIOConfig.RecurrentStatePair> recurrentStates =
                ModelIOConfig.findRecurrentStatePairs(decoder, ioConfig);
        log.info("[GGUF-KV] Found {} recurrent state pairs: {} (fixedBuffers={} prefillSeqLen={} actualPrefillLen={} maxKvLen={})",
                recurrentStates.size(), recurrentStates, fixedBuffers, prefillSeqLen, actualPrefillLen, maxKvLen);

        // ══════════════════════════════════════════════════════════════════════
        // STEP 1: PREFILL -- full prompt, empty KV cache, extract per-layer K/V
        // DSP stays enabled throughout — never disable it.
        // ══════════════════════════════════════════════════════════════════════

        // ── PREFILL EXTERNAL INPUTS ──────────────────────────────────────────────────────────────
        // Forward-fix (Design A): on reuse the prefill plan is already frozen/captured (frozen
        // multi-plan switch freezes it too), so its external inputs MUST keep STABLE device addresses
        // or the captured prefill graph replays against a dangling/stale address → silent gen-3+
        // degeneration. So REUSE the cached prefill tensors and overwrite the VARYING ones in place
        // (input_ids = new prompt, causal mask = new actual length); the CONSTANT ones (zero position /
        // cache-position scalars, empty-KV sentinels, zero recurrent state) are reused as-is and
        // re-zeroed defensively. A fresh tensor per generate is only correct on the non-reused path.
        DataType maskDtype = DataType.FLOAT;
        if (causalMaskName != null && decoder.hasVariable(causalMaskName)) {
            maskDtype = decoder.getVariable(causalMaskName).dataType();
        }
        Map<String, INDArray> prefillInputMap;
        INDArray prefillInputIds;
        if (reuseState != null && reuseState.prefillInputMap != null) {
            // Reuse path: overwrite in place, keep every address stable.
            prefillInputMap = reuseState.prefillInputMap;
            prefillInputIds = prefillInputMap.get(inputIdsName);
            INDArray freshMask = null;
            try (INDArray freshIds = Nd4j.createFromArray(effectiveTokenIds)
                    .reshape(1, prefillSeqLen).castTo(DataType.INT64)) {
                prefillInputIds.assign(freshIds);
                if (posOffsetName != null && prefillInputMap.containsKey(posOffsetName)) {
                    prefillInputMap.get(posOffsetName).assign(0);
                }
                if (cachePosName != null && prefillInputMap.containsKey(cachePosName)) {
                    prefillInputMap.get(cachePosName).assign(0);
                }
                if (prefillInputMap.containsKey(ACTUAL_SEQUENCE_LENGTH_NAME)) {
                    prefillInputMap.get(ACTUAL_SEQUENCE_LENGTH_NAME).assign(actualPrefillLen);
                }
                if (causalMaskName != null && prefillInputMap.containsKey(causalMaskName)) {
                    freshMask = fixedBuffers
                            ? buildPaddedPrefillCausalMask(actualPrefillLen, prefillSeqLen, maxKvLen, maskDtype)
                            : DecoderInputBuilder.buildInGraphCausalMask(prefillSeqLen, maxKvLen, maskDtype);
                    prefillInputMap.get(causalMaskName).assign(freshMask);
                }
                // Empty-KV sentinels reused as-is (no content). Re-zero recurrent state inputs (no history).
                for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
                    INDArray st = prefillInputMap.get(pair.inputName);
                    if (st != null) st.assign(0);
                }

                // CUDA assignment is asynchronous. Keep both source buffers alive through commit;
                // closing either one earlier lets the allocator reuse its device memory while the
                // copy into the stable captured-plan input is still queued.
                Nd4j.getExecutioner().commit();
            } finally {
                if (freshMask != null && !freshMask.wasClosed()) freshMask.close();
            }
            for (INDArray input : prefillInputMap.values()) {
                if (input != null && !input.wasClosed() && input.length() > 0) {
                    input.syncToDevice();
                }
            }
        } else {
            prefillInputIds = Nd4j.createFromArray(effectiveTokenIds)
                    .reshape(1, prefillSeqLen).castTo(DataType.INT64);

            prefillInputMap = new HashMap<>();
            prefillInputMap.put(inputIdsName, prefillInputIds);

            if (posOffsetName != null && decoder.hasVariable(posOffsetName)) {
                prefillInputMap.put(posOffsetName, Nd4j.scalar(DataType.INT64, 0));
            }
            if (cachePosName != null && decoder.hasVariable(cachePosName)) {
                prefillInputMap.put(cachePosName, Nd4j.scalar(DataType.INT64, 0));
            }
            if (decoder.hasVariable(ACTUAL_SEQUENCE_LENGTH_NAME)) {
                prefillInputMap.put(ACTUAL_SEQUENCE_LENGTH_NAME,
                        Nd4j.scalar(DataType.INT64, actualPrefillLen));
            }
            if (causalMaskName != null && decoder.hasVariable(causalMaskName)) {
                if (fixedBuffers) {
                    // Build causal mask that only attends to actual token positions,
                    // masking out the padding region even though the buffer is full-size.
                    prefillInputMap.put(causalMaskName,
                            buildPaddedPrefillCausalMask(actualPrefillLen, prefillSeqLen, maxKvLen, maskDtype));
                } else {
                    prefillInputMap.put(causalMaskName,
                            DecoderInputBuilder.buildInGraphCausalMask(prefillSeqLen, maxKvLen, maskDtype));
                }
            }

            // Empty KV cache inputs — signals attention op to skip in-place scatter
            for (String keyName : kvInputNames.keyNames) {
                if (decoder.hasVariable(keyName)) {
                    prefillInputMap.put(keyName, Nd4j.empty(decoder.getVariable(keyName).dataType()));
                }
            }
            for (String valName : kvInputNames.valueNames) {
                if (decoder.hasVariable(valName)) {
                    prefillInputMap.put(valName, Nd4j.empty(decoder.getVariable(valName).dataType()));
                }
            }

            // Zero-filled recurrent state inputs for prefill (zeros = no prior history)
            // Shapes are derived from the ops that consume each state placeholder.
            for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
                if (decoder.hasVariable(pair.inputName)) {
                    DataType dt = decoder.getVariable(pair.inputName).dataType();
                    long[] stateShape = deriveRecurrentStateShape(decoder, pair.inputName);
                    if (stateShape != null) {
                        prefillInputMap.put(pair.inputName, Nd4j.zeros(dt, stateShape));
                    } else {
                        log.warn("[GGUF-KV] Cannot derive state shape for '{}' from graph", pair.inputName);
                    }
                }
            }
        }

        // ── Prefill last-position logits optimisation ─────────────────────────────────────────────
        // When enabled AND the model exposes "lm_logits_last" (slice of hidden at S-1 before
        // lm_head), request that output instead of full "lm_logits". DSP will then compile a plan
        // that skips the all-positions vocab projection and only computes the last-pos result —
        // a TTFT win proportional to (S-1) for large S. Decode (S=1) is unchanged.
        // Imported GGUF graphs that do NOT have "lm_logits_last" fall back silently.
        // Fixed-buffer-compatible graphs derive lm_logits_last from actual_sequence_length-1,
        // not from the padded buffer tail, so the optimized projection still samples the final
        // real token while preserving a stable output shape.
        final String LAST_POS_LOGITS_NAME = "lm_logits_last";
        boolean usePrefillLastPos = config.isEffectivePrefillLastPositionLogitsEnabled()
                && prefillSeqLen > 1
                && decoder.hasVariable(LAST_POS_LOGITS_NAME)
                && decoder.outputs().contains(LAST_POS_LOGITS_NAME);
        // The effective logits output name for prefill: either "lm_logits_last" or the full one.
        String effectiveLogitsName = usePrefillLastPos ? LAST_POS_LOGITS_NAME : logitsName;

        // Request outputs: logits + per-layer KV outputs + recurrent state outputs
        List<String> prefillOutputNames = new ArrayList<>();
        prefillOutputNames.add(effectiveLogitsName);
        if (useNativeMtp) prefillOutputNames.add(TARGET_HIDDEN_STATES_NAME);
        for (int i = 0; i < numLayers; i++) {
            int layerIdx = extractLayerIndex(kvInputNames.keyNames.get(i));
            prefillOutputNames.add("k_rope_" + layerIdx);
            prefillOutputNames.add("v_heads_" + layerIdx);
        }
        for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
            prefillOutputNames.add(pair.outputName);
        }
        Map<String, INDArray> prefillOutputs;
        try {
            prefillOutputs = decoder.output(
                    prefillInputMap, prefillOutputNames.toArray(new String[0]));
        } catch (Exception e) {
            log.error("[GGUF-KV] Prefill decoder.output() failed", e);
            throw e;
        }

        log.info("[GGUF-KV] Prefill returned {} outputs: {} (prefillLastPos={})",
                prefillOutputs.size(), prefillOutputs.keySet(), usePrefillLastPos);
        INDArray targetPrefillHidden = useNativeMtp ? prefillOutputs.get(TARGET_HIDDEN_STATES_NAME) : null;
        if (useNativeMtp && targetPrefillHidden == null) {
            throw new IllegalStateException("Bundled MTP graph did not return " + TARGET_HIDDEN_STATES_NAME
                    + " during target prefill");
        }

        // Sample first token from prefill logits.
        // When fixedBuffers is active, the logits tensor has shape [1, prefillSeqLen, vocab]
        // where prefillSeqLen includes padding. The real last token is at actualPrefillLen-1.
        // When usePrefillLastPos=true, the shape is [B, 1, vocab] and the only valid index is 0.
        INDArray prefillLogits = prefillOutputs.get(effectiveLogitsName);
        if (prefillLogits == null) {
            throw new RuntimeException("[GGUF-KV] Prefill logits '" + effectiveLogitsName
                    + "' not found in outputs: " + prefillOutputs.keySet());
        }
        // Clamp to the actual logits seq dim (see generateNative): no-op for the padded
        // [1, prefillSeqLen, vocab] case here, but guards an UNCHECKED OOB read if a model
        // ever exports last-position-only [1, 1, vocab] logits on this path too.
        // When usePrefillLastPos=true, shape is [1, 1, vocab] → samplePos=0 always.
        int logitsSamplePos = Math.min(actualPrefillLen - 1, (int) prefillLogits.shape()[1] - 1);
        if (usePrefillLastPos) {
            // lm_logits_last is already last-position only — always sample from index 0
            logitsSamplePos = 0;
        }
        log.info("[GGUF-KV] Prefill logits shape: {} samplePos={} (actualPrefillLen={} lastPosOpt={})",
                Arrays.toString(prefillLogits.shape()), logitsSamplePos, actualPrefillLen, usePrefillLastPos);
        {
            INDArray lastPosLogits = prefillLogits.get(
                    NDArrayIndex.point(0),
                    NDArrayIndex.point(logitsSamplePos),
                    NDArrayIndex.all()).dup();
            log.info("[GGUF-KV] Prefill last-pos logits: dtype={} min={} max={} mean={} hasNaN={}",
                    lastPosLogits.dataType(), lastPosLogits.minNumber(), lastPosLogits.maxNumber(),
                    lastPosLogits.meanNumber(), Double.isNaN(lastPosLogits.meanNumber().doubleValue()));
            lastPosLogits.close();
        }
        List<Integer> generatedSoFar = new ArrayList<>();
        ConstraintMasker constraintMasker = null;
        if (sampling.hasConstraint()) {
            ConstraintConfig constraintConfig = sampling.getConstraintConfig();
            constraintMasker = new ConstraintMasker(
                    constraintConfig.buildConstraint(),
                    constraintConfig.getEvalTopK(),
                    maxNewTokens,
                    sampling.getMaxOutputBlockTokens(),
                    sampling.getStructuredOutputTokenReserve());
            log.info("[Constraint] Constrained in-graph KV decoding active: type={} evalTopK={} "
                            + "maxOutputBlockTokens={} structuredOutputTokenReserve={}",
                    constraintConfig.getType(), constraintConfig.getEvalTopK(),
                    sampling.getMaxOutputBlockTokens(),
                    sampling.getStructuredOutputTokenReserve());
        }
        suppressStopsUnderFloor(prefillLogits, logitsSamplePos, sampling, generatedSoFar.size(), stopTokenIds);
        int firstTokenId = sampleToken(
                prefillLogits, logitsSamplePos, sampling, generatedSoFar, rng,
                constraintMasker, tokenizer, stopTokenIds);
        generatedSoFar.add(firstTokenId);
        prefillLogits.close();
        // prefillInputIds is retained in prefillInputMap (cached in the InGraphKvState) for the
        // fixed-buffer reuse path so the frozen prefill plan replays against a stable address —
        // freed via state.close(), not here. The two early-return terminals below close it inline.

        long firstTokenMs = System.currentTimeMillis() - startTime;

        log.info("[GGUF-KV] First token: {} (eos={})", firstTokenId, stopTokenIds.contains(firstTokenId));

        if (stopTokenIds.contains(firstTokenId)) {
            closePrefillOutputs(prefillOutputs, effectiveLogitsName);
            // Non-reuse owns prefillInputMap and frees it here; on reuse it is the retained state's own
            // field and is freed by the caller via reuseState.close() (the caller drops the cache).
            if (reuseState == null) {
                for (INDArray v : prefillInputMap.values()) { if (v != null && !v.wasClosed()) v.close(); }
            }
            List<Integer> tokens = new ArrayList<>();
            tokens.add(firstTokenId);
            InGraphKvState terminal = new InGraphKvState();
            terminal.terminalResult = buildResult(tokens, promptTokenIds, stopTokenIds, startTime, firstTokenMs);
            terminal.generatedSoFar = tokens;
            terminal.eosReached = true;
            terminal.closed = true;
            terminal.actualPrefillLen = actualPrefillLen;
            terminal.promptTokenCount = prefillSeqLen;
            terminal.maxKvLen = maxKvLen;
            return terminal;
        }

        // ══════════════════════════════════════════════════════════════════════
        // STEP 2: Initialize static KV buffers from prefill K/V outputs
        // Shape: [batch, maxKvLen, numKVHeads, headDim] (BSHD layout)
        // ══════════════════════════════════════════════════════════════════════

        // QUANTIZED strategy validation: reject combination with rotating KV.
        // Rotating KV changes physical write slots per step; the quantize-archive step (which
        // runs between generate calls, not per-decode-step) would need to track eviction order.
        // Defer support for this combination to v2.
        boolean isQuantizedKv = config.getKvCacheStrategy() == KvCacheStrategy.QUANTIZED
                && config.getKvQuantFormat() > 0;
        if (isQuantizedKv && config.isRotatingKvEnabled()) {
            throw new IllegalStateException(
                    "KvCacheStrategy.QUANTIZED is not compatible with rotatingKvEnabled=true. "
                    + "Use STATIC strategy with rotating KV, or QUANTIZED without rotating KV.");
        }
        int kvQuantFormat = isQuantizedKv ? config.getKvQuantFormat() : 0;

        log.info("[GGUF-KV] STEP 2: numLayers={} keyNames.size={} quantized={} format={}",
                numLayers, kvInputNames.keyNames.size(), isQuantizedKv, kvQuantFormat);
        Map<String, INDArray> staticKvBuffers = reuseState != null && reuseState.staticKvBuffers != null
                ? reuseState.staticKvBuffers : new LinkedHashMap<>();
        // QUANTIZED: separate INT8-compressed buffers + scales, sized only to the active
        // prefill region (maxKvLen float elements → maxKvLen INT8 elements, 4x smaller).
        Map<String, INDArray> quantizedKvBuffers = isQuantizedKv
                ? (reuseState != null && reuseState.quantizedKvBuffers != null
                        ? reuseState.quantizedKvBuffers : new LinkedHashMap<>())
                : null;
        Map<String, INDArray> kvScaleBuffers = isQuantizedKv
                ? (reuseState != null && reuseState.kvScaleBuffers != null
                        ? reuseState.kvScaleBuffers : new LinkedHashMap<>())
                : null;
        DataType kvDtype = null;
        for (int i = 0; i < numLayers; i++) {
            int layerIdx = extractLayerIndex(kvInputNames.keyNames.get(i));

            INDArray kRoped = prefillOutputs.get("k_rope_" + layerIdx);
            INDArray vHeads = prefillOutputs.get("v_heads_" + layerIdx);

            if (kRoped == null || vHeads == null) {
                log.warn("Missing prefill K/V for layer {}", layerIdx);
                continue;
            }

            if (kvDtype == null) kvDtype = kRoped.dataType();

            // kRoped/vHeads shape: [batch, prefillLen, numKVHeads, headDim] (4D) or [batch, prefillLen, dim] (3D)
            long[] kvShape = kRoped.shape();
            long batch, numKVHeads, headDim;
            if (kvShape.length == 4) {
                batch = kvShape[0]; numKVHeads = kvShape[2]; headDim = kvShape[3];
            } else {
                // 3D: infer head dimensions from KV cache placeholder shape
                batch = kvShape[0];
                long[] cacheShape = decoder.getVariable(kvInputNames.keyNames.get(i)).getShape();
                numKVHeads = cacheShape[2]; headDim = cacheShape[3];
                kRoped = kRoped.reshape(batch, prefillSeqLen, numKVHeads, headDim);
                vHeads = vHeads.reshape(batch, prefillSeqLen, numKVHeads, headDim);
            }

            // Full-size buffer; write prefill data at positions 0..prefillLen-1. On reuse, zero the
            // RETAINED buffer (stable address — the frozen decode plan baked it) and re-write in place;
            // otherwise allocate fresh. Padding positions [prefillLen..maxKvLen) stay zero.
            String keyName = kvInputNames.keyNames.get(i);
            INDArray keyBuf;
            if (reuseState != null && staticKvBuffers.containsKey(keyName)) {
                keyBuf = staticKvBuffers.get(keyName);
                keyBuf.assign(0);
            } else {
                keyBuf = Nd4j.zeros(kvDtype, batch, maxKvLen, numKVHeads, headDim);
            }
            keyBuf.get(NDArrayIndex.all(), NDArrayIndex.interval(0, prefillSeqLen),
                    NDArrayIndex.all(), NDArrayIndex.all()).assign(kRoped);
            staticKvBuffers.put(keyName, keyBuf);

            String valName = kvInputNames.valueNames.get(i);
            INDArray valBuf;
            if (reuseState != null && staticKvBuffers.containsKey(valName)) {
                valBuf = staticKvBuffers.get(valName);
                valBuf.assign(0);
            } else {
                valBuf = Nd4j.zeros(kvDtype, batch, maxKvLen, numKVHeads, headDim);
            }
            valBuf.get(NDArrayIndex.all(), NDArrayIndex.interval(0, prefillSeqLen),
                    NDArrayIndex.all(), NDArrayIndex.all()).assign(vHeads);
            staticKvBuffers.put(valName, valBuf);

            if (isQuantizedKv) {
                // Quantize the full KV buffer (prefill region + zero padding) into INT8.
                // We quantize the full staticKvBuffer (not just the prefill slice) so
                // scale indices match the flat-row layout of the full [B,maxKvLen,kvH,headDim] tensor.
                // Scale shape: [batch, maxKvLen, numKVHeads] — one scale per token-head row.
                // ADR 0107 V2 ROW-INLINE: quantize into row-inline INT8 tensors
                // [batch, maxKvLen, numKVHeads, headDim+4] — each row holds headDim int8 values
                // followed by that row's float32 scale. The scales live INSIDE the logical tensor,
                // so DSP ext-input staging, H2D re-stage, CUDA-graph capture and device migration
                // all preserve them by construction. No separate scale graph input, no registry.
                INDArray keyCacheInt8 = Nd4j.getExecutioner().exec(
                        new KVCacheQuantize(keyBuf.castTo(DataType.FLOAT), kvQuantFormat, true))[0];
                INDArray valCacheInt8 = Nd4j.getExecutioner().exec(
                        new KVCacheQuantize(valBuf.castTo(DataType.FLOAT), kvQuantFormat, true))[0];
                // Quantize ran on device — sync to HOST so any forced placeholder H2D re-stages
                // the same bytes (values + in-row scales) instead of stale host zeros.
                Nd4j.getExecutioner().commit();
                keyCacheInt8.syncToHost();
                valCacheInt8.syncToHost();

                String keyQName = keyName + "_q";
                String valQName = valName + "_q";

                // Reuse or replace existing quantized buffers on reuse path
                replaceQuantizedBuffer(quantizedKvBuffers, keyQName, keyCacheInt8);
                replaceQuantizedBuffer(quantizedKvBuffers, valQName, valCacheInt8);
                // kvScaleBuffers intentionally NOT populated in V2 row-inline mode — the scale
                // lives inside each KV row, so the native-op scale side-channel is disabled.

                log.info("[GGUF-KV] Layer {} quantized (row-inline scale): keyCache={} ({}x compression vs float)",
                        layerIdx, Arrays.toString(keyCacheInt8.shape()),
                        DataType.FLOAT.width() / DataType.INT8.width());
            }

            log.info("[GGUF-KV] Layer {} KV: kRoped shape={} min={} max={}, keyBuf shape={} min={} max={}",
                    layerIdx, Arrays.toString(kRoped.shape()), kRoped.minNumber(), kRoped.maxNumber(),
                    Arrays.toString(keyBuf.shape()), keyBuf.minNumber(), keyBuf.maxNumber());
            kRoped.close();
            vHeads.close();
        }

        // ── V2 QUANTIZED: free float KV after prefill quantize ────────────────────────────────────
        // ADR 0107 §prefill, §Migration Decision 8: after quantizing the full prefill region into
        // INT8, free the float staticKvBuffers immediately. The quantizedKvBuffers (INT8) + kvScaleBuffers
        // (float scales) become the ONLY live KV storage for the decode phase.
        //
        // V2 wiring: re-index quantizedKvBuffers under the ORIGINAL KV variable names
        // (same as staticKvBuffers keys) so the frozen decode plan's ext-input indices remain valid
        // — the plan sees the same slot index, now pointing to an INT8 buffer.
        boolean isQuantizedV2 = isQuantizedKv && (quantizedKvBuffers != null && !quantizedKvBuffers.isEmpty());
        if (isQuantizedV2) {
            // Build a new INT8 map under ORIGINAL key names (not _q suffixed)
            Map<String, INDArray> int8KvByOrigName = new LinkedHashMap<>();
            for (int i = 0; i < numLayers; i++) {
                String keyName = kvInputNames.keyNames.get(i);
                String valName = kvInputNames.valueNames.get(i);
                INDArray keyQ = quantizedKvBuffers.get(keyName + "_q");
                INDArray valQ = quantizedKvBuffers.get(valName + "_q");
                if (keyQ != null && valQ != null) {
                    int8KvByOrigName.put(keyName, keyQ);
                    int8KvByOrigName.put(valName, valQ);
                }
            }

            // Free the float staticKvBuffers — this is the live-memory reduction.
            // After this point, staticKvBuffers entries are closed and the map is cleared.
            if (reuseState == null) {
                // Fresh session: close all float buffers and release the map.
                for (INDArray arr : staticKvBuffers.values()) {
                    if (arr != null && !arr.wasClosed()) {
                        try { arr.close(); } catch (Exception e) {
                            log.warn("[GGUF-KV-V2] Error closing float KV buffer: {}", e.getMessage());
                        }
                    }
                }
                staticKvBuffers.clear();
                staticKvBuffers = null;
            } else {
                // Reuse path: the retained state's staticKvBuffers already ARE the float buffers
                // (reused in-place above). Close and null them on the retained state.
                if (reuseState.staticKvBuffers != null) {
                    for (INDArray arr : reuseState.staticKvBuffers.values()) {
                        if (arr != null && !arr.wasClosed()) {
                            try { arr.close(); } catch (Exception e) {
                                log.warn("[GGUF-KV-V2] Error closing reuse float KV buffer: {}", e.getMessage());
                            }
                        }
                    }
                    reuseState.staticKvBuffers.clear();
                    reuseState.staticKvBuffers = null;
                }
                staticKvBuffers = null;
            }

            // Replace quantizedKvBuffers with the original-name-keyed INT8 map.
            // The _q and _scale entries remain in kvScaleBuffers / quantizedKvBuffers under
            // original names for scale lookup in the native decode.
            quantizedKvBuffers = int8KvByOrigName;

            log.info("[GGUF-KV-V2] V2 quantized path active: float KV freed, {} INT8 buffers live",
                    int8KvByOrigName.size());
        }

        // Create recurrent state buffers from prefill outputs (keyed by input name). On reuse, assign
        // the new prefill state into the RETAINED buffer in place (stable address); else dup fresh.
        // CUDA assign/dup is asynchronous, so retain each source until the warmup's host-visible logits
        // prove that the dependent decode work has completed.
        List<INDArray> prefillRecurrentCopyDonors = new ArrayList<>();
        Map<String, INDArray> recurrentStateBuffers = reuseState != null && reuseState.recurrentStateBuffers != null
                ? reuseState.recurrentStateBuffers : new LinkedHashMap<>();
        for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
            INDArray stateOut = prefillOutputs.get(pair.outputName);
            if (stateOut != null) {
                INDArray existing = recurrentStateBuffers.get(pair.inputName);
                if (reuseState != null && existing != null) {
                    existing.assign(stateOut);
                } else {
                    recurrentStateBuffers.put(pair.inputName, stateOut.dup());
                }
                prefillRecurrentCopyDonors.add(stateOut);
                log.info("[GGUF-KV] Recurrent state {} → {} shape={}", pair.inputName, pair.outputName,
                        Arrays.toString(recurrentStateBuffers.get(pair.inputName).shape()));
            } else {
                log.warn("[GGUF-KV] Missing prefill output for recurrent state '{}'", pair.outputName);
            }
        }

        // ══════════════════════════════════════════════════════════════════════
        // STEP 3: Warmup decode step -- compile DSP plan for decode shapes
        //
        // When fixedBuffers=true, shapes are ALWAYS the same (seqLen=1 decode,
        // mask width=maxKvLen). Position tracking uses actualPrefillLen so the
        // decode step writes to the correct KV position after real tokens.
        // ══════════════════════════════════════════════════════════════════════
        // The first decode position. With fixed buffers, the model's internal KV
        // cache was filled at 0..prefillSeqLen-1, but only 0..actualPrefillLen-1
        // are real content. We write the first decode token at actualPrefillLen.
        int firstDecodePos = actualPrefillLen;

        // Freeze the decode graph at the policy's fixed W envelope. Inactive slots remain
        // allocated and masked, preserving DSP/CUDA-graph pointer and shape stability.
        DecodePolicy freezePolicy = resolveDecodePolicy(sampling, config);
        int decodeWidth = Math.max(1, freezePolicy.windowMax);
        // A frozen pipeline's window envelope is a property of the compiled plan,
        // not of the per-generation sampling policy: a narrower policy (e.g. greedy
        // selected at runtime after a speculative freeze) runs as activeWindow=1 on
        // the SAME W-wide plan — greedy is the 1x1 special case of the substrate
        // (ADR 0106). Only widening beyond the frozen envelope forces a re-freeze.
        if (reuseState != null && reuseState.decodeInputIds != null
                && reuseState.decodeInputIds.size(1) > decodeWidth) {
            decodeWidth = (int) reuseState.decodeInputIds.size(1);
        }
        INDArray decodeInputIds;
        if (reuseState != null && reuseState.decodeInputIds != null
                && reuseState.decodeInputIds.size(1) == decodeWidth) {
            decodeInputIds = reuseState.decodeInputIds;
            decodeInputIds.assign(0);
        } else {
            decodeInputIds = Nd4j.zeros(DataType.INT64, 1, decodeWidth);
        }
        decodeInputIds.putScalar(new long[]{0, 0}, firstTokenId);

        INDArray decodeCausalMask = null;
        if (causalMaskName != null && decoder.hasVariable(causalMaskName)) {
            INDArray freshMask = decodeWidth > 1
                    ? DecoderInputBuilder.buildInGraphWindowMask(
                            DecoderInputBuilder.chainParents(1, decodeWidth),
                            firstDecodePos, 1, decodeWidth, maxKvLen, maskDtype)
                    : DecoderInputBuilder.buildInGraphDecodeMask(firstDecodePos, maxKvLen, maskDtype);
            if (reuseState != null && reuseState.decodeCausalMask != null
                    && Arrays.equals(reuseState.decodeCausalMask.shape(), freshMask.shape())) {
                decodeCausalMask = reuseState.decodeCausalMask;
                decodeCausalMask.assign(freshMask);
                freshMask.close();
            } else {
                decodeCausalMask = freshMask;
            }
            log.info("[GGUF-KV] Decode mask shape={} dtype={} firstDecodePos={} min={} max={} hasNaN={}",
                    Arrays.toString(decodeCausalMask.shape()), decodeCausalMask.dataType(),
                    firstDecodePos,
                    decodeCausalMask.minNumber(), decodeCausalMask.maxNumber(),
                    Double.isNaN(decodeCausalMask.meanNumber().doubleValue()));
        }
        INDArray decodePositionOffset = null;
        if (posOffsetName != null && decoder.hasVariable(posOffsetName)) {
            if (reuseState != null && reuseState.decodePositionOffset != null) {
                decodePositionOffset = reuseState.decodePositionOffset;
                decodePositionOffset.putScalar(new long[]{}, (long) firstDecodePos);
            } else {
                decodePositionOffset = Nd4j.scalar(DataType.INT64, firstDecodePos);
            }
        }
        INDArray decodeCachePosition = null;
        if (cachePosName != null && decoder.hasVariable(cachePosName)) {
            if (reuseState != null && reuseState.decodeCachePosition != null) {
                decodeCachePosition = reuseState.decodeCachePosition;
                decodeCachePosition.putScalar(new long[]{}, (long) firstDecodePos);
            } else {
                decodeCachePosition = Nd4j.scalar(DataType.INT64, firstDecodePos);
            }
        }
        INDArray decodeActualSequenceLength = null;
        if (decoder.hasVariable(ACTUAL_SEQUENCE_LENGTH_NAME)) {
            if (reuseState != null && reuseState.decodeActualSequenceLength != null) {
                decodeActualSequenceLength = reuseState.decodeActualSequenceLength;
                decodeActualSequenceLength.putScalar(new long[]{}, 1L);
            } else {
                decodeActualSequenceLength = Nd4j.scalar(DataType.INT64, 1L);
            }
        }

        Map<String, INDArray> decodeInputMap = new HashMap<>();
        decodeInputMap.put(inputIdsName, decodeInputIds);
        if (decodeCausalMask != null) decodeInputMap.put(causalMaskName, decodeCausalMask);
        if (decodePositionOffset != null) decodeInputMap.put(posOffsetName, decodePositionOffset);
        if (decodeCachePosition != null) decodeInputMap.put(cachePosName, decodeCachePosition);
        if (decodeActualSequenceLength != null) {
            decodeInputMap.put(ACTUAL_SEQUENCE_LENGTH_NAME, decodeActualSequenceLength);
        }
        // V2 QUANTIZED: use INT8 quantizedKvBuffers as KV ext inputs (float freed).
        // V1 / non-quantized: use float staticKvBuffers as before.
        Map<String, INDArray> kvSourceMap = (isQuantizedV2 && quantizedKvBuffers != null)
                ? quantizedKvBuffers : staticKvBuffers;
        if (kvSourceMap != null) {
            for (Map.Entry<String, INDArray> entry : kvSourceMap.entrySet()) {
                if (decoder.hasVariable(entry.getKey())) {
                    decodeInputMap.put(entry.getKey(), entry.getValue());
                }
            }
        }
        // Add GDN/conv state buffers to decode input map
        for (Map.Entry<String, INDArray> entry : recurrentStateBuffers.entrySet()) {
            if (decoder.hasVariable(entry.getKey())) {
                decodeInputMap.put(entry.getKey(), entry.getValue());
            }
        }
        // The last-position projection is a prefill-only optimization. Decode is S=1 and must use
        // the canonical logits output; DSP already selects a distinct plan for the decode shape.
        // Retain the remaining ordered state outputs so the native recurrent-state contract is stable.
        List<String> decodeOutputNames = new ArrayList<>(prefillOutputNames);
        decodeOutputNames.set(0, logitsName);
        int kvBufCount = (isQuantizedV2 && quantizedKvBuffers != null) ? quantizedKvBuffers.size()
                : (staticKvBuffers != null ? staticKvBuffers.size() : 0);
        log.info("[GGUF-KV] STEP 3: warmup decode with {} KV buffers (V2={}), {} recurrent state buffers, {} inputs",
                kvBufCount, isQuantizedV2, recurrentStateBuffers.size(), decodeInputMap.size());

        Map<String, INDArray> decodeOutputs;
        try {
            decodeOutputs = decoder.output(decodeInputMap, decodeOutputNames.toArray(new String[0]));
        } catch (Exception e) {
            log.error("[GGUF-KV] STEP 3 warmup decode failed", e);
            throw e;
        }

        INDArray targetWarmupHidden = useNativeMtp ? decodeOutputs.get(TARGET_HIDDEN_STATES_NAME) : null;
        if (useNativeMtp && targetWarmupHidden == null) {
            throw new IllegalStateException("Bundled MTP graph did not return " + TARGET_HIDDEN_STATES_NAME
                    + " during target warmup");
        }
        INDArray decodeLogits = decodeOutputs.get(logitsName);
        suppressStopsUnderFloor(decodeLogits, 0, sampling, generatedSoFar.size(), stopTokenIds);
        int secondTokenId = sampleToken(
                decodeLogits, 0, sampling, generatedSoFar, rng,
                constraintMasker, tokenizer);
        generatedSoFar.add(secondTokenId);
        log.info("[GGUF-KV] STEP 3 second token: {}", secondTokenId);
        decodeLogits.close();

        // The warmup consumed firstTokenId and sampled secondTokenId. The native loop starts at the
        // following cache position, so its first execution must consume secondTokenId with matching
        // position/cache scalars. The ONNX path performs this same handoff before entering native
        // decode; omitting it here replays stale warmup inputs during DSP phase transitions.
        prepareInGraphNativeDecodeHandoff(
                decodeInputIds, decodePositionOffset, decodeCachePosition,
                secondTokenId, firstDecodePos + 1L);

        // Reading the sampled token is the existing host-visible completion boundary for warmup. The
        // prefill-copy sources (and any prior two-token run's deferred warmup sources) are now safe to
        // release without introducing a manual device or stream synchronization.
        for (INDArray donor : prefillRecurrentCopyDonors) {
            if (donor != null && !donor.wasClosed()) donor.close();
        }
        prefillRecurrentCopyDonors.clear();
        if (reuseState != null) reuseState.releaseRecurrentCopyDonors();

        // ══════════════════════════════════════════════════════════════════════
        // STEP 4: Get native plan handle, freeze shapes, resolve ext indices
        // ══════════════════════════════════════════════════════════════════════
        InferenceSession session = decoder.getOrCreateSession();
        DynamicShapePlanExecutor executor = session != null ? session.getDynamicShapePlanExecutor() : null;
        Pointer planHandle = executor != null ? executor.getNativePlanHandle() : null;
        boolean nativePlanAvailable = planHandle != null && !planHandle.isNull();

        // Update recurrent state buffers with outputs from warmup decode. Keep each CUDA D2D source
        // alive until the native decode returns host-visible token/timing data.
        List<INDArray> warmupRecurrentCopyDonors = new ArrayList<>();
        for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
            INDArray updated = decodeOutputs.get(pair.outputName);
            if (updated != null) {
                INDArray buf = recurrentStateBuffers.get(pair.inputName);
                if (nativePlanAvailable && buf != null) {
                    buf.assign(updated);
                    warmupRecurrentCopyDonors.add(updated);
                } else {
                    updated.close();
                }
            }
        }
        if (!warmupRecurrentCopyDonors.isEmpty()) {
            // assign() is queued on the array's backend stream, while the native decode plan
            // executes on its dedicated DSP stream. Establish a one-time completion boundary
            // before handing these device-managed recurrent inputs to the native loop; retaining
            // the donors protects their lifetime but does not order work between the two streams.
            Nd4j.getExecutioner().commit();
        }

        if (!nativePlanAvailable) {
            closeGeneratedKvOutputs(decodeOutputs, kvInputNames);
            if (targetPrefillHidden != null && !targetPrefillHidden.wasClosed()) targetPrefillHidden.close();
            if (targetWarmupHidden != null && !targetWarmupHidden.wasClosed()) targetWarmupHidden.close();
            log.warn("Native plan handle not available for GGUF -- returning partial result");
            // On reuse these buffers are the retained state's own — leave them for the caller to free
            // via reuseState.close(); only the non-reuse path owns and frees them inline here.
            if (reuseState == null) {
                decodeInputIds.close();
                if (decodeCausalMask != null) decodeCausalMask.close();
                if (decodePositionOffset != null) decodePositionOffset.close();
                if (decodeCachePosition != null) decodeCachePosition.close();
                if (decodeActualSequenceLength != null) decodeActualSequenceLength.close();
                for (INDArray kv : staticKvBuffers.values()) kv.close();
                for (INDArray rs : recurrentStateBuffers.values()) rs.close();
                for (INDArray v : prefillInputMap.values()) { if (v != null && !v.wasClosed()) v.close(); }
            }

            List<Integer> tokens = new ArrayList<>();
            tokens.add(firstTokenId);
            if (!stopTokenIds.contains(firstTokenId)) tokens.add(secondTokenId);
            InGraphKvState terminal = new InGraphKvState();
            terminal.terminalResult = buildResult(tokens, promptTokenIds, stopTokenIds, startTime, firstTokenMs);
            terminal.generatedSoFar = tokens;
            terminal.closed = true;
            terminal.actualPrefillLen = actualPrefillLen;
            terminal.promptTokenCount = prefillSeqLen;
            terminal.maxKvLen = maxKvLen;
            return terminal;
        }

        if (reuseState == null && executor.getCurrentPlan() != null) {
            // Reuse: the plan is already frozen + KV-cache-configured from the first generate; a second
            // setShapesFrozen(true) / configureMaxAllocation would perturb the live captured plan. Skip.
            executor.setMaxKvCacheLength((int) maxKvLen);
            executor.configureMaxAllocationForKvCache(decodeOutputs);
            log.info("[Perf] GGUF configured KV cache max-allocation: maxKvLen={}", maxKvLen);
            boolean forcedSlotBySlot = decoder.getGraphExecutionMode() == GraphExecutionMode.SLOT_BY_SLOT
                    || Nd4j.getEnvironment().tritonSkipKernels();
            if (forcedSlotBySlot) {
                log.info("[Perf] GGUF keeping slot-by-slot plan unfrozen (mode={} tritonSkipKernels={} planPhase={} pointersStable={})",
                        decoder.getGraphExecutionMode(), Nd4j.getEnvironment().tritonSkipKernels(),
                        executor.getPlanPhase(), executor.arePointersStable());
            } else {
                executor.setShapesFrozen(true);
                log.info("[Perf] GGUF shapes frozen after warmup decode (planPhase={} pointersStable={})",
                        executor.getPlanPhase(), executor.arePointersStable());
                if ("true".equalsIgnoreCase(System.getProperty(ND4JSystemProperties.VLM_BENCHMARK_OP_TIMING, "false"))) {
                    executor.setExecutionTimingEnabled(true);
                    log.info("[Perf] GGUF decoder execution timing enabled");
                }
            }
        }

        // Decode requested K/V outputs only to keep the DSP output contract identical to prefill.
        // The native loop updates the retained KV inputs in-place, so these one-step arrays are not owners.
        closeGeneratedKvOutputs(decodeOutputs, kvInputNames);

        int inputIdsExtIdx = resolveExtInputIdx(executor, inputIdsName);
        int causalMaskExtIdx = causalMaskName != null ? resolveExtInputIdx(executor, causalMaskName) : -1;
        int posOffsetExtIdx = posOffsetName != null ? resolveExtInputIdx(executor, posOffsetName) : -1;
        int cachePosExtIdx = cachePosName != null ? resolveExtInputIdx(executor, cachePosName) : -1;
        int actualSeqLenExtIdx = decoder.hasVariable(ACTUAL_SEQUENCE_LENGTH_NAME)
                ? resolveExtInputIdx(executor, ACTUAL_SEQUENCE_LENGTH_NAME) : -1;
        int logitsOutputIdx = resolveOutputIdx(executor, logitsName);
        int targetHiddenOutputIdx = useNativeMtp
                ? resolveOutputIdx(executor, TARGET_HIDDEN_STATES_NAME) : -1;

        int embeddingsExtIdx = -1;
        int maskExtIdx = -1;
        int posIdsExtIdx = -1;

        int numKvPairs = numLayers;
        int[] kvInputExtIndices = new int[2 * numKvPairs];
        int[] kvOutputIndices = new int[0];
        int ki = 0;
        for (String keyName : kvInputNames.keyNames) {
            kvInputExtIndices[ki++] = resolveExtInputIdx(executor, keyName);
        }
        for (String valName : kvInputNames.valueNames) {
            kvInputExtIndices[ki++] = resolveExtInputIdx(executor, valName);
        }

        // Resolve recurrent state ext input and output indices, split by op type
        // (AutoregressiveDecode C++ op expects GDN and conv indices separately)
        List<Integer> gdnExtList = new ArrayList<>(), gdnOutList = new ArrayList<>();
        List<Integer> convExtList = new ArrayList<>(), convOutList = new ArrayList<>();
        for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
            int extIdx = resolveExtInputIdx(executor, pair.inputName);
            int outIdx = resolveOutputIdx(executor, pair.outputName);
            if (pair.isGdn()) {
                gdnExtList.add(extIdx);
                gdnOutList.add(outIdx);
            } else {
                // All non-GDN recurrent states (conv, or any future type) go in the conv slot
                convExtList.add(extIdx);
                convOutList.add(outIdx);
            }
        }
        int[] gdnStateExtIndices = gdnExtList.stream().mapToInt(Integer::intValue).toArray();
        int[] gdnStateOutputIndices = gdnOutList.stream().mapToInt(Integer::intValue).toArray();
        int[] convStateExtIndices = convExtList.stream().mapToInt(Integer::intValue).toArray();
        int[] convStateOutputIndices = convOutList.stream().mapToInt(Integer::intValue).toArray();

        MtpPreparedState preparedMtp = null;
        if (useNativeMtp) {
            preparedMtp = prepareBundledMtp(
                    reuseState,
                    effectiveTokenIds,
                    prefillSeqLen,
                    actualPrefillLen,
                    maxKvLen,
                    firstDecodePos,
                    firstTokenId,
                    secondTokenId,
                    targetPrefillHidden,
                    targetWarmupHidden);
            // This source backs the queued h_P → MTP carry copy. The native decode's returned token
            // count is the next natural completion boundary, so retain it with the other warmup donors.
            warmupRecurrentCopyDonors.add(targetWarmupHidden);
        }

        // ══════════════════════════════════════════════════════════════════════
        // Build the retained decode state. The native decode loop (former STEP 5) now lives in
        // runInGraphNativeDecode(), shared by the one-shot path and by continuation. cachePosition
        // starts at firstDecodePos+1 (== actualPrefillLen+1): the position where the warmup's
        // secondToken (the last unwritten token) will be written when it is fed.
        // ══════════════════════════════════════════════════════════════════════
        Pointer contextHandle = executor.getCachedOpContext();
        int numPlanExternalInputs = executor.getCurrentPlan() != null
                ? executor.getCurrentPlan().getExternalInputKeys().length : 0;
        int numPlanOutputs = decodeOutputNames.size();

        // On reuse, write back into the SAME retained state object (its buffers ARE the ones just
        // refilled in place) so no buffer is aliased by two states → no double-free. The index / handle
        // / running-state fields are refreshed below; the buffer fields are identity assignments.
        InGraphKvState state = (reuseState != null) ? reuseState : new InGraphKvState();
        // V2 QUANTIZED: float staticKvBuffers are freed; quantizedKvBuffers holds INT8 live storage.
        // V1 / STATIC: staticKvBuffers holds float live storage; quantizedKvBuffers is archive or null.
        state.staticKvBuffers = isQuantizedV2 ? null : staticKvBuffers;
        state.recurrentStateBuffers = recurrentStateBuffers;
        // QUANTIZED KV buffers: in V2, this is the INT8 live store under original variable names.
        state.quantizedKvBuffers = quantizedKvBuffers;
        state.kvScaleBuffers = kvScaleBuffers;
        state.kvQuantFormat = kvQuantFormat;
        state.isQuantizedV2 = isQuantizedV2;
        state.decodeInputIds = decodeInputIds;
        state.decodeCausalMask = decodeCausalMask;
        state.decodePositionOffset = decodePositionOffset;
        state.decodeCachePosition = decodeCachePosition;
        state.decodeActualSequenceLength = decodeActualSequenceLength;
        state.executor = executor;
        state.planHandle = planHandle;
        state.contextHandle = contextHandle;
        state.inputIdsExtIdx = inputIdsExtIdx;
        state.causalMaskExtIdx = causalMaskExtIdx;
        state.posOffsetExtIdx = posOffsetExtIdx;
        state.cachePosExtIdx = cachePosExtIdx;
        state.actualSeqLenExtIdx = actualSeqLenExtIdx;
        state.logitsOutputIdx = logitsOutputIdx;
        state.targetHiddenOutputIdx = targetHiddenOutputIdx;
        state.embeddingsExtIdx = embeddingsExtIdx;
        state.maskExtIdx = maskExtIdx;
        state.posIdsExtIdx = posIdsExtIdx;
        state.kvInputExtIndices = kvInputExtIndices;
        state.kvOutputIndices = kvOutputIndices;
        state.gdnStateExtIndices = gdnStateExtIndices;
        state.gdnStateOutputIndices = gdnStateOutputIndices;
        state.convStateExtIndices = convStateExtIndices;
        state.convStateOutputIndices = convStateOutputIndices;
        state.numPlanExternalInputs = numPlanExternalInputs;
        state.numPlanOutputs = numPlanOutputs;
        state.numKvPairs = numKvPairs;
        if (preparedMtp != null) {
            state.mtpKvBuffers = preparedMtp.kvBuffers;
            state.mtpPrefillInputMap = preparedMtp.prefillInputMap;
            state.mtpInputIds = preparedMtp.inputIds;
            state.mtpTargetHiddenStates = preparedMtp.targetHiddenStates;
            state.mtpCausalMask = preparedMtp.causalMask;
            state.mtpPositionOffset = preparedMtp.positionOffset;
            state.mtpCachePosition = preparedMtp.cachePosition;
            state.mtpSession = preparedMtp.session;
            state.mtpExecutor = preparedMtp.executor;
            state.mtpPlanHandle = preparedMtp.planHandle;
            state.mtpContextHandle = preparedMtp.contextHandle;
            state.mtpInputIdsExtIdx = preparedMtp.inputIdsExtIdx;
            state.mtpTargetHiddenExtIdx = preparedMtp.targetHiddenExtIdx;
            state.mtpCausalMaskExtIdx = preparedMtp.causalMaskExtIdx;
            state.mtpPosOffsetExtIdx = preparedMtp.positionOffsetExtIdx;
            state.mtpCachePosExtIdx = preparedMtp.cachePositionExtIdx;
            state.mtpKvInputExtIndices = preparedMtp.kvInputExtIndices;
            state.mtpLogitsOutputIdx = preparedMtp.logitsOutputIdx;
            state.mtpHiddenOutputIdx = preparedMtp.hiddenOutputIdx;
            state.mtpNumPlanExternalInputs = preparedMtp.numPlanExternalInputs;
            state.mtpNumPlanOutputs = preparedMtp.numPlanOutputs;
        }
        state.kvInputNames = kvInputNames;
        state.recurrentStates = recurrentStates;
        state.decodeOutputNames = decodeOutputNames;
        state.cachePosition = firstDecodePos + 1;
        state.lastGeneratedToken = secondTokenId;
        state.generatedSoFar = generatedSoFar;   // holds [firstToken, secondToken] at this point
        state.rng = rng;
        state.sampling = sampling;
        state.constraintMasker = constraintMasker;
        state.stopTokenIds = stopTokenIds;
        state.eosTokenId = eosTokenId;
        state.maxKvLen = maxKvLen;
        state.actualPrefillLen = actualPrefillLen;
        state.prefillSeqLen = prefillSeqLen;
        state.promptTokenCount = prefillSeqLen;
        state.maskDtype = maskDtype;
        // ── Rotating KV cache wiring ──────────────────────────────────────────────────────────────
        // Only initialise when the config requests it AND when reuse is null (fresh state) — on the
        // fixed-buffer reuse path the slot map is already set on the retained state object.
        if (config.isRotatingKvEnabled() && reuseState == null) {
            int effectiveSinkCount = RotatingKvSlotMap.resolveSinkCount(config.getRotatingKvSinkCount());
            state.rotatingSlotMap = new RotatingKvSlotMap((int) maxKvLen, effectiveSinkCount);
            log.info("[RotatingKV] enabled: maxKvLen={} sinkCount={} ringSize={}",
                    maxKvLen, effectiveSinkCount, state.rotatingSlotMap.getRingSize());
        } else if (reuseState != null) {
            // reuse path: keep whatever was on the state (set during first create)
            state.rotatingSlotMap = reuseState.rotatingSlotMap;
        }
        // else: rotating disabled — rotatingSlotMap stays null (default); all existing paths unchanged.
        state.inputIdsName = inputIdsName;
        state.logitsName = logitsName;
        state.causalMaskName = causalMaskName;
        state.posOffsetName = posOffsetName;
        state.cachePosName = cachePosName;
        state.actualSeqLenName = decoder.hasVariable(ACTUAL_SEQUENCE_LENGTH_NAME) ? ACTUAL_SEQUENCE_LENGTH_NAME : null;
        state.prefillInputMap = prefillInputMap;
        state.retainRecurrentCopyDonors(warmupRecurrentCopyDonors);
        if (reuseState != null) {
            // Clear transient per-generate flags carried over from the previous generate.
            state.eosReached = false;
            state.cancelRequested = false;
            state.terminalResult = null;
            state.closed = false;
        }
        return state;
    }

    /**
     * Move the Java warmup result into the address-stable arrays consumed by native in-graph decode.
     * Package-private for the handoff regression test.
     */
    static void prepareInGraphNativeDecodeHandoff(
            INDArray decodeInputIds,
            INDArray decodePositionOffset,
            INDArray decodeCachePosition,
            int nextInputTokenId,
            long nextCachePosition) {
        decodeInputIds.putScalar(new long[]{0, 0}, nextInputTokenId);
        if (decodePositionOffset != null) {
            decodePositionOffset.putScalar(new long[]{}, nextCachePosition);
        }
        if (decodeCachePosition != null) {
            decodeCachePosition.putScalar(new long[]{}, nextCachePosition);
        }

        // Preserve the stable array identities used by the frozen plan while making the Java writes
        // visible before the native loop switches to its dedicated DSP stream.
        Nd4j.getExecutioner().commit();
        decodeInputIds.syncToDevice();
        if (decodePositionOffset != null) decodePositionOffset.syncToDevice();
        if (decodeCachePosition != null) decodeCachePosition.syncToDevice();
    }

    /**
     * Tokenize a prompt using the tokenizer-owned chat-template handling.
     * Shared by generation, streaming, and continuation sessions.
     */
    private int[] encodePromptToIds(String prompt) {
        int[] promptTokenIds = tokenizer.encodePrompt(prompt, effectiveChatTemplateText()).getIds();
        return requirePromptTokenIds(promptTokenIds);
    }

    private int[] encodeFormattedChatToIds(String formattedPrompt) {
        int[] promptTokenIds = tokenizer.ensureLeadingBos(
                tokenizer.encode(formattedPrompt, false)).getIds();
        return requirePromptTokenIds(promptTokenIds);
    }

    private static int[] requirePromptTokenIds(int[] promptTokenIds) {
        if (promptTokenIds == null || promptTokenIds.length == 0) {
            throw new IllegalArgumentException("Prompt encoding produced no tokens");
        }
        return promptTokenIds;
    }

    // ==================== Cross-request KV Prefix Cache ====================

    /**
     * Container returned by {@link #attemptPrefixCacheHit} when the prefix cache
     * has a match and blocks have been restored into pre-allocated static KV buffers.
     */
    private static final class PrefixHitContext {
        /** Number of tokens restored from cache. Always a positive multiple of blockSize. */
        final int matchedTokenCount;
        /** Pre-allocated static KV buffers with blocks 0..matchedTokenCount/blockSize-1 filled. */
        final Map<String, INDArray> restoredKvBuffers;
        /**
         * Recurrent state snapshot from the cached prefix, or null.
         * Non-null only on exact-block-boundary hits where a snapshot was stored.
         * When null, the suffix prefill must use zero recurrent state (correct for non-recurrent
         * models; for GDN models this triggers a full fallback — see attemptPrefixCacheHit docs).
         */
        final Map<String, INDArray> recurrentSnapshot;

        PrefixHitContext(int matchedTokenCount, Map<String, INDArray> restoredKvBuffers,
                         Map<String, INDArray> recurrentSnapshot) {
            this.matchedTokenCount = matchedTokenCount;
            this.restoredKvBuffers = restoredKvBuffers;
            this.recurrentSnapshot = recurrentSnapshot;
        }
    }

    /**
     * Attempt a prefix cache lookup and restore KV blocks.
     *
     * <p>Returns a {@link PrefixHitContext} if the cache has a usable hit, otherwise {@code null}.
     * On a hit, device-resident KV blocks are D2D-copied into freshly allocated static KV buffers
     * (shaped {@code [1, maxKvLen, kvH, headDim]}) and the context carries the number of tokens
     * that were restored.</p>
     *
     * <p><strong>Recurrent-state policy:</strong> a recurrent-state snapshot is included in the
     * context only if the match is an exact block-boundary hit AND a snapshot was registered for
     * that boundary. Callers must check {@code context.recurrentSnapshot != null}:</p>
     * <ul>
     *   <li>Non-null: GDN/recurrent models can reuse the snapshot for the suffix prefill.</li>
     *   <li>Null: the suffix prefill uses zero recurrent state. For non-recurrent models (most
     *       transformers) this is correct. For GDN models on a partial-boundary hit this is
     *       INCORRECT — the caller must fall back to full prefill by ignoring the hit context.</li>
     * </ul>
     *
     * @param promptTokenIds  full prompt token sequence
     * @param kvInputNames    KV cache layer names (keys then values)
     * @param maxKvLen        total KV buffer length (for buffer allocation)
     * @param recurrentStates the recurrent state pairs discovered from the model graph
     * @return hit context or null
     */
    private PrefixHitContext attemptPrefixCacheHit(int[] promptTokenIds,
                                                    ModelIOConfig.KVCacheNames kvInputNames,
                                                    long maxKvLen,
                                                    List<ModelIOConfig.RecurrentStatePair> recurrentStates) {
        if (prefixBlockPool == null) return null;

        PrefixLookupResult lookup = prefixBlockPool.getRadixCache().lookup(promptTokenIds);
        if (!lookup.hasMatch()) return null;

        int matchedTokenCount = lookup.getMatchedTokenCount();
        if (matchedTokenCount <= 0) return null;

        log.info("[PrefixCache] HIT: {} tokens matched ({} blocks saved), prompt={} tokens",
                matchedTokenCount, lookup.getSharedBlockIds().length, promptTokenIds.length);

        // --- GDN guard: if the model has recurrent state and we do NOT have an exact-boundary
        // snapshot, fall back to full prefill to avoid producing wrong recurrent state.
        boolean hasRecurrentState = recurrentStates != null && !recurrentStates.isEmpty();
        Map<String, INDArray> recurrentSnapshot = null;
        if (hasRecurrentState) {
            recurrentSnapshot = prefixBlockPool.getRecurrentSnapshot(matchedTokenCount);
            if (recurrentSnapshot == null) {
                // No exact-boundary snapshot — we cannot safely continue with partial hit for GDN.
                log.warn("[PrefixCache] GDN model with partial-boundary prefix hit (matched={} tokens, "
                        + "no recurrent snapshot) — falling back to full prefill. "
                        + "To get prefix reuse for this model, ensure the cached prefix is exactly "
                        + "block-aligned AND the previous prefill stored a recurrent snapshot.",
                        matchedTokenCount);
                return null;
            }
        }

        // Allocate fresh static KV buffers (zero-initialized) and restore the cached blocks.
        // Buffer shape is inferred from the decoder variable shapes (same as in prefillWarmupAndFreeze).
        int numLayers = kvInputNames.keyNames.size();
        List<String> allLayerNames = new ArrayList<>();
        allLayerNames.addAll(kvInputNames.keyNames);
        allLayerNames.addAll(kvInputNames.valueNames);

        Map<String, INDArray> restoredKvBuffers = new LinkedHashMap<>();
        boolean allocFailed = false;
        for (String layerName : allLayerNames) {
            if (!decoder.hasVariable(layerName)) continue;
            long[] varShape = decoder.getVariable(layerName).getShape();
            if (varShape == null || varShape.length < 4) {
                log.warn("[PrefixCache] Cannot determine shape for KV layer '{}' — skipping cache hit", layerName);
                allocFailed = true;
                break;
            }
            // varShape is [batch, maxKvLen (placeholder), kvH, headDim] from the model variable
            // The actual maxKvLen is our computed value.
            long batch = varShape[0] > 0 ? varShape[0] : 1L;
            long kvH = varShape[2];
            long headDim = varShape[3];
            org.nd4j.linalg.api.buffer.DataType kvDtype = decoder.getVariable(layerName).dataType();
            restoredKvBuffers.put(layerName, Nd4j.zeros(kvDtype, batch, maxKvLen, kvH, headDim));
        }

        if (allocFailed) {
            // Clean up partially allocated buffers
            for (INDArray buf : restoredKvBuffers.values()) {
                if (buf != null && !buf.wasClosed()) buf.close();
            }
            return null;
        }

        // Restore blocks from the pool into the allocated buffers
        int restoredTokens = prefixBlockPool.restoreBlocks(promptTokenIds, restoredKvBuffers, allLayerNames);
        if (restoredTokens <= 0) {
            log.warn("[PrefixCache] restoreBlocks returned 0 (eviction race?) — falling back to full prefill");
            for (INDArray buf : restoredKvBuffers.values()) {
                if (buf != null && !buf.wasClosed()) buf.close();
            }
            return null;
        }

        // Sync restored buffers to device
        Nd4j.getExecutioner().commit();
        for (INDArray buf : restoredKvBuffers.values()) {
            if (buf != null && !buf.isEmpty()) buf.syncToDevice();
        }

        log.info("[PrefixCache] restored {} tokens into {} KV buffers (recurrentSnapshot={})",
                restoredTokens, restoredKvBuffers.size(), recurrentSnapshot != null ? "yes" : "no");
        return new PrefixHitContext(restoredTokens, restoredKvBuffers, recurrentSnapshot);
    }

    /**
     * Run suffix-only prefill for a prefix cache hit, then complete warmup and freeze.
     *
     * <p>Called when a prefix cache lookup found a hit covering tokens 0..M-1 where M =
     * {@code hit.matchedTokenCount}. The caller provides pre-filled static KV buffers for the
     * prefix region; this method fills positions M..P-1 (the suffix) by running the decoder on
     * suffix tokens with position offset M, then proceeds with the warmup decode and plan freeze
     * exactly as in {@link #prefillWarmupAndFreeze}.</p>
     *
     * @param promptTokenIds  full original prompt token IDs (not just the suffix)
     * @param maxNewTokens    decode budget (used to size the KV buffer)
     * @param kvInputNames    KV cache layer names
     * @param startTime       wall-clock start time for TTFT calculation
     * @param hit             the prefix hit context from {@link #attemptPrefixCacheHit}
     * @return an {@link InGraphKvState} ready for native decode, identical in contract to
     *         {@link #prefillWarmupAndFreeze}'s return value
     */
    private InGraphKvState prefillSuffixOnlyAndFreeze(int[] promptTokenIds, int maxNewTokens,
                                                       ModelIOConfig.KVCacheNames kvInputNames,
                                                       long startTime,
                                                       PrefixHitContext hit) {
        int matchedLen = hit.matchedTokenCount;
        int fullLen = promptTokenIds.length;
        int suffixLen = fullLen - matchedLen;

        // If the entire prompt is cached (suffix is empty), skip suffix prefill entirely —
        // the next token is sampled from a decode step on the last cached token's logits.
        // We still need warmup + freeze, so we treat the last cached position as the "prefill" result.
        // For simplicity, if suffixLen == 0, treat it as matchedLen-1 suffix of length 1 (the last token).
        // This is unusual (exact full-prompt hit), but handle gracefully.
        if (suffixLen <= 0) {
            // Fall back to full prefill — the prompt exactly matches a cached prefix (all P tokens).
            // The warmup step will re-sample, but we still save P-1 tokens of prefill time.
            // We do the warmup on just the last token (position P-1) rather than full prefill.
            // Simplest correct approach: just run full prefill (cache hit saves nothing if suffix=0,
            // but this is an edge case; correctness trumps optimization here).
            log.info("[PrefixCache] Exact full-prompt cache hit (suffixLen=0) — using full prefill for warmup correctness");
            for (INDArray buf : hit.restoredKvBuffers.values()) {
                if (buf != null && !buf.wasClosed()) buf.close();
            }
            return prefillWarmupAndFreeze(promptTokenIds, maxNewTokens, kvInputNames, startTime, null);
        }

        int[] suffixTokenIds = new int[suffixLen];
        System.arraycopy(promptTokenIds, matchedLen, suffixTokenIds, 0, suffixLen);

        // Compute the KV buffer length (same logic as in prefillWarmupAndFreeze)
        int actualPrefillLen = fullLen;  // the full prompt is the "prefill" from the model's perspective
        int prefillSeqLen = fullLen;
        long maxKvLen = prefillSeqLen + maxNewTokens;
        int kvCap = config.getMaxKvCacheLength();
        if (kvCap > 0 && maxKvLen > kvCap) {
            maxKvLen = kvCap;
            maxNewTokens = (int) (maxKvLen - prefillSeqLen);
            if (maxNewTokens < 1) maxNewTokens = 1;
        }
        int numLayers = kvInputNames.keyNames.size();
        SamplingConfig sampling = activeDecodeSampling();
        int eosTokenId = resolveEosTokenId(sampling);
        Set<Integer> stopTokenIds = buildStopTokenIds(eosTokenId);
        DecodePolicy decodePolicy = activeDecodePolicy();
        requireNativeSubstrateAvailable(decodePolicy, sampling);
        java.util.Random rng = sampling.getSeed() != null
                ? new java.util.Random(sampling.getSeed()) : new java.util.Random();

        String inputIdsName  = ioConfig.getInputIdsName() != null ? ioConfig.getInputIdsName() : "input_ids";
        String logitsName    = ioConfig.getLogitsOutputName() != null ? ioConfig.getLogitsOutputName() : "lm_logits";
        String posOffsetName = ioConfig.getPositionOffsetName();
        String cachePosName  = ioConfig.getCachePositionName();
        String causalMaskName = ioConfig.getCausalMaskName();

        List<ModelIOConfig.RecurrentStatePair> recurrentStates =
                ModelIOConfig.findRecurrentStatePairs(decoder, ioConfig);

        DataType maskDtype = DataType.FLOAT;
        if (causalMaskName != null && decoder.hasVariable(causalMaskName)) {
            maskDtype = decoder.getVariable(causalMaskName).dataType();
        }

        // ── SUFFIX PREFILL ─────────────────────────────────────────────────────────────────────────
        // Run the decoder on suffix tokens M..P-1.  Position offset = M so RoPE is correct.
        // The causal mask must allow the suffix to attend to ALL of 0..P-1:
        //   - Lower-left block (0..M-1) is already filled by the cached KV — attention mask = 0 (attend)
        //   - Causal within the suffix: standard lower-triangular
        // Shape: [1, suffixLen, P] where P = fullLen.
        // We represent this as a standard [1,1,suffixLen,P] mask per the model convention.
        // DecoderInputBuilder.buildInGraphCausalMask builds [1,1,seqLen,kvLen] where the rightmost
        // kvLen-seqLen columns are masked. We want kvLen=fullLen, seqLen=suffixLen, but positions
        // within [0..matchedLen-1] must be UN-masked (cached KV present). buildInGraphCausalMask
        // does NOT do this — it masks columns > seqLen-1. We need a custom mask.
        INDArray suffixCausalMask = buildSuffixPrefillMask(suffixLen, matchedLen, (int) maxKvLen, maskDtype);

        Map<String, INDArray> suffixInputMap = new HashMap<>();
        suffixInputMap.put(inputIdsName,
                Nd4j.createFromArray(suffixTokenIds).reshape(1, suffixLen).castTo(DataType.INT64));
        if (posOffsetName != null && decoder.hasVariable(posOffsetName)) {
            suffixInputMap.put(posOffsetName, Nd4j.scalar(DataType.INT64, (long) matchedLen));
        }
        if (cachePosName != null && decoder.hasVariable(cachePosName)) {
            suffixInputMap.put(cachePosName, Nd4j.scalar(DataType.INT64, (long) matchedLen));
        }
        if (decoder.hasVariable(ACTUAL_SEQUENCE_LENGTH_NAME)) {
            suffixInputMap.put(ACTUAL_SEQUENCE_LENGTH_NAME, Nd4j.scalar(DataType.INT64, (long) suffixLen));
        }
        if (causalMaskName != null && decoder.hasVariable(causalMaskName)) {
            suffixInputMap.put(causalMaskName, suffixCausalMask);
        }

        // Feed ALREADY-FILLED KV buffers (not empty sentinels) so the attention op can attend to
        // cached prefix positions.  The in-graph scatter will write the suffix K/V at positions
        // matchedLen..fullLen-1 in place.
        for (String keyName : kvInputNames.keyNames) {
            INDArray buf = hit.restoredKvBuffers.get(keyName);
            if (buf != null && decoder.hasVariable(keyName)) suffixInputMap.put(keyName, buf);
        }
        for (String valName : kvInputNames.valueNames) {
            INDArray buf = hit.restoredKvBuffers.get(valName);
            if (buf != null && decoder.hasVariable(valName)) suffixInputMap.put(valName, buf);
        }

        // Recurrent state: use snapshot for exact-boundary hits, zeros otherwise.
        Map<String, INDArray> recurrentStateBuffers = new LinkedHashMap<>();
        for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
            if (!decoder.hasVariable(pair.inputName)) continue;
            INDArray initState = null;
            if (hit.recurrentSnapshot != null) {
                INDArray snap = hit.recurrentSnapshot.get(pair.inputName);
                if (snap != null) {
                    initState = snap.dup();
                }
            }
            if (initState == null) {
                DataType dt = decoder.getVariable(pair.inputName).dataType();
                long[] stateShape = GenerationPipeline.deriveRecurrentStateShape(decoder, pair.inputName);
                if (stateShape != null) {
                    initState = Nd4j.zeros(dt, stateShape);
                }
            }
            if (initState != null) {
                suffixInputMap.put(pair.inputName, initState);
                recurrentStateBuffers.put(pair.inputName, initState);
            }
        }

        // Request outputs: logits + per-layer KV outputs (at suffix positions) + recurrent state
        List<String> suffixOutputNames = new ArrayList<>();
        suffixOutputNames.add(logitsName);
        for (int i = 0; i < numLayers; i++) {
            int layerIdx = extractLayerIndex(kvInputNames.keyNames.get(i));
            suffixOutputNames.add("k_rope_" + layerIdx);
            suffixOutputNames.add("v_heads_" + layerIdx);
        }
        for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
            suffixOutputNames.add(pair.outputName);
        }

        log.info("[PrefixCache] Suffix prefill: suffixLen={} matchedLen={} maxKvLen={} inputs={}",
                suffixLen, matchedLen, maxKvLen, suffixInputMap.keySet());

        Map<String, INDArray> suffixOutputs;
        try {
            suffixOutputs = decoder.output(suffixInputMap, suffixOutputNames.toArray(new String[0]));
        } catch (Exception e) {
            log.error("[PrefixCache] Suffix prefill failed — falling back to full prefill", e);
            // Clean up and fall back
            for (INDArray v : suffixInputMap.values()) { if (v != null && !v.wasClosed()) v.close(); }
            for (INDArray buf : hit.restoredKvBuffers.values()) {
                if (buf != null && !buf.wasClosed()) buf.close();
            }
            suffixCausalMask.close();
            return prefillWarmupAndFreeze(promptTokenIds, maxNewTokens, kvInputNames, startTime, null);
        }

        // All per-layer suffix K/V outputs are REQUIRED before we mutate any state. A missing
        // layer would leave zeros in that layer's KV region past matchedLen and silently corrupt
        // attention (the zeroed-KV garbage-decode failure class) — abort the suffix path and
        // recompute the full prefill instead.
        List<String> missingSuffixKv = new ArrayList<>();
        for (int i = 0; i < numLayers; i++) {
            int layerIdx = extractLayerIndex(kvInputNames.keyNames.get(i));
            if (suffixOutputs.get("k_rope_" + layerIdx) == null) missingSuffixKv.add("k_rope_" + layerIdx);
            if (suffixOutputs.get("v_heads_" + layerIdx) == null) missingSuffixKv.add("v_heads_" + layerIdx);
        }
        if (!missingSuffixKv.isEmpty()) {
            log.warn("[PrefixCache] Suffix prefill missing required K/V outputs {} — model output naming "
                    + "does not match k_rope_N/v_heads_N; falling back to full prefill", missingSuffixKv);
            for (INDArray v : suffixInputMap.values()) { if (v != null && !v.wasClosed()) v.close(); }
            for (INDArray out : suffixOutputs.values()) { if (out != null && !out.wasClosed()) out.close(); }
            for (INDArray buf : hit.restoredKvBuffers.values()) {
                if (buf != null && !buf.wasClosed()) buf.close();
            }
            suffixCausalMask.close();
            return prefillWarmupAndFreeze(promptTokenIds, maxNewTokens, kvInputNames, startTime, null);
        }

        // ── Sample first token from suffix prefill logits ────────────────────────────────────────────
        INDArray prefillLogits = suffixOutputs.get(logitsName);
        if (prefillLogits == null) {
            log.warn("[PrefixCache] Suffix prefill logits not found — falling back to full prefill");
            for (INDArray v : suffixInputMap.values()) { if (v != null && !v.wasClosed()) v.close(); }
            for (INDArray buf : hit.restoredKvBuffers.values()) {
                if (buf != null && !buf.wasClosed()) buf.close();
            }
            suffixCausalMask.close();
            return prefillWarmupAndFreeze(promptTokenIds, maxNewTokens, kvInputNames, startTime, null);
        }
        // Last position in suffix logits corresponds to the last prompt token (index suffixLen-1)
        int logitsSamplePos = suffixLen - 1;
        List<Integer> generatedSoFar = new ArrayList<>();
        suppressStopsUnderFloor(prefillLogits, logitsSamplePos, sampling, 0, stopTokenIds);
        int firstTokenId = sampleToken(prefillLogits, logitsSamplePos, sampling, generatedSoFar, rng);
        generatedSoFar.add(firstTokenId);
        prefillLogits.close();

        long firstTokenMs = System.currentTimeMillis() - startTime;
        log.info("[PrefixCache] Suffix prefill complete: firstToken={} ({}ms TTFT — prefix was {} tokens)",
                firstTokenId, firstTokenMs, matchedLen);

        // ── Populate static KV buffers with suffix K/V at positions matchedLen..fullLen-1 ─────────
        // The restored buffers already have prefix data at 0..matchedLen-1.
        // The suffix prefill ran the attention ops against those restored buffers AND wrote the
        // suffix K/V into the same buffers in place (since we passed the restored buffers as inputs).
        // However, the model outputs k_rope_N / v_heads_N are the suffix K/V values — we must write
        // them into the static buffers at the correct positions.
        Map<String, INDArray> staticKvBuffers = hit.restoredKvBuffers;
        org.nd4j.linalg.api.buffer.DataType kvDtype = null;

        for (int i = 0; i < numLayers; i++) {
            int layerIdx = extractLayerIndex(kvInputNames.keyNames.get(i));
            INDArray kSuffix = suffixOutputs.get("k_rope_" + layerIdx);
            INDArray vSuffix = suffixOutputs.get("v_heads_" + layerIdx);
            if (kSuffix == null || vSuffix == null) {
                // Unreachable: the pre-scan above already falls back to full prefill on any
                // missing layer output. Continuing here would leave zeroed KV — hard-fail.
                throw new IllegalStateException("[PrefixCache] Suffix K/V for layer " + layerIdx
                        + " missing after pre-scan — invariant violation");
            }
            if (kvDtype == null) kvDtype = kSuffix.dataType();
            // kSuffix shape: [1, suffixLen, kvH, headDim]
            // Write into the static buffer at positions matchedLen..matchedLen+suffixLen-1
            String keyName = kvInputNames.keyNames.get(i);
            String valName = kvInputNames.valueNames.get(i);
            INDArray keyBuf = staticKvBuffers.get(keyName);
            INDArray valBuf = staticKvBuffers.get(valName);
            if (keyBuf == null) {
                // Allocate if not present (shouldn't happen given pre-allocation above)
                long[] bufShape = {1, maxKvLen, kSuffix.shape()[2], kSuffix.shape()[3]};
                keyBuf = Nd4j.zeros(kvDtype, bufShape);
                staticKvBuffers.put(keyName, keyBuf);
            }
            if (valBuf == null) {
                long[] bufShape = {1, maxKvLen, vSuffix.shape()[2], vSuffix.shape()[3]};
                valBuf = Nd4j.zeros(kvDtype, bufShape);
                staticKvBuffers.put(valName, valBuf);
            }
            keyBuf.get(NDArrayIndex.all(), NDArrayIndex.interval(matchedLen, matchedLen + suffixLen),
                    NDArrayIndex.all(), NDArrayIndex.all()).assign(kSuffix);
            valBuf.get(NDArrayIndex.all(), NDArrayIndex.interval(matchedLen, matchedLen + suffixLen),
                    NDArrayIndex.all(), NDArrayIndex.all()).assign(vSuffix);
            kSuffix.close();
            vSuffix.close();
        }

        // Update recurrent state buffers from suffix outputs
        for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
            INDArray updated = suffixOutputs.get(pair.outputName);
            if (updated != null) {
                INDArray buf = recurrentStateBuffers.get(pair.inputName);
                if (buf != null) {
                    buf.assign(updated);
                } else {
                    recurrentStateBuffers.put(pair.inputName, updated.dup());
                }
                updated.close();
            }
        }

        // Close suffix input scalars (not the KV buffers which are now staticKvBuffers)
        closeSuffixInputScalars(suffixInputMap, kvInputNames, recurrentStateBuffers);
        suffixCausalMask.close();

        // Handle EOS on first token
        if (stopTokenIds.contains(firstTokenId)) {
            for (INDArray buf : staticKvBuffers.values()) { if (buf != null && !buf.wasClosed()) buf.close(); }
            for (INDArray buf : recurrentStateBuffers.values()) { if (buf != null && !buf.wasClosed()) buf.close(); }
            List<Integer> tokens = new ArrayList<>();
            tokens.add(firstTokenId);
            InGraphKvState terminal = new InGraphKvState();
            terminal.terminalResult = buildResult(tokens, promptTokenIds, stopTokenIds, startTime, firstTokenMs);
            terminal.generatedSoFar = tokens;
            terminal.eosReached = true;
            terminal.closed = true;
            terminal.actualPrefillLen = actualPrefillLen;
            terminal.promptTokenCount = prefillSeqLen;
            terminal.maxKvLen = maxKvLen;
            return terminal;
        }

        // ── Store the completed prefill in the prefix cache ────────────────────────────────────────
        // We store the FULL prompt (not just suffix) so future requests with longer shared prefixes
        // can get a bigger hit. The pool will handle deduplication via the radix trie.
        storePrefillInPrefixCache(promptTokenIds, actualPrefillLen, staticKvBuffers,
                kvInputNames, recurrentStateBuffers, recurrentStates);

        // ── Warmup decode step (STEP 3 of prefillWarmupAndFreeze) ────────────────────────────────
        // Use cachePosition = actualPrefillLen (the slot for the first decode token).
        int firstDecodePos = actualPrefillLen;
        INDArray decodeInputIds = Nd4j.createFromArray(new int[]{firstTokenId})
                .reshape(1, 1).castTo(DataType.INT64);
        INDArray decodeCausalMask = null;
        if (causalMaskName != null && decoder.hasVariable(causalMaskName)) {
            decodeCausalMask = DecoderInputBuilder.buildInGraphDecodeMask(firstDecodePos, maxKvLen, maskDtype);
        }
        INDArray decodePositionOffset = null;
        if (posOffsetName != null && decoder.hasVariable(posOffsetName)) {
            decodePositionOffset = Nd4j.scalar(DataType.INT64, firstDecodePos);
        }
        INDArray decodeCachePosition = null;
        if (cachePosName != null && decoder.hasVariable(cachePosName)) {
            decodeCachePosition = Nd4j.scalar(DataType.INT64, firstDecodePos);
        }
        INDArray decodeActualSequenceLength = null;
        if (decoder.hasVariable(ACTUAL_SEQUENCE_LENGTH_NAME)) {
            decodeActualSequenceLength = Nd4j.scalar(DataType.INT64, 1L);
        }

        Map<String, INDArray> decodeInputMap = new HashMap<>();
        decodeInputMap.put(inputIdsName, decodeInputIds);
        if (decodeCausalMask != null) decodeInputMap.put(causalMaskName, decodeCausalMask);
        if (decodePositionOffset != null) decodeInputMap.put(posOffsetName, decodePositionOffset);
        if (decodeCachePosition != null) decodeInputMap.put(cachePosName, decodeCachePosition);
        if (decodeActualSequenceLength != null) decodeInputMap.put(ACTUAL_SEQUENCE_LENGTH_NAME, decodeActualSequenceLength);
        for (Map.Entry<String, INDArray> e : staticKvBuffers.entrySet()) {
            if (decoder.hasVariable(e.getKey())) decodeInputMap.put(e.getKey(), e.getValue());
        }
        for (Map.Entry<String, INDArray> e : recurrentStateBuffers.entrySet()) {
            if (decoder.hasVariable(e.getKey())) decodeInputMap.put(e.getKey(), e.getValue());
        }

        List<String> decodeOutputNames = new ArrayList<>();
        decodeOutputNames.add(logitsName);
        for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
            decodeOutputNames.add(pair.outputName);
        }

        Map<String, INDArray> decodeOutputs;
        try {
            decodeOutputs = decoder.output(decodeInputMap, decodeOutputNames.toArray(new String[0]));
        } catch (Exception e) {
            log.error("[PrefixCache] Warmup decode failed after suffix prefill", e);
            throw new RuntimeException("Warmup decode failed after prefix-cache suffix prefill", e);
        }

        INDArray decodeLogits = decodeOutputs.get(logitsName);
        suppressStopsUnderFloor(decodeLogits, 0, sampling, generatedSoFar.size(), stopTokenIds);
        int secondTokenId = sampleToken(decodeLogits, 0, sampling, generatedSoFar, rng);
        generatedSoFar.add(secondTokenId);
        decodeLogits.close();

        // Update recurrent state with warmup decode output
        for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
            INDArray updated = decodeOutputs.get(pair.outputName);
            if (updated != null) {
                INDArray buf = recurrentStateBuffers.get(pair.inputName);
                if (buf != null) buf.assign(updated);
                updated.close();
            }
        }

        // ── Freeze plan (STEP 4) — same as in prefillWarmupAndFreeze ────────────────────────────
        InferenceSession session = decoder.getOrCreateSession();
        DynamicShapePlanExecutor executor = session != null ? session.getDynamicShapePlanExecutor() : null;
        Pointer planHandle = executor != null ? executor.getNativePlanHandle() : null;

        if (planHandle == null || planHandle.isNull()) {
            log.warn("[PrefixCache] Native plan handle not available after suffix prefill warmup");
            decodeInputIds.close();
            if (decodeCausalMask != null) decodeCausalMask.close();
            if (decodePositionOffset != null) decodePositionOffset.close();
            if (decodeCachePosition != null) decodeCachePosition.close();
            if (decodeActualSequenceLength != null) decodeActualSequenceLength.close();
            for (INDArray kv : staticKvBuffers.values()) kv.close();
            for (INDArray rs : recurrentStateBuffers.values()) rs.close();
            List<Integer> tokens = new ArrayList<>();
            tokens.add(firstTokenId);
            if (!stopTokenIds.contains(firstTokenId)) tokens.add(secondTokenId);
            InGraphKvState terminal = new InGraphKvState();
            terminal.terminalResult = buildResult(tokens, promptTokenIds, stopTokenIds, startTime, firstTokenMs);
            terminal.generatedSoFar = tokens;
            terminal.closed = true;
            terminal.actualPrefillLen = actualPrefillLen;
            terminal.promptTokenCount = prefillSeqLen;
            terminal.maxKvLen = maxKvLen;
            return terminal;
        }

        if (executor.getCurrentPlan() != null) {
            executor.setMaxKvCacheLength((int) maxKvLen);
            executor.configureMaxAllocationForKvCache(decodeOutputs);
            boolean forcedSlotBySlot = decoder.getGraphExecutionMode() == GraphExecutionMode.SLOT_BY_SLOT
                    || Nd4j.getEnvironment().tritonSkipKernels();
            if (!forcedSlotBySlot) {
                executor.setShapesFrozen(true);
                log.info("[PrefixCache] Shapes frozen after suffix prefill warmup (planPhase={} pointersStable={})",
                        executor.getPlanPhase(), executor.arePointersStable());
            }
        }

        // Build InGraphKvState (mirrors the tail of prefillWarmupAndFreeze)
        int inputIdsExtIdx = resolveExtInputIdx(executor, inputIdsName);
        int causalMaskExtIdx = causalMaskName != null ? resolveExtInputIdx(executor, causalMaskName) : -1;
        int posOffsetExtIdx = posOffsetName != null ? resolveExtInputIdx(executor, posOffsetName) : -1;
        int cachePosExtIdx = cachePosName != null ? resolveExtInputIdx(executor, cachePosName) : -1;
        int actualSeqLenExtIdx = decoder.hasVariable(ACTUAL_SEQUENCE_LENGTH_NAME)
                ? resolveExtInputIdx(executor, ACTUAL_SEQUENCE_LENGTH_NAME) : -1;
        int logitsOutputIdx = resolveOutputIdx(executor, logitsName);
        int numKvPairs = numLayers;
        int[] kvInputExtIndices = new int[2 * numKvPairs];
        int ki = 0;
        for (String keyName : kvInputNames.keyNames) kvInputExtIndices[ki++] = resolveExtInputIdx(executor, keyName);
        for (String valName : kvInputNames.valueNames) kvInputExtIndices[ki++] = resolveExtInputIdx(executor, valName);

        List<Integer> gdnExtList = new ArrayList<>(), gdnOutList = new ArrayList<>();
        List<Integer> convExtList = new ArrayList<>(), convOutList = new ArrayList<>();
        for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
            int extIdx = resolveExtInputIdx(executor, pair.inputName);
            int outIdx = resolveOutputIdx(executor, pair.outputName);
            if (pair.isGdn()) { gdnExtList.add(extIdx); gdnOutList.add(outIdx); }
            else { convExtList.add(extIdx); convOutList.add(outIdx); }
        }

        Pointer contextHandle = executor.getCachedOpContext();
        int numPlanExternalInputs = executor.getCurrentPlan() != null
                ? executor.getCurrentPlan().getExternalInputKeys().length : 0;
        int numPlanOutputs = decodeOutputNames.size();

        // Build a fresh prefillInputMap (the suffix-prefill path didn't use a retained one)
        Map<String, INDArray> prefillInputMap = new HashMap<>();
        prefillInputMap.put(inputIdsName,
                Nd4j.createFromArray(promptTokenIds).reshape(1, fullLen).castTo(DataType.INT64));
        if (posOffsetName != null && decoder.hasVariable(posOffsetName))
            prefillInputMap.put(posOffsetName, Nd4j.scalar(DataType.INT64, 0L));
        if (cachePosName != null && decoder.hasVariable(cachePosName))
            prefillInputMap.put(cachePosName, Nd4j.scalar(DataType.INT64, 0L));
        if (decoder.hasVariable(ACTUAL_SEQUENCE_LENGTH_NAME))
            prefillInputMap.put(ACTUAL_SEQUENCE_LENGTH_NAME, Nd4j.scalar(DataType.INT64, (long) actualPrefillLen));
        if (causalMaskName != null && decoder.hasVariable(causalMaskName))
            prefillInputMap.put(causalMaskName,
                    DecoderInputBuilder.buildInGraphCausalMask(fullLen, (int) maxKvLen, maskDtype));
        for (String keyName : kvInputNames.keyNames) {
            if (decoder.hasVariable(keyName))
                prefillInputMap.put(keyName, Nd4j.empty(decoder.getVariable(keyName).dataType()));
        }
        for (String valName : kvInputNames.valueNames) {
            if (decoder.hasVariable(valName))
                prefillInputMap.put(valName, Nd4j.empty(decoder.getVariable(valName).dataType()));
        }
        for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
            if (decoder.hasVariable(pair.inputName)) {
                DataType dt = decoder.getVariable(pair.inputName).dataType();
                long[] sh = deriveRecurrentStateShape(decoder, pair.inputName);
                if (sh != null) prefillInputMap.put(pair.inputName, Nd4j.zeros(dt, sh));
            }
        }

        InGraphKvState state = new InGraphKvState();
        state.staticKvBuffers = staticKvBuffers;
        state.recurrentStateBuffers = recurrentStateBuffers;
        state.quantizedKvBuffers = null;
        state.kvScaleBuffers = null;
        state.kvQuantFormat = 0;
        state.decodeInputIds = decodeInputIds;
        state.decodeCausalMask = decodeCausalMask;
        state.decodePositionOffset = decodePositionOffset;
        state.decodeCachePosition = decodeCachePosition;
        state.decodeActualSequenceLength = decodeActualSequenceLength;
        state.executor = executor;
        state.planHandle = planHandle;
        state.contextHandle = contextHandle;
        state.inputIdsExtIdx = inputIdsExtIdx;
        state.causalMaskExtIdx = causalMaskExtIdx;
        state.posOffsetExtIdx = posOffsetExtIdx;
        state.cachePosExtIdx = cachePosExtIdx;
        state.actualSeqLenExtIdx = actualSeqLenExtIdx;
        state.logitsOutputIdx = logitsOutputIdx;
        state.embeddingsExtIdx = -1;
        state.maskExtIdx = -1;
        state.posIdsExtIdx = -1;
        state.kvInputExtIndices = kvInputExtIndices;
        state.kvOutputIndices = new int[0];
        state.gdnStateExtIndices = gdnExtList.stream().mapToInt(Integer::intValue).toArray();
        state.gdnStateOutputIndices = gdnOutList.stream().mapToInt(Integer::intValue).toArray();
        state.convStateExtIndices = convExtList.stream().mapToInt(Integer::intValue).toArray();
        state.convStateOutputIndices = convOutList.stream().mapToInt(Integer::intValue).toArray();
        state.numPlanExternalInputs = numPlanExternalInputs;
        state.numPlanOutputs = numPlanOutputs;
        state.numKvPairs = numKvPairs;
        state.kvInputNames = kvInputNames;
        state.recurrentStates = recurrentStates;
        state.decodeOutputNames = decodeOutputNames;
        state.cachePosition = firstDecodePos + 1;
        state.lastGeneratedToken = secondTokenId;
        state.generatedSoFar = generatedSoFar;
        state.rng = rng;
        state.sampling = sampling;
        state.stopTokenIds = stopTokenIds;
        state.eosTokenId = eosTokenId;
        state.maxKvLen = maxKvLen;
        state.actualPrefillLen = actualPrefillLen;
        state.prefillSeqLen = prefillSeqLen;
        state.promptTokenCount = prefillSeqLen;
        state.maskDtype = maskDtype;
        state.rotatingSlotMap = null;   // prefix cache + rotating KV is rejected at config creation time
        state.inputIdsName = inputIdsName;
        state.logitsName = logitsName;
        state.causalMaskName = causalMaskName;
        state.posOffsetName = posOffsetName;
        state.cachePosName = cachePosName;
        state.actualSeqLenName = decoder.hasVariable(ACTUAL_SEQUENCE_LENGTH_NAME) ? ACTUAL_SEQUENCE_LENGTH_NAME : null;
        state.prefillInputMap = prefillInputMap;
        state.eosReached = false;
        state.cancelRequested = false;
        state.terminalResult = null;
        state.closed = false;
        return state;
    }

    /**
     * Build a causal mask for suffix-only prefill of tokens at positions [matchedLen..matchedLen+suffixLen-1].
     *
     * <p>Shape: {@code [1, 1, suffixLen, maxKvLen]}.
     * - Columns 0..matchedLen-1: all 0.0 (prefix KV is present — attend freely)
     * - Columns matchedLen..matchedLen+suffixLen-1: standard lower-triangular causal within suffix
     * - Columns matchedLen+suffixLen..maxKvLen-1: masked (-1e9)
     */
    private static INDArray buildSuffixPrefillMask(int suffixLen, int matchedLen, int maxKvLen, DataType dtype) {
        float maskVal = (dtype == DataType.HALF || dtype == DataType.FLOAT16) ? -65504.0f : -1e9f;
        float[] data = new float[suffixLen * maxKvLen];
        // Default: all masked
        java.util.Arrays.fill(data, maskVal);
        for (int row = 0; row < suffixLen; row++) {
            int absPos = matchedLen + row;   // absolute position of this suffix token
            // Attend to all prefix positions (0..matchedLen-1)
            for (int col = 0; col < matchedLen; col++) {
                data[row * maxKvLen + col] = 0.0f;
            }
            // Causal within suffix: attend to suffix tokens at positions <= absPos
            for (int col = matchedLen; col <= absPos && col < maxKvLen; col++) {
                data[row * maxKvLen + col] = 0.0f;
            }
        }
        INDArray mask = Nd4j.create(data, new long[]{1, 1, suffixLen, maxKvLen}, 'c');
        if (dtype != DataType.FLOAT) {
            INDArray cast = mask.castTo(dtype);
            mask.close();
            return cast;
        }
        return mask;
    }

    /**
     * Close suffix-prefill input scalars without closing the KV buffers or recurrent state buffers
     * that are now owned by the InGraphKvState.
     */
    private static void closeSuffixInputScalars(Map<String, INDArray> inputMap,
                                                 ModelIOConfig.KVCacheNames kvInputNames,
                                                 Map<String, INDArray> recurrentStateBuffers) {
        java.util.Set<INDArray> keepAlive = new java.util.HashSet<>();
        // KV buffers are now owned by static KV state — do NOT close
        // Recurrent state inputs were moved to recurrentStateBuffers — do NOT close
        if (recurrentStateBuffers != null) keepAlive.addAll(recurrentStateBuffers.values());

        java.util.Set<String> kvKeys = new java.util.HashSet<>();
        if (kvInputNames != null) {
            kvKeys.addAll(kvInputNames.keyNames);
            kvKeys.addAll(kvInputNames.valueNames);
        }

        for (Map.Entry<String, INDArray> e : inputMap.entrySet()) {
            if (kvKeys.contains(e.getKey())) continue;   // KV buffer — skip
            INDArray v = e.getValue();
            if (v != null && keepAlive.contains(v)) continue;
            if (v != null && !v.wasClosed()) {
                try { v.close(); } catch (Exception ignore) {}
            }
        }
    }

    /**
     * Store completed prefill KV data into the cross-request prefix block pool.
     *
     * <p>Called after a successful prefill (both full and suffix paths) when
     * {@code prefixBlockPool != null}. Extracts block-aligned slices from the static KV
     * buffers and registers them in the radix trie for future prefix lookups.</p>
     */
    private void storePrefillInPrefixCache(int[] promptTokenIds, int prefillLen,
                                            Map<String, INDArray> staticKvBuffers,
                                            ModelIOConfig.KVCacheNames kvInputNames,
                                            Map<String, INDArray> recurrentStateBuffers,
                                            List<ModelIOConfig.RecurrentStatePair> recurrentStates) {
        if (prefixBlockPool == null) return;
        List<String> allLayerNames = new ArrayList<>();
        allLayerNames.addAll(kvInputNames.keyNames);
        allLayerNames.addAll(kvInputNames.valueNames);
        // Build recurrent snapshot map (input-name → state INDArray)
        Map<String, INDArray> recurrentSnapshot = null;
        if (recurrentStates != null && !recurrentStates.isEmpty() && recurrentStateBuffers != null) {
            recurrentSnapshot = new LinkedHashMap<>(recurrentStateBuffers);
        }
        prefixBlockPool.storeCompletedPrefill(
                promptTokenIds, prefillLen, staticKvBuffers, allLayerNames, recurrentSnapshot);
    }

    /**
     * One-shot in-graph-KV generation: prefill + warmup + freeze, then run the native decode loop
     * once and free everything. Behaviorally identical to the pre-refactor implementation.
     * For resumable ("continue generation") decoding, use {@link #startSession(String, int)}.
     */
    private GenerationResult generateSimpleWithInGraphKvCache(int[] promptTokenIds, int maxNewTokens,
                                                              ModelIOConfig.KVCacheNames kvInputNames) {
        long startTime = System.currentTimeMillis();
        if (activeSession.get() != null) {
            throw new IllegalStateException(
                    "A GenerationSession is active on this pipeline; close it before calling generate(). "
                    + "While a session is open, decode through the session's generate()/continueGeneration().");
        }

        // ── Prefix cache lookup ──────────────────────────────────────────────────────────────────
        if (prefixBlockPool != null) {
            // A prefix-cache hit builds a fresh suffix-prefill (or GDN-fallback full prefill) with its
            // own executor freeze. Any retained one-shot fixed-buffer state from a PRIOR generate still
            // holds that prior prompt's frozen CUDA-graph plan/buffers; leaving it live lets the prior
            // captured decode alias this generate and replay the prior prompt's tokens (observed:
            // promptB-with-cache reproduced promptA's continuation). Drop it here so the hit path
            // starts from a clean executor — mirrors the session path (see startSession).
            if (cachedFixedBufferState != null) {
                cachedFixedBufferState.close();
                cachedFixedBufferState = null;
            }
            // Discover recurrent state pairs for the GDN guard in attemptPrefixCacheHit
            List<ModelIOConfig.RecurrentStatePair> recurrentStates =
                    ModelIOConfig.findRecurrentStatePairs(decoder, ioConfig);
            // Rough maxKvLen estimate for buffer allocation (will be refined in the suffix path)
            long roughMaxKvLen = promptTokenIds.length + maxNewTokens;
            int kvCap = config.getMaxKvCacheLength();
            if (kvCap > 0 && roughMaxKvLen > kvCap) roughMaxKvLen = kvCap;
            PrefixHitContext hit = attemptPrefixCacheHit(promptTokenIds, kvInputNames, roughMaxKvLen, recurrentStates);
            if (hit != null) {
                InGraphKvState state = prefillSuffixOnlyAndFreeze(promptTokenIds, maxNewTokens,
                        kvInputNames, startTime, hit);
                if (state.terminalResult != null) return state.terminalResult;
                try {
                    return runInGraphNativeDecode(state, maxNewTokens, false, startTime);
                } finally {
                    state.close();
                }
            }
        }

        // FORWARD-FIX: on the fixed-buffer path, reuse the cached frozen state across generates so the
        // captured decode plan replays (no per-generate re-warm). The reuse path keeps the plan, refills
        // the retained (stable-address) buffers in place, and skips the re-freeze. Fresh path otherwise.
        boolean fixedBuffers = config.getMaxPrefillLength() > 0;
        InGraphKvState reuse = null;
        if (fixedBuffers && cachedFixedBufferState != null) {
            InGraphKvState candidate = cachedFixedBufferState;
            long requestedMaxKvLen = (long) config.getMaxPrefillLength() + maxNewTokens;
            int configuredKvCap = config.getMaxKvCacheLength();
            if (configuredKvCap > 0 && requestedMaxKvLen > configuredKvCap) {
                requestedMaxKvLen = configuredKvCap;
            }
            if (!candidate.closed && candidate.maxKvLen == requestedMaxKvLen) {
                reuse = candidate;
            } else {
                // maxNewTokens participates in the frozen causal-mask/KV shapes. Reusing a
                // different-capacity state makes the in-place prefill assignment shape-invalid.
                cachedFixedBufferState = null;
                candidate.close();
                log.info("[Lifecycle] Discarded incompatible cached fixed-buffer state "
                                + "(cachedMaxKvLen={}, requestedMaxKvLen={})",
                        candidate.maxKvLen, requestedMaxKvLen);
            }
        }
        InGraphKvState state = prefillWarmupAndFreeze(promptTokenIds, maxNewTokens, kvInputNames, startTime, reuse);
        if (state.terminalResult != null) {
            // Terminal (early-EOS / no plan handle): the reused state is spent — close it and drop the
            // cache so the next generate rebuilds from scratch. (state is a fresh terminal, != reuse.)
            if (reuse != null) reuse.close();
            cachedFixedBufferState = null;
            return state.terminalResult;
        }
        if (fixedBuffers) {
            // Store the completed prefill in the prefix cache before returning
            if (prefixBlockPool != null && state.staticKvBuffers != null) {
                List<ModelIOConfig.RecurrentStatePair> recurrentStates =
                        ModelIOConfig.findRecurrentStatePairs(decoder, ioConfig);
                storePrefillInPrefixCache(promptTokenIds, state.actualPrefillLen,
                        state.staticKvBuffers, kvInputNames, state.recurrentStateBuffers, recurrentStates);
            }
            // Retain for the next generate; do NOT close here — the buffers/plan are reused in place.
            cachedFixedBufferState = state;
            return runInGraphNativeDecode(state, maxNewTokens, false, startTime);
        }
        // Store the completed prefill in the prefix cache
        if (prefixBlockPool != null && state.staticKvBuffers != null) {
            List<ModelIOConfig.RecurrentStatePair> recurrentStates =
                    ModelIOConfig.findRecurrentStatePairs(decoder, ioConfig);
            storePrefillInPrefixCache(promptTokenIds, state.actualPrefillLen,
                    state.staticKvBuffers, kvInputNames, state.recurrentStateBuffers, recurrentStates);
        }
        try {
            return runInGraphNativeDecode(state, maxNewTokens, false, startTime);
        } finally {
            state.close();
        }
    }

    /**
     * Execute the native {@code autoregressive_decode} loop against a retained {@link InGraphKvState},
     * WITHOUT freeing the retained buffers (the caller / session owns their lifecycle).
     *
     * <p>On the first (non-continuation) call this reproduces the original decode handoff exactly: the
     * two tokens already sampled in Java (prefill + warmup) prefix the native tokens and
     * {@code remainingTokens = maxNewTokens - 2}. On a continuation call there is no Java prefill/warmup,
     * so the result carries only the native tokens, {@code remainingTokens = maxNewTokens}, and decode
     * resumes by feeding {@code state.lastGeneratedToken} at {@code state.cachePosition}.</p>
     *
     * <p>Advances {@code state.cachePosition}, {@code state.lastGeneratedToken},
     * {@code state.generatedSoFar} and {@code state.eosReached} in place.</p>
     */
    private GenerationResult runInGraphNativeDecode(InGraphKvState state, int maxNewTokens,
                                                    boolean isContinuation, long startTime) {
        int remainingTokens = isContinuation ? maxNewTokens : (maxNewTokens - 2);

        // ── Prepare the current decode-step inputs: feed the last generated token at cachePosition ──
        state.decodeInputIds.assign(0);
        state.decodeInputIds.putScalar(new long[]{0, 0}, state.lastGeneratedToken);
        if (state.decodeCausalMask != null) {
            if (state.rotatingSlotMap != null) {
                // ── Rotating KV mask: build per-step based on which physical slots are live ──────
                // Always rebuild from scratch (the ring may have wrapped since the last step).
                // CUDA-graph safe: same buffer address, only values change via D2D staging.
                float maskVal = (state.maskDtype == DataType.HALF || state.maskDtype == DataType.FLOAT16)
                        ? -65504.0f : -1e9f;
                float[] maskData = state.rotatingSlotMap.buildRotatingDecodeMask(
                        state.cachePosition, maskVal);
                INDArray fresh = Nd4j.create(maskData, new long[]{1, 1, 1, state.maxKvLen}, 'c');
                if (state.maskDtype != DataType.FLOAT) {
                    INDArray cast = fresh.castTo(state.maskDtype);
                    fresh.close();
                    fresh = cast;
                }
                state.decodeCausalMask.assign(fresh);
                fresh.close();
            } else if (isContinuation) {
                // Standard (non-rotating) resume: rebuild the mask fresh to unmask exactly
                // 0..cachePosition-1 (all committed K/V), independent of whatever the prior native op
                // left it in. Reuse the SAME buffer (assign, not realloc) to keep the frozen-plan
                // ext-input pointer stable.
                int decodeWidth = (int) state.decodeInputIds.size(1);
                INDArray fresh = decodeWidth > 1
                        ? DecoderInputBuilder.buildInGraphWindowMask(
                                DecoderInputBuilder.chainParents(1, decodeWidth),
                                state.cachePosition - 1, 1, decodeWidth,
                                state.maxKvLen, state.maskDtype)
                        : DecoderInputBuilder.buildInGraphDecodeMask(
                                state.cachePosition - 1, state.maxKvLen, state.maskDtype);
                state.decodeCausalMask.assign(fresh);
                fresh.close();
            } else {
                // Standard first call: the warmup already built the decode mask; unmask the firstToken
                // position (== cachePosition-1) so the second token attends to it.
                state.decodeCausalMask.putScalar(new long[]{0, 0, 0, state.cachePosition - 1}, 0.0f);
            }
        }
        if (state.decodePositionOffset != null) {
            // positionOffset drives RoPE on the NEW token being fed — always the global position.
            state.decodePositionOffset.putScalar(new long[]{}, (long) state.cachePosition);
        }
        if (state.decodeCachePosition != null) {
            // cachePosition fed to the C++ op is the PHYSICAL slot where the current token's K/V will
            // be written. In non-rotating mode this equals the global position. In rotating mode it may
            // be a recycled slot (ring wrap). The fixed KV buffer shape is unchanged — only the write
            // target changes.
            long physicalWriteSlot = (state.rotatingSlotMap != null)
                    ? state.rotatingSlotMap.physicalSlot(state.cachePosition)
                    : state.cachePosition;
            state.decodeCachePosition.putScalar(new long[]{}, physicalWriteSlot);
        }
        if (state.decodeActualSequenceLength != null) {
            state.decodeActualSequenceLength.putScalar(new long[]{}, 1L);
        }

        Nd4j.getExecutioner().commit();
        state.decodeInputIds.syncToDevice();
        if (state.decodeCausalMask != null) state.decodeCausalMask.syncToDevice();
        if (state.decodePositionOffset != null) state.decodePositionOffset.syncToDevice();
        if (state.decodeCachePosition != null) state.decodeCachePosition.syncToDevice();
        if (state.decodeActualSequenceLength != null) state.decodeActualSequenceLength.syncToDevice();
        // V2 QUANTIZED: sync INT8 live buffers (and their scales); float staticKvBuffers are null.
        if (state.isQuantizedV2 && state.quantizedKvBuffers != null) {
            for (INDArray kvBuf : state.quantizedKvBuffers.values()) {
                if (kvBuf != null && !kvBuf.isEmpty()) kvBuf.syncToDevice();
            }
            if (state.kvScaleBuffers != null) {
                for (INDArray scBuf : state.kvScaleBuffers.values()) {
                    if (scBuf != null && !scBuf.isEmpty()) scBuf.syncToDevice();
                }
            }
        } else if (state.staticKvBuffers != null) {
            for (INDArray kvBuf : state.staticKvBuffers.values()) kvBuf.syncToDevice();
        }
        for (INDArray rsBuf : state.recurrentStateBuffers.values()) rsBuf.syncToDevice();

        if (state.constraintMasker != null) {
            return runInGraphConstrainedDecode(
                    state, remainingTokens, isContinuation, startTime);
        }

        int firstTokenId = !state.generatedSoFar.isEmpty() ? state.generatedSoFar.get(0) : state.lastGeneratedToken;
        int secondTokenId = state.generatedSoFar.size() > 1 ? state.generatedSoFar.get(1) : state.lastGeneratedToken;

        // Assemble the KV buffer array in (all keys, then all values) order — same object refs as the
        // frozen plan's ext inputs, so the device pointers stay stable across calls.
        // V2 QUANTIZED: use INT8 quantizedKvBuffers (under original variable names).
        // V1 / STATIC: use float staticKvBuffers.
        Map<String, INDArray> kvLiveMap = state.isQuantizedV2 ? state.quantizedKvBuffers : state.staticKvBuffers;
        INDArray[] staticKvArray = new INDArray[2 * state.numKvPairs];
        int idx = 0;
        for (String keyName : state.kvInputNames.keyNames) {
            staticKvArray[idx++] = kvLiveMap != null ? kvLiveMap.get(keyName) : null;
        }
        for (String valName : state.kvInputNames.valueNames) {
            staticKvArray[idx++] = kvLiveMap != null ? kvLiveMap.get(valName) : null;
        }

        // ADR 0107 V2: assemble scale buffer array (key scales then value scales).
        // Scale arrays are FLOAT32 [batch, maxKvLen, kvHeads] — one float per (token, kv_head).
        // Null when not in V2 quantized mode.
        INDArray[] kvScaleArray = null;
        if (state.isQuantizedV2 && state.kvScaleBuffers != null && !state.kvScaleBuffers.isEmpty()) {
            kvScaleArray = new INDArray[2 * state.numKvPairs];
            int si = 0;
            for (String keyName : state.kvInputNames.keyNames) {
                kvScaleArray[si++] = state.kvScaleBuffers.get(keyName + "_scale");
            }
            for (String valName : state.kvInputNames.valueNames) {
                kvScaleArray[si++] = state.kvScaleBuffers.get(valName + "_scale");
            }
        }
        INDArray[] mtpKvArray = state.mtpKvBuffers != null
                ? new INDArray[]{
                        state.mtpKvBuffers.get(MTP_KEY_CACHE_NAME),
                        state.mtpKvBuffers.get(MTP_VALUE_CACHE_NAME)}
                : null;

        List<Integer> nativeTokens = new ArrayList<>();   // tokens produced by the native op this call
        int nativeCount = 0;
        float tokPerSec = 0f, lateSteadyTokPerSec = 0f;
        int totalSpeculative = 0, totalAccepted = 0, speculativeSteps = 0;

        if (remainingTokens > 0) {
            if (state.rotatingSlotMap != null) {
                // ── Rotating KV: single-step Java loop ──────────────────────────────────────────
                // The C++ inner loop increments currentPosition monotonically. With rotating KV the
                // physical write slot wraps modulo ringSize, so we cannot pass remainingTokens > 1 to
                // the native op — it would use global positions beyond maxKvLen as physical write
                // offsets and corrupt the attention mask. Instead we call the op one step at a time
                // from Java, recalculating the physical slot and rotating mask before each step.
                // CUDA-graph safety: each single-step exec is a replay of the same frozen graph (the
                // plan shape is fixed: 1 input token, 1 output token). No shape change occurs.
                DecodePolicy decodePolicy = resolveDecodePolicy(state.sampling, config);
                if (decodePolicy.kind == DecodePolicyKind.SPECULATIVE
                        && state.mtpPlanHandle != null && !state.mtpPlanHandle.isNull()) {
                    throw new IllegalStateException(
                            "Bundled MTP requires the standard monotonic KV cache; rotating KV uses "
                                    + "a per-token physical-slot loop and cannot preserve the MTP cache prefix.");
                }
                int generatedTokenOffset = state.generatedSoFar != null ? state.generatedSoFar.size() : 0;

                for (int step = 0; step < remainingTokens; step++) {
                    INDArray dummyEmbeddings = Nd4j.zeros(DataType.FLOAT, 1, 1, 1);
                    INDArray dummyEmbTable = Nd4j.zeros(DataType.FLOAT, 1, 1);

                    // The physical write slot for this step — passed as startingCachePosition.
                    // The C++ op runs exactly 1 step (maxNewTokens=1) so it uses this position once.
                    int physSlot = state.rotatingSlotMap.physicalSlot(state.cachePosition);

                    AutoregressiveDecode op = new AutoregressiveDecode(
                            dummyEmbeddings, dummyEmbTable, state.decodeInputIds,
                            state.decodeCausalMask, null, staticKvArray,
                            state.planHandle, state.contextHandle,
                            state.numPlanExternalInputs, state.numPlanOutputs,
                            state.embeddingsExtIdx, state.maskExtIdx, state.causalMaskExtIdx,
                            state.posIdsExtIdx, state.inputIdsExtIdx, state.logitsOutputIdx,
                            -1,                       // attnMaskReformatExtIdx
                            state.posOffsetExtIdx,    // position_offset ext index
                            state.cachePosExtIdx,     // cache_position ext index
                            state.kvInputExtIndices, state.kvOutputIndices,
                            state.gdnStateExtIndices, state.gdnStateOutputIndices,
                            state.convStateExtIndices, state.convStateOutputIndices,
                            1 /* single step */, state.eosTokenId, state.numKvPairs,
                            physSlot,    // starting physical slot for this one step
                            state.sampling.isGreedy() ? 0.0 : state.sampling.getTemperature(),
                            state.sampling.isGreedy() ? 0 : state.sampling.getTopK(),
                            state.sampling.isGreedy() ? 0.0 : state.sampling.getTopP(),
                            state.sampling.getRepetitionPenalty(),
                            state.stopTokenIds);
                    // ADR 0107 V2: append scale buffers and set bit 7 in optionalMask.
                    if (kvScaleArray != null) op.withQuantisedKvScales(kvScaleArray);
                    applyNativePolicy(op, decodePolicy, state.sampling, generatedTokenOffset + step,
                            (int) state.decodeInputIds.size(1));
                    // ADR 0106 Phase 2: wire n-gram speculation when the policy uses the window substrate.
                    // specK > 0 only when decodePolicy.kind == SPECULATIVE and windowMax = specK+1.
                    if (decodePolicy.kind == DecodePolicyKind.SPECULATIVE && config != null
                            && config.getMaxSpeculativeTokens() > 0) {
                        op.withSpeculativeDecoding(config.getMaxSpeculativeTokens(), 1 /* NGRAM */);
                        op.withActualSequenceLengthExtIdx(state.actualSeqLenExtIdx);
                    }

                    INDArray[] results = Nd4j.getExecutioner().exec(op);
                    INDArray nativeTokenIds = results[0];
                    INDArray nativeTokenCount = results[1];
                    INDArray nativeTimingInfo = results[2];
                    int stepCount = nativeTokenCount.getInt(0);
                    tokPerSec = nativeTimingInfo.getFloat(2);
                    lateSteadyTokPerSec = nativeTimingInfo.length() > 5 ? nativeTimingInfo.getFloat(5) : tokPerSec;
                    if (nativeTimingInfo.length() > 9) {
                        totalSpeculative += nativeTimingInfo.getInt(7);
                        totalAccepted += nativeTimingInfo.getInt(8);
                        speculativeSteps += nativeTimingInfo.getInt(9);
                    }
                    int tok = stepCount > 0 ? (int) nativeTokenIds.getLong(0) : 0;

                    closeOutputs(results);
                    dummyEmbeddings.close();
                    dummyEmbTable.close();

                    if (stepCount <= 0) break;
                    nativeTokens.add(tok);
                    nativeCount++;
                    state.cachePosition++;

                    if (state.stopTokenIds.contains(tok)) break;

                    // Update inputs for next step: new token, rotating mask, physical slot.
                    state.decodeInputIds.putScalar(new long[]{0, 0}, tok);
                    if (state.decodeCausalMask != null) {
                        float maskVal = (state.maskDtype == DataType.HALF || state.maskDtype == DataType.FLOAT16)
                                ? -65504.0f : -1e9f;
                        float[] maskData = state.rotatingSlotMap.buildRotatingDecodeMask(
                                state.cachePosition, maskVal);
                        INDArray fresh = Nd4j.create(maskData, new long[]{1, 1, 1, state.maxKvLen}, 'c');
                        if (state.maskDtype != DataType.FLOAT) {
                            INDArray cast = fresh.castTo(state.maskDtype);
                            fresh.close();
                            fresh = cast;
                        }
                        state.decodeCausalMask.assign(fresh);
                        fresh.close();
                    }
                    if (state.decodePositionOffset != null)
                        state.decodePositionOffset.putScalar(new long[]{}, (long) state.cachePosition);
                    if (state.decodeCachePosition != null) {
                        long nextPhysSlot = state.rotatingSlotMap.physicalSlot(state.cachePosition);
                        state.decodeCachePosition.putScalar(new long[]{}, nextPhysSlot);
                    }
                    Nd4j.getExecutioner().commit();
                    state.decodeInputIds.syncToDevice();
                    if (state.decodeCausalMask != null) state.decodeCausalMask.syncToDevice();
                    if (state.decodePositionOffset != null) state.decodePositionOffset.syncToDevice();
                    if (state.decodeCachePosition != null) state.decodeCachePosition.syncToDevice();
                }
                // cachePosition already advanced inside the loop — don't advance again below.
                // Adjust nativeCount already counted; set nativeCountForAdvance = 0 so the advance
                // below (state.cachePosition += nativeCount) is a no-op.
                nativeCount = 0;   // suppress the post-loop advance; done per-step above
            } else {
                // ── Standard (non-rotating) multi-step native op ─────────────────────────────────
                INDArray dummyEmbeddings = Nd4j.zeros(DataType.FLOAT, 1, 1, 1);
                INDArray dummyEmbTable = Nd4j.zeros(DataType.FLOAT, 1, 1);

                AutoregressiveDecode op = new AutoregressiveDecode(
                        dummyEmbeddings, dummyEmbTable, state.decodeInputIds,
                        state.decodeCausalMask, null, staticKvArray,
                        state.planHandle, state.contextHandle,
                        state.numPlanExternalInputs, state.numPlanOutputs,
                        state.embeddingsExtIdx, state.maskExtIdx, state.causalMaskExtIdx,
                        state.posIdsExtIdx, state.inputIdsExtIdx, state.logitsOutputIdx,
                        -1,                       // attnMaskReformatExtIdx
                        state.posOffsetExtIdx,    // position_offset ext index
                        state.cachePosExtIdx,     // cache_position ext index
                        state.kvInputExtIndices, state.kvOutputIndices,
                        state.gdnStateExtIndices, state.gdnStateOutputIndices,
                        state.convStateExtIndices, state.convStateOutputIndices,
                        remainingTokens, state.eosTokenId, state.numKvPairs,
                        state.cachePosition,      // starting cachePosition for the C++ decode loop
                        state.sampling.isGreedy() ? 0.0 : state.sampling.getTemperature(),
                        state.sampling.isGreedy() ? 0 : state.sampling.getTopK(),
                        state.sampling.isGreedy() ? 0.0 : state.sampling.getTopP(),
                        state.sampling.getRepetitionPenalty(),
                        state.stopTokenIds);
                // ADR 0107 V2: append scale buffers and set bit 7 in optionalMask.
                if (kvScaleArray != null) op.withQuantisedKvScales(kvScaleArray);
                int generatedTokenOffset = state.generatedSoFar != null ? state.generatedSoFar.size() : 0;
                DecodePolicy decodePolicy = resolveDecodePolicy(state.sampling, config);
                applyNativePolicy(op, decodePolicy, state.sampling, generatedTokenOffset,
                        (int) state.decodeInputIds.size(1));
                if (decodePolicy.kind == DecodePolicyKind.SPECULATIVE && config != null
                        && config.getMaxSpeculativeTokens() > 0) {
                    boolean hasMtpPlan = state.mtpPlanHandle != null && !state.mtpPlanHandle.isNull();
                    op.withSpeculativeDecoding(
                            config.getMaxSpeculativeTokens(),
                            hasMtpPlan ? AutoregressiveDecode.SPECULATOR_TYPE_MTP
                                    : AutoregressiveDecode.SPECULATOR_TYPE_NGRAM);
                    op.withActualSequenceLengthExtIdx(state.actualSeqLenExtIdx);
                    if (hasMtpPlan) {
                        op.withMtpPlan(
                                state.mtpInputIds,
                                state.mtpTargetHiddenStates,
                                state.mtpCausalMask,
                                state.mtpPositionOffset,
                                state.mtpCachePosition,
                                mtpKvArray,
                                state.mtpPlanHandle,
                                state.mtpContextHandle,
                                state.mtpNumPlanExternalInputs,
                                state.mtpNumPlanOutputs,
                                state.mtpInputIdsExtIdx,
                                state.mtpTargetHiddenExtIdx,
                                state.mtpCausalMaskExtIdx,
                                state.mtpPosOffsetExtIdx,
                                state.mtpCachePosExtIdx,
                                state.mtpKvInputExtIndices,
                                state.mtpLogitsOutputIdx,
                                state.mtpHiddenOutputIdx,
                                state.targetHiddenOutputIdx);
                    }
                }

                INDArray[] results = Nd4j.getExecutioner().exec(op);
                INDArray nativeTokenIds = results[0];
                INDArray nativeTokenCount = results[1];
                INDArray nativeTimingInfo = results[2];
                nativeCount = nativeTokenCount.getInt(0);
                for (int i = 0; i < nativeCount; i++) {
                    int tok = (int) nativeTokenIds.getLong(i);
                    nativeTokens.add(tok);
                    if (state.stopTokenIds.contains(tok)) break;
                }
                tokPerSec = nativeTimingInfo.getFloat(2);
                lateSteadyTokPerSec = nativeTimingInfo.length() > 5 ? nativeTimingInfo.getFloat(5) : tokPerSec;
                if (nativeTimingInfo.length() > 9) {
                    totalSpeculative = nativeTimingInfo.getInt(7);
                    totalAccepted = nativeTimingInfo.getInt(8);
                    speculativeSteps = nativeTimingInfo.getInt(9);
                }

                closeOutputs(results);
                dummyEmbeddings.close();
                dummyEmbTable.close();
            }
            // Token count/timing reads above are the native op's existing host-visible boundary.
            state.releaseRecurrentCopyDonors();
        }

        // ── Assemble this call's returned tokens + advance the session state ──
        List<Integer> callTokens = new ArrayList<>();
        if (!isContinuation) {
            // First call: prefix with the two Java-sampled tokens, honoring early stops (matches original).
            callTokens.add(firstTokenId);
            if (!state.stopTokenIds.contains(firstTokenId)) {
                callTokens.add(secondTokenId);
                if (!state.stopTokenIds.contains(secondTokenId)) {
                    callTokens.addAll(nativeTokens);
                }
            }
            state.generatedSoFar = new ArrayList<>(callTokens);
        } else {
            callTokens.addAll(nativeTokens);
            state.generatedSoFar.addAll(nativeTokens);
        }
        // Each native step advances the write position by one; the last sampled token stays unwritten.
        state.cachePosition += nativeCount;
        if (!nativeTokens.isEmpty()) {
            state.lastGeneratedToken = nativeTokens.get(nativeTokens.size() - 1);
        }

        int[] tokenIds = callTokens.stream().mapToInt(Integer::intValue).toArray();
        boolean hitEos = tokenIds.length > 0 && state.stopTokenIds.contains(tokenIds[tokenIds.length - 1]);
        state.eosReached = state.eosReached || hitEos;

        String text = decodeGeneratedText(tokenIds, state.stopTokenIds);
        long timeMs = System.currentTimeMillis() - startTime;
        log.info("[GGUF-KV] decode complete (continuation={}): nativeCount={} callTokens={} cachePosition={} eos={}",
                isContinuation, nativeCount, callTokens.size(), state.cachePosition, hitEos);

        return GenerationResult.builder()
                .text(text).tokenIds(tokenIds)
                .generatedTokenCount(tokenIds.length)
                .promptTokenCount(state.promptTokenCount)
                .totalTokenCount(state.promptTokenCount + state.generatedSoFar.size())
                .finishReason(hitEos ? GenerationResult.FinishReason.EOS : GenerationResult.FinishReason.MAX_TOKENS)
                .generationTimeMs(timeMs)
                .tokensPerSecond(timeMs > 0 ? (tokenIds.length * 1000.0 / timeMs) : 0)
                .steadyStateTokensPerSecond(tokPerSec)
                .lateSteadyStateTokensPerSecond(lateSteadyTokPerSec)
                .totalSpeculativeTokens(totalSpeculative)
                .totalAcceptedTokens(totalAccepted)
                .speculativeSteps(speculativeSteps)
                .averageAcceptanceRate(totalSpeculative > 0 ? (double) totalAccepted / totalSpeculative : 0.0)
                .effectiveTokensPerSecond(timeMs > 0 ? (tokenIds.length * 1000.0 / timeMs) : 0.0)
                .sessionId(state.sessionId)   // 0 for one-shot state; the real id for any session call
                .build();
    }

    /**
     * Decode an in-graph KV state with token-level structural masking.
     *
     * <p>The native autoregressive loop cannot invoke the Java constraint automaton between
     * tokens, so constrained requests execute the already-frozen decode graph one token at a
     * time through {@link SameDiff#outputDirect(Map, String...)}. This is still the configured
     * DSP plan and the same retained KV/recurrent buffers; only token selection stays in Java
     * so invalid tool names or envelopes are never sampled.</p>
     */
    private GenerationResult runInGraphConstrainedDecode(
            InGraphKvState state,
            int remainingTokens,
            boolean isContinuation,
            long startTime) {
        if (state.decodeInputIds.size(1) != 1) {
            throw new IllegalStateException(
                    "Constrained in-graph decoding requires the scalar decode substrate; got width="
                            + state.decodeInputIds.size(1));
        }

        List<Integer> callGenerated = new ArrayList<>();
        Map<String, INDArray> liveKv =
                state.isQuantizedV2 ? state.quantizedKvBuffers : state.staticKvBuffers;
        int steps = Math.max(0, remainingTokens);

        log.info("[Constraint] Entering constrained in-graph KV decode loop: remainingTokens={}",
                steps);
        for (int step = 0; step < steps; step++) {
            if (state.stopTokenIds.contains(state.lastGeneratedToken)
                    || state.cachePosition >= state.maxKvLen) {
                break;
            }

            state.decodeInputIds.assign(0);
            state.decodeInputIds.putScalar(
                    new long[]{0, 0}, state.lastGeneratedToken);

            if (state.decodeCausalMask != null) {
                INDArray freshMask;
                if (state.rotatingSlotMap != null) {
                    float maskValue =
                            state.maskDtype == DataType.HALF
                                    || state.maskDtype == DataType.FLOAT16
                                    ? -65504.0f : -1e9f;
                    float[] maskData = state.rotatingSlotMap.buildRotatingDecodeMask(
                            state.cachePosition, maskValue);
                    freshMask = Nd4j.create(
                            maskData, new long[]{1, 1, 1, state.maxKvLen}, 'c');
                    if (state.maskDtype != DataType.FLOAT) {
                        INDArray cast = freshMask.castTo(state.maskDtype);
                        freshMask.close();
                        freshMask = cast;
                    }
                } else {
                    freshMask = DecoderInputBuilder.buildInGraphDecodeMask(
                            state.cachePosition - 1, state.maxKvLen, state.maskDtype);
                }
                state.decodeCausalMask.assign(freshMask);
                freshMask.close();
            }
            if (state.decodePositionOffset != null) {
                state.decodePositionOffset.putScalar(
                        new long[]{}, (long) state.cachePosition);
            }
            if (state.decodeCachePosition != null) {
                long physicalSlot = state.rotatingSlotMap != null
                        ? state.rotatingSlotMap.physicalSlot(state.cachePosition)
                        : state.cachePosition;
                state.decodeCachePosition.putScalar(new long[]{}, physicalSlot);
            }
            if (state.decodeActualSequenceLength != null) {
                state.decodeActualSequenceLength.putScalar(new long[]{}, 1L);
            }

            Nd4j.getExecutioner().commit();
            state.decodeInputIds.syncToDevice();
            if (state.decodeCausalMask != null) {
                state.decodeCausalMask.syncToDevice();
            }
            if (state.decodePositionOffset != null) {
                state.decodePositionOffset.syncToDevice();
            }
            if (state.decodeCachePosition != null) {
                state.decodeCachePosition.syncToDevice();
            }
            if (state.decodeActualSequenceLength != null) {
                state.decodeActualSequenceLength.syncToDevice();
            }
            if (liveKv != null) {
                for (INDArray kvBuffer : liveKv.values()) {
                    if (kvBuffer != null && !kvBuffer.wasClosed()) {
                        kvBuffer.syncToDevice();
                    }
                }
            }
            for (INDArray recurrentBuffer : state.recurrentStateBuffers.values()) {
                recurrentBuffer.syncToDevice();
            }

            Map<String, INDArray> decodeInputs = new HashMap<>();
            decodeInputs.put(state.inputIdsName, state.decodeInputIds);
            if (state.causalMaskName != null && state.decodeCausalMask != null) {
                decodeInputs.put(state.causalMaskName, state.decodeCausalMask);
            }
            if (state.posOffsetName != null && state.decodePositionOffset != null) {
                decodeInputs.put(state.posOffsetName, state.decodePositionOffset);
            }
            if (state.cachePosName != null && state.decodeCachePosition != null) {
                decodeInputs.put(state.cachePosName, state.decodeCachePosition);
            }
            if (state.actualSeqLenName != null
                    && state.decodeActualSequenceLength != null) {
                decodeInputs.put(
                        state.actualSeqLenName, state.decodeActualSequenceLength);
            }
            if (liveKv != null) {
                for (Map.Entry<String, INDArray> entry : liveKv.entrySet()) {
                    if (decoder.hasVariable(entry.getKey())) {
                        decodeInputs.put(entry.getKey(), entry.getValue());
                    }
                }
            }
            for (Map.Entry<String, INDArray> entry
                    : state.recurrentStateBuffers.entrySet()) {
                if (decoder.hasVariable(entry.getKey())) {
                    decodeInputs.put(entry.getKey(), entry.getValue());
                }
            }

            Map<String, INDArray> outputs = decoder.outputDirect(
                    decodeInputs, state.decodeOutputNames.toArray(new String[0]));
            INDArray logits = outputs.get(state.logitsName);
            if (logits == null) {
                throw new IllegalStateException(
                        "Constrained decode did not return logits '" + state.logitsName
                                + "': " + outputs.keySet());
            }
            suppressStopsUnderFloor(
                    logits, 0, state.sampling, state.generatedSoFar.size(),
                    state.stopTokenIds);
            int nextToken = sampleToken(
                    logits, 0, state.sampling, state.generatedSoFar, state.rng,
                    state.constraintMasker, tokenizer, state.stopTokenIds);
            logits.close();

            boolean copiedRecurrentState = false;
            for (ModelIOConfig.RecurrentStatePair pair : state.recurrentStates) {
                INDArray updated = outputs.get(pair.outputName);
                INDArray retained = state.recurrentStateBuffers.get(pair.inputName);
                if (updated != null && retained != null) {
                    retained.assign(updated);
                    copiedRecurrentState = true;
                }
            }
            if (copiedRecurrentState) {
                Nd4j.getExecutioner().commit();
            }
            closeGeneratedKvOutputs(outputs, state.kvInputNames);
            for (INDArray output : outputs.values()) {
                if (output != null && !output.wasClosed()) {
                    output.close();
                }
            }

            state.releaseRecurrentCopyDonors();
            state.cachePosition++;
            state.lastGeneratedToken = nextToken;
            state.generatedSoFar.add(nextToken);
            callGenerated.add(nextToken);

            if (state.stopTokenIds.contains(nextToken)) {
                break;
            }
        }

        List<Integer> callTokens = isContinuation
                ? callGenerated
                : new ArrayList<>(state.generatedSoFar);
        int[] tokenIds = callTokens.stream().mapToInt(Integer::intValue).toArray();
        boolean hitEos = tokenIds.length > 0
                && state.stopTokenIds.contains(tokenIds[tokenIds.length - 1]);
        state.eosReached = state.eosReached || hitEos;

        String text = decodeGeneratedText(tokenIds, state.stopTokenIds);
        long timeMs = System.currentTimeMillis() - startTime;
        log.info("[Constraint] Constrained in-graph KV decode complete: "
                        + "callTokens={} cachePosition={} eos={} emittedText={}",
                callTokens.size(), state.cachePosition, hitEos,
                state.constraintMasker.getEmittedText());

        return GenerationResult.builder()
                .text(text)
                .tokenIds(tokenIds)
                .generatedTokenCount(tokenIds.length)
                .promptTokenCount(state.promptTokenCount)
                .totalTokenCount(state.promptTokenCount + state.generatedSoFar.size())
                .finishReason(hitEos
                        ? GenerationResult.FinishReason.EOS
                        : GenerationResult.FinishReason.MAX_TOKENS)
                .generationTimeMs(timeMs)
                .tokensPerSecond(
                        timeMs > 0 ? tokenIds.length * 1000.0 / timeMs : 0.0)
                .steadyStateTokensPerSecond(0.0)
                .lateSteadyStateTokensPerSecond(0.0)
                .effectiveTokensPerSecond(
                        timeMs > 0 ? tokenIds.length * 1000.0 / timeMs : 0.0)
                .sessionId(state.sessionId)
                .build();
    }

    // ==================== Continue-generation session API ====================

    /**
     * Start a resumable {@link GenerationSession}, sized to the pipeline's context ceiling.
     * Capacity is resolved as (first non-zero): {@code config.sessionCapacity}, then
     * {@code config.maxKvCacheLength - promptLen}, then {@code config.maxNewTokens}. To use the model's
     * full context window as the ceiling, set {@code maxKvCacheLength} to the context window (e.g. 512).
     *
     * @see #startSession(String, int)
     */
    public GenerationSession startSession(String prompt) {
        return startSession(prompt, -1);
    }

    /**
     * Start a resumable generation session with an explicit total continuation capacity (in new tokens).
     *
     * <p>The session prefills the prompt ONCE and pre-sizes its STATIC KV buffer to
     * {@code promptLen + capacity}. Subsequent {@link GenerationSession#continueGeneration(int)} calls
     * resume autoregressive decoding from the retained in-graph KV cache and current cache position —
     * no session reset, no re-prefill. Decoding is hard-bounded by the buffer: once it fills, continue
     * returns {@link GenerationResult.FinishReason#MAX_TOKENS} ("context full").</p>
     *
     * <p><b>Continuation contract.</b> For greedy decoding with the default repetition penalty, one
     * logical generation spread over K session calls is <em>numerically identical</em> to a single
     * {@code generate()} of the summed budget — {@code generate(N)} then {@code continueGeneration(M)}
     * equals {@code generate(N+M)}, token-for-token. With sampling ({@code temperature>0}) or a
     * repetition penalty {@code != 1.0} the continuation is valid but not bit-identical to a single call
     * (the Java/C++ RNG and the per-invocation penalty history do not carry across the seam).</p>
     *
     * <p><b>Threading.</b> A session is thread-confined: it must be driven from the thread that created
     * it (the decoder's {@code InferenceSession} and frozen plan are thread-affine). Only one session may
     * be active on a pipeline at a time; opening a second, or calling {@code generate(String,int)} while a
     * session is open, throws. Coordination is lock-free — no monitor is held across a native decode, so
     * misuse fails fast rather than deadlocking.</p>
     *
     * @param prompt   the input text prompt (chat template applied if configured)
     * @param capacity total new-token capacity for the session's lifetime; {@code <= 0} resolves from config
     * @return an open {@link GenerationSession}; call {@code generate(...)} to produce the first tokens
     */
    public GenerationSession startSession(String prompt, int capacity) {
        int restoreDevice = switchToDecoderDevice("start-session");
        OpaqueDataBuffer.suppressCrossDeviceRouting(true);
        try {
            if (embedTokens != null || !ModelIOConfig.isInGraphKvCache(decoder)) {
                throw new UnsupportedOperationException(
                        "GenerationSession continuation is currently supported only for single-model "
                        + "in-graph-KV (GGUF) decoders.");
            }
            if (config.getKvContinuationMode() == KvContinuationMode.GROWABLE) {
                throw new UnsupportedOperationException(
                        "KvContinuationMode.GROWABLE is not yet implemented (follow-up ADR). Use "
                        + "STATIC_CONTEXT_CEILING and size maxKvCacheLength to the model's context window.");
            }
            if (activeSession.get() != null) {
                throw new IllegalStateException("A GenerationSession is already active on this pipeline; close it first.");
            }
            int[] promptTokenIds = encodePromptToIds(prompt);
            int resolvedCapacity = resolveSessionCapacity(capacity, promptTokenIds.length);
            long startTime = System.currentTimeMillis();
            ModelIOConfig.KVCacheNames kvInputNames = ModelIOConfig.findKVCacheInputNames(decoder);

            // Transfer a compatible fixed-buffer state from the pipeline cache into this session.
            // Capacity participates in the native plan shape, so a mismatch requires a full teardown;
            // identical capacities reuse the plan, CUDA graph, KV/recurrent buffers, and input addresses.
            boolean fixedBuffers = config.getMaxPrefillLength() > 0;
            InGraphKvState reuseState = null;
            if (cachedFixedBufferState != null) {
                InGraphKvState candidate = cachedFixedBufferState;
                cachedFixedBufferState = null;
                long requestedMaxKvLen = (long) config.getMaxPrefillLength() + resolvedCapacity;
                int configuredKvCap = config.getMaxKvCacheLength();
                if (configuredKvCap > 0 && requestedMaxKvLen > configuredKvCap) {
                    requestedMaxKvLen = configuredKvCap;
                }
                if (fixedBuffers && !candidate.closed && candidate.maxKvLen == requestedMaxKvLen) {
                    reuseState = candidate;
                    log.info("[GenerationSession] reusing fixed-buffer DSP state (maxKvLen={})",
                            requestedMaxKvLen);
                } else {
                    candidate.close();
                }
            }

            // ── Prefix cache lookup for session start ──────────────────────────────────────────
            // A retained fixed-buffer state is the stronger cache: it preserves captured addresses
            // and refills the complete prompt in place, so do not replace it with a suffix plan.
            InGraphKvState state = null;
            if (prefixBlockPool != null && reuseState == null) {
                List<ModelIOConfig.RecurrentStatePair> recurrentStates =
                        ModelIOConfig.findRecurrentStatePairs(decoder, ioConfig);
                long roughMaxKvLen = (long) promptTokenIds.length + resolvedCapacity;
                int kvCap = config.getMaxKvCacheLength();
                if (kvCap > 0 && roughMaxKvLen > kvCap) roughMaxKvLen = kvCap;
                PrefixHitContext hit = attemptPrefixCacheHit(
                        promptTokenIds, kvInputNames, roughMaxKvLen, recurrentStates);
                if (hit != null) {
                    state = prefillSuffixOnlyAndFreeze(promptTokenIds, resolvedCapacity, kvInputNames, startTime, hit);
                }
            }
            if (state == null) {
                state = prefillWarmupAndFreeze(
                        promptTokenIds, resolvedCapacity, kvInputNames, startTime, reuseState);
                if (state.terminalResult != null && reuseState != null && state != reuseState) {
                    reuseState.close();
                }
                // Store completed prefill in prefix cache for future reuse
                if (prefixBlockPool != null && state.staticKvBuffers != null && state.terminalResult == null) {
                    List<ModelIOConfig.RecurrentStatePair> recurrentStates =
                            ModelIOConfig.findRecurrentStatePairs(decoder, ioConfig);
                    storePrefillInPrefixCache(promptTokenIds, state.actualPrefillLen,
                            state.staticKvBuffers, kvInputNames, state.recurrentStateBuffers, recurrentStates);
                }
            }
            state.sessionId = SESSION_ID_COUNTER.incrementAndGet();
            state.ownerThreadId = Thread.currentThread().getId();

            GenerationSession session = new GenerationSession(this, state, startTime);
            if (!activeSession.compareAndSet(null, session)) {
                state.close();
                throw new IllegalStateException("A GenerationSession opened concurrently on this pipeline.");
            }
            log.info("[GenerationSession] started id={} promptLen={} capacity={} maxKvLen={}",
                    state.sessionId, promptTokenIds.length, resolvedCapacity, state.maxKvLen);
            return session;
        } finally {
            OpaqueDataBuffer.suppressCrossDeviceRouting(false);
            restoreDevice(restoreDevice, "start-session");
        }
    }

    /**
     * Continue decoding from the pipeline's active session — a result-threaded convenience over
     * {@link GenerationSession#continueGeneration(int)}. The downstream self-heal loop calls this while
     * {@code prior.isTruncated()} and there is remaining context budget.
     *
     * @param prior         the previous result from the active session (its {@code sessionId} is validated)
     * @param maxNewTokens  up to this many additional tokens
     */
    public GenerationResult continueFrom(GenerationResult prior, int maxNewTokens) {
        GenerationSession session = activeSession.get();
        if (session == null) {
            throw new IllegalStateException("No active GenerationSession on this pipeline; call startSession() first.");
        }
        if (prior != null && prior.getSessionId() != 0L && prior.getSessionId() != session.getSessionId()) {
            throw new IllegalArgumentException("prior GenerationResult does not belong to the active session (id "
                    + prior.getSessionId() + " != " + session.getSessionId() + ").");
        }
        return session.continueGeneration(maxNewTokens);
    }

    private int resolveSessionCapacity(int capacityArg, int prefillLen) {
        if (capacityArg > 0) return capacityArg;
        if (config.getSessionCapacity() > 0) return config.getSessionCapacity();
        int kvCap = config.getMaxKvCacheLength();
        if (kvCap > 0) return Math.max(1, kvCap - prefillLen);
        return config.getMaxNewTokens();
    }

    /**
     * Resolve the default byte budget for the cross-request KV prefix block pool.
     * Heuristic: 10% of the device's total memory, capped at 512 MB, minimum 64 MB.
     */
    private static long resolvePrefixCacheDefaultBytes() {
        try {
            DeviceMemoryManager mgr = DeviceMemoryManager.getInstance();
            var dev = mgr.getDefaultDevice();
            long totalDeviceBytes = (dev != null) ? dev.getTotalMemory() : 0L;
            if (totalDeviceBytes <= 0) return 256L * 1024 * 1024; // 256 MB fallback
            long tenPercent = totalDeviceBytes / 10;
            long cap = 512L * 1024 * 1024; // 512 MB
            long floor = 64L * 1024 * 1024; // 64 MB
            return Math.max(floor, Math.min(cap, tenPercent));
        } catch (Exception e) {
            return 256L * 1024 * 1024; // 256 MB fallback
        }
    }

    /** Device-guarded native decode step for a session (with capacity clamp for continuation). */
    GenerationResult decodeInSession(InGraphKvState state, int maxNewTokens, boolean isContinuation) {
        int restoreDevice = switchToDecoderDevice("session-decode");
        OpaqueDataBuffer.suppressCrossDeviceRouting(true);
        try {
            if (isContinuation) {
                int available = state.remainingCapacity();
                if (available <= 0 || maxNewTokens <= 0) {
                    return GenerationResult.builder()
                            .text("").tokenIds(new int[0]).generatedTokenCount(0)
                            .promptTokenCount(state.promptTokenCount)
                            .totalTokenCount(state.promptTokenCount + state.generatedSoFar.size())
                            .finishReason(GenerationResult.FinishReason.MAX_TOKENS)
                            .sessionId(state.sessionId)
                            .build();
                }
                maxNewTokens = Math.min(maxNewTokens, available);
            }
            return runInGraphNativeDecode(state, maxNewTokens, isContinuation, System.currentTimeMillis());
        } finally {
            OpaqueDataBuffer.suppressCrossDeviceRouting(false);
            restoreDevice(restoreDevice, "session-decode");
        }
    }

    /**
     * Feed tokens into the frozen decode graph WITHOUT sampling (a "continue:" nudge), writing their
     * K/V and advancing the cache position so the next continueGeneration() resumes after them.
     *
     * <p>To keep the resume invariant ("the last token is unwritten"), this writes the current
     * last-generated token plus all but the final appended token, and designates the final appended
     * token as the new unwritten last.</p>
     */
    void appendInSession(InGraphKvState state, int[] tokens) {
        if (tokens == null || tokens.length == 0) return;
        int restoreDevice = switchToDecoderDevice("session-append");
        OpaqueDataBuffer.suppressCrossDeviceRouting(true);
        try {
            // Tokens whose K/V must be committed now: [oldLast, tokens[0..k-2]]. tokens[k-1] becomes the
            // new unwritten last.
            List<Integer> toFeed = new ArrayList<>();
            toFeed.add(state.lastGeneratedToken);
            for (int i = 0; i < tokens.length - 1; i++) toFeed.add(tokens[i]);

            for (int feed : toFeed) {
                if (state.remainingCapacity() <= 0) {
                    log.warn("[GenerationSession] append hit KV capacity at cachePosition={}", state.cachePosition);
                    break;
                }
                state.decodeInputIds.putScalar(new long[]{0, 0}, feed);
                if (state.decodeCausalMask != null) {
                    if (state.rotatingSlotMap != null) {
                        float maskVal = (state.maskDtype == DataType.HALF || state.maskDtype == DataType.FLOAT16)
                                ? -65504.0f : -1e9f;
                        float[] maskData = state.rotatingSlotMap.buildRotatingDecodeMask(
                                state.cachePosition, maskVal);
                        INDArray fresh = Nd4j.create(maskData, new long[]{1, 1, 1, state.maxKvLen}, 'c');
                        if (state.maskDtype != DataType.FLOAT) {
                            INDArray cast = fresh.castTo(state.maskDtype);
                            fresh.close();
                            fresh = cast;
                        }
                        state.decodeCausalMask.assign(fresh);
                        fresh.close();
                    } else {
                        INDArray fresh = DecoderInputBuilder.buildInGraphDecodeMask(
                                state.cachePosition - 1, state.maxKvLen, state.maskDtype);
                        state.decodeCausalMask.assign(fresh);
                        fresh.close();
                    }
                }
                if (state.decodePositionOffset != null) state.decodePositionOffset.putScalar(new long[]{}, (long) state.cachePosition);
                if (state.decodeCachePosition != null) {
                    long physSlot = (state.rotatingSlotMap != null)
                            ? state.rotatingSlotMap.physicalSlot(state.cachePosition)
                            : state.cachePosition;
                    state.decodeCachePosition.putScalar(new long[]{}, physSlot);
                }
                if (state.decodeActualSequenceLength != null) state.decodeActualSequenceLength.putScalar(new long[]{}, 1L);
                Nd4j.getExecutioner().commit();
                state.decodeInputIds.syncToDevice();
                if (state.decodeCausalMask != null) state.decodeCausalMask.syncToDevice();
                if (state.decodePositionOffset != null) state.decodePositionOffset.syncToDevice();
                if (state.decodeCachePosition != null) state.decodeCachePosition.syncToDevice();
                if (state.decodeActualSequenceLength != null) state.decodeActualSequenceLength.syncToDevice();
                for (INDArray kv : state.staticKvBuffers.values()) kv.syncToDevice();
                for (INDArray rs : state.recurrentStateBuffers.values()) rs.syncToDevice();

                Map<String, INDArray> decodeInputMap = new HashMap<>();
                decodeInputMap.put(state.inputIdsName, state.decodeInputIds);
                if (state.causalMaskName != null && state.decodeCausalMask != null) decodeInputMap.put(state.causalMaskName, state.decodeCausalMask);
                if (state.posOffsetName != null && state.decodePositionOffset != null) decodeInputMap.put(state.posOffsetName, state.decodePositionOffset);
                if (state.cachePosName != null && state.decodeCachePosition != null) decodeInputMap.put(state.cachePosName, state.decodeCachePosition);
                if (state.actualSeqLenName != null && state.decodeActualSequenceLength != null) {
                    decodeInputMap.put(state.actualSeqLenName, state.decodeActualSequenceLength);
                }
                for (Map.Entry<String, INDArray> e : state.staticKvBuffers.entrySet()) {
                    if (decoder.hasVariable(e.getKey())) decodeInputMap.put(e.getKey(), e.getValue());
                }
                for (Map.Entry<String, INDArray> e : state.recurrentStateBuffers.entrySet()) {
                    if (decoder.hasVariable(e.getKey())) decodeInputMap.put(e.getKey(), e.getValue());
                }

                // outputDirect stays on the frozen plan (plain output() may clear session caches).
                Map<String, INDArray> outputs = decoder.outputDirect(
                        decodeInputMap, state.decodeOutputNames.toArray(new String[0]));
                INDArray logits = outputs.get(state.logitsName);
                if (logits != null) logits.close();   // appended tokens are given, not sampled
                for (ModelIOConfig.RecurrentStatePair pair : state.recurrentStates) {
                    INDArray updated = outputs.get(pair.outputName);
                    if (updated != null) {
                        INDArray buf = state.recurrentStateBuffers.get(pair.inputName);
                        if (buf != null) buf.assign(updated);
                        updated.close();
                    }
                }
                closeGeneratedKvOutputs(outputs, state.kvInputNames);
                state.cachePosition += 1;
            }
            for (int t : tokens) state.generatedSoFar.add(t);
            state.lastGeneratedToken = tokens[tokens.length - 1];
            if (state.stopTokenIds.contains(state.lastGeneratedToken)) state.eosReached = true;
        } finally {
            OpaqueDataBuffer.suppressCrossDeviceRouting(false);
            restoreDevice(restoreDevice, "session-append");
        }
    }

    /**
     * End a logical session. Fixed-buffer state is returned to the pipeline cache instead of being
     * destroyed, because the frozen DSP plan captures its device addresses. Variable-shape and
     * terminal states are released immediately.
     */
    void closeSession(GenerationSession session, InGraphKvState state) {
        clearActiveSession(session);
        int restoreDevice = switchToDecoderDevice("session-close");
        try {
            if (config.getMaxPrefillLength() > 0
                    && !state.closed
                    && state.terminalResult == null) {
                InGraphKvState previous = cachedFixedBufferState;
                cachedFixedBufferState = state;
                if (previous != null && previous != state) previous.close();
                log.info("[GenerationSession] returned fixed-buffer DSP state to pipeline cache (maxKvLen={})",
                        state.maxKvLen);
            } else {
                state.close();
            }
        } finally {
            restoreDevice(restoreDevice, "session-close");
        }
    }

    /** Lock-free clear of the active session (only if {@code session} is still the current one). */
    void clearActiveSession(GenerationSession session) {
        activeSession.compareAndSet(session, null);
    }

    /**
     * A resumable text-generation session that owns the retained in-graph KV cache and cache position,
     * so decoding can be continued across multiple calls without a session reset or re-prefill.
     *
     * <p>Obtain one via {@link GenerationPipeline#startSession(String, int)}. The first
     * {@link #generate(int)} runs the initial decode (prefill + warmup happened at start); subsequent
     * {@link #generate(int)} / {@link #continueGeneration(int)} continue from the retained cache. See the
     * continuation contract and threading notes on {@code startSession}. Always {@link #close()} the
     * session (try-with-resources) to release or return its native buffers to the fixed-buffer pool.</p>
     */
    public static final class GenerationSession implements AutoCloseable {
        private final GenerationPipeline pipeline;
        private final InGraphKvState state;
        private final long sessionId;
        private boolean firstGenerateDone;
        private boolean closed;

        GenerationSession(GenerationPipeline pipeline, InGraphKvState state, long createTime) {
            this.pipeline = pipeline;
            this.state = state;
            this.sessionId = state.sessionId;
        }

        /** Opaque id used by {@link GenerationPipeline#continueFrom(GenerationResult, int)}. */
        public long getSessionId() { return sessionId; }
        /** True once a real EOS / stop token has been produced — continuation is then refused. */
        public boolean isEosReached() { requireOpen(); return state.eosReached; }
        /** Absolute position of the next-fed token in the KV buffer. */
        public int getCachePosition() { requireOpen(); return state.cachePosition; }
        /** New-token capacity remaining before the KV buffer is full. */
        public int getRemainingCapacity() { requireOpen(); return Math.max(0, state.remainingCapacity()); }
        /** All tokens generated so far across every call in this session (prompt excluded). */
        public int[] getAllTokens() {
            requireOpen();
            return state.generatedSoFar == null ? new int[0]
                    : state.generatedSoFar.stream().mapToInt(Integer::intValue).toArray();
        }
        /** The clean cumulative text of the whole session (decoded from all tokens at once). */
        public String getFullText() { return pipeline.tokenizer.decode(getAllTokens(), false); }
        /** Cooperatively stop a running {@link #continueToCompletion(int)} loop at the next boundary. */
        public void cancel() { requireOpen(); state.cancelRequested = true; }

        private void requireOpen() {
            if (closed || state.closed) {
                throw new IllegalStateException("GenerationSession is closed.");
            }
        }

        private void checkThread() {
            if (Thread.currentThread().getId() != state.ownerThreadId) {
                throw new IllegalStateException("GenerationSession is thread-confined: created on thread "
                        + state.ownerThreadId + " but used on thread " + Thread.currentThread().getId()
                        + ". Drive the session from its creating thread.");
            }
        }

        /**
         * Generate up to {@code maxNewTokens} tokens. The FIRST call runs the initial decode (prefill and
         * warmup were done at {@code startSession}); subsequent calls continue from the retained KV cache
         * (equivalent to {@link #continueGeneration(int)}).
         */
        public GenerationResult generate(int maxNewTokens) {
            checkThread();
            requireOpen();
            if (!firstGenerateDone) {
                firstGenerateDone = true;
                if (state.terminalResult != null) {
                    state.eosReached = true;
                    return state.terminalResult;   // prefill produced EOS / no plan handle
                }
                return pipeline.decodeInSession(state, maxNewTokens, false);
            }
            return continueGeneration(maxNewTokens);
        }

        /**
         * Continue decoding from the retained KV cache; up to {@code maxNewTokens} more tokens. The
         * returned result's {@code tokenIds} are only this call's new tokens (concatenate across calls to
         * reconstruct a single-shot sequence). Throws if the session has reached EOS.
         */
        public GenerationResult continueGeneration(int maxNewTokens) {
            checkThread();
            requireOpen();
            if (!firstGenerateDone) {
                throw new IllegalStateException("Call generate() first to run the initial decode.");
            }
            if (state.eosReached) {
                throw new IllegalStateException("Session reached EOS/stop; cannot continue past end-of-sequence.");
            }
            return pipeline.decodeInSession(state, maxNewTokens, true);
        }

        /**
         * Continue in fixed-size steps until EOS, buffer-full, degenerate repetition, or {@link #cancel()}.
         * Returns a single combined result whose {@code tokenIds} are all tokens produced by this loop's
         * continuation calls. Uses the pipeline's configured {@link RepetitionGuard}.
         */
        public GenerationResult continueToCompletion(int stepTokens) {
            return continueToCompletion(stepTokens, pipeline.config.getRepetitionGuard());
        }

        /** As {@link #continueToCompletion(int)} but with an explicit degenerate-loop guard. */
        public GenerationResult continueToCompletion(int stepTokens, RepetitionGuard guard) {
            checkThread();
            requireOpen();
            if (stepTokens <= 0) stepTokens = 1;
            List<Integer> loopTokens = new ArrayList<>();
            long start = System.currentTimeMillis();
            while (!state.eosReached && !state.cancelRequested && getRemainingCapacity() > 0) {
                GenerationResult step = continueGeneration(stepTokens);
                for (int t : step.getTokenIds()) loopTokens.add(t);
                if (step.getTokenIds().length == 0) break;   // nothing produced (capacity exhausted)
                if (guard != null && guard.isDegenerate(state.generatedSoFar)) {
                    return buildLoopResult(loopTokens, GenerationResult.FinishReason.REPETITION, start);
                }
            }
            GenerationResult.FinishReason reason =
                    state.eosReached ? GenerationResult.FinishReason.EOS
                    : state.cancelRequested ? GenerationResult.FinishReason.CANCELLED
                    : GenerationResult.FinishReason.MAX_TOKENS;
            return buildLoopResult(loopTokens, reason, start);
        }

        private GenerationResult buildLoopResult(List<Integer> loopTokens,
                                                 GenerationResult.FinishReason reason, long start) {
            int[] ids = loopTokens.stream().mapToInt(Integer::intValue).toArray();
            long timeMs = System.currentTimeMillis() - start;
            return GenerationResult.builder()
                    .text(pipeline.decodeGeneratedText(ids, state.stopTokenIds))
                    .tokenIds(ids).generatedTokenCount(ids.length)
                    .promptTokenCount(state.promptTokenCount)
                    .totalTokenCount(state.promptTokenCount + state.generatedSoFar.size())
                    .finishReason(reason)
                    .generationTimeMs(timeMs)
                    .tokensPerSecond(timeMs > 0 ? (ids.length * 1000.0 / timeMs) : 0)
                    .sessionId(state.sessionId)
                    .build();
        }

        /**
         * Feed {@code tokens} into the model WITHOUT sampling (a "continue:" nudge), writing their K/V and
         * advancing the cache position so the next {@link #continueGeneration(int)} resumes after them.
         * Must be called after the first {@link #generate(int)}.
         */
        public void append(int[] tokens) {
            checkThread();
            requireOpen();
            if (!firstGenerateDone) throw new IllegalStateException("Call generate() before append().");
            if (state.eosReached) throw new IllegalStateException("Session reached EOS; cannot append.");
            pipeline.appendInSession(state, tokens);
        }

        /**
         * Total byte count of the static (float) KV buffers retained for this session.
         * Returns 0 if no buffers have been allocated yet.
         */
        public long getStaticKvTotalBytes() {
            requireOpen();
            if (state.staticKvBuffers == null) return 0L;
            long total = 0L;
            for (INDArray a : state.staticKvBuffers.values()) {
                if (a != null && !a.wasClosed()) total += a.length() * a.dataType().width();
            }
            return total;
        }

        /**
         * Total byte count of the INT8/FP8-compressed quantized KV buffers for this session.
         * Returns 0 when {@code kvCacheStrategy != QUANTIZED} or before prefill completes.
         */
        public long getQuantizedKvTotalBytes() {
            requireOpen();
            if (state.quantizedKvBuffers == null) return 0L;
            long total = 0L;
            for (INDArray a : state.quantizedKvBuffers.values()) {
                if (a != null && !a.wasClosed()) total += a.length() * a.dataType().width();
            }
            return total;
        }

        /**
         * The KV quantization format active for this session (0 = not quantized, 1 = INT8, etc.).
         */
        public int getKvQuantFormat() {
            requireOpen();
            return state.kvQuantFormat;
        }

        /**
         * Whether this session is running with V2 live-quantized KV (float buffers freed, INT8 is live).
         * Returns true when {@link KvCacheStrategy#QUANTIZED} is active with {@code kvQuantFormat > 0}
         * and the float {@code staticKvBuffers} have been freed after prefill.
         * ADR 0107: when true, {@code getStaticKvTotalBytes()} returns 0 and the live KV footprint
         * is measurably smaller than an equivalent STATIC session.
         */
        public boolean isQuantizedV2() {
            requireOpen();
            return state.isQuantizedV2;
        }

        /**
         * Total byte count of the KV scale buffers (float32 per-token-per-head scales for V2 INT8).
         * Returns 0 when not in V2 mode.
         */
        public long getKvScaleTotalBytes() {
            requireOpen();
            if (state.kvScaleBuffers == null) return 0L;
            long total = 0L;
            for (INDArray a : state.kvScaleBuffers.values()) {
                if (a != null && !a.wasClosed()) total += a.length() * a.dataType().width();
            }
            return total;
        }

        /**
         * End this logical session. Idempotent and thread-confined. In fixed-buffer mode ownership of
         * the retained native buffers transfers back to the pipeline for the next compatible request;
         * the state is physically released when replaced or when the pipeline closes.
         */
        @Override
        public void close() {
            if (closed) return;
            checkThread();
            closed = true;
            pipeline.closeSession(this, state);
        }
    }

    /** Extract layer index from a KV cache input name like "past_key_values.3.key" → 3. */
    private static int extractLayerIndex(String kvInputName) {
        // Pattern: past_key_values.{N}.key or past_key_values.{N}.value
        String[] parts = kvInputName.split("\\.");
        for (int i = 0; i < parts.length; i++) {
            try {
                return Integer.parseInt(parts[i]);
            } catch (NumberFormatException ignore) {
            }
        }
        return 0;
    }

    /**
     * Derive the initial shape for a recurrent state placeholder by walking the graph.
     * Dispatches to GDN or conv derivation based on the consuming op type.
     */
    /**
     * Derive the zero-state shape for a recurrent state placeholder (GDN or causal-conv)
     * from the ops that consume it. Public so external teacher-forcing/scoring callers
     * (distillation target extraction, perplexity over hybrid architectures) can build
     * complete prefill input maps: {@link DecoderInputBuilder#buildDecoderInputMap}
     * covers ids/masks/positions/KV caches, and recurrent states are zero-filled with
     * the shape this method returns.
     */
    public static long[] deriveRecurrentStateShape(SameDiff sd, String stateName) {
        Variable var = sd.getVariables().get(stateName);
        if (var == null || var.getInputsForOp() == null || var.getInputsForOp().isEmpty()) {
            return null;
        }

        for (String opName : var.getInputsForOp()) {
            DifferentialFunction op;
            try { op = sd.getOpById(opName); } catch (Exception e) { log.debug("deriveRecurrentStateShape: getOpById('{}') failed", opName, e); continue; }
            if (op == null) continue;

            if (op instanceof GatedDeltaRule) {
                return deriveGdnStateShapeFromOp(sd, op, stateName);
            } else if ("causal_conv1d".equals(op.opName())) {
                return deriveConvStateShapeFromOp(sd, op);
            }
        }
        return null;
    }

    /**
     * Derive GDN state shape [1, H, D_k, D_v] from a GatedDeltaRule op.
     * Walks Q and V input chains backward through reshapes to find the constant H/D values.
     */
    private static long[] deriveGdnStateShapeFromOp(SameDiff sd, DifferentialFunction op, String stateName) {
        String[] inputNames = sd.getInputsForOp(op);
        if (inputNames == null || inputNames.length < 3) return null;

        long[] qDims = resolveReshapeHeadDims(sd, inputNames[0]);
        long[] vDims = resolveReshapeHeadDims(sd, inputNames[2]);
        if (qDims != null && vDims != null) {
            return new long[]{1, qDims[0], qDims[1], vDims[1]};
        }
        log.warn("[state-shape] Could not resolve Q/V dims for GDN state '{}'", stateName);
        return null;
    }

    /**
     * Derive conv state shape [1, D, K-1] from a CausalConv1d op's weight input.
     */
    private static long[] deriveConvStateShapeFromOp(SameDiff sd, DifferentialFunction op) {
        String[] inputNames = sd.getInputsForOp(op);
        if (inputNames == null || inputNames.length < 2) return null;

        SDVariable weightVar = sd.getVariable(inputNames[1]);
        if (weightVar != null && weightVar.getArr() != null) {
            long[] wShape = weightVar.getArr().shape();
            if (wShape.length == 2) {
                return new long[]{1, wShape[0], wShape[1] - 1};
            }
        }
        return null;
    }

    /**
     * Walk backwards from a variable through its producing ops to find a reshape
     * whose shape argument is a stack of [batchDim, seqDim, const(H), const(D)].
     * Returns [H, D] extracted from the constant elements.
     */
    private static long[] resolveReshapeHeadDims(SameDiff sd, String varName) {
        String current = varName;
        for (int depth = 0; depth < 15; depth++) {
            DifferentialFunction producingOp = sd.getVariableOutputOp(current);
            if (producingOp == null) return null;

            String[] inputs = sd.getInputsForOp(producingOp);
            if (inputs == null || inputs.length == 0) return null;

            if ("reshape".equals(producingOp.opName()) && inputs.length >= 2) {
                long[] result = tryExtractStackConstants(sd, inputs[1]);
                if (result != null) return result;
            }

            current = inputs[0];
        }
        return null;
    }

    /**
     * Given a variable name that should be the output of a stack op with
     * [batchDim, seqDim, const(H), const(D)], extract [H, D].
     */
    private static long[] tryExtractStackConstants(SameDiff sd, String shapeVarName) {
        DifferentialFunction shapeOp = sd.getVariableOutputOp(shapeVarName);
        if (shapeOp == null || !"stack".equals(shapeOp.opName())) return null;

        String[] stackInputs = sd.getInputsForOp(shapeOp);
        if (stackInputs == null || stackInputs.length < 4) return null;

        SDVariable hVar = sd.getVariable(stackInputs[2]);
        SDVariable dVar = sd.getVariable(stackInputs[3]);
        if (hVar == null || dVar == null) return null;
        if (hVar.getVariableType() != VariableType.CONSTANT || dVar.getVariableType() != VariableType.CONSTANT) {
            return null;
        }

        INDArray hArr = hVar.getArr();
        INDArray dArr = dVar.getArr();
        if (hArr == null || dArr == null) return null;

        return new long[]{hArr.getLong(0), dArr.getLong(0)};
    }

    /**
     * Select the request-level EOS token when one is configured; otherwise use
     * the tokenizer metadata. Package-private for lightweight precedence tests.
     */
    static int selectEosTokenId(SamplingConfig sampling, int tokenizerEosTokenId) {
        return selectEosTokenId(sampling, -1, tokenizerEosTokenId);
    }

    /**
     * EOS precedence is request override, imported model metadata, then tokenizer metadata.
     * The importer value is distinct because a source container can carry protocol metadata
     * that is not present in a separately supplied tokenizer.json.
     */
    static int selectEosTokenId(
            SamplingConfig sampling,
            int importedEosTokenId,
            int tokenizerEosTokenId) {
        if (sampling != null && sampling.getEosTokenId() >= 0) {
            return sampling.getEosTokenId();
        }
        return importedEosTokenId >= 0 ? importedEosTokenId : tokenizerEosTokenId;
    }

    private int resolveEosTokenId(SamplingConfig sampling) {
        int tokenizerEosTokenId = tokenizer.getEosTokenId();
        int importedEosTokenId = modelMetadata.getEosTokenId();
        int eosTokenId = selectEosTokenId(
                sampling, importedEosTokenId, tokenizerEosTokenId);
        if (sampling != null && sampling.getEosTokenId() >= 0) {
            log.info("[Generation] Resolved eosTokenId={} from active sampling config", eosTokenId);
        } else if (importedEosTokenId >= 0) {
            log.info("[Generation] Resolved eosTokenId={} from imported model metadata", eosTokenId);
        } else {
            log.info("[Generation] Resolved eosTokenId={} from tokenizer", eosTokenId);
        }
        return eosTokenId;
    }

    /**
     * Build the stop-token set from the already-resolved EOS token and configured
     * additional stop tokens.
     */
    private Set<Integer> buildStopTokenIds(int eosTokenId) {
        Set<Integer> stopTokenIds = new HashSet<>();
        if (eosTokenId >= 0) {
            stopTokenIds.add(eosTokenId);
        }
        stopTokenIds.addAll(modelMetadata.getStopTokenIds());
        stopTokenIds.addAll(activeChatStopTokenIds);
        if (config.getAdditionalStopTokenIds() != null) {
            stopTokenIds.addAll(config.getAdditionalStopTokenIds());
        }
        return stopTokenIds;
    }

    /** Close non-logits prefill outputs. */
    /**
     * Build a causal mask for padded prefill: positions within the actual prompt
     * attend causally to each other, but padding positions are fully masked out
     * so they don't corrupt attention scores.
     *
     * <p>Shape: [1, 1, prefillSeqLen, maxKvLen] — same as the un-padded mask
     * from {@link DecoderInputBuilder#buildInGraphCausalMask}. Padding queries retain key 0 as a
     * numerically safe attention target; fully masked rows become all {@code -Infinity} after an
     * FP32 mask is cast inside FP16 attention, and softmax over such a row is undefined. Real query
     * rows still mask every future and padding key, so padding cannot affect real-token outputs.</p>
     *
     * @param actualLen  real token count (un-padded)
     * @param paddedLen  padded prefill length (= maxPrefillLength)
     * @param maxKvLen   total KV buffer size (paddedLen + maxNewTokens)
     * @param dtype      mask data type (FLOAT or HALF)
     * @return attention bias [1, 1, paddedLen, maxKvLen]
     */
    static INDArray buildPaddedPrefillCausalMask(int actualLen, int paddedLen,
                                                  long maxKvLen, DataType dtype) {
        int Q = paddedLen;
        int K = (int) maxKvLen;
        float maskVal = (dtype == DataType.HALF || dtype == DataType.FLOAT16) ? -65504.0f : -1e9f;
        float[] data = new float[Q * K];

        for (int q = 0; q < Q; q++) {
            int rowOffset = q * K;
            if (q < actualLen) {
                // Real token row: causal mask (attend to 0..q, mask q+1..K-1)
                for (int k = q + 1; k < K; k++) {
                    data[rowOffset + k] = maskVal;
                }
            } else {
                // Padding outputs are discarded, but their softmax must remain numerically defined.
                // Leave key 0 unmasked and mask every other key. Real rows above still mask all
                // padding keys, so this cannot feed padding content back into a real-token output.
                for (int k = 1; k < K; k++) {
                    data[rowOffset + k] = maskVal;
                }
            }
        }

        INDArray mask = Nd4j.create(data, new long[]{1, 1, paddedLen, maxKvLen}, 'c');
        if (dtype != DataType.FLOAT) {
            INDArray cast = mask.castTo(dtype);
            mask.close();
            return cast;
        }
        return mask;
    }

    private boolean hasBundledMtpGraph() {
        if (!decoder.hasVariable(TARGET_HIDDEN_STATES_NAME)) return false;

        String[] required = {
                MTP_INPUT_IDS_NAME,
                MTP_TARGET_HIDDEN_NAME,
                MTP_POSITION_OFFSET_NAME,
                MTP_CACHE_POSITION_NAME,
                MTP_CAUSAL_MASK_NAME,
                MTP_KEY_CACHE_NAME,
                MTP_VALUE_CACHE_NAME,
                MTP_KEY_STATES_NAME,
                MTP_VALUE_STATES_NAME,
                MTP_HIDDEN_STATES_NAME,
                MTP_LOGITS_NAME
        };
        for (String name : required) {
            if (!decoder.hasVariable(name)) {
                throw new IllegalStateException("Incomplete bundled MTP graph: missing variable '" + name + "'");
            }
        }
        return true;
    }

    /**
     * Execute a SameDiff branch through an explicitly supplied session. This is the same inference
     * contract used by SameDiff.output(), but lets the target and MTP branches own independent DSP
     * executors while sharing immutable graph weights.
     */
    private static Map<String, INDArray> outputWithSession(
            InferenceSession session, Map<String, INDArray> placeholders, List<String> outputs) {
        for (INDArray array : placeholders.values()) {
            if (array != null) array.setCloseable(false);
        }
        try {
            ExecutionResult result = session.output(
                    outputs,
                    placeholders,
                    Collections.emptyMap(),
                    null,
                    Collections.emptyList(),
                    Collections.emptyList(),
                    At.defaultAt());
            if (result.getOutputs() != null) {
                return ExecutionResult.unpack(result.getOutputs());
            }
            Map<String, INDArray> unpacked = new LinkedHashMap<>();
            if (result.getValueOutputs() != null) {
                result.getValueOutputs().forEach((name, value) ->
                        unpacked.put(name, value != null ? value.getTensorValue() : null));
            }
            return unpacked;
        } finally {
            for (INDArray array : placeholders.values()) {
                if (array != null) {
                    try {
                        array.setCloseable(true);
                    } catch (Exception ignored) {
                        // Match SameDiff.output(): placeholder ownership always returns to the caller.
                    }
                }
            }
        }
    }

    private static final class MtpPreparedState {
        Map<String, INDArray> kvBuffers;
        Map<String, INDArray> prefillInputMap;
        INDArray inputIds;
        INDArray targetHiddenStates;
        INDArray causalMask;
        INDArray positionOffset;
        INDArray cachePosition;
        InferenceSession session;
        DynamicShapePlanExecutor executor;
        Pointer planHandle;
        Pointer contextHandle;
        int inputIdsExtIdx;
        int targetHiddenExtIdx;
        int causalMaskExtIdx;
        int positionOffsetExtIdx;
        int cachePositionExtIdx;
        int[] kvInputExtIndices;
        int logitsOutputIdx;
        int hiddenOutputIdx;
        int numPlanExternalInputs;
        int numPlanOutputs;
    }

    /**
     * Build or replay the isolated Qwen3.5 MTP prefill and scalar-decode plans.
     *
     * <p>Alignment is intentionally explicit: prefill row {@code t} receives token {@code x_t}
     * and target hidden {@code h_(t-1)} (row zero is all-zero); scalar warmup then consumes the
     * first target-sampled token with the final prompt hidden. Before returning, the retained
     * target-hidden input is advanced to the target warmup hidden so native drafting starts from
     * the second sampled token.</p>
     */
    private MtpPreparedState prepareBundledMtp(
            InGraphKvState reuseState,
            int[] effectiveTokenIds,
            int prefillSeqLen,
            int actualPrefillLen,
            long maxKvLen,
            int firstDecodePos,
            int firstTokenId,
            int secondTokenId,
            INDArray targetPrefillHidden,
            INDArray targetWarmupHidden) {

        if (targetPrefillHidden.rank() != 3 || targetWarmupHidden.rank() != 3) {
            throw new IllegalStateException("MTP target hidden outputs must be rank 3, got prefill="
                    + Arrays.toString(targetPrefillHidden.shape()) + " warmup="
                    + Arrays.toString(targetWarmupHidden.shape()));
        }
        long hidden = targetPrefillHidden.size(2);
        if (targetWarmupHidden.size(2) != hidden || actualPrefillLen < 1
                || actualPrefillLen > targetPrefillHidden.size(1)) {
            throw new IllegalStateException("Invalid MTP hidden alignment: prefill="
                    + Arrays.toString(targetPrefillHidden.shape()) + " warmup="
                    + Arrays.toString(targetWarmupHidden.shape()) + " actualPrefillLen="
                    + actualPrefillLen);
        }

        MtpPreparedState prepared = new MtpPreparedState();
        prepared.session = reuseState != null && reuseState.mtpSession != null
                ? reuseState.mtpSession : decoder.getInferenceFactory().create(decoder);
        prepared.prefillInputMap = reuseState != null && reuseState.mtpPrefillInputMap != null
                ? reuseState.mtpPrefillInputMap : new LinkedHashMap<>();
        prepared.kvBuffers = reuseState != null && reuseState.mtpKvBuffers != null
                ? reuseState.mtpKvBuffers : new LinkedHashMap<>();

        DataType mtpDtype = decoder.getVariable(MTP_TARGET_HIDDEN_NAME).dataType();

        INDArray prefillIds = prepared.prefillInputMap.get(MTP_INPUT_IDS_NAME);
        try (INDArray freshIds = Nd4j.createFromArray(effectiveTokenIds)
                .reshape(1, prefillSeqLen).castTo(DataType.INT64)) {
            if (prefillIds == null) {
                prefillIds = freshIds.dup();
                prepared.prefillInputMap.put(MTP_INPUT_IDS_NAME, prefillIds);
            } else {
                prefillIds.assign(freshIds);
            }
        }

        INDArray shiftedHidden = prepared.prefillInputMap.get(MTP_TARGET_HIDDEN_NAME);
        if (shiftedHidden == null) {
            shiftedHidden = Nd4j.zeros(mtpDtype, 1, prefillSeqLen, hidden);
            prepared.prefillInputMap.put(MTP_TARGET_HIDDEN_NAME, shiftedHidden);
        } else {
            shiftedHidden.assign(0);
        }
        if (prefillSeqLen > 1) {
            shiftedHidden.get(
                    NDArrayIndex.all(),
                    NDArrayIndex.interval(1, prefillSeqLen),
                    NDArrayIndex.all()).assign(
                    targetPrefillHidden.get(
                            NDArrayIndex.all(),
                            NDArrayIndex.interval(0, prefillSeqLen - 1),
                            NDArrayIndex.all()));
        }

        INDArray prefillPosition = prepared.prefillInputMap.get(MTP_POSITION_OFFSET_NAME);
        if (prefillPosition == null) {
            prefillPosition = Nd4j.scalar(DataType.INT64, 0L);
            prepared.prefillInputMap.put(MTP_POSITION_OFFSET_NAME, prefillPosition);
        } else {
            prefillPosition.assign(0);
        }
        INDArray prefillCachePosition = prepared.prefillInputMap.get(MTP_CACHE_POSITION_NAME);
        if (prefillCachePosition == null) {
            prefillCachePosition = Nd4j.scalar(DataType.INT64, 0L);
            prepared.prefillInputMap.put(MTP_CACHE_POSITION_NAME, prefillCachePosition);
        } else {
            prefillCachePosition.assign(0);
        }

        INDArray freshPrefillMask = config.getMaxPrefillLength() > 0
                ? buildPaddedPrefillCausalMask(
                        actualPrefillLen, prefillSeqLen, maxKvLen, DataType.FLOAT)
                : DecoderInputBuilder.buildInGraphCausalMask(
                        prefillSeqLen, maxKvLen, DataType.FLOAT);
        INDArray prefillMask = prepared.prefillInputMap.get(MTP_CAUSAL_MASK_NAME);
        if (prefillMask == null) {
            prefillMask = freshPrefillMask;
            prepared.prefillInputMap.put(MTP_CAUSAL_MASK_NAME, prefillMask);
        } else {
            prefillMask.assign(freshPrefillMask);
            freshPrefillMask.close();
        }

        if (!prepared.prefillInputMap.containsKey(MTP_KEY_CACHE_NAME)) {
            prepared.prefillInputMap.put(MTP_KEY_CACHE_NAME, Nd4j.empty(mtpDtype));
        }
        if (!prepared.prefillInputMap.containsKey(MTP_VALUE_CACHE_NAME)) {
            prepared.prefillInputMap.put(MTP_VALUE_CACHE_NAME, Nd4j.empty(mtpDtype));
        }

        List<String> prefillOutputsRequested = Arrays.asList(
                MTP_KEY_STATES_NAME, MTP_VALUE_STATES_NAME);
        Map<String, INDArray> mtpPrefillOutputs = outputWithSession(
                prepared.session, prepared.prefillInputMap, prefillOutputsRequested);
        INDArray keyStates = mtpPrefillOutputs.get(MTP_KEY_STATES_NAME);
        INDArray valueStates = mtpPrefillOutputs.get(MTP_VALUE_STATES_NAME);
        if (keyStates == null || valueStates == null || keyStates.rank() != 4 || valueStates.rank() != 4) {
            throw new IllegalStateException("MTP prefill did not return rank-4 K/V states: "
                    + mtpPrefillOutputs.keySet());
        }

        long kvHeads = keyStates.size(2);
        long headDim = keyStates.size(3);
        INDArray keyCache = prepared.kvBuffers.get(MTP_KEY_CACHE_NAME);
        if (keyCache == null) {
            keyCache = Nd4j.zeros(keyStates.dataType(), 1, maxKvLen, kvHeads, headDim);
            prepared.kvBuffers.put(MTP_KEY_CACHE_NAME, keyCache);
        } else {
            keyCache.assign(0);
        }
        INDArray valueCache = prepared.kvBuffers.get(MTP_VALUE_CACHE_NAME);
        if (valueCache == null) {
            valueCache = Nd4j.zeros(valueStates.dataType(), 1, maxKvLen, kvHeads, headDim);
            prepared.kvBuffers.put(MTP_VALUE_CACHE_NAME, valueCache);
        } else {
            valueCache.assign(0);
        }
        keyCache.get(
                NDArrayIndex.all(),
                NDArrayIndex.interval(0, actualPrefillLen),
                NDArrayIndex.all(),
                NDArrayIndex.all()).assign(
                keyStates.get(
                        NDArrayIndex.all(),
                        NDArrayIndex.interval(0, actualPrefillLen),
                        NDArrayIndex.all(),
                        NDArrayIndex.all()));
        valueCache.get(
                NDArrayIndex.all(),
                NDArrayIndex.interval(0, actualPrefillLen),
                NDArrayIndex.all(),
                NDArrayIndex.all()).assign(
                valueStates.get(
                        NDArrayIndex.all(),
                        NDArrayIndex.interval(0, actualPrefillLen),
                        NDArrayIndex.all(),
                        NDArrayIndex.all()));

        prepared.inputIds = reuseState != null ? reuseState.mtpInputIds : null;
        if (prepared.inputIds == null) prepared.inputIds = Nd4j.zeros(DataType.INT64, 1, 1);
        prepared.inputIds.putScalar(new long[]{0, 0}, firstTokenId);

        prepared.targetHiddenStates = reuseState != null ? reuseState.mtpTargetHiddenStates : null;
        if (prepared.targetHiddenStates == null) {
            prepared.targetHiddenStates = Nd4j.zeros(mtpDtype, 1, 1, hidden);
        }
        prepared.targetHiddenStates.assign(
                targetPrefillHidden.get(
                        NDArrayIndex.all(),
                        NDArrayIndex.interval(actualPrefillLen - 1, actualPrefillLen),
                        NDArrayIndex.all()));

        INDArray freshDecodeMask = DecoderInputBuilder.buildInGraphDecodeMask(
                firstDecodePos, maxKvLen, DataType.FLOAT);
        prepared.causalMask = reuseState != null ? reuseState.mtpCausalMask : null;
        if (prepared.causalMask == null
                || !Arrays.equals(prepared.causalMask.shape(), freshDecodeMask.shape())) {
            prepared.causalMask = freshDecodeMask;
        } else {
            prepared.causalMask.assign(freshDecodeMask);
            freshDecodeMask.close();
        }

        prepared.positionOffset = reuseState != null ? reuseState.mtpPositionOffset : null;
        if (prepared.positionOffset == null) {
            prepared.positionOffset = Nd4j.scalar(DataType.INT64, firstDecodePos);
        } else {
            prepared.positionOffset.putScalar(new long[]{}, (long) firstDecodePos);
        }
        prepared.cachePosition = reuseState != null ? reuseState.mtpCachePosition : null;
        if (prepared.cachePosition == null) {
            prepared.cachePosition = Nd4j.scalar(DataType.INT64, firstDecodePos);
        } else {
            prepared.cachePosition.putScalar(new long[]{}, (long) firstDecodePos);
        }

        Map<String, INDArray> decodeInputs = new LinkedHashMap<>();
        decodeInputs.put(MTP_INPUT_IDS_NAME, prepared.inputIds);
        decodeInputs.put(MTP_TARGET_HIDDEN_NAME, prepared.targetHiddenStates);
        decodeInputs.put(MTP_POSITION_OFFSET_NAME, prepared.positionOffset);
        decodeInputs.put(MTP_CACHE_POSITION_NAME, prepared.cachePosition);
        decodeInputs.put(MTP_CAUSAL_MASK_NAME, prepared.causalMask);
        decodeInputs.put(MTP_KEY_CACHE_NAME, keyCache);
        decodeInputs.put(MTP_VALUE_CACHE_NAME, valueCache);

        List<String> decodeOutputsRequested = Arrays.asList(
                MTP_LOGITS_NAME, MTP_HIDDEN_STATES_NAME);
        Map<String, INDArray> mtpDecodeOutputs = outputWithSession(
                prepared.session, decodeInputs, decodeOutputsRequested);
        INDArray mtpLogits = mtpDecodeOutputs.get(MTP_LOGITS_NAME);
        INDArray mtpHidden = mtpDecodeOutputs.get(MTP_HIDDEN_STATES_NAME);
        if (mtpLogits == null || mtpHidden == null) {
            throw new IllegalStateException("MTP scalar warmup did not return logits and hidden states: "
                    + mtpDecodeOutputs.keySet());
        }

        // Reading one predictor logit is the natural host-visible completion boundary for all
        // prefill-cache copies consumed by this warmup; no manual stream/device synchronization.
        double warmupProbe = mtpLogits.getDouble(0);
        log.info("[MTP] Scalar warmup complete: prefill={} actual={} hidden={} kvHeads={} headDim={} probe={}",
                prefillSeqLen, actualPrefillLen, hidden, kvHeads, headDim, warmupProbe);

        prepared.executor = prepared.session.getDynamicShapePlanExecutor();
        prepared.planHandle = prepared.executor != null
                ? prepared.executor.getNativePlanHandle() : null;
        if (prepared.executor == null || prepared.planHandle == null || prepared.planHandle.isNull()) {
            throw new IllegalStateException("Native MTP DSP plan handle is unavailable after scalar warmup");
        }

        if ((reuseState == null || reuseState.mtpExecutor == null)
                && prepared.executor.getCurrentPlan() != null) {
            prepared.executor.setMaxKvCacheLength((int) maxKvLen);
            prepared.executor.configureMaxAllocationForKvCache(mtpDecodeOutputs);
            boolean forcedSlotBySlot = decoder.getGraphExecutionMode() == GraphExecutionMode.SLOT_BY_SLOT
                    || Nd4j.getEnvironment().tritonSkipKernels();
            if (!forcedSlotBySlot) prepared.executor.setShapesFrozen(true);
        }

        prepared.contextHandle = prepared.executor.getCachedOpContext();
        prepared.inputIdsExtIdx = resolveExtInputIdx(prepared.executor, MTP_INPUT_IDS_NAME);
        prepared.targetHiddenExtIdx = resolveExtInputIdx(prepared.executor, MTP_TARGET_HIDDEN_NAME);
        prepared.positionOffsetExtIdx = resolveExtInputIdx(prepared.executor, MTP_POSITION_OFFSET_NAME);
        prepared.cachePositionExtIdx = resolveExtInputIdx(prepared.executor, MTP_CACHE_POSITION_NAME);
        prepared.causalMaskExtIdx = resolveExtInputIdx(prepared.executor, MTP_CAUSAL_MASK_NAME);
        prepared.kvInputExtIndices = new int[]{
                resolveExtInputIdx(prepared.executor, MTP_KEY_CACHE_NAME),
                resolveExtInputIdx(prepared.executor, MTP_VALUE_CACHE_NAME)
        };
        prepared.logitsOutputIdx = resolveOutputIdx(prepared.executor, MTP_LOGITS_NAME);
        prepared.hiddenOutputIdx = resolveOutputIdx(prepared.executor, MTP_HIDDEN_STATES_NAME);
        prepared.numPlanExternalInputs = prepared.executor.getCurrentPlan() != null
                ? prepared.executor.getCurrentPlan().getExternalInputKeys().length : 0;
        prepared.numPlanOutputs = decodeOutputsRequested.size();

        // Native drafting starts with the second target token and therefore needs h_P, produced by
        // the target warmup that consumed the first token at position P.
        prepared.targetHiddenStates.assign(
                targetWarmupHidden.get(
                        NDArrayIndex.all(),
                        NDArrayIndex.interval(0, 1),
                        NDArrayIndex.all()));
        prepared.inputIds.putScalar(new long[]{0, 0}, secondTokenId);

        keyStates.close();
        valueStates.close();
        targetPrefillHidden.close();
        mtpLogits.close();
        mtpHidden.close();
        return prepared;
    }

    private static void closePrefillOutputs(Map<String, INDArray> outputs, String logitsName) {
        for (Map.Entry<String, INDArray> entry : outputs.entrySet()) {
            if (!entry.getKey().equals(logitsName) && entry.getValue() != null) {
                entry.getValue().close();
            }
        }
    }

    /**
     * Close the per-step GGUF K/V outputs requested solely to keep prefill and decode on one DSP plan.
     * Retained cache ownership belongs to the past_key_values input buffers, not these output arrays.
     */
    private static void closeGeneratedKvOutputs(
            Map<String, INDArray> outputs, ModelIOConfig.KVCacheNames kvInputNames) {
        if (outputs == null || kvInputNames == null) return;
        for (String keyInputName : kvInputNames.keyNames) {
            int layerIdx = extractLayerIndex(keyInputName);
            closeOutput(outputs.get("k_rope_" + layerIdx));
            closeOutput(outputs.get("v_heads_" + layerIdx));
        }
    }

    private static void closeOutput(INDArray array) {
        if (array != null && !array.wasClosed()) array.close();
    }

    private static void closeOutputs(INDArray[] arrays) {
        if (arrays == null) return;
        for (INDArray array : arrays) closeOutput(array);
    }

    /**
     * Sample a token from logits at a specific sequence position using the full sampling pipeline.
     *
     * <p>When {@code sampling.isGreedy()}, performs argmax. Otherwise applies:
     * repetition penalty → temperature scaling → top-k filtering → top-p filtering → multinomial sampling.</p>
     *
     * <p>When {@code masker} is non-null, constraint masking is applied first (after extracting the
     * vocab slice), before any other sampling transforms. The masker also receives the selected token
     * via {@link ConstraintMasker#tokenEmitted} so it can update its accumulated-text state.</p>
     *
     * @param logits         logits tensor of shape [1, seqLen, vocab]
     * @param seqPos         sequence position to sample from
     * @param sampling       sampling configuration
     * @param generatedSoFar previously generated token IDs (for repetition penalty)
     * @param rng            random number generator (ignored for greedy)
     * @return sampled token ID
     */
    private int sampleToken(INDArray logits, long seqPos, SamplingConfig sampling,
                                   List<Integer> generatedSoFar, Random rng) {
        return sampleToken(logits, seqPos, sampling, generatedSoFar, rng, null, null);
    }

    /**
     * Sample a token from logits at a specific sequence position using the full sampling pipeline,
     * optionally applying a {@link ConstraintMasker} for structured-output / constrained decoding.
     *
     * <p>Constraint masking (v1 — ADR 0111): when {@code masker} is non-null, the raw logit slice
     * is converted to a {@code float[]} and passed through {@link ConstraintMasker#maskLogits} before
     * any other sampling filter. The resulting masked float[] is then wrapped back into an INDArray
     * for the standard pipeline (penalties → temperature → top-k → top-p → sample). After a token
     * is selected, {@link ConstraintMasker#tokenEmitted} is called to advance the automaton state.</p>
     *
     * <p>Performance note: the float[] round-trip adds a copy per step. On CPU this is negligible
     * relative to the forward pass. See ADR 0111 for the native-path masking design (Phase 2).</p>
     *
     * @param logits         logits tensor of shape [1, seqLen, vocab]
     * @param seqPos         sequence position to sample from
     * @param sampling       sampling configuration
     * @param generatedSoFar previously generated token IDs (for repetition penalty)
     * @param rng            random number generator (ignored for greedy)
     * @param masker         optional constraint masker; null = unconstrained (identical behavior)
     * @param tokenizer      tokenizer, used to decode token pieces for the masker; may be null when masker is null
     * @return sampled token ID
     */
    private int sampleToken(INDArray logits, long seqPos, SamplingConfig sampling,
                                   List<Integer> generatedSoFar, Random rng,
                                   ConstraintMasker masker, Tokenizer tokenizer) {
        return sampleToken(
                logits, seqPos, sampling, generatedSoFar, rng, masker, tokenizer, null);
    }

    private int sampleToken(INDArray logits, long seqPos, SamplingConfig sampling,
                                   List<Integer> generatedSoFar, Random rng,
                                   ConstraintMasker masker, Tokenizer tokenizer,
                                   Set<Integer> stopTokenIds) {
        INDArray slice = logits.get(
                NDArrayIndex.point(0),
                NDArrayIndex.point(seqPos),
                NDArrayIndex.all()).dup();

        int eosId = sampling.getEosTokenId();
        Set<Integer> effectiveStopTokenIds = stopTokenIds != null
                ? stopTokenIds
                : (eosId >= 0 ? Collections.singleton(eosId) : Collections.emptySet());

        // Apply constraint masking before all other sampling transforms.
        if (masker != null && tokenizer != null) {
            masker.enforceOutputBlockBudget(
                    generatedSoFar == null ? 0 : generatedSoFar.size());
            float[] rawLogits = slice.toFloatVector();
            String nonFiniteFailure = nonFiniteLogitsFailure(rawLogits, masker, generatedSoFar);
            if (nonFiniteFailure != null) {
                slice.close();
                throw new IllegalStateException(nonFiniteFailure);
            }
            float[] masked = masker.maskLogitsByDecodedCandidate(
                    rawLogits,
                    effectiveStopTokenIds,
                    specialTokenIds,
                    token -> constraintTokenPiece(tokenizer, token),
                    token -> decodeConstraintCandidate(token, generatedSoFar, tokenizer),
                    specialTokenPieces);
            String emittedText = masker.getEmittedText();
            constraintCandidateDiagnostics.captureAndLog(
                    emittedText,
                    rawLogits,
                    masked,
                    sampling.isGreedy(),
                    token -> constraintDiagnosticTokenText(
                            token, emittedText, generatedSoFar, tokenizer));
            INDArray maskedSlice = Nd4j.create(masked, new long[]{masked.length}, slice.dataType());
            slice.close();
            slice = maskedSlice;
        }

        if (sampling.isGreedy()) {
            while (true) {
                int token = SamplerUtils.argmax(slice);
                double selectedLogit = slice.getDouble(token);
                if (!Double.isFinite(selectedLogit)) {
                    String failure = masker != null
                            ? constraintDeadEndMessage(masker, generatedSoFar)
                            : nonFiniteLogitsFailure(slice.toFloatVector(), null, generatedSoFar);
                    slice.close();
                    throw new IllegalStateException(failure);
                }
                if (acceptConstraintCandidate(
                        token, generatedSoFar, masker, tokenizer, effectiveStopTokenIds)) {
                    slice.close();
                    return token;
                }
                slice.putScalar(token, Double.NEGATIVE_INFINITY);
            }
        }

        // Apply seen-token penalties
        if (sampling.hasTokenPenalties() && generatedSoFar != null && !generatedSoFar.isEmpty()) {
            int[] prev = generatedSoFar.stream().mapToInt(Integer::intValue).toArray();
            INDArray penalized = SamplerUtils.applyTokenPenalties(slice, prev,
                    sampling.getRepetitionPenalty(), sampling.getFrequencyPenalty(), sampling.getPresencePenalty());
            if (penalized != slice) slice.close();
            slice = penalized;
        }

        // Apply min-p filtering
        if (sampling.hasMinP()) {
            INDArray filtered = SamplerUtils.minPFilter(slice, sampling.getMinP());
            if (filtered != slice) slice.close();
            slice = filtered;
        }

        // Apply typical-p filtering (same placement as the native token_sample pipeline:
        // after penalties and min-p, before temperature)
        if (sampling.hasTypicalP()) {
            INDArray filtered = SamplerUtils.typicalPFilter(slice, sampling.getTypicalP());
            if (filtered != slice) slice.close();
            slice = filtered;
        }

        // Apply XTC (exclude top choices) filtering
        if (sampling.hasXtc()) {
            INDArray filtered = SamplerUtils.xtcFilter(slice, sampling.getXtcProbability(),
                    sampling.getXtcThreshold(), rng);
            if (filtered != slice) slice.close();
            slice = filtered;
        }

        // Apply temperature
        if (sampling.getTemperature() != 1.0 && sampling.getTemperature() > 0) {
            INDArray scaled = SamplerUtils.applyTemperature(slice, sampling.getTemperature());
            if (scaled != slice) slice.close();
            slice = scaled;
        }

        // Apply top-k filtering
        if (sampling.hasTopK()) {
            INDArray filtered = SamplerUtils.topKFilter(slice, sampling.getTopK());
            if (filtered != slice) slice.close();
            slice = filtered;
        }

        // Apply top-p filtering
        if (sampling.hasTopP()) {
            INDArray filtered = SamplerUtils.topPFilter(slice, sampling.getTopP());
            if (filtered != slice) slice.close();
            slice = filtered;
        }

        // Convert to probabilities and sample. If exact full-sequence decoding rejects the
        // sampled candidate, remove it and resample from the renormalized distribution.
        while (true) {
            double bestLogit = slice.maxNumber().doubleValue();
            if (!Double.isFinite(bestLogit)) {
                String failure = masker != null
                        ? constraintDeadEndMessage(masker, generatedSoFar)
                        : nonFiniteLogitsFailure(slice.toFloatVector(), null, generatedSoFar);
                slice.close();
                throw new IllegalStateException(failure);
            }
            INDArray probs = SamplerUtils.softmax(slice);
            int token = SamplerUtils.multinomialSample(probs, rng);
            probs.close();
            if (acceptConstraintCandidate(
                    token, generatedSoFar, masker, tokenizer, effectiveStopTokenIds)) {
                slice.close();
                return token;
            }
            slice.putScalar(token, Double.NEGATIVE_INFINITY);
        }
    }

    private static String decodeConstraintCandidate(
            int token, List<Integer> generatedSoFar, Tokenizer tokenizer) {
        int size = generatedSoFar == null ? 0 : generatedSoFar.size();
        int[] candidateIds = new int[size + 1];
        for (int index = 0; index < size; index++) {
            candidateIds[index] = generatedSoFar.get(index);
        }
        candidateIds[size] = token;
        return tokenizer.decode(candidateIds, false);
    }

    private static String constraintDiagnosticTokenText(
            int token,
            String emittedText,
            List<Integer> generatedSoFar,
            Tokenizer tokenizer) {
        String piece = constraintTokenPiece(tokenizer, token);
        String decodedCandidate = decodeConstraintCandidate(token, generatedSoFar, tokenizer);
        String emitted = emittedText == null ? "" : emittedText;
        String extension = decodedCandidate != null && decodedCandidate.startsWith(emitted)
                ? decodedCandidate.substring(emitted.length()) : decodedCandidate;
        if (Objects.equals(piece, extension)) {
            return extension;
        }
        return "piece=" + piece + " extension=" + extension;
    }

    private boolean acceptConstraintCandidate(
            int token,
            List<Integer> generatedSoFar,
            ConstraintMasker masker,
            Tokenizer tokenizer,
            Set<Integer> stopTokenIds) {
        if (masker == null || tokenizer == null) {
            return true;
        }
        Set<Integer> terminals = stopTokenIds == null
                ? Collections.emptySet() : stopTokenIds;
        if (terminals.contains(token) && masker.isComplete()) {
            return true;
        }
        if (specialTokenIds.contains(token)) {
            String piece = constraintTokenPiece(tokenizer, token);
            if (!masker.allowsSpecialToken(piece)) {
                return false;
            }
            masker.specialTokenEmitted(piece);
            return true;
        }
        String decoded = decodeConstraintCandidate(token, generatedSoFar, tokenizer);
        if (!masker.allowsDecodedText(decoded, specialTokenPieces)) {
            return false;
        }
        masker.decodedTextEmitted(decoded);
        return true;
    }

    private String nonFiniteLogitsFailure(
            float[] rawLogits, ConstraintMasker masker, List<Integer> generatedSoFar) {
        int finite = 0;
        int nan = 0;
        int positiveInfinity = 0;
        int negativeInfinity = 0;
        for (float value : rawLogits) {
            if (Float.isFinite(value)) {
                finite++;
            } else if (Float.isNaN(value)) {
                nan++;
            } else if (value > 0.0f) {
                positiveInfinity++;
            } else {
                negativeInfinity++;
            }
        }
        if (finite > 0) {
            return null;
        }

        StringBuilder dsp = new StringBuilder();
        try {
            DspHandle handle = decoder.dsp();
            if (handle.isCompiled()) {
                int firstNonFiniteSlot = firstNonFiniteDspSlot(handle);
                dsp.append(", firstNonFiniteSlot=").append(firstNonFiniteSlot);
                if (firstNonFiniteSlot >= 0) {
                    DynamicShapePlanExecutor executor =
                            decoder.getOrCreateSession().getDynamicShapePlanExecutor();
                    DynamicShapePlan plan = executor == null ? null : executor.getCurrentPlan();
                    DynamicShapeSlot producer = producerForOutputSlot(plan, firstNonFiniteSlot);
                    if (producer != null) {
                        dsp.append(", firstNonFiniteOp=").append(producer.getOpName())
                                .append(", firstNonFiniteInputs=")
                                .append(Arrays.toString(producer.getInputVarNames()))
                                .append(", firstNonFiniteOutputs=")
                                .append(Arrays.toString(producer.getOutputVarNames()));
                        appendDspSlotStats(
                                dsp, handle, firstNonFiniteSlot, "firstNonFinite",
                                producer.getOpName());
                        appendDspInputLineage(
                                dsp, handle, decoder, plan, producer, 10, new HashSet<>());
                    } else {
                        dsp.append(", firstNonFiniteOp=unknown-output-slot");
                    }
                }
            }
        } catch (RuntimeException diagnosticFailure) {
            dsp.append(", dspDiagnosticFailure=")
                    .append(diagnosticFailure.getClass().getSimpleName())
                    .append(':').append(diagnosticFailure.getMessage());
        }

        String constraintType = masker == null || masker.getConstraint() == null
                ? "none" : masker.getConstraint().getClass().getSimpleName();
        int tokenCount = generatedSoFar == null ? 0 : generatedSoFar.size();
        return "Model produced no finite logits before constraint masking"
                + " [constraint=" + constraintType
                + ", generatedTokens=" + tokenCount
                + ", finite=" + finite
                + ", nan=" + nan
                + ", positiveInfinity=" + positiveInfinity
                + ", negativeInfinity=" + negativeInfinity
                + dsp + "]";
    }

    /**
     * Locate the first output slot containing an actual NaN or infinity.
     *
     * <p>{@link DspHandle#firstNaNSlot()}
     * uses a reduction sum, which can overflow for large finite FP16 tensors and
     * therefore misidentify an early slot. This slower element-wise scan runs
     * only after generation has already failed with wholly non-finite logits.</p>
     */
    private static int firstNonFiniteDspSlot(DspHandle handle) {
        for (int slotIndex = 0; slotIndex < handle.totalSlots(); slotIndex++) {
            INDArray slot = handle.getSlotOutput(slotIndex);
            if (slot == null || !slot.dataType().isNumerical()) {
                continue;
            }
            float[] values = slot.toFloatVector();
            for (float value : values) {
                if (!Float.isFinite(value)) {
                    return slotIndex;
                }
            }
        }
        return -1;
    }

    private static DynamicShapeSlot producerForOutputSlot(
            DynamicShapePlan plan, int outputSlotIndex) {
        if (plan == null || plan.getSlots() == null) {
            return null;
        }
        for (DynamicShapeSlot slot : plan.getSlots()) {
            for (int candidate : slot.getOutputSlotIndices()) {
                if (candidate == outputSlotIndex) {
                    return slot;
                }
            }
        }
        return null;
    }

    /**
     * Trace authoritative input output-slot indices rather than resolving by variable name.
     * SameDiff names may be rewritten after graph construction, while the compiled source
     * indices are the exact wiring consumed by DSP.
     */
    private static void appendDspInputLineage(
            StringBuilder diagnostic,
            DspHandle handle,
            SameDiff graph,
            DynamicShapePlan plan,
            DynamicShapeSlot consumer,
            int remainingDepth,
            Set<Integer> visited) {
        if (remainingDepth <= 0 || consumer == null || plan == null || plan.getSlots() == null) {
            return;
        }
        int[] sources = consumer.getInputSourceIndices();
        byte[] sourceTypes = consumer.getInputSourceTypes();
        String[] inputNames = consumer.getInputVarNames();
        if (sources == null) {
            return;
        }
        for (int i = 0; i < sources.length; i++) {
            int outputSlot = sources[i];
            String inputName = inputNames != null && i < inputNames.length && inputNames[i] != null
                    ? inputNames[i] : consumer.getOpName() + ".input" + i;
            boolean opOutput = outputSlot >= 0
                    && (sourceTypes == null || i >= sourceTypes.length
                    || sourceTypes[i] == DynamicShapeSlot.SOURCE_OP_OUTPUT);
            if (!opOutput) {
                INDArray externalValue = graph == null ? null : graph.getArrForVarName(inputName);
                appendDspArrayStats(
                        diagnostic, externalValue, inputName,
                        "externalSource=" + (sourceTypes != null && i < sourceTypes.length
                                ? sourceTypes[i] : "unknown"));
                continue;
            }
            if (!visited.add(outputSlot)) {
                continue;
            }
            DynamicShapeSlot producer = producerForOutputSlot(plan, outputSlot);
            appendDspSlotStats(diagnostic, handle, outputSlot, inputName,
                    producer == null ? "unknown" : producer.getOpName());
            appendDspInputLineage(
                    diagnostic, handle, graph, plan, producer, remainingDepth - 1, visited);
        }
    }

    private static void appendDspSlotStats(
            StringBuilder diagnostic,
            DspHandle handle,
            int outputSlot,
            String label) {
        appendDspSlotStats(diagnostic, handle, outputSlot, label, "unknown");
    }

    private static void appendDspSlotStats(
            StringBuilder diagnostic,
            DspHandle handle,
            int outputSlot,
            String label,
            String producerName) {
        try {
            INDArray value = handle.getSlotOutput(outputSlot);
            appendDspArrayStats(
                    diagnostic, value, label,
                    "slot=" + outputSlot + ", producer=" + producerName);
        } catch (RuntimeException diagnosticFailure) {
            diagnostic.append("\n  dspLineage[").append(label).append("]={slot=")
                    .append(outputSlot).append(", producer=").append(producerName)
                    .append(", unavailable=")
                    .append(diagnosticFailure.getClass().getSimpleName()).append('}');
        }
    }

    private static void appendDspArrayStats(
            StringBuilder diagnostic,
            INDArray value,
            String label,
            String provenance) {
        diagnostic.append("\n  dspLineage[").append(label).append("]={")
                .append(provenance);
        if (value == null) {
            diagnostic.append(", value=unavailable}");
            return;
        }
        if (!value.dataType().isNumerical()) {
            diagnostic.append(", dtype=").append(value.dataType())
                    .append(", shape=").append(Arrays.toString(value.shape()))
                    .append(", value=non-numerical}");
            return;
        }
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        int finite = 0;
        int nan = 0;
        int positiveInfinity = 0;
        int negativeInfinity = 0;
        for (float element : value.toFloatVector()) {
            if (Float.isNaN(element)) {
                nan++;
            } else if (element == Float.POSITIVE_INFINITY) {
                positiveInfinity++;
            } else if (element == Float.NEGATIVE_INFINITY) {
                negativeInfinity++;
            } else {
                finite++;
                min = Math.min(min, element);
                max = Math.max(max, element);
            }
        }
        diagnostic.append(", dtype=").append(value.dataType())
                .append(", shape=").append(Arrays.toString(value.shape()))
                .append(", finite=").append(finite)
                .append(", min=").append(min)
                .append(", max=").append(max)
                .append(", nan=").append(nan)
                .append(", +inf=").append(positiveInfinity)
                .append(", -inf=").append(negativeInfinity)
                .append('}');
    }

    private static String constraintDeadEndMessage(
            ConstraintMasker masker, List<Integer> generatedSoFar) {
        String emitted = masker == null ? "" : masker.getEmittedText();
        String escaped = emitted == null ? "" : emitted
                .replace("\\", "\\\\")
                .replace("\r", "\\r")
                .replace("\n", "\\n")
                .replace("\t", "\\t");
        int maxChars = 512;
        String bounded = escaped.length() <= maxChars
                ? escaped : "…" + escaped.substring(escaped.length() - maxChars);
        String constraintType = masker == null || masker.getConstraint() == null
                ? "none" : masker.getConstraint().getClass().getSimpleName();
        int tokenCount = generatedSoFar == null ? 0 : generatedSoFar.size();
        return "Constraint rejected every candidate token after exact sequence decode"
                + " [constraint=" + constraintType
                + ", generatedTokens=" + tokenCount
                + ", complete=" + (masker != null && masker.isComplete())
                + ", emitted=\"" + bounded + "\"]";
    }

    /**
     * Resolve the text actually contributed by one token. Added/special tokens are often absent
     * from {@link Tokenizer#getToken(int)} even though decode retains them; native tool sentinels
     * are one such case in LFM2.5. Prefer the cheap vocabulary lookup and fall back to a one-token
     * decode so a valid sentinel is never mistaken for an unavailable constraint transition.
     */
    private static String constraintTokenPiece(Tokenizer tokenizer, int tokenId) {
        String piece = tokenizer.getToken(tokenId);
        return piece == null || piece.isEmpty()
                ? tokenizer.decode(new int[]{tokenId}, false)
                : piece;
    }

    private static List<String> decodeSpecialTokenPieces(
            Tokenizer tokenizer, Collection<Integer> tokenIds) {
        if (tokenizer == null || tokenIds == null || tokenIds.isEmpty()) {
            return List.of();
        }
        List<String> pieces = new ArrayList<>(tokenIds.size());
        for (Integer tokenId : tokenIds) {
            if (tokenId == null || tokenId < 0) {
                continue;
            }
            String piece = constraintTokenPiece(tokenizer, tokenId);
            if (piece != null && !piece.isEmpty() && !pieces.contains(piece)) {
                pieces.add(piece);
            }
        }
        return List.copyOf(pieces);
    }

    private GenerationResult buildResult(List<Integer> generatedTokens, int[] promptTokenIds,
                                          Set<Integer> stopTokenIds, long startTime, long firstTokenMs) {
        return buildResult(generatedTokens, promptTokenIds, stopTokenIds, startTime, firstTokenMs, 0);
    }

    /**
     * Decode loop for merged-If decoders (optimum decoder_model_merged exports):
     * true-length dynamic KV — each step feeds the previous step's presents back as
     * pasts verbatim (the graph derives positions from shape(past), so lengths must
     * be REAL). Runs on the interpreted session; per-step map, no static buffers.
     */
    private GenerationResult dynamicMergedIfDecodeLoop(Map<String, INDArray> prefillOutputs,
            ModelIOConfig.KVCacheNames kvNames, String useCacheBranchName,
            SamplingConfig sampling, Random rng, Set<Integer> stopTokenIds,
            int maxNewTokens, int actualPrefillLen, INDArray encoderOutputs,
            int[] promptTokenIds, long startTime, List<String> allOutputNames) {

        String logitsOut = ioConfig.getLogitsOutputName() != null
                && prefillOutputs.containsKey(ioConfig.getLogitsOutputName())
                ? ioConfig.getLogitsOutputName()
                : allOutputNames.get(0);
        INDArray logits = prefillOutputs.get(logitsOut);
        if (logits == null) {
            throw new IllegalStateException("Merged-If decode: prefill produced no logits ('"
                    + logitsOut + "'); outputs=" + prefillOutputs.keySet());
        }
        long firstTokenMs = System.currentTimeMillis() - startTime;

        List<Integer> tokens = new ArrayList<>();
        int samplePos = Math.min(actualPrefillLen - 1, (int) logits.shape()[1] - 1);
        suppressStopsUnderFloor(logits, samplePos, sampling, 0, stopTokenIds);
        int token = sampleToken(logits, samplePos, sampling, tokens, rng);
        tokens.add(token);

        // presents (this step) -> pasts (next step), true-length.
        Map<String, INDArray> pasts = new LinkedHashMap<>();
        List<String> presentNames = new ArrayList<>(kvNames.keyNames);
        presentNames.addAll(kvNames.valueNames);
        for (String presentName : presentNames) {
            INDArray present = prefillOutputs.get(presentName);
            if (present == null) {
                throw new IllegalStateException("Merged-If decode: prefill missing present '" + presentName + "'");
            }
            pasts.put(ioConfig.presentToInputName(presentName), present);
        }
        if (log.isInfoEnabled()) {
            StringBuilder kvStats = new StringBuilder();
            int shown = 0;
            for (String presentName : presentNames) {
                if (shown++ >= 4) break;
                INDArray p = prefillOutputs.get(presentName);
                kvStats.append(' ').append(presentName).append("=amean:")
                        .append(String.format("%.4f", p.amean().getDouble(0)))
                        .append("/shape:").append(java.util.Arrays.toString(p.shape()));
            }
            log.info("Merged-If prefill KV hand-off stats:{}", kvStats);
        }

        String encName = ioConfig.getEncoderHiddenStatesName();
        // Experiment/diagnostic mode: recompute the WHOLE prefix through the no-past
        // branch each step instead of incremental with-past decode. Isolates
        // with-past-branch defects (that subgraph is otherwise only exercised here).
        boolean recomputeMode = Boolean.getBoolean("merged.decode.recompute");
        List<Long> prefixIds = null;
        if (recomputeMode) {
            prefixIds = new ArrayList<>();
            for (int pid : promptTokenIds) prefixIds.add((long) pid);
            prefixIds.add((long) token);
            log.info("Merged-If decode: RECOMPUTE mode (full prefix through no-past branch each step)");
        }
        while (tokens.size() < maxNewTokens && !stopTokenIds.contains(token)) {
            Map<String, INDArray> stepMap;
            if (recomputeMode) {
                stepMap = new LinkedHashMap<>();
                long[] ids = prefixIds.stream().mapToLong(Long::longValue).toArray();
                stepMap.put(ioConfig.getInputIdsName(), Nd4j.createFromArray(new long[][]{ids}));
                stepMap.put(useCacheBranchName, Nd4j.scalar(false));
            } else {
                stepMap = new LinkedHashMap<>(pasts);
                stepMap.put(ioConfig.getInputIdsName(), Nd4j.createFromArray(new long[][]{{token}}));
                stepMap.put(useCacheBranchName, Nd4j.scalar(true));
            }
            if (encoderOutputs != null && encName != null && decoder.hasVariable(encName)) {
                stepMap.put(encName, encoderOutputs);
            }

            Map<String, INDArray> stepOut = decoder.output(stepMap, allOutputNames.toArray(new String[0]));
            logits = stepOut.get(logitsOut);
            if (logits == null) {
                throw new IllegalStateException("Merged-If decode step " + tokens.size()
                        + ": no logits in outputs " + stepOut.keySet());
            }
            suppressStopsUnderFloor(logits, logits.shape()[1] - 1, sampling, tokens.size(), stopTokenIds);
            if (log.isInfoEnabled() && tokens.size() <= 6) {
                logTopK(logits, logits.shape()[1] - 1, tokens.size());
            }
            token = sampleToken(logits, logits.shape()[1] - 1, sampling, tokens, rng);
            tokens.add(token);

            if (recomputeMode) {
                prefixIds.add((long) token);
                continue;
            }
            Map<String, INDArray> nextPasts = new LinkedHashMap<>();
            for (String presentName : presentNames) {
                INDArray present = stepOut.get(presentName);
                String pastName = ioConfig.presentToInputName(presentName);
                // Optimum merged-decoder CONTRACT: the with-past branch emits EMPTY
                // constant dummies for presents it does not update (encoder/cross-attn
                // KV — proto: Constant() with [0,...] value). Empty present means
                // "unchanged": RETAIN the existing past; only real tensors replace.
                if (present == null || present.isEmpty() || present.length() == 0) {
                    INDArray retained = pasts.get(pastName);
                    if (retained == null) {
                        throw new IllegalStateException("Merged-If decode step " + tokens.size()
                                + ": present '" + presentName + "' empty and no prior past to retain");
                    }
                    nextPasts.put(pastName, retained);
                } else {
                    nextPasts.put(pastName, present);
                }
            }
            pasts = nextPasts;
        }

        log.info("Merged-If dynamic decode: {} tokens ({} max), stop={}",
                tokens.size(), maxNewTokens, stopTokenIds.contains(token));
        return buildResult(tokens, promptTokenIds, stopTokenIds, startTime, firstTokenMs);
    }

    /** Debug: top-5 token ids/logits at a position — distinguishes near-miss from broken conditioning. */
    private static void logTopK(INDArray logits, long seqPos, int step) {
        INDArray row = logits.get(NDArrayIndex.point(0), NDArrayIndex.point(seqPos), NDArrayIndex.all()).dup();
        float[] vals = row.data().asFloat();
        Integer[] idx = new Integer[vals.length];
        for (int i = 0; i < vals.length; i++) idx[i] = i;
        java.util.Arrays.sort(idx, (a, b) -> Float.compare(vals[b], vals[a]));
        StringBuilder sb = new StringBuilder();
        for (int k = 0; k < 5; k++) {
            sb.append(String.format(" %d:%.2f", idx[k], vals[idx[k]]));
        }
        log.info("Decode step {} top-5:{}", step, sb);
        row.close();
    }

    /** While under sampling.minNewTokens, mask stop-token logits to -1e9 in-place at the sampled position. */
    private static void suppressStopsUnderFloor(INDArray logits, long seqPos, SamplingConfig sampling,
                                                int generatedSoFar, Set<Integer> stopTokenIds) {
        if (sampling.getMinNewTokens() <= 0 || generatedSoFar >= sampling.getMinNewTokens()) {
            return;
        }
        for (int stopId : stopTokenIds) {
            logits.putScalar(new long[]{0, seqPos, stopId}, -1e9);
        }
    }

    /**
     * Stop tokens remain in tokenIds for protocol-aware consumers, but plain generated
     * text excludes trailing terminal tokens at the token boundary. This avoids every
     * caller re-parsing model-specific marker strings.
     */
    static int contentTokenLength(int[] tokenIds, Set<Integer> stopTokenIds) {
        if (tokenIds == null) {
            return 0;
        }
        Set<Integer> stops = stopTokenIds == null
                ? Collections.emptySet() : stopTokenIds;
        int length = tokenIds.length;
        while (length > 0 && stops.contains(tokenIds[length - 1])) {
            length--;
        }
        return length;
    }

    private String decodeGeneratedText(int[] tokenIds, Set<Integer> stopTokenIds) {
        int contentLength = contentTokenLength(tokenIds, stopTokenIds);
        int[] contentIds = contentLength == tokenIds.length
                ? tokenIds : Arrays.copyOf(tokenIds, contentLength);
        return tokenizer.decode(contentIds, false);
    }

    private GenerationResult buildResult(List<Integer> generatedTokens, int[] promptTokenIds,
                                          Set<Integer> stopTokenIds, long startTime, long firstTokenMs,
                                          double steadyStateTokPerSec) {
        long endTime = System.currentTimeMillis();
        long timeMs = endTime - startTime;
        int[] tokenIds = generatedTokens.stream().mapToInt(Integer::intValue).toArray();
        String text = decodeGeneratedText(tokenIds, stopTokenIds);
        boolean hitEos = tokenIds.length > 0 && stopTokenIds.contains(tokenIds[tokenIds.length - 1]);

        return GenerationResult.builder()
                .text(text)
                .tokenIds(tokenIds)
                .generatedTokenCount(tokenIds.length)
                .promptTokenCount(promptTokenIds.length)
                .totalTokenCount(promptTokenIds.length + tokenIds.length)
                .finishReason(hitEos ? GenerationResult.FinishReason.EOS : GenerationResult.FinishReason.MAX_TOKENS)
                .generationTimeMs(timeMs)
                .firstTokenLatencyMs(firstTokenMs)
                .tokensPerSecond(timeMs > 0 ? (tokenIds.length * 1000.0 / timeMs) : 0)
                .steadyStateTokensPerSecond(steadyStateTokPerSec)
                .build();
    }

    private static void closeKvOutputs(Map<String, INDArray> outputs, ModelIOConfig.KVCacheNames kvNames,
                                        String logitsName) {
        for (String name : kvNames.keyNames) {
            INDArray arr = outputs.get(name);
            if (arr != null) arr.close();
        }
        for (String name : kvNames.valueNames) {
            INDArray arr = outputs.get(name);
            if (arr != null) arr.close();
        }
    }

    /**
     * Generate text from pre-built embeddings (for VLM or custom embedding pipelines).
     *
     * <p>This is the entry point for vision-language models where embeddings
     * are constructed by merging vision encoder output with text token embeddings
     * before calling the decoder.</p>
     *
     * @param prefillEmbeddings merged embeddings [1, seqLen, hiddenSize]
     * @param promptTokenIds the prompt token IDs (used for input_ids at step 0)
     * @return generation result
     */
    public GenerationResult generate(INDArray prefillEmbeddings, int[] promptTokenIds) {
        return generate(prefillEmbeddings, promptTokenIds, config.getMaxNewTokens());
    }

    /**
     * Generate text from pre-built embeddings with a specified max token count.
     *
     * <p>Runs the native {@code autoregressive_decode} C++ op.</p>
     *
     * @param prefillEmbeddings merged embeddings [1, seqLen, hiddenSize]
     * @param promptTokenIds the prompt token IDs (used for input_ids at step 0)
     * @param maxNewTokens maximum number of tokens to generate
     * @return generation result
     */
    public GenerationResult generate(INDArray prefillEmbeddings, int[] promptTokenIds, int maxNewTokens) {
        return generate(prefillEmbeddings, promptTokenIds, maxNewTokens, null);
    }

    /**
     * Generate text from pre-built embeddings with per-call decode options.
     *
     * <p>Runs the native {@code autoregressive_decode} C++ op. The {@link DecodeOptions}
     * parameter is accepted for API compatibility; per-call overrides that require
     * Java-side control are not supported by the native op.</p>
     *
     * @param prefillEmbeddings merged embeddings [1, seqLen, hiddenSize]
     * @param promptTokenIds the prompt token IDs (used for input_ids at step 0)
     * @param maxNewTokens maximum number of tokens to generate
     * @param options per-call decode options (nullable)
     * @return generation result
     */
    public GenerationResult generate(INDArray prefillEmbeddings, int[] promptTokenIds,
                                     int maxNewTokens, DecodeOptions options) {
        int restoreDevice = switchToDecoderDevice("embedding-generation");
        // Suppress cross-device routing for the entire generation (same rationale
        // as generate(String, int): pool-reserved memory causes false OOM routing).
        OpaqueDataBuffer.suppressCrossDeviceRouting(true);
        try {
            // Use the native AutoregressiveDecode C++ op for maximum performance.
            // The native path does: prefill → warmup → freeze → C++ decode loop with
            // zero JNI round-trips. Bugs that previously caused EOS-on-step-2 have been
            // fixed: causal mask pre-unmask, segment splitting for value-key ops,
            // mixed-type GEMV handling.
            return generateNative(prefillEmbeddings, promptTokenIds, maxNewTokens, options);
        } finally {
            OpaqueDataBuffer.suppressCrossDeviceRouting(false);
            restoreDevice(restoreDevice, "embedding-generation");
        }
    }

    private int switchToDecoderDevice(String reason) {
        int decoderDevice = resolveDecoderDeviceId();
        if (decoderDevice < 0) {
            return -1;
        }

        DeviceMemoryManager deviceMgr = DeviceMemoryManager.getInstance();
        int currentDevice = deviceMgr.getCurrentDeviceId();
        if (currentDevice != decoderDevice) {
            deviceMgr.switchDevice(decoderDevice, "GenerationPipeline", reason);
            return currentDevice;
        }
        return -1;
    }

    private void restoreDevice(int restoreDevice, String reason) {
        if (restoreDevice >= 0) {
            DeviceMemoryManager.getInstance().switchDevice(
                    restoreDevice, "GenerationPipeline", reason + "-restore");
        }
    }

    private int resolveDecoderDeviceId() {
        int device = firstArrayDevice(decoder.getVariablesArrays());
        if (device >= 0) {
            return device;
        }
        return firstArrayDevice(decoder.getConstantArrays());
    }

    private static int firstArrayDevice(ArrayHolder holder) {
        if (holder == null || holder.arrayNames() == null) {
            return -1;
        }

        for (String name : holder.arrayNames()) {
            try {
                INDArray array = holder.getArray(name);
                if (array != null && !array.wasClosed() && array.hasDeviceBuffer()) {
                    int device = array.getNativeDeviceId();
                    if (device >= 0) {
                        return device;
                    }
                }
            } catch (Exception e) {
                log.debug("Unable to resolve array device for '{}': {}", name, e.getMessage());
            }
        }
        return -1;
    }

    /**
     * Generate text using the native {@code AutoregressiveDecode} C++ op.
     *
     * <p>VLM generation with native decode loop. Steps:
     * <ol>
     *   <li>Execute a prefill step via decoder.output() to produce initial KV caches
     *       and trigger DSP compilation.</li>
     *   <li>Pad KV caches to static size, sample first token.</li>
     *   <li>Build decode-step arrays, execute a warmup decode step (compiles DSP
     *       plan for decode shapes), sample second token, scatter KV.</li>
     *   <li>Freeze shapes, resolve plan handle and ext input indices.</li>
     *   <li>Execute AutoregressiveDecode native op for all remaining tokens
     *       entirely in C++ with zero JNI round-trips.</li>
     * </ol>
     * </p>
     */
    private GenerationResult generateNative(INDArray prefillEmbeddings, int[] promptTokenIds, int maxNewTokens) {
        return generateNative(prefillEmbeddings, promptTokenIds, maxNewTokens, null);
    }

    private GenerationResult generateNative(INDArray prefillEmbeddings, int[] promptTokenIds,
                                            int maxNewTokens, DecodeOptions options) {
        long startTime = System.currentTimeMillis();

        int maxPrefill = config.getMaxPrefillLength();
        boolean fixedBuffers = maxPrefill > 0;
        // Track whether we're reusing a frozen plan from a previous call.
        // When true, ext input arrays are retrieved from the executor and reused
        // (new data written via assign()) instead of allocating fresh arrays.
        // This preserves device pointer stability for CUDA graph replay.
        boolean reusingFrozenPlan = false;
        INDArray[] frozenExtInputSnapshot = null;
        String[] frozenExtInputKeys = null;

        if (fixedBuffers) {
            // Fixed-size buffers: skip DSP reset — reuse the frozen plan.
            InferenceSession existingSession = decoder.getOrCreateSession();
            if (existingSession != null) {
                DynamicShapePlanExecutor existingExecutor = existingSession.getDynamicShapePlanExecutor();
                if (existingExecutor != null && existingExecutor.isShapesFrozen()) {
                    log.info("[Lifecycle] Reusing frozen DSP plan for native generation (fixedBuffers=true, maxPrefill={})", maxPrefill);
                    // Keep both the plan and its node-output buffers alive. The captured graph tracks
                    // these external-input addresses, so overwrite the retained arrays in place below.
                    // clearNodeOutputsOnly() must not be used here: despite its name it closes the
                    // session DynamicShapePlan and forces a new native borrower on the next dispatch.
                    frozenExtInputSnapshot = existingExecutor.getExternalInputsSnapshot();
                    DynamicShapePlan frozenPlan = existingExecutor.getCurrentPlan();
                    frozenExtInputKeys = frozenPlan != null ? frozenPlan.getExternalInputKeys() : null;
                    reusingFrozenPlan = frozenExtInputSnapshot != null && frozenExtInputSnapshot.length > 0
                            && frozenExtInputKeys != null
                            && frozenExtInputKeys.length == frozenExtInputSnapshot.length;
                    if (reusingFrozenPlan) {
                        log.info("[Lifecycle] Captured {} ext input arrays for pointer-stable reuse",
                                frozenExtInputSnapshot.length);
                    }
                }
            }
        } else {
            // If a previous generation left the DSP executor frozen, reset the decoder
            // session completely.  Reusing a frozen plan for a new generation causes
            // PLAN_CACHE_BUG (stale plan handle) and LIFECYCLE_VALIDATION_FAILED
            // (stale output buffer snapshots).  Plan phases are strictly linear and
            // cannot be unwound; the only correct reset path is session destruction.
            // We also clear the native plan cache because cached CUDA graphs capture
            // device pointers that become stale when the session's output buffers are
            // freed. Reusing a cached graph with stale pointers produces silent
            // divergence (wrong tokens) or crashes.
            //
            // Only reset when a session actually exists — unconditional reset would
            // destroy externally-compiled plans (e.g., benchmark compile phase).
            InferenceSession existingSession = decoder.getOrCreateSession();
            if (existingSession != null) {
                DynamicShapePlanExecutor existingExecutor = existingSession.getDynamicShapePlanExecutor();
                if (existingExecutor != null && existingExecutor.isShapesFrozen()) {
                    log.info("[Lifecycle] Resetting frozen DSP executor for new generation");
                    decoder.resetSession();
                    decoder.clearDynamicShapePlanCache();
                    // Do not carry deferred plan frees into the replacement generation.
                    SameDiffMemoryUtils.trimAllDevicePools();
                }
            }
        }

        // Resolve decode policy and sampling config. Greedy/sample use the current scalar native op;
        // speculative/contrastive/beam require the ADR 0106 masked multi-position native substrate.
        SamplingConfig sampling = activeDecodeSampling();

        int eosTokenId = resolveEosTokenId(sampling);
        Set<Integer> stopTokenIds = buildStopTokenIds(eosTokenId);

        DecodePolicy decodePolicy = activeDecodePolicy();
        requireNativeSubstrateAvailable(decodePolicy, sampling);
        Random rng = sampling.getSeed() != null ? new Random(sampling.getSeed()) : new Random();

        // Discover KV output names from decoder
        ModelIOConfig.KVCacheNames kvNames = ioConfig.getKvCacheNames();
        if (kvNames == null) {
            kvNames = ModelIOConfig.findKVCacheOutputNames(decoder);
        }
        int numKvPairs = kvNames.keyNames.size();

        // Build all output names: logits + present KV
        List<String> allOutputNames = new ArrayList<>();
        allOutputNames.add(ioConfig.getLogitsOutputName());
        allOutputNames.addAll(kvNames.keyNames);
        allOutputNames.addAll(kvNames.valueNames);

        int actualPrefillLen = promptTokenIds.length;

        // When fixedBuffers is enabled, pad/truncate prompt to maxPrefillLength
        int prefillSeqLen;
        int[] effectiveTokenIds;
        INDArray effectiveEmbeddings;
        if (fixedBuffers) {
            prefillSeqLen = maxPrefill;
            if (actualPrefillLen > maxPrefill) {
                log.warn("[Native] Prompt length {} exceeds maxPrefillLength {} — truncating",
                        actualPrefillLen, maxPrefill);
                effectiveTokenIds = new int[maxPrefill];
                System.arraycopy(promptTokenIds, actualPrefillLen - maxPrefill,
                        effectiveTokenIds, 0, maxPrefill);
                // Slice embeddings to last maxPrefill tokens
                effectiveEmbeddings = prefillEmbeddings.get(NDArrayIndex.all(),
                        NDArrayIndex.interval(actualPrefillLen - maxPrefill, actualPrefillLen),
                        NDArrayIndex.all());
                actualPrefillLen = maxPrefill;
            } else if (actualPrefillLen < maxPrefill) {
                // Right-pad tokens and embeddings
                effectiveTokenIds = new int[maxPrefill];
                System.arraycopy(promptTokenIds, 0, effectiveTokenIds, 0, actualPrefillLen);
                // Pad embeddings with zeros
                long[] embShape = prefillEmbeddings.shape();
                effectiveEmbeddings = Nd4j.zeros(prefillEmbeddings.dataType(),
                        embShape[0], maxPrefill, embShape[2]);
                effectiveEmbeddings.get(NDArrayIndex.all(),
                        NDArrayIndex.interval(0, actualPrefillLen),
                        NDArrayIndex.all()).assign(prefillEmbeddings);
                log.info("[Native] Padded prompt from {} to {} tokens", actualPrefillLen, maxPrefill);
            } else {
                effectiveTokenIds = promptTokenIds;
                effectiveEmbeddings = prefillEmbeddings;
            }
        } else {
            prefillSeqLen = actualPrefillLen;
            effectiveTokenIds = promptTokenIds;
            effectiveEmbeddings = prefillEmbeddings;
        }

        int kvCap = config.getMaxKvCacheLength();
        if (kvCap > 0 && prefillSeqLen >= kvCap) {
            throw new IllegalArgumentException(
                    "Prompt token count "
                            + prefillSeqLen
                            + " leaves no room for generation within "
                            + "maxKvCacheLength="
                            + kvCap
            );
        }

        long maxKvLen =
                prefillSeqLen + maxNewTokens;

        // Cap to configured KV cache length to keep buffer shapes stable across
        // calls with different maxNewTokens — avoids plan recompilation.
        if (kvCap > 0 && maxKvLen > kvCap) {
            maxNewTokens = kvCap - prefillSeqLen;
            maxKvLen = kvCap;
        }
        boolean dspActive = decoder.isDspAutoCompileEnabled();
        List<String> decoderInputNames = decoder.inputs();
        Map<String, INDArray> reusableInputs = new HashMap<>();

        // ══════════════════════════════════════════════════════════════════════
        // STEP 1: Prefill — run decoder with full prompt to get initial KV caches.
        // DSP stays enabled throughout — never disable it. The shape-keyed plan
        // cache handles prefill vs decode shape differences automatically.
        // ══════════════════════════════════════════════════════════════════════

        INDArray currentInputIds = Nd4j.createFromArray(effectiveTokenIds)
                .reshape(1, prefillSeqLen).castTo(DataType.INT64);

        // Encoder-decoder models (Whisper): the encoder output feeds the decoder's
        // cross-attention input. Supplied per-call via DecodeOptions; constant across
        // decode steps, so prefill-time binding persists through the frozen plan's
        // external-input table for the native decode loop.
        INDArray encoderOutputs = options != null ? options.getEncoderOutputs() : null;
        INDArray encoderAttentionMask = options != null ? options.getEncoderAttentionMask() : null;

        Map<String, INDArray> prefillInputMap = DecoderInputBuilder.buildDecoderInputMap(
                ioConfig, decoderInputNames, decoder,
                effectiveEmbeddings, currentInputIds,
                0, prefillSeqLen,
                null, maxKvLen, 0,
                false, hiddenSize,
                reusableInputs, dspActive,
                encoderOutputs, encoderAttentionMask,
                actualPrefillLen);

        Map<String, INDArray> prefillOutputs = decoder.output(
                prefillInputMap, allOutputNames.toArray(new String[0]));

        // ── Merged-If decoders (use_cache_branch): dynamic-KV decode ────────────
        // The static-pad KV scheme below is semantically INCOMPATIBLE with these
        // graphs: their with-past branch derives the position from shape(past)[2],
        // so padding to maxKvLen indexes past the position-embedding table (empty
        // slice → garbage/crash). These graphs also run on the interpreted session
        // (If frames — no native plan), so static replay buys nothing. Decode with
        // TRUE-LENGTH pasts: feed each step's presents back verbatim.
        String mergedUcbName = ioConfig.getUseCacheBranchName();
        if (mergedUcbName != null && decoder.hasVariable(mergedUcbName)) {
            Random mergedRng = sampling.getSeed() != null ? new Random(sampling.getSeed()) : new Random();
            return dynamicMergedIfDecodeLoop(prefillOutputs, kvNames, mergedUcbName,
                    sampling, mergedRng, stopTokenIds, maxNewTokens, actualPrefillLen,
                    encoderOutputs, promptTokenIds, startTime, allOutputNames);
        }

        // ══════════════════════════════════════════════════════════════════════
        // STEP 2: Pad KV caches to static size and prepare decode-step state
        // ══════════════════════════════════════════════════════════════════════
        // Guard prefill KV close: when DSP is active, the C++ slotArrayCache_ still
        // holds raw NDArray* pointers to these outputs. Java close() deletes the C++
        // NDArray, leaving dangling pointers → use-after-free on next execution step.
        // Matches StaticKvCacheDecodeLoop lines 633-648.
        InferenceSession prefillSession = decoder.getOrCreateSession();
        boolean prefillDspActive = prefillSession.getDynamicShapePlanExecutor() != null
                && prefillSession.getDynamicShapePlanExecutor().getCurrentPlan() != null;

        Map<String, INDArray> staticKvBuffers = new LinkedHashMap<>();
        for (String keyName : kvNames.keyNames) {
            INDArray presentKv = prefillOutputs.get(keyName);
            if (presentKv == null) {
                StringBuilder state = new StringBuilder();
                for (Map.Entry<String, INDArray> e : prefillOutputs.entrySet()) {
                    state.append(' ').append(e.getKey()).append('=')
                            .append(e.getValue() == null ? "NULL"
                                    : (e.getValue().isEmpty() ? "EMPTY" : Arrays.toString(e.getValue().shape())));
                }
                throw new IllegalStateException("Prefill produced NULL for present KV output '" + keyName
                        + "'. Full prefill output state:" + state);
            }
            String inputName = ioConfig.presentToInputName(keyName);
            INDArray padded;
            if (reusingFrozenPlan) {
                // Reuse the existing ext input array to preserve device pointers.
                int extIdx = resolveExtInputIdx(frozenExtInputKeys, inputName);
                padded = (extIdx >= 0 && extIdx < frozenExtInputSnapshot.length)
                        ? frozenExtInputSnapshot[extIdx] : null;
                if (padded != null) {
                    // Zero the buffer, then copy prefill KV data into the reused array.
                    padded.assign(0);
                    long seqLen = presentKv.shape()[2];
                    padded.get(NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.interval(0, seqLen), NDArrayIndex.all()).assign(presentKv);
                } else {
                    // Fallback: ext input not found — allocate fresh (breaks pointer stability)
                    log.warn("[Lifecycle] KV ext input '{}' not found in snapshot (extIdx={}), allocating fresh", inputName, extIdx);
                    padded = padKvToStaticSize(presentKv, maxKvLen);
                    padded.setCloseable(false);
                }
            } else {
                padded = padKvToStaticSize(presentKv, maxKvLen);
                padded.setCloseable(false);
            }
            staticKvBuffers.put(inputName, padded);
            if (!prefillDspActive) {
                presentKv.close();
            }
        }
        for (String valName : kvNames.valueNames) {
            INDArray presentKv = prefillOutputs.get(valName);
            String inputName = ioConfig.presentToInputName(valName);
            INDArray padded;
            if (reusingFrozenPlan) {
                int extIdx = resolveExtInputIdx(frozenExtInputKeys, inputName);
                padded = (extIdx >= 0 && extIdx < frozenExtInputSnapshot.length)
                        ? frozenExtInputSnapshot[extIdx] : null;
                if (padded != null) {
                    padded.assign(0);
                    long seqLen = presentKv.shape()[2];
                    padded.get(NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.interval(0, seqLen), NDArrayIndex.all()).assign(presentKv);
                } else {
                    log.warn("[Lifecycle] KV ext input '{}' not found in snapshot (extIdx={}), allocating fresh", inputName, extIdx);
                    padded = padKvToStaticSize(presentKv, maxKvLen);
                    padded.setCloseable(false);
                }
            } else {
                padded = padKvToStaticSize(presentKv, maxKvLen);
                padded.setCloseable(false);
            }
            staticKvBuffers.put(inputName, padded);
            if (!prefillDspActive) {
                presentKv.close();
            }
        }
        Nd4j.getExecutioner().commit();

        // Sample first token from prefill logits using the active SamplingConfig.
        // When fixedBuffers is active, sample from actualPrefillLen-1, not the end.
        INDArray prefillLogits = prefillOutputs.get(ioConfig.getLogitsOutputName());
        // Sample the LAST-token logits. The decoder may export full/padded logits
        // [1, N>=actualPrefillLen, vocab] (real last token at actualPrefillLen-1, before any
        // fixedBuffers padding) OR last-position-only [1, 1, vocab] (optimized use_cache ONNX
        // exports — the only valid index is 0). Clamp to the actual logits seq dim: without it,
        // point(actualPrefillLen-1) on a size-1 dim is an UNCHECKED OOB read (BaseNDArray.get
        // has no bounds check) → reads position-0-adjacent GPU memory → garbage first-token
        // logits → token 11126 "User", making the model ignore the image.
        int logitsSamplePos = Math.min(actualPrefillLen - 1, (int) prefillLogits.shape()[1] - 1);
        List<Integer> warmupGeneratedTokens = new ArrayList<>();

        // Build constraint masker if a ConstraintConfig was set on the SamplingConfig.
        // The masker enforces structural validity at each token selection step by masking
        // disallowed logits before the sampling pipeline runs. When active, all sampleToken
        // calls below (warmup + constrained Java loop) route through the masker-aware overload.
        // The native AutoregressiveDecode op is bypassed in favour of the Java decode loop when
        // the masker is non-null (the native loop cannot call back into Java per step).
        ConstraintMasker constraintMasker = null;
        if (sampling.hasConstraint()) {
            ConstraintConfig cc = sampling.getConstraintConfig();
            constraintMasker = new ConstraintMasker(cc.buildConstraint(), cc.getEvalTopK());
            log.info("[Constraint] Constrained decoding active: type={} evalTopK={}", cc.getType(), cc.getEvalTopK());
        }

        suppressStopsUnderFloor(prefillLogits, logitsSamplePos, sampling, warmupGeneratedTokens.size(), stopTokenIds);
        int firstTokenId = sampleToken(prefillLogits, logitsSamplePos, sampling, warmupGeneratedTokens, rng,
                constraintMasker, tokenizer);
        warmupGeneratedTokens.add(firstTokenId);
        log.info("[Native] Prefill firstTokenId={} logitsShape={} samplePos={} (actualPrefillLen={})",
                firstTokenId, Arrays.toString(prefillLogits.shape()), logitsSamplePos, actualPrefillLen);
        prefillLogits.close();

        // ══════════════════════════════════════════════════════════════════════
        // STEP 3: Build decode-step arrays ONCE, then execute a warmup step.
        //
        // These arrays have the EXACT shapes the C++ native loop will use.
        // The warmup compiles the DSP plan for these shapes, and the native
        // loop reuses the same plan handle — no shape mismatch, no plan swap.
        // ══════════════════════════════════════════════════════════════════════
        // Mask dimension: maxKvLen + 1 (past KV positions + current query position).
        // The model's internal attention ops produce [1,1,1,maxKvLen+1] tensors and
        // add them to the causal mask — shapes MUST match.
        long totalSeqLen = maxKvLen + 1;

        INDArray decodeEmbeddings;
        INDArray decodeInputIds;
        INDArray decodeCausalMask;
        INDArray decodeAttentionMask;
        INDArray decodePosIds;

        if (reusingFrozenPlan) {
            // ── Reuse existing ext input arrays for pointer stability ──────────
            // The frozen plan tracks device pointer addresses. Fresh allocations
            // give new pointers, breaking pointer stability and preventing CUDA
            // graph capture/replay. Retrieve the existing arrays and write new
            // values into them via assign()/putScalar().
            String embedsName = ioConfig.getInputEmbeddingsName();
            int embExtIdx = (embedsName != null && decoder.hasVariable(embedsName))
                    ? resolveExtInputIdx(frozenExtInputKeys, embedsName) : -1;
            int idsExtIdx = resolveExtInputIdx(frozenExtInputKeys, ioConfig.getInputIdsName());
            int causalExtIdx = ioConfig.getCausalMaskName() != null
                    ? resolveExtInputIdx(frozenExtInputKeys, ioConfig.getCausalMaskName()) : -1;
            int maskExtIdx = resolveExtInputIdx(frozenExtInputKeys, ioConfig.getAttentionMaskName());
            int posExtIdx = resolveExtInputIdx(frozenExtInputKeys, ioConfig.getPositionIdsName());

            // Embeddings: write first token embedding into reused array
            decodeEmbeddings = (embExtIdx >= 0 && embExtIdx < frozenExtInputSnapshot.length)
                    ? frozenExtInputSnapshot[embExtIdx] : null;
            if (decodeEmbeddings != null) {
                INDArray firstEmbed = embeddingTable.getRow(firstTokenId).reshape(1, 1, hiddenSize);
                decodeEmbeddings.assign(firstEmbed);
                firstEmbed.close();
            } else {
                decodeEmbeddings = embeddingTable.getRow(firstTokenId).reshape(1, 1, hiddenSize).dup();
            }

            // Input IDs
            decodeInputIds = (idsExtIdx >= 0 && idsExtIdx < frozenExtInputSnapshot.length)
                    ? frozenExtInputSnapshot[idsExtIdx] : null;
            if (decodeInputIds != null) {
                decodeInputIds.putScalar(new long[]{0, 0}, firstTokenId);
            } else {
                decodeInputIds = Nd4j.createFromArray(new int[]{firstTokenId})
                        .reshape(1, 1).castTo(DataType.INT64);
            }

            // Causal mask: rewrite values into existing array
            decodeCausalMask = (causalExtIdx >= 0 && causalExtIdx < frozenExtInputSnapshot.length)
                    ? frozenExtInputSnapshot[causalExtIdx] : null;
            if (decodeCausalMask != null) {
                float maskFill = ModelIOConfig.MASK_FILL;
                for (int i = 0; i < (int) totalSeqLen; i++) {
                    decodeCausalMask.putScalar(new long[]{0, 0, 0, i},
                            (i <= actualPrefillLen) ? 0.0f : maskFill);
                }
            } else {
                float[] causalData = new float[(int) totalSeqLen];
                float maskFill = ModelIOConfig.MASK_FILL;
                for (int i = 0; i < (int) totalSeqLen; i++) {
                    causalData[i] = (i <= actualPrefillLen) ? 0.0f : maskFill;
                }
                decodeCausalMask = Nd4j.createFromArray(causalData).reshape(1, 1, 1, totalSeqLen);
            }

            // Attention mask: rewrite values into existing array
            decodeAttentionMask = (maskExtIdx >= 0 && maskExtIdx < frozenExtInputSnapshot.length)
                    ? frozenExtInputSnapshot[maskExtIdx] : null;
            if (decodeAttentionMask != null) {
                decodeAttentionMask.assign(0);
                for (int i = 0; i < actualPrefillLen; i++) {
                    decodeAttentionMask.putScalar(new long[]{0, i}, 1);
                }
                decodeAttentionMask.putScalar(new long[]{0, totalSeqLen - 1}, 1);
            } else {
                long[] maskData = new long[(int) totalSeqLen];
                for (int i = 0; i < actualPrefillLen; i++) maskData[i] = 1;
                maskData[(int) (totalSeqLen - 1)] = 1;
                decodeAttentionMask = Nd4j.createFromArray(maskData).reshape(1, totalSeqLen);
            }

            // Position IDs
            decodePosIds = (posExtIdx >= 0 && posExtIdx < frozenExtInputSnapshot.length)
                    ? frozenExtInputSnapshot[posExtIdx] : null;
            if (decodePosIds != null) {
                decodePosIds.putScalar(new long[]{0, 0}, actualPrefillLen);
            } else {
                decodePosIds = Nd4j.createFromArray(new long[]{actualPrefillLen}).reshape(1, 1);
            }

            log.info("[Lifecycle] Reused {} ext input arrays for pointer-stable decode step",
                    frozenExtInputSnapshot.length);
        } else {
            // ── First call: allocate fresh decode arrays ───────────────────────
            // dup() is MANDATORY: getRow().reshape() returns a VIEW into embeddingTable.
            // assign() at line below writes into this buffer — without dup(), that corrupts
            // the persistent weight matrix, making subsequent runs non-deterministic.
            decodeEmbeddings = embeddingTable.getRow(firstTokenId).reshape(1, 1, hiddenSize).dup();
            decodeInputIds = Nd4j.createFromArray(new int[]{firstTokenId})
                    .reshape(1, 1).castTo(DataType.INT64);

            // Causal mask: [1, 1, 1, totalSeqLen] FLOAT — MASK_FILL for unfilled positions, 0.0 for filled.
            float[] causalData = new float[(int) totalSeqLen];
            float maskFill = ModelIOConfig.MASK_FILL;
            for (int i = 0; i < (int) totalSeqLen; i++) {
                causalData[i] = (i <= actualPrefillLen) ? 0.0f : maskFill;
            }
            decodeCausalMask = Nd4j.createFromArray(causalData).reshape(1, 1, 1, totalSeqLen);

            // Attention mask: [1, totalSeqLen] LONG (0/1 values, updated per step by C++ kernel).
            long[] maskData = new long[(int) totalSeqLen];
            for (int i = 0; i < actualPrefillLen; i++) maskData[i] = 1;
            maskData[(int) (totalSeqLen - 1)] = 1;
            decodeAttentionMask = Nd4j.createFromArray(maskData).reshape(1, totalSeqLen);

            // Position IDs: [1, 1] INT64 — first decode position is after real tokens
            decodePosIds = Nd4j.createFromArray(new long[]{actualPrefillLen}).reshape(1, 1);
        }

        // ── attn_mask_reformat override ──────────────────────────────────────
        // The model's internal attn_mask_reformat subgraph is correct for single-
        // token decode (seqLen=1). No override needed. Matches reference benchmark
        // at 48 tok/s (commit 11005b4ae6, needsAttnOverride=false).
        String attnReformatNode = ioConfig.getAttnMaskReformatOutput();
        INDArray decodeAttnMaskReformat = null;
        boolean needsAttnOverride = false;

        // Build decode input map directly from these arrays — single source of truth.
        Map<String, INDArray> decodeInputMap = new HashMap<>();
        String embedsName = ioConfig.getInputEmbeddingsName();
        if (embedsName != null && decoder.hasVariable(embedsName)) decodeInputMap.put(embedsName, decodeEmbeddings);
        String idsName = ioConfig.getInputIdsName();
        if (idsName != null) decodeInputMap.put(idsName, decodeInputIds);
        String maskName = ioConfig.getAttentionMaskName();
        if (maskName != null && decoder.hasVariable(maskName))
            decodeInputMap.put(maskName, decodeAttentionMask);
        String causalName = ioConfig.getCausalMaskName();
        if (causalName != null && decoder.hasVariable(causalName))
            decodeInputMap.put(causalName, decodeCausalMask);
        String posName = ioConfig.getPositionIdsName();
        if (posName != null && decoder.hasVariable(posName))
            decodeInputMap.put(posName, decodePosIds);
        if (decodeAttnMaskReformat != null && decoder.hasVariable(attnReformatNode)) {
            decodeInputMap.put(attnReformatNode, decodeAttnMaskReformat);
        }
        // Static KV buffers
        for (Map.Entry<String, INDArray> entry : staticKvBuffers.entrySet()) {
            if (decoder.hasVariable(entry.getKey())) {
                decodeInputMap.put(entry.getKey(), entry.getValue());
            }
        }
        // Encoder-decoder cross-attention input (Whisper): same constant tensor as
        // prefill — every decode step's cross-attention reads the full encoder output.
        String encName = ioConfig.getEncoderHiddenStatesName();
        if (encoderOutputs != null && encName != null && decoder.hasVariable(encName)) {
            decodeInputMap.put(encName, encoderOutputs);
        }
        String encMaskName = ioConfig.getEncoderAttentionMaskName();
        if (encoderAttentionMask != null && encMaskName != null && decoder.hasVariable(encMaskName)) {
            decodeInputMap.put(encMaskName, encoderAttentionMask);
        }
        // Merged-decoder branch selector: decode steps carry past KV — with-past branch.
        String useCacheBranchName = ioConfig.getUseCacheBranchName();
        if (useCacheBranchName != null && decoder.hasVariable(useCacheBranchName)) {
            decodeInputMap.put(useCacheBranchName, Nd4j.scalar(true));
        }
        // Associate internal model inputs (non-placeholder variables)
        DecoderInputBuilder.associateInternalModelInputs(ioConfig,
                new ArrayList<>(decodeInputMap.keySet()), decoder, decodeInputMap);

        // ── Ensure all decode inputs are device-coherent (CUDA) ─────
        // Arrays constructed via Nd4j.createFromArray() are host-primary. The C++
        // executor handles H2D sync automatically for non-frozen paths. Commit any
        // pending CUDA ops from KV padding before proceeding.
        Nd4j.getExecutioner().commit();

        // ── Warmup decode step: compiles the DSP plan for decode shapes ──────
        // The shape-keyed plan cache automatically creates a new plan when the
        // decode shapes (seqLen=1) differ from prefill shapes (seqLen=N).
        // Do NOT clear the plan cache here — clearNodeOutputsOnly() destroys
        // intermediate node outputs (including attn_mask_reformat) that the
        // session needs to recompute the graph correctly. The reference
        // (11005b4ae6, 48 tok/s) goes directly from associateInternalModelInputs
        // to decoder.output() with no plan clearing.
        Map<String, INDArray> decodeOutputs = decoder.output(
                decodeInputMap, allOutputNames.toArray(new String[0]));

        // Sample second token using the active SamplingConfig.
        INDArray decodeLogits = decodeOutputs.get(ioConfig.getLogitsOutputName());
        suppressStopsUnderFloor(decodeLogits, 0, sampling, warmupGeneratedTokens.size(), stopTokenIds);
        int secondTokenId = sampleToken(decodeLogits, 0, sampling, warmupGeneratedTokens, rng,
                constraintMasker, tokenizer);
        if (maxNewTokens >= 2 && !stopTokenIds.contains(firstTokenId)) {
            warmupGeneratedTokens.add(secondTokenId);
        }
        decodeLogits.close();

        // Scatter decode step KV into static buffers (position = prefillSeqLen)
        for (String keyName : kvNames.keyNames) {
            INDArray presentKv = decodeOutputs.get(keyName);
            INDArray staticBuf = staticKvBuffers.get(ioConfig.presentToInputName(keyName));
            if (presentKv != null && staticBuf != null) {
                scatterKvToStatic(presentKv, staticBuf, actualPrefillLen);
            }
        }
        for (String valName : kvNames.valueNames) {
            INDArray presentKv = decodeOutputs.get(valName);
            INDArray staticBuf = staticKvBuffers.get(ioConfig.presentToInputName(valName));
            if (presentKv != null && staticBuf != null) {
                scatterKvToStatic(presentKv, staticBuf, actualPrefillLen);
            }
        }

        // ══════════════════════════════════════════════════════════════════════
        // STEP 4: Get native plan handle and resolve external input indices
        // ══════════════════════════════════════════════════════════════════════
        InferenceSession session = decoder.getOrCreateSession();
        DynamicShapePlanExecutor executor = session != null ? session.getDynamicShapePlanExecutor() : null;
        Pointer planHandle = executor != null ? executor.getNativePlanHandle() : null;

        // If plan not available, fall back to returning just the two tokens we already have
        if (planHandle == null || planHandle.isNull()) {
            log.warn("Native plan handle not available — returning partial result from warmup steps");
            currentInputIds.close();
            decodeEmbeddings.close();
            decodeInputIds.close();
            decodeCausalMask.close();
            decodeAttentionMask.close();
            decodePosIds.close();
            if (decodeAttnMaskReformat != null) decodeAttnMaskReformat.close();

            List<Integer> tokens = new ArrayList<>();
            tokens.add(firstTokenId);
            if (maxNewTokens >= 2 && !stopTokenIds.contains(firstTokenId)) {
                tokens.add(secondTokenId);
            }

            int[] tokenIds = tokens.stream().mapToInt(Integer::intValue).toArray();
            String text = decodeGeneratedText(tokenIds, stopTokenIds);
            long endTime = System.currentTimeMillis();
            long timeMs = endTime - startTime;

            for (INDArray kv : staticKvBuffers.values()) {
                kv.setCloseable(true);
                kv.close();
            }

            return GenerationResult.builder()
                    .text(text)
                    .tokenIds(tokenIds)
                    .generatedTokenCount(tokenIds.length)
                    .promptTokenCount(prefillSeqLen)
                    .totalTokenCount(prefillSeqLen + tokenIds.length)
                    .finishReason(stopTokenIds.contains(tokenIds[tokenIds.length-1])
                            ? GenerationResult.FinishReason.EOS : GenerationResult.FinishReason.MAX_TOKENS)
                    .generationTimeMs(timeMs)
                    .tokensPerSecond(timeMs > 0 ? (tokenIds.length * 1000.0 / timeMs) : 0)
                    .steadyStateTokensPerSecond(0)
                    .build();
        }

        // Freeze shapes after warmup decode — the plan has seen decode-shape data
        // flow through once and compiled for it. Freezing enables CUDA graph capture
        // and stable Triton cache keys.
        if (executor != null && executor.getCurrentPlan() != null) {
            executor.setShapesFrozen(true);
            log.info("[Perf] Shapes frozen after warmup decode (planPhase={} pointersStable={})",
                    executor.getPlanPhase(), executor.arePointersStable());
            if ("true".equalsIgnoreCase(System.getProperty(ND4JSystemProperties.VLM_BENCHMARK_OP_TIMING, "false"))) {
                executor.setExecutionTimingEnabled(true);
                log.info("[Perf] Decoder execution timing enabled");
            }
        }

        // Resolve ext input indices by name
        int embeddingsExtIdx = decoder.hasVariable(ioConfig.getInputEmbeddingsName())
                ? resolveExtInputIdx(executor, ioConfig.getInputEmbeddingsName()) : -1;
        int maskExtIdx = resolveExtInputIdx(executor, ioConfig.getAttentionMaskName());
        int causalMaskExtIdx = ioConfig.getCausalMaskName() != null
                ? resolveExtInputIdx(executor, ioConfig.getCausalMaskName()) : -1;
        int posIdsExtIdx = resolveExtInputIdx(executor, ioConfig.getPositionIdsName());
        int inputIdsExtIdx = resolveExtInputIdx(executor, ioConfig.getInputIdsName());
        int attnMaskReformatExtIdx = (attnReformatNode != null && decoder.hasVariable(attnReformatNode))
                ? resolveExtInputIdx(executor, attnReformatNode) : -1;

        // Resolve logits output index
        int logitsOutputIdx = resolveOutputIdx(executor, ioConfig.getLogitsOutputName());

        // Resolve KV indices
        int[] kvInputExtIndices = new int[2 * numKvPairs];
        int[] kvOutputIndices = new int[2 * numKvPairs];
        int ki = 0;
        for (String keyName : kvNames.keyNames) {
            String inputName = ioConfig.presentToInputName(keyName);
            kvInputExtIndices[ki] = resolveExtInputIdx(executor, inputName);
            kvOutputIndices[ki] = resolveOutputIdx(executor, keyName);
            ki++;
        }
        for (String valName : kvNames.valueNames) {
            String inputName = ioConfig.presentToInputName(valName);
            kvInputExtIndices[ki] = resolveExtInputIdx(executor, inputName);
            kvOutputIndices[ki] = resolveOutputIdx(executor, valName);
            ki++;
        }

        // ══════════════════════════════════════════════════════════════════════
        // STEP 5: Execute the native AutoregressiveDecode op.
        //         Remaining tokens (maxNewTokens - 2 already generated) are
        //         generated entirely in C++ with zero JNI round-trips.
        //
        //         Reuse the decode-step arrays from STEP 3 — same shapes,
        //         same plan, no cache miss. Just update values for step 2.
        // ══════════════════════════════════════════════════════════════════════
        int remainingTokens = maxNewTokens - 2;  // 2 already generated (prefill + 1 decode step)

        // Update the decode arrays for the second token (values only, shapes unchanged).
        INDArray secondEmbed = embeddingTable.getRow(secondTokenId).reshape(1, 1, hiddenSize);
        decodeEmbeddings.assign(secondEmbed);
        secondEmbed.close();
        decodeInputIds.putScalar(new long[]{0, 0}, secondTokenId);
        // Attention mask: unmask the KV position written by the warmup step.
        decodeAttentionMask.putScalar(new long[]{0, actualPrefillLen}, 1);
        // Position IDs: advance to actualPrefillLen + 1
        decodePosIds.putScalar(new long[]{0, 0}, actualPrefillLen + 1);
        // Causal mask: unmask position written by the warmup step.
        decodeCausalMask.putScalar(new long[]{0, 0, 0, actualPrefillLen}, 0.0f);
        // attn_mask_reformat: same
        if (decodeAttnMaskReformat != null) {
            decodeAttnMaskReformat.putScalar(new long[]{0, 0, 0, actualPrefillLen}, 0.0f);
        }

        // Force H2D sync so device buffers are current BEFORE the native decode op.
        // .assign() / .putScalar() write to the host side, leaving isPrimaryActual=true
        // and isSpecialActual=false. When the C++ plan captures a CUDA graph on its
        // first execution, syncToSpecial sees isSpecialActual=false and records an H2D
        // memcpy node into the graph. On every subsequent replay, that captured H2D
        // overwrites the decode loop's kernel updates (embed lookup, mask update, etc.)
        // with stale capture-time host data → degenerate output.
        // By syncing here, isSpecialActual becomes true, the capture skips the H2D
        // recording, and graph replay sees the decode loop's fresh device-side updates.
        Nd4j.getExecutioner().commit();
        decodeEmbeddings.syncToDevice();
        decodeInputIds.syncToDevice();
        decodeAttentionMask.syncToDevice();
        decodePosIds.syncToDevice();
        decodeCausalMask.syncToDevice();
        if (decodeAttnMaskReformat != null) {
            decodeAttnMaskReformat.syncToDevice();
        }
        for (INDArray kvBuf : staticKvBuffers.values()) {
            kvBuf.syncToDevice();
        }

        // Collect static KV buffers as array
        INDArray[] staticKvArray = new INDArray[2 * numKvPairs];
        int idx = 0;
        for (String keyName : kvNames.keyNames) {
            staticKvArray[idx++] = staticKvBuffers.get(ioConfig.presentToInputName(keyName));
        }
        for (String valName : kvNames.valueNames) {
            staticKvArray[idx++] = staticKvBuffers.get(ioConfig.presentToInputName(valName));
        }

        // Get the persistent OpaqueContext that has all ext inputs registered.
        Pointer contextHandle = executor.getCachedOpContext();
        int numPlanExternalInputs = executor.getCurrentPlan() != null
                ? executor.getCurrentPlan().getExternalInputKeys().length : 0;
        int numPlanOutputs = allOutputNames.size();

        // Execute the native decode op
        long decodeLoopStart = System.currentTimeMillis();
        List<Integer> allTokens = new ArrayList<>();
        allTokens.add(firstTokenId);
        // Only add the second warmup token if maxNewTokens >= 2.
        // The warmup decode step runs unconditionally (needed for DSP plan compilation
        // with decode shapes), but when maxNewTokens==1 we must not include its token.
        if (maxNewTokens >= 2 && !stopTokenIds.contains(firstTokenId)) {
            allTokens.add(secondTokenId);
        }

        INDArray nativeTimingInfo = null;  // Hoisted for late-steady metric access after block
        if (remainingTokens > 0 && !stopTokenIds.contains(firstTokenId) && !stopTokenIds.contains(secondTokenId)) {

            // ── Constrained Java decode loop (ADR 0111) ────────────────────────────
            // When a ConstraintMasker is active the native AutoregressiveDecode C++ loop
            // cannot be used — it has no Java callback mechanism for per-step masking.
            // Instead we run a Java step-by-step loop that mirrors the decode-step update
            // sequence used above for the warmup: update input arrays → decoder.output() →
            // sampleToken with masker → scatter KV. Performance is comparable to the
            // Java slot-by-slot path (~8 tok/s CPU) rather than CUDA-graph-replay; this is
            // the v1 trade-off documented in the ADR (native path masking = Phase 2).
            if (constraintMasker != null) {
                log.info("[Constraint] Entering constrained Java decode loop: remainingTokens={}", remainingTokens);
                int currentToken = secondTokenId;
                long currentCachePos = actualPrefillLen + 1; // warmup wrote at actualPrefillLen+0 and +1

                for (int step = 0; step < remainingTokens; step++) {
                    if (stopTokenIds.contains(currentToken)) break;
                    if (constraintMasker.isComplete()) {
                        // Constraint satisfied: allow the model to emit EOS naturally.
                        log.debug("[Constraint] Constraint satisfied at step {}, emitting final token(s)", step);
                    }

                    // Build decode input map for this step (reuse existing static arrays).
                    // Update input values (shapes stay the same — no plan reshaping).
                    INDArray stepEmbed = embeddingTable.getRow(currentToken).reshape(1, 1, hiddenSize);
                    decodeEmbeddings.assign(stepEmbed);
                    stepEmbed.close();
                    decodeInputIds.putScalar(new long[]{0, 0}, currentToken);
                    decodeAttentionMask.putScalar(new long[]{0, currentCachePos - 1}, 1);
                    decodePosIds.putScalar(new long[]{0, 0}, currentCachePos);
                    decodeCausalMask.putScalar(new long[]{0, 0, 0, currentCachePos - 1}, 0.0f);
                    if (decodeAttnMaskReformat != null) {
                        decodeAttnMaskReformat.putScalar(new long[]{0, 0, 0, currentCachePos - 1}, 0.0f);
                    }

                    // cache_position if the model uses it (GGUF)
                    String cachePosNameJ = ioConfig.getCachePositionName();

                    Map<String, INDArray> stepInputMap = new LinkedHashMap<>();
                    String embedsNameJ = ioConfig.getInputEmbeddingsName();
                    if (embedsNameJ != null && decoder.hasVariable(embedsNameJ)) {
                        stepInputMap.put(embedsNameJ, decodeEmbeddings);
                    }
                    stepInputMap.put(ioConfig.getInputIdsName(), decodeInputIds);
                    stepInputMap.put(ioConfig.getAttentionMaskName(), decodeAttentionMask);
                    String posNameJ = ioConfig.getPositionIdsName();
                    if (posNameJ != null && decoder.hasVariable(posNameJ)) {
                        stepInputMap.put(posNameJ, decodePosIds);
                    }
                    String causalNameJ = ioConfig.getCausalMaskName();
                    if (causalNameJ != null && decoder.hasVariable(causalNameJ)) {
                        stepInputMap.put(causalNameJ, decodeCausalMask);
                    }
                    if (decodeAttnMaskReformat != null && attnReformatNode != null && decoder.hasVariable(attnReformatNode)) {
                        stepInputMap.put(attnReformatNode, decodeAttnMaskReformat);
                    }
                    if (cachePosNameJ != null && decoder.hasVariable(cachePosNameJ)) {
                        INDArray cpArr = Nd4j.scalar(DataType.INT64, currentCachePos);
                        stepInputMap.put(cachePosNameJ, cpArr);
                    }
                    for (String kn : kvNames.keyNames) {
                        stepInputMap.put(ioConfig.presentToInputName(kn), staticKvBuffers.get(ioConfig.presentToInputName(kn)));
                    }
                    for (String vn : kvNames.valueNames) {
                        stepInputMap.put(ioConfig.presentToInputName(vn), staticKvBuffers.get(ioConfig.presentToInputName(vn)));
                    }

                    Map<String, INDArray> stepOutputs = decoder.output(
                            stepInputMap, allOutputNames.toArray(new String[0]));

                    INDArray stepLogits = stepOutputs.get(ioConfig.getLogitsOutputName());
                    suppressStopsUnderFloor(stepLogits, 0, sampling, allTokens.size(), stopTokenIds);
                    int nextToken = sampleToken(stepLogits, 0, sampling, allTokens, rng,
                            constraintMasker, tokenizer, stopTokenIds);
                    stepLogits.close();

                    // Scatter KV outputs back into static buffers.
                    for (String kn : kvNames.keyNames) {
                        INDArray kv = stepOutputs.get(kn);
                        if (kv != null) scatterKvToStatic(kv, staticKvBuffers.get(ioConfig.presentToInputName(kn)), currentCachePos);
                    }
                    for (String vn : kvNames.valueNames) {
                        INDArray kv = stepOutputs.get(vn);
                        if (kv != null) scatterKvToStatic(kv, staticKvBuffers.get(ioConfig.presentToInputName(vn)), currentCachePos);
                    }
                    for (Map.Entry<String, INDArray> e : stepOutputs.entrySet()) {
                        if (!e.getKey().equals(ioConfig.getLogitsOutputName())) {
                            if (e.getValue() != null) e.getValue().close();
                        }
                    }

                    allTokens.add(nextToken);
                    currentToken = nextToken;
                    currentCachePos++;

                    if (stopTokenIds.contains(nextToken)) break;
                }
                log.info("[Constraint] Constrained Java decode loop complete: totalTokens={} emittedText={}",
                        allTokens.size(), constraintMasker.getEmittedText());
                // Skip the native op block below — jump to result building.
            } else {

            // Resolve cache_position ext idx if the model has it (GGUF models).
            // VLM/ONNX models typically don't, so this resolves to -1.
            String cachePosName = ioConfig.getCachePositionName();
            int cachePositionExtIdx = (cachePosName != null && decoder.hasVariable(cachePosName))
                    ? resolveExtInputIdx(executor, cachePosName) : -1;

            AutoregressiveDecode op = new AutoregressiveDecode(
                    decodeEmbeddings,
                    embeddingTable,
                    decodeInputIds,
                    decodeAttentionMask,
                    decodePosIds,
                    staticKvArray,
                    planHandle,
                    contextHandle,
                    numPlanExternalInputs,
                    numPlanOutputs,
                    embeddingsExtIdx,
                    maskExtIdx,
                    causalMaskExtIdx,
                    posIdsExtIdx,
                    inputIdsExtIdx,
                    logitsOutputIdx,
                    attnMaskReformatExtIdx,
                    cachePositionExtIdx,
                    kvInputExtIndices,
                    kvOutputIndices,
                    remainingTokens,
                    eosTokenId,
                    numKvPairs,
                    actualPrefillLen + 1,  // current position after 2 warmup steps
                    sampling.isGreedy() ? 0.0 : sampling.getTemperature(),
                    sampling.isGreedy() ? 0 : sampling.getTopK(),
                    sampling.isGreedy() ? 0.0 : sampling.getTopP(),
                    sampling.getRepetitionPenalty(),
                    stopTokenIds);
            applyNativePolicy(op, decodePolicy, sampling, allTokens.size());

            INDArray[] results = Nd4j.getExecutioner().exec(op);
            INDArray nativeTokenIds = results[0];
            INDArray nativeTokenCountArr = results[1];
            nativeTimingInfo = results[2];
            int nativeCount = nativeTokenCountArr.getInt(0);

            // Collect tokens from native op output — use actual token count, not buffer length.
            // Token ID 0 is a valid vocabulary token; do NOT treat it as padding.
            for (int i = 0; i < nativeCount; i++) {
                int tid = nativeTokenIds.getInt(i);
                allTokens.add(tid);
                if (stopTokenIds.contains(tid)) break;
            }
            closeOutput(nativeTokenIds);
            closeOutput(nativeTokenCountArr);
            log.info("[Perf] Decoder after native loop: planPhase={} pointersStable={}",
                    executor.getPlanPhase(), executor.arePointersStable());
            logDspReplayState("after native loop");

            // INVARIANT: if planPhase=REPLAYING, at least one segment must have replayed.
            // A replayCount=0 across all segments means every step ran slot-by-slot at
            // ~8 tok/s instead of the ~65 tok/s CUDA-graph-replay target.  No exception is
            // thrown (the generation result is still valid) but the degradation is severe
            // enough that it must be surfaced as a hard ERROR even without any debug flags.
            try {
                DspDebugger.GraphReplayReport replayReport =
                        DspDebugger.attach(decoder).analyzeGraphReplay();
                if (replayReport.errorMessage == null) {
                    if (replayReport.planPhase == PlanPhase.REPLAYING) {
                        int totalReplays = 0;
                        for (DspDebugger.SegmentReplayInfo seg : replayReport.segments) {
                            totalReplays += seg.replayCount;
                        }
                        if (totalReplays == 0) {
                            log.error("[DSP INVARIANT VIOLATION] planPhase=REPLAYING but " +
                                    "totalReplayCount=0 across {} segment(s). " +
                                    "All execution steps ran slot-by-slot. " +
                                    "Performance is SEVERELY degraded. " +
                                    "Check logs for 'DSP ERROR' or 'DSP WARN' messages.",
                                    replayReport.numSegments);
                        }
                    } else if (replayReport.planPhase != null
                            && replayReport.planPhase != PlanPhase.REPLAYING) {
                        // Plan never reached REPLAYING — every step was slot-by-slot or shapes-only.
                        log.error("[DSP] planPhase={} after decode loop — CUDA graphs were never " +
                                "captured. All segments resolved to slot-by-slot. " +
                                "Performance matches slot-by-slot baseline (~8 tok/s vs ~65 tok/s target).",
                                replayReport.planPhase);
                    }
                }
            } catch (Throwable t) {
                log.warn("[DSP] Post-loop replay invariant check failed: {}", t.getMessage());
            }
            } // end else (native op path — skipped when constraintMasker != null)
        } // end if (remainingTokens > 0)

        long decodeLoopEnd = System.currentTimeMillis();
        long decodeLoopMs = decodeLoopEnd - decodeLoopStart;
        int decodeSteps = allTokens.size() - 1;  // exclude first token (from prefill)
        double tokPerSec = decodeSteps > 0 && decodeLoopMs > 0
                ? (decodeSteps * 1000.0 / decodeLoopMs) : 0;
        float lateSteadyTokPerSec = nativeTimingInfo != null && nativeTimingInfo.length() > 5
                ? nativeTimingInfo.getFloat(5) : (float) tokPerSec;
        closeOutput(nativeTimingInfo);

        int[] tokenIds = allTokens.stream().mapToInt(Integer::intValue).toArray();
        String text = decodeGeneratedText(tokenIds, stopTokenIds);
        long endTime = System.currentTimeMillis();
        long timeMs = endTime - startTime;

        boolean hitEos = tokenIds.length > 0 && stopTokenIds.contains(tokenIds[tokenIds.length - 1]);

        log.info("[Perf] Native decode: {} tokens in {} ms ({} tok/s, lateSteady={} tok/s)",
                decodeSteps, decodeLoopMs, String.format("%.1f", tokPerSec),
                String.format("%.1f", lateSteadyTokPerSec));

        // Cleanup — fixed-buffer decode inputs belong to the frozen executor plan.
        // Closing them would destroy device memory tracked by the native plan and
        // break pointer-stable replay on the next page.
        currentInputIds.close();
        if (!fixedBuffers) {
            decodeEmbeddings.close();
            decodeInputIds.close();
            decodeCausalMask.close();
            decodeAttentionMask.close();
            decodePosIds.close();
            if (decodeAttnMaskReformat != null) decodeAttnMaskReformat.close();

            for (INDArray kv : staticKvBuffers.values()) {
                kv.setCloseable(true);  // Unpin before closing
                kv.close();
            }
        }

        return GenerationResult.builder()
                .text(text)
                .tokenIds(tokenIds)
                .generatedTokenCount(tokenIds.length)
                .promptTokenCount(prefillSeqLen)
                .totalTokenCount(prefillSeqLen + tokenIds.length)
                .finishReason(hitEos ? GenerationResult.FinishReason.EOS : GenerationResult.FinishReason.MAX_TOKENS)
                .generationTimeMs(timeMs)
                .tokensPerSecond(timeMs > 0 ? (tokenIds.length * 1000.0 / timeMs) : 0)
                .steadyStateTokensPerSecond(tokPerSec)
                .lateSteadyStateTokensPerSecond(lateSteadyTokPerSec)
                .build();
    }

    /**
     * Scatter a present KV [batch, heads, 1, dim] into a static buffer at cachePos.
     */
    private static void scatterKvToStatic(INDArray presentKv, INDArray staticBuf, long cachePos) {
        // presentKv shape: [batch, heads, seqLen, dim] — take last position
        long seqLen = presentKv.shape()[2];
        INDArray slice = presentKv.get(
                NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(seqLen - 1), NDArrayIndex.all());
        staticBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(cachePos), NDArrayIndex.all()).assign(slice);
    }

    private void logDspReplayState(String phase) {
        if (!Boolean.getBoolean("vlm.benchmark.dspState")) {
            return;
        }
        try {
            DspDebugger.GraphReplayReport replay = DspDebugger.attach(decoder).analyzeGraphReplay();
            if (replay.errorMessage != null) {
                log.info("[DSP] {} replay unavailable: {}", phase, replay.errorMessage);
                return;
            }
            log.info("[DSP] {} planPhase={} pointersStable={} fullyReplaying={} frozenExec={} segments={} replaying={} captureFailures={} stuck={}",
                    phase,
                    replay.planPhase,
                    replay.pointersStable,
                    replay.isFullyReplaying(),
                    replay.frozenExecutionCount,
                    replay.numSegments,
                    replay.getReplayingSegments().size(),
                    replay.getCaptureFailures().size(),
                    replay.getStuckSegments().size());
            for (DspDebugger.SegmentReplayInfo segment : replay.segments) {
                log.info("[DSP] {} {}", phase, segment);
            }
        } catch (Throwable t) {
            log.warn("[DSP] {} replay state unavailable: {}", phase, t.getMessage());
        }
    }

    /**
     * Resolve an external input index by name from a captured plan key order.
     */
    private static int resolveExtInputIdx(String[] externalInputKeys, String name) {
        if (name == null || externalInputKeys == null) return -1;
        for (int i = 0; i < externalInputKeys.length; i++) {
            if (name.equals(externalInputKeys[i])) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Resolve an external input index by name from the plan executor.
     */
    private static int resolveExtInputIdx(DynamicShapePlanExecutor executor, String name) {
        if (name == null || executor == null) return -1;
        return executor.findExternalInputIndex(name);
    }

    /**
     * Resolve an output index by name from the plan executor.
     */
    private static int resolveOutputIdx(DynamicShapePlanExecutor executor, String name) {
        if (name == null || executor == null) return -1;
        return executor.findOutputIndex(name);
    }

    /**
     * Pad a KV cache tensor from [batch, heads, seqLen, dim] to [batch, heads, maxKvLen, dim].
     */
    /**
     * Replace an entry in a quantized-buffer map. When an existing buffer occupies the slot,
     * close it first to release native memory, then insert the new array.
     * Used in the QUANTIZED KV path (STEP 2) to update the INT8 / scale archives.
     */
    private static void replaceQuantizedBuffer(Map<String, INDArray> map, String key, INDArray newBuf) {
        INDArray old = map.put(key, newBuf);
        if (old != null && !old.wasClosed()) {
            try { old.close(); } catch (Exception e) { /* best-effort close */ }
        }
    }

    private static INDArray padKvToStaticSize(INDArray presentKv, long maxKvLen) {
        long[] shape = presentKv.shape();
        long batch = shape[0], heads = shape[1], seqLen = shape[2], dim = shape[3];
        if (seqLen >= maxKvLen) {
            return presentKv.dup();
        }
        // Allocate the full static buffer as zeros, then copy prefill data into it.
        // This avoids Nd4j.concat which allocates a temp padding array + a concat result.
        INDArray result = Nd4j.zeros(presentKv.dataType(), batch, heads, maxKvLen, dim);
        result.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, seqLen), NDArrayIndex.all()).assign(presentKv);
        return result;
    }

    // ==================== Streaming ====================

    /**
     * Generate text with streaming token-by-token callback.
     *
     * <p>Each token is decoded and passed to the callback as it is generated.
     * The callback receives the decoded text for each individual token.
     * Pipeline defaults are used for decode options.</p>
     *
     * @param prompt the input text prompt
     * @param tokenCallback called with each generated token's decoded text
     */
    public void generateStream(String prompt, Consumer<String> tokenCallback) {
        generateStream(prompt, config.getMaxNewTokens(), tokenCallback, null);
    }

    /**
     * Generate text with streaming and a specified max token count.
     *
     * <p>Pipeline defaults are used for decode options.</p>
     *
     * @param prompt the input text prompt
     * @param maxNewTokens maximum number of tokens to generate
     * @param tokenCallback called with each generated token's decoded text
     */
    public void generateStream(String prompt, int maxNewTokens, Consumer<String> tokenCallback) {
        generateStream(prompt, maxNewTokens, tokenCallback, null);
    }

    /**
     * Generate text with streaming, a specified max token count, and per-call decode options.
     *
     * <p>Each token is decoded and passed to the callback as it is generated.
     * The {@link DecodeOptions} parameter is accepted for API compatibility.
     * Pass {@code null} to use pipeline defaults.</p>
     *
     * <p>Runs the full native {@code autoregressive_decode} C++ op, then streams the
     * result token by token via the callback. This post-hoc streaming is sufficient
     * for most use cases.</p>
     *
     * @param prompt the input text prompt
     * @param maxNewTokens maximum number of tokens to generate
     * @param tokenCallback called with each generated token's decoded text
     * @param options per-call decode options (nullable)
     */
    public void generateStream(String prompt, int maxNewTokens, Consumer<String> tokenCallback,
                               DecodeOptions options) {
        int[] promptTokenIds = encodePromptToIds(prompt);
        INDArray embeddings = embedTokens(promptTokenIds);

        // Run the full native decode, then stream the result token by token.
        GenerationResult result = generate(embeddings, promptTokenIds, maxNewTokens, options);

        // Stream decoded tokens
        if (result.getTokenIds() != null) {
            for (int tokenId : result.getTokenIds()) {
                String tokenText = tokenizer.decode(new int[]{tokenId}, false);
                tokenCallback.accept(tokenText);
            }
        }
    }

    // ==================== Lifecycle ====================

    /**
     * Suspend the pipeline on Android device-lost / pause events.
     *
     * <p>This is the Java-side hook for the Android {@code onPause()} / device-lost
     * lifecycle. It clears the DSP plan cache so that stale compiled plans
     * (which may hold native device pointers that become invalid after a device
     * loss) are not replayed after the device is recovered.</p>
     *
     * <p>Callers should invoke this method before the Android surface is destroyed
     * and before calling {@link #close()} (if a full teardown follows). On resume,
     * the pipeline will recompile plans lazily on the next generation call.</p>
     *
     * <p>This method is a no-op if the decoder is null or has already been closed.</p>
     */
    public void suspend() {
        // Close any open continuation session — it holds KV buffers and a plan handle
        // into native memory that may be invalidated by a device-lost event.
        GenerationSession openSession = activeSession.getAndSet(null);
        if (openSession != null) {
            try {
                openSession.state.close();
            } catch (Exception e) {
                log.warn("[lifecycle] Error closing active GenerationSession during suspend: {}", e.getMessage());
            }
        }
        // Clear DSP plan cache — plans bake device addresses that become stale after
        // a Vulkan device-lost event.  New plans will be compiled lazily on next use.
        if (decoder != null) {
            try {
                decoder.clearDynamicShapePlanCache();
                log.info("[lifecycle] DSP plan cache cleared on suspend");
            } catch (Exception e) {
                log.warn("[lifecycle] Error clearing DSP plan cache on suspend: {}", e.getMessage());
            }
        }
    }

    /**
     * Release all resources held by this pipeline.
     *
     * <p>Closes models that were loaded from paths (owned by the pipeline).
     * Pre-loaded models passed via the config are NOT closed -- the caller
     * retains ownership.</p>
     */
    @Override
    public void close() {
        // Close any open continuation session first — it holds retained KV buffers and a plan handle
        // into the decoder's native plan; freeing the decoder first would leave those dangling. Free the
        // retained state directly (bypassing the session's thread-affinity check) since close() may run on
        // a shutdown thread.
        GenerationSession openSession = activeSession.getAndSet(null);
        if (openSession != null) {
            try {
                openSession.state.close();
            } catch (Exception e) {
                log.warn("Error closing active GenerationSession state: {}", e.getMessage());
            }
        }
        // Free any retained one-shot fixed-buffer decode state (the forward-fix cache).
        InGraphKvState cachedOneShot = cachedFixedBufferState;
        cachedFixedBufferState = null;
        if (cachedOneShot != null) {
            try {
                cachedOneShot.close();
            } catch (Exception e) {
                log.warn("Error closing cached fixed-buffer state: {}", e.getMessage());
            }
        }
        // ADR 0107 V2: restore the KV placeholders' original dtype/shape (mutated to INT8
        // row-inline for quantised decode). The decoder SameDiff may be shared with later
        // pipelines using a non-quantised strategy — without the restore they would allocate
        // INT8 KV buffers and take the quantised attention path with a float-sized cache.
        if (kvPlaceholderOriginalDtypes != null && decoder != null) {
            try {
                Map<String, DataType> restore = new LinkedHashMap<>();
                for (Map.Entry<String, DataType> e : kvPlaceholderOriginalDtypes.entrySet()) {
                    if (decoder.hasVariable(e.getKey())
                            && decoder.getVariable(e.getKey()).dataType() != e.getValue()) {
                        restore.put(e.getKey(), e.getValue());
                    }
                }
                if (!restore.isEmpty()) decoder.convertDataTypes(restore);
                for (Map.Entry<String, long[]> e : kvPlaceholderOriginalShapes.entrySet()) {
                    if (e.getValue() != null && decoder.hasVariable(e.getKey())) {
                        decoder.getVariable(e.getKey()).setShape(e.getValue());
                    }
                }
                log.info("[GGUF-KV] Restored {} KV cache placeholders to original dtype/shape on close",
                        kvPlaceholderOriginalDtypes.size());
            } catch (Exception e) {
                log.warn("Error restoring KV placeholder dtypes on close: {}", e.getMessage());
            }
            kvPlaceholderOriginalDtypes = null;
            kvPlaceholderOriginalShapes = null;
        }
        // DSP cache ownership follows decoder ownership. SameDiff.close() tears down
        // live sessions before deleting native cache entries for an owned decoder. A
        // pre-loaded decoder is shared state and must remain untouched for its owner.
        // Order matters: close() first, THEN freeModelArrays(). The native DSP plan
        // teardown inside close() (releaseGpuIntermediates) walks slot NDArrays that
        // reference the model's DataBuffers — freeing those buffers first leaves the
        // teardown reading freed heap (SIGSEGV in the slot-release loops).
        // freeModelArrays() after close() is idempotent: already-closed buffers are
        // skipped via wasClosed().
        if (ownsDecoder && decoder != null) {
            try {
                decoder.close();
                SameDiffMemoryUtils.freeModelArrays(decoder);
            } catch (Exception e) {
                log.warn("Error closing decoder: {}", e.getMessage());
            }
        }
        if (ownsEmbedTokens && embedTokens != null) {
            try {
                embedTokens.close();
                SameDiffMemoryUtils.freeModelArrays(embedTokens);
            } catch (Exception e) {
                log.warn("Error closing embedTokens: {}", e.getMessage());
            }
        }
        if (ownsDraftDecoder && draftDecoder != null) {
            try {
                draftDecoder.close();
                SameDiffMemoryUtils.freeModelArrays(draftDecoder);
            } catch (Exception e) {
                log.warn("Error closing draftDecoder: {}", e.getMessage());
            }
        }
        // Free the cross-request KV prefix block pool (device buffers).
        if (prefixBlockPool != null) {
            try {
                prefixBlockPool.close();
            } catch (Exception e) {
                log.warn("Error closing prefix block pool: {}", e.getMessage());
            }
        }
    }

    // ==================== Internal Helpers ====================

    /**
     * Embed token IDs into embeddings using the extracted embedding table or the embed_tokens model.
     *
     * <p>Uses direct table lookup when an embedding table was extracted at pipeline creation,
     * falling back to a full SameDiff.output() call otherwise.</p>
     *
     * @param tokenIds token IDs to embed
     * @return embeddings [1, seqLen, hiddenSize]
     */
    public INDArray embedTokens(int[] tokenIds) {
        if (embeddingTable != null) {
            // Direct table lookup via pullRows — single native op instead of per-token
            // row copies in a loop. embeddingTable is [vocabSize, hiddenSize]; pullRows
            // with sourceDimension=1 gathers entire rows (tensors along dim 1) into
            // [seqLen, hiddenSize].
            INDArray gathered = Nd4j.pullRows(embeddingTable, 1, tokenIds);
            INDArray emb = gathered.reshape(1, tokenIds.length, hiddenSize);
            // Ensure FLOAT output (embedding table may be FLOAT already but be safe)
            if (emb.dataType() != DataType.FLOAT) {
                INDArray cast = emb.castTo(DataType.FLOAT);
                emb.close();
                emb = cast;
            }
            return emb;
        }

        // Fallback: run embed_tokens model via SameDiff
        if (embedTokens == null) {
            throw new IllegalStateException(
                    "No embedding table found and no embedTokens model provided. "
                    + "Single-model mode requires a discoverable embedding table in the decoder.");
        }
        INDArray inputIdsTensor = Nd4j.createFromArray(tokenIds)
                .reshape(1, tokenIds.length).castTo(DataType.LONG);
        try {
            Map<String, INDArray> embedOutputs = embedTokens.output(
                    Map.of(embedInputName, inputIdsTensor), embedOutputNames);

            INDArray result = null;
            for (Map.Entry<String, INDArray> entry : embedOutputs.entrySet()) {
                result = entry.getValue().dup();
            }
            if (result == null) {
                throw new IllegalStateException("embed_tokens model produced no output");
            }
            return result;
        } finally {
            SameDiffMemoryUtils.safeClose(inputIdsTensor);
        }
    }

    /**
     * Build a {@link DraftModelSpeculator} from the draft decoder model.
     */
    private Speculator buildDraftSpeculator() {
        ModelIOConfig draftIOConfig = ModelIOConfig.discover(draftDecoder);

        // Extract draft embedding table
        INDArray draftEmbedTable = extractEmbeddingTable(draftDecoder);
        long draftHidden = detectHiddenSize(draftDecoder, draftEmbedTable);

        // Determine draft vocab size
        long draftVocabSize = draftEmbedTable != null ? draftEmbedTable.shape()[0] : 0;

        // Embed function: direct table lookup
        final INDArray finalDraftEmbedTable = draftEmbedTable;
        final long finalDraftHidden = draftHidden;
        Function<int[], INDArray> embedFn;
        if (finalDraftEmbedTable != null) {
            embedFn = tokenIds -> {
                INDArray emb = Nd4j.zeros(DataType.FLOAT, 1, tokenIds.length, finalDraftHidden);
                for (int i = 0; i < tokenIds.length; i++) {
                    int clampedId = (int) Math.min(tokenIds[i], finalDraftEmbedTable.shape()[0] - 1);
                    emb.get(NDArrayIndex.point(0), NDArrayIndex.point(i), NDArrayIndex.all())
                            .assign(finalDraftEmbedTable.getRow(clampedId));
                }
                return emb;
            };
        } else {
            // Fallback: use draft decoder's own embed
            embedFn = tokenIds -> {
                INDArray inputIds = Nd4j.createFromArray(tokenIds)
                        .reshape(1, tokenIds.length).castTo(DataType.LONG);
                // Simple fallback -- just use zeros (draft model quality doesn't matter much)
                return Nd4j.zeros(DataType.FLOAT, 1, tokenIds.length, finalDraftHidden);
            };
        }

        // Decode function: greedy argmax from logits — Java-side argmax
        Function<INDArray, Integer> decodeFn = logits -> {
            // logits shape: [1, seqLen, vocabSize] -- take last position
            long seqLen = logits.shape()[1];
            INDArray lastLogits = logits.get(
                    NDArrayIndex.point(0),
                    NDArrayIndex.point(seqLen - 1),
                    NDArrayIndex.all()).dup();
            float[] vals = lastLogits.data().asFloat();
            int bestIdx = 0;
            float bestVal = vals[0];
            for (int j = 1; j < vals.length; j++) {
                if (vals[j] > bestVal) {
                    bestVal = vals[j];
                    bestIdx = j;
                }
            }
            lastLogits.close();
            return bestIdx;
        };

        return new DraftModelSpeculator(
                "draft-speculator",
                draftDecoder,
                embedFn,
                decodeFn,
                draftIOConfig,
                draftHidden,
                draftVocabSize,
                config.getMaxSpeculativeTokens(),
                256);
    }

    // ==================== Static Helpers ====================

    /**
     * Load a SameDiff model from a file path.
     *
     * <p>Uses {@link SameDiff#load(File, boolean)} for native SDZ/SDNB formats. For other
     * formats (ONNX, etc.), a custom {@link ModelLoader} must be provided via the config.
     * This avoids coupling samediff-llm to format-specific import modules.</p>
     *
     * @param modelPath path to the model file
     * @param modelLoader optional custom loader for non-native formats (may be null)
     * @return the loaded SameDiff model
     * @throws IOException if loading fails
     */
    static SameDiff loadModel(String modelPath, ModelLoader modelLoader) throws IOException {
        File modelFile = new File(modelPath);
        if (!modelFile.exists()) {
            throw new IOException("Model file not found: " + modelPath);
        }

        String name = modelFile.getName().toLowerCase();

        // Native SameDiff formats (SDZ, SDNB, flatbuffers)
        if (name.endsWith(".sdz") || name.endsWith(".sdnb") || name.endsWith(".fb")) {
            log.info("Loading SameDiff model: {}", modelFile.getName());
            long start = System.currentTimeMillis();
            SameDiff sd = SameDiff.load(modelFile, false);
            log.info("Loaded model in {}ms: {}", System.currentTimeMillis() - start, modelFile.getName());
            return sd;
        }

        // Custom loader for other formats (ONNX, etc.)
        if (modelLoader != null) {
            log.info("Loading model via custom loader: {}", modelFile.getName());
            long start = System.currentTimeMillis();
            SameDiff sd = modelLoader.load(modelPath);
            log.info("Custom loader completed in {}ms: {}", System.currentTimeMillis() - start, modelFile.getName());
            return sd;
        }

        throw new IOException("No loader available for model format: " + modelPath
                + ". Provide a modelLoader in GenerationPipelineConfig for non-SDZ formats.");
    }

    /**
     * Pluggable model loader for non-native formats (e.g., ONNX).
     *
     * <p>Implementations live in the module that has the import dependency
     * (e.g., samediff-import-onnx), keeping samediff-llm decoupled.</p>
     */
    @FunctionalInterface
    public interface ModelLoader {
        SameDiff load(String path) throws IOException;

        /**
         * Return metadata retained while loading {@code path}. Loaders that import
         * container formats should override this instead of discarding protocol
         * metadata when returning the SameDiff graph.
         */
        default ModelMetadata getModelMetadata(String path) throws IOException {
            return ModelMetadata.empty();
        }
    }

    /**
     * Model-owned generation metadata preserved across an import boundary.
     */
    public static final class ModelMetadata {
        private static final ModelMetadata EMPTY = new ModelMetadata(
                -1, -1, -1, null, Collections.emptySet(), Collections.emptySet());

        private final int bosTokenId;
        private final int eosTokenId;
        private final int padTokenId;
        private final String chatTemplate;
        private final Set<Integer> stopTokenIds;
        private final Set<Integer> specialTokenIds;

        private ModelMetadata(
                int bosTokenId,
                int eosTokenId,
                int padTokenId,
                String chatTemplate,
                Set<Integer> stopTokenIds,
                Set<Integer> specialTokenIds) {
            this.bosTokenId = bosTokenId;
            this.eosTokenId = eosTokenId;
            this.padTokenId = padTokenId;
            this.chatTemplate = chatTemplate;
            this.stopTokenIds = immutableTokenIds(stopTokenIds);
            this.specialTokenIds = immutableTokenIds(specialTokenIds);
        }

        public static ModelMetadata empty() {
            return EMPTY;
        }

        public static ModelMetadata of(
                int bosTokenId,
                int eosTokenId,
                int padTokenId,
                String chatTemplate,
                Set<Integer> stopTokenIds,
                Set<Integer> specialTokenIds) {
            return new ModelMetadata(bosTokenId, eosTokenId, padTokenId,
                    chatTemplate, stopTokenIds, specialTokenIds);
        }

        private static Set<Integer> immutableTokenIds(Set<Integer> tokenIds) {
            if (tokenIds == null || tokenIds.isEmpty()) {
                return Collections.emptySet();
            }
            Set<Integer> valid = new LinkedHashSet<>();
            for (Integer tokenId : tokenIds) {
                if (tokenId != null && tokenId >= 0) {
                    valid.add(tokenId);
                }
            }
            return Collections.unmodifiableSet(valid);
        }

        public int getBosTokenId() {
            return bosTokenId;
        }

        public int getEosTokenId() {
            return eosTokenId;
        }

        public int getPadTokenId() {
            return padTokenId;
        }

        public String getChatTemplate() {
            return chatTemplate;
        }

        public Set<Integer> getStopTokenIds() {
            return stopTokenIds;
        }

        public Set<Integer> getSpecialTokenIds() {
            return specialTokenIds;
        }
    }

    /**
     * Extract the embedding weight table from a SameDiff model.
     *
     * <p>Searches for the largest rank-2 CONSTANT or VARIABLE array in the model,
     * which is typically the token embedding matrix [vocabSize, hiddenSize].</p>
     *
     * @param model the SameDiff model (typically embed_tokens or a decoder with shared weights)
     * @return the embedding table, or null if not found
     */
    static INDArray extractEmbeddingTable(SameDiff model) {
        INDArray embeddingTable = null;
        for (SDVariable var : model.variables()) {
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
            // If the optimizer quantized constants to HALF for memory savings,
            // cast the embedding table back to FLOAT for the autoregressive_decode op.
            if (embeddingTable.dataType() == DataType.HALF || embeddingTable.dataType() == DataType.BFLOAT16) {
                log.info("Extracted embedding table is {} — casting to FLOAT for decode op", embeddingTable.dataType());
                embeddingTable = embeddingTable.castTo(DataType.FLOAT);
            }
            log.info("Extracted embedding table: shape={}", Arrays.toString(embeddingTable.shape()));
        } else {
            log.warn("Could not extract embedding table from model");
        }
        return embeddingTable;
    }

    /**
     * Detect the hidden size from a model's embedding table or variable shapes.
     *
     * @param model the SameDiff model
     * @return detected hidden size, or 0 if unable to detect
     */
    private static long detectHiddenSize(SameDiff model) {
        INDArray embedTable = extractEmbeddingTable(model);
        return detectHiddenSize(model, embedTable);
    }

    /**
     * Detect the hidden size from a pre-extracted embedding table.
     */
    private static long detectHiddenSize(SameDiff model, INDArray embedTable) {
        if (embedTable != null && embedTable.rank() == 2) {
            // Embedding table shape: [vocabSize, hiddenSize]
            return embedTable.shape()[1];
        }

        // Fallback: search for output variables with 3D shape [batch, seq, hidden]
        for (String outputName : model.outputs()) {
            SDVariable var = model.getVariable(outputName);
            if (var != null && var.getShape() != null && var.getShape().length == 3) {
                long candidate = var.getShape()[2];
                if (candidate > 0) {
                    log.info("Detected hidden size {} from output variable '{}'", candidate, outputName);
                    return candidate;
                }
            }
        }

        log.warn("Could not auto-detect hidden size from model");
        return 0;
    }
}
