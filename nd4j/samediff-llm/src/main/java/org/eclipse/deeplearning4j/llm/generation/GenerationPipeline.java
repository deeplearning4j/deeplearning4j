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
import org.eclipse.deeplearning4j.llm.generation.sampling.Sampler;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplerUtils;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.generation.speculative.Speculator;
import org.eclipse.deeplearning4j.llm.generation.speculative.DraftModelSpeculator;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfig;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfigApplier;
import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.common.config.ND4JSystemProperties;

import org.nd4j.linalg.api.ops.impl.transforms.custom.AutoregressiveDecode;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRule;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.execution.DspCompilationMode;
import org.nd4j.autodiff.samediff.execution.DspDebugger;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.nativeblas.OpaqueDataBuffer;
import org.bytedeco.javacpp.Pointer;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Function;

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
 * @see GenerationPipelineConfig
 * @see AutoregressiveDecode
 * @see GenerationResult
 * @see ModelIOConfig
 */
@Slf4j
public class GenerationPipeline implements AutoCloseable {

    // ---- Core components ----

    @Getter
    private final SameDiff decoder;

    @Getter
    private final SameDiff embedTokens;

    private final Tokenizer tokenizer;

    @Getter
    private final ModelIOConfig ioConfig;

    private final GenerationPipelineConfig config;

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

    private GenerationPipeline(
            SameDiff decoder, boolean ownsDecoder,
            SameDiff embedTokens, boolean ownsEmbedTokens,
            Tokenizer tokenizer,
            ModelIOConfig ioConfig,
            INDArray embeddingTable,
            long hiddenSize,
            String embedInputName, String[] embedOutputNames,
            SameDiff draftDecoder, boolean ownsDraftDecoder,
            GenerationPipelineConfig config) {
        this.decoder = decoder;
        this.ownsDecoder = ownsDecoder;
        this.embedTokens = embedTokens;
        this.ownsEmbedTokens = ownsEmbedTokens;
        this.tokenizer = tokenizer;
        this.ioConfig = ioConfig;
        this.embeddingTable = embeddingTable;
        this.hiddenSize = hiddenSize;
        this.embedInputName = embedInputName;
        this.embedOutputNames = embedOutputNames;
        this.draftDecoder = draftDecoder;
        this.ownsDraftDecoder = ownsDraftDecoder;
        this.config = config;

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
            java.util.concurrent.CompletableFuture<SameDiff> decoderFuture =
                    java.util.concurrent.CompletableFuture.supplyAsync(() -> {
                        try {
                            return loadModel(config.getDecoderPath(), modelLoader);
                        } catch (IOException e) {
                            throw new java.util.concurrent.CompletionException(e);
                        }
                    });
            java.util.concurrent.CompletableFuture<SameDiff> embedFuture =
                    java.util.concurrent.CompletableFuture.supplyAsync(() -> {
                        try {
                            return loadModel(config.getEmbedTokensPath(), modelLoader);
                        } catch (IOException e) {
                            throw new java.util.concurrent.CompletionException(e);
                        }
                    });
            try {
                decoder = decoderFuture.join();
                ownsDecoder = true;
                embedTokens = embedFuture.join();
                ownsEmbedTokens = true;
            } catch (java.util.concurrent.CompletionException e) {
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

        // 1b. Run graph optimizer (sigmoid*x→swish/silu, SwiGLU fusion, etc.)
        // Controlled via GenerationPipelineConfig.graphOptimizerEnabled (default: false).
        // Default is false because OnnxModelCache already runs the full optimizer during import.
        // Excludes QuantizationOptimizations (FP16 weight quantization handled separately).
        if (config.isGraphOptimizerEnabled()) {
            int opsBefore = decoder.getOps().size();
            long optStart = System.currentTimeMillis();
            List<String> outputs = decoder.outputs() != null
                    ? new ArrayList<>(decoder.outputs()) : new ArrayList<>();
            List<org.nd4j.autodiff.samediff.optimize.OptimizerSet> optimizations =
                    GraphOptimizer.defaultOptimizations().stream()
                            .filter(s -> !(s instanceof org.nd4j.autodiff.samediff.optimize.optimizations.QuantizationOptimizations))
                            .collect(java.util.stream.Collectors.toList());
            SameDiff originalDecoder = decoder;
            boolean ownsOriginalDecoder = ownsDecoder;
            SameDiff optimizedDecoder = GraphOptimizer.optimize(decoder, outputs, optimizations);
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

        log.info("GenerationPipeline created: decoder ops={}, embedTokens={}, hiddenSize={}, "
                        + "embeddingTable={}, draftDecoder={}, kvStrategy={}, dsp={}",
                decoder.getOps().size(),
                embedTokens != null ? embedTokens.getOps().size() + " ops" : "single-model mode (using decoder)",
                resolvedHiddenSize,
                embeddingTable != null ? java.util.Arrays.toString(embeddingTable.shape()) : "null (fallback to SameDiff.output())",
                draftDecoder != null ? draftDecoder.getOps().size() + " ops" : "disabled",
                config.getKvCacheStrategy(),
                config.isDspEnabled());

        GenerationPipeline pipeline = new GenerationPipeline(
                decoder, ownsDecoder,
                embedTokens, ownsEmbedTokens,
                config.getTokenizer(),
                ioConfig,
                embeddingTable,
                resolvedHiddenSize,
                embedInputName, embedOutputNames,
                draftDecoder, ownsDraftDecoder,
                config);

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
        int restoreDevice = switchToDecoderDevice("text-generation");
        // Suppress cross-device routing for the entire generation. Model weights,
        // prompt tensors, and KV caches must stay on the decoder's execution device.
        // The CUDA async memory pool reports pool-reserved memory as "used" to
        // cudaMemGetInfo, causing false OOM detection and unnecessary cross-device
        // routing. The pool handles actual OOM via trim+retry.
        OpaqueDataBuffer.suppressCrossDeviceRouting(true);
        try {
            return generateInternal(prompt, maxNewTokens);
        } finally {
            OpaqueDataBuffer.suppressCrossDeviceRouting(false);
            restoreDevice(restoreDevice, "text-generation");
        }
    }

    private GenerationResult generateInternal(String prompt, int maxNewTokens) {
        // Apply chat template if configured and prompt doesn't already contain template markers
        String effectivePrompt = prompt;
        boolean addSpecialTokens = true;
        boolean promptAlreadyFormatted = prompt.contains("<|im_start|>") || prompt.contains("[INST]");
        String chatTemplateText = resolveChatTemplate();
        if (chatTemplateText != null && !chatTemplateText.isEmpty()
                && !promptAlreadyFormatted) {
            List<ChatTemplate.Message> messages = new ArrayList<>();
            messages.add(ChatTemplate.Message.user(prompt));
            ChatTemplate chatTemplate = new ChatTemplate(
                    chatTemplateText,
                    tokenizer.getBosToken(),
                    tokenizer.getEosToken());
            effectivePrompt = chatTemplate.apply(messages, true);
            addSpecialTokens = false;
            log.debug("Applied chat template, prompt length: {} -> {}", prompt.length(), effectivePrompt.length());
        } else if (promptAlreadyFormatted) {
            addSpecialTokens = false;
        }

        // Tokenize
        int[] promptTokenIds = tokenizer.encode(effectivePrompt, addSpecialTokens).getIds();
        if (promptTokenIds == null || promptTokenIds.length == 0) {
            throw new IllegalArgumentException("Prompt encoding produced no tokens");
        }

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

    private String resolveChatTemplate() {
        if (config.getChatTemplate() != null && !config.getChatTemplate().isBlank()) {
            return config.getChatTemplate();
        }
        return tokenizer.getChatTemplate();
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
    private GenerationResult generateSimpleWithInGraphKvCache(int[] promptTokenIds, int maxNewTokens,
                                                                ModelIOConfig.KVCacheNames kvInputNames) {
        long startTime = System.currentTimeMillis();

        int maxPrefill = config.getMaxPrefillLength();
        boolean fixedBuffers = maxPrefill > 0;

        if (fixedBuffers) {
            // Fixed-size buffers: skip DSP reset — reuse the frozen plan.
            // Shapes are identical across calls since everything is pre-allocated
            // at maxPrefillLength. Only reset session state (KV caches, recurrent
            // states) — the plan itself stays intact.
            InferenceSession existSession = decoder.getOrCreateSession();
            if (existSession != null) {
                DynamicShapePlanExecutor existExecutor = existSession.getDynamicShapePlanExecutor();
                if (existExecutor != null && existExecutor.isShapesFrozen()) {
                    log.info("[Lifecycle] Reusing frozen DSP plan (fixedBuffers=true, maxPrefill={})", maxPrefill);
                    // Clear node outputs but keep the plan — we need fresh KV/state
                    // buffers but the compiled execution plan is shape-stable.
                    existSession.clearNodeOutputsOnly();
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
                }
            }
        }

        int eosTokenId = tokenizer.getEosTokenId();
        Set<Integer> stopTokenIds = buildStopTokenIds();

        SamplingConfig sampling = config.getSamplingConfig() != null
                ? config.getSamplingConfig() : SamplingConfig.defaultConfig();
        Random rng = sampling.getSeed() != null ? new Random(sampling.getSeed()) : new Random();

        String inputIdsName = ioConfig.getInputIdsName() != null ? ioConfig.getInputIdsName() : "input_ids";
        String logitsName = ioConfig.getLogitsOutputName() != null ? ioConfig.getLogitsOutputName() : "lm_logits";
        String posOffsetName = ioConfig.getPositionOffsetName();
        String cachePosName = ioConfig.getCachePositionName();
        String causalMaskName = ioConfig.getCausalMaskName();

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

        long maxKvLen = prefillSeqLen + maxNewTokens;
        // Cap to configured KV cache length to keep buffer shapes stable across
        // calls with different maxNewTokens — avoids plan recompilation.
        int kvCap = config.getMaxKvCacheLength();
        if (kvCap > 0 && maxKvLen > kvCap) {
            maxKvLen = kvCap;
            maxNewTokens = (int) (maxKvLen - prefillSeqLen);
            if (maxNewTokens < 1) maxNewTokens = 1;
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

        INDArray prefillInputIds = Nd4j.createFromArray(effectiveTokenIds)
                .reshape(1, prefillSeqLen).castTo(DataType.INT64);

        Map<String, INDArray> prefillInputMap = new HashMap<>();
        prefillInputMap.put(inputIdsName, prefillInputIds);

        if (posOffsetName != null && decoder.hasVariable(posOffsetName)) {
            prefillInputMap.put(posOffsetName, Nd4j.scalar(DataType.INT64, 0));
        }
        if (cachePosName != null && decoder.hasVariable(cachePosName)) {
            prefillInputMap.put(cachePosName, Nd4j.scalar(DataType.INT64, 0));
        }
        DataType maskDtype = DataType.FLOAT;
        if (causalMaskName != null && decoder.hasVariable(causalMaskName)) {
            maskDtype = decoder.getVariable(causalMaskName).dataType();
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

        // Request outputs: logits + per-layer KV outputs + recurrent state outputs
        List<String> prefillOutputNames = new ArrayList<>();
        prefillOutputNames.add(logitsName);
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

        log.info("[GGUF-KV] Prefill returned {} outputs: {}", prefillOutputs.size(), prefillOutputs.keySet());

        // Sample first token from prefill logits.
        // When fixedBuffers is active, the logits tensor has shape [1, prefillSeqLen, vocab]
        // where prefillSeqLen includes padding. The real last token is at actualPrefillLen-1.
        INDArray prefillLogits = prefillOutputs.get(logitsName);
        if (prefillLogits == null) {
            throw new RuntimeException("[GGUF-KV] Prefill logits '" + logitsName + "' not found in outputs: " + prefillOutputs.keySet());
        }
        // Clamp to the actual logits seq dim (see generateNative): no-op for the padded
        // [1, prefillSeqLen, vocab] case here, but guards an UNCHECKED OOB read if a model
        // ever exports last-position-only [1, 1, vocab] logits on this path too.
        int logitsSamplePos = Math.min(actualPrefillLen - 1, (int) prefillLogits.shape()[1] - 1);
        log.info("[GGUF-KV] Prefill logits shape: {} samplePos={} (actualPrefillLen={})",
                java.util.Arrays.toString(prefillLogits.shape()), logitsSamplePos, actualPrefillLen);
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
        int firstTokenId = sampleToken(prefillLogits, logitsSamplePos, sampling, generatedSoFar, rng);
        generatedSoFar.add(firstTokenId);
        prefillLogits.close();
        prefillInputIds.close();

        long firstTokenMs = System.currentTimeMillis() - startTime;

        log.info("[GGUF-KV] First token: {} (eos={})", firstTokenId, stopTokenIds.contains(firstTokenId));

        if (stopTokenIds.contains(firstTokenId)) {
            closePrefillOutputs(prefillOutputs, logitsName);
            List<Integer> tokens = new ArrayList<>();
            tokens.add(firstTokenId);
            return buildResult(tokens, promptTokenIds, stopTokenIds, startTime, firstTokenMs);
        }

        // ══════════════════════════════════════════════════════════════════════
        // STEP 2: Initialize static KV buffers from prefill K/V outputs
        // Shape: [batch, maxKvLen, numKVHeads, headDim] (BSHD layout)
        // ══════════════════════════════════════════════════════════════════════
        log.info("[GGUF-KV] STEP 2: numLayers={} keyNames.size={}", numLayers, kvInputNames.keyNames.size());
        Map<String, INDArray> staticKvBuffers = new LinkedHashMap<>();
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

            // Create full-size zero buffer and write prefill data at positions 0..prefillLen-1
            INDArray keyBuf = Nd4j.zeros(kvDtype, batch, maxKvLen, numKVHeads, headDim);
            keyBuf.get(NDArrayIndex.all(), NDArrayIndex.interval(0, prefillSeqLen),
                    NDArrayIndex.all(), NDArrayIndex.all()).assign(kRoped);
            staticKvBuffers.put(kvInputNames.keyNames.get(i), keyBuf);

            INDArray valBuf = Nd4j.zeros(kvDtype, batch, maxKvLen, numKVHeads, headDim);
            valBuf.get(NDArrayIndex.all(), NDArrayIndex.interval(0, prefillSeqLen),
                    NDArrayIndex.all(), NDArrayIndex.all()).assign(vHeads);
            staticKvBuffers.put(kvInputNames.valueNames.get(i), valBuf);

            log.info("[GGUF-KV] Layer {} KV: kRoped shape={} min={} max={}, keyBuf shape={} min={} max={}",
                    layerIdx, java.util.Arrays.toString(kRoped.shape()), kRoped.minNumber(), kRoped.maxNumber(),
                    java.util.Arrays.toString(keyBuf.shape()), keyBuf.minNumber(), keyBuf.maxNumber());
            kRoped.close();
            vHeads.close();
        }

        // Create recurrent state buffers from prefill outputs (keyed by input name)
        Map<String, INDArray> recurrentStateBuffers = new LinkedHashMap<>();
        for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
            INDArray stateOut = prefillOutputs.get(pair.outputName);
            if (stateOut != null) {
                recurrentStateBuffers.put(pair.inputName, stateOut.dup());
                stateOut.close();
                log.info("[GGUF-KV] Recurrent state {} → {} shape={}", pair.inputName, pair.outputName,
                        java.util.Arrays.toString(recurrentStateBuffers.get(pair.inputName).shape()));
            } else {
                log.warn("[GGUF-KV] Missing prefill output for recurrent state '{}'", pair.outputName);
            }
        }

        Nd4j.getExecutioner().commit();

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

        INDArray decodeInputIds = Nd4j.createFromArray(new int[]{firstTokenId})
                .reshape(1, 1).castTo(DataType.INT64);

        INDArray decodeCausalMask = null;
        if (causalMaskName != null && decoder.hasVariable(causalMaskName)) {
            // Decode mask attends to positions 0..firstDecodePos (inclusive of current).
            // Shape [1,1,1,maxKvLen] is FIXED regardless of actual prompt length.
            decodeCausalMask = DecoderInputBuilder.buildInGraphDecodeMask(firstDecodePos, maxKvLen, maskDtype);
            log.info("[GGUF-KV] Decode mask shape={} dtype={} firstDecodePos={} min={} max={} hasNaN={}",
                    java.util.Arrays.toString(decodeCausalMask.shape()), decodeCausalMask.dataType(),
                    firstDecodePos,
                    decodeCausalMask.minNumber(), decodeCausalMask.maxNumber(),
                    Double.isNaN(decodeCausalMask.meanNumber().doubleValue()));
        }
        INDArray decodePositionOffset = null;
        if (posOffsetName != null && decoder.hasVariable(posOffsetName)) {
            decodePositionOffset = Nd4j.scalar(DataType.INT64, firstDecodePos);
        }
        INDArray decodeCachePosition = null;
        if (cachePosName != null && decoder.hasVariable(cachePosName)) {
            decodeCachePosition = Nd4j.scalar(DataType.INT64, firstDecodePos);
        }

        Map<String, INDArray> decodeInputMap = new HashMap<>();
        decodeInputMap.put(inputIdsName, decodeInputIds);
        if (decodeCausalMask != null) decodeInputMap.put(causalMaskName, decodeCausalMask);
        if (decodePositionOffset != null) decodeInputMap.put(posOffsetName, decodePositionOffset);
        if (decodeCachePosition != null) decodeInputMap.put(cachePosName, decodeCachePosition);
        for (Map.Entry<String, INDArray> entry : staticKvBuffers.entrySet()) {
            if (decoder.hasVariable(entry.getKey())) {
                decodeInputMap.put(entry.getKey(), entry.getValue());
            }
        }
        // Add GDN/conv state buffers to decode input map
        for (Map.Entry<String, INDArray> entry : recurrentStateBuffers.entrySet()) {
            if (decoder.hasVariable(entry.getKey())) {
                decodeInputMap.put(entry.getKey(), entry.getValue());
            }
        }

        List<String> decodeOutputNames = new ArrayList<>();
        decodeOutputNames.add(logitsName);
        for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
            decodeOutputNames.add(pair.outputName);
        }

        log.info("[GGUF-KV] STEP 3: warmup decode with {} KV buffers, {} recurrent state buffers, {} inputs",
                staticKvBuffers.size(), recurrentStateBuffers.size(), decodeInputMap.size());

        Map<String, INDArray> decodeOutputs;
        try {
            decodeOutputs = decoder.output(
                    decodeInputMap, decodeOutputNames.toArray(new String[0]));
        } catch (Exception e) {
            log.error("[GGUF-KV] STEP 3 warmup decode failed", e);
            throw e;
        }

        INDArray decodeLogits = decodeOutputs.get(logitsName);
        int secondTokenId = sampleToken(decodeLogits, 0, sampling, generatedSoFar, rng);
        generatedSoFar.add(secondTokenId);
        log.info("[GGUF-KV] STEP 3 second token: {}", secondTokenId);
        decodeLogits.close();

        // Update recurrent state buffers with outputs from warmup decode
        for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
            INDArray updated = decodeOutputs.get(pair.outputName);
            if (updated != null) {
                INDArray buf = recurrentStateBuffers.get(pair.inputName);
                if (buf != null) buf.assign(updated);
                updated.close();
            }
        }

        // ══════════════════════════════════════════════════════════════════════
        // STEP 4: Get native plan handle, freeze shapes, resolve ext indices
        // ══════════════════════════════════════════════════════════════════════
        InferenceSession session = decoder.getOrCreateSession();
        DynamicShapePlanExecutor executor = session != null ? session.getDynamicShapePlanExecutor() : null;
        Pointer planHandle = executor != null ? executor.getNativePlanHandle() : null;

        if (planHandle == null || planHandle.isNull()) {
            log.warn("Native plan handle not available for GGUF -- returning partial result");
            decodeInputIds.close();
            if (decodeCausalMask != null) decodeCausalMask.close();
            if (decodePositionOffset != null) decodePositionOffset.close();
            if (decodeCachePosition != null) decodeCachePosition.close();
            for (INDArray kv : staticKvBuffers.values()) kv.close();
            for (INDArray rs : recurrentStateBuffers.values()) rs.close();

            List<Integer> tokens = new ArrayList<>();
            tokens.add(firstTokenId);
            if (!stopTokenIds.contains(firstTokenId)) tokens.add(secondTokenId);
            return buildResult(tokens, promptTokenIds, stopTokenIds, startTime, firstTokenMs);
        }

        if (executor.getCurrentPlan() != null) {
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

        int inputIdsExtIdx = resolveExtInputIdx(executor, inputIdsName);
        int causalMaskExtIdx = causalMaskName != null ? resolveExtInputIdx(executor, causalMaskName) : -1;
        int posOffsetExtIdx = posOffsetName != null ? resolveExtInputIdx(executor, posOffsetName) : -1;
        int cachePosExtIdx = cachePosName != null ? resolveExtInputIdx(executor, cachePosName) : -1;
        int logitsOutputIdx = resolveOutputIdx(executor, logitsName);

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

        // ══════════════════════════════════════════════════════════════════════
        // STEP 5: Execute native AutoregressiveDecode op
        // ══════════════════════════════════════════════════════════════════════
        int remainingTokens = maxNewTokens - 2;

        decodeInputIds.putScalar(new long[]{0, 0}, secondTokenId);
        if (decodeCausalMask != null) {
            decodeCausalMask.putScalar(new long[]{0, 0, 0, firstDecodePos}, 0.0f);
        }
        if (decodePositionOffset != null) {
            decodePositionOffset.putScalar(new long[]{}, (long)(firstDecodePos + 1));
        }
        if (decodeCachePosition != null) {
            decodeCachePosition.putScalar(new long[]{}, (long)(firstDecodePos + 1));
        }

        Nd4j.getExecutioner().commit();
        decodeInputIds.syncToDevice();
        if (decodeCausalMask != null) decodeCausalMask.syncToDevice();
        if (decodePositionOffset != null) decodePositionOffset.syncToDevice();
        if (decodeCachePosition != null) decodeCachePosition.syncToDevice();
        for (INDArray kvBuf : staticKvBuffers.values()) kvBuf.syncToDevice();
        for (INDArray rsBuf : recurrentStateBuffers.values()) rsBuf.syncToDevice();

        INDArray[] staticKvArray = new INDArray[2 * numKvPairs];
        int idx = 0;
        for (String keyName : kvInputNames.keyNames) {
            staticKvArray[idx++] = staticKvBuffers.get(keyName);
        }
        for (String valName : kvInputNames.valueNames) {
            staticKvArray[idx++] = staticKvBuffers.get(valName);
        }

        Pointer contextHandle = executor.getCachedOpContext();
        int numPlanExternalInputs = executor.getCurrentPlan() != null
                ? executor.getCurrentPlan().getExternalInputKeys().length : 0;
        int numPlanOutputs = decodeOutputNames.size();

        if (remainingTokens > 0) {
            // GGUF models handle their own embedding lookup internally
            INDArray dummyEmbeddings = Nd4j.zeros(DataType.FLOAT, 1, 1, 1);
            INDArray dummyEmbTable = Nd4j.zeros(DataType.FLOAT, 1, 1);

            AutoregressiveDecode op = new AutoregressiveDecode(
                    dummyEmbeddings, dummyEmbTable, decodeInputIds,
                    decodeCausalMask, null, staticKvArray,
                    planHandle, contextHandle,
                    numPlanExternalInputs, numPlanOutputs,
                    embeddingsExtIdx, maskExtIdx, causalMaskExtIdx,
                    posIdsExtIdx, inputIdsExtIdx, logitsOutputIdx,
                    -1,                 // attnMaskReformatExtIdx
                    posOffsetExtIdx,    // position_offset ext index
                    cachePosExtIdx,     // cache_position ext index
                    kvInputExtIndices, kvOutputIndices,
                    gdnStateExtIndices, gdnStateOutputIndices,
                    convStateExtIndices, convStateOutputIndices,
                    remainingTokens, eosTokenId, numKvPairs,
                    firstDecodePos + 1,
                    sampling.isGreedy() ? 0.0 : sampling.getTemperature(),
                    sampling.isGreedy() ? 0 : sampling.getTopK(),
                    sampling.isGreedy() ? 0.0 : sampling.getTopP(),
                    sampling.getRepetitionPenalty(),
                    stopTokenIds);

            INDArray[] results = Nd4j.getExecutioner().exec(op);
            INDArray nativeTokenIds = results[0];
            INDArray nativeTokenCount = results[1];
            INDArray nativeTimingInfo = results[2];
            int nativeCount = nativeTokenCount.getInt(0);

            List<Integer> allTokens = new ArrayList<>();
            allTokens.add(firstTokenId);
            if (!stopTokenIds.contains(firstTokenId)) {
                allTokens.add(secondTokenId);
                if (!stopTokenIds.contains(secondTokenId)) {
                    for (int i = 0; i < nativeCount; i++) {
                        int tok = (int) nativeTokenIds.getLong(i);
                        allTokens.add(tok);
                        if (stopTokenIds.contains(tok)) break;
                    }
                }
            }

            int[] tokenIds = allTokens.stream().mapToInt(Integer::intValue).toArray();
            String text = tokenizer.decode(tokenIds, false);
            log.info("[GGUF-KV] STEP 5 complete: nativeCount={} allTokens={} text='{}'",
                    nativeCount, allTokens, text.length() > 200 ? text.substring(0, 200) : text);
            long endTime = System.currentTimeMillis();
            long timeMs = endTime - startTime;
            float tokPerSec = nativeTimingInfo.getFloat(2);
            float lateSteadyTokPerSec = nativeTimingInfo.length() > 5 ? nativeTimingInfo.getFloat(5) : tokPerSec;
            boolean hitEos = tokenIds.length > 0 && stopTokenIds.contains(tokenIds[tokenIds.length - 1]);

            dummyEmbeddings.close();
            dummyEmbTable.close();
            decodeInputIds.close();
            if (decodeCausalMask != null) decodeCausalMask.close();
            if (decodePositionOffset != null) decodePositionOffset.close();
            if (decodeCachePosition != null) decodeCachePosition.close();
            for (INDArray kv : staticKvBuffers.values()) kv.close();
            for (INDArray rs : recurrentStateBuffers.values()) rs.close();

            return GenerationResult.builder()
                    .text(text).tokenIds(tokenIds)
                    .generatedTokenCount(tokenIds.length).promptTokenCount(prefillSeqLen)
                    .totalTokenCount(prefillSeqLen + tokenIds.length)
                    .finishReason(hitEos ? GenerationResult.FinishReason.EOS : GenerationResult.FinishReason.MAX_TOKENS)
                    .generationTimeMs(timeMs)
                    .tokensPerSecond(timeMs > 0 ? (tokenIds.length * 1000.0 / timeMs) : 0)
                    .steadyStateTokensPerSecond(tokPerSec)
                    .lateSteadyStateTokensPerSecond(lateSteadyTokPerSec)
                    .build();
        } else {
            decodeInputIds.close();
            if (decodeCausalMask != null) decodeCausalMask.close();
            if (decodePositionOffset != null) decodePositionOffset.close();
            if (decodeCachePosition != null) decodeCachePosition.close();

            List<Integer> allTokens = new ArrayList<>();
            if (maxNewTokens >= 1) allTokens.add(firstTokenId);
            if (maxNewTokens >= 2 && !stopTokenIds.contains(firstTokenId)) allTokens.add(secondTokenId);

            int[] tokenIds = allTokens.stream().mapToInt(Integer::intValue).toArray();
            String text = tokenizer.decode(tokenIds, false);
            long endTime = System.currentTimeMillis();
            long timeMs = endTime - startTime;
            boolean hitEos = tokenIds.length > 0 && stopTokenIds.contains(tokenIds[tokenIds.length - 1]);

            for (INDArray kv : staticKvBuffers.values()) kv.close();
            for (INDArray rs : recurrentStateBuffers.values()) rs.close();

            return GenerationResult.builder()
                    .text(text).tokenIds(tokenIds)
                    .generatedTokenCount(tokenIds.length).promptTokenCount(prefillSeqLen)
                    .totalTokenCount(prefillSeqLen + tokenIds.length)
                    .finishReason(hitEos ? GenerationResult.FinishReason.EOS : GenerationResult.FinishReason.MAX_TOKENS)
                    .generationTimeMs(timeMs)
                    .tokensPerSecond(timeMs > 0 ? (tokenIds.length * 1000.0 / timeMs) : 0)
                    .steadyStateTokensPerSecond(0)
                    .build();
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
    private static long[] deriveRecurrentStateShape(SameDiff sd, String stateName) {
        Variable var = sd.getVariables().get(stateName);
        if (var == null || var.getInputsForOp() == null || var.getInputsForOp().isEmpty()) {
            return null;
        }

        for (String opName : var.getInputsForOp()) {
            DifferentialFunction op;
            try { op = sd.getOpById(opName); } catch (Exception e) { continue; }
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
     * Resolve and validate the EOS token ID from the tokenizer, building the stop token set.
     */
    private Set<Integer> buildStopTokenIds() {
        int eosTokenId = tokenizer.getEosTokenId();
        log.info("[Generation] Resolved eosTokenId={} from tokenizer", eosTokenId);
        Set<Integer> stopTokenIds = new HashSet<>();
        if (eosTokenId >= 0) {
            stopTokenIds.add(eosTokenId);
        }
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
     * from {@link DecoderInputBuilder#buildInGraphCausalMask}, but positions
     * actualPrefillLen..prefillSeqLen-1 have ALL-MASK rows.</p>
     *
     * @param actualLen  real token count (un-padded)
     * @param paddedLen  padded prefill length (= maxPrefillLength)
     * @param maxKvLen   total KV buffer size (paddedLen + maxNewTokens)
     * @param dtype      mask data type (FLOAT or HALF)
     * @return attention bias [1, 1, paddedLen, maxKvLen]
     */
    private static INDArray buildPaddedPrefillCausalMask(int actualLen, int paddedLen,
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
                // Padding row: fully masked so padding tokens produce zero-contribution logits
                for (int k = 0; k < K; k++) {
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

    private static void closePrefillOutputs(Map<String, INDArray> outputs, String logitsName) {
        for (Map.Entry<String, INDArray> entry : outputs.entrySet()) {
            if (!entry.getKey().equals(logitsName) && entry.getValue() != null) {
                entry.getValue().close();
            }
        }
    }

    /**
     * Sample a token from logits at a specific sequence position using the full sampling pipeline.
     *
     * <p>When {@code sampling.isGreedy()}, performs argmax. Otherwise applies:
     * repetition penalty → temperature scaling → top-k filtering → top-p filtering → multinomial sampling.</p>
     *
     * @param logits         logits tensor of shape [1, seqLen, vocab]
     * @param seqPos         sequence position to sample from
     * @param sampling       sampling configuration
     * @param generatedSoFar previously generated token IDs (for repetition penalty)
     * @param rng            random number generator (ignored for greedy)
     * @return sampled token ID
     */
    private static int sampleToken(INDArray logits, long seqPos, SamplingConfig sampling,
                                   List<Integer> generatedSoFar, Random rng) {
        INDArray slice = logits.get(
                NDArrayIndex.point(0),
                NDArrayIndex.point(seqPos),
                NDArrayIndex.all()).dup();

        if (sampling.isGreedy()) {
            int token = SamplerUtils.argmax(slice);
            slice.close();
            return token;
        }

        // Apply repetition penalty
        if (sampling.hasRepetitionPenalty() && generatedSoFar != null && !generatedSoFar.isEmpty()) {
            int[] prev = generatedSoFar.stream().mapToInt(Integer::intValue).toArray();
            INDArray penalized = SamplerUtils.applyRepetitionPenalty(slice, prev, sampling.getRepetitionPenalty());
            if (penalized != slice) slice.close();
            slice = penalized;
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

        // Convert to probabilities and sample
        INDArray probs = SamplerUtils.softmax(slice);
        int token = SamplerUtils.multinomialSample(probs, rng);
        probs.close();
        slice.close();
        return token;
    }

    private GenerationResult buildResult(List<Integer> generatedTokens, int[] promptTokenIds,
                                          Set<Integer> stopTokenIds, long startTime, long firstTokenMs) {
        return buildResult(generatedTokens, promptTokenIds, stopTokenIds, startTime, firstTokenMs, 0);
    }

    private GenerationResult buildResult(List<Integer> generatedTokens, int[] promptTokenIds,
                                          Set<Integer> stopTokenIds, long startTime, long firstTokenMs,
                                          double steadyStateTokPerSec) {
        long endTime = System.currentTimeMillis();
        long timeMs = endTime - startTime;
        int[] tokenIds = generatedTokens.stream().mapToInt(Integer::intValue).toArray();
        String text = tokenizer.decode(tokenIds, false);
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
            return generateNative(prefillEmbeddings, promptTokenIds, maxNewTokens);
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
                    // Snapshot ext inputs BEFORE clearing node outputs — these are the
                    // arrays whose device pointers the native plan tracks for stability.
                    frozenExtInputSnapshot = existingExecutor.getExternalInputsSnapshot();
                    DynamicShapePlan frozenPlan = existingExecutor.getCurrentPlan();
                    frozenExtInputKeys = frozenPlan != null ? frozenPlan.getExternalInputKeys() : null;
                    reusingFrozenPlan = frozenExtInputSnapshot != null && frozenExtInputSnapshot.length > 0
                            && frozenExtInputKeys != null
                            && frozenExtInputKeys.length == frozenExtInputSnapshot.length;
                    existingSession.clearNodeOutputsOnly();
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
                }
            }
        }

        // Resolve EOS token and stop tokens — add eosTokenId unconditionally.
        // The buildStopTokenIds() helper suppresses tokens < 100, but SmolDocling
        // uses eosTokenId=2 which is valid. The native AutoregressiveDecode op
        // needs both EOS and <end_of_utterance> in the stop set.
        int eosTokenId = tokenizer.getEosTokenId();
        Set<Integer> stopTokenIds = new HashSet<>();
        stopTokenIds.add(eosTokenId);
        Integer endOfUtteranceTokenId = tokenizer.getTokenId("<end_of_utterance>");
        if (endOfUtteranceTokenId != null) {
            stopTokenIds.add(endOfUtteranceTokenId);
        }
        if (config.getAdditionalStopTokenIds() != null) {
            stopTokenIds.addAll(config.getAdditionalStopTokenIds());
        }

        // Resolve sampling config
        SamplingConfig sampling = config.getSamplingConfig() != null
                ? config.getSamplingConfig() : SamplingConfig.defaultConfig();

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

        long maxKvLen = prefillSeqLen + maxNewTokens;
        // Cap to configured KV cache length to keep buffer shapes stable across
        // calls with different maxNewTokens — avoids plan recompilation.
        int kvCap = config.getMaxKvCacheLength();
        if (kvCap > 0 && maxKvLen > kvCap) {
            maxKvLen = kvCap;
            // Clamp maxNewTokens so the decode loop doesn't run past the buffer
            maxNewTokens = (int) (maxKvLen - prefillSeqLen);
            if (maxNewTokens < 1) maxNewTokens = 1;
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

        Map<String, INDArray> prefillInputMap = DecoderInputBuilder.buildDecoderInputMap(
                ioConfig, decoderInputNames, decoder,
                effectiveEmbeddings, currentInputIds,
                0, prefillSeqLen,
                null, maxKvLen, 0,
                false, hiddenSize,
                reusableInputs, dspActive);

        Map<String, INDArray> prefillOutputs = decoder.output(
                prefillInputMap, allOutputNames.toArray(new String[0]));

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

        // Sample first token from prefill logits — Java-side argmax.
        // When fixedBuffers is active, sample from actualPrefillLen-1, not the end.
        INDArray prefillLogits = prefillOutputs.get(ioConfig.getLogitsOutputName());
        // Sample the LAST-token logits. The decoder may export full/padded logits
        // [1, N>=actualPrefillLen, vocab] (real last token at actualPrefillLen-1, before any
        // fixedBuffers padding) OR last-position-only [1, 1, vocab] (optimized use_cache ONNX
        // exports — the only valid index is 0). Clamp to the actual logits seq dim: without it,
        // point(actualPrefillLen-1) on a size-1 dim is an UNCHECKED OOB read (BaseNDArray.get
        // has no bounds check) → reads position-0-adjacent GPU memory → garbage first-token
        // logits → argmax = token 11126 "User", making the model ignore the image.
        int logitsSamplePos = Math.min(actualPrefillLen - 1, (int) prefillLogits.shape()[1] - 1);
        INDArray firstLogitsSlice = prefillLogits.get(NDArrayIndex.point(0),
                NDArrayIndex.point(logitsSamplePos),
                NDArrayIndex.all()).dup();
        float[] firstLogitValues = firstLogitsSlice.data().asFloat();
        int firstTokenId = 0;
        float firstMaxVal = firstLogitValues[0];
        for (int j = 1; j < firstLogitValues.length; j++) {
            if (firstLogitValues[j] > firstMaxVal) {
                firstMaxVal = firstLogitValues[j];
                firstTokenId = j;
            }
        }
        firstLogitsSlice.close();
        log.info("[Native] Prefill firstTokenId={} maxVal={} logitsShape={} samplePos={} (actualPrefillLen={})",
                firstTokenId, firstMaxVal,
                java.util.Arrays.toString(prefillLogits.shape()), logitsSamplePos, actualPrefillLen);
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

        // Sample second token — Java-side argmax
        INDArray decodeLogits = decodeOutputs.get(ioConfig.getLogitsOutputName());
        INDArray secondLogitsSlice = decodeLogits.get(NDArrayIndex.point(0),
                NDArrayIndex.point(0),
                NDArrayIndex.all()).dup();
        float[] secondLogitValues = secondLogitsSlice.data().asFloat();
        int secondTokenId = 0;
        float secondMaxVal = secondLogitValues[0];
        for (int j = 1; j < secondLogitValues.length; j++) {
            if (secondLogitValues[j] > secondMaxVal) {
                secondMaxVal = secondLogitValues[j];
                secondTokenId = j;
            }
        }


        secondLogitsSlice.close();
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
            String text = tokenizer.decode(tokenIds, false);
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
        }

        long decodeLoopEnd = System.currentTimeMillis();
        long decodeLoopMs = decodeLoopEnd - decodeLoopStart;
        int decodeSteps = allTokens.size() - 1;  // exclude first token (from prefill)
        double tokPerSec = decodeSteps > 0 && decodeLoopMs > 0
                ? (decodeSteps * 1000.0 / decodeLoopMs) : 0;
        float lateSteadyTokPerSec = nativeTimingInfo != null && nativeTimingInfo.length() > 5
                ? nativeTimingInfo.getFloat(5) : (float) tokPerSec;

        int[] tokenIds = allTokens.stream().mapToInt(Integer::intValue).toArray();
        String text = tokenizer.decode(tokenIds, false);
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
        // Tokenize and embed
        String effectivePrompt = prompt;
        boolean addSpecialTokens = true;
        boolean promptAlreadyFormatted = prompt.contains("<|im_start|>") || prompt.contains("[INST]");
        String chatTemplateText = resolveChatTemplate();
        if (chatTemplateText != null && !chatTemplateText.isEmpty() && !promptAlreadyFormatted) {
            List<ChatTemplate.Message> messages = new ArrayList<>();
            messages.add(ChatTemplate.Message.user(prompt));
            effectivePrompt = new ChatTemplate(chatTemplateText, tokenizer.getBosToken(), tokenizer.getEosToken())
                    .apply(messages, true);
            addSpecialTokens = false;
        } else if (promptAlreadyFormatted) {
            addSpecialTokens = false;
        }

        int[] promptTokenIds = tokenizer.encode(effectivePrompt, addSpecialTokens).getIds();
        if (promptTokenIds == null || promptTokenIds.length == 0) {
            throw new IllegalArgumentException("Prompt encoding produced no tokens");
        }

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
     * Release all resources held by this pipeline.
     *
     * <p>Closes models that were loaded from paths (owned by the pipeline).
     * Pre-loaded models passed via the config are NOT closed -- the caller
     * retains ownership.</p>
     */
    @Override
    public void close() {
        if (ownsDecoder && decoder != null) {
            try {
                SameDiffMemoryUtils.freeModelArrays(decoder);
                decoder.close();
            } catch (Exception e) {
                log.warn("Error closing decoder: {}", e.getMessage());
            }
        }
        if (ownsEmbedTokens && embedTokens != null) {
            try {
                SameDiffMemoryUtils.freeModelArrays(embedTokens);
                embedTokens.close();
            } catch (Exception e) {
                log.warn("Error closing embedTokens: {}", e.getMessage());
            }
        }
        if (ownsDraftDecoder && draftDecoder != null) {
            try {
                SameDiffMemoryUtils.freeModelArrays(draftDecoder);
                draftDecoder.close();
            } catch (Exception e) {
                log.warn("Error closing draftDecoder: {}", e.getMessage());
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
            log.info("Extracted embedding table: shape={}", java.util.Arrays.toString(embeddingTable.shape()));
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
