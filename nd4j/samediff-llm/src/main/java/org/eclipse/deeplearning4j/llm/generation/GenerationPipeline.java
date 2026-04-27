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
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.common.config.ND4JSystemProperties;

import org.nd4j.linalg.api.ops.impl.transforms.custom.AutoregressiveDecode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.bytedeco.javacpp.Pointer;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
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

        // Enable DSP auto-compile on models if not disabled
        enableDspIfConfigured(decoder, "decoder");
        if (embedTokens != null) {
            enableDspIfConfigured(embedTokens, "embedTokens");
        }
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

        // 1. Load or use pre-loaded models
        ModelLoader modelLoader = config.getModelLoader();
        SameDiff decoder = config.getDecoder();
        boolean ownsDecoder = false;
        if (decoder == null) {
            if (config.getDecoderPath() == null) {
                throw new IllegalArgumentException("Either decoder or decoderPath must be provided");
            }
            decoder = loadModel(config.getDecoderPath(), modelLoader);
            ownsDecoder = true;
        }

        SameDiff embedTokens = config.getEmbedTokens();
        boolean ownsEmbedTokens = false;
        if (embedTokens == null && config.getEmbedTokensPath() != null) {
            embedTokens = loadModel(config.getEmbedTokensPath(), modelLoader);
            ownsEmbedTokens = true;
        }
        // embedTokens may be null here — single-model mode uses decoder for embeddings

        // 2. Auto-discover I/O names
        ModelIOConfig ioConfig = config.getIoConfig();
        if (ioConfig == null) {
            ioConfig = ModelIOConfig.discover(decoder);
        }

        // 3. Resolve embed_tokens input/output names
        SameDiff embedSource = embedTokens != null ? embedTokens : decoder;
        String embedInputName = embedSource.inputs().isEmpty()
                ? (ioConfig.getInputIdsName() != null ? ioConfig.getInputIdsName() : "input_ids")
                : embedSource.inputs().get(0);
        String[] embedOutputNames = embedSource.outputs().toArray(new String[0]);

        // 4. Auto-detect hidden size (use embedTokens if available, else decoder)
        long resolvedHiddenSize = config.getHiddenSize();
        if (resolvedHiddenSize <= 0) {
            resolvedHiddenSize = detectHiddenSize(embedSource);
        }

        // 5. Extract embedding table for direct lookup (from embedTokens or decoder)
        INDArray embeddingTable = extractEmbeddingTable(embedSource);

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

        return new GenerationPipeline(
                decoder, ownsDecoder,
                embedTokens, ownsEmbedTokens,
                config.getTokenizer(),
                ioConfig,
                embeddingTable,
                resolvedHiddenSize,
                embedInputName, embedOutputNames,
                draftDecoder, ownsDraftDecoder,
                config);
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
        // Tokenize
        int[] promptTokenIds = tokenizer.encode(prompt, true).getIds();
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

        // Check for external KV cache outputs (ONNX pattern: present_* outputs)
        ModelIOConfig.KVCacheNames kvNames = ioConfig.getKvCacheNames();
        if (kvNames == null) {
            kvNames = ModelIOConfig.findKVCacheOutputNames(decoder);
        }
        if (kvNames != null && !kvNames.keyNames.isEmpty()) {
            return generateSimpleWithKvCache(promptTokenIds, maxNewTokens, kvNames);
        }
        return generateSimpleNoKvCache(promptTokenIds, maxNewTokens);
    }

    /**
     * KV-cached autoregressive generation for single-model GGUF.
     *
     * <p>Prefill: full prompt → logits + present KV. Then decode with static KV buffers
     * at seqLen=1 per step. After warmup, shapes are stable and DSP replay kicks in.</p>
     */
    private GenerationResult generateSimpleWithKvCache(int[] promptTokenIds, int maxNewTokens,
                                                        ModelIOConfig.KVCacheNames kvNames) {
        long startTime = System.currentTimeMillis();

        int eosTokenId = tokenizer.getEosTokenId();
        Set<Integer> stopTokenIds = new HashSet<>();
        stopTokenIds.add(eosTokenId);
        if (config.getAdditionalStopTokenIds() != null) {
            stopTokenIds.addAll(config.getAdditionalStopTokenIds());
        }

        String inputIdsName = ioConfig.getInputIdsName() != null ? ioConfig.getInputIdsName() : "input_ids";
        String logitsName = ioConfig.getLogitsOutputName() != null ? ioConfig.getLogitsOutputName() : "logits";
        int prefillSeqLen = promptTokenIds.length;
        long maxKvLen = prefillSeqLen + maxNewTokens;
        boolean dspActive = decoder.isDspAutoCompileEnabled();
        List<String> decoderInputNames = decoder.inputs();
        Map<String, INDArray> reusableInputs = new HashMap<>();

        // Build all output names: logits + present KV
        List<String> allOutputNames = new ArrayList<>();
        allOutputNames.add(logitsName);
        allOutputNames.addAll(kvNames.keyNames);
        allOutputNames.addAll(kvNames.valueNames);

        // ══════════════════════════════════════════════════════════════════════
        // PREFILL: run decoder with full prompt to get initial logits + KV
        // ══════════════════════════════════════════════════════════════════════
        INDArray prefillInputIds = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, prefillSeqLen).castTo(DataType.INT64);

        Map<String, INDArray> prefillInputMap = DecoderInputBuilder.buildDecoderInputMap(
                ioConfig, decoderInputNames, decoder,
                null, prefillInputIds,
                0, prefillSeqLen,
                null, maxKvLen, 0,
                false, hiddenSize,
                reusableInputs, dspActive);

        Map<String, INDArray> prefillOutputs = decoder.output(
                prefillInputMap, allOutputNames.toArray(new String[0]));

        // Sample first token from prefill logits
        INDArray prefillLogits = prefillOutputs.get(logitsName);
        int firstToken = javaArgmax(prefillLogits, prefillLogits.shape()[1] - 1);
        prefillLogits.close();
        prefillInputIds.close();

        List<Integer> generatedTokens = new ArrayList<>();
        generatedTokens.add(firstToken);
        long firstTokenMs = System.currentTimeMillis() - startTime;

        if (stopTokenIds.contains(firstToken)) {
            closeKvOutputs(prefillOutputs, kvNames, logitsName);
            return buildResult(generatedTokens, promptTokenIds, stopTokenIds, startTime, firstTokenMs);
        }

        // ══════════════════════════════════════════════════════════════════════
        // PAD KV caches to static size for fixed-shape decode
        // ══════════════════════════════════════════════════════════════════════
        Map<String, INDArray> staticKvBuffers = new LinkedHashMap<>();
        for (String keyName : kvNames.keyNames) {
            INDArray presentKv = prefillOutputs.get(keyName);
            INDArray padded = padKvToStaticSize(presentKv, maxKvLen);
            String inputName = ioConfig.presentToInputName(keyName);
            staticKvBuffers.put(inputName, padded);
            presentKv.close();
        }
        for (String valName : kvNames.valueNames) {
            INDArray presentKv = prefillOutputs.get(valName);
            INDArray padded = padKvToStaticSize(presentKv, maxKvLen);
            String inputName = ioConfig.presentToInputName(valName);
            staticKvBuffers.put(inputName, padded);
            presentKv.close();
        }

        // ══════════════════════════════════════════════════════════════════════
        // DECODE LOOP: seqLen=1 per step with static KV buffers
        // ══════════════════════════════════════════════════════════════════════
        int currentToken = firstToken;
        long steadyStart = System.currentTimeMillis();
        int steadyTokens = 0;

        for (int step = 1; step < maxNewTokens; step++) {
            long cachePos = prefillSeqLen + step - 1;

            INDArray decodeInputIds = Nd4j.createFromArray(new int[]{currentToken})
                    .reshape(1, 1).castTo(DataType.INT64);

            Map<String, INDArray> decodeInputMap = DecoderInputBuilder.buildDecoderInputMap(
                    ioConfig, decoderInputNames, decoder,
                    null, decodeInputIds,
                    cachePos, 1,
                    staticKvBuffers, maxKvLen, cachePos,
                    true, hiddenSize,
                    reusableInputs, dspActive);

            Map<String, INDArray> decodeOutputs = decoder.output(
                    decodeInputMap, allOutputNames.toArray(new String[0]));

            // Sample next token
            INDArray decodeLogits = decodeOutputs.get(logitsName);
            int nextToken = javaArgmax(decodeLogits, 0);
            decodeLogits.close();
            decodeInputIds.close();

            // Scatter decode KV into static buffers
            for (String keyName : kvNames.keyNames) {
                INDArray presentKv = decodeOutputs.get(keyName);
                INDArray staticBuf = staticKvBuffers.get(ioConfig.presentToInputName(keyName));
                if (presentKv != null && staticBuf != null) {
                    scatterKvToStatic(presentKv, staticBuf, cachePos);
                    presentKv.close();
                }
            }
            for (String valName : kvNames.valueNames) {
                INDArray presentKv = decodeOutputs.get(valName);
                INDArray staticBuf = staticKvBuffers.get(ioConfig.presentToInputName(valName));
                if (presentKv != null && staticBuf != null) {
                    scatterKvToStatic(presentKv, staticBuf, cachePos);
                    presentKv.close();
                }
            }

            generatedTokens.add(nextToken);
            steadyTokens++;
            currentToken = nextToken;
            if (stopTokenIds.contains(nextToken)) break;
        }

        // Cleanup
        for (INDArray kv : staticKvBuffers.values()) kv.close();

        long endTime = System.currentTimeMillis();
        long steadyMs = endTime - steadyStart;
        double steadyTokPerSec = steadyMs > 0 ? (steadyTokens * 1000.0 / steadyMs) : 0;
        return buildResult(generatedTokens, promptTokenIds, stopTokenIds, startTime, firstTokenMs, steadyTokPerSec);
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

        // Reset frozen DSP executor from previous generation (same as generateNative)
        InferenceSession existSession = decoder.getOrCreateSession();
        if (existSession != null) {
            DynamicShapePlanExecutor existExecutor = existSession.getDynamicShapePlanExecutor();
            if (existExecutor != null && existExecutor.isShapesFrozen()) {
                log.info("[Lifecycle] Resetting frozen DSP executor for new GGUF generation");
                decoder.clearDynamicShapePlanCache();
                decoder.resetSession();
            }
        }

        int eosTokenId = tokenizer.getEosTokenId();
        Set<Integer> stopTokenIds = new HashSet<>();
        stopTokenIds.add(eosTokenId);
        if (config.getAdditionalStopTokenIds() != null) {
            stopTokenIds.addAll(config.getAdditionalStopTokenIds());
        }

        SamplingConfig sampling = config.getSamplingConfig() != null
                ? config.getSamplingConfig() : SamplingConfig.greedy();

        String inputIdsName = ioConfig.getInputIdsName() != null ? ioConfig.getInputIdsName() : "input_ids";
        String logitsName = ioConfig.getLogitsOutputName() != null ? ioConfig.getLogitsOutputName() : "logits";
        String posOffsetName = ioConfig.getPositionOffsetName();
        String cachePosName = ioConfig.getCachePositionName();
        String causalMaskName = ioConfig.getCausalMaskName();

        int prefillSeqLen = promptTokenIds.length;
        long maxKvLen = prefillSeqLen + maxNewTokens;
        int numLayers = kvInputNames.keyNames.size();

        // ══════════════════════════════════════════════════════════════════════
        // STEP 1: PREFILL -- full prompt, empty KV cache, extract per-layer K/V
        // ══════════════════════════════════════════════════════════════════════
        INDArray prefillInputIds = Nd4j.createFromArray(promptTokenIds)
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
            prefillInputMap.put(causalMaskName,
                    DecoderInputBuilder.buildInGraphCausalMask(prefillSeqLen, maxKvLen, maskDtype));
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

        // Request outputs: logits + per-layer k_rope_{L} and v_heads_{L}
        List<String> prefillOutputNames = new ArrayList<>();
        prefillOutputNames.add(logitsName);
        for (int i = 0; i < numLayers; i++) {
            int layerIdx = extractLayerIndex(kvInputNames.keyNames.get(i));
            prefillOutputNames.add("k_rope_" + layerIdx);
            prefillOutputNames.add("v_heads_" + layerIdx);
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

        // Sample first token from prefill logits
        INDArray prefillLogits = prefillOutputs.get(logitsName);
        if (prefillLogits == null) {
            throw new RuntimeException("[GGUF-KV] Prefill logits '" + logitsName + "' not found in outputs: " + prefillOutputs.keySet());
        }
        log.info("[GGUF-KV] Prefill logits shape: {}", java.util.Arrays.toString(prefillLogits.shape()));
        int firstTokenId = javaArgmax(prefillLogits, prefillLogits.shape()[1] - 1);
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

            // kRoped/vHeads shape: [batch, prefillLen, numKVHeads, headDim]
            long[] kvShape = kRoped.shape();
            long batch = kvShape[0], numKVHeads = kvShape[2], headDim = kvShape[3];

            // Create full-size zero buffer and write prefill data at positions 0..prefillLen-1
            INDArray keyBuf = Nd4j.zeros(kvDtype, batch, maxKvLen, numKVHeads, headDim);
            keyBuf.get(NDArrayIndex.all(), NDArrayIndex.interval(0, prefillSeqLen),
                    NDArrayIndex.all(), NDArrayIndex.all()).assign(kRoped);
            staticKvBuffers.put(kvInputNames.keyNames.get(i), keyBuf);

            INDArray valBuf = Nd4j.zeros(kvDtype, batch, maxKvLen, numKVHeads, headDim);
            valBuf.get(NDArrayIndex.all(), NDArrayIndex.interval(0, prefillSeqLen),
                    NDArrayIndex.all(), NDArrayIndex.all()).assign(vHeads);
            staticKvBuffers.put(kvInputNames.valueNames.get(i), valBuf);

            kRoped.close();
            vHeads.close();
        }
        Nd4j.getExecutioner().commit();

        // ══════════════════════════════════════════════════════════════════════
        // STEP 3: Warmup decode step -- compile DSP plan for decode shapes
        // ══════════════════════════════════════════════════════════════════════
        INDArray decodeInputIds = Nd4j.createFromArray(new int[]{firstTokenId})
                .reshape(1, 1).castTo(DataType.INT64);

        INDArray decodeCausalMask = null;
        if (causalMaskName != null && decoder.hasVariable(causalMaskName)) {
            decodeCausalMask = DecoderInputBuilder.buildInGraphDecodeMask(prefillSeqLen, maxKvLen, maskDtype);
        }
        INDArray decodePositionOffset = null;
        if (posOffsetName != null && decoder.hasVariable(posOffsetName)) {
            decodePositionOffset = Nd4j.scalar(DataType.INT64, prefillSeqLen);
        }
        INDArray decodeCachePosition = null;
        if (cachePosName != null && decoder.hasVariable(cachePosName)) {
            decodeCachePosition = Nd4j.scalar(DataType.INT64, prefillSeqLen);
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

        List<String> decodeOutputNames = new ArrayList<>();
        decodeOutputNames.add(logitsName);

        log.info("[GGUF-KV] STEP 3: warmup decode with {} KV buffers, {} inputs",
                staticKvBuffers.size(), decodeInputMap.size());

        Map<String, INDArray> decodeOutputs;
        try {
            decodeOutputs = decoder.output(
                    decodeInputMap, decodeOutputNames.toArray(new String[0]));
        } catch (Exception e) {
            log.error("[GGUF-KV] STEP 3 warmup decode failed", e);
            throw e;
        }

        log.info("[GGUF-KV] STEP 3 complete, outputs: {}", decodeOutputs.keySet());
        INDArray decodeLogits = decodeOutputs.get(logitsName);
        int secondTokenId = javaArgmax(decodeLogits, 0);
        decodeLogits.close();

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

            List<Integer> tokens = new ArrayList<>();
            tokens.add(firstTokenId);
            if (!stopTokenIds.contains(firstTokenId)) tokens.add(secondTokenId);
            return buildResult(tokens, promptTokenIds, stopTokenIds, startTime, firstTokenMs);
        }

        if (executor.getCurrentPlan() != null) {
            executor.setShapesFrozen(true);
            log.info("[Perf] GGUF shapes frozen after warmup decode (planPhase={} pointersStable={})",
                    executor.getPlanPhase(), executor.arePointersStable());
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

        // ══════════════════════════════════════════════════════════════════════
        // STEP 5: Execute native AutoregressiveDecode op
        // ══════════════════════════════════════════════════════════════════════
        int remainingTokens = maxNewTokens - 2;

        decodeInputIds.putScalar(new long[]{0, 0}, secondTokenId);
        if (decodeCausalMask != null) {
            decodeCausalMask.putScalar(new long[]{0, 0, 0, prefillSeqLen}, 0.0f);
        }
        if (decodePositionOffset != null) {
            decodePositionOffset.putScalar(new long[]{}, (long)(prefillSeqLen + 1));
        }
        if (decodeCachePosition != null) {
            decodeCachePosition.putScalar(new long[]{}, (long)(prefillSeqLen + 1));
        }

        Nd4j.getExecutioner().commit();
        decodeInputIds.syncToDevice();
        if (decodeCausalMask != null) decodeCausalMask.syncToDevice();
        if (decodePositionOffset != null) decodePositionOffset.syncToDevice();
        if (decodeCachePosition != null) decodeCachePosition.syncToDevice();
        for (INDArray kvBuf : staticKvBuffers.values()) kvBuf.syncToDevice();

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
                    remainingTokens, eosTokenId, numKvPairs,
                    prefillSeqLen + 1,
                    sampling.isGreedy() ? 0.0 : sampling.getTemperature(),
                    sampling.isGreedy() ? 0 : sampling.getTopK(),
                    sampling.isGreedy() ? 0.0 : sampling.getTopP(),
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
            long endTime = System.currentTimeMillis();
            long timeMs = endTime - startTime;
            float tokPerSec = nativeTimingInfo.getFloat(2);
            boolean hitEos = tokenIds.length > 0 && stopTokenIds.contains(tokenIds[tokenIds.length - 1]);

            dummyEmbeddings.close();
            dummyEmbTable.close();
            decodeInputIds.close();
            if (decodeCausalMask != null) decodeCausalMask.close();
            if (decodePositionOffset != null) decodePositionOffset.close();
            if (decodeCachePosition != null) decodeCachePosition.close();
            for (INDArray kv : staticKvBuffers.values()) kv.close();

            return GenerationResult.builder()
                    .text(text).tokenIds(tokenIds)
                    .generatedTokenCount(tokenIds.length).promptTokenCount(prefillSeqLen)
                    .totalTokenCount(prefillSeqLen + tokenIds.length)
                    .finishReason(hitEos ? GenerationResult.FinishReason.EOS : GenerationResult.FinishReason.MAX_TOKENS)
                    .generationTimeMs(timeMs)
                    .tokensPerSecond(timeMs > 0 ? (tokenIds.length * 1000.0 / timeMs) : 0)
                    .steadyStateTokensPerSecond(tokPerSec)
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

    /** Close non-logits prefill outputs. */
    private static void closePrefillOutputs(Map<String, INDArray> outputs, String logitsName) {
        for (Map.Entry<String, INDArray> entry : outputs.entrySet()) {
            if (!entry.getKey().equals(logitsName) && entry.getValue() != null) {
                entry.getValue().close();
            }
        }
    }

    /**
     * Fallback: no KV cache available, shapes grow each step. No replay possible.
     */
    private GenerationResult generateSimpleNoKvCache(int[] promptTokenIds, int maxNewTokens) {
        long startTime = System.currentTimeMillis();

        int eosTokenId = tokenizer.getEosTokenId();
        Set<Integer> stopTokenIds = new HashSet<>();
        stopTokenIds.add(eosTokenId);
        if (config.getAdditionalStopTokenIds() != null) {
            stopTokenIds.addAll(config.getAdditionalStopTokenIds());
        }

        String inputIdsName = ioConfig.getInputIdsName() != null ? ioConfig.getInputIdsName() : "input_ids";
        String logitsName = ioConfig.getLogitsOutputName() != null ? ioConfig.getLogitsOutputName() : "logits";

        int[] currentIds = promptTokenIds;
        List<Integer> generatedTokens = new ArrayList<>();
        long firstTokenMs = 0;

        for (int step = 0; step < maxNewTokens; step++) {
            INDArray inputIds = Nd4j.createFromArray(currentIds)
                    .reshape(1, currentIds.length).castTo(DataType.INT64);

            Map<String, INDArray> outputs = decoder.output(
                    Map.of(inputIdsName, inputIds), logitsName);

            INDArray logits = outputs.get(logitsName);
            int nextToken = javaArgmax(logits, logits.shape()[1] - 1);
            logits.close();
            inputIds.close();

            if (step == 0) {
                firstTokenMs = System.currentTimeMillis() - startTime;
            }

            generatedTokens.add(nextToken);
            if (stopTokenIds.contains(nextToken)) break;

            int[] newIds = new int[currentIds.length + 1];
            System.arraycopy(currentIds, 0, newIds, 0, currentIds.length);
            newIds[currentIds.length] = nextToken;
            currentIds = newIds;
        }

        return buildResult(generatedTokens, promptTokenIds, stopTokenIds, startTime, firstTokenMs);
    }

    /** Java-side argmax at a specific sequence position in [1, seqLen, vocab] logits. */
    private static int javaArgmax(INDArray logits, long seqPos) {
        INDArray slice = logits.get(
                NDArrayIndex.point(0),
                NDArrayIndex.point(seqPos),
                NDArrayIndex.all()).dup();
        float[] values = slice.data().asFloat();
        int best = 0;
        float bestVal = values[0];
        for (int j = 1; j < values.length; j++) {
            if (values[j] > bestVal) {
                bestVal = values[j];
                best = j;
            }
        }
        slice.close();
        return best;
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
        return generateNative(prefillEmbeddings, promptTokenIds, maxNewTokens);
    }

    /**
     * Generate text using the native {@code autoregressive_decode} C++ op.
     *
     * <p>The native op runs the full decode loop in C++, eliminating per-step
     * Java to C++ round-trips. Steps:
     * <ol>
     *   <li>Execute a prefill step via decoder.output() to (a) produce initial KV caches
     *       and (b) trigger DSP compilation so we can get the native plan handle.</li>
     *   <li>Pad KV caches to static size, prepare decode-step inputs (seqLen=1).</li>
     *   <li>Execute the first decode step to warm up decode-shape compilation.</li>
     *   <li>Extract the native plan handle and external input metadata.</li>
     *   <li>Invoke AutoregressiveDecode op — the C++ side does the rest:
     *       plan.execute() per step, token sampling, KV scatter, embedding lookup,
     *       mask/pos/inputIds updates, stop condition checking.</li>
     * </ol>
     * </p>
     */
    private GenerationResult generateNative(INDArray prefillEmbeddings, int[] promptTokenIds, int maxNewTokens) {
        long startTime = System.currentTimeMillis();

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
                decoder.clearDynamicShapePlanCache();
                decoder.resetSession();
            }
        }

        // Resolve EOS token and stop tokens
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
                ? config.getSamplingConfig() : SamplingConfig.greedy();

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

        int prefillSeqLen = promptTokenIds.length;
        long maxKvLen = prefillSeqLen + maxNewTokens;
        boolean dspActive = decoder.isDspAutoCompileEnabled();
        List<String> decoderInputNames = decoder.inputs();
        Map<String, INDArray> reusableInputs = new HashMap<>();

        // ══════════════════════════════════════════════════════════════════════
        // STEP 1: Prefill — run decoder with full prompt to get initial KV caches
        //         and trigger DSP plan compilation.
        // ══════════════════════════════════════════════════════════════════════
        INDArray currentInputIds = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, prefillSeqLen).castTo(DataType.INT64);

        Map<String, INDArray> prefillInputMap = DecoderInputBuilder.buildDecoderInputMap(
                ioConfig, decoderInputNames, decoder,
                prefillEmbeddings, currentInputIds,
                0, prefillSeqLen,
                null, maxKvLen, 0,
                false, hiddenSize,
                reusableInputs, dspActive);

        Map<String, INDArray> prefillOutputs = decoder.output(
                prefillInputMap, allOutputNames.toArray(new String[0]));

        // ══════════════════════════════════════════════════════════════════════
        // STEP 2: Pad KV caches to static size and prepare decode-step state
        // ══════════════════════════════════════════════════════════════════════
        Map<String, INDArray> staticKvBuffers = new LinkedHashMap<>();
        for (String keyName : kvNames.keyNames) {
            INDArray presentKv = prefillOutputs.get(keyName);
            INDArray padded = padKvToStaticSize(presentKv, maxKvLen);
            String inputName = ioConfig.presentToInputName(keyName);
            staticKvBuffers.put(inputName, padded);
            presentKv.close();
        }
        for (String valName : kvNames.valueNames) {
            INDArray presentKv = prefillOutputs.get(valName);
            INDArray padded = padKvToStaticSize(presentKv, maxKvLen);
            String inputName = ioConfig.presentToInputName(valName);
            staticKvBuffers.put(inputName, padded);
            presentKv.close();
        }
        Nd4j.getExecutioner().commit();

        // Sample first token from prefill logits — Java-side argmax
        INDArray prefillLogits = prefillOutputs.get(ioConfig.getLogitsOutputName());
        INDArray firstLogitsSlice = prefillLogits.get(NDArrayIndex.point(0),
                NDArrayIndex.point(prefillLogits.shape()[1] - 1),
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
        prefillLogits.close();

        // ══════════════════════════════════════════════════════════════════════
        // STEP 3: Build decode-step arrays ONCE, then execute a warmup step.
        //
        // These arrays have the EXACT shapes the C++ native loop will use.
        // The warmup compiles the DSP plan for these shapes, and the native
        // loop reuses the same plan handle — no shape mismatch, no plan swap.
        // ══════════════════════════════════════════════════════════════════════
        // dup() is MANDATORY: getRow().reshape() returns a VIEW into embeddingTable.
        // assign() at line 642 writes into this buffer — without dup(), that corrupts
        // the persistent weight matrix, making subsequent runs non-deterministic.
        INDArray decodeEmbeddings = embeddingTable.getRow(firstTokenId).reshape(1, 1, hiddenSize).dup();
        INDArray decodeInputIds = Nd4j.createFromArray(new int[]{firstTokenId})
                .reshape(1, 1).castTo(DataType.INT64);

        // Mask dimension: maxKvLen + 1 (past KV positions + current query position).
        // The model's internal attention ops produce [1,1,1,maxKvLen+1] tensors and
        // add them to the causal mask — shapes MUST match.
        long totalSeqLen = maxKvLen + 1;

        // Causal mask: [1, 1, 1, totalSeqLen] FLOAT — MASK_FILL for unfilled positions, 0.0 for filled.
        // Uses PADDED layout (query at totalSeqLen-1), matching DecoderInputBuilder with dspActive=true.
        // This is the layout the old StaticKvCacheDecodeLoop uses and it produces correct output.
        //   1. Fill everything with MASK_FILL (-3.4028235e+38f) = masked
        //   2. Unmask [0..prefillSeqLen] with 0.0f (filled KV positions from prefill)
        // The C++ kernel updates this per step: unmask position currentPosition with 0.0f.
        INDArray decodeCausalMask = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, totalSeqLen);
        decodeCausalMask.assign(ModelIOConfig.MASK_FILL);
        if (prefillSeqLen + 1 > 0) {
            decodeCausalMask.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.interval(0, prefillSeqLen + 1)).assign(0.0f);
        }

        // Attention mask: [1, totalSeqLen] LONG (0/1 values, updated per step by C++ kernel).
        // Uses PADDED layout (query at totalSeqLen-1), matching DecoderInputBuilder with dspActive=true.
        //   1. Valid past KV positions: [0, prefillSeqLen-1] = 1
        //   2. Query position: totalSeqLen-1 = 1
        //   3. Future padding: [prefillSeqLen, totalSeqLen-2] = 0
        INDArray decodeAttentionMask = Nd4j.zeros(DataType.LONG, 1, totalSeqLen);
        decodeAttentionMask.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, prefillSeqLen)).assign(1);
        decodeAttentionMask.putScalar(0, totalSeqLen - 1, 1);

        // Position IDs: [1, 1] INT64
        INDArray decodePosIds = Nd4j.createFromArray(new long[]{prefillSeqLen})
                .reshape(1, 1);

        // ── attn_mask_reformat override ──────────────────────────────────────
        // The model's internal attn_mask_reformat subgraph can produce incorrect
        // masks for multi-token padded static-KV decode (seqLen > 1). For single-
        // token decode the subgraph is correct — the override is only needed for
        // speculative decoding where seqLen = K+1 > 1. Matches FrozenDecodeStep:
        // it only adds the override when seqLen > 1.
        String attnReformatNode = ioConfig.getAttnMaskReformatOutput();
        INDArray decodeAttnMaskReformat = null;
        // Decode embeddings shape: [1, 1, hiddenSize] → seqLen = 1. No override.
        boolean needsAttnOverride = false;
        if (needsAttnOverride && attnReformatNode != null && decoder.hasVariable(attnReformatNode)) {
            decoder.addPlaceholderOverride(attnReformatNode);
            decoder.getVariable(attnReformatNode).setShape(-1, -1, -1, -1);
            // Build initial bias: [1, 1, 1, maskLen] FLOAT.
            //   [0, prefillSeqLen) = 0.0f (already unmasked past KV)
            //   [prefillSeqLen, maxKvLen) = MASK_FILL (future empty KV)
            //   [maxKvLen] = 0.0f (query position)
            decodeAttnMaskReformat = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, totalSeqLen);
            if (prefillSeqLen < maxKvLen) {
                decodeAttnMaskReformat.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(prefillSeqLen, maxKvLen)).assign(ModelIOConfig.MASK_FILL);
            }
        }

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
                scatterKvToStatic(presentKv, staticBuf, prefillSeqLen);
            }
        }
        for (String valName : kvNames.valueNames) {
            INDArray presentKv = decodeOutputs.get(valName);
            INDArray staticBuf = staticKvBuffers.get(ioConfig.presentToInputName(valName));
            if (presentKv != null && staticBuf != null) {
                scatterKvToStatic(presentKv, staticBuf, prefillSeqLen);
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
            if (!stopTokenIds.contains(firstTokenId)) {
                tokens.add(secondTokenId);
            }

            int[] tokenIds = tokens.stream().mapToInt(Integer::intValue).toArray();
            String text = tokenizer.decode(tokenIds, false);
            long endTime = System.currentTimeMillis();
            long timeMs = endTime - startTime;

            for (INDArray kv : staticKvBuffers.values()) kv.close();

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

        // Freeze shapes explicitly after warmup decode, matching the old
        // StaticKvCacheDecodeLoop behavior. This ensures the Java executor
        // and native plan both agree that shapes are frozen, enabling the
        // fastest replay path and stable Triton cache keys.
        if (executor != null && executor.getCurrentPlan() != null) {
            executor.setShapesFrozen(true);
            log.info("[Perf] Shapes frozen after warmup decode (planPhase={} pointersStable={})",
                    executor.getPlanPhase(), executor.arePointersStable());
        }

        // Resolve ext input indices by name
        // The plan's external input keys array tells us which index maps to each named input
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
        // Embeddings: overwrite with secondTokenId embedding
        INDArray secondEmbed = embeddingTable.getRow(secondTokenId).reshape(1, 1, hiddenSize);
        decodeEmbeddings.assign(secondEmbed);
        secondEmbed.close();
        // Input IDs: overwrite with secondTokenId
        decodeInputIds.putScalar(new long[]{0, 0}, secondTokenId);
        // Attention mask: unmask the KV position written by the warmup step.
        // Padded layout: the query stays at totalSeqLen-1. Each step unmasks the
        // KV position that was just written (cachePos = prefillSeqLen after warmup).
        decodeAttentionMask.putScalar(new long[]{0, prefillSeqLen}, 1);
        // Position IDs: advance to prefillSeqLen + 1
        decodePosIds.putScalar(new long[]{0, 0}, prefillSeqLen + 1);
        // Causal mask: unmask the KV position written by the warmup step.
        // Mirrors DecoderInputBuilder delta update: putScalar(cachePos, 0.0f)
        decodeCausalMask.putScalar(new long[]{0, 0, 0, prefillSeqLen}, 0.0f);
        // attn_mask_reformat: unmask the KV position written by the warmup step
        if (decodeAttnMaskReformat != null) {
            decodeAttnMaskReformat.putScalar(new long[]{0, 0, 0, prefillSeqLen}, 0.0f);
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
        // This context is reused across executeNative() calls by the Java executor.
        Pointer contextHandle = executor.getCachedOpContext();
        int numPlanExternalInputs = executor.getCurrentPlan() != null
                ? executor.getCurrentPlan().getExternalInputKeys().length : 0;
        int numPlanOutputs = allOutputNames.size();

        // Execute the native decode op
        if (remainingTokens > 0) {
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
                    kvInputExtIndices,
                    kvOutputIndices,
                    remainingTokens,
                    eosTokenId,
                    numKvPairs,
                    prefillSeqLen + 1,  // current position after 2 warmup steps
                    sampling.isGreedy() ? 0.0 : sampling.getTemperature(),
                    sampling.isGreedy() ? 0 : sampling.getTopK(),
                    sampling.isGreedy() ? 0.0 : sampling.getTopP(),
                    stopTokenIds);

            INDArray[] results = Nd4j.getExecutioner().exec(op);
            INDArray nativeTokenIds = results[0];
            INDArray nativeTokenCount = results[1];
            INDArray nativeTimingInfo = results[2];

            int nativeCount = nativeTokenCount.getInt(0);

            // Combine: firstToken + secondToken + native tokens
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
            long endTime = System.currentTimeMillis();
            long timeMs = endTime - startTime;

            // Use native timing for steady-state metrics
            float totalMs = nativeTimingInfo.getFloat(0);
            float tokPerSec = nativeTimingInfo.getFloat(2);

            boolean hitEos = tokenIds.length > 0 && stopTokenIds.contains(tokenIds[tokenIds.length - 1]);

            // Cleanup
            currentInputIds.close();
            decodeEmbeddings.close();
            decodeInputIds.close();
            decodeCausalMask.close();
            decodeAttentionMask.close();
            decodePosIds.close();
            if (decodeAttnMaskReformat != null) decodeAttnMaskReformat.close();

            for (INDArray kv : staticKvBuffers.values()) kv.close();

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
                    .build();
        } else {
            // Only 2 warmup tokens generated, no remaining to decode natively
            currentInputIds.close();
            decodeEmbeddings.close();
            decodeInputIds.close();
            decodeCausalMask.close();
            decodeAttentionMask.close();
            decodePosIds.close();
            if (decodeAttnMaskReformat != null) decodeAttnMaskReformat.close();

            // Clamp returned tokens to maxNewTokens — the warmup always
            // generates 2 tokens internally, but the caller asked for fewer.
            List<Integer> allTokens = new ArrayList<>();
            if (maxNewTokens >= 1) {
                allTokens.add(firstTokenId);
            }
            if (maxNewTokens >= 2 && !stopTokenIds.contains(firstTokenId)) {
                allTokens.add(secondTokenId);
            }

            int[] tokenIds = allTokens.stream().mapToInt(Integer::intValue).toArray();
            String text = tokenizer.decode(tokenIds, false);
            long endTime = System.currentTimeMillis();
            long timeMs = endTime - startTime;
            boolean hitEos = tokenIds.length > 0 && stopTokenIds.contains(tokenIds[tokenIds.length - 1]);

            for (INDArray kv : staticKvBuffers.values()) kv.close();

            return GenerationResult.builder()
                    .text(text)
                    .tokenIds(tokenIds)
                    .generatedTokenCount(tokenIds.length)
                    .promptTokenCount(prefillSeqLen)
                    .totalTokenCount(prefillSeqLen + tokenIds.length)
                    .finishReason(hitEos ? GenerationResult.FinishReason.EOS : GenerationResult.FinishReason.MAX_TOKENS)
                    .generationTimeMs(timeMs)
                    .tokensPerSecond(timeMs > 0 ? (tokenIds.length * 1000.0 / timeMs) : 0)
                    .steadyStateTokensPerSecond(0)
                    .build();
        }
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
        INDArray padding = Nd4j.zeros(presentKv.dataType(), batch, heads, maxKvLen - seqLen, dim);
        INDArray result = Nd4j.concat(2, presentKv, padding);
        padding.close();
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
        int[] promptTokenIds = tokenizer.encode(prompt, true).getIds();
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

    /**
     * Free all GPU memory held by a vision encoder's model arrays.
     *
     * <p>Call this after vision encoding is complete and the encoder is no longer needed.
     * Clears placeholders, resets sessions, frees all constant/variable arrays, and
     * closes the model. This reclaims the ~5GB+ of GPU memory used by vision encoder weights.</p>
     *
     * @param visionEncoder the vision encoder SameDiff model to free
     */
    public static void freeVisionEncoder(SameDiff visionEncoder) {
        if (visionEncoder == null) {
            return;
        }
        try {
            visionEncoder.clearPlaceholders(true);
            visionEncoder.clearOpInputs();
            visionEncoder.resetSession();
            SameDiffMemoryUtils.freeModelArrays(visionEncoder);
            visionEncoder.close();
            log.info("Vision encoder freed");
        } catch (Exception e) {
            log.warn("Error freeing vision encoder: {}", e.getMessage());
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
            // Direct table lookup -- avoids full SameDiff.output() per call
            INDArray emb = Nd4j.zeros(DataType.FLOAT, 1, tokenIds.length, hiddenSize);
            for (int i = 0; i < tokenIds.length; i++) {
                emb.get(NDArrayIndex.point(0), NDArrayIndex.point(i), NDArrayIndex.all())
                        .assign(embeddingTable.getRow(tokenIds[i]));
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
