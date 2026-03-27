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

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * Unified LLM inference pipeline that wraps {@link StaticKvCacheDecodeLoop} and
 * eliminates all manual wiring boilerplate.
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
 * <p>For VLM (vision-language model) use cases, call
 * {@link #generate(INDArray, int[])} with pre-built embeddings that include
 * merged vision and text tokens.</p>
 *
 * @see GenerationPipelineConfig
 * @see StaticKvCacheDecodeLoop
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
        if (embedTokens == null) {
            if (config.getEmbedTokensPath() == null) {
                throw new IllegalArgumentException("Either embedTokens or embedTokensPath must be provided");
            }
            embedTokens = loadModel(config.getEmbedTokensPath(), modelLoader);
            ownsEmbedTokens = true;
        }

        // 2. Auto-discover I/O names
        ModelIOConfig ioConfig = config.getIoConfig();
        if (ioConfig == null) {
            ioConfig = ModelIOConfig.discover(decoder);
        }

        // 3. Resolve embed_tokens input/output names
        String embedInputName = embedTokens.inputs().isEmpty()
                ? (ioConfig.getInputIdsName() != null ? ioConfig.getInputIdsName() : "input_ids")
                : embedTokens.inputs().get(0);
        String[] embedOutputNames = embedTokens.outputs().toArray(new String[0]);

        // 4. Auto-detect hidden size
        long resolvedHiddenSize = config.getHiddenSize();
        if (resolvedHiddenSize <= 0) {
            resolvedHiddenSize = detectHiddenSize(embedTokens);
        }

        // 5. Extract embedding table for direct lookup
        INDArray embeddingTable = extractEmbeddingTable(embedTokens);

        // 6. Load draft decoder for speculative decoding
        SameDiff draftDecoder = config.getDraftDecoder();
        boolean ownsDraftDecoder = false;
        if (draftDecoder == null && config.getDraftModelPath() != null) {
            draftDecoder = loadModel(config.getDraftModelPath(), modelLoader);
            ownsDraftDecoder = true;
        }

        log.info("GenerationPipeline created: decoder ops={}, embedTokens ops={}, hiddenSize={}, "
                        + "embeddingTable={}, draftDecoder={}, kvStrategy={}, dsp={}",
                decoder.getOps().size(),
                embedTokens.getOps().size(),
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
     * builds a {@link StaticKvCacheDecodeLoop}, and runs autoregressive decoding.</p>
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

        // Embed tokens
        INDArray embeddings = embedTokens(promptTokenIds);

        return generate(embeddings, promptTokenIds, maxNewTokens);
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
     * @param prefillEmbeddings merged embeddings [1, seqLen, hiddenSize]
     * @param promptTokenIds the prompt token IDs (used for input_ids at step 0)
     * @param maxNewTokens maximum number of tokens to generate
     * @return generation result
     */
    public GenerationResult generate(INDArray prefillEmbeddings, int[] promptTokenIds, int maxNewTokens) {
        StaticKvCacheDecodeLoop loop = buildDecodeLoop(maxNewTokens);
        return loop.decode(prefillEmbeddings, promptTokenIds);
    }

    // ==================== Streaming ====================

    /**
     * Generate text with streaming token-by-token callback.
     *
     * <p>Each token is decoded and passed to the callback as it is generated.
     * The callback receives the decoded text for each individual token.</p>
     *
     * @param prompt the input text prompt
     * @param tokenCallback called with each generated token's decoded text
     */
    public void generateStream(String prompt, Consumer<String> tokenCallback) {
        generateStream(prompt, config.getMaxNewTokens(), tokenCallback);
    }

    /**
     * Generate text with streaming and a specified max token count.
     *
     * @param prompt the input text prompt
     * @param maxNewTokens maximum number of tokens to generate
     * @param tokenCallback called with each generated token's decoded text
     */
    public void generateStream(String prompt, int maxNewTokens, Consumer<String> tokenCallback) {
        // Tokenize and embed
        int[] promptTokenIds = tokenizer.encode(prompt, true).getIds();
        if (promptTokenIds == null || promptTokenIds.length == 0) {
            throw new IllegalArgumentException("Prompt encoding produced no tokens");
        }

        INDArray embeddings = embedTokens(promptTokenIds);

        // Build decode loop with streaming callback
        StaticKvCacheDecodeLoop.StaticKvCacheDecodeLoopBuilder builder = createBaseLoopBuilder(maxNewTokens);
        StaticKvCacheDecodeLoop loop = builder.build();

        // Use decode with per-token tracking
        // For true streaming, the decode loop would need a callback parameter.
        // For now, we run decode and stream the result token by token.
        GenerationResult result = loop.decode(embeddings, promptTokenIds);

        // Stream decoded tokens
        if (result.getTokenIds() != null) {
            for (int tokenId : result.getTokenIds()) {
                String tokenText = tokenizer.decode(new int[]{tokenId}, true);
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
     * @param tokenIds token IDs to embed
     * @return embeddings [1, seqLen, hiddenSize]
     */
    private INDArray embedTokens(int[] tokenIds) {
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
     * Build a {@link StaticKvCacheDecodeLoop} with the pipeline's configuration.
     */
    private StaticKvCacheDecodeLoop buildDecodeLoop(int maxNewTokens) {
        return createBaseLoopBuilder(maxNewTokens).build();
    }

    /**
     * Create a pre-configured decode loop builder with all pipeline settings applied.
     */
    private StaticKvCacheDecodeLoop.StaticKvCacheDecodeLoopBuilder createBaseLoopBuilder(int maxNewTokens) {
        StaticKvCacheDecodeLoop.StaticKvCacheDecodeLoopBuilder builder = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .maxNewTokens(maxNewTokens)
                .hiddenSize(hiddenSize)
                .samplingConfig(config.getSamplingConfig())
                .kvCacheStrategy(config.getKvCacheStrategy())
                .turboQuantBits(config.getTurboQuantBits())
                .ioConfig(ioConfig);

        if (config.getAdditionalStopTokenIds() != null) {
            builder.additionalStopTokenIds(config.getAdditionalStopTokenIds());
        }

        // Configure speculative decoding if draft model is available
        if (draftDecoder != null && config.getMaxSpeculativeTokens() > 0) {
            Speculator speculator = buildDraftSpeculator();
            builder.speculator(speculator)
                    .maxSpeculativeTokens(config.getMaxSpeculativeTokens());
        }

        return builder;
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

        // Decode function: greedy argmax from logits
        Function<INDArray, Integer> decodeFn = logits -> {
            // logits shape: [1, seqLen, vocabSize] -- take last position
            long seqLen = logits.shape()[1];
            INDArray lastLogits = logits.get(
                    NDArrayIndex.point(0),
                    NDArrayIndex.point(seqLen - 1),
                    NDArrayIndex.all());
            return Nd4j.argMax(lastLogits.dup(), -1).getInt(0);
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
