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

package org.eclipse.deeplearning4j.vlm.model;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.config.ModelConfig;
import org.eclipse.deeplearning4j.llm.generation.batch.BatchGenerationState;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.SameDiffMemoryUtils;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplerUtils;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.eclipse.deeplearning4j.llm.tokenizer.Encoding;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.vlm.model.patching.SameDiffGraphPatch;
import org.eclipse.deeplearning4j.vlm.model.encoder.VisionEncoderIOConfig;
import org.eclipse.deeplearning4j.vlm.output.protocol.VlmOutputProtocolRegistry;
import org.eclipse.deeplearning4j.vlm.output.protocol.VlmProtocolOutput;
import org.eclipse.deeplearning4j.vlm.output.protocol.VlmProtocolPlan;
import org.eclipse.deeplearning4j.vlm.output.protocol.VlmProtocolRequest;
import org.eclipse.deeplearning4j.vlm.model.encoder.VisionEncoderUtils;
import org.eclipse.deeplearning4j.vlm.model.encoder.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.loading.MultiPartModelLoader;
import org.eclipse.deeplearning4j.vlm.model.loading.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipelineConfig;
import org.eclipse.deeplearning4j.llm.generation.ModelIOConfig;
import org.eclipse.deeplearning4j.llm.generation.kvcache.KvCacheManager;
import org.eclipse.deeplearning4j.llm.generation.kvcache.KvCacheStrategy;
import org.eclipse.deeplearning4j.llm.generation.speculative.NgramSpeculator;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfig;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.memory.ArrayCacheMemoryMgr;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.MultiBackendWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.IOException;
import java.util.IdentityHashMap;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Vision-Language Model wrapper for multi-modal AI.
 *
 * This class provides a unified interface for vision-language models that
 * combine image understanding with text generation. It supports models like:
 * - SmolDocling (document understanding)
 * - LLaVA / LLaVA-Next (visual question answering)
 * - Idefics (interleaved image-text)
 *
 * <p>Models must be in SDZ (SameDiff ZIP) format. To convert from ONNX:</p>
 * <pre>{@code
 * // Import ONNX model and save as SDZ
 * OnnxFrameworkImporter importer = new OnnxFrameworkImporter();
 * SameDiff model = importer.runImport("model.onnx", Collections.emptyMap(), false, false);
 * SDZSerializer.save(model, new File("model.sdz"), false, null);
 * }</pre>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * // Load a SmolDocling model (SDZ format)
 * VisionLanguageModel vlm = VisionLanguageModel.fromDirectory(
 *     new File("SmolDocling-256M-sdz")
 * );
 *
 * // Process a document image
 * INDArray image = vlm.preprocessImage(new File("document.png"));
 * String output = vlm.generate(image, "Convert this document to markdown");
 *
 * vlm.close();
 * }</pre>
 *
 * Adam Gibson
 */
@Slf4j
@Getter
public class VisionLanguageModel implements AutoCloseable {

    private final SameDiff visionEncoder;
    private final SameDiff embedTokens;
    private final SameDiff decoder;
    private final Tokenizer tokenizer;
    private final VLMImagePreprocessor imagePreprocessor;
    private final ModelConfig config;
    private final VlmOutputProtocolRegistry outputProtocols;

    // Device-aware fields for multi-chip support
    @Setter
    private DeviceDescriptor targetDevice;

    @Setter
    private MultiBackendWorkspace workspace;

    private volatile boolean closed = false;

    /** KV cache strategy for autoregressive decoding. Default: STATIC. */
    @Setter
    private KvCacheStrategy kvCacheStrategy = KvCacheStrategy.STATIC;

    /** Maximum KV cache sequence length. 0 means auto-detect. */
    @Setter
    private int maxKvLen = 0;

    /** Optional benchmark config applied after GenerationPipeline construction. */
    @Setter
    private BenchmarkConfig benchmarkConfig;

    /**
     * Cached GenerationPipeline for multi-page decode reuse.
     * Created on first page with fixedBuffers enabled (maxPrefillLength > 0)
     * so the frozen DSP plan is reused across pages without recompilation.
     */
    private GenerationPipeline cachedPipeline;
    private int cachedPromptTokenCount;
    /** Physical padded prefill capacity captured by {@link #cachedPipeline}; zero in dynamic mode. */
    private int cachedMaxPrefillLength;
    private String cachedProtocolKey;

    /** Cached decoder I/O configuration — discovered once, reused across pages. */
    private ModelIOConfig cachedDecoderIOConfig;

    /** External KV cache manager for managed autoregressive decoding. */
    @Setter
    private KvCacheManager externalKvCacheManager;

    /** Discovered vision encoder I/O configuration (input/output variable names). */
    @Getter
    private final VisionEncoderIOConfig visionEncoderIOConfig;

    @Builder
    private VisionLanguageModel(
            SameDiff visionEncoder,
            SameDiff embedTokens,
            SameDiff decoder,
            Tokenizer tokenizer,
            VLMImagePreprocessor imagePreprocessor,
            ModelConfig config,
            VlmOutputProtocolRegistry outputProtocols,
            DeviceDescriptor targetDevice,
            MultiBackendWorkspace workspace,
            VisionEncoderIOConfig visionEncoderIOConfig,
            KvCacheStrategy kvCacheStrategy,
            int maxKvLen,
            KvCacheManager externalKvCacheManager) {
        this.visionEncoder = visionEncoder;
        this.embedTokens = embedTokens;
        this.decoder = decoder;
        this.tokenizer = tokenizer;
        this.imagePreprocessor = imagePreprocessor;
        this.config = config;
        this.outputProtocols = outputProtocols != null ? outputProtocols : VlmOutputProtocolRegistry.fallback();
        this.targetDevice = targetDevice;
        this.workspace = workspace;
        this.kvCacheStrategy = kvCacheStrategy != null ? kvCacheStrategy : KvCacheStrategy.STATIC;
        this.maxKvLen = maxKvLen;
        this.externalKvCacheManager = externalKvCacheManager;
        // Use externally-provided IOConfig if given, otherwise auto-discover
        if (visionEncoderIOConfig != null) {
            this.visionEncoderIOConfig = visionEncoderIOConfig;
            log.info("Using externally-provided VisionEncoderIOConfig: pixelValues={}, primaryOutput={}",
                    visionEncoderIOConfig.getPixelValuesName(), visionEncoderIOConfig.getPrimaryOutputName());
        } else {
            this.visionEncoderIOConfig = visionEncoder != null
                    ? VisionEncoderIOConfig.discover(visionEncoder) : null;
        }

        // Enable DSP auto-compile on all model components if not disabled
        if (visionEncoder != null) enableDspIfConfigured(visionEncoder, "visionEncoder");
        if (decoder != null) enableDspIfConfigured(decoder, "decoder");
        if (embedTokens != null) enableDspIfConfigured(embedTokens, "embedTokens");
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

    /**
     * Load a VLM from a directory containing SDZ model files.
     *
     * <p>Expected directory structure:</p>
     * <pre>
     * model_dir/
     * ├── config.json
     * ├── tokenizer.json
     * ├── preprocessor_config.json
     * ├── vision_encoder.sdz
     * ├── embed_tokens.sdz (optional)
     * └── decoder.sdz
     * </pre>
     *
     * @param modelDir the model directory containing SDZ files
     * @return the loaded model
     * @throws IOException if loading fails
     */
    public static VisionLanguageModel fromDirectory(File modelDir) throws IOException {
        return fromDirectory(modelDir, null);
    }

    /**
     * Load a VLM from a directory with an optional externally-provided IOConfig.
     *
     * @param modelDir the model directory containing SDZ files
     * @param ioConfig external vision encoder IO config, or null to auto-discover
     * @return the loaded model
     * @throws IOException if loading fails
     */
    public static VisionLanguageModel fromDirectory(File modelDir, VisionEncoderIOConfig ioConfig) throws IOException {
        MultiPartModelLoader.LoadedModel loaded = MultiPartModelLoader.load(modelDir);
        return loaded.toVisionLanguageModel(ioConfig);
    }

    /**
     * Load a SmolDocling model from a directory containing SDZ files.
     *
     * <p>SmolDocling is a document understanding model that can convert
     * documents to structured formats like markdown.</p>
     *
     * @param modelDir the SmolDocling model directory containing SDZ files
     * @return the loaded model
     * @throws IOException if loading fails
     */
    public static VisionLanguageModel loadSmolDocling(File modelDir) throws IOException {
        return loadSmolDocling(modelDir, null);
    }

    /**
     * Load a SmolDocling model with an optional externally-provided IOConfig.
     *
     * @param modelDir the SmolDocling model directory containing SDZ files
     * @param ioConfig external vision encoder IO config, or null to auto-discover
     * @return the loaded model
     * @throws IOException if loading fails
     */
    public static VisionLanguageModel loadSmolDocling(File modelDir, VisionEncoderIOConfig ioConfig) throws IOException {
        MultiPartModelLoader.LoadedModel loaded = MultiPartModelLoader.loadSmolDocling(modelDir);
        return loaded.toVisionLanguageModel(ioConfig);
    }

    /**
     * Load a VLM from ONNX model files with automatic SDZ caching.
     *
     * <p>On the first call, imports all ONNX models in parallel and caches
     * the results as SDZ files alongside the ONNX files. On subsequent calls,
     * loads directly from the cached SDZ files, reducing load time from
     * ~5 minutes to ~30 seconds.</p>
     *
     * @param visionEncoderOnnx path to the vision encoder ONNX file
     * @param decoderOnnx path to the decoder ONNX file
     * @param embedTokensOnnx path to the embed tokens ONNX file
     * @param tokenizerFile path to the tokenizer.json file
     * @return the loaded model with workspace mode enabled
     * @throws IOException if loading fails
     */
    public static VisionLanguageModel fromOnnx(
            File visionEncoderOnnx,
            File decoderOnnx,
            File embedTokensOnnx,
            File tokenizerFile) throws IOException {
        return fromOnnx(visionEncoderOnnx, decoderOnnx, embedTokensOnnx, tokenizerFile,
                8 * 1024 * 1024, null);
    }

    /**
     * Load a VLM from ONNX model files with an optional externally-provided IOConfig.
     */
    public static VisionLanguageModel fromOnnx(
            File visionEncoderOnnx,
            File decoderOnnx,
            File embedTokensOnnx,
            File tokenizerFile,
            VisionEncoderIOConfig ioConfig) throws IOException {
        return fromOnnx(visionEncoderOnnx, decoderOnnx, embedTokensOnnx, tokenizerFile,
                8 * 1024 * 1024, ioConfig);
    }

    /**
     * Load a VLM from ONNX model files with automatic SDZ caching and configurable workspace.
     *
     * @param visionEncoderOnnx path to the vision encoder ONNX file
     * @param decoderOnnx path to the decoder ONNX file
     * @param embedTokensOnnx path to the embed tokens ONNX file
     * @param tokenizerFile path to the tokenizer.json file
     * @param workspaceSize native workspace size in bytes (0 to disable)
     * @return the loaded model
     * @throws IOException if loading fails
     */
    public static VisionLanguageModel fromOnnx(
            File visionEncoderOnnx,
            File decoderOnnx,
            File embedTokensOnnx,
            File tokenizerFile,
            long workspaceSize) throws IOException {
        return fromOnnx(visionEncoderOnnx, decoderOnnx, embedTokensOnnx, tokenizerFile, workspaceSize, null);
    }

    /**
     * Load a VLM from ONNX model files with configurable workspace and optional IOConfig.
     */
    public static VisionLanguageModel fromOnnx(
            File visionEncoderOnnx,
            File decoderOnnx,
            File embedTokensOnnx,
            File tokenizerFile,
            long workspaceSize,
            VisionEncoderIOConfig ioConfig) throws IOException {
        long start = System.currentTimeMillis();

        // Load all 3 models in parallel with SDZ caching
        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionEncoderOnnx.getAbsolutePath(),
                decoderOnnx.getAbsolutePath(),
                embedTokensOnnx.getAbsolutePath()
        );

        SameDiff visionEncoder = models[0];
        SameDiff decoder = models[1];
        SameDiff embedTokens = models[2];

        // Enable native workspace for C++ op buffer safety
        if (workspaceSize > 0) {
            visionEncoder.enableWorkspaceMode(workspaceSize);
            decoder.enableWorkspaceMode(workspaceSize);
            embedTokens.enableWorkspaceMode(workspaceSize);
        }

        // Load tokenizer
        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerFile);

        long elapsed = System.currentTimeMillis() - start;
        log.info("VLM loaded from ONNX in {}ms (with caching)", elapsed);

        VisionLanguageModelBuilder builder = applyOnnxPackageMetadata(
                VisionLanguageModel.builder()
                        .visionEncoder(visionEncoder)
                        .decoder(decoder)
                        .embedTokens(embedTokens)
                        .tokenizer(tokenizer), tokenizerFile);
        if (ioConfig != null) {
            builder.visionEncoderIOConfig(ioConfig);
        }
        return builder.build();
    }

    /**
     * Load a VLM from ONNX model files with optional patches applied after import.
     *
     * @param visionEncoderOnnx path to the vision encoder ONNX file
     * @param decoderOnnx path to the decoder ONNX file
     * @param embedTokensOnnx path to the embed tokens ONNX file
     * @param tokenizerFile path to the tokenizer.json file
     * @param workspaceSize native workspace size in bytes (0 to disable)
     * @param ioConfig optional externally-provided IOConfig (null for auto-discover)
     * @param visionPatches patches to apply to the vision encoder (null for none)
     * @param decoderPatches patches to apply to the decoder (null for none)
     * @param embedPatches patches to apply to embed_tokens (null for none)
     * @return the loaded and patched model
     * @throws IOException if loading fails
     */
    public static VisionLanguageModel fromOnnxWithPatches(
            File visionEncoderOnnx,
            File decoderOnnx,
            File embedTokensOnnx,
            File tokenizerFile,
            long workspaceSize,
            VisionEncoderIOConfig ioConfig,
            List<SameDiffGraphPatch> visionPatches,
            List<SameDiffGraphPatch> decoderPatches,
            List<SameDiffGraphPatch> embedPatches) throws IOException {
        long start = System.currentTimeMillis();

        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionEncoderOnnx.getAbsolutePath(),
                decoderOnnx.getAbsolutePath(),
                embedTokensOnnx.getAbsolutePath()
        );

        SameDiff visionEncoder = models[0];
        SameDiff decoder = models[1];
        SameDiff embedTokens = models[2];

        // Apply patches
        applyPatches(visionEncoder, visionPatches, "visionEncoder");
        applyPatches(decoder, decoderPatches, "decoder");
        applyPatches(embedTokens, embedPatches, "embedTokens");

        if (workspaceSize > 0) {
            visionEncoder.enableWorkspaceMode(workspaceSize);
            decoder.enableWorkspaceMode(workspaceSize);
            embedTokens.enableWorkspaceMode(workspaceSize);
        }

        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerFile);

        long elapsed = System.currentTimeMillis() - start;
        log.info("VLM loaded from ONNX with patches in {}ms", elapsed);

        VisionLanguageModelBuilder builder = applyOnnxPackageMetadata(
                VisionLanguageModel.builder()
                        .visionEncoder(visionEncoder)
                        .decoder(decoder)
                        .embedTokens(embedTokens)
                        .tokenizer(tokenizer), tokenizerFile);
        if (ioConfig != null) {
            builder.visionEncoderIOConfig(ioConfig);
        }
        return builder.build();
    }

    private static VisionLanguageModelBuilder applyOnnxPackageMetadata(
            VisionLanguageModelBuilder builder, File tokenizerFile) throws IOException {
        File modelDirectory = tokenizerFile == null ? null : tokenizerFile.getAbsoluteFile().getParentFile();
        if (modelDirectory == null) return builder.outputProtocols(VlmOutputProtocolRegistry.fallback());
        builder.config(MultiPartModelLoader.loadConfig(modelDirectory));
        builder.outputProtocols(VlmOutputProtocolRegistry.load(modelDirectory));
        if (new File(modelDirectory, "preprocessor_config.json").isFile()) {
            builder.imagePreprocessor(VLMImagePreprocessor.fromModelDirectory(modelDirectory));
        }
        return builder;
    }

    private static void applyPatches(SameDiff model, List<SameDiffGraphPatch> patches, String label) {
        if (patches == null || patches.isEmpty()) return;
        for (SameDiffGraphPatch patch : patches) {
            if (patch.appliesTo(model)) {
                boolean applied = patch.apply(model);
                log.info("Patch '{}' on {}: {}", patch.name(), label, applied ? "applied" : "skipped");
            }
        }
    }

    /**
     * Preprocess an image for model input.
     *
     * @param imageFile the image file
     * @return the preprocessed image tensor
     * @throws IOException if loading fails
     */
    public INDArray preprocessImage(File imageFile) throws IOException {
        return imagePreprocessor.preprocess(imageFile);
    }

    /**
     * Preprocess an image tensor.
     *
     * @param image the raw image tensor [C, H, W] or [H, W, C]
     * @return the preprocessed image tensor
     */
    public INDArray preprocessImage(INDArray image) {
        return imagePreprocessor.preprocess(image);
    }

    /**
     * Encode an image through the vision encoder.
     *
     * @param image the preprocessed image tensor
     * @return the image embeddings
     */
    public INDArray encodeImage(INDArray image) {
        checkNotClosed();

        INDArray normalizedImage = normalizeVisionInputShape(image);
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put(visionEncoderIOConfig.getPixelValuesName(), normalizedImage);

        if (visionEncoderIOConfig.hasPixelAttentionMask()) {
            long h = normalizedImage.size(normalizedImage.rank() - 2);
            long w = normalizedImage.size(normalizedImage.rank() - 1);
            long[] maskShape;
            if (normalizedImage.rank() == 5) {
                // [batch, frames, C, H, W] -> mask [batch, frames, H, W]
                maskShape = new long[]{normalizedImage.size(0), normalizedImage.size(1), h, w};
            } else {
                // [batch, C, H, W] -> mask [batch, H, W]
                maskShape = new long[]{normalizedImage.size(0), h, w};
            }
            INDArray mask = Nd4j.ones(DataType.BOOL, maskShape);
            inputs.put(visionEncoderIOConfig.getPixelAttentionMaskName(), mask);
        }

        Map<String, INDArray> outputs = visionEncoder.output(inputs, visionEncoderIOConfig.getOutputNames());
        INDArray result = outputs.get(visionEncoderIOConfig.getPrimaryOutputName());

        // Close input arrays that we created (mask, normalizedImage if it differs from input).
        // These are only needed during visionEncoder.output() and leak GPU memory if not freed.
        if (visionEncoderIOConfig.hasPixelAttentionMask()) {
            INDArray mask = inputs.get(visionEncoderIOConfig.getPixelAttentionMaskName());
            if (mask != null && !mask.wasClosed()) {
                mask.setCloseable(true);
                mask.close();
            }
        }
        // Close non-primary outputs from the vision encoder
        for (var entry : outputs.entrySet()) {
            if (entry.getValue() != result && entry.getValue() != null && !entry.getValue().wasClosed()) {
                entry.getValue().setCloseable(true);
                entry.getValue().close();
            }
        }

        return result;
    }

    /**
     * Embed text tokens.
     *
     * @param tokenIds the token IDs
     * @return the text embeddings
     */
    public INDArray embedText(int[] tokenIds) {
        checkNotClosed();

        INDArray inputIds = Nd4j.createFromArray(tokenIds).reshape(1, tokenIds.length);

        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("input_ids", inputIds);

        Map<String, INDArray> outputs = embedTokens.output(inputs, "inputs_embeds");
        INDArray result = outputs.get("inputs_embeds");

        // Close inputIds — no longer needed after embedTokens.output() completes
        if (inputIds != null && !inputIds.wasClosed()) {
            inputIds.setCloseable(true);
            inputIds.close();
        }

        return result;
    }

    /**
     * Embed text tokens for a batch of sequences in a single graph execution.
     *
     * <p>This is more efficient than calling {@link #embedText(int[])} N times
     * because it executes the embedding graph once with a batched input tensor
     * instead of N separate executions.</p>
     *
     * @param tokenIds array of token IDs, one per sequence in the batch
     * @param batchSize number of sequences
     * @return the text embeddings [batchSize, 1, hidden]
     */
    public INDArray embedTextBatch(int[] tokenIds, int batchSize) {
        checkNotClosed();

        INDArray inputIds = Nd4j.createFromArray(tokenIds).reshape(batchSize, 1).castTo(DataType.LONG);

        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("input_ids", inputIds);

        Map<String, INDArray> outputs = embedTokens.output(inputs, "inputs_embeds");
        return outputs.get("inputs_embeds");
    }

    /**
     * Generate text from an image and prompt.
     *
     * @param image the preprocessed image
     * @param prompt the text prompt
     * @return the generated text
     */
    public String generate(INDArray image, String prompt) {
        return generate(image, prompt, 0, 1.0, true);
    }

    /**
     * Generate text from an image and prompt with parameters.
     *
     * @param image the preprocessed image
     * @param prompt the text prompt
     * @param maxNewTokens maximum tokens to generate
     * @param temperature sampling temperature
     * @param doSample whether to sample (false = greedy)
     * @return the generated text
     */
    public String generate(INDArray image, String prompt, int maxNewTokens,
                          double temperature, boolean doSample) {
        return generateWithMetrics(image, prompt, maxNewTokens, temperature, doSample).getText();
    }

    /**
     * Generate text from an image and prompt, returning detailed metrics.
     *
     * @param image the preprocessed image
     * @param prompt the text prompt
     * @return the generation result with metrics
     */
    public GenerationResult generateWithMetrics(INDArray image, String prompt) {
        return generateWithMetrics(image, prompt, 0, 1.0, true);
    }

    /**
     * Generate text from an image and prompt with parameters, returning detailed metrics.
     *
     * @param image the preprocessed image
     * @param prompt the text prompt
     * @param maxNewTokens maximum tokens to generate
     * @param temperature sampling temperature
     * @param doSample whether to sample (false = greedy)
     * @return the generation result with metrics
     */
    public GenerationResult generateWithMetrics(INDArray image, String prompt, int maxNewTokens,
                                                double temperature, boolean doSample) {
        return generateWithMetrics(image,
                VlmProtocolRequest.builder().promptOverride(prompt).build(),
                maxNewTokens, temperature, doSample);
    }

    /** Direct-image generation through a package/request-selected output protocol. */
    public GenerationResult generateWithMetrics(INDArray image, VlmProtocolRequest protocolRequest,
                                                int maxNewTokens, double temperature, boolean doSample) {
        SamplingConfig samplingConfig = SamplingConfig.builder()
                .maxNewTokens(maxNewTokens)
                .temperature(temperature)
                .doSample(doSample)
                .build();
        return generateWithMetrics(image, protocolRequest, samplingConfig);
    }

    /** Direct-image generation with the complete sampling policy preserved. */
    public GenerationResult generateWithMetrics(INDArray image, VlmProtocolRequest protocolRequest,
                                                SamplingConfig samplingConfig) {
        checkNotClosed();
        SamplingConfig requestedSampling = samplingConfig != null
                ? samplingConfig : SamplingConfig.greedy();

        VlmProtocolPlan protocolPlan = outputProtocols.prepare(protocolRequest, tokenizer, config);

        INDArray imageEmbeddings = null;
        INDArray textEmbeddings = null;
        INDArray combinedEmbeddings = null;
        try {
            imageEmbeddings = encodeImage(image);
            Encoding promptEncoding = protocolPlan.isApplyChatTemplate()
                    ? tokenizer.encodePrompt(protocolPlan.getPrompt())
                    : tokenizer.encode(protocolPlan.getPrompt(), false);
            int[] textPromptIds = promptEncoding.getIds();
            textEmbeddings = embedText(textPromptIds);
            combinedEmbeddings = combineEmbeddings(imageEmbeddings, textEmbeddings);

            int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
            int imageTokenCount = (int) imageEmbeddings.size(1);
            int[] combinedPromptIds = combinedPromptIds(
                    imageTokenCount, imageTokenId, textPromptIds);
            int effectiveMaxTokens = resolveGenerationBudget(
                    combinedPromptIds.length, requestedSampling.getMaxNewTokens());
            return decodeWithStaticLoop(combinedEmbeddings, combinedPromptIds,
                    requestedSampling.toBuilder().maxNewTokens(effectiveMaxTokens).build(), protocolPlan);
        } finally {
            SameDiffMemoryUtils.safeClose(combinedEmbeddings);
            SameDiffMemoryUtils.safeClose(textEmbeddings);
            SameDiffMemoryUtils.safeClose(imageEmbeddings);
        }
    }

    /**
     * Combine image and text embeddings.
     *
     * @param imageEmbeddings the image embeddings
     * @param textEmbeddings the text embeddings
     * @return combined embeddings
     */
    public INDArray combineEmbeddings(INDArray imageEmbeddings, INDArray textEmbeddings) {
        // For most VLMs: [batch, image_tokens, hidden] + [batch, text_tokens, hidden]
        // Concatenate along sequence dimension
        return Nd4j.concat(1, imageEmbeddings, textEmbeddings);
    }

    // =========================================================================
    // Batch Generation Methods for Throughput Optimization
    // =========================================================================

    /**
     * Generate text from multiple images in parallel (batch processing).
     *
     * <p>Batch processing amortizes the per-step overhead (shape calculation,
     * memory management, graph traversal) across multiple sequences. This
     * significantly improves throughput (tokens/second) at the cost of
     * slightly higher latency to first token.</p>
     *
     * <p>All sequences must use the same prompt template. Each image is
     * processed through the vision encoder and decoder in parallel.</p>
     *
     * @param images list of preprocessed image tensors, each [1, C, H, W] or [tiles, C, H, W]
     * @param prompt the shared text prompt
     * @param maxNewTokens maximum tokens to generate per sequence
     * @return array of generation results, one per image
     */
    public GenerationResult[] generateBatch(List<INDArray> images, String prompt, int maxNewTokens) {
        return generateBatch(images, prompt, maxNewTokens, false, 0.0);
    }

    /**
     * Generate per-page outputs for a multi-page document.
     *
     * <p>This is a semantic wrapper around {@link #generateBatch(List, String, int, boolean, double)}
     * that treats each image as a page in one document.</p>
     *
     * @param pageImages preprocessed page tensors (one image per page)
     * @param prompt prompt applied to each page
     * @param maxNewTokens max tokens per page
     * @return one generation result per page, in input order
     */
    public GenerationResult[] generatePages(List<INDArray> pageImages, String prompt, int maxNewTokens) {
        return generatePages(pageImages, prompt, maxNewTokens, false, 0.0);
    }

    /**
     * Generate per-page outputs for a multi-page document with sampling control.
     *
     * @param pageImages preprocessed page tensors (one image per page)
     * @param prompt prompt applied to each page
     * @param maxNewTokens max tokens per page
     * @param doSample whether to sample (false = greedy)
     * @param temperature sampling temperature (used when doSample=true)
     * @return one generation result per page, in input order
     */
    public GenerationResult[] generatePages(List<INDArray> pageImages, String prompt, int maxNewTokens,
                                            boolean doSample, double temperature) {
        return generateBatch(pageImages, prompt, maxNewTokens, doSample, temperature);
    }

    /**
     * Generate one combined document string from multiple page images.
     *
     * <p>Each page is generated independently, then concatenated using a page delimiter.</p>
     *
     * @param pageImages preprocessed page tensors (one image per page)
     * @param prompt prompt applied to each page
     * @param maxNewTokens max tokens per page
     * @return combined document text
     */
    public String generateDocument(List<INDArray> pageImages, String prompt, int maxNewTokens) {
        return generateDocument(pageImages, prompt, maxNewTokens, false, 0.0, "\n\n<page_break/>\n\n");
    }

    /**
     * Generate one combined document string from multiple page images with full controls.
     *
     * @param pageImages preprocessed page tensors (one image per page)
     * @param prompt prompt applied to each page
     * @param maxNewTokens max tokens per page
     * @param doSample whether to sample (false = greedy)
     * @param temperature sampling temperature (used when doSample=true)
     * @param pageDelimiter delimiter inserted between page outputs
     * @return combined document text
     */
    public String generateDocument(List<INDArray> pageImages, String prompt, int maxNewTokens,
                                   boolean doSample, double temperature, String pageDelimiter) {
        GenerationResult[] pageResults = generatePages(pageImages, prompt, maxNewTokens, doSample, temperature);
        String delimiter = pageDelimiter == null ? "" : pageDelimiter;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < pageResults.length; i++) {
            if (i > 0) {
                sb.append(delimiter);
            }
            if (pageResults[i] != null && pageResults[i].getText() != null) {
                sb.append(pageResults[i].getText());
            }
        }
        return sb.toString();
    }

    /**
     * Generate text from multiple images in parallel with sampling control.
     *
     * @param images list of preprocessed image tensors
     * @param prompt the shared text prompt
     * @param maxNewTokens maximum tokens to generate
     * @param doSample whether to sample (false = greedy)
     * @param temperature sampling temperature (only used if doSample=true)
     * @return array of generation results
     */
    public GenerationResult[] generateBatch(List<INDArray> images, String prompt,
                                             int maxNewTokens, boolean doSample, double temperature) {
        checkNotClosed();
        int batchSize = images.size();
        if (batchSize == 0) {
            return new GenerationResult[0];
        }

        long startNanos = System.nanoTime();

        // Encode prompt once (shared across all sequences)
        Encoding promptEncoding = tokenizer.encode(prompt, true);
        int[] promptIds = promptEncoding.getIds();
        int promptTokenCount = promptIds.length;
        int[] promptTokenCounts = new int[batchSize];
        for (int i = 0; i < batchSize; i++) {
            promptTokenCounts[i] = promptTokenCount;
        }

        // Encode all images through vision encoder (can be batched if images have same shape)
        List<INDArray> imageEmbeddingsList = new ArrayList<>();
        for (INDArray image : images) {
            INDArray imageEmbed = encodeImage(image);
            imageEmbeddingsList.add(imageEmbed);
        }

        // Stack image embeddings: [batchSize, imageSeqLen, hidden]
        INDArray batchedImageEmbeddings = Nd4j.concat(0, imageEmbeddingsList.toArray(new INDArray[0]));

        // Get text embeddings and replicate for batch
        INDArray singleTextEmbeddings = embedText(promptIds); // [1, textSeqLen, hidden]
        INDArray batchedTextEmbeddings = Nd4j.tile(singleTextEmbeddings, batchSize, 1, 1);

        // Combine: [batchSize, imageSeqLen + textSeqLen, hidden]
        INDArray currentEmbeddings = combineEmbeddings(batchedImageEmbeddings, batchedTextEmbeddings);

        // Discover decoder inputs/outputs for KV cache
        List<String> decoderInputNames = decoder.inputs();
        String logitsOutputName = ModelIOConfig.findLogitsOutputName(decoder);
        ModelIOConfig.KVCacheNames kvNames = ModelIOConfig.findKVCacheOutputNames(decoder);
        boolean useKvCache = !kvNames.keyNames.isEmpty() && !kvNames.valueNames.isEmpty();
        long hiddenSize = config != null && config.getHiddenSize() != null ? config.getHiddenSize() : 0;

        // Build output request list
        List<String> allOutputNames = new ArrayList<>();
        allOutputNames.add(logitsOutputName != null ? logitsOutputName : "logits");
        if (useKvCache) {
            allOutputNames.addAll(kvNames.keyNames);
            allOutputNames.addAll(kvNames.valueNames);
        }

        // Initialize batch state
        BatchGenerationState state = new BatchGenerationState(batchSize, tokenizer.getEosTokenId());
        Map<String, INDArray> kvCache = useKvCache ? new HashMap<>() : null;
        long pastSeqLen = 0;

        // N-gram speculative decoding: predict future tokens from patterns in generated sequence.
        // SmolDocling outputs repetitive DocTags — ideal for n-gram speculation.
        boolean speculativeEnabled = Boolean.parseBoolean(
                System.getProperty(ND4JSystemProperties.VLM_SPECULATIVE, "true")) && useKvCache;
        int ngramSize = Integer.getInteger(ND4JSystemProperties.VLM_SPECULATIVE_NGRAM_SIZE, 3);
        int maxSpecTokens = Integer.getInteger(ND4JSystemProperties.VLM_SPECULATIVE_MAX_TOKENS, 5);
        NgramSpeculator speculator = speculativeEnabled ? new NgramSpeculator(ngramSize, maxSpecTokens) : null;
        int totalSpeculativeAccepted = 0;
        int totalSpeculativeAttempts = 0;

        // Autoregressive generation loop
        for (int step = 0; step < maxNewTokens; step++) {
            // Build decoder inputs
            Map<String, INDArray> decoderInputMap = new HashMap<>();
            long currentSeqLen = currentEmbeddings.shape()[1];
            long totalSeqLen = currentSeqLen + pastSeqLen;

            for (String inputName : decoderInputNames) {
                if (inputName.equals("inputs_embeds")) {
                    decoderInputMap.put(inputName, currentEmbeddings);
                } else if (inputName.equals("attention_mask")) {
                    INDArray attentionMask = Nd4j.ones(DataType.LONG, batchSize, totalSeqLen);
                    // Zero out attention for finished sequences so decoder ignores them
                    for (int i = 0; i < batchSize; i++) {
                        if (state.isFinished(i)) {
                            attentionMask.putRow(i, Nd4j.zeros(DataType.LONG, totalSeqLen));
                        }
                    }
                    decoderInputMap.put(inputName, attentionMask);
                } else if (inputName.equals("_causal_mask")) {
                    decoderInputMap.put(inputName, ModelIOConfig.buildCausalMask(batchSize, currentSeqLen, totalSeqLen));
                } else if (inputName.equals("position_ids")) {
                    INDArray posIds = Nd4j.arange(pastSeqLen, pastSeqLen + currentSeqLen)
                            .reshape(1, currentSeqLen).castTo(DataType.LONG);
                    decoderInputMap.put(inputName, Nd4j.tile(posIds, batchSize, 1));
                } else if (useKvCache && inputName.startsWith("past_key_values.")) {
                    String presentName = inputName.replace("past_key_values", "present");
                    if (kvCache.containsKey(presentName)) {
                        decoderInputMap.put(inputName, kvCache.get(presentName));
                    } else {
                        decoderInputMap.put(inputName, ModelIOConfig.createEmptyKvCache(
                                decoder, inputName, batchSize, hiddenSize));
                    }
                }
            }

            // Run decoder
            long t0 = System.nanoTime();
            Map<String, INDArray> outputs = decoder.output(decoderInputMap,
                    allOutputNames.toArray(new String[0]));
            INDArray logits = outputs.get(allOutputNames.get(0));
            long tDecodeNs = System.nanoTime() - t0;

            // Close per-step input arrays (attention_mask, position_ids, causal_mask).
            // These are freshly allocated every step and leak GPU memory if not freed.
            // Skip inputs_embeds (owned by caller) and past_key_values (managed by kvCache).
            for (var entry : decoderInputMap.entrySet()) {
                String name = entry.getKey();
                if (name.equals("inputs_embeds") || name.startsWith("past_key_values.")) continue;
                SameDiffMemoryUtils.safeClose(entry.getValue());
            }

            // Update KV cache
            long t1 = System.nanoTime();
            if (useKvCache) {
                for (String presentName : kvNames.keyNames) {
                    INDArray pv = outputs.get(presentName);
                    if (pv != null) {
                        INDArray old = kvCache.put(presentName, pv);
                        SameDiffMemoryUtils.safeClose(old);
                    }
                }
                for (String presentName : kvNames.valueNames) {
                    INDArray pv = outputs.get(presentName);
                    if (pv != null) {
                        INDArray old = kvCache.put(presentName, pv);
                        SameDiffMemoryUtils.safeClose(old);
                    }
                }
            }
            long tKvCacheNs = System.nanoTime() - t1;

            // Get logits for last position: [batchSize, seqLen, vocab] -> [batchSize, vocab]
            INDArray lastLogits;
            if (logits.rank() == 3) {
                lastLogits = logits.get(NDArrayIndex.all(),
                        NDArrayIndex.point(logits.size(1) - 1), NDArrayIndex.all());
            } else {
                lastLogits = logits;
            }

            // Sample next tokens for all sequences
            long t2 = System.nanoTime();
            int[] nextTokenIds;
            if (!doSample || temperature <= 0) {
                nextTokenIds = SamplerUtils.argmaxBatch(lastLogits);
            } else {
                INDArray scaledLogits = lastLogits.div(temperature);
                INDArray probs = SamplerUtils.softmax(scaledLogits);
                nextTokenIds = SamplerUtils.multinomialSampleBatch(probs, new java.util.Random());
            }
            long tSampleNs = System.nanoTime() - t2;

            // Record tokens
            long stepNanos = System.nanoTime() - startNanos;
            state.recordTokens(nextTokenIds, stepNanos);

            // N-gram speculative decoding: after sufficient context, attempt to predict
            // and verify multiple future tokens in a single forward pass.
            int speculativeAccepted = 0;
            if (speculator != null && step >= ngramSize && useKvCache && !state.allFinished()) {
                NgramSpeculator.SpeculationResult specResult = speculator.speculateBatch(
                        state.getGeneratedTokens(), state.getFinished());
                if (specResult.hasSpeculation()) {
                    totalSpeculativeAttempts++;
                    int K = specResult.getCommonLength();
                    int[][] specTokens = specResult.getPerSequenceTokens();

                    // Build multi-token embeddings: [batchSize, K, hidden]
                    // Each sequence gets its K speculative tokens embedded.
                    // Advance pastSeqLen past the greedy step's tokens (will be used
                    // for position_ids in the verification pass).
                    long specPastSeqLen = pastSeqLen + currentSeqLen;
                    List<INDArray> specEmbedList = new ArrayList<>();
                    for (int k = 0; k < K; k++) {
                        int[] tokensAtK = new int[batchSize];
                        for (int b = 0; b < batchSize; b++) {
                            if (state.isFinished(b) || specTokens[b].length <= k) {
                                tokensAtK[b] = tokenizer.getEosTokenId();
                            } else {
                                tokensAtK[b] = specTokens[b][k];
                            }
                        }
                        specEmbedList.add(embedTextBatch(tokensAtK, batchSize));
                    }
                    // Concat along seq dim: [batchSize, K, hidden]
                    INDArray specEmbeddings = Nd4j.concat(1, specEmbedList.toArray(new INDArray[0]));
                    for (INDArray emb : specEmbedList) {
                        if (emb != specEmbeddings) SameDiffMemoryUtils.safeClose(emb);
                    }

                    // Run verification forward pass with K tokens
                    long currentSpecSeqLen = specEmbeddings.shape()[1];
                    long totalSpecSeqLen = currentSpecSeqLen + specPastSeqLen;
                    Map<String, INDArray> specDecoderInputMap = new HashMap<>();
                    for (String inputName : decoderInputMap.keySet()) {
                        if (inputName.equals("inputs_embeds")) {
                            specDecoderInputMap.put(inputName, specEmbeddings);
                        } else if (inputName.equals("attention_mask")) {
                            specDecoderInputMap.put(inputName, Nd4j.ones(DataType.LONG, batchSize, totalSpecSeqLen));
                        } else if (inputName.equals("_causal_mask")) {
                            specDecoderInputMap.put(inputName, ModelIOConfig.buildCausalMask(batchSize, currentSpecSeqLen, totalSpecSeqLen));
                        } else if (inputName.equals("position_ids")) {
                            INDArray posIds = Nd4j.arange(specPastSeqLen, specPastSeqLen + currentSpecSeqLen)
                                    .reshape(1, currentSpecSeqLen).castTo(DataType.LONG);
                            specDecoderInputMap.put(inputName, Nd4j.tile(posIds, batchSize, 1));
                        } else if (inputName.startsWith("past_key_values.")) {
                            // Reuse KV cache from the greedy step
                            String presentName = inputName.replace("past_key_values", "present");
                            if (kvCache.containsKey(presentName)) {
                                specDecoderInputMap.put(inputName, kvCache.get(presentName));
                            }
                        }
                    }

                    Map<String, INDArray> specOutputs = decoder.output(specDecoderInputMap,
                            allOutputNames.toArray(new String[0]));

                    // Close speculative pass input arrays (attention_mask, position_ids, causal_mask)
                    for (var entry : specDecoderInputMap.entrySet()) {
                        String name = entry.getKey();
                        if (name.equals("inputs_embeds") || name.startsWith("past_key_values.")) continue;
                        SameDiffMemoryUtils.safeClose(entry.getValue());
                    }

                    INDArray specLogits = specOutputs.get(allOutputNames.get(0));

                    // Verify speculation for sequence 0 (representative for greedy decoding)
                    // specLogits shape: [batchSize, K, vocabSize]
                    if (specLogits.rank() == 3 && specLogits.size(1) == K) {
                        // Extract logits for first batch element: [K, vocab]
                        float[][] logitsPerPos = new float[K][];
                        for (int k = 0; k < K; k++) {
                            INDArray posLogits = specLogits.get(NDArrayIndex.point(0),
                                    NDArrayIndex.point(k), NDArrayIndex.all());
                            logitsPerPos[k] = posLogits.toFloatVector();
                        }

                        int accepted = NgramSpeculator.verifySpeculation(specTokens[0], logitsPerPos);
                        if (accepted > 0) {
                            // Accept verified tokens for all sequences
                            for (int a = 0; a < accepted; a++) {
                                int[] acceptedTokenIds = new int[batchSize];
                                for (int b = 0; b < batchSize; b++) {
                                    if (state.isFinished(b) || specTokens[b].length <= a) {
                                        acceptedTokenIds[b] = tokenizer.getEosTokenId();
                                    } else {
                                        acceptedTokenIds[b] = specTokens[b][a];
                                    }
                                }
                                state.recordTokens(acceptedTokenIds, System.nanoTime() - startNanos);
                                step++; // Count each accepted speculative token as a step
                            }
                            speculativeAccepted = accepted;
                            totalSpeculativeAccepted += accepted;
                        }

                        // Update KV cache from speculative pass
                        if (useKvCache) {
                            for (String presentName : kvNames.keyNames) {
                                INDArray pv = specOutputs.get(presentName);
                                if (pv != null) {
                                    INDArray old = kvCache.put(presentName, pv);
                                    SameDiffMemoryUtils.safeClose(old);
                                }
                            }
                            for (String presentName : kvNames.valueNames) {
                                INDArray pv = specOutputs.get(presentName);
                                if (pv != null) {
                                    INDArray old = kvCache.put(presentName, pv);
                                    SameDiffMemoryUtils.safeClose(old);
                                }
                            }
                            // Advance pastSeqLen past both the greedy token and speculative tokens
                            pastSeqLen = specPastSeqLen + currentSpecSeqLen;
                        }
                    }
                    SameDiffMemoryUtils.safeClose(specEmbeddings);
                    decoder.clearPlaceholders(false);

                    if (speculativeAccepted > 0) {
                        log.debug("Step {}: speculation accepted {}/{} tokens", step, speculativeAccepted, K);
                    }
                }
            }

            // Check if all sequences are done
            if (state.allFinished()) {
                if (log.isDebugEnabled()) {
                    log.debug("Step {}: decode={}ms, kvCache={}ms, sample={}ms (finished)",
                            step, tDecodeNs / 1_000_000, tKvCacheNs / 1_000_000, tSampleNs / 1_000_000);
                }
                break;
            }

            // Prepare embeddings for next step
            long t3 = System.nanoTime();
            if (useKvCache) {
                if (speculativeAccepted == 0) {
                    pastSeqLen += currentSeqLen;
                }

                // Embed next tokens for all sequences in one batch call: [batchSize, 1, hidden]
                INDArray prevEmbeddings = currentEmbeddings;
                int[] batchTokenIds = new int[batchSize];
                for (int i = 0; i < batchSize; i++) {
                    if (state.isFinished(i)) {
                        batchTokenIds[i] = tokenizer.getEosTokenId();
                    } else {
                        // Use last accepted token (either greedy or last speculative)
                        List<Integer> seqTokens = state.getTokensForSequence(i);
                        batchTokenIds[i] = seqTokens.get(seqTokens.size() - 1);
                    }
                }
                currentEmbeddings = embedTextBatch(batchTokenIds, batchSize);
                if (prevEmbeddings != batchedImageEmbeddings) {
                    SameDiffMemoryUtils.safeClose(prevEmbeddings);
                }
                decoder.clearPlaceholders(false);
            } else {
                // No KV cache: grow embeddings (inefficient but correct)
                int[] batchTokenIds = new int[batchSize];
                for (int i = 0; i < batchSize; i++) {
                    batchTokenIds[i] = state.isFinished(i) ? tokenizer.getEosTokenId() : nextTokenIds[i];
                }
                INDArray batchedNewTokenEmbeds = embedTextBatch(batchTokenIds, batchSize);
                currentEmbeddings = Nd4j.concat(1, currentEmbeddings, batchedNewTokenEmbeds);
            }
            long tEmbedNs = System.nanoTime() - t3;

            // After prefill (step 0), reassign devices with fresh memory budgets.
            // Vision encoder may have been freed, making secondary devices available.
            if (step == 0 && useKvCache) {
                decoder.reassignDynamicShapePlanDevices();
            }

            // Per-step profiling (every 10 steps or on step 0)
            if (step % 10 == 0 && log.isInfoEnabled()) {
                log.info("Step {}: decode={}ms, kvCache={}ms, sample={}ms, embed={}ms",
                        step, tDecodeNs / 1_000_000, tKvCacheNs / 1_000_000,
                        tSampleNs / 1_000_000, tEmbedNs / 1_000_000);
            }
        }

        // Log speculative decoding summary
        if (speculator != null && totalSpeculativeAttempts > 0) {
            log.info("Speculative decoding: {}/{} attempts accepted {} total tokens (avg {}/attempt)",
                    totalSpeculativeAttempts, totalSpeculativeAttempts, totalSpeculativeAccepted,
                    totalSpeculativeAttempts > 0 ? String.format("%.1f", (double) totalSpeculativeAccepted / totalSpeculativeAttempts) : "0");
        }

        // Mark any remaining sequences as max tokens
        for (int i = 0; i < batchSize; i++) {
            state.markMaxTokens(i);
        }

        // Clean up KV cache
        if (kvCache != null) {
            for (INDArray v : kvCache.values()) {
                SameDiffMemoryUtils.safeClose(v);
            }
        }

        // Decode all sequences
        long totalNanos = System.nanoTime() - startNanos;
        String[] texts = new String[batchSize];
        for (int i = 0; i < batchSize; i++) {
            texts[i] = tokenizer.decode(state.getTokenArrayForSequence(i), true);
        }

        log.info("Batch generation complete: {} sequences, {} total tokens in {}ms ({} tokens/sec)",
                batchSize, state.getMaxTokenCount() * batchSize, totalNanos / 1_000_000,
                String.format("%.1f", (state.getMaxTokenCount() * batchSize * 1_000_000_000.0) / totalNanos));

        return state.buildResults(texts, promptTokenCounts, totalNanos);
    }

    /**
     * Encode multiple images through the vision encoder in a single batch.
     *
     * <p>This is more efficient than encoding images one at a time when
     * images have the same dimensions (number of tiles).</p>
     *
     * @param images list of preprocessed image tensors with same shape
     * @return batched image embeddings [batchSize, seqLen, hidden]
     */
    public INDArray encodeImagesBatch(List<INDArray> images) {
        checkNotClosed();

        if (images.isEmpty()) {
            return Nd4j.empty(DataType.FLOAT);
        }

        // Check if all images have same shape (required for true batching)
        long[] firstShape = images.get(0).shape();
        boolean sameShape = true;
        for (INDArray img : images) {
            if (!java.util.Arrays.equals(img.shape(), firstShape)) {
                sameShape = false;
                break;
            }
        }

        if (sameShape && firstShape[0] == 1) {
            // All images have same shape - stack and process together
            List<INDArray> normalized = new ArrayList<>(images.size());
            for (INDArray img : images) {
                normalized.add(normalizeVisionInputShape(img));
            }
            INDArray batched = Nd4j.concat(0, normalized.toArray(new INDArray[0]));
            Map<String, INDArray> inputs = new HashMap<>();
            inputs.put(visionEncoderIOConfig.getPixelValuesName(), batched);
            Map<String, INDArray> outputs = visionEncoder.output(inputs,
                    java.util.Arrays.asList(visionEncoderIOConfig.getOutputNames()));
            return outputs.get(visionEncoderIOConfig.getPrimaryOutputName());
        } else {
            // Different shapes - process individually and stack
            List<INDArray> embeddings = new ArrayList<>();
            for (INDArray img : images) {
                embeddings.add(encodeImage(img));
            }
            return Nd4j.concat(0, embeddings.toArray(new INDArray[0]));
        }
    }

    /**
     * Encode an image that has been split into tiles (frames) via {@link ImageTiler}.
     *
     * <p>This method runs the vision encoder frame-by-frame, creates pixel attention masks
     * per frame, selects the output via {@link VisionEncoderUtils#selectVisionOutput},
     * and concatenates all frame embeddings along the sequence dimension.</p>
     *
     * @param splitResult the result from {@link ImageTiler#splitImageForVLM}
     * @param targetSize the target frame size (e.g. 512 for SmolDocling)
     * @return concatenated vision embeddings [1, totalSeqLen, hiddenDim]
     */
    public INDArray encodeImageTiled(ImageTiler.SplitImageResult splitResult, int targetSize) {
        checkNotClosed();

        List<INDArray> frameEmbeddings = new ArrayList<>();
        int numFrames = splitResult.frames.size();

        for (int f = 0; f < numFrames; f++) {
            java.awt.image.BufferedImage frame = splitResult.frames.get(f);
            INDArray frameTensor = imagePreprocessor.preprocess(frame);
            // Use normalizeVisionInputShape to match what encodeImage() does — it checks
            // whether the encoder expects rank-5 [batch,frames,C,H,W] or rank-4 [batch,C,H,W].
            // Unconditionally reshaping to rank-5 breaks models that expect rank-4 (e.g.,
            // SmolDocling's conv2d pipeline has permute ops with 4-element permutation vectors).
            INDArray visionFrameInput = normalizeVisionInputShape(frameTensor);

            Map<String, INDArray> inputs = new HashMap<>();
            inputs.put(visionEncoderIOConfig.getPixelValuesName(), visionFrameInput);

            // Create pixel attention mask for this frame
            ImageTiler.ContentRegion region = splitResult.contentRegions.get(f);
            INDArray pixelMask = ImageTiler.createPixelAttentionMask(
                    region.width, region.height, targetSize);
            // Only add if the encoder accepts it
            if (visionEncoderIOConfig.hasPixelAttentionMask()) {
                inputs.put(visionEncoderIOConfig.getPixelAttentionMaskName(), pixelMask);
            }

            // Run vision encoder for this frame using discovered output names
            Map<String, INDArray> outputs = visionEncoder.output(inputs, visionEncoderIOConfig.getOutputNames());
            VisionEncoderUtils.VisionOutput selected = VisionEncoderUtils.selectVisionOutput(outputs);

            if (selected == null || selected.tensor == null) {
                log.warn("Frame {}/{}: no usable vision output, skipping", f + 1, numFrames);
                continue;
            }

            if (VisionEncoderUtils.isAllZeroOrNaN(selected.tensor)) {
                log.warn("Frame {}/{}: vision output is all zeros/NaN, skipping", f + 1, numFrames);
                continue;
            }

            // Ensure rank-3: [1, seqLen, hiddenDim]
            INDArray embedding = selected.tensor;
            if (embedding.rank() == 2) {
                embedding = embedding.reshape(1, embedding.size(0), embedding.size(1));
            }

            frameEmbeddings.add(embedding.dup());

            if (log.isDebugEnabled()) {
                log.debug("Frame {}/{}: output '{}' shape={}, min={}, max={}",
                        f + 1, numFrames, selected.name,
                        java.util.Arrays.toString(embedding.shape()),
                        embedding.minNumber(), embedding.maxNumber());
            } else {
                log.info("Frame {}/{}: output '{}' shape={}",
                        f + 1, numFrames, selected.name,
                        java.util.Arrays.toString(embedding.shape()));
            }

            // Do NOT close plan-owned output arrays — their DataBuffers may be
            // the same GPU allocation the DSP plan's slot buffers reference.
            // Closing them frees GPU memory that the next execution still reads,
            // causing Xid 13 (Out Of Range Address). The .dup() above already
            // copied what we need; the plan will manage its own buffers.

            SameDiffMemoryUtils.safeClose(frameTensor);
            SameDiffMemoryUtils.safeClose(pixelMask);
        }

        if (frameEmbeddings.isEmpty()) {
            throw new IllegalStateException("No valid vision embeddings from any frame");
        }

        // Concatenate along sequence dimension
        INDArray result;
        if (frameEmbeddings.size() == 1) {
            result = frameEmbeddings.get(0);
        } else {
            result = Nd4j.concat(1, frameEmbeddings.toArray(new INDArray[0]));
            // Close individual frame embeddings — concat created a new array,
            // so the originals are no longer needed. Without this, ~6-12 MB leaks per page.
            for (INDArray frameEmb : frameEmbeddings) {
                SameDiffMemoryUtils.safeClose(frameEmb);
            }
        }

        log.info("Encoded {} frames -> vision embeddings shape={}",
                numFrames, java.util.Arrays.toString(result.shape()));
        return result;
    }

    /**
     * Generate per-page outputs from tiled multi-page inputs.
     *
     * <p>Each page is encoded with per-frame pixel attention masks, then decoded independently
     * using Docling-style image prompt construction.</p>
     *
     * @param pageSplitResults tiled pages (one split result per page)
     * @param userPrompt prompt appended after image tokens
     * @param maxNewTokens max tokens per page
     * @param doSample whether to sample (false = greedy)
     * @param temperature sampling temperature
     * @param targetSize tile size used for mask generation (e.g. 512)
     * @return one generation result per page, in input order
     */
    public GenerationResult[] generatePagesTiled(List<ImageTiler.SplitImageResult> pageSplitResults,
                                                 String userPrompt,
                                                 int maxNewTokens,
                                                 boolean doSample,
                                                 double temperature,
                                                 int targetSize) {
        return generatePagesTiled(pageSplitResults,
                VlmProtocolRequest.builder().promptOverride(userPrompt).build(),
                maxNewTokens, doSample, temperature, targetSize);
    }

    /** Generate tiled pages through a package/request-selected output protocol. */
    public GenerationResult[] generatePagesTiled(List<ImageTiler.SplitImageResult> pageSplitResults,
                                                 VlmProtocolRequest protocolRequest,
                                                 int maxNewTokens,
                                                 boolean doSample,
                                                 double temperature,
                                                 int targetSize) {
        SamplingConfig samplingConfig = SamplingConfig.builder()
                .maxNewTokens(maxNewTokens)
                .doSample(doSample)
                .temperature(temperature)
                .build();
        return generatePagesTiled(pageSplitResults, protocolRequest, samplingConfig, targetSize);
    }

    /** Generate tiled pages while preserving the complete caller sampling policy. */
    public GenerationResult[] generatePagesTiled(List<ImageTiler.SplitImageResult> pageSplitResults,
                                                 VlmProtocolRequest protocolRequest,
                                                 SamplingConfig samplingConfig,
                                                 int targetSize) {
        checkNotClosed();
        if (pageSplitResults == null || pageSplitResults.isEmpty()) {
            return new GenerationResult[0];
        }

        VlmProtocolPlan protocolPlan = outputProtocols.prepare(protocolRequest, tokenizer, config);

        SamplingConfig requestedSampling = samplingConfig != null
                ? samplingConfig : SamplingConfig.greedy();

        GenerationResult[] results = new GenerationResult[pageSplitResults.size()];
        for (int i = 0; i < pageSplitResults.size(); i++) {
            ImageTiler.SplitImageResult splitResult = pageSplitResults.get(i);
            if (splitResult == null || splitResult.getTotalFrames() <= 0) {
                results[i] = GenerationResult.builder()
                        .text("")
                        .tokenIds(new int[0])
                        .generatedTokenCount(0)
                        .promptTokenCount(0)
                        .totalTokenCount(0)
                        .finishReason(GenerationResult.FinishReason.MAX_TOKENS)
                        .firstTokenLatencyMs(0)
                        .generationTimeMs(0)
                        .tokensPerSecond(0.0)
                        .build();
                continue;
            }

            // Reset sessions between pages to free DSP intermediates and CUDA pool memory.
            // Without this, ~14 GB of GPU memory accumulates across pages.
            if (i > 0) {
                log.info("Page {}/{}: resetting sessions to reclaim GPU memory", i + 1, pageSplitResults.size());
                resetSessions();
            }

            INDArray visionEmbeddings = encodeImageTiled(splitResult, targetSize);
            try {
                int totalFrames = splitResult.getTotalFrames();
                int imageSeqLenPerFrame = (int) (visionEmbeddings.size(1) / Math.max(totalFrames, 1));
                results[i] = generateFromEmbeddings(
                        visionEmbeddings,
                        protocolPlan,
                        requestedSampling,
                        splitResult.numRows,
                        splitResult.numCols,
                        imageSeqLenPerFrame);
            } finally {
                SameDiffMemoryUtils.safeClose(visionEmbeddings);
            }
        }

        return results;
    }

    /**
     * Generate per-page outputs from tiled multi-page inputs (greedy defaults).
     *
     * @param pageSplitResults tiled pages (one split result per page)
     * @param userPrompt prompt appended after image tokens
     * @param maxNewTokens max tokens per page
     * @param targetSize tile size used for mask generation (e.g. 512)
     * @return one generation result per page, in input order
     */
    public GenerationResult[] generatePagesTiled(List<ImageTiler.SplitImageResult> pageSplitResults,
                                                 String userPrompt,
                                                 int maxNewTokens,
                                                 int targetSize) {
        return generatePagesTiled(pageSplitResults, userPrompt, maxNewTokens, false, 0.0, targetSize);
    }

    /**
     * Generate one combined document output from tiled multi-page inputs.
     *
     * @param pageSplitResults tiled pages (one split result per page)
     * @param userPrompt prompt appended after image tokens
     * @param maxNewTokens max tokens per page
     * @param doSample whether to sample (false = greedy)
     * @param temperature sampling temperature
     * @param targetSize tile size used for mask generation (e.g. 512)
     * @param pageDelimiter delimiter inserted between page outputs
     * @return combined document text
     */
    public String generateDocumentTiled(List<ImageTiler.SplitImageResult> pageSplitResults,
                                        String userPrompt,
                                        int maxNewTokens,
                                        boolean doSample,
                                        double temperature,
                                        int targetSize,
                                        String pageDelimiter) {
        GenerationResult[] pageResults = generatePagesTiled(
                pageSplitResults, userPrompt, maxNewTokens, doSample, temperature, targetSize);
        String delimiter = pageDelimiter == null ? "" : pageDelimiter;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < pageResults.length; i++) {
            if (i > 0) {
                sb.append(delimiter);
            }
            if (pageResults[i] != null && pageResults[i].getText() != null) {
                sb.append(pageResults[i].getText());
            }
        }
        return sb.toString();
    }

    /**
     * Generate one combined document output from tiled multi-page inputs (greedy defaults).
     *
     * @param pageSplitResults tiled pages (one split result per page)
     * @param userPrompt prompt appended after image tokens
     * @param maxNewTokens max tokens per page
     * @param targetSize tile size used for mask generation (e.g. 512)
     * @return combined document text
     */
    public String generateDocumentTiled(List<ImageTiler.SplitImageResult> pageSplitResults,
                                        String userPrompt,
                                        int maxNewTokens,
                                        int targetSize) {
        return generateDocumentTiled(
                pageSplitResults, userPrompt, maxNewTokens, false, 0.0, targetSize, "\n\n<page_break/>\n\n");
    }

    /**
     * Generate text from pre-computed vision embeddings using the embedding merger approach.
     *
     * <p>This method takes vision embeddings (from {@link #encodeImageTiled} or
     * {@link #encodeImage}), builds the prompt with image tokens, uses
     * {@link EmbeddingMerger} to replace {@code <image>} token positions with
     * vision embeddings, and runs the decode loop.</p>
     *
     * @param visionEmbeddings the pre-computed vision embeddings [1, visionSeqLen, hiddenDim]
     * @param userPrompt the text prompt from the user
     * @param config sampling configuration
     * @param imageRows number of tile rows (0 if no tiling)
     * @param imageCols number of tile columns (0 if no tiling)
     * @param imageSeqLenPerFrame number of vision tokens per frame
     * @return generation result with text and metrics
     */
    public GenerationResult generateFromEmbeddings(
            INDArray visionEmbeddings,
            String userPrompt,
            SamplingConfig config,
            int imageRows, int imageCols,
            int imageSeqLenPerFrame) {
        VlmProtocolPlan plan = outputProtocols.prepare(
                VlmProtocolRequest.builder().promptOverride(userPrompt).build(), tokenizer, this.config);
        return generateFromEmbeddings(visionEmbeddings, plan, config,
                imageRows, imageCols, imageSeqLenPerFrame);
    }

    /** Protocol-aware embedding generation used by document/OCR pipelines. */
    public GenerationResult generateFromEmbeddings(
            INDArray visionEmbeddings,
            VlmProtocolPlan protocolPlan,
            SamplingConfig config,
            int imageRows, int imageCols,
            int imageSeqLenPerFrame) {
        checkNotClosed();

        int requestedMaxTokens = config.getMaxNewTokens();
        double temperature = config.getTemperature();
        boolean doSample = config.isDoSample();

        // Build prompt with image tokens
        int totalFrames = (imageRows > 0 && imageCols > 0)
                ? imageRows * imageCols + 1  // tiles + global
                : 1;  // global only
        int totalImageTokens = totalFrames * imageSeqLenPerFrame;
        String imagePrompt = ImagePromptBuilder.buildImagePromptString(
                imageRows, imageCols, imageSeqLenPerFrame);

        // Apply chat template if available — wraps image + text in model-specific format
        // (e.g., Idefics3: <|im_start|>User:<image tokens>prompt<end_of_utterance>\nAssistant:)
        String fullPrompt;
        String userPrompt = protocolPlan.getPrompt();
        String chatTemplateText = tokenizer.getChatTemplate();
        if (protocolPlan.isApplyChatTemplate()
                && chatTemplateText != null && !chatTemplateText.isEmpty()) {
            List<ChatTemplate.ContentPart> parts = new ArrayList<>();
            parts.add(ChatTemplate.ContentPart.image(imagePrompt));
            parts.add(ChatTemplate.ContentPart.text(userPrompt));
            ChatTemplate.Message msg = new ChatTemplate.Message("user", parts);
            ChatTemplate chatTemplate = new ChatTemplate(
                    chatTemplateText,
                    tokenizer.getBosToken(),
                    tokenizer.getEosToken());
            fullPrompt = chatTemplate.apply(List.of(msg), true);
            log.info("Applied chat template to VLM prompt, length: {} chars", fullPrompt.length());
        } else {
            fullPrompt = imagePrompt + userPrompt;
        }

        // Tokenize
        Encoding encoding = tokenizer.encode(fullPrompt, false);
        int[] promptTokenIds = encoding.getIds();
        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        int promptTokenCount = promptTokenIds.length;
        int maxTokens = resolveGenerationBudget(promptTokenCount, requestedMaxTokens);

        log.info("generateFromEmbeddings: prompt has {} tokens, {} image tokens expected, vision shape={}",
                promptTokenCount, totalImageTokens,
                java.util.Arrays.toString(visionEmbeddings.shape()));

        // Embed text tokens
        INDArray textEmbeddings = embedText(promptTokenIds);

        // Merge: replace <image> positions with vision embeddings
        INDArray inputsEmbeds = EmbeddingMerger.mergeEmbeddings(
                textEmbeddings, visionEmbeddings, promptTokenIds, imageTokenId);
        SameDiffMemoryUtils.safeClose(textEmbeddings);

        // Run decode loop using the merged embeddings directly
        try {
            return decodeWithStaticLoop(inputsEmbeds, promptTokenIds,
                    config.toBuilder().maxNewTokens(maxTokens).build(), protocolPlan);
        } finally {
            // inputsEmbeds is not closed by the decode loop (it treats it as externally owned).
            // We must close it here to prevent ~5 MB GPU memory leak per page.
            SameDiffMemoryUtils.safeClose(inputsEmbeds);
        }
    }

    /**
     * Core decode loop that generates text from pre-combined embeddings.
     *
     * @param combinedEmbeddings the input embeddings [1, seqLen, hidden]
     * @param promptTokenCount number of prompt tokens (for metrics)
     * @param maxNewTokens maximum tokens to generate
     * @param temperature sampling temperature
     * @param doSample whether to sample (false = greedy)
     * @return generation result
     */
    private GenerationResult decodeFromEmbeddings(INDArray combinedEmbeddings, int promptTokenCount,
                                                   int maxNewTokens, double temperature, boolean doSample) {
        List<String> decoderInputNames = decoder.inputs();
        String logitsOutputName = ModelIOConfig.findLogitsOutputName(decoder);
        ModelIOConfig.KVCacheNames kvNames = ModelIOConfig.findKVCacheOutputNames(decoder);
        boolean useKvCache = !kvNames.keyNames.isEmpty() && !kvNames.valueNames.isEmpty();
        long hiddenSize = config != null && config.getHiddenSize() != null ? config.getHiddenSize() : 0;

        List<String> allOutputNames = new ArrayList<>();
        allOutputNames.add(logitsOutputName != null ? logitsOutputName : "logits");
        if (useKvCache) {
            allOutputNames.addAll(kvNames.keyNames);
            allOutputNames.addAll(kvNames.valueNames);
        }

        StringBuilder generated = new StringBuilder();
        List<Integer> generatedTokenIds = new ArrayList<>();
        long startNanos = System.nanoTime();
        long firstTokenLatencyNanos = 0;
        int generatedTokenCount = 0;
        GenerationResult.FinishReason finishReason = GenerationResult.FinishReason.MAX_TOKENS;

        Map<String, INDArray> kvCache = useKvCache ? new HashMap<>() : null;
        INDArray currentEmbeddings = combinedEmbeddings;
        long pastSeqLen = 0;
        long batchSize = 1;

        // Pre-allocate FIXED-SHAPE reusable buffers for attention_mask, _causal_mask, and position_ids.
        // CRITICAL: These buffers must have CONSTANT shape across all decode steps to prevent
        // DSP shape-hash changes that trigger multi-plan switches on unwarmed plans (→ garbage output).
        // Pattern matches GenerationPipeline's fixed-shape approach with delta putScalar updates.
        boolean needsAttentionMask2 = decoderInputNames.contains("attention_mask");
        boolean needsCausalMask2 = decoderInputNames.contains("_causal_mask");
        boolean needsPositionIds2 = decoderInputNames.contains("position_ids");
        long initialSeqLen2 = currentEmbeddings.shape()[1];
        long maxTotalSeqLen2 = initialSeqLen2 + maxNewTokens;

        // attention_mask: fixed [1, maxTotalSeqLen2] — ones for valid positions, zeros for future
        INDArray attentionMaskBuffer2 = null;
        if (needsAttentionMask2) {
            attentionMaskBuffer2 = Nd4j.zeros(DataType.LONG, batchSize, maxTotalSeqLen2);
            // Prefill: mark initial positions as valid
            attentionMaskBuffer2.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, initialSeqLen2)).assign(1);
        }

        // _causal_mask: fixed [1, 1, 1, maxTotalSeqLen2] — 0.0 for visible positions, MASK_FILL for masked
        // During decode (seqLen=1), this is a 1D attention bias: attend to past, mask future
        INDArray causalMaskBuffer2 = null;
        if (needsCausalMask2) {
            causalMaskBuffer2 = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, maxTotalSeqLen2);
            // Prefill: all positions beyond initialSeqLen are masked
            causalMaskBuffer2.assign(ModelIOConfig.MASK_FILL);
            causalMaskBuffer2.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.interval(0, initialSeqLen2)).assign(0.0f);
        }

        INDArray positionIdsBuffer2 = needsPositionIds2 ?
                Nd4j.zeros(DataType.LONG, 1, Math.max(initialSeqLen2, 1)) : null;

        for (int i = 0; i < maxNewTokens; i++) {
            Map<String, INDArray> decoderInputMap = new HashMap<>();
            long currentSeqLen = currentEmbeddings.shape()[1];
            long totalSeqLen = currentSeqLen + pastSeqLen;

            // Track temporary input arrays so we can free them after output()
            List<INDArray> tempInputs = new ArrayList<>();

            for (String inputName : decoderInputNames) {
                if (inputName.equals("inputs_embeds")) {
                    decoderInputMap.put(inputName, currentEmbeddings);
                } else if (inputName.equals("attention_mask")) {
                    // FIXED-SHAPE buffer: delta update — mark the new position as valid.
                    // Shape stays [1, maxTotalSeqLen2] every step → no DSP plan switches.
                    if (i > 0) {
                        // Unmask the position for the token generated in the previous step
                        attentionMaskBuffer2.putScalar(0, totalSeqLen - 1, 1);
                    }
                    decoderInputMap.put(inputName, attentionMaskBuffer2);
                } else if (inputName.equals("_causal_mask")) {
                    // FIXED-SHAPE buffer: delta update — unmask one more position.
                    // Shape stays [1, 1, 1, maxTotalSeqLen2] every step → no DSP plan switches.
                    if (i == 0) {
                        // Prefill: use the pre-initialized buffer directly (already set up above)
                        // For prefill with currentSeqLen > 1, we need a proper causal mask
                        if (currentSeqLen > 1) {
                            INDArray prefillMask = ModelIOConfig.buildCausalMask(currentSeqLen, totalSeqLen);
                            decoderInputMap.put(inputName, prefillMask);
                            tempInputs.add(prefillMask);
                        } else {
                            decoderInputMap.put(inputName, causalMaskBuffer2);
                        }
                    } else {
                        // Decode: unmask position for the newly generated token
                        causalMaskBuffer2.putScalar(new long[]{0, 0, 0, totalSeqLen - 1}, 0.0f);
                        decoderInputMap.put(inputName, causalMaskBuffer2);
                    }
                } else if (inputName.equals("position_ids")) {
                    // For KV cache decode (single token), reuse buffer and set value
                    if (useKvCache && i > 0 && currentSeqLen == 1) {
                        positionIdsBuffer2.putScalar(0, 0, pastSeqLen);
                        decoderInputMap.put(inputName, positionIdsBuffer2);
                    } else {
                        // Prefill or non-KV: need full range
                        INDArray posIds = Nd4j.arange(pastSeqLen, pastSeqLen + currentSeqLen)
                                .reshape(1, currentSeqLen).castTo(DataType.LONG);
                        decoderInputMap.put(inputName, posIds);
                        tempInputs.add(posIds);
                    }
                } else if (useKvCache && inputName.startsWith("past_key_values.")) {
                    String presentName = inputName.replace("past_key_values", "present");
                    if (kvCache.containsKey(presentName)) {
                        decoderInputMap.put(inputName, kvCache.get(presentName));
                    } else {
                        INDArray emptyKv = ModelIOConfig.createEmptyKvCache(
                                decoder, inputName, batchSize, hiddenSize);
                        decoderInputMap.put(inputName, emptyKv);
                        tempInputs.add(emptyKv);
                    }
                }
            }

            Map<String, INDArray> outputs = decoder.output(decoderInputMap,
                    allOutputNames.toArray(new String[0]));
            INDArray logits = outputs.get(allOutputNames.get(0));

            // Free temporary input arrays (poisoned by directExecHelper, so use safeClose)
            for (INDArray tmp : tempInputs) {
                SameDiffMemoryUtils.safeClose(tmp);
            }

            if (useKvCache) {
                for (String presentName : kvNames.keyNames) {
                    INDArray pv = outputs.get(presentName);
                    if (pv != null) {
                        INDArray old = kvCache.put(presentName, pv);
                        SameDiffMemoryUtils.safeClose(old);
                    }
                }
                for (String presentName : kvNames.valueNames) {
                    INDArray pv = outputs.get(presentName);
                    if (pv != null) {
                        INDArray old = kvCache.put(presentName, pv);
                        SameDiffMemoryUtils.safeClose(old);
                    }
                }
            }

            int nextTokenId;
            if (!doSample || temperature <= 0) {
                INDArray lastLogits = logits.get(NDArrayIndex.point(0),
                        NDArrayIndex.point(logits.shape()[1] - 1));
                nextTokenId = SamplerUtils.argmax(lastLogits);
            } else {
                INDArray scaledLogits = logits.div(temperature);
                INDArray probs = Nd4j.nn().softmax(scaledLogits, 2);
                INDArray lastProbs = probs.get(NDArrayIndex.point(0),
                        NDArrayIndex.point(probs.shape()[1] - 1));
                nextTokenId = SamplerUtils.argmax(lastProbs);
                SameDiffMemoryUtils.safeClose(scaledLogits);
                SameDiffMemoryUtils.safeClose(probs);
            }

            // Free logits — we've extracted the token ID already
            SameDiffMemoryUtils.safeClose(logits);

            if (generatedTokenCount == 0) {
                firstTokenLatencyNanos = System.nanoTime() - startNanos;
            }
            generatedTokenCount++;
            generatedTokenIds.add(nextTokenId);

            if (nextTokenId == tokenizer.getEosTokenId()) {
                finishReason = GenerationResult.FinishReason.EOS;
                break;
            }

            String tokenText = tokenizer.decode(new int[]{nextTokenId}, true);
            generated.append(tokenText);

            if (useKvCache) {
                pastSeqLen += currentSeqLen;
                INDArray prevEmbeddings = currentEmbeddings;
                currentEmbeddings = embedText(new int[]{nextTokenId});
                if (prevEmbeddings != combinedEmbeddings) {
                    SameDiffMemoryUtils.safeClose(prevEmbeddings);
                }
                decoder.clearPlaceholders(false);
            } else {
                INDArray newTokenEmbed = embedText(new int[]{nextTokenId});
                currentEmbeddings = Nd4j.concat(1, currentEmbeddings, newTokenEmbed);
            }
        }

        // Clean up pre-allocated buffers
        SameDiffMemoryUtils.safeClose(attentionMaskBuffer2);
        SameDiffMemoryUtils.safeClose(causalMaskBuffer2);
        SameDiffMemoryUtils.safeClose(positionIdsBuffer2);

        // Free final KV cache entries
        if (kvCache != null) {
            for (INDArray v : kvCache.values()) {
                SameDiffMemoryUtils.safeClose(v);
            }
        }

        // Free final embeddings if not the original input
        if (currentEmbeddings != combinedEmbeddings) {
            SameDiffMemoryUtils.safeClose(currentEmbeddings);
        }

        // Flush pending GPU ops and trim memory pool to release freed CUDA memory
        SameDiffMemoryUtils.trimAllDevicePools();

        long totalNanos = System.nanoTime() - startNanos;
        long totalMs = totalNanos / 1_000_000;
        long firstTokenMs = firstTokenLatencyNanos / 1_000_000;
        int[] tokenIdArray = generatedTokenIds.stream().mapToInt(Integer::intValue).toArray();

        return GenerationResult.builder()
                .text(generated.toString())
                .tokenIds(tokenIdArray)
                .generatedTokenCount(generatedTokenCount)
                .promptTokenCount(promptTokenCount)
                .totalTokenCount(promptTokenCount + generatedTokenCount)
                .finishReason(finishReason)
                .firstTokenLatencyMs(firstTokenMs)
                .generationTimeMs(totalMs)
                .tokensPerSecond(totalNanos > 0 ? (generatedTokenCount * 1_000_000_000.0) / totalNanos : 0)
                .build();
    }

    /**
     * Reset all inference sessions to free GPU memory between pages.
     * Call this after processing each page to release cached intermediates
     * before starting the next page.
     */
    public void resetSessions() {
        log.info("Resetting sessions between pages (freeing cached intermediates)");
        // Reset vision encoder and embedTokens — their shapes vary per page.
        resetDspForModel("visionEncoder", visionEncoder);
        resetDspForModel("embedTokens", embedTokens);
        // Preserve decoder DSP plan if cachedPipeline is active (fixedBuffers mode).
        // The frozen decode plan has constant shapes and is valid for replay.
        if (cachedPipeline == null) {
            resetDspForModel("decoder", decoder);
        } else {
            log.info("Decoder DSP plan preserved for replay in resetSessions()");
        }

        // Drain ArrayCacheMemoryMgr exactly once at the page boundary. Component DSP resets
        // intentionally defer this thread-local sweep because only this owner can protect buffers
        // retained by every surviving model, including the cached decoder pipeline.
        // IMPORTANT: Pass model constant/variable buffers as protected so that force-closing
        // "constant-poisoned" intermediates doesn't destroy buffers the native DSP plan still
        // references. Without this, CUDA graph replay fails (0 replays, ~13x slowdown).
        IdentityHashMap<DataBuffer, Boolean> protectedBuffers = collectAllModelProtectedBuffers();
        ArrayCacheMemoryMgr.closeDeferredBuffers(protectedBuffers);
        ArrayCacheMemoryMgr.clearCacheState();
        ArrayCacheMemoryMgr.setEnableCache(false);

        // Force CUDA memory pool to release retained memory back to device allocator.
        // Without this, freed GPU allocations stay in the pool (not visible to cudaMemGetInfo)
        // and subsequent pages fail with OOM even though the memory is logically free.
        SameDiffMemoryUtils.trimAllDevicePools();
    }

    /**
     * GPU memory usage threshold (as fraction of total) above which decoder CUDA graphs
     * are fully reset instead of preserved. Default: 0.80 (80%).
     *
     * <p>Configurable via system property {@code kompile.vlm.gpu.memory.threshold}
     * or environment variable {@code KOMPILE_VLM_GPU_MEMORY_THRESHOLD}.</p>
     */
    private static final double GPU_MEMORY_PRESERVE_THRESHOLD;
    static {
        String prop = System.getProperty(ND4JSystemProperties.KOMPILE_VLM_GPU_MEMORY_THRESHOLD);
        if (prop == null) {
            prop = System.getenv("KOMPILE_VLM_GPU_MEMORY_THRESHOLD");
        }
        double threshold = 0.80;
        if (prop != null) {
            try {
                threshold = Double.parseDouble(prop);
                if (threshold <= 0.0 || threshold > 1.0) {
                    threshold = 0.80;
                }
            } catch (NumberFormatException ignored) {
                // use default
            }
        }
        GPU_MEMORY_PRESERVE_THRESHOLD = threshold;
    }

    /** Tracks whether a full decoder reset was forced due to memory pressure on the previous page. */
    private boolean lastDecoderResetWasFull = false;

    /**
     * Reset all inference sessions for DECODE page-to-page reuse with memory-budgeted
     * decoder graph preservation.
     *
     * <p><b>Strategy:</b></p>
     * <ul>
     *   <li>Vision encoder, embedTokens, and decoder all preserve their frozen DSP plans.
     *       Shape changes (different tile counts per page) are handled by the frozen
     *       multi-plan switch — the C++ NativePlanCache maintains shape-keyed plans
     *       and Java switches between them automatically.</li>
     *   <li>After preserve, GPU free memory is checked against a configurable threshold
     *       (default 80% usage). If usage exceeds the threshold, the decoder falls back to
     *       a full reset for this page to reclaim memory</li>
     *   <li>On subsequent pages after a forced full reset, the decoder returns to preserve
     *       mode (re-captured graphs from that page are reusable)</li>
     * </ul>
     *
     * <p>The threshold is configurable via:</p>
     * <ul>
     *   <li>System property: {@code -Dkompile.vlm.gpu.memory.threshold=0.80}</li>
     *   <li>Environment variable: {@code KOMPILE_VLM_GPU_MEMORY_THRESHOLD=0.80}</li>
     * </ul>
     *
     * <p>Call this between pages when decode shapes are invariant (always [1,1] for input_ids/position_ids).</p>
     */
    public void resetSessionsForDecode() {
        // Vision encoder and embedTokens: PRESERVE DSP plans across pages.
        // Their shapes vary per page (different tile counts, image sizes), but the
        // frozen multi-plan switch mechanism handles this — the C++ NativePlanCache
        // maintains shape-keyed plans and the Java executor switches between them
        // when placeholder shapes change. Destroying plans here wastes ~5-10s per
        // page for recompilation + CUDA graph recapture.
        log.info("Vision encoder and embedTokens DSP plans preserved (frozen multi-plan switch handles shape changes)");

        // PRESERVE the decoder DSP plan for replay across pages.
        //
        // With fixedBuffers mode (maxPrefillLength > 0 in GenerationPipelineConfig),
        // the decoder's DSP plan is compiled for constant shapes:
        //   - Prefill: [1, maxPrefillLength] (padded prompts)
        //   - Decode:  [1, 1] (single-token steps)
        //   - KV:      [batch, heads, maxKvLen, dim] (static buffers)
        //
        // These shapes are identical across pages, so the frozen plan (including
        // captured CUDA graphs) is valid for replay. The fixedBuffers path retains
        // pointer-stable prefill/decode/KV arrays and refills their contents in place
        // before replay; it must not clear the session or replace captured addresses.
        //
        // The previous full-reset approach destroyed the decoder plan between pages,
        // causing ~5s recompilation overhead per page and CUDA graph capture conflicts
        // that leaked ~3GB/page. The stale-KV-pointer crash that motivated the full
        // reset only occurs WITHOUT fixedBuffers — with fixedBuffers, retained KV and
        // decode arrays are refilled in place and the external-input identity stays stable.
        boolean memoryPressureHigh = isGpuMemoryPressureHigh();
        if (cachedPipeline != null && !memoryPressureHigh) {
            log.info("Decoder DSP plan preserved for replay (fixedBuffers mode active)");
            lastDecoderResetWasFull = false;
        } else {
            if (memoryPressureHigh) {
                log.warn("GPU memory pressure exceeded the preservation threshold; releasing the cached decoder pipeline");
                resetCachedDecoderPipelineForMemoryPressure();
            } else {
                log.info("No cached generation pipeline; resetting the decoder DSP plan");
                resetDspForModel("decoder", decoder);
            }
            lastDecoderResetWasFull = true;
        }

        // Drain deferred-close buffers and CUDA memory pool.
        // Without this, freed GPU allocations stay in the pool and
        // accumulate across pages until OOM.
        IdentityHashMap<DataBuffer, Boolean> protectedBuffers = collectAllModelProtectedBuffers();
        ArrayCacheMemoryMgr.closeDeferredBuffers(protectedBuffers);
        ArrayCacheMemoryMgr.clearCacheState();
        ArrayCacheMemoryMgr.setEnableCache(false);
        SameDiffMemoryUtils.trimAllDevicePools();

        logGpuMemoryStatus(lastDecoderResetWasFull
                ? "after decode reset (decoder plan released)"
                : "after decode reset (decoder plan preserved)");
    }

    /**
     * Fully release reusable decoder state at a page/generation boundary under memory pressure.
     * The pre-loaded decoder remains owned by this VLM; only its cached generation state,
     * SameDiff session, and captured dynamic plans are discarded. The next page recreates and
     * freezes a coherent pipeline lazily.
     */
    private void resetCachedDecoderPipelineForMemoryPressure() {
        if (cachedPipeline != null) {
            cachedPipeline.close();
            if (cachedPipeline.hasRetainedEmbeddingInputs()) {
                throw new IllegalStateException("Decoder borrower retirement failed; cached generation "
                        + "pipeline retained for a safe teardown retry");
            }
            cachedPipeline = null;
            cachedDecoderIOConfig = null;
        }
        cachedPromptTokenCount = 0;
        cachedMaxPrefillLength = 0;
        cachedProtocolKey = null;
        if (decoder != null) {
            decoder.resetSession();
            decoder.clearDynamicShapePlanCache();
        }
    }

    /**
     * End one logical adaptive generation batch while keeping model weights loaded.
     * Region replay is preserved within the batch; at its boundary every shape-keyed
     * vision/embed/decoder session and plan is retired so region shapes cannot leak into
     * the next full page.
     */
    public void releaseCachedGenerationPipeline() {
        log.info("Releasing cached generation pipeline at explicit batch boundary");
        resetCachedDecoderPipelineForMemoryPressure();
        retireModelSession("visionEncoder", visionEncoder);
        retireModelSession("embedTokens", embedTokens);
        // The sessions above have deliberately been retired. Protect model-owned weights only;
        // asking them for a DSP executor here would recreate empty sessions immediately after
        // retirement and leave another lifecycle object behind.
        IdentityHashMap<DataBuffer, Boolean> protectedBuffers = collectAllModelWeightBuffers();
        ArrayCacheMemoryMgr.closeDeferredBuffers(protectedBuffers);
        ArrayCacheMemoryMgr.clearCacheState();
        ArrayCacheMemoryMgr.setEnableCache(false);
        SameDiffMemoryUtils.trimAllDevicePools();
        logGpuMemoryStatus("after explicit generation batch release");
    }

    /**
     * Complete an adaptive page while preserving all fixed-shape replay plans.
     *
     * <p>Vision tiles are normalized to the model input shape before inference, and the
     * generation pipeline pads decoder inputs to its fixed envelope. Destroying the vision/embed
     * sessions after a crop therefore does not release a unique crop plan; it discards the warmed
     * plan needed by the next page and can spend minutes rebuilding it under memory pressure.
     * Emergency reclamation remains in {@link #reclaimAdaptiveInputPlansIfNeeded()} and runs only
     * when the configured GPU pressure threshold is actually exceeded.</p>
     */
    public void retireAdaptiveInputPlansPreservingDecoder() {
        resetSessionsForDecode();
        if (lastDecoderResetWasFull) {
            log.info("Adaptive page complete; preserved vision/embed replay and released decoder state under pressure");
        } else {
            log.info("Adaptive page complete; preserved warmed vision, embed, and padded decoder plans");
        }
        logGpuMemoryStatus(lastDecoderResetWasFull
                ? "after adaptive page cleanup (decoder pressure release)"
                : "after adaptive page cleanup (all replay plans preserved)");
    }

    /**
     * Reclaim accumulated crop-shape plans before the next adaptive sibling can OOM.
     * The decoder uses one fixed padded shape and remains protected/replayable.
     *
     * @return true when vision/embed plans were retired
     */
    public boolean reclaimAdaptiveInputPlansIfNeeded() {
        if (!isGpuMemoryPressureHigh()) return false;
        log.warn("Adaptive crop memory pressure is high; retiring superseded vision/embed plans "
                + "before the next crop while preserving the padded decoder plan");
        IdentityHashMap<DataBuffer, Boolean> protectedBuffers = collectAllModelWeightBuffers();
        collectDspExtInputBuffers(decoder, protectedBuffers);
        retireModelSession("visionEncoder", visionEncoder);
        retireModelSession("embedTokens", embedTokens);
        ArrayCacheMemoryMgr.closeDeferredBuffers(protectedBuffers);
        ArrayCacheMemoryMgr.clearCacheState();
        ArrayCacheMemoryMgr.setEnableCache(false);
        SameDiffMemoryUtils.trimAllDevicePools();
        logGpuMemoryStatus("after adaptive crop-plan reclamation");
        return true;
    }

    private void retireModelSession(String label, SameDiff model) {
        if (model == null) return;
        model.resetSession();
        model.clearDynamicShapePlanCache();
        log.info("Retired {} session and shape-keyed DSP plan cache at adaptive batch boundary", label);
    }

    /**
     * Query GPU memory and determine if usage exceeds the preservation threshold.
     *
     * <p>Uses {@code NativeOps.getDeviceFreeMemory()} and {@code getDeviceTotalMemory()}
     * to compute usage fraction. If the query fails (e.g., CPU-only mode), returns false
     * (assumes memory is fine).</p>
     *
     * @return true if GPU memory usage exceeds {@link #GPU_MEMORY_PRESERVE_THRESHOLD}
     */
    private boolean isGpuMemoryPressureHigh() {
        try {
            org.nd4j.nativeblas.NativeOps nativeOps =
                    org.nd4j.nativeblas.NativeOpsHolder.getInstance().getDeviceNativeOps();
            // Flush pending ops so free memory reflects actual state
            Nd4j.getExecutioner().commit();

            int device = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            long totalMem = nativeOps.getDeviceTotalMemory(device);
            long freeMem = nativeOps.getDeviceFreeMemory(device);

            if (totalMem <= 0) {
                log.debug("GPU memory query returned totalMem={}, skipping pressure check", totalMem);
                return false;
            }

            double usageFraction = 1.0 - ((double) freeMem / totalMem);
            long usedMB = (totalMem - freeMem) / (1024 * 1024);
            long totalMB = totalMem / (1024 * 1024);
            long freeMB = freeMem / (1024 * 1024);

            log.info("GPU memory: used={}MB / total={}MB (free={}MB, usage={}%, threshold={}%)",
                    usedMB, totalMB, freeMB,
                    String.format("%.1f", usageFraction * 100),
                    String.format("%.1f", GPU_MEMORY_PRESERVE_THRESHOLD * 100));

            return usageFraction > GPU_MEMORY_PRESERVE_THRESHOLD;
        } catch (Exception e) {
            log.debug("GPU memory pressure check failed (CPU mode?): {}", e.getMessage());
            return false;
        }
    }

    /**
     * Log current GPU memory status with a context label.
     */
    private void logGpuMemoryStatus(String context) {
        try {
            org.nd4j.nativeblas.NativeOps nativeOps =
                    org.nd4j.nativeblas.NativeOpsHolder.getInstance().getDeviceNativeOps();
            int device = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            long totalMem = nativeOps.getDeviceTotalMemory(device);
            long freeMem = nativeOps.getDeviceFreeMemory(device);
            if (totalMem > 0) {
                long usedMB = (totalMem - freeMem) / (1024 * 1024);
                long totalMB = totalMem / (1024 * 1024);
                double usagePct = (1.0 - (double) freeMem / totalMem) * 100;
                log.info("GPU memory {}: used={}MB / total={}MB ({}%)", context, usedMB, totalMB,
                        String.format("%.1f", usagePct));
            }
        } catch (Exception e) {
            // silent
        }
    }

    /**
     * Reset the DSP executor for a single SameDiff model, freeing its cached intermediates and
     * native plan state while deferring shared thread-local array-cache cleanup to the page owner.
     */
    private void resetDspForModel(String label, SameDiff model) {
        if (model == null) return;
        InferenceSession session = model.getOrCreateSession();
        DynamicShapePlanExecutor dsp = session.getDynamicShapePlanExecutor();
        if (dsp != null) {
            // ArrayCacheMemoryMgr is thread-local and shared by the decoder, vision encoder,
            // and embed model. Defer its cleanup until the page-level owner has reset every
            // disposable executor and collected the preserved decoder's external inputs.
            dsp.resetForNextPage(false);
            log.debug("DSP resetForNextPage completed for {} (shared cache cleanup deferred)", label);
        }
    }

    /**
     * Reset the DSP executor for a single SameDiff model, preserving decode-invariant state.
     */
    private void resetDspForModelDecode(String label, SameDiff model) {
        if (model == null) return;
        InferenceSession session = model.getOrCreateSession();
        DynamicShapePlanExecutor dsp = session.getDynamicShapePlanExecutor();
        if (dsp != null) {
            dsp.resetForNextPageDecode();
            log.debug("DSP resetForNextPageDecode completed for {}", label);
        }
    }

    /**
     * Collect DataBuffers from ALL sub-models' constants and variables that must NOT be
     * force-closed during ArrayCacheMemoryMgr cleanup. This prevents "constant-poisoned"
     * intermediates that share DataBuffers with model weights from being destroyed,
     * which would leave the native DSP plan with stale pointers and break CUDA graph replay.
     */
    private IdentityHashMap<DataBuffer, Boolean> collectAllModelProtectedBuffers() {
        IdentityHashMap<DataBuffer, Boolean> protectedBuffers = collectAllModelWeightBuffers();
        // Protect ext input arrays from frozen DSP plans.
        // These arrays (inputIds, posIds, causalMask, attentionMask, embeddings, etc.)
        // are tracked by the native plan for pointer stability. If trimAllDevicePools()
        // frees their device memory, the next execution sees stale pointers
        // → pointersStableCount never increases → no CUDA graph replay.
        collectDspExtInputBuffers(decoder, protectedBuffers);
        collectDspExtInputBuffers(visionEncoder, protectedBuffers);
        collectDspExtInputBuffers(embedTokens, protectedBuffers);
        return protectedBuffers;
    }

    private IdentityHashMap<DataBuffer, Boolean> collectAllModelWeightBuffers() {
        IdentityHashMap<DataBuffer, Boolean> protectedBuffers = new IdentityHashMap<>();
        collectModelBuffers(decoder, protectedBuffers);
        collectModelBuffers(visionEncoder, protectedBuffers);
        collectModelBuffers(embedTokens, protectedBuffers);
        return protectedBuffers;
    }

    private static void collectDspExtInputBuffers(SameDiff model, IdentityHashMap<DataBuffer, Boolean> out) {
        if (model == null) return;
        InferenceSession session = model.getOrCreateSession();
        DynamicShapePlanExecutor dsp = session.getDynamicShapePlanExecutor();
        if (dsp == null) return;
        // A frozen executor may alternate between multiple shape-keyed native plans. The active
        // snapshot alone is insufficient: page prefill can be current while an inactive decode
        // plan still owns the pointer-stable attention mask reused by the next page.
        dsp.collectRetainedExternalInputBuffers(out);
    }

    private static void collectModelBuffers(SameDiff model, IdentityHashMap<DataBuffer, Boolean> out) {
        if (model == null) return;
        for (SDVariable variable : model.variables()) {
            if (variable == null) continue;
            VariableType type = variable.getVariableType();
            if (type != VariableType.CONSTANT && type != VariableType.VARIABLE) continue;
            INDArray arr = variable.getArr();
            if (arr != null && !arr.wasClosed() && arr.data() != null) {
                out.put(arr.data(), Boolean.TRUE);
            }
        }
    }

    /**
     * Decode using GenerationPipeline for optimized throughput (Triton, CUDA graphs, native KV scatter).
     *
     * <p>By default the pipeline is cached after first creation with {@code maxPrefillLength}
     * set, enabling fixed-buffer mode in {@code generateNative()}. This means:</p>
     * <ul>
     *   <li>All prompts are padded/truncated to a fixed prefill length</li>
     *   <li>The DSP plan is compiled once on the first page and frozen</li>
     *   <li>Subsequent pages reuse the frozen plan via CUDA graph replay</li>
     *   <li>Only KV cache contents change between pages — no plan recompilation</li>
     * </ul>
     *
     * <p>Set {@code -Dvlm.benchmark.dynamicBuffers=true} for benchmark comparison with
     * the platform GenerationPipeline tests, which do not use fixed prefill buffers.</p>
     */
    private GenerationResult decodeWithStaticLoop(INDArray combinedEmbeddings, int[] promptIds,
                                                   int maxNewTokens, double temperature, boolean doSample) {
        VlmProtocolPlan defaultPlan = outputProtocols.prepare(
                VlmProtocolRequest.raw(""), tokenizer, config);
        return decodeWithStaticLoop(combinedEmbeddings, promptIds, maxNewTokens,
                temperature, doSample, defaultPlan);
    }

    private GenerationResult decodeWithStaticLoop(INDArray combinedEmbeddings, int[] promptIds,
                                                   int maxNewTokens, double temperature, boolean doSample,
                                                   VlmProtocolPlan protocolPlan) {
        SamplingConfig samplingCfg = doSample
                ? SamplingConfig.builder()
                        .maxNewTokens(maxNewTokens).temperature(temperature)
                        .topP(0.95).doSample(true).build()
                : SamplingConfig.greedy().toBuilder().maxNewTokens(maxNewTokens).build();
        return decodeWithStaticLoop(combinedEmbeddings, promptIds, samplingCfg, protocolPlan);
    }

    private GenerationResult decodeWithStaticLoop(INDArray combinedEmbeddings, int[] promptIds,
                                                   SamplingConfig requestedSampling,
                                                   VlmProtocolPlan protocolPlan) {
        SamplingConfig samplingCfg = requestedSampling != null
                ? requestedSampling : SamplingConfig.greedy();
        int maxNewTokens = samplingCfg.getMaxNewTokens();
        java.util.List<Integer> modelEosTokenIds = modelEosTokenIds();
        Integer modelEosTokenId = modelEosTokenIds.isEmpty() ? null : modelEosTokenIds.get(0);
        if ((protocolPlan == null || protocolPlan.isInheritModelEos()) && modelEosTokenId != null) {
            samplingCfg = samplingCfg.toBuilder().eosTokenId(modelEosTokenId).build();
        }
        // Secondary model EOS values remain alternate scalar terminators. Task/output protocol
        // stops are independent ordered sequences resolved from package metadata below.
        java.util.Set<Integer> additionalModelStopTokenIds = protocolPlan != null
                && !protocolPlan.isInheritModelEos() ? java.util.Collections.emptySet()
                : modelEosTokenIds.size() <= 1 ? java.util.Collections.emptySet()
                : new java.util.LinkedHashSet<>(modelEosTokenIds.subList(1, modelEosTokenIds.size()));
        String protocolKey = protocolPlan == null ? "fallback" : protocolPlan.cacheKey();

        try {
            boolean promptIncompatible = cachedMaxPrefillLength > 0
                    ? promptIds.length > cachedMaxPrefillLength
                    : cachedPromptTokenCount > 0 && promptIds.length != cachedPromptTokenCount;
            boolean protocolIncompatible = !java.util.Objects.equals(protocolKey, cachedProtocolKey);
            if (cachedPipeline != null && (promptIncompatible || protocolIncompatible)) {
                log.info("Rebuilding cached generation pipeline for capacity/protocol change "
                                + "prompt {} -> {} (padded capacity={}), protocol {} -> {}",
                        cachedPromptTokenCount, promptIds.length, cachedMaxPrefillLength,
                        cachedProtocolKey, protocolKey);
                GenerationPipeline incompatiblePipeline = cachedPipeline;
                incompatiblePipeline.close();
                if (incompatiblePipeline.hasRetainedEmbeddingInputs()) {
                    throw new IllegalStateException("Decoder borrower retirement failed while rebuilding the "
                            + "cached generation pipeline; retaining it for a safe teardown retry");
                }
                cachedPipeline = null;
                cachedDecoderIOConfig = null;
                cachedPromptTokenCount = 0;
                cachedMaxPrefillLength = 0;
                cachedProtocolKey = null;
            }
            if (cachedPipeline == null) {
                cachedDecoderIOConfig = ModelIOConfig.discover(decoder);
                boolean dynamicBenchmarkBuffers = Boolean.getBoolean("vlm.benchmark.dynamicBuffers");
                int maxPrefill = dynamicBenchmarkBuffers ? 0 : computeMaxPrefillLength(promptIds.length, maxNewTokens);
                int effectiveMaxKvCacheLength = dynamicBenchmarkBuffers ? 0 : resolveContextWindow();
                log.info("Creating cached GenerationPipeline (mode={}, maxPrefillLength={}, promptLen={}, maxNewTokens={}, maxKvCacheLength={})",
                        dynamicBenchmarkBuffers ? "dynamic-benchmark" : "fixedBuffers",
                        maxPrefill, promptIds.length, maxNewTokens, effectiveMaxKvCacheLength);

                GenerationPipelineConfig.GenerationPipelineConfigBuilder pipelineConfigBuilder = GenerationPipelineConfig.builder()
                        .decoder(decoder)
                        .embedTokens(embedTokens)
                        .tokenizer(tokenizer)
                        .ioConfig(cachedDecoderIOConfig)
                        .samplingConfig(samplingCfg)
                        .additionalStopTokenIds(additionalModelStopTokenIds)
                        .additionalStopTokenSequences(protocolPlan == null
                                ? java.util.Collections.emptyList()
                                : protocolPlan.tokenSequences())
                        .inheritModelStopTokenIds(protocolPlan == null || protocolPlan.isInheritModelEos())
                        .inheritDefaultEos(protocolPlan == null || protocolPlan.isInheritModelEos())
                        .inheritChatTemplateStopTokenIds(protocolPlan == null
                                || protocolPlan.isInheritChatTemplateStops())
                        .maxNewTokens(maxNewTokens)
                        .maxKvCacheLength(effectiveMaxKvCacheLength)
                        .hiddenSize(config != null && config.getHiddenSize() != null ? config.getHiddenSize() : 0)
                        // MultiPartModelLoader/SameDiffOptimizationCache already loaded the
                        // optimized decoder artifact. Re-optimizing here creates an owned graph
                        // copy (including every model weight) each time pressure rebuilds the
                        // generation borrower, defeating cleanup and exhausting GPU memory.
                        .graphOptimizerEnabled(false)
                        .benchmarkConfig(benchmarkConfig);
                if (!dynamicBenchmarkBuffers) {
                    pipelineConfigBuilder.maxPrefillLength(maxPrefill);
                }

                cachedPipeline = GenerationPipeline.create(pipelineConfigBuilder.build());
                cachedPromptTokenCount = promptIds.length;
                cachedMaxPrefillLength = maxPrefill;
                cachedProtocolKey = protocolKey;
            } else {
                log.info("Reusing cached GenerationPipeline (promptLen={}, padded capacity={}, decoder plan preserved)",
                        promptIds.length, cachedMaxPrefillLength);
                cachedPromptTokenCount = promptIds.length;
            }

            // Sampling is pure logits post-processing and does not alter frozen plan shapes.
            // Refresh it on every call so adaptive repetition penalties and filters are not
            // silently replaced by the first full-page request's policy.
            cachedPipeline.setSamplingConfig(samplingCfg);
            return cachedPipeline.generate(combinedEmbeddings, promptIds, maxNewTokens);
        } catch (IOException e) {
            throw new RuntimeException("Failed to create generation pipeline", e);
        }
    }

    /** Resolve one request against the package-declared protocol registry. */
    public VlmProtocolPlan prepareOutputProtocol(VlmProtocolRequest request) {
        return outputProtocols.prepare(request, tokenizer, config);
    }

    /** Parse/render one generation with the same selected protocol used for prompting. */
    public VlmProtocolOutput processOutputProtocol(VlmProtocolRequest request, GenerationResult result) {
        VlmProtocolPlan plan = outputProtocols.prepare(request, tokenizer, config);
        return outputProtocols.process(request, plan, result, tokenizer);
    }

    public GenerationResult mergeOutputProtocolRegions(VlmProtocolRequest request,
                                                       java.util.List<GenerationResult> regions) {
        VlmProtocolPlan plan = outputProtocols.prepare(request, tokenizer, config);
        return outputProtocols.mergeRegions(request, plan, regions, tokenizer);
    }

    public String getDefaultOutputProtocolId() {
        return outputProtocols.defaultProtocolId();
    }

    /** Resolve an EOS-oriented generation budget from the actual prompt and model context. */
    int resolveGenerationBudget(int promptTokenCount, int requestedMaxTokens) {
        int contextWindow = resolveContextWindow();
        if (contextWindow <= 0) {
            return requestedMaxTokens > 0 ? requestedMaxTokens : 512;
        }
        int available = contextWindow - promptTokenCount;
        if (available <= 0) {
            throw new IllegalStateException("Prompt length " + promptTokenCount
                    + " exhausts the model context window " + contextWindow);
        }
        return requestedMaxTokens > 0 ? Math.min(requestedMaxTokens, available) : available;
    }

    private int resolveContextWindow() {
        int contextWindow = config != null && config.getMaxPositionEmbeddings() != null
                ? config.getMaxPositionEmbeddings() : 0;
        if (maxKvLen > 0) {
            contextWindow = contextWindow > 0 ? Math.min(contextWindow, maxKvLen) : maxKvLen;
        }
        return contextWindow;
    }

    java.util.List<Integer> modelEosTokenIds() {
        return config == null ? java.util.Collections.emptyList() : config.getEosTokenIds();
    }

    static int[] combinedPromptIds(int imageTokenCount, int imageTokenId, int[] textPromptIds) {
        if (imageTokenCount < 0) throw new IllegalArgumentException("imageTokenCount must be >= 0");
        int[] textIds = textPromptIds == null ? new int[0] : textPromptIds;
        int[] combined = new int[imageTokenCount + textIds.length];
        java.util.Arrays.fill(combined, 0, imageTokenCount, imageTokenId);
        System.arraycopy(textIds, 0, combined, imageTokenCount, textIds.length);
        return combined;
    }

    /**
     * Compute the fixed prefill length for stable DSP plan shapes across pages.
     *
     * <p>The physical prefill buffer is rounded to the next 256-token boundary with 20%
     * headroom and capped by the model context. Padding is only a replay shape: semantic
     * generation capacity remains based on the real prompt length.</p>
     *
     * @param currentPromptLen the prompt token count from the first page
     * @param maxNewTokens maximum tokens to generate per page
     * @return the fixed prefill length for all subsequent pages
     */
    private int computeMaxPrefillLength(int currentPromptLen, int maxNewTokens) {
        int contextWindow = resolveContextWindow();
        if (contextWindow > 0 && currentPromptLen > contextWindow) {
            throw new IllegalStateException("Prompt length " + currentPromptLen
                    + " exceeds fixed KV/context capacity " + contextWindow);
        }

        // Select a capacity, not the first request's exact length. The logical prompt length
        // still drives cache_position and maxNewTokens, so this padding never consumes context.
        int prefill = (int) Math.ceil(currentPromptLen * 1.2);
        prefill = Math.max(prefill, 512);
        prefill = ((prefill + 255) / 256) * 256;
        if (contextWindow > 0) {
            prefill = Math.min(prefill, contextWindow);
        }
        prefill = Math.max(prefill, currentPromptLen);
        log.info("maxPrefillLength computed: {} (promptLen={}, maxNewTokens={}, maxKvLen={})",
                prefill, currentPromptLen, maxNewTokens, this.maxKvLen);
        return prefill;
    }

    /**
     * Compile all sub-models with Triton MAX_AUTOTUNE for optimal inference performance.
     * This triggers Triton kernel compilation (CONST_GEN, GATHER, CONCAT, SPLIT, STACK,
     * NORMALIZATION, ATTENTION) and CUDA graph capture.
     * Must be called AFTER model loading and AFTER Environment.applyOptimalLLMConfig().
     * First call is slow (compiles kernels); subsequent calls are fast (cached).
     */
    public void compileWithTriton() {
        try {
            org.nd4j.autodiff.samediff.execution.DspCompilationMode mode =
                    org.nd4j.autodiff.samediff.execution.DspCompilationMode.MAX_AUTOTUNE;
            log.info("Compiling VLM sub-models with Triton MAX_AUTOTUNE...");
            long start = System.currentTimeMillis();

            if (decoder != null) {
                log.info("  Compiling decoder...");
                decoder.compileNativeDynamicShapePlan(mode);
            }
            if (visionEncoder != null) {
                log.info("  Compiling vision encoder...");
                visionEncoder.compileNativeDynamicShapePlan(mode);
            }
            if (embedTokens != null) {
                log.info("  Compiling embed_tokens...");
                embedTokens.compileNativeDynamicShapePlan(mode);
            }

            long elapsed = System.currentTimeMillis() - start;
            log.info("Triton compilation complete in {}ms", elapsed);
        } catch (Exception e) {
            log.warn("Triton compilation failed (will fall back to standard execution): {}", e.getMessage());
        }
    }

    /**
     * Check if the model is valid and usable.
     *
     * @return true if the model can be used
     */
    public boolean isValid() {
        return !closed && visionEncoder != null && decoder != null && tokenizer != null;
    }

    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("Model has been closed");
        }
    }


    // =========================================================================
    // Device-Aware Methods for Multi-Chip Backend Support
    // =========================================================================

    /**
     * Preprocess and encode an image on a specific device.
     *
     * @param imageFile the image file
     * @param device the target device for processing
     * @return image embeddings on the specified device
     * @throws IOException if loading fails
     */
    public INDArray encodeImageOnDevice(File imageFile, DeviceDescriptor device) throws IOException {
        checkNotClosed();

        INDArray image = imagePreprocessor.preprocessOnDevice(imageFile, device);
        return encodeImageOnDevice(image, device);
    }

    /**
     * Encode an already preprocessed image on a specific device.
     *
     * @param image the preprocessed image tensor
     * @param device the target device
     * @return image embeddings on the specified device
     */
    public INDArray encodeImageOnDevice(INDArray image, DeviceDescriptor device) {
        checkNotClosed();

        INDArray normalized = normalizeVisionInputShape(image);
        // Ensure image is on target device
        ensureOnDevice(normalized, device);

        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put(visionEncoderIOConfig.getPixelValuesName(), normalized);

        Map<String, INDArray> outputs = visionEncoder.output(inputs, visionEncoderIOConfig.getOutputNames());
        INDArray result = outputs.get(visionEncoderIOConfig.getPrimaryOutputName());

        // Ensure output is on target device
        ensureOnDevice(result, device);
        return result;
    }

    private INDArray normalizeVisionInputShape(INDArray image) {
        if (image == null) {
            throw new IllegalArgumentException("image must not be null");
        }
        // Always pass rank-4 [batch, C, H, W] to the vision encoder.
        // The ONNX model may declare a rank-5 placeholder [batch, frames, C, H, W],
        // but the internal reshape_no_copy ops use ONNX zero-semantics shape constants
        // (e.g. [0,0,0,0]) that copy dims positionally from the input. With rank-5
        // input, [0,0,0,0] copies [batch, frames, C, H] instead of [batch, C, H, W],
        // producing a shape mismatch. Rank-4 input makes the reshape identity-correct.
        // SameDiff's dynamic shape system accepts any input rank regardless of what
        // the placeholder declared.
        log.info("normalizeVisionInputShape: image.rank()={}, shape={}",
                image.rank(), java.util.Arrays.toString(image.shape()));
        if (image.rank() == 5) {
            // [batch, frames, C, H, W] -> [batch*frames, C, H, W]
            long[] shape = image.shape();
            long batchFrames = shape[0] * shape[1];
            INDArray reshaped = image.reshape(batchFrames, shape[2], shape[3], shape[4]);
            log.info("normalizeVisionInputShape: collapsed rank-5 to rank-4 {}", java.util.Arrays.toString(reshaped.shape()));
            return reshaped;
        }
        return image;
    }

    private boolean visionEncoderExpectsFramedInput() {
        SDVariable pixelValues = visionEncoder.getVariable(visionEncoderIOConfig.getPixelValuesName());
        if (pixelValues == null) {
            log.info("visionEncoderExpectsFramedInput: pixelValues variable is null");
            return false;
        }
        long[] shape = pixelValues.getShape();
        log.info("visionEncoderExpectsFramedInput: pixelValues.getShape()={}, length={}",
                shape != null ? java.util.Arrays.toString(shape) : "null",
                shape != null ? shape.length : -1);
        return shape != null && shape.length == 5;
    }

    /**
     * Embed text tokens on a specific device.
     *
     * @param tokenIds the token IDs
     * @param device the target device
     * @return text embeddings on the specified device
     */
    public INDArray embedTextOnDevice(int[] tokenIds, DeviceDescriptor device) {
        checkNotClosed();

        INDArray inputIds = Nd4j.createFromArray(tokenIds).reshape(1, tokenIds.length);
        ensureOnDevice(inputIds, device);

        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("input_ids", inputIds);

        Map<String, INDArray> outputs = embedTokens.output(inputs, "inputs_embeds");
        INDArray result = outputs.get("inputs_embeds");

        ensureOnDevice(result, device);
        return result;
    }

    /**
     * Generate text from an image on a specific device.
     *
     * @param image the preprocessed image
     * @param prompt the text prompt
     * @param device the target device for inference
     * @return the generated text
     */
    public String generateOnDevice(INDArray image, String prompt, DeviceDescriptor device) {
        return generateOnDevice(image, prompt, 512, 1.0, true, device);
    }

    /**
     * Generate text from an image on a specific device with full parameters.
     *
     * @param image the preprocessed image
     * @param prompt the text prompt
     * @param maxNewTokens maximum tokens to generate
     * @param temperature sampling temperature
     * @param doSample whether to sample
     * @param device the target device
     * @return the generated text
     */
    public String generateOnDevice(INDArray image, String prompt, int maxNewTokens,
                                   double temperature, boolean doSample, DeviceDescriptor device) {
        return generateOnDeviceWithMetrics(image, prompt, maxNewTokens, temperature, doSample, device).getText();
    }

    /**
     * Generate text from an image on a specific device, returning detailed metrics.
     *
     * @param image the preprocessed image
     * @param prompt the text prompt
     * @param device the target device for inference
     * @return the generation result with metrics
     */
    public GenerationResult generateOnDeviceWithMetrics(INDArray image, String prompt, DeviceDescriptor device) {
        return generateOnDeviceWithMetrics(image, prompt, 512, 1.0, true, device);
    }

    /**
     * Generate text from an image on a specific device with full parameters, returning detailed metrics.
     *
     * @param image the preprocessed image
     * @param prompt the text prompt
     * @param maxNewTokens maximum tokens to generate
     * @param temperature sampling temperature
     * @param doSample whether to sample
     * @param device the target device
     * @return the generation result with metrics
     */
    public GenerationResult generateOnDeviceWithMetrics(INDArray image, String prompt, int maxNewTokens,
                                                        double temperature, boolean doSample, DeviceDescriptor device) {
        checkNotClosed();

        // Encode image on device
        INDArray imageEmbeddings = encodeImageOnDevice(image, device);

        // Encode prompt on device
        Encoding promptEncoding = tokenizer.encode(prompt, true);
        int[] promptIds = promptEncoding.getIds();
        int promptTokenCount = promptIds.length;
        INDArray textEmbeddings = embedTextOnDevice(promptIds, device);

        // Combine embeddings
        INDArray combinedEmbeddings = combineEmbeddings(imageEmbeddings, textEmbeddings);
        ensureOnDevice(combinedEmbeddings, device);

        // Autoregressive generation with timing
        StringBuilder generated = new StringBuilder();
        List<Integer> generatedTokenIds = new ArrayList<>();
        long startNanos = System.nanoTime();
        long firstTokenLatencyNanos = 0;
        int generatedTokenCount = 0;
        GenerationResult.FinishReason finishReason = GenerationResult.FinishReason.MAX_TOKENS;

        for (int i = 0; i < maxNewTokens; i++) {
            Map<String, INDArray> decoderInputs = new HashMap<>();
            decoderInputs.put("inputs_embeds", combinedEmbeddings);

            Map<String, INDArray> outputs = decoder.output(decoderInputs, "logits");
            INDArray logits = outputs.get("logits");

            int nextTokenId;
            if (!doSample || temperature <= 0) {
                INDArray lastLogits = logits.get(NDArrayIndex.point(0),
                        NDArrayIndex.point(logits.shape()[1] - 1));
                nextTokenId = SamplerUtils.argmax(lastLogits);
            } else {
                INDArray scaledLogits = logits.div(temperature);
                INDArray probs = Nd4j.nn().softmax(scaledLogits, 2);
                INDArray lastProbs = probs.get(NDArrayIndex.point(0),
                        NDArrayIndex.point(probs.shape()[1] - 1));
                nextTokenId = SamplerUtils.argmax(lastProbs);
            }

            // Record first token latency
            if (generatedTokenCount == 0) {
                firstTokenLatencyNanos = System.nanoTime() - startNanos;
            }
            generatedTokenCount++;
            generatedTokenIds.add(nextTokenId);

            if (nextTokenId == tokenizer.getEosTokenId()) {
                finishReason = GenerationResult.FinishReason.EOS;
                break;
            }

            String tokenText = tokenizer.decode(new int[]{nextTokenId}, true);
            generated.append(tokenText);

            INDArray newTokenEmbed = embedTextOnDevice(new int[]{nextTokenId}, device);
            combinedEmbeddings = Nd4j.concat(1, combinedEmbeddings, newTokenEmbed);
        }

        long totalNanos = System.nanoTime() - startNanos;
        long totalMs = totalNanos / 1_000_000;
        long firstTokenMs = firstTokenLatencyNanos / 1_000_000;
        int[] tokenIdArray = generatedTokenIds.stream().mapToInt(Integer::intValue).toArray();

        return GenerationResult.builder()
                .text(generated.toString())
                .tokenIds(tokenIdArray)
                .generatedTokenCount(generatedTokenCount)
                .promptTokenCount(promptTokenCount)
                .totalTokenCount(promptTokenCount + generatedTokenCount)
                .finishReason(finishReason)
                .firstTokenLatencyMs(firstTokenMs)
                .generationTimeMs(totalMs)
                .tokensPerSecond(totalNanos > 0 ? (generatedTokenCount * 1_000_000_000.0) / totalNanos : 0)
                .build();
    }

    /**
     * Generate using workspace memory on a specific device.
     *
     * @param image the preprocessed image
     * @param prompt the text prompt
     * @param workspace the workspace for memory allocation
     * @param device the target device
     * @return the generated text
     */
    public String generateInWorkspace(INDArray image, String prompt,
                                      MultiBackendWorkspace workspace, DeviceDescriptor device) {
        try (MemoryWorkspace ws = workspace.notifyScopeEntered()) {
            return generateOnDevice(image, prompt, device);
        }
    }

    /**
     * Prefetch image data to the target device asynchronously.
     *
     * @param imageFile the image file to prefetch
     * @param device the target device
     * @return future that completes when prefetch is done
     */
    public CompletableFuture<INDArray> prefetchImage(File imageFile, DeviceDescriptor device) {
        return imagePreprocessor.preprocessAsync(imageFile, device);
    }

    /**
     * Transfer model inputs to a specific device.
     *
     * @param inputs map of input name to array
     * @param device target device
     */
    public void transferInputsToDevice(Map<String, INDArray> inputs, DeviceDescriptor device) {
        for (INDArray array : inputs.values()) {
            ensureOnDevice(array, device);
        }
    }

    /**
     * Get model inputs on a specific device.
     *
     * @param image the preprocessed image
     * @param prompt the text prompt
     * @param device the target device
     * @return map of model inputs on the device
     */
    public Map<String, INDArray> getModelInputsOnDevice(INDArray image, String prompt,
                                                         DeviceDescriptor device) {
        checkNotClosed();

        INDArray imageEmbeddings = encodeImageOnDevice(image, device);
        Encoding promptEncoding = tokenizer.encode(prompt, true);
        INDArray textEmbeddings = embedTextOnDevice(promptEncoding.getIds(), device);
        INDArray combinedEmbeddings = combineEmbeddings(imageEmbeddings, textEmbeddings);

        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("inputs_embeds", combinedEmbeddings);
        inputs.put("image_embeddings", imageEmbeddings);
        inputs.put("text_embeddings", textEmbeddings);

        return inputs;
    }

    /**
     * Ensure an array is available on the specified device.
     *
     * @param array the array to transfer
     * @param device the target device
     */
    private void ensureOnDevice(INDArray array, DeviceDescriptor device) {
        DeviceDescriptor effectiveDevice = device != null ? device : this.targetDevice;
        if (effectiveDevice == null) {
            return;
        }

        DataBuffer buffer = array.data();
        if (buffer != null && buffer.isHybrid()) {
            buffer.asHybrid().ensureAvailableOn(effectiveDevice);
        }
    }

    /**
     * Create a VLM configured for a specific device.
     *
     * @param modelDir the model directory
     * @param device the target device
     * @return VLM configured for the device
     * @throws IOException if loading fails
     */
    public static VisionLanguageModel forDevice(File modelDir, DeviceDescriptor device) throws IOException {
        MultiPartModelLoader.LoadedModel loaded = MultiPartModelLoader.load(modelDir);
        return VisionLanguageModel.builder()
                .visionEncoder(loaded.getVisionEncoder())
                .embedTokens(loaded.getEmbedTokens())
                .decoder(loaded.getDecoder())
                .tokenizer(loaded.getTokenizer())
                .imagePreprocessor(VLMImagePreprocessor.forDevice(loaded.getImagePreprocessor().getConfig(), device))
                .config(loaded.getConfig())
                .outputProtocols(loaded.getOutputProtocols())
                .targetDevice(device)
                .build();
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            if (cachedPipeline != null) {
                try {
                    cachedPipeline.close();
                } catch (Exception e) {
                    log.warn("Error closing cached pipeline", e);
                }
                cachedPipeline = null;
                cachedDecoderIOConfig = null;
                cachedPromptTokenCount = 0;
                cachedMaxPrefillLength = 0;
                cachedProtocolKey = null;
            }
            closeModel("decoder", decoder);
            closeModel("embedTokens", embedTokens);
            closeModel("visionEncoder", visionEncoder);

            if (tokenizer != null) {
                try {
                    tokenizer.close();
                } catch (Exception e) {
                    log.warn("Error closing tokenizer", e);
                }
            }
            if (imagePreprocessor != null) {
                imagePreprocessor.shutdown();
            }
            if (workspace != null) {
                try {
                    workspace.close();
                } catch (Exception e) {
                    log.warn("Error closing workspace", e);
                }
            }
        }
    }

    private void closeModel(String label, SameDiff model) {
        if (model == null) {
            return;
        }

        try {
            model.clearPlaceholders(true);
        } catch (Exception e) {
            log.warn("Error clearing placeholders for {}", label, e);
        }
        try {
            model.clearOpInputs();
        } catch (Exception e) {
            log.warn("Error clearing op inputs for {}", label, e);
        }
        try {
            model.resetSession();
        } catch (Exception e) {
            log.warn("Error resetting session for {}", label, e);
        }
        try {
            SameDiffMemoryUtils.freeModelArrays(model);
        } catch (Exception e) {
            log.warn("Error freeing model arrays for {}", label, e);
        }
        try {
            model.close();
        } catch (Exception e) {
            log.warn("Error closing {}", label, e);
        }
    }
}
