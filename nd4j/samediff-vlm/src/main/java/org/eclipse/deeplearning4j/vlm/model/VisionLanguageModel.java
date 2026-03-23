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
import org.eclipse.deeplearning4j.llm.generation.BatchGenerationState;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.SameDiffMemoryUtils;
import org.eclipse.deeplearning4j.llm.generation.SamplerUtils;
import org.eclipse.deeplearning4j.llm.tokenizer.Encoding;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.eclipse.deeplearning4j.llm.generation.DecoderUtils;
import org.eclipse.deeplearning4j.llm.generation.NgramSpeculator;
import org.eclipse.deeplearning4j.llm.generation.SamplingConfig;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
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

    // Device-aware fields for multi-chip support
    @Setter
    private DeviceDescriptor targetDevice;

    @Setter
    private MultiBackendWorkspace workspace;

    private volatile boolean closed = false;

    @Builder
    private VisionLanguageModel(
            SameDiff visionEncoder,
            SameDiff embedTokens,
            SameDiff decoder,
            Tokenizer tokenizer,
            VLMImagePreprocessor imagePreprocessor,
            ModelConfig config,
            DeviceDescriptor targetDevice,
            MultiBackendWorkspace workspace) {
        this.visionEncoder = visionEncoder;
        this.embedTokens = embedTokens;
        this.decoder = decoder;
        this.tokenizer = tokenizer;
        this.imagePreprocessor = imagePreprocessor;
        this.config = config;
        this.targetDevice = targetDevice;
        this.workspace = workspace;
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
        MultiPartModelLoader.LoadedModel loaded = MultiPartModelLoader.load(modelDir);
        return loaded.toVisionLanguageModel();
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
        MultiPartModelLoader.LoadedModel loaded = MultiPartModelLoader.loadSmolDocling(modelDir);
        return loaded.toVisionLanguageModel();
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
                8 * 1024 * 1024);
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

        return VisionLanguageModel.builder()
                .visionEncoder(visionEncoder)
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .imagePreprocessor(VLMImagePreprocessor.defaultPreprocessor())
                .build();
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

        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("pixel_values", normalizeVisionInputShape(image));

        Map<String, INDArray> outputs = visionEncoder.output(inputs, "image_embeds");
        return outputs.get("image_embeds");
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
        return outputs.get("inputs_embeds");
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
        return generate(image, prompt, 512, 1.0, true);
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
        return generateWithMetrics(image, prompt, 512, 1.0, true);
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
        checkNotClosed();

        // Encode image
        INDArray imageEmbeddings = encodeImage(image);

        // Encode prompt
        Encoding promptEncoding = tokenizer.encode(prompt, true);
        int[] promptIds = promptEncoding.getIds();
        int promptTokenCount = promptIds.length;
        INDArray textEmbeddings = embedText(promptIds);

        // Combine embeddings (image before text for most VLMs)
        INDArray combinedEmbeddings = combineEmbeddings(imageEmbeddings, textEmbeddings);

        // Discover decoder inputs/outputs for KV cache
        List<String> decoderInputNames = decoder.inputs();
        String logitsOutputName = DecoderUtils.findLogitsOutputName(decoder);
        DecoderUtils.KVCacheNames kvNames = DecoderUtils.findKVCacheOutputNames(decoder);
        boolean useKvCache = !kvNames.keyNames.isEmpty() && !kvNames.valueNames.isEmpty();
        long hiddenSize = config != null && config.getHiddenSize() != null ? config.getHiddenSize() : 0;

        // Build list of all outputs to request
        List<String> allOutputNames = new ArrayList<>();
        allOutputNames.add(logitsOutputName != null ? logitsOutputName : "logits");
        if (useKvCache) {
            allOutputNames.addAll(kvNames.keyNames);
            allOutputNames.addAll(kvNames.valueNames);
        }

        // Autoregressive generation with KV cache
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

        for (int i = 0; i < maxNewTokens; i++) {
            // Build decoder inputs
            Map<String, INDArray> decoderInputMap = new HashMap<>();
            long currentSeqLen = currentEmbeddings.shape()[1];
            long totalSeqLen = currentSeqLen + pastSeqLen;

            for (String inputName : decoderInputNames) {
                if (inputName.equals("inputs_embeds")) {
                    decoderInputMap.put(inputName, currentEmbeddings);
                } else if (inputName.equals("attention_mask")) {
                    decoderInputMap.put(inputName, Nd4j.ones(DataType.LONG, batchSize, totalSeqLen));
                } else if (inputName.equals("_causal_mask")) {
                    decoderInputMap.put(inputName, DecoderUtils.buildCausalMask(currentSeqLen, totalSeqLen));
                } else if (inputName.equals("position_ids")) {
                    decoderInputMap.put(inputName, Nd4j.arange(pastSeqLen, pastSeqLen + currentSeqLen)
                            .reshape(1, currentSeqLen).castTo(DataType.LONG));
                } else if (useKvCache && inputName.startsWith("past_key_values.")) {
                    String presentName = inputName.replace("past_key_values", "present");
                    if (kvCache.containsKey(presentName)) {
                        decoderInputMap.put(inputName, kvCache.get(presentName));
                    } else {
                        decoderInputMap.put(inputName, DecoderUtils.createEmptyKvCache(
                                decoder, inputName, batchSize, hiddenSize));
                    }
                }
            }

            // Run decoder, requesting logits + KV cache outputs
            Map<String, INDArray> outputs = decoder.output(decoderInputMap,
                    allOutputNames.toArray(new String[0]));
            INDArray logits = outputs.get(allOutputNames.get(0));

            // Update KV cache
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

            // Get next token (greedy or sampling)
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

            // Check for EOS
            if (nextTokenId == tokenizer.getEosTokenId()) {
                finishReason = GenerationResult.FinishReason.EOS;
                break;
            }

            // Decode token
            String tokenText = tokenizer.decode(new int[]{nextTokenId}, true);
            generated.append(tokenText);

            // For next iteration: only embed the new token (not the full sequence)
            if (useKvCache) {
                pastSeqLen += currentSeqLen;
                INDArray prevEmbeddings = currentEmbeddings;
                currentEmbeddings = embedText(new int[]{nextTokenId});
                if (prevEmbeddings != combinedEmbeddings) {
                    SameDiffMemoryUtils.safeClose(prevEmbeddings);
                }
                decoder.clearPlaceholders(false);
            } else {
                // Fallback: no KV cache, grow embeddings (O(n²) per token)
                INDArray newTokenEmbed = embedText(new int[]{nextTokenId});
                currentEmbeddings = Nd4j.concat(1, currentEmbeddings, newTokenEmbed);
            }
        }

        // Clean up KV cache
        if (kvCache != null) {
            for (INDArray v : kvCache.values()) {
                SameDiffMemoryUtils.safeClose(v);
            }
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
        String logitsOutputName = DecoderUtils.findLogitsOutputName(decoder);
        DecoderUtils.KVCacheNames kvNames = DecoderUtils.findKVCacheOutputNames(decoder);
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
                    decoderInputMap.put(inputName, DecoderUtils.buildCausalMask(batchSize, currentSeqLen, totalSeqLen));
                } else if (inputName.equals("position_ids")) {
                    INDArray posIds = Nd4j.arange(pastSeqLen, pastSeqLen + currentSeqLen)
                            .reshape(1, currentSeqLen).castTo(DataType.LONG);
                    decoderInputMap.put(inputName, Nd4j.tile(posIds, batchSize, 1));
                } else if (useKvCache && inputName.startsWith("past_key_values.")) {
                    String presentName = inputName.replace("past_key_values", "present");
                    if (kvCache.containsKey(presentName)) {
                        decoderInputMap.put(inputName, kvCache.get(presentName));
                    } else {
                        decoderInputMap.put(inputName, DecoderUtils.createEmptyKvCache(
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
                            specDecoderInputMap.put(inputName, DecoderUtils.buildCausalMask(batchSize, currentSpecSeqLen, totalSpecSeqLen));
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
            inputs.put("pixel_values", batched);
            Map<String, INDArray> outputs = visionEncoder.output(inputs, "image_embeds");
            return outputs.get("image_embeds");
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
            INDArray visionFrameInput = frameTensor.reshape(1, 1, frameTensor.size(1), frameTensor.size(2), frameTensor.size(3));

            Map<String, INDArray> inputs = new HashMap<>();
            inputs.put("pixel_values", visionFrameInput);

            // Create pixel attention mask for this frame
            ImageTiler.ContentRegion region = splitResult.contentRegions.get(f);
            INDArray pixelMask = ImageTiler.createPixelAttentionMask(
                    region.width, region.height, targetSize);
            // Only add if the encoder accepts it
            if (visionEncoder.getVariable("pixel_attention_mask") != null) {
                inputs.put("pixel_attention_mask", pixelMask);
            }

            // Run vision encoder for this frame
            Map<String, INDArray> outputs = visionEncoder.output(inputs);
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

            log.info("Frame {}/{}: output '{}' shape={}, min={}, max={}",
                    f + 1, numFrames, selected.name,
                    java.util.Arrays.toString(embedding.shape()),
                    embedding.minNumber(), embedding.maxNumber());

            for (var entry : outputs.entrySet()) {
                INDArray arr = entry.getValue();
                if (arr != null && !arr.wasClosed()) {
                    arr.setCloseable(true);
                    arr.close();
                }
            }

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
        checkNotClosed();
        if (pageSplitResults == null || pageSplitResults.isEmpty()) {
            return new GenerationResult[0];
        }

        SamplingConfig samplingConfig = SamplingConfig.builder()
                .maxNewTokens(maxNewTokens)
                .doSample(doSample)
                .temperature(temperature)
                .build();

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

            INDArray visionEmbeddings = encodeImageTiled(splitResult, targetSize);
            try {
                int totalFrames = splitResult.getTotalFrames();
                int imageSeqLenPerFrame = (int) (visionEmbeddings.size(1) / Math.max(totalFrames, 1));
                results[i] = generateFromEmbeddings(
                        visionEmbeddings,
                        userPrompt,
                        samplingConfig,
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
        checkNotClosed();

        long startNanos = System.nanoTime();
        int maxTokens = config.getMaxNewTokens();
        double temperature = config.getTemperature();
        boolean doSample = config.isDoSample();

        // Build prompt with image tokens
        int totalFrames = (imageRows > 0 && imageCols > 0)
                ? imageRows * imageCols + 1  // tiles + global
                : 1;  // global only
        int totalImageTokens = totalFrames * imageSeqLenPerFrame;
        String imagePrompt = ImagePromptBuilder.buildImagePromptString(
                imageRows, imageCols, imageSeqLenPerFrame);
        String fullPrompt = imagePrompt + userPrompt;

        // Tokenize
        Encoding encoding = tokenizer.encode(fullPrompt, false);
        int[] promptTokenIds = encoding.getIds();
        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        int promptTokenCount = promptTokenIds.length;

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
        return decodeFromEmbeddings(inputsEmbeds, promptTokenCount, maxTokens, temperature, doSample);
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
        String logitsOutputName = DecoderUtils.findLogitsOutputName(decoder);
        DecoderUtils.KVCacheNames kvNames = DecoderUtils.findKVCacheOutputNames(decoder);
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

        for (int i = 0; i < maxNewTokens; i++) {
            Map<String, INDArray> decoderInputMap = new HashMap<>();
            long currentSeqLen = currentEmbeddings.shape()[1];
            long totalSeqLen = currentSeqLen + pastSeqLen;

            for (String inputName : decoderInputNames) {
                if (inputName.equals("inputs_embeds")) {
                    decoderInputMap.put(inputName, currentEmbeddings);
                } else if (inputName.equals("attention_mask")) {
                    decoderInputMap.put(inputName, Nd4j.ones(DataType.LONG, batchSize, totalSeqLen));
                } else if (inputName.equals("_causal_mask")) {
                    decoderInputMap.put(inputName, DecoderUtils.buildCausalMask(currentSeqLen, totalSeqLen));
                } else if (inputName.equals("position_ids")) {
                    decoderInputMap.put(inputName, Nd4j.arange(pastSeqLen, pastSeqLen + currentSeqLen)
                            .reshape(1, currentSeqLen).castTo(DataType.LONG));
                } else if (useKvCache && inputName.startsWith("past_key_values.")) {
                    String presentName = inputName.replace("past_key_values", "present");
                    if (kvCache.containsKey(presentName)) {
                        decoderInputMap.put(inputName, kvCache.get(presentName));
                    } else {
                        decoderInputMap.put(inputName, DecoderUtils.createEmptyKvCache(
                                decoder, inputName, batchSize, hiddenSize));
                    }
                }
            }

            Map<String, INDArray> outputs = decoder.output(decoderInputMap,
                    allOutputNames.toArray(new String[0]));
            INDArray logits = outputs.get(allOutputNames.get(0));

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
            }

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

        if (kvCache != null) {
            for (INDArray v : kvCache.values()) {
                SameDiffMemoryUtils.safeClose(v);
            }
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
        inputs.put("pixel_values", normalized);

        Map<String, INDArray> outputs = visionEncoder.output(inputs, "image_embeds");
        INDArray result = outputs.get("image_embeds");

        // Ensure output is on target device
        ensureOnDevice(result, device);
        return result;
    }

    private INDArray normalizeVisionInputShape(INDArray image) {
        if (image == null) {
            throw new IllegalArgumentException("image must not be null");
        }
        if (image.rank() == 4 && visionEncoderExpectsFramedInput()) {
            if (image.size(0) == 1) {
                // [1, C, H, W] -> [1, 1, C, H, W]
                return image.reshape(1, 1, image.size(1), image.size(2), image.size(3));
            }
            // [frames, C, H, W] -> [1, frames, C, H, W]
            return image.reshape(1, image.size(0), image.size(1), image.size(2), image.size(3));
        }
        return image;
    }

    private boolean visionEncoderExpectsFramedInput() {
        SDVariable pixelValues = visionEncoder.getVariable("pixel_values");
        if (pixelValues == null) {
            return false;
        }
        long[] shape = pixelValues.getShape();
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
                .targetDevice(device)
                .build();
    }

    /**
     * Create a model-parallel VLM across multiple devices.
     *
     * @param modelDir the model directory
     * @param devices devices for [visionEncoder, embedTokens, decoder]
     * @return model-parallel VLM
     * @throws IOException if loading fails
     */
    public static ModelParallelVLM forMultipleDevices(File modelDir, DeviceDescriptor... devices)
            throws IOException {
        return ModelParallelVLM.fromDirectory(modelDir, devices);
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
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
            // SameDiff instances are GC'd
        }
    }
}
