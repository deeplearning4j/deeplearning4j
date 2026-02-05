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
import org.eclipse.deeplearning4j.llm.generation.SamplerUtils;
import org.eclipse.deeplearning4j.llm.tokenizer.Encoding;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.eclipse.deeplearning4j.llm.generation.DecoderUtils;
import org.nd4j.autodiff.samediff.SameDiff;
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
 * @author Eclipse Deeplearning4j Contributors
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
        inputs.put("pixel_values", image);

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
                        if (old != null) old.close();
                    }
                }
                for (String presentName : kvNames.valueNames) {
                    INDArray pv = outputs.get(presentName);
                    if (pv != null) {
                        INDArray old = kvCache.put(presentName, pv);
                        if (old != null) old.close();
                    }
                }
            }

            // Get next token (greedy or sampling)
            int nextTokenId;
            if (!doSample || temperature <= 0) {
                nextTokenId = Nd4j.argMax(logits.get(NDArrayIndex.point(0),
                        NDArrayIndex.point(logits.shape()[1] - 1)), 0).getInt(0);
            } else {
                INDArray scaledLogits = logits.div(temperature);
                INDArray probs = Nd4j.nn().softmax(scaledLogits, 2);
                nextTokenId = Nd4j.argMax(probs.get(NDArrayIndex.point(0),
                        NDArrayIndex.point(probs.shape()[1] - 1)), 0).getInt(0);
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
                    prevEmbeddings.close();
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
                if (v != null) v.close();
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
        INDArray batchedImageEmbeddings = Nd4j.vstack(imageEmbeddingsList.toArray(new INDArray[0]));

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
                    decoderInputMap.put(inputName, Nd4j.ones(DataType.LONG, batchSize, totalSeqLen));
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
            Map<String, INDArray> outputs = decoder.output(decoderInputMap,
                    allOutputNames.toArray(new String[0]));
            INDArray logits = outputs.get(allOutputNames.get(0));

            // Update KV cache
            if (useKvCache) {
                for (String presentName : kvNames.keyNames) {
                    INDArray pv = outputs.get(presentName);
                    if (pv != null) {
                        INDArray old = kvCache.put(presentName, pv);
                        if (old != null) old.close();
                    }
                }
                for (String presentName : kvNames.valueNames) {
                    INDArray pv = outputs.get(presentName);
                    if (pv != null) {
                        INDArray old = kvCache.put(presentName, pv);
                        if (old != null) old.close();
                    }
                }
            }

            // Get logits for last position: [batchSize, seqLen, vocab] -> [batchSize, vocab]
            INDArray lastLogits;
            if (logits.rank() == 3) {
                lastLogits = logits.get(NDArrayIndex.all(),
                        NDArrayIndex.point(logits.size(1) - 1), NDArrayIndex.all());
            } else {
                lastLogits = logits;
            }

            // Sample next tokens for all sequences
            int[] nextTokenIds;
            if (!doSample || temperature <= 0) {
                nextTokenIds = SamplerUtils.argmaxBatch(lastLogits);
            } else {
                INDArray scaledLogits = lastLogits.div(temperature);
                INDArray probs = SamplerUtils.softmax(scaledLogits);
                nextTokenIds = SamplerUtils.multinomialSampleBatch(probs, new java.util.Random());
            }

            // Record tokens
            long stepNanos = System.nanoTime() - startNanos;
            state.recordTokens(nextTokenIds, stepNanos);

            // Check if all sequences are done
            if (state.allFinished()) {
                break;
            }

            // Prepare embeddings for next step
            if (useKvCache) {
                pastSeqLen += currentSeqLen;

                // Embed next tokens for all sequences: [batchSize, 1, hidden]
                INDArray prevEmbeddings = currentEmbeddings;
                INDArray[] nextEmbeds = new INDArray[batchSize];
                for (int i = 0; i < batchSize; i++) {
                    if (!state.isFinished(i)) {
                        nextEmbeds[i] = embedText(new int[]{nextTokenIds[i]});
                    } else {
                        // Finished sequences: embed EOS token (padding)
                        nextEmbeds[i] = embedText(new int[]{tokenizer.getEosTokenId()});
                    }
                }
                currentEmbeddings = Nd4j.vstack(nextEmbeds);
                if (prevEmbeddings != batchedImageEmbeddings) {
                    prevEmbeddings.close();
                }
                decoder.clearPlaceholders(false);
            } else {
                // No KV cache: grow embeddings (inefficient but correct)
                INDArray[] newTokenEmbeds = new INDArray[batchSize];
                for (int i = 0; i < batchSize; i++) {
                    newTokenEmbeds[i] = embedText(new int[]{nextTokenIds[i]});
                }
                INDArray batchedNewTokenEmbeds = Nd4j.vstack(newTokenEmbeds);
                currentEmbeddings = Nd4j.concat(1, currentEmbeddings, batchedNewTokenEmbeds);
            }
        }

        // Mark any remaining sequences as max tokens
        for (int i = 0; i < batchSize; i++) {
            state.markMaxTokens(i);
        }

        // Clean up KV cache
        if (kvCache != null) {
            for (INDArray v : kvCache.values()) {
                if (v != null) v.close();
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
            INDArray batched = Nd4j.vstack(images.toArray(new INDArray[0]));
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
            return Nd4j.vstack(embeddings.toArray(new INDArray[0]));
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

        // Ensure image is on target device
        ensureOnDevice(image, device);

        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("pixel_values", image);

        Map<String, INDArray> outputs = visionEncoder.output(inputs, "image_embeds");
        INDArray result = outputs.get("image_embeds");

        // Ensure output is on target device
        ensureOnDevice(result, device);
        return result;
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
                nextTokenId = Nd4j.argMax(logits.get(NDArrayIndex.point(0),
                        NDArrayIndex.point(logits.shape()[1] - 1)), 0).getInt(0);
            } else {
                INDArray scaledLogits = logits.div(temperature);
                INDArray probs = Nd4j.nn().softmax(scaledLogits, 2);
                nextTokenId = Nd4j.argMax(probs.get(NDArrayIndex.point(0),
                        NDArrayIndex.point(probs.shape()[1] - 1)), 0).getInt(0);
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

