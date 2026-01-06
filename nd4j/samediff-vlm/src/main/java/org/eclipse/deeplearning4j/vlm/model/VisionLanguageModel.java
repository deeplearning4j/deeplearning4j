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
import org.eclipse.deeplearning4j.llm.tokenizer.Encoding;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.MultiBackendWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
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
        checkNotClosed();

        // Encode image
        INDArray imageEmbeddings = encodeImage(image);

        // Encode prompt
        Encoding promptEncoding = tokenizer.encode(prompt, true);
        INDArray textEmbeddings = embedText(promptEncoding.getIds());

        // Combine embeddings (image before text for most VLMs)
        INDArray combinedEmbeddings = combineEmbeddings(imageEmbeddings, textEmbeddings);

        // Autoregressive generation
        StringBuilder generated = new StringBuilder();
        int[] currentIds = promptEncoding.getIds();

        for (int i = 0; i < maxNewTokens; i++) {
            // Run decoder
            Map<String, INDArray> decoderInputs = new HashMap<>();
            decoderInputs.put("inputs_embeds", combinedEmbeddings);

            // Get logits for next token
            Map<String, INDArray> outputs = decoder.output(decoderInputs, "logits");
            INDArray logits = outputs.get("logits");

            // Get next token (greedy or sampling)
            int nextTokenId;
            if (!doSample || temperature <= 0) {
                nextTokenId = Nd4j.argMax(logits.get(NDArrayIndex.point(0),
                        NDArrayIndex.point(logits.shape()[1] - 1)), 0).getInt(0);
            } else {
                // Apply temperature and sample
                INDArray scaledLogits = logits.div(temperature);
                INDArray probs = Nd4j.nn().softmax(scaledLogits, 2);
                // Multinomial sampling would go here
                nextTokenId = Nd4j.argMax(probs.get(NDArrayIndex.point(0),
                        NDArrayIndex.point(probs.shape()[1] - 1)), 0).getInt(0);
            }

            // Check for EOS
            if (nextTokenId == tokenizer.getEosTokenId()) {
                break;
            }

            // Decode token
            String tokenText = tokenizer.decode(new int[]{nextTokenId}, true);
            generated.append(tokenText);

            // Update embeddings for next iteration
            INDArray newTokenEmbed = embedText(new int[]{nextTokenId});
            combinedEmbeddings = Nd4j.concat(1, combinedEmbeddings, newTokenEmbed);
        }

        return generated.toString();
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
        checkNotClosed();

        // Encode image on device
        INDArray imageEmbeddings = encodeImageOnDevice(image, device);

        // Encode prompt on device
        Encoding promptEncoding = tokenizer.encode(prompt, true);
        INDArray textEmbeddings = embedTextOnDevice(promptEncoding.getIds(), device);

        // Combine embeddings
        INDArray combinedEmbeddings = combineEmbeddings(imageEmbeddings, textEmbeddings);
        ensureOnDevice(combinedEmbeddings, device);

        // Autoregressive generation
        StringBuilder generated = new StringBuilder();

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

            if (nextTokenId == tokenizer.getEosTokenId()) {
                break;
            }

            String tokenText = tokenizer.decode(new int[]{nextTokenId}, true);
            generated.append(tokenText);

            INDArray newTokenEmbed = embedTextOnDevice(new int[]{nextTokenId}, device);
            combinedEmbeddings = Nd4j.concat(1, combinedEmbeddings, newTokenEmbed);
        }

        return generated.toString();
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
                .imagePreprocessor(VLMImagePreprocessor.forDevice(loaded.getPreprocessorConfig(), device))
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

// Import for NDArrayIndex - would need proper import
class NDArrayIndex {
    public static org.nd4j.linalg.indexing.INDArrayIndex point(long i) {
        return org.nd4j.linalg.indexing.NDArrayIndex.point(i);
    }
}
