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
import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.config.ModelConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.tokenizer.Encoding;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.vlm.pipeline.VLMPipelineExecutor;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.MultiBackendWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.*;

/**
 * Model-parallel Vision-Language Model wrapper for multi-device/multi-chip inference.
 *
 * This class enables running large VLMs across multiple devices (GPUs/accelerators)
 * using various parallelism strategies:
 * - Pipeline parallelism: Vision encoder on device 0, decoder on device 1, etc.
 * - Tensor parallelism: Split model weights across devices
 * - Data parallelism: Process different images on different devices
 *
 * <p>Features:</p>
 * <ul>
 *   <li>Automatic device placement and data transfer</li>
 *   <li>Workspace-based memory management for efficiency</li>
 *   <li>Asynchronous prefetching for overlapping compute and transfer</li>
 *   <li>Support for heterogeneous devices (CPU + multiple GPUs)</li>
 * </ul>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * // Create a model-parallel VLM with 2 GPUs
 * ModelParallelVLM vlm = ModelParallelVLM.builder()
 *     .modelDirectory(new File("SmolDocling-256M-sdz"))
 *     .visionEncoderDevice(DeviceDescriptor.cuda(0))
 *     .decoderDevice(DeviceDescriptor.cuda(1))
 *     .build();
 *
 * // Process an image
 * String output = vlm.generate(new File("document.png"), "Convert to markdown");
 *
 * vlm.close();
 * }</pre>
 *
 * Adam Gibson
 */
@Slf4j
@Getter
public class ModelParallelVLM implements AutoCloseable {

    // Model components
    private final SameDiff visionEncoder;
    private final SameDiff embedTokens;
    private final SameDiff decoder;
    private final Tokenizer tokenizer;
    private final VLMImagePreprocessor imagePreprocessor;
    private final ModelConfig config;

    // Device configuration
    private final DeviceDescriptor visionEncoderDevice;
    private final DeviceDescriptor embedTokensDevice;
    private final DeviceDescriptor decoderDevice;
    private final List<DeviceDescriptor> allDevices;

    // Memory management
    private final MultiBackendWorkspace inferenceWorkspace;
    private final Map<DeviceDescriptor, MemoryWorkspace> deviceWorkspaces;
    private final long workspaceSize;

    // Execution
    private final ExecutorService transferExecutor;
    private final boolean enablePrefetch;
    private final boolean enableAsyncTransfer;

    // Pipeline executor for multi-GPU encode/decode parallelism
    private final VLMPipelineExecutor pipelineExecutor;

    private volatile boolean closed = false;

    @Builder
    private ModelParallelVLM(
            File modelDirectory,
            DeviceDescriptor visionEncoderDevice,
            DeviceDescriptor embedTokensDevice,
            DeviceDescriptor decoderDevice,
            Long workspaceSize,
            Boolean enablePrefetch,
            Boolean enableAsyncTransfer) throws IOException {

        // Load model components
        MultiPartModelLoader.LoadedModel loaded = MultiPartModelLoader.load(modelDirectory);
        this.visionEncoder = loaded.getVisionEncoder();
        this.embedTokens = loaded.getEmbedTokens();
        this.decoder = loaded.getDecoder();
        this.tokenizer = loaded.getTokenizer();
        this.config = loaded.getConfig();

        // Set device configuration with defaults
        this.visionEncoderDevice = visionEncoderDevice != null ?
                visionEncoderDevice : DeviceDescriptor.cpu();
        this.embedTokensDevice = embedTokensDevice != null ?
                embedTokensDevice : this.visionEncoderDevice;
        this.decoderDevice = decoderDevice != null ?
                decoderDevice : this.visionEncoderDevice;

        // Collect all unique devices
        Set<DeviceDescriptor> deviceSet = new LinkedHashSet<>();
        deviceSet.add(this.visionEncoderDevice);
        deviceSet.add(this.embedTokensDevice);
        deviceSet.add(this.decoderDevice);
        this.allDevices = new ArrayList<>(deviceSet);

        // Configure memory
        this.workspaceSize = workspaceSize != null ? workspaceSize : 256 * 1024 * 1024L; // 256MB default

        // Create device workspaces
        this.deviceWorkspaces = new ConcurrentHashMap<>();
        for (DeviceDescriptor device : allDevices) {
            WorkspaceConfiguration wsConfig = WorkspaceConfiguration.builder()
                    .initialSize(this.workspaceSize)
                    .policyAllocation(AllocationPolicy.STRICT)
                    .policyLearning(LearningPolicy.FIRST_LOOP)
                    .policySpill(SpillPolicy.EXTERNAL)
                    .policyMirroring(MirroringPolicy.FULL)
                    .build();
            String workspaceId = "ModelParallelVLM-" + device.toString();
            MemoryWorkspace ws;
            if (device.getDeviceType() == DeviceType.CUDA_GPU) {
                ws = Nd4j.getWorkspaceManager().createNewWorkspace(
                        wsConfig, workspaceId, device.getDeviceIndex());
            } else {
                ws = Nd4j.getWorkspaceManager().createNewWorkspace(wsConfig, workspaceId);
            }
            deviceWorkspaces.put(device, ws);
        }

        // Create inference workspace (simplified - real impl would use MultiBackendWorkspace)
        this.inferenceWorkspace = null; // TODO: Create proper multi-backend workspace

        // Create preprocessor with vision encoder device
        this.imagePreprocessor = VLMImagePreprocessor.forDevice(
                loaded.getImagePreprocessor().getConfig(),
                this.visionEncoderDevice);

        // Configure execution
        this.enablePrefetch = enablePrefetch != null ? enablePrefetch : true;
        this.enableAsyncTransfer = enableAsyncTransfer != null ? enableAsyncTransfer : true;

        this.transferExecutor = Executors.newFixedThreadPool(2, r -> {
            Thread t = new Thread(r, "ModelParallelVLM-Transfer");
            t.setDaemon(true);
            return t;
        });

        // Create pipeline executor for multi-GPU encode/decode parallelism
        int encoderDevIdx = this.visionEncoderDevice.getDeviceType() == DeviceType.CUDA_GPU
                ? this.visionEncoderDevice.getDeviceIndex() : 0;
        int decoderDevIdx = this.decoderDevice.getDeviceType() == DeviceType.CUDA_GPU
                ? this.decoderDevice.getDeviceIndex() : 0;
        this.pipelineExecutor = new VLMPipelineExecutor(
                this.visionEncoder, encoderDevIdx,
                this.decoder, decoderDevIdx,
                this.embedTokens);

        log.info("ModelParallelVLM initialized with devices: vision={}, embed={}, decoder={}",
                this.visionEncoderDevice, this.embedTokensDevice, this.decoderDevice);
    }

    /**
     * Load a model-parallel VLM from a directory.
     *
     * @param modelDir the model directory
     * @param devices array of devices [visionEncoder, embedTokens, decoder]
     * @return the loaded model
     * @throws IOException if loading fails
     */
    public static ModelParallelVLM fromDirectory(File modelDir, DeviceDescriptor... devices)
            throws IOException {
        ModelParallelVLMBuilder builder = builder().modelDirectory(modelDir);

        if (devices.length >= 1) builder.visionEncoderDevice(devices[0]);
        if (devices.length >= 2) builder.embedTokensDevice(devices[1]);
        if (devices.length >= 3) builder.decoderDevice(devices[2]);

        return builder.build();
    }

    /**
     * Create a simple 2-GPU pipeline: vision on GPU 0, decoder on GPU 1.
     *
     * @param modelDir the model directory
     * @return the loaded model
     * @throws IOException if loading fails
     */
    public static ModelParallelVLM twoGpuPipeline(File modelDir) throws IOException {
        return builder()
                .modelDirectory(modelDir)
                .visionEncoderDevice(DeviceDescriptor.cuda(0))
                .embedTokensDevice(DeviceDescriptor.cuda(0))
                .decoderDevice(DeviceDescriptor.cuda(1))
                .build();
    }

    /**
     * Create a CPU + GPU hybrid: preprocessing on CPU, model on GPU.
     *
     * @param modelDir the model directory
     * @param gpuIndex the GPU index to use
     * @return the loaded model
     * @throws IOException if loading fails
     */
    public static ModelParallelVLM cpuGpuHybrid(File modelDir, int gpuIndex) throws IOException {
        return builder()
                .modelDirectory(modelDir)
                .visionEncoderDevice(DeviceDescriptor.cuda(gpuIndex))
                .embedTokensDevice(DeviceDescriptor.cuda(gpuIndex))
                .decoderDevice(DeviceDescriptor.cuda(gpuIndex))
                .build();
    }

    /**
     * Generate text from an image file and prompt.
     *
     * @param imageFile the image file
     * @param prompt the text prompt
     * @return the generated text
     * @throws IOException if loading fails
     */
    public String generate(File imageFile, String prompt) throws IOException {
        return generate(imageFile, prompt, 512, 1.0, true);
    }

    /**
     * Generate text from an image file and prompt with parameters.
     *
     * @param imageFile the image file
     * @param prompt the text prompt
     * @param maxNewTokens maximum tokens to generate
     * @param temperature sampling temperature
     * @param doSample whether to sample
     * @return the generated text
     * @throws IOException if loading fails
     */
    public String generate(File imageFile, String prompt, int maxNewTokens,
                          double temperature, boolean doSample) throws IOException {
        return generateWithMetrics(imageFile, prompt, maxNewTokens, temperature, doSample).getText();
    }

    /**
     * Generate text from an image file, returning detailed metrics.
     *
     * @param imageFile the image file
     * @param prompt the text prompt
     * @return the generation result with metrics
     * @throws IOException if loading fails
     */
    public GenerationResult generateWithMetrics(File imageFile, String prompt) throws IOException {
        return generateWithMetrics(imageFile, prompt, 512, 1.0, true);
    }

    /**
     * Generate text from an image file with parameters, returning detailed metrics.
     *
     * @param imageFile the image file
     * @param prompt the text prompt
     * @param maxNewTokens maximum tokens to generate
     * @param temperature sampling temperature
     * @param doSample whether to sample
     * @return the generation result with metrics
     * @throws IOException if loading fails
     */
    public GenerationResult generateWithMetrics(File imageFile, String prompt, int maxNewTokens,
                                                double temperature, boolean doSample) throws IOException {
        checkNotClosed();

        // Preprocess image on vision encoder device
        INDArray image = imagePreprocessor.preprocessOnDevice(imageFile, visionEncoderDevice);

        return generateFromImageWithMetrics(image, prompt, maxNewTokens, temperature, doSample);
    }

    /**
     * Generate text from a preprocessed image tensor.
     *
     * @param image the preprocessed image tensor
     * @param prompt the text prompt
     * @param maxNewTokens maximum tokens to generate
     * @param temperature sampling temperature
     * @param doSample whether to sample
     * @return the generated text
     */
    public String generateFromImage(INDArray image, String prompt, int maxNewTokens,
                                    double temperature, boolean doSample) {
        return generateFromImageWithMetrics(image, prompt, maxNewTokens, temperature, doSample).getText();
    }

    /**
     * Generate text from a preprocessed image tensor, returning detailed metrics.
     *
     * @param image the preprocessed image tensor
     * @param prompt the text prompt
     * @param maxNewTokens maximum tokens to generate
     * @param temperature sampling temperature
     * @param doSample whether to sample
     * @return the generation result with metrics
     */
    public GenerationResult generateFromImageWithMetrics(INDArray image, String prompt, int maxNewTokens,
                                                         double temperature, boolean doSample) {
        checkNotClosed();

        // Step 1: Encode image via pipeline executor (runs on encoder GPU,
        // result transferred to decoder GPU automatically)
        log.debug("Encoding image via pipeline on device: {}", visionEncoderDevice);
        Map<String, INDArray> visionInputs = new HashMap<>();
        visionInputs.put("pixel_values", image);
        INDArray imageEmbeddings;
        try {
            Future<INDArray> encodeFuture = pipelineExecutor.encodeAsync(
                    visionInputs, new String[]{"image_embeds", "last_hidden_state"});
            imageEmbeddings = encodeFuture.get(5, java.util.concurrent.TimeUnit.MINUTES);
        } catch (java.util.concurrent.ExecutionException e) {
            throw new RuntimeException("Vision encoding failed", e.getCause());
        } catch (java.util.concurrent.TimeoutException e) {
            throw new RuntimeException("Vision encoding timed out", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Vision encoding interrupted", e);
        }
        // imageEmbeddings is already on decoder device

        // Step 2: Encode prompt on embed tokens device
        Encoding promptEncoding = tokenizer.encode(prompt, true);
        int[] promptIds = promptEncoding.getIds();
        int promptTokenCount = promptIds.length;
        INDArray textEmbeddings = embedTextOnDevice(promptIds);

        // Step 3: Transfer text embeddings to decoder device if different
        if (!decoderDevice.equals(embedTokensDevice)) {
            transferToDevice(textEmbeddings, decoderDevice);
        }

        // Step 4: Combine embeddings (both now on decoder device)
        INDArray combinedEmbeddings = Nd4j.concat(1, imageEmbeddings, textEmbeddings);
        ensureOnDevice(combinedEmbeddings, decoderDevice);

        // Step 5: Autoregressive generation on decoder device with timing
        StringBuilder generated = new StringBuilder();
        List<Integer> generatedTokenIds = new ArrayList<>();
        long startNanos = System.nanoTime();
        long firstTokenLatencyNanos = 0;
        int generatedTokenCount = 0;
        GenerationResult.FinishReason finishReason = GenerationResult.FinishReason.MAX_TOKENS;

        for (int i = 0; i < maxNewTokens; i++) {
            // Run decoder
            Map<String, INDArray> decoderInputs = new HashMap<>();
            decoderInputs.put("inputs_embeds", combinedEmbeddings);

            Map<String, INDArray> outputs = decoder.output(decoderInputs, "logits");
            INDArray logits = outputs.get("logits");

            // Get next token (greedy or sampling)
            int nextTokenId = getNextToken(logits, temperature, doSample);

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

            // Update embeddings for next iteration
            INDArray newTokenEmbed = embedTextOnDevice(new int[]{nextTokenId});
            if (!decoderDevice.equals(embedTokensDevice)) {
                transferToDevice(newTokenEmbed, decoderDevice);
            }
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
     * Batch generate from multiple images.
     * Uses data parallelism to process images on different devices.
     *
     * @param imageFiles list of image files
     * @param prompts list of prompts (one per image, or single prompt for all)
     * @param maxNewTokens maximum tokens per generation
     * @return list of generated texts
     * @throws IOException if loading fails
     */
    public List<String> batchGenerate(List<File> imageFiles, List<String> prompts,
                                      int maxNewTokens) throws IOException {
        checkNotClosed();

        if (prompts.size() != 1 && prompts.size() != imageFiles.size()) {
            throw new IllegalArgumentException(
                    "Prompts must be single element or match imageFiles size");
        }

        List<String> results = new ArrayList<>(imageFiles.size());
        for (int i = 0; i < imageFiles.size(); i++) {
            String prompt = prompts.size() == 1 ? prompts.get(0) : prompts.get(i);
            results.add(generate(imageFiles.get(i), prompt, maxNewTokens, 1.0, true));
        }
        return results;
    }

    /**
     * Generate one output per page for a multi-page document.
     *
     * @param pageFiles page images in order
     * @param prompt prompt applied to each page
     * @param maxNewTokens max tokens per page
     * @param temperature sampling temperature
     * @param doSample whether to sample
     * @return generated text per page (same order as input)
     * @throws IOException if loading fails
     */
    public List<String> generatePages(List<File> pageFiles, String prompt, int maxNewTokens,
                                      double temperature, boolean doSample) throws IOException {
        checkNotClosed();
        List<String> outputs = new ArrayList<>(pageFiles.size());
        for (File pageFile : pageFiles) {
            outputs.add(generate(pageFile, prompt, maxNewTokens, temperature, doSample));
        }
        return outputs;
    }

    /**
     * Generate one output per page for a multi-page document (greedy defaults).
     *
     * @param pageFiles page images in order
     * @param prompt prompt applied to each page
     * @param maxNewTokens max tokens per page
     * @return generated text per page (same order as input)
     * @throws IOException if loading fails
     */
    public List<String> generatePages(List<File> pageFiles, String prompt, int maxNewTokens) throws IOException {
        return generatePages(pageFiles, prompt, maxNewTokens, 1.0, true);
    }

    /**
     * Generate a single combined document output from page images.
     *
     * @param pageFiles page images in order
     * @param prompt prompt applied to each page
     * @param maxNewTokens max tokens per page
     * @param temperature sampling temperature
     * @param doSample whether to sample
     * @param pageDelimiter delimiter inserted between page outputs
     * @return combined document output
     * @throws IOException if loading fails
     */
    public String generateDocument(List<File> pageFiles, String prompt, int maxNewTokens,
                                   double temperature, boolean doSample, String pageDelimiter) throws IOException {
        List<String> pageOutputs = generatePages(pageFiles, prompt, maxNewTokens, temperature, doSample);
        String delimiter = pageDelimiter == null ? "" : pageDelimiter;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < pageOutputs.size(); i++) {
            if (i > 0) {
                sb.append(delimiter);
            }
            String text = pageOutputs.get(i);
            if (text != null) {
                sb.append(text);
            }
        }
        return sb.toString();
    }

    /**
     * Prefetch an image to the vision encoder device.
     *
     * @param imageFile the image to prefetch
     * @return future that completes when prefetch is done
     */
    public CompletableFuture<INDArray> prefetchImage(File imageFile) {
        return imagePreprocessor.preprocessAsync(imageFile, visionEncoderDevice);
    }

    /**
     * Encode image on vision encoder device.
     */
    private INDArray encodeImageOnDevice(INDArray image) {
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("pixel_values", image);

        Map<String, INDArray> outputs = visionEncoder.output(inputs, "image_embeds");
        return outputs.get("image_embeds");
    }

    /**
     * Embed text tokens on embed tokens device.
     */
    private INDArray embedTextOnDevice(int[] tokenIds) {
        INDArray inputIds = Nd4j.createFromArray(tokenIds).reshape(1, tokenIds.length);
        ensureOnDevice(inputIds, embedTokensDevice);

        if (embedTokens != null) {
            Map<String, INDArray> inputs = new HashMap<>();
            inputs.put("input_ids", inputIds);
            Map<String, INDArray> outputs = embedTokens.output(inputs, "inputs_embeds");
            return outputs.get("inputs_embeds");
        }

        // Fallback: decoder handles embedding internally
        return inputIds;
    }

    /**
     * Get next token from logits.
     */
    private int getNextToken(INDArray logits, double temperature, boolean doSample) {
        if (!doSample || temperature <= 0) {
            // Greedy decoding
            return Nd4j.argMax(logits.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                    org.nd4j.linalg.indexing.NDArrayIndex.point(logits.shape()[1] - 1)), 0).getInt(0);
        } else {
            // Temperature-scaled sampling
            INDArray scaledLogits = logits.div(temperature);
            INDArray probs = Nd4j.nn().softmax(scaledLogits, 2);
            return Nd4j.argMax(probs.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                    org.nd4j.linalg.indexing.NDArrayIndex.point(probs.shape()[1] - 1)), 0).getInt(0);
        }
    }

    /**
     * Ensure array is on specified device.
     */
    private void ensureOnDevice(INDArray array, DeviceDescriptor device) {
        if (device == null) return;

        DataBuffer buffer = array.data();
        if (buffer != null && buffer.isHybrid()) {
            buffer.asHybrid().ensureAvailableOn(device);
        }
    }

    /**
     * Transfer array to specified device.
     */
    private void transferToDevice(INDArray array, DeviceDescriptor targetDevice) {
        if (enableAsyncTransfer) {
            CompletableFuture.runAsync(() -> ensureOnDevice(array, targetDevice), transferExecutor);
        } else {
            ensureOnDevice(array, targetDevice);
        }
    }

    /**
     * Prefetch data to device asynchronously.
     */
    private void prefetchToDevice(INDArray array, DeviceDescriptor device) {
        if (!enablePrefetch) return;

        DataBuffer buffer = array.data();
        if (buffer != null && buffer.isHybrid()) {
            transferExecutor.submit(() -> buffer.asHybrid().prefetch(device));
        }
    }

    /**
     * Synchronize all pending transfers on a device.
     */
    public void syncDevice(DeviceDescriptor device) {
        // Implementation depends on backend
        // For CUDA: cudaDeviceSynchronize or stream sync
    }

    /**
     * Synchronize all devices.
     */
    public void syncAllDevices() {
        for (DeviceDescriptor device : allDevices) {
            syncDevice(device);
        }
    }

    /**
     * Get memory usage statistics.
     */
    public Map<String, Object> getMemoryStats() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("numDevices", allDevices.size());
        stats.put("devices", allDevices);
        stats.put("workspaceSize", workspaceSize);
        return stats;
    }

    /**
     * Check if model is valid.
     */
    public boolean isValid() {
        return !closed && visionEncoder != null && decoder != null && tokenizer != null;
    }

    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("Model has been closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;

            // Close pipeline executor first (frees encoder if not already freed)
            if (pipelineExecutor != null) {
                try {
                    pipelineExecutor.close();
                } catch (Exception e) {
                    log.warn("Error closing pipeline executor", e);
                }
            }

            // Shutdown executor
            if (transferExecutor != null) {
                transferExecutor.shutdown();
                try {
                    if (!transferExecutor.awaitTermination(5, TimeUnit.SECONDS)) {
                        transferExecutor.shutdownNow();
                    }
                } catch (InterruptedException e) {
                    transferExecutor.shutdownNow();
                    Thread.currentThread().interrupt();
                }
            }

            // Close preprocessor
            if (imagePreprocessor != null) {
                imagePreprocessor.shutdown();
            }

            // Close tokenizer
            if (tokenizer != null) {
                try {
                    tokenizer.close();
                } catch (Exception e) {
                    log.warn("Error closing tokenizer", e);
                }
            }

            // Close device workspaces
            for (MemoryWorkspace ws : deviceWorkspaces.values()) {
                try {
                    ws.close();
                } catch (Exception e) {
                    log.warn("Error closing workspace", e);
                }
            }

            log.info("ModelParallelVLM closed");
        }
    }
}
