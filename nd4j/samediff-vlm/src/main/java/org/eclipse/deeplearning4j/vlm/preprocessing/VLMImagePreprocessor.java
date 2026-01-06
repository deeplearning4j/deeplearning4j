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

package org.eclipse.deeplearning4j.vlm.preprocessing;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.MultiBackendWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Image preprocessor for Vision-Language Models.
 *
 * Handles the preprocessing pipeline for VLM image inputs:
 * 1. Load image from file/stream
 * 2. Resize to model's expected dimensions
 * 3. Rescale pixel values (typically 0-255 to 0-1)
 * 4. Normalize with mean and std
 * 5. Convert to model's expected format (NCHW)
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * VLMImagePreprocessor preprocessor = VLMImagePreprocessor.fromConfig(
 *     new File("preprocessor_config.json")
 * );
 *
 * INDArray image = preprocessor.preprocess(new File("image.jpg"));
 * // image shape: [1, 3, height, width]
 * }</pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
@Getter
public class VLMImagePreprocessor {

    private final PreprocessorConfig config;
    private final int targetHeight;
    private final int targetWidth;
    private final int numChannels;

    // Device-aware fields for multi-chip support
    @Setter
    private DeviceDescriptor targetDevice;

    @Setter
    private MultiBackendWorkspace workspace;

    private final ExecutorService prefetchExecutor;

    @Builder
    private VLMImagePreprocessor(PreprocessorConfig config) {
        this(config, null, null);
    }

    /**
     * Create a preprocessor with device placement.
     *
     * @param config the preprocessor config
     * @param targetDevice the target device for output tensors
     * @param workspace optional workspace for memory management
     */
    public VLMImagePreprocessor(PreprocessorConfig config, DeviceDescriptor targetDevice, MultiBackendWorkspace workspace) {
        this.config = config != null ? config : PreprocessorConfig.forClip();
        this.targetHeight = this.config.getTargetHeight();
        this.targetWidth = this.config.getTargetWidth();
        this.numChannels = this.config.getNumChannels() != null ? this.config.getNumChannels() : 3;
        this.targetDevice = targetDevice;
        this.workspace = workspace;
        this.prefetchExecutor = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "VLMImagePreprocessor-Prefetch");
            t.setDaemon(true);
            return t;
        });
    }

    /**
     * Create a preprocessor from a config file.
     *
     * @param configFile the preprocessor_config.json file
     * @return the preprocessor
     * @throws IOException if loading fails
     */
    public static VLMImagePreprocessor fromConfig(File configFile) throws IOException {
        PreprocessorConfig config = PreprocessorConfig.fromFile(configFile);
        return new VLMImagePreprocessor(config);
    }

    /**
     * Create a preprocessor from a config object.
     *
     * @param config the preprocessor config
     * @return the preprocessor
     */
    public static VLMImagePreprocessor fromConfig(PreprocessorConfig config) {
        return new VLMImagePreprocessor(config);
    }

    /**
     * Create a default preprocessor (CLIP-style).
     *
     * @return default preprocessor
     */
    public static VLMImagePreprocessor defaultPreprocessor() {
        return new VLMImagePreprocessor(PreprocessorConfig.forClip());
    }

    /**
     * Create a preprocessor for SmolDocling.
     *
     * @return SmolDocling preprocessor
     */
    public static VLMImagePreprocessor forSmolDocling() {
        return new VLMImagePreprocessor(PreprocessorConfig.forSmolDocling());
    }

    /**
     * Create a preprocessor for LLaVA models.
     *
     * @return LLaVA preprocessor
     */
    public static VLMImagePreprocessor forLlava() {
        PreprocessorConfig config = PreprocessorConfig.forClip();
        config.setSize(new PreprocessorConfig.ImageSize(336, 336));
        return new VLMImagePreprocessor(config);
    }

    /**
     * Preprocess an image from a file.
     *
     * @param imageFile the image file
     * @return preprocessed tensor [1, C, H, W]
     * @throws IOException if loading fails
     */
    public INDArray preprocess(File imageFile) throws IOException {
        BufferedImage image = ImageIO.read(imageFile);
        if (image == null) {
            throw new IOException("Failed to load image: " + imageFile.getAbsolutePath());
        }
        return preprocess(image);
    }

    /**
     * Preprocess an image from an input stream.
     *
     * @param inputStream the image stream
     * @return preprocessed tensor [1, C, H, W]
     * @throws IOException if loading fails
     */
    public INDArray preprocess(InputStream inputStream) throws IOException {
        BufferedImage image = ImageIO.read(inputStream);
        if (image == null) {
            throw new IOException("Failed to load image from stream");
        }
        return preprocess(image);
    }

    /**
     * Preprocess a BufferedImage.
     *
     * @param image the BufferedImage
     * @return preprocessed tensor [1, C, H, W]
     */
    public INDArray preprocess(BufferedImage image) {
        // Resize to target dimensions
        BufferedImage resized = resize(image, targetWidth, targetHeight);

        // Convert to tensor [1, C, H, W]
        INDArray tensor = imageToTensor(resized);

        return preprocessLoaded(tensor);
    }

    /**
     * Preprocess an already-loaded image tensor.
     *
     * @param image the image tensor (any shape, will be reshaped)
     * @return preprocessed tensor [1, C, H, W]
     */
    public INDArray preprocess(INDArray image) {
        // Ensure correct shape
        INDArray formatted = ensureNCHW(image);
        return preprocessLoaded(formatted);
    }

    /**
     * Preprocess multiple images.
     *
     * @param imageFiles the image files
     * @return list of preprocessed tensors
     * @throws IOException if loading fails
     */
    public List<INDArray> preprocessBatch(List<File> imageFiles) throws IOException {
        List<INDArray> results = new ArrayList<>(imageFiles.size());
        for (File file : imageFiles) {
            results.add(preprocess(file));
        }
        return results;
    }

    /**
     * Preprocess multiple images and stack into batch.
     *
     * @param imageFiles the image files
     * @return batched tensor [N, C, H, W]
     * @throws IOException if loading fails
     */
    public INDArray preprocessBatched(List<File> imageFiles) throws IOException {
        List<INDArray> images = preprocessBatch(imageFiles);
        return Nd4j.vstack(images);
    }

    private BufferedImage resize(BufferedImage original, int targetWidth, int targetHeight) {
        BufferedImage resized = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = resized.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2d.drawImage(original, 0, 0, targetWidth, targetHeight, null);
        g2d.dispose();
        return resized;
    }

    private INDArray imageToTensor(BufferedImage image) {
        int height = image.getHeight();
        int width = image.getWidth();

        // Create tensor [1, C, H, W]
        float[] data = new float[numChannels * height * width];
        int idx = 0;

        for (int c = 0; c < numChannels; c++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int rgb = image.getRGB(x, y);
                    float value;
                    switch (c) {
                        case 0:
                            value = ((rgb >> 16) & 0xFF); // R
                            break;
                        case 1:
                            value = ((rgb >> 8) & 0xFF);  // G
                            break;
                        default:
                            value = (rgb & 0xFF);         // B
                            break;
                    }
                    data[idx++] = value;
                }
            }
        }

        return Nd4j.create(data, new int[]{1, numChannels, height, width});
    }

    private INDArray preprocessLoaded(INDArray image) {
        INDArray result = image;

        // Rescale (0-255 to 0-1 typically)
        if (config.isDoRescale()) {
            result = result.mul(config.getRescaleFactor());
        }

        // Normalize with mean and std
        if (config.isDoNormalize()) {
            result = normalize(result, config.getMean(), config.getStd());
        }

        return result;
    }

    private INDArray normalize(INDArray image, double[] mean, double[] std) {
        // image shape: [1, C, H, W]
        // Normalize each channel: (x - mean) / std

        long[] shape = image.shape();
        int numChannels = (int) shape[1];

        for (int c = 0; c < numChannels && c < mean.length && c < std.length; c++) {
            INDArray channel = image.get(
                    NDArrayIndex.all(),
                    NDArrayIndex.point(c),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
            );
            channel.subi(mean[c]).divi(std[c]);
        }

        return image;
    }

    private INDArray ensureNCHW(INDArray image) {
        long[] shape = image.shape();

        if (shape.length == 3) {
            // [C, H, W] -> [1, C, H, W]
            if (shape[0] == 3 || shape[0] == 1) {
                return image.reshape(1, shape[0], shape[1], shape[2]);
            }
            // [H, W, C] -> [1, C, H, W]
            if (shape[2] == 3 || shape[2] == 1) {
                return image.permute(2, 0, 1).reshape(1, shape[2], shape[0], shape[1]);
            }
        } else if (shape.length == 4) {
            // [N, H, W, C] -> [N, C, H, W]
            if (shape[3] == 3 || shape[3] == 1) {
                return image.permute(0, 3, 1, 2);
            }
            // Already [N, C, H, W]
            return image;
        }

        // Assume it's already correct
        return image;
    }

    /**
     * Extract patches from an image for Vision Transformer.
     *
     * @param image the preprocessed image [1, C, H, W]
     * @param patchSize the patch size (e.g., 16)
     * @return patches [1, num_patches, patch_dim]
     */
    public INDArray extractPatches(INDArray image, int patchSize) {
        long[] shape = image.shape();
        int channels = (int) shape[1];
        int height = (int) shape[2];
        int width = (int) shape[3];

        int numPatchesH = height / patchSize;
        int numPatchesW = width / patchSize;
        int numPatches = numPatchesH * numPatchesW;
        int patchDim = channels * patchSize * patchSize;

        INDArray patches = Nd4j.zeros(1, numPatches, patchDim);

        int patchIdx = 0;
        for (int ph = 0; ph < numPatchesH; ph++) {
            for (int pw = 0; pw < numPatchesW; pw++) {
                int startH = ph * patchSize;
                int startW = pw * patchSize;

                INDArray patch = image.get(
                        NDArrayIndex.point(0),
                        NDArrayIndex.all(),
                        NDArrayIndex.interval(startH, startH + patchSize),
                        NDArrayIndex.interval(startW, startW + patchSize)
                );

                patches.put(new org.nd4j.linalg.indexing.INDArrayIndex[]{
                        NDArrayIndex.point(0),
                        NDArrayIndex.point(patchIdx),
                        NDArrayIndex.all()
                }, patch.reshape(patchDim));

                patchIdx++;
            }
        }

        return patches;
    }

    // =========================================================================
    // Device-Aware Methods for Multi-Chip Backend Support
    // =========================================================================

    /**
     * Preprocess an image and place the result on a specific device.
     *
     * @param imageFile the image file
     * @param device the target device
     * @return preprocessed tensor on the specified device
     * @throws IOException if loading fails
     */
    public INDArray preprocessOnDevice(File imageFile, DeviceDescriptor device) throws IOException {
        INDArray result = preprocess(imageFile);
        ensureOnDevice(result, device);
        return result;
    }

    /**
     * Preprocess an image using workspace memory on a specific device.
     *
     * @param imageFile the image file
     * @param workspace the workspace for memory allocation
     * @param device the target device
     * @return preprocessed tensor in the workspace on the specified device
     * @throws IOException if loading fails
     */
    public INDArray preprocessInWorkspace(File imageFile, MultiBackendWorkspace workspace,
                                          DeviceDescriptor device) throws IOException {
        try (MemoryWorkspace ws = workspace.notifyScopeEntered()) {
            INDArray result = preprocess(imageFile);
            ensureOnDevice(result, device);
            return result;
        }
    }

    /**
     * Preprocess a BufferedImage and place on a specific device.
     *
     * @param image the BufferedImage
     * @param device the target device
     * @return preprocessed tensor on the specified device
     */
    public INDArray preprocessOnDevice(BufferedImage image, DeviceDescriptor device) {
        INDArray result = preprocess(image);
        ensureOnDevice(result, device);
        return result;
    }

    /**
     * Preprocess multiple images in parallel and place on specific devices.
     * Each image can go to a different device for distributed processing.
     *
     * @param imageFiles the image files
     * @param devices the target devices (one per image, or single device for all)
     * @return list of preprocessed tensors on their respective devices
     * @throws IOException if loading fails
     */
    public List<INDArray> preprocessBatchOnDevices(List<File> imageFiles,
                                                   List<DeviceDescriptor> devices) throws IOException {
        if (devices.size() != 1 && devices.size() != imageFiles.size()) {
            throw new IllegalArgumentException(
                    "Devices list must have 1 element or match imageFiles size");
        }

        List<INDArray> results = new ArrayList<>(imageFiles.size());
        for (int i = 0; i < imageFiles.size(); i++) {
            DeviceDescriptor device = devices.size() == 1 ? devices.get(0) : devices.get(i);
            results.add(preprocessOnDevice(imageFiles.get(i), device));
        }
        return results;
    }

    /**
     * Asynchronously preprocess an image and prefetch to a device.
     * Returns immediately with a future that completes when preprocessing and transfer are done.
     *
     * @param imageFile the image file
     * @param device the target device
     * @return future containing the preprocessed tensor on the device
     */
    public CompletableFuture<INDArray> preprocessAsync(File imageFile, DeviceDescriptor device) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return preprocessOnDevice(imageFile, device);
            } catch (IOException e) {
                throw new RuntimeException("Failed to preprocess image: " + imageFile, e);
            }
        }, prefetchExecutor);
    }

    /**
     * Prefetch an image to a device for later use.
     * The image is preprocessed and transferred to the device in the background.
     *
     * @param imageFile the image file to prefetch
     * @param device the target device
     * @return future that completes when prefetch is done
     */
    public CompletableFuture<Void> prefetchImage(File imageFile, DeviceDescriptor device) {
        return CompletableFuture.runAsync(() -> {
            try {
                INDArray result = preprocess(imageFile);
                DataBuffer buffer = result.data();
                if (buffer != null && buffer.isHybrid()) {
                    buffer.asHybrid().prefetch(device);
                }
            } catch (IOException e) {
                log.warn("Failed to prefetch image: {}", imageFile, e);
            }
        }, prefetchExecutor);
    }

    /**
     * Preprocess and extract patches on a specific device.
     *
     * @param imageFile the image file
     * @param patchSize the patch size for ViT
     * @param device the target device
     * @return patches tensor on the specified device
     * @throws IOException if loading fails
     */
    public INDArray preprocessAndExtractPatchesOnDevice(File imageFile, int patchSize,
                                                        DeviceDescriptor device) throws IOException {
        INDArray preprocessed = preprocessOnDevice(imageFile, device);
        INDArray patches = extractPatches(preprocessed, patchSize);
        ensureOnDevice(patches, device);
        return patches;
    }

    /**
     * Create a preprocessor configured for a specific device.
     *
     * @param config the preprocessor config
     * @param device the target device
     * @return preprocessor that will place outputs on the specified device
     */
    public static VLMImagePreprocessor forDevice(PreprocessorConfig config, DeviceDescriptor device) {
        return new VLMImagePreprocessor(config, device, null);
    }

    /**
     * Create a preprocessor with workspace-based allocation on a specific device.
     *
     * @param config the preprocessor config
     * @param device the target device
     * @param workspace the workspace for memory management
     * @return preprocessor using workspace allocation
     */
    public static VLMImagePreprocessor forDeviceWithWorkspace(PreprocessorConfig config,
                                                               DeviceDescriptor device,
                                                               MultiBackendWorkspace workspace) {
        return new VLMImagePreprocessor(config, device, workspace);
    }

    /**
     * Ensure an INDArray is available on the specified device.
     * If targetDevice is set and device is null, uses targetDevice.
     *
     * @param array the array to transfer
     * @param device the target device (or null to use default targetDevice)
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
     * Shutdown the prefetch executor.
     * Should be called when the preprocessor is no longer needed.
     */
    public void shutdown() {
        if (prefetchExecutor != null && !prefetchExecutor.isShutdown()) {
            prefetchExecutor.shutdown();
        }
    }
}
