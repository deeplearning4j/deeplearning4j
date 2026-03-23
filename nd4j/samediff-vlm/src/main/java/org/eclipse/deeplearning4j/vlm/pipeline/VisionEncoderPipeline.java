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

package org.eclipse.deeplearning4j.vlm.pipeline;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.vlm.model.VisionEncoderUtils;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.vlm.preprocessing.PreprocessorConfig;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Encapsulates the entire vision encoding lifecycle for VLM inference:
 * image tiling, preprocessing, per-frame encoder execution, embedding
 * concatenation, and cleanup.
 *
 * <p>This class extracts the vision encoding flow that was previously
 * inlined in test code, making it reusable across applications. It handles
 * proper GPU memory management by closing intermediate arrays and providing
 * an explicit {@link #freeEncoder()} method to release model weights when
 * encoding is complete (before decode begins).</p>
 *
 * <p>Usage:</p>
 * <pre>{@code
 * SameDiff visionEncoder = OnnxModelCache.importVisionEncoder(...);
 * VisionEncoderPipeline pipeline = new VisionEncoderPipeline(visionEncoder, 512);
 *
 * // Encode an image into vision embeddings
 * VisionEncoderPipeline.EncodeResult result = pipeline.encode(pdfImage);
 * INDArray visionEmbeddings = result.getEmbeddings();
 * int numFrames = result.getNumFrames();
 * int numRows = result.getNumRows();
 * int numCols = result.getNumCols();
 *
 * // Free encoder GPU memory before starting decode
 * pipeline.freeEncoder();
 *
 * // ... use visionEmbeddings in decoder pipeline ...
 * pipeline.close();
 * }</pre>
 */
@Slf4j
public class VisionEncoderPipeline implements AutoCloseable {

    private final SameDiff visionEncoder;
    private final VLMImagePreprocessor preprocessor;
    @Getter
    private final int targetSize;
    @Getter
    private final int maxTiles;
    @Getter
    private final int longestEdgeResize;
    private boolean encoderFreed = false;
    private boolean closed = false;

    /**
     * Result of encoding an image, containing the concatenated vision embeddings
     * and tiling metadata needed for prompt construction.
     */
    @Getter
    public static class EncodeResult {
        private final INDArray embeddings;
        private final int numFrames;
        private final int numRows;
        private final int numCols;

        public EncodeResult(INDArray embeddings, int numFrames, int numRows, int numCols) {
            this.embeddings = embeddings;
            this.numFrames = numFrames;
            this.numRows = numRows;
            this.numCols = numCols;
        }
    }

    /**
     * Create a VisionEncoderPipeline with default SmolDocling-style preprocessing
     * (rescale 1/255, normalize mean=0.5 std=0.5, longestEdge=2048, maxTiles=9).
     *
     * @param visionEncoder the imported SameDiff vision encoder graph
     * @param targetSize    the tile size for the vision encoder (e.g. 512)
     */
    public VisionEncoderPipeline(SameDiff visionEncoder, int targetSize) {
        this(visionEncoder, targetSize, buildDefaultConfig(targetSize), 2048, 9);
    }

    /**
     * Create a VisionEncoderPipeline with custom preprocessing configuration.
     *
     * @param visionEncoder     the imported SameDiff vision encoder graph
     * @param targetSize        the tile size for the vision encoder (e.g. 512)
     * @param preprocessorConfig preprocessing configuration (rescale, normalize, etc.)
     * @param longestEdgeResize resize the longest edge to this before tiling (e.g. 2048)
     * @param maxTiles          maximum number of content tiles (-1 for unlimited)
     */
    public VisionEncoderPipeline(SameDiff visionEncoder, int targetSize,
                                  PreprocessorConfig preprocessorConfig,
                                  int longestEdgeResize, int maxTiles) {
        if (visionEncoder == null) {
            throw new IllegalArgumentException("visionEncoder must not be null");
        }
        if (targetSize <= 0) {
            throw new IllegalArgumentException("targetSize must be positive, got " + targetSize);
        }
        this.visionEncoder = visionEncoder;
        this.targetSize = targetSize;
        this.longestEdgeResize = longestEdgeResize;
        this.maxTiles = maxTiles;
        this.preprocessor = VLMImagePreprocessor.fromConfig(preprocessorConfig);
    }

    /**
     * Encode a single image into vision embeddings.
     *
     * <p>The full pipeline is:</p>
     * <ol>
     *   <li>Resize image so longest edge matches {@code longestEdgeResize}</li>
     *   <li>Split into tiles via {@link ImageTiler#splitImageForVLM}</li>
     *   <li>Preprocess all frames (rescale + normalize) into a 5D tensor</li>
     *   <li>Run the vision encoder on each frame sequentially</li>
     *   <li>Concatenate frame embeddings along the sequence dimension</li>
     * </ol>
     *
     * @param image the input image (e.g. a rendered PDF page)
     * @return EncodeResult containing the concatenated vision embeddings and tiling metadata
     * @throws IllegalStateException if the encoder has already been freed or the pipeline is closed
     */
    public EncodeResult encode(BufferedImage image) {
        checkState();
        if (image == null) {
            throw new IllegalArgumentException("image must not be null");
        }

        long startTime = System.currentTimeMillis();

        // Step 1: Resize longest edge
        BufferedImage resizedForTiling = ImageTiler.resizeLongestEdge(image, longestEdgeResize);

        // Step 2: Split into tiles
        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(resizedForTiling, targetSize, maxTiles);
        int numFrames = splitResult.getTotalFrames();
        log.info("Image tiled into {} frames ({}x{} grid + 1 global)", numFrames,
                splitResult.numRows, splitResult.numCols);

        // Step 3: Preprocess frames
        INDArray imageInput = VisionEncoderUtils.preprocessFrames(splitResult.frames, preprocessor, targetSize);

        // Step 4: Run vision encoder on each frame
        List<String> visionInputNames = visionEncoder.inputs();
        String[] visionOutputNames = visionEncoder.outputs().toArray(new String[0]);

        List<INDArray> frameEmbeddings = new ArrayList<>(numFrames);
        for (int frameIdx = 0; frameIdx < numFrames; frameIdx++) {
            // Extract single frame from the 5D tensor [1, numFrames, 3, H, W]
            INDArray frameSlice = imageInput.get(
                    NDArrayIndex.point(0), NDArrayIndex.point(frameIdx),
                    NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
            INDArray singleFrame = frameSlice.reshape(1, 1, 3, targetSize, targetSize).dup();

            // Build input map
            Map<String, INDArray> visionInputMap = new HashMap<>();
            for (String inputName : visionInputNames) {
                if (inputName.equals("pixel_values")) {
                    visionInputMap.put(inputName, singleFrame);
                } else if (inputName.equals("pixel_attention_mask")) {
                    ImageTiler.ContentRegion region = splitResult.contentRegions.get(frameIdx);
                    visionInputMap.put(inputName,
                            ImageTiler.createPixelAttentionMask(region.width, region.height, targetSize));
                }
            }

            // Execute vision encoder
            Map<String, INDArray> visionOutputs = visionEncoder.output(visionInputMap, visionOutputNames);

            // Select and dup the best output
            VisionEncoderUtils.VisionOutput selected = VisionEncoderUtils.selectVisionOutput(visionOutputs);
            if (selected == null || selected.tensor == null) {
                throw new IllegalStateException("Vision encoder produced no usable output for frame " + frameIdx);
            }

            INDArray out = selected.tensor.dup();
            frameEmbeddings.add(out);

            // Close intermediate outputs to prevent GPU leaks
            for (Map.Entry<String, INDArray> entry : visionOutputs.entrySet()) {
                INDArray arr = entry.getValue();
                if (arr != null && arr.closeable() && !arr.wasClosed()) {
                    arr.close();
                }
            }
            singleFrame.close();

            log.debug("Encoded frame {}/{}: output shape={}", frameIdx + 1, numFrames,
                    java.util.Arrays.toString(out.shape()));
        }

        // Clean up vision encoder session state
        visionEncoder.clearPlaceholders(false);
        visionEncoder.clearOpInputs();
        visionEncoder.resetSession();
        Nd4j.getExecutioner().commit();

        // Step 5: Concatenate frame embeddings
        INDArray visionEmbeddings;
        if (frameEmbeddings.size() == 1) {
            visionEmbeddings = frameEmbeddings.get(0).dup();
        } else {
            visionEmbeddings = Nd4j.concat(1, frameEmbeddings.toArray(new INDArray[0])).dup();
        }

        // Close individual frame embeddings
        for (INDArray fe : frameEmbeddings) {
            if (fe != null && fe.closeable() && !fe.wasClosed()) {
                fe.close();
            }
        }

        // Close the preprocessed image input
        imageInput.close();

        long elapsed = System.currentTimeMillis() - startTime;
        log.info("Vision encoding complete [{}ms]: {} frames, embeddings shape={}",
                elapsed, numFrames, java.util.Arrays.toString(visionEmbeddings.shape()));

        return new EncodeResult(visionEmbeddings, numFrames, splitResult.numRows, splitResult.numCols);
    }

    /**
     * Free the vision encoder model weights to recover GPU memory.
     * Call this after encoding is complete and before starting decode.
     *
     * <p>After calling this method, {@link #encode(BufferedImage)} will throw
     * {@link IllegalStateException}.</p>
     */
    public void freeEncoder() {
        if (encoderFreed) {
            return;
        }
        encoderFreed = true;

        int closedArrays = 0;
        long closedBytes = 0;
        for (ArrayHolder holder : new ArrayHolder[]{visionEncoder.getConstantArrays(), visionEncoder.getVariablesArrays()}) {
            for (String name : new ArrayList<>(holder.arrayNames())) {
                INDArray arr = holder.removeArray(name);
                if (arr != null && !arr.wasClosed()) {
                    closedBytes += arr.length() * arr.dataType().width();
                    arr.data().setConstant(false);
                    arr.close();
                    closedArrays++;
                }
            }
        }
        Nd4j.getExecutioner().commit();
        log.info("Freed {} vision encoder arrays ({}MB)", closedArrays, closedBytes / (1024 * 1024));
    }

    /**
     * Returns whether the encoder has been freed via {@link #freeEncoder()}.
     *
     * @return true if the encoder weights have been released
     */
    public boolean isEncoderFreed() {
        return encoderFreed;
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;

        if (!encoderFreed) {
            freeEncoder();
        }

        preprocessor.shutdown();
        log.info("VisionEncoderPipeline closed");
    }

    private void checkState() {
        if (closed) {
            throw new IllegalStateException("VisionEncoderPipeline has been closed");
        }
        if (encoderFreed) {
            throw new IllegalStateException("Vision encoder has been freed; cannot encode");
        }
    }

    private static PreprocessorConfig buildDefaultConfig(int targetSize) {
        PreprocessorConfig config = new PreprocessorConfig();
        config.setSize(new PreprocessorConfig.ImageSize(targetSize, targetSize));
        config.setDoRescale(true);
        config.setRescaleFactor(1.0 / 255.0);
        config.setDoNormalize(true);
        config.setImageMean(new double[]{0.5, 0.5, 0.5});
        config.setImageStd(new double[]{0.5, 0.5, 0.5});
        return config;
    }
}
