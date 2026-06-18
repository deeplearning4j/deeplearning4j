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

package org.eclipse.deeplearning4j.vlm.model.encoder;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Model-agnostic vision encoder for VLM pipelines.
 *
 * <p>Handles the complete image-to-embeddings path: image tiling, preprocessing,
 * per-frame SameDiff forward passes, and concatenation into a single embedding
 * tensor. Not tied to any specific VLM architecture (SmolDocling, LLaVA, etc.).</p>
 *
 * <p>Usage:</p>
 * <pre>{@code
 * VisionEncoder encoder = VisionEncoder.builder()
 *     .model(visionEncoderSd)
 *     .preprocessor(VLMImagePreprocessor.fromConfig(ppConfig))
 *     .targetSize(512)
 *     .maxTiles(-1)
 *     .build();
 *
 * VisionEncoder.Result result = encoder.encode(image, splitResult);
 * INDArray visionEmbeddings = result.getEmbeddings();
 * encoder.close();
 * }</pre>
 *
 * <p>For pipelined multi-page encoding with CPU/GPU overlap, use
 * {@link PipelinedVisionEncoder} instead.</p>
 *
 * @see PipelinedVisionEncoder
 * @see VisionEncoderUtils
 * @see EmbeddingMerger
 *
 * Adam Gibson
 */
@Slf4j
@Builder
public class VisionEncoder implements AutoCloseable {

    private final SameDiff model;

    /**
     * Optional preprocessor reserved for a future {@code encode(BufferedImage)} convenience method
     * that would handle tiling and normalization internally. The current
     * {@link #encode(INDArray, int, ImageTiler.SplitImageResult)} entry-point takes already
     * pre-processed, pre-tiled input, so this field is not used today.
     */
    private final VLMImagePreprocessor preprocessor;
    @Builder.Default private final int targetSize = 512;
    @Builder.Default private final int maxTiles = -1;

    /**
     * Encode a pre-tiled image into vision embeddings.
     *
     * <p>Runs the vision encoder model on each frame from the tiled image,
     * then concatenates the per-frame embeddings along the sequence dimension.</p>
     *
     * @param tiledImage  pre-tiled image tensor [1, numFrames, C, H, W]
     * @param numFrames   number of frames (tiles) in the tiled image tensor
     * @param splitResult tiling result ({@link ImageTiler.SplitImageResult}) that provides the
     *                    per-frame {@link ImageTiler.ContentRegion} values used to build pixel
     *                    attention masks
     * @return encoding result with concatenated embeddings
     */
    public Result encode(INDArray tiledImage, int numFrames, ImageTiler.SplitImageResult splitResult) {
        long startMs = System.currentTimeMillis();

        List<String> inputNames = model.inputs();
        String[] outputNames = model.outputs().toArray(new String[0]);

        List<INDArray> frameEmbeddings = new ArrayList<>();
        List<INDArray> frameInputs = new ArrayList<>();
        boolean frameLoopSucceeded = false;
        try {
            for (int frameIdx = 0; frameIdx < numFrames; frameIdx++) {
                INDArray frameSlice = tiledImage.get(
                        NDArrayIndex.point(0), NDArrayIndex.point(frameIdx),
                        NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
                // SmolDocling's vision encoder expects rank-4 NCHW input per frame.
                // Passing rank-5 [1, 1, C, H, W] breaks ONNX zero-copy reshape constants.
                INDArray singleFrame = frameSlice.reshape(1, 3, targetSize, targetSize).dup();
                frameInputs.add(singleFrame);

                Map<String, INDArray> inputMap = new HashMap<>();
                for (String inputName : inputNames) {
                    if (inputName.equals("pixel_values")) {
                        inputMap.put(inputName, singleFrame);
                    } else if (inputName.equals("pixel_attention_mask")) {
                        ImageTiler.ContentRegion region = splitResult.contentRegions.get(frameIdx);
                        INDArray attentionMask = ImageTiler.createPixelAttentionMask(region.width, region.height, targetSize);
                        inputMap.put(inputName, attentionMask);
                        frameInputs.add(attentionMask);
                    }
                }

                Map<String, INDArray> outputs = model.output(inputMap, outputNames);

                VisionEncoderUtils.VisionOutput selected = VisionEncoderUtils.selectVisionOutput(outputs);
                if (selected == null || selected.tensor == null) {
                    throw new IllegalStateException("Vision encoder produced no usable output for frame " + frameIdx);
                }

                INDArray out = selected.tensor.dup();
                frameEmbeddings.add(out);

            }
            frameLoopSucceeded = true;
        } finally {
            for (INDArray frameInput : frameInputs) {
                if (frameInput != null && frameInput.closeable() && !frameInput.wasClosed()) frameInput.close();
            }
            if (!frameLoopSucceeded) {
                for (INDArray fe : frameEmbeddings) {
                    if (fe != null && fe.closeable() && !fe.wasClosed()) fe.close();
                }
            }
        }

        // Concatenate frame embeddings along sequence dimension
        INDArray embeddings;
        if (frameEmbeddings.size() == 1) {
            embeddings = frameEmbeddings.get(0).dup();
        } else {
            embeddings = Nd4j.concat(1, frameEmbeddings.toArray(new INDArray[0])).dup();
        }

        // Clean up individual frame embeddings
        for (INDArray fe : frameEmbeddings) {
            if (fe != null && fe.closeable() && !fe.wasClosed()) fe.close();
        }

        long encodingTimeMs = System.currentTimeMillis() - startMs;
        log.info("Vision encoder done [{}ms]: {} frames, embeddings shape={}",
                encodingTimeMs, numFrames, Arrays.toString(embeddings.shape()));

        return new Result(embeddings, numFrames, encodingTimeMs);
    }

    /**
     * Free the vision encoder's model memory.
     *
     * <p>Releases all constant/variable arrays from the SameDiff model. Call this
     * after encoding is complete but before starting decode, to free GPU memory
     * for the decoder model.</p>
     */
    public void freeModelMemory() {
        org.eclipse.deeplearning4j.llm.generation.SameDiffMemoryUtils.freeVisionEncoder(model);
    }

    @Override
    public void close() {
        // The model is NOT closed here — the caller owns it.
        // Call freeModelMemory() explicitly to release GPU memory.
    }

    /**
     * Result of vision encoding.
     */
    @Getter
    public static class Result {
        private final INDArray embeddings;    // [1, totalSeqLen, hidden]
        private final int totalFrames;
        private final long encodingTimeMs;

        public Result(INDArray embeddings, int totalFrames, long encodingTimeMs) {
            this.embeddings = embeddings;
            this.totalFrames = totalFrames;
            this.encodingTimeMs = encodingTimeMs;
        }
    }
}
