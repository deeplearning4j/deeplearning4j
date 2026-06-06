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
import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.vlm.model.VisionEncoderUtils;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Preprocesses video frames for Vision-Language Models.
 *
 * <p>Takes a list of extracted video frames and produces a 5D tensor
 * [1, numFrames, 3, H, W] suitable for video VLM input.</p>
 *
 * <p>All configuration is via the builder — resolution, sampler, temporal
 * patch size, and thread count are all parameters, not model-specific
 * factory methods.</p>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * VideoPreprocessor preprocessor = VideoPreprocessor.builder()
 *     .imagePreprocessor(VLMImagePreprocessor.fromConfig(myConfig))
 *     .sampler(VideoFrameSampler.builder()
 *         .strategy(VideoFrameSampler.Strategy.UNIFORM)
 *         .maxFrames(16)
 *         .build())
 *     .targetHeight(384)
 *     .targetWidth(384)
 *     .temporalPatchSize(2)
 *     .build();
 *
 * INDArray tensor = preprocessor.preprocessVideo(new File("video.mp4"));
 * }</pre>
 *
 * @see VideoFrameExtractor
 * @see VideoFrameSampler
 * @see VLMImagePreprocessor
 */
@Slf4j
@Getter
@Builder
public class VideoPreprocessor {

    private VLMImagePreprocessor imagePreprocessor;

    @Builder.Default
    private VideoFrameSampler sampler = VideoFrameSampler.builder().build();

    @Builder.Default
    private int targetHeight = 384;

    @Builder.Default
    private int targetWidth = 384;

    @Builder.Default
    private int numPreprocessThreads = 4;

    @Builder.Default
    private int temporalPatchSize = 1;

    /**
     * Preprocess a video file end-to-end: extract frames, sample, and convert to tensor.
     *
     * @param videoFile the video file
     * @return preprocessed tensor [1, numFrames, 3, H, W]
     * @throws IOException if the video cannot be read
     */
    public INDArray preprocessVideo(File videoFile) throws IOException {
        List<BufferedImage> frames = sampler.sample(videoFile);
        return preprocessFrames(frames);
    }

    /**
     * Preprocess a list of video frames into a 5D tensor.
     *
     * <p>Uses parallel preprocessing when numPreprocessThreads > 1.</p>
     *
     * @param frames the extracted video frames
     * @return preprocessed tensor [1, numFrames, 3, targetHeight, targetWidth]
     */
    public INDArray preprocessFrames(List<BufferedImage> frames) {
        if (frames == null || frames.isEmpty()) {
            return Nd4j.create(DataType.FLOAT, 1, 0, 3, targetHeight, targetWidth);
        }

        int targetSize = Math.max(targetHeight, targetWidth);

        if (numPreprocessThreads > 1 && frames.size() > 1) {
            return VisionEncoderUtils.preprocessFramesParallel(
                    frames,
                    this::createThreadLocalPreprocessor,
                    targetSize,
                    numPreprocessThreads
            );
        } else {
            return VisionEncoderUtils.preprocessFrames(frames, imagePreprocessor, targetSize);
        }
    }

    /**
     * Preprocess a video file and pad frames to be divisible by temporalPatchSize.
     *
     * <p>Models with temporal patching (e.g., temporal_patch_size=2) require
     * the frame count to be divisible by the temporal factor. This method pads
     * by duplicating the last frame if needed.</p>
     *
     * @param videoFile the video file
     * @return preprocessed tensor with frame count divisible by temporalPatchSize
     * @throws IOException if the video cannot be read
     */
    public INDArray preprocessVideoTemporalAligned(File videoFile) throws IOException {
        List<BufferedImage> frames = sampler.sample(videoFile);
        return preprocessFramesTemporalAligned(frames);
    }

    /**
     * Preprocess frames with temporal alignment padding.
     *
     * @param frames the extracted video frames
     * @return preprocessed tensor with frame count divisible by temporalPatchSize
     */
    public INDArray preprocessFramesTemporalAligned(List<BufferedImage> frames) {
        if (temporalPatchSize <= 1) {
            return preprocessFrames(frames);
        }

        int numFrames = frames.size();
        int remainder = numFrames % temporalPatchSize;
        if (remainder != 0) {
            int padCount = temporalPatchSize - remainder;
            BufferedImage lastFrame = frames.get(numFrames - 1);
            for (int i = 0; i < padCount; i++) {
                frames.add(lastFrame);
            }
            log.info("Padded {} frames to {} for temporal_patch_size={}",
                    numFrames, frames.size(), temporalPatchSize);
        }

        return preprocessFrames(frames);
    }

    /**
     * Estimate the number of vision tokens for a given number of frames.
     *
     * @param numFrames the number of input frames
     * @param patchSize the vision encoder patch size (e.g., 14 for SigLIP, 16 for Qwen-VL)
     * @param pixelShuffleFactor the pixel shuffle compression factor (1 = no shuffle)
     * @return estimated number of vision tokens
     */
    public int estimateVisionTokens(int numFrames, int patchSize, int pixelShuffleFactor) {
        int patchesPerFrame = (targetHeight / patchSize) * (targetWidth / patchSize);
        int tokensPerFrame = patchesPerFrame / (pixelShuffleFactor * pixelShuffleFactor);
        int effectiveFrames = temporalPatchSize > 1 ? numFrames / temporalPatchSize : numFrames;
        return effectiveFrames * tokensPerFrame;
    }

    private VLMImagePreprocessor createThreadLocalPreprocessor() {
        return VLMImagePreprocessor.fromConfig(imagePreprocessor.getConfig());
    }
}
