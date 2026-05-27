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
import org.eclipse.deeplearning4j.llm.config.PreprocessorConfig;
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
 * <p>Takes a list of extracted video frames ({@link BufferedImage}) and produces
 * a 5D tensor suitable for video VLM input. Supports model-specific configurations
 * for different video VLMs:</p>
 * <ul>
 *   <li>SmolVLM2: 384x384, SigLIP normalization, frames as image sequence</li>
 *   <li>Qwen3-VL: dynamic resolution, ViT normalization, temporal patch pairs</li>
 *   <li>MiniCPM-V 4.5: 384x384, SigLIP2 normalization, 6-frame groups</li>
 * </ul>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * VideoPreprocessor preprocessor = VideoPreprocessor.forSmolVLM2();
 *
 * // From a video file
 * INDArray frames = preprocessor.preprocessVideo(new File("video.mp4"));
 *
 * // From pre-extracted frames
 * List<BufferedImage> frameList = sampler.sample(videoFile);
 * INDArray tensor = preprocessor.preprocessFrames(frameList);
 * // tensor shape: [1, numFrames, 3, H, W]
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

    @Builder.Default
    private VLMImagePreprocessor imagePreprocessor = VLMImagePreprocessor.defaultPreprocessor();

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
     * <p>Some models (e.g., Qwen3-VL with temporal_patch_size=2) require the number
     * of frames to be divisible by a temporal patch factor. This method pads by
     * duplicating the last frame if needed.</p>
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
     * Get the number of vision tokens that will be produced for a given number of frames.
     *
     * <p>This is model-dependent:</p>
     * <ul>
     *   <li>SmolVLM2: (H/14)*(W/14) / 9 tokens per frame (14px patches, 3x3 pixel shuffle)</li>
     *   <li>Qwen3-VL: (H/16)*(W/16) * (T/2) tokens (16px patches, temporal_patch_size=2)</li>
     *   <li>MiniCPM-V: 64 tokens per 6-frame group (3D-Resampler)</li>
     * </ul>
     *
     * @param numFrames the number of input frames
     * @param patchSize the vision encoder patch size
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

    // =========================================================================
    // Model-specific factory methods
    // =========================================================================

    /**
     * Create a preprocessor for SmolVLM2 video models.
     *
     * <p>SmolVLM2 uses SigLIP-so400m-patch14-384 as the vision encoder.
     * Frames are processed independently at 384x384, then compressed via 3x3 pixel shuffle.</p>
     *
     * @return SmolVLM2 video preprocessor
     */
    public static VideoPreprocessor forSmolVLM2() {
        PreprocessorConfig sigLipConfig = PreprocessorConfig.forViT();
        sigLipConfig.setSize(new PreprocessorConfig.ImageSize(384, 384));

        return VideoPreprocessor.builder()
                .imagePreprocessor(VLMImagePreprocessor.fromConfig(sigLipConfig))
                .sampler(VideoFrameSampler.forSmolVLM2())
                .targetHeight(384)
                .targetWidth(384)
                .temporalPatchSize(1)
                .numPreprocessThreads(4)
                .build();
    }

    /**
     * Create a preprocessor for Qwen3-VL video models.
     *
     * <p>Qwen3-VL uses SigLIP-2 with temporal_patch_size=2 (pairs of frames merged).
     * Frames are processed at 384x384 and the temporal dimension must be even.</p>
     *
     * @return Qwen3-VL video preprocessor
     */
    public static VideoPreprocessor forQwen3VL() {
        PreprocessorConfig sigLip2Config = PreprocessorConfig.forViT();
        sigLip2Config.setSize(new PreprocessorConfig.ImageSize(384, 384));

        return VideoPreprocessor.builder()
                .imagePreprocessor(VLMImagePreprocessor.fromConfig(sigLip2Config))
                .sampler(VideoFrameSampler.forQwen3VL())
                .targetHeight(384)
                .targetWidth(384)
                .temporalPatchSize(2)
                .numPreprocessThreads(4)
                .build();
    }

    /**
     * Create a preprocessor for MiniCPM-V 4.5 video models.
     *
     * <p>MiniCPM-V 4.5 uses SigLIP2-400M with a 3D-Resampler that groups
     * 6 consecutive frames into 64 tokens. Frame count should be divisible by 6.</p>
     *
     * @return MiniCPM-V 4.5 video preprocessor
     */
    public static VideoPreprocessor forMiniCPMV() {
        PreprocessorConfig sigLip2Config = PreprocessorConfig.forViT();
        sigLip2Config.setSize(new PreprocessorConfig.ImageSize(384, 384));

        return VideoPreprocessor.builder()
                .imagePreprocessor(VLMImagePreprocessor.fromConfig(sigLip2Config))
                .sampler(VideoFrameSampler.forMiniCPMV())
                .targetHeight(384)
                .targetWidth(384)
                .temporalPatchSize(6)
                .numPreprocessThreads(4)
                .build();
    }

    /**
     * Create a preprocessor for Qwen2.5-VL video models.
     *
     * <p>Similar to Qwen3-VL but using the older SigLIP vision encoder.</p>
     *
     * @return Qwen2.5-VL video preprocessor
     */
    public static VideoPreprocessor forQwen25VL() {
        PreprocessorConfig sigLipConfig = PreprocessorConfig.forViT();
        sigLipConfig.setSize(new PreprocessorConfig.ImageSize(384, 384));

        return VideoPreprocessor.builder()
                .imagePreprocessor(VLMImagePreprocessor.fromConfig(sigLipConfig))
                .sampler(VideoFrameSampler.builder()
                        .strategy(VideoFrameSampler.Strategy.FIXED_FPS)
                        .targetFPS(2.0)
                        .maxFrames(64)
                        .build())
                .targetHeight(384)
                .targetWidth(384)
                .temporalPatchSize(2)
                .numPreprocessThreads(4)
                .build();
    }
}
