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
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.vlm.preprocessing.VideoFrameExtractor;
import org.eclipse.deeplearning4j.vlm.preprocessing.VideoFrameSampler;
import org.eclipse.deeplearning4j.vlm.preprocessing.VideoPreprocessor;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Video-capable Vision-Language Model for understanding video content.
 *
 * <p>Extends the image-based {@link VisionLanguageModel} pipeline to handle
 * video inputs. Supports multiple video VLM architectures:</p>
 * <ul>
 *   <li><b>SmolVLM2</b>: Frames processed as image sequence, pixel shuffle compression</li>
 *   <li><b>Qwen3-VL</b>: Temporal patches with M-RoPE, DeepStack fusion</li>
 *   <li><b>MiniCPM-V 4.5</b>: 3D-Resampler grouping 6 frames into 64 tokens</li>
 * </ul>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * // Load a video VLM
 * VideoVisionLanguageModel videoVLM = VideoVisionLanguageModel.fromDirectory(
 *     new File("SmolVLM2-256M-Video-Instruct-sdz")
 * );
 *
 * // Generate description from a video file
 * String description = videoVLM.generate(new File("video.mp4"), "Describe this video.");
 *
 * // Or from pre-extracted frames
 * List<BufferedImage> frames = ...;
 * String description = videoVLM.generate(frames, "What is happening in this video?");
 *
 * videoVLM.close();
 * }</pre>
 *
 * @see VisionLanguageModel
 * @see VideoPreprocessor
 * @see VideoFrameSampler
 */
@Slf4j
@Getter
public class VideoVisionLanguageModel implements AutoCloseable {

    private final VisionLanguageModel vlm;
    private final VideoPreprocessor videoPreprocessor;

    @Setter
    private int maxFrames = 32;

    @Setter
    private int maxNewTokens = 512;

    @Setter
    private double temperature = 1.0;

    @Setter
    private boolean doSample = true;

    @Builder
    private VideoVisionLanguageModel(
            VisionLanguageModel vlm,
            VideoPreprocessor videoPreprocessor,
            Integer maxFrames,
            Integer maxNewTokens,
            Double temperature,
            Boolean doSample) {
        this.vlm = vlm;
        this.videoPreprocessor = videoPreprocessor != null
                ? videoPreprocessor : VideoPreprocessor.builder().build();
        if (maxFrames != null) this.maxFrames = maxFrames;
        if (maxNewTokens != null) this.maxNewTokens = maxNewTokens;
        if (temperature != null) this.temperature = temperature;
        if (doSample != null) this.doSample = doSample;
    }

    // =========================================================================
    // Video generation from file
    // =========================================================================

    /**
     * Generate text from a video file and prompt.
     *
     * <p>Performs the full pipeline: extract frames -> sample -> preprocess -> encode -> decode.</p>
     *
     * @param videoFile the video file (MP4, AVI, MKV, WebM, etc.)
     * @param prompt the text prompt
     * @return generated text describing the video
     * @throws IOException if the video cannot be read
     */
    public String generate(File videoFile, String prompt) throws IOException {
        return generate(videoFile, prompt, maxNewTokens, temperature, doSample);
    }

    /**
     * Generate text from a video file with full parameter control.
     *
     * @param videoFile the video file
     * @param prompt the text prompt
     * @param maxNewTokens maximum tokens to generate
     * @param temperature sampling temperature
     * @param doSample whether to sample (false = greedy)
     * @return generated text
     * @throws IOException if the video cannot be read
     */
    public String generate(File videoFile, String prompt, int maxNewTokens,
                          double temperature, boolean doSample) throws IOException {
        return generateWithMetrics(videoFile, prompt, maxNewTokens, temperature, doSample).getText();
    }

    /**
     * Generate text from a video file, returning detailed metrics.
     *
     * @param videoFile the video file
     * @param prompt the text prompt
     * @return generation result with text and metrics
     * @throws IOException if the video cannot be read
     */
    public GenerationResult generateWithMetrics(File videoFile, String prompt) throws IOException {
        return generateWithMetrics(videoFile, prompt, maxNewTokens, temperature, doSample);
    }

    /**
     * Generate text from a video file with full parameter control, returning metrics.
     *
     * @param videoFile the video file
     * @param prompt the text prompt
     * @param maxNewTokens maximum tokens to generate
     * @param temperature sampling temperature
     * @param doSample whether to sample (false = greedy)
     * @return generation result with text and metrics
     * @throws IOException if the video cannot be read
     */
    public GenerationResult generateWithMetrics(File videoFile, String prompt, int maxNewTokens,
                                                double temperature, boolean doSample) throws IOException {
        long startTime = System.currentTimeMillis();

        INDArray frameTensor = videoPreprocessor.preprocessVideoTemporalAligned(videoFile);
        log.info("Video preprocessing took {}ms, frame tensor shape: {}",
                System.currentTimeMillis() - startTime,
                java.util.Arrays.toString(frameTensor.shape()));

        try {
            return generateFromFrameTensor(frameTensor, prompt, maxNewTokens, temperature, doSample);
        } finally {
            frameTensor.close();
        }
    }

    // =========================================================================
    // Video generation from pre-extracted frames
    // =========================================================================

    /**
     * Generate text from pre-extracted video frames.
     *
     * @param frames the video frames
     * @param prompt the text prompt
     * @return generated text
     */
    public String generate(List<BufferedImage> frames, String prompt) {
        return generate(frames, prompt, maxNewTokens, temperature, doSample);
    }

    /**
     * Generate text from pre-extracted video frames with parameters.
     *
     * @param frames the video frames
     * @param prompt the text prompt
     * @param maxNewTokens maximum tokens to generate
     * @param temperature sampling temperature
     * @param doSample whether to sample (false = greedy)
     * @return generated text
     */
    public String generate(List<BufferedImage> frames, String prompt, int maxNewTokens,
                          double temperature, boolean doSample) {
        return generateWithMetrics(frames, prompt, maxNewTokens, temperature, doSample).getText();
    }

    /**
     * Generate text from pre-extracted frames, returning metrics.
     *
     * @param frames the video frames
     * @param prompt the text prompt
     * @param maxNewTokens maximum tokens to generate
     * @param temperature sampling temperature
     * @param doSample whether to sample (false = greedy)
     * @return generation result with text and metrics
     */
    public GenerationResult generateWithMetrics(List<BufferedImage> frames, String prompt,
                                                int maxNewTokens, double temperature,
                                                boolean doSample) {
        INDArray frameTensor = videoPreprocessor.preprocessFramesTemporalAligned(frames);
        try {
            return generateFromFrameTensor(frameTensor, prompt, maxNewTokens, temperature, doSample);
        } finally {
            frameTensor.close();
        }
    }

    // =========================================================================
    // Core frame tensor processing
    // =========================================================================

    /**
     * Generate from a preprocessed frame tensor.
     *
     * <p>The frame tensor should be [1, numFrames, 3, H, W]. Each frame is
     * encoded through the vision encoder independently, and the resulting
     * embeddings are concatenated along the sequence dimension.</p>
     *
     * @param frameTensor preprocessed frames [1, numFrames, 3, H, W]
     * @param prompt the text prompt
     * @param maxNewTokens maximum tokens to generate
     * @param temperature sampling temperature
     * @param doSample whether to sample
     * @return generation result
     */
    public GenerationResult generateFromFrameTensor(INDArray frameTensor, String prompt,
                                                     int maxNewTokens, double temperature,
                                                     boolean doSample) {
        long numFrames = frameTensor.size(1);
        log.info("Encoding {} video frames through vision encoder", numFrames);

        long encodeStart = System.currentTimeMillis();

        // Encode each frame through the vision encoder and collect embeddings
        List<INDArray> frameEmbeddings = new ArrayList<>((int) numFrames);
        for (int f = 0; f < numFrames; f++) {
            // Extract single frame: [1, 3, H, W]
            INDArray singleFrame = frameTensor.get(
                    NDArrayIndex.point(0),
                    NDArrayIndex.point(f),
                    NDArrayIndex.all(),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
            ).reshape(1, frameTensor.size(2), frameTensor.size(3), frameTensor.size(4));

            INDArray embedding = vlm.encodeImage(singleFrame);
            frameEmbeddings.add(embedding);

            if (f == 0) {
                log.info("Per-frame embedding shape: {}", java.util.Arrays.toString(embedding.shape()));
            }
        }

        long encodeTime = System.currentTimeMillis() - encodeStart;
        log.info("Encoded {} frames in {}ms ({:.1f}ms/frame)",
                numFrames, encodeTime, (double) encodeTime / numFrames);

        // Concatenate all frame embeddings along sequence dimension
        // Each embedding is [1, tokens_per_frame, hidden_dim]
        // Result is [1, total_vision_tokens, hidden_dim]
        INDArray allFrameEmbeddings;
        if (frameEmbeddings.size() == 1) {
            allFrameEmbeddings = frameEmbeddings.get(0);
        } else {
            INDArray[] embeddingArray = frameEmbeddings.toArray(new INDArray[0]);
            allFrameEmbeddings = Nd4j.concat(1, embeddingArray);
            // Close individual frame embeddings after concat
            for (INDArray emb : frameEmbeddings) {
                if (!emb.wasClosed()) {
                    emb.close();
                }
            }
        }

        log.info("Combined vision embeddings shape: {} ({} total tokens)",
                java.util.Arrays.toString(allFrameEmbeddings.shape()),
                allFrameEmbeddings.size(1));

        // Determine vision tokens per frame for prompt building
        int tokensPerFrame = (int) (allFrameEmbeddings.size(1) / numFrames);

        // Use the VLM's generateFromEmbeddings which handles prompt building,
        // token merging, and autoregressive decode
        SamplingConfig samplingConfig = SamplingConfig.builder()
                .maxNewTokens(maxNewTokens)
                .temperature(temperature)
                .doSample(doSample)
                .build();

        try {
            return vlm.generateFromEmbeddings(
                    allFrameEmbeddings,
                    prompt,
                    samplingConfig,
                    0, 0,  // no spatial tiling for video frames
                    tokensPerFrame
            );
        } finally {
            if (!allFrameEmbeddings.wasClosed()) {
                allFrameEmbeddings.close();
            }
        }
    }

    /**
     * Get video metadata without processing frames.
     *
     * @param videoFile the video file
     * @return video metadata (width, height, fps, duration, etc.)
     * @throws IOException if the video cannot be read
     */
    public VideoFrameExtractor.VideoMetadata getVideoMetadata(File videoFile) throws IOException {
        VideoFrameExtractor extractor = VideoFrameExtractor.builder().build();
        return extractor.getMetadata(videoFile);
    }

    // =========================================================================
    // Factory methods
    // =========================================================================

    /**
     * Load a video VLM from a directory containing SDZ model files.
     *
     * @param modelDir the model directory
     * @return the video VLM
     * @throws IOException if loading fails
     */
    public static VideoVisionLanguageModel fromDirectory(File modelDir) throws IOException {
        return fromDirectory(modelDir, null);
    }

    /**
     * Load a video VLM from a directory with a specific video preprocessor.
     *
     * @param modelDir the model directory
     * @param videoPreprocessor the video preprocessor (null for default)
     * @return the video VLM
     * @throws IOException if loading fails
     */
    public static VideoVisionLanguageModel fromDirectory(File modelDir,
                                                          VideoPreprocessor videoPreprocessor) throws IOException {
        VisionLanguageModel vlm = VisionLanguageModel.fromDirectory(modelDir);
        return VideoVisionLanguageModel.builder()
                .vlm(vlm)
                .videoPreprocessor(videoPreprocessor)
                .build();
    }

    /**
     * Create a video VLM wrapping an existing VisionLanguageModel.
     *
     * @param vlm the base vision-language model
     * @return the video VLM with default preprocessing
     */
    public static VideoVisionLanguageModel fromVLM(VisionLanguageModel vlm) {
        return VideoVisionLanguageModel.builder()
                .vlm(vlm)
                .build();
    }

    @Override
    public void close() {
        if (vlm != null) {
            vlm.close();
        }
    }
}
