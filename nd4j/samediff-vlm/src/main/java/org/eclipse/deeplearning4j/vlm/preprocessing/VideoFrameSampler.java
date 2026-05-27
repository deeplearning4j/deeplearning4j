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

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Configurable frame sampling strategies for video VLMs.
 *
 * <p>Different video VLMs require different frame sampling approaches:</p>
 * <ul>
 *   <li>{@link Strategy#UNIFORM} - Evenly spaced frames (SmolVLM2 default)</li>
 *   <li>{@link Strategy#FIXED_FPS} - Extract at a specific frame rate (Qwen-VL dynamic FPS)</li>
 *   <li>{@link Strategy#KEYFRAME} - Only I-frames/keyframes (efficient for long videos)</li>
 * </ul>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * VideoFrameSampler sampler = VideoFrameSampler.builder()
 *     .strategy(VideoFrameSampler.Strategy.UNIFORM)
 *     .maxFrames(32)
 *     .build();
 *
 * List<BufferedImage> frames = sampler.sample(new File("video.mp4"));
 * }</pre>
 *
 * @see VideoFrameExtractor
 * @see VideoPreprocessor
 */
@Slf4j
@Builder
public class VideoFrameSampler {

    public enum Strategy {
        UNIFORM,
        FIXED_FPS,
        KEYFRAME
    }

    @Getter
    @Builder.Default
    private Strategy strategy = Strategy.UNIFORM;

    @Getter
    @Builder.Default
    private int maxFrames = 32;

    @Getter
    @Builder.Default
    private double targetFPS = 1.0;

    @Getter
    @Builder.Default
    private int minFrameInterval = 1;

    /**
     * Sample frames from a video file using the configured strategy.
     *
     * @param videoFile the video file
     * @return sampled frames
     * @throws IOException if the video cannot be read
     */
    public List<BufferedImage> sample(File videoFile) throws IOException {
        VideoFrameExtractor extractor = VideoFrameExtractor.builder().build();

        List<BufferedImage> frames;
        switch (strategy) {
            case UNIFORM:
                frames = extractor.extractFrames(videoFile, maxFrames);
                break;
            case FIXED_FPS:
                frames = extractor.extractAtFPS(videoFile, targetFPS, maxFrames);
                break;
            case KEYFRAME:
                frames = extractor.extractKeyframes(videoFile, maxFrames);
                break;
            default:
                throw new IllegalStateException("Unknown sampling strategy: " + strategy);
        }

        if (minFrameInterval > 1 && frames.size() > maxFrames / minFrameInterval) {
            List<BufferedImage> thinned = new ArrayList<>();
            for (int i = 0; i < frames.size(); i += minFrameInterval) {
                thinned.add(frames.get(i));
                if (thinned.size() >= maxFrames) break;
            }
            frames = thinned;
        }

        log.info("Sampled {} frames using {} strategy from {}",
                frames.size(), strategy, videoFile.getName());
        return frames;
    }

    /**
     * Sample frames from a pre-extracted frame list (e.g., for re-sampling).
     *
     * @param allFrames all available frames
     * @return sampled subset
     */
    public List<BufferedImage> sampleFromFrames(List<BufferedImage> allFrames) {
        if (allFrames.size() <= maxFrames) {
            return new ArrayList<>(allFrames);
        }

        List<BufferedImage> sampled = new ArrayList<>(maxFrames);
        double step = (double) allFrames.size() / maxFrames;

        for (int i = 0; i < maxFrames; i++) {
            int index = Math.min((int) Math.round(i * step), allFrames.size() - 1);
            sampled.add(allFrames.get(index));
        }

        return sampled;
    }

    /**
     * Create a sampler for SmolVLM2 (uniform, moderate frame count).
     */
    public static VideoFrameSampler forSmolVLM2() {
        return VideoFrameSampler.builder()
                .strategy(Strategy.UNIFORM)
                .maxFrames(16)
                .build();
    }

    /**
     * Create a sampler for Qwen3-VL (dynamic FPS with more frames).
     */
    public static VideoFrameSampler forQwen3VL() {
        return VideoFrameSampler.builder()
                .strategy(Strategy.FIXED_FPS)
                .targetFPS(2.0)
                .maxFrames(64)
                .build();
    }

    /**
     * Create a sampler for MiniCPM-V 4.5 (groups of 6 frames).
     */
    public static VideoFrameSampler forMiniCPMV() {
        return VideoFrameSampler.builder()
                .strategy(Strategy.FIXED_FPS)
                .targetFPS(10.0)
                .maxFrames(180)
                .build();
    }
}
