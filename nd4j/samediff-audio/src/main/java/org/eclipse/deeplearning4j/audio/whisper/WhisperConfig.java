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

package org.eclipse.deeplearning4j.audio.whisper;

import lombok.Builder;
import lombok.Data;

/**
 * Configuration for Whisper model variants.
 * Contains hyperparameters for all Whisper model sizes (tiny through large-v3/turbo).
 */
@Data
@Builder
public class WhisperConfig {

    private String modelName;
    private int numEncoderLayers;
    private int numDecoderLayers;
    private int hiddenSize;
    private int numAttentionHeads;
    private int numMelBins;

    @Builder.Default
    private int maxSourcePositions = 1500;

    @Builder.Default
    private int maxTargetPositions = 448;

    @Builder.Default
    private int vocabSize = 51865;

    @Builder.Default
    private int sampleRate = 16000;

    @Builder.Default
    private int nFft = 400;

    @Builder.Default
    private int hopLength = 160;

    @Builder.Default
    private int chunkLengthSeconds = 30;

    public static WhisperConfig tiny() {
        return WhisperConfig.builder()
                .modelName("tiny")
                .numEncoderLayers(4).numDecoderLayers(4)
                .hiddenSize(384).numAttentionHeads(6)
                .numMelBins(80).build();
    }

    public static WhisperConfig base() {
        return WhisperConfig.builder()
                .modelName("base")
                .numEncoderLayers(6).numDecoderLayers(6)
                .hiddenSize(512).numAttentionHeads(8)
                .numMelBins(80).build();
    }

    public static WhisperConfig small() {
        return WhisperConfig.builder()
                .modelName("small")
                .numEncoderLayers(12).numDecoderLayers(12)
                .hiddenSize(768).numAttentionHeads(12)
                .numMelBins(80).build();
    }

    public static WhisperConfig medium() {
        return WhisperConfig.builder()
                .modelName("medium")
                .numEncoderLayers(24).numDecoderLayers(24)
                .hiddenSize(1024).numAttentionHeads(16)
                .numMelBins(80).build();
    }

    public static WhisperConfig largeV2() {
        return WhisperConfig.builder()
                .modelName("large-v2")
                .numEncoderLayers(32).numDecoderLayers(32)
                .hiddenSize(1280).numAttentionHeads(20)
                .numMelBins(80).build();
    }

    public static WhisperConfig largeV3() {
        return WhisperConfig.builder()
                .modelName("large-v3")
                .numEncoderLayers(32).numDecoderLayers(32)
                .hiddenSize(1280).numAttentionHeads(20)
                .numMelBins(128).build();
    }

    public static WhisperConfig turbo() {
        return WhisperConfig.builder()
                .modelName("turbo")
                .numEncoderLayers(32).numDecoderLayers(4)
                .hiddenSize(1280).numAttentionHeads(20)
                .numMelBins(128).build();
    }

    /**
     * Get the number of audio frames for a full 30-second chunk.
     */
    public int getNumFrames() {
        return (sampleRate * chunkLengthSeconds) / hopLength;
    }

    /**
     * Get the number of samples for a full chunk.
     */
    public int getChunkLengthSamples() {
        return sampleRate * chunkLengthSeconds;
    }
}
