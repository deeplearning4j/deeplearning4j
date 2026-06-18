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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff.vlm;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.vlm.model.projector.TemporalPatchEmbed;
import org.eclipse.deeplearning4j.vlm.preprocessing.VideoFrameSampler;
import org.eclipse.deeplearning4j.vlm.preprocessing.VideoPreprocessor;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.eclipse.deeplearning4j.llm.config.PreprocessorConfig;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedMRoPE;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Qwen3-VL specific components: M-RoPE, DeepStack, temporal patches,
 * and video preprocessing with temporal alignment.
 *
 * All tests use synthetic data — no model download required.
 */
@Slf4j
@DisplayName("Qwen3-VL Video Component Tests")
public class TestQwen3VLVideoGeneration extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    @DisplayName("VideoFrameSampler builder with FIXED_FPS strategy")
    public void testVideoFrameSamplerFixedFps() {
        VideoFrameSampler sampler = VideoFrameSampler.builder()
                .strategy(VideoFrameSampler.Strategy.FIXED_FPS)
                .targetFPS(2.0)
                .maxFrames(64)
                .build();
        assertEquals(VideoFrameSampler.Strategy.FIXED_FPS, sampler.getStrategy());
        assertEquals(64, sampler.getMaxFrames());
        assertEquals(2.0, sampler.getTargetFPS(), 0.01);
    }

    @Test
    @DisplayName("VideoPreprocessor with temporalPatchSize=2 pads odd frame count")
    public void testVideoPreprocessorTemporalAlignment() {
        PreprocessorConfig config = PreprocessorConfig.forViT();
        config.setSize(new PreprocessorConfig.ImageSize(384, 384));

        VideoPreprocessor preprocessor = VideoPreprocessor.builder()
                .imagePreprocessor(VLMImagePreprocessor.fromConfig(config))
                .targetHeight(384)
                .targetWidth(384)
                .temporalPatchSize(2)
                .build();

        // 3 frames should be padded to 4 (next multiple of temporalPatchSize=2)
        Color[] colors = {Color.RED, Color.GREEN, Color.BLUE};
        List<BufferedImage> frames = new ArrayList<>(3);
        for (Color c : colors) {
            BufferedImage img = new BufferedImage(448, 448, BufferedImage.TYPE_3BYTE_BGR);
            Graphics2D g = img.createGraphics();
            g.setColor(c);
            g.fillRect(0, 0, 448, 448);
            g.dispose();
            frames.add(img);
        }

        var result = preprocessor.preprocessFramesTemporalAligned(frames);
        assertNotNull(result);

        long[] shape = result.shape();
        assertEquals(5, shape.length);
        assertEquals(1, shape[0]);
        assertEquals(4, shape[1], "3 frames must be padded to 4 for temporalPatchSize=2");
        assertEquals(3, shape[2]);
    }

    @Test
    @DisplayName("FusedMRoPE op creation with Qwen3-VL defaults")
    public void testFusedMRoPEOpCreation() {
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 10, 32, 64);
        SDVariable posT = sd.placeHolder("pos_t", DataType.INT64, 1, 10);
        SDVariable posH = sd.placeHolder("pos_h", DataType.INT64, 1, 10);
        SDVariable posW = sd.placeHolder("pos_w", DataType.INT64, 1, 10);

        FusedMRoPE op = FusedMRoPE.forQwen3VL(sd, input, posT, posH, posW);

        assertNotNull(op);
        assertEquals("fused_mrope", op.opName());
        assertEquals(24, op.getSectionT());
        assertEquals(20, op.getSectionH());
        assertEquals(20, op.getSectionW());
        assertTrue(op.isInterleaved());
    }

    @Test
    @DisplayName("TemporalPatchEmbed builder with correct patch sizes")
    public void testTemporalPatchEmbedBuilder() {
        TemporalPatchEmbed embed = TemporalPatchEmbed.builder()
                .temporalPatchSize(2)
                .spatialPatchSize(16)
                .channels(3)
                .build();

        assertEquals(2, embed.getTemporalPatchSize());
        assertEquals(16, embed.getSpatialPatchSize());
    }
}
