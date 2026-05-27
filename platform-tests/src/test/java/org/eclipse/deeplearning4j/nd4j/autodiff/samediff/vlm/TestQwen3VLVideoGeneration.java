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
import org.eclipse.deeplearning4j.safetensors.architecture.SafeTensorsArchitectureRegistry;
import org.eclipse.deeplearning4j.vlm.model.DeepStackMerger;
import org.eclipse.deeplearning4j.vlm.model.TemporalPatchEmbed;
import org.eclipse.deeplearning4j.vlm.preprocessing.VideoFrameSampler;
import org.eclipse.deeplearning4j.vlm.preprocessing.VideoPreprocessor;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.architecture.ArchitectureRegistry;
import org.nd4j.ggml.format.MultimodalGGUFLoader;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedMRoPE;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for the Qwen3-VL video VLM pipeline.
 *
 * Covers architecture registration, video frame sampling, temporal-aligned frame
 * preprocessing, FusedMRoPE op creation, DeepStack merger construction,
 * TemporalPatchEmbed configuration, SafeTensors architecture registration, and
 * MultimodalGGUFLoader detection.
 *
 * All tests use synthetic data — no real model weights are required.
 *
 * @author Adam Gibson
 */
@Slf4j
@DisplayName("Qwen3-VL Video Generation Tests")
public class TestQwen3VLVideoGeneration extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    // =========================================================================
    // 1. Architecture registry — GGML ArchitectureRegistry
    // =========================================================================

    @Test
    @DisplayName("testQwen3VLArchitectureRegistration - verify Qwen3VLArchitecture in ArchitectureRegistry")
    public void testQwen3VLArchitectureRegistration() {
        assertTrue(ArchitectureRegistry.hasArchitecture("qwen3vl"),
                "ArchitectureRegistry must have 'qwen3vl' registered");
        assertNotNull(ArchitectureRegistry.getArchitecture("qwen3vl"),
                "getArchitecture('qwen3vl') must return non-null");

        // Check supported variants
        assertTrue(ArchitectureRegistry.hasArchitecture("qwen3_vl"),
                "ArchitectureRegistry must have variant 'qwen3_vl'");
        assertTrue(ArchitectureRegistry.hasArchitecture("qwen2_vl"),
                "ArchitectureRegistry must have variant 'qwen2_vl'");
        assertTrue(ArchitectureRegistry.hasArchitecture("qwen2.5_vl"),
                "ArchitectureRegistry must have variant 'qwen2.5_vl'");
    }

    // =========================================================================
    // 2. VideoFrameSampler defaults for Qwen3-VL
    // =========================================================================

    @Test
    @DisplayName("testVideoFrameSamplerQwen3VL - forQwen3VL returns FIXED_FPS strategy with 64 maxFrames")
    public void testVideoFrameSamplerQwen3VL() {
        VideoFrameSampler sampler = VideoFrameSampler.forQwen3VL();
        assertNotNull(sampler, "forQwen3VL() must return a non-null sampler");
        assertEquals(VideoFrameSampler.Strategy.FIXED_FPS, sampler.getStrategy(),
                "Qwen3-VL sampler strategy must be FIXED_FPS");
        assertEquals(64, sampler.getMaxFrames(),
                "Qwen3-VL sampler maxFrames must be 64");
    }

    // =========================================================================
    // 3. VideoPreprocessor for Qwen3-VL — synthetic frames with temporal padding
    // =========================================================================

    @Test
    @DisplayName("testVideoPreprocessorQwen3VL - 3 synthetic 448x448 frames padded to 4 produce [1,4,3,384,384]")
    public void testVideoPreprocessorQwen3VL() {
        VideoPreprocessor preprocessor = VideoPreprocessor.forQwen3VL();
        assertNotNull(preprocessor, "forQwen3VL() must return a non-null preprocessor");

        // Create 3 synthetic 448x448 BGR frames filled with distinct solid colors
        Color[] colors = {Color.RED, Color.GREEN, Color.BLUE};
        List<BufferedImage> frames = new ArrayList<>(3);
        for (Color c : colors) {
            BufferedImage img = new BufferedImage(448, 448, BufferedImage.TYPE_3BYTE_BGR);
            Graphics2D g2d = img.createGraphics();
            g2d.setColor(c);
            g2d.fillRect(0, 0, 448, 448);
            g2d.dispose();
            frames.add(img);
        }

        // preprocessFramesTemporalAligned pads 3 frames to 4 (next multiple of temporalPatchSize=2)
        INDArray result = preprocessor.preprocessFramesTemporalAligned(frames);
        assertNotNull(result, "preprocessFramesTemporalAligned must return a non-null tensor");

        long[] shape = result.shape();
        assertEquals(5, shape.length, "Output tensor must be 5D");
        assertEquals(1, shape[0], "Batch dimension must be 1");
        assertEquals(4, shape[1],
                "Frame dimension must be 4 (3 frames padded to next multiple of temporalPatchSize=2)");
        assertEquals(3, shape[2], "Channel dimension must be 3");
        // Qwen3-VL preprocessor targets 384x384 (Math.max(targetHeight=384, targetWidth=384))
        assertEquals(384, shape[3], "Height dimension must be 384 (Qwen3-VL target size)");
        assertEquals(384, shape[4], "Width dimension must be 384 (Qwen3-VL target size)");
    }

    // =========================================================================
    // 4. FusedMRoPE op creation for Qwen3-VL
    // =========================================================================

    @Test
    @DisplayName("testFusedMRoPEOpCreation - forQwen3VL produces op with name fused_mrope and sections [24,20,20]")
    public void testFusedMRoPEOpCreation() {
        SameDiff sd = SameDiff.create();

        // input: [1, 10, 32, 64] FLOAT
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 10, 32, 64);

        // position tensors: [1, 10] INT64
        SDVariable posT = sd.placeHolder("pos_t", DataType.INT64, 1, 10);
        SDVariable posH = sd.placeHolder("pos_h", DataType.INT64, 1, 10);
        SDVariable posW = sd.placeHolder("pos_w", DataType.INT64, 1, 10);

        FusedMRoPE op = FusedMRoPE.forQwen3VL(sd, input, posT, posH, posW);

        assertNotNull(op, "FusedMRoPE.forQwen3VL must return a non-null op");
        assertEquals("fused_mrope", op.opName(),
                "Op name must be 'fused_mrope'");
        assertEquals(24, op.getSectionT(),
                "Temporal section (sectionT) must be 24");
        assertEquals(20, op.getSectionH(),
                "Height section (sectionH) must be 20");
        assertEquals(20, op.getSectionW(),
                "Width section (sectionW) must be 20");
        assertTrue(op.isInterleaved(),
                "Qwen3-VL FusedMRoPE must use interleaved mode (interleaved=1)");
    }

    // =========================================================================
    // 5. DeepStackMerger creation for Qwen3-VL
    // =========================================================================

    @Test
    @DisplayName("testDeepStackMergerCreation - forQwen3VL returns merger with visualIndexes [8,16,24]")
    public void testDeepStackMergerCreation() {
        DeepStackMerger merger = DeepStackMerger.forQwen3VL();
        assertNotNull(merger, "DeepStackMerger.forQwen3VL() must return a non-null merger");

        int[] visualIndexes = merger.getVisualIndexes();
        assertNotNull(visualIndexes, "visualIndexes must not be null");
        assertArrayEquals(new int[]{8, 16, 24}, visualIndexes,
                "Qwen3-VL DeepStack extraction layers must be [8, 16, 24]");

        assertEquals(1152, merger.getVisionHiddenSize(),
                "Qwen3-VL visionHiddenSize must be 1152");
        assertEquals(4096, merger.getLlmHiddenSize(),
                "Qwen3-VL llmHiddenSize must be 4096");
    }

    // =========================================================================
    // 6. TemporalPatchEmbed creation for Qwen3-VL
    // =========================================================================

    @Test
    @DisplayName("testTemporalPatchEmbedCreation - forQwen3VL returns temporalPatchSize=2 spatialPatchSize=16")
    public void testTemporalPatchEmbedCreation() {
        TemporalPatchEmbed embed = TemporalPatchEmbed.forQwen3VL();
        assertNotNull(embed, "TemporalPatchEmbed.forQwen3VL() must return a non-null embed");
        assertEquals(2, embed.getTemporalPatchSize(),
                "Qwen3-VL temporal_patch_size must be 2");
        assertEquals(16, embed.getSpatialPatchSize(),
                "Qwen3-VL spatial_patch_size must be 16");
    }

    // =========================================================================
    // 7. SafeTensors architecture registry for Qwen3-VL
    // =========================================================================

    @Test
    @DisplayName("testQwen3VLSafeTensorsRegistration - Qwen3VLSafeTensorsBuilder registered and canHandle config")
    public void testQwen3VLSafeTensorsRegistration() {
        assertTrue(SafeTensorsArchitectureRegistry.has("qwen3_vl"),
                "SafeTensorsArchitectureRegistry must have 'qwen3_vl' registered");

        var builder = SafeTensorsArchitectureRegistry.get("qwen3_vl");
        assertNotNull(builder, "get('qwen3_vl') must return a non-null builder");

        Map<String, Object> config = Map.of(
                "architectures", List.of("Qwen3VLForConditionalGeneration")
        );
        assertTrue(builder.canHandle(config),
                "Qwen3VLSafeTensorsBuilder must canHandle a config with 'Qwen3VLForConditionalGeneration'");
    }

    // =========================================================================
    // 8. MultimodalGGUFLoader — mmproj detection from an empty directory
    // =========================================================================

    @Test
    @DisplayName("testMultimodalGGUFLoaderDetection - loadFromDirectory throws IOException for empty directory")
    public void testMultimodalGGUFLoaderDetection() throws IOException {
        File tempDir = Files.createTempDirectory("gguf-test-").toFile();
        try {
            // An empty directory has no .gguf files; loadFromDirectory must throw IOException
            assertThrows(IOException.class,
                    () -> MultimodalGGUFLoader.loadFromDirectory(tempDir),
                    "loadFromDirectory should throw IOException when no .gguf files are present");
        } finally {
            tempDir.delete();
        }
    }
}
