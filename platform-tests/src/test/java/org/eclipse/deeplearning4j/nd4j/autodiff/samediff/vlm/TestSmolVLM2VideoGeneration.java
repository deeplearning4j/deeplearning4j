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
import org.eclipse.deeplearning4j.vlm.model.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.PixelShuffleProjector;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.VideoFrameSampler;
import org.eclipse.deeplearning4j.vlm.preprocessing.VideoPreprocessor;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.ggml.architecture.ArchitectureRegistry;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for the SmolVLM2 video VLM pipeline.
 *
 * Covers architecture registration, video frame sampling, frame preprocessing,
 * pixel shuffle projection, embedding merging, SafeTensors registration, and
 * image prompt construction.
 *
 * All tests use synthetic data — no real model weights are required.
 *
 * @author Adam Gibson
 */
@Slf4j
@DisplayName("SmolVLM2 Video Generation Tests")
public class TestSmolVLM2VideoGeneration extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    // =========================================================================
    // 1. Architecture registry — GGML ArchitectureRegistry
    // =========================================================================

    @Test
    @DisplayName("testSmolVLM2ArchitectureRegistration - verify SmolVLM2Architecture in ArchitectureRegistry")
    public void testSmolVLM2ArchitectureRegistration() {
        assertTrue(ArchitectureRegistry.hasArchitecture("smolvlm2"),
                "ArchitectureRegistry must have 'smolvlm2' registered");
        assertNotNull(ArchitectureRegistry.getArchitecture("smolvlm2"),
                "getArchitecture('smolvlm2') must return non-null");

        // Check supported variants
        assertTrue(ArchitectureRegistry.hasArchitecture("smolvlm"),
                "ArchitectureRegistry must have variant 'smolvlm'");
        assertTrue(ArchitectureRegistry.hasArchitecture("smolvlm2-video"),
                "ArchitectureRegistry must have variant 'smolvlm2-video'");
        assertTrue(ArchitectureRegistry.hasArchitecture("idefics3"),
                "ArchitectureRegistry must have variant 'idefics3'");
    }

    // =========================================================================
    // 2. VideoFrameSampler defaults for SmolVLM2
    // =========================================================================

    @Test
    @DisplayName("testVideoFrameSamplerDefaults - forSmolVLM2 returns UNIFORM strategy with 16 maxFrames")
    public void testVideoFrameSamplerDefaults() {
        VideoFrameSampler sampler = VideoFrameSampler.forSmolVLM2();
        assertNotNull(sampler, "forSmolVLM2() must return a non-null sampler");
        assertEquals(VideoFrameSampler.Strategy.UNIFORM, sampler.getStrategy(),
                "SmolVLM2 sampler strategy must be UNIFORM");
        assertEquals(16, sampler.getMaxFrames(),
                "SmolVLM2 sampler maxFrames must be 16");
    }

    // =========================================================================
    // 3. VideoPreprocessor for SmolVLM2 — synthetic frames
    // =========================================================================

    @Test
    @DisplayName("testVideoPreprocessorForSmolVLM2 - 4 synthetic 384x384 frames produce [1,4,3,384,384]")
    public void testVideoPreprocessorForSmolVLM2() {
        VideoPreprocessor preprocessor = VideoPreprocessor.forSmolVLM2();
        assertNotNull(preprocessor, "forSmolVLM2() must return a non-null preprocessor");

        // Create 4 synthetic 384x384 BGR frames, each filled with a distinct solid color
        Color[] colors = {Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW};
        List<BufferedImage> frames = new ArrayList<>(4);
        for (Color c : colors) {
            BufferedImage img = new BufferedImage(384, 384, BufferedImage.TYPE_3BYTE_BGR);
            Graphics2D g2d = img.createGraphics();
            g2d.setColor(c);
            g2d.fillRect(0, 0, 384, 384);
            g2d.dispose();
            frames.add(img);
        }

        INDArray result = preprocessor.preprocessFrames(frames);
        assertNotNull(result, "preprocessFrames must return a non-null tensor");

        long[] shape = result.shape();
        assertEquals(5, shape.length, "Output must be 5-dimensional [1, numFrames, 3, H, W]");
        assertEquals(1, shape[0], "Batch dimension must be 1");
        assertEquals(4, shape[1], "Frame dimension must equal number of input frames (4)");
        assertEquals(3, shape[2], "Channel dimension must be 3");
        assertEquals(384, shape[3], "Height must be 384");
        assertEquals(384, shape[4], "Width must be 384");
    }

    // =========================================================================
    // 4. PixelShuffleProjector for SmolVLM2
    // =========================================================================

    @Test
    @DisplayName("testPixelShuffleProjector - [1,729,1152] compressed to [1,81,10368] by forSmolVLM2()")
    public void testPixelShuffleProjector() {
        PixelShuffleProjector projector = PixelShuffleProjector.forSmolVLM2();
        assertNotNull(projector, "forSmolVLM2() must return a non-null projector");

        // SigLIP-so400m-patch14-384: 384/14 = 27 patches per side, 27*27 = 729 tokens, hidden=1152
        int patchGridH = 27;
        int patchGridW = 27;
        int seqLen = patchGridH * patchGridW; // 729
        int hidden = 1152;

        INDArray visionTokens = Nd4j.rand(DataType.FLOAT, 1, seqLen, hidden);

        INDArray compressed = projector.compress(visionTokens, patchGridH, patchGridW);
        assertNotNull(compressed, "compress() must return a non-null result");

        long[] shape = compressed.shape();
        assertEquals(3, shape.length, "Compressed output must be 3-dimensional [1, reducedSeq, newHidden]");
        assertEquals(1, shape[0], "Batch dimension must be 1");

        // shuffleFactor=3 → 729 / (3*3) = 81 tokens
        long expectedSeqLen = seqLen / (3 * 3);
        assertEquals(expectedSeqLen, shape[1],
                "Reduced sequence length must be 729/9 = 81 tokens");

        log.info("PixelShuffleProjector compressed shape: {}", Arrays.toString(shape));
    }

    // =========================================================================
    // 5. EmbeddingMerger — video token merging
    // =========================================================================

    @Test
    @DisplayName("testEmbeddingMergerVideoTokens - 12 video token slots replaced by 3 frames x 4 tokens")
    public void testEmbeddingMergerVideoTokens() {
        int videoTokenId = 49190;
        int seqLen = 20;
        int hiddenDim = 512;
        int tokensPerFrame = 4;
        int numFrames = 3;
        int totalVisionTokens = numFrames * tokensPerFrame; // 12

        // Text embeddings [1, 20, 512]
        INDArray textEmbeddings = Nd4j.rand(DataType.FLOAT, 1, seqLen, hiddenDim);

        // 3 frame embeddings, each [1, 4, 512]
        List<INDArray> frameEmbeddings = new ArrayList<>(numFrames);
        for (int i = 0; i < numFrames; i++) {
            frameEmbeddings.add(Nd4j.rand(DataType.FLOAT, 1, tokensPerFrame, hiddenDim));
        }

        // Token IDs: 12 video token positions, 8 regular text tokens
        int[] tokenIds = new int[seqLen];
        int videoCount = 0;
        for (int i = 0; i < seqLen; i++) {
            if (videoCount < totalVisionTokens) {
                tokenIds[i] = videoTokenId;
                videoCount++;
            } else {
                tokenIds[i] = 100 + i; // arbitrary non-video token
            }
        }

        INDArray merged = EmbeddingMerger.mergeVideoEmbeddings(
                textEmbeddings, frameEmbeddings, tokenIds, videoTokenId);

        assertNotNull(merged, "mergeVideoEmbeddings must return a non-null result");

        long[] shape = merged.shape();
        assertEquals(3, shape.length, "Merged output must be 3-dimensional [1, newSeqLen, hiddenDim]");
        assertEquals(1, shape[0], "Batch dimension must be 1");
        // videoSlots == totalVisionTokens (12==12): newSeqLen = 20 - 12 + 12 = 20
        assertEquals(seqLen, shape[1],
                "Merged sequence length must be 20 (8 text + 12 vision tokens)");
        assertEquals(hiddenDim, shape[2], "Hidden dimension must be preserved");

        log.info("EmbeddingMerger merged shape: {}", Arrays.toString(shape));
    }

    // =========================================================================
    // 6. SafeTensors registry — SmolVLM2SafeTensorsBuilder
    // =========================================================================

    @Test
    @DisplayName("testSmolVLM2SafeTensorsRegistration - verify SmolVLM2SafeTensorsBuilder in registry")
    public void testSmolVLM2SafeTensorsRegistration() {
        assertTrue(SafeTensorsArchitectureRegistry.has("smolvlm2"),
                "SafeTensorsArchitectureRegistry must have 'smolvlm2' registered");

        // Verify canHandle with a config map containing the standard HuggingFace architecture field
        Map<String, Object> config = new HashMap<>();
        config.put("architectures", List.of("SmolVLM2ForConditionalGeneration"));

        var builder = SafeTensorsArchitectureRegistry.get("smolvlm2");
        assertNotNull(builder, "get('smolvlm2') must return a non-null builder");
        assertTrue(builder.canHandle(config),
                "SmolVLM2SafeTensorsBuilder must canHandle config with 'SmolVLM2ForConditionalGeneration'");
    }

    // =========================================================================
    // 7. ImagePromptBuilder — video token prompt construction
    // =========================================================================

    @Test
    @DisplayName("testImagePromptBuilderVideoToken - rows=0,cols=0,seqLen=5 produces global-img with 5 image tokens")
    public void testImagePromptBuilderVideoToken() {
        // rows=0, cols=0 → no tiling, uses global-img path
        String prompt = ImagePromptBuilder.buildImagePromptString(0, 0, 5);
        assertNotNull(prompt, "buildImagePromptString must return a non-null string");
        assertFalse(prompt.isEmpty(), "buildImagePromptString must return a non-empty string");

        assertTrue(prompt.contains("<global-img>"),
                "Prompt must contain '<global-img>' token");

        // Count occurrences of "<image>"
        int count = 0;
        int idx = 0;
        while ((idx = prompt.indexOf("<image>", idx)) != -1) {
            count++;
            idx += "<image>".length();
        }
        assertEquals(5, count,
                "Prompt must contain exactly 5 '<image>' tokens for imageSeqLen=5");

        log.info("ImagePromptBuilder output: {}", prompt);
    }
}
