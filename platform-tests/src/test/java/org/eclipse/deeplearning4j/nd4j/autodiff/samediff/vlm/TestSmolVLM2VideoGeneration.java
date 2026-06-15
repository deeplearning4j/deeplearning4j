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
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.preprocessing.VideoFrameSampler;
import org.eclipse.deeplearning4j.vlm.preprocessing.VideoPreprocessor;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.eclipse.deeplearning4j.llm.config.PreprocessorConfig;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for the SmolVLM2 video VLM pipeline.
 *
 * Downloads SmolVLM2-256M-Video-Instruct Q8_0 GGUF (~175MB), imports it
 * as a SameDiff graph, and validates the model structure and video pipeline
 * components.
 *
 * Run with:
 * <pre>
 * cd platform-tests && mvn test \
 *   -Dtest=TestSmolVLM2VideoGeneration \
 *   -Dbackend.artifactId=nd4j-cuda-12.9
 * </pre>
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
@DisplayName("SmolVLM2 Video Generation Tests")
public class TestSmolVLM2VideoGeneration {

    private static SameDiff model;
    private static File modelFile;
    private static boolean modelLoaded = false;

    @BeforeAll
    public static void setup() throws Exception {
        log.info("Downloading SmolVLM2-256M-Video-Instruct GGUF...");
        long t0 = System.currentTimeMillis();
        VLMModelDownloader.DownloadResult result = VLMModelDownloader.download(
                VLMModelDownloader.VLMModel.SMOLVLM2_256M_VIDEO_GGUF);
        modelFile = result.getModelFile();
        long downloadMs = System.currentTimeMillis() - t0;
        log.info("Download: {}ms, file: {}, size: {} bytes",
                downloadMs, modelFile.getAbsolutePath(), result.getFileSizeBytes());

        log.info("Importing GGUF -> SameDiff...");
        long t1 = System.currentTimeMillis();
        model = GGMLModelImport.importModel(modelFile.getAbsolutePath(),
                ConversionOptions.forInference());
        long importMs = System.currentTimeMillis() - t1;
        log.info("Import: {}ms, ops: {}, variables: {}",
                importMs, model.ops().length, model.variables().size());
        modelLoaded = true;
    }

    @AfterAll
    public static void teardown() {
        if (model != null) {
            try {
                model.close();
            } catch (Exception e) {
                log.warn("Error closing model: {}", e.getMessage());
            }
            model = null;
        }
    }

    @Test
    @Order(1)
    @DisplayName("Model imports successfully with non-trivial graph")
    public void testModelImport() {
        assertTrue(modelLoaded, "Model must load successfully");
        assertNotNull(model, "SameDiff model must not be null");
        assertTrue(model.ops().length > 10,
                "Model must have a non-trivial number of ops, got: " + model.ops().length);
        assertTrue(model.variables().size() > 10,
                "Model must have a non-trivial number of variables, got: " + model.variables().size());
        log.info("Model graph: {} ops, {} variables", model.ops().length, model.variables().size());
    }

    @Test
    @Order(2)
    @DisplayName("Model has expected LLM decoder variables (SmolLM2 architecture)")
    public void testModelHasDecoderVariables() {
        assertTrue(modelLoaded, "Model must be loaded");

        // Log variable names first so we can see them even on failure
        log.info("Sample variable names (first 30):");
        model.variables().stream()
                .limit(30)
                .forEach(v -> log.info("  {}: shape={}", v.name(),
                        v.getArr() != null ? Arrays.toString(v.getArr().shape()) : "null"));

        // SmolVLM2 uses SmolLM2 (VLlama3) as its LLM decoder
        // GGUF tensor names may be raw (blk.0.attn_q) or mapped (model.layers.0.self_attn)
        boolean hasAttnWeights = model.variables().stream()
                .anyMatch(v -> (v.name().contains("attn") || v.name().contains("self_attn"))
                        && (v.name().contains("blk.0") || v.name().contains("layers.0")
                            || v.name().contains("layer.0")));
        assertTrue(hasAttnWeights,
                "Model must contain layer-0 attention variables (SmolLM2 decoder)");

        boolean hasEmbedding = model.variables().stream()
                .anyMatch(v -> v.name().contains("token_embd") || v.name().contains("embed"));
        assertTrue(hasEmbedding,
                "Model must contain token embedding variables");
    }

    @Test
    @Order(3)
    @DisplayName("VideoFrameSampler builder produces correct config")
    public void testVideoFrameSamplerBuilder() {
        VideoFrameSampler sampler = VideoFrameSampler.builder()
                .strategy(VideoFrameSampler.Strategy.UNIFORM)
                .maxFrames(16)
                .build();
        assertEquals(VideoFrameSampler.Strategy.UNIFORM, sampler.getStrategy());
        assertEquals(16, sampler.getMaxFrames());

        // sampleFromFrames should reduce 30 frames to 16
        List<BufferedImage> allFrames = new ArrayList<>();
        for (int i = 0; i < 30; i++) {
            allFrames.add(new BufferedImage(100, 100, BufferedImage.TYPE_3BYTE_BGR));
        }
        List<BufferedImage> sampled = sampler.sampleFromFrames(allFrames);
        assertEquals(16, sampled.size(),
                "sampleFromFrames must reduce 30 frames to maxFrames=16");
    }

    @Test
    @Order(4)
    @DisplayName("VideoPreprocessor produces correct 5D tensor shape")
    public void testVideoPreprocessor() {
        PreprocessorConfig sigLipConfig = PreprocessorConfig.forViT();
        sigLipConfig.setSize(new PreprocessorConfig.ImageSize(384, 384));

        VideoPreprocessor preprocessor = VideoPreprocessor.builder()
                .imagePreprocessor(VLMImagePreprocessor.fromConfig(sigLipConfig))
                .targetHeight(384)
                .targetWidth(384)
                .build();

        List<BufferedImage> frames = new ArrayList<>();
        Color[] colors = {Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW};
        for (Color c : colors) {
            BufferedImage img = new BufferedImage(384, 384, BufferedImage.TYPE_3BYTE_BGR);
            Graphics2D g = img.createGraphics();
            g.setColor(c);
            g.fillRect(0, 0, 384, 384);
            g.dispose();
            frames.add(img);
        }

        INDArray result = preprocessor.preprocessFrames(frames);
        assertNotNull(result);

        long[] shape = result.shape();
        assertEquals(5, shape.length, "Must be 5D [batch, frames, channels, H, W]");
        assertEquals(1, shape[0]);
        assertEquals(4, shape[1]);
        assertEquals(3, shape[2]);
        assertEquals(384, shape[3]);
        assertEquals(384, shape[4]);
    }

    @Test
    @Order(5)
    @DisplayName("EmbeddingMerger scatters vision tokens into text sequence via native op")
    public void testEmbeddingMergerNative() {
        int imageTokenId = 49190;
        int seqLen = 20;
        int hiddenDim = 64;
        int numVisionTokens = 8;

        INDArray textEmb = Nd4j.ones(DataType.FLOAT, 1, seqLen, hiddenDim);
        INDArray visionEmb = Nd4j.ones(DataType.FLOAT, 1, numVisionTokens, hiddenDim).mul(99.0);

        int[] tokenIds = new int[seqLen];
        for (int i = 0; i < numVisionTokens; i++) tokenIds[i] = imageTokenId;
        for (int i = numVisionTokens; i < seqLen; i++) tokenIds[i] = 100 + i;

        INDArray merged = EmbeddingMerger.mergeEmbeddings(textEmb, visionEmb, tokenIds, imageTokenId);

        assertNotNull(merged);
        long[] shape = merged.shape();
        assertEquals(1, shape[0]);
        assertEquals(seqLen, shape[1]);
        assertEquals(hiddenDim, shape[2]);

        float visionVal = merged.getFloat(0, 0, 0);
        float textVal = merged.getFloat(0, seqLen - 1, 0);
        assertEquals(99.0f, visionVal, 0.1f,
                "Vision token position must contain vision embedding value (99.0)");
        assertEquals(1.0f, textVal, 0.1f,
                "Text token position must contain original text embedding value (1.0)");
    }

    @Test
    @Order(6)
    @DisplayName("EmbeddingMerger handles multi-frame video token merging")
    public void testEmbeddingMergerVideoFrames() {
        int videoTokenId = 49190;
        int seqLen = 24;
        int hiddenDim = 64;
        int tokensPerFrame = 4;
        int numFrames = 3;
        int totalVision = numFrames * tokensPerFrame; // 12

        INDArray textEmb = Nd4j.zeros(DataType.FLOAT, 1, seqLen, hiddenDim);

        List<INDArray> frameEmbeddings = new ArrayList<>();
        for (int f = 0; f < numFrames; f++) {
            frameEmbeddings.add(
                    Nd4j.ones(DataType.FLOAT, 1, tokensPerFrame, hiddenDim).mul((f + 1) * 10.0));
        }

        int[] tokenIds = new int[seqLen];
        for (int i = 0; i < totalVision; i++) tokenIds[i] = videoTokenId;
        for (int i = totalVision; i < seqLen; i++) tokenIds[i] = 500 + i;

        INDArray merged = EmbeddingMerger.mergeVideoEmbeddings(
                textEmb, frameEmbeddings, tokenIds, videoTokenId);

        assertNotNull(merged);
        assertEquals(seqLen, merged.size(1));

        float frame0Val = merged.getFloat(0, 0, 0);
        float frame1Val = merged.getFloat(0, 4, 0);
        float frame2Val = merged.getFloat(0, 8, 0);
        float textVal = merged.getFloat(0, 12, 0);

        assertEquals(10.0f, frame0Val, 0.1f, "Frame 0 position must have value 10.0");
        assertEquals(20.0f, frame1Val, 0.1f, "Frame 1 position must have value 20.0");
        assertEquals(30.0f, frame2Val, 0.1f, "Frame 2 position must have value 30.0");
        assertEquals(0.0f, textVal, 0.1f, "Text position must have original value 0.0");
    }
}
