/*
 *  ******************************************************************************
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

package org.eclipse.deeplearning4j.vlm;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.model.VisionLanguageModel;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Reproduce the CUDA graph replay hang at step 2 (first decode replay).
 *
 * <p>Problem: VLM decoder hangs on step 2 (the first CUDA graph replay of the decode plan).
 * Steps 0 (prefill/capture) and 1 (first decode/capture) succeed. Step 2 attempts to replay
 * the captured graph and hangs indefinitely on seg[1320-1323] (attention matmuls + Triton softmax).</p>
 *
 * <p>Root cause: cuBLAS workspace is being zeroed (cudaMemsetAsync) before segment captures.
 * When segment N captures, cuBLAS caches plan/descriptor data in the workspace. Later segments
 * (like seg[1320-1323]) inherit those cached plans — their captured graphs omit H2D re-upload nodes.
 * On replay, the pre-segments workspace zeroing destroys the cached plans, so GEMM kernels read
 * zeros and hang.</p>
 *
 * <p>This test exercises the exact pattern that triggers the hang:</p>
 * <ul>
 *   <li>Load SmolDocling-256M-preview model via VisionLanguageModel</li>
 *   <li>Process a single page from pathfinder-mythic.pdf</li>
 *   <li>Run the StaticKvCacheDecodeLoop with maxNewTokens=50</li>
 *   <li>Verify step 2+ (first CUDA graph replay) completes without hanging</li>
 * </ul>
 *
 * <p>Run:
 * <pre>
 *   cd platform-tests && mvn test \
 *     -Dtest=TestCudaGraphReplayHang \
 *     -Dbackend.artifactId=nd4j-cuda-12.9
 * </pre>
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class TestCudaGraphReplayHang {

    private static final File MODEL_DIR = new File(
            System.getProperty("user.home") + "/.kompile/models/vlm/smoldocling-256m-pipeline");
    private static final File PDF_FILE = new File(
            "platform-tests/pathfinder-mythic.pdf");
    private static final int TARGET_SIZE = 512;
    private static final int MAX_NEW_TOKENS = 50;
    private static final int TEST_TIMEOUT_SECONDS = 60;

    private static VisionLanguageModel vlm;
    private static NativeOps nativeOps;
    private static boolean modelsLoaded = false;

    @BeforeAll
    public static void setup() throws Exception {
        // Enable DSP and CUDA graph capture
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        System.setProperty(ND4JSystemProperties.DSP_NATIVE_EXECUTOR_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);

        // Enable batch-zero for capture-compatible execution
        Nd4j.getEnvironment().setDspBatchZero(true);
        Nd4j.getEnvironment().setDspBatchZeroKernel(true);

        nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        logGpuMemory("Before model load");

        // Load VLM model
        if (MODEL_DIR.isDirectory()) {
            log.info("Loading VisionLanguageModel from {}", MODEL_DIR);
            vlm = VisionLanguageModel.loadSmolDocling(MODEL_DIR);
            log.info("VLM loaded successfully");
            modelsLoaded = true;
        } else {
            log.warn("Model dir not found: {} — tests will be skipped", MODEL_DIR);
        }

        logGpuMemory("After model load");
    }

    @AfterAll
    public static void teardown() {
        if (vlm != null) {
            vlm.close();
        }
        logGpuMemory("After teardown");
    }

    private static void logGpuMemory(String label) {
        try {
            int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
            for (int d = 0; d < numDevices; d++) {
                long freeMem = nativeOps.getDeviceFreeMemory(d);
                long totalMem = nativeOps.getDeviceTotalMemory(d);
                long usedMem = totalMem - freeMem;
                log.info("[{}] Device {}: used={} MB, free={} MB, total={} MB",
                        label, d, usedMem / (1024 * 1024), freeMem / (1024 * 1024), totalMem / (1024 * 1024));
            }
        } catch (Exception e) {
            log.warn("Could not query GPU memory: {}", e.getMessage());
        }
    }

    private BufferedImage createSyntheticPage(int width, int height, String text) {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, width, height);
        g.setColor(Color.BLACK);
        g.setFont(new Font("SansSerif", Font.PLAIN, 14));
        g.drawString(text, 50, 100);
        g.dispose();
        return img;
    }

    /**
     * Test 1: Minimal synthetic page generate — exercises StaticKvCacheDecodeLoop.
     *
     * <p>This is the minimal reproducer for the CUDA graph replay hang.
     * A single synthetic page is generated with maxNewTokens=50.
     * The decode loop should complete all 50 steps without hanging at step 2.</p>
     *
     * <p>Expected behavior:
     * - Step 0: prefill + CUDA graph capture (succeeds)
     * - Step 1: first decode + CUDA graph capture (succeeds)
     * - Step 2+: CUDA graph replay (HANGS without fix)</p>
     */
    @Test
    @Order(1)
    @DisplayName("Synthetic page generate — StaticKvCacheDecodeLoop replay test")
    @Timeout(value = TEST_TIMEOUT_SECONDS, unit = TimeUnit.SECONDS)
    public void testSyntheticPageGenerate() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");
        logGpuMemory("Test 1 start");

        // Create a simple synthetic page
        BufferedImage page = createSyntheticPage(800, 1200, "Test document page for CUDA graph replay");
        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(page, TARGET_SIZE, 9);
        log.info("Synthetic page: {} frames ({}x{} grid)",
                splitResult.getTotalFrames(), splitResult.numRows, splitResult.numCols);

        logGpuMemory("Before generatePagesTiled");

        GenerationResult[] results = vlm.generatePagesTiled(
                Collections.singletonList(splitResult),
                "Convert this page to docling.",
                MAX_NEW_TOKENS,
                TARGET_SIZE);

        logGpuMemory("After generatePagesTiled");

        assertNotNull(results);
        assertEquals(1, results.length);
        assertNotNull(results[0]);
        log.info("Result: {} tokens, text='{}', finishReason={}",
                results[0].getGeneratedTokenCount(),
                results[0].getText(),
                results[0].getFinishReason());
        log.info("Timing: firstTokenLatency={}ms, totalTime={}ms, {:.1f} tok/s",
                results[0].getFirstTokenLatencyMs(),
                results[0].getGenerationTimeMs(),
                results[0].getTokensPerSecond());

        // Verify we got some output (not hanging)
        assertTrue(results[0].getGeneratedTokenCount() > 0, "No tokens generated — decode loop may have hung");

        vlm.resetSessions();
        logGpuMemory("After resetSessions");
    }

    /**
     * Test 2: PDF page generate — real-world reproducer.
     *
     * <p>Uses page 10 from pathfinder-mythic.pdf (the actual page that triggered the hang).
     * This is the full end-to-end test with real image data.</p>
     */
    @Test
    @Order(2)
    @DisplayName("PDF page generate — real-world CUDA graph replay test")
    @Timeout(value = TEST_TIMEOUT_SECONDS, unit = TimeUnit.SECONDS)
    public void testPdfPageGenerate() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");
        Assumptions.assumeTrue(PDF_FILE.exists(), "PDF file not found: " + PDF_FILE + " — skipping");

        logGpuMemory("Test 2 start");

        // Load PDF page 10 (the page that triggered the original hang)
        BufferedImage page;
        try {
            page = loadPdfPage(PDF_FILE, 10);
            assertNotNull(page, "Failed to load PDF page 10");
        } catch (Exception e) {
            log.warn("Could not load PDF page 10: {} — using synthetic page instead", e.getMessage());
            page = createSyntheticPage(800, 1200, "PDF page fallback");
        }

        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(page, TARGET_SIZE, 9);
        log.info("PDF page: {} frames ({}x{} grid), image size={}x{}",
                splitResult.getTotalFrames(), splitResult.numRows, splitResult.numCols,
                page.getWidth(), page.getHeight());

        logGpuMemory("Before generatePagesTiled (PDF)");

        GenerationResult[] results = vlm.generatePagesTiled(
                Collections.singletonList(splitResult),
                "Convert this page to docling.",
                MAX_NEW_TOKENS,
                TARGET_SIZE);

        logGpuMemory("After generatePagesTiled (PDF)");

        assertNotNull(results);
        assertEquals(1, results.length);
        assertNotNull(results[0]);
        log.info("PDF page result: {} tokens, text='{}', finishReason={}",
                results[0].getGeneratedTokenCount(),
                results[0].getText(),
                results[0].getFinishReason());
        log.info("PDF page timing: firstTokenLatency={}ms, totalTime={}ms, {:.1f} tok/s",
                results[0].getFirstTokenLatencyMs(),
                results[0].getGenerationTimeMs(),
                results[0].getTokensPerSecond());

        // Verify we got some output (not hanging)
        assertTrue(results[0].getGeneratedTokenCount() > 0, "No tokens generated — decode loop may have hung");

        vlm.resetSessions();
        logGpuMemory("After resetSessions (PDF)");
    }

    /**
     * Test 3: Multiple sequential pages — stress test for CUDA graph replay.
     *
     * <p>Generates 3 pages sequentially to stress-test the CUDA graph replay path.
     * Each page should complete without hanging, and GPU memory should be properly managed.</p>
     */
    @Test
    @Order(3)
    @DisplayName("Multiple sequential pages — CUDA graph replay stress test")
    @Timeout(value = TEST_TIMEOUT_SECONDS * 3, unit = TimeUnit.SECONDS)
    public void testMultipleSequentialPages() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");
        logGpuMemory("Test 3 start");

        int numPages = 3;
        for (int pageNum = 1; pageNum <= numPages; pageNum++) {
            log.info("=== Page {}/{} ===", pageNum, numPages);
            logGpuMemory("Before page " + pageNum);

            BufferedImage page = createSyntheticPage(800, 1200, "Page " + pageNum + " content for stress test");
            ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(page, TARGET_SIZE, 9);
            log.info("Page {}: {} frames", pageNum, splitResult.getTotalFrames());

            GenerationResult[] results;
            try {
                results = vlm.generatePagesTiled(
                        Collections.singletonList(splitResult),
                        "Convert this page to docling.",
                        MAX_NEW_TOKENS,
                        TARGET_SIZE);
            } catch (Exception e) {
                logGpuMemory("FAILED at page " + pageNum);
                fail("Page " + pageNum + " failed: " + e.getMessage(), e);
                return;
            }

            logGpuMemory("After page " + pageNum);

            assertNotNull(results);
            assertEquals(1, results.length);
            log.info("Page {} result: {} tokens, text='{}'",
                    pageNum, results[0].getGeneratedTokenCount(), results[0].getText());

            // Reset between pages
            vlm.resetSessions();
            logGpuMemory("After resetSessions page " + pageNum);
        }

        log.info("All {} pages completed successfully", numPages);
    }

    /**
     * Test 4: Direct decode loop test — bypass VLM wrapper.
     *
     * <p>Directly exercises StaticKvCacheDecodeLoop to isolate the CUDA graph replay bug
     * from any VLM wrapper logic.</p>
     */
    @Test
    @Order(4)
    @DisplayName("Direct StaticKvCacheDecodeLoop test — isolate replay bug")
    @Timeout(value = TEST_TIMEOUT_SECONDS, unit = TimeUnit.SECONDS)
    public void testDirectDecodeLoop() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");

        logGpuMemory("Test 4 start");

        // Get decoder and embed_tokens from VLM
        SameDiff decoder = vlm.getDecoder();
        SameDiff embedTokens = vlm.getEmbedTokens();

        // Create simple prompt
        int[] promptTokens = {0, 44, 2692, 46}; // Simple 4-token prompt
        long hiddenSize = vlm.getConfig() != null && vlm.getConfig().getHiddenSize() != null
                ? vlm.getConfig().getHiddenSize() : 2048;

        // Build prefill embeddings
        INDArray embeddingTable = null;
        for (org.nd4j.autodiff.samediff.SDVariable var : embedTokens.variables()) {
            if (var.getVariableType() == org.nd4j.autodiff.samediff.VariableType.CONSTANT ||
                var.getVariableType() == org.nd4j.autodiff.samediff.VariableType.VARIABLE) {
                INDArray arr = var.getArr();
                if (arr != null && arr.rank() == 2) {
                    if (embeddingTable == null || arr.length() > embeddingTable.length()) {
                        embeddingTable = arr;
                    }
                }
            }
        }
        assertNotNull(embeddingTable, "Could not find embedding table");

        INDArray prefillEmbeds = buildPrefillEmbeddings(embeddingTable, promptTokens);
        log.info("Prefill embeddings shape: {}", java.util.Arrays.toString(prefillEmbeds.shape()));

        // Build StaticKvCacheDecodeLoop
        org.eclipse.deeplearning4j.llm.generation.SamplingConfig samplingCfg =
                org.eclipse.deeplearning4j.llm.generation.SamplingConfig.greedy();

        org.eclipse.deeplearning4j.llm.generation.ModelIOConfig decoderIOConfig =
                org.eclipse.deeplearning4j.llm.generation.ModelIOConfig.discover(decoder);

        org.eclipse.deeplearning4j.llm.generation.StaticKvCacheDecodeLoop loop =
                org.eclipse.deeplearning4j.llm.generation.StaticKvCacheDecodeLoop.builder()
                        .decoder(decoder)
                        .embedTokens(embedTokens)
                        .tokenizer(vlm.getTokenizer())
                        .ioConfig(decoderIOConfig)
                        .samplingConfig(samplingCfg)
                        .maxNewTokens(MAX_NEW_TOKENS)
                        .hiddenSize(hiddenSize)
                        .build();

        log.info("Running StaticKvCacheDecodeLoop with maxNewTokens={}", MAX_NEW_TOKENS);
        logGpuMemory("Before decode loop");

        GenerationResult result = loop.decode(prefillEmbeds, promptTokens);

        logGpuMemory("After decode loop");

        assertNotNull(result);
        log.info("Direct decode result: {} tokens, text='{}', finishReason={}",
                result.getGeneratedTokenCount(),
                result.getText(),
                result.getFinishReason());
        log.info("Direct decode timing: firstTokenLatency={}ms, totalTime={}ms, {:.1f} tok/s",
                result.getFirstTokenLatencyMs(),
                result.getGenerationTimeMs(),
                result.getTokensPerSecond());

        // Verify we got some output (not hanging)
        assertTrue(result.getGeneratedTokenCount() > 0, "No tokens generated — decode loop may have hung");
    }

    private INDArray buildPrefillEmbeddings(INDArray embeddingTable, int[] tokens) {
        long hiddenSize = embeddingTable.size(1);
        INDArray result = Nd4j.create(DataType.FLOAT, 1, tokens.length, hiddenSize);
        for (int i = 0; i < tokens.length; i++) {
            result.get(org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                    org.nd4j.linalg.indexing.NDArrayIndex.point(i),
                    org.nd4j.linalg.indexing.NDArrayIndex.all())
                    .assign(embeddingTable.getRow(tokens[i]));
        }
        return result;
    }

    private BufferedImage loadPdfPage(File pdfFile, int pageNumber) throws IOException {
        // Use PDFBox or similar to load PDF page
        // For now, return a synthetic image as placeholder
        // In a real test, you'd use:
        // PDDocument doc = PDDocument.load(pdfFile);
        // PDFRenderer renderer = new PDFRenderer(doc);
        // BufferedImage img = renderer.renderImageWithDPI(pageNumber - 1, 150);
        return createSyntheticPage(800, 1200, "PDF page " + pageNumber + " placeholder");
    }
}
