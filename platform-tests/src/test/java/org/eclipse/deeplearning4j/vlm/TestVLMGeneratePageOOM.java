package org.eclipse.deeplearning4j.vlm;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.vlm.model.VisionLanguageModel;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.junit.jupiter.api.*;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Reproduce the GPU OOM that occurs during decoder token generation
 * in the VLM subprocess. The vision encoder succeeds (all frames),
 * but the decoder runs out of GPU memory during generatePagesTiled().
 *
 * This exercises the same code path as VlmDocumentPipeline.processPdf():
 *   1. ImageTiler.splitImageForVLM()
 *   2. vlm.generatePagesTiled() -> encodeImageTiled() + generateFromEmbeddings() + decodeFromEmbeddings()
 *   3. vlm.resetSessions() between pages
 *
 * Run:
 *   cd platform-tests && mvn test \
 *     -Dtest=TestVLMGeneratePageOOM \
 *     -Dbackend.artifactId=nd4j-cuda-12.9
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestVLMGeneratePageOOM {

    private static final File MODEL_DIR = new File(
            System.getProperty("user.home") + "/.kompile/models/vlm/smoldocling-256m-pipeline");
    private static final int TARGET_SIZE = 512;
    private static final int MAX_NEW_TOKENS = 50;

    private static VisionLanguageModel vlm;

    @BeforeAll
    public static void setup() throws Exception {
        Assumptions.assumeTrue(MODEL_DIR.isDirectory(),
                "Model dir not found: " + MODEL_DIR + " — skip test");

        logGpuMemory("Before model load");
        log.info("Loading VisionLanguageModel from {}", MODEL_DIR);
        vlm = VisionLanguageModel.loadSmolDocling(MODEL_DIR);
        log.info("VLM loaded successfully");
        logGpuMemory("After model load");
    }

    @AfterAll
    public static void teardown() {
        if (vlm != null) vlm.close();
    }

    private static void logGpuMemory(String label) {
        try {
            var nativeOps = org.nd4j.nativeblas.NativeOpsHolder.getInstance().getDeviceNativeOps();
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
     * Single page generation — exercises the full pipeline:
     * vision encoder (all frames) + embed tokens + decoder loop.
     * This is the minimal repro for the OOM in decodeFromEmbeddings().
     */
    @Test
    @Order(1)
    @DisplayName("Single page generatePagesTiled — full encode + decode")
    public void testSinglePageGenerate() {
        BufferedImage page = createSyntheticPage(800, 1200, "Test document page 1");
        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(page, TARGET_SIZE, 9);
        log.info("Page split into {} frames ({}x{} grid)",
                splitResult.getTotalFrames(), splitResult.numRows, splitResult.numCols);

        logGpuMemory("Before generatePagesTiled (page 1)");

        GenerationResult[] results = vlm.generatePagesTiled(
                Collections.singletonList(splitResult),
                "Convert this page to docling.",
                MAX_NEW_TOKENS,
                TARGET_SIZE);

        logGpuMemory("After generatePagesTiled (page 1)");

        assertNotNull(results);
        assertEquals(1, results.length);
        assertNotNull(results[0]);
        log.info("Page 1 result: {} tokens, text='{}', finishReason={}",
                results[0].getGeneratedTokenCount(),
                results[0].getText(),
                results[0].getFinishReason());
        log.info("Page 1 timing: firstTokenLatency={}ms, totalTime={}ms, {:.1f} tok/s",
                results[0].getFirstTokenLatencyMs(),
                results[0].getGenerationTimeMs(),
                results[0].getTokensPerSecond());

        // Reset sessions like VlmDocumentPipeline does
        vlm.resetSessions();
        logGpuMemory("After resetSessions (page 1)");
    }

    /**
     * Two pages sequentially — the pattern that triggers OOM in the subprocess.
     * Page 1 succeeds, page 2 fails because GPU memory wasn't fully reclaimed.
     */
    @Test
    @Order(2)
    @DisplayName("Two pages sequentially with resetSessions — OOM repro")
    public void testTwoPagesSequential() {
        for (int pageNum = 1; pageNum <= 2; pageNum++) {
            BufferedImage page = createSyntheticPage(800, 1200, "Page " + pageNum + " content");
            ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(page, TARGET_SIZE, 9);
            log.info("Page {}: {} frames ({}x{} grid)", pageNum,
                    splitResult.getTotalFrames(), splitResult.numRows, splitResult.numCols);

            logGpuMemory("Before page " + pageNum);

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

            // Reset between pages — same as VlmDocumentPipeline
            vlm.resetSessions();
            logGpuMemory("After resetSessions page " + pageNum);
        }
    }

    /**
     * Small single-frame image — minimal GPU memory for encode+decode.
     * Should succeed even under memory pressure.
     */
    @Test
    @Order(3)
    @DisplayName("Small image (512x512, single frame) — minimal memory")
    public void testSmallImageGenerate() {
        BufferedImage page = createSyntheticPage(512, 512, "Small page");
        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(page, TARGET_SIZE, 9);
        log.info("Small image: {} frames", splitResult.getTotalFrames());

        logGpuMemory("Before small image generate");

        GenerationResult[] results = vlm.generatePagesTiled(
                Collections.singletonList(splitResult),
                "Convert this page to docling.",
                MAX_NEW_TOKENS,
                TARGET_SIZE);

        logGpuMemory("After small image generate");

        assertNotNull(results);
        assertEquals(1, results.length);
        log.info("Small image result: {} tokens, text='{}'",
                results[0].getGeneratedTokenCount(), results[0].getText());

        vlm.resetSessions();
    }
}
