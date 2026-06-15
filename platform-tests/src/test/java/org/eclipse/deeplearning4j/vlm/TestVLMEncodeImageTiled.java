package org.eclipse.deeplearning4j.vlm;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.generation.SameDiffMemoryUtils;
import org.eclipse.deeplearning4j.vlm.model.VisionEncoderUtils;
import org.eclipse.deeplearning4j.vlm.model.VisionLanguageModel;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test VisionLanguageModel.encodeImageTiled() — the same code path used by
 * the kompile VLM subprocess (VlmDocumentPipeline).
 *
 * Uses the pre-downloaded model at ~/.kompile/models/vlm/smoldocling-256m-pipeline/
 * which contains SDZ + ONNX + tokenizer.json + preprocessor_config.json.
 *
 * Run:
 *   cd platform-tests && mvn test \
 *     -Dtest=TestVLMEncodeImageTiled \
 *     -Dbackend.artifactId=nd4j-cuda-12.9
 */
@Slf4j
public class TestVLMEncodeImageTiled {

    private static final File MODEL_DIR = new File(
            System.getProperty("user.home") + "/.kompile/models/vlm/smoldocling-256m-pipeline");
    private static final int TARGET_SIZE = 512;

    private static VisionLanguageModel vlm;

    @BeforeAll
    public static void setup() throws Exception {
        Assumptions.assumeTrue(MODEL_DIR.isDirectory(),
                "Model dir not found: " + MODEL_DIR + " — skip test");

        log.info("Loading VisionLanguageModel from {}", MODEL_DIR);
        vlm = VisionLanguageModel.loadSmolDocling(MODEL_DIR);
        log.info("VLM loaded successfully");
    }

    @AfterAll
    public static void teardown() {
        if (vlm != null) vlm.close();
    }

    @Test
    @DisplayName("encodeImageTiled with single 512x512 synthetic image")
    public void testEncodeImageTiledSingleFrame() {
        // Create a synthetic 512x512 image (same as what PDF rendering produces)
        BufferedImage testImage = new BufferedImage(TARGET_SIZE, TARGET_SIZE, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = testImage.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, TARGET_SIZE, TARGET_SIZE);
        g.setColor(Color.BLACK);
        g.drawString("Test page content", 50, 100);
        g.dispose();

        // Tile it — same path as VlmDocumentPipeline
        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(testImage, TARGET_SIZE, 9);
        assertNotNull(splitResult);
        assertTrue(splitResult.getTotalFrames() > 0, "No frames produced");
        log.info("Split into {} frames ({}x{} grid)", splitResult.getTotalFrames(),
                splitResult.numRows, splitResult.numCols);

        // This is the call that fails in the subprocess
        INDArray visionEmbeddings = vlm.encodeImageTiled(splitResult, TARGET_SIZE);
        assertNotNull(visionEmbeddings);
        log.info("Vision embeddings shape: {}", Arrays.toString(visionEmbeddings.shape()));
        assertTrue(visionEmbeddings.rank() >= 2, "Expected rank >= 2, got " + visionEmbeddings.rank());

        visionEmbeddings.close();
    }

    @Test
    @DisplayName("encodeImageTiled with larger image that produces multiple tiles")
    public void testEncodeImageTiledMultiFrame() {
        // Create a larger image that will be split into multiple tiles
        int w = 800, h = 1200;
        BufferedImage testImage = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = testImage.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, w, h);
        g.setColor(Color.BLACK);
        g.drawString("Multi-tile test", 50, 100);
        g.dispose();

        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(testImage, TARGET_SIZE, 9);
        assertNotNull(splitResult);
        log.info("Split {}x{} image into {} frames ({}x{} grid)",
                w, h, splitResult.getTotalFrames(), splitResult.numRows, splitResult.numCols);

        INDArray visionEmbeddings = vlm.encodeImageTiled(splitResult, TARGET_SIZE);
        assertNotNull(visionEmbeddings);
        log.info("Vision embeddings shape: {}", Arrays.toString(visionEmbeddings.shape()));
        assertTrue(visionEmbeddings.rank() >= 2);

        visionEmbeddings.close();
    }

    /**
     * Isolate the exact failure: call visionEncoder.output() twice in a row
     * with the same rank-4 input — mimics what encodeImageTiled does per frame.
     * This bypasses VisionLanguageModel entirely to pinpoint DSP plan reuse bug.
     */
    @Test
    @DisplayName("Raw SameDiff vision encoder: two consecutive calls with same rank-4 input")
    public void testRawVisionEncoderTwoConsecutiveCalls() {
        SameDiff visionEncoder = vlm.getVisionEncoder();
        assertNotNull(visionEncoder, "Vision encoder is null");

        INDArray pixelValues = Nd4j.rand(DataType.FLOAT, 1, 3, TARGET_SIZE, TARGET_SIZE);
        INDArray pixelMask = Nd4j.ones(DataType.BOOL, 1, TARGET_SIZE, TARGET_SIZE);

        List<String> outputNames = List.of("image_features");

        // Call 1 — should succeed (compiles DSP plan)
        log.info("=== Call 1: rank-4 [1,3,512,512] ===");
        Map<String, INDArray> inputs1 = new HashMap<>();
        inputs1.put("pixel_values", pixelValues);
        inputs1.put("pixel_attention_mask", pixelMask);
        Map<String, INDArray> out1 = visionEncoder.output(inputs1, outputNames);
        assertNotNull(out1.get("image_features"));
        log.info("Call 1 output shape: {}", Arrays.toString(out1.get("image_features").shape()));

        // Close outputs like encodeImageTiled does
        for (var entry : out1.entrySet()) {
            INDArray arr = entry.getValue();
            if (arr != null && !arr.wasClosed()) {
                arr.setCloseable(true);
                arr.close();
            }
        }

        // Close INPUTS too — this is what encodeImageTiled does (lines 1342-1343)
        pixelValues.close();
        pixelMask.close();

        // Call 2 — this is what fails on frame 2
        log.info("=== Call 2: same rank-4 [1,3,512,512] ===");
        INDArray pixelValues2 = Nd4j.rand(DataType.FLOAT, 1, 3, TARGET_SIZE, TARGET_SIZE);
        INDArray pixelMask2 = Nd4j.ones(DataType.BOOL, 1, TARGET_SIZE, TARGET_SIZE);
        Map<String, INDArray> inputs2 = new HashMap<>();
        inputs2.put("pixel_values", pixelValues2);
        inputs2.put("pixel_attention_mask", pixelMask2);

        Map<String, INDArray> out2;
        try {
            out2 = visionEncoder.output(inputs2, outputNames);
        } catch (Exception e) {
            log.error("Call 2 FAILED: {}", e.getMessage());
            fail("Second call to visionEncoder.output() failed: " + e.getMessage(), e);
            return;
        }
        assertNotNull(out2.get("image_features"));
        log.info("Call 2 output shape: {}", Arrays.toString(out2.get("image_features").shape()));

        pixelValues.close();
        pixelMask.close();
        pixelValues2.close();
        pixelMask2.close();
    }

    /**
     * Call encodeImageTiled twice with single-frame images.
     * If frame 2 within a single call fails, does calling it separately also fail?
     */
    @Test
    @DisplayName("encodeImageTiled called twice with single-frame images")
    public void testEncodeImageTiledCalledTwice() {
        BufferedImage img1 = new BufferedImage(TARGET_SIZE, TARGET_SIZE, BufferedImage.TYPE_INT_RGB);
        Graphics2D g1 = img1.createGraphics();
        g1.setColor(Color.WHITE); g1.fillRect(0, 0, TARGET_SIZE, TARGET_SIZE);
        g1.dispose();

        ImageTiler.SplitImageResult split1 = ImageTiler.splitImageForVLM(img1, TARGET_SIZE, 9);
        INDArray emb1 = vlm.encodeImageTiled(split1, TARGET_SIZE);
        assertNotNull(emb1);
        log.info("Call 1 embeddings shape: {}", Arrays.toString(emb1.shape()));
        emb1.close();

        // Second call — same setup, fresh image
        BufferedImage img2 = new BufferedImage(TARGET_SIZE, TARGET_SIZE, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2 = img2.createGraphics();
        g2.setColor(Color.GRAY); g2.fillRect(0, 0, TARGET_SIZE, TARGET_SIZE);
        g2.dispose();

        ImageTiler.SplitImageResult split2 = ImageTiler.splitImageForVLM(img2, TARGET_SIZE, 9);
        INDArray emb2;
        try {
            emb2 = vlm.encodeImageTiled(split2, TARGET_SIZE);
        } catch (Exception e) {
            log.error("Second encodeImageTiled call FAILED: {}", e.getMessage());
            fail("Second call to encodeImageTiled failed: " + e.getMessage(), e);
            return;
        }
        assertNotNull(emb2);
        log.info("Call 2 embeddings shape: {}", Arrays.toString(emb2.shape()));
        emb2.close();
    }

    /**
     * Mimic encodeImageTiled exactly: split a large image, preprocess each frame,
     * call visionEncoder.output() per frame.
     */
    @Test
    @DisplayName("Manual encodeImageTiled reproduction: preprocess + output per frame")
    public void testManualEncodeImageTiledRepro() {
        SameDiff visionEncoder = vlm.getVisionEncoder();
        VLMImagePreprocessor preprocessor = vlm.getImagePreprocessor();
        assertNotNull(visionEncoder);
        assertNotNull(preprocessor);

        // Create multi-tile image
        int w = 800, h = 1200;
        BufferedImage testImage = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = testImage.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, w, h);
        g.setColor(Color.BLACK);
        g.drawString("Manual repro test", 50, 100);
        g.dispose();

        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(testImage, TARGET_SIZE, 9);
        int numFrames = splitResult.getTotalFrames();
        log.info("Testing {} frames manually", numFrames);

        var ioConfig = vlm.getVisionEncoderIOConfig();

        for (int f = 0; f < numFrames; f++) {
            BufferedImage frame = splitResult.frames.get(f);
            INDArray frameTensor = preprocessor.preprocess(frame);
            log.info("Frame {}/{}: preprocessed shape={} rank={}",
                    f + 1, numFrames, Arrays.toString(frameTensor.shape()), frameTensor.rank());

            Map<String, INDArray> inputs = new HashMap<>();
            inputs.put(ioConfig.getPixelValuesName(), frameTensor);

            ImageTiler.ContentRegion region = splitResult.contentRegions.get(f);
            INDArray pixelMask = ImageTiler.createPixelAttentionMask(
                    region.width, region.height, TARGET_SIZE);
            if (ioConfig.hasPixelAttentionMask()) {
                inputs.put(ioConfig.getPixelAttentionMaskName(), pixelMask);
            }

            Map<String, INDArray> outputs;
            try {
                outputs = visionEncoder.output(inputs, "image_features");
            } catch (Exception e) {
                log.error("Frame {}/{} FAILED: {}", f + 1, numFrames, e.getMessage());
                fail("Frame " + (f + 1) + "/" + numFrames + " failed: " + e.getMessage(), e);
                return;
            }

            // Do exactly what encodeImageTiled does
            VisionEncoderUtils.VisionOutput selected = VisionEncoderUtils.selectVisionOutput(outputs);
            assertNotNull(selected, "Frame " + (f + 1) + " selected output null");
            INDArray embedding = selected.tensor;
            if (embedding.rank() == 2) {
                embedding = embedding.reshape(1, embedding.size(0), embedding.size(1));
            }
            // Check for zeros/NaN like encodeImageTiled does
            if (VisionEncoderUtils.isAllZeroOrNaN(selected.tensor)) {
                log.warn("Frame {}/{}: all zeros/NaN", f + 1, numFrames);
            }
            INDArray dup = embedding.dup();
            log.info("Frame {}/{}: output shape={}, min={}, max={}", f + 1, numFrames,
                    Arrays.toString(dup.shape()),
                    embedding.minNumber(), embedding.maxNumber());

            // Close outputs EXACTLY like encodeImageTiled does
            for (var entry : outputs.entrySet()) {
                INDArray arr = entry.getValue();
                if (arr != null && !arr.wasClosed()) {
                    arr.setCloseable(true);
                    arr.close();
                }
            }
            SameDiffMemoryUtils.safeClose(frameTensor);
            SameDiffMemoryUtils.safeClose(pixelMask);
            dup.close();
        }
    }

    /**
     * Exact copy of VisionLanguageModel.encodeImageTiled() logic — to find the exact line that differs.
     */
    @Test
    @DisplayName("Exact copy of encodeImageTiled logic")
    public void testExactCopyOfEncodeImageTiled() {
        SameDiff visionEncoder = vlm.getVisionEncoder();
        VLMImagePreprocessor preprocessor = vlm.getImagePreprocessor();
        var ioConfig = vlm.getVisionEncoderIOConfig();

        int w = 800, h = 1200;
        BufferedImage testImage = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = testImage.createGraphics();
        g.setColor(Color.WHITE); g.fillRect(0, 0, w, h);
        g.dispose();

        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(testImage, TARGET_SIZE, 9);
        int numFrames = splitResult.frames.size();
        log.info("Exact copy test: {} frames", numFrames);

        List<INDArray> frameEmbeddings = new ArrayList<>();

        for (int f = 0; f < numFrames; f++) {
            BufferedImage frame = splitResult.frames.get(f);
            INDArray frameTensor = preprocessor.preprocess(frame);
            // normalizeVisionInputShape is a no-op

            Map<String, INDArray> inputs = new HashMap<>();
            inputs.put(ioConfig.getPixelValuesName(), frameTensor);

            ImageTiler.ContentRegion region = splitResult.contentRegions.get(f);
            INDArray pixelMask = ImageTiler.createPixelAttentionMask(
                    region.width, region.height, TARGET_SIZE);
            if (ioConfig.hasPixelAttentionMask()) {
                inputs.put(ioConfig.getPixelAttentionMaskName(), pixelMask);
            }

            Map<String, INDArray> outputs;
            try {
                outputs = visionEncoder.output(inputs, ioConfig.getOutputNames());
            } catch (Exception e) {
                log.error("Exact copy frame {}/{} FAILED: {}", f + 1, numFrames, e.getMessage());
                fail("Exact copy frame " + (f + 1) + " failed: " + e.getMessage(), e);
                return;
            }

            VisionEncoderUtils.VisionOutput selected = VisionEncoderUtils.selectVisionOutput(outputs);
            assertNotNull(selected);

            if (VisionEncoderUtils.isAllZeroOrNaN(selected.tensor)) {
                log.warn("Frame {}/{}: all zeros/NaN, skipping", f + 1, numFrames);
                continue;
            }

            INDArray embedding = selected.tensor;
            if (embedding.rank() == 2) {
                embedding = embedding.reshape(1, embedding.size(0), embedding.size(1));
            }

            frameEmbeddings.add(embedding.dup());

            log.info("Frame {}/{}: output '{}' shape={}, min={}, max={}",
                    f + 1, numFrames, selected.name,
                    Arrays.toString(embedding.shape()),
                    embedding.minNumber(), embedding.maxNumber());

            for (var entry : outputs.entrySet()) {
                INDArray arr = entry.getValue();
                if (arr != null && !arr.wasClosed()) {
                    arr.setCloseable(true);
                    arr.close();
                }
            }

            SameDiffMemoryUtils.safeClose(frameTensor);
            SameDiffMemoryUtils.safeClose(pixelMask);
        }

        assertEquals(numFrames, frameEmbeddings.size(), "Not all frames produced embeddings");
        for (INDArray fe : frameEmbeddings) fe.close();
    }

    /**
     * Same as above but with resetSession() between calls — test if resetting fixes it.
     */
    @Test
    @DisplayName("Raw SameDiff vision encoder: two calls with resetSession between them")
    public void testRawVisionEncoderWithResetBetweenCalls() {
        SameDiff visionEncoder = vlm.getVisionEncoder();
        assertNotNull(visionEncoder, "Vision encoder is null");

        List<String> outputNames = List.of("image_features");

        // Call 1
        log.info("=== Call 1 (with reset) ===");
        INDArray pv1 = Nd4j.rand(DataType.FLOAT, 1, 3, TARGET_SIZE, TARGET_SIZE);
        INDArray pm1 = Nd4j.ones(DataType.BOOL, 1, TARGET_SIZE, TARGET_SIZE);
        Map<String, INDArray> in1 = Map.of("pixel_values", pv1, "pixel_attention_mask", pm1);
        Map<String, INDArray> out1 = visionEncoder.output(in1, outputNames);
        assertNotNull(out1.get("image_features"));
        log.info("Call 1 output: {}", Arrays.toString(out1.get("image_features").shape()));
        for (var e : out1.entrySet()) {
            if (e.getValue() != null && !e.getValue().wasClosed()) {
                e.getValue().setCloseable(true);
                e.getValue().close();
            }
        }
        pv1.close();
        pm1.close();

        // Reset session between calls
        log.info("=== Resetting session ===");
        visionEncoder.resetSession();

        // Call 2
        log.info("=== Call 2 (after reset) ===");
        INDArray pv2 = Nd4j.rand(DataType.FLOAT, 1, 3, TARGET_SIZE, TARGET_SIZE);
        INDArray pm2 = Nd4j.ones(DataType.BOOL, 1, TARGET_SIZE, TARGET_SIZE);
        Map<String, INDArray> in2 = Map.of("pixel_values", pv2, "pixel_attention_mask", pm2);
        Map<String, INDArray> out2;
        try {
            out2 = visionEncoder.output(in2, outputNames);
        } catch (Exception e) {
            log.error("Call 2 (after reset) FAILED: {}", e.getMessage());
            fail("Second call after resetSession() failed: " + e.getMessage(), e);
            return;
        }
        assertNotNull(out2.get("image_features"));
        log.info("Call 2 output: {}", Arrays.toString(out2.get("image_features").shape()));

        pv2.close();
        pm2.close();
    }
}
