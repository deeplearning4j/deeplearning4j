/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  ******************************************************************************
 */
package org.eclipse.deeplearning4j.vlm;

import lombok.extern.slf4j.Slf4j;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.eclipse.deeplearning4j.llm.config.PreprocessorConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.SameDiffMemoryUtils;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfig;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfigApplier;
import org.eclipse.deeplearning4j.vlm.model.VisionLanguageModel;
import org.eclipse.deeplearning4j.vlm.model.loading.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Regression coverage for changing OCR regions through one frozen vision plan. */
@Slf4j
public class VisionEncoderRegionReplayTest {

    private static final int TARGET_SIZE = 512;
    private static final String DEFAULT_MODEL_DIR =
            "/home/agibsonccc/Documents/GitHub/kompile/data/models/vlm-pipelines/smoldocling-256m";
    private static final String DEFAULT_PDF =
            "/home/agibsonccc/Documents/GitHub/kompile/kompile-vlm-demo/sample-pdfs/sample.pdf";

    @Test
    void testRegionSequenceMatchesFreshReplay() throws Exception {
        File modelDir = new File(System.getProperty("vlm.test.modelDir", DEFAULT_MODEL_DIR));
        File pdfFile = new File(System.getProperty("vlm.test.pdf.path", DEFAULT_PDF));
        Assumptions.assumeTrue(new File(modelDir, "vision_encoder.onnx").isFile(),
                "SmolDocling vision encoder is not available");
        Assumptions.assumeTrue(pdfFile.isFile(), "OCR regression PDF is not available");

        SameDiff visionEncoder = OnnxModelCache.importWithCache(
                new File(modelDir, "vision_encoder.onnx").getAbsolutePath());
        visionEncoder.setDspAutoCompileEnabled(true);
        visionEncoder.setDspNativeAutoCompileEnabled(true);
        PreprocessorConfig config = PreprocessorConfig.fromFile(
                new File(modelDir, "preprocessor_config.json"));
        VLMImagePreprocessor preprocessor = VLMImagePreprocessor.fromConfig(config);

        try (PDDocument document = PDDocument.load(pdfFile)) {
            BufferedImage page = new PDFRenderer(document)
                    .renderImageWithDPI(7, 144, ImageType.RGB);
            assertEquals(1224, page.getWidth(), "Unexpected page-8 render width");

            BufferedImage parent = page.getSubimage(0, 0, 522, 1584);
            BufferedImage target = page.getSubimage(0, 0, 280, 1584);

            INDArray initial = encodeRegion(visionEncoder, preprocessor, target, "initial");
            INDArray fresh = encodeRegion(visionEncoder, preprocessor, target, "repeat");
            dumpFingerprintsIfRequested(visionEncoder);

            // Separate the reference and production-order scenarios without clearing mid-scenario.
            visionEncoder.resetSession();
            visionEncoder.clearDynamicShapePlanCache();

            SameDiffMemoryUtils.safeClose(encodeRegion(visionEncoder, preprocessor, parent, "parent"));
            INDArray replayed = encodeRegion(visionEncoder, preprocessor, target, "after-parent");
            visionEncoder.resetSession();
            visionEncoder.clearDynamicShapePlanCache();
            INDArray recomputed = encodeRegion(visionEncoder, preprocessor, target, "post-reset");

            try {
                double repeatDifference = maxAbsDifference(initial, fresh);
                double replayDifference = maxAbsDifference(fresh, replayed);
                double recomputedDifference = maxAbsDifference(fresh, recomputed);
                double initialRecomputedDifference = maxAbsDifference(initial, recomputed);
                log.info("Vision lifecycle comparison: initialSum={}, freshSum={}, replayedSum={}, "
                                + "recomputedSum={}, repeatDifference={}, replayDifference={}, "
                                + "recomputedDifference={}, initialRecomputedDifference={}",
                        initial.sumNumber(), fresh.sumNumber(), replayed.sumNumber(),
                        recomputed.sumNumber(), repeatDifference, replayDifference,
                        recomputedDifference, initialRecomputedDifference);
                assertEquals(0.0, repeatDifference, 1e-3,
                        "Repeated target execution changed while the DSP plan remained active");
                assertEquals(0.0, recomputedDifference, 1e-3,
                        "Target remained corrupted after rebuilding only the DSP plan; model state was mutated");
                assertEquals(0.0, replayDifference, 1e-3,
                        "Vision output changed after replaying preceding OCR regions");
            } finally {
                SameDiffMemoryUtils.safeClose(initial);
                SameDiffMemoryUtils.safeClose(fresh);
                SameDiffMemoryUtils.safeClose(replayed);
                SameDiffMemoryUtils.safeClose(recomputed);
            }
        } finally {
            preprocessor.shutdown();
            SameDiffMemoryUtils.freeVisionEncoder(visionEncoder);
        }
    }

    @Test
    void testFullPageThenColumnMatchesFreshReplay() throws Exception {
        File modelDir = new File(System.getProperty("vlm.test.modelDir", DEFAULT_MODEL_DIR));
        File pdfFile = new File(System.getProperty("vlm.test.pdf.path", DEFAULT_PDF));
        Assumptions.assumeTrue(new File(modelDir, "vision_encoder.onnx").isFile(),
                "SmolDocling vision encoder is not available");
        Assumptions.assumeTrue(pdfFile.isFile(), "OCR regression PDF is not available");

        SameDiff visionEncoder = OnnxModelCache.importWithCache(
                new File(modelDir, "vision_encoder.onnx").getAbsolutePath());
        visionEncoder.setDspAutoCompileEnabled(true);
        visionEncoder.setDspNativeAutoCompileEnabled(true);
        PreprocessorConfig config = PreprocessorConfig.fromFile(
                new File(modelDir, "preprocessor_config.json"));
        VLMImagePreprocessor preprocessor = VLMImagePreprocessor.fromConfig(config);

        try (PDDocument document = PDDocument.load(pdfFile)) {
            BufferedImage page = new PDFRenderer(document)
                    .renderImageWithDPI(7, 144, ImageType.RGB);
            assertEquals(1224, page.getWidth(), "Unexpected page-8 render width");
            BufferedImage column = page.getSubimage(0, 0, 522, 1584);

            // Production dense OCR encodes the full page before re-entering the
            // vision encoder with this nine-frame first column.
            SameDiffMemoryUtils.safeClose(encodeRegion(
                    visionEncoder, preprocessor, page, "full-page"));

            // Decoder pipeline construction applies its process-global OPTIMAL
            // profile after full-page vision capture. An optional isolated cache
            // directory lets diagnostics compile current IR without deleting or
            // overwriting the normal Triton cache.
            BenchmarkConfigApplier.apply(BenchmarkConfig.optimal());
            String isolatedCacheDir = System.getProperty("vlm.test.tritonCacheDir", "");
            if (!isolatedCacheDir.isEmpty()) {
                Nd4j.getEnvironment().setTritonCacheDir(isolatedCacheDir);
            }
            if (Boolean.getBoolean("vlm.test.disableConsolidatedArgTable")) {
                Nd4j.getEnvironment().setTritonConsolidatedArgTable(false);
            }
            if (Boolean.getBoolean("vlm.test.disableSectionFusion")) {
                Nd4j.getEnvironment().setTritonSectionFusion(false);
            }
            if (Boolean.getBoolean("vlm.test.disableGraphCapture")) {
                Nd4j.getEnvironment().setTritonGraphCapture(false);
            }
            INDArray columnAfterFullPage = encodeRegion(
                    visionEncoder, preprocessor, column, "column-after-full-page");
            dumpFingerprintsIfRequested(visionEncoder);

            // Rebuild only the vision session/plan while retaining the same
            // process-global profile. Comparing equivalent compilation profiles
            // isolates lifecycle corruption from valid floating-point differences
            // between Triton partitioning/reduction configurations.
            visionEncoder.resetSession();
            visionEncoder.clearDynamicShapePlanCache();
            INDArray columnReference = encodeRegion(
                    visionEncoder, preprocessor, column, "column-reference");
            dumpFingerprintsIfRequested(visionEncoder);
            try {
                double difference = maxAbsDifference(columnReference, columnAfterFullPage);
                log.info("Full-page to column comparison: referenceSum={}, replayedSum={}, difference={}",
                        columnReference.sumNumber(), columnAfterFullPage.sumNumber(), difference);
                assertEquals(0.0, difference, 1e-3,
                        "Vision output changed after the dense OCR full-page to column transition");
            } finally {
                SameDiffMemoryUtils.safeClose(columnReference);
                SameDiffMemoryUtils.safeClose(columnAfterFullPage);
            }
        } finally {
            preprocessor.shutdown();
            SameDiffMemoryUtils.freeVisionEncoder(visionEncoder);
        }
    }

    @Test
    void testFullPageColumnLeftLeafVisionSequenceMatchesRebuiltLeaf() throws Exception {
        File modelDir = new File(System.getProperty("vlm.test.modelDir", DEFAULT_MODEL_DIR));
        File pdfFile = new File(System.getProperty("vlm.test.pdf.path", DEFAULT_PDF));
        Assumptions.assumeTrue(new File(modelDir, "vision_encoder.onnx").isFile(),
                "SmolDocling vision encoder is not available");
        Assumptions.assumeTrue(pdfFile.isFile(), "OCR regression PDF is not available");

        SameDiff visionEncoder = OnnxModelCache.importWithCache(
                new File(modelDir, "vision_encoder.onnx").getAbsolutePath());
        visionEncoder.setDspAutoCompileEnabled(true);
        visionEncoder.setDspNativeAutoCompileEnabled(true);
        PreprocessorConfig config = PreprocessorConfig.fromFile(
                new File(modelDir, "preprocessor_config.json"));
        VLMImagePreprocessor preprocessor = VLMImagePreprocessor.fromConfig(config);

        try (PDDocument document = PDDocument.load(pdfFile)) {
            BufferedImage page = new PDFRenderer(document)
                    .renderImageWithDPI(7, 144, ImageType.RGB);
            BufferedImage column = page.getSubimage(0, 0, 522, 1584);
            BufferedImage leftColumn = page.getSubimage(0, 0, 280, 1584);
            BufferedImage leaf = page.getSubimage(0, 0, 280, 538);

            SameDiffMemoryUtils.safeClose(encodeRegion(
                    visionEncoder, preprocessor, page, "full-page"));
            BenchmarkConfigApplier.apply(BenchmarkConfig.optimal());
            SameDiffMemoryUtils.safeClose(encodeRegion(
                    visionEncoder, preprocessor, column, "column"));
            SameDiffMemoryUtils.safeClose(encodeRegion(
                    visionEncoder, preprocessor, leftColumn, "left-column"));
            INDArray sequenceLeaf = encodeRegion(
                    visionEncoder, preprocessor, leaf, "sequence-leaf");

            visionEncoder.resetSession();
            visionEncoder.clearDynamicShapePlanCache();
            INDArray rebuiltLeaf = encodeRegion(
                    visionEncoder, preprocessor, leaf, "rebuilt-leaf");
            try {
                double difference = maxAbsDifference(rebuiltLeaf, sequenceLeaf);
                log.info("Full-page/column/left/leaf vision comparison: rebuiltSum={}, "
                                + "sequenceSum={}, difference={}",
                        rebuiltLeaf.sumNumber(), sequenceLeaf.sumNumber(), difference);
                assertEquals(0.0, difference, 1e-3,
                        "Vision output changed across the exact dense OCR region sequence");
            } finally {
                SameDiffMemoryUtils.safeClose(sequenceLeaf);
                SameDiffMemoryUtils.safeClose(rebuiltLeaf);
            }
        } finally {
            preprocessor.shutdown();
            SameDiffMemoryUtils.freeVisionEncoder(visionEncoder);
        }
    }

    @Test
    void testFourthMultimodalPrefillMatchesFreshLeaf() throws Exception {
        File modelDir = new File(System.getProperty("vlm.test.modelDir", DEFAULT_MODEL_DIR));
        File pdfFile = new File(System.getProperty("vlm.test.pdf.path", DEFAULT_PDF));
        Assumptions.assumeTrue(modelDir.isDirectory(), "SmolDocling model is not available");
        Assumptions.assumeTrue(pdfFile.isFile(), "OCR regression PDF is not available");

        final String prompt = System.getProperty("vlm.test.prompt",
                "Convert this document to DocTags format with structure tags and bounding boxes.");
        try (PDDocument document = PDDocument.load(pdfFile)) {
            BufferedImage page = new PDFRenderer(document)
                    .renderImageWithDPI(7, 144, ImageType.RGB);
            assertEquals(1224, page.getWidth(), "Unexpected page-8 render width");
            BufferedImage column = page.getSubimage(0, 0, 522, 1584);
            BufferedImage leftColumn = page.getSubimage(0, 0, 280, 1584);
            BufferedImage leaf = page.getSubimage(0, 0, 280, 538);
            ImageTiler.SplitImageResult leafSplit =
                    ImageTiler.splitImageForVLMPreservingScale(leaf, TARGET_SIZE, -1);

            try (VisionLanguageModel reusedModel = VisionLanguageModel.loadSmolDocling(modelDir)) {
                reusedModel.setMaxKvLen(4096);
                int fullPageToken = firstGeneratedToken(
                        reusedModel, page, prompt, false, "full-page");
                reusedModel.resetSessionsForDecode();
                int columnToken = firstGeneratedToken(
                        reusedModel, column, prompt, true, "column");
                int leftColumnToken = firstGeneratedToken(
                        reusedModel, leftColumn, prompt, true, "left-column");
                INDArray sequenceLeafVision = reusedModel.encodeImageTiled(leafSplit, TARGET_SIZE);
                try {
                    int reusedLeafToken = firstGeneratedTokenFromEmbeddings(
                            reusedModel, sequenceLeafVision, leafSplit, prompt,
                            "reused-leaf");

                    log.info("Multimodal prefill transition: fullPage={}, column={}, leftColumn={}, "
                                    + "expectedLeaf=216, reusedLeaf={}",
                            fullPageToken, columnToken, leftColumnToken, reusedLeafToken);
                    assertEquals(216, reusedLeafToken,
                            "Fourth SmolDocling prefill diverged from the established page-8 leaf oracle");
                } finally {
                    SameDiffMemoryUtils.safeClose(sequenceLeafVision);
                }
            }
        }
    }

    @Test
    void testRepeatedLeafPrefillCompileTransition() throws Exception {
        File modelDir = new File(System.getProperty("vlm.test.modelDir", DEFAULT_MODEL_DIR));
        File pdfFile = new File(System.getProperty("vlm.test.pdf.path", DEFAULT_PDF));
        Assumptions.assumeTrue(modelDir.isDirectory(), "SmolDocling model is not available");
        Assumptions.assumeTrue(pdfFile.isFile(), "OCR regression PDF is not available");

        final String prompt = System.getProperty("vlm.test.prompt",
                "Convert this document to DocTags format with structure tags and bounding boxes.");
        try (PDDocument document = PDDocument.load(pdfFile);
             VisionLanguageModel model = VisionLanguageModel.loadSmolDocling(modelDir)) {
            BufferedImage page = new PDFRenderer(document)
                    .renderImageWithDPI(7, 144, ImageType.RGB);
            BufferedImage leaf = page.getSubimage(0, 0, 280, 538);
            model.setMaxKvLen(4096);

            int[] tokens = new int[4];
            for (int generation = 0; generation < tokens.length; generation++) {
                tokens[generation] = firstGeneratedToken(
                        model, leaf, prompt, true, "leaf-" + generation);
            }

            for (int generation = 1; generation < tokens.length; generation++) {
                assertEquals(tokens[0], tokens[generation],
                        "Repeated SmolDocling leaf prefill diverged at generation " + generation);
            }
        }
    }

    @Test
    void testRepeatedLeafPrefillWithFixedVisionEmbeddings() throws Exception {
        File modelDir = new File(System.getProperty("vlm.test.modelDir", DEFAULT_MODEL_DIR));
        File pdfFile = new File(System.getProperty("vlm.test.pdf.path", DEFAULT_PDF));
        Assumptions.assumeTrue(modelDir.isDirectory(), "SmolDocling model is not available");
        Assumptions.assumeTrue(pdfFile.isFile(), "OCR regression PDF is not available");

        final String prompt = System.getProperty("vlm.test.prompt",
                "Convert this document to DocTags format with structure tags and bounding boxes.");
        try (PDDocument document = PDDocument.load(pdfFile);
             VisionLanguageModel model = VisionLanguageModel.loadSmolDocling(modelDir)) {
            BufferedImage page = new PDFRenderer(document)
                    .renderImageWithDPI(7, 144, ImageType.RGB);
            BufferedImage leaf = page.getSubimage(0, 0, 280, 538);
            ImageTiler.SplitImageResult split =
                    ImageTiler.splitImageForVLMPreservingScale(leaf, TARGET_SIZE, -1);
            model.setMaxKvLen(4096);
            model.setBenchmarkConfig(BenchmarkConfig.optimal());

            INDArray visionEmbeddings = model.encodeImageTiled(split, TARGET_SIZE);
            try {
                int imageSeqLenPerFrame =
                        (int) (visionEmbeddings.size(1) / split.getTotalFrames());
                SamplingConfig sampling = SamplingConfig.greedy().toBuilder()
                        .maxNewTokens(2)
                        .build();
                int[] tokens = new int[4];
                for (int generation = 0; generation < tokens.length; generation++) {
                    GenerationResult result = model.generateFromEmbeddings(
                            visionEmbeddings, prompt, sampling,
                            split.numRows, split.numCols, imageSeqLenPerFrame);
                    assertNotNull(result, "Fixed-vision generation " + generation + " returned null");
                    assertTrue(result.getTokenIds() != null && result.getTokenIds().length > 0,
                            "Fixed-vision generation " + generation + " returned no tokens");
                    tokens[generation] = result.getTokenIds()[0];
                    log.info("Fixed-vision prefill generation {}: token={}", generation, tokens[generation]);
                }
                for (int generation = 1; generation < tokens.length; generation++) {
                    assertEquals(tokens[0], tokens[generation],
                            "Fixed-vision SmolDocling prefill diverged at generation " + generation);
                }
            } finally {
                SameDiffMemoryUtils.safeClose(visionEmbeddings);
            }
        }
    }

    private int firstGeneratedToken(VisionLanguageModel model,
                                    BufferedImage image,
                                    String prompt,
                                    boolean preserveScale,
                                    String label) {
        ImageTiler.SplitImageResult split = preserveScale
                ? ImageTiler.splitImageForVLMPreservingScale(image, TARGET_SIZE, -1)
                : ImageTiler.splitImageForVLM(image, TARGET_SIZE, -1);
        GenerationResult result = model.generatePagesTiled(
                Collections.singletonList(split), prompt,
                2, false, 0.0, TARGET_SIZE)[0];
        assertNotNull(result, label + " generation returned null");
        assertTrue(result.getTokenIds() != null && result.getTokenIds().length > 0,
                label + " generation returned no token IDs");
        log.info("Multimodal prefill {}: token={}, promptTokens={}, finishReason={}",
                label, result.getTokenIds()[0], result.getPromptTokenCount(),
                result.getFinishReason());
        return result.getTokenIds()[0];
    }

    private int firstGeneratedTokenFromEmbeddings(VisionLanguageModel model,
                                                  INDArray visionEmbeddings,
                                                  ImageTiler.SplitImageResult split,
                                                  String prompt,
                                                  String label) {
        int imageSeqLenPerFrame =
                (int) (visionEmbeddings.size(1) / split.getTotalFrames());
        SamplingConfig sampling = SamplingConfig.greedy().toBuilder()
                .maxNewTokens(2)
                .build();
        GenerationResult result = model.generateFromEmbeddings(
                visionEmbeddings, prompt, sampling,
                split.numRows, split.numCols, imageSeqLenPerFrame);
        assertNotNull(result, label + " generation returned null");
        assertTrue(result.getTokenIds() != null && result.getTokenIds().length > 0,
                label + " generation returned no token IDs");
        log.info("Multimodal prefill {}: token={}, promptTokens={}, finishReason={}",
                label, result.getTokenIds()[0], result.getPromptTokenCount(),
                result.getFinishReason());
        return result.getTokenIds()[0];
    }

    private INDArray encodeRegion(SameDiff visionEncoder,
                                  VLMImagePreprocessor preprocessor,
                                  BufferedImage region,
                                  String label) {
        ImageTiler.SplitImageResult split =
                ImageTiler.splitImageForVLMPreservingScale(region, TARGET_SIZE, -1);
        String outputName = visionEncoder.outputs().get(0);
        List<INDArray> encodedFrames = new ArrayList<>();
        for (int i = 0; i < split.frames.size(); i++) {
            INDArray pixels = preprocessor.preprocess(split.frames.get(i));
            if (pixels.rank() == 3) {
                pixels = pixels.reshape(1, 3, TARGET_SIZE, TARGET_SIZE);
            }
            ImageTiler.ContentRegion content = split.contentRegions.get(i);
            INDArray mask = ImageTiler.createPixelAttentionMask(
                    content.width, content.height, TARGET_SIZE);
            Map<String, INDArray> placeholders = new LinkedHashMap<>();
            placeholders.put("pixel_values", pixels);
            placeholders.put("pixel_attention_mask", mask);
            try {
                INDArray encodedFrame = visionEncoder.output(placeholders, outputName)
                        .get(outputName).dup();
                encodedFrames.add(encodedFrame);
                log.info("Vision frame output: label={}, frame={}/{}, content={}x{}, shape={}, sum={}",
                        label, i + 1, split.frames.size(), content.width, content.height,
                        encodedFrame.shape(), encodedFrame.sumNumber());
            } finally {
                SameDiffMemoryUtils.safeClose(pixels);
                SameDiffMemoryUtils.safeClose(mask);
            }
        }

        INDArray encoded = encodedFrames.size() == 1
                ? encodedFrames.get(0)
                : Nd4j.concat(1, encodedFrames.toArray(new INDArray[0]));
        if (encodedFrames.size() > 1) {
            for (INDArray frame : encodedFrames) {
                SameDiffMemoryUtils.safeClose(frame);
            }
        }
        return encoded;
    }

    private double maxAbsDifference(INDArray expected, INDArray actual) {
        INDArray difference = expected.sub(actual);
        try {
            return Nd4j.math().abs(difference).maxNumber().doubleValue();
        } finally {
            SameDiffMemoryUtils.safeClose(difference);
        }
    }

    private void dumpFingerprintsIfRequested(SameDiff visionEncoder) {
        if (!Boolean.getBoolean("vlm.test.dumpFingerprints")) {
            return;
        }
        org.bytedeco.javacpp.Pointer planHandle = visionEncoder.dsp().getNativePlanHandle();
        if (planHandle == null) {
            log.info("Vision fingerprint ring unavailable: no native plan handle");
            return;
        }
        Nd4j.getNativeOps().drainPlanFingerprintRing(planHandle);
        log.info("Vision fingerprint ring: {}",
                Nd4j.getNativeOps().getPlanFingerprintJson(planHandle));
        int startSlot = Integer.getInteger("vlm.test.dumpPlanStart", -1);
        int endSlot = Integer.getInteger("vlm.test.dumpPlanEnd", -1);
        if (startSlot >= 0 && endSlot >= startSlot) {
            int numSlots = Nd4j.getNativeOps().getPlanNumSlots(planHandle);
            for (int slot = startSlot; slot <= endSlot && slot < numSlots; slot++) {
                log.info("Vision plan slot {}: op={}", slot,
                        Nd4j.getNativeOps().getPlanSlotOpName(planHandle, slot));
            }
        }
    }
}
