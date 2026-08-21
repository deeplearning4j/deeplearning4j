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
import org.eclipse.deeplearning4j.llm.generation.SameDiffMemoryUtils;
import org.eclipse.deeplearning4j.vlm.model.loading.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

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
    }
}
