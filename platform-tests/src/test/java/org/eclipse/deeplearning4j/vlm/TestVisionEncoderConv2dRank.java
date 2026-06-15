package org.eclipse.deeplearning4j.vlm;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.loading.OnnxModelCache;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Focused test: run SmolDocling vision encoder with both rank-4 and rank-5 input
 * to debug the conv2d rank mismatch in native DSP execution.
 *
 * Run:
 *   cd platform-tests && mvn test \
 *     -Dtest=TestVisionEncoderConv2dRank \
 *     -Dbackend.artifactId=nd4j-cuda-12.9
 */
@Slf4j
public class TestVisionEncoderConv2dRank {

    private static SameDiff visionEncoder;

    @BeforeAll
    public static void setup() throws Exception {
        File visionDir = VLMModelDownloader.download(
                VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER).getModelFile();
        visionEncoder = OnnxModelCache.importWithCache(visionDir.getAbsolutePath());
        log.info("Vision encoder loaded: {} ops", visionEncoder.ops().length);
        log.info("Inputs: {}", visionEncoder.inputs());
        log.info("Outputs: {}", visionEncoder.outputs());

        visionEncoder.setDspAutoCompileEnabled(true);
        visionEncoder.setDspNativeAutoCompileEnabled(true);
    }

    @AfterAll
    public static void teardown() {
        if (visionEncoder != null) visionEncoder.close();
    }

    @Test
    @DisplayName("Vision encoder with rank-5 input [1,1,3,512,512] - matches benchmark test")
    public void testRank5Input() {
        int targetSize = 512;
        INDArray pixelValues = Nd4j.rand(DataType.FLOAT, 1, 1, 3, targetSize, targetSize);
        INDArray pixelMask = Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);

        log.info("pixel_values shape: {} rank: {}", Arrays.toString(pixelValues.shape()), pixelValues.rank());
        log.info("pixel_attention_mask shape: {} rank: {}", Arrays.toString(pixelMask.shape()), pixelMask.rank());

        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("pixel_values", pixelValues);
        inputs.put("pixel_attention_mask", pixelMask);

        List<String> outputNames = List.of("image_features");

        Map<String, INDArray> outputs;
        try {
            outputs = visionEncoder.output(inputs, outputNames);
        } catch (Exception e) {
            log.error("Rank-5 input failed: {}", e.getMessage(), e);
            fail("Rank-5 [1,1,3,512,512] failed: " + e.getMessage(), e);
            return;
        }

        INDArray imageFeatures = outputs.get("image_features");
        assertNotNull(imageFeatures);
        log.info("Rank-5 output shape: {}", Arrays.toString(imageFeatures.shape()));

        pixelValues.close();
        pixelMask.close();
    }

    @Test
    @DisplayName("Vision encoder with rank-4 input [1,3,512,512] - matches VisionLanguageModel path")
    public void testRank4Input() {
        int targetSize = 512;
        INDArray pixelValues = Nd4j.rand(DataType.FLOAT, 1, 3, targetSize, targetSize);
        INDArray pixelMask = Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);

        log.info("pixel_values shape: {} rank: {}", Arrays.toString(pixelValues.shape()), pixelValues.rank());
        log.info("pixel_attention_mask shape: {} rank: {}", Arrays.toString(pixelMask.shape()), pixelMask.rank());

        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("pixel_values", pixelValues);
        inputs.put("pixel_attention_mask", pixelMask);

        List<String> outputNames = List.of("image_features");

        Map<String, INDArray> outputs;
        try {
            outputs = visionEncoder.output(inputs, outputNames);
        } catch (Exception e) {
            log.error("Rank-4 input failed: {}", e.getMessage(), e);
            fail("Rank-4 [1,3,512,512] failed: " + e.getMessage(), e);
            return;
        }

        INDArray imageFeatures = outputs.get("image_features");
        assertNotNull(imageFeatures);
        log.info("Rank-4 output shape: {}", Arrays.toString(imageFeatures.shape()));

        pixelValues.close();
        pixelMask.close();
    }
}
