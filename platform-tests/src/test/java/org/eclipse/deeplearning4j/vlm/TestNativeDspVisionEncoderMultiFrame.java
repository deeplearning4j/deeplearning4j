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

package org.eclipse.deeplearning4j.vlm;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Reproduces native DSP executor failure when the SmolDocling vision encoder
 * is called multiple times with the same compiled plan.
 *
 * <p>The vision encoder expects rank-4 pixel_values [batch, C, H, W] and has
 * a conv2d op that requires exactly rank-4 input. When the native C++ executor
 * caches shapes from execution 1 and reuses them on execution 2 (even with
 * shapesFrozen=false), upstream reshape/permute ops produce wrong-rank tensors
 * and conv2d fails with "rank of input array must be equal to 4, but got 5".</p>
 *
 * <p>Run with:
 * <pre>
 *   cd platform-tests && mvn test \
 *     -Dtest=TestNativeDspVisionEncoderMultiFrame \
 *     -Dbackend.artifactId=nd4j-cuda-12.9
 * </pre>
 */
@Slf4j
public class TestNativeDspVisionEncoderMultiFrame {

    private static SameDiff visionEncoder;
    private static boolean modelsLoaded = false;

    @BeforeAll
    public static void setup() throws Exception {
        // Download and import the SmolDocling vision encoder
        log.info("Downloading SmolDocling vision encoder...");
        File visionDir = VLMModelDownloader.download(
                VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER).getModelFile();

        log.info("Importing ONNX model...");
        visionEncoder = OnnxModelCache.importWithCache(visionDir.getAbsolutePath());
        log.info("Vision encoder loaded: {} ops", visionEncoder.ops().length);

        // Enable DSP with native execution
        visionEncoder.setDspAutoCompileEnabled(true);
        visionEncoder.setDspNativeAutoCompileEnabled(true);
        log.info("DSP native auto-compile enabled");

        modelsLoaded = true;
    }

    @AfterAll
    public static void teardown() {
        if (visionEncoder != null) {
            visionEncoder.close();
        }
    }

    /**
     * Tests that the native DSP executor can handle multiple consecutive calls
     * to the vision encoder with the same plan. Frame 1 typically succeeds,
     * but frame 2 fails because the C++ native plan retains stale cached shapes.
     *
     * <p>The specific failure chain is:
     * <ol>
     *   <li>Frame 1: pixel_values [1,3,512,512] (rank 4) → native plan compiles and executes OK</li>
     *   <li>Frame 2: pixel_values [1,3,512,512] (rank 4) → native plan reuses stale shape cache,
     *       permute op gets mismatched permutation vector size (4) vs input rank (5),
     *       conv2d receives rank-5 input → KERNEL_FAILURE</li>
     * </ol>
     */
    @Test
    @DisplayName("Native DSP: vision encoder must handle multiple frames without shape cache corruption")
    public void testMultiFrameNativeDspExecution() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");

        // Discover placeholder names
        String pixelValuesName = "pixel_values";
        String pixelAttentionMaskName = "pixel_attention_mask";

        // Discover output name
        List<String> outputNames = List.of("image_features");

        int targetSize = 512;
        int numFrames = 3;

        for (int frame = 0; frame < numFrames; frame++) {
            log.info("=== Frame {}/{} ===", frame + 1, numFrames);

            // Create rank-4 pixel values: [1, 3, 512, 512]
            INDArray pixelValues = Nd4j.rand(DataType.FLOAT, 1, 3, targetSize, targetSize);

            // Create pixel attention mask: [1, 512, 512]
            INDArray pixelMask = Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);

            Map<String, INDArray> inputs = new HashMap<>();
            inputs.put(pixelValuesName, pixelValues);
            inputs.put(pixelAttentionMaskName, pixelMask);

            // This should not throw — if native DSP shape cache is stale,
            // frame 2+ will fail with conv2d rank mismatch
            Map<String, INDArray> outputs;
            try {
                outputs = visionEncoder.output(inputs, outputNames);
            } catch (Exception e) {
                fail("Frame " + (frame + 1) + " failed: " + e.getMessage() +
                        "\nThis indicates the native DSP executor's shape cache is corrupted " +
                        "between consecutive executions of the same plan.", e);
                return;
            }

            INDArray imageFeatures = outputs.get("image_features");
            assertNotNull(imageFeatures, "Frame " + (frame + 1) + ": image_features output is null");

            log.info("Frame {}/{}: output shape={}, min={}, max={}",
                    frame + 1, numFrames,
                    Arrays.toString(imageFeatures.shape()),
                    imageFeatures.minNumber(),
                    imageFeatures.maxNumber());

            // Expected output: [1, 64, 576]
            assertEquals(3, imageFeatures.rank(),
                    "Frame " + (frame + 1) + ": expected rank-3 output");
            assertEquals(1, imageFeatures.size(0),
                    "Frame " + (frame + 1) + ": expected batch=1");

            // Clean up to simulate frame-by-frame processing
            pixelValues.close();
            pixelMask.close();
            for (var entry : outputs.entrySet()) {
                INDArray arr = entry.getValue();
                if (arr != null && !arr.wasClosed()) {
                    arr.setCloseable(true);
                    arr.close();
                }
            }
        }

        log.info("All {} frames succeeded with native DSP execution", numFrames);
    }

    /**
     * Minimal reproduction: calls the vision encoder exactly twice with identical
     * input shapes. If this fails on the second call, the native plan's shape cache
     * is not being invalidated between executions.
     */
    @Test
    @DisplayName("Native DSP: two identical calls must not corrupt shape cache")
    public void testTwoIdenticalCalls() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");

        int targetSize = 512;
        List<String> outputNames = List.of("image_features");

        // Call 1
        INDArray pv1 = Nd4j.rand(DataType.FLOAT, 1, 3, targetSize, targetSize);
        INDArray mask1 = Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);
        Map<String, INDArray> in1 = Map.of(
                "pixel_values", pv1,
                "pixel_attention_mask", mask1);

        Map<String, INDArray> out1;
        try {
            out1 = visionEncoder.output(in1, outputNames);
        } catch (Exception e) {
            fail("Call 1 (should always work) failed: " + e.getMessage(), e);
            return;
        }
        log.info("Call 1 output shape: {}", Arrays.toString(out1.get("image_features").shape()));
        // Close outputs
        out1.values().forEach(a -> { if (a != null && !a.wasClosed()) { a.setCloseable(true); a.close(); } });
        pv1.close();
        mask1.close();

        // Call 2 — same shapes, this is where stale cache causes failure
        INDArray pv2 = Nd4j.rand(DataType.FLOAT, 1, 3, targetSize, targetSize);
        INDArray mask2 = Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);
        Map<String, INDArray> in2 = Map.of(
                "pixel_values", pv2,
                "pixel_attention_mask", mask2);

        Map<String, INDArray> out2;
        try {
            out2 = visionEncoder.output(in2, outputNames);
        } catch (Exception e) {
            fail("Call 2 failed — native DSP shape cache corruption: " + e.getMessage(), e);
            return;
        }
        log.info("Call 2 output shape: {}", Arrays.toString(out2.get("image_features").shape()));

        assertNotNull(out2.get("image_features"));
        assertEquals(3, out2.get("image_features").rank());

        out2.values().forEach(a -> { if (a != null && !a.wasClosed()) { a.setCloseable(true); a.close(); } });
        pv2.close();
        mask2.close();
    }
}
