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
 * <p><b>Bug</b>: The first call to visionEncoder.output() succeeds, but the
 * second call (with identical input shapes) fails at a divide op (slot ~1241)
 * with KERNEL_FAILURE status 1. This happens because the native DSP plan
 * retains stale outputSlots_ pointers from execution 1 that are freed/invalid
 * by execution 2.</p>
 *
 * <p>The ONNX model declares pixel_values as rank-5: [batch_size, num_images, 3, 512, 512].
 * VisionLanguageModel.normalizeVisionInputShape() reshapes rank-4 to rank-5 before calling
 * output(). This test uses the same rank-5 input shape [1, 1, 3, 512, 512].</p>
 *
 * <p>Run with:
 * <pre>
 *   cd platform-tests && mvn test \
 *     -Dtest=TestNativeDspVisionEncoderMultiFrame \
 *     -Dbackend.artifactId=nd4j-cuda-12.9
 * </pre>
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestNativeDspVisionEncoderMultiFrame {

    private static SameDiff visionEncoder;
    private static boolean modelsLoaded = false;

    @BeforeAll
    public static void setup() throws Exception {
        // Download and import the SmolDocling vision encoder
        log.info("Downloading SmolDocling vision encoder...");
        File visionDir = VLMModelDownloader.download(
                VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER).getModelFile();

        log.info("Importing ONNX model from: {}", visionDir);
        visionEncoder = OnnxModelCache.importWithCache(visionDir.getAbsolutePath());
        log.info("Vision encoder loaded: {} ops", visionEncoder.ops().length);

        // Log placeholder shapes to confirm rank-5
        var pixelVar = visionEncoder.getVariable("pixel_values");
        if (pixelVar != null) {
            log.info("pixel_values placeholder shape: {}", Arrays.toString(pixelVar.getShape()));
        }

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
            log.debug("Could not query GPU memory: {}", e.getMessage());
        }
    }

    /**
     * Core reproducer: calls the vision encoder exactly twice with rank-5
     * input [1, 1, 3, 512, 512] — the same shape VisionLanguageModel uses.
     *
     * Call 1 succeeds. Call 2 fails at divide op (slot ~1241) with status 1
     * if the native DSP plan has stale state from execution 1.
     */
    @Test
    @Order(1)
    @DisplayName("DSP re-execution: two identical rank-5 calls must both succeed")
    public void testTwoIdenticalRank5Calls() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");

        int targetSize = 512;
        List<String> outputNames = List.of("image_features");

        // === Call 1 ===
        log.info("=== Call 1: rank-5 [1,1,3,{},{}] ===", targetSize, targetSize);
        logGpuMemory("Before call 1");

        INDArray pv1 = Nd4j.rand(DataType.FLOAT, 1, 1, 3, targetSize, targetSize);
        INDArray mask1 = Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);
        Map<String, INDArray> in1 = new LinkedHashMap<>();
        in1.put("pixel_values", pv1);
        in1.put("pixel_attention_mask", mask1);

        Map<String, INDArray> out1;
        try {
            out1 = visionEncoder.output(in1, outputNames);
        } catch (Exception e) {
            fail("Call 1 failed (should always work): " + e.getMessage(), e);
            return;
        }

        INDArray features1 = out1.get("image_features");
        assertNotNull(features1, "Call 1: image_features is null");
        log.info("Call 1 SUCCESS: output shape={}", Arrays.toString(features1.shape()));
        logGpuMemory("After call 1");

        // Close call 1 outputs
        for (INDArray a : out1.values()) {
            if (a != null && !a.wasClosed()) { a.setCloseable(true); a.close(); }
        }
        pv1.close();
        mask1.close();

        // === Call 2 — same shapes, this is where the bug manifests ===
        log.info("=== Call 2: rank-5 [1,1,3,{},{}] (same shape as call 1) ===", targetSize, targetSize);
        logGpuMemory("Before call 2");

        INDArray pv2 = Nd4j.rand(DataType.FLOAT, 1, 1, 3, targetSize, targetSize);
        INDArray mask2 = Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);
        Map<String, INDArray> in2 = new LinkedHashMap<>();
        in2.put("pixel_values", pv2);
        in2.put("pixel_attention_mask", mask2);

        Map<String, INDArray> out2;
        try {
            out2 = visionEncoder.output(in2, outputNames);
        } catch (Exception e) {
            logGpuMemory("FAILED at call 2");
            fail("Call 2 failed — DSP re-execution bug. " +
                    "Native plan retains stale state from call 1. " +
                    "Error: " + e.getMessage(), e);
            return;
        }

        INDArray features2 = out2.get("image_features");
        assertNotNull(features2, "Call 2: image_features is null");
        log.info("Call 2 SUCCESS: output shape={}", Arrays.toString(features2.shape()));
        logGpuMemory("After call 2");

        // Verify both calls produce same-shaped output
        assertEquals(3, features2.rank(), "Expected rank-3 output [batch, seq, dim]");

        for (INDArray a : out2.values()) {
            if (a != null && !a.wasClosed()) { a.setCloseable(true); a.close(); }
        }
        pv2.close();
        mask2.close();
    }

    /**
     * Extended reproducer: 3 frames sequentially, like the VLM tiled pipeline.
     * If call 2 fails, this confirms the re-execution bug.
     * If calls 2-3 fail but call 1 succeeds, the native plan state is not
     * properly reset between executions.
     */
    @Test
    @Order(2)
    @DisplayName("DSP re-execution: three sequential rank-5 frames")
    public void testThreeSequentialRank5Frames() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");

        int targetSize = 512;
        int numFrames = 3;
        List<String> outputNames = List.of("image_features");

        for (int frame = 0; frame < numFrames; frame++) {
            log.info("=== Frame {}/{}: rank-5 [1,1,3,{},{}] ===",
                    frame + 1, numFrames, targetSize, targetSize);
            logGpuMemory("Before frame " + (frame + 1));

            INDArray pixelValues = Nd4j.rand(DataType.FLOAT, 1, 1, 3, targetSize, targetSize);
            INDArray pixelMask = Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);

            Map<String, INDArray> inputs = new LinkedHashMap<>();
            inputs.put("pixel_values", pixelValues);
            inputs.put("pixel_attention_mask", pixelMask);

            Map<String, INDArray> outputs;
            try {
                outputs = visionEncoder.output(inputs, outputNames);
            } catch (Exception e) {
                logGpuMemory("FAILED at frame " + (frame + 1));
                fail("Frame " + (frame + 1) + "/" + numFrames + " failed. " +
                        "DSP native plan re-execution bug — stale state from prior execution. " +
                        "Error: " + e.getMessage(), e);
                return;
            }

            INDArray imageFeatures = outputs.get("image_features");
            assertNotNull(imageFeatures, "Frame " + (frame + 1) + ": image_features is null");

            log.info("Frame {}/{} SUCCESS: output shape={}, min={:.4f}, max={:.4f}",
                    frame + 1, numFrames,
                    Arrays.toString(imageFeatures.shape()),
                    imageFeatures.minNumber().floatValue(),
                    imageFeatures.maxNumber().floatValue());
            logGpuMemory("After frame " + (frame + 1));

            // Expected output: [1, 64, 576]
            assertEquals(3, imageFeatures.rank(),
                    "Frame " + (frame + 1) + ": expected rank-3 output");
            assertEquals(1, imageFeatures.size(0),
                    "Frame " + (frame + 1) + ": expected batch=1");

            // Close to simulate per-frame processing like VisionLanguageModel.encodeImageTiled()
            for (INDArray a : outputs.values()) {
                if (a != null && !a.wasClosed()) { a.setCloseable(true); a.close(); }
            }
            pixelValues.close();
            pixelMask.close();
        }

        log.info("All {} frames succeeded with native DSP re-execution", numFrames);
    }

    /**
     * Control test: verify that non-DSP (Java InferenceSession) execution works
     * correctly for multiple calls. This confirms the bug is in the native DSP
     * plan, not in the SameDiff graph or ONNX import.
     */
    @Test
    @Order(3)
    @DisplayName("Control: Java InferenceSession handles re-execution correctly")
    public void testJavaSessionReexecution() throws Exception {
        // Load a separate instance with DSP disabled
        File visionDir = VLMModelDownloader.download(
                VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER).getModelFile();
        SameDiff javaEncoder = OnnxModelCache.importWithCache(visionDir.getAbsolutePath());

        // Explicitly disable DSP
        javaEncoder.setDspAutoCompileEnabled(false);
        javaEncoder.setDspNativeAutoCompileEnabled(false);
        log.info("Control test: DSP disabled, using Java InferenceSession");

        int targetSize = 512;
        List<String> outputNames = List.of("image_features");

        for (int call = 1; call <= 2; call++) {
            log.info("=== Control call {} ===", call);

            INDArray pv = Nd4j.rand(DataType.FLOAT, 1, 1, 3, targetSize, targetSize);
            INDArray mask = Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);
            Map<String, INDArray> inputs = new LinkedHashMap<>();
            inputs.put("pixel_values", pv);
            inputs.put("pixel_attention_mask", mask);

            Map<String, INDArray> outputs = javaEncoder.output(inputs, outputNames);

            INDArray features = outputs.get("image_features");
            assertNotNull(features, "Control call " + call + ": image_features is null");
            log.info("Control call {} SUCCESS: shape={}", call, Arrays.toString(features.shape()));

            for (INDArray a : outputs.values()) {
                if (a != null && !a.wasClosed()) { a.setCloseable(true); a.close(); }
            }
            pv.close();
            mask.close();

            javaEncoder.resetSession();
        }

        javaEncoder.close();
    }

    /**
     * Test with resetSession() between DSP calls — does resetting the session
     * fix the re-execution issue?
     */
    @Test
    @Order(4)
    @DisplayName("DSP re-execution with resetSession() between calls")
    public void testDspWithResetSessionBetweenCalls() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");

        int targetSize = 512;
        List<String> outputNames = List.of("image_features");

        for (int call = 1; call <= 2; call++) {
            log.info("=== DSP+reset call {} ===", call);
            logGpuMemory("Before DSP+reset call " + call);

            INDArray pv = Nd4j.rand(DataType.FLOAT, 1, 1, 3, targetSize, targetSize);
            INDArray mask = Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);
            Map<String, INDArray> inputs = new LinkedHashMap<>();
            inputs.put("pixel_values", pv);
            inputs.put("pixel_attention_mask", mask);

            Map<String, INDArray> outputs;
            try {
                outputs = visionEncoder.output(inputs, outputNames);
            } catch (Exception e) {
                logGpuMemory("FAILED at DSP+reset call " + call);
                fail("DSP+reset call " + call + " failed: " + e.getMessage(), e);
                return;
            }

            INDArray features = outputs.get("image_features");
            assertNotNull(features, "DSP+reset call " + call + ": image_features is null");
            log.info("DSP+reset call {} SUCCESS: shape={}", call, Arrays.toString(features.shape()));

            for (INDArray a : outputs.values()) {
                if (a != null && !a.wasClosed()) { a.setCloseable(true); a.close(); }
            }
            pv.close();
            mask.close();

            // Reset session between calls — this is what VisionLanguageModel.resetSessions() does
            visionEncoder.resetSession();
            logGpuMemory("After resetSession call " + call);
        }

        log.info("DSP with resetSession() between calls: both succeeded");
    }
}
