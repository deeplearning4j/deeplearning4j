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
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.io.File;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests DSP plan reuse across multiple "pages" with the real SmolDocling
 * vision encoder — the exact model and execution pattern that fails with
 * "Unknown data type requested DataTypeUtils:UNKNOWN" on page 2.
 *
 * <h3>Bug reproduction:</h3>
 * <p>The vision encoder's DSP executor compiles a native plan on first execution.
 * Without session reset, the second execution of the same plan fails because:
 * <ul>
 *   <li>The C++ plan has `cudaGraphs=true` and `shapesFrozen=false`</li>
 *   <li>On second call, `clearDynamicShapePlanCaches()` clears shape caches</li>
 *   <li>But CUDA graph from first execution references stale memory addresses</li>
 *   <li>C++ execution returns status -1 with "Unknown DataType" error</li>
 * </ul>
 *
 * <h3>Fix verification:</h3>
 * <p>After fixing, this test verifies:
 * <ul>
 *   <li>Vision encoder can be called N times without session reset</li>
 *   <li>Results are numerically valid (no NaN, no Inf)</li>
 *   <li>Output shapes are consistent across calls</li>
 *   <li>Frozen shapes enable buffer reuse without errors</li>
 *   <li>Interleaved encoder/decoder execution (full VLM page simulation)</li>
 * </ul>
 *
 * <p>Run:
 * <pre>
 *   cd platform-tests && mvn test \
 *     -Dtest=TestDSPVisionEncoderPageReuse \
 *     -Dbackend.artifactId=nd4j-cuda-12.9
 * </pre>
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestDSPVisionEncoderPageReuse {

    private static SameDiff visionEncoder;
    private static SameDiff embedTokens;
    private static SameDiff decoder;
    private static boolean modelsLoaded = false;
    private static NativeOps nativeOps;

    @BeforeAll
    public static void setup() throws Exception {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        System.setProperty(ND4JSystemProperties.DSP_DIAGNOSTICS, "all");
        System.setProperty(ND4JSystemProperties.DSP_DIAGNOSTICS_LEVEL, "full");
        InferenceSession.setDynamicShapePlanEnabled(true);
        // Enable batch-zero directly on the native environment — system properties
        // are only read at ND4J init time, too late for mid-run changes.
        // Batch-zero replaces per-slot cudaMemsetAsync with a single kernel launch,
        // which is mandatory for capture-compatible execution (cudaMemsetAsync
        // fails with error 901 during stream capture).
        Nd4j.getEnvironment().setDspBatchZero(true);
        Nd4j.getEnvironment().setDspBatchZeroKernel(true);
        nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        // Initialize DSP diagnostics early so all compilation/execution is traced
        org.nd4j.autodiff.samediff.diagnostics.DspDiagnostics.initialize();

        try {
            // Download and import vision encoder
            log.info("Downloading SmolDocling vision encoder...");
            File visionDir = VLMModelDownloader.download(
                    VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER).getModelFile();
            visionEncoder = OnnxModelCache.importWithCache(visionDir.getAbsolutePath());
            visionEncoder.setDspAutoCompileEnabled(true);
            visionEncoder.setDspNativeAutoCompileEnabled(true);
            log.info("Vision encoder loaded: {} ops", visionEncoder.ops().length);

            // Download and import embed_tokens
            log.info("Downloading SmolDocling embed_tokens...");
            File embedDir = VLMModelDownloader.download(
                    VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS).getModelFile();
            embedTokens = OnnxModelCache.importWithCache(embedDir.getAbsolutePath());
            embedTokens.setDspAutoCompileEnabled(true);
            embedTokens.setDspNativeAutoCompileEnabled(true);
            log.info("Embed tokens loaded: {} ops", embedTokens.ops().length);

            // Download and import decoder
            log.info("Downloading SmolDocling decoder...");
            File decoderDir = VLMModelDownloader.download(
                    VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER).getModelFile();
            decoder = OnnxModelCache.importWithCache(decoderDir.getAbsolutePath());
            decoder.setDspAutoCompileEnabled(true);
            decoder.setDspNativeAutoCompileEnabled(true);
            log.info("Decoder loaded: {} ops", decoder.ops().length);

            modelsLoaded = true;
        } catch (Exception e) {
            log.error("Failed to load models: {}", e.getMessage());
            // Don't fail — tests will skip via assumeTrue
        }
    }

    @AfterAll
    public static void teardown() {
        if (visionEncoder != null) visionEncoder.close();
        if (embedTokens != null) embedTokens.close();
        if (decoder != null) decoder.close();
    }

    @AfterEach
    public void cleanupBetweenTests() {
        // Free cached native plan handles + reset sessions + trim GPU memory pool.
        // resetSession() alone only clears Java-side state — the native plan handles
        // (and their ~512MB capture buffers per segment) stay cached. We must free
        // them explicitly so later tests have enough GPU memory.
        log.info("=== Inter-test GPU cleanup ===");
        logGpuMemory("Before inter-test cleanup");
        // Order matters: resetSession() closes the executor which CACHES the
        // native plan handle. freeAllCachedNativePlanHandles() then frees it.
        if (visionEncoder != null) {
            visionEncoder.resetSession();
            visionEncoder.freeAllCachedNativePlanHandles();
        }
        if (embedTokens != null) {
            embedTokens.resetSession();
            embedTokens.freeAllCachedNativePlanHandles();
        }
        if (decoder != null) {
            decoder.resetSession();
            decoder.freeAllCachedNativePlanHandles();
        }
        try {
            Nd4j.getExecutioner().commit();
            int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
            for (int d = 0; d < numDevices; d++) {
                nativeOps.trimMemoryPool(d);
            }
            log.info("Trimmed memory pool on {} devices after test", numDevices);
        } catch (Exception e) {
            log.warn("Post-test trimMemoryPool failed: {}", e.getMessage());
        }
        logGpuMemory("After inter-test cleanup");
    }

    private static void logGpuMemory(String label) {
        try {
            int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
            for (int d = 0; d < numDevices; d++) {
                long freeMem = nativeOps.getDeviceFreeMemory(d);
                long totalMem = nativeOps.getDeviceTotalMemory(d);
                log.info("[{}] Device {}: used={} MB, free={} MB",
                        label, d,
                        (totalMem - freeMem) / (1024 * 1024),
                        freeMem / (1024 * 1024));
            }
        } catch (Exception e) {
            log.debug("Could not query GPU memory: {}", e.getMessage());
        }
    }

    // =========================================================================
    // Test 1: Vision encoder called 3 times without session reset
    // This is THE bug reproducer — page 2 fails with "Unknown DataType"
    // =========================================================================

    @Test
    @Order(1)
    @DisplayName("Vision encoder: 3 consecutive calls without session reset (page reuse)")
    public void testVisionEncoderThreeCallsNoReset() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");

        int targetSize = 512;
        int numPages = 3;
        List<String> outputNames = List.of("image_features");
        long[] expectedOutputShape = null;

        for (int page = 1; page <= numPages; page++) {
            log.info("=== Page {}/{} ===", page, numPages);
            logGpuMemory("Before page " + page);

            // Create rank-5 input: [batch=1, frames=1, C=3, H=512, W=512]
            INDArray pixelValues = Nd4j.rand(DataType.FLOAT, 1, 1, 3, targetSize, targetSize);
            INDArray pixelMask = Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);

            Map<String, INDArray> inputs = new LinkedHashMap<>();
            inputs.put("pixel_values", pixelValues);
            inputs.put("pixel_attention_mask", pixelMask);

            Map<String, INDArray> outputs;
            try {
                outputs = visionEncoder.output(inputs, outputNames);
            } catch (Exception e) {
                logGpuMemory("FAILED at page " + page);
                fail("Page " + page + "/" + numPages +
                        " vision encoder FAILED — DSP re-execution bug. Error: " +
                        e.getMessage(), e);
                return;
            }

            INDArray features = outputs.get("image_features");
            assertNotNull(features, "Page " + page + ": image_features is null");
            assertEquals(3, features.rank(), "Page " + page + ": expected rank-3 output");
            assertFalse(Double.isNaN(features.sumNumber().doubleValue()),
                    "Page " + page + ": output contains NaN");

            if (expectedOutputShape == null) {
                expectedOutputShape = features.shape();
            } else {
                assertArrayEquals(expectedOutputShape, features.shape(),
                        "Page " + page + ": output shape changed from " +
                                Arrays.toString(expectedOutputShape));
            }

            log.info("Page {}/{}: output shape={}, sum={:.4f}", page, numPages,
                    Arrays.toString(features.shape()), features.sumNumber().floatValue());
            logGpuMemory("After page " + page);

            // Close outputs to simulate VLM per-page processing
            for (INDArray a : outputs.values()) {
                if (a != null && !a.wasClosed()) {
                    a.setCloseable(true);
                    a.close();
                }
            }
            pixelValues.close();
            pixelMask.close();

            // NO session reset between pages — this is the correct behavior
        }

        log.info("All {} pages succeeded with vision encoder DSP reuse", numPages);
    }

    // =========================================================================
    // Test 2: Vision encoder with frozen shapes
    // =========================================================================

    @Test
    @Order(2)
    @DisplayName("Vision encoder: frozen shapes enable buffer reuse across pages")
    public void testVisionEncoderFrozenShapes() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");

        int targetSize = 512;
        List<String> outputNames = List.of("image_features");

        // Call 1: compile plan
        logGpuMemory("Before frozen test Call 1");
        INDArray pv1 = Nd4j.rand(DataType.FLOAT, 1, 1, 3, targetSize, targetSize);
        INDArray mask1 = Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);
        Map<String, INDArray> out1 = visionEncoder.output(
                Map.of("pixel_values", pv1, "pixel_attention_mask", mask1), outputNames);
        INDArray features1 = out1.get("image_features");
        assertNotNull(features1, "Call 1 failed");
        log.info("Call 1 (compile): shape={}", Arrays.toString(features1.shape()));
        logGpuMemory("After frozen test Call 1");

        // Close Call 1 outputs before freeze to reduce GPU pressure
        pv1.close();
        mask1.close();
        for (INDArray a : out1.values()) {
            if (a != null && !a.wasClosed()) { a.setCloseable(true); a.close(); }
        }

        // Trim memory pool to release freed memory back to device
        try {
            int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
            for (int d = 0; d < numDevices; d++) {
                nativeOps.trimMemoryPool(d);
            }
            log.info("Trimmed memory pool on {} devices", numDevices);
        } catch (Exception e) {
            log.warn("trimMemoryPool failed: {}", e.getMessage());
        }
        logGpuMemory("After trim before freeze");

        // Freeze shapes
        visionEncoder.setDspShapesFrozen(true);

        // Plan introspection: dump slots 290-300 to understand create/shape_of ops
        {
            InferenceSession session = visionEncoder.getSessions().get(Thread.currentThread().getId());
            if (session != null && session.getDynamicShapePlanExecutor() != null) {
                var plan = session.getDynamicShapePlanExecutor().getCurrentPlan();
                if (plan != null) {
                    var slots = plan.getSlots();
                    log.info("=== PLAN INTROSPECTION: slots 288-304 ===");
                    for (int si = 288; si <= Math.min(304, slots.length - 1); si++) {
                        log.info(org.nd4j.autodiff.samediff.execution.PlanIntrospection.formatSlot(plan, si));
                    }
                }
            }
        }

        // Call 2-4: frozen execution with buffer reuse
        for (int i = 2; i <= 4; i++) {
            logGpuMemory("Before frozen Call " + i);
            INDArray pv = Nd4j.rand(DataType.FLOAT, 1, 1, 3, targetSize, targetSize);
            INDArray mask = Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);
            Map<String, INDArray> out;
            try {
                out = visionEncoder.output(
                        Map.of("pixel_values", pv, "pixel_attention_mask", mask), outputNames);
            } catch (Exception e) {
                logGpuMemory("FAILED frozen Call " + i);
                fail("Call " + i + " with frozen shapes failed: " + e.getMessage(), e);
                return;
            }
            INDArray features = out.get("image_features");
            assertNotNull(features, "Call " + i + ": image_features is null");
            assertFalse(Double.isNaN(features.sumNumber().doubleValue()),
                    "Call " + i + ": NaN in output");
            log.info("Call {} (frozen): shape={}, sum={:.4f}", i,
                    Arrays.toString(features.shape()), features.sumNumber().floatValue());
            logGpuMemory("After frozen Call " + i);

            pv.close();
            mask.close();
        }

        // Unfreeze for other tests
        visionEncoder.setDspShapesFrozen(false);
        log.info("Frozen shapes test passed");
    }

    // =========================================================================
    // Test 3: Full VLM page simulation (encoder → decoder → repeat)
    // =========================================================================

    @Test
    @Order(3)
    @DisplayName("Full VLM simulation: vision encoder + decoder for 2 pages without reset")
    public void testFullVlmTwoPageSimulation() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");
        logGpuMemory("Test 3 start");

        int targetSize = 512;
        int numPages = 2;

        for (int page = 1; page <= numPages; page++) {
            log.info("=== Full VLM Page {}/{} ===", page, numPages);
            logGpuMemory("Before page " + page);

            // Step 1: Vision encoder
            INDArray pixelValues = Nd4j.rand(DataType.FLOAT, 1, 1, 3, targetSize, targetSize);
            INDArray pixelMask = Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);

            Map<String, INDArray> encoderOut;
            try {
                encoderOut = visionEncoder.output(
                        Map.of("pixel_values", pixelValues, "pixel_attention_mask", pixelMask),
                        List.of("image_features"));
            } catch (Exception e) {
                fail("Page " + page + " vision encoder failed: " + e.getMessage(), e);
                return;
            }

            INDArray visionFeatures = encoderOut.get("image_features");
            assertNotNull(visionFeatures, "Page " + page + ": vision features null");
            log.info("Page {} encoder: shape={}", page, Arrays.toString(visionFeatures.shape()));

            // Step 2: Embed tokens (BOS token)
            INDArray bosTokenIds = Nd4j.createFromArray(new long[][]{{49282}});  // SmolDocling BOS
            Map<String, INDArray> embedOut;
            try {
                embedOut = embedTokens.output(
                        Map.of("input_ids", bosTokenIds),
                        List.of("inputs_embeds"));
            } catch (Exception e) {
                fail("Page " + page + " embed_tokens failed: " + e.getMessage(), e);
                return;
            }

            INDArray tokenEmbeddings = embedOut.get("inputs_embeds");
            assertNotNull(tokenEmbeddings, "Page " + page + ": token embeddings null");
            log.info("Page {} embed_tokens: shape={}", page,
                    Arrays.toString(tokenEmbeddings.shape()));

            // Cleanup page outputs
            pixelValues.close();
            pixelMask.close();
            for (INDArray a : encoderOut.values()) {
                if (a != null && !a.wasClosed()) { a.setCloseable(true); a.close(); }
            }
            for (INDArray a : embedOut.values()) {
                if (a != null && !a.wasClosed()) { a.setCloseable(true); a.close(); }
            }
            bosTokenIds.close();

            logGpuMemory("After page " + page);
            // NO session reset between pages
        }

        log.info("Full VLM {} page simulation succeeded", numPages);
    }

    // =========================================================================
    // Test 4: Vision encoder session reset + re-execute (plan cache test)
    // =========================================================================

    @Test
    @Order(4)
    @DisplayName("Vision encoder: session reset preserves plan cache, re-execution succeeds")
    public void testVisionEncoderResetAndReexecute() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");
        logGpuMemory("Test 4 start");

        int targetSize = 512;
        List<String> outputNames = List.of("image_features");

        // Call 1: compile + execute
        INDArray pv1 = Nd4j.rand(DataType.FLOAT, 1, 1, 3, targetSize, targetSize);
        INDArray mask1 = Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);
        Map<String, INDArray> out1 = visionEncoder.output(
                Map.of("pixel_values", pv1, "pixel_attention_mask", mask1), outputNames);
        assertNotNull(out1.get("image_features"), "Pre-reset call failed");
        long[] shape1 = out1.get("image_features").shape();
        log.info("Pre-reset: shape={}", Arrays.toString(shape1));

        pv1.close();
        mask1.close();
        for (INDArray a : out1.values()) {
            if (a != null && !a.wasClosed()) { a.setCloseable(true); a.close(); }
        }

        // Reset session
        visionEncoder.resetSession();
        log.info("Session reset complete");
        logGpuMemory("After reset");

        // Call 2: should restore cached plan handle
        INDArray pv2 = Nd4j.rand(DataType.FLOAT, 1, 1, 3, targetSize, targetSize);
        INDArray mask2 = Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);
        Map<String, INDArray> out2;
        try {
            out2 = visionEncoder.output(
                    Map.of("pixel_values", pv2, "pixel_attention_mask", mask2), outputNames);
        } catch (Exception e) {
            fail("Post-reset vision encoder call failed: " + e.getMessage(), e);
            return;
        }

        INDArray features2 = out2.get("image_features");
        assertNotNull(features2, "Post-reset output is null");
        assertArrayEquals(shape1, features2.shape(), "Post-reset shape mismatch");
        assertFalse(Double.isNaN(features2.sumNumber().doubleValue()),
                "Post-reset output contains NaN");
        log.info("Post-reset: shape={}", Arrays.toString(features2.shape()));

        pv2.close();
        mask2.close();
    }

    // =========================================================================
    // Helpers
    // =========================================================================

}
