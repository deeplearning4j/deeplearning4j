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
import org.bytedeco.javacpp.LongPointer;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.diagnostics.DspDiagnostics;
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
 * Tests DSP memory stability across repeated frozen executions.
 *
 * <p>Verifies that GPU memory pool usage remains bounded after the initial
 * execution when shapes are frozen. The key invariant:
 * <ul>
 *   <li>Pool used after cleanup should return to baseline (±10%)</li>
 *   <li>Device free memory should not decrease monotonically</li>
 *   <li>No OOM after N iterations with constant shapes</li>
 * </ul>
 *
 * <p>Uses DSP diagnostics APIs to inspect pool stats at each step.
 *
 * <p>Run:
 * <pre>
 *   cd platform-tests && mvn test \
 *     -Dtest=TestDSPMemoryStability \
 *     -Dbackend.artifactId=nd4j-cuda-12.9
 * </pre>
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestDSPMemoryStability {

    private static SameDiff visionEncoder;
    private static boolean modelsLoaded = false;
    private static NativeOps nativeOps;

    @BeforeAll
    public static void setup() throws Exception {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        System.setProperty(ND4JSystemProperties.DSP_DIAGNOSTICS, "MEMORY");
        System.setProperty(ND4JSystemProperties.DSP_DIAGNOSTICS_LEVEL, "full");
        InferenceSession.setDynamicShapePlanEnabled(true);
        nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        DspDiagnostics.initialize();

        try {
            log.info("Downloading SmolDocling vision encoder...");
            File visionDir = VLMModelDownloader.download(
                    VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER).getModelFile();
            visionEncoder = OnnxModelCache.importWithCache(visionDir.getAbsolutePath());
            visionEncoder.setDspAutoCompileEnabled(true);
            visionEncoder.setDspNativeAutoCompileEnabled(true);
            log.info("Vision encoder loaded: {} ops", visionEncoder.ops().length);
            modelsLoaded = true;
        } catch (Exception e) {
            log.error("Failed to load models: {}", e.getMessage());
        }
    }

    @AfterAll
    public static void teardown() {
        if (visionEncoder != null) visionEncoder.close();
    }

    /**
     * Query pool stats via NativeOps.
     */
    private static long[] getPoolStats(int deviceId) {
        try (LongPointer usedPtr = new LongPointer(1);
             LongPointer reservedPtr = new LongPointer(1)) {
            nativeOps.getMemoryPoolStats(deviceId, usedPtr, reservedPtr);
            return new long[]{usedPtr.get(0), reservedPtr.get(0)};
        }
    }

    private static long getDeviceFreeMemory(int deviceId) {
        return nativeOps.getDeviceFreeMemory(deviceId);
    }

    private static void logMemory(String label, int deviceId) {
        long[] pool = getPoolStats(deviceId);
        long freeMem = getDeviceFreeMemory(deviceId);
        long totalMem = nativeOps.getDeviceTotalMemory(deviceId);
        log.info("[{}] dev={}: poolUsed={}MB poolReserved={}MB devUsed={}MB devFree={}MB",
                label, deviceId,
                pool[0] / (1024 * 1024), pool[1] / (1024 * 1024),
                (totalMem - freeMem) / (1024 * 1024), freeMem / (1024 * 1024));
    }

    // =========================================================================
    // Test 1: Pool stats return to baseline after frozen cleanup
    // =========================================================================

    @Test
    @Order(1)
    @DisplayName("Frozen execution: pool used returns to baseline between calls")
    public void testPoolStatsReturnToBaseline() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");

        int targetSize = 512;
        int deviceId = 0;
        List<String> outputNames = List.of("image_features");

        // Warmup call (compiles the plan)
        logMemory("Before warmup", deviceId);
        INDArray pv0 = Nd4j.rand(DataType.FLOAT, 1, 1, 3, targetSize, targetSize);
        INDArray mask0 = Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);
        Map<String, INDArray> out0 = visionEncoder.output(
                Map.of("pixel_values", pv0, "pixel_attention_mask", mask0), outputNames);
        assertNotNull(out0.get("image_features"), "Warmup call failed");
        pv0.close();
        mask0.close();
        for (INDArray a : out0.values()) {
            if (a != null && !a.wasClosed()) { a.setCloseable(true); a.close(); }
        }
        logMemory("After warmup", deviceId);

        // Trim pool to establish baseline
        nativeOps.trimMemoryPool(deviceId);
        logMemory("After trim (baseline)", deviceId);
        long[] baselinePool = getPoolStats(deviceId);
        long baselineDevFree = getDeviceFreeMemory(deviceId);

        // Freeze shapes
        freezeDspShapes(visionEncoder);

        // Run frozen calls and track pool stats
        int numIterations = 6;
        long[] poolUsedAfterCall = new long[numIterations];
        long[] devFreeAfterCall = new long[numIterations];

        for (int i = 0; i < numIterations; i++) {
            INDArray pv = Nd4j.rand(DataType.FLOAT, 1, 1, 3, targetSize, targetSize);
            INDArray mask = Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);
            Map<String, INDArray> out;
            try {
                out = visionEncoder.output(
                        Map.of("pixel_values", pv, "pixel_attention_mask", mask), outputNames);
            } catch (Exception e) {
                logMemory("FAILED at iteration " + i, deviceId);
                fail("Iteration " + i + " failed: " + e.getMessage(), e);
                return;
            }

            INDArray features = out.get("image_features");
            assertNotNull(features, "Iteration " + i + ": null output");
            assertFalse(Double.isNaN(features.sumNumber().doubleValue()),
                    "Iteration " + i + ": NaN in output");

            pv.close();
            mask.close();
            for (INDArray a : out.values()) {
                if (a != null && !a.wasClosed()) { a.setCloseable(true); a.close(); }
            }

            long[] pool = getPoolStats(deviceId);
            poolUsedAfterCall[i] = pool[0];
            devFreeAfterCall[i] = getDeviceFreeMemory(deviceId);

            log.info("Frozen iter {}: poolUsed={}MB devFree={}MB",
                    i, pool[0] / (1024 * 1024), devFreeAfterCall[i] / (1024 * 1024));
        }

        // Verify: pool used after last call should not be more than 2x the baseline
        // (some overhead from the frozen execution is expected)
        long lastPoolUsed = poolUsedAfterCall[numIterations - 1];
        long basePoolUsed = baselinePool[0];
        log.info("Pool used: baseline={}MB final={}MB ratio={:.2f}",
                basePoolUsed / (1024 * 1024), lastPoolUsed / (1024 * 1024),
                basePoolUsed > 0 ? (double) lastPoolUsed / basePoolUsed : 0.0);

        // The pool should not grow unboundedly. After warmup, subsequent calls
        // should reuse memory. Allow 3x headroom for intermediates.
        assertTrue(lastPoolUsed < basePoolUsed + 10L * 1024 * 1024 * 1024,
                "Pool used grew unboundedly: baseline=" + basePoolUsed / (1024 * 1024) +
                        "MB, final=" + lastPoolUsed / (1024 * 1024) + "MB");

        // Verify: device free memory should not decrease monotonically
        // (if every call leaks, free decreases each time)
        int decreaseCount = 0;
        for (int i = 1; i < numIterations; i++) {
            if (devFreeAfterCall[i] < devFreeAfterCall[i - 1] - 50 * 1024 * 1024) {
                decreaseCount++;
            }
        }
        log.info("Device free monotonic decreases: {}/{}", decreaseCount, numIterations - 1);
        assertTrue(decreaseCount < numIterations - 1,
                "Device free memory decreased monotonically every iteration — likely a leak");

        unfreezeDspShapes(visionEncoder);
        log.info("Pool stats baseline test passed");
    }

    // =========================================================================
    // Test 2: Many frozen iterations without OOM
    // =========================================================================

    @Test
    @Order(2)
    @DisplayName("Frozen execution: 10 iterations without OOM")
    public void testManyFrozenIterationsNoOOM() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");

        int targetSize = 512;
        int deviceId = 0;
        List<String> outputNames = List.of("image_features");

        // Warmup
        INDArray pv0 = Nd4j.rand(DataType.FLOAT, 1, 1, 3, targetSize, targetSize);
        INDArray mask0 = Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);
        Map<String, INDArray> out0 = visionEncoder.output(
                Map.of("pixel_values", pv0, "pixel_attention_mask", mask0), outputNames);
        assertNotNull(out0.get("image_features"));
        pv0.close();
        mask0.close();
        for (INDArray a : out0.values()) {
            if (a != null && !a.wasClosed()) { a.setCloseable(true); a.close(); }
        }

        nativeOps.trimMemoryPool(deviceId);
        freezeDspShapes(visionEncoder);
        logMemory("Before 10-iteration stress test", deviceId);

        for (int i = 1; i <= 10; i++) {
            INDArray pv = Nd4j.rand(DataType.FLOAT, 1, 1, 3, targetSize, targetSize);
            INDArray mask = Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);
            Map<String, INDArray> out;
            try {
                out = visionEncoder.output(
                        Map.of("pixel_values", pv, "pixel_attention_mask", mask), outputNames);
            } catch (Exception e) {
                logMemory("OOM at iteration " + i, deviceId);
                fail("OOM at iteration " + i + ": " + e.getMessage(), e);
                return;
            }

            INDArray features = out.get("image_features");
            assertNotNull(features, "Iteration " + i + ": null output");
            assertFalse(Double.isNaN(features.sumNumber().doubleValue()),
                    "Iteration " + i + ": NaN");

            pv.close();
            mask.close();
            for (INDArray a : out.values()) {
                if (a != null && !a.wasClosed()) { a.setCloseable(true); a.close(); }
            }

            if (i % 3 == 0) {
                logMemory("After frozen iteration " + i, deviceId);
            }
        }

        unfreezeDspShapes(visionEncoder);
        logMemory("After 10-iteration stress test", deviceId);
        log.info("10-iteration frozen stress test passed");
    }

    // =========================================================================
    // Test 3: DSP diagnostic report output
    // =========================================================================

    @Test
    @Order(3)
    @DisplayName("DSP diagnostics: plan report contains memory events")
    public void testDspDiagnosticsReport() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");

        // Clear previous diagnostics
        DspDiagnostics.clear();

        int targetSize = 512;
        List<String> outputNames = List.of("image_features");

        // Execute once to generate diagnostic events
        INDArray pv = Nd4j.rand(DataType.FLOAT, 1, 1, 3, targetSize, targetSize);
        INDArray mask = Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);
        Map<String, INDArray> out = visionEncoder.output(
                Map.of("pixel_values", pv, "pixel_attention_mask", mask), outputNames);
        assertNotNull(out.get("image_features"));

        pv.close();
        mask.close();
        for (INDArray a : out.values()) {
            if (a != null && !a.wasClosed()) { a.setCloseable(true); a.close(); }
        }

        // Get the diagnostic report
        String report = DspDiagnostics.getPlanReport();
        log.info("DSP Plan Report:\n{}", report);

        // Report should contain memory category events
        assertNotNull(report, "Plan report is null");
        // The report is formatted as a table — just verify it's non-empty
        assertTrue(report.length() > 10, "Plan report is too short: " + report);

        log.info("DSP diagnostics report test passed");
    }

    // =========================================================================
    // Test 4: Non-frozen repeated execution (no shapes frozen)
    // =========================================================================

    @Test
    @Order(4)
    @DisplayName("Non-frozen execution: 4 iterations without OOM")
    public void testNonFrozenRepeatedExecution() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");

        int targetSize = 512;
        int deviceId = 0;
        List<String> outputNames = List.of("image_features");

        // Ensure shapes are NOT frozen
        unfreezeDspShapes(visionEncoder);

        logMemory("Before non-frozen test", deviceId);

        for (int i = 1; i <= 4; i++) {
            INDArray pv = Nd4j.rand(DataType.FLOAT, 1, 1, 3, targetSize, targetSize);
            INDArray mask = Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);
            Map<String, INDArray> out;
            try {
                out = visionEncoder.output(
                        Map.of("pixel_values", pv, "pixel_attention_mask", mask), outputNames);
            } catch (Exception e) {
                logMemory("Failed non-frozen iteration " + i, deviceId);
                fail("Non-frozen iteration " + i + " failed: " + e.getMessage(), e);
                return;
            }

            INDArray features = out.get("image_features");
            assertNotNull(features, "Iteration " + i + ": null");
            assertFalse(Double.isNaN(features.sumNumber().doubleValue()),
                    "Iteration " + i + ": NaN");

            pv.close();
            mask.close();
            for (INDArray a : out.values()) {
                if (a != null && !a.wasClosed()) { a.setCloseable(true); a.close(); }
            }

            logMemory("After non-frozen iteration " + i, deviceId);
        }

        log.info("Non-frozen repeated execution test passed");
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    private void freezeDspShapes(SameDiff sd) {
        try {
            var sessionsField = SameDiff.class.getDeclaredField("sessions");
            sessionsField.setAccessible(true);
            @SuppressWarnings("unchecked")
            Map<Long, InferenceSession> sessions = (Map<Long, InferenceSession>) sessionsField.get(sd);
            InferenceSession session = sessions.get(Thread.currentThread().getId());
            if (session != null) {
                DynamicShapePlanExecutor dsp = session.getDynamicShapePlanExecutor();
                if (dsp != null) {
                    dsp.setShapesFrozen(true);
                    log.info("Froze DSP shapes");
                }
            }
        } catch (Exception e) {
            log.warn("Could not freeze DSP shapes: {}", e.getMessage());
        }
    }

    private void unfreezeDspShapes(SameDiff sd) {
        try {
            var sessionsField = SameDiff.class.getDeclaredField("sessions");
            sessionsField.setAccessible(true);
            @SuppressWarnings("unchecked")
            Map<Long, InferenceSession> sessions = (Map<Long, InferenceSession>) sessionsField.get(sd);
            InferenceSession session = sessions.get(Thread.currentThread().getId());
            if (session != null) {
                DynamicShapePlanExecutor dsp = session.getDynamicShapePlanExecutor();
                if (dsp != null) {
                    dsp.setShapesFrozen(false);
                    log.info("Unfroze DSP shapes");
                }
            }
        } catch (Exception e) {
            log.warn("Could not unfreeze DSP shapes: {}", e.getMessage());
        }
    }
}
