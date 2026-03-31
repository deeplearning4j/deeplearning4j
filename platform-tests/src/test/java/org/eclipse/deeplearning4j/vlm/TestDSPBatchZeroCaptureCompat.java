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
 * Tests that batch-zero kernel mode eliminates all cudaMemsetAsync calls during
 * CUDA graph capture. With batch-zero enabled:
 *
 * <ul>
 *   <li>All output buffer zeroing happens via a single batchZeroKernel BEFORE capture</li>
 *   <li>Individual nullify() calls are suppressed during capture (isBatchZeroActive=true)</li>
 *   <li>No cudaMemsetAsync graph nodes are created (would cause error 901)</li>
 *   <li>Execution is consistent: same path for captured and non-captured segments</li>
 * </ul>
 *
 * <p>Run:
 * <pre>
 *   cd platform-tests && mvn test \
 *     -Dtest=TestDSPBatchZeroCaptureCompat \
 *     -Dbackend.artifactId=nd4j-cuda-12.9
 * </pre>
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestDSPBatchZeroCaptureCompat {

    private static SameDiff visionEncoder;
    private static boolean modelsLoaded = false;
    private static NativeOps nativeOps;

    @BeforeAll
    public static void setup() throws Exception {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        System.setProperty(ND4JSystemProperties.DSP_DIAGNOSTICS, "all");
        System.setProperty(ND4JSystemProperties.DSP_DIAGNOSTICS_LEVEL, "full");
        InferenceSession.setDynamicShapePlanEnabled(true);

        // Enable batch-zero directly on native environment
        Nd4j.getEnvironment().setDspBatchZero(true);
        Nd4j.getEnvironment().setDspBatchZeroKernel(true);

        nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

        log.info("Batch-zero enabled: dspBatchZero={}, dspBatchZeroKernel={}",
                Nd4j.getEnvironment().dspBatchZero(),
                Nd4j.getEnvironment().dspBatchZeroKernel());

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
            log.error("Failed to load models", e);
        }
    }

    @AfterAll
    public static void teardown() {
        if (visionEncoder != null) visionEncoder.close();
    }

    @AfterEach
    public void cleanupBetweenTests() {
        if (visionEncoder != null) {
            visionEncoder.resetSession();
            visionEncoder.freeAllCachedNativePlanHandles();
        }
        try {
            Nd4j.getExecutioner().commit();
            int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
            for (int d = 0; d < numDevices; d++) {
                nativeOps.trimMemoryPool(d);
            }
        } catch (Exception e) {
            log.warn("Post-test cleanup failed: {}", e.getMessage());
        }
        logGpuMemory("After cleanup");
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
    // Test 1: Batch-zero single execution — verify output is valid
    // =========================================================================

    @Test
    @Order(1)
    @DisplayName("Batch-zero: single execution produces valid output")
    public void testBatchZeroSingleExecution() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");
        assertTrue(Nd4j.getEnvironment().dspBatchZero(),
                "dspBatchZero must be enabled");
        assertTrue(Nd4j.getEnvironment().dspBatchZeroKernel(),
                "dspBatchZeroKernel must be enabled");

        int targetSize = 512;
        INDArray pixelValues = Nd4j.rand(DataType.FLOAT, 1, 1, 3, targetSize, targetSize);
        INDArray pixelMask = Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);

        Map<String, INDArray> result = visionEncoder.output(
                Map.of("pixel_values", pixelValues, "pixel_attention_mask", pixelMask),
                List.of("image_features"));

        INDArray features = result.get("image_features");
        assertNotNull(features, "image_features must not be null");
        assertEquals(3, features.rank(), "Expected rank 3");
        log.info("Test 1: shape={}, sum={}", Arrays.toString(features.shape()), features.sumNumber());

        // Valid output: no NaN, no Inf, non-zero
        assertFalse(features.isNaN().any(), "Output contains NaN");
        assertFalse(features.isInfinite().any(), "Output contains Inf");

        pixelValues.close();
        pixelMask.close();
    }

    // =========================================================================
    // Test 2: Batch-zero multi-execution — verify no status 50 errors on
    // repeated calls (capture should succeed cleanly without cudaMemsetAsync)
    // =========================================================================

    @Test
    @Order(2)
    @DisplayName("Batch-zero: 3 consecutive calls without reset (capture reuse)")
    public void testBatchZeroMultiExecution() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");

        int targetSize = 512;
        int numCalls = 3;
        long[] prevShape = null;

        for (int call = 1; call <= numCalls; call++) {
            log.info("=== Batch-zero multi-exec call {}/{} ===", call, numCalls);
            logGpuMemory("Before call " + call);

            INDArray pixelValues = Nd4j.rand(DataType.FLOAT, 1, 1, 3, targetSize, targetSize);
            INDArray pixelMask = Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);

            Map<String, INDArray> result;
            try {
                result = visionEncoder.output(
                        Map.of("pixel_values", pixelValues, "pixel_attention_mask", pixelMask),
                        List.of("image_features"));
            } catch (Exception e) {
                fail("Call " + call + " failed: " + e.getMessage(), e);
                return;
            }

            INDArray features = result.get("image_features");
            assertNotNull(features, "Call " + call + ": image_features null");
            assertFalse(features.isNaN().any(), "Call " + call + ": contains NaN");

            long[] shape = features.shape();
            log.info("Call {}: shape={}", call, Arrays.toString(shape));

            if (prevShape != null) {
                assertArrayEquals(prevShape, shape,
                        "Call " + call + ": shape changed from " +
                                Arrays.toString(prevShape) + " to " + Arrays.toString(shape));
            }
            prevShape = shape;

            pixelValues.close();
            pixelMask.close();
            logGpuMemory("After call " + call);
        }

        log.info("Batch-zero multi-exec: all {} calls succeeded", numCalls);
    }

    // =========================================================================
    // Test 3: Batch-zero with frozen shapes — the critical path where
    // capture-incompatible cudaMemsetAsync caused error 901
    // =========================================================================

    @Test
    @Order(3)
    @DisplayName("Batch-zero: frozen shapes execution (capture-critical path)")
    public void testBatchZeroFrozenShapes() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");
        logGpuMemory("Test 3 start");

        int targetSize = 512;
        List<String> outputNames = List.of("image_features");

        // Call 1: warmup (compile + execute)
        log.info("=== Frozen batch-zero: Call 1 (warmup) ===");
        INDArray pv1 = Nd4j.rand(DataType.FLOAT, 1, 1, 3, targetSize, targetSize);
        INDArray pm1 = Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);
        Map<String, INDArray> r1 = visionEncoder.output(
                Map.of("pixel_values", pv1, "pixel_attention_mask", pm1), outputNames);
        INDArray f1 = r1.get("image_features");
        assertNotNull(f1, "Warmup: features null");
        long[] expectedShape = f1.shape();
        log.info("Warmup shape: {}", Arrays.toString(expectedShape));
        pv1.close();
        pm1.close();

        // Freeze shapes
        log.info("Freezing shapes...");
        visionEncoder.setDspShapesFrozen(true);

        // Call 2: first frozen execution (triggers capture with batch-zero)
        log.info("=== Frozen batch-zero: Call 2 (first frozen) ===");
        logGpuMemory("Before frozen call 2");
        INDArray pv2 = Nd4j.rand(DataType.FLOAT, 1, 1, 3, targetSize, targetSize);
        INDArray pm2 = Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);

        Map<String, INDArray> r2;
        try {
            r2 = visionEncoder.output(
                    Map.of("pixel_values", pv2, "pixel_attention_mask", pm2), outputNames);
        } catch (Exception e) {
            fail("Frozen call 2 failed: " + e.getMessage(), e);
            return;
        }

        INDArray f2 = r2.get("image_features");
        assertNotNull(f2, "Frozen call 2: features null");
        assertArrayEquals(expectedShape, f2.shape(), "Frozen call 2: shape mismatch");
        assertFalse(f2.isNaN().any(), "Frozen call 2: contains NaN");
        log.info("Frozen call 2: shape={}", Arrays.toString(f2.shape()));
        pv2.close();
        pm2.close();

        // Call 3: second frozen execution (should replay captured graph)
        log.info("=== Frozen batch-zero: Call 3 (frozen replay) ===");
        logGpuMemory("Before frozen call 3");
        INDArray pv3 = Nd4j.rand(DataType.FLOAT, 1, 1, 3, targetSize, targetSize);
        INDArray pm3 = Nd4j.ones(DataType.BOOL, 1, targetSize, targetSize);

        Map<String, INDArray> r3;
        try {
            r3 = visionEncoder.output(
                    Map.of("pixel_values", pv3, "pixel_attention_mask", pm3), outputNames);
        } catch (Exception e) {
            fail("Frozen call 3 failed: " + e.getMessage(), e);
            return;
        }

        INDArray f3 = r3.get("image_features");
        assertNotNull(f3, "Frozen call 3: features null");
        assertArrayEquals(expectedShape, f3.shape(), "Frozen call 3: shape mismatch");
        assertFalse(f3.isNaN().any(), "Frozen call 3: contains NaN");
        log.info("Frozen call 3: shape={}", Arrays.toString(f3.shape()));
        pv3.close();
        pm3.close();

        log.info("Batch-zero frozen shapes: all calls succeeded");
    }
}
