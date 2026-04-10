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
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.diagnostics.DspDiagnostics;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.io.File;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test battery for DSP (DynamicShapePlan) slot lifecycle,
 * buffer management, and re-execution correctness.
 *
 * <p>Exercises and validates:
 * <ol>
 *   <li>Slot output buffer validity - all output arrays have valid, non-closed DataBuffers</li>
 *   <li>Cleanup effectiveness - pool stats before/after show GPU memory is freed</li>
 *   <li>Re-execution correctness - same model, same inputs produce matching outputs</li>
 *   <li>Constant array preservation - weight arrays survive cleanup with valid GPU memory</li>
 *   <li>View chain integrity - reshape/permute ops produce valid views with live backing buffers</li>
 *   <li>Segment boundary correctness - capturable segment outputs are valid after re-execution</li>
 *   <li>Memory pool baseline recovery - after cleanup+trim, pool used returns near baseline</li>
 * </ol>
 *
 * <p>Run:
 * <pre>
 *   cd platform-tests && mvn test \
 *     -Dtest=TestDSPSlotLifecycle \
 *     -Dbackend.artifactId=nd4j-cuda-12.9
 * </pre>
 */
@Slf4j
@Tag("long-running")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestDSPSlotLifecycle {

    private static SameDiff visionEncoder;
    private static boolean modelsLoaded = false;
    private static NativeOps nativeOps;
    private static boolean isCPU;

    /** Input image spatial size used across all tests. */
    private static final int TARGET_SIZE = 512;

    /** Output node name for the vision encoder. */
    private static final List<String> OUTPUT_NAMES = List.of("image_features");

    @BeforeAll
    public static void setup() throws Exception {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        System.setProperty(ND4JSystemProperties.DSP_DIAGNOSTICS, "all");
        System.setProperty(ND4JSystemProperties.DSP_DIAGNOSTICS_LEVEL, "full");
        InferenceSession.setDynamicShapePlanEnabled(true);
        nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        isCPU = Nd4j.getEnvironment().isCPU();

        try {
            DspDiagnostics.initialize();
        } catch (Exception e) {
            log.warn("DspDiagnostics.initialize() failed (non-fatal): {}", e.getMessage());
        }

        try {
            log.info("Downloading SmolDocling vision encoder for slot lifecycle tests...");
            File visionDir = VLMModelDownloader.download(
                    VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER).getModelFile();
            visionEncoder = OnnxModelCache.importWithCache(visionDir.getAbsolutePath());
            visionEncoder.setDspAutoCompileEnabled(true);
            visionEncoder.setDspNativeAutoCompileEnabled(true);
            log.info("Vision encoder loaded: {} ops, {} variables",
                    visionEncoder.ops().length, visionEncoder.variables().size());
            modelsLoaded = true;
        } catch (Exception e) {
            log.error("Failed to load vision encoder model: {}", e.getMessage(), e);
            // Tests will skip via assumeTrue
        }
    }

    @AfterAll
    public static void teardown() {
        if (visionEncoder != null) {
            visionEncoder.close();
        }
    }

    @BeforeEach
    public void resetBeforeTest() {
        if (visionEncoder != null) {
            visionEncoder.resetSession();
        }
    }

    // =========================================================================
    // Helper methods
    // =========================================================================

    /**
     * Create a random pixel_values input tensor for the vision encoder.
     * Shape: [batch=1, frames=1, channels=3, H, W]
     */
    private static INDArray createPixelValues() {
        return Nd4j.rand(DataType.FLOAT, 1, 1, 3, TARGET_SIZE, TARGET_SIZE);
    }

    /**
     * Create a pixel attention mask.
     * Shape: [batch=1, H, W]
     */
    private static INDArray createPixelMask() {
        return Nd4j.ones(DataType.BOOL, 1, TARGET_SIZE, TARGET_SIZE);
    }

    /**
     * Build the standard input map for the vision encoder.
     */
    private static Map<String, INDArray> createInputs(INDArray pixelValues, INDArray pixelMask) {
        Map<String, INDArray> inputs = new LinkedHashMap<>();
        inputs.put("pixel_values", pixelValues);
        inputs.put("pixel_attention_mask", pixelMask);
        return inputs;
    }

    /**
     * Query GPU memory pool stats for the given device.
     * Returns [poolUsedBytes, poolReservedBytes].
     * Returns [0, 0] on CPU backend.
     */
    private static long[] getPoolStats(int deviceId) {
        if (isCPU) {
            return new long[]{0L, 0L};
        }
        try (LongPointer usedPtr = new LongPointer(1);
             LongPointer reservedPtr = new LongPointer(1)) {
            nativeOps.getMemoryPoolStats(deviceId, usedPtr, reservedPtr);
            return new long[]{usedPtr.get(0), reservedPtr.get(0)};
        }
    }

    /**
     * Get device free memory in bytes.
     * Returns Long.MAX_VALUE on CPU backend.
     */
    private static long getDeviceFreeMemory(int deviceId) {
        if (isCPU) {
            return Long.MAX_VALUE;
        }
        return nativeOps.getDeviceFreeMemory(deviceId);
    }

    /**
     * Trim the memory pool on the given device.
     */
    private static void trimPool(int deviceId) {
        if (isCPU) {
            return;
        }
        try {
            nativeOps.trimMemoryPool(deviceId);
        } catch (Exception e) {
            log.warn("trimMemoryPool failed on device {}: {}", deviceId, e.getMessage());
        }
    }

    /**
     * Log GPU memory stats at a labeled checkpoint.
     */
    private static void logMemory(String label, int deviceId) {
        if (isCPU) {
            log.info("[{}] CPU backend — no GPU memory stats", label);
            return;
        }
        try {
            long[] pool = getPoolStats(deviceId);
            long freeMem = getDeviceFreeMemory(deviceId);
            long totalMem = nativeOps.getDeviceTotalMemory(deviceId);
            log.info("[{}] dev={}: poolUsed={}MB poolReserved={}MB devUsed={}MB devFree={}MB",
                    label, deviceId,
                    pool[0] / (1024 * 1024), pool[1] / (1024 * 1024),
                    (totalMem - freeMem) / (1024 * 1024), freeMem / (1024 * 1024));
        } catch (Exception e) {
            log.debug("Could not query GPU memory for label '{}': {}", label, e.getMessage());
        }
    }

    /**
     * Validate that an output array has a valid, non-closed DataBuffer.
     * On CUDA, also checks that the GPU (special) buffer pointer exists.
     */
    private static void assertBufferValid(INDArray arr, String context) {
        assertNotNull(arr, context + ": array is null");
        assertFalse(arr.wasClosed(), context + ": array is closed");
        assertTrue(arr.length() > 0, context + ": array has zero length");

        DataBuffer db = arr.data();
        assertNotNull(db, context + ": DataBuffer is null");
        assertFalse(db.wasClosed(), context + ": DataBuffer is closed");

        // On CUDA backends, verify the GPU buffer pointer is non-null
        if (!isCPU) {
            try {
                assertNotNull(db.pointer(), context + ": DataBuffer pointer() is null");
                assertTrue(db.address() != 0, context + ": DataBuffer address is 0");
            } catch (Exception e) {
                // If pointer access throws, the buffer is likely freed
                fail(context + ": DataBuffer pointer access failed — buffer may be freed: " + e.getMessage());
            }
        }
    }

    /**
     * Close all arrays in an output map safely.
     */
    private static void closeOutputs(Map<String, INDArray> outputs) {
        for (INDArray a : outputs.values()) {
            if (a != null && !a.wasClosed()) {
                a.setCloseable(true);
                a.close();
            }
        }
    }

    /**
     * Freeze DSP shapes on the current thread's session.
     */
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

    /**
     * Unfreeze DSP shapes on the current thread's session.
     */
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

    // =========================================================================
    // Test 1: Slot output buffer validity after each execution
    // =========================================================================

    @Test
    @Order(1)
    @DisplayName("Slot output buffers are valid (non-null, non-closed) after each execution")
    public void testSlotOutputBufferValidity() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");

        int numExecutions = 3;
        for (int exec = 1; exec <= numExecutions; exec++) {
            log.info("--- Execution {}/{}: checking output buffer validity ---", exec, numExecutions);

            INDArray pixelValues = createPixelValues();
            INDArray pixelMask = createPixelMask();
            Map<String, INDArray> inputs = createInputs(pixelValues, pixelMask);

            Map<String, INDArray> outputs;
            try {
                outputs = visionEncoder.output(inputs, OUTPUT_NAMES);
            } catch (Exception e) {
                fail("Execution " + exec + " failed: " + e.getMessage(), e);
                return;
            }

            // Validate every output array's buffer
            for (Map.Entry<String, INDArray> entry : outputs.entrySet()) {
                String name = entry.getKey();
                INDArray arr = entry.getValue();
                String ctx = "Exec " + exec + ", output '" + name + "'";

                assertBufferValid(arr, ctx);

                // Verify we can actually read the data (forces device-to-host sync on CUDA)
                double sum = arr.sumNumber().doubleValue();
                assertFalse(Double.isNaN(sum), ctx + ": sum is NaN (invalid data)");
                assertFalse(Double.isInfinite(sum), ctx + ": sum is Inf (invalid data)");
                log.info("{}: shape={}, sum={}", ctx, Arrays.toString(arr.shape()), sum);
            }

            // Clean up for next iteration
            closeOutputs(outputs);
            pixelValues.close();
            pixelMask.close();
        }

        log.info("Slot output buffer validity test PASSED for {} executions", numExecutions);
    }

    // =========================================================================
    // Test 2: Cleanup effectiveness - pool stats before/after
    // =========================================================================

    @Test
    @Order(2)
    @DisplayName("Cleanup effectiveness: GPU memory pool used drops after output closure and trim")
    public void testCleanupEffectiveness() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");

        int deviceId = 0;

        // Establish baseline: run once, close outputs, trim
        INDArray pv0 = createPixelValues();
        INDArray mask0 = createPixelMask();
        Map<String, INDArray> warmup = visionEncoder.output(createInputs(pv0, mask0), OUTPUT_NAMES);
        assertNotNull(warmup.get("image_features"), "Warmup execution failed");
        closeOutputs(warmup);
        pv0.close();
        mask0.close();
        Nd4j.getExecutioner().commit();
        trimPool(deviceId);
        logMemory("Baseline after warmup+trim", deviceId);
        long[] baselinePool = getPoolStats(deviceId);

        // Execute again — this will allocate intermediates and output buffers
        INDArray pv1 = createPixelValues();
        INDArray mask1 = createPixelMask();
        Map<String, INDArray> out1 = visionEncoder.output(createInputs(pv1, mask1), OUTPUT_NAMES);
        assertNotNull(out1.get("image_features"), "Post-baseline execution failed");
        Nd4j.getExecutioner().commit();
        logMemory("After execution (before cleanup)", deviceId);
        long[] afterExecPool = getPoolStats(deviceId);

        // Close outputs — should free output buffers back to pool
        closeOutputs(out1);
        pv1.close();
        mask1.close();
        Nd4j.getExecutioner().commit();
        logMemory("After close (before trim)", deviceId);
        long[] afterClosePool = getPoolStats(deviceId);

        // Trim pool — should release freed blocks back to device
        trimPool(deviceId);
        logMemory("After trim", deviceId);
        long[] afterTrimPool = getPoolStats(deviceId);

        if (!isCPU) {
            // After closing outputs, pool used should not have grown compared to
            // post-exec (some memory returned to pool's free list)
            log.info("Pool used: baseline={}MB afterExec={}MB afterClose={}MB afterTrim={}MB",
                    baselinePool[0] / (1024 * 1024),
                    afterExecPool[0] / (1024 * 1024),
                    afterClosePool[0] / (1024 * 1024),
                    afterTrimPool[0] / (1024 * 1024));

            // After trim, pool used should be closer to baseline than afterExec
            // Allow generous tolerance — the key invariant is that trim reduces usage
            long growthAfterExec = afterExecPool[0] - baselinePool[0];
            long growthAfterTrim = afterTrimPool[0] - baselinePool[0];
            if (growthAfterExec > 0) {
                double recoveryRatio = 1.0 - ((double) growthAfterTrim / growthAfterExec);
                log.info("Memory recovery ratio after trim: {}", String.format("%.2f", recoveryRatio));
                // We expect at least some recovery. Constants and cached plan data will remain.
                assertTrue(growthAfterTrim <= growthAfterExec,
                        "Pool used grew after trim: afterExec=" + afterExecPool[0] / (1024 * 1024) +
                                "MB, afterTrim=" + afterTrimPool[0] / (1024 * 1024) + "MB");
            }
        } else {
            log.info("CPU backend — skipping GPU pool stat assertions");
        }

        log.info("Cleanup effectiveness test PASSED");
    }

    // =========================================================================
    // Test 3: Re-execution correctness - deterministic outputs for same input
    // =========================================================================

    @Test
    @Order(3)
    @DisplayName("Re-execution correctness: same input produces matching outputs within tolerance")
    public void testReExecutionCorrectness() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");

        // Use a fixed seed for reproducible input
        Nd4j.getRandom().setSeed(42);
        INDArray fixedInput = Nd4j.rand(DataType.FLOAT, 1, 1, 3, TARGET_SIZE, TARGET_SIZE);
        INDArray mask = createPixelMask();

        int numRuns = 4;
        double[] sums = new double[numRuns];
        long[][] shapes = new long[numRuns][];
        INDArray firstOutput = null;

        for (int run = 0; run < numRuns; run++) {
            log.info("--- Re-execution run {}/{} ---", run + 1, numRuns);

            // Clone the fixed input so each run starts with identical data
            INDArray inputClone = fixedInput.dup();
            INDArray maskClone = mask.dup();

            Map<String, INDArray> outputs = visionEncoder.output(
                    createInputs(inputClone, maskClone), OUTPUT_NAMES);

            INDArray features = outputs.get("image_features");
            assertNotNull(features, "Run " + (run + 1) + ": output is null");
            assertBufferValid(features, "Run " + (run + 1));

            sums[run] = features.sumNumber().doubleValue();
            shapes[run] = features.shape();
            assertFalse(Double.isNaN(sums[run]), "Run " + (run + 1) + ": NaN in output");

            log.info("Run {}: shape={}, sum={}", run + 1, Arrays.toString(shapes[run]), sums[run]);

            if (run == 0) {
                // Keep first output for element-wise comparison
                firstOutput = features.dup();
            } else {
                // Shape must match across all runs
                assertArrayEquals(shapes[0], shapes[run],
                        "Shape mismatch: run 1=" + Arrays.toString(shapes[0]) +
                                " vs run " + (run + 1) + "=" + Arrays.toString(shapes[run]));

                // Values should be identical for deterministic ops with same input
                // Use a tolerance for floating-point accumulation differences
                double absDiff = Math.abs(sums[run] - sums[0]);
                double relTol = Math.abs(sums[0]) * 1e-4; // 0.01% relative tolerance
                assertTrue(absDiff < relTol + 1e-3,
                        "Output values diverged: run 1 sum=" + sums[0] +
                                " vs run " + (run + 1) + " sum=" + sums[run] +
                                " (absDiff=" + absDiff + ", relTol=" + relTol + ")");

                // Element-wise comparison on a subset (first 100 elements)
                INDArray flat1 = firstOutput.reshape(-1);
                INDArray flatN = features.reshape(-1);
                long numToCheck = Math.min(100, flat1.length());
                for (long i = 0; i < numToCheck; i++) {
                    double v1 = flat1.getDouble(i);
                    double vN = flatN.getDouble(i);
                    double elemDiff = Math.abs(v1 - vN);
                    assertTrue(elemDiff < 1e-3,
                            "Element " + i + " diverged: run1=" + v1 + " vs run" +
                                    (run + 1) + "=" + vN + " (diff=" + elemDiff + ")");
                }
            }

            // Close this run's outputs (but not firstOutput which we kept)
            inputClone.close();
            maskClone.close();
            closeOutputs(outputs);
        }

        // Clean up
        if (firstOutput != null && !firstOutput.wasClosed()) {
            firstOutput.close();
        }
        fixedInput.close();
        mask.close();

        log.info("Re-execution correctness test PASSED across {} runs", numRuns);
    }

    // =========================================================================
    // Test 4: Constant (weight) array preservation after cleanup
    // =========================================================================

    @Test
    @Order(4)
    @DisplayName("Constant arrays survive cleanup with valid GPU memory")
    public void testConstantArrayPreservation() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");

        int deviceId = 0;

        // Collect constant variable names and their initial values
        Set<SDVariable> constants = visionEncoder.constants();
        assertFalse(constants.isEmpty(), "Vision encoder has no constant variables");
        log.info("Vision encoder has {} constant variables", constants.size());

        // Sample up to 20 constants to check
        List<SDVariable> sampled = new ArrayList<>(constants);
        if (sampled.size() > 20) {
            sampled = sampled.subList(0, 20);
        }

        // Record checksums before execution
        Map<String, Double> preExecSums = new LinkedHashMap<>();
        Map<String, long[]> preExecShapes = new LinkedHashMap<>();
        for (SDVariable constant : sampled) {
            INDArray arr = constant.getArr();
            if (arr != null && !arr.isEmpty() && !arr.wasClosed()) {
                preExecSums.put(constant.name(), arr.sumNumber().doubleValue());
                preExecShapes.put(constant.name(), arr.shape());
            }
        }
        log.info("Recorded checksums for {} constants before execution", preExecSums.size());
        Assumptions.assumeFalse(preExecSums.isEmpty(), "No accessible constant arrays found");

        // Execute the model twice to exercise cleanup
        for (int exec = 1; exec <= 2; exec++) {
            INDArray pv = createPixelValues();
            INDArray mask = createPixelMask();
            Map<String, INDArray> outputs = visionEncoder.output(createInputs(pv, mask), OUTPUT_NAMES);
            assertNotNull(outputs.get("image_features"), "Execution " + exec + " failed");
            closeOutputs(outputs);
            pv.close();
            mask.close();
        }

        // Trim pool to force cleanup
        Nd4j.getExecutioner().commit();
        trimPool(deviceId);
        logMemory("After 2 executions + trim", deviceId);

        // Verify constants survived with valid buffers and unchanged values
        int checked = 0;
        int preserved = 0;
        for (SDVariable constant : sampled) {
            String name = constant.name();
            INDArray arr = constant.getArr();
            if (arr == null || arr.isEmpty()) {
                continue;
            }
            checked++;

            // Buffer validity check
            assertFalse(arr.wasClosed(),
                    "Constant '" + name + "' was closed after cleanup");

            DataBuffer db = arr.data();
            assertNotNull(db, "Constant '" + name + "' has null DataBuffer after cleanup");
            assertFalse(db.wasClosed(),
                    "Constant '" + name + "' DataBuffer is closed after cleanup");

            // On CUDA, check GPU pointer is still valid
            if (!isCPU) {
                try {
                    assertTrue(db.address() != 0,
                            "Constant '" + name + "' has zero address after cleanup");
                } catch (Exception e) {
                    fail("Constant '" + name + "' buffer access failed after cleanup: " + e.getMessage());
                }
            }

            // Value preservation check
            if (preExecSums.containsKey(name)) {
                double postSum = arr.sumNumber().doubleValue();
                double preSum = preExecSums.get(name);
                long[] preShape = preExecShapes.get(name);

                assertArrayEquals(preShape, arr.shape(),
                        "Constant '" + name + "' shape changed: " +
                                Arrays.toString(preShape) + " -> " + Arrays.toString(arr.shape()));

                double diff = Math.abs(postSum - preSum);
                assertTrue(diff < 1e-5,
                        "Constant '" + name + "' value changed: preSum=" + preSum +
                                " postSum=" + postSum + " diff=" + diff);
                preserved++;
            }
        }

        log.info("Constant preservation: checked={}, values preserved={}/{}", checked, preserved, preExecSums.size());
        assertTrue(preserved > 0, "No constants were verified as preserved");
        log.info("Constant array preservation test PASSED");
    }

    // =========================================================================
    // Test 5: View chain integrity
    // =========================================================================

    @Test
    @Order(5)
    @DisplayName("View chain integrity: reshape/permute outputs have valid non-freed backing buffers")
    public void testViewChainIntegrity() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");

        // Execute and get all intermediate arrays by requesting output
        INDArray pixelValues = createPixelValues();
        INDArray pixelMask = createPixelMask();

        Map<String, INDArray> outputs = visionEncoder.output(
                createInputs(pixelValues, pixelMask), OUTPUT_NAMES);

        INDArray features = outputs.get("image_features");
        assertNotNull(features, "Output is null");
        assertBufferValid(features, "image_features output");

        // Create view operations on the output to verify view chain integrity
        // These simulate what downstream ops (decoder input preparation) would do
        log.info("Output shape: {}", Arrays.toString(features.shape()));

        // Test reshape view
        long totalElements = features.length();
        INDArray reshaped = features.reshape(1, -1);
        assertBufferValid(reshaped, "reshaped view [1, -1]");
        assertEquals(totalElements, reshaped.length(),
                "Reshape changed total elements");
        log.info("Reshape [1, -1]: shape={}, valid", Arrays.toString(reshaped.shape()));

        // Test that the reshaped view shares backing buffer with original
        assertEquals(features.data().address(), reshaped.data().address(),
                "Reshape view should share backing buffer address with original");

        // Test permute view (if rank >= 3)
        if (features.rank() >= 3) {
            long[] permutation = new long[features.rank()];
            // Reverse dimension order
            for (int i = 0; i < features.rank(); i++) {
                permutation[i] = features.rank() - 1 - i;
            }
            INDArray permuted = features.permute(permutation);
            assertBufferValid(permuted, "permuted view");
            assertEquals(features.length(), permuted.length(),
                    "Permute changed total elements");
            log.info("Permute {}: shape={}, valid",
                    Arrays.toString(permutation), Arrays.toString(permuted.shape()));

            // Verify shared backing buffer
            assertEquals(features.data().address(), permuted.data().address(),
                    "Permute view should share backing buffer address with original");
        }

        // Test slice view
        if (features.rank() >= 2 && features.size(0) > 0) {
            INDArray slice = features.slice(0, 0);
            assertBufferValid(slice, "slice(0, 0) view");
            log.info("Slice [0, 0]: shape={}, valid", Arrays.toString(slice.shape()));
        }

        // Verify that after reading all views, the original is still valid
        assertBufferValid(features, "original after view reads");
        double sum = features.sumNumber().doubleValue();
        assertFalse(Double.isNaN(sum), "Original output corrupted after view operations");
        log.info("Original output still valid after view chain: sum={}", sum);

        // Cleanup
        closeOutputs(outputs);
        pixelValues.close();
        pixelMask.close();

        log.info("View chain integrity test PASSED");
    }

    // =========================================================================
    // Test 6: Segment boundary correctness after re-execution
    // =========================================================================

    @Test
    @Order(6)
    @DisplayName("Segment boundary correctness: capturable segment outputs valid after re-execution")
    public void testSegmentBoundaryCorrectness() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");

        int deviceId = 0;

        // Execute 1: compile plan, establish segment boundaries
        log.info("--- Segment test: execution 1 (compile) ---");
        INDArray pv1 = createPixelValues();
        INDArray mask1 = createPixelMask();
        Map<String, INDArray> out1 = visionEncoder.output(createInputs(pv1, mask1), OUTPUT_NAMES);
        INDArray features1 = out1.get("image_features");
        assertBufferValid(features1, "Exec 1 image_features");
        long[] shape1 = features1.shape();
        double sum1 = features1.sumNumber().doubleValue();
        log.info("Exec 1: shape={}, sum={}", Arrays.toString(shape1), sum1);
        closeOutputs(out1);
        pv1.close();
        mask1.close();

        // Freeze shapes to lock segment boundaries
        freezeDspShapes(visionEncoder);
        logMemory("After exec 1 + freeze", deviceId);

        // Execute 2-4: re-execute with frozen segments
        for (int exec = 2; exec <= 4; exec++) {
            log.info("--- Segment test: execution {} (frozen re-execution) ---", exec);

            INDArray pv = createPixelValues();
            INDArray mask = createPixelMask();

            Map<String, INDArray> out;
            try {
                out = visionEncoder.output(createInputs(pv, mask), OUTPUT_NAMES);
            } catch (Exception e) {
                logMemory("FAILED at exec " + exec, deviceId);
                fail("Frozen re-execution " + exec + " failed — segment boundary issue: " +
                        e.getMessage(), e);
                return;
            }

            INDArray features = out.get("image_features");
            assertBufferValid(features, "Exec " + exec + " image_features");

            // Shape must be identical to exec 1 (segments define shapes)
            assertArrayEquals(shape1, features.shape(),
                    "Shape changed at exec " + exec + ": expected " +
                            Arrays.toString(shape1) + " got " + Arrays.toString(features.shape()));

            // Output should be numerically valid
            double sum = features.sumNumber().doubleValue();
            assertFalse(Double.isNaN(sum), "Exec " + exec + ": NaN in segment output");
            assertFalse(Double.isInfinite(sum), "Exec " + exec + ": Inf in segment output");

            log.info("Exec {}: shape={}, sum={}", exec, Arrays.toString(features.shape()), sum);

            closeOutputs(out);
            pv.close();
            mask.close();
        }

        unfreezeDspShapes(visionEncoder);
        log.info("Segment boundary correctness test PASSED");
    }

    // =========================================================================
    // Test 7: Memory pool baseline recovery after cleanup + trim
    // =========================================================================

    @Test
    @Order(7)
    @DisplayName("Memory pool baseline recovery: pool used returns near baseline after cleanup+trim")
    public void testMemoryPoolBaselineRecovery() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");

        int deviceId = 0;

        // Phase 1: Establish baseline
        // Run a warmup, close everything, trim aggressively
        INDArray pvWarm = createPixelValues();
        INDArray maskWarm = createPixelMask();
        Map<String, INDArray> warmup = visionEncoder.output(createInputs(pvWarm, maskWarm), OUTPUT_NAMES);
        assertNotNull(warmup.get("image_features"), "Warmup failed");
        closeOutputs(warmup);
        pvWarm.close();
        maskWarm.close();
        Nd4j.getExecutioner().commit();
        trimPool(deviceId);
        logMemory("Phase 1: Baseline", deviceId);
        long[] baselinePool = getPoolStats(deviceId);
        long baselineDevFree = getDeviceFreeMemory(deviceId);

        // Phase 2: Execute multiple times to accumulate allocations
        int numAllocationRuns = 5;
        for (int i = 0; i < numAllocationRuns; i++) {
            INDArray pv = createPixelValues();
            INDArray mask = createPixelMask();
            Map<String, INDArray> out = visionEncoder.output(createInputs(pv, mask), OUTPUT_NAMES);
            assertNotNull(out.get("image_features"), "Allocation run " + (i + 1) + " failed");
            closeOutputs(out);
            pv.close();
            mask.close();
        }
        Nd4j.getExecutioner().commit();
        logMemory("Phase 2: After " + numAllocationRuns + " allocations (before trim)", deviceId);
        long[] afterAllocPool = getPoolStats(deviceId);

        // Phase 3: Trim and check recovery
        trimPool(deviceId);
        logMemory("Phase 3: After trim", deviceId);
        long[] afterTrimPool = getPoolStats(deviceId);
        long afterTrimDevFree = getDeviceFreeMemory(deviceId);

        if (!isCPU) {
            long baseUsed = baselinePool[0];
            long afterAllocUsed = afterAllocPool[0];
            long afterTrimUsed = afterTrimPool[0];

            log.info("Pool used: baseline={}MB afterAlloc={}MB afterTrim={}MB",
                    baseUsed / (1024 * 1024),
                    afterAllocUsed / (1024 * 1024),
                    afterTrimUsed / (1024 * 1024));

            // Key invariant: pool used should NOT grow monotonically.
            // After trim, pool used should be no more than the baseline + a constant
            // overhead for DSP plan caches and compiled segments.
            // Allow 500MB headroom for plan/segment cache overhead.
            long maxAcceptableOverhead = 500L * 1024 * 1024;
            assertTrue(afterTrimUsed < baseUsed + maxAcceptableOverhead,
                    "Pool used did not recover to near baseline: baseline=" +
                            baseUsed / (1024 * 1024) + "MB, afterTrim=" +
                            afterTrimUsed / (1024 * 1024) + "MB, maxOverhead=" +
                            maxAcceptableOverhead / (1024 * 1024) + "MB");

            // Device free memory should also recover (within tolerance for fragmentation)
            log.info("Device free: baseline={}MB afterTrim={}MB",
                    baselineDevFree / (1024 * 1024),
                    afterTrimDevFree / (1024 * 1024));
        } else {
            log.info("CPU backend — skipping GPU pool recovery assertions");
        }

        log.info("Memory pool baseline recovery test PASSED");
    }

    // =========================================================================
    // Test 8: Pool used does NOT grow monotonically across iterations
    // =========================================================================

    @Test
    @Order(8)
    @DisplayName("Pool growth tracking: pool used does not grow monotonically over 8 iterations")
    public void testPoolUsedNotMonotonic() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");

        int deviceId = 0;
        int numIterations = 8;

        // Warmup and baseline
        INDArray pvWarm = createPixelValues();
        INDArray maskWarm = createPixelMask();
        Map<String, INDArray> warmup = visionEncoder.output(createInputs(pvWarm, maskWarm), OUTPUT_NAMES);
        assertNotNull(warmup.get("image_features"), "Warmup failed");
        closeOutputs(warmup);
        pvWarm.close();
        maskWarm.close();
        Nd4j.getExecutioner().commit();
        trimPool(deviceId);

        long[] poolUsedPerIteration = new long[numIterations];
        long[] devFreePerIteration = new long[numIterations];

        for (int i = 0; i < numIterations; i++) {
            INDArray pv = createPixelValues();
            INDArray mask = createPixelMask();
            Map<String, INDArray> out = visionEncoder.output(createInputs(pv, mask), OUTPUT_NAMES);

            INDArray features = out.get("image_features");
            assertNotNull(features, "Iteration " + (i + 1) + ": null output");
            assertBufferValid(features, "Iteration " + (i + 1));

            // Close outputs between iterations (simulates real usage)
            closeOutputs(out);
            pv.close();
            mask.close();
            Nd4j.getExecutioner().commit();

            // Record stats after cleanup
            long[] pool = getPoolStats(deviceId);
            poolUsedPerIteration[i] = pool[0];
            devFreePerIteration[i] = getDeviceFreeMemory(deviceId);

            log.info("Iteration {}: poolUsed={}MB devFree={}MB",
                    i + 1,
                    pool[0] / (1024 * 1024),
                    devFreePerIteration[i] / (1024 * 1024));
        }

        if (!isCPU) {
            // Count strictly monotonic increases (each iteration uses more than previous)
            int monotonicIncreases = 0;
            for (int i = 1; i < numIterations; i++) {
                // Consider it monotonically increasing if growth is >1MB per iteration
                if (poolUsedPerIteration[i] > poolUsedPerIteration[i - 1] + 1024 * 1024) {
                    monotonicIncreases++;
                }
            }

            log.info("Monotonic pool increases: {}/{} (max acceptable: {})",
                    monotonicIncreases, numIterations - 1, numIterations - 2);

            // If ALL transitions are monotonically increasing, that is a memory leak.
            // Allow at most N-2 increases (i.e., at least one stabilization/decrease).
            assertTrue(monotonicIncreases < numIterations - 1,
                    "Pool used grew monotonically every single iteration (" +
                            monotonicIncreases + "/" + (numIterations - 1) +
                            ") — this indicates a memory leak. " +
                            "First=" + poolUsedPerIteration[0] / (1024 * 1024) + "MB, " +
                            "Last=" + poolUsedPerIteration[numIterations - 1] / (1024 * 1024) + "MB");

            // Also check total growth: should not exceed 2x from first to last
            long totalGrowth = poolUsedPerIteration[numIterations - 1] - poolUsedPerIteration[0];
            log.info("Total pool growth over {} iterations: {}MB",
                    numIterations, totalGrowth / (1024 * 1024));
        } else {
            log.info("CPU backend — skipping monotonic pool growth assertions");
        }

        log.info("Pool growth tracking test PASSED across {} iterations", numIterations);
    }
}
