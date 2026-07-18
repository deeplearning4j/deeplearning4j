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
package org.eclipse.deeplearning4j.nd4j.backends;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.shape.DiagPart;
import org.nd4j.linalg.api.ops.impl.shape.Gather;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRoPE;
import org.nd4j.linalg.api.ops.impl.transforms.custom.LayerNorm;
import org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNorm;
import org.nd4j.linalg.api.ops.impl.transforms.strict.GELU;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Swish;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Mixed Vulkan integration and eager numerical tests.
 *
 * The SameDiff matmul and persistent-RNG cases below inspect DSP plan state,
 * require "vulkan-native", and prohibit slot-by-slot fallback; those cases are
 * replay evidence for their actual graphs. The wave tests use eager ND4J calls
 * through runNEagerSteps(). They validate numerical/backend integration only:
 * repeating an eager call is not DSP replay and is not emitted-kernel coverage.
 *
 * VulkanKernelEmitterStrictReplayTest is the authoritative per-op kernel gate.
 * Skipped cases and software Vulkan devices do not count as hardware evidence.
 *
 * Prerequisites:
 *   1. nd4j-vulkan is included by the test-vulkan Maven profile.
 *   2. A Vulkan device enumerates through the normal loader.
 *   3. The Vulkan chip exposes HAVE_MLIR=1 and Triton-enabled DSP.
 *
 * Run from platform-tests:
 * <pre>
 *   /home/agibsonccc/dev-apps/mvn/bin/mvn --log-file vulkan-integration.log -Ptest-vulkan -Dlibnd4j.triton=ON -Dnd4j.vulkan.test.requireHardware=true -Dtest=VulkanReplayEquivalenceTest test
 * </pre>
 */
@Slf4j
@Tag(TagNames.VULKAN)
@DisplayName("Vulkan DSP integration and eager numerical equivalence")
public class VulkanReplayEquivalenceTest {

    private static final String VULKAN_BINDINGS_CLASS = "org.nd4j.linalg.vulkan.bindings.Nd4jVulkan";

    /** NativeOps singleton, null when bindings absent. */
    private static Object nativeOps;

    /** True when at least one Vulkan device enumerates. */
    private static boolean vulkanDevicePresent = false;

    /** Device selected through the backend's ordinary setDevice API. */
    private static int selectedDeviceId = 0;
    private static String selectedDeviceName;

    /**
     * True when the chip library was built with HAVE_MLIR=1.
     */
    private static boolean mlirEnabled = false;

    @BeforeAll
    static void setup() {
        try {
            Class<?> bindingsClass = Class.forName(VULKAN_BINDINGS_CLASS);
            nativeOps = bindingsClass.getDeclaredConstructor().newInstance();
            log.info("Loaded Vulkan NativeOps: {}", bindingsClass.getName());

            int count = (int) bindingsClass.getMethod("getAvailableDevices").invoke(nativeOps);
            log.info("Vulkan device count: {}", count);
            vulkanDevicePresent = (count > 0);

            if (vulkanDevicePresent) {
                String deviceNameRegex =
                        System.getProperty("nd4j.vulkan.test.deviceNameRegex", "").trim();
                Pattern requestedDevice = deviceNameRegex.isEmpty()
                        ? null : Pattern.compile(deviceNameRegex);
                selectedDeviceId = -1;
                for (int deviceId = 0; deviceId < count; deviceId++) {
                    String name = (String) bindingsClass.getMethod(
                            "getDeviceName", int.class).invoke(nativeOps, deviceId);
                    log.info("Vulkan device[{}]: '{}'", deviceId, name);
                    if (requestedDevice == null || requestedDevice.matcher(name).find()) {
                        selectedDeviceId = deviceId;
                        selectedDeviceName = name;
                        break;
                    }
                }
                if (selectedDeviceId < 0) {
                    throw new IllegalStateException(
                            "No Vulkan device name matched regex: " + deviceNameRegex);
                }
                int setResult = (int) bindingsClass.getMethod(
                        "setDevice", int.class).invoke(nativeOps, selectedDeviceId);
                if (setResult != 1) {
                    throw new IllegalStateException(
                            "setDevice(" + selectedDeviceId + ") failed");
                }
                log.info("Selected Vulkan replay device: id={} name='{}'",
                        selectedDeviceId, selectedDeviceName);
            }

            boolean mlirProbed = false;
            try {
                Object value = bindingsClass.getField("HAVE_MLIR").get(null);
                mlirProbed = value instanceof Number && ((Number) value).intValue() == 1;
            } catch (NoSuchFieldException noGeneratedConstant) {
                try {
                    Method isMlir = bindingsClass.getMethod("isMlirEnabled");
                    mlirProbed = (boolean) isMlir.invoke(nativeOps);
                } catch (NoSuchMethodException noCapabilityMethod) {
                    Method getIntConst = bindingsClass.getMethod("getConfigIntValue", String.class);
                    Object value = getIntConst.invoke(nativeOps, "HAVE_MLIR");
                    mlirProbed = value instanceof Number && ((Number) value).intValue() == 1;
                }
            }
            mlirEnabled = mlirProbed;
            log.info("Vulkan MLIR enabled: {}", mlirEnabled);

            if (Boolean.getBoolean("nd4j.vulkan.test.requireTriton")) {
                boolean tritonAvailable = (boolean) bindingsClass
                        .getMethod("isTritonAvailable").invoke(nativeOps);
                if (!tritonAvailable) {
                    throw new IllegalStateException(
                            "Vulkan was built without the shared Triton DSP/compiler stack");
                }
                log.info("Vulkan Triton DSP/compiler stack available: true");
            }

        } catch (ClassNotFoundException e) {
            log.warn("VulkanReplayEquivalenceTest: Vulkan NativeOps not on the test classpath: {}",
                     VULKAN_BINDINGS_CLASS);
            nativeOps = null;
            vulkanDevicePresent = false;
            selectedDeviceId = 0;
            selectedDeviceName = null;
            mlirEnabled = false;
        } catch (Exception e) {
            throw new IllegalStateException(
                    "Vulkan NativeOps loaded but device/capability initialization failed", e);
        }
    }

    @AfterAll
    static void teardown() {
        // Nothing to clean up — Nd4j handles its own lifecycle.
    }

    // ── skip helpers ──────────────────────────────────────────────────────────

    private static void requireVulkanDevice() {
        assumeTrue(nativeOps != null,
                "Vulkan NativeOps (" + VULKAN_BINDINGS_CLASS + ") not on classpath — run with -Ptest-vulkan");
        assumeTrue(vulkanDevicePresent,
                "getAvailableDevices()==0 — no Vulkan device present; lavapipe not installed?");
        activateSelectedDevice();
    }

    private static void activateSelectedDevice() {
        Object result = invokeNative(
                "setDevice", new Class<?>[]{int.class}, selectedDeviceId);
        assertEquals(1, ((Number) result).intValue(),
                "setDevice(" + selectedDeviceId + ") must succeed");
    }

    private static void requireMlir() {
        requireVulkanDevice();
        assumeTrue(mlirEnabled,
                "HAVE_MLIR=0 in this chip build — the Vulkan integration suite requires MLIR. "
                + "Build the Vulkan Maven profile with MLIR and Triton enabled.");
    }

    /**
     * Returns true if the selected Vulkan device advertises fp16 (HALF) compute
     * support via the isFp16Supported(int deviceId) JNI method.
     * Returns false if the method is not available (capability-gate falls back to skip).
     */
    private static boolean isFp16Supported() {
        try {
            java.lang.reflect.Method m = nativeOps.getClass().getMethod("isFp16Supported", int.class);
            Object val = m.invoke(nativeOps, selectedDeviceId);
            if (val instanceof Boolean) return (Boolean) val;
        } catch (Exception e) {
            log.debug("isFp16Supported() not available on this build: {}", e.getMessage());
        }
        return false;
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    private long readReplayDispatchCount() {
        try {
            Method m = nativeOps.getClass().getMethod("getDspCounterValue", String.class, String.class);
            Object val = m.invoke(nativeOps, "GRAPH_REPLAY", "vulkan_backend REPLAY_DONE");
            if (val instanceof Number) return ((Number) val).longValue();
        } catch (Exception e) {
            log.debug("getDspCounterValue not available: {}", e.getMessage());
        }
        return -1L;
    }

    private static Object invokeNative(String methodName, Class<?>[] parameterTypes, Object... args) {
        try {
            Method method = nativeOps.getClass().getMethod(methodName, parameterTypes);
            return method.invoke(nativeOps, args);
        } catch (ReflectiveOperationException e) {
            throw new AssertionError("Required Vulkan NativeOps method unavailable: " + methodName, e);
        }
    }

    private static long jsonLongMetric(String json, String key) {
        Matcher matcher = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*([0-9]+)")
                .matcher(json);
        assertTrue(matcher.find(), () -> "Missing Vulkan diagnostic metric '" + key + "':\n" + json);
        return Long.parseLong(matcher.group(1));
    }

    private static String jsonStringMetric(String json, String key) {
        Matcher matcher = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*\"([^\"]+)\"")
                .matcher(json);
        assertTrue(matcher.find(), () -> "Missing Vulkan diagnostic metric '" + key + "':\n" + json);
        return matcher.group(1);
    }

    private static void assertWithinTolerance(float[] expected, float[] actual, float tol,
                                               String context) {
        assertEquals(expected.length, actual.length, context + ": array length mismatch");
        for (int i = 0; i < expected.length; i++) {
            float diff = Math.abs(expected[i] - actual[i]);
            assertTrue(diff <= tol,
                    context + ": element[" + i + "] expected=" + expected[i]
                    + " actual=" + actual[i] + " diff=" + diff + " > tol=" + tol);
        }
    }

    /** Run N eager executions and return the last result; this helper is not DSP replay evidence. */
    private float[] runNEagerSteps(int steps, java.util.function.Supplier<INDArray> supplier) {
        float[] lastResult = null;
        for (int step = 0; step < steps; step++) {
            INDArray result = supplier.get();
            Nd4j.getExecutioner().commit();
            lastResult = result.toFloatVector();
        }
        return lastResult;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // T3.1 — matmul replay (pre-Wave-1, original test)
    // ══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("T3.1: SameDiff matmul chain executes and replays real Vulkan kernels")
    void testMatmulReplayEquivalence() {
        assertNotNull(nativeOps, "Vulkan NativeOps must be present under -Ptest-vulkan");
        assertTrue(vulkanDevicePresent, "At least one Vulkan device must enumerate");
        assertTrue(mlirEnabled, "The Vulkan chip must expose HAVE_MLIR=1");
        activateSelectedDevice();

        invokeNative("dspDiagClear", new Class<?>[0]);
        invokeNative("dspDiagSetCategories", new Class<?>[]{int.class}, 1 << 16);
        invokeNative("dspDiagSetLevel", new Class<?>[]{int.class}, 2);

        String enumeratedDevice = (String) invokeNative(
                "getDeviceName", new Class<?>[]{int.class}, selectedDeviceId);
        assertNotNull(enumeratedDevice,
                "Selected Vulkan device " + selectedDeviceId + " has no name");
        assertFalse(enumeratedDevice.isBlank(),
                "Selected Vulkan device " + selectedDeviceId + " has an empty name");

        float[] bData = new float[8 * 4];
        for (int i = 0; i < 4; i++) {
            bData[i * 4 + i] = 1.0f;
        }
        float[] cData = {
                1.0f, 0.0f,
                0.0f, 1.0f,
                1.0f, 0.0f,
                0.0f, 1.0f
        };

        SameDiff sameDiff = SameDiff.create();
        String diagnosticJson;
        try {
            SDVariable a = sameDiff.placeHolder("a", DataType.FLOAT, 4, 8);
            SDVariable b = sameDiff.placeHolder("b", DataType.FLOAT, 8, 4);
            SDVariable c = sameDiff.placeHolder("c", DataType.FLOAT, 4, 2);
            SDVariable first = sameDiff.mmul("first", a, b);
            sameDiff.mmul("out", first, c);

            sameDiff.getSessions().clear();
            assertEquals("VULKAN", System.getProperty(ND4JSystemProperties.DSP_GRAPH_EXECUTION_MODE),
                    "Strict Vulkan replay must be selected through the universal DSP graph-execution property");
            sameDiff.setDspAutoCompileEnabled(true);
            sameDiff.setDspNativeAutoCompileEnabled(true);

            INDArray aInput = Nd4j.zeros(DataType.FLOAT, 4, 8);
            Map<String, INDArray> inputs = new LinkedHashMap<>();
            inputs.put("a", aInput);
            inputs.put("b", Nd4j.create(bData, new long[]{8, 4}));
            inputs.put("c", Nd4j.create(cData, new long[]{4, 2}));

            for (int step = 0; step < 24; step++) {
                float[] aData = new float[4 * 8];
                for (int i = 0; i < aData.length; i++) {
                    aData[i] = step + i * 0.125f;
                }
                aInput.assign(Nd4j.create(aData, new long[]{4, 8}));

                INDArray actual = sameDiff.output(inputs, "out").get("out");
                assertNotNull(actual, "SameDiff returned no output at step " + step);

                float[] expected = new float[4 * 2];
                for (int row = 0; row < 4; row++) {
                    int aOffset = row * 8;
                    expected[row * 2] = aData[aOffset] + aData[aOffset + 2];
                    expected[row * 2 + 1] = aData[aOffset + 1] + aData[aOffset + 3];
                }
                assertWithinTolerance(expected, actual.toFloatVector(), 1e-5f,
                        "Vulkan matmul replay step " + step);
            }

            int planPhase = DspPlanAssertions.getPlanPhase(sameDiff);
            int totalReplays = DspPlanAssertions.getTotalGraphReplays(sameDiff);
            String segmentState = DspPlanAssertions.snapshotSegmentState(sameDiff, 0);
            log.info("Vulkan DSP state: phase={} totalReplays={} segment={}",
                    planPhase, totalReplays, segmentState);

            assertEquals(2, planPhase, "Vulkan DSP plan did not reach REPLAYING");
            assertTrue(totalReplays > 0, "Vulkan DSP plan reported zero graph replays");
            DspPlanAssertions.assertAllSegmentsCompiledWith(
                    sameDiff, "vulkan-native", "real Vulkan kernel dispatch gate");
            DspPlanAssertions.assertNoSlotBySlotFallback(
                    sameDiff, "real Vulkan kernel dispatch gate");

            diagnosticJson = (String) invokeNative(
                    "dspDiagGetJsonReport", new Class<?>[0]);
        } finally {
            sameDiff.close();
        }

        assertNotNull(diagnosticJson, "Vulkan DSP diagnostic report is null");
        assertTrue(diagnosticJson.contains("vulkan_backend REPLAY_DONE"),
                () -> "No Vulkan REPLAY_DONE event was emitted:\n" + diagnosticJson);

        String replayDevice = jsonStringMetric(diagnosticJson, "device_name");
        long replayCount = jsonLongMetric(diagnosticJson, "replay_count");
        long dispatches = jsonLongMetric(diagnosticJson, "num_dispatches");

        assertEquals(enumeratedDevice, replayDevice,
                "Vulkan replay ran on a different device than selected device "
                        + selectedDeviceId);
        assertTrue(replayCount > 0,
                () -> "Vulkan replay_count must be positive:\n" + diagnosticJson);
        assertTrue(dispatches > 0,
                () -> "Vulkan num_dispatches must be positive:\n" + diagnosticJson);

        if (Boolean.getBoolean("nd4j.vulkan.test.requireHardware")) {
            String lowerName = replayDevice.toLowerCase(Locale.ROOT);
            assertFalse(lowerName.contains("llvmpipe") || lowerName.contains("lavapipe"),
                    "Strict hardware gate selected a software Vulkan device: " + replayDevice);
        }

        log.info("Real Vulkan replay proven: deviceId={} device='{}' replayCount={} dispatches={}",
                selectedDeviceId, replayDevice, replayCount, dispatches);
    }

    @Test
    @DisplayName("T3.1b: persistent uniform RNG replay consumes evolving generator state")
    void testPersistentUniformRandomStateEvolution() {
        requireMlir();

        final double from = -1.75;
        final double to = 2.25;
        final int length = 256;
        SameDiff sameDiff = SameDiff.create();
        try {
            SDVariable random = sameDiff.random.uniform(
                    from, to, DataType.FLOAT, length);
            String outputName = random.name();

            sameDiff.getSessions().clear();
            sameDiff.setDspAutoCompileEnabled(true);
            sameDiff.setDspNativeAutoCompileEnabled(true);

            Map<String, INDArray> inputs = new LinkedHashMap<>();
            float[] previousValues = null;
            for (int step = 0; step < 24; step++) {
                INDArray output = sameDiff.output(inputs, outputName).get(outputName);
                assertNotNull(output, "Uniform RNG returned no output at step " + step);
                float[] values = output.toFloatVector();
                assertEquals(length, values.length,
                        "Uniform RNG output length mismatch at step " + step);
                for (int index = 0; index < values.length; index++) {
                    float value = values[index];
                    assertTrue(Float.isFinite(value) && value >= from && value < to,
                            "Uniform RNG value outside [" + from + ", " + to + ") at step "
                                    + step + " index " + index + ": " + value);
                }
                if (previousValues != null) {
                    assertFalse(Arrays.equals(previousValues, values),
                            "Vulkan reused frozen RNG state at step " + step);
                }
                previousValues = values;
            }

            assertEquals(2, DspPlanAssertions.getPlanPhase(sameDiff),
                    "Uniform RNG plan did not reach REPLAYING");
            assertTrue(DspPlanAssertions.getTotalGraphReplays(sameDiff) >= 2,
                    "Uniform RNG plan did not report multiple native graph replays");
            DspPlanAssertions.assertAllSegmentsCompiledWith(
                    sameDiff, "vulkan-native", "persistent Vulkan RNG replay gate");
            DspPlanAssertions.assertNoSlotBySlotFallback(
                    sameDiff, "persistent Vulkan RNG replay gate");
        } finally {
            sameDiff.close();
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // T3.2 — Empty-capture guard (original test)
    // ══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("T3.2: an unmapped descriptor is rejected without replay or host execution")
    void testEmptyCaptureGuard() {
        requireMlir();

        int n = 16;
        float[] data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float) (i + 1);

        long dispatchCountBefore = readReplayDispatchCount();
        IllegalStateException failure = assertThrows(IllegalStateException.class,
                () -> Nd4j.cumsum(Nd4j.create(data, new long[]{n}), 0));
        assertNotNull(failure.getMessage());
        assertTrue(failure.getMessage().contains(
                        "Vulkan eager execution does not support this descriptor hash"),
                "Unmapped Vulkan descriptors must fail at the device admission boundary: "
                        + failure.getMessage());

        long dispatchCountAfter = readReplayDispatchCount();
        if (dispatchCountBefore >= 0) {
            assertEquals(dispatchCountBefore, dispatchCountAfter,
                    "GRAPH_REPLAY counter must not increase for a rejected descriptor");
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // T3.3 — Abort-cleanliness (original test)
    // ══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("T3.3: rejected capture leaves subsequent Vulkan execution clean")
    void testAbortCleanliness() {
        requireMlir();

        int n = 8;
        float[] aData = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f};
        float[] bData = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f};

        IllegalStateException failure = assertThrows(IllegalStateException.class, () -> {
            INDArray a = Nd4j.create(aData, new long[]{n});
            INDArray b = Nd4j.create(bData, new long[]{n});
            INDArray sum = a.add(b);
            Nd4j.cumsum(sum, 0);
        });
        assertNotNull(failure.getMessage());
        assertTrue(failure.getMessage().contains(
                        "Vulkan eager execution does not support this descriptor hash"),
                "Rejected capture must report the Vulkan admission failure: "
                        + failure.getMessage());

        INDArray cleanA = Nd4j.create(new float[]{1f, 2f, 3f, 4f}, new long[]{4});
        INDArray cleanB = Nd4j.create(new float[]{1f, 1f, 1f, 1f}, new long[]{4});
        INDArray cleanResult = assertDoesNotThrow(() -> cleanA.add(cleanB),
                "Subsequent Vulkan op after rejection should not throw");
        Nd4j.getExecutioner().commit();
        assertWithinTolerance(new float[]{2f, 3f, 4f, 5f}, cleanResult.toFloatVector(),
                1e-6f, "post-rejection Vulkan op");
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Wave 1 T3 tests
    // ══════════════════════════════════════════════════════════════════════════

    // ── EAGER.W1.add ─────────────────────────────────────────────────────────────

    /**
     * EAGER.W1.add: Verify elementwise add produces CPU-oracle-matching results
     * across three eager executions.
     */
    @Test
    @DisplayName("EAGER.W1.add: elementwise add eager execution matches CPU oracle")
    void testAddEagerEquivalence() {
        requireMlir();

        int n = 32;
        float[] aData = new float[n];
        float[] bData = new float[n];
        for (int i = 0; i < n; i++) { aData[i] = (i % 11) * 0.1f - 0.5f; bData[i] = (i % 7) * 0.2f + 0.1f; }

        float[] expectedData = Nd4j.create(aData, new long[]{n})
                .add(Nd4j.create(bData, new long[]{n})).toFloatVector();

        float[] lastResult = runNEagerSteps(3, () ->
                Nd4j.create(aData, new long[]{n}).add(Nd4j.create(bData, new long[]{n})));

        assertNotNull(lastResult);
        assertWithinTolerance(expectedData, lastResult, 1e-5f, "add step-N vs CPU oracle");
        log.info("EAGER.W1.add: PASS, n={}, result[0]={}", n, lastResult[0]);
    }

    // ── EAGER.W1.subtract ────────────────────────────────────────────────────────

    /**
     * EAGER.W1.subtract: Verify elementwise subtract (Wave 1 ElementwiseBinaryToSpirv).
     */
    @Test
    @DisplayName("EAGER.W1.subtract: elementwise subtract eager execution matches CPU oracle")
    void testSubtractEagerEquivalence() {
        requireMlir();

        int n = 32;
        float[] aData = new float[n];
        float[] bData = new float[n];
        for (int i = 0; i < n; i++) { aData[i] = (i % 13) * 0.15f; bData[i] = (i % 9) * 0.05f; }

        float[] expectedData = Nd4j.create(aData, new long[]{n})
                .sub(Nd4j.create(bData, new long[]{n})).toFloatVector();

        float[] lastResult = runNEagerSteps(3, () ->
                Nd4j.create(aData, new long[]{n}).sub(Nd4j.create(bData, new long[]{n})));

        assertNotNull(lastResult);
        assertWithinTolerance(expectedData, lastResult, 1e-5f, "subtract step-N vs CPU oracle");
        log.info("EAGER.W1.subtract: PASS, n={}, result[0]={}", n, lastResult[0]);
    }

    // ── EAGER.W1.multiply ────────────────────────────────────────────────────────

    /**
     * EAGER.W1.multiply: Verify elementwise multiply (Wave 1 ElementwiseBinaryToSpirv).
     */
    @Test
    @DisplayName("EAGER.W1.multiply: elementwise multiply eager execution matches CPU oracle")
    void testMultiplyEagerEquivalence() {
        requireMlir();

        int n = 32;
        float[] aData = new float[n];
        float[] bData = new float[n];
        for (int i = 0; i < n; i++) { aData[i] = (i % 5) * 0.1f + 0.1f; bData[i] = (i % 3) * 0.2f + 0.2f; }

        float[] expectedData = Nd4j.create(aData, new long[]{n})
                .mul(Nd4j.create(bData, new long[]{n})).toFloatVector();

        float[] lastResult = runNEagerSteps(3, () ->
                Nd4j.create(aData, new long[]{n}).mul(Nd4j.create(bData, new long[]{n})));

        assertNotNull(lastResult);
        assertWithinTolerance(expectedData, lastResult, 1e-5f, "multiply step-N vs CPU oracle");
        log.info("EAGER.W1.multiply: PASS, n={}, result[0]={}", n, lastResult[0]);
    }

    // ── EAGER.W1.divide ──────────────────────────────────────────────────────────

    /**
     * EAGER.W1.divide: Verify elementwise divide (Wave 1 ElementwiseBinaryToSpirv).
     * Uses non-zero denominator to avoid NaN.
     */
    @Test
    @DisplayName("EAGER.W1.divide: elementwise divide eager execution matches CPU oracle")
    void testDivideEagerEquivalence() {
        requireMlir();

        int n = 32;
        float[] aData = new float[n];
        float[] bData = new float[n];
        for (int i = 0; i < n; i++) { aData[i] = (i % 7) * 0.5f + 1.0f; bData[i] = (i % 5) * 0.3f + 0.5f; }

        float[] expectedData = Nd4j.create(aData, new long[]{n})
                .div(Nd4j.create(bData, new long[]{n})).toFloatVector();

        float[] lastResult = runNEagerSteps(3, () ->
                Nd4j.create(aData, new long[]{n}).div(Nd4j.create(bData, new long[]{n})));

        assertNotNull(lastResult);
        assertWithinTolerance(expectedData, lastResult, 1e-4f, "divide step-N vs CPU oracle");
        log.info("EAGER.W1.divide: PASS, n={}, result[0]={}", n, lastResult[0]);
    }



    // ── EAGER.W1.swish ────────────────────────────────────────────────────────────

    /**
     * EAGER.W1.swish: Verify silu(x) = x * sigmoid(x) matches CPU oracle across ≥3 steps.

     */
    @Test
    @DisplayName("EAGER.W1.swish: silu activation eager execution matches CPU oracle")
    void testSwishEagerEquivalence() {
        requireMlir();

        int n = 32;
        float[] data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (i - n / 2) * 0.3f;

        // CPU oracle: x * sigmoid(x)
        INDArray xOracle = Nd4j.create(data, new long[]{n});
        float[] expectedData = xOracle.mul(Transforms.sigmoid(xOracle.dup())).toFloatVector();

        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray x = Nd4j.create(data, new long[]{n});
            // Swish is ND4J's concrete SiLU-equivalent transform op.
            return Nd4j.exec(new Swish(x));
        });

        assertNotNull(lastResult);
        // Tolerance 1e-4 because polynomial approximation of sigmoid differs slightly.
        assertWithinTolerance(expectedData, lastResult, 1e-4f, "swish step-N vs CPU oracle");
        log.info("EAGER.W1.swish: PASS, n={}, result[0]={}", n, lastResult[0]);
    }

    // ── EAGER.W1.gelu ────────────────────────────────────────────────────────────

    /**
     * EAGER.W1.gelu: Verify gelu(x) ≈ 0.5*x*(1+tanh(√(2/π)*(x+0.044715*x³)))
     * matches CPU oracle across ≥3 steps.

     */
    @Test
    @DisplayName("EAGER.W1.gelu: gelu activation eager execution matches CPU oracle")
    void testGeluEagerEquivalence() {
        requireMlir();

        int n = 32;
        float[] data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (i - n / 2) * 0.2f;

        INDArray xOracle = Nd4j.create(data, new long[]{n});
        float[] expectedData = Nd4j.exec(new GELU(xOracle)).toFloatVector();

        float[] lastResult = runNEagerSteps(3, () ->
                Nd4j.exec(new GELU(Nd4j.create(data, new long[]{n}))));

        assertNotNull(lastResult);
        // Tolerance slightly higher due to tanh polynomial approximation.
        assertWithinTolerance(expectedData, lastResult, 2e-4f, "gelu step-N vs CPU oracle");
        log.info("EAGER.W1.gelu: PASS, n={}, result[0]={}", n, lastResult[0]);
    }

    // ── EAGER.W1.tanh ────────────────────────────────────────────────────────────

    /**
     * EAGER.W1.tanh: Verify tanh activation matches CPU oracle across ≥3 steps.

     */
    @Test
    @DisplayName("EAGER.W1.tanh: tanh activation eager execution matches CPU oracle")
    void testTanhEagerEquivalence() {
        requireMlir();

        int n = 32;
        float[] data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (i - n / 2) * 0.15f;

        float[] expectedData = Transforms.tanh(Nd4j.create(data, new long[]{n}), true).toFloatVector();

        float[] lastResult = runNEagerSteps(3, () ->
                Transforms.tanh(Nd4j.create(data, new long[]{n}), true));

        assertNotNull(lastResult);
        assertWithinTolerance(expectedData, lastResult, 2e-4f, "tanh step-N vs CPU oracle");
        log.info("EAGER.W1.tanh: PASS, n={}, result[0]={}", n, lastResult[0]);
    }

    // ── EAGER.W1.sigmoid ─────────────────────────────────────────────────────────

    /**
     * EAGER.W1.sigmoid: Verify sigmoid activation matches CPU oracle across ≥3 steps.

     */
    @Test
    @DisplayName("EAGER.W1.sigmoid: sigmoid activation eager execution matches CPU oracle")
    void testSigmoidEagerEquivalence() {
        requireMlir();

        int n = 32;
        float[] data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (i - n / 2) * 0.25f;

        float[] expectedData = Transforms.sigmoid(Nd4j.create(data, new long[]{n}), true).toFloatVector();

        float[] lastResult = runNEagerSteps(3, () ->
                Transforms.sigmoid(Nd4j.create(data, new long[]{n}), true));

        assertNotNull(lastResult);
        assertWithinTolerance(expectedData, lastResult, 1e-4f, "sigmoid step-N vs CPU oracle");
        log.info("EAGER.W1.sigmoid: PASS, n={}, result[0]={}", n, lastResult[0]);
    }

    // ── EAGER.W1.relu ────────────────────────────────────────────────────────────

    /**
     * EAGER.W1.relu: Verify relu activation matches CPU oracle across ≥3 steps.

     */
    @Test
    @DisplayName("EAGER.W1.relu: relu activation eager execution matches CPU oracle")
    void testReluEagerEquivalence() {
        requireMlir();

        int n = 32;
        float[] data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (i - n / 2) * 0.2f;

        float[] expectedData = Transforms.relu(Nd4j.create(data, new long[]{n}), true).toFloatVector();

        float[] lastResult = runNEagerSteps(3, () ->
                Transforms.relu(Nd4j.create(data, new long[]{n}), true));

        assertNotNull(lastResult);
        assertWithinTolerance(expectedData, lastResult, 1e-6f, "relu step-N vs CPU oracle");
        log.info("EAGER.W1.relu: PASS, n={}, result[0]={}", n, lastResult[0]);
    }

    // ── EAGER.W1.softmax ─────────────────────────────────────────────────────────

    /**
     * EAGER.W1.softmax: Verify row-wise softmax (last-dim reduce) matches CPU oracle.
     * Input: [4, 8] — rows=4, dim=8.

     *
     * Verification:
     *   (a) outputs match CPU oracle within 1e-5 tolerance
     *   (b) each row sums to ≈1.0 (sanity check)
     */
    @Test
    @DisplayName("EAGER.W1.softmax: row-wise softmax eager execution matches CPU oracle")
    void testSoftmaxEagerEquivalence() {
        requireMlir();

        int rows = 4;
        int dim  = 8;
        float[] data = new float[rows * dim];
        for (int i = 0; i < data.length; i++) data[i] = (i % 11) * 0.3f - 1.5f;

        // CPU oracle via Nd4j built-in softmax (axis=-1 == dim 1 for 2-D).
        INDArray xOracle = Nd4j.create(data, new long[]{rows, dim});
        float[] expectedData = Transforms.softmax(xOracle, true).toFloatVector();

        float[] lastResult = runNEagerSteps(3, () ->
                Transforms.softmax(Nd4j.create(data, new long[]{rows, dim}), true));

        assertNotNull(lastResult);
        assertWithinTolerance(expectedData, lastResult, 1e-5f, "softmax step-N vs CPU oracle");

        // Sanity: each row sums to 1.0.
        for (int r = 0; r < rows; r++) {
            float rowSum = 0;
            for (int c = 0; c < dim; c++) rowSum += lastResult[r * dim + c];
            assertTrue(Math.abs(rowSum - 1.0f) < 1e-4f,
                    "softmax row " + r + " sum=" + rowSum + " expected ≈1.0");
        }
        log.info("EAGER.W1.softmax: PASS, shape=[{},{}], result[0]={}", rows, dim, lastResult[0]);
    }

    // ── EAGER.W1.layer_norm ──────────────────────────────────────────────────────

    /**
     * EAGER.W1.layer_norm: Verify layer normalization (mean+variance two-pass, f32)
     * matches CPU oracle across ≥3 steps.
     * Input: [4, 16] — rows=4, hidden=16.

     *
     * CPU oracle: manual computation (mean, variance, normalize) to avoid
     * depending on Nd4j's layer_norm implementation path.
     */
    @Test
    @DisplayName("EAGER.W1.layer_norm: layer normalization eager execution matches CPU oracle")
    void testLayerNormEagerEquivalence() {
        requireMlir();

        int rows   = 4;
        int hidden = 16;
        float eps  = 1e-5f;

        float[] data  = new float[rows * hidden];
        float[] gamma = new float[hidden];
        float[] beta  = new float[hidden];
        for (int i = 0; i < data.length; i++) data[i] = (i % 13) * 0.2f - 1.0f;
        for (int i = 0; i < hidden; i++) { gamma[i] = 1.0f + i * 0.05f; beta[i] = i * 0.01f; }

        // CPU oracle: manual layer_norm.
        float[] expectedData = new float[rows * hidden];
        for (int r = 0; r < rows; r++) {
            // Pass 1: mean
            float mean = 0;
            for (int c = 0; c < hidden; c++) mean += data[r * hidden + c];
            mean /= hidden;
            // Pass 2: variance
            float variance = 0;
            for (int c = 0; c < hidden; c++) {
                float diff = data[r * hidden + c] - mean;
                variance += diff * diff;
            }
            variance /= hidden;
            float normFactor = (float)(1.0 / Math.sqrt(variance + eps));
            // Pass 3: normalize + scale + bias
            for (int c = 0; c < hidden; c++) {
                float normed = (data[r * hidden + c] - mean) * normFactor;
                expectedData[r * hidden + c] = normed * gamma[c] + beta[c];
            }
        }

        INDArray gammaArr = Nd4j.create(gamma, new long[]{hidden});
        INDArray betaArr  = Nd4j.create(beta,  new long[]{hidden});

        // Exercise the concrete layer_norm eager op; strict replay owns emitter evidence.
        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray x = Nd4j.create(data, new long[]{rows, hidden});
            return Nd4j.exec(new LayerNorm(
                    x, gammaArr.dup(), betaArr.dup(), x.dup(), true, 1))[0];
        });

        assertNotNull(lastResult);
        assertWithinTolerance(expectedData, lastResult, 5e-4f, "layer_norm step-N vs CPU oracle");
        log.info("EAGER.W1.layer_norm: PASS, shape=[{},{}], result[0]={}", rows, hidden, lastResult[0]);
    }

    // ── EAGER.W1.rms_norm_f16 ────────────────────────────────────────────────────

    /**
     * EAGER.W1.rms_norm_f16: Verify RMS-norm with f16 input matches f32 CPU oracle
     * within the wider tolerance expected from half-precision.
     * Tests the Wave 1 f16 extension of RmsNormToSpirv.
     *
     * Skip condition: device must support f16 (checked via isFp16Supported()
     * NativeOps method; skip-clean if not present or returns false).
     */
    @Test
    @DisplayName("EAGER.W1.rms_norm_f16: f16 rms_norm eager execution matches f32 CPU oracle within fp16 tolerance")
    void testRmsNormF16EagerEquivalence() {
        requireMlir();

        // Probe fp16 support on the device.
        boolean fp16Supported = false;
        try {
            Method m = nativeOps.getClass().getMethod("isFp16Supported", int.class);
            fp16Supported = (boolean) m.invoke(nativeOps, 0);
        } catch (Exception e) {
            log.debug("isFp16Supported() not available: {}", e.getMessage());
        }
        assumeTrue(fp16Supported, "Vulkan device does not advertise fp16 support — skipping f16 rms_norm test");

        int rows   = 4;
        int hidden = 32;
        float eps  = 1e-6f;

        float[] data  = new float[rows * hidden];
        float[] gamma = new float[hidden];
        for (int i = 0; i < data.length; i++) data[i] = (i % 7) * 0.1f - 0.3f;
        for (int i = 0; i < hidden; i++) gamma[i] = 1.0f;

        // f32 CPU oracle: rms_norm(x, gamma).
        float[] expectedData = new float[rows * hidden];
        for (int r = 0; r < rows; r++) {
            float sumSq = 0;
            for (int c = 0; c < hidden; c++) sumSq += data[r * hidden + c] * data[r * hidden + c];
            float normFactor = (float)(1.0 / Math.sqrt(sumSq / hidden + eps));
            for (int c = 0; c < hidden; c++) {
                expectedData[r * hidden + c] = data[r * hidden + c] * normFactor * gamma[c];
            }
        }

        // Run 3 steps with f16 input arrays.
        // Nd4j.create(data, shape, DataType.HALF) creates a fp16 NDArray.
        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray xF16 = Nd4j.create(data, new long[]{rows, hidden},
                                        DataType.HALF);
            INDArray gF16 = Nd4j.create(gamma, new long[]{hidden},
                                        DataType.HALF);
            // Exercise the concrete rms_norm eager op; strict replay owns emitter evidence.
            return Nd4j.exec(new RmsNorm(xF16, gF16, xF16.dup(), eps))[0];
        });

        assertNotNull(lastResult);
        // fp16 tolerance is much wider than fp32 (~1%).
        assertWithinTolerance(expectedData, lastResult, 0.02f, "rms_norm_f16 step-N vs f32 CPU oracle");
        log.info("EAGER.W1.rms_norm_f16: PASS, shape=[{},{}], result[0]={}", rows, hidden, lastResult[0]);
    }

    // ── EAGER.W1.rope_f16 ────────────────────────────────────────────────────────

    /**
     * EAGER.W1.rope_f16: Verify RoPE with f16 input matches f32 CPU oracle within
     * fp16 tolerance. This is eager numerical evidence only.
     *
     * Skip condition: same fp16 device support check as rms_norm_f16.
     */
    @Test
    @DisplayName("EAGER.W1.rope_f16: f16 rope eager execution matches f32 CPU oracle within fp16 tolerance")
    void testRopeF16EagerEquivalence() {
        requireMlir();

        boolean fp16Supported = false;
        try {
            Method m = nativeOps.getClass().getMethod("isFp16Supported", int.class);
            fp16Supported = (boolean) m.invoke(nativeOps, 0);
        } catch (Exception e) {
            log.debug("isFp16Supported() not available: {}", e.getMessage());
        }
        assumeTrue(fp16Supported, "Vulkan device does not advertise fp16 support — skipping f16 rope test");

        // Small RoPE shape: [B=1, S=4, H=2, D=8], halfD=4
        int B = 1, S = 4, H = 2, D = 8, halfD = 4;
        float[] xData   = new float[B * S * H * D];
        float[] cosData = new float[S * halfD];
        float[] sinData = new float[S * halfD];
        for (int i = 0; i < xData.length; i++) xData[i] = (i % 7) * 0.1f - 0.3f;
        for (int i = 0; i < cosData.length; i++) cosData[i] = (float)Math.cos(i * 0.1);
        for (int i = 0; i < sinData.length; i++) sinData[i] = (float)Math.sin(i * 0.1);

        // f32 CPU oracle: apply RoPE pair rotation.
        float[] expectedData = new float[xData.length];
        for (int b = 0; b < B; b++) for (int s = 0; s < S; s++)
          for (int h = 0; h < H; h++) for (int p = 0; p < halfD; p++) {
            int baseIdx = ((b * S + s) * H + h) * D;
            int cosIdx  = s * halfD + p;
            float x1 = xData[baseIdx + 2*p];
            float x2 = xData[baseIdx + 2*p+1];
            float c  = cosData[cosIdx];
            float sn = sinData[cosIdx];
            expectedData[baseIdx + 2*p]   = x1 * c - x2 * sn;
            expectedData[baseIdx + 2*p+1] = x2 * c + x1 * sn;
        }

        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray xF16   = Nd4j.create(xData, new long[]{B, S, H, D},
                                          DataType.HALF);
            INDArray cosF16 = Nd4j.create(cosData, new long[]{S, halfD},
                                          DataType.HALF);
            INDArray sinF16 = Nd4j.create(sinData, new long[]{S, halfD},
                                          DataType.HALF);
            // Exercise fused_rope with precomputed cos/sin caches directly.
            return Nd4j.exec(new FusedRoPE(
                    xF16, cosF16, sinF16, xF16.dup(), FusedRoPE.ROPE_TYPE_STANDARD))[0];
        });

        assertNotNull(lastResult);
        // fp16 tolerance.
        assertWithinTolerance(expectedData, lastResult, 0.02f, "rope_f16 step-N vs f32 CPU oracle");
        log.info("EAGER.W1.rope_f16: PASS, shape=[{},{},{},{}], result[0]={}",
                 B, S, H, D, lastResult[0]);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Wave 2 T3 tests — transforms/shape group
    // ══════════════════════════════════════════════════════════════════════════

    // ── EAGER.W2.gather ──────────────────────────────────────────────────────────

    /**
     * EAGER.W2.gather: Verify axis-0 gather (embedding lookup) matches CPU oracle
     * across ≥3 execution steps.
     *
     * Shape: table=[V=8, D=16], indices=[I=4] (int32), output=[I=4, D=16].

     *
     * Capability gate: skipped if device/MLIR not present (clean skip).
     */
    @Test
    @DisplayName("EAGER.W2.gather: axis-0 gather (embedding lookup) eager execution matches CPU oracle")
    void testGatherEagerEquivalence() {
        requireMlir();

        int V = 8;   // vocabulary / table rows
        int D = 16;  // feature dimension
        int I = 4;   // number of lookup indices

        // Build f32 embedding table: table[v, d] = v * 10 + d * 0.1
        float[] tableData = new float[V * D];
        for (int v = 0; v < V; v++) {
            for (int d = 0; d < D; d++) {
                tableData[v * D + d] = v * 10.0f + d * 0.1f;
            }
        }

        // Indices to look up (using values in [0, V))
        int[] idxData = {2, 5, 0, 7};

        // CPU oracle: manual gather
        float[] expectedData = new float[I * D];
        for (int i = 0; i < I; i++) {
            for (int d = 0; d < D; d++) {
                expectedData[i * D + d] = tableData[idxData[i] * D + d];
            }
        }

        // Execute the current eager Gather op directly (axis 0).
        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray t = Nd4j.create(tableData, new long[]{V, D});
            INDArray idx = Nd4j.createFromArray(idxData);
            return Nd4j.exec(new Gather(t, idx, 0))[0];
        });

        assertNotNull(lastResult, "gather returned null result");
        assertWithinTolerance(expectedData, lastResult, 1e-5f, "gather step-N vs CPU oracle");
        log.info("EAGER.W2.gather: PASS, table=[{},{}], indices={}, result[0]={}",
                 V, D, Arrays.toString(idxData), lastResult[0]);
    }

    // ── EAGER.W2.concat ──────────────────────────────────────────────────────────

    /**
     * EAGER.W2.concat: Verify concat along axis 1 (sequence dim) matches CPU oracle.
     *
     * Shapes: A=[2, 4, 8], B=[2, 3, 8] → C=[2, 7, 8] (axis=1).
     * This mimics KV-cache concat (past + current KV along sequence).

     */
    @Test
    @DisplayName("EAGER.W2.concat: concat along axis-1 (KV-cache-like) eager execution matches CPU oracle")
    void testConcatEagerEquivalence() {
        requireMlir();

        // Past KV slice: [batch=2, pastSeq=4, dim=8]
        int B = 2, pastS = 4, curS = 3, D = 8;

        float[] aData = new float[B * pastS * D];
        float[] bData = new float[B * curS * D];
        for (int i = 0; i < aData.length; i++) aData[i] = i * 0.01f;
        for (int i = 0; i < bData.length; i++) bData[i] = 100.0f + i * 0.01f;

        // CPU oracle: concatenate along axis 1
        INDArray aOracle = Nd4j.create(aData, new long[]{B, pastS, D});
        INDArray bOracle = Nd4j.create(bData, new long[]{B, curS, D});
        float[] expectedData = Nd4j.concat(1, aOracle, bOracle).toFloatVector();

        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray a = Nd4j.create(aData, new long[]{B, pastS, D});
            INDArray b = Nd4j.create(bData, new long[]{B, curS, D});
            return Nd4j.concat(1, a, b);
        });

        assertNotNull(lastResult, "concat returned null result");
        assertWithinTolerance(expectedData, lastResult, 1e-5f, "concat step-N vs CPU oracle");
        log.info("EAGER.W2.concat: PASS, A=[{},{},{}]+B=[{},{},{}]→C=[{},{},{}], result[0]={}",
                 B, pastS, D, B, curS, D, B, pastS + curS, D, lastResult[0]);
    }

    /**
     * EAGER.W2.concat_axis0: Verify concat along axis 0 matches CPU oracle.
     * Shape: A=[3, 8], B=[5, 8] → C=[8, 8] (axis=0).
     */
    @Test
    @DisplayName("EAGER.W2.concat_axis0: concat along axis-0 (row-stack) eager execution matches CPU oracle")
    void testConcatAxis0EagerEquivalence() {
        requireMlir();

        int r1 = 3, r2 = 5, D = 8;
        float[] aData = new float[r1 * D];
        float[] bData = new float[r2 * D];
        for (int i = 0; i < aData.length; i++) aData[i] = i * 0.1f;
        for (int i = 0; i < bData.length; i++) bData[i] = 50.0f + i * 0.1f;

        INDArray aOracle = Nd4j.create(aData, new long[]{r1, D});
        INDArray bOracle = Nd4j.create(bData, new long[]{r2, D});
        float[] expectedData = Nd4j.concat(0, aOracle, bOracle).toFloatVector();

        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray a = Nd4j.create(aData, new long[]{r1, D});
            INDArray b = Nd4j.create(bData, new long[]{r2, D});
            return Nd4j.concat(0, a, b);
        });

        assertNotNull(lastResult, "concat axis-0 returned null result");
        assertWithinTolerance(expectedData, lastResult, 1e-5f, "concat_axis0 step-N vs CPU oracle");
        log.info("EAGER.W2.concat_axis0: PASS, A=[{},{}]+B=[{},{}]→C=[{},{}], result[0]={}",
                 r1, D, r2, D, r1 + r2, D, lastResult[0]);
    }

    // ── EAGER.W2.transpose ───────────────────────────────────────────────────────

    /**
     * EAGER.W2.transpose_2d: Verify 2-D matrix transpose [M, N] → [N, M].

     */
    @Test
    @DisplayName("EAGER.W2.transpose_2d: 2-D matrix transpose eager execution matches CPU oracle")
    void testTranspose2DEagerEquivalence() {
        requireMlir();

        int M = 6, N = 4;
        float[] data = new float[M * N];
        for (int i = 0; i < data.length; i++) data[i] = i * 0.5f + 0.1f;

        INDArray xOracle = Nd4j.create(data, new long[]{M, N});
        float[] expectedData = xOracle.transpose().dup().toFloatVector();

        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray x = Nd4j.create(data, new long[]{M, N});
            return x.transpose().dup();
        });

        assertNotNull(lastResult, "transpose 2D returned null");
        assertWithinTolerance(expectedData, lastResult, 1e-6f, "transpose_2d step-N vs CPU oracle");
        log.info("EAGER.W2.transpose_2d: PASS, [{},{}]→[{},{}], result[0]={}",
                 M, N, N, M, lastResult[0]);
    }

    /**
     * EAGER.W2.permute_4d_bnsh_bsnh: Verify the BNSH→BSNH permutation (0,2,1,3)
     * used in every attention block matches CPU oracle across ≥3 steps.
     *
     * Input shape: [B=2, N=4, S=8, H=16]  (BNSH layout)
     * Output shape: [B=2, S=8, N=4, H=16] (BSNH layout)
     * Permutation: [0, 2, 1, 3]
     *

     */
    @Test
    @DisplayName("EAGER.W2.permute_4d_bnsh_bsnh: BNSH→BSNH (0,2,1,3) permute eager execution matches CPU oracle")
    void testPermute4dBnshBsnhEagerEquivalence() {
        requireMlir();

        int B = 2, N = 4, S = 8, H = 16;
        float[] data = new float[B * N * S * H];
        for (int i = 0; i < data.length; i++) data[i] = (i % 31) * 0.1f - 1.5f;

        // CPU oracle: permute(0, 2, 1, 3)
        INDArray xOracle = Nd4j.create(data, new long[]{B, N, S, H});
        float[] expectedData = xOracle.permute(0, 2, 1, 3).dup().toFloatVector();

        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray x = Nd4j.create(data, new long[]{B, N, S, H});
            return x.permute(0, 2, 1, 3).dup();
        });

        assertNotNull(lastResult, "permute 4D BNSH→BSNH returned null");
        assertWithinTolerance(expectedData, lastResult, 1e-6f,
                "permute_4d_bnsh_bsnh step-N vs CPU oracle");
        log.info("EAGER.W2.permute_4d_bnsh_bsnh: PASS, [{},{},{},{}]→[{},{},{},{}], result[0]={}",
                 B, N, S, H, B, S, N, H, lastResult[0]);
    }

    /**
     * EAGER.W2.transpose_3d: Verify 3-D transpose / permute (0,2,1) matches CPU oracle.
     * Input: [B=2, S=6, D=8] → [B=2, D=8, S=6]
     */
    @Test
    @DisplayName("EAGER.W2.transpose_3d: 3-D permute (0,2,1) eager execution matches CPU oracle")
    void testTranspose3DEagerEquivalence() {
        requireMlir();

        int B = 2, S = 6, D = 8;
        float[] data = new float[B * S * D];
        for (int i = 0; i < data.length; i++) data[i] = i * 0.2f - 5.0f;

        INDArray xOracle = Nd4j.create(data, new long[]{B, S, D});
        float[] expectedData = xOracle.permute(0, 2, 1).dup().toFloatVector();

        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray x = Nd4j.create(data, new long[]{B, S, D});
            return x.permute(0, 2, 1).dup();
        });

        assertNotNull(lastResult, "transpose 3D returned null");
        assertWithinTolerance(expectedData, lastResult, 1e-6f,
                "transpose_3d step-N vs CPU oracle");
        log.info("EAGER.W2.transpose_3d: PASS, [{},{},{}]→[{},{},{}], result[0]={}",
                 B, S, D, B, D, S, lastResult[0]);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Wave 3 T3 tests — reductions
    // ══════════════════════════════════════════════════════════════════════════

    // ── EAGER.W3.reduce_sum ──────────────────────────────────────────────────────

    /**
     * EAGER.W3.reduce_sum (last-dim): Verify reduce_sum over last axis matches CPU oracle.
     *
     * Input:  [6, 8] f32
     * Output: [6]    f32  (keepDims=false, axis=1)

     *
     * Capability gate: skipped if device/MLIR not present (clean skip).
     */
    @Test
    @DisplayName("EAGER.W3.reduce_sum_last_dim: reduce_sum over axis-1 eager execution matches CPU oracle")
    void testReduceSumLastDimEagerEquivalence() {
        requireMlir();

        int rows = 6, D = 8;
        float[] data = new float[rows * D];
        for (int i = 0; i < data.length; i++) data[i] = (i % 11) * 0.5f - 2.5f;

        // CPU oracle: sum over last dim
        INDArray xOracle = Nd4j.create(data, new long[]{rows, D});
        float[] expectedData = xOracle.sum(1).toFloatVector();

        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray x = Nd4j.create(data, new long[]{rows, D});
            return x.sum(1);
        });

        assertNotNull(lastResult, "reduce_sum returned null");
        assertEquals(rows, lastResult.length, "reduce_sum output length mismatch");
        assertWithinTolerance(expectedData, lastResult, 1e-4f,
                "reduce_sum_last_dim step-N vs CPU oracle");
        log.info("EAGER.W3.reduce_sum_last_dim: PASS, shape=[{},{}]→[{}], result[0]={}",
                 rows, D, rows, lastResult[0]);
    }

    /**
     * EAGER.W3.reduce_sum_full: Verify full-tensor reduce_sum (all axes) matches CPU oracle.
     *
     * Input:  [4, 10] f32
     * Output: scalar

     */
    @Test
    @DisplayName("EAGER.W3.reduce_sum_full: full-tensor reduce_sum eager execution matches CPU oracle")
    void testReduceSumFullEagerEquivalence() {
        requireMlir();

        int rows = 4, D = 10;
        float[] data = new float[rows * D];
        for (int i = 0; i < data.length; i++) data[i] = (i % 7) * 0.3f - 1.0f;

        INDArray xOracle = Nd4j.create(data, new long[]{rows, D});
        float expectedScalar = xOracle.sumNumber().floatValue();

        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray x = Nd4j.create(data, new long[]{rows, D});
            return x.sum();
        });

        assertNotNull(lastResult, "reduce_sum full returned null");
        assertEquals(1, lastResult.length, "reduce_sum full output should be scalar (length 1)");
        assertTrue(Math.abs(lastResult[0] - expectedScalar) < 1e-3f,
                "reduce_sum_full: expected=" + expectedScalar + " actual=" + lastResult[0]
                + " diff=" + Math.abs(lastResult[0] - expectedScalar));
        log.info("EAGER.W3.reduce_sum_full: PASS, shape=[{},{}]→scalar={}", rows, D, lastResult[0]);
    }

    // ── EAGER.W3.reduce_mean ─────────────────────────────────────────────────────

    /**
     * EAGER.W3.reduce_mean_last_dim: Verify reduce_mean over last axis matches CPU oracle.
     *
     * Input:  [5, 12] f32
     * Output: [5]     f32

     */
    @Test
    @DisplayName("EAGER.W3.reduce_mean_last_dim: reduce_mean over axis-1 eager execution matches CPU oracle")
    void testReduceMeanLastDimEagerEquivalence() {
        requireMlir();

        int rows = 5, D = 12;
        float[] data = new float[rows * D];
        for (int i = 0; i < data.length; i++) data[i] = (i % 9) * 0.4f - 1.8f;

        INDArray xOracle = Nd4j.create(data, new long[]{rows, D});
        float[] expectedData = xOracle.mean(1).toFloatVector();

        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray x = Nd4j.create(data, new long[]{rows, D});
            return x.mean(1);
        });

        assertNotNull(lastResult, "reduce_mean returned null");
        assertEquals(rows, lastResult.length, "reduce_mean output length mismatch");
        assertWithinTolerance(expectedData, lastResult, 1e-5f,
                "reduce_mean_last_dim step-N vs CPU oracle");
        log.info("EAGER.W3.reduce_mean_last_dim: PASS, shape=[{},{}]→[{}], result[0]={}",
                 rows, D, rows, lastResult[0]);
    }

    // ── EAGER.W3.reduce_max ──────────────────────────────────────────────────────

    /**
     * EAGER.W3.reduce_max_last_dim: Verify reduce_max over last axis matches CPU oracle.
     *
     * Input:  [4, 8] f32
     * Output: [4]    f32

     * Sanity: each output value must be ≥ all elements in its input row.
     */
    @Test
    @DisplayName("EAGER.W3.reduce_max_last_dim: reduce_max over axis-1 eager execution matches CPU oracle")
    void testReduceMaxLastDimEagerEquivalence() {
        requireMlir();

        int rows = 4, D = 8;
        float[] data = new float[rows * D];
        for (int i = 0; i < data.length; i++) data[i] = (i % 13) * 0.7f - 4.0f;

        INDArray xOracle = Nd4j.create(data, new long[]{rows, D});
        float[] expectedData = xOracle.max(1).toFloatVector();

        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray x = Nd4j.create(data, new long[]{rows, D});
            return x.max(1);
        });

        assertNotNull(lastResult, "reduce_max returned null");
        assertEquals(rows, lastResult.length, "reduce_max output length mismatch");
        assertWithinTolerance(expectedData, lastResult, 1e-5f,
                "reduce_max_last_dim step-N vs CPU oracle");

        // Sanity: max(row) >= all elements in that row
        for (int r = 0; r < rows; r++) {
            for (int d = 0; d < D; d++) {
                float elem = data[r * D + d];
                assertTrue(lastResult[r] >= elem - 1e-5f,
                        "reduce_max row " + r + " result=" + lastResult[r]
                        + " < element[" + d + "]=" + elem);
            }
        }
        log.info("EAGER.W3.reduce_max_last_dim: PASS, shape=[{},{}]→[{}], result[0]={}",
                 rows, D, rows, lastResult[0]);
    }

    // ── EAGER.W3.reduce_min ──────────────────────────────────────────────────────

    /**
     * EAGER.W3.reduce_min_last_dim: Verify reduce_min over last axis matches CPU oracle.
     *
     * Input:  [4, 8] f32
     * Output: [4]    f32

     * Sanity: each output value must be ≤ all elements in its input row.
     */
    @Test
    @DisplayName("EAGER.W3.reduce_min_last_dim: reduce_min over axis-1 eager execution matches CPU oracle")
    void testReduceMinLastDimEagerEquivalence() {
        requireMlir();

        int rows = 4, D = 8;
        float[] data = new float[rows * D];
        for (int i = 0; i < data.length; i++) data[i] = (i % 13) * 0.7f - 4.0f;

        INDArray xOracle = Nd4j.create(data, new long[]{rows, D});
        float[] expectedData = xOracle.min(1).toFloatVector();

        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray x = Nd4j.create(data, new long[]{rows, D});
            return x.min(1);
        });

        assertNotNull(lastResult, "reduce_min returned null");
        assertEquals(rows, lastResult.length, "reduce_min output length mismatch");
        assertWithinTolerance(expectedData, lastResult, 1e-5f,
                "reduce_min_last_dim step-N vs CPU oracle");

        // Sanity: min(row) <= all elements in that row
        for (int r = 0; r < rows; r++) {
            for (int d = 0; d < D; d++) {
                float elem = data[r * D + d];
                assertTrue(lastResult[r] <= elem + 1e-5f,
                        "reduce_min row " + r + " result=" + lastResult[r]
                        + " > element[" + d + "]=" + elem);
            }
        }
        log.info("EAGER.W3.reduce_min_last_dim: PASS, shape=[{},{}]→[{}], result[0]={}",
                 rows, D, rows, lastResult[0]);
    }

    // ── EAGER.W3.reduce_prod ─────────────────────────────────────────────────────

    /**
     * EAGER.W3.reduce_prod_last_dim: Verify reduce_prod over last axis matches CPU oracle.
     *
     * Input:  [3, 6] f32 (small values to avoid overflow)
     * Output: [3]    f32

     */
    @Test
    @DisplayName("EAGER.W3.reduce_prod_last_dim: reduce_prod over axis-1 eager execution matches CPU oracle")
    void testReduceProdLastDimEagerEquivalence() {
        requireMlir();

        int rows = 3, D = 6;
        float[] data = new float[rows * D];
        // Use small values in (0, 2] to keep products finite
        for (int i = 0; i < data.length; i++) data[i] = 0.5f + (i % 5) * 0.3f;

        // CPU oracle: product over last dim
        float[] expectedData = new float[rows];
        for (int r = 0; r < rows; r++) {
            float prod = 1.0f;
            for (int d = 0; d < D; d++) prod *= data[r * D + d];
            expectedData[r] = prod;
        }

        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray x = Nd4j.create(data, new long[]{rows, D});
            return x.prod(1);
        });

        assertNotNull(lastResult, "reduce_prod returned null");
        assertEquals(rows, lastResult.length, "reduce_prod output length mismatch");
        // Product tolerances are wider (relative error ~1e-4 for f32 chain multiply)
        for (int r = 0; r < rows; r++) {
            float expected = expectedData[r];
            float actual   = lastResult[r];
            float relErr   = Math.abs(actual - expected) / (Math.abs(expected) + 1e-6f);
            assertTrue(relErr < 1e-3f,
                    "reduce_prod row " + r + ": expected=" + expected
                    + " actual=" + actual + " relErr=" + relErr);
        }
        log.info("EAGER.W3.reduce_prod_last_dim: PASS, shape=[{},{}]→[{}], result[0]={}",
                 rows, D, rows, lastResult[0]);
    }

    /**
     * EAGER.W3.reduce_max_full: Verify full-tensor reduce_max matches CPU oracle.
     *
     * Input:  [5, 8] f32
     * Output: scalar

     */
    @Test
    @DisplayName("EAGER.W3.reduce_max_full: full-tensor reduce_max eager execution matches CPU oracle")
    void testReduceMaxFullEagerEquivalence() {
        requireMlir();

        int rows = 5, D = 8;
        float[] data = new float[rows * D];
        for (int i = 0; i < data.length; i++) data[i] = (i % 17) * 0.6f - 5.0f;

        INDArray xOracle = Nd4j.create(data, new long[]{rows, D});
        float expectedScalar = xOracle.maxNumber().floatValue();

        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray x = Nd4j.create(data, new long[]{rows, D});
            return x.max();
        });

        assertNotNull(lastResult, "reduce_max full returned null");
        assertEquals(1, lastResult.length, "reduce_max full output should be scalar");
        assertTrue(Math.abs(lastResult[0] - expectedScalar) < 1e-5f,
                "reduce_max_full: expected=" + expectedScalar + " actual=" + lastResult[0]);
        log.info("EAGER.W3.reduce_max_full: PASS, shape=[{},{}]→scalar={}", rows, D, lastResult[0]);
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Wave 4 Tests
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * EAGER.W4.attention_decomposition: Verify unfused SDPA (scaled_dot_product_attention) output
     * matches a manual CPU oracle: O = softmax(Q * K^T / sqrt(Dk)) * V.
     *
     * Shapes:
     *   Q: [B=1, nH=2, Sq=4, Dk=8]
     *   K: [B=1, nH=2, Sk=6, Dk=8]
     *   V: [B=1, nH=2, Sk=6, Dv=8]
     *   O: [B=1, nH=2, Sq=4, Dv=8]
     *
     * This is the eager mathematical decomposition, not the fused SDPA op.
     * f32 only, no attention mask.
     */
    @Test
    @DisplayName("EAGER.W4.attention_decomposition: eager attention decomposition matches manual CPU oracle")
    void testAttentionDecompositionEagerEquivalence() {
        requireMlir();

        int B = 1, nH = 2, Sq = 4, Sk = 6, Dk = 8, Dv = 8;
        long[] qShape = {B, nH, Sq, Dk};
        long[] kShape = {B, nH, Sk, Dk};
        long[] vShape = {B, nH, Sk, Dv};

        // Generate deterministic f32 inputs
        float[] qData = new float[B * nH * Sq * Dk];
        float[] kData = new float[B * nH * Sk * Dk];
        float[] vData = new float[B * nH * Sk * Dv];
        for (int i = 0; i < qData.length; i++) qData[i] = ((i % 11) - 5) * 0.1f;
        for (int i = 0; i < kData.length; i++) kData[i] = ((i % 13) - 6) * 0.08f;
        for (int i = 0; i < vData.length; i++) vData[i] = ((i % 7) - 3) * 0.12f;

        float scale = 1.0f / (float) Math.sqrt(Dk);

        // ── CPU oracle ──────────────────────────────────────────────────────
        // For each (b, h): O[q,d] = Σ_k softmax(Q[q,:] · K[k,:] * scale)[k] * V[k,d]
        float[] expected = new float[B * nH * Sq * Dv];
        for (int b = 0; b < B; b++) {
            for (int h = 0; h < nH; h++) {
                for (int q = 0; q < Sq; q++) {
                    // Compute scores[k] = scale * Σ_d Q[b,h,q,d]*K[b,h,k,d]
                    float[] scores = new float[Sk];
                    for (int k = 0; k < Sk; k++) {
                        float dot = 0;
                        for (int d = 0; d < Dk; d++) {
                            int qi = ((b * nH + h) * Sq + q) * Dk + d;
                            int ki = ((b * nH + h) * Sk + k) * Dk + d;
                            dot += qData[qi] * kData[ki];
                        }
                        scores[k] = dot * scale;
                    }
                    // Softmax over Sk
                    float maxS = Float.NEGATIVE_INFINITY;
                    for (float s : scores) maxS = Math.max(maxS, s);
                    float sumExp = 0;
                    float[] attn = new float[Sk];
                    for (int k = 0; k < Sk; k++) {
                        attn[k] = (float) Math.exp(scores[k] - maxS);
                        sumExp += attn[k];
                    }
                    for (int k = 0; k < Sk; k++) attn[k] /= sumExp;
                    // O[b,h,q,dv] = Σ_k attn[k] * V[b,h,k,dv]
                    for (int dv = 0; dv < Dv; dv++) {
                        float o = 0;
                        for (int k = 0; k < Sk; k++) {
                            int vi = ((b * nH + h) * Sk + k) * Dv + dv;
                            o += attn[k] * vData[vi];
                        }
                        expected[((b * nH + h) * Sq + q) * Dv + dv] = o;
                    }
                }
            }
        }

        // ── Eager attention decomposition ─────────────────────────────────────
        // This explicitly composes softmax(Q * K^T / sqrt(Dk)) * V with INDArray
        // operations. It is numerical integration coverage, not a fused SDPA op
        // and not Vulkan recorder or replay evidence.
        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray Q = Nd4j.create(qData, qShape);
            INDArray K = Nd4j.create(kData, kShape);
            INDArray V = Nd4j.create(vData, vShape);
            // Compose the attention formula manually with eager ND4J operations.
            // Flatten batch+head dims to treat as 2D for each (b,h) pair
            INDArray Q2 = Q.reshape(B * nH, Sq, Dk);  // [B*nH, Sq, Dk]
            INDArray K2 = K.reshape(B * nH, Sk, Dk);  // [B*nH, Sk, Dk]
            INDArray V2 = V.reshape(B * nH, Sk, Dv);  // [B*nH, Sk, Dv]
            // S = Q2 @ K2^T * scale → [B*nH, Sq, Sk]
            INDArray Kt = K2.permute(0, 2, 1);         // [B*nH, Dk, Sk]
            INDArray S  = Nd4j.matmul(Q2, Kt).muli(scale); // [B*nH, Sq, Sk]
            // Softmax over last dim (Sk) — flatten batch+head+query dims for 2-D softmax
            INDArray Attn = org.nd4j.linalg.ops.transforms.Transforms
                                .softmax(S.reshape(B * nH * Sq, Sk), false)
                                .reshape(B * nH, Sq, Sk);
            // O = Attn @ V → [B*nH, Sq, Dv]
            INDArray O2 = Nd4j.matmul(Attn, V2);       // [B*nH, Sq, Dv]
            return O2.reshape(B, nH, Sq, Dv);
        });

        assertNotNull(lastResult, "SDPA returned null");
        assertEquals(expected.length, lastResult.length, "SDPA output length mismatch");
        assertWithinTolerance(expected, lastResult, 1e-4f, "EAGER.W4.attention_decomposition");
        log.info("EAGER.W4.attention_decomposition: PASS B={} nH={} Sq={} Sk={} Dk={} Dv={} result[0]={}",
                 B, nH, Sq, Sk, Dk, Dv, lastResult[0]);
    }

    /**
     * EAGER.W4.gather_f16: Verify gather with f16 table matches f32 oracle.
     *
     * Gate: skipped when isFp16Supported() returns false.
     * Shape: table [V=16, D=8] f16; indices [I=4] int32; output [I, D] f16.
     */
    @Test
    @DisplayName("EAGER.W4.gather_f16: f16 gather eager execution matches f32 oracle (capability-gated)")
    void testGatherF16EagerEquivalence() {
        requireMlir();
        org.junit.jupiter.api.Assumptions.assumeTrue(isFp16Supported(),
                "Skipping EAGER.W4.gather_f16: device does not support f16");

        int V = 16, D = 8, I = 4;
        float[] tableF32 = new float[V * D];
        for (int i = 0; i < tableF32.length; i++) tableF32[i] = (i % 7) * 0.5f - 1.5f;
        int[] idxData = {0, 3, 7, 12};

        // CPU oracle: gather rows
        float[] expected = new float[I * D];
        for (int i = 0; i < I; i++) {
            int row = idxData[i];
            for (int d = 0; d < D; d++) {
                expected[i * D + d] = tableF32[row * D + d];
            }
        }

        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray tableF16 = Nd4j.create(tableF32, new long[]{V, D}).castTo(org.nd4j.linalg.api.buffer.DataType.HALF);
            INDArray indices   = Nd4j.createFromArray(idxData);
            INDArray out = Nd4j.exec(new Gather(tableF16, indices, 0))[0];
            return out.castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT);
        });

        assertNotNull(lastResult, "gather f16 returned null");
        assertEquals(expected.length, lastResult.length, "gather f16 output length mismatch");
        // f16 tolerance is wider (~1e-3 for fp16 rounding)
        assertWithinTolerance(expected, lastResult, 2e-3f, "EAGER.W4.gather_f16");
        log.info("EAGER.W4.gather_f16: PASS V={} D={} I={} result[0]={}", V, D, I, lastResult[0]);
    }

    /**
     * EAGER.W4.concat_f16: Verify concat with f16 tensors matches f32 oracle.
     *
     * Gate: skipped when isFp16Supported() returns false.
     * Shape: two [4, 8] f16 → concat along axis-1 → [4, 16] f16.
     */
    @Test
    @DisplayName("EAGER.W4.concat_f16: f16 concat eager execution matches f32 oracle (capability-gated)")
    void testConcatF16EagerEquivalence() {
        requireMlir();
        org.junit.jupiter.api.Assumptions.assumeTrue(isFp16Supported(),
                "Skipping EAGER.W4.concat_f16: device does not support f16");

        int rows = 4, D = 8;
        float[] aData = new float[rows * D];
        float[] bData = new float[rows * D];
        for (int i = 0; i < aData.length; i++) aData[i] = (i % 5) * 0.3f;
        for (int i = 0; i < bData.length; i++) bData[i] = (i % 7) * 0.2f - 0.5f;

        // CPU oracle: concat along axis-1
        float[] expected = new float[rows * (D + D)];
        for (int r = 0; r < rows; r++) {
            for (int d = 0; d < D; d++) expected[r * (2 * D) + d]     = aData[r * D + d];
            for (int d = 0; d < D; d++) expected[r * (2 * D) + D + d] = bData[r * D + d];
        }

        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray af16 = Nd4j.create(aData, new long[]{rows, D}).castTo(org.nd4j.linalg.api.buffer.DataType.HALF);
            INDArray bf16 = Nd4j.create(bData, new long[]{rows, D}).castTo(org.nd4j.linalg.api.buffer.DataType.HALF);
            INDArray out = Nd4j.concat(1, af16, bf16);
            return out.castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT);
        });

        assertNotNull(lastResult, "concat f16 returned null");
        assertEquals(expected.length, lastResult.length, "concat f16 output length mismatch");
        assertWithinTolerance(expected, lastResult, 2e-3f, "EAGER.W4.concat_f16");
        log.info("EAGER.W4.concat_f16: PASS shape=[{},{}]+[{},{}]→[{},{}] result[0]={}",
                 rows, D, rows, D, rows, 2*D, lastResult[0]);
    }

    /**
     * EAGER.W4.transpose_f16: Verify transpose with f16 matches f32 oracle.
     *
     * Gate: skipped when isFp16Supported() returns false.
     * Shape: [3, 5] f16 → transpose → [5, 3] f16.
     */
    @Test
    @DisplayName("EAGER.W4.transpose_f16: f16 transpose eager execution matches f32 oracle (capability-gated)")
    void testTransposeF16EagerEquivalence() {
        requireMlir();
        org.junit.jupiter.api.Assumptions.assumeTrue(isFp16Supported(),
                "Skipping EAGER.W4.transpose_f16: device does not support f16");

        int R = 3, C = 5;
        float[] data = new float[R * C];
        for (int i = 0; i < data.length; i++) data[i] = (i % 9) * 0.4f - 1.8f;

        // CPU oracle: transpose [R, C] → [C, R]
        float[] expected = new float[C * R];
        for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {
                expected[c * R + r] = data[r * C + c];
            }
        }

        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray xf16 = Nd4j.create(data, new long[]{R, C}).castTo(org.nd4j.linalg.api.buffer.DataType.HALF);
            INDArray out  = xf16.permute(1, 0);
            return out.castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT);
        });

        assertNotNull(lastResult, "transpose f16 returned null");
        assertEquals(expected.length, lastResult.length, "transpose f16 output length mismatch");
        assertWithinTolerance(expected, lastResult, 2e-3f, "EAGER.W4.transpose_f16");
        log.info("EAGER.W4.transpose_f16: PASS shape=[{},{}]→[{},{}] result[0]={}",
                 R, C, C, R, lastResult[0]);
    }

    /**
     * EAGER.W4.reduce_sum_nd: Verify rank-3 reduce_sum matches CPU oracle.
     *
     * Input:  [2, 3, 4] f32
     * Axis:   [2] (reduce last dim → [2, 3])

     */
    @Test
    @DisplayName("EAGER.W4.reduce_sum_nd: rank-3 reduce_sum eager execution matches CPU oracle")
    void testReduceSumNDEagerEquivalence() {
        requireMlir();

        int d0 = 2, d1 = 3, d2 = 4;
        float[] data = new float[d0 * d1 * d2];
        for (int i = 0; i < data.length; i++) data[i] = (i % 11) * 0.5f - 2.5f;

        // CPU oracle: sum over axis 2 → [d0, d1]
        float[] expected = new float[d0 * d1];
        for (int i = 0; i < d0; i++) {
            for (int j = 0; j < d1; j++) {
                float s = 0;
                for (int k = 0; k < d2; k++) {
                    s += data[(i * d1 + j) * d2 + k];
                }
                expected[i * d1 + j] = s;
            }
        }

        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray x = Nd4j.create(data, new long[]{d0, d1, d2});
            return x.sum(2);
        });

        assertNotNull(lastResult, "reduce_sum_nd returned null");
        assertEquals(expected.length, lastResult.length, "reduce_sum_nd output length mismatch");
        assertWithinTolerance(expected, lastResult, 1e-4f, "EAGER.W4.reduce_sum_nd");
        log.info("EAGER.W4.reduce_sum_nd: PASS shape=[{},{},{}]→[{},{}] result[0]={}",
                 d0, d1, d2, d0, d1, lastResult[0]);
    }

    /**
     * EAGER.W4.reduce_mean_nd: Verify rank-4 reduce_mean matches CPU oracle.
     *
     * Input:  [2, 3, 4, 5] f32
     * Axes:   [2, 3] (reduce last two dims → [2, 3])

     */
    @Test
    @DisplayName("EAGER.W4.reduce_mean_nd: rank-4 reduce_mean eager execution matches CPU oracle")
    void testReduceMeanNDEagerEquivalence() {
        requireMlir();

        int d0 = 2, d1 = 3, d2 = 4, d3 = 5;
        float[] data = new float[d0 * d1 * d2 * d3];
        for (int i = 0; i < data.length; i++) data[i] = (i % 13) * 0.3f - 1.9f;

        // CPU oracle: mean over axes 2 and 3 → [d0, d1]
        float[] expected = new float[d0 * d1];
        int reduceN = d2 * d3;
        for (int i = 0; i < d0; i++) {
            for (int j = 0; j < d1; j++) {
                float s = 0;
                for (int k = 0; k < d2; k++) {
                    for (int l = 0; l < d3; l++) {
                        s += data[((i * d1 + j) * d2 + k) * d3 + l];
                    }
                }
                expected[i * d1 + j] = s / reduceN;
            }
        }

        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray x = Nd4j.create(data, new long[]{d0, d1, d2, d3});
            return x.mean(2, 3);
        });

        assertNotNull(lastResult, "reduce_mean_nd returned null");
        assertEquals(expected.length, lastResult.length, "reduce_mean_nd output length mismatch");
        assertWithinTolerance(expected, lastResult, 1e-4f, "EAGER.W4.reduce_mean_nd");
        log.info("EAGER.W4.reduce_mean_nd: PASS shape=[{},{},{},{}]→[{},{}] result[0]={}",
                 d0, d1, d2, d3, d0, d1, lastResult[0]);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Wave 5 T3 tests
    // ══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("EAGER.W5.squared_relu: squared_relu eager execution matches CPU oracle")
    void testSquaredReluEagerEquivalence() {
        requireMlir();
        int n = 32;
        float[] data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (i - n / 2) * 0.3f;
        float[] expected = new float[n];
        for (int i = 0; i < n; i++) {
            float r = Math.max(data[i], 0.0f);
            expected[i] = r * r;
        }
        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray x = Nd4j.create(data, new long[]{n});
            INDArray r = Transforms.relu(x, true);
            return r.mul(r);
        });
        assertNotNull(lastResult);
        assertWithinTolerance(expected, lastResult, 1e-5f, "squared_relu vs CPU oracle");
        log.info("EAGER.W5.squared_relu: PASS n={} result[0]={}", n, lastResult[0]);
    }

    @Test
    @DisplayName("EAGER.W5.abs: abs eager execution matches CPU oracle")
    void testAbsEagerEquivalence() {
        requireMlir();
        int n = 32;
        float[] data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (i - n / 2) * 0.4f;
        float[] expected = new float[n];
        for (int i = 0; i < n; i++) expected[i] = Math.abs(data[i]);
        float[] lastResult = runNEagerSteps(3, () ->
            Transforms.abs(Nd4j.create(data, new long[]{n}), true));
        assertNotNull(lastResult);
        assertWithinTolerance(expected, lastResult, 1e-6f, "abs vs CPU oracle");
        log.info("EAGER.W5.abs: PASS n={} result[0]={}", n, lastResult[0]);
    }

    @Test
    @DisplayName("EAGER.W5.neg: neg eager execution matches CPU oracle")
    void testNegEagerEquivalence() {
        requireMlir();
        int n = 32;
        float[] data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (i % 7) * 0.3f - 1.0f;
        float[] expected = new float[n];
        for (int i = 0; i < n; i++) expected[i] = -data[i];
        float[] lastResult = runNEagerSteps(3, () ->
            Nd4j.create(data, new long[]{n}).neg());
        assertNotNull(lastResult);
        assertWithinTolerance(expected, lastResult, 1e-6f, "neg vs CPU oracle");
        log.info("EAGER.W5.neg: PASS n={} result[0]={}", n, lastResult[0]);
    }

    @Test
    @DisplayName("EAGER.W5.sqrt: sqrt eager execution matches CPU oracle")
    void testSqrtEagerEquivalence() {
        requireMlir();
        int n = 16;
        float[] data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (i + 1) * 0.5f;  // positive values
        float[] expected = new float[n];
        for (int i = 0; i < n; i++) expected[i] = (float) Math.sqrt(data[i]);
        float[] lastResult = runNEagerSteps(3, () ->
            Transforms.sqrt(Nd4j.create(data, new long[]{n}), true));
        assertNotNull(lastResult);
        assertWithinTolerance(expected, lastResult, 1e-3f, "sqrt vs CPU oracle");
        log.info("EAGER.W5.sqrt: PASS n={} result[0]={}", n, lastResult[0]);
    }

    @Test
    @DisplayName("EAGER.W5.square: x^2 eager execution matches CPU oracle")
    void testSquareEagerEquivalence() {
        requireMlir();
        int n = 32;
        float[] data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (i % 9) * 0.2f - 0.8f;
        float[] expected = new float[n];
        for (int i = 0; i < n; i++) expected[i] = data[i] * data[i];
        float[] lastResult = runNEagerSteps(3, () ->
            Nd4j.create(data, new long[]{n}).mul(Nd4j.create(data, new long[]{n})));
        assertNotNull(lastResult);
        assertWithinTolerance(expected, lastResult, 1e-5f, "square vs CPU oracle");
        log.info("EAGER.W5.square: PASS n={} result[0]={}", n, lastResult[0]);
    }

    @Test
    @DisplayName("EAGER.W5.minimum: elementwise min eager execution matches CPU oracle")
    void testMinimumEagerEquivalence() {
        requireMlir();
        int n = 32;
        float[] aData = new float[n], bData = new float[n];
        for (int i = 0; i < n; i++) { aData[i] = (i % 7) * 0.3f; bData[i] = (i % 5) * 0.4f; }
        float[] expected = new float[n];
        for (int i = 0; i < n; i++) expected[i] = Math.min(aData[i], bData[i]);
        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray a = Nd4j.create(aData, new long[]{n});
            INDArray b = Nd4j.create(bData, new long[]{n});
            return Transforms.min(a, b, true);
        });
        assertNotNull(lastResult);
        assertWithinTolerance(expected, lastResult, 1e-6f, "minimum vs CPU oracle");
        log.info("EAGER.W5.minimum: PASS n={} result[0]={}", n, lastResult[0]);
    }

    @Test
    @DisplayName("EAGER.W5.maximum: elementwise max eager execution matches CPU oracle")
    void testMaximumEagerEquivalence() {
        requireMlir();
        int n = 32;
        float[] aData = new float[n], bData = new float[n];
        for (int i = 0; i < n; i++) { aData[i] = (i % 11) * 0.2f - 1.0f; bData[i] = (i % 7) * 0.3f - 0.5f; }
        float[] expected = new float[n];
        for (int i = 0; i < n; i++) expected[i] = Math.max(aData[i], bData[i]);
        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray a = Nd4j.create(aData, new long[]{n});
            INDArray b = Nd4j.create(bData, new long[]{n});
            return Transforms.max(a, b, true);
        });
        assertNotNull(lastResult);
        assertWithinTolerance(expected, lastResult, 1e-6f, "maximum vs CPU oracle");
        log.info("EAGER.W5.maximum: PASS n={} result[0]={}", n, lastResult[0]);
    }

    @Test
    @DisplayName("EAGER.W5.elu: elu activation eager execution matches CPU oracle")
    void testEluEagerEquivalence() {
        requireMlir();
        int n = 32;
        float[] data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (i - n / 2) * 0.25f;
        float[] expected = new float[n];
        for (int i = 0; i < n; i++) {
            float x = data[i];
            expected[i] = (x >= 0) ? x : (float)(Math.exp(x) - 1.0);
        }
        float[] lastResult = runNEagerSteps(3, () ->
            Transforms.elu(Nd4j.create(data, new long[]{n}), true));
        assertNotNull(lastResult);
        assertWithinTolerance(expected, lastResult, 1e-4f, "elu vs CPU oracle");
        log.info("EAGER.W5.elu: PASS n={} result[0]={}", n, lastResult[0]);
    }

    @Test
    @DisplayName("EAGER.W5.leaky_relu: leaky relu eager execution matches CPU oracle")
    void testLeakyReluEagerEquivalence() {
        requireMlir();
        int n = 32;
        float[] data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (i - n / 2) * 0.3f;
        float[] expected = new float[n];
        for (int i = 0; i < n; i++) {
            float x = data[i];
            expected[i] = (x >= 0) ? x : 0.01f * x;
        }
        float[] lastResult = runNEagerSteps(3, () ->
            Transforms.leakyRelu(Nd4j.create(data, new long[]{n}), 0.01, true));
        assertNotNull(lastResult);
        assertWithinTolerance(expected, lastResult, 1e-5f, "leaky_relu vs CPU oracle");
        log.info("EAGER.W5.leaky_relu: PASS n={} result[0]={}", n, lastResult[0]);
    }

    @Test
    @DisplayName("EAGER.W5.assign: assign (identity copy) eager execution matches CPU oracle")
    void testAssignEagerEquivalence() {
        requireMlir();
        int n = 24;
        float[] data = new float[n];
        for (int i = 0; i < n; i++) data[i] = i * 0.1f - 1.2f;
        float[] expected = Arrays.copyOf(data, n);
        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray x = Nd4j.create(data, new long[]{n});
            INDArray y = Nd4j.zerosLike(x);
            return y.assign(x);
        });
        assertNotNull(lastResult);
        assertWithinTolerance(expected, lastResult, 1e-6f, "assign vs CPU oracle");
        log.info("EAGER.W5.assign: PASS n={} result[0]={}", n, lastResult[0]);
    }

    @Test
    @DisplayName("EAGER.W5.batched_gemm: rank-3 batched matmul eager execution matches CPU oracle")
    void testBatchedGemmEagerEquivalence() {
        requireMlir();
        int B = 2, M = 4, K = 6, N = 3;
        float[] aData = new float[B * M * K];
        float[] bData = new float[B * K * N];
        for (int i = 0; i < aData.length; i++) aData[i] = (i % 7) * 0.1f - 0.3f;
        for (int i = 0; i < bData.length; i++) bData[i] = (i % 5) * 0.15f + 0.05f;
        // CPU oracle: manual batched matmul
        float[] expected = new float[B * M * N];
        for (int b = 0; b < B; b++) {
            for (int m = 0; m < M; m++) {
                for (int nIdx = 0; nIdx < N; nIdx++) {
                    float s = 0;
                    for (int k = 0; k < K; k++) {
                        s += aData[(b * M + m) * K + k] * bData[(b * K + k) * N + nIdx];
                    }
                    expected[(b * M + m) * N + nIdx] = s;
                }
            }
        }
        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray a = Nd4j.create(aData, new long[]{B, M, K});
            INDArray bArr = Nd4j.create(bData, new long[]{B, K, N});
            return Nd4j.matmul(a, bArr);
        });
        assertNotNull(lastResult, "batched_gemm returned null");
        assertEquals(expected.length, lastResult.length, "batched_gemm output length mismatch");
        assertWithinTolerance(expected, lastResult, 1e-4f, "batched_gemm vs CPU oracle");
        log.info("EAGER.W5.batched_gemm: PASS B={} M={} K={} N={} result[0]={}",
                 B, M, K, N, lastResult[0]);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Wave 6 T3 tests — f16 extensions and new EW/index-map ops
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * EAGER.W6.softmax_f16: Verify row-wise softmax with f16 input matches f32 oracle.
     *
     * Gate: skipped when isFp16Supported() returns false.
     * Shape: [4, 8] f16. This is eager numerical evidence only.
     */
    @Test
    @DisplayName("EAGER.W6.softmax_f16: f16 softmax eager execution matches f32 oracle (capability-gated)")
    void testSoftmaxF16EagerEquivalence() {
        requireMlir();
        assumeTrue(isFp16Supported(), "Skipping EAGER.W6.softmax_f16: device does not support f16");

        int rows = 4, dim = 8;
        float[] data = new float[rows * dim];
        for (int i = 0; i < data.length; i++) data[i] = (i % 11) * 0.3f - 1.5f;

        // f32 CPU oracle
        INDArray xOracle = Nd4j.create(data, new long[]{rows, dim});
        float[] expected = org.nd4j.linalg.ops.transforms.Transforms.softmax(xOracle, true).toFloatVector();

        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray xF16 = Nd4j.create(data, new long[]{rows, dim}).castTo(DataType.HALF);
            INDArray outF16 = org.nd4j.linalg.ops.transforms.Transforms.softmax(xF16, true);
            return outF16.castTo(DataType.FLOAT);
        });

        assertNotNull(lastResult);
        // Sanity: each row sums to 1
        for (int r = 0; r < rows; r++) {
            float rowSum = 0;
            for (int c = 0; c < dim; c++) rowSum += lastResult[r * dim + c];
            assertTrue(Math.abs(rowSum - 1.0f) < 0.02f,
                    "softmax_f16 row " + r + " sum=" + rowSum);
        }
        assertWithinTolerance(expected, lastResult, 0.05f, "softmax_f16 vs f32 oracle");
        log.info("EAGER.W6.softmax_f16: PASS shape=[{},{}] result[0]={}", rows, dim, lastResult[0]);
    }

    /**
     * EAGER.W6.reduce_sum_f16: Verify reduce_sum with f16 input matches f32 oracle.
     *
     * Gate: skipped when isFp16Supported() returns false.
     * Shape: [4, 8] f16 → sum over axis 1 → [4]. This is eager numerical evidence only.
     */
    @Test
    @DisplayName("EAGER.W6.reduce_sum_f16: f16 reduce_sum eager execution matches f32 oracle (capability-gated)")
    void testReduceSumF16EagerEquivalence() {
        requireMlir();
        assumeTrue(isFp16Supported(), "Skipping EAGER.W6.reduce_sum_f16: device does not support f16");

        int rows = 4, D = 8;
        float[] data = new float[rows * D];
        for (int i = 0; i < data.length; i++) data[i] = (i % 7) * 0.5f - 1.5f;

        // f32 CPU oracle
        INDArray xOracle = Nd4j.create(data, new long[]{rows, D});
        float[] expected = xOracle.sum(1).toFloatVector();

        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray xF16 = Nd4j.create(data, new long[]{rows, D}).castTo(DataType.HALF);
            INDArray out = xF16.sum(1);
            return out.castTo(DataType.FLOAT);
        });

        assertNotNull(lastResult);
        assertEquals(rows, lastResult.length);
        // f16 tolerance ~2%
        assertWithinTolerance(expected, lastResult, 0.05f, "reduce_sum_f16 vs f32 oracle");
        log.info("EAGER.W6.reduce_sum_f16: PASS shape=[{},{}]→[{}] result[0]={}", rows, D, rows, lastResult[0]);
    }

    /**
     * EAGER.W6.reduce_max_f16: Verify reduce_max with f16 input matches f32 oracle.
     *
     * Gate: skipped when isFp16Supported() returns false.
     * Shape: [4, 8] f16 → max over axis 1 → [4].
     */
    @Test
    @DisplayName("EAGER.W6.reduce_max_f16: f16 reduce_max eager execution matches f32 oracle (capability-gated)")
    void testReduceMaxF16EagerEquivalence() {
        requireMlir();
        assumeTrue(isFp16Supported(), "Skipping EAGER.W6.reduce_max_f16: device does not support f16");

        int rows = 4, D = 8;
        float[] data = new float[rows * D];
        for (int i = 0; i < data.length; i++) data[i] = (i % 13) * 0.4f - 2.5f;

        INDArray xOracle = Nd4j.create(data, new long[]{rows, D});
        float[] expected = xOracle.max(1).toFloatVector();

        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray xF16 = Nd4j.create(data, new long[]{rows, D}).castTo(DataType.HALF);
            INDArray out = xF16.max(1);
            return out.castTo(DataType.FLOAT);
        });

        assertNotNull(lastResult);
        assertEquals(rows, lastResult.length);
        assertWithinTolerance(expected, lastResult, 0.05f, "reduce_max_f16 vs f32 oracle");
        log.info("EAGER.W6.reduce_max_f16: PASS shape=[{},{}]→[{}] result[0]={}", rows, D, rows, lastResult[0]);
    }

    /**
     * EAGER.W6.attention_decomposition_f16: Verify SDPA with f16 Q/K/V matches f32 oracle.
     *
     * Gate: skipped when isFp16Supported() returns false.
     * Shape: Q=[1,2,4,8] K=[1,2,6,8] V=[1,2,6,8] f16. This tests the eager decomposition only.
     */
    @Test
    @DisplayName("EAGER.W6.attention_decomposition_f16: f16 eager attention decomposition matches f32 oracle (capability-gated)")
    void testAttentionDecompositionF16EagerEquivalence() {
        requireMlir();
        assumeTrue(isFp16Supported(), "Skipping EAGER.W6.attention_decomposition_f16: device does not support f16");

        int B = 1, nH = 2, Sq = 4, Sk = 4, Dk = 8, Dv = 8;
        float[] qData = new float[B * nH * Sq * Dk];
        float[] kData = new float[B * nH * Sk * Dk];
        float[] vData = new float[B * nH * Sk * Dv];
        for (int i = 0; i < qData.length; i++) qData[i] = ((i % 7) - 3) * 0.1f;
        for (int i = 0; i < kData.length; i++) kData[i] = ((i % 5) - 2) * 0.1f;
        for (int i = 0; i < vData.length; i++) vData[i] = ((i % 9) - 4) * 0.08f;
        float scale = 1.0f / (float) Math.sqrt(Dk);

        // f32 CPU oracle (manual SDPA)
        float[] expected = new float[B * nH * Sq * Dv];
        for (int b = 0; b < B; b++) {
            for (int h = 0; h < nH; h++) {
                for (int q = 0; q < Sq; q++) {
                    float[] scores = new float[Sk];
                    for (int k = 0; k < Sk; k++) {
                        float dot = 0;
                        for (int d = 0; d < Dk; d++) {
                            dot += qData[((b * nH + h) * Sq + q) * Dk + d]
                                 * kData[((b * nH + h) * Sk + k) * Dk + d];
                        }
                        scores[k] = dot * scale;
                    }
                    float maxS = Float.NEGATIVE_INFINITY;
                    for (float s : scores) maxS = Math.max(maxS, s);
                    float sumExp = 0;
                    float[] attn = new float[Sk];
                    for (int k = 0; k < Sk; k++) {
                        attn[k] = (float) Math.exp(scores[k] - maxS);
                        sumExp += attn[k];
                    }
                    for (int k = 0; k < Sk; k++) attn[k] /= sumExp;
                    for (int dv = 0; dv < Dv; dv++) {
                        float o = 0;
                        for (int k = 0; k < Sk; k++) {
                            o += attn[k] * vData[((b * nH + h) * Sk + k) * Dv + dv];
                        }
                        expected[((b * nH + h) * Sq + q) * Dv + dv] = o;
                    }
                }
            }
        }

        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray Q = Nd4j.create(qData, new long[]{B, nH, Sq, Dk}).castTo(DataType.HALF);
            INDArray K = Nd4j.create(kData, new long[]{B, nH, Sk, Dk}).castTo(DataType.HALF);
            INDArray V = Nd4j.create(vData, new long[]{B, nH, Sk, Dv}).castTo(DataType.HALF);
            INDArray Q2 = Q.reshape(B * nH, Sq, Dk);
            INDArray K2 = K.reshape(B * nH, Sk, Dk);
            INDArray V2 = V.reshape(B * nH, Sk, Dv);
            INDArray Kt = K2.permute(0, 2, 1);
            INDArray S  = Nd4j.matmul(Q2, Kt).muli((double)scale);
            INDArray Attn = org.nd4j.linalg.ops.transforms.Transforms
                                .softmax(S.reshape(B * nH * Sq, Sk).castTo(DataType.FLOAT), false)
                                .reshape(B * nH, Sq, Sk).castTo(DataType.HALF);
            INDArray O2 = Nd4j.matmul(Attn, V2);
            return O2.reshape(B, nH, Sq, Dv).castTo(DataType.FLOAT);
        });

        assertNotNull(lastResult);
        assertEquals(expected.length, lastResult.length);
        // f16 tolerance wider: ~5%
        assertWithinTolerance(expected, lastResult, 0.1f, "sdpa_f16 vs f32 oracle");
        log.info("EAGER.W6.attention_decomposition_f16: PASS B={} nH={} Sq={} Sk={} result[0]={}", B, nH, Sq, Sk, lastResult[0]);
    }

    /**
     * EAGER.W6.reduce_mean_f16: Verify reduce_mean with f16 input matches f32 oracle.
     *
     * Gate: skipped when isFp16Supported() returns false.
     * Shape: [4, 8] f16 → mean over axis 1 → [4].
     */
    @Test
    @DisplayName("EAGER.W6.reduce_mean_f16: f16 reduce_mean eager execution matches f32 oracle (capability-gated)")
    void testReduceMeanF16EagerEquivalence() {
        requireMlir();
        assumeTrue(isFp16Supported(), "Skipping EAGER.W6.reduce_mean_f16: device does not support f16");

        int rows = 4, D = 8;
        float[] data = new float[rows * D];
        for (int i = 0; i < data.length; i++) data[i] = (i % 9) * 0.4f - 1.6f;

        INDArray xOracle = Nd4j.create(data, new long[]{rows, D});
        float[] expected = xOracle.mean(1).toFloatVector();

        float[] lastResult = runNEagerSteps(3, () -> {
            INDArray xF16 = Nd4j.create(data, new long[]{rows, D}).castTo(DataType.HALF);
            INDArray out = xF16.mean(1);
            return out.castTo(DataType.FLOAT);
        });

        assertNotNull(lastResult);
        assertEquals(rows, lastResult.length);
        assertWithinTolerance(expected, lastResult, 0.05f, "reduce_mean_f16 vs f32 oracle");
        log.info("EAGER.W6.reduce_mean_f16: PASS shape=[{},{}]→[{}] result[0]={}", rows, D, rows, lastResult[0]);
    }

    // Image/convolution coverage belongs in VulkanKernelEmitterStrictReplayTest once
    // those real SameDiff ops have Vulkan emitters and hardware dispatch evidence.
}
