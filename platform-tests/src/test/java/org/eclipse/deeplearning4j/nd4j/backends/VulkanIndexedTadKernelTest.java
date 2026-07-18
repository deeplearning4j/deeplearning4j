/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.nd4j.backends;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Real-pipeline coverage for the descriptor-driven indexed-TAD movement family.
 *
 * <p>The public NDArray factory paths call the Vulkan eager executor, which uses
 * the same descriptor catalogue, recorder, MLIR lowering, command capture, and
 * replay submission as DSP. There is no host implementation or unsupported-op
 * fallback in that path. Each probe therefore requires one captured dispatch
 * and one real replay submission on the selected Vulkan device.</p>
 */
@Slf4j
@Tag(TagNames.VULKAN)
@DisplayName("Vulkan indexed-TAD kernels")
public class VulkanIndexedTadKernelTest {

    private static final String VULKAN_BINDINGS_CLASS =
            "org.nd4j.linalg.vulkan.bindings.Nd4jVulkan";

    private static Object nativeOps;
    private static int selectedDeviceId;
    private static String selectedDeviceName;

    @BeforeAll
    static void setupVulkan() {
        try {
            Class<?> bindingsClass = Class.forName(VULKAN_BINDINGS_CLASS);
            nativeOps = bindingsClass.getDeclaredConstructor().newInstance();

            int deviceCount = ((Number) bindingsClass.getMethod("getAvailableDevices")
                    .invoke(nativeOps)).intValue();
            assertTrue(deviceCount > 0,
                    "Indexed-TAD kernel tests require an enumerated Vulkan device");

            String requestedRegex =
                    System.getProperty("nd4j.vulkan.test.deviceNameRegex", "").trim();
            Pattern requested = requestedRegex.isEmpty()
                    ? null : Pattern.compile(requestedRegex);
            selectedDeviceId = -1;
            for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
                String name = (String) bindingsClass
                        .getMethod("getDeviceName", int.class)
                        .invoke(nativeOps, deviceId);
                log.info("Vulkan indexed-TAD device[{}]='{}'", deviceId, name);
                if (requested == null || requested.matcher(name).find()) {
                    selectedDeviceId = deviceId;
                    selectedDeviceName = name;
                    break;
                }
            }
            assertTrue(selectedDeviceId >= 0,
                    "No Vulkan device matched nd4j.vulkan.test.deviceNameRegex="
                            + requestedRegex);

            assertTrue(probeMlir(bindingsClass),
                    "Vulkan indexed-TAD kernels require HAVE_MLIR=1");
            if (Boolean.getBoolean("nd4j.vulkan.test.requireTriton")) {
                assertTrue((Boolean) bindingsClass.getMethod("isTritonAvailable")
                                .invoke(nativeOps),
                        "The Vulkan artifact must include the shared Triton DSP stack");
            }

            String factoryClass = Nd4j.getNDArrayFactory().getClass()
                    .getName().toLowerCase(Locale.ROOT);
            assertTrue(factoryClass.contains("vulkan"),
                    "The test-vulkan profile must select the Vulkan NDArray factory, got "
                            + factoryClass);

            activateSelectedDevice();
            assertHardwareDeviceWhenRequired();
        } catch (ClassNotFoundException e) {
            fail("Vulkan bindings are absent; run this suite with -Ptest-vulkan", e);
        } catch (ReflectiveOperationException e) {
            fail("Vulkan bindings do not expose the indexed-TAD test contract", e);
        }
    }

    @BeforeEach
    void selectDeviceForTestThread() {
        activateSelectedDevice();
    }

    @Test
    @DisplayName("pullRows rank-1 emits one real integer pipeline")
    void pullRowsRankOneExecutesRealKernel() {
        try (INDArray source = Nd4j.createFromArray(10, 20, 30, 40, 50);
             INDArray result = executeBackendPullRows(
                     "pullRows rank-1 INT",
                     source,
                     0,
                     new int[]{4, 0, 2})) {
            assertArrayEquals(new long[]{3}, result.shape());
            assertArrayEquals(
                    new double[]{50, 10, 30},
                    result.toDoubleVector(),
                    0.0);
        }
    }

    @Test
    @DisplayName("pullRows rank-2 rows and columns emit stride-aware real pipelines")
    void pullRowsRankTwoAxesExecuteRealKernels() {
        try (INDArray rows = Nd4j.create(
                new float[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                new long[]{4, 3});
             INDArray pulledRows = executePullRows(
                     "pullRows rank-2 rows FLOAT",
                     rows,
                     1,
                     new int[]{3, 1})) {
            assertArrayEquals(new long[]{2, 3}, pulledRows.shape());
            assertArrayEquals(
                    new double[]{9, 10, 11, 3, 4, 5},
                    pulledRows.toDoubleVector(),
                    0.0);
        }

        try (INDArray columns = Nd4j.create(
                new double[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                new long[]{3, 4});
             INDArray pulledColumns = executePullRows(
                     "pullRows rank-2 columns DOUBLE",
                     columns,
                     0,
                     new int[]{2, 0})) {
            assertArrayEquals(new long[]{3, 2}, pulledColumns.shape());
            assertArrayEquals(
                    new double[]{2, 0, 6, 4, 10, 8},
                    pulledColumns.toDoubleVector(),
                    0.0);
        }
    }

    @Test
    @DisplayName("rank-1 shuffle emits one in-place real pipeline")
    void shuffleRankOneExecutesRealKernel() {
        final int seed = 12345;
        int[] map = ArrayUtil.buildInterleavedVector(new Random(seed), 8);
        double[] input = {10, 11, 12, 13, 14, 15, 16, 17};
        double[] expected = applyPairMap(input, 1, map);

        try (INDArray vector = Nd4j.createFromArray(
                10, 11, 12, 13, 14, 15, 16, 17)) {
            runOnePipeline("shuffle rank-1 INT",
                    () -> Nd4j.shuffle(vector, new Random(seed), 0));
            assertArrayEquals(expected, vector.toDoubleVector(), 0.0);
        }
    }

    @Test
    @DisplayName("multi-array rank-3 shuffle stays one shared real pipeline")
    void shuffleMultipleRankThreeArraysExecuteOneRealKernel() {
        final int seed = 9876;
        final int itemCount = 6;
        final int tadLength = 6;

        float[] featureValues = new float[itemCount * tadLength];
        double[] labelValues = new double[itemCount * tadLength];
        for (int item = 0; item < itemCount; item++) {
            for (int element = 0; element < tadLength; element++) {
                int offset = item * tadLength + element;
                featureValues[offset] = item * 10.0f + element;
                labelValues[offset] = 1000.0 + item * 10.0 + element;
            }
        }

        int[] map = ArrayUtil.buildInterleavedVector(
                new Random(seed), itemCount);
        double[] expectedFeatures = applyPairMap(
                toDouble(featureValues), tadLength, map);
        double[] expectedLabels = applyPairMap(
                labelValues, tadLength, map);

        try (INDArray features = Nd4j.create(
                featureValues, new long[]{itemCount, 2, 3});
             INDArray labels = Nd4j.create(
                     labelValues, new long[]{itemCount, 3, 2})) {
            List<INDArray> arrays = Arrays.asList(features, labels);
            runOnePipeline("shuffle two rank-3 arrays",
                    () -> Nd4j.shuffle(
                            arrays,
                            new Random(seed),
                            Collections.singletonList(new long[]{1, 2})));

            assertArrayEquals(
                    expectedFeatures, features.toDoubleVector(), 0.0);
            assertArrayEquals(
                    expectedLabels, labels.toDoubleVector(), 0.0);
        }
    }

    private static INDArray executePullRows(
            String label, INDArray source, int dimension, int[] indexes) {
        beginPipelineProbe();
        INDArray result = Nd4j.pullRows(source, dimension, indexes);
        Nd4j.getExecutioner().commit();
        assertOnePipeline(label);
        return result;
    }

    private static INDArray executeBackendPullRows(
            String label, INDArray source, int dimension, int[] indexes) {
        beginPipelineProbe();
        INDArray result = Nd4j.getNDArrayFactory().pullRows(
                source, dimension, indexes);
        Nd4j.getExecutioner().commit();
        assertOnePipeline(label);
        return result;
    }

    private static void runOnePipeline(String label, Runnable operation) {
        beginPipelineProbe();
        operation.run();
        Nd4j.getExecutioner().commit();
        assertOnePipeline(label);
    }

    private static void beginPipelineProbe() {
        invokeNative("dspDiagClear", new Class<?>[0]);
        invokeNative("dspDiagSetCategories", new Class<?>[]{int.class}, -1);
        invokeNative("dspDiagSetLevel", new Class<?>[]{int.class}, 2);
    }

    private static void assertOnePipeline(String label) {
        String report = (String) invokeNative(
                "dspDiagGetJsonReport", new Class<?>[0]);
        assertNotNull(report, label + ": Vulkan diagnostic report is null");
        assertTrue(report.contains("vulkan_backend CAPTURE_DONE"),
                () -> label + ": no Vulkan capture event was emitted:\n" + report);
        assertTrue(report.contains("vulkan_backend REPLAY_DONE"),
                () -> label + ": no Vulkan replay event was emitted:\n" + report);
        assertEquals(1L, jsonLongMetric(report, "num_dispatches"),
                () -> label + ": expected exactly one captured compute dispatch:\n"
                        + report);
        assertTrue(jsonLongMetric(report, "replay_count") > 0,
                () -> label + ": replay_count must be positive:\n" + report);
        assertEquals(selectedDeviceName, jsonStringMetric(report, "device_name"),
                label + ": eager replay ran on the wrong Vulkan device");
        assertHardwareDeviceWhenRequired();
    }

    private static double[] applyPairMap(
            double[] original, int itemLength, int[] map) {
        double[] result = original.clone();
        for (int item = 0; item < map.length; item++) {
            int target = map[item];
            if (target < 0) {
                continue;
            }
            for (int element = 0; element < itemLength; element++) {
                int left = item * itemLength + element;
                int right = target * itemLength + element;
                double value = result[left];
                result[left] = result[right];
                result[right] = value;
            }
        }
        return result;
    }

    private static double[] toDouble(float[] values) {
        double[] result = new double[values.length];
        for (int index = 0; index < values.length; index++) {
            result[index] = values[index];
        }
        return result;
    }

    private static void activateSelectedDevice() {
        Object result = invokeNative(
                "setDevice", new Class<?>[]{int.class}, selectedDeviceId);
        assertEquals(1, ((Number) result).intValue(),
                "setDevice(" + selectedDeviceId + ") must succeed");
        Nd4j.getAffinityManager().setDeviceForCurrentThread(selectedDeviceId);
        assertEquals(selectedDeviceId,
                Nd4j.getAffinityManager().getDeviceForCurrentThread(),
                "ND4J affinity must select the same Vulkan device");
    }

    private static boolean probeMlir(Class<?> bindingsClass)
            throws ReflectiveOperationException {
        try {
            Object value = bindingsClass.getField("HAVE_MLIR").get(null);
            return value instanceof Number
                    && ((Number) value).intValue() == 1;
        } catch (NoSuchFieldException missingGeneratedConstant) {
            try {
                Method method = bindingsClass.getMethod("isMlirEnabled");
                return (Boolean) method.invoke(nativeOps);
            } catch (NoSuchMethodException missingCapabilityMethod) {
                Object value = bindingsClass.getMethod(
                                "getConfigIntValue", String.class)
                        .invoke(nativeOps, "HAVE_MLIR");
                return value instanceof Number
                        && ((Number) value).intValue() == 1;
            }
        }
    }

    private static void assertHardwareDeviceWhenRequired() {
        if (!Boolean.getBoolean("nd4j.vulkan.test.requireHardware")) {
            return;
        }
        String lowerName = selectedDeviceName.toLowerCase(Locale.ROOT);
        assertFalse(lowerName.contains("lavapipe")
                        || lowerName.contains("llvmpipe"),
                "Strict hardware mode selected a software Vulkan device: "
                        + selectedDeviceName);
    }

    private static Object invokeNative(
            String methodName, Class<?>[] parameterTypes, Object... arguments) {
        assertNotNull(nativeOps, "Vulkan NativeOps must be initialized");
        try {
            return nativeOps.getClass().getMethod(methodName, parameterTypes)
                    .invoke(nativeOps, arguments);
        } catch (ReflectiveOperationException e) {
            throw new AssertionError(
                    "Required Vulkan NativeOps method unavailable: "
                            + methodName,
                    e);
        }
    }

    private static long jsonLongMetric(String json, String key) {
        Matcher matcher = Pattern.compile(
                        "\"" + Pattern.quote(key)
                                + "\"\\s*:\\s*([0-9]+)")
                .matcher(json);
        assertTrue(matcher.find(),
                () -> "Missing Vulkan diagnostic metric '" + key
                        + "':\n" + json);
        return Long.parseLong(matcher.group(1));
    }

    private static String jsonStringMetric(String json, String key) {
        Matcher matcher = Pattern.compile(
                        "\"" + Pattern.quote(key)
                                + "\"\\s*:\\s*\"([^\"]+)\"")
                .matcher(json);
        assertTrue(matcher.find(),
                () -> "Missing Vulkan diagnostic metric '" + key
                        + "':\n" + json);
        return matcher.group(1);
    }
}
