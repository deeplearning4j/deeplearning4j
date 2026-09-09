/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.eclipse.deeplearning4j.llm.generation.ModelIOConfig;
import org.eclipse.deeplearning4j.llm.generation.kvcache.UnifiedKvCacheManager;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Caller-visible attention state must survive both migration directions and replay. */
public class DspAttentionCacheWritebackTest extends BaseND4JTest {
    @Override
    public DataType getDataType() { return DataType.FLOAT; }

    @ParameterizedTest
    @CsvSource({"0,1,FLOAT", "1,0,FLOAT", "0,1,HALF", "1,0,HALF",
            "1,-1,FLOAT", "1,-1,HALF"})
    void retainedCachesAreWrittenAcrossDevices(int callerDevice, int attentionDevice, DataType dtype) {
        checkRetainedCaches(callerDevice, attentionDevice, dtype);
    }

    @ParameterizedTest
    @CsvSource({"0,1,FLOAT", "1,0,FLOAT", "0,1,HALF", "1,0,HALF",
            "1,-1,FLOAT", "1,-1,HALF"})
    void managerCachesAreWrittenAcrossDevices(int callerDevice, int attentionDevice, DataType dtype) {
        assumeTrue(Nd4j.backends().isCudaAvailable(), "requires CUDA");
        assumeTrue(Nd4j.getAffinityManager().getNumberOfDevices() >= 2, "requires two CUDA devices");
        int savedDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        Nd4j.getAffinityManager().setDeviceForCurrentThread(callerDevice);
        final int capacity = 17;
        ModelIOConfig.KVCacheNames names = new ModelIOConfig.KVCacheNames(
                java.util.List.of("present.0.key"), java.util.List.of("present.0.value"));
        try (UnifiedKvCacheManager manager = initializedManager(dtype, capacity);
             SameDiff sd = SameDiff.create()) {
            INDArray keys = manager.getStaticKvBuffers().get("past_key_values.0.key");
            INDArray values = manager.getStaticKvBuffers().get("past_key_values.0.value");
            assertEquals(callerDevice, NativeOpsHolder.getInstance().getDeviceNativeOps()
                    .dbDeviceId(keys.data().opaqueBuffer()));
            Nd4j.getAffinityManager().setDeviceForCurrentThread(0);
            SDVariable q = sd.placeHolder("q", dtype, 1, 1, 2);
            SDVariable k = sd.placeHolder("k", dtype, 1, 1, 2);
            SDVariable v = sd.placeHolder("v", dtype, 1, 1, 2);
            SDVariable pastK = sd.placeHolder("past_key_values.0.key", dtype, 1, 1, capacity, 2);
            SDVariable pastV = sd.placeHolder("past_key_values.0.value", dtype, 1, 1, capacity, 2);
            SDVariable mask = sd.placeHolder("mask", dtype, 1, 1, 1, capacity + 1);
            SDVariable placement = sd.placeHolder("placement", dtype, 256);
            SDVariable[] outputs = new org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention(
                    sd, q.add(placement.sum()), k, v, mask, pastK, pastV, 1, 1.0, false).outputVariables();
            String[] outputNames = {"out", "present.0.key", "present.0.value"};
            for (int i = 0; i < outputNames.length; i++) sd.updateVariableNameAndReference(outputs[i], outputNames[i]);
            var plan = sd.compileDynamicShapePlan(outputNames);
            for (var slot : plan.getSlots()) slot.setTargetDeviceId(attentionDevice);
            sd.compileNativeDynamicShapePlan(outputNames);
            double[] expectedK = new double[capacity * 2];
            double[] expectedV = new double[capacity * 2];
            for (int step = 1; step < capacity; step++) {
                assertEquals(step, manager.getCachePosition());
                try (INDArray query = Nd4j.zeros(dtype, 1, 1, 2);
                     INDArray key = Nd4j.createFromArray((float) step, (float) step + 1).castTo(dtype).reshape(1, 1, 2);
                     INDArray value = Nd4j.createFromArray((float) (2 * step), (float) (2 * step + 1)).castTo(dtype).reshape(1, 1, 2);
                     INDArray bias = Nd4j.zeros(dtype, 1, 1, 1, capacity + 1);
                     INDArray weight = Nd4j.zeros(dtype, 256)) {
                    // Concat mode appends the new entry AFTER the full padded past.
                    // Mask unused past slots but retain the appended entry.
                    for (int i = step; i < capacity; i++) bias.putScalar(i, -10000);
                    Map<String, INDArray> feeds = new HashMap<>(Map.of(
                            "q", query, "k", key, "v", value, "mask", bias, "placement", weight));
                    manager.prepareInputs(feeds, sd, 2, true);
                    assertSame(keys, feeds.get("past_key_values.0.key"));
                    assertSame(values, feeds.get("past_key_values.0.value"));
                    Map<String, INDArray> result = sd.output(feeds, outputNames);
                    try {
                        expectedK[step * 2] = step;
                        expectedK[step * 2 + 1] = step + 1;
                        expectedV[step * 2] = 2 * step;
                        expectedV[step * 2 + 1] = 2 * step + 1;
                        double[] expectedOut = new double[2];
                        for (int i = 0; i <= step; i++) {
                            expectedOut[0] += expectedV[i * 2] / (step + 1);
                            expectedOut[1] += expectedV[i * 2 + 1] / (step + 1);
                        }
                        assertArrayEquals(expectedOut, result.get("out").data().asDouble(),
                                dtype == DataType.HALF ? 0.1 : 1e-4, "manager attention step " + step);
                        // Production manager loop: scatter the present outputs, then
                        // retire them. The manager remains the static-buffer owner.
                        manager.scatterNewEntries(result, names);
                        assertEquals(step + 1, manager.getCachePosition());
                        assertArrayEquals(expectedK, keys.data().asDouble(), 0.0, "manager keys step " + step);
                        assertArrayEquals(expectedV, values.data().asDouble(), 0.0, "manager values step " + step);
                        assertSame(keys, manager.getStaticKvBuffers().get("past_key_values.0.key"));
                    } finally {
                        result.values().forEach(INDArray::close);
                    }
                }
            }
            assertTrue(DspPlanAssertions.getTotalGraphReplays(sd) > 0, "manager decode must reach replay");
            DspPlanAssertions.assertNoCaptureFailures(sd, "manager prepare/scatter lifecycle");
        } finally {
            Nd4j.getAffinityManager().setDeviceForCurrentThread(savedDevice);
        }
    }

    private static UnifiedKvCacheManager initializedManager(DataType dtype, int capacity) {
        UnifiedKvCacheManager manager = new UnifiedKvCacheManager();
        try (INDArray key = Nd4j.zeros(dtype, 1, 1, 1, 2);
             INDArray value = Nd4j.zeros(dtype, 1, 1, 1, 2)) {
            manager.initializeFromPrefill(Map.of("present.0.key", key, "present.0.value", value),
                    new ModelIOConfig.KVCacheNames(java.util.List.of("present.0.key"),
                            java.util.List.of("present.0.value")), capacity - 1, 1);
            return manager;
        } catch (RuntimeException | Error failure) {
            manager.close();
            throw failure;
        }
    }

    private void checkRetainedCaches(int callerDevice, int attentionDevice, DataType dtype) {
        assumeTrue(Nd4j.backends().isCudaAvailable(), "requires CUDA");
        assumeTrue(Nd4j.getAffinityManager().getNumberOfDevices() >= 2, "requires two CUDA devices");
        int savedDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        // Assigned cases exercise both segment directions. Automatic cases
        // use a larger device-0 feed to keep execution away from the caller cache.
        Nd4j.getAffinityManager().setDeviceForCurrentThread(callerDevice);
        try (INDArray keyCache = Nd4j.zeros(dtype, 1, 4, 1, 2);
             INDArray valueCache = Nd4j.zeros(dtype, 1, 4, 1, 2);
             SameDiff sd = SameDiff.create()) {
            assertEquals(callerDevice, NativeOpsHolder.getInstance().getDeviceNativeOps()
                    .dbDeviceId(keyCache.data().opaqueBuffer()), "initial key cache device");
            assertEquals(callerDevice, NativeOpsHolder.getInstance().getDeviceNativeOps()
                    .dbDeviceId(valueCache.data().opaqueBuffer()), "initial value cache device");
            Nd4j.getAffinityManager().setDeviceForCurrentThread(0);
            SDVariable q = sd.placeHolder("q", dtype, 1, 1, 1, 2);
            SDVariable k = sd.placeHolder("k", dtype, 1, 1, 1, 2);
            SDVariable v = sd.placeHolder("v", dtype, 1, 1, 1, 2);
            SDVariable keys = sd.placeHolder("keys", dtype, 1, 4, 1, 2);
            SDVariable values = sd.placeHolder("values", dtype, 1, 4, 1, 2);
            SDVariable position = sd.placeHolder("position", DataType.LONG);
            SDVariable bias = sd.placeHolder("bias", dtype, 1, 1, 1, 4);
            SDVariable effectiveQuery = attentionDevice < 0
                    ? q.add(sd.placeHolder("placement", dtype, 64).sum()) : q;
            SDVariable out = sd.nn.dotProductAttentionV2("out", effectiveQuery, v, k, null, null,
                    keys, values, position, bias, 1.0, 0.0, false, false);
            // A subsequent reader on the other device must see the writer's
            // publication. It must not reclassify the state as input-only.
            keys.add(out.sum()).sum("probe");
            var plan = sd.compileDynamicShapePlan("out", "probe");
            plan.assignDevices(Map.of(0, 1L, 1, 1L));
            for (var slot : plan.getSlots()) {
                slot.setTargetDeviceId(attentionDevice < 0 ? -1
                        : "dot_product_attention_v2".equals(slot.getOpName())
                                ? attentionDevice : 1 - attentionDevice);
            }
            sd.compileNativeDynamicShapePlan("out", "probe");
            {
                double[] expectedK = new double[8];
                double[] expectedV = new double[8];
                for (int iteration = 0; iteration < 16; iteration++) {
                    int pos = iteration % 4;
                    expectedK[2 * pos] = iteration + 1;
                    expectedK[2 * pos + 1] = iteration + 2;
                    expectedV[2 * pos] = 2 * iteration + 1;
                    expectedV[2 * pos + 1] = 2 * iteration + 2;
                    try (INDArray query = Nd4j.zeros(dtype, 1, 1, 1, 2);
                         INDArray key = Nd4j.createFromArray((float) iteration + 1, (float) iteration + 2)
                                 .castTo(dtype).reshape(1, 1, 1, 2);
                         INDArray value = Nd4j.createFromArray((float) (2 * iteration + 1), (float) (2 * iteration + 2))
                                 .castTo(dtype).reshape(1, 1, 1, 2);
                         INDArray cachePosition = Nd4j.scalar(DataType.LONG, pos);
                         INDArray mask = Nd4j.zeros(dtype, 1, 1, 1, 4);
                         INDArray placement = Nd4j.zeros(dtype, 64)) {
                        for (int i = pos + 1; i < 4; i++) mask.putScalar(i, -10000);
                        Map<String, INDArray> feeds = new HashMap<>(Map.of("q", query, "k", key, "v", value,
                                "keys", keyCache, "values", valueCache, "position", cachePosition, "bias", mask));
                        if (attentionDevice < 0) feeds.put("placement", placement);
                        Map<String, INDArray> result = sd.output(feeds, "out", "probe");
                        try {
                            if (attentionDevice < 0) {
                                assertEquals(0, Nd4j.getAffinityManager().getDeviceForCurrentThread(),
                                        "automatic placement must exercise remote caller state");
                            }
                            assertArrayEquals(expectedK, keyCache.data().asDouble(), 0.0, "retained keys " + iteration);
                            assertArrayEquals(expectedV, valueCache.data().asDouble(), 0.0, "retained values " + iteration);
                            double[] expectedOut = new double[2];
                            for (int i = 0; i <= pos; i++) {
                                expectedOut[0] += expectedV[2 * i] / (pos + 1);
                                expectedOut[1] += expectedV[2 * i + 1] / (pos + 1);
                            }
                            double tolerance = dtype == DataType.HALF ? 0.1 : 1e-4;
                            assertArrayEquals(expectedOut, result.get("out").data().asDouble(), tolerance);
                            double expectedProbe = Arrays.stream(expectedK).sum()
                                    + 8 * (expectedOut[0] + expectedOut[1]);
                            assertEquals(expectedProbe, result.get("probe").getDouble(0),
                                    dtype == DataType.HALF ? 2.0 : 1e-3);
                            var executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
                            var nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                            for (String name : new String[]{"keys", "values"}) {
                                int index = Arrays.asList(executor.getCurrentPlan().getExternalInputKeys()).indexOf(name);
                                assertTrue(nativeOps.getPlanIsExternalInputVariable(executor.getNativePlanHandle(), index));
                                assertFalse(nativeOps.getPlanIsExternalInputPlaceholder(executor.getNativePlanHandle(), index));
                                INDArray caller = name.equals("keys") ? keyCache : valueCache;
                                assertSame(caller, executor.getExternalInputsSnapshot()[index],
                                        "Java must preserve caller state");
                                assertEquals(callerDevice, nativeOps.dbDeviceId(caller.data().opaqueBuffer()),
                                        "native execution must preserve caller state device");
                            }
                        } finally {
                            result.values().forEach(INDArray::close);
                        }
                    }
                }
                assertTrue(DspPlanAssertions.getTotalGraphReplays(sd) > 0, "stable state replicas must permit replay");
                DspPlanAssertions.assertNoCaptureFailures(sd, "attention state writeback");
            }
        } finally {
            Nd4j.getAffinityManager().setDeviceForCurrentThread(savedDevice);
        }
    }
}
