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
        checkRetainedCaches(callerDevice, attentionDevice, dtype, false);
    }

    @ParameterizedTest
    @CsvSource({"0,1,FLOAT", "1,0,FLOAT", "0,1,HALF", "1,0,HALF",
            "1,-1,FLOAT", "1,-1,HALF"})
    void managerCachesAreWrittenAcrossDevices(int callerDevice, int attentionDevice, DataType dtype) {
        checkRetainedCaches(callerDevice, attentionDevice, dtype, true);
    }

    private static UnifiedKvCacheManager initializedManager(DataType dtype) {
        UnifiedKvCacheManager manager = new UnifiedKvCacheManager();
        try (INDArray key = Nd4j.zeros(dtype, 1, 1, 1, 2);
             INDArray value = Nd4j.zeros(dtype, 1, 1, 1, 2)) {
            manager.initializeFromPrefill(Map.of("present.0.key", key, "present.0.value", value),
                    new ModelIOConfig.KVCacheNames(java.util.List.of("present.0.key"),
                            java.util.List.of("present.0.value")), 3, 1);
            return manager;
        } catch (RuntimeException | Error failure) {
            manager.close();
            throw failure;
        }
    }

    private void checkRetainedCaches(int callerDevice, int attentionDevice, DataType dtype,
                                     boolean managerBacked) {
        assumeTrue(Nd4j.backends().isCudaAvailable(), "requires CUDA");
        assumeTrue(Nd4j.getAffinityManager().getNumberOfDevices() >= 2, "requires two CUDA devices");
        int savedDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        // Assigned cases exercise both segment directions. Automatic cases
        // use a larger device-0 feed to keep execution away from the caller cache.
        Nd4j.getAffinityManager().setDeviceForCurrentThread(callerDevice);
        // Manager owns BHSD storage. Persistent BSHD views preserve that exact
        // backing allocation for DPA's in-graph writes; do not scatter twice.
        try (UnifiedKvCacheManager manager = managerBacked ? initializedManager(dtype) : null;
             INDArray keyCache = manager == null ? Nd4j.zeros(dtype, 1, 4, 1, 2)
                     : manager.getStaticKvBuffers().get("past_key_values.0.key").permute(0, 2, 1, 3);
             INDArray valueCache = manager == null ? Nd4j.zeros(dtype, 1, 4, 1, 2)
                     : manager.getStaticKvBuffers().get("past_key_values.0.value").permute(0, 2, 1, 3);
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
                    if (manager != null) {
                        manager.setCachePosition(pos);
                        assertEquals(pos, manager.getCachePosition());
                        assertSame(manager.getStaticKvBuffers().get("past_key_values.0.key").data(), keyCache.data());
                        assertSame(manager.getStaticKvBuffers().get("past_key_values.0.value").data(), valueCache.data());
                    }
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
                            if (iteration == 0) {
                                var executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
                                var nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                                for (String name : new String[]{"keys", "values"}) {
                                    int index = Arrays.asList(executor.getCurrentPlan().getExternalInputKeys()).indexOf(name);
                                    INDArray caller = name.equals("keys") ? keyCache : valueCache;
                                    INDArray bound = executor.getExternalInputsSnapshot()[index];
                                    System.out.println("KV_WRITEBACK_BOUND callerDevice=" + callerDevice
                                            + " attentionDevice=" + attentionDevice + " name=" + name
                                            + " callerDb=" + caller.data().opaqueBuffer().address()
                                            + " boundDb=" + bound.data().opaqueBuffer().address()
                                            + " callerNativeDevice=" + nativeOps.dbDeviceId(caller.data().opaqueBuffer())
                                            + " boundNativeDevice=" + nativeOps.dbDeviceId(bound.data().opaqueBuffer())
                                            + " variable=" + nativeOps.getPlanIsExternalInputVariable(executor.getNativePlanHandle(), index)
                                            + " placeholder=" + nativeOps.getPlanIsExternalInputPlaceholder(executor.getNativePlanHandle(), index));
                                }
                            }
                            try {
                                assertArrayEquals(expectedK, keyCache.data().asDouble(), 0.0, "retained keys " + iteration);
                            } catch (AssertionError failure) {
                                var ops = NativeOpsHolder.getInstance().getDeviceNativeOps();
                                System.out.println("KV_FAILED_READ constant=" + ops.dbIsConstant(keyCache.data().opaqueBuffer()));
                                ops.dbForceSyncToPrimary(keyCache.data().opaqueBuffer());
                                System.out.println("KV_FAILED_READ deviceCopy=" + Arrays.toString(keyCache.data().asDouble()));
                                throw failure; // Diagnostic only: never turn a stale read into a pass.
                            }
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
