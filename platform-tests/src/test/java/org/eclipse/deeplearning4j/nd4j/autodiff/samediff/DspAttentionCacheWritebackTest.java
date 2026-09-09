/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

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
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Caller-visible attention state must survive both migration directions and replay. */
public class DspAttentionCacheWritebackTest extends BaseND4JTest {
    @Override
    public DataType getDataType() { return DataType.FLOAT; }

    @ParameterizedTest
    @CsvSource({"0,1,FLOAT", "1,0,FLOAT", "0,1,HALF", "1,0,HALF"})
    void retainedCachesAreWrittenAcrossDevices(int callerDevice, int attentionDevice, DataType dtype) {
        assumeTrue(Nd4j.backends().isCudaAvailable(), "requires CUDA");
        assumeTrue(Nd4j.getAffinityManager().getNumberOfDevices() >= 2, "requires two CUDA devices");
        int savedDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        // The executor stays on device 0; the reverse case exercises the Java
        // input-replica path as well as the native secondary-segment path.
        Nd4j.getAffinityManager().setDeviceForCurrentThread(callerDevice);
        try (INDArray keyCache = Nd4j.zeros(dtype, 1, 4, 1, 2);
             INDArray valueCache = Nd4j.zeros(dtype, 1, 4, 1, 2);
             SameDiff sd = SameDiff.create()) {
            Nd4j.getAffinityManager().setDeviceForCurrentThread(0);
            SDVariable q = sd.placeHolder("q", dtype, 1, 1, 1, 2);
            SDVariable k = sd.placeHolder("k", dtype, 1, 1, 1, 2);
            SDVariable v = sd.placeHolder("v", dtype, 1, 1, 1, 2);
            SDVariable keys = sd.placeHolder("keys", dtype, 1, 4, 1, 2);
            SDVariable values = sd.placeHolder("values", dtype, 1, 4, 1, 2);
            SDVariable position = sd.placeHolder("position", DataType.LONG);
            SDVariable bias = sd.placeHolder("bias", dtype, 1, 1, 1, 4);
            SDVariable out = sd.nn.dotProductAttentionV2("out", q, v, k, null, null,
                    keys, values, position, bias, 1.0, 0.0, false, false);
            // A subsequent reader on the other device must see the writer's
            // publication. It must not reclassify the state as input-only.
            keys.add(out.sum()).sum("probe");
            var plan = sd.compileDynamicShapePlan("out", "probe");
            plan.assignDevices(Map.of(0, 1L, 1, 1L));
            for (var slot : plan.getSlots()) {
                slot.setTargetDeviceId("dot_product_attention_v2".equals(slot.getOpName())
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
                         INDArray mask = Nd4j.zeros(dtype, 1, 1, 1, 4)) {
                        for (int i = pos + 1; i < 4; i++) mask.putScalar(i, -10000);
                        Map<String, INDArray> result = sd.output(Map.of("q", query, "k", key, "v", value,
                                "keys", keyCache, "values", valueCache, "position", cachePosition, "bias", mask),
                                "out", "probe");
                        try {
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
                                assertSame(name.equals("keys") ? keyCache : valueCache,
                                        executor.getExternalInputsSnapshot()[index], "Java must preserve caller state");
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
