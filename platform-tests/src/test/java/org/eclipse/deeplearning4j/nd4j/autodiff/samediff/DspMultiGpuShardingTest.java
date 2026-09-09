/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.eclipse.deeplearning4j.llm.generation.SameDiffMemoryUtils;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.ExecutionPhase;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutorTestAccess.*;

/**
 * Isolated behavior tests for automatic DSP multi-GPU op-segment sharding.
 *
 * <p>These flesh out the non-P2P cross-device execution path — input migration,
 * secondary-device constant replication, and output-back-migration — on minimal
 * MLPs with a single-GPU reference, so each behavior can be verified and
 * regressed far faster than the full BGE model.</p>
 *
 * <p>Requires &gt;1 CUDA device; every test is skipped otherwise.</p>
 */
@Slf4j
public class DspMultiGpuShardingTest extends BaseND4JTest {

    @Override
    public DataType getDataType() {
        return DataType.FLOAT;
    }

    @Override
    public long getTimeoutMilliseconds() {
        return 10 * 60 * 1000L;
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /**
     * A small tanh MLP.  Same seed → same weight initialisation so single-GPU
     * and sharded instances are numerically identical before any execution.
     */
    private static SameDiff buildMlp(int inDim, int hidden, int layers, int outDim, long seed) {
        Nd4j.getRandom().setSeed(seed);
        SameDiff sd = SameDiff.create();
        SDVariable h = sd.placeHolder("x", DataType.FLOAT, -1, inDim);
        int dim = inDim;
        for (int l = 0; l < layers; l++) {
            SDVariable w = sd.var("w" + l, Nd4j.rand(DataType.FLOAT, dim, hidden).muli(0.1));
            SDVariable b = sd.var("b" + l, Nd4j.rand(DataType.FLOAT, hidden).muli(0.01));
            h = sd.math().tanh(h.mmul(w).add(b));
            dim = hidden;
        }
        SDVariable wo = sd.var("wo", Nd4j.rand(DataType.FLOAT, dim, outDim).muli(0.1));
        SDVariable out = h.mmul(wo);
        out.rename("out");
        return sd;
    }

    /** A small fan-out graph whose parallel matmuls share one predecessor. */
    private static SameDiff buildParallelMlp(int inDim, int branches, int outDim, long seed) {
        Nd4j.getRandom().setSeed(seed);
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, inDim);
        SDVariable shared = sd.math().tanh(x);
        SDVariable sum = null;
        for (int branch = 0; branch < branches; branch++) {
            SDVariable weight = sd.var("pw" + branch,
                    Nd4j.rand(DataType.FLOAT, inDim, outDim).muli(0.1));
            SDVariable value = shared.mmul(weight);
            sum = sum == null ? value : sum.add(value);
        }
        sum.rename("out");
        return sd;
    }

    /** Device-local staging must not repeat plan-wide H2D preparation on a secondary GPU. */
    @Test
    public void testPreReplayKeepsPrimaryOnlyInputOnItsDevice() {
        assumeTrue(Nd4j.backends().isCudaAvailable(), "requires CUDA");
        assumeTrue(Nd4j.getAffinityManager().getNumberOfDevices() == 2, "requires two CUDA devices");
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int originalDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        boolean originalDsp = InferenceSession.isDynamicShapePlanEnabled();
        long originalLimit = Nd4j.getEnvironment().getDeviceLimit(1);
        final int width = 4 * 1024 * 1024;
        SameDiff sd = null;
        INDArray input = null;
        try {
            InferenceSession.setDynamicShapePlanEnabled(true);
            Nd4j.getAffinityManager().setDeviceForCurrentThread(0);
            input = Nd4j.create(DataType.FLOAT, 1, width).assign(1.0);
            nativeOps.dbSyncToSpecial(input.data().opaqueBuffer());
            sd = SameDiff.create();
            SDVariable x = sd.placeHolder("primaryOnly", DataType.FLOAT, 1, width);
            x.sum("primarySum", 1).add("out", 1.0);
            DynamicShapePlan plan = sd.compileDynamicShapePlan("out");
            for (var slot : plan.getSlots()) {
                slot.setTargetDeviceId(Arrays.asList(slot.getOutputVarNames()).contains("out") ? 1 : 0);
            }
            sd.compileNativeDynamicShapePlan("out");
            long limit = Nd4j.getEnvironment().getDeviceCounter(1) + 8L * 1024 * 1024;
            if (originalLimit > 0) limit = Math.min(limit, originalLimit);
            Nd4j.getEnvironment().setDeviceLimit(1, limit);
            for (int iteration = 0; iteration < 6; iteration++) {
                Nd4j.getAffinityManager().setDeviceForCurrentThread(0);
                input.assign((iteration + 1) * 0.125);
                INDArray output = sd.output(Map.of("primaryOnly", input), "out").get("out");
                try {
                    assertEquals(0, nativeOps.dbDeviceId(input.data().opaqueBuffer()),
                            "secondary staging must not migrate a primary-only input");
                    assertTrue(Nd4j.getEnvironment().getDeviceCounter(1) <= limit);
                    assertEquals(width * (iteration + 1) * 0.125 + 1.0, output.getDouble(0), 0.0);
                } finally {
                    SameDiffMemoryUtils.safeClose(output);
                }
            }
            assertTrue(countAssignedSlots(plan, 0) > 0);
            assertTrue(countAssignedSlots(plan, 1) > 0);
        } finally {
            try {
                if (sd != null) sd.close();
            } finally {
                SameDiffMemoryUtils.safeClose(input);
                Nd4j.getEnvironment().setDeviceLimit(1, originalLimit);
                InferenceSession.setDynamicShapePlanEnabled(originalDsp);
                Nd4j.getAffinityManager().setDeviceForCurrentThread(originalDevice);
                SameDiffMemoryUtils.reclaimClosedGraphResources();
            }
        }
    }

    private static int countAssignedSlots(DynamicShapePlan plan, int deviceId) {
        int count = 0;
        for (var slot : plan.getSlots()) {
            if (slot.getTargetDeviceId() == deviceId) count++;
        }
        return count;
    }

    private static long poolAwareAvailableMemory(NativeOps nativeOps, int deviceId) {
        long cudaFree = nativeOps.getDeviceFreeMemory(deviceId);
        try (LongPointer used = new LongPointer(1);
             LongPointer reserved = new LongPointer(1)) {
            nativeOps.getMemoryPoolStats(deviceId, used, reserved);
            return cudaFree + Math.max(0L, reserved.get() - used.get());
        }
    }

    /**
     * Run {@code sd} once with DSP enabled. Automatic multi-GPU placement is the default;
     * {@code singleGpu} exercises the explicit opt-out used for a reference result.
     * Returns a {@code dup()} of the output (avoids CUDA view-staleness on the caller side).
     * Saves and restores both the DSP-enabled flag and the single-GPU system property.
     */
    private static INDArray runOnce(SameDiff sd, INDArray x, boolean singleGpu) {
        boolean prevDsp = InferenceSession.isDynamicShapePlanEnabled();
        String prevSingleGpu = System.getProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
        InferenceSession.setDynamicShapePlanEnabled(true);
        if (singleGpu) System.setProperty(ND4JSystemProperties.DSP_SINGLE_GPU, "true");
        else System.clearProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
        try {
            Map<String, INDArray> res = sd.output(Collections.singletonMap("x", x), "out");
            return res.get("out").dup();
        } finally {
            InferenceSession.setDynamicShapePlanEnabled(prevDsp);
            if (prevSingleGpu != null) System.setProperty(ND4JSystemProperties.DSP_SINGLE_GPU, prevSingleGpu);
            else System.clearProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
        }
    }

    private static void assertUsesEveryCudaDevice(SameDiff sd) {
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        assertNotNull(executor, "automatic multi-GPU run did not create a DSP executor");
        DynamicShapePlan plan = executor.getCurrentPlan();
        assertNotNull(plan, "automatic multi-GPU run did not retain its DSP plan");

        Set<Integer> assignedDevices = new HashSet<>();
        for (var slot : plan.getSlots()) {
            if (slot.getTargetDeviceId() >= 0) assignedDevices.add(slot.getTargetDeviceId());
        }
        int availableDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        for (int device = 0; device < availableDevices; device++) {
            assertTrue(assignedDevices.contains(device),
                    "automatic DSP placement omitted CUDA device " + device + ": "
                            + plan.getDeviceAssignmentSummary());
        }
        assertEquals(availableDevices, plan.getNumDistinctDevices(),
                "DSP plan must report every available CUDA device");
    }

    // -----------------------------------------------------------------------
    // Test 1 — sharded first-pass correctness
    // -----------------------------------------------------------------------

    /**
     * Behavior 1 (correctness): a sharded plan must produce the same output as the
     * single-GPU plan.  Exercises cross-device input migration, secondary-device
     * constant replication, and output-back-migration end-to-end on a 6-layer MLP.
     */
    @Test
    public void testShardedMlpMatchesSingleGpu() {
        assumeTrue(Nd4j.getAffinityManager().getNumberOfDevices() > 1,
                "multi-GPU sharding requires >1 CUDA device");

        INDArray x = Nd4j.rand(DataType.FLOAT, 8, 64);
        SameDiff single  = buildMlp(64, 128, 6, 16, 12345L);
        SameDiff sharded = buildMlp(64, 128, 6, 16, 12345L);

        INDArray ref = runOnce(single,  x.dup(), true);
        INDArray got = runOnce(sharded, x.dup(), false);
        assertUsesEveryCudaDevice(sharded);

        assertArrayEquals(ref.shape(), got.shape(), "sharded output shape must match single-GPU");
        double maxDiff = ref.sub(got).amaxNumber().doubleValue();
        log.info("testShardedMlpMatchesSingleGpu: maxAbsDiff={}, refMaxAbs={}", maxDiff, ref.amaxNumber());
        assertFalse(got.isNaN().any(), "sharded output has NaN");
        assertTrue(got.equalsWithEps(ref, 1e-3),
                "sharded output must match single-GPU within 1e-3 (maxAbsDiff=" + maxDiff + ")");
    }

    // -----------------------------------------------------------------------
    // Test 2 — steady-state replay: same SameDiff instance, N iterations
    // -----------------------------------------------------------------------

    /**
     * Behavior 2 (steady-state replay): run the same sharded SameDiff instance
     * {@value #REPLAY_ITERATIONS} times with the same input shapes and assert every
     * iteration still matches the single-GPU reference within 1e-3.
     *
     * <p>The first few iterations are slot-by-slot warmup; later ones replay the
     * captured per-device CUDA graphs.  Regressions here indicate
     * capture→replay divergence, not just first-pass correctness.</p>
     */
    private static final int REPLAY_ITERATIONS = 10;

    @Test
    public void testRepeatedShardedExecutionMatchesSingleGpu() {
        assumeTrue(Nd4j.getAffinityManager().getNumberOfDevices() > 1,
                "multi-GPU sharding requires >1 CUDA device");

        INDArray x = Nd4j.rand(DataType.FLOAT, 8, 64);

        // Single-GPU reference evaluated once — its output is stable.
        SameDiff single = buildMlp(64, 128, 6, 16, 42L);
        INDArray ref = runOnce(single, x.dup(), true);

        // ONE sharded instance reused across all iterations to exercise steady state.
        SameDiff sharded = buildMlp(64, 128, 6, 16, 42L);

        boolean prevDsp   = InferenceSession.isDynamicShapePlanEnabled();
        String prevSingleGpu = System.getProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
        InferenceSession.setDynamicShapePlanEnabled(true);
        System.clearProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
        try {
            for (int i = 0; i < REPLAY_ITERATIONS; i++) {
                Map<String, INDArray> res = sharded.output(Collections.singletonMap("x", x.dup()), "out");
                INDArray got = res.get("out").dup();
                assertUsesEveryCudaDevice(sharded);

                assertArrayEquals(ref.shape(), got.shape(),
                        "sharded output shape mismatch at iteration " + i);
                assertFalse(got.isNaN().any(),
                        "sharded output has NaN at iteration " + i);

                double maxDiff = ref.sub(got).amaxNumber().doubleValue();
                log.info("testRepeatedShardedExecutionMatchesSingleGpu iteration {}: maxAbsDiff={}", i, maxDiff);
                assertTrue(got.equalsWithEps(ref, 1e-3),
                        "sharded output diverged at iteration " + i
                                + " (maxAbsDiff=" + maxDiff + ")");
            }
        } finally {
            InferenceSession.setDynamicShapePlanEnabled(prevDsp);
            if (prevSingleGpu != null) System.setProperty(ND4JSystemProperties.DSP_SINGLE_GPU, prevSingleGpu);
            else System.clearProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
        }
    }

    /** A reused sharded plan must not retain the previous caller's stream/device. */
    @Test
    public void testShardedReplayWithAlternatingCallerDevice() {
        assumeTrue(Nd4j.backends().isCudaAvailable(), "requires CUDA");
        int deviceCount = Nd4j.getAffinityManager().getNumberOfDevices();
        assumeTrue(deviceCount > 1, "requires multiple CUDA devices");
        int originalDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        SameDiff single = null;
        SameDiff sharded = null;
        INDArray input = null;
        INDArray reference = null;
        boolean originalDsp = InferenceSession.isDynamicShapePlanEnabled();
        String originalSingleGpu = System.getProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
        try {
            InferenceSession.setDynamicShapePlanEnabled(true);
            System.clearProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
            Nd4j.getAffinityManager().setDeviceForCurrentThread(0);
            input = Nd4j.rand(DataType.FLOAT, 8, 64);
            single = buildViewMlp(64, 128, 6, 16, 42L);
            reference = runOnce(single, input, true);
            sharded = buildViewMlp(64, 128, 6, 16, 42L);
            sharded.setGraphExecutionMode(GraphExecutionMode.TRITON);
            for (int iteration = 0; iteration < REPLAY_ITERATIONS; iteration++) {
                int callerDevice = iteration % deviceCount;
                Nd4j.getAffinityManager().setDeviceForCurrentThread(callerDevice);
                Map<String, INDArray> result = sharded.output(Collections.singletonMap("x", input), "out");
                assertEquals(callerDevice, Nd4j.getAffinityManager().getDeviceForCurrentThread(),
                        "execution/readback changed caller device at iteration " + iteration);
                try (INDArray got = result.get("out").dup()) {
                    assertUsesEveryCudaDevice(sharded);
                    assertTrue(got.equalsWithEps(reference, 1e-3),
                            "alternating-device replay diverged at iteration " + iteration);
                }
            }
            DspPlanAssertions.assertNoCaptureFailures(sharded, "alternating caller device");
        } finally {
            try {
                if (sharded != null) sharded.close();
                if (single != null) single.close();
                SameDiffMemoryUtils.safeClose(reference);
                SameDiffMemoryUtils.safeClose(input);
            } finally {
                InferenceSession.setDynamicShapePlanEnabled(originalDsp);
                if (originalSingleGpu == null) System.clearProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
                else System.setProperty(ND4JSystemProperties.DSP_SINGLE_GPU, originalSingleGpu);
                Nd4j.getAffinityManager().setDeviceForCurrentThread(originalDevice);
            }
        }
    }

    /** Exercise the native nested-plan entry, not only Java executor dispatch. */
    @Test
    public void testNativeDecodeAfterShardedJavaWarmup() {
        runNativeDecodeAfterShardedJavaWarmup(false);
    }

    @Test
    public void testNativeKvAttentionAfterShardedJavaWarmup() {
        runNativeDecodeAfterShardedJavaWarmup(true);
    }

    @Test
    public void testNativeFullPlanReentryFromForeignDevice() {
        runNativeDecodeAfterShardedJavaWarmup(true, true);
    }

    private void runNativeDecodeAfterShardedJavaWarmup(boolean withKvAttention) {
        runNativeDecodeAfterShardedJavaWarmup(withKvAttention, false);
    }

    private void runNativeDecodeAfterShardedJavaWarmup(boolean withKvAttention, boolean fullPlanReentry) {
        assumeTrue(Nd4j.backends().isCudaAvailable(), "requires CUDA");
        assumeTrue(Nd4j.getAffinityManager().getNumberOfDevices() == 2, "requires two CUDA devices");
        int originalDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        boolean originalDsp = InferenceSession.isDynamicShapePlanEnabled();
        String originalSingleGpu = System.getProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
        SameDiff graph = null;
        java.util.List<INDArray> owned = new java.util.ArrayList<>();
        boolean originalAllocationLogging = Nd4j.getEnvironment().isLogNativeNDArrayCreation();
        try {
            if (Boolean.getBoolean("dsp.test.traceNativeLifetime")) {
                Nd4j.getEnvironment().setLogNativeNDArrayCreation(true);
            }
            InferenceSession.setDynamicShapePlanEnabled(true);
            System.clearProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
            Nd4j.getAffinityManager().setDeviceForCurrentThread(0);
            graph = SameDiff.create();
            graph.setGraphExecutionMode(GraphExecutionMode.TRITON);
            Map<String, INDArray> feeds = new java.util.LinkedHashMap<>();
            java.util.List<INDArray> kvArrays = new java.util.ArrayList<>();
            java.util.List<String> kvNames = new java.util.ArrayList<>();
            SDVariable cachePosition = null;
            if (withKvAttention) {
                cachePosition = graph.placeHolder("cache_position", DataType.LONG);
                INDArray position = Nd4j.scalar(DataType.LONG, 0);
                owned.add(position);
                feeds.put("cache_position", position);
            }
            SDVariable h = graph.placeHolder("embeddings", DataType.FLOAT, 1, 1, 16).reshape(1, 16);
            for (int layer = 0; layer < 6; layer++) {
                INDArray weights = Nd4j.eye(16).castTo(DataType.FLOAT);
                owned.add(weights);
                h = graph.nn.relu(graph.linalg.mmul(h, graph.constant("w" + layer, weights)), 0);
                if (withKvAttention) {
                    String keyName = "past_key_values." + layer + ".key";
                    String valueName = "past_key_values." + layer + ".value";
                    INDArray key = Nd4j.ones(DataType.HALF, 1, 8, 1, 16);
                    INDArray value = Nd4j.ones(DataType.HALF, 1, 8, 1, 16);
                    Collections.addAll(owned, key, value);
                    Collections.addAll(kvArrays, key, value);
                    Collections.addAll(kvNames, keyName, valueName);
                    feeds.put(keyName, key);
                    feeds.put(valueName, value);
                    SDVariable qkv = h.reshape(1, 1, 1, 16);
                    h = graph.nn.dotProductAttentionV2("attention" + layer, qkv, qkv, qkv, null, null,
                            graph.placeHolder(keyName, DataType.HALF, 1, 8, 1, 16),
                            graph.placeHolder(valueName, DataType.HALF, 1, 8, 1, 16),
                            cachePosition, null, 0.0, 0.0, false, false).reshape(1, 16);
                }
            }
            float[][] logitsWeights = new float[16][8];
            for (int row = 0; row < 16; row++) {
                for (int col = 0; col < 8; col++) logitsWeights[row][col] = (col + 1) / 16.0f;
            }
            INDArray projection = Nd4j.createFromArray(logitsWeights);
            owned.add(projection);
            graph.reshape("logits", graph.linalg.mmul(h, graph.constant("projection", projection)), 1, 1, 8);
            INDArray embeddings = Nd4j.ones(DataType.FLOAT, 1, 1, 16);
            INDArray table = Nd4j.ones(DataType.FLOAT, 8, 16);
            INDArray ids = Nd4j.zeros(DataType.LONG, 1, 1);
            INDArray mask = Nd4j.ones(DataType.FLOAT, 1, 8);
            INDArray positions = Nd4j.zeros(DataType.LONG, 1, 1);
            Collections.addAll(owned, embeddings, table, ids, mask, positions);
            feeds.put("embeddings", embeddings);
            for (int iteration = 0; iteration < 12; iteration++) {
                Map<String, INDArray> warmup = graph.output(feeds, "logits");
                if (withKvAttention) {
                    float[] actual = warmup.get("logits").data().asFloat();
                    for (int token = 0; token < 8; token++) {
                        assertEquals(token + 1, actual[token], 1e-4,
                                "Java KV warmup iteration=" + iteration + " logit=" + token);
                    }
                }
            }
            assertUsesEveryCudaDevice(graph);
            assertTrue(DspPlanAssertions.getTotalGraphReplays(graph) > 0, "warmup must reach graph replay");
            DynamicShapePlanExecutor executor = graph.getOrCreateSession().getDynamicShapePlanExecutor();
            String[] externalKeys = executor.getCurrentPlan().getExternalInputKeys();
            int embeddingIndex = java.util.Arrays.asList(externalKeys).indexOf("embeddings");
            assertTrue(embeddingIndex >= 0);
            int cachePositionIndex = java.util.Arrays.asList(externalKeys).indexOf("cache_position");
            int[] kvIndices = kvNames.stream().mapToInt(name -> java.util.Arrays.asList(externalKeys).indexOf(name)).toArray();
            for (int index : kvIndices) assertTrue(index >= 0, "KV input missing from native plan");
            if (withKvAttention) assertTrue(cachePositionIndex >= 0);
            Nd4j.getAffinityManager().setDeviceForCurrentThread(1);
            if (fullPlanReentry) {
                // Enter the full native lifecycle directly, as executeSteadyState does
                // after invalidation. Do not let Java's executor choose the plan device.
                NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
                int status = nativeOps.executeDynamicShapePlan(executor.getNativePlanHandle(),
                        executor.getCachedOpContext(), nativeOps.dspGetExecutionStream(executor.getNativePlanHandle()));
                assertEquals(0, status, "foreign-device full-plan reentry: " + nativeOps.lastErrorMessage());
                assertEquals(1, Nd4j.getAffinityManager().getDeviceForCurrentThread(),
                        "full-plan reentry must restore the caller device");
            }
            var decode = new org.nd4j.linalg.api.ops.impl.transforms.custom.AutoregressiveDecode(
                    embeddings, table, ids, mask, positions,
                    withKvAttention ? kvArrays.toArray(new INDArray[0]) : null,
                    executor.getNativePlanHandle(), executor.getCachedOpContext(),
                    externalKeys.length, 1, embeddingIndex, -1, -1, -1, -1, 0, -1, cachePositionIndex,
                    kvIndices, new int[0], 3, -1, kvArrays.size() / 2, 1, 0.0, 0, 0.0, 1.0,
                    Collections.emptySet());
            INDArray[] outputs = Nd4j.getExecutioner().exec(decode);
            Collections.addAll(owned, outputs);
            assertEquals(3, outputs[1].getLong(0), "native loop must execute all three steps");
            for (int token = 0; token < 3; token++) {
                assertEquals(7, outputs[0].getLong(token), "wrong token at native step " + token);
            }
            DspPlanAssertions.assertNoCaptureFailures(graph, "native decode handoff");
        } finally {
            try {
                if (graph != null) graph.close();
                for (INDArray array : owned) SameDiffMemoryUtils.safeClose(array);
            } finally {
                Nd4j.getEnvironment().setLogNativeNDArrayCreation(originalAllocationLogging);
                InferenceSession.setDynamicShapePlanEnabled(originalDsp);
                if (originalSingleGpu == null) System.clearProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
                else System.setProperty(ND4JSystemProperties.DSP_SINGLE_GPU, originalSingleGpu);
                Nd4j.getAffinityManager().setDeviceForCurrentThread(originalDevice);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 3 — cross-device output back-migration
    // -----------------------------------------------------------------------

    /**
     * Behavior 3 (cross-device back-migration): a minimal 2-layer graph is executed
     * with sharding ON so the last segment may run on the secondary device; the output
     * must be migrated back to device 0 before Java reads it.  Verifies that the
     * back-migration path preserves values accurately (tol 1e-4, tighter than the
     * 6-layer MLP because the graph is tiny and accumulates less FP rounding).
     *
     * <p>We rely entirely on DSP's own segment-placement logic.  If DSP does not split
     * the 2-layer graph the test still passes (maxAbsDiff will be ~0); the log message
     * records whether any divergence was observed so it is obvious whether the
     * migration path was actually exercised.</p>
     */
    @Test
    public void testCrossDeviceTransferPreservesValues() {
        assumeTrue(Nd4j.getAffinityManager().getNumberOfDevices() > 1,
                "multi-GPU sharding requires >1 CUDA device");

        final long seed = 7777L;
        // Small input — 4 rows × 32 features keeps the test fast.
        INDArray x = Nd4j.rand(DataType.FLOAT, 4, 32);

        SameDiff single  = buildMlp(32, 64, 2, 8, seed);
        SameDiff sharded = buildMlp(32, 64, 2, 8, seed);

        INDArray ref = runOnce(single,  x.dup(), true);
        INDArray got = runOnce(sharded, x.dup(), false);
        assertUsesEveryCudaDevice(sharded);

        assertArrayEquals(ref.shape(), got.shape(),
                "cross-device: output shape must match single-GPU");
        assertFalse(got.isNaN().any(),     "cross-device output has NaN");
        assertFalse(got.isInfinite().any(), "cross-device output has Inf");

        double maxDiff = ref.sub(got).amaxNumber().doubleValue();
        log.info("testCrossDeviceTransferPreservesValues: maxAbsDiff={} "
                + "(0 = same device, >0 = migration path exercised)", maxDiff);
        assertTrue(got.equalsWithEps(ref, 1e-4),
                "cross-device output must match single-GPU within 1e-4 (maxAbsDiff=" + maxDiff + ")");
    }

    // -----------------------------------------------------------------------
    // Test 4 — no NaN/Inf across multiple batch sizes (parameterized)
    // -----------------------------------------------------------------------

    /** Batch sizes swept by {@link #testShardedMlpNoNaNAcrossShapes(int)}. */
    static Stream<Integer> batchSizes() {
        return Stream.of(4, 8, 16);
    }

    /**
     * Behavior 4 (multi-shape stability): run the sharded 6-layer MLP at several batch
     * sizes and assert no NaN/Inf and that the output shape and values match the
     * single-GPU reference within 1e-3.
     *
     * <p>Each batch size drives a distinct shape-keyed plan entry in the DSP plan cache.
     * Regressions here typically indicate per-device plan-keying bugs or workspace
     * sizing that depends on a hardcoded batch dimension.</p>
     */
    @ParameterizedTest
    @MethodSource("batchSizes")
    public void testShardedMlpNoNaNAcrossShapes(int batch) {
        assumeTrue(Nd4j.getAffinityManager().getNumberOfDevices() > 1,
                "multi-GPU sharding requires >1 CUDA device");

        INDArray x = Nd4j.rand(DataType.FLOAT, batch, 64);

        SameDiff single  = buildMlp(64, 128, 6, 16, 55555L);
        SameDiff sharded = buildMlp(64, 128, 6, 16, 55555L);

        INDArray ref = runOnce(single,  x.dup(), true);
        INDArray got = runOnce(sharded, x.dup(), false);
        assertUsesEveryCudaDevice(sharded);

        assertArrayEquals(new long[]{batch, 16}, got.shape(),
                "unexpected output shape for batch=" + batch);
        assertFalse(got.isNaN().any(),
                "NaN in sharded output for batch=" + batch);
        assertFalse(got.isInfinite().any(),
                "Inf in sharded output for batch=" + batch);

        double maxDiff = ref.sub(got).amaxNumber().doubleValue();
        log.info("testShardedMlpNoNaNAcrossShapes batch={}: maxAbsDiff={}", batch, maxDiff);
        assertTrue(got.equalsWithEps(ref, 1e-3),
                "sharded output diverged from single-GPU for batch=" + batch
                        + " (maxAbsDiff=" + maxDiff + ")");
    }

    // -----------------------------------------------------------------------
    // Test 5 — explicit single-GPU override is deterministic
    // -----------------------------------------------------------------------

    /**
     * Behavior 5 (safety): the explicit single-GPU override remains deterministic.
     */
    @Test
    public void testSingleGpuOverrideIsDeterministic() {
        assumeTrue(Nd4j.getAffinityManager().getNumberOfDevices() > 1,
                "requires >1 CUDA device");

        INDArray x = Nd4j.rand(DataType.FLOAT, 8, 64);
        SameDiff a = buildMlp(64, 128, 6, 16, 999L);
        SameDiff b = buildMlp(64, 128, 6, 16, 999L);

        INDArray r1 = runOnce(a, x.dup(), true);
        INDArray r2 = runOnce(b, x.dup(), true);

        assertTrue(r1.equalsWithEps(r2, 1e-6), "single-GPU runs must be deterministic/identical");
    }

    // -----------------------------------------------------------------------
    // Test 6 — VIEW ops across the device boundary (aliasing lifetime)
    // -----------------------------------------------------------------------

    /**
     * A tanh MLP with a transpose/transpose-back (view-capable) op after every layer.
     * When a view op lands at a device-segment boundary it reads a cross-segment input,
     * which the sharded executor migrates to the consuming device as a per-segment copy.
     * A view aliases its input's DataBuffer — so if that migrated copy is freed at segment
     * cleanup while the view output still references it, the alias dangles (garbage / crash),
     * and on captured replay the baked buffer address is stale.
     */
    private static SameDiff buildViewMlp(int inDim, int hidden, int layers, int outDim, long seed) {
        Nd4j.getRandom().setSeed(seed);
        SameDiff sd = SameDiff.create();
        SDVariable h = sd.placeHolder("x", DataType.FLOAT, -1, inDim);
        int dim = inDim;
        for (int l = 0; l < layers; l++) {
            SDVariable w = sd.var("w" + l, Nd4j.rand(DataType.FLOAT, dim, hidden).muli(0.1));
            SDVariable b = sd.var("b" + l, Nd4j.rand(DataType.FLOAT, hidden).muli(0.01));
            h = sd.math().tanh(h.mmul(w).add(b));
            // View-capable ops: transpose then transpose back. Each permute shares its
            // input's DataBuffer (a view), so it exercises the alias-of-migrated-input path.
            SDVariable ht = sd.permute("ht" + l, h, 1, 0);   // [B,H] -> [H,B] (view)
            h = sd.permute("hb" + l, ht, 1, 0);              // [H,B] -> [B,H] (view of view)
            dim = hidden;
        }
        SDVariable wo = sd.var("wo", Nd4j.rand(DataType.FLOAT, dim, outDim).muli(0.1));
        // Make the PLAN OUTPUT itself a view (permute of the final matmul). This exercises
        // output-view materialization + device-N -> device-0 output back-migration of a view
        // in the sharded path, on top of the interior cross-segment view aliasing above.
        SDVariable out = sd.permute("out", h.mmul(wo), 1, 0);  // [B,outDim] -> [outDim,B] (view output)
        return sd;
    }

    /**
     * Behavior 6: a sharded model containing view-capable ops (transpose) at layer
     * boundaries must match the single-GPU reference across {@value #REPLAY_ITERATIONS}
     * iterations — i.e. a view aliasing a cross-device migrated input must not dangle
     * after per-segment migration cleanup, in either slot-by-slot warmup or captured replay.
     */
    @Test
    public void testRepeatedShardedViewOpsMatchSingleGpu() {
        assumeTrue(Nd4j.getAffinityManager().getNumberOfDevices() > 1,
                "multi-GPU sharding requires >1 CUDA device");

        INDArray x = Nd4j.rand(DataType.FLOAT, 8, 64);

        SameDiff single = buildViewMlp(64, 128, 6, 16, 7L);
        INDArray ref = runOnce(single, x.dup(), true);

        SameDiff sharded = buildViewMlp(64, 128, 6, 16, 7L);

        boolean prevDsp  = InferenceSession.isDynamicShapePlanEnabled();
        String prevSingleGpu = System.getProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
        InferenceSession.setDynamicShapePlanEnabled(true);
        System.clearProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
        try {
            for (int i = 0; i < REPLAY_ITERATIONS; i++) {
                Map<String, INDArray> res = sharded.output(Collections.singletonMap("x", x.dup()), "out");
                INDArray got = res.get("out").dup();
                assertUsesEveryCudaDevice(sharded);

                assertArrayEquals(ref.shape(), got.shape(),
                        "sharded view-op output shape mismatch at iteration " + i);
                assertFalse(got.isNaN().any(),
                        "sharded view-op output has NaN at iteration " + i);
                double maxDiff = ref.sub(got).amaxNumber().doubleValue();
                log.info("testRepeatedShardedViewOpsMatchSingleGpu iteration {}: maxAbsDiff={}", i, maxDiff);
                assertTrue(got.equalsWithEps(ref, 1e-3),
                        "sharded view-op output diverged at iteration " + i
                                + " (maxAbsDiff=" + maxDiff + ")");
            }
        } finally {
            InferenceSession.setDynamicShapePlanEnabled(prevDsp);
            if (prevSingleGpu != null) System.setProperty(ND4JSystemProperties.DSP_SINGLE_GPU, prevSingleGpu);
            else System.clearProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
        }
    }

    /** Behavior 7: automatic placement must exercise a non-P2P view boundary correctly. */
    @Test
    public void testAutomaticShardedViewInputBoundary() {
        assumeTrue(Nd4j.getAffinityManager().getNumberOfDevices() > 1,
                "multi-GPU sharding requires >1 CUDA device");

        INDArray x = Nd4j.rand(DataType.FLOAT, 8, 64);
        SameDiff single = buildViewMlp(64, 128, 6, 16, 99L);
        SameDiff sharded = buildViewMlp(64, 128, 6, 16, 99L);
        INDArray ref = runOnce(single, x.dup(), true);
        INDArray got = runOnce(sharded, x.dup(), false);

        assertUsesEveryCudaDevice(sharded);
        assertArrayEquals(ref.shape(), got.shape(), "automatic sharded output shape mismatch");
        assertFalse(got.isNaN().any(), "automatic sharded output has NaN");
        double maxDiff = ref.sub(got).amaxNumber().doubleValue();
        assertTrue(got.equalsWithEps(ref, 1e-3),
                "automatic sharded view output diverged (maxAbsDiff=" + maxDiff + ")");
    }

    /**
     * Functional execution must retire migrated external weights, not pin a fresh
     * replica until plan teardown on every invocation. Each weight is 256 KiB,
     * larger than the 64 KiB steady-state allowance, so even one retained replica
     * per call cannot hide behind the allowance. No captured CUDA addresses are used.
     *
     * <p>CUDA pool used bytes are physical allocations (including deferred pinned
     * frees), unlike MemoryCounter. Reserved bytes include reusable cache capacity
     * and are logged, not asserted. Run this test in the serialized GPU lane.</p>
     */
    @Test
    public void testFunctionalShardedExternalWeightsPoolPlateaus() {
        assumeTrue(Nd4j.backends().isCudaAvailable(), "requires CUDA");
        int deviceCount = Nd4j.getAffinityManager().getNumberOfDevices();
        assumeTrue(deviceCount == 2, "bounded regression requires two CUDA devices");
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        assumeTrue(nativeOps.isMemoryPoolEnabled(), "physical retention check requires CUDA pools");

        final long allowance = 64L * 1024 * 1024;
        final long plateauSlack = 64L * 1024;
        final int warmupLast = 4;
        int originalDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        boolean originalDsp = InferenceSession.isDynamicShapePlanEnabled();
        String originalSingleGpu = System.getProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
        long[] originalLimits = new long[deviceCount];
        long[] limits = new long[deviceCount];
        long[] plateau = new long[deviceCount];
        for (int device = 0; device < deviceCount; device++) {
            originalLimits[device] = Nd4j.getEnvironment().getDeviceLimit(device);
            assumeTrue(poolAwareAvailableMemory(nativeOps, device) >= allowance,
                    "insufficient memory for 64 MiB allowance on device " + device);
        }

        SameDiff sharded = null;
        INDArray x = null;
        try {
            Nd4j.getAffinityManager().setDeviceForCurrentThread(0);
            InferenceSession.setDynamicShapePlanEnabled(true);
            System.clearProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
            for (int device = 0; device < deviceCount; device++) {
                limits[device] = Nd4j.getEnvironment().getDeviceCounter(device) + allowance;
                Nd4j.getEnvironment().setDeviceLimit(device, limits[device]);
            }
            sharded = SameDiff.create();
            sharded.setGraphExecutionMode(GraphExecutionMode.EMULATED_REPLAY);
            SDVariable h = sharded.placeHolder("x", DataType.FLOAT, 2, 256);
            // VARIABLE, not CONSTANT: these primary-device weights become
            // SOURCE_VARIABLE external inputs to secondary-device segments.
            for (int layer = 0; layer < 8; layer++) {
                SDVariable weight = sharded.var("migrationWeight" + layer,
                        Nd4j.create(DataType.FLOAT, 256, 256).assign(1.0 / 256));
                h = sharded.math().tanh(h.mmul(weight));
            }
            h.rename("out");
            x = Nd4j.create(DataType.FLOAT, 2, 256);
            for (int iteration = 0; iteration < REPLAY_ITERATIONS; iteration++) {
                // Changing values catches stale external-input publication while
                // retaining shapes and caller allocations for a clean pool sample.
                double expected = 0.125 + iteration * 0.03125;
                x.assign(expected);
                for (int layer = 0; layer < 8; layer++) expected = Math.tanh(expected);
                try (INDArray got = runOnce(sharded, x, false)) {
                    assertUsesEveryCudaDevice(sharded);
                    assertEquals(DataType.FLOAT, got.dataType());
                    assertArrayEquals(new long[]{2, 256}, got.shape());
                    // Bulk host read avoids allocating comparison temporaries in
                    // the device pool we are measuring.
                    for (float value : got.data().asFloat()) {
                        assertTrue(Float.isFinite(value), "non-finite output at " + iteration);
                        assertEquals(expected, value, 1e-5, "parity at iteration " + iteration);
                    }
                }
                assertEquals(0, Nd4j.getAffinityManager().getDeviceForCurrentThread(),
                        "functional sharding must restore caller device");
                if (iteration >= warmupLast) {
                    DspPlanAssertions.assertPhaseReached(sharded, PlanPhase.SHAPES_FROZEN,
                            "pool sample must exercise frozen input pinning");
                    DspPlanAssertions.assertAllCapturableSegmentsReachedPhase(
                            sharded, ExecutionPhase.REPLAYING, "functional sharded pool sample");
                    assertTrue(DspPlanAssertions.getTotalGraphReplays(sharded) > 0,
                            "must execute the published functional program");
                }
                Nd4j.getExecutioner().commit();
                for (int device = 0; device < deviceCount; device++) {
                    // Drains pending cudaFreeAsync streams before querying used
                    // bytes. Trimming cannot free a live graph-baked pinned block.
                    nativeOps.trimMemoryPool(device);
                    try (LongPointer used = new LongPointer(1);
                         LongPointer reserved = new LongPointer(1)) {
                        nativeOps.getMemoryPoolStats(device, used, reserved);
                        assertTrue(used.get() > 0, "pool stats unavailable on device " + device);
                        log.info("Functional migration iteration={} device={} poolUsed={} poolReserved={}",
                                iteration, device, used.get(), reserved.get());
                        if (iteration == warmupLast) plateau[device] = used.get();
                        if (iteration > warmupLast) {
                            assertTrue(used.get() <= plateau[device] + plateauSlack,
                                    "retained migration allocation on device " + device
                                            + " iteration=" + iteration + " baseline=" + plateau[device]
                                            + " used=" + used.get());
                        }
                    }
                    assertTrue(Nd4j.getEnvironment().getDeviceCounter(device) <= limits[device],
                            "device " + device + " exceeded bounded allowance");
                }
                assertEquals(0, nativeOps.lastErrorCode(), "native error at iteration " + iteration);
            }
            DspPlanAssertions.assertNoCaptureFailures(sharded, "functional migration regression");
        } finally {
            try {
                if (sharded != null) sharded.close();
            } finally {
                try {
                    SameDiffMemoryUtils.safeClose(x);
                    Nd4j.getExecutioner().commit();
                    for (int device = 0; device < deviceCount; device++) nativeOps.trimMemoryPool(device);
                } finally {
                    for (int device = 0; device < deviceCount; device++) {
                        Nd4j.getEnvironment().setDeviceLimit(device, originalLimits[device]);
                    }
                    InferenceSession.setDynamicShapePlanEnabled(originalDsp);
                    if (originalSingleGpu == null) System.clearProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
                    else System.setProperty(ND4JSystemProperties.DSP_SINGLE_GPU, originalSingleGpu);
                    Nd4j.getAffinityManager().setDeviceForCurrentThread(originalDevice);
                }
            }
        }
    }

    /**
     * Mutable migration storage must be bounded independently of Java GC. Force both
     * placeholders to be consumed on BOTH GPUs: Java copies device-1 inputs to the primary,
     * and native pre-replay can move those very DataBuffers back to the secondary.
     * Exercise FLOAT GDN-size (1 MiB) and HALF KV-size (640 KiB) inputs, A/B/A shape leases,
     * an offset/stepped input, and ordinary/direct output interleaving without changing caps.
     */
    @Test
    public void testMutableShardedInputReplicasAreFreshAndPoolPlateaus() {
        assumeTrue(Nd4j.backends().isCudaAvailable(), "requires CUDA");
        assumeTrue(Nd4j.getAffinityManager().getNumberOfDevices() == 2, "requires two real CUDA devices");
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        assumeTrue(nativeOps.isMemoryPoolEnabled(), "requires physical used-pool accounting");
        final int gdnWidth = 262144;
        final int kvWidth = 327680;
        final int warmupCalls = 18; // six executions of each shape/layout before measuring
        final long slack = 256L * 1024; // smaller than either single leaked migration buffer
        int originalDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        boolean originalDsp = InferenceSession.isDynamicShapePlanEnabled();
        String originalSingleGpu = System.getProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
        List<INDArray> callerRoots = new ArrayList<>();
        INDArray[] gdnInputs = new INDArray[3];
        INDArray[] kvInputs = new INDArray[3];
        SameDiff sd = null;
        try {
            InferenceSession.setDynamicShapePlanEnabled(true);
            System.clearProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
            Nd4j.getAffinityManager().setDeviceForCurrentThread(0);
            sd = SameDiff.create();
            SDVariable gdn = sd.placeHolder("mutableGdn", DataType.FLOAT, -1, gdnWidth);
            SDVariable kv = sd.placeHolder("mutableKv", DataType.HALF, -1, kvWidth);
            gdn.add("gdnPrimary", 1.0).add("gdnOut", gdn);
            kv.add("kvPrimary", 1.0).add("kvOut", kv);
            DynamicShapePlan plan = sd.compileDynamicShapePlan("gdnOut", "kvOut");
            plan.assignDevices(Map.of(0, 1L, 1, 1L));
            for (var slot : plan.getSlots()) {
                boolean primary = Arrays.asList(slot.getOutputVarNames()).contains("gdnPrimary")
                        || Arrays.asList(slot.getOutputVarNames()).contains("kvPrimary");
                slot.setTargetDeviceId(primary ? 0 : 1);
            }
            sd.compileNativeDynamicShapePlan("gdnOut", "kvOut");
            for (String input : new String[]{"mutableGdn", "mutableKv"}) {
                Set<Integer> consumers = new HashSet<>();
                for (var slot : plan.getSlots()) {
                    if (Arrays.asList(slot.getInputVarNames()).contains(input)) {
                        consumers.add(slot.getTargetDeviceId());
                    }
                }
                assertEquals(Set.of(0, 1), consumers, "must exercise primary migration and secondary use: " + input);
            }

            // Establish the execution device from primary-resident caller inputs, not a
            // private executor field or a forced execution mode. DSP advances normally.
            INDArray primeGdn = Nd4j.create(DataType.FLOAT, 1, gdnWidth).assign(0.0);
            INDArray primeKv = Nd4j.create(DataType.HALF, 1, kvWidth).assign(0.0);
            callerRoots.add(primeGdn);
            callerRoots.add(primeKv);
            Map<String, INDArray> primed = sd.output(Map.of("mutableGdn", primeGdn, "mutableKv", primeKv),
                    "gdnOut", "kvOut");
            for (INDArray value : primed.values()) SameDiffMemoryUtils.safeClose(value);

            Nd4j.getAffinityManager().setDeviceForCurrentThread(1);
            for (int scenario = 0; scenario < 3; scenario++) {
                int rows = scenario == 1 ? 2 : 1;
                if (scenario == 2) {
                    INDArray gdnRoot = Nd4j.create(DataType.FLOAT, 3, 2L * gdnWidth + 1).assign(-123.0);
                    INDArray kvRoot = Nd4j.create(DataType.HALF, 3, 2L * kvWidth + 1).assign(-123.0);
                    callerRoots.add(gdnRoot);
                    callerRoots.add(kvRoot);
                    gdnInputs[scenario] = gdnRoot.get(NDArrayIndex.interval(1, 2),
                            NDArrayIndex.interval(1, 2, 2L * gdnWidth + 1));
                    kvInputs[scenario] = kvRoot.get(NDArrayIndex.interval(1, 2),
                            NDArrayIndex.interval(1, 2, 2L * kvWidth + 1));
                    assertTrue(gdnInputs[scenario].isView());
                    assertTrue(gdnInputs[scenario].offset() > 0);
                } else {
                    char order = scenario == 1 ? 'f' : 'c';
                    gdnInputs[scenario] = Nd4j.createUninitialized(DataType.FLOAT,
                            new long[]{rows, gdnWidth}, order);
                    kvInputs[scenario] = Nd4j.createUninitialized(DataType.HALF,
                            new long[]{rows, kvWidth}, order);
                    callerRoots.add(gdnInputs[scenario]);
                    callerRoots.add(kvInputs[scenario]);
                }
                assertEquals(1, nativeOps.dbDeviceId(gdnInputs[scenario].data().opaqueBuffer()));
                assertEquals(1, nativeOps.dbDeviceId(kvInputs[scenario].data().opaqueBuffer()));
            }

            long[] plateau = new long[2];
            Map<Long, INDArray[]> ownersByLease = new LinkedHashMap<>();
            boolean sawSecondaryReplica = false;
            // Finish warmup with the largest Java readback cache, after every
            // native shape lease is resident. A smaller final cache understates
            // the steady-state peak when the two-row lease is revisited.
            int[] scenarioOrder = {0, 2, 1};
            for (int iteration = 0; iteration < 42; iteration++) {
                int scenario = scenarioOrder[(iteration / 6) % scenarioOrder.length];
                INDArray gdnInput = gdnInputs[scenario];
                INDArray kvInput = kvInputs[scenario];
                double value = 0.125 * (iteration + 1); // exactly representable in HALF
                gdnInput.assign(value);
                kvInput.assign(value + 0.5);
                gdnInput.putScalar(new long[]{gdnInput.size(0) - 1, gdnWidth - 1}, value + 0.25);
                kvInput.putScalar(new long[]{kvInput.size(0) - 1, kvWidth - 1}, value + 0.75);
                // The last producer is a device kernel on device 1. Do not read
                // the host or commit between it and DSP's migration/packing path.
                Nd4j.getAffinityManager().setDeviceForCurrentThread(1);
                gdnInput.addi(0.125);
                kvInput.addi(0.125);
                Map<String, INDArray> inputs = Map.of("mutableGdn", gdnInput, "mutableKv", kvInput);
                boolean direct = iteration % 4 != 0;
                Map<String, INDArray> result = direct ? sd.outputDirect(inputs, "gdnOut", "kvOut")
                        : sd.output(inputs, "gdnOut", "kvOut");
                try {
                    assertMutableMigrationValues(result.get("gdnOut"), gdnInput.shape(), DataType.FLOAT,
                            2 * value + 1.25, iteration);
                    assertMutableMigrationValues(result.get("kvOut"), kvInput.shape(), DataType.HALF,
                            2 * value + 2.25, iteration);
                } catch (AssertionError failure) {
                    try {
                        snapshotMutableKvFailure(sd, plan, kvInput, iteration);
                    } catch (Throwable diagnosticFailure) {
                        failure.addSuppressed(diagnosticFailure);
                    }
                    throw failure;
                }
                if (!direct) {
                    // output() owns independent copies; outputDirect() borrows its cached arrays.
                    for (INDArray output : result.values()) SameDiffMemoryUtils.safeClose(output);
                }
                DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
                INDArray[] installed = executor.getExternalInputsSnapshot();
                long handleAddress = executor.getNativePlanHandle().address();
                Map<Long, INDArray[]> retainedByHandle = retainedExternalInputsByPlanHandle(executor);
                INDArray[] retained = retainedByHandle.get(handleAddress);
                assertNotNull(retained, "Successful execution must publish its retained input owners");
                assertEquals(installed.length, retained.length);
                for (int index = 0; index < installed.length; index++) {
                    assertSame(installed[index], retained[index],
                            "Retained snapshot still names a pre-migration source: iteration=" + iteration
                                    + " input=" + index);
                }
                INDArray[] previous = ownersByLease.putIfAbsent(handleAddress, installed);
                for (String name : inputs.keySet()) {
                    int index = Arrays.asList(plan.getExternalInputKeys()).indexOf(name);
                    assertNotSame(inputs.get(name), installed[index], "caller buffer must not become the owned replica");
                    sawSecondaryReplica |= nativeOps.dbDeviceId(installed[index].data().opaqueBuffer()) == 1;
                    if (previous != null) assertSame(previous[index], installed[index],
                            "mutable replica changed for the same lease/input at iteration " + iteration);
                    assertFalse(inputs.get(name).wasClosed(), "caller array closed: " + name);
                    assertFalse(inputs.get(name).data().wasClosed(), "caller buffer closed: " + name);
                }
                if (iteration >= warmupCalls) {
                    DspPlanAssertions.assertPhaseReached(sd, PlanPhase.SHAPES_FROZEN, "mutable migration plateau");
                    assertTrue(DspPlanAssertions.getTotalGraphReplays(sd) > 0, "must exercise replay");
                }
                Nd4j.getExecutioner().commit();
                for (int device = 0; device < 2; device++) {
                    nativeOps.trimMemoryPool(device); // drain deferred frees; never GC
                    try (LongPointer used = new LongPointer(1); LongPointer reserved = new LongPointer(1)) {
                        nativeOps.getMemoryPoolStats(device, used, reserved);
                        assertTrue(used.get() > 0, "pool stats unavailable: " + device);
                        if (iteration < warmupCalls && iteration % 6 == 5) {
                            plateau[device] = Math.max(plateau[device], used.get());
                            log.info("Mutable migration warmup scenario={} device={} used={} reserved={}",
                                    scenario, device, used.get(), reserved.get());
                        }
                        if (iteration >= warmupCalls) assertTrue(used.get() <= plateau[device] + slack,
                                "mutable migration retention: iteration=" + iteration + " device=" + device
                                        + " used=" + used.get() + " warmupPeak=" + plateau[device]);
                    }
                }
                assertEquals(0, nativeOps.lastErrorCode(), "native failure at iteration " + iteration);
            }
            assertTrue(ownersByLease.size() >= 2, "must exercise multiple native shape leases");
            assertTrue(sawSecondaryReplica, "must exercise native relocation of a primary-allocated replica");
            assertEquals(-123.0f, callerRoots.get(callerRoots.size() - 2).getFloat(0, 0),
                    "GDN view copy overwrote the caller's sentinel");
            assertEquals(-123.0f, callerRoots.get(callerRoots.size() - 1).getFloat(0, 0),
                    "KV view copy overwrote the caller's sentinel");
            DspPlanAssertions.assertNoCaptureFailures(sd, "mutable migration");
            sd.close();
            sd = null;
            for (INDArray[] owned : ownersByLease.values()) {
                for (String name : new String[]{"mutableGdn", "mutableKv"}) {
                    int index = Arrays.asList(plan.getExternalInputKeys()).indexOf(name);
                    assertFalse(DynamicShapePlanExecutor.isArrayLive(owned[index]),
                            "teardown retained an owned migration replica: " + name);
                }
            }
            for (INDArray caller : callerRoots) {
                assertTrue(DynamicShapePlanExecutor.isArrayLive(caller), "teardown closed a caller allocation");
            }
            for (int scenario = 0; scenario < 3; scenario++) {
                assertTrue(DynamicShapePlanExecutor.isArrayLive(gdnInputs[scenario]), "closed caller alias");
                assertTrue(DynamicShapePlanExecutor.isArrayLive(kvInputs[scenario]), "closed caller alias");
            }
        } finally {
            try {
                if (sd != null) sd.close();
            } finally {
                for (INDArray root : callerRoots) SameDiffMemoryUtils.safeClose(root);
                InferenceSession.setDynamicShapePlanEnabled(originalDsp);
                if (originalSingleGpu == null) System.clearProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
                else System.setProperty(ND4JSystemProperties.DSP_SINGLE_GPU, originalSingleGpu);
                Nd4j.getAffinityManager().setDeviceForCurrentThread(originalDevice);
            }
        }
    }

    /** Eviction must return logical headroom BEFORE incoming allocation or execution. */
    @Test
    public void testMutableReplicaEvictionFreesBeforeAdmission() throws Exception {
        assumeTrue(Nd4j.backends().isCudaAvailable(), "requires CUDA");
        assumeTrue(Nd4j.getAffinityManager().getNumberOfDevices() == 2, "requires two real CUDA devices");
        final int width = 262144;
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int originalDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        boolean originalDsp = InferenceSession.isDynamicShapePlanEnabled();
        String originalSingle = System.getProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
        String budgetProperty = "nd4j.dsp.planLeaseBudgetFraction";
        String originalBudget = System.getProperty(budgetProperty);
        long[] originalLimits = {Nd4j.getEnvironment().getDeviceLimit(0),
                Nd4j.getEnvironment().getDeviceLimit(1)};
        List<INDArray> callers = new ArrayList<>();
        SameDiff sd = null;
        try {
            InferenceSession.setDynamicShapePlanEnabled(true);
            System.clearProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
            System.setProperty(budgetProperty, "0.000001");
            Nd4j.getAffinityManager().setDeviceForCurrentThread(0);
            sd = SameDiff.create();
            configureMutableReplicaGraph(sd, width);
            INDArray primeGdn = Nd4j.create(DataType.FLOAT, 1, width);
            INDArray primeKv = Nd4j.create(DataType.HALF, 1, width);
            callers.add(primeGdn);
            callers.add(primeKv);
            runMutableReplicaInputs(sd, primeGdn, primeKv, 0.0);

            Nd4j.getAffinityManager().setDeviceForCurrentThread(1);
            INDArray aGdn = Nd4j.create(DataType.FLOAT, 1, width);
            INDArray aKv = Nd4j.create(DataType.HALF, 1, width);
            INDArray bGdn = Nd4j.create(DataType.FLOAT, 2, width);
            INDArray bKv = Nd4j.create(DataType.HALF, 2, width);
            Collections.addAll(callers, aGdn, aKv, bGdn, bKv);
            for (INDArray input : new INDArray[]{aGdn, aKv, bGdn, bKv}) assertEquals(1,
                    nativeOps.dbDeviceId(input.data().opaqueBuffer()), "must start on the secondary device");
            for (int i = 0; i < 6; i++) runMutableReplicaInputs(sd, aGdn, aKv, i * 0.125);
            DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
            long aHandle = executor.getNativePlanHandle().address();
            List<INDArray> aReplicas = new ArrayList<>(mutableReplicaCaches(executor).get(aHandle).values());
            for (int i = 0; i < 6; i++) runMutableReplicaInputs(sd, bGdn, bKv, 1.0 + i * 0.125);
            long bHandle = executor.getNativePlanHandle().address();
            assertNotEquals(aHandle, bHandle);
            DspPlanAssertions.assertPhaseReached(sd, PlanPhase.SHAPES_FROZEN, "eviction victim must be frozen");
            assertTrue(DspPlanAssertions.getTotalGraphReplays(sd) > 0, "eviction must release replay borrowers");
            List<INDArray> victims = new ArrayList<>(mutableReplicaCaches(executor).get(bHandle).values());
            assertEquals(2, victims.size());

            // Dispatch A without rebinding: B is inactive, but cachedOpContext still has B's
            // raw input pointers. Eviction must detach this precise hazardous context too.
            redispatchForCurrentShapes(executor,
                    Map.of("mutableGdn", aGdn, "mutableKv", aKv), true);
            assertEquals(aHandle, executor.getNativePlanHandle().address());
            assertNotNull(executor.getCachedOpContext());
            long ownedBytes = victims.stream().mapToLong(a -> a.length() * a.dataType().width()).sum();
            INDArray admissionTemplate = victims.get(0);
            int admissionDevice = nativeOps.dbDeviceId(admissionTemplate.data().opaqueBuffer());
            DataType admissionType = admissionTemplate.dataType();
            long[] admissionShape = admissionTemplate.shape().clone();
            long admissionBytes = admissionTemplate.length() * admissionType.width();
            long before = logicalDeviceBytes();
            long[] tightLimits = new long[2];
            for (int device = 0; device < 2; device++) {
                tightLimits[device] = Nd4j.getEnvironment().getDeviceCounter(device) + 4096;
                Nd4j.getEnvironment().setDeviceLimit(device, tightLimits[device]);
            }
            assertTrue(admissionBytes > 4096, "incoming allocation must not fit before eviction");
            evictPinnedLeasesForCapacity(executor, Long.MIN_VALUE, -1);

            // No incoming execution, GC, pool trimming or extra commit between eviction and
            // these checks: queued retirement until a future successful call fails decisively.
            assertFalse(mutableReplicaCaches(executor).containsKey(bHandle));
            assertNull(executor.getCachedOpContext());
            assertTrue(before - logicalDeviceBytes() >= ownedBytes,
                    "eviction did not immediately return owned replica bytes to logical admission");
            for (INDArray victim : victims) assertFalse(DynamicShapePlanExecutor.isArrayLive(victim));
            for (INDArray active : aReplicas) assertTrue(DynamicShapePlanExecutor.isArrayLive(active),
                    "eviction closed an active graph-baked replica");
            for (INDArray caller : callers) assertTrue(DynamicShapePlanExecutor.isArrayLive(caller),
                    "eviction closed caller-owned storage");
            assertTrue(retiredMigrationArrays(executor).isEmpty());
            Map<?, Long> identities = pinnedPlanHandlesByIdentity(executor);
            Map<?, Long> costs = pinnedLeaseEstimatedBytes(executor);
            assertFalse(identities.containsValue(bHandle));
            assertEquals(identities.keySet(), costs.keySet(), "eviction left stale lease costs");

            Nd4j.getAffinityManager().setDeviceForCurrentThread(admissionDevice);
            try (INDArray admitted = Nd4j.createUninitialized(admissionType, admissionShape)) {
                admitted.assign(0.25);
                assertTrue(DynamicShapePlanExecutor.isArrayLive(admitted));
                for (int device = 0; device < 2; device++) assertTrue(
                        Nd4j.getEnvironment().getDeviceCounter(device) <= tightLimits[device]);
            }
            // A's captured addresses survived B's context detachment. Caps remain unchanged.
            runMutableReplicaInputs(sd, aGdn, aKv, 2.0);
        } finally {
            try {
                if (sd != null) sd.close();
            } finally {
                for (int device = 0; device < 2; device++) Nd4j.getEnvironment().setDeviceLimit(device, originalLimits[device]);
                for (INDArray caller : callers) SameDiffMemoryUtils.safeClose(caller);
                InferenceSession.setDynamicShapePlanEnabled(originalDsp);
                if (originalSingle == null) System.clearProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
                else System.setProperty(ND4JSystemProperties.DSP_SINGLE_GPU, originalSingle);
                if (originalBudget == null) System.clearProperty(budgetProperty);
                else System.setProperty(budgetProperty, originalBudget);
                Nd4j.getAffinityManager().setDeviceForCurrentThread(originalDevice);
            }
        }
    }

    /** Abort after real packed-view copies, before/after binding and during replay; retry in place. */
    @Test
    public void testMutableReplicaFailureRetryReleasesOwnedCopies() throws Exception {
        assumeTrue(Nd4j.backends().isCudaAvailable(), "requires CUDA");
        assumeTrue(Nd4j.getAffinityManager().getNumberOfDevices() == 2, "requires two real CUDA devices");
        final int width = 262144;
        int originalDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        boolean originalDsp = InferenceSession.isDynamicShapePlanEnabled();
        String originalSingle = System.getProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
        long[] originalLimits = {Nd4j.getEnvironment().getDeviceLimit(0),
                Nd4j.getEnvironment().getDeviceLimit(1)};
        List<INDArray> callers = new ArrayList<>();
        SameDiff sd = null;
        try {
            InferenceSession.setDynamicShapePlanEnabled(true);
            System.clearProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
            Nd4j.getAffinityManager().setDeviceForCurrentThread(0);
            sd = SameDiff.create();
            configureMutableReplicaGraph(sd, width);
            INDArray primeGdn = Nd4j.create(DataType.FLOAT, 1, width);
            INDArray primeKv = Nd4j.create(DataType.HALF, 1, width);
            Collections.addAll(callers, primeGdn, primeKv);
            for (int i = 0; i < 6; i++) {
                try {
                    runMutableReplicaInputs(sd, primeGdn, primeKv, 0.0);
                } catch (AssertionError failure) {
                    try {
                        snapshotMutableKvFailure(sd,
                                sd.getOrCreateSession().getDynamicShapePlanExecutor().getCurrentPlan(), primeKv, i);
                    } catch (Exception | AssertionError diagnosticFailure) {
                        failure.addSuppressed(diagnosticFailure);
                    }
                    throw failure;
                }
            }
            Nd4j.getAffinityManager().setDeviceForCurrentThread(1);
            INDArray gdnRoot = Nd4j.create(DataType.FLOAT, 3, 2L * width + 1).assign(-123.0);
            INDArray kvRoot = Nd4j.create(DataType.HALF, 3, 2L * width + 1).assign(-123.0);
            Collections.addAll(callers, gdnRoot, kvRoot);
            INDArray gdnView = gdnRoot.get(NDArrayIndex.interval(1, 2), NDArrayIndex.interval(1, 2, 2L * width + 1));
            INDArray kvView = kvRoot.get(NDArrayIndex.interval(1, 2), NDArrayIndex.interval(1, 2, 2L * width + 1));
            assertTrue(gdnView.isView() && gdnView.offset() > 0);
            gdnView.assign(0.5);
            kvView.assign(1.0);
            DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
            DynamicShapePlan plan = executor.getCurrentPlan();
            Map<String, INDArray> inputs = Map.of("mutableGdn", gdnView, "mutableKv", kvView);
            redispatchForCurrentShapes(executor, inputs, true);
            long stridedHandle = executor.getNativePlanHandle().address();
            for (int device = 0; device < 2; device++) Nd4j.getEnvironment().setDeviceLimit(device,
                    Nd4j.getEnvironment().getDeviceCounter(device) + 64L * 1024 * 1024);

            for (int attempt = 0; attempt < 4; attempt++) {
                boolean afterBinding = attempt == 2;
                boolean existingLease = attempt == 3;
                if (existingLease) {
                    // Match the injection boundary: outputDirect materializes these views,
                    // selecting a contiguous lease instead of the raw strided lease.
                    for (int i = 0; i < 6; i++) {
                        try {
                            runMutableReplicaInputs(executor, gdnView, kvView, 1.0 + i * 0.125);
                        } catch (AssertionError failure) {
                            try {
                                System.err.println("MUTABLE_FAILURE phase=existing-lease-warmup");
                                snapshotMutableKvFailure(sd, plan, kvView, i);
                            } catch (Exception | AssertionError diagnosticFailure) {
                                failure.addSuppressed(diagnosticFailure);
                            }
                            throw failure;
                        }
                        assertEquals(stridedHandle, executor.getNativePlanHandle().address(),
                                "warm-up must establish the same strided lease used by failure injection");
                    }
                    DspPlanAssertions.assertPhaseReached(sd, PlanPhase.SHAPES_FROZEN, "retry before replay failure");
                    assertTrue(DspPlanAssertions.getTotalGraphReplays(sd) > 0);
                }
                Map<String, INDArray> established = existingLease
                        ? new LinkedHashMap<>(mutableReplicaCaches(executor).get(executor.getNativePlanHandle().address()))
                        : Collections.emptyMap();
                RuntimeException injected = new IllegalStateException("abort after packed mutable migration " + attempt);
                List<INDArray> packed = retiredMigrationArrays(executor);
                List<INDArray> failedAllocations = new ArrayList<>();
                Map<String, INDArray> failing = new LinkedHashMap<>(inputs) {
                    @Override
                    public boolean containsKey(Object key) {
                        // The migration loop checks the next input only AFTER the first
                        // refresh has allocated, packed, and enqueued its real device copy.
                        if (!packed.isEmpty() && (!afterBinding
                                || migrationInputsBound(executor))) {
                            failedAllocations.addAll(packed);
                            Map<String, INDArray> owned = mutableReplicaCaches(executor)
                                    .get(executor.getNativePlanHandle().address());
                            assertNotNull(owned);
                            if (!existingLease) failedAllocations.addAll(owned.values());
                            throw injected;
                        }
                        return super.containsKey(key);
                    }
                };
                long[] beforeDevices = {Nd4j.getEnvironment().getDeviceCounter(0),
                        Nd4j.getEnvironment().getDeviceCounter(1)};
                long before = beforeDevices[0] + beforeDevices[1];
                System.err.println("MUTABLE_FAILURE_ACCOUNTING before attempt=" + attempt
                        + " devices=" + Arrays.toString(beforeDevices));
                RuntimeException failure = assertThrows(RuntimeException.class,
                        () -> executeNative(executor, plan, failing));
                assertSame(injected, failure, "cleanup replaced the original execution error");
                assertEquals(stridedHandle, executor.getNativePlanHandle().address(),
                        "failure injection must not switch shape leases");
                assertEquals(0, failure.getSuppressed().length, "cleanup itself failed");
                assertTrue(failedAllocations.size() >= (existingLease ? 1 : afterBinding ? 4 : 2),
                        "must exercise packed sources and new owned destinations at the requested failure boundary");
                for (INDArray owned : failedAllocations) assertFalse(DynamicShapePlanExecutor.isArrayLive(owned),
                        "failed call retained an owned copy until later execution/teardown");
                assertTrue(packed.isEmpty(), "packed source cleanup must finish before returning the failure");
                if (existingLease) {
                    Map<String, INDArray> current = mutableReplicaCaches(executor).get(stridedHandle);
                    assertNotNull(current, "failure must retain the established strided lease's replicas");
                    for (Map.Entry<String, INDArray> entry : established.entrySet()) {
                        assertSame(entry.getValue(), current.get(entry.getKey()));
                        assertTrue(DynamicShapePlanExecutor.isArrayLive(entry.getValue()),
                                "failure closed a previously successful graph-baked replica");
                    }
                } else {
                    assertFalse(mutableReplicaCaches(executor).containsKey(executor.getNativePlanHandle().address()));
                }
                long[] afterDevices = {Nd4j.getEnvironment().getDeviceCounter(0),
                        Nd4j.getEnvironment().getDeviceCounter(1)};
                long after = afterDevices[0] + afterDevices[1];
                System.err.println("MUTABLE_FAILURE_ACCOUNTING after attempt=" + attempt
                        + " devices=" + Arrays.toString(afterDevices) + " delta=" + (after - before));
                assertTrue(after <= before, "failure leaked logical device bytes: attempt=" + attempt
                        + " before=" + Arrays.toString(beforeDevices) + " after=" + Arrays.toString(afterDevices)
                        + " delta=" + (after - before));
                for (INDArray caller : callers) assertTrue(DynamicShapePlanExecutor.isArrayLive(caller));
                assertTrue(DynamicShapePlanExecutor.isArrayLive(gdnView));
                assertTrue(DynamicShapePlanExecutor.isArrayLive(kvView));
            }
            // Same executor, same strided lease, same caps: no reset/recompile/GC recovery.
            for (int i = 0; i < 6; i++) {
                try {
                    runMutableReplicaInputs(executor, gdnView, kvView, 1.0 + i * 0.125);
                } catch (AssertionError failure) {
                    try {
                        System.err.println("MUTABLE_FAILURE phase=post-injection-retry");
                        snapshotMutableKvFailure(sd, plan, kvView, i);
                    } catch (Exception | AssertionError diagnosticFailure) {
                        failure.addSuppressed(diagnosticFailure);
                    }
                    throw failure;
                }
                assertEquals(stridedHandle, executor.getNativePlanHandle().address(),
                        "retry must execute the failed lease, not a normalized replacement");
            }
            DspPlanAssertions.assertPhaseReached(sd, PlanPhase.SHAPES_FROZEN, "retry must reach frozen execution");
            assertTrue(DspPlanAssertions.getTotalGraphReplays(sd) > 0);
            DspPlanAssertions.assertNoCaptureFailures(sd, "mutable failure/retry");
            assertEquals(-123.0f, gdnRoot.getFloat(0, 0));
            assertEquals(-123.0f, kvRoot.getFloat(0, 0));
        } finally {
            try {
                if (sd != null) sd.close();
            } finally {
                for (int device = 0; device < 2; device++) Nd4j.getEnvironment().setDeviceLimit(device, originalLimits[device]);
                for (INDArray caller : callers) SameDiffMemoryUtils.safeClose(caller);
                InferenceSession.setDynamicShapePlanEnabled(originalDsp);
                if (originalSingle == null) System.clearProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
                else System.setProperty(ND4JSystemProperties.DSP_SINGLE_GPU, originalSingle);
                Nd4j.getAffinityManager().setDeviceForCurrentThread(originalDevice);
            }
        }
    }

    @Test
    public void testOutputReadbackFailureCompletesBeforeCrossThreadRelease() throws Exception {
        assumeTrue(Nd4j.backends().isCudaAvailable(), "requires CUDA");
        assumeTrue(Nd4j.getAffinityManager().getNumberOfDevices() == 2, "requires two real CUDA devices");
        int originalDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        boolean originalDsp = InferenceSession.isDynamicShapePlanEnabled();
        String originalSingle = System.getProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
        SameDiff sd = null;
        INDArray gdn = null;
        INDArray kv = null;
        var releaser = java.util.concurrent.Executors.newSingleThreadExecutor();
        try {
            InferenceSession.setDynamicShapePlanEnabled(true);
            System.clearProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
            Nd4j.getAffinityManager().setDeviceForCurrentThread(0);
            sd = SameDiff.create();
            configureOutputReadbackGraph(sd, 262144);
            gdn = Nd4j.create(DataType.FLOAT, 1, 262144);
            kv = Nd4j.create(DataType.HALF, 1, 262144);
            for (boolean fresh : new boolean[]{false, true}) {
                for (int i = 0; i < 6; i++) {
                    try {
                        runMutableReplicaInputs(sd, gdn, kv, 0.125 * i);
                    } catch (AssertionError failure) {
                        snapshotMutableKvFailure(sd, sd.getOrCreateSession()
                                .getDynamicShapePlanExecutor().getCurrentPlan(), kv, i);
                        throw failure;
                    }
                }
                DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
                assertTrue(mutableReplicaCaches(executor).isEmpty(),
                        "output-only readback fixture must not need mutable input replicas");
                if (fresh) closeZeroCopyOutputCache(executor);
                List<String> names = cachedRequestedOutputNames(executor);
                assertNotNull(names);
                List<INDArray> retired = retiredMigrationArrays(executor);
                List<INDArray> abandoned = new ArrayList<>();
                RuntimeException injected = new IllegalStateException("abort after first output readback fresh=" + fresh);
                setCachedRequestedOutputNames(executor, new java.util.AbstractList<String>() {
                    @Override public int size() { return names.size(); }
                    @Override public String get(int index) {
                        if (index == 1) {
                            assertSame(Thread.currentThread(), outputReadbackThread(executor),
                                    "first real copy must be pending at injection");
                            Set<Integer> pending = outputReadbackDevices(executor);
                            assertEquals(Set.of(0), pending);
                            Set<Integer> migrations = migrationCopyDevices(executor);
                            assertTrue(migrations.isEmpty(), "readback completion must not rely on input migration: " + migrations);
                            abandoned.addAll(retired);
                            throw injected;
                        }
                        return names.get(index);
                    }
                });
                Map<String, INDArray> inputs = Map.of("mutableGdn", gdn, "mutableKv", kv);
                try {
                    RuntimeException failure = assertThrows(RuntimeException.class, () -> executeNative(executor,
                            executor.getCurrentPlan(), inputs));
                    assertSame(injected, failure);
                    assertEquals(0, failure.getSuppressed().length);
                } finally {
                    setCachedRequestedOutputNames(executor, names);
                }
                assertNull(outputReadbackThread(executor), "completion precedes exception return");
                assertTrue(outputReadbackDevices(executor).isEmpty());
                assertTrue(retired.isEmpty());
                if (fresh) assertFalse(abandoned.isEmpty(), "fresh destination must have been retained");
                for (INDArray array : abandoned) assertFalse(DynamicShapePlanExecutor.isArrayLive(array));
                // No test-side commit, host read, GC or trim before release on a different LC/thread.
                releaser.submit(() -> {
                    assertNull(outputReadbackThread(executor));
                    executor.releaseGpuIntermediates();
                }).get(30, java.util.concurrent.TimeUnit.SECONDS);
                for (int i = 0; i < 6; i++) runMutableReplicaInputs(sd, gdn, kv, 1.0 + 0.125 * i);
                DspPlanAssertions.assertPhaseReached(sd, PlanPhase.SHAPES_FROZEN, "output readback failure retry");
                DspPlanAssertions.assertNoCaptureFailures(sd, "output readback failure retry");
            }
        } finally {
            releaser.shutdownNow();
            try {
                if (sd != null) sd.close();
            } finally {
                SameDiffMemoryUtils.safeClose(gdn);
                SameDiffMemoryUtils.safeClose(kv);
                InferenceSession.setDynamicShapePlanEnabled(originalDsp);
                if (originalSingle == null) System.clearProperty(ND4JSystemProperties.DSP_SINGLE_GPU);
                else System.setProperty(ND4JSystemProperties.DSP_SINGLE_GPU, originalSingle);
                Nd4j.getAffinityManager().setDeviceForCurrentThread(originalDevice);
            }
        }
    }

    /** Keep real two-device execution without Java mutable-input migration masking readback cleanup. */
    private static void configureOutputReadbackGraph(SameDiff sd, int width) {
        SDVariable gdn = sd.placeHolder("mutableGdn", DataType.FLOAT, -1, width);
        SDVariable kv = sd.placeHolder("mutableKv", DataType.HALF, -1, width);
        gdn.add("gdnPrimary", gdn).add("gdnOut", 1.0);
        kv.add("kvPrimary", kv).add("kvOut", 1.0);
        DynamicShapePlan plan = sd.compileDynamicShapePlan("gdnOut", "kvOut");
        plan.assignDevices(Map.of(0, 1L, 1, 1L));
        for (var slot : plan.getSlots()) {
            boolean consumesInput = Arrays.asList(slot.getOutputVarNames()).contains("gdnPrimary")
                    || Arrays.asList(slot.getOutputVarNames()).contains("kvPrimary");
            // External inputs have one consumer device; only intermediates cross back to GPU0.
            slot.setTargetDeviceId(consumesInput ? 1 : 0);
        }
        sd.compileNativeDynamicShapePlan("gdnOut", "kvOut");
    }

    private static void configureMutableReplicaGraph(SameDiff sd, int width) {
        SDVariable gdn = sd.placeHolder("mutableGdn", DataType.FLOAT, -1, width);
        SDVariable kv = sd.placeHolder("mutableKv", DataType.HALF, -1, width);
        gdn.add("gdnPrimary", 1.0).add("gdnOut", gdn);
        kv.add("kvPrimary", 1.0).add("kvOut", kv);
        DynamicShapePlan plan = sd.compileDynamicShapePlan("gdnOut", "kvOut");
        plan.assignDevices(Map.of(0, 1L, 1, 1L));
        for (var slot : plan.getSlots()) {
            boolean primary = Arrays.asList(slot.getOutputVarNames()).contains("gdnPrimary")
                    || Arrays.asList(slot.getOutputVarNames()).contains("kvPrimary");
            slot.setTargetDeviceId(primary ? 0 : 1);
        }
        sd.compileNativeDynamicShapePlan("gdnOut", "kvOut");
    }

    private static void runMutableReplicaInputs(SameDiff sd, INDArray gdn, INDArray kv, double value) {
        gdn.assign(value);
        kv.assign(value + 0.5);
        gdn.putScalar(new long[]{gdn.size(0) - 1, gdn.size(1) - 1}, value + 0.25);
        kv.putScalar(new long[]{kv.size(0) - 1, kv.size(1) - 1}, value + 0.75);
        Map<String, INDArray> outputs = sd.outputDirect(Map.of("mutableGdn", gdn, "mutableKv", kv), "gdnOut", "kvOut");
        assertMutableMigrationValues(outputs.get("gdnOut"), gdn.shape(), DataType.FLOAT, 2 * value + 1, 0);
        assertMutableMigrationValues(outputs.get("kvOut"), kv.shape(), DataType.HALF, 2 * value + 2, 0);
    }

    /** Exercise the executor's packed-view migration, without session-side materialization. */
    private static void runMutableReplicaInputs(DynamicShapePlanExecutor executor,
                                                INDArray gdn, INDArray kv, double value) {
        gdn.assign(value);
        kv.assign(value + 0.5);
        gdn.putScalar(new long[]{gdn.size(0) - 1, gdn.size(1) - 1}, value + 0.25);
        kv.putScalar(new long[]{kv.size(0) - 1, kv.size(1) - 1}, value + 0.75);
        Map<String, INDArray> outputs = executeNative(executor, executor.getCurrentPlan(),
                Map.of("mutableGdn", gdn, "mutableKv", kv));
        assertMutableMigrationValues(outputs.get("gdnOut"), gdn.shape(), DataType.FLOAT, 2 * value + 1, 0);
        assertMutableMigrationValues(outputs.get("kvOut"), kv.shape(), DataType.HALF, 2 * value + 2, 0);
    }

    private static long logicalDeviceBytes() {
        return Nd4j.getEnvironment().getDeviceCounter(0) + Nd4j.getEnvironment().getDeviceCounter(1);
    }

    /** Capture only after an assertion fails, so diagnostics cannot repair staging before execution. */
    private static void snapshotMutableKvFailure(SameDiff sd, DynamicShapePlan plan, INDArray caller, int iteration) {
        NativeOps ops = NativeOpsHolder.getInstance().getDeviceNativeOps();
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        Pointer handle = executor.getNativePlanHandle();
        int index = sd.dsp().extInputIndex("mutableKv");
        // This address was recorded during execute; querying it does not dereference
        // or migrate a native array. Do not dereference it after the call either.
        // Keep failure evidence visible even with the test runtime's NOP SLF4J binding.
        System.err.println("MUTABLE_FAILURE iteration=" + iteration + " extIndex=" + index
                + " execute-recorded-address=" + Long.toHexString(ops.getPlanLastExternalInputAddress(handle, index)));
        for (int slot = 0; slot < plan.getSlots().length; slot++) {
            String[] names = plan.getSlots()[slot].getOutputVarNames();
            if (Arrays.asList(names).contains("kvPrimary") || Arrays.asList(names).contains("kvOut")) {
                try (INDArray snapshot = sd.dsp().getSlotOutput(slot)) {
                    // getSlotOutput is a synchronizing readback, not a raw device probe.
                    logMutableSnapshot(iteration, "post-failure-readback slot=" + slot + " "
                            + Arrays.toString(names), snapshot);
                }
            }
        }
        // Do not inspect staging through specialBuffer(): both the opaque-array
        // getter and getPlanStagingBufferAddress can migrate the buffer. Likewise,
        // copyPlanStagingToBuffer refreshes it. None proves its execution-time contents.
        // Use native pre-replay tracing for that evidence, not these post-failure reads.
        logMutableSnapshot(iteration, "post-failure-owned-replica", executor.getExternalInputsSnapshot()[index]);
        logMutableSnapshot(iteration, "post-failure-caller", caller);
    }

    private static void logMutableSnapshot(int iteration, String label, INDArray snapshot) {
        if (snapshot == null) {
            System.err.println("MUTABLE_FAILURE iteration=" + iteration + " " + label + "=null");
            return;
        }
        if (snapshot.isView()) {
            // A stepped view's DataBuffer is the caller's entire root, including
            // sentinel gaps. Materialize only for this already-failed diagnostic.
            try (INDArray logicalCopy = snapshot.dup('c')) {
                logMutableSnapshot(iteration, label + " (logical-view-copy)", logicalCopy);
            }
            return;
        }
        float[] values = snapshot.data().asFloat();
        System.err.println("MUTABLE_FAILURE iteration=" + iteration + " " + label
                + " dtype=" + snapshot.dataType() + " shape=" + Arrays.toString(snapshot.shape())
                + " first=" + values[0] + " middle=" + values[values.length / 2]
                + " last=" + values[values.length - 1]);
    }

    private static void assertMutableMigrationValues(INDArray output, long[] shape, DataType dtype,
                                                     double expected, int iteration) {
        assertEquals(dtype, output.dataType());
        assertArrayEquals(shape, output.shape());
        float[] values = output.data().asFloat();
        assertEquals(output.length(), values.length);
        for (int index = 0; index < values.length; index++) {
            double expectedValue = expected + (index == values.length - 1 ? 0.5 : 0.0);
            if (values[index] != expectedValue) {
                // Avoid millions of per-element diagnostic strings/allocations (and GC)
                // in the successful pool-measurement path.
                assertEquals(expectedValue, values[index], 0.0,
                        "fresh mutable input at iteration " + iteration + " index=" + index);
            }
        }
    }

    /**
     * Behavior 8: automatic placement must use each device's remaining configured
     * allocation allowance, and later placement logic must preserve that weighted split.
     */
    @Test
    public void testAutomaticPlacementHonorsPerDeviceMemoryLimits() {
        int deviceCount = Nd4j.getAffinityManager().getNumberOfDevices();
        assumeTrue(deviceCount > 1, "per-device DSP placement requires >1 CUDA device");

        int originalDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int preferredDevice = 0;
        long smallestPhysicalAvailable = Long.MAX_VALUE;
        for (int device = 0; device < deviceCount; device++) {
            long physicalAvailable = poolAwareAvailableMemory(nativeOps, device);
            if (physicalAvailable < smallestPhysicalAvailable) {
                preferredDevice = device;
                smallestPhysicalAvailable = physicalAvailable;
            }
        }
        long mib = 1024L * 1024L;
        long preferredMultiplier = Math.max(8L, 2L * deviceCount);
        long smallAllowance = Math.min(64L * mib,
                smallestPhysicalAvailable / (2L * preferredMultiplier));
        assumeTrue(smallAllowance >= 16L * mib,
                "insufficient free device memory for bounded placement regression");
        long preferredAllowance = smallAllowance * preferredMultiplier;
        long[] originalLimits = new long[deviceCount];
        long[] configuredLimits = new long[deviceCount];
        for (int device = 0; device < deviceCount; device++) {
            originalLimits[device] = Nd4j.getEnvironment().getDeviceLimit(device);
        }

        SameDiff single = null;
        SameDiff sharded = null;
        INDArray x = null;
        INDArray singleInput = null;
        INDArray shardedInput = null;
        INDArray ref = null;
        INDArray got = null;
        try {
            Nd4j.getAffinityManager().setDeviceForCurrentThread(0);
            x = Nd4j.rand(DataType.FLOAT, 8, 64);
            single = buildParallelMlp(64, 16, 32, 20260905L);
            sharded = buildParallelMlp(64, 16, 32, 20260905L);

            singleInput = x.dup();
            ref = runOnce(single, singleInput, true);
            single.close();
            single = null;
            SameDiffMemoryUtils.safeClose(singleInput);
            singleInput = null;

            for (int device = 0; device < deviceCount; device++) {
                long allowance = device == preferredDevice ? preferredAllowance : smallAllowance;
                configuredLimits[device] = Nd4j.getEnvironment().getDeviceCounter(device) + allowance;
                Nd4j.getEnvironment().setDeviceLimit(device, configuredLimits[device]);
            }

            shardedInput = x.dup();
            got = runOnce(sharded, shardedInput, false);
            DynamicShapePlan plan = sharded.getOrCreateSession()
                    .getDynamicShapePlanExecutor().getCurrentPlan();
            assertNotNull(plan, "capped automatic run did not retain its DSP plan");
            assertUsesEveryCudaDevice(sharded);

            int preferredDeviceSlots = countAssignedSlots(plan, preferredDevice);
            log.info("Capped DSP placement: physically least-available device {} received {}MiB allowance "
                            + "versus {}MiB on peers; {}",
                    preferredDevice, preferredAllowance / mib, smallAllowance / mib,
                    plan.getDeviceAssignmentSummary());
            for (int device = 0; device < deviceCount; device++) {
                if (device != preferredDevice) {
                    assertTrue(preferredDeviceSlots > countAssignedSlots(plan, device),
                            "larger allowance must receive more slots than each smaller allowance: "
                                    + plan.getDeviceAssignmentSummary());
                }
                assertTrue(Nd4j.getEnvironment().getDeviceCounter(device) <= configuredLimits[device],
                        "device " + device + " exceeded configured allocation limit");
            }
            assertEquals(0, Nd4j.getAffinityManager().getDeviceForCurrentThread(),
                    "sharded execution must restore the caller's device");
            assertFalse(got.isNaN().any(), "capped sharded output has NaN");
            assertTrue(got.equalsWithEps(ref, 1e-3),
                    "capped sharded output must match the single-GPU reference");
            assertEquals(0, Nd4j.getNativeOps().lastErrorCode(),
                    "capped sharded execution left a native error");

            // A three-device ratio verifies cumulative apportionment without access to
            // three physical GPUs. A round-robin parallel-group rewrite changes these counts.
            Map<Integer, Long> weightedBudgets = new LinkedHashMap<>();
            weightedBudgets.put(0, 8L);
            weightedBudgets.put(1, 1L);
            weightedBudgets.put(2, 1L);
            plan.assignDevices(weightedBudgets);
            int expectedDeviceZeroSlots = (int) Math.round(0.8 * plan.getSlots().length);
            int expectedDeviceOneSlots = (int) Math.round(0.9 * plan.getSlots().length)
                    - expectedDeviceZeroSlots;
            assertEquals(expectedDeviceZeroSlots, countAssignedSlots(plan, 0),
                    "parallel-group placement must preserve the weighted device budget");
            assertEquals(expectedDeviceOneSlots, countAssignedSlots(plan, 1),
                    "second device must receive its cumulative weighted share");
            assertEquals(plan.getSlots().length - expectedDeviceZeroSlots - expectedDeviceOneSlots,
                    countAssignedSlots(plan, 2), "last device must receive only the final remainder");

            plan.assignDevices(Collections.singletonMap(preferredDevice, 1L));
            assertEquals(plan.getSlots().length, countAssignedSlots(plan, preferredDevice),
                    "a sole usable device must receive every slot");
            assertEquals(1, plan.getNumDistinctDevices(),
                    "singleton placement must report one distinct device");
        } finally {
            try {
                Nd4j.getExecutioner().commit();
                if (sharded != null) sharded.close();
                if (single != null) single.close();
                SameDiffMemoryUtils.safeClose(got);
                SameDiffMemoryUtils.safeClose(ref);
                SameDiffMemoryUtils.safeClose(shardedInput);
                SameDiffMemoryUtils.safeClose(singleInput);
                SameDiffMemoryUtils.safeClose(x);
            } finally {
                for (int device = 0; device < deviceCount; device++) {
                    Nd4j.getEnvironment().setDeviceLimit(device, originalLimits[device]);
                }
                Nd4j.getAffinityManager().setDeviceForCurrentThread(originalDevice);
            }
        }

        for (int device = 0; device < deviceCount; device++) {
            assertEquals(originalLimits[device], Nd4j.getEnvironment().getDeviceLimit(device),
                    "test must restore device " + device + " memory limit");
        }
        assertEquals(originalDevice, Nd4j.getAffinityManager().getDeviceForCurrentThread(),
                "test must restore the original caller device");
    }
}
