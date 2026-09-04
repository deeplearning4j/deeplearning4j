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
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

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
}
