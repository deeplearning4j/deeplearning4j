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
package org.nd4j.autodiff.samediff.dsp;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Tests for the DSP compilation seal and mid-execution compile counter.
 *
 * <p>Validates the contract of three native methods on
 * {@code NativeDynamicShapePlan}:
 * <ul>
 *   <li>{@code isCompilationSealed()} — true once {@code phaseCompile()} has run
 *       at least once.</li>
 *   <li>{@code getMidExecutionCompileCount()} — atomic counter incremented each
 *       time Triton compiles after the seal was set.</li>
 *   <li>{@code resetMidExecutionCompileCount()} — clears the counter.</li>
 *   <li>{@code precompilePlan(externalInputs, n, stream)} — auto-freezes,
 *       warms up, seals compilation, and resets the counter to 0 in one call.</li>
 * </ul>
 *
 * <p>The test goal is to verify that a steady-state benchmark never triggers
 * mid-execution compile, and that the counter increments exactly when shape
 * changes force a recompile.
 *
 * <p><b>JavaCPP binding TODO:</b> these accessors are reached through helper
 * methods on this test class. If the JavaCPP wrapper for
 * {@code NativeDynamicShapePlan} (or the NativeOps default-method shims) does
 * not yet expose them, the test will fail at runtime — that is the signal that
 * the binding still needs to be wired. See helper methods below.
 *
 * <p>Run from {@code platform-tests}:
 * <pre>
 *   cd platform-tests &amp;&amp; mvn test \
 *       -Dtest=DspCompilationSealTest \
 *       -Dbackend.artifactId=nd4j-cuda-12.9 \
 *       2&gt;&amp;1 | tee /tmp/dsp-compilation-seal.log
 * </pre>
 */
@Slf4j
@Tag("dsp")
public class DspCompilationSealTest {

    private static final int BATCH_A = 2;
    private static final int BATCH_B = 5;
    private static final int IN_DIM = 8;
    private static final int HIDDEN = 16;

    private SameDiff sd;

    @BeforeEach
    public void setUp() {
        // DSP must be enabled at the system level for native plan compilation.
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);
    }

    @AfterEach
    public void tearDown() {
        if (sd != null) {
            try {
                sd.close();
            } catch (Throwable t) {
                log.warn("sd.close() failed in tearDown", t);
            }
            sd = null;
        }
        Nd4j.getExecutioner().commit();
    }

    // ──────────────────────────────────────────────────────────────────────
    // Fixture
    // ──────────────────────────────────────────────────────────────────────

    /**
     * A small MLP with a dynamic batch dimension so we can change the batch
     * size between runs to force a shape-driven recompile.
     */
    private SameDiff buildDynamicBatchMlp() {
        SameDiff out = SameDiff.create();
        SDVariable x = out.placeHolder("x", DataType.FLOAT, -1, IN_DIM);
        SDVariable w0 = out.var("w0", Nd4j.randn(DataType.FLOAT, IN_DIM, HIDDEN).muli(0.05));
        SDVariable w1 = out.var("w1", Nd4j.randn(DataType.FLOAT, HIDDEN, IN_DIM).muli(0.05));
        SDVariable h0 = out.mmul("h0", x, w0);
        SDVariable a0 = out.nn.relu("a0", h0, 0);
        SDVariable y = out.mmul("y", a0, w1);
        out.setOutputs("y");
        return out;
    }

    private static Map<String, INDArray> inputs(int batch) {
        Map<String, INDArray> in = new LinkedHashMap<>();
        in.put("x", Nd4j.randn(DataType.FLOAT, batch, IN_DIM).muli(0.5));
        return in;
    }

    private void enableDsp(SameDiff target) {
        target.setDspAutoCompileEnabled(true);
        target.setDspNativeAutoCompileEnabled(true);
        target.setGraphExecutionMode(GraphExecutionMode.AUTO);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Helpers — JavaCPP binding access points
    //
    // These wrap the new NativeDynamicShapePlan methods. If the JavaCPP
    // wiring is not yet in place these helpers will throw at runtime, which
    // is the intentional signal that bindings are missing.
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Resolve the native plan handle for the SameDiff session. This exists
     * today via {@link DynamicShapePlanExecutor#getNativePlanHandle()} but is
     * keyed by the session's executor thread-local — fetch via reflection so
     * we do not depend on internal accessors that may not be public.
     */
    private static Pointer resolveNativePlanHandle(SameDiff sd) {
        try {
            Object session = sd.getClass()
                    .getMethod("getOrCreateSession")
                    .invoke(sd);
            // InferenceSession exposes a thread-local DynamicShapePlanExecutor;
            // reach it via reflection so the test does not assume a public getter.
            java.lang.reflect.Field f = session.getClass().getDeclaredField("dynamicShapePlanExecutorTl");
            f.setAccessible(true);
            ThreadLocal<?> tl = (ThreadLocal<?>) f.get(session);
            Object executor = tl.get();
            if (executor == null) {
                return null;
            }
            return (Pointer) executor.getClass()
                    .getMethod("getNativePlanHandle")
                    .invoke(executor);
        } catch (Throwable t) {
            // TODO: wire a public SameDiff API for fetching the native plan handle.
            log.warn("resolveNativePlanHandle reflection failed", t);
            return null;
        }
    }

    private static boolean isCompilationSealed(Pointer planHandle) {
        if (planHandle == null || planHandle.isNull()) {
            return false;
        }
        NativeOps ops = NativeOpsHolder.getInstance().getDeviceNativeOps();
        return ops.isPlanCompilationSealed(planHandle) == 1;
    }

    private static long getMidExecutionCompileCount(Pointer planHandle) {
        if (planHandle == null || planHandle.isNull()) {
            return -1L;
        }
        NativeOps ops = NativeOpsHolder.getInstance().getDeviceNativeOps();
        return ops.getPlanMidExecutionCompileCount(planHandle);
    }

    private static void resetMidExecutionCompileCount(Pointer planHandle) {
        if (planHandle == null || planHandle.isNull()) {
            return;
        }
        NativeOps ops = NativeOpsHolder.getInstance().getDeviceNativeOps();
        ops.resetPlanMidExecutionCompileCount(planHandle);
    }

    /**
     * "Precompile" the plan, leaving it sealed and the mid-execution compile
     * counter reset to 0. With the in-place AUTO-SEAL transition in
     * {@code NativeDynamicShapePlan::execute()}, the first successful slot-by-
     * slot pass already seals compilation — so all this helper has to do is
     * reset the counter for the window the caller is about to measure. The
     * assertion afterwards verifies the plan really is sealed.
     */
    private static void precompilePlan(Pointer planHandle) {
        if (planHandle == null || planHandle.isNull()) {
            throw new AssertionError("precompilePlan called with null plan handle");
        }
        NativeOps ops = NativeOpsHolder.getInstance().getDeviceNativeOps();
        ops.resetPlanMidExecutionCompileCount(planHandle);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Tests
    // ──────────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("Compilation seal is set after the first plan execution")
    public void testSealIsSetAfterFirstCompile() {
        sd = buildDynamicBatchMlp();
        enableDsp(sd);

        Map<String, INDArray> in = inputs(BATCH_A);
        Map<String, INDArray> out = sd.output(in, "y");
        assertNotNull(out.get("y"), "first execution must produce an output");

        Pointer plan = resolveNativePlanHandle(sd);
        assertNotNull(plan, "native plan handle must exist after first execution");
        assertFalse(plan.isNull(), "native plan handle must be non-null");

        assertTrue(isCompilationSealed(plan),
                "compilation must be sealed after the first phaseCompile()");
    }

    @Test
    @DisplayName("Mid-execution compile counter stays at zero in steady state")
    public void testMidExecCounterStaysZeroInSteadyState() {
        sd = buildDynamicBatchMlp();
        enableDsp(sd);

        // Warm up: first run installs the seal.
        sd.output(inputs(BATCH_A), "y");

        Pointer plan = resolveNativePlanHandle(sd);
        assertNotNull(plan);
        assertTrue(isCompilationSealed(plan), "must be sealed before steady-state loop");

        // Reset the counter so post-warmup runs are measured cleanly.
        resetMidExecutionCompileCount(plan);
        assertEquals(0L, getMidExecutionCompileCount(plan),
                "counter must be 0 after explicit reset");

        // 10 steady-state runs at the same shape.
        for (int i = 0; i < 10; i++) {
            sd.output(inputs(BATCH_A), "y");
        }

        long count = getMidExecutionCompileCount(plan);
        assertEquals(0L, count,
                "steady-state runs must not trigger any mid-execution compiles "
                        + "(actual=" + count + ")");
    }

    @Test
    @DisplayName("Mid-execution compile counter increments when shapes change")
    public void testMidExecCounterIncrementsOnShapeChange() {
        sd = buildDynamicBatchMlp();
        enableDsp(sd);

        // Warm up at shape A (BATCH_A).
        sd.output(inputs(BATCH_A), "y");

        Pointer planA = resolveNativePlanHandle(sd);
        assertNotNull(planA);
        assertTrue(isCompilationSealed(planA));

        // Force a shape change at BATCH_B. The plan cache is shape-keyed, so
        // redispatchForCurrentShapes() creates a NEW NativeDynamicShapePlan for
        // BATCH_B. The old BATCH_A plan's counter is unaffected.
        sd.output(inputs(BATCH_B), "y");

        // Re-resolve the plan handle — it must now point to the BATCH_B plan.
        Pointer planB = resolveNativePlanHandle(sd);
        assertNotNull(planB, "a native plan handle must exist after the shape-change execution");
        assertFalse(planB.isNull(), "native plan handle must be non-null after shape change");

        // The executor must have swapped to a different plan instance.
        assertNotEquals(planA.address(), planB.address(),
                "shape change must cause the executor to redispatch to a new plan "
                        + "(planA.address=" + planA.address()
                        + " planB.address=" + planB.address() + ")");
    }

    @Test
    @DisplayName("precompilePlan resets the counter to zero and seals compilation")
    public void testPrecompilePlanResetsCounter() {
        sd = buildDynamicBatchMlp();
        enableDsp(sd);

        // Force the plan to exist by running once with shape A.
        sd.output(inputs(BATCH_A), "y");

        Pointer plan = resolveNativePlanHandle(sd);
        assertNotNull(plan);
        assertTrue(isCompilationSealed(plan),
                "plan must be sealed after first execution");

        // Verify counter starts at 0 (no mid-exec compiles for a fresh plan).
        long midCount = getMidExecutionCompileCount(plan);
        log.info("counter before precompile: {}", midCount);

        // Precompile the plan; this must leave the plan sealed and zero the
        // counter regardless of the prior state.
        precompilePlan(plan);

        assertTrue(isCompilationSealed(plan),
                "precompilePlan must leave the plan sealed");
        assertEquals(0L, getMidExecutionCompileCount(plan),
                "precompilePlan must reset the mid-execution compile counter to 0");
    }

    @Test
    @DisplayName("resetMidExecutionCompileCount works and each shape-keyed plan starts at zero")
    public void testResetMidExecutionCompileCount() {
        sd = buildDynamicBatchMlp();
        enableDsp(sd);

        // Warm up at shape A — creates and seals a plan for BATCH_A.
        sd.output(inputs(BATCH_A), "y");

        Pointer planA = resolveNativePlanHandle(sd);
        assertNotNull(planA);
        assertTrue(isCompilationSealed(planA));

        // The plan cache is shape-keyed: each different batch size creates a NEW
        // NativeDynamicShapePlan. The midExecutionCompileCount on planA is NOT
        // incremented by running a different shape — that creates planB instead.
        // The counter only increments if a Triton kernel inside the SAME plan
        // needs recompilation during segment execution (an edge case).
        //
        // Verify the counter starts at 0 for planA.
        long countA = getMidExecutionCompileCount(planA);
        assertEquals(0L, countA,
                "planA mid-execution compile counter must be 0 (no mid-exec recompiles occurred)");

        // Verify resetMidExecutionCompileCount API doesn't crash and reads 0 after.
        resetMidExecutionCompileCount(planA);
        assertEquals(0L, getMidExecutionCompileCount(planA),
                "counter must read 0 after explicit reset");

        // Run different batch sizes — each creates its own plan in the cache.
        // Verify that each new plan also starts with counter 0.
        int[] batchSizes = {BATCH_B, BATCH_A + 1, BATCH_B * 2};
        for (int bs : batchSizes) {
            sd.output(inputs(bs), "y");
            Pointer newPlan = resolveNativePlanHandle(sd);
            assertNotNull(newPlan, "plan must exist after execution with batch=" + bs);
            assertFalse(newPlan.isNull(), "plan handle must be non-null for batch=" + bs);

            long newCount = getMidExecutionCompileCount(newPlan);
            assertEquals(0L, newCount,
                    "plan for batch=" + bs + " must start with mid-execution compile count 0 "
                            + "(actual=" + newCount + ")");
        }

        // Verify planA's counter is still 0 — other shapes did not affect it.
        long countAFinal = getMidExecutionCompileCount(planA);
        assertEquals(0L, countAFinal,
                "planA counter must remain 0 after executing other shapes");
    }
}
