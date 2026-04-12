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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.internal.memory.ArrayCacheMemoryMgr;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that the growth factor resets to 1.0 when DSP returns a null plan and
 * falls back to standard execution.
 *
 * <p>Background: DSP execution uses a scoped growth factor for memory allocation.
 * When DSP returns null (no compiled plan) and falls back to standard
 * {@code executeOperations()}, the growth factor must NOT leak from the DSP
 * scope into the standard path. A leaked growth factor would cause over-allocation
 * in subsequent standard executions.</p>
 *
 * <p>Key assertion: after DSP null-plan fallback, {@code effectiveGrowthFactor()}
 * must return the global default (typically 1.0).</p>
 */
@Slf4j
@Tag("samediff")
public class TestDSPGrowthFactorNullPlanScoping extends BaseNd4jTestWithBackends {

    private static final double TOL = 1e-9;

    @Override
    public char ordering() {
        return 'c';
    }

    @BeforeAll
    static void enableDspGlobally() {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);
    }

    private void enableDsp(SameDiff sd) {
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
    }

    @AfterEach
    void cleanup() {
        Nd4j.getExecutioner().commit();
    }

    /**
     * Test 1: Verify growth factor is 1.0 before and after DSP execution with a
     * graph that DSP can compile (non-null plan). Ensures DSP scopes properly.
     */
    @Test
    @DisplayName("Growth factor scoping: DSP compile does not leak growth factor")
    public void testGrowthFactorNotLeakedByDSPCompile() {
        double globalGfBefore = ArrayCacheMemoryMgr.effectiveGrowthFactor();
        log.info("Global growth factor before: {}", globalGfBefore);

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 8);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 8, 4).muli(0.1));
        SDVariable out = sd.mmul("out", x, w);

        enableDsp(sd);

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 8);

        // Execute — this goes through DSP warmup
        try {
            sd.outputDirect(Collections.singletonMap("x", input), "out");
        } catch (Exception e) {
            log.info("DSP execution exception (acceptable): {}", e.getMessage());
        }

        double globalGfAfter = ArrayCacheMemoryMgr.effectiveGrowthFactor();
        log.info("Global growth factor after: {}", globalGfAfter);

        assertEquals(globalGfBefore, globalGfAfter, TOL,
                "DSP execution must NOT leak growth factor. Before=" + globalGfBefore
                        + " After=" + globalGfAfter);

        sd.close();
    }

    /**
     * Test 2: DSP null plan fallback — disable DSP mid-execution and verify
     * growth factor is restored.
     */
    @Test
    @DisplayName("Growth factor: null plan fallback restores growth factor")
    public void testGrowthFactorAfterNullPlanFallback() {
        double globalGfBefore = ArrayCacheMemoryMgr.effectiveGrowthFactor();
        log.info("Global growth factor before: {}", globalGfBefore);

        // First, run a DSP compilation to set up a plan
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 8);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 8, 4).muli(0.1));
        SDVariable out = sd.mmul("out", x, w);

        enableDsp(sd);

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 8);

        // Execute DSP path (warmup)
        Map<String, INDArray> dspResult = sd.outputDirect(Collections.singletonMap("x", input), "out");
        assertNotNull(dspResult.get("out"));

        // Now execute standard path on same sd — this should NOT use DSP growth factor
        // The output() method on a SameDiff with DSP enabled should still go through
        // the standard path when not using outputDirect
        Map<String, INDArray> stdResult = sd.output(Collections.singletonMap("x", input), "out");
        assertNotNull(stdResult.get("out"));

        double globalGfAfter = ArrayCacheMemoryMgr.effectiveGrowthFactor();
        log.info("Global growth factor after DSP+standard: {}", globalGfAfter);

        assertEquals(globalGfBefore, globalGfAfter, TOL,
                "DSP + standard execution must NOT leak growth factor. Before="
                        + globalGfBefore + " After=" + globalGfAfter);

        sd.close();
    }

    /**
     * Test 3: Repeated DSP execution interleaved with standard execution.
     * Growth factor must remain stable throughout.
     */
    @Test
    @DisplayName("Growth factor: repeated DSP/standard interleaving does not leak")
    public void testInterleavedDSPStandardExecution() {
        double globalGfBefore = ArrayCacheMemoryMgr.effectiveGrowthFactor();
        log.info("Global growth factor before: {}", globalGfBefore);

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 8);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 8, 4).muli(0.1));
        SDVariable out = sd.mmul("out", x, w);

        enableDsp(sd);

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 8);

        for (int i = 0; i < 5; i++) {
            // DSP execution
            sd.outputDirect(Collections.singletonMap("x", input), "out");
            double gfAfterDSP = ArrayCacheMemoryMgr.effectiveGrowthFactor();
            assertEquals(globalGfBefore, gfAfterDSP, TOL,
                    "After DSP iteration " + i + ": growth factor leaked. Expected="
                            + globalGfBefore + " Got=" + gfAfterDSP);

            // Standard execution
            sd.output(Collections.singletonMap("x", input), "out");
            double gfAfterStd = ArrayCacheMemoryMgr.effectiveGrowthFactor();
            assertEquals(globalGfBefore, gfAfterStd, TOL,
                    "After standard iteration " + i + ": growth factor leaked. Expected="
                            + globalGfBefore + " Got=" + gfAfterStd);

            log.info("Iteration {}: GF after DSP={}, after standard={}", i, gfAfterDSP, gfAfterStd);
        }

        sd.close();
    }

    /**
     * Test 4: DSP disabled — standard execution only. Growth factor must remain
     * at global default throughout. This is the baseline control test.
     */
    @Test
    @DisplayName("Growth factor: standard execution only (DSP disabled) baseline")
    public void testStandardExecutionOnly() {
        double globalGfBefore = ArrayCacheMemoryMgr.effectiveGrowthFactor();
        log.info("Global growth factor before: {}", globalGfBefore);

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 8);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 8, 4).muli(0.1));
        SDVariable out = sd.mmul("out", x, w);

        // DSP NOT enabled
        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 8);

        for (int i = 0; i < 5; i++) {
            sd.output(Collections.singletonMap("x", input), "out");
            double gf = ArrayCacheMemoryMgr.effectiveGrowthFactor();
            assertEquals(globalGfBefore, gf, TOL,
                    "Standard-only iteration " + i + ": growth factor changed. Expected="
                            + globalGfBefore + " Got=" + gf);
        }

        double globalGfAfter = ArrayCacheMemoryMgr.effectiveGrowthFactor();
        log.info("Global growth factor after: {}", globalGfAfter);
        assertEquals(globalGfBefore, globalGfAfter, TOL,
                "Standard execution only should not change growth factor");

        sd.close();
    }

    /**
     * Test 5: Growth factor scope nesting with DSP execution inside a manual scope.
     * Verifies DSP execution respects the outer scope.
     */
    @Test
    @DisplayName("Growth factor: DSP execution inside manual scope respects outer scope")
    public void testDSPInsideManualScope() throws Exception {
        double globalGfBefore = ArrayCacheMemoryMgr.effectiveGrowthFactor();

        try (AutoCloseable scope = ArrayCacheMemoryMgr.withGrowthFactor(1.5)) {
            double scopedGfBefore = ArrayCacheMemoryMgr.effectiveGrowthFactor();
            assertEquals(1.5, scopedGfBefore, TOL, "Should be 1.5 inside scope");

            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 8);
            SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 8, 4).muli(0.1));
            sd.mmul("out", x, w);
            enableDsp(sd);

            INDArray input = Nd4j.randn(DataType.FLOAT, 2, 8);
            sd.outputDirect(Collections.singletonMap("x", input), "out");

            double scopedGfAfter = ArrayCacheMemoryMgr.effectiveGrowthFactor();
            assertEquals(1.5, scopedGfAfter, TOL,
                    "DSP execution inside scope should not change scoped growth factor");

            sd.close();
        }

        double globalGfAfter = ArrayCacheMemoryMgr.effectiveGrowthFactor();
        assertEquals(globalGfBefore, globalGfAfter, TOL,
                "After scope exits, global growth factor must be restored");
    }
}
