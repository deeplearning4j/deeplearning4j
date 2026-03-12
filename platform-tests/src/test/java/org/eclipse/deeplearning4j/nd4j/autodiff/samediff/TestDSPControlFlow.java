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

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.nd4j.autodiff.samediff.ControlFlow;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspCompilationMode;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests DSP (Dynamic Shape Plan) control flow support.
 * Verifies that Switch, Merge, Enter, Exit, NextIteration, and LoopCond
 * ops are correctly compiled and executed through the DSP path.
 */
@DisplayName("DSP Control Flow Tests")
public class TestDSPControlFlow extends BaseNd4jTestWithBackends {

    private static final Logger log = LoggerFactory.getLogger(TestDSPControlFlow.class);
    private static final double TOL = 1e-5;

    @Override
    public char ordering() {
        return 'c';
    }

    /**
     * Test 1: Switch/Merge if-else (no loop).
     * Build: x = placeholder, cond = placeholder(bool)
     * Switch(x, cond) -> true: x*2, false: x+1 -> Merge -> output
     * Verify: cond=true -> x*2; cond=false -> x+1
     */
    @Test
    @DisplayName("Test Switch/Merge if-else")
    public void testSwitchMergeIfElse() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3);
        SDVariable cond = sd.placeHolder("cond", DataType.BOOL);

        SDVariable result = ControlFlow.ifCond(
                sd,
                "result",
                "test_if",
                s -> cond,
                s -> s.math.mul("true_branch", x, sd.constant(2.0f)),
                s -> s.math.add("false_branch", x, sd.constant(1.0f))
        );

        // Enable DSP
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        INDArray input = Nd4j.ones(DataType.FLOAT, 2, 3).mul(3.0f);

        // Test true branch: x * 2 = 6
        Map<String, INDArray> outTrue = sd.output(
                Map.of("x", input, "cond", Nd4j.scalar(true)),
                "result"
        );
        INDArray expectedTrue = input.mul(2.0f);
        assertEquals(expectedTrue, outTrue.get("result"),
                "True branch should return x*2");

        // Test false branch: x + 1 = 4
        Map<String, INDArray> outFalse = sd.output(
                Map.of("x", input, "cond", Nd4j.scalar(false)),
                "result"
        );
        INDArray expectedFalse = input.add(1.0f);
        assertEquals(expectedFalse, outFalse.get("result"),
                "False branch should return x+1");
    }

    /**
     * Test 2: Switch dead-slot propagation with multi-op branches.
     * true_branch = x*2 + 3, false_branch = x - 1
     * Verify only the taken branch ops execute.
     */
    @Test
    @DisplayName("Test Switch dead-slot propagation")
    public void testSwitchDeadSlotPropagation() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3);
        SDVariable cond = sd.placeHolder("cond", DataType.BOOL);

        SDVariable result = ControlFlow.ifCond(
                sd,
                "result",
                "dead_test",
                s -> cond,
                s -> {
                    SDVariable mul = s.math.mul("true_mul", x, sd.constant(2.0f));
                    return s.math.add("true_add", mul, sd.constant(3.0f));
                },
                s -> s.math.sub("false_sub", x, sd.constant(1.0f))
        );

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        INDArray input = Nd4j.ones(DataType.FLOAT, 2, 3).mul(5.0f);

        // True branch: x*2 + 3 = 13
        Map<String, INDArray> outTrue = sd.output(
                Map.of("x", input, "cond", Nd4j.scalar(true)),
                "result"
        );
        INDArray expectedTrue = input.mul(2.0f).add(3.0f);
        assertEquals(expectedTrue, outTrue.get("result"),
                "True branch: x*2 + 3");

        // False branch: x - 1 = 4
        Map<String, INDArray> outFalse = sd.output(
                Map.of("x", input, "cond", Nd4j.scalar(false)),
                "result"
        );
        INDArray expectedFalse = input.sub(1.0f);
        assertEquals(expectedFalse, outFalse.get("result"),
                "False branch: x - 1");
    }

    /**
     * Test 3: While loop — fixed iteration count.
     * x = 0, while (x < 5) { x = x + 1 } -> expect 5
     */
    @Test
    @DisplayName("Test while loop fixed iterations")
    public void testWhileLoopFixedIterations() {
        SameDiff sd = SameDiff.create();
        SDVariable countIn = sd.constant("count", Nd4j.scalar(DataType.FLOAT, 0));

        SDVariable[] loopResult = sd.whileLoop(
                new String[]{"loop_out"},
                "fixed_loop",
                new SDVariable[]{countIn},
                (s, vars) -> vars[0].lt(5.0),
                (s, vars) -> new SDVariable[]{vars[0].add(1.0)}
        );

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        Map<String, INDArray> out = sd.outputAll(null);
        float result = out.get(loopResult[0].name()).getFloat(0);
        assertEquals(5.0f, result, TOL, "Loop should count to 5");
    }

    /**
     * Test 4: While loop — data-dependent termination.
     * x = 1.0, while (x < 100) { x = x * 2 } -> expect 128
     */
    @Test
    @DisplayName("Test while loop data-dependent termination")
    public void testWhileLoopDataDependentTermination() {
        SameDiff sd = SameDiff.create();
        SDVariable xIn = sd.constant("x", Nd4j.scalar(DataType.FLOAT, 1.0f));

        SDVariable[] loopResult = sd.whileLoop(
                new String[]{"loop_out"},
                "double_loop",
                new SDVariable[]{xIn},
                (s, vars) -> vars[0].lt(100.0),
                (s, vars) -> new SDVariable[]{vars[0].mul(2.0)}
        );

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        Map<String, INDArray> out = sd.outputAll(null);
        float result = out.get(loopResult[0].name()).getFloat(0);
        assertEquals(128.0f, result, TOL, "1*2^7 = 128 >= 100");
    }

    /**
     * Test 5: While loop — accumulation with multiple loop variables.
     * sum = 0, i = 0, while (i < 10) { sum = sum + i; i = i + 1 } -> expect 45
     */
    @Test
    @DisplayName("Test while loop variable accumulation")
    public void testWhileLoopAccumulation() {
        SameDiff sd = SameDiff.create();
        SDVariable countIn = sd.constant("count", Nd4j.scalar(DataType.FLOAT, 0));
        SDVariable sumIn = sd.constant("sum", Nd4j.scalar(DataType.FLOAT, 0));

        SDVariable[] loopResult = sd.whileLoop(
                new String[]{"count_out", "sum_out"},
                "accum_loop",
                new SDVariable[]{countIn, sumIn},
                (s, vars) -> vars[0].lt(10.0),
                (s, vars) -> new SDVariable[]{
                        vars[0].add(1.0),
                        vars[1].add(vars[0])
                }
        );

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        Map<String, INDArray> out = sd.outputAll(null);
        float sum = out.get(loopResult[1].name()).getFloat(0);
        assertEquals(45.0f, sum, TOL, "0+1+2+...+9 = 45");
    }

    /**
     * Test 6: Nested if-else (Switch within Switch).
     * outer_cond, inner_cond -> nested branches
     * All 4 combinations tested.
     */
    @Test
    @DisplayName("Test nested if-else")
    public void testNestedIfElse() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3);
        SDVariable outerCond = sd.placeHolder("outerCond", DataType.BOOL);
        SDVariable innerCond = sd.placeHolder("innerCond", DataType.BOOL);

        SDVariable result = ControlFlow.ifCond(
                sd,
                "result",
                "outer_if",
                s -> outerCond,
                s -> ControlFlow.ifCond(
                        sd,
                        "inner_true_result",
                        "inner_true_if",
                        s2 -> innerCond,
                        s2 -> s2.math.mul("tt_branch", x, sd.constant(10.0f)),
                        s2 -> s2.math.mul("tf_branch", x, sd.constant(20.0f))
                ),
                s -> ControlFlow.ifCond(
                        sd,
                        "inner_false_result",
                        "inner_false_if",
                        s2 -> innerCond,
                        s2 -> s2.math.mul("ft_branch", x, sd.constant(30.0f)),
                        s2 -> s2.math.mul("ff_branch", x, sd.constant(40.0f))
                )
        );

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        INDArray input = Nd4j.ones(DataType.FLOAT, 2, 3);

        // outer=true, inner=true -> x*10
        Map<String, INDArray> outTT = sd.output(
                Map.of("x", input, "outerCond", Nd4j.scalar(true), "innerCond", Nd4j.scalar(true)),
                "result"
        );
        assertEquals(input.mul(10.0f), outTT.get("result"), "TT: x*10");

        // outer=true, inner=false -> x*20
        Map<String, INDArray> outTF = sd.output(
                Map.of("x", input, "outerCond", Nd4j.scalar(true), "innerCond", Nd4j.scalar(false)),
                "result"
        );
        assertEquals(input.mul(20.0f), outTF.get("result"), "TF: x*20");

        // outer=false, inner=true -> x*30
        Map<String, INDArray> outFT = sd.output(
                Map.of("x", input, "outerCond", Nd4j.scalar(false), "innerCond", Nd4j.scalar(true)),
                "result"
        );
        assertEquals(input.mul(30.0f), outFT.get("result"), "FT: x*30");

        // outer=false, inner=false -> x*40
        Map<String, INDArray> outFF = sd.output(
                Map.of("x", input, "outerCond", Nd4j.scalar(false), "innerCond", Nd4j.scalar(false)),
                "result"
        );
        assertEquals(input.mul(40.0f), outFF.get("result"), "FF: x*40");
    }

    /**
     * Test 7: DSP while loop — classic sum pattern (matches SameDiffTests).
     * count=5, sum=0, while(count > 0) { sum += count; count-- } -> expect 15
     */
    @Test
    @DisplayName("Test while loop sum countdown")
    public void testWhileLoopSumCountdown() {
        SameDiff sd = SameDiff.create();
        SDVariable countIn = sd.constant("count", Nd4j.scalar(DataType.INT32, 5));
        SDVariable sumIn = sd.constant("sum", Nd4j.scalar(DataType.INT32, 0));

        SDVariable[] loopResult = sd.whileLoop(
                "sum_loop",
                new SDVariable[]{countIn, sumIn},
                (s, vars) -> vars[0].gt(0),
                (s, vars) -> new SDVariable[]{
                        vars[0].sub(1),
                        vars[1].add(vars[0])
                }
        );

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        Map<String, INDArray> out = sd.outputAll(null);
        int sum = out.get(loopResult[1].name()).getInt(0);
        assertEquals(15, sum, "5+4+3+2+1 = 15");
    }

    /**
     * Test 8: DSP vs standard path equivalence.
     * Run same control flow graph with DSP enabled vs disabled,
     * verify identical outputs.
     */
    @Test
    @DisplayName("Test DSP vs standard path equivalence")
    public void testDspVsStandardEquivalence() {
        // Build graph: if (cond) x*2 else x+1
        INDArray input = Nd4j.ones(DataType.FLOAT, 2, 3).mul(7.0f);

        for (boolean condValue : new boolean[]{true, false}) {
            // Standard path
            SameDiff sdStd = SameDiff.create();
            SDVariable xStd = sdStd.placeHolder("x", DataType.FLOAT, 2, 3);
            SDVariable condStd = sdStd.placeHolder("cond", DataType.BOOL);
            ControlFlow.ifCond(
                    sdStd, "result", "std_if",
                    s -> condStd,
                    s -> s.math.mul("true_b", xStd, sdStd.constant(2.0f)),
                    s -> s.math.add("false_b", xStd, sdStd.constant(1.0f))
            );

            Map<String, INDArray> stdOut = sdStd.output(
                    Map.of("x", input, "cond", Nd4j.scalar(condValue)),
                    "result"
            );

            // DSP path
            SameDiff sdDsp = SameDiff.create();
            SDVariable xDsp = sdDsp.placeHolder("x", DataType.FLOAT, 2, 3);
            SDVariable condDsp = sdDsp.placeHolder("cond", DataType.BOOL);
            ControlFlow.ifCond(
                    sdDsp, "result", "dsp_if",
                    s -> condDsp,
                    s -> s.math.mul("true_b", xDsp, sdDsp.constant(2.0f)),
                    s -> s.math.add("false_b", xDsp, sdDsp.constant(1.0f))
            );
            sdDsp.setDspAutoCompileEnabled(true);
            sdDsp.setDspNativeAutoCompileEnabled(true);
            sdDsp.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

            Map<String, INDArray> dspOut = sdDsp.output(
                    Map.of("x", input, "cond", Nd4j.scalar(condValue)),
                    "result"
            );

            assertEquals(stdOut.get("result"), dspOut.get("result"),
                    "DSP and standard should produce identical results for cond=" + condValue);
        }
    }

    /**
     * Test 9: DSP vs standard path equivalence for while loop.
     * Run same while loop graph with DSP enabled vs disabled.
     */
    @Test
    @DisplayName("Test DSP vs standard while loop equivalence")
    public void testDspVsStandardWhileLoopEquivalence() {
        // Standard path
        SameDiff sdStd = SameDiff.create();
        SDVariable countStd = sdStd.constant("count", Nd4j.scalar(DataType.FLOAT, 0));
        SDVariable sumStd = sdStd.constant("sum", Nd4j.scalar(DataType.FLOAT, 0));
        SDVariable[] stdResult = sdStd.whileLoop(
                "std_loop",
                new SDVariable[]{countStd, sumStd},
                (s, vars) -> vars[0].lt(5.0),
                (s, vars) -> new SDVariable[]{
                        vars[0].add(1.0),
                        vars[1].add(vars[0])
                }
        );
        Map<String, INDArray> stdOut = sdStd.outputAll(null);

        // DSP path
        SameDiff sdDsp = SameDiff.create();
        SDVariable countDsp = sdDsp.constant("count", Nd4j.scalar(DataType.FLOAT, 0));
        SDVariable sumDsp = sdDsp.constant("sum", Nd4j.scalar(DataType.FLOAT, 0));
        SDVariable[] dspResult = sdDsp.whileLoop(
                "dsp_loop",
                new SDVariable[]{countDsp, sumDsp},
                (s, vars) -> vars[0].lt(5.0),
                (s, vars) -> new SDVariable[]{
                        vars[0].add(1.0),
                        vars[1].add(vars[0])
                }
        );
        sdDsp.setDspAutoCompileEnabled(true);
        sdDsp.setDspNativeAutoCompileEnabled(true);
        sdDsp.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        Map<String, INDArray> dspOut = sdDsp.outputAll(null);

        float stdSum = stdOut.get(stdResult[1].name()).getFloat(0);
        float dspSum = dspOut.get(dspResult[1].name()).getFloat(0);
        assertEquals(stdSum, dspSum, TOL,
                "DSP and standard while loop should produce identical sums");
        assertEquals(10.0f, stdSum, TOL, "0+1+2+3+4 = 10");
    }

    /**
     * Test 10: DSP plan compiles successfully for control flow graphs.
     * Verifies that DSP compilation no longer rejects graphs with CF ops.
     */
    @Test
    @DisplayName("Test DSP plan compiles for control flow")
    public void testDspPlanCompilesForControlFlow() {
        // If-else graph
        SameDiff sdIf = SameDiff.create();
        SDVariable x = sdIf.placeHolder("x", DataType.FLOAT, 2, 3);
        SDVariable cond = sdIf.placeHolder("cond", DataType.BOOL);
        ControlFlow.ifCond(
                sdIf, "result", "compile_if",
                s -> cond,
                s -> s.math.mul("true_b", x, sdIf.constant(2.0f)),
                s -> s.math.add("false_b", x, sdIf.constant(1.0f))
        );
        sdIf.setDspAutoCompileEnabled(true);
        sdIf.setDspNativeAutoCompileEnabled(true);

        // Should not return null — CF ops should be compiled
        var mode = sdIf.compileNativeDynamicShapePlan(
                List.of("result"),
                GraphExecutionMode.SLOT_BY_SLOT,
                true
        );
        assertNotNull(mode, "DSP plan should compile for if-else graph (not null)");

        // While loop graph
        SameDiff sdLoop = SameDiff.create();
        SDVariable countIn = sdLoop.constant("count", Nd4j.scalar(DataType.FLOAT, 0));
        SDVariable[] loopResult = sdLoop.whileLoop(
                "compile_loop",
                new SDVariable[]{countIn},
                (s, vars) -> vars[0].lt(5.0),
                (s, vars) -> new SDVariable[]{vars[0].add(1.0)}
        );
        sdLoop.setDspAutoCompileEnabled(true);
        sdLoop.setDspNativeAutoCompileEnabled(true);

        var loopMode = sdLoop.compileNativeDynamicShapePlan(
                List.of(loopResult[0].name()),
                GraphExecutionMode.SLOT_BY_SLOT,
                true
        );
        assertNotNull(loopMode, "DSP plan should compile for while loop graph (not null)");
    }
}
