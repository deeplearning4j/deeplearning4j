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
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that verify control flow ops (if-else, while loops) work correctly
 * under CUDA graph capture/replay and other non-SLOT_BY_SLOT execution modes.
 *
 * <p>Each test builds the same control flow graph pattern as
 * {@link TestDSPControlFlow} but executes it under different execution modes
 * (CUDA_GRAPHS, EMULATED_REPLAY, AUTO) and verifies correctness against a
 * SLOT_BY_SLOT reference.</p>
 *
 * <p>Run:
 * <pre>
 *   cd platform-tests && mvn test -Dtest=DSPControlFlowCudaGraphsTest
 * </pre>
 */
@DisplayName("DSP Control Flow under CUDA Graphs and replay modes")
public class DSPControlFlowCudaGraphsTest extends BaseNd4jTestWithBackends {

    private static final Logger log = LoggerFactory.getLogger(DSPControlFlowCudaGraphsTest.class);
    private static final double TOL = 1e-5;

    @Override
    public char ordering() {
        return 'c';
    }

    // ─── Helpers ─────────────────────────────────────────────────────────────

    private void enableDsp(SameDiff sd, GraphExecutionMode mode) {
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(mode);
    }

    /**
     * Build an if-else graph: if (cond) x*2 else x+1.
     * Returns the result variable name.
     */
    private SameDiff buildIfElseGraph() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3);
        SDVariable cond = sd.placeHolder("cond", DataType.BOOL);

        ControlFlow.ifCond(
                sd,
                "result",
                "test_if",
                s -> cond,
                s -> s.math.mul("true_branch", x, sd.constant(2.0f)),
                s -> s.math.add("false_branch", x, sd.constant(1.0f))
        );
        return sd;
    }

    /**
     * Build a while-loop graph: x=0, while(x < 5) { x = x + 1 }.
     * Returns the SameDiff instance and the output variable name.
     */
    private SameDiff buildWhileLoopFixedIterations() {
        SameDiff sd = SameDiff.create();
        SDVariable countIn = sd.constant("count", Nd4j.scalar(DataType.FLOAT, 0));

        sd.whileLoop(
                new String[]{"loop_out"},
                "fixed_loop",
                new SDVariable[]{countIn},
                (s, vars) -> vars[0].lt(5.0),
                (s, vars) -> new SDVariable[]{vars[0].add(1.0)}
        );
        return sd;
    }

    /**
     * Build a while-loop accumulation graph:
     * sum=0, i=0, while(i < 10) { sum = sum + i; i = i + 1 }.
     */
    private SameDiff buildWhileLoopAccumulation() {
        SameDiff sd = SameDiff.create();
        SDVariable countIn = sd.constant("count", Nd4j.scalar(DataType.FLOAT, 0));
        SDVariable sumIn = sd.constant("sum", Nd4j.scalar(DataType.FLOAT, 0));

        sd.whileLoop(
                new String[]{"count_out", "sum_out"},
                "accum_loop",
                new SDVariable[]{countIn, sumIn},
                (s, vars) -> vars[0].lt(10.0),
                (s, vars) -> new SDVariable[]{
                        vars[0].add(1.0),
                        vars[1].add(vars[0])
                }
        );
        return sd;
    }

    // =========================================================================
    // Test 1: if-else under CUDA_GRAPHS
    // =========================================================================

    @Test
    @DisplayName("testIfElseUnderCudaGraphs — both branches produce correct output")
    public void testIfElseUnderCudaGraphs() {
        SameDiff sd = buildIfElseGraph();
        enableDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        INDArray input = Nd4j.ones(DataType.FLOAT, 2, 3).mul(3.0f);

        // True branch: x * 2 = 6
        Map<String, INDArray> outTrue = sd.output(
                Map.of("x", input, "cond", Nd4j.scalar(true)),
                "result"
        );
        INDArray expectedTrue = input.mul(2.0f);
        assertEquals(expectedTrue, outTrue.get("result"),
                "CUDA_GRAPHS true branch should return x*2");

        // False branch: x + 1 = 4
        Map<String, INDArray> outFalse = sd.output(
                Map.of("x", input, "cond", Nd4j.scalar(false)),
                "result"
        );
        INDArray expectedFalse = input.add(1.0f);
        assertEquals(expectedFalse, outFalse.get("result"),
                "CUDA_GRAPHS false branch should return x+1");

        log.info("PASS: testIfElseUnderCudaGraphs");
        sd.close();
    }

    // =========================================================================
    // Test 2: if-else under EMULATED_REPLAY
    // =========================================================================

    @Test
    @DisplayName("testIfElseUnderEmulatedReplay — both branches produce correct output")
    public void testIfElseUnderEmulatedReplay() {
        SameDiff sd = buildIfElseGraph();
        enableDsp(sd, GraphExecutionMode.EMULATED_REPLAY);

        INDArray input = Nd4j.ones(DataType.FLOAT, 2, 3).mul(3.0f);

        // True branch
        Map<String, INDArray> outTrue = sd.output(
                Map.of("x", input, "cond", Nd4j.scalar(true)),
                "result"
        );
        INDArray expectedTrue = input.mul(2.0f);
        assertEquals(expectedTrue, outTrue.get("result"),
                "EMULATED_REPLAY true branch should return x*2");

        // False branch
        Map<String, INDArray> outFalse = sd.output(
                Map.of("x", input, "cond", Nd4j.scalar(false)),
                "result"
        );
        INDArray expectedFalse = input.add(1.0f);
        assertEquals(expectedFalse, outFalse.get("result"),
                "EMULATED_REPLAY false branch should return x+1");

        log.info("PASS: testIfElseUnderEmulatedReplay");
        sd.close();
    }

    // =========================================================================
    // Test 3: if-else under AUTO mode
    // =========================================================================

    @Test
    @DisplayName("testIfElseUnderAutoMode — both branches produce correct output")
    public void testIfElseUnderAutoMode() {
        SameDiff sd = buildIfElseGraph();
        enableDsp(sd, GraphExecutionMode.AUTO);

        INDArray input = Nd4j.ones(DataType.FLOAT, 2, 3).mul(3.0f);

        // True branch
        Map<String, INDArray> outTrue = sd.output(
                Map.of("x", input, "cond", Nd4j.scalar(true)),
                "result"
        );
        INDArray expectedTrue = input.mul(2.0f);
        assertEquals(expectedTrue, outTrue.get("result"),
                "AUTO true branch should return x*2");

        // False branch
        Map<String, INDArray> outFalse = sd.output(
                Map.of("x", input, "cond", Nd4j.scalar(false)),
                "result"
        );
        INDArray expectedFalse = input.add(1.0f);
        assertEquals(expectedFalse, outFalse.get("result"),
                "AUTO false branch should return x+1");

        log.info("PASS: testIfElseUnderAutoMode");
        sd.close();
    }

    // =========================================================================
    // Test 4: while-loop (fixed iterations) under CUDA_GRAPHS
    // =========================================================================

    @Test
    @DisplayName("testWhileLoopUnderCudaGraphs — fixed iterations produce correct count")
    public void testWhileLoopUnderCudaGraphs() {
        SameDiff sd = buildWhileLoopFixedIterations();
        enableDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        Map<String, INDArray> out = sd.outputAll(null);
        float result = out.get("loop_out").getFloat(0);
        assertEquals(5.0f, result, TOL, "CUDA_GRAPHS loop should count to 5");

        log.info("PASS: testWhileLoopUnderCudaGraphs, result={}", result);
        sd.close();
    }

    // =========================================================================
    // Test 5: while-loop (fixed iterations) under EMULATED_REPLAY
    // =========================================================================

    @Test
    @DisplayName("testWhileLoopUnderEmulatedReplay — fixed iterations produce correct count")
    public void testWhileLoopUnderEmulatedReplay() {
        SameDiff sd = buildWhileLoopFixedIterations();
        enableDsp(sd, GraphExecutionMode.EMULATED_REPLAY);

        Map<String, INDArray> out = sd.outputAll(null);
        float result = out.get("loop_out").getFloat(0);
        assertEquals(5.0f, result, TOL, "EMULATED_REPLAY loop should count to 5");

        log.info("PASS: testWhileLoopUnderEmulatedReplay, result={}", result);
        sd.close();
    }

    // =========================================================================
    // Test 6: multi-step replay with changing condition inputs
    // =========================================================================

    @Test
    @DisplayName("testControlFlowMultiStepReplay — if-else over 10+ iterations with changing conditions")
    public void testControlFlowMultiStepReplay() {
        SameDiff sd = buildIfElseGraph();
        enableDsp(sd, GraphExecutionMode.CUDA_GRAPHS);

        INDArray baseInput = Nd4j.ones(DataType.FLOAT, 2, 3);
        int numSteps = 12;
        List<INDArray> outputs = new ArrayList<>();
        List<INDArray> expectedOutputs = new ArrayList<>();

        for (int step = 0; step < numSteps; step++) {
            // Alternate condition each step: true, false, true, false, ...
            boolean condValue = (step % 2 == 0);
            INDArray input = baseInput.mul(step + 1);
            Map<String, INDArray> result = sd.output(
                    Map.of("x", input, "cond", Nd4j.scalar(condValue)),
                    "result"
            );
            outputs.add(result.get("result").dup());

            INDArray expected = condValue ? input.mul(2.0f) : input.add(1.0f);
            expectedOutputs.add(expected);

            log.info("Step {}: cond={}, output[0]={}, expected[0]={}",
                    step, condValue,
                    result.get("result").getFloat(0, 0),
                    expected.getFloat(0, 0));
        }

        // Verify all steps produced correct output
        for (int step = 0; step < numSteps; step++) {
            double diff = outputs.get(step).sub(expectedOutputs.get(step)).norm2Number().doubleValue();
            double refNorm = expectedOutputs.get(step).norm2Number().doubleValue();
            double relErr = refNorm > 0 ? diff / refNorm : diff;
            assertTrue(relErr < TOL,
                    "Step " + step + " diverged from expected. relErr=" + relErr);
        }

        log.info("PASS: testControlFlowMultiStepReplay — all {} steps correct", numSteps);
        sd.close();
    }

    // =========================================================================
    // Test 7: mode equivalence — SLOT_BY_SLOT vs CUDA_GRAPHS vs EMULATED_REPLAY
    // =========================================================================

    @Test
    @DisplayName("testControlFlowModeEquivalence — all three modes produce identical output")
    public void testControlFlowModeEquivalence() {
        INDArray input = Nd4j.ones(DataType.FLOAT, 2, 3).mul(7.0f);

        // Run under SLOT_BY_SLOT (reference)
        // NOTE: Do NOT call sd.close() between instances — SameDiff.close() can
        // corrupt native constant pool entries, causing SIGABRT in subsequent instances.
        Map<String, INDArray> slotBySlotTrue;
        Map<String, INDArray> slotBySlotFalse;
        {
            SameDiff sd = buildIfElseGraph();
            sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
            slotBySlotTrue = sd.output(
                    Map.of("x", input, "cond", Nd4j.scalar(true)),
                    "result"
            );
            slotBySlotFalse = sd.output(
                    Map.of("x", input, "cond", Nd4j.scalar(false)),
                    "result"
            );
            // Dup outputs before they could be invalidated
            slotBySlotTrue = Map.of("result", slotBySlotTrue.get("result").dup());
            slotBySlotFalse = Map.of("result", slotBySlotFalse.get("result").dup());
        }

        // Run under CUDA_GRAPHS
        Map<String, INDArray> cudaGraphsTrue;
        Map<String, INDArray> cudaGraphsFalse;
        {
            SameDiff sd = buildIfElseGraph();
            enableDsp(sd, GraphExecutionMode.CUDA_GRAPHS);
            cudaGraphsTrue = sd.output(
                    Map.of("x", input, "cond", Nd4j.scalar(true)),
                    "result"
            );
            cudaGraphsFalse = sd.output(
                    Map.of("x", input, "cond", Nd4j.scalar(false)),
                    "result"
            );
            cudaGraphsTrue = Map.of("result", cudaGraphsTrue.get("result").dup());
            cudaGraphsFalse = Map.of("result", cudaGraphsFalse.get("result").dup());
        }

        // Run under EMULATED_REPLAY
        Map<String, INDArray> emulatedTrue;
        Map<String, INDArray> emulatedFalse;
        {
            SameDiff sd = buildIfElseGraph();
            enableDsp(sd, GraphExecutionMode.EMULATED_REPLAY);
            emulatedTrue = sd.output(
                    Map.of("x", input, "cond", Nd4j.scalar(true)),
                    "result"
            );
            emulatedFalse = sd.output(
                    Map.of("x", input, "cond", Nd4j.scalar(false)),
                    "result"
            );
        }

        // Compare true branch outputs
        double diffCudaTrue = cudaGraphsTrue.get("result").sub(slotBySlotTrue.get("result"))
                .norm2Number().doubleValue();
        double diffEmuTrue = emulatedTrue.get("result").sub(slotBySlotTrue.get("result"))
                .norm2Number().doubleValue();
        double normTrue = slotBySlotTrue.get("result").norm2Number().doubleValue();
        double relErrCudaTrue = normTrue > 0 ? diffCudaTrue / normTrue : diffCudaTrue;
        double relErrEmuTrue = normTrue > 0 ? diffEmuTrue / normTrue : diffEmuTrue;

        assertTrue(relErrCudaTrue < TOL,
                "CUDA_GRAPHS true branch diverged from SLOT_BY_SLOT. relErr=" + relErrCudaTrue);
        assertTrue(relErrEmuTrue < TOL,
                "EMULATED_REPLAY true branch diverged from SLOT_BY_SLOT. relErr=" + relErrEmuTrue);

        // Compare false branch outputs
        double diffCudaFalse = cudaGraphsFalse.get("result").sub(slotBySlotFalse.get("result"))
                .norm2Number().doubleValue();
        double diffEmuFalse = emulatedFalse.get("result").sub(slotBySlotFalse.get("result"))
                .norm2Number().doubleValue();
        double normFalse = slotBySlotFalse.get("result").norm2Number().doubleValue();
        double relErrCudaFalse = normFalse > 0 ? diffCudaFalse / normFalse : diffCudaFalse;
        double relErrEmuFalse = normFalse > 0 ? diffEmuFalse / normFalse : diffEmuFalse;

        assertTrue(relErrCudaFalse < TOL,
                "CUDA_GRAPHS false branch diverged from SLOT_BY_SLOT. relErr=" + relErrCudaFalse);
        assertTrue(relErrEmuFalse < TOL,
                "EMULATED_REPLAY false branch diverged from SLOT_BY_SLOT. relErr=" + relErrEmuFalse);

        log.info("PASS: testControlFlowModeEquivalence — all three modes match");
    }

    // =========================================================================
    // Test 8: while-loop accumulation under CUDA_GRAPHS and EMULATED_REPLAY
    // =========================================================================

    @Test
    @DisplayName("testWhileLoopAccumulation under multiple modes")
    public void testWhileLoopAccumulationModes() {
        float expectedSum = 45.0f; // 0+1+2+...+9 = 45

        // SLOT_BY_SLOT reference
        float slotBySlotSum;
        {
            SameDiff sd = buildWhileLoopAccumulation();
            sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
            Map<String, INDArray> out = sd.outputAll(null);
            slotBySlotSum = out.get("sum_out").getFloat(0);
            sd.close();
        }

        // CUDA_GRAPHS
        float cudaGraphsSum;
        {
            SameDiff sd = buildWhileLoopAccumulation();
            enableDsp(sd, GraphExecutionMode.CUDA_GRAPHS);
            Map<String, INDArray> out = sd.outputAll(null);
            cudaGraphsSum = out.get("sum_out").getFloat(0);
            sd.close();
        }

        // EMULATED_REPLAY
        float emulatedSum;
        {
            SameDiff sd = buildWhileLoopAccumulation();
            enableDsp(sd, GraphExecutionMode.EMULATED_REPLAY);
            Map<String, INDArray> out = sd.outputAll(null);
            emulatedSum = out.get("sum_out").getFloat(0);
            sd.close();
        }

        assertEquals(expectedSum, slotBySlotSum, TOL,
                "SLOT_BY_SLOT accumulation should be 45");
        assertEquals(slotBySlotSum, cudaGraphsSum, TOL,
                "CUDA_GRAPHS accumulation should match SLOT_BY_SLOT");
        assertEquals(slotBySlotSum, emulatedSum, TOL,
                "EMULATED_REPLAY accumulation should match SLOT_BY_SLOT");

        log.info("PASS: testWhileLoopAccumulationModes — slotBySlot={}, cudaGraphs={}, emulated={}",
                slotBySlotSum, cudaGraphsSum, emulatedSum);
    }
}
