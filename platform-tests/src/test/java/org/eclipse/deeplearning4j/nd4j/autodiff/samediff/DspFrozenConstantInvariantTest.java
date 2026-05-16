/*
 *  ******************************************************************************
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
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Frozen constant invariant tests — verifies the single-pass slot classification
 * and buffer safety guarantees:
 *
 * <ol>
 *   <li><b>Frozen buffer validity</b> — every frozen slot must have a non-null,
 *       non-closed DataBuffer at freeze time and at every subsequent execution.
 *       The native code throws FROZEN_BUFFER_INVALID / FREEZE_NULL_OUTPUT if
 *       violated.</li>
 *   <li><b>Variable-input propagation</b> — ops that read from variable inputs
 *       (placeholders, recurrent state) must NOT be frozen, regardless of whether
 *       their warmup outputs happened to match.</li>
 *   <li><b>Data-dependent op stability</b> — data-dependent ops with ALL frozen
 *       constant inputs ARE correctly frozen (dependsOnExternal handles this).
 *       Data-dependent ops with variable inputs are NOT frozen.</li>
 *   <li><b>Output determinism</b> — frozen slots produce identical output across
 *       multiple executions. Non-frozen slots produce correct (not stale) output
 *       when inputs change.</li>
 * </ol>
 *
 * <p><b>Run:</b>
 * <pre>
 *   cd platform-tests && mvn test \
 *       -Dtest=DspFrozenConstantInvariantTest \
 *       2>&1 | tee /tmp/frozen-invariant.log
 * </pre>
 */
@Slf4j
@Tag("dsp")
@DisplayName("DSP frozen constant invariant tests")
public class DspFrozenConstantInvariantTest {

    private SameDiff sd;

    @BeforeEach
    public void setUp() {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);
    }

    @AfterEach
    public void tearDown() {
        if (sd != null) {
            try { sd.close(); } catch (Throwable t) { /* ignore */ }
            sd = null;
        }
    }

    // ── Test 1: Constant-only graph freezes everything ──────────────────────

    @ParameterizedTest(name = "constantOnlyGraph_{0}")
    @EnumSource(value = GraphExecutionMode.class,
                names = {"SLOT_BY_SLOT", "AUTO", "CUDA_GRAPHS"})
    public void testConstantOnlyGraphFreezesAll(GraphExecutionMode mode) {
        // Graph: constant W * constant B = output (all inputs are weights/constants)
        // Every op should be frozen after warmup.
        sd = SameDiff.create();
        SDVariable w = sd.constant("W", Nd4j.randn(DataType.FLOAT, 4, 4));
        SDVariable b = sd.constant("B", Nd4j.randn(DataType.FLOAT, 1, 4));
        SDVariable out = sd.math().add("output", sd.mmul("mm", w, w), b);

        sd.setGraphExecutionMode(mode);

        // Warmup: 3 executions to get past freeze detection
        Map<String, INDArray> inputs = new LinkedHashMap<>();
        INDArray result1 = null, result2 = null, result3 = null;
        for (int i = 0; i < 3; i++) {
            Map<String, INDArray> results = sd.output(inputs, "output");
            if (i == 0) result1 = results.get("output").dup();
            if (i == 1) result2 = results.get("output").dup();
            if (i == 2) result3 = results.get("output").dup();
        }

        // All executions must produce identical output (constant graph)
        assertNotNull(result1);
        assertNotNull(result2);
        assertNotNull(result3);
        assertEquals(result1, result2, "Constant graph: exec 1 != exec 2");
        assertEquals(result2, result3, "Constant graph: exec 2 != exec 3");
    }

    // ── Test 2: Variable inputs prevent freezing ────────────────────────────

    @ParameterizedTest(name = "variableInputNotFrozen_{0}")
    @EnumSource(value = GraphExecutionMode.class,
                names = {"SLOT_BY_SLOT", "AUTO", "CUDA_GRAPHS"})
    public void testVariableInputPreventsFreezing(GraphExecutionMode mode) {
        // Graph: placeholder X * constant W = output
        // The matmul depends on X (variable), so it must NOT be frozen.
        // Changing X must produce different output.
        sd = SameDiff.create();
        SDVariable x = sd.placeHolder("X", DataType.FLOAT, -1, 4);
        SDVariable w = sd.constant("W", Nd4j.eye(4).castTo(DataType.FLOAT));
        SDVariable out = sd.mmul("output", x, w);

        sd.setGraphExecutionMode(mode);

        // Execute with input A
        INDArray inputA = Nd4j.ones(DataType.FLOAT, 2, 4);
        Map<String, INDArray> mapA = new LinkedHashMap<>();
        mapA.put("X", inputA);

        // Warmup
        for (int i = 0; i < 3; i++) sd.output(mapA, "output");
        INDArray resultA = sd.output(mapA, "output").get("output").dup();

        // Execute with input B (different values)
        INDArray inputB = Nd4j.ones(DataType.FLOAT, 2, 4).mul(5.0);
        Map<String, INDArray> mapB = new LinkedHashMap<>();
        mapB.put("X", inputB);
        INDArray resultB = sd.output(mapB, "output").get("output").dup();

        // Results MUST differ — if they're equal, the matmul was incorrectly frozen
        assertNotEquals(resultA, resultB,
            "Variable-input op produced same output for different inputs — " +
            "it was incorrectly frozen. Mode=" + mode);

        // Verify correctness: X * I = X
        assertEquals(inputA, resultA, "output != inputA (identity matmul)");
        assertEquals(inputB, resultB, "output != inputB (identity matmul)");
    }

    // ── Test 3: Data-dependent op with constant inputs IS frozen ────────────

    @ParameterizedTest(name = "dataDepConstInputFrozen_{0}")
    @EnumSource(value = GraphExecutionMode.class,
                names = {"SLOT_BY_SLOT", "AUTO", "CUDA_GRAPHS"})
    public void testDataDepWithConstantInputsIsFrozen(GraphExecutionMode mode) {
        // Graph: concat of two constants along axis 0
        // Even though concat is data-dependent, all inputs are constants.
        // dependsOnExternal is false for all inputs → safe to freeze.
        // Output must be identical across all executions.
        sd = SameDiff.create();
        SDVariable a = sd.constant("A", Nd4j.ones(DataType.FLOAT, 2, 3));
        SDVariable b = sd.constant("B", Nd4j.ones(DataType.FLOAT, 2, 3).mul(2));
        SDVariable out = sd.concat("output", 0, a, b);

        sd.setGraphExecutionMode(mode);

        Map<String, INDArray> inputs = new LinkedHashMap<>();
        INDArray result1 = null, result2 = null;
        for (int i = 0; i < 4; i++) {
            Map<String, INDArray> results = sd.output(inputs, "output");
            if (i == 1) result1 = results.get("output").dup();
            if (i == 3) result2 = results.get("output").dup();
        }

        assertNotNull(result1);
        assertNotNull(result2);
        assertEquals(result1, result2,
            "Data-dep op with constant inputs should produce identical output across " +
            "all executions (should be frozen). Mode=" + mode);
    }

    // ── Test 4: Data-dependent op with variable inputs is NOT frozen ────────

    @ParameterizedTest(name = "dataDepVariableInputLive_{0}")
    @EnumSource(value = GraphExecutionMode.class,
                names = {"SLOT_BY_SLOT", "AUTO", "CUDA_GRAPHS"})
    public void testDataDepWithVariableInputIsLive(GraphExecutionMode mode) {
        // Graph: concat(placeholder X, constant B) along axis 0
        // X changes per step → concat output must change.
        sd = SameDiff.create();
        SDVariable x = sd.placeHolder("X", DataType.FLOAT, 2, 3);
        SDVariable b = sd.constant("B", Nd4j.ones(DataType.FLOAT, 2, 3).mul(2));
        SDVariable out = sd.concat("output", 0, x, b);

        sd.setGraphExecutionMode(mode);

        INDArray inputA = Nd4j.ones(DataType.FLOAT, 2, 3);
        Map<String, INDArray> mapA = new LinkedHashMap<>();
        mapA.put("X", inputA);

        // Warmup
        for (int i = 0; i < 3; i++) sd.output(mapA, "output");
        INDArray resultA = sd.output(mapA, "output").get("output").dup();

        // Change input
        INDArray inputB = Nd4j.ones(DataType.FLOAT, 2, 3).mul(99);
        Map<String, INDArray> mapB = new LinkedHashMap<>();
        mapB.put("X", inputB);
        INDArray resultB = sd.output(mapB, "output").get("output").dup();

        assertNotEquals(resultA, resultB,
            "Data-dep op with variable input produced same output for different inputs — " +
            "it was incorrectly frozen. Mode=" + mode);
    }

    // ── Test 5: Frozen output consistency across many executions ─────────────

    @Test
    @DisplayName("Frozen outputs remain bit-identical over 10 executions")
    public void testFrozenOutputBitIdentical() {
        // Constant-only graph: output must be bit-identical every execution.
        sd = SameDiff.create();
        SDVariable w = sd.constant("W", Nd4j.randn(DataType.FLOAT, 8, 8));
        SDVariable out = sd.mmul("output", w, w);

        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        Map<String, INDArray> inputs = new LinkedHashMap<>();
        INDArray reference = null;
        for (int i = 0; i < 10; i++) {
            Map<String, INDArray> results = sd.output(inputs, "output");
            INDArray current = results.get("output").dup();
            if (i == 0) {
                reference = current;
            } else {
                assertEquals(reference, current,
                    "Frozen output changed at execution " + i + " — buffer was corrupted");
            }
        }
    }

    // ── Test 6: Mixed graph — constants frozen, variables live ──────────────

    @ParameterizedTest(name = "mixedGraph_{0}")
    @EnumSource(value = GraphExecutionMode.class,
                names = {"SLOT_BY_SLOT", "AUTO", "CUDA_GRAPHS"})
    public void testMixedGraphCorrectness(GraphExecutionMode mode) {
        // Graph: output = (W * W) + X
        // W * W is constant → should be frozen.
        // + X depends on placeholder X → should be live.
        // Changing X must change output, but W * W component stays the same.
        sd = SameDiff.create();
        SDVariable w = sd.constant("W", Nd4j.eye(4).castTo(DataType.FLOAT).mul(2));
        SDVariable ww = sd.mmul("WW", w, w);
        SDVariable x = sd.placeHolder("X", DataType.FLOAT, 4, 4);
        SDVariable out = sd.math().add("output", ww, x);

        sd.setGraphExecutionMode(mode);

        INDArray inputA = Nd4j.zeros(DataType.FLOAT, 4, 4);
        Map<String, INDArray> mapA = new LinkedHashMap<>();
        mapA.put("X", inputA);

        // Warmup
        for (int i = 0; i < 3; i++) sd.output(mapA, "output");
        INDArray resultA = sd.output(mapA, "output").get("output").dup();

        // W * W = (2I)(2I) = 4I, so output = 4I + 0 = 4I
        INDArray expected4I = Nd4j.eye(4).castTo(DataType.FLOAT).mul(4);
        assertEquals(expected4I, resultA, "Mixed graph: output != 4I + 0 with mode=" + mode);

        // Change X to ones
        INDArray inputB = Nd4j.ones(DataType.FLOAT, 4, 4);
        Map<String, INDArray> mapB = new LinkedHashMap<>();
        mapB.put("X", inputB);
        INDArray resultB = sd.output(mapB, "output").get("output").dup();

        // output = 4I + ones
        INDArray expected4IplusOnes = expected4I.add(inputB);
        assertEquals(expected4IplusOnes, resultB, "Mixed graph: output != 4I + ones with mode=" + mode);

        assertNotEquals(resultA, resultB,
            "Mixed graph produced same output for different X — live ops were frozen. Mode=" + mode);
    }
}
