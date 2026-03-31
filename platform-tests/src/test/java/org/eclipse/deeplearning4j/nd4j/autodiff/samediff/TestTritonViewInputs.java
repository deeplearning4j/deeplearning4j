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
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that Triton-compiled kernels correctly handle non-contiguous view inputs.
 *
 * <p>When a view-producing op (permute, transpose) creates a non-contiguous view,
 * downstream Triton kernels must use the actual array strides, not assume C-contiguous
 * layout. This test class exercises each view op + Triton-compilable op combination
 * to identify which kernels silently produce wrong results.</p>
 *
 * <p>The pattern is:
 * <pre>
 *   input → matmul → VIEW_OP → TRITON_OP → output
 *                     ↑ non-contiguous    ↑ compiled by Triton
 * </pre>
 *
 * <p>Reference is taken from the first unfrozen execution (slot-by-slot, correct).
 * Frozen executions trigger Triton compilation; if strides are wrong, output diverges.</p>
 *
 * <p>Run:
 * <pre>
 *   cd platform-tests && mvn test -Dtest=TestTritonViewInputs \
 *       -Dnd4j.dsp.diagnostics=ALL -Dnd4j.dsp.diagnostics.level=full
 * </pre>
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestTritonViewInputs extends BaseNd4jTestWithBackends {

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

    /**
     * Run a frozen DSP test using the unfrozen warmup output as reference.
     * Warmup uses slot-by-slot execution (correct). Frozen triggers Triton.
     * If Triton mishandles strides, frozen output diverges from reference.
     */
    private void runFrozenTest(String graphName, SameDiff sd, INDArray input,
                               int frozenIters, double tolerance) {
        enableDsp(sd);

        // Warmup (unfrozen) — slot-by-slot, produces correct reference
        Map<String, INDArray> result = sd.output(
                Collections.singletonMap("input", input), "output");
        INDArray reference = result.get("output").dup();
        log.info("{} warmup: shape={} sum={}", graphName,
                java.util.Arrays.toString(reference.shape()),
                reference.sumNumber().doubleValue());

        // Freeze shapes — next executions trigger Triton compilation
        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor executor = session.getDynamicShapePlanExecutor();
        if (executor != null) {
            executor.setShapesFrozen(true);
            log.info("{}: shapes frozen", graphName);
        }

        // Frozen iterations — Triton kernels active
        for (int i = 0; i < frozenIters; i++) {
            result = sd.output(Collections.singletonMap("input", input), "output");
            INDArray out = result.get("output").dup();

            assertFalse(out.isNaN().any(),
                    graphName + " frozen call " + i + ": NaN detected");
            assertFalse(out.isInfinite().any(),
                    graphName + " frozen call " + i + ": Inf detected");

            double maxDiff = reference.sub(out).amaxNumber().doubleValue();
            double outSum = out.sumNumber().doubleValue();
            double refSum = reference.sumNumber().doubleValue();
            log.info("{} frozen call {}: maxDiff={} outSum={} refSum={}",
                    graphName, i, maxDiff, outSum, refSum);

            // Log first 4 values
            float[] outVals = out.dup('c').data().asFloat();
            float[] refVals = reference.dup('c').data().asFloat();
            int show = Math.min(4, outVals.length);
            log.info("  out[0:{}]=[{}, {}, {}, {}]", show,
                    show > 0 ? outVals[0] : "N/A",
                    show > 1 ? outVals[1] : "N/A",
                    show > 2 ? outVals[2] : "N/A",
                    show > 3 ? outVals[3] : "N/A");
            log.info("  ref[0:{}]=[{}, {}, {}, {}]", show,
                    show > 0 ? refVals[0] : "N/A",
                    show > 1 ? refVals[1] : "N/A",
                    show > 2 ? refVals[2] : "N/A",
                    show > 3 ? refVals[3] : "N/A");

            assertTrue(maxDiff < tolerance,
                    graphName + " frozen call " + i + ": maxDiff=" + maxDiff
                            + " exceeds tolerance=" + tolerance
                            + " (outSum=" + outSum + ", refSum=" + refSum + ")");
            out.close();
        }

        reference.close();
        sd.close();
    }

    // ─────────────────────────────────────────────────────────────────────
    // Test 1: permute(0,2,1) → add
    // ─────────────────────────────────────────────────────────────────────

    @Test
    @Order(1)
    @DisplayName("permute(0,2,1) → add: Triton must use actual strides")
    public void testPermuteToAdd() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4, 8);
        SDVariable w = sd.constant("weight", Nd4j.randn(DataType.FLOAT, 8, 8));
        SDVariable mm = sd.mmul("mm", input, w);
        SDVariable perm = sd.permute("perm", mm, 0, 2, 1);
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 8, 4).muli(0.1));
        perm.add("output", bias);

        runFrozenTest("permute→add", sd, Nd4j.randn(DataType.FLOAT, 2, 4, 8), 3, 1e-3);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Test 2: permute(0,2,1) → mul
    // ─────────────────────────────────────────────────────────────────────

    @Test
    @Order(2)
    @DisplayName("permute(0,2,1) → mul: Triton must use actual strides")
    public void testPermuteToMul() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4, 8);
        SDVariable w = sd.constant("weight", Nd4j.randn(DataType.FLOAT, 8, 8));
        SDVariable mm = sd.mmul("mm", input, w);
        SDVariable perm = sd.permute("perm", mm, 0, 2, 1);
        SDVariable scale = sd.constant("scale", Nd4j.randn(DataType.FLOAT, 1, 8, 4).muli(0.5).addi(1.0));
        perm.mul("output", scale);

        runFrozenTest("permute→mul", sd, Nd4j.randn(DataType.FLOAT, 2, 4, 8), 3, 1e-3);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Test 3: permute(0,2,1) → sigmoid (unary op through view)
    // ─────────────────────────────────────────────────────────────────────

    @Test
    @Order(3)
    @DisplayName("permute(0,2,1) → sigmoid: unary Triton with view input")
    public void testPermuteToSigmoid() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4, 8);
        SDVariable w = sd.constant("weight", Nd4j.randn(DataType.FLOAT, 8, 8));
        SDVariable mm = sd.mmul("mm", input, w);
        SDVariable perm = sd.permute("perm", mm, 0, 2, 1);
        sd.nn().sigmoid("output", perm);

        runFrozenTest("permute→sigmoid", sd, Nd4j.randn(DataType.FLOAT, 2, 4, 8), 3, 1e-3);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Test 4: permute(0,2,1) → relu
    // ─────────────────────────────────────────────────────────────────────

    @Test
    @Order(4)
    @DisplayName("permute(0,2,1) → relu: unary Triton with view input")
    public void testPermuteToRelu() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4, 8);
        SDVariable w = sd.constant("weight", Nd4j.randn(DataType.FLOAT, 8, 8));
        SDVariable mm = sd.mmul("mm", input, w);
        SDVariable perm = sd.permute("perm", mm, 0, 2, 1);
        sd.nn().relu("output", perm, 0.0);

        runFrozenTest("permute→relu", sd, Nd4j.randn(DataType.FLOAT, 2, 4, 8), 3, 1e-3);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Test 5: permute(0,2,1) → tanh
    // ─────────────────────────────────────────────────────────────────────

    @Test
    @Order(5)
    @DisplayName("permute(0,2,1) → tanh: unary Triton with view input")
    public void testPermuteToTanh() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4, 8);
        SDVariable w = sd.constant("weight", Nd4j.randn(DataType.FLOAT, 8, 8));
        SDVariable mm = sd.mmul("mm", input, w);
        SDVariable perm = sd.permute("perm", mm, 0, 2, 1);
        sd.math().tanh("output", perm);

        runFrozenTest("permute→tanh", sd, Nd4j.randn(DataType.FLOAT, 2, 4, 8), 3, 1e-3);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Test 6: permute(2,0,1) → add (different permutation)
    // ─────────────────────────────────────────────────────────────────────

    @Test
    @Order(6)
    @DisplayName("permute(2,0,1) → add: non-trivial permutation strides")
    public void testPermute201ToAdd() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4, 8);
        SDVariable w = sd.constant("weight", Nd4j.randn(DataType.FLOAT, 8, 8));
        SDVariable mm = sd.mmul("mm", input, w);
        SDVariable perm = sd.permute("perm", mm, 2, 0, 1);
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 1, 4).muli(0.1));
        perm.add("output", bias);

        runFrozenTest("permute(2,0,1)→add", sd, Nd4j.randn(DataType.FLOAT, 2, 4, 8), 3, 1e-3);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Test 7: permute → add + mul chain
    // ─────────────────────────────────────────────────────────────────────

    @Test
    @Order(7)
    @DisplayName("permute → add → mul: chained Triton ops after view")
    public void testPermuteToAddMulChain() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4, 8);
        SDVariable w = sd.constant("weight", Nd4j.randn(DataType.FLOAT, 8, 8));
        SDVariable mm = sd.mmul("mm", input, w);
        SDVariable perm = sd.permute("perm", mm, 0, 2, 1);
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 8, 4).muli(0.1));
        SDVariable added = perm.add("added", bias);
        SDVariable scale = sd.constant("scale", Nd4j.randn(DataType.FLOAT, 1, 8, 4).muli(0.5).addi(1.0));
        added.mul("output", scale);

        runFrozenTest("permute→add→mul", sd, Nd4j.randn(DataType.FLOAT, 2, 4, 8), 3, 1e-3);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Test 8: 4D permute(0,2,3,1) → add (attention-style)
    // ─────────────────────────────────────────────────────────────────────

    @Test
    @Order(8)
    @DisplayName("4D permute(0,2,3,1) → add: attention-style head rearrangement")
    public void testPermute4DToAdd() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 2, 4, 8);
        SDVariable w = sd.constant("weight", Nd4j.randn(DataType.FLOAT, 8, 8));
        SDVariable mm = sd.mmul("mm", input, w);
        SDVariable perm = sd.permute("perm", mm, 0, 2, 3, 1);
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 1, 8, 2).muli(0.1));
        perm.add("output", bias);

        runFrozenTest("4D permute(0,2,3,1)→add", sd,
                Nd4j.randn(DataType.FLOAT, 1, 2, 4, 8), 3, 1e-3);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Test 9: permute → reduce_sum
    // ─────────────────────────────────────────────────────────────────────

    @Test
    @Order(9)
    @DisplayName("permute → reduce_sum: reduction Triton kernel with view input")
    public void testPermuteToReduceSum() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4, 8);
        SDVariable w = sd.constant("weight", Nd4j.randn(DataType.FLOAT, 8, 8));
        SDVariable mm = sd.mmul("mm", input, w);
        SDVariable perm = sd.permute("perm", mm, 0, 2, 1);
        perm.sum("output", 2);

        runFrozenTest("permute→reduce_sum", sd, Nd4j.randn(DataType.FLOAT, 2, 4, 8), 3, 1e-3);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Test 10: permute → sub
    // ─────────────────────────────────────────────────────────────────────

    @Test
    @Order(10)
    @DisplayName("permute → sub: binary subtraction with view input")
    public void testPermuteToSub() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4, 8);
        SDVariable w = sd.constant("weight", Nd4j.randn(DataType.FLOAT, 8, 8));
        SDVariable mm = sd.mmul("mm", input, w);
        SDVariable perm = sd.permute("perm", mm, 0, 2, 1);
        SDVariable offset = sd.constant("offset", Nd4j.randn(DataType.FLOAT, 1, 8, 4).muli(0.5));
        perm.sub("output", offset);

        runFrozenTest("permute→sub", sd, Nd4j.randn(DataType.FLOAT, 2, 4, 8), 3, 1e-3);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Test 11: permute → div
    // ─────────────────────────────────────────────────────────────────────

    @Test
    @Order(11)
    @DisplayName("permute → div: binary division with view input")
    public void testPermuteToDiv() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4, 8);
        SDVariable w = sd.constant("weight", Nd4j.randn(DataType.FLOAT, 8, 8));
        SDVariable mm = sd.mmul("mm", input, w);
        SDVariable perm = sd.permute("perm", mm, 0, 2, 1);
        SDVariable divisor = sd.constant("divisor", Nd4j.randn(DataType.FLOAT, 1, 8, 4).muli(0.5).addi(2.0));
        perm.div("output", divisor);

        runFrozenTest("permute→div", sd, Nd4j.randn(DataType.FLOAT, 2, 4, 8), 3, 1e-3);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Test 12: No view — sanity check that Triton works with contiguous
    // ─────────────────────────────────────────────────────────────────────

    @Test
    @Order(12)
    @DisplayName("No view (contiguous) → add: sanity check")
    public void testContiguousToAdd() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4, 8);
        SDVariable w = sd.constant("weight", Nd4j.randn(DataType.FLOAT, 8, 8));
        SDVariable mm = sd.mmul("mm", input, w);
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 4, 8).muli(0.1));
        mm.add("output", bias);

        runFrozenTest("contiguous→add", sd, Nd4j.randn(DataType.FLOAT, 2, 4, 8), 3, 1e-3);
    }
}
