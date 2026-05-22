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
package org.eclipse.deeplearning4j.nd4j.autodiff.optimization;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for CommonSubexpressionElimination optimizer.
 * Verifies that duplicate ops with identical inputs and arguments are eliminated
 * while preserving numerical correctness.
 */
@Tag(TagNames.DL4J_OLD_API)
public class TestCommonSubexpressionElimination extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    @Override
    public long getTimeoutMilliseconds() {
        return 1_000_000_000L;
    }

    /**
     * Two identical unary ops (tanh(x) and tanh(x)) should be deduplicated.
     * The downstream add that consumes both should see the same canonical output.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDuplicateUnaryOpEliminated(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);

        // Two tanh ops on the same input
        SDVariable tanh1 = sd.math().tanh("tanh1", x);
        SDVariable tanh2 = sd.math().tanh("tanh2", x);
        SDVariable out = tanh1.add("out", tanh2);
        sd.setOutputs(Collections.singletonList("out"));

        // Run BEFORE optimization to get reference result
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 4);
        Map<String, INDArray> ph = Collections.singletonMap("x", input);
        INDArray refResult = sd.outputSingle(ph, "out");

        int opsBefore = sd.getOps().size();

        // Use only CSE (skip other passes that might interfere)
        List<org.nd4j.autodiff.samediff.optimize.OptimizerSet> cseOnly = Arrays.asList(
                new org.nd4j.autodiff.samediff.optimize.optimizations.CommonSubexpressionElimination()
        );
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("out"), cseOnly);
        int opsAfter = optimized.getOps().size();

        // CSE should eliminate one tanh
        assertTrue(opsAfter < opsBefore,
                "CSE should reduce op count. Before: " + opsBefore + ", After: " + opsAfter);

        // Verify numerical correctness
        INDArray optResult = optimized.outputSingle(ph, "out");
        assertEquals(refResult, optResult);
    }

    /**
     * Ops with different inputs should NOT be deduplicated even if the opName matches.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDifferentInputsNotEliminated(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, -1, 4);

        // Two tanh ops on DIFFERENT inputs — must NOT be deduplicated
        SDVariable tanhX = sd.math().tanh("tanhX", x);
        SDVariable tanhY = sd.math().tanh("tanhY", y);
        SDVariable out = tanhX.add("out", tanhY);
        sd.setOutputs(Collections.singletonList("out"));

        INDArray xArr = Nd4j.rand(DataType.FLOAT, 2, 4);
        INDArray yArr = Nd4j.rand(DataType.FLOAT, 2, 4);
        Map<String, INDArray> ph = Map.of("x", xArr, "y", yArr);
        INDArray refResult = sd.outputSingle(ph, "out");

        SameDiff optimized = GraphOptimizer.optimize(sd, "out");

        // Both tanh ops should survive — different inputs
        long tanhCount = optimized.getOps().values().stream()
                .filter(op -> op.getOp() != null && "tanh".equals(op.getOp().opName()))
                .count();
        assertEquals(2, tanhCount, "tanh ops on different inputs must both survive");

        INDArray optResult = optimized.outputSingle(ph, "out");
        assertEquals(refResult, optResult);
    }

    /**
     * Ops with the same inputs but different arguments (e.g., different reduce axes)
     * should NOT be deduplicated.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDifferentArgsNotEliminated(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);

        // Two reduce_mean on same input, different axes
        SDVariable mean0 = sd.math().mean("mean0", x, 0);  // shape [4]
        SDVariable mean1 = sd.math().mean("mean1", x, 1);  // shape [3]

        // Use both as separate outputs to avoid shape-mismatch issues
        sd.setOutputs(Arrays.asList("mean0", "mean1"));

        INDArray input = Nd4j.rand(DataType.FLOAT, 3, 4);
        Map<String, INDArray> ph = Collections.singletonMap("x", input);
        Map<String, INDArray> refResults = sd.output(ph, "mean0", "mean1");

        SameDiff optimized = GraphOptimizer.optimize(sd, "mean0", "mean1");

        // Correctness is the key check — different axes must produce different results
        Map<String, INDArray> optResults = optimized.output(ph, "mean0", "mean1");
        assertEquals(refResults.get("mean0"), optResults.get("mean0"));
        assertEquals(refResults.get("mean1"), optResults.get("mean1"));
    }

    /**
     * Duplicate binary ops (a + b and a + b again) should be deduplicated.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDuplicateBinaryOpEliminated(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, -1, 4);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, -1, 4);

        // Two identical additions: a + b
        SDVariable add1 = a.add("add1", b);
        SDVariable add2 = a.add("add2", b);
        SDVariable out = add1.mul("out", add2);
        sd.setOutputs(Collections.singletonList("out"));

        INDArray aArr = Nd4j.rand(DataType.FLOAT, 2, 4);
        INDArray bArr = Nd4j.rand(DataType.FLOAT, 2, 4);
        Map<String, INDArray> ph = Map.of("a", aArr, "b", bArr);
        INDArray refResult = sd.outputSingle(ph, "out");

        int opsBefore = sd.getOps().size();

        List<org.nd4j.autodiff.samediff.optimize.OptimizerSet> cseOnly = Arrays.asList(
                new org.nd4j.autodiff.samediff.optimize.optimizations.CommonSubexpressionElimination()
        );
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("out"), cseOnly);
        int opsAfter = optimized.getOps().size();

        assertTrue(opsAfter < opsBefore,
                "CSE should eliminate duplicate add. Before: " + opsBefore + ", After: " + opsAfter);

        INDArray optResult = optimized.outputSingle(ph, "out");
        assertEquals(refResult, optResult);
    }

    /**
     * When both duplicate ops are requested as outputs, the graph output
     * protection should prevent CSE from eliminating them.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGraphOutputsPreserved(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);

        // Two identical ops, both requested as outputs
        SDVariable tanh1 = sd.math().tanh("out1", x);
        SDVariable tanh2 = sd.math().tanh("out2", x);
        sd.setOutputs(Arrays.asList("out1", "out2"));

        SameDiff optimized = GraphOptimizer.optimize(sd, "out1", "out2");

        // Both outputs must be resolvable
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 4);
        Map<String, INDArray> ph = Collections.singletonMap("x", input);

        Map<String, INDArray> results = optimized.output(ph, "out1", "out2");
        assertNotNull(results.get("out1"), "out1 must be resolvable");
        assertNotNull(results.get("out2"), "out2 must be resolvable");
        assertEquals(results.get("out1"), results.get("out2"),
                "Both outputs should be numerically identical");
    }
}
