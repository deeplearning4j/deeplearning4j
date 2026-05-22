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
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.Mmul;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for horizontal fusion: batching multiple independent ops of the same type.
 *
 * Primary pattern: Parallel Matmul Fusion
 * When multiple matmuls share the same activation input with different constant weights,
 * they are fused into a single matmul with concatenated weights then split.
 */
@Tag(TagNames.SAMEDIFF)
public class TestHorizontalFusion {

    /**
     * Two matmuls sharing the same input with different constant weights.
     * matmul(X, W1) and matmul(X, W2) → matmul(X, concat(W1, W2)) + split
     */
    @Test
    void testTwoParallelMatmuls() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 4, 3));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 4, 5));
        SDVariable out1 = sd.mmul("out1", x, w1);
        SDVariable out2 = sd.mmul("out2", x, w2);
        SDVariable result = out1.add("result", out2.sum(1).reshape(-1, 1).mul(sd.constant(Nd4j.ones(DataType.FLOAT, 1, 3))));
        sd.setOutputs("out1", "out2");

        // Run unoptimized to get reference values
        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 4);
        Map<String, INDArray> refOutputs = sd.output(Map.of("x", input), "out1", "out2");
        INDArray refOut1 = refOutputs.get("out1").dup();
        INDArray refOut2 = refOutputs.get("out2").dup();

        // Count matmul ops before optimization
        int mmulCountBefore = countOpsOfType(sd, Mmul.class);
        assertEquals(2, mmulCountBefore, "Should have 2 matmuls before optimization");

        // Optimize
        SameDiff optimized = GraphOptimizer.optimize(sd, "out1", "out2");

        // Count matmul ops after optimization — should be 1 (fused)
        int mmulCountAfter = countOpsOfType(optimized, Mmul.class);
        assertEquals(1, mmulCountAfter, "Should have 1 fused matmul after optimization");

        // Verify results match
        Map<String, INDArray> optOutputs = optimized.output(Map.of("x", input), "out1", "out2");
        INDArray optOut1 = optOutputs.get("out1");
        INDArray optOut2 = optOutputs.get("out2");

        double diff1 = refOut1.sub(optOut1).amaxNumber().doubleValue();
        double diff2 = refOut2.sub(optOut2).amaxNumber().doubleValue();
        assertTrue(diff1 < 1e-5, "out1 max diff = " + diff1);
        assertTrue(diff2 < 1e-5, "out2 max diff = " + diff2);
    }

    /**
     * Three matmuls sharing the same input — simulates Q/K/V projections.
     */
    @Test
    void testThreeParallelMatmuls_QKV() {
        SameDiff sd = SameDiff.create();
        int d = 64, hq = 32, hk = 32, hv = 32;
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, d);
        SDVariable wq = sd.constant("wq", Nd4j.randn(DataType.FLOAT, d, hq));
        SDVariable wk = sd.constant("wk", Nd4j.randn(DataType.FLOAT, d, hk));
        SDVariable wv = sd.constant("wv", Nd4j.randn(DataType.FLOAT, d, hv));
        sd.mmul("Q", x, wq);
        sd.mmul("K", x, wk);
        sd.mmul("V", x, wv);
        sd.setOutputs("Q", "K", "V");

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, d);
        Map<String, INDArray> refOutputs = sd.output(Map.of("x", input), "Q", "K", "V");

        // Optimize
        SameDiff optimized = GraphOptimizer.optimize(sd, "Q", "K", "V");

        // Should fuse 3 matmuls into 1
        int mmulCountAfter = countOpsOfType(optimized, Mmul.class);
        assertEquals(1, mmulCountAfter, "Should have 1 fused matmul for Q/K/V");

        // Verify numerical correctness — FP32 matmul accumulation order changes
        // when fusing, so allow slightly larger tolerance than element-wise ops.
        Map<String, INDArray> optOutputs = optimized.output(Map.of("x", input), "Q", "K", "V");
        for (String name : new String[]{"Q", "K", "V"}) {
            double diff = refOutputs.get(name).sub(optOutputs.get(name)).amaxNumber().doubleValue();
            assertTrue(diff < 0.01, name + " max diff = " + diff);
        }
    }

    /**
     * Matmuls with different first inputs should NOT be fused.
     */
    @Test
    void testDifferentInputs_NoFusion() {
        SameDiff sd = SameDiff.create();
        SDVariable x1 = sd.placeHolder("x1", DataType.FLOAT, -1, 4);
        SDVariable x2 = sd.placeHolder("x2", DataType.FLOAT, -1, 4);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 4, 3));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 4, 5));
        sd.mmul("out1", x1, w1);
        sd.mmul("out2", x2, w2);
        sd.setOutputs("out1", "out2");

        SameDiff optimized = GraphOptimizer.optimize(sd, "out1", "out2");

        // Should remain 2 separate matmuls — different inputs
        int mmulCount = countOpsOfType(optimized, Mmul.class);
        assertEquals(2, mmulCount, "Should still have 2 matmuls (different inputs)");
    }

    /**
     * Matmul with non-constant weight should NOT be fused.
     */
    @Test
    void testNonConstantWeight_NoFusion() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 4, 3));
        SDVariable w2 = sd.placeHolder("w2", DataType.FLOAT, 4, 5);  // placeholder, not constant
        sd.mmul("out1", x, w1);
        sd.mmul("out2", x, w2);
        sd.setOutputs("out1", "out2");

        SameDiff optimized = GraphOptimizer.optimize(sd, "out1", "out2");

        // Should remain 2 separate matmuls — w2 is not a constant
        int mmulCount = countOpsOfType(optimized, Mmul.class);
        assertEquals(2, mmulCount, "Should still have 2 matmuls (non-constant weight)");
    }

    /**
     * Single matmul should NOT be fused (nothing to fuse with).
     */
    @Test
    void testSingleMatmul_NoFusion() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 3));
        sd.mmul("out", x, w);
        sd.setOutputs("out");

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 4);
        INDArray refOut = sd.output(Map.of("x", input), "out").get("out").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "out");
        INDArray optOut = optimized.output(Map.of("x", input), "out").get("out");

        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "Single matmul should be unchanged, diff = " + diff);
    }

    private int countOpsOfType(SameDiff sd, Class<?> opType) {
        int count = 0;
        for (SameDiffOp op : sd.getOps().values()) {
            if (opType.isInstance(op.getOp())) {
                count++;
            }
        }
        return count;
    }
}
