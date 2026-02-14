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
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for native executor graph topology handling.
 *
 * Covers: chains, diamonds, fan-out, fan-in, multiple outputs,
 * residual connections, attention patterns, concat/split patterns.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class NativeExecutorGraphTopologyTest extends BaseNd4jTestWithBackends {

    private static final double TOL = 1e-5;

    @Test
    public void testLinearChain() {
        // a → b → c → d (4-step linear chain)
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable a = x.add("a", 1.0);
        SDVariable b = a.mul("b", 2.0);
        SDVariable c = b.sub("c", 3.0);
        SDVariable d = c.div("d", 4.0);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "d");
        NativeExecutorTestUtils.assertValidSerialization(plan);
        assertTrue(plan.getSlots().length >= 4, "Linear chain should have >= 4 slots");

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 2, 4).mul(5);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "d");
        // d = ((5+1)*2 - 3) / 4 = (12 - 3)/4 = 9/4 = 2.25
        assertEquals(2.25, result.get("d").getDouble(0), TOL, "chain result");

        plan.close();
    }

    @Test
    public void testDiamondTopology() {
        SameDiff sd = NativeExecutorTestUtils.createDiamondGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "c");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 1, 4).mul(5);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "c");
        // c = (x+1) + (x*2) = 6 + 10 = 16
        assertEquals(16.0, result.get("c").getDouble(0), TOL, "diamond result");

        plan.close();
    }

    @Test
    public void testFanOutMultipleOutputs() {
        SameDiff sd = NativeExecutorTestUtils.createFanOutGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "a", "b", "c");
        NativeExecutorTestUtils.assertValidSerialization(plan);
        assertEquals(3, plan.getOutputNameToSlotIndex().size(), "Should have 3 outputs");

        plan.close();
    }

    @Test
    public void testFanIn() {
        // Multiple inputs converge to single output
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, -1, 4);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, -1, 4);
        SDVariable c = sd.placeHolder("c", DataType.FLOAT, -1, 4);
        SDVariable sum = a.add(b).add("z", c);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray ones = Nd4j.ones(DataType.FLOAT, 2, 4);
        Map<String, INDArray> result = sd.output(Map.of("a", ones, "b", ones.mul(2), "c", ones.mul(3)), "z");
        assertEquals(6.0, result.get("z").getDouble(0), TOL, "fan-in sum");

        plan.close();
    }

    @Test
    public void testResidualConnection() {
        // y = relu(x @ w + b) + x (residual)
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable w = sd.var("w", Nd4j.eye(4).castTo(DataType.FLOAT));
        SDVariable b = sd.var("b", Nd4j.zeros(DataType.FLOAT, 1, 4));
        SDVariable hidden = sd.nn().relu(sd.mmul(x, w).add(b), 0.0);
        SDVariable out = hidden.add("z", x); // residual

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 2, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        // relu(x @ I + 0) + x = relu(x) + x = x + x = 2*x = 2
        assertEquals(2.0, result.get("z").getDouble(0), TOL, "residual");

        plan.close();
    }

    @Test
    public void testConcatSplitPattern() {
        // Concat two inputs, then slice apart
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, -1, 2);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, -1, 3);
        SDVariable cat = sd.concat("cat", 1, a, b);
        // Take first 2 columns back
        SDVariable z = sd.slice("z", cat, new int[]{0, 0}, new int[]{-1, 2});

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray aArr = Nd4j.ones(DataType.FLOAT, 2, 2).mul(10);
        INDArray bArr = Nd4j.ones(DataType.FLOAT, 2, 3).mul(20);
        Map<String, INDArray> result = sd.output(Map.of("a", aArr, "b", bArr), "z");
        assertArrayEquals(new long[]{2, 2}, result.get("z").shape());

        plan.close();
    }

    @Test
    public void testWideGraph() {
        // Many independent branches from single input
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable sum = null;
        for (int i = 0; i < 10; i++) {
            SDVariable branch = x.mul(i + 1.0);
            sum = (sum == null) ? branch : sum.add(branch);
        }
        sum.rename("z");

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        plan.close();
    }

    @Test
    public void testDeepChain() {
        // 20-op deep chain
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable cur = x;
        for (int i = 0; i < 20; i++) {
            cur = cur.add(0.01); // tiny increment
        }
        cur.rename("z");

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);
        assertTrue(plan.getSlots().length >= 20, "Deep chain should have >= 20 slots");

        INDArray xArr = Nd4j.zeros(DataType.FLOAT, 1, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertEquals(0.2, result.get("z").getDouble(0), 1e-4, "20 * 0.01 = 0.2");

        plan.close();
    }

    @Test
    public void testMultipleOutputVariables() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable a = x.add("a", 1);
        SDVariable b = x.mul("b", 2);
        SDVariable c = a.add("c", b);

        // Request all intermediate + final outputs
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "a", "b", "c");
        NativeExecutorTestUtils.assertValidSerialization(plan);
        assertEquals(3, plan.getOutputNameToSlotIndex().size());

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 1, 4).mul(3);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "a", "b", "c");
        assertEquals(4.0, result.get("a").getDouble(0), TOL);
        assertEquals(6.0, result.get("b").getDouble(0), TOL);
        assertEquals(10.0, result.get("c").getDouble(0), TOL);

        plan.close();
    }

    @Test
    public void testSubsetOutputRequest() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable a = x.add("a", 1);
        SDVariable b = a.mul("b", 2);
        SDVariable c = b.add("c", 3);

        // Only request final output — intermediate ops still execute but don't need to be returned
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "c");
        NativeExecutorTestUtils.assertValidSerialization(plan);
        assertEquals(1, plan.getOutputNameToSlotIndex().size());

        plan.close();
    }

    @Test
    public void testTwoLayerMLP() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, 4, 8).mul(0.01));
        SDVariable b1 = sd.var("b1", Nd4j.zeros(DataType.FLOAT, 1, 8));
        SDVariable w2 = sd.var("w2", Nd4j.randn(DataType.FLOAT, 8, 2).mul(0.01));
        SDVariable b2 = sd.var("b2", Nd4j.zeros(DataType.FLOAT, 1, 2));

        SDVariable h = sd.nn().relu(sd.mmul(x, w1).add(b1), 0.0);
        SDVariable out = sd.nn().softmax("z", sd.mmul(h, w2).add(b2), -1);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 3, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertArrayEquals(new long[]{3, 2}, result.get("z").shape());
        // Softmax outputs should sum to 1 per row
        for (int i = 0; i < 3; i++) {
            double rowSum = result.get("z").getRow(i).sumNumber().doubleValue();
            assertEquals(1.0, rowSum, 1e-4, "Softmax row sum for row " + i);
        }

        plan.close();
    }

    @Test
    public void testAttentionPattern() {
        // Simplified attention: Q*K^T / sqrt(d) → softmax → * V
        int batchSize = 2, seqLen = 4, dim = 8;
        SameDiff sd = SameDiff.create();
        SDVariable q = sd.placeHolder("q", DataType.FLOAT, batchSize, seqLen, dim);
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, batchSize, seqLen, dim);
        SDVariable v = sd.placeHolder("v", DataType.FLOAT, batchSize, seqLen, dim);

        // scores = Q @ K^T (using mmul which supports 3D batched matmul)
        SDVariable kt = sd.permute(k, 0, 2, 1);
        SDVariable scores = sd.mmul("scores", q, kt);
        // scale
        SDVariable scale = sd.constant(Nd4j.scalar(DataType.FLOAT, 1.0f / (float) Math.sqrt(dim)));
        SDVariable scaled = scores.mul(scale);
        // softmax
        SDVariable attn = sd.nn().softmax(scaled, -1);
        // apply to values
        SDVariable out = sd.mmul("z", attn, v);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray qArr = Nd4j.randn(DataType.FLOAT, batchSize, seqLen, dim);
        INDArray kArr = Nd4j.randn(DataType.FLOAT, batchSize, seqLen, dim);
        INDArray vArr = Nd4j.randn(DataType.FLOAT, batchSize, seqLen, dim);
        Map<String, INDArray> result = sd.output(Map.of("q", qArr, "k", kArr, "v", vArr), "z");
        assertArrayEquals(new long[]{batchSize, seqLen, dim}, result.get("z").shape());

        plan.close();
    }

    @Test
    public void testSharedConstant() {
        // Same constant used by multiple ops
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable scale = sd.constant("scale", Nd4j.ones(DataType.FLOAT, 1, 4).mul(2));
        SDVariable a = x.mul(scale);
        SDVariable b = x.add(scale);
        SDVariable z = a.add("z", b);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 1, 4).mul(3);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        // z = (3*2) + (3+2) = 6 + 5 = 11
        assertEquals(11.0, result.get("z").getDouble(0), TOL);

        plan.close();
    }

    @Test
    public void testReleaseScheduleExists() {
        SameDiff sd = NativeExecutorTestUtils.createChainGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        // Verify release schedule is populated
        assertNotNull(plan.getReleaseAtStep());
        assertEquals(plan.getSlots().length, plan.getReleaseAtStep().length,
                "Release schedule should have entry per step");

        // At least some steps should release intermediate slots
        int totalReleased = 0;
        for (int[] releases : plan.getReleaseAtStep()) {
            totalReleased += releases.length;
        }
        assertTrue(totalReleased > 0, "Should release at least some intermediate slots");

        plan.close();
    }

    @Test
    public void testExternalInputMapping() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable w = sd.var("w", Nd4j.ones(DataType.FLOAT, 1, 4));
        SDVariable c = sd.constant("c", Nd4j.ones(DataType.FLOAT, 1, 4).mul(2));
        SDVariable z = x.mul(w).add("z", c);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        // externalInputKeys should contain the placeholder, variable, and constant names
        String[] keys = plan.getExternalInputKeys();
        assertNotNull(keys);
        assertTrue(keys.length >= 3, "Should have at least 3 external inputs (x, w, c)");

        plan.close();
    }
}
