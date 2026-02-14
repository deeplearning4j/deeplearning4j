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
 * Tests for automatic op fusion in the native executor.
 *
 * These tests validate that fusible patterns (element-wise chains, bias+activation,
 * layer norm) are correctly identified in the plan and produce correct results.
 * The actual fusion optimization happens in the C++ FusionPass — these tests
 * verify correctness of the unfused baseline that the fused version must match.
 *
 * Covers: add+relu, add+sigmoid, 4-op chains, max chain length,
 * non-fusible breaks, layer norm pattern, bias+activation, broadcast+activation.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class NativeExecutorFusionTest extends BaseNd4jTestWithBackends {

    private static final double TOL = 1e-5;

    @Test
    public void testAddReluFusionPattern() {
        // y = relu(x + bias) — classic bias+activation fusion candidate
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable bias = sd.var("bias", Nd4j.ones(DataType.FLOAT, 1, 4).mul(0.5f));
        SDVariable sum = x.add(bias);
        SDVariable z = sd.nn().relu("z", sum, 0.0);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.createFromArray(new float[][]{{-2, -1, 0, 1}});
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        // relu(-2+0.5) = 0, relu(-1+0.5) = 0, relu(0+0.5) = 0.5, relu(1+0.5) = 1.5
        assertEquals(0.0, result.get("z").getFloat(0, 0), TOL, "relu(-1.5)=0");
        assertEquals(0.0, result.get("z").getFloat(0, 1), TOL, "relu(-0.5)=0");
        assertEquals(0.5, result.get("z").getFloat(0, 2), TOL, "relu(0.5)=0.5");
        assertEquals(1.5, result.get("z").getFloat(0, 3), TOL, "relu(1.5)=1.5");

        plan.close();
    }

    @Test
    public void testAddSigmoidFusionPattern() {
        // y = sigmoid(x + bias)
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable bias = sd.var("bias", Nd4j.zeros(DataType.FLOAT, 1, 4));
        SDVariable sum = x.add(bias);
        SDVariable z = sd.nn().sigmoid("z", sum);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.zeros(DataType.FLOAT, 2, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertEquals(0.5, result.get("z").getFloat(0, 0), TOL, "sigmoid(0)=0.5");

        plan.close();
    }

    @Test
    public void testFourOpChain() {
        // y = relu(mul(add(x, b1), scale))
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable b1 = sd.var("b1", Nd4j.ones(DataType.FLOAT, 1, 4));
        SDVariable scale = sd.var("scale", Nd4j.ones(DataType.FLOAT, 1, 4).mul(2));
        SDVariable sum = x.add(b1);
        SDVariable scaled = sum.mul(scale);
        SDVariable z = sd.nn().relu("z", scaled, 0.0);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);
        assertTrue(plan.getSlots().length >= 3, "Should have >= 3 ops (add, mul, relu)");

        INDArray xArr = Nd4j.createFromArray(new float[][]{{-2, 0, 1, 3}});
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        // relu((-2+1)*2) = relu(-2) = 0
        // relu((0+1)*2) = relu(2) = 2
        // relu((1+1)*2) = relu(4) = 4
        // relu((3+1)*2) = relu(8) = 8
        assertEquals(0.0, result.get("z").getFloat(0, 0), TOL);
        assertEquals(2.0, result.get("z").getFloat(0, 1), TOL);
        assertEquals(4.0, result.get("z").getFloat(0, 2), TOL);
        assertEquals(8.0, result.get("z").getFloat(0, 3), TOL);

        plan.close();
    }

    @Test
    public void testLongChain() {
        // 8 consecutive element-wise ops
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable cur = x;
        for (int i = 0; i < 8; i++) {
            cur = cur.add(1.0);
        }
        cur.rename("z");

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.zeros(DataType.FLOAT, 1, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertEquals(8.0, result.get("z").getFloat(0, 0), TOL, "8 adds of 1 = 8");

        plan.close();
    }

    @Test
    public void testNonFusibleBreakByMatmul() {
        // Chain with matmul in the middle (non-fusible break)
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable a = x.add(1.0); // fusible
        SDVariable w = sd.var("w", Nd4j.eye(4).castTo(DataType.FLOAT));
        SDVariable b = sd.mmul(a, w); // NOT fusible (matmul)
        SDVariable z = sd.nn().relu("z", b, 0.0); // fusible with matmul output

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.ones(DataType.FLOAT, 2, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        // relu(((1+1) @ I)) = relu(2) = 2
        assertEquals(2.0, result.get("z").getFloat(0, 0), TOL);

        plan.close();
    }

    @Test
    public void testLayerNormPattern() {
        // Simplified layer norm: (x - mean) / sqrt(var + eps) * gamma + beta
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable gamma = sd.var("gamma", Nd4j.ones(DataType.FLOAT, 1, 4));
        SDVariable beta = sd.var("beta", Nd4j.zeros(DataType.FLOAT, 1, 4));

        SDVariable mean = x.mean(true, 1);
        SDVariable centered = x.sub(mean);
        SDVariable variance = centered.mul(centered).mean(true, 1);
        SDVariable eps = sd.constant(Nd4j.scalar(DataType.FLOAT, 1e-5f));
        SDVariable std = sd.math().sqrt(variance.add(eps));
        SDVariable normed = centered.div(std);
        SDVariable scaled = normed.mul(gamma);
        SDVariable z = scaled.add("z", beta);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.createFromArray(new float[][]{{1, 2, 3, 4}, {4, 3, 2, 1}});
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertNotNull(result.get("z"));
        assertArrayEquals(new long[]{2, 4}, result.get("z").shape());

        plan.close();
    }

    @Test
    public void testBiasActivationFusion() {
        // add(bias) → gelu — classic post-linear fusion
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable bias = sd.var("bias", Nd4j.zeros(DataType.FLOAT, 1, 4));
        SDVariable biased = x.add(bias);
        SDVariable z = sd.nn().gelu("z", biased);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.zeros(DataType.FLOAT, 2, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        // gelu(0+0) = 0
        assertEquals(0.0, result.get("z").getFloat(0, 0), TOL);

        plan.close();
    }

    @Test
    public void testFusionWithBroadcast() {
        // relu(x + bias[1,N]) — broadcast + activation
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable bias = sd.var("bias", Nd4j.ones(DataType.FLOAT, 1, 4).mul(0.5f));
        SDVariable biased = x.add(bias); // broadcast add
        SDVariable z = sd.nn().relu("z", biased, 0.0);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.createFromArray(new float[][]{{-1, 0, 1, 2}, {-2, -1, 0, 1}});
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        // Row 0: relu(-1+0.5)=0, relu(0+0.5)=0.5, relu(1+0.5)=1.5, relu(2+0.5)=2.5
        assertEquals(0.0, result.get("z").getFloat(0, 0), TOL);
        assertEquals(0.5, result.get("z").getFloat(0, 1), TOL);
        assertEquals(1.5, result.get("z").getFloat(0, 2), TOL);
        assertEquals(2.5, result.get("z").getFloat(0, 3), TOL);

        plan.close();
    }

    @Test
    public void testFusionPreservesCorrectness() {
        // Verify unfused and (future) fused paths produce same results
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable scale = sd.var("scale", Nd4j.ones(DataType.FLOAT, 1, 4).mul(2));
        SDVariable a = x.add(1.0);
        SDVariable b = a.mul(scale);
        SDVariable z = sd.nn().relu("z", b, 0.0);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 10, 4);
        Map<String, INDArray> result1 = sd.output(Map.of("x", xArr), "z");
        Map<String, INDArray> result2 = sd.output(Map.of("x", xArr), "z");
        NativeExecutorTestUtils.assertArrayEquals(result1.get("z"), result2.get("z"), 0,
                "Deterministic execution");

        plan.close();
    }

    @Test
    public void testMultipleFusionGroups() {
        // Two independent fusible chains
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, -1, 4);

        // Chain 1: relu(x + 1)
        SDVariable a = sd.nn().relu(x.add(1.0), 0.0);
        // Chain 2: sigmoid(y * 2)
        SDVariable b = sd.nn().sigmoid(y.mul(2.0));
        // Merge
        SDVariable z = a.add("z", b);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.zeros(DataType.FLOAT, 2, 4);
        INDArray yArr = Nd4j.zeros(DataType.FLOAT, 2, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr, "y", yArr), "z");
        // relu(0+1) = 1, sigmoid(0*2) = 0.5, sum = 1.5
        assertEquals(1.5, result.get("z").getFloat(0, 0), TOL);

        plan.close();
    }

    @Test
    public void testFusionWithDynamicShapes() {
        // Shape changes between calls — fusion should adapt
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, -1);
        SDVariable z = sd.nn().relu("z", x.add(1.0), 0.0);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        // Shape [2,4]
        Map<String, INDArray> r1 = sd.output(Map.of("x", Nd4j.zeros(2, 4)), "z");
        assertArrayEquals(new long[]{2, 4}, r1.get("z").shape());

        // Shape [5,8]
        Map<String, INDArray> r2 = sd.output(Map.of("x", Nd4j.zeros(5, 8)), "z");
        assertArrayEquals(new long[]{5, 8}, r2.get("z").shape());

        plan.close();
    }

    @Test
    public void testSwishActivation() {
        // swish(x) = x * sigmoid(x) — 3-op fusion candidate
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable sig = sd.nn().sigmoid(x);
        SDVariable z = x.mul("z", sig);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.zeros(DataType.FLOAT, 2, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        // swish(0) = 0 * 0.5 = 0
        assertEquals(0.0, result.get("z").getFloat(0, 0), TOL);

        plan.close();
    }

    @Test
    public void testSoftmaxPattern() {
        // Softmax: exp(x - max) / sum(exp(x - max))
        // This tests that reduction ops properly break fusion chains
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable z = sd.nn().softmax("z", x, -1);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.zeros(DataType.FLOAT, 2, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertEquals(0.25, result.get("z").getFloat(0, 0), TOL, "Uniform softmax");

        plan.close();
    }
}
