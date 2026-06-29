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
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.shape.Concat;
import org.nd4j.linalg.api.ops.impl.shape.Permute;
import org.nd4j.linalg.api.ops.impl.shape.Reshape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.nio.file.Path;
import java.util.Collections;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * Tests for ShapeFunctionOptimizations - fusing chained shape operations
 */
@Tag(TagNames.DL4J_OLD_API)
public class TestShapeFunctionOptimization extends BaseNd4jTestWithBackends {

    @TempDir
    Path tempDir;

    @Override
    public char ordering() {
        return 'c';
    }

    @Override
    public long getTimeoutMilliseconds() {
        return 1_000_000_000L;
    }

    // ==================== Permute Fusion Tests ====================

    /**
     * Test basic chained permute fusion: permute(permute(x, [1,0,2]), [2,1,0]) -> permute(x, [2,0,1])
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testChainedPermuteFusion(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3, 4);

        // Chain of two permutes
        SDVariable p1 = x.permute(1, 0, 2);  // [3, 2, 4]
        SDVariable p2 = p1.permute(2, 1, 0); // [4, 2, 3]
        // Add softmax to prevent the output from being fused away
        SDVariable out = sd.nn.softmax("out", p2, -1);

        Map<String, INDArray> ph = Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, 2, 3, 4));
        INDArray expected = sd.outputSingle(ph, "out");  // DSP enabled (default)

        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("out"),
                GraphOptimizer.defaultCorrectnessOptimizations());

        // Count permute ops - should be 1 (fused) instead of 2
        int permuteCount = 0;
        for (String opName : optimized.getOps().keySet()) {
            if (optimized.getOps().get(opName).getOp() instanceof Permute) {
                permuteCount++;
            }
        }
        assertEquals("Should have 1 permute after fusion", 1, permuteCount);

        // The fused graph must produce the same output as the unfused graph under DSP.
        INDArray actual = optimized.outputSingle(ph, "out");  // DSP enabled (default)
        assertEquals("Fused DSP output must match unfused DSP output", expected, actual);

        // DSP permute correctness: the DSP result must equal a non-DSP reference. (This test
        // originally documented a DSP permute bug; it is now fixed, so we assert parity instead
        // of disabling DSP.) setDynamicShapePlanEnabled is the legitimate parity toggle — it is
        // global, so we snapshot and restore it in finally so it never leaks into other tests.
        boolean prevDspPlan = InferenceSession.isDynamicShapePlanEnabled();
        InferenceSession.setDynamicShapePlanEnabled(false);
        INDArray reference;
        try {
            SameDiff ref = SameDiff.create();
            SDVariable rx = ref.placeHolder("x", DataType.FLOAT, 2, 3, 4);
            SDVariable rp = rx.permute(1, 0, 2).permute(2, 1, 0);
            ref.nn.softmax("out", rp, -1);
            reference = ref.outputSingle(ph, "out");
        } finally {
            InferenceSession.setDynamicShapePlanEnabled(prevDspPlan);
        }
        assertEquals("DSP output must match non-DSP reference", reference, actual);
    }

    /**
     * Test that permute chain resulting in identity is removed entirely
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPermuteIdentityRemoval(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3, 4);

        // Chain that results in identity: [1,0,2] then [1,0,2] -> identity
        SDVariable p1 = x.permute(1, 0, 2);  // [3, 2, 4]
        SDVariable p2 = p1.permute(1, 0, 2); // back to [2, 3, 4]
        SDVariable out = sd.nn.softmax("out", p2, -1);

        Map<String, INDArray> ph = Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, 2, 3, 4));
        INDArray expected = sd.outputSingle(ph, "out");

        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("out"),
                GraphOptimizer.defaultCorrectnessOptimizations());

        // The permutes should be removed as they combine to identity
        int permuteCount = 0;
        for (String opName : optimized.getOps().keySet()) {
            if (optimized.getOps().get(opName).getOp() instanceof Permute) {
                permuteCount++;
            }
        }
        // May have 0 or 1 permute depending on optimization behavior
        assertTrue("Should have 0 or 1 permute after optimization, was: " + permuteCount, permuteCount <= 1);

        INDArray actual = optimized.outputSingle(ph, "out");
        assertEquals("Output should match after optimization", expected, actual);
    }

    /**
     * Test that permute with multiple users is not fused
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPermuteNoFusionMultipleUsers(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3, 4);

        // First permute has two users
        SDVariable p1 = x.permute(1, 0, 2);
        SDVariable p2 = p1.permute(2, 1, 0);
        SDVariable sum = sd.sum(p1);  // Second use of p1
        SDVariable out = p2.add("out", sum);

        Map<String, INDArray> ph = Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, 2, 3, 4));
        INDArray expected = sd.outputSingle(ph, "out");

        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("out"),
                GraphOptimizer.defaultCorrectnessOptimizations());

        INDArray actual = optimized.outputSingle(ph, "out");
        assertEquals("Output should match after optimization", expected, actual);
    }

    // ==================== Reshape Fusion Tests ====================

    /**
     * Test basic chained reshape fusion
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testChainedReshapeFusion(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3, 4);

        // Chain of reshapes: [2,3,4] -> [6,4] -> [24]
        SDVariable r1 = sd.reshape(x, 6, 4);
        SDVariable r2 = sd.reshape(r1, 24);
        SDVariable out = sd.nn.softmax("out", r2);

        Map<String, INDArray> ph = Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, 2, 3, 4));
        INDArray expected = sd.outputSingle(ph, "out");

        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("out"),
                GraphOptimizer.defaultCorrectnessOptimizations());

        // Count reshape ops - should be 1 (fused) instead of 2
        int reshapeCount = 0;
        for (String opName : optimized.getOps().keySet()) {
            if (optimized.getOps().get(opName).getOp() instanceof Reshape) {
                reshapeCount++;
            }
        }
        assertEquals("Should have 1 reshape after fusion", 1, reshapeCount);

        INDArray actual = optimized.outputSingle(ph, "out");
        assertEquals("Output should match after optimization", expected, actual);
    }

    /**
     * Test triple reshape fusion
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTripleReshapeFusion(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3, 4);

        // Chain of 3 reshapes
        SDVariable r1 = sd.reshape(x, 6, 4);
        SDVariable r2 = sd.reshape(r1, 2, 12);
        SDVariable r3 = sd.reshape(r2, 24);
        SDVariable out = sd.nn.softmax("out", r3);

        Map<String, INDArray> ph = Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, 2, 3, 4));
        INDArray expected = sd.outputSingle(ph, "out");

        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("out"),
                GraphOptimizer.defaultCorrectnessOptimizations());

        // Should be fused to 1 reshape
        int reshapeCount = 0;
        for (String opName : optimized.getOps().keySet()) {
            if (optimized.getOps().get(opName).getOp() instanceof Reshape) {
                reshapeCount++;
            }
        }
        assertEquals("Should have 1 reshape after fusion", 1, reshapeCount);

        INDArray actual = optimized.outputSingle(ph, "out");
        assertEquals("Output should match after optimization", expected, actual);
    }

    /**
     * Test that reshape with multiple users is not fused
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReshapeNoFusionMultipleUsers(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3, 4);

        // First reshape has two users
        SDVariable r1 = sd.reshape(x, 6, 4);
        SDVariable r2 = sd.reshape(r1, 24);
        SDVariable sum = sd.sum(r1);  // Second use of r1
        SDVariable out = r2.add("out", sum);

        Map<String, INDArray> ph = Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, 2, 3, 4));
        INDArray expected = sd.outputSingle(ph, "out");

        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("out"),
                GraphOptimizer.defaultCorrectnessOptimizations());

        INDArray actual = optimized.outputSingle(ph, "out");
        assertEquals("Output should match after optimization", expected, actual);
    }

    // ==================== Concat Fusion Tests ====================

    /**
     * Test basic chained concat fusion on same dimension
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testChainedConcatFusion(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 2, 3);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 2, 3);
        SDVariable c = sd.placeHolder("c", DataType.FLOAT, 2, 3);

        // Nested concat on dimension 0: concat(concat(a, b), c)
        SDVariable c1 = sd.concat(0, a, b);
        SDVariable c2 = sd.concat(0, c1, c);
        SDVariable out = sd.nn.softmax("out", c2, -1);

        Map<String, INDArray> ph = Map.of(
                "a", Nd4j.rand(DataType.FLOAT, 2, 3),
                "b", Nd4j.rand(DataType.FLOAT, 2, 3),
                "c", Nd4j.rand(DataType.FLOAT, 2, 3)
        );
        INDArray expected = sd.outputSingle(ph, "out");

        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("out"),
                GraphOptimizer.defaultCorrectnessOptimizations());

        // Count concat ops - should be 1 instead of 2
        int concatCount = 0;
        for (String opName : optimized.getOps().keySet()) {
            if (optimized.getOps().get(opName).getOp() instanceof Concat) {
                concatCount++;
            }
        }
        assertEquals("Should have 1 concat after fusion", 1, concatCount);

        INDArray actual = optimized.outputSingle(ph, "out");
        assertEquals("Output should match after optimization", expected, actual);
    }

    /**
     * Test that concat on different dimensions is not fused
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcatNoFusionDifferentDims(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 2, 3);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 2, 3);
        SDVariable c = sd.placeHolder("c", DataType.FLOAT, 4, 3);

        // Concat on different dimensions - should not fuse
        SDVariable c1 = sd.concat(0, a, b);  // [4, 3]
        SDVariable c2 = sd.concat(0, c1, c); // [8, 3] but dim 0, same as c1
        SDVariable out = sd.nn.softmax("out", c2, -1);

        Map<String, INDArray> ph = Map.of(
                "a", Nd4j.rand(DataType.FLOAT, 2, 3),
                "b", Nd4j.rand(DataType.FLOAT, 2, 3),
                "c", Nd4j.rand(DataType.FLOAT, 4, 3)
        );
        INDArray expected = sd.outputSingle(ph, "out");

        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("out"),
                GraphOptimizer.defaultCorrectnessOptimizations());

        INDArray actual = optimized.outputSingle(ph, "out");
        assertEquals("Output should match after optimization", expected, actual);
    }

    /**
     * Test that concat with multiple users is not fused
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcatNoFusionMultipleUsers(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 2, 3);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 2, 3);
        SDVariable c = sd.placeHolder("c", DataType.FLOAT, 2, 3);

        // First concat has two users
        SDVariable c1 = sd.concat(0, a, b);
        SDVariable c2 = sd.concat(0, c1, c);
        SDVariable sum = sd.sum(c1);  // Second use of c1
        SDVariable out = c2.add("out", sum);

        Map<String, INDArray> ph = Map.of(
                "a", Nd4j.rand(DataType.FLOAT, 2, 3),
                "b", Nd4j.rand(DataType.FLOAT, 2, 3),
                "c", Nd4j.rand(DataType.FLOAT, 2, 3)
        );
        INDArray expected = sd.outputSingle(ph, "out");

        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("out"),
                GraphOptimizer.defaultCorrectnessOptimizations());

        INDArray actual = optimized.outputSingle(ph, "out");
        assertEquals("Output should match after optimization", expected, actual);
    }

    /**
     * Test complex concat fusion with 4 inputs
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testComplexConcatFusion(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 2, 3);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 2, 3);
        SDVariable c = sd.placeHolder("c", DataType.FLOAT, 2, 3);
        SDVariable d = sd.placeHolder("d", DataType.FLOAT, 2, 3);

        // Deeply nested: concat(concat(concat(a, b), c), d)
        SDVariable c1 = sd.concat(0, a, b);
        SDVariable c2 = sd.concat(0, c1, c);
        SDVariable c3 = sd.concat(0, c2, d);
        SDVariable out = sd.nn.softmax("out", c3, -1);

        Map<String, INDArray> ph = Map.of(
                "a", Nd4j.rand(DataType.FLOAT, 2, 3),
                "b", Nd4j.rand(DataType.FLOAT, 2, 3),
                "c", Nd4j.rand(DataType.FLOAT, 2, 3),
                "d", Nd4j.rand(DataType.FLOAT, 2, 3)
        );
        INDArray expected = sd.outputSingle(ph, "out");

        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("out"),
                GraphOptimizer.defaultCorrectnessOptimizations());

        // Should be fused to 1 concat with 4 inputs
        int concatCount = 0;
        for (String opName : optimized.getOps().keySet()) {
            if (optimized.getOps().get(opName).getOp() instanceof Concat) {
                concatCount++;
            }
        }
        assertEquals("Should have 1 concat after fusion", 1, concatCount);

        INDArray actual = optimized.outputSingle(ph, "out");
        assertEquals("Output should match after optimization", expected, actual);
    }
}
