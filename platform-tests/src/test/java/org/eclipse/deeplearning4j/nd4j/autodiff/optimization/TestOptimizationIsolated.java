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
import org.nd4j.autodiff.samediff.optimize.OptimizerSet;
import org.nd4j.autodiff.samediff.optimize.optimizations.*;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Isolated tests to identify which optimizer is causing the wrong result.
 */
@Tag(TagNames.DL4J_OLD_API)
public class TestOptimizationIsolated extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    @Override
    public long getTimeoutMilliseconds() {
        return 1_000_000_000L;
    }

    /**
     * Test 1: Verify the original graph gives -1.0
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void test1_OriginalGraph(Nd4jBackend nd4jBackend) {
        SameDiff sd = createTestGraph();

        INDArray result = sd.outputSingle(Collections.emptyMap(), "out");
        System.out.println("Test 1 - Original graph output: " + result);
        assertEquals("Original graph should output -1.0", -1.0, result.getDouble(0), 1e-5);
    }

    /**
     * Test 2: Verify graph.dup() preserves correct output
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void test2_GraphDup(Nd4jBackend nd4jBackend) {
        SameDiff sd = createTestGraph();
        INDArray originalResult = sd.outputSingle(Collections.emptyMap(), "out");

        SameDiff copy = sd.dup();
        INDArray copyResult = copy.outputSingle(Collections.emptyMap(), "out");

        System.out.println("Test 2 - Original: " + originalResult + ", Dup: " + copyResult);
        assertEquals("Dup should preserve output", originalResult.getDouble(0), copyResult.getDouble(0), 1e-5);
        assertEquals("Both should be -1.0", -1.0, copyResult.getDouble(0), 1e-5);
    }

    /**
     * Test 3: Verify dup() doesn't corrupt original
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void test3_DupDoesNotCorruptOriginal(Nd4jBackend nd4jBackend) {
        SameDiff sd = createTestGraph();
        INDArray resultBefore = sd.outputSingle(Collections.emptyMap(), "out");

        SameDiff copy = sd.dup();
        copy.outputSingle(Collections.emptyMap(), "out");  // Execute the copy

        INDArray resultAfter = sd.outputSingle(Collections.emptyMap(), "out");

        System.out.println("Test 3 - Before dup exec: " + resultBefore + ", After dup exec: " + resultAfter);
        assertEquals("Original should not be corrupted by dup", resultBefore.getDouble(0), resultAfter.getDouble(0), 1e-5);
    }

    /**
     * Test 4: Empty optimizations list (just cloning)
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void test4_EmptyOptimizations(Nd4jBackend nd4jBackend) {
        SameDiff sd = createTestGraph();
        INDArray originalResult = sd.outputSingle(Collections.emptyMap(), "out");

        List<OptimizerSet> emptyOptimizations = Collections.emptyList();
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("out"), emptyOptimizations);

        INDArray optimizedResult = optimized.outputSingle(Collections.emptyMap(), "out");

        System.out.println("Test 4 - Original: " + originalResult + ", Empty opts: " + optimizedResult);
        assertEquals("Empty optimizations should preserve output", originalResult.getDouble(0), optimizedResult.getDouble(0), 1e-5);
    }

    /**
     * Test 5: Just UnusedFunctionOptimizations
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void test5_OnlyUnusedFunctionOptimizations(Nd4jBackend nd4jBackend) {
        SameDiff sd = createTestGraph();
        INDArray originalResult = sd.outputSingle(Collections.emptyMap(), "out");

        List<OptimizerSet> opts = Arrays.asList(new UnusedFunctionOptimizations());
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("out"), opts);

        INDArray optimizedResult = optimized.outputSingle(Collections.emptyMap(), "out");

        System.out.println("Test 5 - Original: " + originalResult + ", UnusedFunctionOptimizations: " + optimizedResult);
        assertEquals("UnusedFunctionOptimizations should preserve output", originalResult.getDouble(0), optimizedResult.getDouble(0), 1e-5);
    }

    /**
     * Test 6: Just ConstantFunctionOptimizations
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void test6_OnlyConstantFunctionOptimizations(Nd4jBackend nd4jBackend) {
        SameDiff sd = createTestGraph();
        INDArray originalResult = sd.outputSingle(Collections.emptyMap(), "out");

        List<OptimizerSet> opts = Arrays.asList(new ConstantFunctionOptimizations());
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("out"), opts);

        INDArray optimizedResult = optimized.outputSingle(Collections.emptyMap(), "out");

        System.out.println("Test 6 - Original: " + originalResult + ", ConstantFunctionOptimizations: " + optimizedResult);
        assertEquals("ConstantFunctionOptimizations should preserve output", originalResult.getDouble(0), optimizedResult.getDouble(0), 1e-5);
    }

    /**
     * Test 7: Just IdentityFunctionOptimizations
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void test7_OnlyIdentityFunctionOptimizations(Nd4jBackend nd4jBackend) {
        SameDiff sd = createTestGraph();
        INDArray originalResult = sd.outputSingle(Collections.emptyMap(), "out");

        List<OptimizerSet> opts = Arrays.asList(new IdentityFunctionOptimizations());
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("out"), opts);

        INDArray optimizedResult = optimized.outputSingle(Collections.emptyMap(), "out");

        System.out.println("Test 7 - Original: " + originalResult + ", IdentityFunctionOptimizations: " + optimizedResult);
        assertEquals("IdentityFunctionOptimizations should preserve output", originalResult.getDouble(0), optimizedResult.getDouble(0), 1e-5);
    }

    /**
     * Test 8: Just ShapeFunctionOptimizations
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void test8_OnlyShapeFunctionOptimizations(Nd4jBackend nd4jBackend) {
        SameDiff sd = createTestGraph();
        INDArray originalResult = sd.outputSingle(Collections.emptyMap(), "out");

        List<OptimizerSet> opts = Arrays.asList(new ShapeFunctionOptimizations());
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("out"), opts);

        INDArray optimizedResult = optimized.outputSingle(Collections.emptyMap(), "out");

        System.out.println("Test 8 - Original: " + originalResult + ", ShapeFunctionOptimizations: " + optimizedResult);
        assertEquals("ShapeFunctionOptimizations should preserve output", originalResult.getDouble(0), optimizedResult.getDouble(0), 1e-5);
    }

    /**
     * Test 9: Just LinearFusionOptimizations
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void test9_OnlyLinearFusionOptimizations(Nd4jBackend nd4jBackend) {
        SameDiff sd = createTestGraph();
        INDArray originalResult = sd.outputSingle(Collections.emptyMap(), "out");

        List<OptimizerSet> opts = Arrays.asList(new LinearFusionOptimizations());
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("out"), opts);

        INDArray optimizedResult = optimized.outputSingle(Collections.emptyMap(), "out");

        System.out.println("Test 9 - Original: " + originalResult + ", LinearFusionOptimizations: " + optimizedResult);
        assertEquals("LinearFusionOptimizations should preserve output", originalResult.getDouble(0), optimizedResult.getDouble(0), 1e-5);
    }

    /**
     * Test 10: Just AttentionFusionOptimizations
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void test10_OnlyAttentionFusionOptimizations(Nd4jBackend nd4jBackend) {
        SameDiff sd = createTestGraph();
        INDArray originalResult = sd.outputSingle(Collections.emptyMap(), "out");

        List<OptimizerSet> opts = Arrays.asList(new AttentionFusionOptimizations());
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("out"), opts);

        INDArray optimizedResult = optimized.outputSingle(Collections.emptyMap(), "out");

        System.out.println("Test 10 - Original: " + originalResult + ", AttentionFusionOptimizations: " + optimizedResult);
        assertEquals("AttentionFusionOptimizations should preserve output", originalResult.getDouble(0), optimizedResult.getDouble(0), 1e-5);
    }

    /**
     * Test 11: Check if original is corrupted after optimization
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void test11_OriginalNotCorruptedAfterOptimization(Nd4jBackend nd4jBackend) {
        SameDiff sd = createTestGraph();
        INDArray resultBefore = sd.outputSingle(Collections.emptyMap(), "out");

        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("out"));
        optimized.outputSingle(Collections.emptyMap(), "out");  // Execute optimized

        INDArray resultAfter = sd.outputSingle(Collections.emptyMap(), "out");

        System.out.println("Test 11 - Before optimization: " + resultBefore + ", After optimization: " + resultAfter);
        assertEquals("Original should not be corrupted by optimization", resultBefore.getDouble(0), resultAfter.getDouble(0), 1e-5);
    }

    private SameDiff createTestGraph() {
        // c = 1.0 (constant)
        // add = c + 1 = 2.0
        // v = 1.0 (variable)
        // out = v - add = 1.0 - 2.0 = -1.0
        SameDiff sd = SameDiff.create();
        SDVariable c = sd.constant("c", Nd4j.scalar(1.0));
        SDVariable c2 = c.add("add", 1);
        SDVariable v = sd.var("variable", Nd4j.scalar(1.0));
        SDVariable out = v.sub("out", c2);
        return sd;
    }
}
