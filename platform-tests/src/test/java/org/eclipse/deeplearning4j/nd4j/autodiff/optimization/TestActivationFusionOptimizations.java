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
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MulOp;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Sigmoid;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Swish;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.nio.file.Path;
import java.util.Collections;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * Tests for ActivationFusionOptimizations - fusing activation patterns like sigmoid(x)*x -> swish
 */
@Tag(TagNames.DL4J_OLD_API)
public class TestActivationFusionOptimizations extends BaseNd4jTestWithBackends {

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

    /**
     * Test sigmoid(x) * x -> swish(x) fusion
     * This pattern is the definition of Swish/SiLU activation
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSigmoidMulToSwish(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);

        // sigmoid(x) * x = swish(x)
        SDVariable sigmoidX = sd.nn.sigmoid(x);
        SDVariable swishManual = sigmoidX.mul(x);
        SDVariable out = sd.nn.softmax("out", swishManual);

        Map<String, INDArray> ph = Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, 2, 4));
        INDArray expected = sd.outputSingle(ph, "out");

        SameDiff optimized = GraphOptimizer.optimize(sd, "out");

        // Should have fused to swish
        boolean foundSwish = false;
        boolean foundSigmoid = false;
        for (String opName : optimized.getOps().keySet()) {
            var op = optimized.getOps().get(opName).getOp();
            if (op instanceof Swish) {
                foundSwish = true;
            }
            if (op instanceof Sigmoid) {
                foundSigmoid = true;
            }
        }
        assertTrue("Should have swish after fusion", foundSwish);
        assertFalse("Sigmoid should be removed after fusion", foundSigmoid);

        INDArray actual = optimized.outputSingle(ph, "out");
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        assertTrue("Output difference should be small, was: " + maxDiff, maxDiff < 1e-5);
    }

    /**
     * Test x * sigmoid(x) -> swish(x) fusion (commutative)
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMulSigmoidToSwish(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);

        // x * sigmoid(x) = swish(x)
        SDVariable sigmoidX = sd.nn.sigmoid(x);
        SDVariable swishManual = x.mul(sigmoidX);
        SDVariable out = sd.nn.softmax("out", swishManual);

        Map<String, INDArray> ph = Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, 2, 4));
        INDArray expected = sd.outputSingle(ph, "out");

        SameDiff optimized = GraphOptimizer.optimize(sd, "out");

        // Should have fused to swish
        boolean foundSwish = false;
        for (String opName : optimized.getOps().keySet()) {
            if (optimized.getOps().get(opName).getOp() instanceof Swish) {
                foundSwish = true;
                break;
            }
        }
        assertTrue("Should have swish after fusion", foundSwish);

        INDArray actual = optimized.outputSingle(ph, "out");
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        assertTrue("Output difference should be small, was: " + maxDiff, maxDiff < 1e-5);
    }

    /**
     * Test that sigmoid(a) * b where a != b is NOT fused to swish
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNoFusionDifferentInputs(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, -1, 4);

        // sigmoid(x) * y where x != y should NOT be fused to swish
        SDVariable sigmoidX = sd.nn.sigmoid(x);
        SDVariable product = sigmoidX.mul(y);
        SDVariable out = sd.nn.softmax("out", product);

        Map<String, INDArray> ph = Map.of(
            "x", Nd4j.rand(DataType.FLOAT, 2, 4),
            "y", Nd4j.rand(DataType.FLOAT, 2, 4)
        );
        INDArray expected = sd.outputSingle(ph, "out");

        SameDiff optimized = GraphOptimizer.optimize(sd, "out");

        // Should NOT have fused to swish since inputs are different
        boolean foundSigmoid = false;
        for (String opName : optimized.getOps().keySet()) {
            if (optimized.getOps().get(opName).getOp() instanceof Sigmoid) {
                foundSigmoid = true;
                break;
            }
        }
        assertTrue("Sigmoid should remain when inputs differ", foundSigmoid);

        INDArray actual = optimized.outputSingle(ph, "out");
        assertEquals("Output should match after optimization", expected, actual);
    }

    /**
     * Test that sigmoid with multiple users is not fused
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNoFusionMultipleUsers(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);

        // sigmoid(x) has multiple users
        SDVariable sigmoidX = sd.nn.sigmoid(x);
        SDVariable product1 = sigmoidX.mul(x);  // Would be swish pattern
        SDVariable product2 = sigmoidX.mul(2.0);  // Another use of sigmoid
        SDVariable out = sd.nn.softmax("out", product1.add(product2));

        Map<String, INDArray> ph = Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, 2, 4));
        INDArray expected = sd.outputSingle(ph, "out");

        SameDiff optimized = GraphOptimizer.optimize(sd, "out");

        // Sigmoid should remain since it has multiple users
        boolean foundSigmoid = false;
        for (String opName : optimized.getOps().keySet()) {
            if (optimized.getOps().get(opName).getOp() instanceof Sigmoid) {
                foundSigmoid = true;
                break;
            }
        }
        assertTrue("Sigmoid should remain when it has multiple users", foundSigmoid);

        INDArray actual = optimized.outputSingle(ph, "out");
        assertEquals("Output should match after optimization", expected, actual);
    }

    /**
     * Test that using swish directly works (baseline)
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSwishDirect(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable swishX = sd.nn().swish(x);
        SDVariable out = sd.nn.softmax("out", swishX);

        Map<String, INDArray> ph = Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, 2, 4));

        // Just verify it runs without error
        INDArray output = sd.outputSingle(ph, "out");
        assertNotNull("Should produce output", output);
        assertEquals("Output should have correct batch size", 2, output.shape()[0]);
        assertEquals("Output should have correct feature size", 4, output.shape()[1]);
    }

    /**
     * Test swish(x) * y pattern detection (SwiGLU component)
     * Note: This is detection only, not fusion yet
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSwiGLUPatternDetection(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, -1, 4);

        // swish(x) * y = SwiGLU component
        SDVariable swishX = sd.nn().swish(x);
        SDVariable swiGLU = swishX.mul(y);
        SDVariable out = sd.nn.softmax("out", swiGLU);

        Map<String, INDArray> ph = Map.of(
            "x", Nd4j.rand(DataType.FLOAT, 2, 4),
            "y", Nd4j.rand(DataType.FLOAT, 2, 4)
        );
        INDArray expected = sd.outputSingle(ph, "out");

        // Just verify it optimizes without error
        SameDiff optimized = GraphOptimizer.optimize(sd, "out");

        INDArray actual = optimized.outputSingle(ph, "out");
        assertEquals("Output should match after optimization", expected, actual);
    }
}
