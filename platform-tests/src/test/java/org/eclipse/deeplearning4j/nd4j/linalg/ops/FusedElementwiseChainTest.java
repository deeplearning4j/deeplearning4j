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

package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedElementwiseChain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.BaseNd4jTestWithBackends;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the fused element-wise chain op.
 * Each test compares the fused result against computing ops separately.
 *
 * IMPORTANT: Nd4j.nn().sigmoid(x) and similar TransformStrict ops run IN-PLACE,
 * modifying x's device buffer. Always use .dup() when computing expected values
 * from the same input array used by the fused op.
 */
public class FusedElementwiseChainTest extends BaseNd4jTestWithBackends {

    private static final double TOL = 1e-4;

    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    public void testMultiplySigmoid() {
        // SiLU/Swish gate pattern: sigmoid(x * y)
        INDArray x = Nd4j.linspace(-2, 2, 100, DataType.FLOAT).reshape(10, 10);
        INDArray y = Nd4j.ones(DataType.FLOAT, 10, 10).muli(0.5);

        // Reference: compute step by step (mul creates new array, so x is safe)
        INDArray mulResult = x.mul(y);
        INDArray expected = Nd4j.nn().sigmoid(mulResult);

        // Fused
        INDArray output = Nd4j.createUninitialized(DataType.FLOAT, x.shape());
        FusedElementwiseChain op = FusedElementwiseChain.builder()
                .input(x)
                .multiply(y)
                .sigmoid()
                .output(output)
                .build();
        Nd4j.exec(op);

        assertArrayEquals(expected.shape(), output.shape());
        assertEquals(expected, output,
                "multiply->sigmoid fused should match separate execution. " +
                        "Max diff: " + expected.sub(output).amaxNumber().doubleValue());
    }

    @Test
    public void testAddRelu() {
        // FFN bias + activation: relu(x + bias)
        INDArray x = Nd4j.linspace(-5, 5, 100, DataType.FLOAT).reshape(10, 10);
        INDArray bias = Nd4j.ones(DataType.FLOAT, 10, 10).muli(0.1);

        // add creates new array, relu on that is safe
        INDArray expected = Nd4j.nn().relu(x.add(bias), 0);

        INDArray output = Nd4j.createUninitialized(DataType.FLOAT, x.shape());
        FusedElementwiseChain op = FusedElementwiseChain.builder()
                .input(x)
                .add(bias)
                .relu()
                .output(output)
                .build();
        Nd4j.exec(op);

        assertEquals(expected, output,
                "add->relu fused should match. Max diff: " +
                        expected.sub(output).amaxNumber().doubleValue());
    }

    @Test
    public void testAddGelu() {
        // Attention/FFN: gelu(x + bias)
        INDArray x = Nd4j.linspace(-3, 3, 100, DataType.FLOAT).reshape(10, 10);
        INDArray bias = Nd4j.ones(DataType.FLOAT, 10, 10).muli(0.2);

        // add creates new array, gelu on that is safe
        INDArray addResult = x.add(bias);
        INDArray expected = Nd4j.nn().gelu(addResult);

        INDArray output = Nd4j.createUninitialized(DataType.FLOAT, x.shape());
        FusedElementwiseChain op = FusedElementwiseChain.builder()
                .input(x)
                .add(bias)
                .gelu()
                .output(output)
                .build();
        Nd4j.exec(op);

        // GELU approximation may differ slightly between fused kernel and native op
        double maxDiff = expected.sub(output).amaxNumber().doubleValue();
        assertTrue(maxDiff < 0.05,
                "add->gelu fused should match within tolerance. Max diff: " + maxDiff);
    }

    @Test
    public void testAddSigmoid() {
        INDArray x = Nd4j.linspace(-3, 3, 100, DataType.FLOAT).reshape(10, 10);
        INDArray bias = Nd4j.ones(DataType.FLOAT, 10, 10).muli(0.3);

        // add creates new array, sigmoid on that is safe
        INDArray expected = Nd4j.nn().sigmoid(x.add(bias));

        INDArray output = Nd4j.createUninitialized(DataType.FLOAT, x.shape());
        FusedElementwiseChain op = FusedElementwiseChain.builder()
                .input(x)
                .add(bias)
                .sigmoid()
                .output(output)
                .build();
        Nd4j.exec(op);

        assertEquals(expected, output,
                "add->sigmoid fused should match. Max diff: " +
                        expected.sub(output).amaxNumber().doubleValue());
    }

    @Test
    public void testAddTanh() {
        INDArray x = Nd4j.linspace(-3, 3, 100, DataType.FLOAT).reshape(10, 10);
        INDArray bias = Nd4j.ones(DataType.FLOAT, 10, 10).muli(0.1);

        // add creates new array, tanh on that is safe
        INDArray expected = Nd4j.math().tanh(x.add(bias));

        INDArray output = Nd4j.createUninitialized(DataType.FLOAT, x.shape());
        FusedElementwiseChain op = FusedElementwiseChain.builder()
                .input(x)
                .add(bias)
                .tanh()
                .output(output)
                .build();
        Nd4j.exec(op);

        assertEquals(expected, output,
                "add->tanh fused should match. Max diff: " +
                        expected.sub(output).amaxNumber().doubleValue());
    }

    @Test
    public void testMultiplyTanh() {
        INDArray x = Nd4j.linspace(-2, 2, 100, DataType.FLOAT).reshape(10, 10);
        INDArray y = Nd4j.ones(DataType.FLOAT, 10, 10).muli(0.5);

        // mul creates new array, tanh on that is safe
        INDArray expected = Nd4j.math().tanh(x.mul(y));

        INDArray output = Nd4j.createUninitialized(DataType.FLOAT, x.shape());
        FusedElementwiseChain op = FusedElementwiseChain.builder()
                .input(x)
                .multiply(y)
                .tanh()
                .output(output)
                .build();
        Nd4j.exec(op);

        assertEquals(expected, output,
                "multiply->tanh fused should match. Max diff: " +
                        expected.sub(output).amaxNumber().doubleValue());
    }

    @Test
    public void testTripleChainAddReluMultiply() {
        // 3-op chain: multiply(relu(add(x, bias)), scale)
        INDArray x = Nd4j.linspace(-5, 5, 100, DataType.FLOAT).reshape(10, 10);
        INDArray bias = Nd4j.ones(DataType.FLOAT, 10, 10).muli(0.5);
        INDArray scale = Nd4j.ones(DataType.FLOAT, 10, 10).muli(2.0);

        INDArray step1 = x.add(bias);
        INDArray step2 = Nd4j.nn().relu(step1, 0);
        INDArray expected = step2.mul(scale);

        INDArray output = Nd4j.createUninitialized(DataType.FLOAT, x.shape());
        FusedElementwiseChain op = FusedElementwiseChain.builder()
                .input(x)
                .add(bias)
                .relu()
                .multiply(scale)
                .output(output)
                .build();
        Nd4j.exec(op);

        assertEquals(expected, output,
                "add->relu->multiply fused should match. Max diff: " +
                        expected.sub(output).amaxNumber().doubleValue());
    }

    @Test
    public void testUnaryOnlyChain() {
        // Pure unary chain: neg(abs(x))
        INDArray x = Nd4j.linspace(-5, 5, 100, DataType.FLOAT).reshape(10, 10);

        // abs creates new array via Nd4j.math().abs(), negi modifies that in place
        INDArray expected = Nd4j.math().abs(x).negi();

        INDArray output = Nd4j.createUninitialized(DataType.FLOAT, x.shape());
        FusedElementwiseChain op = FusedElementwiseChain.builder()
                .input(x)
                .abs()
                .neg()
                .output(output)
                .build();
        Nd4j.exec(op);

        assertEquals(expected, output,
                "abs->neg fused should match. Max diff: " +
                        expected.sub(output).amaxNumber().doubleValue());
    }

    @Test
    public void testSingleUnaryOp() {
        // Single op chain: just sigmoid
        // NOTE: Nd4j.nn().sigmoid(x) runs IN-PLACE on x, so dup first
        INDArray x = Nd4j.linspace(-3, 3, 100, DataType.FLOAT).reshape(10, 10);
        INDArray expected = Nd4j.nn().sigmoid(x.dup());

        INDArray output = Nd4j.createUninitialized(DataType.FLOAT, x.shape());
        FusedElementwiseChain op = FusedElementwiseChain.builder()
                .input(x)
                .sigmoid()
                .output(output)
                .build();
        Nd4j.exec(op);

        assertEquals(expected, output,
                "single sigmoid fused should match. Max diff: " +
                        expected.sub(output).amaxNumber().doubleValue());
    }

    @Test
    public void testBroadcastSecondaryInput() {
        // Binary op with scalar broadcast: add(x, scalar)
        INDArray x = Nd4j.linspace(-3, 3, 100, DataType.FLOAT).reshape(10, 10);
        INDArray scalar = Nd4j.scalar(DataType.FLOAT, 0.5f);

        // add creates new array, sigmoid on that is safe
        INDArray expected = Nd4j.nn().sigmoid(x.add(scalar));

        INDArray output = Nd4j.createUninitialized(DataType.FLOAT, x.shape());
        FusedElementwiseChain op = FusedElementwiseChain.builder()
                .input(x)
                .add(scalar)
                .sigmoid()
                .output(output)
                .build();
        Nd4j.exec(op);

        assertEquals(expected, output,
                "add(scalar)->sigmoid fused should match. Max diff: " +
                        expected.sub(output).amaxNumber().doubleValue());
    }

    @Test
    public void testSwish() {
        // Single swish op: x * sigmoid(x)
        // Must dup x because sigmoid runs in-place AND x.mul reads x
        INDArray x = Nd4j.linspace(-3, 3, 100, DataType.FLOAT).reshape(10, 10);
        INDArray xDup = x.dup();
        INDArray sigX = Nd4j.nn().sigmoid(xDup.dup());
        INDArray expected = xDup.mul(sigX);

        INDArray output = Nd4j.createUninitialized(DataType.FLOAT, x.shape());
        FusedElementwiseChain op = FusedElementwiseChain.builder()
                .input(x)
                .swish()
                .output(output)
                .build();
        Nd4j.exec(op);

        double maxDiff = expected.sub(output).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL,
                "swish fused should match. Max diff: " + maxDiff);
    }

    @Test
    public void testMish() {
        // Single mish op: x * tanh(softplus(x))
        // Must dup x because exp/tanh run in-place
        INDArray x = Nd4j.linspace(-3, 3, 100, DataType.FLOAT).reshape(10, 10);
        INDArray xDup = x.dup();
        INDArray softplus = Nd4j.math().log(Nd4j.math().exp(xDup.dup()).addi(1.0));
        INDArray expected = xDup.mul(Nd4j.math().tanh(softplus));

        INDArray output = Nd4j.createUninitialized(DataType.FLOAT, x.shape());
        FusedElementwiseChain op = FusedElementwiseChain.builder()
                .input(x)
                .mish()
                .output(output)
                .build();
        Nd4j.exec(op);

        double maxDiff = expected.sub(output).amaxNumber().doubleValue();
        assertTrue(maxDiff < 0.01,
                "mish fused should match. Max diff: " + maxDiff);
    }

    @Test
    public void testHalfMishNonlinearChain() {
        INDArray x = Nd4j.linspace(-3, 3, 64, DataType.FLOAT).reshape(8, 8).castTo(DataType.HALF);
        INDArray xFloat = x.castTo(DataType.FLOAT);
        INDArray softplus = Nd4j.math().log(Nd4j.math().exp(xFloat.dup()).addi(1.0));
        INDArray expected = xFloat.mul(Nd4j.math().tanh(softplus)).castTo(DataType.HALF);

        INDArray output = Nd4j.createUninitialized(DataType.HALF, x.shape());
        FusedElementwiseChain op = FusedElementwiseChain.builder()
                .input(x)
                .mish()
                .output(output)
                .build();
        Nd4j.exec(op);

        double maxDiff = expected.castTo(DataType.FLOAT).sub(output.castTo(DataType.FLOAT)).amaxNumber().doubleValue();
        assertTrue(maxDiff < 0.01,
                "half mish fused should match float reference cast to half. Max diff: " + maxDiff);
    }

    @Test
    public void testHalfExpLogSqrtChain() {
        INDArray x = Nd4j.linspace(0.25, 2.25, 64, DataType.FLOAT).reshape(8, 8).castTo(DataType.HALF);
        INDArray exp = Nd4j.math().exp(x.castTo(DataType.FLOAT).dup());
        INDArray log = Nd4j.math().log(exp);
        INDArray expected = Nd4j.math().sqrt(log).castTo(DataType.HALF);

        INDArray output = Nd4j.createUninitialized(DataType.HALF, x.shape());
        FusedElementwiseChain op = FusedElementwiseChain.builder()
                .input(x)
                .exp()
                .log()
                .sqrt()
                .output(output)
                .build();
        Nd4j.exec(op);

        double maxDiff = expected.castTo(DataType.FLOAT).sub(output.castTo(DataType.FLOAT)).amaxNumber().doubleValue();
        assertTrue(maxDiff < 0.01,
                "half exp->log->sqrt fused should match float reference cast to half. Max diff: " + maxDiff);
    }

    @Test
    public void testCodegenApiUnaryOnly() {
        // Test the codegen-generated Nd4j.nn().fusedElementwiseChain() API
        INDArray x = Nd4j.linspace(-3, 3, 100, DataType.FLOAT).reshape(10, 10);
        INDArray expected = Nd4j.nn().sigmoid(x.dup());

        // Use codegen API: just unary ops, no secondary inputs
        INDArray output = Nd4j.nn().fusedElementwiseChain(x, new INDArray[0], new int[]{FusedElementwiseChain.OP_SIGMOID});

        double maxDiff = expected.sub(output).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL,
                "codegen API sigmoid should match. Max diff: " + maxDiff);
    }

    @Test
    public void testCodegenApiWithSecondary() {
        // Test codegen API with secondary inputs
        INDArray x = Nd4j.linspace(-2, 2, 100, DataType.FLOAT).reshape(10, 10);
        INDArray y = Nd4j.ones(DataType.FLOAT, 10, 10).muli(0.5);

        // mul creates new array, sigmoid on that is safe
        INDArray expected = Nd4j.nn().sigmoid(x.mul(y));

        // Use codegen API: multiply(y) -> sigmoid
        INDArray output = Nd4j.nn().fusedElementwiseChain(x,
                new INDArray[]{y},
                new int[]{FusedElementwiseChain.OP_MUL, FusedElementwiseChain.OP_SIGMOID});

        double maxDiff = expected.sub(output).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL,
                "codegen API multiply->sigmoid should match. Max diff: " + maxDiff);
    }

    @Test
    public void testWithPreallocatedOutput() {
        // Use a pre-allocated output buffer
        // NOTE: dup x because sigmoid runs in-place
        INDArray x = Nd4j.linspace(-3, 3, 100, DataType.FLOAT).reshape(10, 10);
        INDArray expected = Nd4j.nn().sigmoid(x.dup());

        INDArray output = Nd4j.zeros(DataType.FLOAT, x.shape());
        FusedElementwiseChain op = FusedElementwiseChain.builder()
                .input(x)
                .sigmoid()
                .output(output)
                .build();
        Nd4j.exec(op);

        double maxDiff = expected.sub(output).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL,
                "preallocated output sigmoid should match. Max diff: " + maxDiff);
    }
}
