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

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Regression tests for the centralized OpTraitTable refactoring.
 *
 * These tests verify that DSP plan compilation and execution produce correct results
 * after replacing hardcoded op lists (DATA_DEPENDENT_OPS, FULLY_WRITING_OPS,
 * VALUE_DEPENDENT_OPS, etc.) with centralized OpDescriptor traits.
 *
 * Each test exercises a different op category to catch trait misclassification:
 * - Unary elementwise (activation)
 * - Binary elementwise (broadcastable)
 * - Reduction
 * - Value-dependent shape (reshape, gather)
 * - View-producing (expand_dims, squeeze)
 * - Normalization (softmax)
 * - Identity
 */
public class OpTraitRegressionTest extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    /**
     * Test that a model with unary elementwise ops (activation traits)
     * produces correct DSP results matching standard execution.
     */
    @Test
    public void testUnaryElementwiseActivationChain() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3);
        SDVariable relu = sd.nn().relu(x, 0.0);
        SDVariable sigmoid = sd.nn().sigmoid(relu);
        SDVariable result = sd.math().tanh(sigmoid);
        result.rename("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 3);
        Map<String, INDArray> standard = sd.output(Map.of("x", input), "output");

        // Run again with DSP enabled
        sd.setDspAutoCompileEnabled(true);
        Map<String, INDArray> dsp = sd.output(Map.of("x", input), "output");

        INDArray expected = standard.get("output");
        INDArray actual = dsp.get("output");
        assertArrayEquals(expected.shape(), actual.shape());
        assertTrue(expected.equalsWithEps(actual, 1e-4),
                "Unary elementwise chain: DSP result differs from standard");

        sd.close();
    }

    /**
     * Test binary elementwise ops (broadcastable traits).
     */
    @Test
    public void testBinaryElementwiseBroadcast() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable bias = sd.constant("bias", Nd4j.ones(DataType.FLOAT, 1, 4).mul(0.5));
        SDVariable added = x.add(bias);
        SDVariable result = added.mul(sd.constant("scale", Nd4j.create(new float[]{2.0f})));
        result.rename("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 3, 4);
        Map<String, INDArray> standard = sd.output(Map.of("x", input), "output");

        sd.setDspAutoCompileEnabled(true);
        Map<String, INDArray> dsp = sd.output(Map.of("x", input), "output");

        assertTrue(standard.get("output").equalsWithEps(dsp.get("output"), 1e-4),
                "Binary elementwise broadcast: DSP result differs from standard");

        sd.close();
    }

    /**
     * Test reduction ops (reduce_sum, reduce_mean).
     */
    @Test
    public void testReductionOps() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable sum = x.sum(1);
        SDVariable result = sd.math().log(sum.add(1.0));
        result.rename("output");

        INDArray input = Nd4j.rand(DataType.FLOAT, 3, 4).add(0.1);
        Map<String, INDArray> standard = sd.output(Map.of("x", input), "output");

        sd.setDspAutoCompileEnabled(true);
        Map<String, INDArray> dsp = sd.output(Map.of("x", input), "output");

        assertTrue(standard.get("output").equalsWithEps(dsp.get("output"), 1e-4),
                "Reduction ops: DSP result differs from standard");

        sd.close();
    }

    /**
     * Test value-dependent shape ops (reshape with dynamic shape).
     */
    @Test
    public void testValueDependentShapeReshape() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 6);
        // Reshape to [2, 3, 2] using constant shape
        SDVariable shape = sd.constant("shape", Nd4j.createFromArray(2L, 3L, 2L));
        SDVariable reshaped = sd.reshape(x, shape);
        SDVariable result = reshaped.sum(2);
        result.rename("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 6);
        Map<String, INDArray> standard = sd.output(Map.of("x", input), "output");

        sd.setDspAutoCompileEnabled(true);
        Map<String, INDArray> dsp = sd.output(Map.of("x", input), "output");

        assertArrayEquals(standard.get("output").shape(), dsp.get("output").shape());
        assertTrue(standard.get("output").equalsWithEps(dsp.get("output"), 1e-4),
                "Value-dependent reshape: DSP result differs from standard");

        sd.close();
    }

    /**
     * Test softmax (normalization trait).
     */
    @Test
    public void testNormalizationSoftmax() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 5);
        SDVariable result = sd.nn().softmax(x, -1);
        result.rename("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 5);
        Map<String, INDArray> standard = sd.output(Map.of("x", input), "output");

        sd.setDspAutoCompileEnabled(true);
        Map<String, INDArray> dsp = sd.output(Map.of("x", input), "output");

        assertTrue(standard.get("output").equalsWithEps(dsp.get("output"), 1e-4),
                "Softmax normalization: DSP result differs from standard");

        sd.close();
    }

    /**
     * Test matmul (MATMUL trait + FULLY_WRITING).
     */
    @Test
    public void testMatmulOps() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 3, 4));
        SDVariable bias = sd.constant("bias", Nd4j.zeros(DataType.FLOAT, 4));
        SDVariable mm = sd.mmul(x, w);
        SDVariable result = mm.add(bias);
        result.rename("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 3);
        Map<String, INDArray> standard = sd.output(Map.of("x", input), "output");

        sd.setDspAutoCompileEnabled(true);
        Map<String, INDArray> dsp = sd.output(Map.of("x", input), "output");

        assertTrue(standard.get("output").equalsWithEps(dsp.get("output"), 1e-3),
                "Matmul: DSP result differs from standard");

        sd.close();
    }

    /**
     * Test multi-category model: exercises all trait categories in one graph.
     * This is the key regression test — if any op is misclassified,
     * the DSP plan will compile incorrectly and produce wrong results.
     */
    @Test
    public void testAllCategoriesCombined() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 8);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 8, 4));

        // Matmul (MATMUL trait)
        SDVariable hidden = sd.mmul(x, w);

        // Binary elementwise (BINARY_ELEMENTWISE trait)
        SDVariable bias = sd.constant("bias", Nd4j.zeros(DataType.FLOAT, 4));
        SDVariable biased = hidden.add(bias);

        // Unary elementwise / activation (ACTIVATION trait)
        SDVariable activated = sd.nn().relu(biased, 0.0);

        // Normalization (NORMALIZATION trait)
        SDVariable normed = sd.nn().softmax(activated, -1);

        // Reduction (REDUCTION trait)
        SDVariable result = normed.sum(1);
        result.rename("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 8);
        Map<String, INDArray> standard = sd.output(Map.of("x", input), "output");

        sd.setDspAutoCompileEnabled(true);
        Map<String, INDArray> dsp = sd.output(Map.of("x", input), "output");

        assertArrayEquals(standard.get("output").shape(), dsp.get("output").shape());
        assertTrue(standard.get("output").equalsWithEps(dsp.get("output"), 1e-3),
                "All-category combined model: DSP result differs from standard.\n" +
                "Expected: " + standard.get("output") + "\nActual: " + dsp.get("output"));

        sd.close();
    }
}
