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

package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Pow;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for validating Pow operation broadcasting behavior.
 *
 * This test was created to verify the fix for ONNX Pow import where
 * a scalar exponent (e.g., 2.0 for squaring) needs to be broadcast
 * over a tensor input. The original bug was that ONNX Pow was mapped
 * to pow_pairwise which doesn't support broadcasting, causing LayerNorm
 * operations to produce NaN/Infinity values.
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@DisplayName("Pow Broadcast Validation Tests")
public class TestPowBroadcastValidation extends BaseOpValidation {

    /**
     * Test that Pow operation works correctly with a scalar exponent
     * broadcast over a tensor. This is the core case that was broken
     * in the ONNX import.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test Pow with scalar exponent broadcasting")
    public void testPowScalarExponentBroadcast(Nd4jBackend backend) {
        // Create a tensor input
        INDArray input = Nd4j.createFromArray(new float[]{
            1.0f, 2.0f, 3.0f, 4.0f,
            -1.0f, -2.0f, 0.5f, 0.25f
        }).reshape(2, 4);

        // Create a scalar exponent (2.0 for squaring, like in LayerNorm)
        INDArray exponent = Nd4j.scalar(DataType.FLOAT, 2.0);

        // Execute Pow operation
        Pow pow = new Pow(input, exponent);
        INDArray[] result = Nd4j.getExecutioner().exec(pow);

        assertNotNull(result);
        assertEquals(1, result.length);
        INDArray output = result[0];

        // Verify output shape matches input shape
        assertArrayEquals(input.shape(), output.shape(),
            "Output shape should match input shape after broadcasting");

        // Verify values are squared correctly
        INDArray expected = Nd4j.createFromArray(new float[]{
            1.0f, 4.0f, 9.0f, 16.0f,
            1.0f, 4.0f, 0.25f, 0.0625f
        }).reshape(2, 4);

        // Check for NaN/Infinity (this was the bug)
        assertFalse(output.isNaN().any(), "Output should not contain NaN values");
        assertFalse(output.isInfinite().any(), "Output should not contain Infinite values");

        // Verify values
        assertTrue(expected.equalsWithEps(output, 1e-5),
            "Output values should match expected squared values. Expected: " + expected + ", Got: " + output);

        log.info("Pow with scalar broadcast test passed. Input shape: {}, Exponent shape: {}, Output shape: {}",
            input.shape(), exponent.shape(), output.shape());
    }

    /**
     * Test Pow in SameDiff with scalar exponent broadcasting.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test Pow in SameDiff with scalar exponent")
    public void testSameDiffPowScalarExponent(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        // Create input variable
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 2, 4);

        // Create scalar exponent constant
        SDVariable exponent = sd.constant("exponent", Nd4j.scalar(DataType.FLOAT, 2.0));

        // Apply Pow operation
        SDVariable output = sd.math().pow(input, exponent);
        output.rename("output");

        // Create input data
        INDArray inputData = Nd4j.createFromArray(new float[]{
            1.0f, 2.0f, 3.0f, 4.0f,
            -1.0f, -2.0f, 0.5f, 0.25f
        }).reshape(2, 4);

        // Execute
        Map<String, INDArray> results = sd.output(Map.of("input", inputData), "output");
        INDArray result = results.get("output");

        // Verify
        assertFalse(result.isNaN().any(), "SameDiff Pow output should not contain NaN");
        assertFalse(result.isInfinite().any(), "SameDiff Pow output should not contain Infinite");

        INDArray expected = inputData.mul(inputData); // x^2 = x * x
        assertTrue(expected.equalsWithEps(result, 1e-5),
            "SameDiff Pow output should be correct");

        log.info("SameDiff Pow with scalar exponent test passed");
    }

    /**
     * Test the full LayerNorm pattern that uses Pow for variance calculation.
     * LayerNorm computes: (x - mean) / sqrt(variance + epsilon)
     * where variance = mean((x - mean)^2)
     *
     * The Pow operation is used to compute (x - mean)^2
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test LayerNorm pattern with Pow")
    public void testLayerNormPatternWithPow(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        // Input with shape [batch=2, features=4]
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 2, 4);

        // Compute mean along last axis (axis 1 for 2D tensor)
        SDVariable mean = input.mean(true, 1); // keepDims=true, shape [2, 1]

        // Compute x - mean (broadcast subtraction)
        SDVariable centered = input.sub(mean);

        // Compute (x - mean)^2 using Pow with scalar exponent
        // This is the critical operation that was broken
        SDVariable exponent = sd.constant("exponent", Nd4j.scalar(DataType.FLOAT, 2.0));
        SDVariable squared = sd.math().pow(centered, exponent);

        // Compute variance = mean((x - mean)^2)
        SDVariable variance = squared.mean(true, 1); // shape [2, 1]

        // Compute 1/sqrt(variance + epsilon)
        SDVariable epsilon = sd.constant("epsilon", Nd4j.scalar(DataType.FLOAT, 1e-5f));
        SDVariable stdInv = sd.math().rsqrt(variance.add(epsilon));

        // Normalize: (x - mean) / sqrt(variance + epsilon)
        SDVariable normalized = centered.mul(stdInv);
        normalized.rename("output");

        // Name intermediate results for debugging
        mean.rename("mean");
        centered.rename("centered");
        squared.rename("squared");
        variance.rename("variance");
        stdInv.rename("stdInv");

        // Create input data
        INDArray inputData = Nd4j.createFromArray(new float[]{
            1.0f, 2.0f, 3.0f, 4.0f,
            -2.0f, 0.0f, 2.0f, 4.0f
        }).reshape(2, 4);

        // Execute and get all intermediate outputs for debugging
        Map<String, INDArray> results = sd.output(
            Map.of("input", inputData),
            "mean", "centered", "squared", "variance", "stdInv", "output"
        );

        // Debug: print all intermediate results
        log.info("Input: {}", inputData);
        log.info("Mean: {}", results.get("mean"));
        log.info("Centered (x - mean): {}", results.get("centered"));
        log.info("Squared (centered^2): {}", results.get("squared"));
        log.info("Variance: {}", results.get("variance"));
        log.info("StdInv (1/sqrt(var+eps)): {}", results.get("stdInv"));
        log.info("Output (normalized): {}", results.get("output"));

        INDArray result = results.get("output");

        // Check the key operation: Pow with scalar exponent broadcasting
        INDArray squaredResult = results.get("squared");
        INDArray centeredResult = results.get("centered");

        // Verify squared result has no NaN/Infinity - this is the core test
        assertFalse(squaredResult.isNaN().any(),
            "Squared (Pow) result should not contain NaN. This would indicate Pow broadcasting failed.");
        assertFalse(squaredResult.isInfinite().any(),
            "Squared (Pow) result should not contain Infinite values");

        // Verify squared result matches expected (centered^2)
        INDArray expectedSquared = centeredResult.mul(centeredResult);
        assertTrue(expectedSquared.equalsWithEps(squaredResult, 1e-4),
            "Squared result should match centered^2. Expected:\n" + expectedSquared + "\nGot:\n" + squaredResult);

        // Verify no NaN or Infinity in final output
        assertFalse(result.isNaN().any(),
            "LayerNorm output should not contain NaN.");
        assertFalse(result.isInfinite().any(),
            "LayerNorm output should not contain Infinite values");

        // Verify normalized values have roughly zero mean and unit variance
        for (int i = 0; i < 2; i++) {
            INDArray row = result.getRow(i);
            double rowMean = row.meanNumber().doubleValue();
            // Use population variance (biased=true) since we're normalizing, not estimating
            double rowStd = Math.sqrt(row.var(true).getDouble(0));

            assertEquals(0.0, rowMean, 1e-3,
                "LayerNorm output row " + i + " should have mean ~0, got: " + rowMean);
            // Tolerance of 0.2 to account for N vs N-1 variance calculation differences
            // The key test is that Pow broadcasting works (no NaN) and values are in the right range
            assertEquals(1.0, rowStd, 0.2,
                "LayerNorm output row " + i + " should have std ~1, got: " + rowStd);
        }

        log.info("LayerNorm pattern with Pow test passed. Output:\n{}", result);
    }

    /**
     * Test Pow with different broadcasting scenarios.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test Pow with various broadcasting scenarios")
    public void testPowBroadcastingScenarios(Nd4jBackend backend) {
        // Scenario 1: [3, 4] base with [1] exponent
        {
            INDArray base = Nd4j.rand(DataType.FLOAT, 3, 4).muli(2).addi(0.5); // positive values
            INDArray exp = Nd4j.createFromArray(new float[]{2.0f});

            Pow pow = new Pow(base, exp);
            INDArray[] result = Nd4j.getExecutioner().exec(pow);

            assertArrayEquals(new long[]{3, 4}, result[0].shape());
            assertFalse(result[0].isNaN().any());

            log.info("Scenario 1 passed: [3,4] ^ [1]");
        }

        // Scenario 2: [2, 3, 4] base with [1] exponent (3D case)
        {
            INDArray base = Nd4j.rand(DataType.FLOAT, 2, 3, 4).muli(2).addi(0.5);
            INDArray exp = Nd4j.scalar(DataType.FLOAT, 2.0);

            Pow pow = new Pow(base, exp);
            INDArray[] result = Nd4j.getExecutioner().exec(pow);

            assertArrayEquals(new long[]{2, 3, 4}, result[0].shape());
            assertFalse(result[0].isNaN().any());

            log.info("Scenario 2 passed: [2,3,4] ^ scalar");
        }

        // Scenario 3: [2, 4] base with [4] exponent (per-element on last axis)
        {
            INDArray base = Nd4j.rand(DataType.FLOAT, 2, 4).muli(2).addi(0.5);
            INDArray exp = Nd4j.createFromArray(new float[]{1.0f, 2.0f, 0.5f, 3.0f});

            Pow pow = new Pow(base, exp);
            INDArray[] result = Nd4j.getExecutioner().exec(pow);

            assertArrayEquals(new long[]{2, 4}, result[0].shape());
            assertFalse(result[0].isNaN().any());

            log.info("Scenario 3 passed: [2,4] ^ [4]");
        }

        // Scenario 4: [2, 1, 4] base with [1, 3, 1] exponent
        {
            INDArray base = Nd4j.rand(DataType.FLOAT, 2, 1, 4).muli(2).addi(0.5);
            INDArray exp = Nd4j.rand(DataType.FLOAT, 1, 3, 1).muli(2).addi(1);

            Pow pow = new Pow(base, exp);
            INDArray[] result = Nd4j.getExecutioner().exec(pow);

            // Should broadcast to [2, 3, 4]
            assertArrayEquals(new long[]{2, 3, 4}, result[0].shape());
            assertFalse(result[0].isNaN().any());

            log.info("Scenario 4 passed: [2,1,4] ^ [1,3,1] = [2,3,4]");
        }
    }

    /**
     * Test that same-shape Pow still works (regression test).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Test Pow with same shapes (no broadcasting)")
    public void testPowSameShapes(Nd4jBackend backend) {
        INDArray base = Nd4j.createFromArray(new float[]{2.0f, 3.0f, 4.0f, 5.0f}).reshape(2, 2);
        INDArray exp = Nd4j.createFromArray(new float[]{2.0f, 2.0f, 0.5f, 3.0f}).reshape(2, 2);

        Pow pow = new Pow(base, exp);
        INDArray[] result = Nd4j.getExecutioner().exec(pow);

        INDArray expected = Nd4j.createFromArray(new float[]{
            4.0f, 9.0f,           // 2^2, 3^2
            2.0f, 125.0f          // 4^0.5, 5^3
        }).reshape(2, 2);

        assertArrayEquals(new long[]{2, 2}, result[0].shape());
        assertTrue(expected.equalsWithEps(result[0], 1e-4),
            "Same-shape Pow should produce correct results");

        log.info("Same-shape Pow test passed");
    }
}
