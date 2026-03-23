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

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that clearing externalPlaceholderBuffers at the start of each output() call
 * does not cause placeholder data buffers to be closed or corrupted during execution.
 *
 * Background: The new code adds externalPlaceholderBuffers.clear() at the start of
 * output(). If this causes previously-registered placeholder DataBuffers to become
 * eligible for GC/close while still referenced by ops in the execution graph, the
 * decode loop would produce corrupt results.
 *
 * Test approach: Pass the SAME placeholder arrays across multiple output() calls
 * (simulating a decode loop that reuses KV cache arrays). Verify the placeholder
 * arrays remain valid and results are correct after each call.
 */
@DisplayName("externalPlaceholderBuffers.clear() isolation test")
public class PlaceholderBufferClearTest extends BaseNd4jTestWithBackends {

    private static final Logger log = LoggerFactory.getLogger(PlaceholderBufferClearTest.class);
    private static final double TOL = 1e-4;

    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    @DisplayName("Reusing same placeholder arrays across multiple output() calls")
    public void testSamePlaceholderArraysAcrossMultipleOutputCalls() {
        int dim = 32;

        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, dim);
        SDVariable weights = sd.var("weights", Nd4j.eye(dim).castTo(DataType.FLOAT));
        SDVariable out = sd.mmul("output", input, weights);

        // Create placeholder array once, reuse across calls
        INDArray inputArr = Nd4j.ones(DataType.FLOAT, 1, dim).mul(42.0f);
        float expectedVal = 42.0f;

        for (int i = 0; i < 10; i++) {
            // Verify input array is still valid before each call
            assertFalse(inputArr.wasClosed(), "Iteration " + i + ": input array was closed before output()");
            assertNotNull(inputArr.data(), "Iteration " + i + ": input data buffer is null before output()");
            assertEquals(expectedVal, inputArr.getFloat(0), TOL,
                    "Iteration " + i + ": input array data corrupted before output()");

            Map<String, INDArray> ph = Collections.singletonMap("input", inputArr);
            Map<String, INDArray> result = sd.output(ph, "output");
            INDArray output = result.get("output");

            // Verify output is correct (identity matmul should produce same values)
            double maxDiff = output.sub(inputArr).amaxNumber().doubleValue();
            log.info("Iteration {}: output first={}, expected={}, maxDiff={}",
                    i, output.getFloat(0), expectedVal, maxDiff);
            assertTrue(maxDiff < TOL,
                    "Iteration " + i + ": output wrong after clear. maxDiff=" + maxDiff);

            // Verify input array is still valid AFTER each call
            assertFalse(inputArr.wasClosed(), "Iteration " + i + ": input array was closed after output()");
            assertNotNull(inputArr.data(), "Iteration " + i + ": input data buffer is null after output()");
            assertEquals(expectedVal, inputArr.getFloat(0), TOL,
                    "Iteration " + i + ": input array data corrupted after output()");
        }
    }

    @Test
    @DisplayName("Different placeholder arrays each call - values change correctly")
    public void testDifferentPlaceholderArraysEachCall() {
        int dim = 16;

        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, dim);
        SDVariable scale = sd.var("scale", Nd4j.ones(DataType.FLOAT, dim).mul(2.0));
        SDVariable out = input.mul("output", scale);

        for (int i = 0; i < 10; i++) {
            float fillVal = (float)(i + 1);
            INDArray inputArr = Nd4j.ones(DataType.FLOAT, 1, dim).mul(fillVal);
            float expectedVal = fillVal * 2.0f;

            Map<String, INDArray> ph = Collections.singletonMap("input", inputArr);
            Map<String, INDArray> result = sd.output(ph, "output");
            INDArray output = result.get("output");

            double maxDiff = Math.abs(output.getFloat(0) - expectedVal);
            log.info("Iteration {}: input={}, expected={}, got={}, diff={}",
                    i, fillVal, expectedVal, output.getFloat(0), maxDiff);
            assertTrue(maxDiff < TOL,
                    "Iteration " + i + ": expected " + expectedVal + " but got " + output.getFloat(0));
        }
    }

    @Test
    @DisplayName("Multiple placeholders with mixed reuse patterns")
    public void testMultiplePlaceholdersMixedReuse() {
        // Simulates a decode loop: some placeholders are reused (like KV cache),
        // some change each step (like input_ids)
        int dim = 16;

        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("changing_input", DataType.FLOAT, 1, dim);
        SDVariable b = sd.placeHolder("stable_input", DataType.FLOAT, 1, dim);
        SDVariable out = a.add("output", b);

        // Stable input stays the same across all calls
        INDArray stableArr = Nd4j.ones(DataType.FLOAT, 1, dim).mul(100.0f);

        for (int i = 0; i < 10; i++) {
            float changingVal = (float)(i + 1);
            INDArray changingArr = Nd4j.ones(DataType.FLOAT, 1, dim).mul(changingVal);
            float expectedVal = changingVal + 100.0f;

            // Verify stable array survives across calls
            assertFalse(stableArr.wasClosed(), "Step " + i + ": stable array closed");
            assertEquals(100.0f, stableArr.getFloat(0), TOL,
                    "Step " + i + ": stable array corrupted");

            Map<String, INDArray> ph = new LinkedHashMap<>();
            ph.put("changing_input", changingArr);
            ph.put("stable_input", stableArr);
            Map<String, INDArray> result = sd.output(ph, "output");
            INDArray output = result.get("output");

            double diff = Math.abs(output.getFloat(0) - expectedVal);
            log.info("Step {}: changing={}, stable=100, expected={}, got={}, diff={}",
                    i, changingVal, expectedVal, output.getFloat(0), diff);
            assertTrue(diff < TOL,
                    "Step " + i + ": expected " + expectedVal + " but got " + output.getFloat(0));
        }

        // Final check: stable array still intact
        assertFalse(stableArr.wasClosed(), "Stable array closed after all iterations");
        assertEquals(100.0f, stableArr.getFloat(0), TOL, "Stable array corrupted after all iterations");
    }

    @Test
    @DisplayName("output() then outputDirect() with same placeholder arrays")
    public void testOutputThenOutputDirectSamePlaceholders() {
        // Decode loop pattern: steps 0-1 via output(), steps 2+ via outputDirect()
        int dim = 16;

        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, dim);
        SDVariable weights = sd.var("weights", Nd4j.eye(dim).castTo(DataType.FLOAT));
        SDVariable out = sd.mmul("output", input, weights);

        INDArray inputArr = Nd4j.ones(DataType.FLOAT, 1, dim).mul(7.0f);

        // Steps 0-1: output()
        for (int step = 0; step < 2; step++) {
            Map<String, INDArray> ph = Collections.singletonMap("input", inputArr);
            Map<String, INDArray> result = sd.output(ph, "output");
            INDArray output = result.get("output");
            double maxDiff = output.sub(inputArr).amaxNumber().doubleValue();
            log.info("output() step {}: maxDiff={}", step, maxDiff);
            assertTrue(maxDiff < TOL, "output() step " + step + " failed");
        }

        // Verify input survived
        assertFalse(inputArr.wasClosed(), "Input closed after output() steps");
        assertEquals(7.0f, inputArr.getFloat(0), TOL, "Input corrupted after output() steps");

        // Steps 2+: outputDirect()
        for (int step = 2; step < 10; step++) {
            Map<String, INDArray> ph = Collections.singletonMap("input", inputArr);
            Map<String, INDArray> result = sd.outputDirect(ph, "output");
            INDArray output = result.get("output");
            double maxDiff = output.sub(inputArr).amaxNumber().doubleValue();
            log.info("outputDirect() step {}: maxDiff={}", step, maxDiff);
            assertTrue(maxDiff < TOL, "outputDirect() step " + step + " failed");
        }

        // Final integrity check
        assertFalse(inputArr.wasClosed(), "Input closed after outputDirect() steps");
        assertEquals(7.0f, inputArr.getFloat(0), TOL, "Input corrupted after outputDirect() steps");
    }
}
