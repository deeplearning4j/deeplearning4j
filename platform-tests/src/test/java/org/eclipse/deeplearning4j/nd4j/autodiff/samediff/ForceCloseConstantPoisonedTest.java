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
 * Tests that force-closing constant-poisoned buffers in InferenceSession cleanup
 * does not destroy external placeholder arrays or graph constants.
 *
 * Background: The new cleanup code in InferenceSession (around line 1599) does:
 *   buf.setConstant(false);
 *   buf.close();
 * for buffers marked constant that are NOT in protectedBuffers. If a buffer is
 * shared between the cleanup scope and external arrays (like KV cache placeholders),
 * force-closing it would destroy the external array.
 *
 * Test approach: Pass external arrays as placeholders, run output(), then verify
 * the external arrays are still valid (not closed, data intact).
 */
@DisplayName("Force-close constant-poisoned buffer isolation test")
public class ForceCloseConstantPoisonedTest extends BaseNd4jTestWithBackends {

    private static final Logger log = LoggerFactory.getLogger(ForceCloseConstantPoisonedTest.class);
    private static final double TOL = 1e-4;

    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    @DisplayName("External placeholder arrays survive output() cleanup")
    public void testExternalPlaceholderArraysSurviveCleanup() {
        int dim = 32;

        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, dim);
        SDVariable weights = sd.var("weights", Nd4j.randn(DataType.FLOAT, dim, dim).mul(0.01));
        SDVariable out = sd.mmul("output", input, weights);

        // Create external placeholder arrays
        INDArray externalInput = Nd4j.randn(DataType.FLOAT, 1, dim);
        float firstVal = externalInput.getFloat(0);
        long externalInputId = externalInput.data().pointer().address();

        Map<String, INDArray> ph = Collections.singletonMap("input", externalInput);
        Map<String, INDArray> result = sd.output(ph, "output");

        // Verify external input array survived cleanup
        assertFalse(externalInput.wasClosed(),
                "External input array was closed by output() cleanup");
        assertNotNull(externalInput.data(),
                "External input data buffer is null after output() cleanup");
        assertEquals(firstVal, externalInput.getFloat(0), TOL,
                "External input data corrupted after output() cleanup");

        log.info("External input survived: wasClosed={}, firstVal={}, expected={}",
                externalInput.wasClosed(), externalInput.getFloat(0), firstVal);
    }

    @Test
    @DisplayName("Graph constants survive output() cleanup")
    public void testGraphConstantsSurviveCleanup() {
        int dim = 16;

        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, dim);
        SDVariable constant = sd.constant("myConst", Nd4j.ones(DataType.FLOAT, dim).mul(3.14));
        SDVariable out = input.add("output", constant);

        INDArray inputArr = Nd4j.ones(DataType.FLOAT, 1, dim);
        Map<String, INDArray> ph = Collections.singletonMap("input", inputArr);

        // Run output multiple times
        for (int i = 0; i < 5; i++) {
            Map<String, INDArray> result = sd.output(ph, "output");
            INDArray output = result.get("output");

            // Expected: 1.0 + 3.14 = 4.14
            double maxDiff = output.sub(Nd4j.ones(DataType.FLOAT, 1, dim).mul(4.14)).amaxNumber().doubleValue();
            log.info("Iteration {}: output first={}, expected=4.14, maxDiff={}", i, output.getFloat(0), maxDiff);
            assertTrue(maxDiff < TOL,
                    "Iteration " + i + ": constant appears destroyed. maxDiff=" + maxDiff);

            // Verify the constant array is still intact in the graph
            INDArray constArr = sd.getVariable("myConst").getArr();
            assertNotNull(constArr, "Iteration " + i + ": constant array is null");
            assertFalse(constArr.wasClosed(), "Iteration " + i + ": constant array was closed");
            assertEquals(3.14f, constArr.getFloat(0), (float) TOL,
                    "Iteration " + i + ": constant value corrupted");
        }
    }

    @Test
    @DisplayName("Shared buffer between placeholder and intermediate does not cause double-close")
    public void testSharedBufferNoDoubleClose() {
        // Build a graph where placeholder data flows through a view-producing op
        // (reshape), so the intermediate shares a DataBuffer with the placeholder.
        // The cleanup must NOT close that shared buffer.
        int dim = 16;

        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, dim);
        SDVariable reshaped = sd.reshape("reshaped", input, sd.constant(Nd4j.createFromArray(new long[]{dim})));
        SDVariable scale = sd.var("scale", Nd4j.ones(DataType.FLOAT, dim).mul(2.0));
        SDVariable out = reshaped.mul("output", scale);

        INDArray externalInput = Nd4j.ones(DataType.FLOAT, 1, dim).mul(5.0f);

        for (int i = 0; i < 5; i++) {
            Map<String, INDArray> ph = Collections.singletonMap("input", externalInput);
            Map<String, INDArray> result = sd.output(ph, "output");
            INDArray output = result.get("output");

            // Expected: 5.0 * 2.0 = 10.0
            double maxDiff = Math.abs(output.getFloat(0) - 10.0f);
            log.info("Iteration {}: output first={}, expected=10.0, diff={}", i, output.getFloat(0), maxDiff);
            assertTrue(maxDiff < TOL,
                    "Iteration " + i + ": wrong result, possible buffer corruption. Got " + output.getFloat(0));

            // Verify external input still intact
            assertFalse(externalInput.wasClosed(),
                    "Iteration " + i + ": external input closed (shared buffer double-close)");
            assertEquals(5.0f, externalInput.getFloat(0), (float) TOL,
                    "Iteration " + i + ": external input data corrupted");
        }
    }

    @Test
    @DisplayName("External arrays survive across output() then outputDirect() transitions")
    public void testExternalArraysSurviveOutputThenOutputDirect() {
        // Decode loop: steps 0-1 via output(), steps 2+ via outputDirect()
        // External arrays must survive both paths
        int dim = 16;

        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, dim);
        SDVariable kvCache = sd.placeHolder("kv_cache", DataType.FLOAT, 1, dim);
        SDVariable out = input.add("output", kvCache);

        INDArray inputArr = Nd4j.ones(DataType.FLOAT, 1, dim);
        INDArray kvCacheArr = Nd4j.ones(DataType.FLOAT, 1, dim).mul(10.0f);

        // Steps 0-1: output()
        for (int step = 0; step < 2; step++) {
            Map<String, INDArray> ph = new LinkedHashMap<>();
            ph.put("input", inputArr);
            ph.put("kv_cache", kvCacheArr);
            Map<String, INDArray> result = sd.output(ph, "output");
            INDArray output = result.get("output");

            double expected = 11.0;
            double diff = Math.abs(output.getFloat(0) - expected);
            log.info("output() step {}: got={}, expected={}, diff={}", step, output.getFloat(0), expected, diff);
            assertTrue(diff < TOL, "output() step " + step + " wrong result");
        }

        // Verify both arrays survived
        assertFalse(inputArr.wasClosed(), "inputArr closed after output() steps");
        assertFalse(kvCacheArr.wasClosed(), "kvCacheArr closed after output() steps");
        assertEquals(1.0f, inputArr.getFloat(0), (float) TOL, "inputArr corrupted");
        assertEquals(10.0f, kvCacheArr.getFloat(0), (float) TOL, "kvCacheArr corrupted");

        // Steps 2+: outputDirect()
        for (int step = 2; step < 10; step++) {
            Map<String, INDArray> ph = new LinkedHashMap<>();
            ph.put("input", inputArr);
            ph.put("kv_cache", kvCacheArr);
            Map<String, INDArray> result = sd.outputDirect(ph, "output");
            INDArray output = result.get("output");

            double expected = 11.0;
            double diff = Math.abs(output.getFloat(0) - expected);
            log.info("outputDirect() step {}: got={}, expected={}, diff={}", step, output.getFloat(0), expected, diff);
            assertTrue(diff < TOL, "outputDirect() step " + step + " wrong result");
        }

        // Final check
        assertFalse(inputArr.wasClosed(), "inputArr closed after outputDirect() steps");
        assertFalse(kvCacheArr.wasClosed(), "kvCacheArr closed after outputDirect() steps");
        assertEquals(1.0f, inputArr.getFloat(0), (float) TOL, "inputArr corrupted after all steps");
        assertEquals(10.0f, kvCacheArr.getFloat(0), (float) TOL, "kvCacheArr corrupted after all steps");
    }
}
