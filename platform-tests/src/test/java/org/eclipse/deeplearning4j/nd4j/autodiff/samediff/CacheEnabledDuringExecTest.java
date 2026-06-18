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

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.memory.ArrayCacheMemoryMgr;
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
 * Tests that ArrayCacheMemoryMgr cache staying enabled during executeOperations()
 * does not corrupt output values.
 *
 * Background: The old code disabled the cache before executeOperations() with
 * ArrayCacheMemoryMgr.setEnableCache(false). The new code keeps the cache enabled
 * (only resetting growth factor to 1.0). If cache-enabled execution produces wrong
 * results (e.g., stale data from cached buffers), this test will catch it.
 *
 * System property knob: -Dnd4j.test.cacheEnabled=true/false to force cache state.
 */
@DisplayName("ArrayCacheMemoryMgr cache enabled during executeOperations isolation test")
public class CacheEnabledDuringExecTest extends BaseNd4jTestWithBackends {

    private static final Logger log = LoggerFactory.getLogger(CacheEnabledDuringExecTest.class);
    private static final double TOL = 1e-4;

    private boolean savedCacheState;
    private double savedGrowthFactor;

    @Override
    public char ordering() {
        return 'c';
    }

    @BeforeEach
    public void saveState() {
        savedCacheState = ArrayCacheMemoryMgr.isCacheEnabled();
        savedGrowthFactor = ArrayCacheMemoryMgr.getGrowthFactor().get();
    }

    @AfterEach
    public void restoreState() {
        ArrayCacheMemoryMgr.setEnableCache(savedCacheState);
        ArrayCacheMemoryMgr.setGrowthFactor(savedGrowthFactor);
    }

    /**
     * Build a simple decoder-layer-like graph: matmul + add bias.
     * output = input @ weights + bias
     */
    private SameDiff buildDecoderGraph(int inputDim, int outputDim) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, inputDim);
        SDVariable weights = sd.var("weights", Nd4j.randn(DataType.FLOAT, inputDim, outputDim).mul(0.01));
        SDVariable bias = sd.var("bias", Nd4j.zeros(DataType.FLOAT, outputDim));
        SDVariable mm = sd.mmul("matmul_out", input, weights);
        SDVariable out = mm.add("output", bias);
        return sd;
    }

    @Test
    @DisplayName("Cache enabled vs disabled produces identical results across multiple calls")
    public void testCacheEnabledVsDisabledCorrectness() {
        int inputDim = 64;
        int outputDim = 32;
        int batchSize = 4;
        int numIterations = 10;

        // Build two identical graphs with the same weights
        SameDiff sdRef = buildDecoderGraph(inputDim, outputDim);
        SameDiff sdTest = SameDiff.create();
        SDVariable input2 = sdTest.placeHolder("input", DataType.FLOAT, -1, inputDim);
        // Copy weights from reference
        INDArray refWeights = sdRef.getVariable("weights").getArr();
        INDArray refBias = sdRef.getVariable("bias").getArr();
        SDVariable weights2 = sdTest.var("weights", refWeights.dup());
        SDVariable bias2 = sdTest.var("bias", refBias.dup());
        SDVariable mm2 = sdTest.mmul("matmul_out", input2, weights2);
        SDVariable out2 = mm2.add("output", bias2);

        for (int i = 0; i < numIterations; i++) {
            INDArray inputArr = Nd4j.randn(DataType.FLOAT, batchSize, inputDim);

            // Reference: cache disabled
            ArrayCacheMemoryMgr.setEnableCache(false);
            ArrayCacheMemoryMgr.setGrowthFactor(1.0);
            Map<String, INDArray> refPlaceholders = new LinkedHashMap<>();
            refPlaceholders.put("input", inputArr.dup());
            Map<String, INDArray> refResult = sdRef.output(refPlaceholders, "output");
            INDArray refOutput = refResult.get("output");

            // Test: cache enabled (the new behavior)
            ArrayCacheMemoryMgr.setEnableCache(true);
            ArrayCacheMemoryMgr.setGrowthFactor(1.0);
            Map<String, INDArray> testPlaceholders = new LinkedHashMap<>();
            testPlaceholders.put("input", inputArr.dup());
            Map<String, INDArray> testResult = sdTest.output(testPlaceholders, "output");
            INDArray testOutput = testResult.get("output");

            log.info("Iteration {}: refOutput[0]={}, testOutput[0]={}", i,
                    refOutput.getFloat(0), testOutput.getFloat(0));

            // They must be numerically identical
            double maxDiff = refOutput.sub(testOutput).amaxNumber().doubleValue();
            log.info("Iteration {}: max absolute difference = {}", i, maxDiff);
            assertTrue(maxDiff < TOL,
                    "Iteration " + i + ": cache-enabled output diverges from cache-disabled. maxDiff=" + maxDiff);
        }
    }

    @Test
    @DisplayName("Cache enabled with growth factor 1.0 does not leak stale data")
    public void testCacheEnabledNoStaleData() {
        // This simulates the decode loop scenario where the same graph is called repeatedly
        // with different inputs. If cached buffers contain stale data from previous calls,
        // outputs will be wrong.
        int dim = 32;

        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, dim);
        SDVariable weights = sd.var("weights", Nd4j.eye(dim).castTo(DataType.FLOAT));
        SDVariable out = sd.mmul("output", input, weights);

        // With identity weights, output should equal input
        ArrayCacheMemoryMgr.setEnableCache(true);
        ArrayCacheMemoryMgr.setGrowthFactor(1.0);

        for (int i = 0; i < 20; i++) {
            float fillVal = (float) (i + 1);
            INDArray inputArr = Nd4j.ones(DataType.FLOAT, 1, dim).mul(fillVal);

            Map<String, INDArray> placeholders = new LinkedHashMap<>();
            placeholders.put("input", inputArr);
            Map<String, INDArray> result = sd.output(placeholders, "output");
            INDArray output = result.get("output");

            // Output should be all fillVal (identity matmul)
            double maxDiff = output.sub(inputArr).amaxNumber().doubleValue();
            log.info("Step {}: expected fill={}, got first={}, maxDiff={}", i, fillVal,
                    output.getFloat(0), maxDiff);
            assertTrue(maxDiff < TOL,
                    "Step " + i + ": stale data detected. Expected all " + fillVal +
                    " but maxDiff=" + maxDiff + ". First value=" + output.getFloat(0));
        }
    }

    @Test
    @DisplayName("System property knob controls cache state")
    public void testSystemPropertyKnob() {
        String propValue = System.getProperty("nd4j.test.cacheEnabled");
        if (propValue == null) {
            log.info("nd4j.test.cacheEnabled not set, running with both states");
            // Run with both states and verify both produce correct results
            for (boolean cacheEnabled : new boolean[]{true, false}) {
                ArrayCacheMemoryMgr.setEnableCache(cacheEnabled);
                ArrayCacheMemoryMgr.setGrowthFactor(1.0);

                SameDiff sd = buildDecoderGraph(16, 8);
                INDArray input = Nd4j.randn(DataType.FLOAT, 2, 16);
                Map<String, INDArray> ph = Collections.singletonMap("input", input);
                Map<String, INDArray> result = sd.output(ph, "output");
                INDArray output = result.get("output");
                assertNotNull(output, "Output should not be null with cacheEnabled=" + cacheEnabled);
                assertFalse(output.isNaN().any(), "Output contains NaN with cacheEnabled=" + cacheEnabled);
                assertFalse(output.isInfinite().any(), "Output contains Inf with cacheEnabled=" + cacheEnabled);
                log.info("cacheEnabled={}: output OK, shape={}", cacheEnabled, java.util.Arrays.toString(output.shape()));
            }
        } else {
            boolean forced = Boolean.parseBoolean(propValue);
            log.info("nd4j.test.cacheEnabled forced to {}", forced);
            ArrayCacheMemoryMgr.setEnableCache(forced);
            ArrayCacheMemoryMgr.setGrowthFactor(1.0);

            SameDiff sd = buildDecoderGraph(16, 8);
            INDArray input = Nd4j.randn(DataType.FLOAT, 2, 16);
            Map<String, INDArray> ph = Collections.singletonMap("input", input);
            Map<String, INDArray> result = sd.output(ph, "output");
            INDArray output = result.get("output");
            assertNotNull(output);
            assertFalse(output.isNaN().any(), "Output contains NaN with forced cacheEnabled=" + forced);
            assertFalse(output.isInfinite().any(), "Output contains Inf with forced cacheEnabled=" + forced);
        }
    }

    @Test
    @DisplayName("outputDirect with cache enabled matches output() results")
    public void testOutputDirectWithCacheEnabled() {
        // Steps 0-1 use output(), steps 2+ use outputDirect() in the decode loop.
        // Both must produce the same results with cache enabled.
        int dim = 16;

        SameDiff sd = buildDecoderGraph(dim, dim);
        ArrayCacheMemoryMgr.setEnableCache(true);
        ArrayCacheMemoryMgr.setGrowthFactor(1.0);

        for (int step = 0; step < 10; step++) {
            INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, dim);
            Map<String, INDArray> ph = Collections.singletonMap("input", inputArr.dup());
            Map<String, INDArray> phDirect = Collections.singletonMap("input", inputArr.dup());

            // output() path (steps 0-1)
            Map<String, INDArray> resultOutput = sd.output(ph, "output");

            // outputDirect() path (steps 2+)
            Map<String, INDArray> resultDirect = sd.outputDirect(phDirect, "output");

            INDArray fromOutput = resultOutput.get("output");
            INDArray fromDirect = resultDirect.get("output");

            double maxDiff = fromOutput.sub(fromDirect).amaxNumber().doubleValue();
            log.info("Step {}: output() vs outputDirect() maxDiff={}", step, maxDiff);
            assertTrue(maxDiff < TOL,
                    "Step " + step + ": output() and outputDirect() diverge with cache enabled. maxDiff=" + maxDiff);
        }
    }
}
