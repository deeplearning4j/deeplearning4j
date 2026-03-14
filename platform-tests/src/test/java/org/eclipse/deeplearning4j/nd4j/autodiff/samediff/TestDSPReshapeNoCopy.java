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
import org.junit.jupiter.api.DisplayName;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests DSP (Dynamic Shape Plan) execution correctness for reshape operations.
 *
 * reshape is remapped to reshape_no_copy in the DSP compiler to enable
 * zero-copy view creation when possible. This requires:
 * 1. DynamicShapePlanCompiler: reshape → reshape_no_copy iArgs remapping + op replacement
 * 2. DynamicShapePlanExecutor: ARRAY_COPY_OFFSET_INPUT_0 flag handling (view creation)
 * 3. C++ reshape_no_copy: defensive filtering of order markers (-99/-102)
 *
 * These tests verify numerical correctness through DSP for various reshape patterns.
 */
public class TestDSPReshapeNoCopy extends BaseNd4jTestWithBackends {

    private static final Logger log = LoggerFactory.getLogger(TestDSPReshapeNoCopy.class);
    private static final double TOL = 1e-5;

    @Override
    public char ordering() {
        return 'c';
    }

    /**
     * Test simple 2D→2D reshape through DSP.
     * Input [2,6] → reshape [3,4]. Verifies data preservation.
     */
    @Test
    @DisplayName("DSP: reshape 2D→2D matches standard execution")
    public void testReshape2Dto2D() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 6);
        SDVariable reshaped = sd.reshape("reshaped", x, 3, 4);
        // Add a downstream op to verify reshape output is usable
        SDVariable out = reshaped.mul("out", 2.0);

        INDArray input = Nd4j.linspace(1, 12, 12, DataType.FLOAT).reshape(2, 6);

        // Standard execution
        Map<String, INDArray> standardResult = sd.output(Map.of("x", input), "out");
        INDArray expected = standardResult.get("out").dup();

        // DSP execution
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
        INDArray actual = dspResult.get("out");

        assertNotNull(actual, "DSP output should not be null");
        assertArrayEquals(expected.shape(), actual.shape(), "Shapes should match");
        assertEquals(expected, actual, "DSP reshape 2D→2D should match standard execution");
        log.info("testReshape2Dto2D PASSED: expected={}, actual={}", expected.shapeInfoToString(), actual.shapeInfoToString());
    }

    /**
     * Test 3D→2D reshape (rank reduction) through DSP.
     * Input [2,3,4] → reshape [6,4]. Common in attention/MLP layers.
     */
    @Test
    @DisplayName("DSP: reshape 3D→2D matches standard execution")
    public void testReshape3Dto2D() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3, 4);
        SDVariable reshaped = sd.reshape("reshaped", x, 6, 4);
        SDVariable out = reshaped.add("out", 1.0);

        INDArray input = Nd4j.linspace(1, 24, 24, DataType.FLOAT).reshape(2, 3, 4);

        Map<String, INDArray> standardResult = sd.output(Map.of("x", input), "out");
        INDArray expected = standardResult.get("out").dup();

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
        INDArray actual = dspResult.get("out");

        assertNotNull(actual);
        assertArrayEquals(expected.shape(), actual.shape());
        assertEquals(expected, actual, "DSP reshape 3D→2D should match standard execution");
    }

    /**
     * Test reshape with -1 dimension inference through DSP.
     * Input [2,3,4] → reshape [2,-1]. The -1 should resolve to 12.
     */
    @Test
    @DisplayName("DSP: reshape with -1 inference matches standard execution")
    public void testReshapeWithNeg1() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3, 4);
        SDVariable reshaped = sd.reshape("reshaped", x, 2, -1);
        SDVariable out = reshaped.mul("out", 3.0);

        INDArray input = Nd4j.linspace(1, 24, 24, DataType.FLOAT).reshape(2, 3, 4);

        Map<String, INDArray> standardResult = sd.output(Map.of("x", input), "out");
        INDArray expected = standardResult.get("out").dup();

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
        INDArray actual = dspResult.get("out");

        assertNotNull(actual);
        assertArrayEquals(new long[]{2, 12}, actual.shape(), "Shape with -1 should resolve to [2,12]");
        assertEquals(expected, actual, "DSP reshape with -1 should match standard execution");
    }

    /**
     * Test reshape → matmul chain through DSP.
     * This is the critical pattern in transformer decode: reshape attention output,
     * then multiply by output projection weight.
     * Input [1,3,4] → reshape [3,4] → matmul [4,8] → [3,8]
     */
    @Test
    @DisplayName("DSP: reshape → matmul chain matches standard execution")
    public void testReshapeMatmulChain() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 3, 4);
        SDVariable reshaped = sd.reshape("reshaped", x, 3, 4);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 8));
        SDVariable out = sd.mmul("out", reshaped, w);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 3, 4);

        Map<String, INDArray> standardResult = sd.output(Map.of("x", input), "out");
        INDArray expected = standardResult.get("out").dup();

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
        INDArray actual = dspResult.get("out");

        assertNotNull(actual);
        assertArrayEquals(new long[]{3, 8}, actual.shape());
        assertTrue(expected.equalsWithEps(actual, 1e-4),
                "DSP reshape→matmul chain output should match within tolerance");
    }

    /**
     * Test multiple reshape ops in a single graph through DSP.
     * Input [2,12] → reshape [2,3,4] → add bias → reshape [6,4] → output
     */
    @Test
    @DisplayName("DSP: multiple reshapes in chain match standard execution")
    public void testMultipleReshapes() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 12);
        SDVariable r1 = sd.reshape("r1", x, 2, 3, 4);
        SDVariable bias = sd.constant("bias", Nd4j.ones(DataType.FLOAT, 2, 3, 4));
        SDVariable added = r1.add("added", bias);
        SDVariable out = sd.reshape("out", added, 6, 4);

        INDArray input = Nd4j.linspace(1, 24, 24, DataType.FLOAT).reshape(2, 12);

        Map<String, INDArray> standardResult = sd.output(Map.of("x", input), "out");
        INDArray expected = standardResult.get("out").dup();

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
        INDArray actual = dspResult.get("out");

        assertNotNull(actual);
        assertArrayEquals(new long[]{6, 4}, actual.shape());
        assertEquals(expected, actual, "DSP multiple reshapes should match standard execution");
    }

    /**
     * Test repeated DSP execution (multiple steps) with reshape.
     * Verifies that cached views don't cause stale data on subsequent steps.
     */
    @Test
    @DisplayName("DSP: reshape correctness over multiple execution steps")
    public void testReshapeMultipleSteps() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 12);
        SDVariable reshaped = sd.reshape("reshaped", x, 3, 4);
        SDVariable out = reshaped.add("out", 0.5);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        for (int step = 0; step < 5; step++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, 12).mul(step + 1);

            // Compute expected via direct Nd4j ops
            INDArray expectedReshaped = input.reshape(3, 4);
            INDArray expected = expectedReshaped.add(0.5);

            Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
            INDArray actual = dspResult.get("out");

            assertNotNull(actual, "Step " + step + ": DSP output should not be null");
            assertArrayEquals(new long[]{3, 4}, actual.shape(), "Step " + step + ": shape mismatch");
            assertTrue(expected.equalsWithEps(actual, 1e-4),
                    "Step " + step + ": DSP reshape should match expected. " +
                    "Expected first 5: " + expected.get(Nd4j.createFromArray(0, 0)) +
                    ", Actual first 5: " + actual.get(Nd4j.createFromArray(0, 0)));
        }
    }

    /**
     * Test reshape followed by concat through DSP.
     * This is the pattern that was failing: reshape_no_copy produces a view,
     * concat uses it as input. Without ARRAY_COPY_OFFSET_INPUT_0 handling,
     * concat gets a fresh buffer instead of a view → wrong shape propagation.
     */
    @Test
    @DisplayName("DSP: reshape → concat chain matches standard execution")
    public void testReshapeConcatChain() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable r1 = sd.reshape("r1", x, 2, 4);
        SDVariable constant = sd.constant("c", Nd4j.ones(DataType.FLOAT, 2, 4));
        SDVariable out = sd.concat("out", 0, r1, constant);

        INDArray input = Nd4j.linspace(1, 8, 8, DataType.FLOAT).reshape(1, 8);

        Map<String, INDArray> standardResult = sd.output(Map.of("x", input), "out");
        INDArray expected = standardResult.get("out").dup();

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
        INDArray actual = dspResult.get("out");

        assertNotNull(actual);
        assertArrayEquals(new long[]{4, 4}, actual.shape(), "Concat output should be [4,4]");
        assertEquals(expected, actual, "DSP reshape→concat should match standard execution");
    }

    /**
     * Test 4D reshape typical in attention heads: [1,12,64] → [1,3,4,64].
     * ONNX attention blocks reshape Q/K/V from [batch,seq,hidden] to [batch,heads,seq,head_dim].
     */
    @Test
    @DisplayName("DSP: attention-style 3D→4D reshape matches standard execution")
    public void testAttentionReshape() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 12, 64);
        SDVariable reshaped = sd.reshape("reshaped", x, 1, 3, 4, 64);
        SDVariable out = reshaped.mul("out", 0.125); // scale factor

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 12, 64);

        Map<String, INDArray> standardResult = sd.output(Map.of("x", input), "out");
        INDArray expected = standardResult.get("out").dup();

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), "out");
        INDArray actual = dspResult.get("out");

        assertNotNull(actual);
        assertArrayEquals(new long[]{1, 3, 4, 64}, actual.shape());
        assertTrue(expected.equalsWithEps(actual, 1e-4),
                "DSP attention reshape should match standard execution");
    }
}
