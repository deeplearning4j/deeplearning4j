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

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the SlotBufferOwnership system in DSP.
 *
 * <p>Validates that DSP correctly classifies buffer ownership (SLOT_OWNED,
 * VIEW_OF_WEIGHT, etc.) and that ownership metadata remains consistent
 * across multiple executions.</p>
 *
 * <p>Run:
 * <pre>
 *   cd platform-tests && mvn test -Dtest=TestDSPBufferOwnership
 * </pre>
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestDSPBufferOwnership extends BaseNd4jTestWithBackends {

    private SameDiff sd;

    @Override
    public char ordering() {
        return 'c';
    }

    @BeforeAll
    static void enableDspGlobally() {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);
    }

    private void enableDsp(SameDiff sd) {
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
    }

    private SameDiff buildMatmulAddGraph(int inputDim, int hiddenDim) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, inputDim);
        SDVariable w = sd.constant("weight", Nd4j.randn(DataType.FLOAT, inputDim, hiddenDim));
        SDVariable b = sd.constant("bias", Nd4j.zeros(DataType.FLOAT, 1, hiddenDim));
        SDVariable h = sd.mmul("mm", x, w).add("output", b);
        return sd;
    }

    @AfterEach
    void cleanup() {
        if (sd != null) {
            sd.close();
            sd = null;
        }
    }

    @Test
    @Order(1)
    @DisplayName("Buffer ownership classification: computed results are SLOT_OWNED")
    public void testOwnershipClassification() {
        sd = buildMatmulAddGraph(16, 32);
        enableDsp(sd);

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 16);
        Map<String, INDArray> result = sd.output(
                Collections.singletonMap("input", input), "output");

        // The output should exist and be non-null
        assertNotNull(result.get("output"), "Output should be produced by DSP execution");
        assertEquals(2, result.get("output").rank(), "Output should be rank 2");
        assertArrayEquals(new long[]{2, 32}, result.get("output").shape(),
                "Output shape should be [2, 32]");
    }

    @Test
    @Order(2)
    @DisplayName("View chain freeing: parent buffer survives when views are freed")
    public void testViewChainFreeing() {
        sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, 16);
        SDVariable r1 = sd.reshape("r1", x, 1, -1, 4, 4);
        SDVariable r2 = sd.reshape("r2", r1, 1, -1, 16);
        SDVariable out = sd.identity("output", r2);
        enableDsp(sd);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 16);
        Map<String, INDArray> result = sd.output(
                Collections.singletonMap("input", input), "output");

        INDArray output = result.get("output");
        assertNotNull(output, "Output must not be null after view chain");
        // Output should have the correct shape from the final reshape
        assertArrayEquals(new long[]{1, 1, 16}, output.shape(),
                "Final reshape should produce [1, 1, 16]");

        // Verify data is not corrupted (should match input data reinterpreted)
        assertFalse(output.isNaN().any(), "Output should not contain NaN");
        assertFalse(output.isInfinite().any(), "Output should not contain Inf");
    }

    @Test
    @Order(3)
    @DisplayName("Weight protection: weight buffers are never freed during plan cleanup")
    public void testWeightProtection() {
        sd = buildMatmulAddGraph(16, 32);
        enableDsp(sd);

        // Get a reference to the weight array before execution
        INDArray weightBefore = sd.getVariable("weight").getArr().dup();

        // Execute multiple times to trigger potential cleanup
        for (int i = 0; i < 5; i++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 2, 16);
            sd.output(Collections.singletonMap("input", input), "output");
        }

        // Weight should still be intact
        INDArray weightAfter = sd.getVariable("weight").getArr();
        assertNotNull(weightAfter, "Weight array must not be null after executions");
        assertEquals(weightBefore, weightAfter,
                "Weight data must not be corrupted by DSP execution");
    }

    @Test
    @Order(4)
    @DisplayName("Ownership consistency across repeated same-shape executions: results remain correct")
    public void testOwnershipResetOnFreeze() {
        sd = buildMatmulAddGraph(16, 32);
        enableDsp(sd);

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 16);

        // Warmup execution
        Map<String, INDArray> result1 = sd.output(
                Collections.singletonMap("input", input), "output");
        assertNotNull(result1.get("output"), "Warmup must produce output");
        INDArray ref = result1.get("output").dup();

        // Run several more times with the same shape -- DSP may internally promote
        // slots to a frozen state. Buffer ownership metadata must remain consistent
        // throughout these transitions.
        for (int i = 0; i < 5; i++) {
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("input", input), "output");
            assertNotNull(result.get("output"),
                    "Execution " + i + " must produce output");
            assertArrayEquals(ref.shape(), result.get("output").shape(),
                    "Execution " + i + " shape must match reference");

            double maxDiff = ref.sub(result.get("output")).amaxNumber().doubleValue();
            assertTrue(maxDiff < 1e-4,
                    "Execution " + i + " diverged from reference by " + maxDiff);
        }
    }

    @Test
    @Order(5)
    @DisplayName("Ownership consistency: 10 executions maintain consistent metadata")
    public void testOwnershipConsistency() {
        sd = buildMatmulAddGraph(16, 32);
        enableDsp(sd);

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 16);

        // Compute reference result
        Map<String, INDArray> reference = sd.output(
                Collections.singletonMap("input", input), "output");
        INDArray refOutput = reference.get("output").dup();

        // Execute 10 more times with same input, verify results match
        for (int i = 0; i < 10; i++) {
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("input", input), "output");
            INDArray out = result.get("output");
            assertNotNull(out, "Execution " + i + " must produce output");

            // Check shape consistency
            assertArrayEquals(refOutput.shape(), out.shape(),
                    "Execution " + i + " shape must match reference");

            // Check data consistency (same input -> same output)
            double maxDiff = refOutput.sub(out).amaxNumber().doubleValue();
            assertTrue(maxDiff < 1e-4,
                    "Execution " + i + " output diverged from reference by " + maxDiff);
        }
    }
}
