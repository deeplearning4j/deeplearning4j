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
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.*;

/**
 * Tests for the DspStreamGuard (CUDA-only).
 *
 * <p>The DspStreamGuard ensures that the DSP execution stream is properly
 * restored when execution scope exits, even on exceptions. These tests
 * validate that guard semantics work correctly.</p>
 *
 * <p>On CPU builds, these tests are skipped via Assumptions.</p>
 *
 * <p>Run:
 * <pre>
 *   cd platform-tests && mvn test -Dtest=TestDSPStreamGuard
 * </pre>
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestDSPStreamGuard extends BaseNd4jTestWithBackends {

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

    private boolean isCudaBackend() {
        return Nd4j.getBackend().getClass().getSimpleName().contains("Cuda");
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
    @DisplayName("Guard restores previous stream value on scope exit")
    public void testGuardRestoresStream() {
        assumeTrue(isCudaBackend(), "Stream guard test requires CUDA backend");

        sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, 8);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 8, 16));
        sd.mmul("output", x, w);
        enableDsp(sd);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 8);

        // Execute once -- the stream guard should save and restore during DSP execution
        Map<String, INDArray> result = sd.output(
                Collections.singletonMap("input", input), "output");
        assertNotNull(result.get("output"),
                "Execution after stream guard should produce valid output");

        // Execute again -- if stream was not restored, this would fail or produce garbage
        Map<String, INDArray> result2 = sd.output(
                Collections.singletonMap("input", input), "output");
        assertNotNull(result2.get("output"),
                "Second execution must work if stream was properly restored");

        // Results should match (same input, same weights)
        double maxDiff = result.get("output").sub(result2.get("output")).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-4,
                "Two identical executions must produce same result, maxDiff=" + maxDiff);
    }

    @Test
    @Order(2)
    @DisplayName("Guard restores even if scope exits via exception")
    public void testGuardOnException() {
        assumeTrue(isCudaBackend(), "Stream guard test requires CUDA backend");

        sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, 8);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 8, 16));
        sd.mmul("output", x, w);
        enableDsp(sd);

        // Normal execution first
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 8);
        sd.output(Collections.singletonMap("input", input), "output");

        // Try execution with null input -- may cause exception
        try {
            sd.output(Collections.singletonMap("input", (INDArray) null), "output");
        } catch (Exception e) {
            log.info("Expected exception on null input: {}", e.getMessage());
        }

        // Stream guard should have restored -- next valid execution should work
        Map<String, INDArray> result = sd.output(
                Collections.singletonMap("input", input), "output");
        assertNotNull(result.get("output"),
                "Execution after exception must work if stream guard restored properly");
    }

    @Test
    @Order(3)
    @DisplayName("Nested guards: inner restores to middle, outer restores to original")
    public void testNestedGuards() {
        assumeTrue(isCudaBackend(), "Stream guard test requires CUDA backend");

        // Create two separate graphs to simulate nested DSP executions
        sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, 8);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 8, 16));
        sd.mmul("output", x, w);
        enableDsp(sd);

        SameDiff sd2 = SameDiff.create();
        SDVariable x2 = sd2.placeHolder("input", DataType.FLOAT, -1, 16);
        SDVariable w2 = sd2.constant("w", Nd4j.randn(DataType.FLOAT, 16, 4));
        sd2.mmul("output", x2, w2);
        enableDsp(sd2);

        try {
            INDArray input1 = Nd4j.randn(DataType.FLOAT, 1, 8);
            Map<String, INDArray> r1 = sd.output(
                    Collections.singletonMap("input", input1), "output");

            // Use output of first as input to second (nested execution)
            Map<String, INDArray> r2 = sd2.output(
                    Collections.singletonMap("input", r1.get("output")), "output");
            assertNotNull(r2.get("output"), "Nested execution must produce output");
            assertArrayEquals(new long[]{1, 4}, r2.get("output").shape());

            // Execute first graph again -- stream must be properly restored
            Map<String, INDArray> r3 = sd.output(
                    Collections.singletonMap("input", input1), "output");
            assertNotNull(r3.get("output"),
                    "Post-nested execution must work with properly restored stream");
        } finally {
            sd2.close();
        }
    }

    @Test
    @Order(4)
    @DisplayName("CPU build: guard is a no-op and does not crash")
    public void testCpuBuildNoOp() {
        // This test runs on ALL backends -- verifies no crash on CPU
        sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, 8);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 8, 16));
        sd.mmul("output", x, w);
        enableDsp(sd);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 8);
        Map<String, INDArray> result = sd.output(
                Collections.singletonMap("input", input), "output");

        assertNotNull(result.get("output"),
                "Execution must succeed regardless of backend");
        assertArrayEquals(new long[]{1, 16}, result.get("output").shape());

        if (!isCudaBackend()) {
            log.info("Running on CPU backend -- stream guard is a no-op, verified no crash");
        }
    }
}
