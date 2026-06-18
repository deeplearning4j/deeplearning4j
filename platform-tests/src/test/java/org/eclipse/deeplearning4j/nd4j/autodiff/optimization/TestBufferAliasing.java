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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for buffer aliasing optimization (in-place execution of unary elementwise ops).
 *
 * The NativePlanCompiler marks UNARY_ELEMENTWISE | FULLY_WRITING ops for in-place execution
 * when their single input has only one consumer. This reduces peak memory by reusing the
 * input buffer as the output buffer. These tests verify correctness across all DSP modes.
 */
@Tag(TagNames.SAMEDIFF)
public class TestBufferAliasing {

    /**
     * Chain of unary elementwise ops: tanh -> abs -> neg
     * Each op's input has only one consumer, so all should be candidates for in-place execution.
     * Verify the output matches manual computation.
     */
    @ParameterizedTest(name = "unaryChain_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    void testUnaryChainInPlace(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, -1);
        SDVariable t = sd.math().tanh("tanh_out", x);
        SDVariable a = sd.math().abs("abs_out", t);
        SDVariable n = sd.math().neg("output", a);
        sd.setOutputs("output");
        sd.setGraphExecutionMode(mode);

        INDArray input = Nd4j.randn(DataType.FLOAT, 4, 8).muli(2.0);

        // Compute expected result manually using dup() to avoid in-place mutation
        INDArray expected = Nd4j.math().tanh(input.dup());
        expected = Nd4j.math().abs(expected.dup());
        expected = expected.neg();

        // Run through DSP
        INDArray result = sd.output(Map.of("x", input), "output").get("output");

        assertEquals(expected.shape()[0], result.shape()[0], mode + ": shape mismatch dim 0");
        assertEquals(expected.shape()[1], result.shape()[1], mode + ": shape mismatch dim 1");

        // Check values match within tolerance
        double maxDiff = expected.sub(result).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-5, mode + ": max diff = " + maxDiff + " (expected < 1e-5)");
    }

    /**
     * Branching: one input feeds two ops. Neither should be in-place since the input
     * has consumer_count > 1. Verify correctness is maintained.
     */
    @ParameterizedTest(name = "branchingNotInPlace_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    void testBranchingNotInPlace(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, -1);
        SDVariable t = sd.math().tanh("tanh_out", x);
        // Both abs and neg consume tanh_out -> consumer_count=2, no in-place for either
        SDVariable a = sd.math().abs("abs_out", t);
        SDVariable n = sd.math().neg("neg_out", t);
        SDVariable out = a.add("output", n);
        sd.setOutputs("output");
        sd.setGraphExecutionMode(mode);

        INDArray input = Nd4j.randn(DataType.FLOAT, 4, 8).muli(2.0);

        // Expected: tanh(x) -> abs(tanh(x)) + neg(tanh(x))
        // Use dup() to prevent in-place mutation of shared intermediate
        INDArray tanhX = Nd4j.math().tanh(input.dup());
        INDArray expected = Nd4j.math().abs(tanhX.dup()).add(tanhX.dup().neg());

        INDArray result = sd.output(Map.of("x", input), "output").get("output");

        double maxDiff = expected.sub(result).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-5, mode + ": max diff = " + maxDiff);
    }

    /**
     * Multiple steps with varying shapes — verify in-place doesn't cause stale buffers.
     */
    @ParameterizedTest(name = "multiStep_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "TRITON", "CUDA_GRAPHS"})
    void testMultiStepInPlace(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable t = sd.math().tanh("t", x);
        SDVariable s = sd.nn().sigmoid("output", t);
        sd.setOutputs("output");
        sd.setGraphExecutionMode(mode);

        int errorCount = 0;
        for (int step = 0; step < 8; step++) {
            int batchSize = (step < 3) ? 4 : 1;
            INDArray input = Nd4j.randn(DataType.FLOAT, batchSize, 16);
            // dup() to prevent in-place mutation by Nd4j ops
            INDArray expected = Nd4j.math().tanh(input.dup());
            expected = Nd4j.nn().sigmoid(expected.dup());

            INDArray result = sd.output(Map.of("x", input), "output").get("output");
            double maxDiff = expected.sub(result).amaxNumber().doubleValue();
            if (maxDiff >= 1e-5) {
                errorCount++;
            }
        }
        assertEquals(0, errorCount, mode + ": " + errorCount + "/8 steps had wrong results");
    }
}
