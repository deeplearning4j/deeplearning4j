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

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanCompiler;
import org.nd4j.autodiff.samediff.execution.ForwardExecutionDAG;
import org.nd4j.autodiff.samediff.execution.ForwardExecutionDAGBuilder;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Shared test utilities for native C++ graph executor tests.
 *
 * Provides helper methods for creating test graphs, compiling plans,
 * and verifying dual-execution parity between Java and native executors.
 */
public class NativeExecutorTestUtils {

    /**
     * Create a simple add graph: z = x + y
     */
    public static SameDiff createAddGraph() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, -1);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, -1, -1);
        SDVariable z = x.add(y);
        z.rename("z");
        return sd;
    }

    /**
     * Create a matmul graph: z = x @ w + b
     */
    public static SameDiff createMatMulGraph(int inputDim, int outputDim) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, inputDim);
        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, inputDim, outputDim).mul(0.01));
        SDVariable b = sd.var("b", Nd4j.zeros(DataType.FLOAT, 1, outputDim));
        SDVariable z = sd.mmul("matmul", x, w);
        SDVariable out = z.add("z", b);
        return sd;
    }

    /**
     * Create a chain graph: z = relu(sigmoid(x + y) * w)
     */
    public static SameDiff createChainGraph() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable y = sd.var("y", Nd4j.ones(DataType.FLOAT, 1, 4));
        SDVariable w = sd.var("w", Nd4j.ones(DataType.FLOAT, 1, 4).mul(0.5));
        SDVariable sum = x.add(y);
        SDVariable sig = sd.nn().sigmoid(sum);
        SDVariable mul = sig.mul(w);
        SDVariable out = sd.nn().relu("z", mul, 0.0);
        return sd;
    }

    /**
     * Create a diamond graph:
     *   x → [a = x+1, b = x*2] → c = a + b
     */
    public static SameDiff createDiamondGraph() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable one = sd.constant("one", Nd4j.ones(DataType.FLOAT, 1, 4));
        SDVariable two = sd.constant("two", Nd4j.ones(DataType.FLOAT, 1, 4).mul(2));
        SDVariable a = x.add("a", one);
        SDVariable b = x.mul("b", two);
        SDVariable c = a.add("c", b);
        return sd;
    }

    /**
     * Create a fan-out graph:
     *   x → [a = x+1, b = x*2, c = relu(x)]
     */
    public static SameDiff createFanOutGraph() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable one = sd.constant("one", Nd4j.ones(DataType.FLOAT, 1, 4));
        SDVariable two = sd.constant("two", Nd4j.ones(DataType.FLOAT, 1, 4).mul(2));
        SDVariable a = x.add("a", one);
        SDVariable b = x.mul("b", two);
        SDVariable c = sd.nn().relu("c", x, 0.0);
        return sd;
    }

    /**
     * Compile a SameDiff graph into a DynamicShapePlan.
     */
    public static DynamicShapePlan compilePlan(SameDiff sd, String... outputs) {
        Set<String> outputSet = new LinkedHashSet<>(Arrays.asList(outputs));
        ForwardExecutionDAGBuilder builder = new ForwardExecutionDAGBuilder(sd);
        ForwardExecutionDAG dag = builder.buildForwardDAG(outputSet);
        return DynamicShapePlanCompiler.compile(sd, dag, outputSet);
    }

    /**
     * Execute a SameDiff graph with the Java executor and return results.
     */
    public static Map<String, INDArray> executeJava(SameDiff sd, Map<String, INDArray> placeholders,
                                                     String... outputs) {
        return sd.output(placeholders, outputs);
    }

    /**
     * Assert that two INDArrays are equal within tolerance.
     */
    public static void assertArrayEquals(INDArray expected, INDArray actual, double tolerance, String message) {
        assertNotNull(expected, message + " - expected is null");
        assertNotNull(actual, message + " - actual is null");
        org.junit.jupiter.api.Assertions.assertArrayEquals(expected.shape(), actual.shape(),
                message + " - shape mismatch: " + Arrays.toString(expected.shape()) +
                        " vs " + Arrays.toString(actual.shape()));
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        assertTrue(maxDiff <= tolerance,
                message + " - max diff " + maxDiff + " exceeds tolerance " + tolerance);
    }

    /**
     * Verify that serialization produces a valid DSP1 binary.
     */
    public static void assertValidSerialization(DynamicShapePlan plan) {
        byte[] data = plan.serialize();
        assertNotNull(data, "Serialized plan is null");
        assertTrue(data.length >= 24, "Serialized plan too small: " + data.length);
        assertTrue(DynamicShapePlan.isValidSerializedPlan(data), "Invalid DSP1 header");

        ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        int magic = buf.getInt();
        assertEquals(0x44535031, magic, "Wrong magic");
        int version = buf.getInt();
        assertEquals(1, version, "Wrong version");
        int numSlots = buf.getInt();
        assertEquals(plan.getSlots().length, numSlots, "Slot count mismatch");
        int totalOutputSlots = buf.getInt();
        assertEquals(plan.getTotalOutputSlots(), totalOutputSlots, "Output slot count mismatch");
        int numExternalInputs = buf.getInt();
        assertEquals(plan.getExternalInputKeys().length, numExternalInputs, "External input count mismatch");
        int numRequestedOutputs = buf.getInt();
        assertEquals(plan.getOutputNameToSlotIndex().size(), numRequestedOutputs, "Requested output count mismatch");
    }
}
