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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.BufferAllocation;
import org.nd4j.autodiff.samediff.execution.BufferAllocKind;
import org.nd4j.autodiff.samediff.execution.ExecutionPlan;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

@NativeTag
@Tag(TagNames.SAMEDIFF)
public class PlanBasedExecutionPoolingTest extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPlanBasedExecutionUsesSharedPool(Nd4jBackend backend) throws Exception {
        SameDiff sd = SameDiff.create();
        sd.setPlanBasedExecution(true);

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 4);
        SDVariable y = x.add("y", 1.0);
        SDVariable z = x.mul("z", 2.0);

        INDArray input = Nd4j.linspace(1.0, 4.0, 4, DataType.FLOAT).reshape(1, 4);
        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        placeholders.put("x", input);
        Map<String, INDArray> outputs = sd.output(placeholders, "y", "z");

        assertEquals(input.add(1.0), outputs.get("y"));
        assertEquals(input.mul(2.0), outputs.get("z"));

        InferenceSession session = sd.getOrCreateSession();
        var executor = session.getPlanExecutor();
        assertNotNull(executor, "PlanExecutor should be initialized for plan-based execution");

        ExecutionPlan plan = executor.getCurrentPlan();
        assertNotNull(plan, "ExecutionPlan should be captured in PlanExecutor");

        INDArray[] bufferPool = executor.getBufferPool();
        Map<DataType, DataBuffer> dataTypePools = executor.getDataTypePools();
        assertNotNull(bufferPool, "Buffer pool should be initialized");
        assertNotNull(dataTypePools, "Data type pools should be initialized");

        BufferAllocation yAlloc = plan.getBufferMap().get("y");
        BufferAllocation zAlloc = plan.getBufferMap().get("z");
        assertNotNull(yAlloc, "Missing buffer allocation for y");
        assertNotNull(zAlloc, "Missing buffer allocation for z");

        INDArray yBuf = bufferPool[yAlloc.getIndex()];
        INDArray zBuf = bufferPool[zAlloc.getIndex()];
        assertNotNull(yBuf, "Buffer for y should exist");
        assertNotNull(zBuf, "Buffer for z should exist");

        DataBuffer pool = dataTypePools.get(DataType.FLOAT);
        assertNotNull(pool, "Expected FLOAT pool for plan buffers");

        assertSame(pool, yBuf.data(), "y buffer should be backed by the shared pool");
        assertSame(pool, zBuf.data(), "z buffer should be backed by the shared pool");
        assertNotSame(yBuf, zBuf, "y and z should be distinct views into the pool");
        assertNotSame(outputs.get("y"), yBuf, "Returned outputs must be detached from the pool");
        assertNotSame(outputs.get("z"), zBuf, "Returned outputs must be detached from the pool");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReuseAllocationSharesOffset(Nd4jBackend backend) throws Exception {
        SameDiff sd = SameDiff.create();
        sd.setPlanBasedExecution(true);

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 4);
        SDVariable a = x.add("a", 1.0);
        SDVariable b = x.mul("b", 2.0);
        SDVariable c = a.add("c", b);
        SDVariable d = c.mul("d", 3.0);
        SDVariable e = d.add("e", 4.0);

        INDArray input = Nd4j.linspace(1.0, 4.0, 4, DataType.FLOAT).reshape(1, 4);
        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        placeholders.put("x", input);
        Map<String, INDArray> outputs = sd.output(placeholders, "e");

        INDArray expected = input.mul(2.0).add(input).add(1.0).mul(3.0).add(4.0);
        assertEquals(expected, outputs.get("e"));

        InferenceSession session = sd.getOrCreateSession();
        var executor = session.getPlanExecutor();
        assertNotNull(executor, "PlanExecutor should be initialized for plan-based execution");

        ExecutionPlan plan = executor.getCurrentPlan();
        assertNotNull(plan, "ExecutionPlan should be captured in PlanExecutor");

        BufferAllocation reuseAlloc = null;
        for (BufferAllocation alloc : plan.getAllocations()) {
            if (alloc.getKind() == BufferAllocKind.REUSE) {
                reuseAlloc = alloc;
                break;
            }
        }
        assertNotNull(reuseAlloc, "Expected at least one REUSE allocation in the plan");

        INDArray[] bufferPool = executor.getBufferPool();
        assertNotNull(bufferPool, "Buffer pool should be initialized");

        int reuseIndex = reuseAlloc.getIndex();
        int sourceIndex = reuseAlloc.getReusesBufferIndex();
        INDArray reuseArray = bufferPool[reuseIndex];
        INDArray sourceArray = bufferPool[sourceIndex];

        assertNotNull(reuseArray, "REUSE buffer should be allocated");
        assertNotNull(sourceArray, "Source buffer for REUSE should be allocated");
        assertSame(sourceArray.data(), reuseArray.data(), "REUSE buffer should share the same DataBuffer");
        assertEquals(sourceArray.offset(), reuseArray.offset(), "REUSE buffer should share the same offset");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDynamicShapeOutputReusesBuffer(Nd4jBackend backend) throws Exception {
        SameDiff sd = SameDiff.create();
        sd.setPlanBasedExecution(true);

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 2);
        SDVariable shape = sd.shape(x);
        SDVariable r = sd.reshape("r", x, shape);

        INDArray input = Nd4j.linspace(1.0, 4.0, 4, DataType.FLOAT).reshape(2, 2);
        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        placeholders.put("x", input);
        Map<String, INDArray> out1 = sd.output(placeholders, "r");

        assertEquals(input, out1.get("r"));

        InferenceSession session = sd.getOrCreateSession();
        var executor = session.getPlanExecutor();
        assertNotNull(executor, "PlanExecutor should be initialized for plan-based execution");

        ExecutionPlan plan = executor.getCurrentPlan();
        BufferAllocation rAlloc = plan.getBufferMap().get("r");
        assertNotNull(rAlloc, "Missing buffer allocation for r");

        INDArray[] bufferPool = executor.getBufferPool();
        INDArray first = bufferPool[rAlloc.getIndex()];
        assertNotNull(first, "Dynamic output buffer should be allocated after first run");

        INDArray input2 = Nd4j.linspace(5.0, 8.0, 4, DataType.FLOAT).reshape(2, 2);
        placeholders.put("x", input2);
        Map<String, INDArray> out2 = sd.output(placeholders, "r");
        assertEquals(input2, out2.get("r"));

        INDArray second = bufferPool[rAlloc.getIndex()];
        assertTrue(first == second, "Dynamic output should reuse the same buffer instance across runs");
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
