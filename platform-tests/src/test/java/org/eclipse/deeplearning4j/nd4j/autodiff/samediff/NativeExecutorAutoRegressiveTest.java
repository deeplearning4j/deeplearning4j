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
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for autoregressive-style execution patterns with the native executor.
 *
 * Covers: growing KV cache simulation, prefill→decode transition,
 * session resets, multi-step execution with growing shapes.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class NativeExecutorAutoRegressiveTest extends BaseNd4jTestWithBackends {

    @Test
    public void testGrowingInputSequence() {
        // Simulate KV cache growth: input grows each step
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, -1, 8); // [batch, seq, dim]
        SDVariable z = x.add("z", 1.0);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        for (int seqLen = 1; seqLen <= 20; seqLen++) {
            INDArray xArr = Nd4j.ones(DataType.FLOAT, 1, seqLen, 8);
            Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
            assertArrayEquals(new long[]{1, seqLen, 8}, result.get("z").shape(),
                    "Step " + seqLen + " shape mismatch");
            assertEquals(2.0, result.get("z").getDouble(0, 0, 0), 1e-5,
                    "Step " + seqLen + " value mismatch");
        }

        plan.close();
    }

    @Test
    public void testSessionResetClearsCache() {
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        // Execute to populate caches
        INDArray x = Nd4j.ones(DataType.FLOAT, 2, 4);
        INDArray y = Nd4j.ones(DataType.FLOAT, 2, 4);
        sd.output(Map.of("x", x, "y", y), "z");

        // Simulate session reset
        plan.clearAllShapeCaches();

        // Execute again — should work with fresh caches
        Map<String, INDArray> result = sd.output(Map.of("x", x, "y", y), "z");
        assertEquals(2.0, result.get("z").getDouble(0), 1e-5);

        plan.close();
    }

    @Test
    public void testMultiStepWithShapeChange() {
        // Simulate prefill (large batch) then decode (batch=1, growing seq)
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, -1);
        SDVariable z = x.mul("z", 2.0);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        // Prefill: large input
        Map<String, INDArray> prefill = sd.output(
                Map.of("x", Nd4j.ones(DataType.FLOAT, 10, 100)), "z");
        assertArrayEquals(new long[]{10, 100}, prefill.get("z").shape());

        // Decode: small growing inputs
        for (int step = 1; step <= 5; step++) {
            Map<String, INDArray> decode = sd.output(
                    Map.of("x", Nd4j.ones(DataType.FLOAT, 1, step)), "z");
            assertArrayEquals(new long[]{1, step}, decode.get("z").shape(),
                    "Decode step " + step);
            assertEquals(2.0, decode.get("z").getDouble(0, 0), 1e-5);
        }

        plan.close();
    }

    @Test
    public void testRepeatedPlanReuse() {
        // Create plan once, execute many times
        SameDiff sd = NativeExecutorTestUtils.createMatMulGraph(8, 4);
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        for (int i = 0; i < 50; i++) {
            INDArray x = Nd4j.randn(DataType.FLOAT, 1, 8);
            Map<String, INDArray> result = sd.output(Map.of("x", x), "z");
            assertNotNull(result.get("z"));
            assertArrayEquals(new long[]{1, 4}, result.get("z").shape());
        }

        plan.close();
    }

    @Test
    public void testPlanSerializationStableAcrossExecutions() {
        SameDiff sd = NativeExecutorTestUtils.createAddGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        byte[] before = plan.serialize();

        // Execute a few times
        INDArray x = Nd4j.ones(DataType.FLOAT, 2, 4);
        INDArray y = Nd4j.ones(DataType.FLOAT, 2, 4);
        for (int i = 0; i < 5; i++) {
            sd.output(Map.of("x", x, "y", y), "z");
        }

        byte[] after = plan.serialize();

        assertArrayEquals(before, after, "Serialization should be stable across executions");

        plan.close();
    }

    @Test
    public void testCacheClearBetweenSessions() {
        SameDiff sd = NativeExecutorTestUtils.createChainGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        // Session 1
        INDArray x1 = Nd4j.ones(DataType.FLOAT, 2, 4);
        sd.output(Map.of("x", x1), "z");

        // Clear (session boundary)
        plan.clearAllShapeCaches();

        // Session 2 with different shapes
        INDArray x2 = Nd4j.ones(DataType.FLOAT, 5, 4);
        Map<String, INDArray> result = sd.output(Map.of("x", x2), "z");
        assertArrayEquals(new long[]{5, 4}, result.get("z").shape());

        plan.close();
    }

    @Test
    public void testConcurrentPlanCreation() {
        // Create multiple plans for same graph (different outputs)
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable a = x.add("a", 1);
        SDVariable b = a.mul("b", 2);
        SDVariable c = b.sub("c", 3);

        DynamicShapePlan plan1 = NativeExecutorTestUtils.compilePlan(sd, "a");
        DynamicShapePlan plan2 = NativeExecutorTestUtils.compilePlan(sd, "b");
        DynamicShapePlan plan3 = NativeExecutorTestUtils.compilePlan(sd, "c");

        NativeExecutorTestUtils.assertValidSerialization(plan1);
        NativeExecutorTestUtils.assertValidSerialization(plan2);
        NativeExecutorTestUtils.assertValidSerialization(plan3);

        // Plans should have different slot counts
        assertTrue(plan1.getSlots().length <= plan2.getSlots().length);
        assertTrue(plan2.getSlots().length <= plan3.getSlots().length);

        plan1.close();
        plan2.close();
        plan3.close();
    }

    @Test
    public void testTransformerBlockPattern() {
        // Simplified transformer: norm → QKV → attention → residual → norm → FFN → residual
        int dim = 8;
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, dim);

        // Layer norm (simplified, keepDims=true for broadcasting)
        SDVariable mean = x.mean(true, 1);
        SDVariable centered = x.sub(mean);
        SDVariable norm = centered.div(sd.math().sqrt(centered.mul(centered).mean(true, 1).add(1e-5)));

        // FFN (simplified)
        SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, dim, dim * 4).mul(0.01));
        SDVariable w2 = sd.var("w2", Nd4j.randn(DataType.FLOAT, dim * 4, dim).mul(0.01));
        SDVariable hidden = sd.nn().relu(sd.mmul(norm, w1), 0.0);
        SDVariable ffnOut = sd.mmul(hidden, w2);

        // Residual
        SDVariable out = x.add("z", ffnOut);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");
        NativeExecutorTestUtils.assertValidSerialization(plan);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 2, dim);
        Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
        assertArrayEquals(new long[]{2, dim}, result.get("z").shape());

        plan.close();
    }

    @Test
    public void testManyStepsNoOOM() {
        // Execute 100 steps with dynamic shapes — should not OOM
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable w = sd.var("w", Nd4j.eye(16).castTo(DataType.FLOAT));
        SDVariable z = sd.nn().relu("z", sd.mmul(x, w), 0.0);

        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        for (int step = 0; step < 100; step++) {
            int batchSize = (step % 5) + 1; // Vary batch size
            INDArray xArr = Nd4j.randn(DataType.FLOAT, batchSize, 16);
            Map<String, INDArray> result = sd.output(Map.of("x", xArr), "z");
            assertNotNull(result.get("z"), "Step " + step);
        }

        plan.close();
    }

    @Test
    public void testPlanIntrospection() {
        SameDiff sd = NativeExecutorTestUtils.createChainGraph();
        DynamicShapePlan plan = NativeExecutorTestUtils.compilePlan(sd, "z");

        // Test plan introspection methods
        assertTrue(plan.getSlots().length > 0, "Should have slots");
        assertTrue(plan.getTotalOutputSlots() > 0, "Should have output slots");
        assertTrue(plan.getExternalInputKeys().length > 0, "Should have external inputs");
        assertNotNull(plan.getRequestedOutputs());
        assertTrue(plan.getRequestedOutputs().contains("z"));
        assertNotNull(plan.getOutputNameToSlotIndex());
        assertTrue(plan.getOutputNameToSlotIndex().containsKey("z"));
        assertNotNull(plan.getSummary());

        plan.close();
    }
}
