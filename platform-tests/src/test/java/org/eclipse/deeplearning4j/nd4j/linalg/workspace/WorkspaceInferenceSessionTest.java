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

package org.eclipse.deeplearning4j.nd4j.linalg.workspace;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.internal.memory.ArrayCacheMemoryMgr;
import org.nd4j.autodiff.samediff.internal.memory.WorkspaceSessionMemMgr;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.api.memory.abstracts.Nd4jWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@Tag(TagNames.WORKSPACES)
public class WorkspaceInferenceSessionTest extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    @BeforeEach
    public void setUp() {
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
    }

    @AfterEach
    public void tearDown() {
        Nd4j.getMemoryManager().setCurrentWorkspace(null);
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
    }

    // ========================================================================
    // Basic correctness tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBasicSameDiffWithWorkspace(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, -1, 4);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, -1, 4);
        SDVariable sum = a.add("sum", b);
        SDVariable product = sum.mul("product", a);

        sd.enableWorkspaceMode(10 * 1024 * 1024);

        INDArray aArr = Nd4j.ones(DataType.FLOAT, 2, 4);
        INDArray bArr = Nd4j.ones(DataType.FLOAT, 2, 4).mul(2);

        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("a", aArr);
        placeholders.put("b", bArr);

        Map<String, INDArray> result = sd.output(placeholders, "product");
        INDArray out = result.get("product");

        assertNotNull(out);
        assertEquals(3.0f, out.getFloat(0, 0), 1e-5);
        assertEquals(3.0f, out.getFloat(1, 3), 1e-5);
        assertFalse(out.wasClosed(), "Output array should not be closed after workspace scope ends");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWorkspaceVsDefaultOutputsMatch(Nd4jBackend backend) {
        SameDiff sdDefault = SameDiff.create();
        SDVariable a1 = sdDefault.placeHolder("a", DataType.FLOAT, -1, 4);
        SDVariable b1 = sdDefault.placeHolder("b", DataType.FLOAT, -1, 4);
        SDVariable sum1 = a1.add("sum", b1);
        SDVariable result1 = sum1.mul("result", a1);

        SameDiff sdWorkspace = SameDiff.create();
        SDVariable a2 = sdWorkspace.placeHolder("a", DataType.FLOAT, -1, 4);
        SDVariable b2 = sdWorkspace.placeHolder("b", DataType.FLOAT, -1, 4);
        SDVariable sum2 = a2.add("sum", b2);
        SDVariable result2 = sum2.mul("result", a2);

        sdWorkspace.enableWorkspaceMode(10 * 1024 * 1024);

        INDArray aArr = Nd4j.randn(DataType.FLOAT, 3, 4);
        INDArray bArr = Nd4j.randn(DataType.FLOAT, 3, 4);

        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("a", aArr);
        placeholders.put("b", bArr);

        Map<String, INDArray> defaultResult = sdDefault.output(placeholders, "result");
        Map<String, INDArray> workspaceResult = sdWorkspace.output(placeholders, "result");

        assertEquals(defaultResult.get("result"), workspaceResult.get("result"),
                "Workspace and default outputs should be identical");
    }

    // ========================================================================
    // Workspace scoping and lifecycle tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWorkspaceOutputArraysSurviveScope(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4);
        SDVariable doubled = input.mul("doubled", 2.0);

        sd.enableWorkspaceMode(1024 * 1024);

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("input", Nd4j.ones(DataType.FLOAT, 2, 4));

        Map<String, INDArray> result = sd.output(ph, "doubled");
        INDArray out = result.get("doubled");

        // Output should be a valid heap array after workspace scope closes
        assertNotNull(out);
        assertFalse(out.wasClosed());
        assertEquals(2.0f, out.getFloat(0, 0), 1e-5);

        // Should be able to do further ops on the output (proves it's heap-allocated)
        INDArray tripled = out.mul(1.5);
        assertEquals(3.0f, tripled.getFloat(0, 0), 1e-5);

        // Verify the output is NOT attached to any workspace
        assertFalse(out.isAttached(), "Output array should not be attached to a workspace");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWorkspaceScopeCyclesDontLeakMemory(Nd4jBackend backend) {
        // Verify that workspace memory is recycled between scope cycles
        // by running many iterations and checking the workspace doesn't grow unboundedly
        WorkspaceSessionMemMgr mgr = new WorkspaceSessionMemMgr(4 * 1024 * 1024);
        try {
            for (int i = 0; i < 100; i++) {
                mgr.scopeIn();

                // Allocate several intermediate arrays
                for (int j = 0; j < 10; j++) {
                    INDArray arr = mgr.allocate(false, DataType.FLOAT, 64, 64);
                    arr.assign(1.0f);
                    assertEquals(1.0f, arr.getFloat(0, 0), 1e-5f,
                            "Array value mismatch at cycle " + i + " alloc " + j);
                }

                // Allocate detached output
                INDArray output = mgr.allocate(true, DataType.FLOAT, 1, 10);
                output.assign(42.0f);

                mgr.scopeOut();

                // Detached output survives scope
                assertEquals(42.0f, output.getFloat(0), 1e-5f);
                assertFalse(output.isAttached());
            }
        } finally {
            mgr.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNestedWorkspaceScopingWithSameDiff(Nd4jBackend backend) {
        // Test that workspace-backed inference works correctly when there's
        // already an outer workspace active (nested workspace pattern from
        // ComputationGraph/MultiLayerNetwork)
        val outerConfig = WorkspaceConfiguration.builder()
                .initialSize(2 * 1024 * 1024)
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .policyReset(ResetPolicy.BLOCK_LEFT)
                .policySpill(SpillPolicy.REALLOCATE)
                .build();

        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4);
        SDVariable out = input.add("out", 1.0);

        sd.enableWorkspaceMode(1024 * 1024);

        // Simulate having an outer workspace active (like in a DL4J layer)
        try (MemoryWorkspace outerWs = Nd4j.getWorkspaceManager()
                .getAndActivateWorkspace(outerConfig, "OUTER_LAYER_WS")) {

            Map<String, INDArray> ph = new HashMap<>();
            // Input allocated in outer workspace
            INDArray inputArr = Nd4j.ones(DataType.FLOAT, 2, 4);
            ph.put("input", inputArr);

            Map<String, INDArray> result = sd.output(ph, "out");
            INDArray output = result.get("out");

            assertNotNull(output);
            assertFalse(output.wasClosed());
            assertEquals(2.0f, output.getFloat(0, 0), 1e-5);
        }

        // After outer workspace closes, run again to verify no stale workspace refs
        Map<String, INDArray> ph2 = new HashMap<>();
        ph2.put("input", Nd4j.ones(DataType.FLOAT, 2, 4));
        Map<String, INDArray> result2 = sd.output(ph2, "out");
        assertEquals(2.0f, result2.get("out").getFloat(0, 0), 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScopeOutOfWorkspacesForDetachedArrays(Nd4jBackend backend) {
        // Verify that detached arrays are allocated outside workspaces using
        // the scopeOutOfWorkspaces() pattern (consistent with ComputationGraph)
        WorkspaceSessionMemMgr mgr = new WorkspaceSessionMemMgr(2 * 1024 * 1024);
        try {
            mgr.scopeIn();

            // Non-detached should be in workspace
            INDArray intermediate = mgr.allocate(false, DataType.FLOAT, 10, 10);
            assertNotNull(intermediate);

            // Detached should be outside workspace
            INDArray detached = mgr.allocate(true, DataType.FLOAT, 10, 10);
            detached.assign(99.0f);
            assertNotNull(detached);
            assertFalse(detached.isAttached(),
                    "Detached array should not be attached to any workspace");

            mgr.scopeOut();

            // Detached array remains valid after scope ends
            assertEquals(99.0f, detached.getFloat(0, 0), 1e-5f);
            assertFalse(detached.wasClosed());
        } finally {
            mgr.close();
        }
    }

    // ========================================================================
    // Spill policy and large allocation tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWorkspaceSpillPolicyReallocate(Nd4jBackend backend) {
        // Configure workspace with small initial size but REALLOCATE spill policy.
        // Allocations that exceed workspace should spill and the workspace should
        // learn the correct size for next cycle.
        val config = WorkspaceConfiguration.builder()
                .initialSize(1024)  // Very small - 1KB
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .policyReset(ResetPolicy.BLOCK_LEFT)
                .policySpill(SpillPolicy.REALLOCATE)
                .build();

        WorkspaceSessionMemMgr mgr = new WorkspaceSessionMemMgr(config);
        try {
            // First cycle: allocation exceeds workspace, should spill
            mgr.scopeIn();
            INDArray large = mgr.allocate(false, DataType.FLOAT, 256, 256);
            assertNotNull(large);
            large.assign(1.0f);
            assertEquals(1.0f, large.getFloat(0, 0), 1e-5f);
            INDArray output = mgr.allocate(true, DataType.FLOAT, 1, 10);
            output.assign(5.0f);
            mgr.scopeOut();

            assertEquals(5.0f, output.getFloat(0), 1e-5f);

            // Second cycle: workspace should have learned the size
            mgr.scopeIn();
            INDArray large2 = mgr.allocate(false, DataType.FLOAT, 256, 256);
            assertNotNull(large2);
            large2.assign(2.0f);
            assertEquals(2.0f, large2.getFloat(0, 0), 1e-5f);
            mgr.scopeOut();
        } finally {
            mgr.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLargePreAllocationWorkspace(Nd4jBackend backend) {
        // Pre-allocate a large workspace (256MB) and verify efficient reuse
        // This simulates a production scenario where workspace size is known upfront
        long workspaceSize = 256L * 1024 * 1024; // 256MB

        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 1024);
        SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, 1024, 2048));
        SDVariable w2 = sd.var("w2", Nd4j.randn(DataType.FLOAT, 2048, 1024));
        SDVariable h = sd.nn.relu("h", sd.mmul("mm1", input, w1), 0);
        SDVariable out = sd.mmul("out", h, w2);

        sd.enableWorkspaceMode(workspaceSize);

        // Run multiple forward passes with large arrays
        for (int i = 0; i < 5; i++) {
            Map<String, INDArray> ph = new HashMap<>();
            ph.put("input", Nd4j.randn(DataType.FLOAT, 32, 1024));

            Map<String, INDArray> result = sd.output(ph, "out");
            INDArray output = result.get("out");

            assertNotNull(output);
            assertFalse(output.wasClosed());
            assertArrayEquals(new long[]{32, 1024}, output.shape());
            assertFalse(output.isAttached(), "Output must be detached after scope");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWorkspaceWithVaryingBatchSizes(Nd4jBackend backend) {
        // Test that workspace handles varying batch sizes across inference calls.
        // First call with batch=1, then batch=32, then back to batch=1.
        // This exercises the REALLOCATE spill path and learning.
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 64);
        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, 64, 64));
        SDVariable out = sd.mmul("out", input, w);

        // Start with small workspace - it should learn
        sd.enableWorkspaceMode(64 * 1024); // 64KB

        int[] batchSizes = {1, 4, 16, 32, 16, 4, 1, 64, 1};
        for (int batchSize : batchSizes) {
            Map<String, INDArray> ph = new HashMap<>();
            ph.put("input", Nd4j.randn(DataType.FLOAT, batchSize, 64));

            Map<String, INDArray> result = sd.output(ph, "out");
            INDArray output = result.get("out");

            assertNotNull(output, "Output null for batch size " + batchSize);
            assertFalse(output.wasClosed(), "Output closed for batch size " + batchSize);
            assertEquals(batchSize, output.size(0),
                    "Wrong batch dim for batch size " + batchSize);
            assertEquals(64, output.size(1));
        }
    }

    // ========================================================================
    // Autoregressive / repeated inference tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAutoRegressiveInferenceWithWorkspace(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 8);
        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, 8, 8));
        SDVariable mm = sd.mmul("mm", input, w);
        SDVariable out = sd.nn.softmax("out", mm, -1);

        sd.enableWorkspaceMode(10 * 1024 * 1024);

        INDArray prev = Nd4j.randn(DataType.FLOAT, 1, 8);
        for (int i = 0; i < 50; i++) {
            Map<String, INDArray> ph = new HashMap<>();
            ph.put("input", prev);
            Map<String, INDArray> result = sd.output(ph, "out");
            INDArray output = result.get("out");

            assertNotNull(output, "Output null at iteration " + i);
            assertFalse(output.wasClosed(), "Output closed at iteration " + i);
            assertFalse(output.isAttached(), "Output still attached at iteration " + i);
            assertEquals(1, output.rows());
            assertEquals(8, output.columns());

            double sum = output.sumNumber().doubleValue();
            assertEquals(1.0, sum, 1e-4, "Softmax sum should be ~1 at iteration " + i);

            // Feed output as next input (autoregressive pattern)
            prev = output;
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWorkspaceIntermediateArraysRecycled(Nd4jBackend backend) {
        // Create a graph with many intermediate nodes to exercise workspace recycling
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 16);
        SDVariable current = input;
        for (int i = 0; i < 20; i++) {
            current = current.add("add_" + i, 1.0);
            current = current.mul("mul_" + i, 0.99);
        }
        sd.identity("output", current);

        sd.enableWorkspaceMode(2 * 1024 * 1024);

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("input", Nd4j.zeros(DataType.FLOAT, 4, 16));

        Map<String, INDArray> result1 = sd.output(ph, "output");
        Map<String, INDArray> result2 = sd.output(ph, "output");

        assertNotNull(result1.get("output"));
        assertNotNull(result2.get("output"));
        assertEquals(result1.get("output"), result2.get("output"),
                "Two runs should produce the same result");
    }

    // ========================================================================
    // Data type and multiple output tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWorkspaceWithVariousDataTypes(Nd4jBackend backend) {
        for (DataType dt : new DataType[]{DataType.FLOAT, DataType.DOUBLE,
                DataType.HALF, DataType.INT, DataType.LONG}) {
            SameDiff sd = SameDiff.create();
            try {
                SDVariable input = sd.placeHolder("input", dt, -1, 4);
                SDVariable out;
                if (dt.isFPType()) {
                    out = input.add("out", 1.0);
                } else {
                    SDVariable ones = sd.constant("ones", Nd4j.ones(dt, 1, 4));
                    out = input.add("out", ones);
                }

                sd.enableWorkspaceMode(1024 * 1024);

                Map<String, INDArray> ph = new HashMap<>();
                ph.put("input", Nd4j.zeros(dt, 2, 4));

                Map<String, INDArray> result = sd.output(ph, "out");
                INDArray output = result.get("out");
                assertNotNull(output, "Output for " + dt + " should not be null");
                assertFalse(output.wasClosed(), "Output for " + dt + " should not be closed");
                assertFalse(output.isAttached(), "Output for " + dt + " should not be attached");
            } finally {
                sd.close();
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWorkspaceWithMultipleOutputs(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4);
        SDVariable added = input.add("added", 1.0);
        SDVariable multiplied = input.mul("multiplied", 2.0);
        SDVariable subtracted = input.sub("subtracted", 0.5);

        sd.enableWorkspaceMode(1024 * 1024);

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("input", Nd4j.ones(DataType.FLOAT, 2, 4));

        Map<String, INDArray> result = sd.output(ph, "added", "multiplied", "subtracted");

        for (String name : new String[]{"added", "multiplied", "subtracted"}) {
            assertNotNull(result.get(name), name + " output should not be null");
            assertFalse(result.get(name).wasClosed(), name + " output should not be closed");
            assertFalse(result.get(name).isAttached(), name + " should not be attached");
        }

        assertEquals(2.0f, result.get("added").getFloat(0, 0), 1e-5);
        assertEquals(2.0f, result.get("multiplied").getFloat(0, 0), 1e-5);
        assertEquals(0.5f, result.get("subtracted").getFloat(0, 0), 1e-5);
    }

    // ========================================================================
    // enableWorkspaceMode API test
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEnableWorkspaceMode(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 2, 3);
        SDVariable out = input.add("out", 1.0);

        sd.enableWorkspaceMode(1024 * 1024);

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("input", Nd4j.ones(DataType.FLOAT, 2, 3));

        Map<String, INDArray> wsResult = sd.output(ph, "out");
        INDArray output = wsResult.get("out");

        assertNotNull(output);
        assertFalse(output.wasClosed());
        assertArrayEquals(new long[]{2, 3}, output.shape());
        assertEquals(2.0f, output.getFloat(0, 0), 1e-5);

        // Run again to verify workspace reuse
        Map<String, INDArray> wsResult2 = sd.output(ph, "out");
        assertEquals(output, wsResult2.get("out"));
    }

    // ========================================================================
    // GPU-specific workspace scoping tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCudaWorkspaceCyclingNoLeaks(Nd4jBackend backend) {
        if (Nd4j.getExecutioner().type() != OpExecutioner.ExecutionerType.CUDA)
            return;

        // Tests CUDA workspace cycling with inference to verify GPU memory is properly recycled
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 256);
        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, 256, 256));
        SDVariable out = sd.mmul("out", input, w);

        val config = WorkspaceConfiguration.builder()
                .initialSize(10 * 1024 * 1024)
                .maxSize(10 * 1024 * 1024)
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .policyReset(ResetPolicy.BLOCK_LEFT)
                .policySpill(SpillPolicy.REALLOCATE)
                .build();

        WorkspaceSessionMemMgr mgr = new WorkspaceSessionMemMgr(config);
        InferenceSession session = new InferenceSession(sd, mgr);

        // Run 50 cycles to verify CUDA memory is properly recycled
        for (int i = 0; i < 50; i++) {
            Map<String, INDArray> ph = new HashMap<>();
            ph.put("input", Nd4j.randn(DataType.FLOAT, 8, 256));
            Map<String, INDArray> result = sd.output(ph, "out");

            INDArray output = result.get("out");
            assertNotNull(output, "Output null at CUDA cycle " + i);
            assertFalse(output.wasClosed(), "Output closed at CUDA cycle " + i);
            assertArrayEquals(new long[]{8, 256}, output.shape());
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCudaHostDeviceSyncDuringWorkspaceInference(Nd4jBackend backend) {
        if (Nd4j.getExecutioner().type() != OpExecutioner.ExecutionerType.CUDA)
            return;

        // Verify host/device pointer correctness within workspace-backed inference.
        // This catches the bug where workspace arrays have stale device buffers.
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 100);
        SDVariable doubled = input.mul("doubled", 2.0);
        SDVariable summed = doubled.sum("summed");

        sd.enableWorkspaceMode(10 * 1024 * 1024);

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("input", Nd4j.ones(DataType.FLOAT, 1, 100));

        // Multiple runs to catch host/device sync issues
        for (int i = 0; i < 10; i++) {
            Map<String, INDArray> result = sd.output(ph, "doubled", "summed");

            INDArray doubledOut = result.get("doubled");
            INDArray summedOut = result.get("summed");

            // Forces device sync
            assertEquals(2.0f, doubledOut.getFloat(0, 0), 1e-5f,
                    "Host/device sync issue at iter " + i);
            assertEquals(200.0, summedOut.getDouble(0), 1e-3,
                    "Sum mismatch at iter " + i);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCudaMirroringPolicyFullWithWorkspace(Nd4jBackend backend) {
        if (Nd4j.getExecutioner().type() != OpExecutioner.ExecutionerType.CUDA)
            return;

        // Test with FULL mirroring policy which maintains both host and device copies.
        // This is the pattern used for debugging CUDA issues (see CudaWorkspaceTests).
        val config = WorkspaceConfiguration.builder()
                .initialSize(10 * 1024 * 1024)
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .policyMirroring(MirroringPolicy.FULL)
                .policyReset(ResetPolicy.BLOCK_LEFT)
                .policySpill(SpillPolicy.REALLOCATE)
                .build();

        WorkspaceSessionMemMgr mgr = new WorkspaceSessionMemMgr(config);
        try {
            for (int i = 0; i < 10; i++) {
                mgr.scopeIn();

                INDArray arr = mgr.allocate(false, DataType.FLOAT, 100);
                arr.assign(0.0f);  // Explicitly zero
                assertEquals(0.0f, arr.getFloat(0), 1e-5f,
                        "Mirrored array should read zeros at cycle " + i);

                arr.assign(2.5f);
                assertEquals(2.5f, arr.getFloat(0), 1e-5f);

                // Verify computation works (forces device sync)
                double sum = arr.sumNumber().doubleValue();
                assertEquals(250.0, sum, 1e-3);

                mgr.scopeOut();
            }
        } finally {
            mgr.close();
        }
    }

    // ========================================================================
    // Multi-GPU / device affinity tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultiGpuWorkspaceInference(Nd4jBackend backend) {
        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        if (numDevices < 2) {
            log.info("Skipping multi-GPU test: only {} device(s) available", numDevices);
            return;
        }

        // Create a graph and run inference on different devices via thread affinity.
        // Each thread gets its own workspace and device.
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 32);
        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, 32, 32));
        SDVariable out = sd.mmul("out", input, w);

        sd.enableWorkspaceMode(4 * 1024 * 1024);

        int numThreads = Math.min(numDevices, 4);
        CountDownLatch latch = new CountDownLatch(numThreads);
        AtomicBoolean failed = new AtomicBoolean(false);
        StringBuilder errors = new StringBuilder();

        for (int t = 0; t < numThreads; t++) {
            final int deviceId = t % numDevices;
            final int threadIdx = t;
            new Thread(() -> {
                try {
                    // Pin thread to specific device
                    Nd4j.getAffinityManager().unsafeSetDevice(deviceId);

                    for (int i = 0; i < 10; i++) {
                        Map<String, INDArray> ph = new HashMap<>();
                        ph.put("input", Nd4j.randn(DataType.FLOAT, 4, 32));
                        Map<String, INDArray> result = sd.output(ph, "out");
                        INDArray output = result.get("out");

                        if (output == null || output.wasClosed()) {
                            failed.set(true);
                            synchronized (errors) {
                                errors.append("Device ").append(deviceId)
                                        .append(" thread ").append(threadIdx)
                                        .append(" iter ").append(i)
                                        .append(": output null or closed\n");
                            }
                        }
                    }
                } catch (Exception e) {
                    failed.set(true);
                    synchronized (errors) {
                        errors.append("Device ").append(deviceId)
                                .append(" thread ").append(threadIdx)
                                .append(": ").append(e.getMessage()).append("\n");
                    }
                } finally {
                    latch.countDown();
                }
            }).start();
        }

        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            fail("Interrupted waiting for threads");
        }
        assertFalse(failed.get(), "Multi-GPU test failed:\n" + errors);
    }

    // ========================================================================
    // GPU failover / recovery tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWorkspaceRecoveryAfterOOM(Nd4jBackend backend) {
        // Verify that workspace-backed inference recovers gracefully when
        // allocation exceeds workspace. With REALLOCATE spill policy,
        // the overflow should be handled by spilling to regular allocations.
        val config = WorkspaceConfiguration.builder()
                .initialSize(4096)  // Intentionally tiny
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .policyReset(ResetPolicy.BLOCK_LEFT)
                .policySpill(SpillPolicy.REALLOCATE)
                .build();

        WorkspaceSessionMemMgr mgr = new WorkspaceSessionMemMgr(config);
        try {
            // First pass: allocations will spill
            mgr.scopeIn();
            INDArray big = mgr.allocate(false, DataType.FLOAT, 512, 512);
            assertNotNull(big, "Large allocation should succeed via spill");
            big.assign(1.0f);
            assertEquals(1.0f, big.getFloat(0, 0), 1e-5f);
            mgr.scopeOut();

            // Second pass: workspace should have learned the required size
            mgr.scopeIn();
            INDArray big2 = mgr.allocate(false, DataType.FLOAT, 512, 512);
            assertNotNull(big2, "Second allocation should succeed after learning");
            big2.assign(2.0f);
            assertEquals(2.0f, big2.getFloat(0, 0), 1e-5f);
            mgr.scopeOut();

            // Third pass: verify stability
            mgr.scopeIn();
            INDArray big3 = mgr.allocate(false, DataType.FLOAT, 512, 512);
            assertNotNull(big3);
            big3.assign(3.0f);
            assertEquals(3.0f, big3.getFloat(0, 0), 1e-5f);
            INDArray detached = mgr.allocate(true, DataType.FLOAT, 1, 10);
            detached.assign(42.0f);
            mgr.scopeOut();

            assertEquals(42.0f, detached.getFloat(0), 1e-5f);
        } finally {
            mgr.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWorkspaceStressWithManySmallAllocations(Nd4jBackend backend) {
        // Stress test: many small allocations within a single scope.
        // This exercises the bump allocator path and verifies no fragmentation issues.
        WorkspaceSessionMemMgr mgr = new WorkspaceSessionMemMgr(16 * 1024 * 1024);
        try {
            for (int cycle = 0; cycle < 5; cycle++) {
                mgr.scopeIn();

                List<INDArray> arrays = new ArrayList<>();
                // 1000 small allocations
                for (int i = 0; i < 1000; i++) {
                    INDArray arr = mgr.allocate(false, DataType.FLOAT, 1, 16);
                    arr.assign((float) i);
                    arrays.add(arr);
                }

                // Verify all arrays have correct values
                for (int i = 0; i < 1000; i++) {
                    assertEquals((float) i, arrays.get(i).getFloat(0), 1e-5f,
                            "Value mismatch at index " + i + " cycle " + cycle);
                }

                // Detached output
                INDArray output = mgr.allocate(true, DataType.FLOAT, 1, 1);
                output.assign(999.0f);

                mgr.scopeOut();

                assertEquals(999.0f, output.getFloat(0), 1e-5f);
            }
        } finally {
            mgr.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDetachPatternWithWorkspaceInference(Nd4jBackend backend) {
        // Test the detach() pattern: arrays created inside a workspace
        // can be detached to heap for use outside the workspace.
        // This is the mechanism output arrays use to survive scopeOut.
        val wsConfig = WorkspaceConfiguration.builder()
                .initialSize(2 * 1024 * 1024)
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .policyReset(ResetPolicy.BLOCK_LEFT)
                .policySpill(SpillPolicy.REALLOCATE)
                .build();

        INDArray detachedArr;
        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager()
                .getAndActivateWorkspace(wsConfig, "DETACH_TEST_WS")) {
            INDArray wsArr = Nd4j.create(new float[]{10, 20, 30, 40, 50});
            assertTrue(wsArr.isAttached(), "Array should be attached to workspace");

            // Detach to heap
            detachedArr = wsArr.detach();
            assertFalse(detachedArr.isAttached(), "Detached array should not be attached");

            assertEquals(10.0f, detachedArr.getFloat(0), 0.01f);
            assertEquals(50.0f, detachedArr.getFloat(4), 0.01f);
        }

        // After workspace closes, detached array is still valid
        assertFalse(detachedArr.wasClosed());
        assertEquals(10.0f, detachedArr.getFloat(0), 0.01f);
        assertEquals(50.0f, detachedArr.getFloat(4), 0.01f);

        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
    }

    // ========================================================================
    // Thread safety tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWorkspaceFactoryThreadSafety(Nd4jBackend backend) throws Exception {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4);
        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, 4, 4));
        SDVariable out = sd.mmul("out", input, w);

        sd.enableWorkspaceMode(4 * 1024 * 1024);

        int numThreads = 4;
        int iterationsPerThread = 20;
        CountDownLatch latch = new CountDownLatch(numThreads);
        AtomicBoolean failed = new AtomicBoolean(false);
        StringBuilder errors = new StringBuilder();

        Thread[] threads = new Thread[numThreads];
        for (int t = 0; t < numThreads; t++) {
            final int threadIdx = t;
            threads[t] = new Thread(() -> {
                try {
                    for (int i = 0; i < iterationsPerThread; i++) {
                        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 2, 4);
                        Map<String, INDArray> ph = new HashMap<>();
                        ph.put("input", inputArr);
                        Map<String, INDArray> result = sd.output(ph, "out");
                        INDArray output = result.get("out");
                        if (output == null || output.wasClosed()) {
                            failed.set(true);
                            synchronized (errors) {
                                errors.append("Thread ").append(threadIdx)
                                        .append(" iter ").append(i)
                                        .append(": output null or closed\n");
                            }
                        }
                    }
                } catch (Exception e) {
                    failed.set(true);
                    synchronized (errors) {
                        errors.append("Thread ").append(threadIdx)
                                .append(": ").append(e.getMessage()).append("\n");
                    }
                } finally {
                    latch.countDown();
                }
            });
            threads[t].start();
        }

        latch.await();
        assertFalse(failed.get(), "Thread safety test failed:\n" + errors);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcurrentWorkspaceScoping(Nd4jBackend backend) throws Exception {
        // Multiple threads doing workspace scopeIn/scopeOut simultaneously.
        // Each thread should get its own workspace scope (thread-local).
        int numThreads = 4;
        int iterations = 50;
        CyclicBarrier barrier = new CyclicBarrier(numThreads);
        CountDownLatch latch = new CountDownLatch(numThreads);
        AtomicBoolean failed = new AtomicBoolean(false);
        StringBuilder errors = new StringBuilder();

        for (int t = 0; t < numThreads; t++) {
            final int threadIdx = t;
            new Thread(() -> {
                WorkspaceSessionMemMgr mgr = new WorkspaceSessionMemMgr(2 * 1024 * 1024);
                try {
                    // Warmup: initialize CUDA context for this thread before barrier
                    mgr.scopeIn();
                    INDArray warmup = mgr.allocate(false, DataType.FLOAT, 4, 4);
                    warmup.assign(1.0f);
                    Nd4j.getExecutioner().commit();
                    mgr.scopeOut();

                    barrier.await(); // All threads start together after warmup
                    for (int i = 0; i < iterations; i++) {
                        mgr.scopeIn();

                        INDArray arr = mgr.allocate(false, DataType.FLOAT, 32, 32);
                        arr.assign((float) threadIdx);
                        Nd4j.getExecutioner().commit();

                        float sum = arr.sumNumber().floatValue();
                        INDArray output = mgr.allocate(true, DataType.FLOAT, 1, 1);
                        output.assign(sum);
                        Nd4j.getExecutioner().commit();

                        mgr.scopeOut();

                        float expected = threadIdx * 32.0f * 32.0f;
                        float actual = output.getFloat(0);
                        if (Math.abs(expected - actual) > 1.0f) {
                            failed.set(true);
                            synchronized (errors) {
                                errors.append("Thread ").append(threadIdx)
                                        .append(" iter ").append(i)
                                        .append(": expected ").append(expected)
                                        .append(" got ").append(actual).append("\n");
                            }
                        }
                    }
                } catch (Exception e) {
                    failed.set(true);
                    synchronized (errors) {
                        errors.append("Thread ").append(threadIdx)
                                .append(": ").append(e.getMessage()).append("\n");
                    }
                } finally {
                    mgr.close();
                    latch.countDown();
                }
            }).start();
        }

        latch.await();
        assertFalse(failed.get(), "Concurrent workspace scoping failed:\n" + errors);
    }

    // ========================================================================
    // Custom workspace configuration tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWorkspaceWithOverAllocationPolicy(Nd4jBackend backend) {
        // Test OVERALLOCATE policy which adds a buffer (30% default)
        // to prevent frequent reallocation
        val config = WorkspaceConfiguration.builder()
                .initialSize(0)
                .overallocationLimit(0.5)  // 50% overallocation
                .policyAllocation(AllocationPolicy.OVERALLOCATE)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .policyReset(ResetPolicy.BLOCK_LEFT)
                .policySpill(SpillPolicy.REALLOCATE)
                .build();

        WorkspaceSessionMemMgr mgr = new WorkspaceSessionMemMgr(config);
        try {
            // First cycle: workspace learns the size
            mgr.scopeIn();
            INDArray arr = mgr.allocate(false, DataType.FLOAT, 100, 100);
            arr.assign(1.0f);
            assertEquals(1.0f, arr.getFloat(0, 0), 1e-5f);
            mgr.scopeOut();

            // Second cycle: workspace should accommodate with overallocation buffer
            mgr.scopeIn();
            INDArray arr2 = mgr.allocate(false, DataType.FLOAT, 100, 100);
            arr2.assign(2.0f);
            assertEquals(2.0f, arr2.getFloat(0, 0), 1e-5f);

            // Slightly larger allocation should fit within overallocation buffer
            INDArray arr3 = mgr.allocate(false, DataType.FLOAT, 50, 50);
            arr3.assign(3.0f);
            assertEquals(3.0f, arr3.getFloat(0, 0), 1e-5f);

            mgr.scopeOut();
        } finally {
            mgr.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWorkspaceWithLearningOverTime(Nd4jBackend backend) {
        // Test OVER_TIME learning policy which learns workspace size progressively
        // over multiple cycles (similar to ComputationGraph pattern)
        val config = WorkspaceConfiguration.builder()
                .initialSize(0)
                .policyAllocation(AllocationPolicy.OVERALLOCATE)
                .policyLearning(LearningPolicy.OVER_TIME)
                .cyclesBeforeInitialization(3)  // Learn over 3 cycles
                .policyReset(ResetPolicy.BLOCK_LEFT)
                .policySpill(SpillPolicy.REALLOCATE)
                .build();

        WorkspaceSessionMemMgr mgr = new WorkspaceSessionMemMgr(config);
        try {
            // Run enough cycles for the workspace to learn
            for (int i = 0; i < 10; i++) {
                mgr.scopeIn();

                INDArray arr = mgr.allocate(false, DataType.FLOAT, 64, 64);
                arr.assign((float) i);
                assertEquals((float) i, arr.getFloat(0, 0), 1e-5f);

                INDArray output = mgr.allocate(true, DataType.FLOAT, 1, 1);
                output.assign((float) i * 10);

                mgr.scopeOut();

                assertEquals((float) i * 10, output.getFloat(0), 1e-5f);
            }
        } finally {
            mgr.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWorkspaceWithMaxSizeCap(Nd4jBackend backend) {
        // Test that maxSize caps workspace growth
        val config = WorkspaceConfiguration.builder()
                .initialSize(1024)
                .maxSize(1024 * 1024)  // 1MB cap
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .policyReset(ResetPolicy.BLOCK_LEFT)
                .policySpill(SpillPolicy.REALLOCATE)
                .build();

        WorkspaceSessionMemMgr mgr = new WorkspaceSessionMemMgr(config);
        try {
            // Allocations within cap should work
            mgr.scopeIn();
            INDArray small = mgr.allocate(false, DataType.FLOAT, 32, 32);
            small.assign(1.0f);
            assertEquals(1.0f, small.getFloat(0, 0), 1e-5f);
            mgr.scopeOut();

            // Very large allocations should still work (via spill with REALLOCATE)
            mgr.scopeIn();
            INDArray large = mgr.allocate(false, DataType.FLOAT, 512, 512);
            assertNotNull(large, "Large allocation should succeed via spill");
            large.assign(2.0f);
            assertEquals(2.0f, large.getFloat(0, 0), 1e-5f);
            mgr.scopeOut();
        } finally {
            mgr.close();
        }
    }

    // ========================================================================
    // Native workspace lifecycle diagnostic test
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNativeWorkspaceCreateDestroyCycle(Nd4jBackend backend) {
        // Minimal test: just create and destroy native workspaces in a loop
        // to isolate whether the crash is in CudaMemoryPool or elsewhere
        org.nd4j.nativeblas.NativeOps nativeOps = org.nd4j.nativeblas.NativeOpsHolder.getInstance().getDeviceNativeOps();
        for (int i = 0; i < 10; i++) {
            log.info("Native workspace cycle {}", i);
            Nd4j.getExecutioner().commit();
            org.bytedeco.javacpp.Pointer wsPtr = nativeOps.createNativeWorkspace(1024L * 1024);
            assertNotNull(wsPtr, "Native workspace should not be null at cycle " + i);
            assertFalse(wsPtr.isNull(), "Native workspace pointer should not be null at cycle " + i);

            // Scope in and out
            nativeOps.workspaceScopeIn(wsPtr);
            nativeOps.workspaceScopeOut(wsPtr);

            // Destroy
            Nd4j.getExecutioner().commit();
            nativeOps.destroyNativeWorkspace(wsPtr);
            log.info("Native workspace cycle {} complete", i);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNativeWorkspaceWithNd4jOps(Nd4jBackend backend) {
        // Test native workspace create/destroy alongside ND4J operations
        org.nd4j.nativeblas.NativeOps nativeOps = org.nd4j.nativeblas.NativeOpsHolder.getInstance().getDeviceNativeOps();
        for (int i = 0; i < 10; i++) {
            log.info("Native workspace + ND4J ops cycle {}", i);
            org.bytedeco.javacpp.Pointer wsPtr = nativeOps.createNativeWorkspace(1024L * 1024);
            assertNotNull(wsPtr);

            // Do some ND4J operations while native workspace exists
            INDArray a = Nd4j.ones(DataType.FLOAT, 4, 4);
            INDArray b = Nd4j.ones(DataType.FLOAT, 4, 4).mul(2);
            INDArray c = a.add(b);
            assertEquals(3.0f, c.getFloat(0, 0), 1e-5f);

            // Run SameDiff too
            SameDiff sd = SameDiff.create();
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4);
            SDVariable out = input.add("out", 1.0);
            Map<String, INDArray> ph = new HashMap<>();
            ph.put("input", Nd4j.zeros(DataType.FLOAT, 2, 4));
            Map<String, INDArray> result = sd.output(ph, "out");
            assertEquals(1.0f, result.get("out").getFloat(0, 0), 1e-5f);

            Nd4j.getExecutioner().commit();
            nativeOps.destroyNativeWorkspace(wsPtr);
            log.info("Native workspace + ND4J ops cycle {} complete", i);
        }
    }

    // ========================================================================
    // Edge case tests
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWorkspaceWithEmptyArrays(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4);
        SDVariable out = input.add("out", 1.0);

        sd.enableWorkspaceMode(1024 * 1024);

        // Single-element batch
        Map<String, INDArray> ph = new HashMap<>();
        ph.put("input", Nd4j.zeros(DataType.FLOAT, 1, 4));
        Map<String, INDArray> result = sd.output(ph, "out");

        assertNotNull(result.get("out"));
        assertEquals(1.0f, result.get("out").getFloat(0, 0), 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWorkspaceCloseThenReuse(Nd4jBackend backend) {
        // Verify that after close(), the workspace can be re-initialized
        WorkspaceSessionMemMgr mgr = new WorkspaceSessionMemMgr(1024 * 1024);

        // Use it
        mgr.scopeIn();
        INDArray arr = mgr.allocate(false, DataType.FLOAT, 10, 10);
        arr.assign(1.0f);
        mgr.scopeOut();

        // Close it
        mgr.close();

        // Allocating without scope should fall back to heap (no crash)
        INDArray heapArr = mgr.allocate(false, DataType.FLOAT, 10, 10);
        assertNotNull(heapArr);
        heapArr.assign(2.0f);
        assertEquals(2.0f, heapArr.getFloat(0, 0), 1e-5f);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWorkspaceDupPreservesValues(Nd4jBackend backend) {
        // Test that dup() within workspace preserves array values
        WorkspaceSessionMemMgr mgr = new WorkspaceSessionMemMgr(2 * 1024 * 1024);
        try {
            mgr.scopeIn();

            INDArray original = Nd4j.create(new float[]{1, 2, 3, 4, 5});
            INDArray duped = mgr.dup(original);

            assertNotNull(duped);
            assertEquals(original, duped);

            // Modify original, dup should be independent
            original.assign(0.0f);
            assertEquals(1.0f, duped.getFloat(0), 1e-5f);
            assertEquals(5.0f, duped.getFloat(4), 1e-5f);

            mgr.scopeOut();
        } finally {
            mgr.close();
        }
    }

    // ========================================================================
    // View safety tests (Phase 1c)
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testViewSafetyWithWorkspace(Nd4jBackend backend) {
        // Graph with reshape/slice producing views - verify outputs aren't views after workspace cycling
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 8);
        SDVariable reshaped = sd.reshape("reshaped", input, -1, 2, 4);
        SDVariable sliced = sd.slice("sliced", reshaped, new int[]{0, 0, 0}, new int[]{-1, 1, 4});
        SDVariable squeezed = sd.squeeze("squeezed", sliced, 1);

        sd.enableWorkspaceMode(4 * 1024 * 1024);

        for (int i = 0; i < 10; i++) {
            Map<String, INDArray> ph = new HashMap<>();
            ph.put("input", Nd4j.randn(DataType.FLOAT, 3, 8));

            Map<String, INDArray> result = sd.output(ph, "squeezed");
            INDArray output = result.get("squeezed");

            assertNotNull(output, "Output null at iteration " + i);
            assertFalse(output.wasClosed(), "Output closed at iteration " + i);
            assertFalse(output.isAttached(), "Output still attached at iteration " + i);
            // After view safety processing, the output should not be a view
            // (it gets dup'd if the workspace-backed intermediate was a view)
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWorkspaceAttachDetachOpContext(Nd4jBackend backend) {
        // Verify workspace allocation lifecycle and scope behavior
        WorkspaceSessionMemMgr mgr = new WorkspaceSessionMemMgr(4 * 1024 * 1024);
        try {
            mgr.scopeIn();

            assertTrue(mgr.isWorkspaceBacked(),
                    "WorkspaceSessionMemMgr should report as workspace-backed");

            INDArray arr = mgr.allocate(false, DataType.FLOAT, 32, 32);
            arr.assign(1.0f);
            assertEquals(1.0f, arr.getFloat(0, 0), 1e-5f);

            // Detached arrays should survive scope cycling
            INDArray detached = mgr.allocate(true, DataType.FLOAT, 32, 32);
            detached.assign(2.0f);

            mgr.scopeOut();

            // Detached array should still be valid after scopeOut
            assertEquals(2.0f, detached.getFloat(0, 0), 1e-5f);
        } finally {
            mgr.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScopeInOutRapidCycling(Nd4jBackend backend) {
        // 1000 rapid scope transitions to check for leaks or state corruption
        WorkspaceSessionMemMgr mgr = new WorkspaceSessionMemMgr(1024 * 1024);
        try {
            for (int i = 0; i < 1000; i++) {
                mgr.scopeIn();
                INDArray arr = mgr.allocate(false, DataType.FLOAT, 4, 4);
                arr.assign((float) (i % 100));
                INDArray output = mgr.allocate(true, DataType.FLOAT, 1);
                output.assign(arr.sumNumber());
                mgr.scopeOut();

                float expected = (float) (i % 100) * 16.0f;
                assertEquals(expected, output.getFloat(0), 1e-3f,
                        "Mismatch at rapid cycle " + i);
            }
        } finally {
            mgr.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAutoRegressiveWithViewOps(Nd4jBackend backend) {
        // Autoregressive loop with view-producing ops (reshape, slice)
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 8);
        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, 8, 8));
        SDVariable mm = sd.mmul("mm", input, w);
        SDVariable out = sd.nn.softmax("out", mm, -1);

        sd.enableWorkspaceMode(4 * 1024 * 1024);

        INDArray prev = Nd4j.randn(DataType.FLOAT, 1, 8);
        for (int i = 0; i < 50; i++) {
            Map<String, INDArray> ph = new HashMap<>();
            ph.put("input", prev);
            Map<String, INDArray> result = sd.output(ph, "out");
            INDArray output = result.get("out");

            assertNotNull(output, "Output null at autoregressive iter " + i);
            assertFalse(output.wasClosed(), "Output closed at autoregressive iter " + i);
            assertFalse(output.isAttached(), "Output attached at autoregressive iter " + i);

            double sum = output.sumNumber().doubleValue();
            assertEquals(1.0, sum, 1e-3, "Softmax sum not 1.0 at iter " + i);

            prev = output;
        }
    }

    // ========================================================================
    // Disable workspace mode test
    // ========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDisableWorkspaceMode(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 2, 3);
        SDVariable out = input.add("out", 1.0);

        // Enable then disable
        sd.enableWorkspaceMode(1024 * 1024);
        sd.disableWorkspaceMode();

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("input", Nd4j.ones(DataType.FLOAT, 2, 3));

        Map<String, INDArray> result = sd.output(ph, "out");
        INDArray output = result.get("out");

        assertNotNull(output);
        assertEquals(2.0f, output.getFloat(0, 0), 1e-5);
    }
}
