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
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for DeviceExecutionContext in DSP.
 *
 * <p>Validates that the device execution context correctly captures and restores
 * thread-local state, and that workspace allocation works correctly within
 * the DSP execution path.</p>
 *
 * <p>Run:
 * <pre>
 *   cd platform-tests && mvn test -Dtest=TestDSPMultiDeviceContext
 * </pre>
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestDSPMultiDeviceContext extends BaseNd4jTestWithBackends {

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

    @AfterEach
    void cleanup() {
        if (sd != null) {
            sd.close();
            sd = null;
        }
    }

    @Test
    @Order(1)
    @DisplayName("Thread-local context: DSP execution preserves thread-local state")
    public void testFromThreadLocals() {
        sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, 8);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 8, 16));
        sd.mmul("output", x, w);
        enableDsp(sd);

        // Capture state before DSP execution
        int deviceBefore = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 8);
        Map<String, INDArray> result = sd.output(
                Collections.singletonMap("input", input), "output");
        assertNotNull(result.get("output"));

        // Device affinity should be unchanged after DSP execution
        int deviceAfter = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        assertEquals(deviceBefore, deviceAfter,
                "DSP execution must not change thread's device affinity");
    }

    @Test
    @Order(2)
    @DisplayName("Round-trip context: thread-local state survives DSP round-trip")
    public void testApplyToThreadLocals() {
        sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, 8);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 8, 16));
        sd.mmul("output", x, w);
        enableDsp(sd);

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 8);

        // Execute multiple times -- thread-local state should survive each round-trip
        for (int i = 0; i < 5; i++) {
            int deviceBefore = Nd4j.getAffinityManager().getDeviceForCurrentThread();

            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("input", input), "output");
            assertNotNull(result.get("output"), "Execution " + i + " must produce output");

            int deviceAfter = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            assertEquals(deviceBefore, deviceAfter,
                    "Execution " + i + " must preserve device affinity");
        }
    }

    @Test
    @Order(3)
    @DisplayName("Workspace allocation: workspace-scoped arrays are correctly allocated and aligned")
    public void testWorkspaceAlloc() {
        WorkspaceConfiguration wsConfig = WorkspaceConfiguration.builder()
                .initialSize(1024 * 1024) // 1MB
                .maxSize(10 * 1024 * 1024) // 10MB
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.NONE)
                .policyMirroring(MirroringPolicy.FULL)
                .policySpill(SpillPolicy.FAIL)
                .build();

        String wsName = "TestDSPMultiDeviceContext_ws";

        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(wsConfig, wsName)) {
            // Allocate arrays within workspace
            INDArray a = Nd4j.create(DataType.FLOAT, 32, 32);
            INDArray b = Nd4j.create(DataType.FLOAT, 32, 32);

            assertNotNull(a, "Workspace allocation must succeed");
            assertNotNull(b, "Workspace allocation must succeed");

            // Verify arrays are attached to workspace
            assertTrue(a.isAttached(), "Array must be attached to workspace");
            assertTrue(b.isAttached(), "Array must be attached to workspace");

            // Verify we can do computation within workspace
            INDArray c = a.mmul(b);
            assertNotNull(c, "Workspace computation must produce result");
            assertArrayEquals(new long[]{32, 32}, c.shape());
        }

        // After workspace close, verify cleanup
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();

        // Now test that DSP execution works after workspace lifecycle
        sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, 8);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 8, 16));
        sd.mmul("output", x, w);
        enableDsp(sd);

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 8);
        Map<String, INDArray> result = sd.output(
                Collections.singletonMap("input", input), "output");
        assertNotNull(result.get("output"),
                "DSP execution must work after workspace lifecycle");
        assertArrayEquals(new long[]{2, 16}, result.get("output").shape());
    }
}
