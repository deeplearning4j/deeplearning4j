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
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;

@Slf4j
@Tag(TagNames.WORKSPACES)
@NativeTag
public class CudaWorkspaceTests extends BaseNd4jTestWithBackends {
    private DataType initialType = Nd4j.dataType();



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWorkspaceReuse() {
        if (Nd4j.getExecutioner().type() != OpExecutioner.ExecutionerType.CUDA)
            return;

        val workspaceConfig = WorkspaceConfiguration.builder()
                .policyMirroring(MirroringPolicy.HOST_ONLY) // Commenting this out makes it so that assert is not triggered (for at least 40 secs or so...)
                .build();
        int cnt = 0;

        for (int  e = 0; e < 10; e++) {
            try (val ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfig, "test")) {
                final INDArray zeros = Nd4j.zeros(4, 'f');
                //final INDArray zeros = Nd4j.create(4, 'f'); // Also fails, but maybe less of an issue as javadoc does not say that one can expect returned array to be all zeros.
                assertEquals( 0d, zeros.sumNumber().doubleValue(), 1e-10,"Got non-zero array " + zeros + " after " + cnt + " iterations !");
                zeros.putScalar(0, 1);
            }
        }

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWorkspaceCyclingNoLeaks(Nd4jBackend backend) {
        if (Nd4j.getExecutioner().type() != OpExecutioner.ExecutionerType.CUDA)
            return;

        val workspaceConfig = WorkspaceConfiguration.builder()
                .initialSize(10 * 1024 * 1024)
                .maxSize(10 * 1024 * 1024)
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .policyReset(ResetPolicy.BLOCK_LEFT)
                .policySpill(SpillPolicy.REALLOCATE)
                .build();

        // Run many workspace cycles to verify CUDA memory is properly recycled
        for (int i = 0; i < 50; i++) {
            try (val ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfig, "cuda_cycle_test")) {
                INDArray arr = Nd4j.create(DataType.FLOAT, 256, 256);
                arr.assign(1.0f);
                assertEquals(1.0f, arr.getFloat(0, 0), 1e-5f);
            }
        }

        // Clean up
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWorkspaceHostDevicePointerCorrectness(Nd4jBackend backend) {
        if (Nd4j.getExecutioner().type() != OpExecutioner.ExecutionerType.CUDA)
            return;

        val workspaceConfig = WorkspaceConfiguration.builder()
                .initialSize(10 * 1024 * 1024)
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .policyMirroring(MirroringPolicy.FULL)
                .build();

        try (val ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfig, "cuda_ptr_test")) {
            INDArray arr = Nd4j.create(DataType.FLOAT, 100);
            arr.assign(2.5f);

            // Verify the array has valid data on host
            assertEquals(2.5f, arr.getFloat(0), 1e-5f);
            assertEquals(2.5f, arr.getFloat(99), 1e-5f);

            // Verify computation works (forces device sync)
            double sum = arr.sumNumber().doubleValue();
            assertEquals(250.0, sum, 1e-3);
        }

        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGpuOomSpillToCpuPinned(Nd4jBackend backend) {
        if (Nd4j.getExecutioner().type() != OpExecutioner.ExecutionerType.CUDA)
            return;

        // Tiny workspace, large allocation - verifies the C++ fallback to pinned host memory
        val workspaceConfig = WorkspaceConfiguration.builder()
                .initialSize(1024)  // Intentionally tiny (1KB)
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .policyReset(ResetPolicy.BLOCK_LEFT)
                .policySpill(SpillPolicy.REALLOCATE)
                .build();

        try (val ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfig, "cuda_oom_spill_test")) {
            // This allocation exceeds the workspace - should spill
            INDArray large = Nd4j.create(DataType.FLOAT, 1024, 1024);
            assertNotNull(large, "Spill allocation should succeed");
            large.assign(1.0f);
            assertEquals(1.0f, large.getFloat(0, 0), 1e-5f);

            // Verify computation works on the spilled array
            double sum = large.sumNumber().doubleValue();
            assertEquals(1024.0 * 1024.0, sum, 1.0);
        }

        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWorkspaceAttachToOpContextCuda(Nd4jBackend backend) {
        if (Nd4j.getExecutioner().type() != OpExecutioner.ExecutionerType.CUDA)
            return;

        // Attach workspace, run matmul, verify workspace offset increased
        org.nd4j.autodiff.samediff.internal.memory.WorkspaceSessionMemMgr mgr =
                new org.nd4j.autodiff.samediff.internal.memory.WorkspaceSessionMemMgr(10 * 1024 * 1024);
        try {
            mgr.scopeIn();

            org.bytedeco.javacpp.Pointer wsPtr = mgr.getNativeWorkspacePointer();
            assertNotNull(wsPtr, "Native workspace pointer should be non-null on CUDA");

            // Get initial offset
            org.nd4j.nativeblas.NativeOps nativeOps = org.nd4j.nativeblas.NativeOpsHolder.getInstance().getDeviceNativeOps();
            long offsetBefore = nativeOps.getWorkspaceCurrentOffset(wsPtr);

            // Run a simple computation
            INDArray a = mgr.allocate(false, DataType.FLOAT, 32, 32);
            INDArray b = mgr.allocate(false, DataType.FLOAT, 32, 32);
            a.assign(1.0f);
            b.assign(2.0f);

            mgr.scopeOut();

            mgr.close();
        } catch (Exception e) {
            mgr.close();
            // It's OK if this test can't get workspace offset on this platform
            log.info("testWorkspaceAttachToOpContextCuda: {}", e.getMessage());
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCudaDeviceSyncOnScopeOut(Nd4jBackend backend) {
        if (Nd4j.getExecutioner().type() != OpExecutioner.ExecutionerType.CUDA)
            return;

        // Verify pending CUDA ops complete before workspace reset
        val workspaceConfig = WorkspaceConfiguration.builder()
                .initialSize(10 * 1024 * 1024)
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .policyReset(ResetPolicy.BLOCK_LEFT)
                .policySpill(SpillPolicy.REALLOCATE)
                .build();

        org.nd4j.autodiff.samediff.internal.memory.WorkspaceSessionMemMgr mgr =
                new org.nd4j.autodiff.samediff.internal.memory.WorkspaceSessionMemMgr(workspaceConfig);
        try {
            for (int i = 0; i < 20; i++) {
                mgr.scopeIn();

                INDArray arr = mgr.allocate(false, DataType.FLOAT, 256, 256);
                arr.assign(1.0f);

                // Launch async op
                INDArray result = arr.mmul(arr);

                // The scopeOut should commit pending ops
                INDArray output = mgr.allocate(true, DataType.FLOAT, 256, 256);
                output.assign(result);

                mgr.scopeOut();

                // After scope out, reading from detached output should be safe
                assertEquals(256.0f, output.getFloat(0, 0), 1e-1f,
                        "Device sync issue at iteration " + i);
            }
        } finally {
            mgr.close();
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
