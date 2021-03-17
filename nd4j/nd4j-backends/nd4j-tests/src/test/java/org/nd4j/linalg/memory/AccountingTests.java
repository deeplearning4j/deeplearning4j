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

package org.nd4j.linalg.memory;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.AllocationsTracker;
import org.nd4j.linalg.api.memory.DeviceAllocationsTracker;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationKind;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;

@Slf4j
@Disabled
public class AccountingTests extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDetached_1(Nd4jBackend backend) {
        val array = Nd4j.createFromArray(1, 2, 3, 4, 5);
        assertEquals(DataType.INT, array.dataType());

        assertTrue(Nd4j.getMemoryManager().allocatedMemory(0) > 0L);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDetached_2(Nd4jBackend backend) {
        val deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        val before = Nd4j.getMemoryManager().allocatedMemory(deviceId);

        val array = Nd4j.createFromArray(1, 2, 3, 4, 5, 6, 7);
        assertEquals(DataType.INT, array.dataType());

        val after = Nd4j.getMemoryManager().allocatedMemory(deviceId);

        assertTrue(after > before);
        assertTrue(AllocationsTracker.getInstance().bytesOnDevice(AllocationKind.CONSTANT, Nd4j.getAffinityManager().getDeviceForCurrentThread()) > 0);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWorkspaceAccounting_1(Nd4jBackend backend) {
        val deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        val wsConf = WorkspaceConfiguration.builder()
                .initialSize(10 * 1024 * 1024)
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .build();

        val before = Nd4j.getMemoryManager().allocatedMemory(deviceId);

        val workspace = Nd4j.getWorkspaceManager().createNewWorkspace(wsConf, "random_name_here");

        val middle = Nd4j.getMemoryManager().allocatedMemory(deviceId);

        workspace.destroyWorkspace(true);

        val after = Nd4j.getMemoryManager().allocatedMemory(deviceId);

        log.info("Before: {}; Middle: {}; After: {}", before, middle, after);
        assertTrue(middle > before);
        assertTrue(after < middle);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWorkspaceAccounting_2(Nd4jBackend backend) {
        val deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        val wsConf = WorkspaceConfiguration.builder()
                .initialSize(0)
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.OVER_TIME)
                .cyclesBeforeInitialization(3)
                .build();

        val before = Nd4j.getMemoryManager().allocatedMemory(deviceId);

        long middle1 = 0;
        try (val workspace = Nd4j.getWorkspaceManager().getAndActivateWorkspace(wsConf, "random_name_here")) {
            val array = Nd4j.create(DataType.DOUBLE, 5, 5);
            middle1 = Nd4j.getMemoryManager().allocatedMemory(deviceId);
        }

        val middle2 = Nd4j.getMemoryManager().allocatedMemory(deviceId);

        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();

        val after = Nd4j.getMemoryManager().allocatedMemory(deviceId);

        log.info("Before: {}; Middle1: {}; Middle2: {}; After: {}", before, middle1, middle2, after);
        assertTrue(middle1 > before);
        assertTrue(after < middle1);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testManualDeallocation_1(Nd4jBackend backend) {
        val deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        val before = Nd4j.getMemoryManager().allocatedMemory(deviceId);

        val array = Nd4j.createFromArray(new byte[] {1, 2, 3});

        val middle = Nd4j.getMemoryManager().allocatedMemory(deviceId);

        array.close();

        val after = Nd4j.getMemoryManager().allocatedMemory(deviceId);

        assertTrue(middle > before);

        // <= here just because possible cache activation
        assertTrue(after <= middle);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTracker_1(Nd4jBackend backend) {
        val tracker = new DeviceAllocationsTracker();

        for (val e: AllocationKind.values()) {
            for (int v = 1; v <= 100; v++) {
                tracker.updateState(e, v);
            }

            assertNotEquals(0, tracker.getState(e));
        }

        for (val e: AllocationKind.values()) {
            for (int v = 1; v <= 100; v++) {
                tracker.updateState(e, -v);
            }

            assertEquals(0, tracker.getState(e));
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
