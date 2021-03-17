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

package org.nd4j.linalg.workspace;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.LocationPolicy;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.api.memory.abstracts.Nd4jWorkspace;

import java.io.File;

import static org.junit.jupiter.api.Assertions.*;
import static org.nd4j.linalg.api.buffer.DataType.DOUBLE;

@Slf4j

public class BasicWorkspaceTests extends BaseNd4jTestWithBackends {
    DataType initialType = Nd4j.dataType();

    private static final WorkspaceConfiguration basicConfig = WorkspaceConfiguration.builder()
                    .initialSize(10 * 1024 * 1024).maxSize(10 * 1024 * 1024).overallocationLimit(0.1)
                    .policyAllocation(AllocationPolicy.STRICT).policyLearning(LearningPolicy.FIRST_LOOP)
                    .policyMirroring(MirroringPolicy.FULL).policySpill(SpillPolicy.EXTERNAL).build();

    private static final WorkspaceConfiguration loopOverTimeConfig =
                    WorkspaceConfiguration.builder().initialSize(0).maxSize(10 * 1024 * 1024).overallocationLimit(0.1)
                                    .policyAllocation(AllocationPolicy.STRICT).policyLearning(LearningPolicy.OVER_TIME)
                                    .policyMirroring(MirroringPolicy.FULL).policySpill(SpillPolicy.EXTERNAL).build();


    private static final WorkspaceConfiguration loopFirstConfig =
                    WorkspaceConfiguration.builder().initialSize(0).maxSize(10 * 1024 * 1024).overallocationLimit(0.1)
                                    .policyAllocation(AllocationPolicy.STRICT).policyLearning(LearningPolicy.FIRST_LOOP)
                                    .policyMirroring(MirroringPolicy.FULL).policySpill(SpillPolicy.EXTERNAL).build();



    @BeforeEach
    public void setUp() {
        Nd4j.setDataType(DOUBLE);
    }

    @AfterEach
    public void shutdown() {
        Nd4j.getMemoryManager().setCurrentWorkspace(null);
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();

        Nd4j.setDataType(initialType);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCold(Nd4jBackend backend) {
        INDArray array = Nd4j.create(10);

        array.addi(1.0);

        assertEquals(10f, array.sumNumber().floatValue(), 0.01f);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMinSize1(Nd4jBackend backend) {
        WorkspaceConfiguration conf = WorkspaceConfiguration.builder().minSize(10 * 1024 * 1024)
                        .overallocationLimit(1.0).policyAllocation(AllocationPolicy.OVERALLOCATE)
                        .policyLearning(LearningPolicy.FIRST_LOOP).policyMirroring(MirroringPolicy.FULL)
                        .policySpill(SpillPolicy.EXTERNAL).build();

        try (Nd4jWorkspace workspace = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(conf, "WT")) {
            INDArray array = Nd4j.create(100);

            assertEquals(0, workspace.getCurrentSize());
        }

        try (Nd4jWorkspace workspace = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(conf, "WT")) {
            INDArray array = Nd4j.create(100);

            assertEquals(10 * 1024 * 1024, workspace.getCurrentSize());
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBreakout2(Nd4jBackend backend) {

        assertEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        INDArray scoped = outScope2();

        assertEquals(null, scoped);

        assertEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBreakout1(Nd4jBackend backend) {

        assertEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        INDArray scoped = outScope1();

        assertEquals(true, scoped.isAttached());

        assertEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    private INDArray outScope2() {
        try {
            try (Nd4jWorkspace wsOne =
                            (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfig, "EXT")) {
                throw new RuntimeException();
            }
        } catch (Exception e) {
            return null;
        }
    }

    private INDArray outScope1() {
        try (Nd4jWorkspace wsOne =
                        (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfig, "EXT")) {
            return Nd4j.create(10);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLeverage3(Nd4jBackend backend) {
        try (Nd4jWorkspace wsOne =
                        (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfig, "EXT")) {
            INDArray array = null;
            try (Nd4jWorkspace wsTwo =
                            (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfig, "INT")) {
                INDArray matrix = Nd4j.create(32, 1, 40);

                INDArray view = matrix.tensorAlongDimension(0, 1, 2);
                view.assign(1.0f);
                assertEquals(40.0f, matrix.sumNumber().floatValue(), 0.01f);
                assertEquals(40.0f, view.sumNumber().floatValue(), 0.01f);
                array = view.leverageTo("EXT");
            }

            assertEquals(40.0f, array.sumNumber().floatValue(), 0.01f);
        }
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLeverageTo2(Nd4jBackend backend) {
        val exp = Nd4j.scalar(15.0);
        try (Nd4jWorkspace wsOne =
                        (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(loopOverTimeConfig, "EXT")) {
            INDArray array1 = Nd4j.create(new double[] {1f, 2f, 3f, 4f, 5f});
            INDArray array3 = null;

            try (Nd4jWorkspace wsTwo =
                            (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfig, "INT")) {
                INDArray array2 = Nd4j.create(new double[] {1f, 2f, 3f, 4f, 5f});

                long reqMemory = 5 * Nd4j.sizeOfDataType(DOUBLE);

                array3 = array2.leverageTo("EXT");

                assertEquals(0, wsOne.getCurrentSize());

                assertEquals(15f, array3.sumNumber().floatValue(), 0.01f);

                array2.assign(0);

                assertEquals(15f, array3.sumNumber().floatValue(), 0.01f);
            }

            try (Nd4jWorkspace wsTwo =
                            (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfig, "INT")) {
                INDArray array2 = Nd4j.create(100);
            }

            assertEquals(15f, array3.sumNumber().floatValue(), 0.01f);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLeverageTo1(Nd4jBackend backend) {
        try (Nd4jWorkspace wsOne =
                        (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfig, "EXT")) {
            INDArray array1 = Nd4j.create(DOUBLE, 5);

            try (Nd4jWorkspace wsTwo =
                            (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfig, "INT")) {
                INDArray array2 = Nd4j.create(new double[] {1f, 2f, 3f, 4f, 5f});

                long reqMemory = 5 * Nd4j.sizeOfDataType(DOUBLE);
                assertEquals(reqMemory + reqMemory % 16, wsOne.getPrimaryOffset());

                array2.leverageTo("EXT");

                assertEquals((reqMemory + reqMemory % 16) * 2, wsOne.getPrimaryOffset());
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testOutOfScope1(Nd4jBackend backend) {
        try (Nd4jWorkspace wsOne =
                        (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfig, "EXT")) {
            INDArray array1 = Nd4j.create(new double[] {1f, 2f, 3f, 4f, 5f});

            long reqMemory = 5 * Nd4j.sizeOfDataType(array1.dataType());
            assertEquals(reqMemory + reqMemory % 16, wsOne.getPrimaryOffset());

            INDArray array2;

            try (MemoryWorkspace workspace = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                array2 = Nd4j.create(new double[] {1f, 2f, 3f, 4f, 5f});
            }
            assertFalse(array2.isAttached());

            log.info("Current workspace: {}", Nd4j.getMemoryManager().getCurrentWorkspace());
            assertTrue(wsOne == Nd4j.getMemoryManager().getCurrentWorkspace());

            INDArray array3 = Nd4j.create(new double[] {1f, 2f, 3f, 4f, 5f});

            reqMemory = 5 * Nd4j.sizeOfDataType(array3.dataType());
            assertEquals((reqMemory + reqMemory % 16) * 2, wsOne.getPrimaryOffset());

            array1.addi(array2);

            assertEquals(30.0f, array1.sumNumber().floatValue(), 0.01f);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLeverage1(Nd4jBackend backend) {
        try (Nd4jWorkspace wsOne =
                        (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfig, "EXT")) {

            assertEquals(0, wsOne.getPrimaryOffset());

            try (Nd4jWorkspace wsTwo =
                            (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfig, "INT")) {

                INDArray array = Nd4j.create(new double[] {1f, 2f, 3f, 4f, 5f});

                assertEquals(0, wsOne.getPrimaryOffset());

                long reqMemory = 5 * Nd4j.sizeOfDataType(array.dataType());
                assertEquals(reqMemory + reqMemory % 16, wsTwo.getPrimaryOffset());

                INDArray copy = array.leverage();

                assertEquals(reqMemory + reqMemory % 16, wsTwo.getPrimaryOffset());
                assertEquals(reqMemory + reqMemory % 16, wsOne.getPrimaryOffset());

                assertNotEquals(null, copy);

                assertTrue(copy.isAttached());

                assertEquals(15.0f, copy.sumNumber().floatValue(), 0.01f);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNoShape1(Nd4jBackend backend) {
        int outDepth = 50;
        int miniBatch = 64;
        int outH = 8;
        int outW = 8;

        try (Nd4jWorkspace wsI =
                        (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfig, "ITER")) {
            INDArray delta = Nd4j.create(new int[] {50, 64, 8, 8}, new int[] {64, 3200, 8, 1}, 'c');
            delta = delta.permute(1, 0, 2, 3);

            assertArrayEquals(new long[] {64, 50, 8, 8}, delta.shape());
            assertArrayEquals(new long[] {3200, 64, 8, 1}, delta.stride());

            INDArray delta2d = Shape.newShapeNoCopy(delta, new int[] {outDepth, miniBatch * outH * outW}, false);

            assertNotNull(delta2d);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCreateDetached1(Nd4jBackend backend) {
        try (Nd4jWorkspace wsI =
                        (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfig, "ITER")) {

            INDArray array1 = Nd4j.create(new double[] {1f, 2f, 3f, 4f, 5f});

            INDArray array2 = Nd4j.createUninitializedDetached(DOUBLE, 5);

            array2.assign(array1);

            long reqMemory = 5 * Nd4j.sizeOfDataType(array1.dataType());
            assertEquals(reqMemory + reqMemory % 16, wsI.getPrimaryOffset());
            assertEquals(array1, array2);

            INDArray array3 = Nd4j.createUninitializedDetached(DataType.FLOAT, new long[0]);
            assertTrue(array3.isScalar());
            assertEquals(1, array3.length());
            assertEquals(1, array3.data().length());
        }
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDetach1(Nd4jBackend backend) {
        INDArray array = null;
        INDArray copy = null;
        try (Nd4jWorkspace wsI =
                        (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfig, "ITER")) {
            array = Nd4j.create(new double[] {1f, 2f, 3f, 4f, 5f});

            // despite we're allocating this array in workspace, it's empty yet, so it's external allocation
            assertTrue(array.isInScope());
            assertTrue(array.isAttached());

            long reqMemory = 5 * Nd4j.sizeOfDataType(array.dataType());
            assertEquals(reqMemory + reqMemory % 16, wsI.getPrimaryOffset());

            copy = array.detach();

            assertTrue(array.isInScope());
            assertTrue(array.isAttached());
            assertEquals(reqMemory + reqMemory % 16, wsI.getPrimaryOffset());

            assertFalse(copy.isAttached());
            assertTrue(copy.isInScope());
            assertEquals(reqMemory + reqMemory % 16, wsI.getPrimaryOffset());
        }

        assertEquals(15.0f, copy.sumNumber().floatValue(), 0.01f);
        assertFalse(array == copy);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScope2(Nd4jBackend backend) {
        INDArray array = null;
        try (Nd4jWorkspace wsI =
                        (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(loopFirstConfig, "ITER")) {
            array = Nd4j.create(DOUBLE, 100);

            // despite we're allocating this array in workspace, it's empty yet, so it's external allocation
            assertTrue(array.isInScope());
            assertEquals(0, wsI.getCurrentSize());
        }


        try (Nd4jWorkspace wsI =
                        (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(loopFirstConfig, "ITER")) {
            array = Nd4j.create(DOUBLE, 100);

            assertTrue(array.isInScope());
            assertEquals(100 * Nd4j.sizeOfDataType(array.dataType()), wsI.getPrimaryOffset());
        }

        assertFalse(array.isInScope());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScope1(Nd4jBackend backend) {
        INDArray array = null;
        try (Nd4jWorkspace wsI =
                        (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfig, "ITER")) {
            array = Nd4j.create(DOUBLE, 100);

            assertTrue(array.isInScope());
        }

        assertFalse(array.isInScope());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIsAttached3(Nd4jBackend backend) {
        INDArray array = Nd4j.create(DOUBLE, 100);
        try (Nd4jWorkspace wsI =
                        (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfig, "ITER")) {
            INDArray arrayL = array.leverageTo("ITER");

            assertFalse(array.isAttached());
            assertFalse(arrayL.isAttached());

        }

        INDArray array2 = Nd4j.create(DOUBLE, 100);

        assertFalse(array.isAttached());
        assertFalse(array2.isAttached());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIsAttached2(Nd4jBackend backend) {
        INDArray array = Nd4j.create(DOUBLE, 100);
        try (Nd4jWorkspace wsI =
                        (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(loopFirstConfig, "ITER")) {
            INDArray arrayL = array.leverageTo("ITER");

            assertFalse(array.isAttached());
            assertFalse(arrayL.isAttached());
        }

        INDArray array2 = Nd4j.create(100);

        assertFalse(array.isAttached());
        assertFalse(array2.isAttached());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIsAttached1(Nd4jBackend backend) {

        try (Nd4jWorkspace wsI =
                        (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(loopFirstConfig, "ITER")) {
            INDArray array = Nd4j.create(DOUBLE, 100);

            assertTrue(array.isAttached());
        }

        INDArray array = Nd4j.create(100);

        assertFalse(array.isAttached());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testOverallocation3(Nd4jBackend backend) {
        WorkspaceConfiguration overallocationConfig = WorkspaceConfiguration.builder().initialSize(0)
                        .maxSize(10 * 1024 * 1024).overallocationLimit(1.0)
                        .policyAllocation(AllocationPolicy.OVERALLOCATE).policyLearning(LearningPolicy.OVER_TIME)
                        .policyMirroring(MirroringPolicy.FULL).policySpill(SpillPolicy.EXTERNAL).build();

        Nd4jWorkspace workspace = (Nd4jWorkspace) Nd4j.getWorkspaceManager().createNewWorkspace(overallocationConfig);

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertEquals(0, workspace.getCurrentSize());

        for (int x = 10; x <= 100; x += 10) {
            try (MemoryWorkspace cW = workspace.notifyScopeEntered()) {
                INDArray array = Nd4j.create(DOUBLE, x);
            }
        }

        assertEquals(0, workspace.getCurrentSize());

        workspace.initializeWorkspace();


        // should be 800 = 100 elements * 4 bytes per element * 2 as overallocation coefficient
        assertEquals(200 * Nd4j.sizeOfDataType(DOUBLE), workspace.getCurrentSize());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testOverallocation2(Nd4jBackend backend) {
        WorkspaceConfiguration overallocationConfig = WorkspaceConfiguration.builder().initialSize(0)
                        .maxSize(10 * 1024 * 1024).overallocationLimit(1.0)
                        .policyAllocation(AllocationPolicy.OVERALLOCATE).policyLearning(LearningPolicy.FIRST_LOOP)
                        .policyMirroring(MirroringPolicy.FULL).policySpill(SpillPolicy.EXTERNAL).build();

        Nd4jWorkspace workspace = (Nd4jWorkspace) Nd4j.getWorkspaceManager().createNewWorkspace(overallocationConfig);

        //Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertEquals(0, workspace.getCurrentSize());

        try (MemoryWorkspace cW = workspace.notifyScopeEntered()) {
            INDArray array = Nd4j.create(DOUBLE, 100);
        }

        // should be 800 = 100 elements * 4 bytes per element * 2 as overallocation coefficient
        assertEquals(200 * Nd4j.sizeOfDataType(DOUBLE), workspace.getCurrentSize());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testOverallocation1(Nd4jBackend backend) {
        WorkspaceConfiguration overallocationConfig = WorkspaceConfiguration.builder().initialSize(1024)
                        .maxSize(10 * 1024 * 1024).overallocationLimit(1.0)
                        .policyAllocation(AllocationPolicy.OVERALLOCATE).policyLearning(LearningPolicy.NONE)
                        .policyMirroring(MirroringPolicy.FULL).policySpill(SpillPolicy.EXTERNAL).build();

        Nd4jWorkspace workspace = (Nd4jWorkspace) Nd4j.getWorkspaceManager().createNewWorkspace(overallocationConfig);

        assertEquals(2048, workspace.getCurrentSize());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testToggle1(Nd4jBackend backend) {
        Nd4jWorkspace workspace = (Nd4jWorkspace) Nd4j.getWorkspaceManager().createNewWorkspace(loopFirstConfig);

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertNotEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        assertEquals(0, workspace.getPrimaryOffset());

        try (MemoryWorkspace cW = workspace.notifyScopeEntered()) {
            INDArray array1 = Nd4j.create(DOUBLE, 100);

            cW.toggleWorkspaceUse(false);

            INDArray arrayDetached = Nd4j.create(DOUBLE, 100);

            arrayDetached.assign(1.0f);

            double sum = arrayDetached.sumNumber().doubleValue();
            assertEquals(100f, sum, 0.01);

            cW.toggleWorkspaceUse(true);

            INDArray array2 = Nd4j.create(DOUBLE, 100);
        }

        assertEquals(0, workspace.getPrimaryOffset());
        assertEquals(200 * Nd4j.sizeOfDataType(DOUBLE), workspace.getCurrentSize());

        log.info("--------------------------");

        try (MemoryWorkspace cW = workspace.notifyScopeEntered()) {
            INDArray array1 = Nd4j.create(DOUBLE, 100);

            cW.toggleWorkspaceUse(false);

            INDArray arrayDetached = Nd4j.create(DOUBLE, 100);

            arrayDetached.assign(1.0f);

            double sum = arrayDetached.sumNumber().doubleValue();
            assertEquals(100f, sum, 0.01);

            cW.toggleWorkspaceUse(true);

            assertEquals(100 * Nd4j.sizeOfDataType(DOUBLE), workspace.getPrimaryOffset());

            INDArray array2 = Nd4j.create(DOUBLE, 100);

            assertEquals(200 * Nd4j.sizeOfDataType(DOUBLE), workspace.getPrimaryOffset());
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLoop4(Nd4jBackend backend) {
        Nd4jWorkspace workspace = (Nd4jWorkspace) Nd4j.getWorkspaceManager().createNewWorkspace(loopFirstConfig);

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertNotEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        assertEquals(0, workspace.getPrimaryOffset());

        try (MemoryWorkspace cW = workspace.notifyScopeEntered()) {
            INDArray array1 = Nd4j.create(DOUBLE, 100);
            INDArray array2 = Nd4j.create(DOUBLE, 100);
        }

        assertEquals(0, workspace.getPrimaryOffset());
        assertEquals(200 * Nd4j.sizeOfDataType(DOUBLE), workspace.getCurrentSize());

        try (MemoryWorkspace cW = workspace.notifyScopeEntered()) {
            INDArray array1 = Nd4j.create(DOUBLE, 100);

            assertEquals(100 * Nd4j.sizeOfDataType(DOUBLE), workspace.getPrimaryOffset());
        }

        assertEquals(0, workspace.getPrimaryOffset());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLoops3(Nd4jBackend backend) {
        Nd4jWorkspace workspace = (Nd4jWorkspace) Nd4j.getWorkspaceManager().createNewWorkspace(loopFirstConfig);

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertNotEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        assertEquals(0, workspace.getPrimaryOffset());

        workspace.notifyScopeEntered();

        INDArray arrayCold1 = Nd4j.create(DOUBLE, 100);
        INDArray arrayCold2 = Nd4j.create(DOUBLE, 10);

        assertEquals(0, workspace.getPrimaryOffset());
        assertEquals(0, workspace.getCurrentSize());

        workspace.notifyScopeLeft();

        assertEquals(0, workspace.getPrimaryOffset());

        long reqMem = 110 * Nd4j.sizeOfDataType(DOUBLE);

        assertEquals(reqMem + reqMem % 8, workspace.getCurrentSize());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLoops2(Nd4jBackend backend) {
        Nd4jWorkspace workspace = (Nd4jWorkspace) Nd4j.getWorkspaceManager().createNewWorkspace(loopOverTimeConfig);

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertNotEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        assertEquals(0, workspace.getPrimaryOffset());

        for (int x = 1; x <= 100; x++) {
            workspace.notifyScopeEntered();

            INDArray arrayCold = Nd4j.create(DOUBLE, x);

            assertEquals(0, workspace.getPrimaryOffset());
            assertEquals(0, workspace.getCurrentSize());

            workspace.notifyScopeLeft();
        }

        workspace.initializeWorkspace();

        long reqMem = 100 * Nd4j.sizeOfDataType(DOUBLE);

        //assertEquals(reqMem + reqMem % 8, workspace.getCurrentSize());
        assertEquals(0, workspace.getPrimaryOffset());

        workspace.notifyScopeEntered();

        INDArray arrayHot = Nd4j.create(DOUBLE, 10);

        reqMem = 10 * Nd4j.sizeOfDataType(DOUBLE);
        assertEquals(reqMem + reqMem % 8, workspace.getPrimaryOffset());

        workspace.notifyScopeLeft();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLoops1(Nd4jBackend backend) {
        Nd4jWorkspace workspace = (Nd4jWorkspace) Nd4j.getWorkspaceManager().createNewWorkspace(loopOverTimeConfig);

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertNotEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        assertEquals(0, workspace.getPrimaryOffset());

        workspace.notifyScopeEntered();

        INDArray arrayCold = Nd4j.create(DOUBLE, 10);

        assertEquals(0, workspace.getPrimaryOffset());
        assertEquals(0, workspace.getCurrentSize());

        arrayCold.assign(1.0f);

        assertEquals(10f, arrayCold.sumNumber().floatValue(), 0.01f);

        workspace.notifyScopeLeft();


        workspace.initializeWorkspace();
        long reqMemory = 11 * Nd4j.sizeOfDataType(arrayCold.dataType());
        assertEquals(reqMemory + reqMemory % 16, workspace.getCurrentSize());


        log.info("-----------------------");

        for (int x = 0; x < 10; x++) {
            assertEquals(0, workspace.getPrimaryOffset());

            workspace.notifyScopeEntered();

            INDArray array = Nd4j.create(DOUBLE, 10);


            long reqMem = 10 * Nd4j.sizeOfDataType(array.dataType());

            assertEquals(reqMem + reqMem % 16, workspace.getPrimaryOffset());

            array.addi(1.0);

            assertEquals(reqMem + reqMem % 16, workspace.getPrimaryOffset());

            assertEquals(10, array.sumNumber().doubleValue(), 0.01,"Failed on iteration " + x);

            workspace.notifyScopeLeft();

            assertEquals(0, workspace.getPrimaryOffset());
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllocation6(Nd4jBackend backend) {
        Nd4jWorkspace workspace = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfig, "testAllocation6");

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertNotEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        assertEquals(0, workspace.getPrimaryOffset());

        INDArray array = Nd4j.rand(DOUBLE, 100, 10);

        // checking if allocation actually happened
        assertEquals(1000 * Nd4j.sizeOfDataType(array.dataType()), workspace.getPrimaryOffset());

        INDArray dup = array.dup();

        assertEquals(2000 * Nd4j.sizeOfDataType(dup.dataType()), workspace.getPrimaryOffset());

        //assertEquals(5, dup.sumNumber().doubleValue(), 0.01);

        workspace.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllocation5(Nd4jBackend backend) {
        Nd4jWorkspace workspace = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfig, "testAllocation5");

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertNotEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        assertEquals(0, workspace.getPrimaryOffset());

        INDArray array = Nd4j.create(DOUBLE, new long[] {1, 5}, 'c');

        // checking if allocation actually happened
        long reqMemory = 5 * Nd4j.sizeOfDataType(DOUBLE);
        assertEquals(reqMemory + reqMemory % 16, workspace.getPrimaryOffset());

        array.assign(1.0f);

        INDArray dup = array.dup();

        assertEquals((reqMemory + reqMemory % 16) * 2, workspace.getPrimaryOffset());

        assertEquals(5, dup.sumNumber().doubleValue(), 0.01);

        workspace.close();
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllocation4(Nd4jBackend backend) {
        WorkspaceConfiguration failConfig = WorkspaceConfiguration.builder().initialSize(1024 * 1024)
                        .maxSize(1024 * 1024).overallocationLimit(0.1).policyAllocation(AllocationPolicy.STRICT)
                        .policyLearning(LearningPolicy.FIRST_LOOP).policyMirroring(MirroringPolicy.FULL)
                        .policySpill(SpillPolicy.FAIL).build();


        Nd4jWorkspace workspace = (Nd4jWorkspace) Nd4j.getWorkspaceManager().createNewWorkspace(failConfig);

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertNotEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        assertEquals(0, workspace.getPrimaryOffset());

        INDArray array = Nd4j.create(DOUBLE, new long[] {1, 5}, 'c');

        // checking if allocation actually happened
        long reqMem = 5 * Nd4j.sizeOfDataType(array.dataType());
        assertEquals(reqMem + reqMem % 16, workspace.getPrimaryOffset());

        try {
            INDArray array2 = Nd4j.create(DOUBLE, 10000000);
            assertTrue(false);
        } catch (ND4JIllegalStateException e) {
            assertTrue(true);
        }

        assertEquals(reqMem + reqMem % 16, workspace.getPrimaryOffset());

        INDArray array2 = Nd4j.create(DOUBLE, new long[] {1, 5}, 'c');

        assertEquals((reqMem + reqMem % 16) * 2, workspace.getPrimaryOffset());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllocation3(Nd4jBackend backend) {
        Nd4jWorkspace workspace = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfig,
                        "testAllocation2");

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertNotEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        assertEquals(0, workspace.getPrimaryOffset());

        INDArray array = Nd4j.create(DOUBLE, new long[] {1, 5}, 'c');

        // checking if allocation actually happened
        long reqMem = 5 * Nd4j.sizeOfDataType(array.dataType());
        assertEquals(reqMem + reqMem % 16, workspace.getPrimaryOffset());

        array.assign(1.0f);

        assertEquals(5, array.sumNumber().doubleValue(), 0.01);

        workspace.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllocation2(Nd4jBackend backend) {
        Nd4jWorkspace workspace = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfig,
                        "testAllocation2");

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertNotEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        assertEquals(0, workspace.getPrimaryOffset());

        INDArray array = Nd4j.create(DOUBLE, 5);

        // checking if allocation actually happened
        long reqMem = 5 * Nd4j.sizeOfDataType(array.dataType());
        assertEquals(reqMem + reqMem % 16, workspace.getPrimaryOffset());

        array.assign(1.0f);

        assertEquals(5, array.sumNumber().doubleValue(), 0.01);

        workspace.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllocation1(Nd4jBackend backend) {



        INDArray exp = Nd4j.create(new double[] {1f, 2f, 3f, 4f, 5f});

        Nd4jWorkspace workspace = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfig,
                "TestAllocation1");

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertNotEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        assertEquals(0, workspace.getPrimaryOffset());

        INDArray array = Nd4j.create(new double[] {1f, 2f, 3f, 4f, 5f});

        // checking if allocation actually happened
        long reqMem = 5 * Nd4j.sizeOfDataType(array.dataType());
        assertEquals(reqMem + reqMem % 16, workspace.getPrimaryOffset());


        assertEquals(exp, array);

        // checking stuff at native side
        double sum = array.sumNumber().doubleValue();
        assertEquals(15.0, sum, 0.01);

        // checking INDArray validity
        assertEquals(1.0, array.getFloat(0), 0.01);
        assertEquals(2.0, array.getFloat(1), 0.01);
        assertEquals(3.0, array.getFloat(2), 0.01);
        assertEquals(4.0, array.getFloat(3), 0.01);
        assertEquals(5.0, array.getFloat(4), 0.01);


        // checking INDArray validity
        assertEquals(1.0, array.getDouble(0), 0.01);
        assertEquals(2.0, array.getDouble(1), 0.01);
        assertEquals(3.0, array.getDouble(2), 0.01);
        assertEquals(4.0, array.getDouble(3), 0.01);
        assertEquals(5.0, array.getDouble(4), 0.01);

        // checking workspace memory space

        INDArray array2 = Nd4j.create(new double[] {5f, 4f, 3f, 2f, 1f});

        sum = array2.sumNumber().doubleValue();
        assertEquals(15.0, sum, 0.01);

        // 44 = 20 + 4 + 20, 4 was allocated as Op.extraArgs for sum
        //assertEquals(44, workspace.getPrimaryOffset());


        array.addi(array2);

        sum = array.sumNumber().doubleValue();
        assertEquals(30.0, sum, 0.01);


        // checking INDArray validity
        assertEquals(6.0, array.getFloat(0), 0.01);
        assertEquals(6.0, array.getFloat(1), 0.01);
        assertEquals(6.0, array.getFloat(2), 0.01);
        assertEquals(6.0, array.getFloat(3), 0.01);
        assertEquals(6.0, array.getFloat(4), 0.01);

        workspace.close();
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMmap1(Nd4jBackend backend) {
        // we don't support MMAP on cuda yet
        if (Nd4j.getExecutioner().getClass().getName().toLowerCase().contains("cuda"))
            return;

        WorkspaceConfiguration mmap = WorkspaceConfiguration.builder()
                .initialSize(1000000)
                .policyLocation(LocationPolicy.MMAP)
                .policyLearning(LearningPolicy.NONE)
                .build();

        MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(mmap, "M2");

        INDArray mArray = Nd4j.create(DOUBLE, 100);
        mArray.assign(10f);

        assertEquals(1000f, mArray.sumNumber().floatValue(), 1e-5);

        ws.close();


        ws.notifyScopeEntered();

        INDArray mArrayR = Nd4j.createUninitialized(DOUBLE, 100);
        assertEquals(1000f, mArrayR.sumNumber().floatValue(), 1e-5);

        ws.close();
    }


    @Test
    @Disabled
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMmap2(Nd4jBackend backend) throws Exception {
        // we don't support MMAP on cuda yet
        if (Nd4j.getExecutioner().getClass().getName().toLowerCase().contains("cuda"))
            return;

        File tmp = File.createTempFile("tmp", "fdsfdf");
        tmp.deleteOnExit();
        Nd4jWorkspace.fillFile(tmp, 100000);

        WorkspaceConfiguration mmap = WorkspaceConfiguration.builder()
                .policyLocation(LocationPolicy.MMAP)
                .tempFilePath(tmp.getAbsolutePath())
                .build();

        MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(mmap, "M3");

        INDArray mArray = Nd4j.create(DOUBLE, 100);
        mArray.assign(10f);

        assertEquals(1000f, mArray.sumNumber().floatValue(), 1e-5);

        ws.notifyScopeLeft();
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testInvalidLeverageMigrateDetach(Nd4jBackend backend){

        try {
            MemoryWorkspace ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(basicConfig, "testInvalidLeverage");

            INDArray invalidArray = null;

            for (int i = 0; i < 10; i++) {
                try (MemoryWorkspace ws2 = ws.notifyScopeEntered()) {
                    invalidArray = Nd4j.linspace(1, 10, 10, DOUBLE);
                }
            }
            assertTrue(invalidArray.isAttached());

            MemoryWorkspace ws2 = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(basicConfig, "testInvalidLeverage2");

            //Leverage
            try (MemoryWorkspace ws3 = ws2.notifyScopeEntered()) {
                invalidArray.leverage();
                fail("Exception should be thrown");
            } catch (ND4JWorkspaceException e) {
                //Expected exception
                log.info("Expected exception: {}", e.getMessage());
            }

            try (MemoryWorkspace ws3 = ws2.notifyScopeEntered()) {
                invalidArray.leverageTo("testInvalidLeverage2");
                fail("Exception should be thrown");
            } catch (ND4JWorkspaceException e) {
                //Expected exception
                log.info("Expected exception: {}", e.getMessage());
            }

            try (MemoryWorkspace ws3 = ws2.notifyScopeEntered()) {
                invalidArray.leverageOrDetach("testInvalidLeverage2");
                fail("Exception should be thrown");
            } catch (ND4JWorkspaceException e) {
                //Expected exception
                log.info("Expected exception: {}", e.getMessage());
            }

            try {
                invalidArray.leverageTo("testInvalidLeverage2");
                fail("Exception should be thrown");
            } catch (ND4JWorkspaceException e) {
                //Expected exception
                log.info("Expected exception: {}", e.getMessage());
            }

            //Detach
            try{
                invalidArray.detach();
                fail("Exception should be thrown");
            } catch (ND4JWorkspaceException e){
                log.info("Expected exception: {}", e.getMessage());
            }


            //Migrate
            try (MemoryWorkspace ws3 = ws2.notifyScopeEntered()) {
                invalidArray.migrate();
                fail("Exception should be thrown");
            } catch (ND4JWorkspaceException e) {
                //Expected exception
                log.info("Expected exception: {}", e.getMessage());
            }

            try {
                invalidArray.migrate(true);
                fail("Exception should be thrown");
            } catch (ND4JWorkspaceException e) {
                //Expected exception
                log.info("Expected exception: {}", e.getMessage());
            }


            //Dup
            try{
                invalidArray.dup();
                fail("Exception should be thrown");
            } catch (ND4JWorkspaceException e){
                log.info("Expected exception: {}", e.getMessage());
            }

            //Unsafe dup:
            try{
                invalidArray.unsafeDuplication();
                fail("Exception should be thrown");
            } catch (ND4JWorkspaceException e){
                log.info("Expected exception: {}", e.getMessage());
            }

            try{
                invalidArray.unsafeDuplication(true);
                fail("Exception should be thrown");
            } catch (ND4JWorkspaceException e){
                log.info("Expected exception: {}", e.getMessage());
            }


        } finally {
            Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBadGenerationLeverageMigrateDetach(Nd4jBackend backend){
        INDArray gen2 = null;

        for (int i = 0; i < 4; i++) {
            MemoryWorkspace wsOuter = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(basicConfig, "testBadGeneration");

            try (MemoryWorkspace wsOuter2 = wsOuter.notifyScopeEntered()) {
                INDArray arr = Nd4j.linspace(1, 10, 10, DOUBLE);
                if (i == 2) {
                    gen2 = arr;
                }

                if (i == 3) {
                    MemoryWorkspace wsInner = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(basicConfig, "testBadGeneration2");
                    try (MemoryWorkspace wsInner2 = wsInner.notifyScopeEntered()) {

                        //Leverage
                        try {
                            gen2.leverage();
                            fail("Exception should be thrown");
                        } catch (ND4JWorkspaceException e) {
                            //Expected exception
                            log.info("Expected exception: {}", e.getMessage());
                        }

                        try {
                            gen2.leverageTo("testBadGeneration2");
                            fail("Exception should be thrown");
                        } catch (ND4JWorkspaceException e) {
                            //Expected exception
                            log.info("Expected exception: {}", e.getMessage());
                        }

                        try {
                            gen2.leverageOrDetach("testBadGeneration2");
                            fail("Exception should be thrown");
                        } catch (ND4JWorkspaceException e) {
                            //Expected exception
                            log.info("Expected exception: {}", e.getMessage());
                        }

                        try {
                            gen2.leverageTo("testBadGeneration2");
                            fail("Exception should be thrown");
                        } catch (ND4JWorkspaceException e) {
                            //Expected exception
                            log.info("Expected exception: {}", e.getMessage());
                        }

                        //Detach
                        try {
                            gen2.detach();
                            fail("Exception should be thrown");
                        } catch (ND4JWorkspaceException e) {
                            log.info("Expected exception: {}", e.getMessage());
                        }


                        //Migrate
                        try {
                            gen2.migrate();
                            fail("Exception should be thrown");
                        } catch (ND4JWorkspaceException e) {
                            //Expected exception
                            log.info("Expected exception: {}", e.getMessage());
                        }

                        try {
                            gen2.migrate(true);
                            fail("Exception should be thrown");
                        } catch (ND4JWorkspaceException e) {
                            //Expected exception
                            log.info("Expected exception: {}", e.getMessage());
                        }


                        //Dup
                        try {
                            gen2.dup();
                            fail("Exception should be thrown");
                        } catch (ND4JWorkspaceException e) {
                            log.info("Expected exception: {}", e.getMessage());
                        }

                        //Unsafe dup:
                        try {
                            gen2.unsafeDuplication();
                            fail("Exception should be thrown");
                        } catch (ND4JWorkspaceException e) {
                            log.info("Expected exception: {}", e.getMessage());
                        }

                        try {
                            gen2.unsafeDuplication(true);
                            fail("Exception should be thrown");
                        } catch (ND4JWorkspaceException e) {
                            log.info("Expected exception: {}", e.getMessage());
                        }
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDtypeLeverage(Nd4jBackend backend){

        for(DataType globalDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
            for (DataType arrayDType : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
                Nd4j.setDefaultDataTypes(globalDtype, globalDtype);

                WorkspaceConfiguration configOuter = WorkspaceConfiguration.builder().initialSize(10 * 1024L * 1024L)
                        .policyAllocation(AllocationPolicy.OVERALLOCATE).policyLearning(LearningPolicy.NONE).build();
                WorkspaceConfiguration configInner = WorkspaceConfiguration.builder().initialSize(10 * 1024L * 1024L)
                        .policyAllocation(AllocationPolicy.OVERALLOCATE).policyLearning(LearningPolicy.NONE).build();

                try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(configOuter, "ws")) {
                    INDArray arr = Nd4j.create(arrayDType, 3, 4);
                    try (MemoryWorkspace wsInner = Nd4j.getWorkspaceManager().getAndActivateWorkspace(configOuter, "wsInner")) {
                        INDArray leveraged = arr.leverageTo("ws");
                        assertTrue(leveraged.isAttached());
                        assertEquals(arrayDType, leveraged.dataType());

                        INDArray detached = leveraged.detach();
                        assertFalse(detached.isAttached());
                        assertEquals(arrayDType, detached.dataType());
                    }
                }
            }
        }
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCircularWorkspaceAsymmetry_1(Nd4jBackend backend) {
        // nothing to test on CPU here
        if (Nd4j.getEnvironment().isCPU())
            return;

        // circular workspace mode
        val configuration = WorkspaceConfiguration.builder().initialSize(10 * 1024 * 1024)
                .policyReset(ResetPolicy.ENDOFBUFFER_REACHED).policyAllocation(AllocationPolicy.STRICT)
                .policySpill(SpillPolicy.FAIL).policyLearning(LearningPolicy.NONE).build();


        try (val ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(configuration, "circular_ws")) {
            val array = Nd4j.create(DataType.FLOAT, 10, 10);

            // we expect that this array has no data/buffer on HOST side
            assertEquals(AffinityManager.Location.DEVICE, Nd4j.getAffinityManager().getActiveLocation(array));

            // since this array doesn't have HOST buffer - it will allocate one now
            array.getDouble(3L);
        }

        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
