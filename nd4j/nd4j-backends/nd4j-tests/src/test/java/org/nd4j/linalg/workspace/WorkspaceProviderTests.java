/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.workspace;

import lombok.extern.slf4j.Slf4j;
import org.junit.After;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.memory.abstracts.Nd4jWorkspace;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@RunWith(Parameterized.class)
public class WorkspaceProviderTests extends BaseNd4jTest {
    private static final WorkspaceConfiguration basicConfiguration = WorkspaceConfiguration.builder().initialSize(81920)
                    .overallocationLimit(0.1).policySpill(SpillPolicy.EXTERNAL).policyLearning(LearningPolicy.NONE)
                    .policyMirroring(MirroringPolicy.FULL).policyAllocation(AllocationPolicy.OVERALLOCATE).build();

    private static final WorkspaceConfiguration bigConfiguration = WorkspaceConfiguration.builder()
                    .initialSize(20 * 1024 * 1024L).overallocationLimit(0.1).policySpill(SpillPolicy.EXTERNAL)
                    .policyLearning(LearningPolicy.NONE).policyMirroring(MirroringPolicy.FULL)
                    .policyAllocation(AllocationPolicy.OVERALLOCATE).build();

    private static final WorkspaceConfiguration loopConfiguration = WorkspaceConfiguration.builder().initialSize(0)
                    .overallocationLimit(0.1).policySpill(SpillPolicy.EXTERNAL).policyLearning(LearningPolicy.OVER_TIME)
                    .policyMirroring(MirroringPolicy.FULL).policyAllocation(AllocationPolicy.STRICT).build();


    private static final WorkspaceConfiguration delayedConfiguration = WorkspaceConfiguration.builder().initialSize(0)
                    .overallocationLimit(0.1).policySpill(SpillPolicy.EXTERNAL).policyLearning(LearningPolicy.OVER_TIME)
                    .policyMirroring(MirroringPolicy.FULL).cyclesBeforeInitialization(3)
                    .policyAllocation(AllocationPolicy.STRICT).build();

    private static final WorkspaceConfiguration reallocateConfiguration = WorkspaceConfiguration.builder()
                    .initialSize(0).overallocationLimit(0.1).policySpill(SpillPolicy.REALLOCATE)
                    .policyLearning(LearningPolicy.OVER_TIME).policyMirroring(MirroringPolicy.FULL)
                    .policyAllocation(AllocationPolicy.STRICT).build();

    private static final WorkspaceConfiguration reallocateDelayedConfiguration = WorkspaceConfiguration.builder()
                    .initialSize(0).overallocationLimit(0.1).policySpill(SpillPolicy.REALLOCATE)
                    .cyclesBeforeInitialization(3).policyLearning(LearningPolicy.OVER_TIME)
                    .policyMirroring(MirroringPolicy.FULL).policyAllocation(AllocationPolicy.STRICT).build();


    private static final WorkspaceConfiguration reallocateUnspecifiedConfiguration = WorkspaceConfiguration.builder()
                    .initialSize(0).overallocationLimit(0.0).policySpill(SpillPolicy.REALLOCATE)
                    .policyLearning(LearningPolicy.OVER_TIME).policyMirroring(MirroringPolicy.FULL)
                    .policyAllocation(AllocationPolicy.OVERALLOCATE).policyReset(ResetPolicy.BLOCK_LEFT).build();



    private static final WorkspaceConfiguration firstConfiguration = WorkspaceConfiguration.builder().initialSize(0)
                    .overallocationLimit(0.1).policySpill(SpillPolicy.EXTERNAL)
                    .policyLearning(LearningPolicy.FIRST_LOOP).policyMirroring(MirroringPolicy.FULL)
                    .policyAllocation(AllocationPolicy.STRICT).build();


    private static final WorkspaceConfiguration circularConfiguration = WorkspaceConfiguration.builder()
                    .minSize(10 * 1024L * 1024L).overallocationLimit(1.0).policySpill(SpillPolicy.EXTERNAL)
                    .policyLearning(LearningPolicy.FIRST_LOOP).policyMirroring(MirroringPolicy.FULL)
                    .policyAllocation(AllocationPolicy.STRICT).policyReset(ResetPolicy.ENDOFBUFFER_REACHED).build();


    private static final WorkspaceConfiguration adsiConfiguration =
                    WorkspaceConfiguration.builder().overallocationLimit(3.0).policySpill(SpillPolicy.REALLOCATE)
                                    .policyLearning(LearningPolicy.FIRST_LOOP).policyMirroring(MirroringPolicy.FULL)
                                    .policyAllocation(AllocationPolicy.OVERALLOCATE)
                                    .policyReset(ResetPolicy.ENDOFBUFFER_REACHED).build();

    DataType initialType;

    public WorkspaceProviderTests(Nd4jBackend backend) {
        super(backend);
        this.initialType = Nd4j.dataType();
    }

    @After
    public void shutUp() throws Exception {
        Nd4j.getMemoryManager().setCurrentWorkspace(null);
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        Nd4j.setDataType(this.initialType);
    }

    /**
     * This simple test checks for over-time learning with coefficient applied
     *
     * @throws Exception
     */
    @Test
    public void testUnboundedLoop2() throws Exception {
        WorkspaceConfiguration configuration =
                        WorkspaceConfiguration.builder().initialSize(0).policyReset(ResetPolicy.ENDOFBUFFER_REACHED)
                                        .policyAllocation(AllocationPolicy.OVERALLOCATE).overallocationLimit(4.0)
                                        .policyLearning(LearningPolicy.OVER_TIME).cyclesBeforeInitialization(5).build();

        Nd4jWorkspace ws1 =
                        (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(configuration, "ITER");

        long requiredMemory = 100 * Nd4j.sizeOfDataType();
        long shiftedSize = ((long) (requiredMemory * 1.3)) + (8 - (((long) (requiredMemory * 1.3)) % 8));

        for (int x = 0; x < 100; x++) {
            try (Nd4jWorkspace wsI = (Nd4jWorkspace) Nd4j.getWorkspaceManager()
                            .getWorkspaceForCurrentThread(configuration, "ITER").notifyScopeEntered()) {
                INDArray array = Nd4j.create(100);
            }

            // only checking after workspace is initialized
            if (x > 4) {
                assertEquals(shiftedSize, ws1.getInitialBlockSize());
                assertEquals(5 * shiftedSize, ws1.getCurrentSize());
            } else if (x < 4) {
                // we're making sure we're not initialize early
                assertEquals("Failed on iteration " + x, 0, ws1.getCurrentSize());
            }
        }

        // maximum allocation amount is 100 elements during learning, and additional coefficient is 4.0. result is workspace of 500 elements
        assertEquals(5 * shiftedSize, ws1.getCurrentSize());

        assertNull(Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    @Test
    public void testUnboundedLoop1() throws Exception {
        WorkspaceConfiguration configuration = WorkspaceConfiguration.builder()
                        .initialSize(100 * 100 * Nd4j.sizeOfDataType()).policyReset(ResetPolicy.ENDOFBUFFER_REACHED)
                        .policyAllocation(AllocationPolicy.STRICT).build();

        for (int x = 0; x < 100; x++) {
            try (Nd4jWorkspace ws1 = (Nd4jWorkspace) Nd4j.getWorkspaceManager()
                            .getWorkspaceForCurrentThread(configuration, "ITER").notifyScopeEntered()) {

                INDArray array = Nd4j.create(100);
            }

            Nd4jWorkspace ws1 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(configuration,
                            "ITER");

            assertEquals((x + 1) * 100 * Nd4j.sizeOfDataType(), ws1.getHostOffset());
        }

        Nd4jWorkspace ws1 =
                        (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(configuration, "ITER");
        assertEquals(100 * 100 * Nd4j.sizeOfDataType(), ws1.getHostOffset());

        // just to trigger reset
        ws1.notifyScopeEntered();

        // confirming reset
        //        assertEquals(0, ws1.getHostOffset());

        ws1.notifyScopeLeft();

        assertNull(Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    @Test
    public void testMultithreading1() throws Exception {
        final List<MemoryWorkspace> workspaces = new CopyOnWriteArrayList<>();
        Nd4j.getWorkspaceManager().setDefaultWorkspaceConfiguration(basicConfiguration);

        Thread[] threads = new Thread[20];
        for (int x = 0; x < threads.length; x++) {
            threads[x] = new Thread(new Runnable() {
                @Override
                public void run() {
                    MemoryWorkspace workspace = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread();
                    workspaces.add(workspace);
                }
            });

            threads[x].start();
        }

        for (int x = 0; x < threads.length; x++) {
            threads[x].join();
        }

        for (int x = 0; x < threads.length; x++) {
            for (int y = 0; y < threads.length; y++) {
                if (x == y)
                    continue;

                assertFalse(workspaces.get(x) == workspaces.get(y));
            }
        }

        assertNull(Nd4j.getMemoryManager().getCurrentWorkspace());
    }


    @Test
    public void testNestedWorkspacesOverlap2() throws Exception {
        Nd4j.getWorkspaceManager().setDefaultWorkspaceConfiguration(basicConfiguration);

        assertFalse(Nd4j.getWorkspaceManager().checkIfWorkspaceExists("WS1"));
        assertFalse(Nd4j.getWorkspaceManager().checkIfWorkspaceExists("WS2"));

        try (Nd4jWorkspace ws1 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS1")
                        .notifyScopeEntered()) {
            INDArray array = Nd4j.create(new float[] {6f, 3f, 1f, 9f, 21f});
            INDArray array3 = null;

            long reqMem = 5 * Nd4j.sizeOfDataType();
            assertEquals(reqMem + reqMem % 8, ws1.getHostOffset());
            try (Nd4jWorkspace ws2 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS2")
                            .notifyScopeEntered()) {

                INDArray array2 = Nd4j.create(new float[] {1f, 2f, 3f, 4f, 5f});

                reqMem = 5 * Nd4j.sizeOfDataType();
                assertEquals(reqMem + reqMem % 8, ws1.getHostOffset());
                assertEquals(reqMem + reqMem % 8, ws2.getHostOffset());

                try (Nd4jWorkspace ws3 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS1")
                                .notifyScopeBorrowed()) {
                    assertTrue(ws1 == ws3);
                    assertTrue(ws1 == Nd4j.getMemoryManager().getCurrentWorkspace());

                    array3 = array2.unsafeDuplication();
                    assertTrue(ws1 == array3.data().getParentWorkspace());
                    assertEquals(reqMem + reqMem % 8, ws2.getHostOffset());
                    assertEquals((reqMem + reqMem % 8) * 2, ws1.getHostOffset());
                }

                log.info("Current workspace: {}", Nd4j.getMemoryManager().getCurrentWorkspace());
                assertTrue(ws2 == Nd4j.getMemoryManager().getCurrentWorkspace());

                assertEquals(reqMem + reqMem % 8, ws2.getHostOffset());
                assertEquals((reqMem + reqMem % 8) * 2, ws1.getHostOffset());

                assertEquals(15f, array3.sumNumber().floatValue(), 0.01f);
            }
        }

        log.info("------");

        assertNull(Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    @Test
    public void testNestedWorkspacesOverlap1() throws Exception {
        Nd4j.getWorkspaceManager().setDefaultWorkspaceConfiguration(basicConfiguration);
        try (Nd4jWorkspace ws1 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS1")
                        .notifyScopeEntered()) {
            INDArray array = Nd4j.create(new float[] {1f, 2f, 3f, 4f, 5f});

            long reqMem = 5 * Nd4j.sizeOfDataType();
            assertEquals(reqMem + reqMem % 8, ws1.getHostOffset());
            try (Nd4jWorkspace ws2 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS2")
                            .notifyScopeEntered()) {

                INDArray array2 = Nd4j.create(new float[] {1f, 2f, 3f, 4f, 5f});

                reqMem = 5 * Nd4j.sizeOfDataType();
                assertEquals(reqMem + reqMem % 8, ws1.getHostOffset());
                assertEquals(reqMem + reqMem % 8, ws2.getHostOffset());

                try (Nd4jWorkspace ws3 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS1")
                                .notifyScopeBorrowed()) {
                    assertTrue(ws1 == ws3);

                    INDArray array3 = Nd4j.create(new float[] {1f, 2f, 3f, 4f, 5f});

                    assertEquals(reqMem + reqMem % 8, ws2.getHostOffset());
                    assertEquals((reqMem + reqMem % 8) * 2, ws1.getHostOffset());
                }
            }
        }

        assertNull(Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    @Test
    public void testWorkspacesSerde3() throws Exception {
        INDArray array = Nd4j.create(10).assign(1.0);
        INDArray restored = null;

        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(bos);
        Nd4j.write(array, dos);

        try (Nd4jWorkspace workspace = (Nd4jWorkspace) Nd4j.getWorkspaceManager()
                        .getAndActivateWorkspace(basicConfiguration, "WS_1")) {

            try (MemoryWorkspace wsO = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                workspace.enableDebug(true);

                ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
                DataInputStream dis = new DataInputStream(bis);
                restored = Nd4j.read(dis);

                assertEquals(0, workspace.getHostOffset());

                assertEquals(array.length(), restored.length());
                assertEquals(1.0f, restored.meanNumber().floatValue(), 1.0f);

                // we want to ensure it's the same cached shapeInfo used here
                assertTrue(array.shapeInfoDataBuffer() == restored.shapeInfoDataBuffer());
            }
        }
    }



    @Test
    public void testWorkspacesSerde2() throws Exception {
        INDArray array = Nd4j.create(10).assign(1.0);
        INDArray restored = null;

        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(bos);
        Nd4j.write(array, dos);

        try (Nd4jWorkspace workspace = (Nd4jWorkspace) Nd4j.getWorkspaceManager()
                        .getAndActivateWorkspace(basicConfiguration, "WS_1")) {
            workspace.enableDebug(true);

            ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
            DataInputStream dis = new DataInputStream(bis);
            restored = Nd4j.read(dis);

            long requiredMemory = 10 * Nd4j.sizeOfDataType();
            assertEquals(requiredMemory + requiredMemory % 8, workspace.getHostOffset());

            assertEquals(array.length(), restored.length());
            assertEquals(1.0f, restored.meanNumber().floatValue(), 1.0f);

            // we want to ensure it's the same cached shapeInfo used here
            assertTrue(array.shapeInfoDataBuffer() == restored.shapeInfoDataBuffer());
        }
    }

    @Test
    public void testWorkspacesSerde1() throws Exception {
        int[] shape = new int[] {17, 57, 79};
        INDArray array = Nd4j.create(shape).assign(1.0);
        INDArray restored = null;

        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(bos);
        Nd4j.write(array, dos);

        try (MemoryWorkspace workspace = Nd4j.getWorkspaceManager().getAndActivateWorkspace(bigConfiguration, "WS_1")) {
            ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
            DataInputStream dis = new DataInputStream(bis);
            restored = Nd4j.read(dis);

            assertEquals(array.length(), restored.length());
            assertEquals(1.0f, restored.meanNumber().floatValue(), 1.0f);

            // we want to ensure it's the same cached shapeInfo used here
            assertTrue(array.shapeInfoDataBuffer() == restored.shapeInfoDataBuffer());
        }
    }


    @Test
    public void testCircularBufferReset1() throws Exception {
        Nd4jWorkspace workspace = (Nd4jWorkspace) Nd4j.getWorkspaceManager()
                        .getWorkspaceForCurrentThread(circularConfiguration, "WSR_1");

        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace("WSR_1")) {
            Nd4j.create(10000);
            assertEquals(0, workspace.getCurrentSize());
            assertEquals(1, workspace.getNumberOfExternalAllocations());
        }

        assertEquals(10 * 1024L * 1024L, workspace.getCurrentSize());
        assertEquals(0, workspace.getHostOffset());
        assertEquals(1, workspace.getNumberOfExternalAllocations());

        for (int i = 0; i < 11 * 1024 * 1024; i += 10000 * Nd4j.sizeOfDataType()) {
            try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace("WSR_1")) {
                Nd4j.create(10000);
            }

            /*
            if (i < 10480000)
                assertEquals("I: " + i,1, workspace.getNumberOfExternalAllocations());
            else
                assertEquals(0, workspace.getNumberOfExternalAllocations());
                */
        }

        assertEquals(0, workspace.getNumberOfExternalAllocations());

    }

    @Test
    public void testVariableInput1() throws Exception {
        Nd4jWorkspace workspace = (Nd4jWorkspace) Nd4j.getWorkspaceManager()
                        .getWorkspaceForCurrentThread(adsiConfiguration, "ADSI");

        INDArray array1 = null;
        INDArray array2 = null;

        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(adsiConfiguration, "ADSI")) {
            // we allocate first element smaller then subsequent;
            array1 = Nd4j.create(8, 128, 100);
        }

        long requiredMemory = 8 * 128 * 100 * Nd4j.sizeOfDataType();
        long shiftedSize = ((long) (requiredMemory * 1.3)) + (8 - (((long) (requiredMemory * 1.3)) % 8));
        assertEquals(shiftedSize, workspace.getInitialBlockSize());
        assertEquals(shiftedSize * 4, workspace.getCurrentSize());
        assertEquals(0, workspace.getHostOffset());
        assertEquals(0, workspace.getDeviceOffset());

        assertEquals(1, workspace.getCyclesCount());
        assertEquals(0, workspace.getStepNumber());


        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(adsiConfiguration, "ADSI")) {
            // allocating same shape
            array1 = Nd4j.create(8, 128, 100);
        }

        assertEquals(workspace.getInitialBlockSize(), workspace.getHostOffset());
        assertEquals(workspace.getInitialBlockSize(), workspace.getDeviceOffset());

        assertEquals(2, workspace.getCyclesCount());
        assertEquals(0, workspace.getStepNumber());


        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(adsiConfiguration, "ADSI")) {
            // allocating bigger shape
            array1 = Nd4j.create(8, 128, 200).assign(1.0);
        }

        // offsets should be intact, allocation happened as pinned
        assertEquals(workspace.getInitialBlockSize(), workspace.getHostOffset());
        assertEquals(workspace.getInitialBlockSize(), workspace.getDeviceOffset());

        assertEquals(1, workspace.getNumberOfPinnedAllocations());

        assertEquals(3, workspace.getCyclesCount());
        assertEquals(0, workspace.getStepNumber());


        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(adsiConfiguration, "ADSI")) {
            // allocating same shape
            array1 = Nd4j.create(8, 128, 100);
        }

        assertEquals(2, workspace.getNumberOfPinnedAllocations());
        assertEquals(0, workspace.getStepNumber());
        assertEquals(4, workspace.getCyclesCount());

        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(adsiConfiguration, "ADSI")) {
            // allocating same shape
            array1 = Nd4j.create(8, 128, 100);
        }

        assertEquals(3, workspace.getNumberOfPinnedAllocations());
        assertEquals(1, workspace.getStepNumber());
        assertEquals(5, workspace.getCyclesCount());

        for (int i = 0; i < 12; i++) {
            try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(adsiConfiguration, "ADSI")) {
                // allocating same shape
                array1 = Nd4j.create(8, 128, 100);
            }
        }

        // Now we know that workspace was reallocated and offset was shifted to the end of workspace
        assertEquals(4, workspace.getStepNumber());

        requiredMemory = 8 * 128 * 200 * Nd4j.sizeOfDataType();
        shiftedSize = ((long) (requiredMemory * 1.3)) + (8 - (((long) (requiredMemory * 1.3)) % 8));

        assertEquals(shiftedSize * 4, workspace.getCurrentSize());
        assertEquals(workspace.getCurrentSize(), workspace.getHostOffset());
        assertEquals(workspace.getCurrentSize(), workspace.getDeviceOffset());

    }

    @Test
    public void testReallocate3() throws Exception {
        MemoryWorkspace workspace = Nd4j.getWorkspaceManager()
                        .getWorkspaceForCurrentThread(reallocateUnspecifiedConfiguration, "WS_1");

        for (int i = 1; i <= 10; i++) {
            try (MemoryWorkspace ws = Nd4j.getWorkspaceManager()
                            .getAndActivateWorkspace(reallocateUnspecifiedConfiguration, "WS_1")) {
                INDArray array = Nd4j.create(100 * i);
            }

            if (i == 3) {
                workspace.initializeWorkspace();
                assertEquals("Failed on iteration " + i, 100 * i * Nd4j.sizeOfDataType(), workspace.getCurrentSize());
            }
        }

        log.info("-----------------------------");

        for (int i = 10; i > 0; i--) {
            try (MemoryWorkspace ws = Nd4j.getWorkspaceManager()
                            .getAndActivateWorkspace(reallocateUnspecifiedConfiguration, "WS_1")) {
                INDArray array = Nd4j.create(100 * i);
            }
        }

        workspace.initializeWorkspace();
        assertEquals("Failed on final", 100 * 10 * Nd4j.sizeOfDataType(), workspace.getCurrentSize());
    }

    @Test
    public void testReallocate2() throws Exception {
        MemoryWorkspace workspace =
                        Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(reallocateDelayedConfiguration, "WS_1");

        for (int i = 1; i <= 10; i++) {
            try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(reallocateDelayedConfiguration,
                            "WS_1")) {
                INDArray array = Nd4j.create(100 * i);
            }

            if (i >= 3)
                assertEquals("Failed on iteration " + i, 100 * i * Nd4j.sizeOfDataType(), workspace.getCurrentSize());
            else
                assertEquals(0, workspace.getCurrentSize());
        }
    }

    @Test
    public void testCircularLearning1() throws Exception {
        INDArray array1;
        INDArray array2;
        for (int i = 0; i < 2; i++) {
            try (MemoryWorkspace workspace =
                            Nd4j.getWorkspaceManager().getAndActivateWorkspace(circularConfiguration, "WSX")) {
                array1 = Nd4j.create(10).assign(1);
            }

            Nd4jWorkspace workspace = (Nd4jWorkspace) Nd4j.getWorkspaceManager()
                            .getWorkspaceForCurrentThread(circularConfiguration, "WSX");
            assertEquals(10 * 1024 * 1024L, workspace.getCurrentSize());
            log.info("Current step number: {}", workspace.getStepNumber());
            if (i == 0)
                assertEquals(0, workspace.getHostOffset());
            else if (i == 1)
                assertEquals(workspace.getInitialBlockSize(), workspace.getHostOffset());
        }

    }

    @Test
    public void testReallocate1() throws Exception {
        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(reallocateConfiguration, "WS_1")) {
            INDArray array = Nd4j.create(100);
        }



        Nd4jWorkspace workspace = (Nd4jWorkspace) Nd4j.getWorkspaceManager()
                        .getWorkspaceForCurrentThread(reallocateConfiguration, "WS_1");
        workspace.initializeWorkspace();

        assertEquals(100 * Nd4j.sizeOfDataType(), workspace.getCurrentSize());

        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(reallocateConfiguration, "WS_1")) {
            INDArray array = Nd4j.create(1000);
        }

        assertEquals(1000 * Nd4j.sizeOfDataType(), workspace.getMaxCycleAllocations());

        workspace.initializeWorkspace();

        assertEquals(1000 * Nd4j.sizeOfDataType(), workspace.getCurrentSize());

        // now we're working on reallocated array, that should be able to hold >100 elements
        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(reallocateConfiguration, "WS_1")) {
            INDArray array = Nd4j.create(500).assign(1.0);

            assertEquals(1.0, array.meanNumber().doubleValue(), 0.01);
        }
    }

    @Test
    public void testNestedWorkspaces11() throws Exception {
        for (int x = 1; x < 10; x++) {
            try (MemoryWorkspace ws1 = Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfiguration, "WS_1")) {
                INDArray array1 = Nd4j.create(100 * x);

                for (int i = 1; i < 10; i++) {
                    try (MemoryWorkspace ws2 =
                                    Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfiguration, "WS_1")) {
                        INDArray array2 = Nd4j.create(100 * x);
                        for (int e = 1; e < 10; e++) {
                            try (MemoryWorkspace ws3 = Nd4j.getWorkspaceManager()
                                            .getWorkspaceForCurrentThread(basicConfiguration, "WS_1")
                                            .notifyScopeBorrowed()) {
                                INDArray array3 = Nd4j.create(100 * x);
                            }
                        }
                    }
                }
            }
        }
    }


    @Test
    public void testNestedWorkspaces10() throws Exception {
        for (int x = 1; x < 10; x++) {
            try (MemoryWorkspace ws1 = Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfiguration, "WS_1")) {
                INDArray array1 = Nd4j.create(100 * x);
                try (MemoryWorkspace ws2 =
                                Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfiguration, "WS_1")) {
                    INDArray array2 = Nd4j.create(100 * x);
                    try (MemoryWorkspace ws3 = Nd4j.getWorkspaceManager()
                                    .getWorkspaceForCurrentThread(basicConfiguration, "WS_1").notifyScopeBorrowed()) {
                        INDArray array3 = Nd4j.create(100 * x);
                    }

                }
            }
        }
    }


    @Test
    public void testNestedWorkspaces9() throws Exception {
        for (int x = 1; x < 10; x++) {
            try (MemoryWorkspace ws =
                            Nd4j.getWorkspaceManager().getAndActivateWorkspace(delayedConfiguration, "WS_1")) {
                INDArray array = Nd4j.create(100 * x);
            }
        }

        Nd4jWorkspace workspace = (Nd4jWorkspace) Nd4j.getWorkspaceManager()
                        .getWorkspaceForCurrentThread(delayedConfiguration, "WS_1");
        workspace.initializeWorkspace();

        assertEquals(300 * Nd4j.sizeOfDataType(), workspace.getCurrentSize());
    }


    @Test
    public void testNestedWorkspaces8() throws Exception {
        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(loopConfiguration, "WS_1")) {
            INDArray array = Nd4j.create(100);
        }



        Nd4jWorkspace workspace = (Nd4jWorkspace) Nd4j.getWorkspaceManager()
                        .getWorkspaceForCurrentThread(loopConfiguration, "WS_1");
        workspace.initializeWorkspace();

        assertEquals(100 * Nd4j.sizeOfDataType(), workspace.getCurrentSize());

        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(loopConfiguration, "WS_1")) {
            INDArray array = Nd4j.create(1000);
        }

        Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(loopConfiguration, "WS_1").initializeWorkspace();

        assertEquals(100 * Nd4j.sizeOfDataType(), workspace.getCurrentSize());
    }

    @Test
    public void testNestedWorkspaces7() throws Exception {
        try (Nd4jWorkspace wsExternal = (Nd4jWorkspace) Nd4j.getWorkspaceManager()
                        .getAndActivateWorkspace(basicConfiguration, "External")) {
            INDArray array1 = Nd4j.create(10);
            INDArray array2 = null;
            INDArray array3 = null;
            INDArray array4 = null;
            INDArray array5 = null;


            try (Nd4jWorkspace wsFeedForward = (Nd4jWorkspace) Nd4j.getWorkspaceManager()
                            .getAndActivateWorkspace(basicConfiguration, "FeedForward")) {
                array2 = Nd4j.create(10);
                assertEquals(true, array2.isAttached());

                try (Nd4jWorkspace borrowed = (Nd4jWorkspace) Nd4j.getWorkspaceManager()
                                .getWorkspaceForCurrentThread("External").notifyScopeBorrowed()) {
                    array3 = Nd4j.create(10);

                    assertTrue(wsExternal == array3.data().getParentWorkspace());

                    try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                        array4 = Nd4j.create(10);
                    }

                    array5 = Nd4j.create(10);
                    log.info("Workspace5: {}", array5.data().getParentWorkspace());
                    assertTrue(null == array4.data().getParentWorkspace());
                    assertFalse(array4.isAttached());
                    assertTrue(wsExternal == array5.data().getParentWorkspace());
                }

                assertEquals(true, array3.isAttached());
                assertEquals(false, array4.isAttached());
                assertEquals(true, array5.isAttached());
            }
        }
    }

    @Test
    public void testNestedWorkspaces6() throws Exception {

        try (Nd4jWorkspace wsExternal = (Nd4jWorkspace) Nd4j.getWorkspaceManager()
                        .getAndActivateWorkspace(firstConfiguration, "External")) {
            INDArray array1 = Nd4j.create(10);
            INDArray array2 = null;
            INDArray array3 = null;
            INDArray array4 = null;


            try (Nd4jWorkspace wsFeedForward = (Nd4jWorkspace) Nd4j.getWorkspaceManager()
                            .getAndActivateWorkspace(firstConfiguration, "FeedForward")) {
                array2 = Nd4j.create(10);
                assertEquals(true, array2.isAttached());

                try (Nd4jWorkspace borrowed = (Nd4jWorkspace) Nd4j.getWorkspaceManager()
                                .getWorkspaceForCurrentThread("External").notifyScopeBorrowed()) {
                    array3 = Nd4j.create(10);

                    assertTrue(wsExternal == array3.data().getParentWorkspace());
                }

                assertEquals(true, array3.isAttached());

                try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    array4 = Nd4j.create(10);
                }

                assertEquals(false, array4.isAttached());
            }


            assertEquals(0, wsExternal.getCurrentSize());
            log.info("------");
        }
    }

    @Test
    public void testNestedWorkspaces5() throws Exception {
        Nd4j.getWorkspaceManager().setDefaultWorkspaceConfiguration(basicConfiguration);
        try (Nd4jWorkspace ws1 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS1")
                        .notifyScopeEntered()) {

            INDArray array1 = Nd4j.create(100);
            try (Nd4jWorkspace ws2 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS1")
                            .notifyScopeEntered()) {

                INDArray array2 = Nd4j.create(100);
            }

            long reqMem = 200 * Nd4j.sizeOfDataType();
            assertEquals(reqMem + reqMem % 8, ws1.getHostOffset());

            INDArray array3 = Nd4j.create(100);

            reqMem = 300 * Nd4j.sizeOfDataType();
            assertEquals(reqMem + reqMem % 8, ws1.getHostOffset());
        }

        assertNull(Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    @Test
    public void testNestedWorkspaces4() throws Exception {
        Nd4j.getWorkspaceManager().setDefaultWorkspaceConfiguration(basicConfiguration);

        try (Nd4jWorkspace ws1 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS1")
                        .notifyScopeEntered()) {

            INDArray array1 = Nd4j.create(100);

            try (Nd4jWorkspace ws2 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS2")
                            .notifyScopeEntered()) {
                INDArray array2 = Nd4j.create(100);

                try (Nd4jWorkspace ws3 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS3")
                                .notifyScopeEntered()) {
                    INDArray array3 = Nd4j.create(100);

                    assertEquals(100 * Nd4j.sizeOfDataType(), ws1.getHostOffset());
                    assertEquals(100 * Nd4j.sizeOfDataType(), ws2.getHostOffset());
                    assertEquals(100 * Nd4j.sizeOfDataType(), ws3.getHostOffset());
                }

                INDArray array2b = Nd4j.create(100);

                assertEquals(200 * Nd4j.sizeOfDataType(), ws2.getHostOffset());
            }

            INDArray array1b = Nd4j.create(100);

            assertEquals(200 * Nd4j.sizeOfDataType(), ws1.getHostOffset());
        }

        Nd4jWorkspace ws1 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS1");
        Nd4jWorkspace ws2 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS2");
        Nd4jWorkspace ws3 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS3");


        assertEquals(0 * Nd4j.sizeOfDataType(), ws1.getHostOffset());
        assertEquals(0 * Nd4j.sizeOfDataType(), ws2.getHostOffset());
        assertEquals(0 * Nd4j.sizeOfDataType(), ws3.getHostOffset());

        assertNull(Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    @Test
    public void testNestedWorkspaces3() throws Exception {
        Nd4j.getWorkspaceManager().setDefaultWorkspaceConfiguration(basicConfiguration);


        // We open top-level workspace
        try (Nd4jWorkspace ws1 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS1")
                        .notifyScopeEntered()) {

            INDArray array1 = Nd4j.create(100);

            assertEquals(100 * Nd4j.sizeOfDataType(), ws1.getHostOffset());

            // we open first nested workspace
            try (Nd4jWorkspace ws2 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS2")
                            .notifyScopeEntered()) {
                assertEquals(0 * Nd4j.sizeOfDataType(), ws2.getHostOffset());

                INDArray array2 = Nd4j.create(100);

                assertEquals(100 * Nd4j.sizeOfDataType(), ws1.getHostOffset());
                assertEquals(100 * Nd4j.sizeOfDataType(), ws2.getHostOffset());
            }

            // and second nexted workspace
            try (Nd4jWorkspace ws3 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS3")
                            .notifyScopeEntered()) {
                assertEquals(0 * Nd4j.sizeOfDataType(), ws3.getHostOffset());

                INDArray array2 = Nd4j.create(100);

                assertEquals(100 * Nd4j.sizeOfDataType(), ws1.getHostOffset());
                assertEquals(100 * Nd4j.sizeOfDataType(), ws3.getHostOffset());
            }

            // this allocation should happen within top-level workspace
            INDArray array1b = Nd4j.create(100);

            assertEquals(200 * Nd4j.sizeOfDataType(), ws1.getHostOffset());
        }

        assertEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        assertNull(Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    @Test
    public void testNestedWorkspaces2() throws Exception {
        Nd4j.getWorkspaceManager().setDefaultWorkspaceConfiguration(basicConfiguration);

        try (Nd4jWorkspace ws1 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS1")
                        .notifyScopeEntered()) {

            INDArray array1 = Nd4j.create(100);

            assertEquals(100 * Nd4j.sizeOfDataType(), ws1.getHostOffset());

            for (int x = 1; x <= 100; x++) {
                try (Nd4jWorkspace ws2 = (Nd4jWorkspace) Nd4j.getWorkspaceManager()
                                .getWorkspaceForCurrentThread(loopConfiguration, "WS2").notifyScopeEntered()) {
                    INDArray array2 = Nd4j.create(x);
                }

                Nd4jWorkspace ws2 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS2");
                long reqMemory = x * Nd4j.sizeOfDataType();
                assertEquals(reqMemory + reqMemory % 8, ws2.getLastCycleAllocations());
            }

            Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS2").initializeWorkspace();

            assertEquals(100 * Nd4j.sizeOfDataType(),
                            ((Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS2"))
                                            .getCurrentSize());
        }

        assertNull(Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    @Test
    public void testNestedWorkspaces1() throws Exception {
        Nd4j.getWorkspaceManager().setDefaultWorkspaceConfiguration(basicConfiguration);


        try (Nd4jWorkspace ws1 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS1")
                        .notifyScopeEntered()) {

            INDArray array1 = Nd4j.create(100);

            assertEquals(100 * Nd4j.sizeOfDataType(), ws1.getHostOffset());

            try (Nd4jWorkspace ws2 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS2")
                            .notifyScopeEntered()) {
                assertEquals(0 * Nd4j.sizeOfDataType(), ws2.getHostOffset());

                INDArray array2 = Nd4j.create(100);

                assertEquals(100 * Nd4j.sizeOfDataType(), ws1.getHostOffset());
                assertEquals(100 * Nd4j.sizeOfDataType(), ws2.getHostOffset());
            }
        }

        assertNull(Nd4j.getMemoryManager().getCurrentWorkspace());
        log.info("---------------");
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
    }

    @Test
    public void testNewWorkspace1() throws Exception {
        MemoryWorkspace workspace1 = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread();

        assertNotEquals(null, workspace1);

        MemoryWorkspace workspace2 = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread();

        assertEquals(workspace1, workspace2);
    }

    @Ignore
    @Test
    public void testMemcpy1() {
        INDArray warmUp = Nd4j.create(100000);
        for (int x = 0; x < 5000; x++) {
            warmUp.addi(0.1);
        }

        WorkspaceConfiguration configuration =
                        WorkspaceConfiguration.builder().policyMirroring(MirroringPolicy.HOST_ONLY)
                                        .initialSize(1024L * 1024L * 1024L).policyLearning(LearningPolicy.NONE).build();

        INDArray array = Nd4j.createUninitialized(150000000);

        MemoryWorkspace workspace =
                        (Nd4jWorkspace) Nd4j.getWorkspaceManager().createNewWorkspace(configuration, "HOST");
        workspace.notifyScopeEntered();


        INDArray memcpy = array.unsafeDuplication(false);


        workspace.notifyScopeLeft();

    }

    @Override
    public char ordering() {
        return 'c';
    }
}
