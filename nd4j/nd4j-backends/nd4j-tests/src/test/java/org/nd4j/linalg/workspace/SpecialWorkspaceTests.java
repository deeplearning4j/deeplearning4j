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
import lombok.val;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.api.memory.abstracts.Nd4jWorkspace;

import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@RunWith(Parameterized.class)
public class SpecialWorkspaceTests extends BaseNd4jTest {
    private DataType initialType;

    public SpecialWorkspaceTests(Nd4jBackend backend) {
        super(backend);
        this.initialType = Nd4j.dataType();
    }

    @After
    public void shutUp() {
        Nd4j.getMemoryManager().setCurrentWorkspace(null);
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        Nd4j.setDataType(this.initialType);
    }

    @Test
    public void testVariableTimeSeries1() {
        WorkspaceConfiguration configuration = WorkspaceConfiguration.builder().initialSize(0).overallocationLimit(3.0)
                        .policyAllocation(AllocationPolicy.OVERALLOCATE).policySpill(SpillPolicy.EXTERNAL)
                        .policyLearning(LearningPolicy.FIRST_LOOP).policyReset(ResetPolicy.ENDOFBUFFER_REACHED).build();

        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(configuration, "WS1")) {
            Nd4j.create(500);
            Nd4j.create(500);
        }

        Nd4jWorkspace workspace = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS1");
//        workspace.enableDebug(true);

        assertEquals(0, workspace.getStepNumber());

        long requiredMemory = 1000 * Nd4j.sizeOfDataType();
        long shiftedSize = ((long) (requiredMemory * 1.3)) + (8 - (((long) (requiredMemory * 1.3)) % 8));
        assertEquals(requiredMemory, workspace.getSpilledSize());
        assertEquals(shiftedSize, workspace.getInitialBlockSize());
        assertEquals(workspace.getInitialBlockSize() * 4, workspace.getCurrentSize());

        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace("WS1")) {
            Nd4j.create(2000);
        }

        assertEquals(0, workspace.getStepNumber());

        assertEquals(1000 * Nd4j.sizeOfDataType(), workspace.getSpilledSize());
        assertEquals(2000 * Nd4j.sizeOfDataType(), workspace.getPinnedSize());

        assertEquals(0, workspace.getDeviceOffset());

        // FIXME: fix this!
        //assertEquals(0, workspace.getHostOffset());

        assertEquals(0, workspace.getThisCycleAllocations());
        log.info("------------------");

        assertEquals(1, workspace.getNumberOfPinnedAllocations());

        for (int e = 0; e < 4; e++) {
            for (int i = 0; i < 4; i++) {
                try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(configuration, "WS1")) {
                    Nd4j.create(500);
                    Nd4j.create(500);
                }

                assertEquals("Failed on iteration " + i, (i + 1) * workspace.getInitialBlockSize(),
                                workspace.getDeviceOffset());
            }

            if (e >= 2) {
                assertEquals("Failed on iteration " + e, 0, workspace.getNumberOfPinnedAllocations());
            } else {
                assertEquals("Failed on iteration " + e, 1, workspace.getNumberOfPinnedAllocations());
            }
        }

        assertEquals(0, workspace.getSpilledSize());
        assertEquals(0, workspace.getPinnedSize());
        assertEquals(0, workspace.getNumberOfPinnedAllocations());
        assertEquals(0, workspace.getNumberOfExternalAllocations());

        log.info("Workspace state after first block: ---------------------------------------------------------");
        Nd4j.getWorkspaceManager().printAllocationStatisticsForCurrentThread();


        log.info("--------------------------------------------------------------------------------------------");

        // we just do huge loop now, with pinned stuff in it
        for (int i = 0; i < 100; i++) {
            try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(configuration, "WS1")) {
                Nd4j.create(500);
                Nd4j.create(500);
                Nd4j.create(500);

                assertEquals(1500 * Nd4j.sizeOfDataType(), workspace.getThisCycleAllocations());
            }
        }

        assertEquals(0, workspace.getSpilledSize());
        assertNotEquals(0, workspace.getPinnedSize());
        assertNotEquals(0, workspace.getNumberOfPinnedAllocations());
        assertEquals(0, workspace.getNumberOfExternalAllocations());


        // and we do another clean loo, without pinned stuff in it, to ensure all pinned allocates are gone
        for (int i = 0; i < 100; i++) {
            try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(configuration, "WS1")) {
                Nd4j.create(500);
                Nd4j.create(500);
            }
        }

        assertEquals(0, workspace.getSpilledSize());
        assertEquals(0, workspace.getPinnedSize());
        assertEquals(0, workspace.getNumberOfPinnedAllocations());
        assertEquals(0, workspace.getNumberOfExternalAllocations());


        log.info("Workspace state after second block: ---------------------------------------------------------");
        Nd4j.getWorkspaceManager().printAllocationStatisticsForCurrentThread();
    }


    @Test
    public void testVariableTimeSeries2() {
        WorkspaceConfiguration configuration = WorkspaceConfiguration.builder().initialSize(0).overallocationLimit(3.0)
                        .policyAllocation(AllocationPolicy.OVERALLOCATE).policySpill(SpillPolicy.REALLOCATE)
                        .policyLearning(LearningPolicy.FIRST_LOOP).policyReset(ResetPolicy.ENDOFBUFFER_REACHED).build();

        Nd4jWorkspace workspace =
                        (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(configuration, "WS1");
//        workspace.enableDebug(true);

        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(configuration, "WS1")) {
            Nd4j.create(500);
            Nd4j.create(500);
        }



        assertEquals(0, workspace.getStepNumber());

        long requiredMemory = 1000 * Nd4j.sizeOfDataType();
        long shiftedSize = ((long) (requiredMemory * 1.3)) + (8 - (((long) (requiredMemory * 1.3)) % 8));
        assertEquals(requiredMemory, workspace.getSpilledSize());
        assertEquals(shiftedSize, workspace.getInitialBlockSize());
        assertEquals(workspace.getInitialBlockSize() * 4, workspace.getCurrentSize());


        for (int i = 0; i < 100; i++) {
            try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(configuration, "WS1")) {
                Nd4j.create(500);
                Nd4j.create(500);
                Nd4j.create(500);
            }
        }


        assertEquals(workspace.getInitialBlockSize() * 4, workspace.getCurrentSize());

        assertEquals(0, workspace.getNumberOfPinnedAllocations());
        assertEquals(0, workspace.getNumberOfExternalAllocations());

        assertEquals(0, workspace.getSpilledSize());
        assertEquals(0, workspace.getPinnedSize());

    }

    @Test
    public void testViewDetach_1() {
        WorkspaceConfiguration configuration = WorkspaceConfiguration.builder().initialSize(10000000).overallocationLimit(3.0)
                .policyAllocation(AllocationPolicy.OVERALLOCATE).policySpill(SpillPolicy.REALLOCATE)
                .policyLearning(LearningPolicy.FIRST_LOOP).policyReset(ResetPolicy.BLOCK_LEFT).build();

        Nd4jWorkspace workspace =
                (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(configuration, "WS109");

        INDArray row = Nd4j.linspace(1, 10, 10);
        INDArray exp = Nd4j.create(10).assign(2.0);
        INDArray result = null;
        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(configuration, "WS109")) {
            INDArray matrix = Nd4j.create(10, 10);
            for (int e = 0; e < matrix.rows(); e++)
                matrix.getRow(e).assign(row);


            INDArray column = matrix.getColumn(1);
            assertTrue(column.isView());
            assertTrue(column.isAttached());
            result = column.detach();
        }

        assertFalse(result.isView());
        assertFalse(result.isAttached());
        assertEquals(exp, result);
    }


    @Test
    public void testAlignment_1() {
        WorkspaceConfiguration initialConfig = WorkspaceConfiguration.builder().initialSize(10 * 1024L * 1024L)
                .policyAllocation(AllocationPolicy.STRICT).policyLearning(LearningPolicy.NONE).build();
        MemoryWorkspace workspace = Nd4j.getWorkspaceManager().getAndActivateWorkspace(initialConfig, "WS132143452343");

        for( int j=0; j<100; j++ ){

            try(MemoryWorkspace ws = workspace.notifyScopeEntered()) {

                for (int x = 0; x < 10; x++) {
                    //System.out.println("Start iteration (" + j + "," + x + ")");
                    INDArray arr = Nd4j.linspace(1,10,10, DataType.DOUBLE).reshape(1,10);
                    INDArray sum = arr.sum(true, 1);
                    Nd4j.create(DataType.BOOL, x+1);        //NOTE: no crash if set to FLOAT/HALF, No crash if removed entirely; same crash for BOOL/UBYTE
                    //System.out.println("End iteration (" + j + "," + x + ")");
                }
            }
        }
    }


    @Test
    public void testNoOpExecution_1() {
        val configuration = WorkspaceConfiguration.builder().initialSize(10000000).overallocationLimit(3.0)
                .policyAllocation(AllocationPolicy.OVERALLOCATE).policySpill(SpillPolicy.REALLOCATE)
                .policyLearning(LearningPolicy.FIRST_LOOP).policyReset(ResetPolicy.BLOCK_LEFT).build();

        int iterations = 10000;

        val array0 = Nd4j.create(new long[]{ 100, 100});
        val array1 = Nd4j.create(new long[]{ 100, 100});
        val array2 = Nd4j.create(new long[]{ 100, 100});
        val array3 = Nd4j.create(new long[]{ 100, 100});
        val array4 = Nd4j.create(new long[]{ 100, 100});
        val array5 = Nd4j.create(new long[]{ 100, 100});
        val array6 = Nd4j.create(new long[]{ 100, 100});
        val array7 = Nd4j.create(new long[]{ 100, 100});
        val array8 = Nd4j.create(new long[]{ 100, 100});
        val array9 = Nd4j.create(new long[]{ 100, 100});

        val timeStart = System.nanoTime();
        for (int e = 0; e < iterations; e++) {

            val op = DynamicCustomOp.builder("noop")
                    .addInputs(array0, array1, array2, array3, array4, array5, array6, array7, array8, array9)
                    .addOutputs(array0, array1, array2, array3, array4, array5, array6, array7, array8, array9)
                    .addIntegerArguments(5, 10)
                    .addFloatingPointArguments(3.0, 10.0)
                    .addBooleanArguments(true, false)
                    .callInplace(true)
                    .build();

            Nd4j.getExecutioner().exec(op);
        }
        val timeEnd = System.nanoTime();
        log.info("{} ns", ((timeEnd - timeStart) / (double) iterations));
    }

    @Test
    public void testWorkspaceOrder_1(){
        WorkspaceConfiguration conf = WorkspaceConfiguration.builder()
                .initialSize(1_000_000)
                .overallocationLimit(0.05)
                .policyLearning(LearningPolicy.NONE)
                .build();

        val exp = Arrays.asList("outer", null, "outer", "inner", "outer", null);
        val res = new ArrayList<String>();

        try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(conf, "outer")){
            try(MemoryWorkspace ws2 = Nd4j.getWorkspaceManager().getAndActivateWorkspace(conf, "inner")){
                try(MemoryWorkspace ws3 = ws.notifyScopeBorrowed()){
                    System.out.println("X: " + Nd4j.getMemoryManager().getCurrentWorkspace());                  //outer
                    res.add(Nd4j.getMemoryManager().getCurrentWorkspace() == null ? null : Nd4j.getMemoryManager().getCurrentWorkspace().getId());
                    try(MemoryWorkspace ws4 = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()){
                        System.out.println("A: " + Nd4j.getMemoryManager().getCurrentWorkspace());              //None (null)
                        res.add(Nd4j.getMemoryManager().getCurrentWorkspace() == null ? null : Nd4j.getMemoryManager().getCurrentWorkspace().getId());
                    }
                    System.out.println("B: " + Nd4j.getMemoryManager().getCurrentWorkspace());                  //outer
                    res.add(Nd4j.getMemoryManager().getCurrentWorkspace() == null ? null : Nd4j.getMemoryManager().getCurrentWorkspace().getId());
                }
                System.out.println("C: " + Nd4j.getMemoryManager().getCurrentWorkspace());                      //inner
                res.add(Nd4j.getMemoryManager().getCurrentWorkspace() == null ? null : Nd4j.getMemoryManager().getCurrentWorkspace().getId());
            }
            System.out.println("D: " + Nd4j.getMemoryManager().getCurrentWorkspace());                          //outer
            res.add(Nd4j.getMemoryManager().getCurrentWorkspace() == null ? null : Nd4j.getMemoryManager().getCurrentWorkspace().getId());
        }
        System.out.println("E: " + Nd4j.getMemoryManager().getCurrentWorkspace());                              //None (null)
        res.add(Nd4j.getMemoryManager().getCurrentWorkspace() == null ? null : Nd4j.getMemoryManager().getCurrentWorkspace().getId());

        assertEquals(exp, res);
    }

    @Test
    public void testMmapedWorkspaceLimits_1() throws Exception {
        if (!Nd4j.getEnvironment().isCPU())
            return;

        val tmpFile = Files.createTempFile("some", "file");
        val mmap = WorkspaceConfiguration.builder()
                .initialSize(200 * 1024L * 1024L) // 200mbs
                .tempFilePath(tmpFile.toAbsolutePath().toString())
                .policyLocation(LocationPolicy.MMAP)
                .build();

        try (val ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(mmap, "M2")) {
            int twoHundredMbsOfFloats = 52_428_800; // 200mbs % 4
            val addMoreFloats = true;
            if (addMoreFloats) {
                twoHundredMbsOfFloats += 1_000;
            }

            val x = Nd4j.rand(DataType.FLOAT, twoHundredMbsOfFloats);
        }
    }

    @Test
    public void testMmapedWorkspace_Path_Limits_1() throws Exception {
        if (!Nd4j.getEnvironment().isCPU())
            return;

        // getting very long file name
        val builder = new StringBuilder("long_file_name_");
        for (int e = 0; e < 100; e++)
            builder.append("9");


        val tmpFile = Files.createTempFile("some", builder.toString());
        val mmap = WorkspaceConfiguration.builder()
                .initialSize(200 * 1024L * 1024L) // 200mbs
                .tempFilePath(tmpFile.toAbsolutePath().toString())
                .policyLocation(LocationPolicy.MMAP)
                .build();

        try (val ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(mmap, "M2")) {
            val x = Nd4j.rand(DataType.FLOAT, 1024);
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
