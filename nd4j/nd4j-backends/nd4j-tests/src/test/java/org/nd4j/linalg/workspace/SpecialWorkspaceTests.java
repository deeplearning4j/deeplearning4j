package org.nd4j.linalg.workspace;

import lombok.extern.slf4j.Slf4j;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.memory.abstracts.Nd4jWorkspace;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@RunWith(Parameterized.class)
public class SpecialWorkspaceTests extends BaseNd4jTest {
    private DataBuffer.Type initialType;

    public SpecialWorkspaceTests(Nd4jBackend backend) {
        super(backend);
        this.initialType = Nd4j.dataType();
    }

    @After
    public void shutUp() throws Exception {
        Nd4j.getMemoryManager().setCurrentWorkspace(null);
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        Nd4j.setDataType(this.initialType);
    }

    @Test
    public void testVariableTimeSeries1() throws Exception {
        WorkspaceConfiguration configuration = WorkspaceConfiguration.builder().initialSize(0).overallocationLimit(3.0)
                        .policyAllocation(AllocationPolicy.OVERALLOCATE).policySpill(SpillPolicy.EXTERNAL)
                        .policyLearning(LearningPolicy.FIRST_LOOP).policyReset(ResetPolicy.ENDOFBUFFER_REACHED).build();

        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(configuration, "WS1")) {
            Nd4j.create(500);
            Nd4j.create(500);
        }

        Nd4jWorkspace workspace = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS1");
        workspace.enableDebug(true);

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
    public void testVariableTimeSeries2() throws Exception {
        WorkspaceConfiguration configuration = WorkspaceConfiguration.builder().initialSize(0).overallocationLimit(3.0)
                        .policyAllocation(AllocationPolicy.OVERALLOCATE).policySpill(SpillPolicy.REALLOCATE)
                        .policyLearning(LearningPolicy.FIRST_LOOP).policyReset(ResetPolicy.ENDOFBUFFER_REACHED).build();

        Nd4jWorkspace workspace =
                        (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(configuration, "WS1");
        workspace.enableDebug(true);

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
    public void testViewDetach_1() throws Exception {
        WorkspaceConfiguration configuration = WorkspaceConfiguration.builder().initialSize(10000000).overallocationLimit(3.0)
                .policyAllocation(AllocationPolicy.OVERALLOCATE).policySpill(SpillPolicy.REALLOCATE)
                .policyLearning(LearningPolicy.FIRST_LOOP).policyReset(ResetPolicy.BLOCK_LEFT).build();

        Nd4jWorkspace workspace =
                (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(configuration, "WS109");

        INDArray row = Nd4j.linspace(1, 10, 10);
        INDArray exp = Nd4j.create(1, 10).assign(2.0);
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

    @Override
    public char ordering() {
        return 'c';
    }
}
