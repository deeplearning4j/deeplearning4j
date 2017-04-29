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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.memory.abstracts.Nd4jWorkspace;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

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
        WorkspaceConfiguration configuration = WorkspaceConfiguration.builder()
                .initialSize(0)
                .overallocationLimit(3.0)
                .policyAllocation(AllocationPolicy.OVERALLOCATE)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .policyReset(ResetPolicy.ENDOFBUFFER_REACHED)
                .build();

        try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(configuration, "WS1")) {
            Nd4j.create(500);
            Nd4j.create(500);
        }

        Nd4jWorkspace workspace = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS1");
        workspace.enableDebug(true);

        assertEquals(0, workspace.getStepNumber());

        assertEquals(1000 * Nd4j.sizeOfDataType(), workspace.getSpilledSize());
        assertEquals(1000 * Nd4j.sizeOfDataType(), workspace.getInitialBlockSize());
        assertEquals((1000 + (1000*3)) * Nd4j.sizeOfDataType(), workspace.getCurrentSize());

        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace("WS1")) {
            Nd4j.create(1100);
        }

        assertEquals(0, workspace.getStepNumber());

        assertEquals(1000 * Nd4j.sizeOfDataType(), workspace.getSpilledSize());
        assertEquals(1100 * Nd4j.sizeOfDataType(), workspace.getPinnedSize());

        assertEquals(0, workspace.getHostOffset());
        assertEquals(0, workspace.getThisCycleAllocations());
        log.info("------------------");

        assertEquals(1, workspace.getNumberOfPinnedAllocations());

        for (int e = 0; e < 4; e++) {
            for (int i = 0; i < 4; i++) {
                try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(configuration, "WS1")) {
                    Nd4j.create(500);
                    Nd4j.create(500);
                }

                assertEquals("Failed on iteration " + i, (i + 1) * 1000 * Nd4j.sizeOfDataType(), workspace.getHostOffset());
            }

            if (e >= 1) {
                assertEquals("Failed on iteration " + e,0, workspace.getNumberOfPinnedAllocations());
            } else {
                assertEquals(1, workspace.getNumberOfPinnedAllocations());
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

    @Override
    public char ordering() {
        return 'c';
    }
}
