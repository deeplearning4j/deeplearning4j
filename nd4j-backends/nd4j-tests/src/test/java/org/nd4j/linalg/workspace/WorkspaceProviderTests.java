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
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
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
public class WorkspaceProviderTests extends BaseNd4jTest {
    private static final WorkspaceConfiguration basicConfiguration = WorkspaceConfiguration.builder()
            .initialSize(81920)
            .overallocationLimit(0.1)
            .policySpill(SpillPolicy.EXTERNAL)
            .policyLearning(LearningPolicy.NONE)
            .policyMirroring(MirroringPolicy.FULL)
            .policyAllocation(AllocationPolicy.OVERALLOCATE)
            .build();

    private static final WorkspaceConfiguration loopConfiguration = WorkspaceConfiguration.builder()
            .initialSize(0)
            .overallocationLimit(0.1)
            .policySpill(SpillPolicy.EXTERNAL)
            .policyLearning(LearningPolicy.OVER_TIME)
            .policyMirroring(MirroringPolicy.FULL)
            .policyAllocation(AllocationPolicy.STRICT)
            .build();

    DataBuffer.Type initialType;

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

    @Test
    public void testNestedWorkspaces2() throws Exception {
        Nd4j.getWorkspaceManager().setDefaultWorkspaceConfiguration(basicConfiguration);

        try(Nd4jWorkspace ws1 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS1").notifyScopeEntered()) {

            INDArray array1 = Nd4j.create(100);

            assertEquals(100 * Nd4j.sizeOfDataType(), ws1.getHostOffset());

            for (int x = 1; x <= 100; x++) {
                try (Nd4jWorkspace ws2 = (Nd4jWorkspace)  Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(loopConfiguration, "WS2").notifyScopeEntered()) {
                    INDArray array2 = Nd4j.create(x);
                }

                Nd4jWorkspace ws2 = (Nd4jWorkspace)  Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS2");
                assertEquals(x * Nd4j.sizeOfDataType(), ws2.getLastCycleAllocations());
            }

            Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS2").initializeWorkspace();

            assertEquals(100 * Nd4j.sizeOfDataType(), ((Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS2")).getCurrentSize());
        }
    }

    @Test
    public void testNestedWorkspaces1() throws Exception {
        Nd4j.getWorkspaceManager().setDefaultWorkspaceConfiguration(basicConfiguration);


        try(Nd4jWorkspace ws1 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS1").notifyScopeEntered()) {

            INDArray array1 = Nd4j.create(100);

            assertEquals(100 * Nd4j.sizeOfDataType(), ws1.getHostOffset());

            try(Nd4jWorkspace ws2 = (Nd4jWorkspace)  Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS2").notifyScopeEntered()) {
                assertEquals(0 * Nd4j.sizeOfDataType(), ws2.getHostOffset());

                INDArray array2 = Nd4j.create(100);

                assertEquals(100 * Nd4j.sizeOfDataType(), ws1.getHostOffset());
                assertEquals(100 * Nd4j.sizeOfDataType(), ws2.getHostOffset());
            }
        }
    }

    @Test
    public void testNewWorkspace1() throws Exception {
        MemoryWorkspace workspace1 = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread();

        assertNotEquals(null, workspace1);

        MemoryWorkspace workspace2 = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread();

        assertEquals(workspace1, workspace2);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
