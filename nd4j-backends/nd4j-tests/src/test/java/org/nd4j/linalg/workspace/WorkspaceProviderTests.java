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
import org.nd4j.linalg.api.memory.enums.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.memory.abstracts.Nd4jWorkspace;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import static org.junit.Assert.*;

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

    /**
     * This simple test checks for over-time learning with coefficient applied
     *
     * @throws Exception
     */
    @Test
    public void testUnboundedLoop2() throws Exception {
        WorkspaceConfiguration configuration = WorkspaceConfiguration.builder()
                .initialSize(0)
                .policyReset(ResetPolicy.ENDOFBUFFER_REACHED)
                .policyAllocation(AllocationPolicy.OVERALLOCATE)
                .overallocationLimit(4.0)
                .policyLearning(LearningPolicy.OVER_TIME)
                .cyclesBeforeInitialization(5)
                .build();

        Nd4jWorkspace ws1 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(configuration, "ITER");

        for (int x = 0; x < 100; x++) {
            try (Nd4jWorkspace wsI = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(configuration, "ITER").notifyScopeEntered()) {
                INDArray array = Nd4j.create(100);
            }

            // only checking after workspace is initialized
            if (x > 5) {
                assertEquals(5 * 100 * Nd4j.sizeOfDataType(), ws1.getCurrentSize());

                // if we've passed 5 iterations - workspace is initialized, and now offset mechanics works
  //              if (x % 5 == 0)
//                    assertEquals(2000, ws1.getHostOffset());
  //              else
//                    assertEquals((x % 5) * 100 * Nd4j.sizeOfDataType(), ws1.getHostOffset());
            } else if (x < 5) {
                // we're making sure we're not initialize early
                assertEquals(0, ws1.getCurrentSize());
            }
        }

        // maximum allocation amount is 100 elements during learning, and additional coefficient is 4.0. result is workspace of 500 elements
        assertEquals(5 * 100 * Nd4j.sizeOfDataType(), ws1.getCurrentSize());

        assertNull(Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    @Test
    public void testUnboundedLoop1() throws Exception {
        WorkspaceConfiguration configuration = WorkspaceConfiguration.builder()
                .initialSize(100 * 100 * Nd4j.sizeOfDataType())
                .policyReset(ResetPolicy.ENDOFBUFFER_REACHED)
                .policyAllocation(AllocationPolicy.STRICT)
                .build();

        for (int x = 0; x < 100; x++) {
            try (Nd4jWorkspace ws1 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(configuration, "ITER").notifyScopeEntered()) {

                INDArray array = Nd4j.create(100);
            }

            Nd4jWorkspace ws1 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(configuration, "ITER");

            assertEquals((x + 1 ) * 100 * Nd4j.sizeOfDataType(), ws1.getHostOffset());
        }

        Nd4jWorkspace ws1 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(configuration, "ITER");
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
        try(Nd4jWorkspace ws1 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS1").notifyScopeEntered()) {
            INDArray array = Nd4j.create(new float[]{6f, 3f, 1f, 9f, 21f});
            INDArray array3 = null;

            long reqMem = 5 * Nd4j.sizeOfDataType();
            assertEquals(reqMem + reqMem % 8, ws1.getHostOffset());
            try(Nd4jWorkspace ws2 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS2").notifyScopeEntered()) {

                INDArray array2 = Nd4j.create(new float[]{1f, 2f, 3f, 4f, 5f});

                reqMem = 5 * Nd4j.sizeOfDataType();
                assertEquals(reqMem + reqMem % 8, ws1.getHostOffset());
                assertEquals(reqMem + reqMem % 8, ws2.getHostOffset());

                try(Nd4jWorkspace ws3 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS1").notifyScopeBorrowed()) {
                    assertTrue(ws1 == ws3);
                    assertTrue(ws1 == Nd4j.getMemoryManager().getCurrentWorkspace());

                    array3 = array2.unsafeDuplication();

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

        assertNull(Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    @Test
    public void testNestedWorkspacesOverlap1() throws Exception {
        Nd4j.getWorkspaceManager().setDefaultWorkspaceConfiguration(basicConfiguration);
        try(Nd4jWorkspace ws1 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS1").notifyScopeEntered()) {
            INDArray array = Nd4j.create(new float[]{1f, 2f, 3f, 4f, 5f});

            long reqMem = 5 * Nd4j.sizeOfDataType();
            assertEquals(reqMem + reqMem % 8, ws1.getHostOffset());
            try(Nd4jWorkspace ws2 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS2").notifyScopeEntered()) {

                INDArray array2 = Nd4j.create(new float[]{1f, 2f, 3f, 4f, 5f});

                reqMem = 5 * Nd4j.sizeOfDataType();
                assertEquals(reqMem + reqMem % 8, ws1.getHostOffset());
                assertEquals(reqMem + reqMem % 8, ws2.getHostOffset());

                try(Nd4jWorkspace ws3 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS1").notifyScopeBorrowed()) {
                    assertTrue(ws1 == ws3);

                    INDArray array3 = Nd4j.create(new float[]{1f, 2f, 3f, 4f, 5f});

                    assertEquals(reqMem + reqMem % 8, ws2.getHostOffset());
                    assertEquals((reqMem + reqMem % 8) * 2, ws1.getHostOffset());
                }
            }
        }

        assertNull(Nd4j.getMemoryManager().getCurrentWorkspace());
    }

    @Test
    public void testNestedWorkspaces5() throws Exception {
        Nd4j.getWorkspaceManager().setDefaultWorkspaceConfiguration(basicConfiguration);
        try(Nd4jWorkspace ws1 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS1").notifyScopeEntered()) {

            INDArray array1 = Nd4j.create(100);
            try(Nd4jWorkspace ws2 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS1").notifyScopeEntered()) {

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

        try(Nd4jWorkspace ws1 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS1").notifyScopeEntered()) {

            INDArray array1 = Nd4j.create(100);

            try(Nd4jWorkspace ws2 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS2").notifyScopeEntered()) {
                INDArray array2 = Nd4j.create(100);

                try(Nd4jWorkspace ws3 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS3").notifyScopeEntered()) {
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
        try(Nd4jWorkspace ws1 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS1").notifyScopeEntered()) {

            INDArray array1 = Nd4j.create(100);

            assertEquals(100 * Nd4j.sizeOfDataType(), ws1.getHostOffset());

            // we open first nested workspace
            try(Nd4jWorkspace ws2 = (Nd4jWorkspace)  Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS2").notifyScopeEntered()) {
                assertEquals(0 * Nd4j.sizeOfDataType(), ws2.getHostOffset());

                INDArray array2 = Nd4j.create(100);

                assertEquals(100 * Nd4j.sizeOfDataType(), ws1.getHostOffset());
                assertEquals(100 * Nd4j.sizeOfDataType(), ws2.getHostOffset());
            }

            // and second nexted workspace
            try(Nd4jWorkspace ws3 = (Nd4jWorkspace)  Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS3").notifyScopeEntered()) {
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

        try(Nd4jWorkspace ws1 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS1").notifyScopeEntered()) {

            INDArray array1 = Nd4j.create(100);

            assertEquals(100 * Nd4j.sizeOfDataType(), ws1.getHostOffset());

            for (int x = 1; x <= 100; x++) {
                try (Nd4jWorkspace ws2 = (Nd4jWorkspace)  Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(loopConfiguration, "WS2").notifyScopeEntered()) {
                    INDArray array2 = Nd4j.create(x);
                }

                Nd4jWorkspace ws2 = (Nd4jWorkspace)  Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS2");
                long reqMemory = x * Nd4j.sizeOfDataType();
                assertEquals(reqMemory + reqMemory % 8, ws2.getLastCycleAllocations());
            }

            Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS2").initializeWorkspace();

            assertEquals(100 * Nd4j.sizeOfDataType(), ((Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS2")).getCurrentSize());
        }

        assertNull(Nd4j.getMemoryManager().getCurrentWorkspace());
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

        assertNull(Nd4j.getMemoryManager().getCurrentWorkspace());
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
