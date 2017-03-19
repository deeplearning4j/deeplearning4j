package org.nd4j.linalg.workspace;

import lombok.extern.slf4j.Slf4j;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.memory.abstracts.Nd4jWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@RunWith(Parameterized.class)
public class BasicWorkspaceTests extends BaseNd4jTest {
    DataBuffer.Type initialType;

    private static final WorkspaceConfiguration basicConfig = WorkspaceConfiguration.builder()
            .initialSize(10 * 1024 * 1024)
            .maxSize(10 * 1024 * 1024)
            .overallocationLimit(0.1)
            .policyAllocation(AllocationPolicy.STRICT)
            .policyLearning(LearningPolicy.FIRST_LOOP)
            .policyMirroring(MirroringPolicy.FULL)
            .policySpill(SpillPolicy.EXTERNAL)
            .build();

    private static final WorkspaceConfiguration loopOverTimeConfig = WorkspaceConfiguration.builder()
            .initialSize(0)
            .maxSize(10 * 1024 * 1024)
            .overallocationLimit(0.1)
            .policyAllocation(AllocationPolicy.STRICT)
            .policyLearning(LearningPolicy.OVER_TIME)
            .policyMirroring(MirroringPolicy.FULL)
            .policySpill(SpillPolicy.EXTERNAL)
            .build();


    private static final WorkspaceConfiguration loopFirstConfig = WorkspaceConfiguration.builder()
            .initialSize(0)
            .maxSize(10 * 1024 * 1024)
            .overallocationLimit(0.1)
            .policyAllocation(AllocationPolicy.STRICT)
            .policyLearning(LearningPolicy.FIRST_LOOP)
            .policyMirroring(MirroringPolicy.FULL)
            .policySpill(SpillPolicy.EXTERNAL)
            .build();

    public BasicWorkspaceTests(Nd4jBackend backend) {
        super(backend);
        this.initialType = Nd4j.dataType();
    }

    @Before
    public void setUp() {
        Nd4j.setDataType(DataBuffer.Type.FLOAT);
    }

    @After
    public void shutdown() {
        Nd4j.getMemoryManager().setCurrentWorkspace(null);

        Nd4j.setDataType(initialType);
    }

    @Test
    public void testCold() throws Exception {
        INDArray array = Nd4j.create(10);

        array.addi(1.0);

        assertEquals(10f, array.sumNumber().floatValue(), 0.01f);
    }



    @Test
    public void testDetach1() throws Exception {
        INDArray array = null;
        INDArray copy = null;
        try (Nd4jWorkspace wsI = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfig, "ITER")) {
            array = Nd4j.create(new float[]{1f, 2f, 3f, 4f, 5f});

            // despite we're allocating this array in workspace, it's empty yet, so it's external allocation
            assertTrue(array.isInScope());
            assertTrue(array.isAttached());
            assertEquals(5 * Nd4j.sizeOfDataType(), wsI.getHostOffset());

            copy = array.detach();

            assertTrue(array.isInScope());
            assertTrue(array.isAttached());
            assertEquals(5 * Nd4j.sizeOfDataType(), wsI.getHostOffset());

            assertFalse(copy.isAttached());
            assertTrue(copy.isInScope());
            assertEquals(5 * Nd4j.sizeOfDataType(), wsI.getHostOffset());
        }

        assertEquals(15.0f, copy.sumNumber().floatValue(), 0.01f);
        assertFalse(array == copy);
    }

    @Test
    public void testScope2() throws Exception {
        INDArray array = null;
        try (Nd4jWorkspace wsI = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(loopFirstConfig, "ITER")) {
            array = Nd4j.create(100);

            // despite we're allocating this array in workspace, it's empty yet, so it's external allocation
            assertTrue(array.isInScope());
            assertEquals(0, wsI.getCurrentSize());
        }


        try (Nd4jWorkspace wsI = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(loopFirstConfig, "ITER")) {
            array = Nd4j.create(100);

            assertTrue(array.isInScope());
            assertEquals(100 * Nd4j.sizeOfDataType(), wsI.getHostOffset());
        }

        assertFalse(array.isInScope());
    }

    @Test
    public void testScope1() throws Exception {
        INDArray array = null;
        try (Nd4jWorkspace wsI = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(basicConfig, "ITER")) {
            array = Nd4j.create(100);

            assertTrue(array.isInScope());
        }

        assertFalse(array.isInScope());
    }

    @Test
    public void testIsAttached1() {

        try (Nd4jWorkspace wsI = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getAndActivateWorkspace(loopFirstConfig, "ITER")) {
            INDArray array = Nd4j.create(100);

            assertTrue(array.isAttached());
        }

        INDArray array = Nd4j.create(100);

        assertFalse(array.isAttached());
    }

    @Test
    public void testOverallocation3() throws Exception {
        WorkspaceConfiguration overallocationConfig = WorkspaceConfiguration.builder()
                .initialSize(0)
                .maxSize(10 * 1024 * 1024)
                .overallocationLimit(1.0)
                .policyAllocation(AllocationPolicy.OVERALLOCATE)
                .policyLearning(LearningPolicy.OVER_TIME)
                .policyMirroring(MirroringPolicy.FULL)
                .policySpill(SpillPolicy.EXTERNAL)
                .build();

        Nd4jWorkspace workspace = new Nd4jWorkspace(overallocationConfig);

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertEquals(0, workspace.getCurrentSize());

        for (int x = 10; x <= 100; x+=10) {
            try (MemoryWorkspace cW = workspace.notifyScopeEntered()) {
                INDArray array = Nd4j.create(x);
            }
        }

        assertEquals(0, workspace.getCurrentSize());

        workspace.initializeWorkspace();


        // should be 800 = 100 elements * 4 bytes per element * 2 as overallocation coefficient
        assertEquals(200 * Nd4j.sizeOfDataType(), workspace.getCurrentSize());
    }

    @Test
    public void testOverallocation2() throws Exception {
        WorkspaceConfiguration overallocationConfig = WorkspaceConfiguration.builder()
                .initialSize(0)
                .maxSize(10 * 1024 * 1024)
                .overallocationLimit(1.0)
                .policyAllocation(AllocationPolicy.OVERALLOCATE)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .policyMirroring(MirroringPolicy.FULL)
                .policySpill(SpillPolicy.EXTERNAL)
                .build();

        Nd4jWorkspace workspace = new Nd4jWorkspace(overallocationConfig);

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertEquals(0, workspace.getCurrentSize());

        try(MemoryWorkspace cW = workspace.notifyScopeEntered()) {
            INDArray array = Nd4j.create(100);
        }

        // should be 800 = 100 elements * 4 bytes per element * 2 as overallocation coefficient
        assertEquals(200 * Nd4j.sizeOfDataType(), workspace.getCurrentSize());
    }

    @Test
    public void testOverallocation1() throws Exception {
        WorkspaceConfiguration overallocationConfig = WorkspaceConfiguration.builder()
                .initialSize(1024)
                .maxSize(10 * 1024 * 1024)
                .overallocationLimit(1.0)
                .policyAllocation(AllocationPolicy.OVERALLOCATE)
                .policyLearning(LearningPolicy.NONE)
                .policyMirroring(MirroringPolicy.FULL)
                .policySpill(SpillPolicy.EXTERNAL)
                .build();

        Nd4jWorkspace workspace = new Nd4jWorkspace(overallocationConfig);

        assertEquals(2048, workspace.getCurrentSize());
    }

    @Test
    public void testToggle1() throws Exception {
        Nd4jWorkspace workspace = new Nd4jWorkspace(loopFirstConfig);

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertNotEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        assertEquals(0, workspace.getHostOffset());

        try(MemoryWorkspace cW = workspace.notifyScopeEntered()) {
            INDArray array1 = Nd4j.create(100);

            cW.toggleWorkspaceUse(false);

            INDArray arrayDetached = Nd4j.create(100);

            arrayDetached.assign(1.0f);

            double sum = arrayDetached.sumNumber().doubleValue();
            assertEquals(100f, sum, 0.01);

            cW.toggleWorkspaceUse(true);

            INDArray array2 = Nd4j.create(100);
        }

        assertEquals(0, workspace.getHostOffset());
        assertEquals(200 * Nd4j.sizeOfDataType(), workspace.getCurrentSize());


        try(MemoryWorkspace cW = workspace.notifyScopeEntered()) {
            INDArray array1 = Nd4j.create(100);

            cW.toggleWorkspaceUse(false);

            INDArray arrayDetached = Nd4j.create(100);

            arrayDetached.assign(1.0f);

            double sum = arrayDetached.sumNumber().doubleValue();
            assertEquals(100f, sum, 0.01);

            cW.toggleWorkspaceUse(true);

            assertEquals(100 * Nd4j.sizeOfDataType(), workspace.getHostOffset());

            INDArray array2 = Nd4j.create(100);

            assertEquals(200 * Nd4j.sizeOfDataType(), workspace.getHostOffset());
        }
    }

    @Test
    public void testLoop4() throws Exception {
        Nd4jWorkspace workspace = new Nd4jWorkspace(loopFirstConfig);

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertNotEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        assertEquals(0, workspace.getHostOffset());

        try(MemoryWorkspace cW = workspace.notifyScopeEntered()) {
            INDArray array1 = Nd4j.create(100);
            INDArray array2 = Nd4j.create(100);
        }

        assertEquals(0, workspace.getHostOffset());
        assertEquals(200 * Nd4j.sizeOfDataType(), workspace.getCurrentSize());

        try(MemoryWorkspace cW = workspace.notifyScopeEntered()) {
            INDArray array1 = Nd4j.create(100);

            assertEquals(100 * Nd4j.sizeOfDataType(), workspace.getHostOffset());
        }

        assertEquals(0, workspace.getHostOffset());
    }

    @Test
    public void testLoops3() throws Exception {
        Nd4jWorkspace workspace = new Nd4jWorkspace(loopFirstConfig);

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertNotEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        assertEquals(0, workspace.getHostOffset());

        workspace.notifyScopeEntered();

        INDArray arrayCold1 = Nd4j.create(100);
        INDArray arrayCold2 = Nd4j.create(10);

        assertEquals(0, workspace.getHostOffset());
        assertEquals(0, workspace.getCurrentSize());

        workspace.notifyScopeLeft();

        assertEquals(0, workspace.getHostOffset());
        assertEquals(110 * Nd4j.sizeOfDataType(), workspace.getCurrentSize());
    }

    @Test
    public void testLoops2() throws Exception {
        Nd4jWorkspace workspace = new Nd4jWorkspace(loopOverTimeConfig);

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertNotEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        assertEquals(0, workspace.getHostOffset());

        for (int x = 1; x <= 100; x++) {
            workspace.notifyScopeEntered();

            INDArray arrayCold = Nd4j.create(x);

            assertEquals(0, workspace.getHostOffset());
            assertEquals(0, workspace.getCurrentSize());

            workspace.notifyScopeLeft();
        }

        workspace.initializeWorkspace();
        assertEquals(100 * Nd4j.sizeOfDataType(), workspace.getCurrentSize());
        assertEquals(0, workspace.getHostOffset());

        workspace.notifyScopeEntered();

        INDArray arrayHot = Nd4j.create(10);
        assertEquals(10 * Nd4j.sizeOfDataType(), workspace.getHostOffset());

        workspace.notifyScopeLeft();
    }

    @Test
    public void testLoops1() throws Exception {
        Nd4jWorkspace workspace = new Nd4jWorkspace(loopOverTimeConfig);

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertNotEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        assertEquals(0, workspace.getHostOffset());

        workspace.notifyScopeEntered();

        INDArray arrayCold = Nd4j.create(10);

        assertEquals(0, workspace.getHostOffset());
        assertEquals(0, workspace.getCurrentSize());

        arrayCold.assign(1.0f);

        assertEquals(10f, arrayCold.sumNumber().floatValue(), 0.01f);

        workspace.notifyScopeLeft();


        workspace.initializeWorkspace();
        assertEquals(11 * Nd4j.sizeOfDataType(), workspace.getCurrentSize());


        log.info("-----------------------");

        for (int x = 0; x < 10; x++) {
            assertEquals(0, workspace.getHostOffset());

            workspace.notifyScopeEntered();

            INDArray array = Nd4j.create(10);

            assertEquals(10 * Nd4j.sizeOfDataType(), workspace.getHostOffset());

            array.addi(1.0f);

            assertEquals(10 * Nd4j.sizeOfDataType(), workspace.getHostOffset());

            assertEquals("Failed on iteration " + x,10, array.sumNumber().doubleValue(), 0.01);

            workspace.notifyScopeLeft();

            assertEquals(0, workspace.getHostOffset());
        }
    }

    @Test
    public void testAllocation5() throws Exception {
        Nd4jWorkspace workspace = new Nd4jWorkspace(basicConfig);

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertNotEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        assertEquals(0, workspace.getHostOffset());

        INDArray array = Nd4j.create(new int[] {1,5}, 'c');

        // checking if allocation actually happened
        assertEquals(5 * Nd4j.sizeOfDataType(), workspace.getHostOffset());

        array.assign(1.0f);

        INDArray dup = array.dup();

        assertEquals(10 * Nd4j.sizeOfDataType(), workspace.getHostOffset());

        assertEquals(5, dup.sumNumber().doubleValue(), 0.01);
    }


    @Test
    public void testAllocation4() throws Exception {
        WorkspaceConfiguration failConfig = WorkspaceConfiguration.builder()
                .initialSize(1024 * 1024)
                .maxSize(1024 * 1024)
                .overallocationLimit(0.1)
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .policyMirroring(MirroringPolicy.FULL)
                .policySpill(SpillPolicy.FAIL)
                .build();


        Nd4jWorkspace workspace = new Nd4jWorkspace(failConfig);

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertNotEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        assertEquals(0, workspace.getHostOffset());

        INDArray array = Nd4j.create(new int[] {1,5}, 'c');

        // checking if allocation actually happened
        assertEquals(5 * Nd4j.sizeOfDataType(), workspace.getHostOffset());

        try {
            INDArray array2 = Nd4j.create(10000000);
            assertTrue(false);
        } catch (ND4JIllegalStateException e) {
            assertTrue(true);
        }

        assertEquals(5 * Nd4j.sizeOfDataType(), workspace.getHostOffset());

        INDArray array2 = Nd4j.create(new int[] {1,5}, 'c');

        assertEquals(10 * Nd4j.sizeOfDataType(), workspace.getHostOffset());
    }

    @Test
    public void testAllocation3() throws Exception {
        Nd4jWorkspace workspace = new Nd4jWorkspace(basicConfig);

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertNotEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        assertEquals(0, workspace.getHostOffset());

        INDArray array = Nd4j.create(new int[] {1,5}, 'c');

        // checking if allocation actually happened
        assertEquals(5 * Nd4j.sizeOfDataType(), workspace.getHostOffset());

        array.assign(1.0f);

        assertEquals(5, array.sumNumber().doubleValue(), 0.01);
    }

    @Test
    public void testAllocation2() throws Exception {
        Nd4jWorkspace workspace = new Nd4jWorkspace(basicConfig);

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertNotEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        assertEquals(0, workspace.getHostOffset());

        INDArray array = Nd4j.create(5);

        // checking if allocation actually happened
        assertEquals(5 * Nd4j.sizeOfDataType(), workspace.getHostOffset());

        array.assign(1.0f);

        assertEquals(5, array.sumNumber().doubleValue(), 0.01);
    }

    @Test
    public void testAllocation1() throws Exception {
        Nd4jWorkspace workspace = new Nd4jWorkspace(basicConfig);

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertNotEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        INDArray array = Nd4j.create(new float[]{1f, 2f, 3f, 4f, 5f});

        // checking if allocation actually happened
        assertEquals(20, workspace.getHostOffset());

        // checking stuff at native side
        double sum = array.sumNumber().doubleValue();
        assertEquals(15.0, sum, 0.01);

        array.getFloat(0);

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

        INDArray array2 = Nd4j.create(new float[]{5f, 4f, 3f, 2f, 1f});

        sum = array2.sumNumber().doubleValue();
        assertEquals(15.0, sum, 0.01);

        // 44 = 20 + 4 + 20, 4 was allocated as Op.extraArgs for sum
        //assertEquals(44, workspace.getHostOffset());


        array.addi(array2);

        sum = array.sumNumber().doubleValue();
        assertEquals(30.0, sum, 0.01);


        // checking INDArray validity
        assertEquals(6.0, array.getFloat(0), 0.01);
        assertEquals(6.0, array.getFloat(1), 0.01);
        assertEquals(6.0, array.getFloat(2), 0.01);
        assertEquals(6.0, array.getFloat(3), 0.01);
        assertEquals(6.0, array.getFloat(4), 0.01);
    }



    @Override
    public char ordering() {
        return 'c';
    }
}
