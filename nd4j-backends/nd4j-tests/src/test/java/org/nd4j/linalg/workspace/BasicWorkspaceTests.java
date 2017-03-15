package org.nd4j.linalg.workspace;

import lombok.extern.slf4j.Slf4j;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

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

    private static final WorkspaceConfiguration loopConfig = WorkspaceConfiguration.builder()
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
    public void testLoops1() throws Exception {
        Nd4jWorkspace workspace = new Nd4jWorkspace(loopConfig);

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertNotEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        assertEquals(0, workspace.getCurrentOffset());

        workspace.notifyScopeEntered();

        INDArray arrayCold = Nd4j.create(10);

        assertEquals(0, workspace.getCurrentOffset());
        assertEquals(0, workspace.getCurrentSize());

        arrayCold.assign(1.0f);

        assertEquals(10f, arrayCold.sumNumber().floatValue(), 0.01f);

        workspace.notifyScopeLeft();


        workspace.initializeWorkspace();
        assertEquals(40, workspace.getCurrentSize());


        log.info("-----------------------");

        for (int x = 0; x < 10; x++) {
            assertEquals(0, workspace.getCurrentOffset());

            workspace.notifyScopeEntered();

            INDArray array = Nd4j.create(10);

            assertEquals(40, workspace.getCurrentOffset());



//            log.info("Array: {}", array);

            array.addi(1.0f);

            //assertEquals(40, workspace.getCurrentOffset());

            assertEquals("Failed on iteration " + x,10, array.sumNumber().doubleValue(), 0.01);

            workspace.notifyScopeLeft();

            assertEquals(0, workspace.getCurrentOffset());
        }
    }

    @Test
    public void testAllocation5() throws Exception {
        Nd4jWorkspace workspace = new Nd4jWorkspace(basicConfig);

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertNotEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        assertEquals(0, workspace.getCurrentOffset());

        INDArray array = Nd4j.create(new int[] {1,5}, 'c');

        // checking if allocation actually happened
        assertEquals(20, workspace.getCurrentOffset());

        array.assign(1.0f);

        INDArray dup = array.dup();

        assertEquals(44, workspace.getCurrentOffset());

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

        assertEquals(0, workspace.getCurrentOffset());

        INDArray array = Nd4j.create(new int[] {1,5}, 'c');

        // checking if allocation actually happened
        assertEquals(20, workspace.getCurrentOffset());

        try {
            INDArray array2 = Nd4j.create(10000000);
            assertTrue(false);
        } catch (ND4JIllegalStateException e) {
            assertTrue(true);
        }

        assertEquals(20, workspace.getCurrentOffset());

        INDArray array2 = Nd4j.create(new int[] {1,5}, 'c');

        assertEquals(40, workspace.getCurrentOffset());
    }

    @Test
    public void testAllocation3() throws Exception {
        Nd4jWorkspace workspace = new Nd4jWorkspace(basicConfig);

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertNotEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        assertEquals(0, workspace.getCurrentOffset());

        INDArray array = Nd4j.create(new int[] {1,5}, 'c');

        // checking if allocation actually happened
        assertEquals(20, workspace.getCurrentOffset());

        array.assign(1.0f);

        assertEquals(5, array.sumNumber().doubleValue(), 0.01);
    }

    @Test
    public void testAllocation2() throws Exception {
        Nd4jWorkspace workspace = new Nd4jWorkspace(basicConfig);

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertNotEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        assertEquals(0, workspace.getCurrentOffset());

        INDArray array = Nd4j.create(5);

        // checking if allocation actually happened
        assertEquals(20, workspace.getCurrentOffset());

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
        assertEquals(20, workspace.getCurrentOffset());

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
        //assertEquals(44, workspace.getCurrentOffset());


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
