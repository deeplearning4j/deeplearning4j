package org.nd4j.linalg.workspace;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
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

/**
 * @author raver119@gmail.com
 */
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
    public void testAllocation1() throws Exception {
        Nd4jWorkspace workspace = new Nd4jWorkspace(basicConfig);

        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);

        assertNotEquals(null, Nd4j.getMemoryManager().getCurrentWorkspace());

        INDArray array = Nd4j.create(new float[]{1f, 2f, 3f, 4f, 5f});

        // checking if allocation actually happened
        assertEquals(20, workspace.getCurrentOffset());

        // checking INDArray validity
        assertEquals(1.0, array.getFloat(0), 0.01);
        assertEquals(2.0, array.getFloat(1), 0.01);
        assertEquals(3.0, array.getFloat(2), 0.01);
        assertEquals(4.0, array.getFloat(3), 0.01);
        assertEquals(5.0, array.getFloat(4), 0.01);

        /*
        // checking INDArray validity
        assertEquals(1.0, array.getDouble(0), 0.01);
        assertEquals(2.0, array.getDouble(1), 0.01);
        assertEquals(3.0, array.getDouble(2), 0.01);
        assertEquals(4.0, array.getDouble(3), 0.01);
        assertEquals(5.0, array.getDouble(4), 0.01);
*/
        // checking workspace memory space
    }



    @Override
    public char ordering() {
        return 'c';
    }
}
