package org.nd4j.linalg.specials;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertEquals;

@Slf4j
@RunWith(Parameterized.class)
public class CudaTests extends BaseNd4jTest {

    DataBuffer.Type initialType;

    public CudaTests(Nd4jBackend backend) {
        super(backend);
        this.initialType = Nd4j.dataType();
    }

    @Before
    public void setUp() {
            Nd4j.setDataType(DataBuffer.Type.FLOAT);
        }

    @After
    public void setDown() {
            Nd4j.setDataType(initialType);
        }

    @Test
    public void testMGrid_1() {
        if (!(Nd4j.getExecutioner() instanceof GridExecutioner))
            return;

        val arrayA = Nd4j.create(128, 128);
        val arrayB = Nd4j.create(128, 128);
        val arrayC = Nd4j.create(128, 128);

        arrayA.muli(arrayB);

        val executioner = (GridExecutioner) Nd4j.getExecutioner();

        assertEquals(1, executioner.getQueueLength());

        arrayA.addi(arrayC);

        assertEquals(1, executioner.getQueueLength());
    }


    @Test
    public void testMGrid_2() {
        if (!(Nd4j.getExecutioner() instanceof GridExecutioner))
            return;

        val exp = Nd4j.create(128, 128).assign(2.0);
        Nd4j.getExecutioner().commit();

        val arrayA = Nd4j.create(128, 128);
        val arrayB = Nd4j.create(128, 128);
        arrayA.muli(arrayB);

        val executioner = (GridExecutioner) Nd4j.getExecutioner();

        assertEquals(1, executioner.getQueueLength());

        arrayA.addi(2.0f);

        assertEquals(0, executioner.getQueueLength());

        Nd4j.getExecutioner().commit();

        assertEquals(exp, arrayA);
    }


    @Override
    public char ordering() {
        return 'c';
    }
}
