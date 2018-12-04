package org.nd4j.linalg.api.buffer;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertEquals;

@RunWith(Parameterized.class)
public class DataBufferTests extends BaseNd4jTest {

    public DataBufferTests(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testNoArgCreateBuffer(){

        //Float
        DataBuffer f = Nd4j.createBuffer(new float[]{1,2,3});
        assertEquals(DataType.FLOAT, f.dataType());
        assertEquals(3, f.length());

        f = Nd4j.createBuffer(new float[]{1,2,3}, 0);
        assertEquals(DataType.FLOAT, f.dataType());
        assertEquals(3, f.length());

        f = Nd4j.createBufferDetached(new float[]{1,2,3});
        assertEquals(DataType.FLOAT, f.dataType());
        assertEquals(3, f.length());

        //Double
        DataBuffer d = Nd4j.createBuffer(new double[]{1,2,3});
        assertEquals(DataType.DOUBLE, d.dataType());
        assertEquals(3, d.length());

        d = Nd4j.createBuffer(new double[]{1,2,3}, 0);
        assertEquals(DataType.DOUBLE, d.dataType());
        assertEquals(3, d.length());

        d = Nd4j.createBufferDetached(new double[]{1,2,3});
        assertEquals(DataType.DOUBLE, d.dataType());
        assertEquals(3, d.length());

        //Int
        DataBuffer i = Nd4j.createBuffer(new int[]{1,2,3});
        assertEquals(DataType.INT, i.dataType());
        assertEquals(3, i.length());

        i = Nd4j.createBuffer(new int[]{1,2,3}, 0);
        assertEquals(DataType.INT, i.dataType());
        assertEquals(3, i.length());

        i = Nd4j.createBufferDetached(new int[]{1,2,3});
        assertEquals(DataType.INT, i.dataType());
        assertEquals(3, i.length());

        //Long
        DataBuffer l = Nd4j.createBuffer(new long[]{1,2,3});
        assertEquals(DataType.LONG, l.dataType());
        assertEquals(3, l.length());

        l = Nd4j.createBuffer(new long[]{1,2,3});
        assertEquals(DataType.LONG, l.dataType());
        assertEquals(3, l.length());

        l = Nd4j.createBufferDetached(new long[]{1,2,3});
        assertEquals(DataType.LONG, l.dataType());
        assertEquals(3, l.length());

        //byte
        DataBuffer b = Nd4j.createBuffer(new byte[]{1,2,3}, 3);
        assertEquals(DataType.BYTE, b.dataType());
        assertEquals(3, b.length());




    }

    @Override
    public char ordering(){
        return 'c';
    }

}
