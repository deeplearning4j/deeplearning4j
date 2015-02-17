package org.nd4j.linalg.api.buffer.test;


import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.ArrayOps;
import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.ops.factory.ElementWiseOpFactories;

/**
 * Created by agibsonccc on 2/14/15.
 */
public abstract class DataBufferTest {


    @Test
    public void testGetSet() {
        double[] d1 = new double[]{1,2,3,4};
        DataBuffer d = Nd4j.createBuffer(d1);
        double[] d2 = d.asDouble();
        assertArrayEquals(d1, d2, 1e-1);
        d.destroy();

    }

    @Test
    public void testDup() {
        double[] d1 = new double[]{1,2,3,4};
        DataBuffer d = Nd4j.createBuffer(d1);
        DataBuffer d2 = d.dup();
        assertEquals(d,d2);
        d.destroy();
        d2.destroy();
    }

    @Test
    public void testPut() {
        double[] d1 = new double[]{1,2,3,4};
        DataBuffer d = Nd4j.createBuffer(d1);
        d.put(0,0.0);
        double[] result = new double[]{0,2,3,4};
        d1 = d.asDouble();
        assertArrayEquals(d1,result,1e-1);
        d.destroy();
    }

    @Test
    public void testApply() {
        INDArray ones = Nd4j.valueArrayOf(5, 2.0);
        ones.toString();
        DataBuffer buffer = ones.data();
        //square
        ElementWiseOp op = new ArrayOps()
                .from(ones).op(ElementWiseOpFactories.pow()).extraArgs(new Object[]{2})
                .build();
        buffer.apply(op);
        INDArray four = Nd4j.valueArrayOf(5,4.0);
        DataBuffer d = four.data();
        assertEquals(buffer,d);
        buffer.destroy();
        d.destroy();



    }

    @Test
    public void testGetRange() {
        DataBuffer buffer = Nd4j.linspace(1,5,5).data();
        double[] get = buffer.getDoublesAt(0,3);
        double[] data = new double[]{1,2,3};
        assertArrayEquals(get,data,1e-1);


        double[] get2 = buffer.asDouble();
        double[] allData = buffer.getDoublesAt(0,buffer.length());
        assertArrayEquals(get2,allData,1e-1);
        buffer.destroy();



    }


    @Test
    public void testGetOffsetRange() {
        DataBuffer buffer = Nd4j.linspace(1,5,5).data();
        double[] get = buffer.getDoublesAt(1,3);
        double[] data = new double[]{2,3,4};
        assertArrayEquals(get,data,1e-1);


        double[] allButLast = new double[]{2,3,4,5};

        double[] allData = buffer.getDoublesAt(1,buffer.length());
        assertArrayEquals(allButLast,allData,1e-1);
        buffer.destroy();


    }




}
