package org.nd4j.linalg.jcublas.buffer;

import org.junit.Test;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class CudaFloatDataBufferTest {

    @Test
    public void getDouble() throws Exception {
        DataBuffer buffer = Nd4j.createBuffer(new float[]{1f,2f,3f,4f});

        assertEquals("CudaFloatDataBuffer", buffer.getClass().getSimpleName());


        System.out.println("Starting check...");

        assertEquals(1.0f, buffer.getFloat(0), 0.001f);
        assertEquals(2.0f, buffer.getFloat(1), 0.001f);
        assertEquals(3.0f, buffer.getFloat(2), 0.001f);
        assertEquals(4.0f, buffer.getFloat(3), 0.001f);

        System.out.println("Data: " + buffer);
    }

    @Test
    public void testPut() throws Exception {
        DataBuffer buffer = Nd4j.createBuffer(new float[]{1f,2f,3f,4f});
        buffer.put(2, 16f);

        assertEquals(16.0f, buffer.getFloat(2), 0.001f);

        System.out.println("Data: " + buffer);
    }

    @Test
    public void testNdArrayView1() throws Exception {
        INDArray array = Nd4j.create(new float[] {1f,2f,3f,4f});

        assertEquals(1.0f, array.getFloat(0), 0.001f);
        assertEquals(2.0f, array.getFloat(1), 0.001f);
        assertEquals(3.0f, array.getFloat(2), 0.001f);
        assertEquals(4.0f, array.getFloat(3), 0.001f);
    }

    @Test
    public void testNdArrayView2() throws Exception {
        INDArray array = Nd4j.create(10, 10);

        long tp1 = array.data().getTrackingPoint();

        array.putScalar(0, 10f);
        assertEquals(10.0f, array.getFloat(0), 0.01f);

        System.out.println("X1 --------------------------------");

        INDArray array2 = array.slice(1);
        long tp2 = array2.data().getTrackingPoint();

        assertEquals(tp1, tp2);

        array2.putScalar(0, 10);
        System.out.println("X2 --------------------------------");
        assertEquals(10.0f, array2.getFloat(0), 0.01f);

        tp2 = array2.data().getTrackingPoint();
        tp1 = array.data().getTrackingPoint();

        assertEquals(tp1, tp2);
    }

    @Test
    public void testINDArrayOffsets2() throws Exception {
        INDArray array = Nd4j.linspace(0, 24, 25).reshape(5, 5);

        assertEquals(6.0f, array.getFloat(6), 0.01f);

        INDArray slice0 = array.slice(0);
        assertEquals(0f, slice0.getFloat(0), 0.01f);

        INDArray slice1 = array.slice(1);
        assertEquals(5f, slice1.getFloat(0), 0.01f);

        INDArray slice2 = array.slice(2);
        assertEquals(10f, slice2.getFloat(0), 0.01f);

        INDArray slice3 = array.slice(3);
        assertEquals(15f, slice3.getFloat(0), 0.01f);

        assertEquals(array.data().getTrackingPoint(), slice0.data().getTrackingPoint());
        assertEquals(array.data().getTrackingPoint(), slice1.data().getTrackingPoint());
        assertEquals(array.data().getTrackingPoint(), slice2.data().getTrackingPoint());
        assertEquals(array.data().getTrackingPoint(), slice3.data().getTrackingPoint());
    }

    @Test
    public void testINDArrayOffsets3() throws Exception {
        INDArray array = Nd4j.linspace(0, 24, 25).reshape(5, 5);
        INDArray array2 = Nd4j.create(5, 5);

        assertEquals(6.0f, array.getFloat(6), 0.01f);

        INDArray slice0 = array.slice(0);
        assertEquals(0f, slice0.getFloat(0), 0.01f);

        array2.putRow(0, slice0);
        assertEquals(slice0, array2.slice(0));

        INDArray slice1 = array.slice(1);
        assertEquals(5f, slice1.getFloat(0), 0.01f);

        System.out.println("---------------------------------------------------------------------");
        array2.putRow(1, slice1);
//        assertFalse(true);
        assertEquals(slice1, array2.slice(1));
        System.out.println("---------------------------------------------------------------------");


        INDArray slice2 = array.slice(2);
        assertEquals(10f, slice2.getFloat(0), 0.01f);

        INDArray slice3 = array.slice(3);
        assertEquals(15f, slice3.getFloat(0), 0.01f);

        assertEquals(array.data().getTrackingPoint(), slice0.data().getTrackingPoint());
        assertEquals(array.data().getTrackingPoint(), slice1.data().getTrackingPoint());
        assertEquals(array.data().getTrackingPoint(), slice2.data().getTrackingPoint());
        assertEquals(array.data().getTrackingPoint(), slice3.data().getTrackingPoint());
    }

    @Test
    public void testDup1() throws Exception {
        INDArray array0 = Nd4j.create(10);
        INDArray array7 = Nd4j.create(10);

        CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();
        context.syncOldStream();

        long time1 = System.nanoTime();
        INDArray array1 =  Nd4j.linspace(0, 9, 1000);
        long time2 = System.nanoTime();
//        context.syncOldStream();
        long time3 = System.nanoTime();
        INDArray array2 = array1.dup();
        long time4 = System.nanoTime();
        //context.syncOldStream();

        System.out.println("Linspace time: " + (time2 - time1));
        System.out.println("Dup time: " + (time4 - time3));
        System.out.println("Total time: " + ((time2 - time1)+ (time4 - time3)));

        assertEquals(array1, array2);
        assertNotEquals(array1.data().getTrackingPoint(), array2.data().getTrackingPoint());
    }
}