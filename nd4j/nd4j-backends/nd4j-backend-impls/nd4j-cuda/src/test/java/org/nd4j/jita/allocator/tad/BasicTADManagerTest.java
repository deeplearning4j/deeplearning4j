package org.nd4j.jita.allocator.tad;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cache.ArrayDescriptor;
import org.nd4j.linalg.cache.TADManager;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;

import static org.junit.Assert.*;

/**
 * Created by raver on 11.05.2016.
 */
public class BasicTADManagerTest {

    TADManager tadManager;

    @Before
    public void setUp() throws Exception {
        tadManager = new BasicTADManager();
    }

    @Test
    public void testTADcreation1() throws Exception {
        INDArray array = Nd4j.create(10, 100);

        DataBuffer tad = tadManager.getTADOnlyShapeInfo(array, new int[]{0}).getFirst();

        System.out.println("TAD: " + tad);
        System.out.println("Shape: " + array.shapeInfoDataBuffer());

        assertEquals(2, tad.getInt(0));
        assertEquals(1, tad.getInt(1));
        assertEquals(10, tad.getInt(2));
        assertEquals(1, tad.getInt(3));
        assertEquals(100, tad.getInt(4));
        assertEquals(0, tad.getInt(5));
        assertEquals(100, tad.getInt(6));
        assertEquals(99, tad.getInt(7));
    }

    @Test
    public void testTADcreation2() throws Exception {
        INDArray array = Nd4j.create(10, 100);

        TADManager tadManager  = new DeviceTADManager();

        DataBuffer tad = tadManager.getTADOnlyShapeInfo(array, new int[]{0}).getFirst();
        DataBuffer tad2 = tadManager.getTADOnlyShapeInfo(array, new int[]{0}).getFirst();

        System.out.println("TAD: " + tad);
        System.out.println("Shape: " + array.shapeInfoDataBuffer());

        CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();

        assertEquals(2, tad.getInt(0));
        assertEquals(1, tad.getInt(1));
        assertEquals(10, tad.getInt(2));
        assertEquals(1, tad.getInt(3));
        assertEquals(100, tad.getInt(4));
        assertEquals(0, tad.getInt(5));
        assertEquals(100, tad.getInt(6));
        assertEquals(99, tad.getInt(7));

        assertFalse(AtomicAllocator.getInstance().getAllocationPoint(tad).isActualOnDeviceSide());

        long tadPointer1 = AtomicAllocator.getInstance().getPointer(tad, context).address();
        long tadPointer2 = AtomicAllocator.getInstance().getPointer(tad2, context).address();

        assertTrue(AtomicAllocator.getInstance().getAllocationPoint(tad).isActualOnDeviceSide());

        System.out.println("tadPointer1: " + tadPointer1);
        System.out.println("tadPointer2: " + tadPointer2);

        assertEquals(tadPointer1, tadPointer2);

        AtomicAllocator.getInstance().moveToConstant(tad);

        long tadPointer3 = AtomicAllocator.getInstance().getPointer(tad, context).address();
        long tadPointer4 = AtomicAllocator.getInstance().getPointer(tad2, context).address();

        assertEquals(tadPointer4, tadPointer3);
        assertNotEquals(tadPointer1, tadPointer3);
    }



    @Test
    public void testArrayDesriptor1() throws Exception {
        ArrayDescriptor descriptor1 = new ArrayDescriptor(new int[] {2, 3, 4});
        ArrayDescriptor descriptor2 = new ArrayDescriptor(new int[] {2, 4, 3});
        ArrayDescriptor descriptor3 = new ArrayDescriptor(new int[] {3, 2, 4});
        ArrayDescriptor descriptor4 = new ArrayDescriptor(new int[] {4, 2, 3});
        ArrayDescriptor descriptor5 = new ArrayDescriptor(new int[] {4, 3, 2});

        assertNotEquals(descriptor1, descriptor2);
        assertNotEquals(descriptor2, descriptor3);
        assertNotEquals(descriptor3, descriptor4);
        assertNotEquals(descriptor4, descriptor5);
        assertNotEquals(descriptor1, descriptor3);
        assertNotEquals(descriptor1, descriptor4);
        assertNotEquals(descriptor1, descriptor5);
        assertNotEquals(descriptor2, descriptor4);
        assertNotEquals(descriptor2, descriptor5);
    }

}