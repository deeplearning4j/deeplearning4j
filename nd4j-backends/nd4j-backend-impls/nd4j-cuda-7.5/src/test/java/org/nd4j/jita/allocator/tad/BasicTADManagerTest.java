package org.nd4j.jita.allocator.tad;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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

        DataBuffer tad = tadManager.getTADOnlyShapeInfo(array, new int[]{0}, 1);

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

}