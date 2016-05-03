package org.nd4j.linalg.api.rng;
import static org.junit.Assert.*;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

/**
 * @author Adam Gibson
 */
@RunWith(Parameterized.class)
public class RngTests extends BaseNd4jTest {
   public RngTests(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testRngConstitency() {
        Nd4j.getRandom().setSeed(123);
        INDArray arr = Nd4j.rand(1,5);
        Nd4j.getRandom().setSeed(123);
        INDArray arr2 = Nd4j.rand(1,5);
        assertEquals(arr,arr2);
    }

    @Test
    public void testRandomWithOrder(){

        Nd4j.getRandom().setSeed(12345);

        INDArray arr = Nd4j.rand('c',3,4);
        assertArrayEquals(new int[]{3,4}, arr.shape());
        assertEquals('c', arr.ordering());
        assertTrue(arr.minNumber().doubleValue() >= 0.0);
        assertTrue(arr.maxNumber().doubleValue() <= 1.0);

        INDArray arr2 = Nd4j.rand('f',3,4);
        assertArrayEquals(new int[]{3,4}, arr2.shape());
        assertEquals('f', arr2.ordering());
        assertTrue(arr2.minNumber().doubleValue() >= 0.0);
        assertTrue(arr2.maxNumber().doubleValue() <= 1.0);

        INDArray arr3 = Nd4j.rand('c',new int[]{3,4,5});
        assertArrayEquals(new int[]{3,4,5}, arr3.shape());
        assertEquals('c', arr3.ordering());
        assertTrue(arr3.minNumber().doubleValue() >= 0.0);
        assertTrue(arr3.maxNumber().doubleValue() <= 1.0);

        INDArray arr4 = Nd4j.rand('f',new int[]{3,4,5});
        assertArrayEquals(new int[]{3,4,5}, arr4.shape());
        assertEquals('f', arr4.ordering());
        assertTrue(arr4.minNumber().doubleValue() >= 0.0);
        assertTrue(arr4.maxNumber().doubleValue() <= 1.0);


        INDArray narr = Nd4j.randn('c',3,4);
        assertArrayEquals(new int[]{3,4}, narr.shape());
        assertEquals('c', narr.ordering());
        assertEquals(narr.meanNumber().doubleValue(), 0.0, 0.5);

        INDArray narr2 = Nd4j.randn('f',3,4);
        assertArrayEquals(new int[]{3,4}, narr2.shape());
        assertEquals('f', narr2.ordering());
        assertEquals(narr2.meanNumber().doubleValue(), 0.0, 0.5);

        INDArray narr3 = Nd4j.randn('c',new int[]{3,4,5});
        assertArrayEquals(new int[]{3,4,5}, narr3.shape());
        assertEquals('c', narr3.ordering());
        assertEquals(narr3.meanNumber().doubleValue(), 0.0, 0.5);

        INDArray narr4 = Nd4j.randn('f',new int[]{3,4,5});
        assertArrayEquals(new int[]{3,4,5}, narr4.shape());
        assertEquals('f', narr4.ordering());
        assertEquals(narr4.meanNumber().doubleValue(), 0.0, 0.5);

    }


    @Override
    public char ordering() {
        return 'f';
    }

}
