package org.nd4j.linalg.api.blas;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class Level2Test {
    @Test
    public void testGemv1() throws Exception {
        INDArray array1 = Nd4j.linspace(1, 1000, 1000).reshape(10, 100);
        INDArray array2 = Nd4j.linspace(1,100, 100).reshape(100,1);

        INDArray array3 = array1.mmul(array2);

        assertEquals(10, array3.length());
        assertEquals(338350f, array3.getFloat(0), 0.001f);
        assertEquals(843350f, array3.getFloat(1), 0.001f);
        assertEquals(1348350f, array3.getFloat(2), 0.001f);
        assertEquals(1853350f, array3.getFloat(3), 0.001f);
    }

    @Test
    public void testGemv2() throws Exception {
        INDArray array1 = Nd4j.linspace(1, 1000, 1000).reshape(10, 100);
        INDArray array2 = Nd4j.linspace(1,100, 100).reshape('f', 100,1);

        INDArray array3 = array1.mmul(array2);

        assertEquals(10, array3.length());
        assertEquals(338350f, array3.getFloat(0), 0.001f);
        assertEquals(843350f, array3.getFloat(1), 0.001f);
        assertEquals(1348350f, array3.getFloat(2), 0.001f);
        assertEquals(1853350f, array3.getFloat(3), 0.001f);
    }

    @Test
    public void testGemv3() throws Exception {
        INDArray array1 = Nd4j.linspace(1, 1000, 1000).reshape('f', 10, 100);
        INDArray array2 = Nd4j.linspace(1,100, 100).reshape('f', 100,1);

        INDArray array3 = array1.mmul(array2);

        assertEquals(10, array3.length());
        assertEquals(3338050f, array3.getFloat(0), 0.001f);
        assertEquals(3343100f, array3.getFloat(1), 0.001f);
        assertEquals(3348150f, array3.getFloat(2), 0.001f);
        assertEquals(3353200f, array3.getFloat(3), 0.001f);
    }

    @Test
    public void testGemv4() throws Exception {
        INDArray array1 = Nd4j.linspace(1, 1000, 1000).reshape('f', 10, 100);
        INDArray array2 = Nd4j.linspace(1,100, 100).reshape(100,1);

        INDArray array3 = array1.mmul(array2);

        assertEquals(10, array3.length());
        assertEquals(3338050f, array3.getFloat(0), 0.001f);
        assertEquals(3343100f, array3.getFloat(1), 0.001f);
        assertEquals(3348150f, array3.getFloat(2), 0.001f);
        assertEquals(3353200f, array3.getFloat(3), 0.001f);
    }
}
