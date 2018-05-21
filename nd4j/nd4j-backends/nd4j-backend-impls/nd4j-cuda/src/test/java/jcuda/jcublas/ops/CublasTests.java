package jcuda.jcublas.ops;

import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
@Ignore
public class CublasTests {
    @Test
    public void testGemm1() throws Exception {
        INDArray array1 = Nd4j.linspace(1, 100, 100).reshape(1, 100);
        INDArray array2 = Nd4j.linspace(1, 100, 100).reshape(100, 1);

        INDArray array3 = array1.mmul(array2);

        assertEquals(338350f, array3.getFloat(0), 0.001f);
    }

    @Test
    public void testGemm2() throws Exception {
        INDArray array1 = Nd4j.linspace(1, 100, 100).reshape('f', 1, 100);
        INDArray array2 = Nd4j.linspace(1, 100, 100).reshape('f', 100, 1);

        INDArray array3 = array1.mmul(array2);

        assertEquals(338350f, array3.getFloat(0), 0.001f);
    }

    @Test
    public void testGemm3() throws Exception {
        INDArray array1 = Nd4j.linspace(1, 1000, 1000).reshape(10, 100);
        INDArray array2 = Nd4j.linspace(1, 1000, 1000).reshape(100, 10);

        INDArray array3 = array1.mmul(array2);


        //System.out.println("Array3: " + Arrays.toString(array3.data().asFloat()));

        assertEquals(3338050.0f, array3.data().getFloat(0),0.001f);
        assertEquals(8298050.0f, array3.data().getFloat(1),0.001f);
        assertEquals(3343100.0f, array3.data().getFloat(10),0.001f);
        assertEquals(8313100.0f, array3.data().getFloat(11),0.001f);
        assertEquals(3348150.0f, array3.data().getFloat(20),0.001f);
        assertEquals(8328150.0f, array3.data().getFloat(21),0.001f);
    }

    @Test
    public void testGemm4() throws Exception {
        INDArray array1 = Nd4j.linspace(1, 1000, 1000).reshape(10, 100);
        INDArray array2 = Nd4j.linspace(1, 1000, 1000).reshape('f', 100, 10);

        INDArray array3 = array1.mmul(array2);

        //System.out.println("Array3: " + Arrays.toString(array3.data().asFloat()));

        assertEquals(338350f, array3.data().getFloat(0),0.001f);
        assertEquals(843350f, array3.data().getFloat(1),0.001f);
        assertEquals(843350f, array3.data().getFloat(10),0.001f);
        assertEquals(2348350f, array3.data().getFloat(11),0.001f);
        assertEquals(1348350f, array3.data().getFloat(20),0.001f);
        assertEquals(3853350f, array3.data().getFloat(21),0.001f);
    }

    @Test
    public void testGemm5() throws Exception {
        INDArray array1 = Nd4j.linspace(1, 1000, 1000).reshape('f', 10, 100);
        INDArray array2 = Nd4j.linspace(1, 1000, 1000).reshape(100, 10);

        INDArray array3 = array1.mmul(array2);

        //System.out.println("Array3: " + Arrays.toString(array3.data().asFloat()));

        assertEquals(3.293408E7f, array3.data().getFloat(0),10f);
        assertEquals(3.29837E7f, array3.data().getFloat(1),10f);
        assertEquals(3.3835E7f, array3.data().getFloat(99),10f);
    }

    @Test
    public void testGemm6() throws Exception {
        INDArray array1 = Nd4j.linspace(1, 1000, 1000).reshape('f', 10, 100);
        INDArray array2 = Nd4j.linspace(1, 1000, 1000).reshape('f', 100, 10);

        INDArray array3 = array1.mmul(array2);

        //System.out.println("Array3: " + Arrays.toString(array3.data().asFloat()));

        assertEquals(3338050.0f, array3.data().getFloat(0),0.001f);
        assertEquals(3343100f, array3.data().getFloat(1),0.001f);
        assertEquals(8298050f, array3.data().getFloat(10),0.001f);
        assertEquals(8313100.0f, array3.data().getFloat(11),0.001f);
        assertEquals(1.325805E7f, array3.data().getFloat(20),5f);
        assertEquals(1.32831E7f, array3.data().getFloat(21),5f);
    }

    @Test
    public void testGemm7() throws Exception {
        INDArray array1 = Nd4j.linspace(1, 1000, 1000).reshape(10, 100);
        INDArray array2 = Nd4j.linspace(1, 1000, 1000).reshape(100, 10);
        INDArray array3 = Nd4j.create(10, 10);


        array1.mmul(array2, array3);

//        System.out.println("Array3: " + Arrays.toString(array3.data().asFloat()));

        assertEquals(3338050.0f, array3.data().getFloat(0),0.001f);
        assertEquals(8298050.0f, array3.data().getFloat(1),0.001f);
        assertEquals(3343100.0f, array3.data().getFloat(10),0.001f);
        assertEquals(8313100.0f, array3.data().getFloat(11),0.001f);
        assertEquals(3348150.0f, array3.data().getFloat(20),0.001f);
        assertEquals(8328150.0f, array3.data().getFloat(21),0.001f);
    }

    @Test
    public void testGemm8() throws Exception {
        INDArray array1 = Nd4j.ones(10, 10);
        INDArray array2 = Nd4j.ones(10, 10);
        INDArray array3 = Nd4j.create(10, 10);

        array1.mmul(array2, array3);

        assertEquals(10.0f, array3.data().getFloat(0),0.001f);
        assertEquals(10.0f, array3.data().getFloat(1),0.001f);
        assertEquals(10.0f, array3.data().getFloat(10),0.001f);
        assertEquals(10.0f, array3.data().getFloat(11),0.001f);
        assertEquals(10.0f, array3.data().getFloat(20),0.001f);
        assertEquals(10.0f, array3.data().getFloat(21),0.001f);
    }

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
