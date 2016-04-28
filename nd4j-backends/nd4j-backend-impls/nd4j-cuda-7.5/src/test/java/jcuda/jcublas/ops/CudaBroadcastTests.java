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
public class CudaBroadcastTests {

    @Test
    public void testPinnedAddiRowVector() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        for (int iter = 0; iter < 100; iter++) {

            INDArray array1 = Nd4j.zeros(15, 15);

            for (int y = 0; y < 15; y++) {
                for (int x = 0; x < 15; x++) {
                    assertEquals("Failed on iteration: ["+iter+"], y.x: ["+y+"."+x+"]", 0.0f, array1.getRow(y).getFloat(x), 0.01);
                }
            }
            INDArray array2 = Nd4j.create(new float[]{2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f});

            for (int i = 0; i < 30; i++) {
                array1.addiRowVector(array2);
            }

            //System.out.println("Array1: " + array1);
            //System.out.println("Array2: " + array2);

            for (int y = 0; y < 15; y++) {
                for (int x = 0; x < 15; x++) {
                    assertEquals("Failed on iteration: ["+iter+"], y.x: ["+y+"."+x+"]", 60.0f, array1.getRow(y).getFloat(x), 0.01);
                }
            }
        }
    }

    @Test
    public void testPinnedSubiRowVector() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        INDArray array1 = Nd4j.zeros(1500,150);
        INDArray array2 = Nd4j.linspace(1,150,150);

        array1.subiRowVector(array2);

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(-1.0f, array1.getRow(0).getFloat(0), 0.01);
    }

    @Test
    public void testPinnedSubiColumnVector2() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        INDArray array1 = Nd4j.zeros(1500,150);
        INDArray array2 = Nd4j.linspace(1,1500,1500).reshape(1500,1);

        array1.subiColumnVector(array2);

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(-1.0f, array1.getRow(0).getFloat(0), 0.01);
        assertEquals(-1.0f, array1.getRow(0).getFloat(0), 0.01);
        assertEquals(-301.0f, array1.getRow(300).getFloat(0), 0.01);
        assertEquals(-1500.0f, array1.getRow(1499).getFloat(0), 0.01);
    }

    @Test
    public void testPinnedSubiRowVector2() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        INDArray array1 = Nd4j.zeros(1500,150);
        INDArray array2 = Nd4j.linspace(1,1500,1500).reshape(1500,1);

        array1.subiRowVector(array2);

        //System.out.println("Array1: " + array1);
        //System.out.println("Array2: " + array2);

        assertEquals(-1.0f, array1.getRow(0).getFloat(0), 0.01);
        assertEquals(-1.0f, array1.getRow(0).getFloat(0), 0.01);
        assertEquals(-301.0f, array1.getRow(300).getFloat(0), 0.01);
        assertEquals(-1500.0f, array1.getRow(1499).getFloat(0), 0.01);
    }

    @Test
    public void testPinnedRSubiRowVector() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.zeros(15,15);
        INDArray array2 = Nd4j.create(new float[]{2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f});

        array1.rsubiRowVector(array2);

        //System.out.println("Array1: " + array1);
        //System.out.println("Array2: " + array2);

        assertEquals(2.0f, array1.getRow(0).getFloat(0), 0.01);
    }

    @Test
    public void testPinnedSubiColumnVector() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        INDArray array1 = Nd4j.zeros(150,3);
        INDArray array2 = Nd4j.linspace(1, 150, 150).reshape(150,1);

        array1.subiColumnVector(array2);

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(-1.0f, array1.getRow(0).getFloat(0), 0.01);
        assertEquals(-2.0f, array1.getRow(1).getFloat(0), 0.01);
        assertEquals(-3.0f, array1.getRow(2).getFloat(0), 0.01);
    }

    @Test
    public void testPinnedMulRowVector() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.zeros(15,15);
        array1.putRow(0, Nd4j.create(new float[]{2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f}));
        array1.putRow(1, Nd4j.create(new float[]{2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f}));
        INDArray array2 = Nd4j.create(new float[]{2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f});

        array1.muliRowVector(array2);

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(4.0f, array1.getRow(0).getFloat(0), 0.01);
    }

    @Test
    public void testPinnedDivRowVector() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.zeros(15,15);
        array1.putRow(0, Nd4j.create(new float[]{2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f}));
        array1.putRow(1, Nd4j.create(new float[]{2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f}));
        INDArray array2 = Nd4j.create(new float[]{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

        array1.diviRowVector(array2);

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(2.0f, array1.getRow(0).getFloat(0), 0.01);
    }

    @Test
    public void testPinnedRDivRowVector() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.zeros(15,15);
        array1.putRow(0, Nd4j.create(new float[]{2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f}));
        array1.putRow(1, Nd4j.create(new float[]{2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f}));
        INDArray array2 = Nd4j.create(new float[]{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

        array1.rdiviRowVector(array2);

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(0.5f, array1.getRow(0).getFloat(0), 0.01);
    }
}

