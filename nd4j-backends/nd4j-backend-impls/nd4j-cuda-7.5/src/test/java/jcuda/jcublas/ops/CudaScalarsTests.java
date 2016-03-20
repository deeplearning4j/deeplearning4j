package jcuda.jcublas.ops;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.allocation.PinnedMemoryStrategy;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.Assert.assertEquals;

/**
 * These tests are set up just to track changes done to various GPU transfer mechanics
 *
 * Each test will address specific memory allocation strategy
 *
 * @author raver119@gmail.com
 */
public class CudaScalarsTests {





    @Test
    public void testPinnedScalarDiv() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{2.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});

        array2.divi(0.5f);

        System.out.println("Divi result: " + array2.getFloat(0));
        assertEquals(4.0f, array2.getFloat(0), 0.01f);
    }

    @Test
    public void testPinnedScalarAdd() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});

        array2.addi(0.5f);

        System.out.println("Addi result: " + array2.getFloat(0));
        assertEquals(1.5f, array2.getFloat(0), 0.01f);
    }

    @Test
    public void testPinnedScalarSub() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{2.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});

        array2.subi(0.5f);

        System.out.println("Subi result: " + array2.getFloat(0));
        assertEquals(1.5f, array2.getFloat(0), 0.01f);
    }


    @Test
    public void testPinnedScalarMul() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});

        array2.muli(2.0f);

        System.out.println("Mul result: " + array2.getFloat(0));
        assertEquals(2.0f, array2.getFloat(0), 0.01f);
    }

    @Test
    public void testPinnedScalarRDiv() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{2.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});

        array2.rdivi(0.5f);

        System.out.println("RDiv result: " + array2.getFloat(0));
        assertEquals(0.25f, array2.getFloat(0), 0.01f);
    }

    @Test
    public void testPinnedScalarRSub() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{2.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});

        array2.rsubi(0.5f);

        System.out.println("RSub result: " + array2.getFloat(0));
        assertEquals(-1.5f, array2.getFloat(0), 0.01f);
    }

    @Test
    public void testPinnedScalarMax() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{2.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});

        INDArray max = Transforms.max(array2, 0.5f, true);

        System.out.println("Max result: " + max);
        assertEquals(2.0f, array2.getFloat(0), 0.01f);
        assertEquals(1.0f, array2.getFloat(1), 0.01f);
    }

    @Test
    public void testPinnedScalarLessOrEqual() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{2.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});

        INDArray max = array2.lti(1.1f);

        System.out.println("LTI result: " + max);
        assertEquals(0.0, array2.getFloat(0), 0.01f);
    }

    @Test
    public void testPinnedScalarGreaterOrEqual() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{2.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});

        INDArray max = array2.gti(1.1f);

        System.out.println("LTI result: " + max);
        assertEquals(1.0, array2.getFloat(0), 0.01f);
        assertEquals(0.0, array2.getFloat(1), 0.01f);
    }

    @Test
    public void testPinnedScalarEquals() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{2.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});

        INDArray max = array2.eqi(2.0f);

        System.out.println("EQI result: " + max);
        assertEquals(1.0, array2.getFloat(0), 0.01f);
        assertEquals(0.0, array2.getFloat(1), 0.01f);
    }

    @Test
    public void testPinnedScalarSet() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{2.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});

        INDArray max = array2.assign(3.0f);

        System.out.println("EQI result: " + max);
        System.out.println("Array2 result: " + array2);
        assertEquals(3.0, array2.getFloat(0), 0.01f);
        assertEquals(3.0, array2.getFloat(1), 0.01f);
    }

    @Test
    public void testScalMul2() throws Exception {
        INDArray array1 = Nd4j.linspace(1, 784000, 784000).reshape(784, 1000);
        INDArray array2 = Nd4j.linspace(1, 784000, 784000).reshape(784, 1000).dup('f');

        array1.muli(0.5f);
        array2.muli(0.5f);
        //System.out.println("MUL result: " + array1);
        assertEquals(0.5f, array1.getDouble(0), 0.0001f);
        assertEquals(392000f, array1.getDouble(783999), 0.0001f);

        assertEquals(array1, array2);
    }

    @Test
    public void testScalMul3() throws Exception {
        INDArray array1 = Nd4j.linspace(1, 784, 784).reshape(1, 784);
        INDArray array2 = Nd4j.linspace(1, 784, 784).reshape(1, 784).dup('f');

        array1.divi(0.5f);
        array2.divi(0.5f);
        //System.out.println("MUL result: " + array1);
        assertEquals(2.0f, array1.getDouble(0), 0.0001f);
        assertEquals(1568f, array1.getDouble(783), 0.0001f);

        assertEquals(array1, array2);
    }

    @Test
    public void testScalMul4() throws Exception {
        INDArray array1 = Nd4j.zeros(1, 784);
        INDArray array2 = Nd4j.zeros(1, 784).dup('f');

        array1.divi(255f);
        array2.divi(255f);
        //System.out.println("MUL result: " + array1);
        assertEquals(0.0f, array1.getDouble(0), 0.0001f);
        assertEquals(0.0f, array1.getDouble(783), 0.0001f);

        assertEquals(array1, array2);
    }

    @Test
    public void testScalMul5() throws Exception {
        INDArray array1 = Nd4j.create(new double[] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 191.0, 253.0, 253.0, 253.0, 60.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 190.0, 251.0, 251.0, 251.0, 230.0, 166.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 32.0, 127.0, 0.0, 190.0, 251.0, 251.0, 251.0, 253.0, 220.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 24.0, 84.0, 221.0, 229.0, 251.0, 139.0, 23.0, 31.0, 225.0, 251.0, 253.0, 248.0, 111.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 194.0, 251.0, 251.0, 251.0, 251.0, 218.0, 39.0, 0.0, 83.0, 193.0, 253.0, 251.0, 126.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 24.0, 194.0, 255.0, 253.0, 253.0, 253.0, 253.0, 255.0, 63.0, 0.0, 0.0, 100.0, 255.0, 253.0, 173.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.0, 186.0, 251.0, 253.0, 251.0, 251.0, 251.0, 251.0, 221.0, 54.0, 0.0, 0.0, 0.0, 253.0, 251.0, 251.0, 149.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.0, 189.0, 251.0, 251.0, 253.0, 251.0, 251.0, 219.0, 126.0, 0.0, 0.0, 0.0, 0.0, 0.0, 253.0, 251.0, 251.0, 188.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 141.0, 251.0, 251.0, 253.0, 251.0, 251.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 114.0, 251.0, 251.0, 244.0, 83.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 127.0, 251.0, 251.0, 253.0, 231.0, 94.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 96.0, 251.0, 251.0, 251.0, 94.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 92.0, 253.0, 253.0, 253.0, 255.0, 63.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 96.0, 253.0, 253.0, 253.0, 95.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 197.0, 251.0, 251.0, 251.0, 161.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 115.0, 251.0, 251.0, 251.0, 94.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 96.0, 251.0, 251.0, 251.0, 172.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 253.0, 251.0, 251.0, 219.0, 47.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 96.0, 251.0, 251.0, 251.0, 94.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.0, 162.0, 253.0, 251.0, 251.0, 148.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 96.0, 251.0, 251.0, 251.0, 94.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 158.0, 181.0, 251.0, 253.0, 251.0, 172.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 96.0, 253.0, 253.0, 253.0, 153.0, 96.0, 96.0, 96.0, 96.0, 96.0, 155.0, 253.0, 253.0, 253.0, 253.0, 255.0, 253.0, 126.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 24.0, 185.0, 251.0, 251.0, 251.0, 253.0, 251.0, 251.0, 251.0, 251.0, 253.0, 251.0, 251.0, 251.0, 251.0, 253.0, 207.0, 31.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.0, 188.0, 251.0, 251.0, 253.0, 251.0, 251.0, 251.0, 251.0, 253.0, 251.0, 251.0, 251.0, 172.0, 126.0, 31.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 95.0, 188.0, 228.0, 253.0, 251.0, 251.0, 251.0, 251.0, 253.0, 243.0, 109.0, 31.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 59.0, 95.0, 94.0, 94.0, 94.0, 193.0, 95.0, 82.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        array1.divi(255f);

        System.out.println("Array1: "+ array1);
    }
}
