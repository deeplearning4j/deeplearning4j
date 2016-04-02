package jcuda.jcublas.ops;

import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.IndexAccumulation;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMin;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.allocation.PinnedMemoryStrategy;
import org.nd4j.linalg.jcublas.context.ContextHolder;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
@Ignore
public class CudaIndexReduceTests {

    @Test
    public void testPinnedIMax() throws Exception {
        // simple way to stop test if we're not on CUDA backend here

        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());


        INDArray array1 = Nd4j.create(new float[]{1.0f, 0.1f, 2.0f, 3.0f, 4.0f, 5.0f});



        int idx =  ((IndexAccumulation) Nd4j.getExecutioner().exec(new IMax(array1))).getFinalResult();

        System.out.println("Array1: " + array1);

        assertEquals(5, idx);
    }

    @Test
    public void testPinnedIMaxLarge() throws Exception {
        // simple way to stop test if we're not on CUDA backend here

        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());


        INDArray array1 = Nd4j.linspace(1,1024,1024);



        int idx =  ((IndexAccumulation) Nd4j.getExecutioner().exec(new IMax(array1))).getFinalResult();

        System.out.println("Array1: " + array1);

        assertEquals(1023, idx);
    }

    @Test
    public void testIMaxDimensional() throws Exception {
        INDArray toArgMax = Nd4j.linspace(1,24,24).reshape(4, 3, 2);
        INDArray valueArray = Nd4j.valueArrayOf(new int[]{4, 2}, 2.0);
        INDArray valueArrayTwo = Nd4j.valueArrayOf(new int[]{3,2},3.0);
        INDArray valueArrayThree = Nd4j.valueArrayOf(new int[]{4,3},1.0);

        INDArray  argMax = Nd4j.argMax(toArgMax, 1);
        assertEquals(valueArray, argMax);

        INDArray argMaxZero = Nd4j.argMax(toArgMax,0);
        assertEquals(valueArrayTwo, argMaxZero);

        INDArray argMaxTwo = Nd4j.argMax(toArgMax,2);
        assertEquals(valueArrayThree,argMaxTwo);
    }

    @Test
    public void testPinnedIMax2() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{6.0f, 0.1f, 2.0f, 3.0f, 7.0f, 5.0f});

        int idx =  ((IndexAccumulation) Nd4j.getExecutioner().exec(new IMax(array1))).getFinalResult();

        System.out.println("Array1: " + array1);

        assertEquals(4, idx);
    }

    @Test
    public void testPinnedIMax3() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{6.0f, 0.1f, 2.0f, 3.0f, 7.0f, 9.0f});

        int idx =  ((IndexAccumulation) Nd4j.getExecutioner().exec(new IMax(array1))).getFinalResult();

        System.out.println("Array1: " + array1);

        assertEquals(5, idx);
    }

    @Test
    public void testPinnedIMin() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.0f, 0.1f, 2.0f, 3.0f, 4.0f, 5.0f});

        int idx =  ((IndexAccumulation) Nd4j.getExecutioner().exec(new IMin(array1))).getFinalResult();

        System.out.println("Array1: " + array1);

        assertEquals(1, idx);
    }

    @Test
    public void testPinnedIMin2() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{0.1f, 1.1f, 2.0f, 3.0f, 4.0f, 5.0f});

        int idx =  ((IndexAccumulation) Nd4j.getExecutioner().exec(new IMin(array1))).getFinalResult();

        System.out.println("Array1: " + array1);
        assertEquals(0, idx);
    }
}
