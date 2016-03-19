package jcuda.jcublas.ops;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.Mean;
import org.nd4j.linalg.api.ops.impl.accum.Norm2;
import org.nd4j.linalg.api.ops.impl.accum.Sum;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.allocation.PinnedMemoryStrategy;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class CudaAccumTests {



    /**
     * Sum call
     * @throws Exception
     */
    @Test
    public void testPinnedSum() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{2.01f, 2.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});


        //INDArray result = Nd4j.getExecutioner().exec(new Sum(array1), 1);

        Sum sum = new Sum(array1);
        Nd4j.getExecutioner().exec(sum, 1);

        Number resu = sum.getFinalResult();

        System.out.println("Result: " + resu);

        assertEquals(17.15f, resu.floatValue(), 0.01f);
    }

    /**
     * Mean call
     * @throws Exception
     */
    @Test
    public void testPinnedMean() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{2.01f, 2.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});


        Mean mean = new Mean(array1);
        Nd4j.getExecutioner().exec(mean, 1);

        Number resu = mean.getFinalResult();


//        INDArray result = Nd4j.getExecutioner().exec(new Mean(array1), 1);

        System.out.println("Array1: " + array1);
//        System.out.println("Result: " + result);

        assertEquals(1.14f, resu.floatValue(), 0.01f);
    }

    @Test
    public void testSum2() {
        INDArray n = Nd4j.create(Nd4j.linspace(1, 8, 8).data(), new int[]{2, 2, 2});
        System.out.println("N result: " + n);
        INDArray test = Nd4j.create(new float[]{3, 7, 11, 15}, new int[]{2, 2});
        System.out.println("Test result: " + test);
        INDArray sum = n.sum(-1);

        System.out.println("elementWiseStride: " + n.elementWiseStride());
        System.out.println("elementStride: " + n.elementStride());

        System.out.println("Sum result: " + sum);
        assertEquals(test, sum);
    }

    @Test
    public void testNorm2() throws Exception {
        INDArray array1 = Nd4j.create(new float[]{2.01f, 2.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});

        INDArray result = array1.norm2(1);

        System.out.println(result);
        assertEquals(4.62f,  result.getDouble(0), 0.001);
    }
}
