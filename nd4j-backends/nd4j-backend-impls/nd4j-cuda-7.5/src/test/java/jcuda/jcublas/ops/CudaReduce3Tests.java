package jcuda.jcublas.ops;

import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.accum.distances.ManhattanDistance;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.allocation.PinnedMemoryStrategy;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
@Ignore
public class CudaReduce3Tests {


    /**
     * Norm2 + cuBlas dot call
     *
     * @throws Exception
     */
    @Test
    public void testPinnedCosineSim() throws Exception {
        // simple way to stop test if we're not on CUDA backend here

        INDArray array1 = Nd4j.create(new float[]{2.01f, 2.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});


        double similarity = Transforms.cosineSim(array1, array2);

        System.out.println("Cosine similarity: " + similarity);
        assertEquals(0.95f, similarity, 0.01f);
    }

    @Test
    public void testPinnedManhattanDistance() throws Exception {
        // simple way to stop test if we're not on CUDA backend here

        INDArray array1 = Nd4j.create(new float[]{0.0f, 1.0f, 2.0f, 3.0f, 4.0f});
        INDArray array2 = Nd4j.create(new float[]{0.5f, 1.5f, 2.5f, 3.5f, 4.5f});


           double result = Nd4j.getExecutioner().execAndReturn(new ManhattanDistance(array1, array2)).getFinalResult().doubleValue();

            System.out.println("Distance: " + result);
            assertEquals(2.5, result, 0.01);

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);
    }

    @Test
    public void testPinnedEuclideanDistance() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        INDArray array1 = Nd4j.create(new float[]{0.0f, 1.0f, 2.0f, 3.0f, 4.0f});
        INDArray array2 = Nd4j.create(new float[]{0.5f, 1.5f, 2.5f, 3.5f, 4.5f});

        double result = Nd4j.getExecutioner().execAndReturn(new EuclideanDistance(array1,array2)).getFinalResult().doubleValue();

        System.out.println("Distance: " + result);
        assertEquals(1.118033988749895, result, 0.01);
    }

    @Test
    public void testPinnedEuclideanDistance2() throws Exception {
        INDArray array1 = Nd4j.linspace(1, 10000, 10000);
        INDArray array2 = Nd4j.linspace(1, 9000, 10000);

        float result = Nd4j.getExecutioner().execAndReturn(new EuclideanDistance(array1,array2)).getFinalResult().floatValue();

        System.out.println("Distance: " + result);
        assertEquals(57736.473f, result, 0.01f);
    }

    @Test
    public void testPinnedManhattanDistance2() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        INDArray array1 = Nd4j.linspace(1, 10000, 10000);
        INDArray array2 = Nd4j.linspace(1, 9000, 10000);

        float result = Nd4j.getExecutioner().execAndReturn(new ManhattanDistance(array1, array2)).getFinalResult().floatValue();

        assertEquals(5000003.0, result, 0.001f);
    }

    @Test
    public void testPinnedEuclideanDistance3() throws Exception {
        INDArray array1 = Nd4j.linspace(1, 10000, 130);
        INDArray array2 = Nd4j.linspace(1, 9000, 130);

        float result = Nd4j.getExecutioner().execAndReturn(new EuclideanDistance(array1,array2)).getFinalResult().floatValue();

        System.out.println("Distance: " + result);
        assertEquals(6595.551f, result, 0.01f);
    }

    @Test
    public void testPinnedManhattanDistance3() throws Exception {
        INDArray array1 = Nd4j.linspace(1, 10000, 98);
        INDArray array2 = Nd4j.linspace(1, 9000, 98);

        float result = Nd4j.getExecutioner().execAndReturn(new ManhattanDistance(array1, array2)).getFinalResult().floatValue();

        assertEquals(49000.004f, result, 0.001f);
    }
}
