package jcuda.jcublas.ops;

import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.accum.distances.ManhattanDistance;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.allocation.PinnedMemoryStrategy;
import org.nd4j.linalg.jcublas.context.ContextHolder;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class CudaReduce3Tests {

    @Test
    @Ignore
    public void testPinnedCosineSimilarity() throws Exception {
        // passthrough atm, no sense testing
    }

    @Test
    public void testPinnedManhattanDistance() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{0.0f, 1.0f, 2.0f, 3.0f, 4.0f});
        INDArray array2 = Nd4j.create(new float[]{0.5f, 1.5f, 2.5f, 3.5f, 4.5f});

        try {
            double result = Nd4j.getExecutioner().execAndReturn(new ManhattanDistance(array1, array2)).getFinalResult().doubleValue();

            System.out.println("Distance: " + result);
            assertEquals(2.5, result, 0.01);
        } catch (Exception e) {

        }

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);
    }

    @Test
    public void testPinnedEuclideanDistance() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{0.0f, 1.0f, 2.0f, 3.0f, 4.0f});
        INDArray array2 = Nd4j.create(new float[]{0.5f, 1.5f, 2.5f, 3.5f, 4.5f});

        double result = Nd4j.getExecutioner().execAndReturn(new EuclideanDistance(array1,array2)).getFinalResult().doubleValue();

        System.out.println("Distance: " + result);
        assertEquals(1.118033988749895, result, 0.01);
    }
}
