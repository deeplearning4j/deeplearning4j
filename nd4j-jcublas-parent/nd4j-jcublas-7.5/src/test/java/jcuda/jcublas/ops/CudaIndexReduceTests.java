package jcuda.jcublas.ops;

import org.junit.Test;
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
public class CudaIndexReduceTests {

    @Test
    public void testPinnedIMax() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        // reset to default MemoryStrategy, most probable is Pinned
        ContextHolder.getInstance().forceMemoryStrategyForThread(new PinnedMemoryStrategy());

        assertEquals("PinnedMemoryStrategy", ContextHolder.getInstance().getMemoryStrategy().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.0f, 0.1f, 2.0f, 3.0f, 4.0f, 5.0f});



        int idx =  ((IndexAccumulation) Nd4j.getExecutioner().exec(new IMax(array1))).getFinalResult();
        System.out.println("Call happened");

        assertEquals(5, idx);
    }

    @Test
    public void testPinnedIMax2() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        // reset to default MemoryStrategy, most probable is Pinned
        ContextHolder.getInstance().forceMemoryStrategyForThread(new PinnedMemoryStrategy());

        assertEquals("PinnedMemoryStrategy", ContextHolder.getInstance().getMemoryStrategy().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{6.0f, 0.1f, 2.0f, 3.0f, 4.0f, 5.0f});



        int idx =  ((IndexAccumulation) Nd4j.getExecutioner().exec(new IMax(array1))).getFinalResult();

        assertEquals(0, idx);
    }

    @Test
    public void testPinnedIMin() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        // reset to default MemoryStrategy, most probable is Pinned
        ContextHolder.getInstance().forceMemoryStrategyForThread(new PinnedMemoryStrategy());

        assertEquals("PinnedMemoryStrategy", ContextHolder.getInstance().getMemoryStrategy().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.0f, 0.1f, 2.0f, 3.0f, 4.0f, 5.0f});



        int idx =  ((IndexAccumulation) Nd4j.getExecutioner().exec(new IMin(array1))).getFinalResult();

        assertEquals(1, idx);
    }

    @Test
    public void testPinnedIMin2() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        // reset to default MemoryStrategy, most probable is Pinned
        ContextHolder.getInstance().forceMemoryStrategyForThread(new PinnedMemoryStrategy());

        assertEquals("PinnedMemoryStrategy", ContextHolder.getInstance().getMemoryStrategy().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{0.1f, 1.1f, 2.0f, 3.0f, 4.0f, 5.0f});



        int idx =  ((IndexAccumulation) Nd4j.getExecutioner().exec(new IMin(array1))).getFinalResult();

        assertEquals(0, idx);
    }
}
