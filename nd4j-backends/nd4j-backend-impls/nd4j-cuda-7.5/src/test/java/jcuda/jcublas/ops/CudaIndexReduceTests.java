package jcuda.jcublas.ops;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.IndexAccumulation;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMin;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
@Ignore
public class CudaIndexReduceTests {

    @Before
    public void setUp() {
        CudaEnvironment.getInstance().getConfiguration()
                .setExecutionModel(Configuration.ExecutionModel.SEQUENTIAL)
                .setFirstMemory(AllocationStatus.DEVICE)
                .setMaximumBlockSize(64)
                .setMaximumGridSize(64)
                .enableDebug(true);

        System.out.println("Init called");
    }

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
    public void testPinnedIMax4() throws Exception {
        // simple way to stop test if we're not on CUDA backend here


        INDArray array1 = Nd4j.create(new float[]{0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 0.0f});

        int idx =  ((IndexAccumulation) Nd4j.getExecutioner().exec(new IMax(array1))).getFinalResult();

        System.out.println("Array1: " + array1);

        assertEquals(3, idx);
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
    public void testIMaxLargeLarge() throws Exception {
        INDArray array1 = Nd4j.linspace(1, 1000, 12800);

        int idx =  ((IndexAccumulation) Nd4j.getExecutioner().exec(new IMax(array1))).getFinalResult();

        assertEquals(12799, idx);
    }

    @Test
    public void testIamaxC() {
        INDArray linspace = Nd4j.linspace(1, 4, 4).dup('c');
        assertEquals(3,Nd4j.getBlasWrapper().iamax(linspace));
    }

    @Test
    public void testIamaxF() {
        INDArray linspace = Nd4j.linspace(1, 4, 4).dup('f');
        assertEquals(3,Nd4j.getBlasWrapper().iamax(linspace));
    }

    @Test
    public void testIMax2() {
        INDArray array1 = Nd4j.linspace(1, 1000, 128000).reshape(128, 1000);

        long time1 = System.currentTimeMillis();
        INDArray  argMax = Nd4j.argMax(array1, 1);
        long time2 = System.currentTimeMillis();

        System.out.println("Execution time: " + (time2 - time1));
        for (int i = 0; i < 128; i++) {
            assertEquals(999f, argMax.getFloat(i), 0.0001f);
        }
    }

    @Test
    public void testIMax3() {
        INDArray array1 = Nd4j.linspace(1, 1000, 128000).reshape(128, 1000);

        INDArray  argMax = Nd4j.argMax(array1, 0);


        System.out.println("ARgmax length: " + argMax.length());
        for (int i = 0; i < 1000; i++) {
            assertEquals("Failed iteration: ["+ i +"]", 127, argMax.getFloat(i), 0.0001f);
        }
    }

    @Test
    public void testIMax4() {
        INDArray array1 = Nd4j.linspace(1, 1000, 128000).reshape(128, 1000);

        long time1 = System.currentTimeMillis();
        INDArray  argMax = Nd4j.argMax(array1, 0,1);
        long time2 = System.currentTimeMillis();

        System.out.println("Execution time: " + (time2 - time1));

        assertEquals(127999f, argMax.getFloat(0), 0.001f);
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
