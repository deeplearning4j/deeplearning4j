package jcuda.jcublas.ops;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * @author raver119@gmail.com
 */
@Ignore
public class CudaPairwiseTrainformsTests {

    @Before
    public void setUp() {
        CudaEnvironment.getInstance().getConfiguration()
                .setExecutionModel(Configuration.ExecutionModel.SEQUENTIAL)
                .setFirstMemory(AllocationStatus.DEVICE)
                .setMaximumBlockSize(256)
                .setMaximumGridSize(64)
                .enableDebug(true);

        System.out.println("Init called");
    }


    @Test
    public void testPinnedAddiRowVector() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f});
        INDArray array2 = Nd4j.create(new float[]{2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f});

        array1.addiRowVector(array2);

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(3.5f, array1.getRow(0).getFloat(0), 0.01);
    }

    @Test
    public void testPinnedDiviRowVector() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f});
        INDArray array2 = Nd4j.create(new float[]{2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f});

        INDArray result = array1.diviRowVector(array2);

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);
        System.out.println("Result: " + result);

        assertEquals(0.75f, array1.getRow(0).getFloat(0), 0.01);
    }

    @Test
    public void testPinnedRDiviRowVector() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f});
        INDArray array2 = Nd4j.create(new float[]{2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f});

        array1.rdiviRowVector(array2);

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(1.33f, array1.getRow(0).getFloat(0), 0.01);
    }

    @Test
    public void testPinnedSubiRowVector() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f});
        INDArray array2 = Nd4j.create(new float[]{2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f});

        array1.subiRowVector(array2);

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(-0.5f, array1.getRow(0).getFloat(0), 0.01);
    }

    @Test
    public void testPinnedRSubiRowVector() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f});
        INDArray array2 = Nd4j.create(new float[]{2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f});

        array1.rsubiRowVector(array2);

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(0.5f, array1.getRow(0).getFloat(0), 0.01);
    }

    @Test
    public void testPinnedMuliRowVector() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f});
        INDArray array2 = Nd4j.create(new float[]{2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f});

        array1.muliRowVector(array2);

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(3.0f, array1.getRow(0).getFloat(0), 0.01);
    }

    @Test
    public void testPinnedMuliColumnVector1() throws Exception {
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.linspace(1, 1280, 128000).reshape(128, 1000);
        INDArray array2 = Nd4j.linspace(1, 1280, 128000).reshape(128, 1000).dup('f');

        array1.muli(array2);

     //   System.out.println("Array1: " + array1);
       // System.out.println("Array2: " + array2);
    }

    @Test
    public void testPinnedMuliColumnVector2() throws Exception {
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.linspace(1, 1280, 128000).reshape(128, 1000);
        INDArray array2 = Nd4j.linspace(1, 1280, 128000).reshape(128, 1000);

        INDArray arrayTest = array1.dup('f');

        assertEquals(array1, arrayTest);

        array1.muli(array2);



    //    System.out.println("Array1: " + array1);

       assertNotEquals(array1, arrayTest);
        //System.out.println("Array2: " + array2);
    }

    @Test
    public void testFOrdering1() throws Exception {
        INDArray array1 = Nd4j.linspace(1, 1280, 128000).reshape(128, 1000);
        INDArray array2 = Nd4j.linspace(1, 1280, 128000).reshape(128, 1000);

        INDArray array3 = Nd4j.linspace(1, 1280, 128000).reshape(128, 1000);
        INDArray array4 = Nd4j.linspace(1, 1280, 128000).reshape(128, 1000).dup('f');

        System.out.println("a4 stride: " + array4.elementWiseStride());

        long time1 = System.currentTimeMillis();
        array1.muli(array2);
        long time2 = System.currentTimeMillis();

        System.out.println("Execution time 1: " + (time2 - time1));

        time1 = System.currentTimeMillis();
        array3.muli(array4);
        time2 = System.currentTimeMillis();
        System.out.println("Execution time 2: " + (time2 - time1));

        assertEquals(array3, array1);
    }

    @Test
    public void testFOrdering2() throws Exception {
        INDArray array1 = Nd4j.linspace(1, 1280, 128000);
        INDArray array2 = Nd4j.linspace(1, 1280, 128000);

        INDArray array3 = Nd4j.linspace(1, 1280, 128000);
        INDArray array4 = Nd4j.linspace(1, 1280, 128000).dup('f');

        array1.muli(array2);

        array3.muli(array4);

        assertEquals(array3, array1);
    }

    @Test
    public void testAssign(){
        int[] shape1 = {3,2,2,2,2,2};
        int[] shape2 = {12,8};
        int length = ArrayUtil.prod(shape1);

        assertEquals(ArrayUtil.prod(shape1),ArrayUtil.prod(shape2));

        INDArray arr = Nd4j.linspace(1,length,length).reshape('c',shape1);
        INDArray arr2c = Nd4j.create(shape2,'c');
        INDArray arr2f = Nd4j.create(shape2,'f');

        arr2c.assign(arr);
        System.out.println("--------------");
        arr2f.assign(arr);

        INDArray exp = Nd4j.linspace(1,length,length).reshape('c',shape2);

        assertEquals(exp,arr2c);
        assertEquals(exp,arr2f);
    }
}
