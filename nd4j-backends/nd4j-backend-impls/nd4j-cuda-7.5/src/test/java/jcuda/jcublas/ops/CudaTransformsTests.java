package jcuda.jcublas.ops;

import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.allocation.PinnedMemoryStrategy;
import org.nd4j.linalg.jcublas.context.ContextHolder;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
@Ignore
public class CudaTransformsTests {

    @Test
    public void testPinnedCosine() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});


        Nd4j.getExecutioner().exec(new Cos(array1, array2));

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(0.53f, array2.getFloat(0), 0.01);
    }

    @Test
    public void testPinnedAbs() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});


        Nd4j.getExecutioner().exec(new Abs(array1, array2));

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(1.01f, array2.getFloat(0), 0.01);
    }

    @Test
    public void testPinnedCeil() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});


        Nd4j.getExecutioner().exec(new Ceil(array1, array2));

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(2.0f, array2.getFloat(0), 0.01);

    }

    @Test
    public void testPinnedExp() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});


        Nd4j.getExecutioner().exec(new Exp(array1, array2));

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(2.75f, array2.getFloat(0), 0.01);
    }

    @Test
    public void testPinnedExp2() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{0.9f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});


        Nd4j.getExecutioner().exec(new Exp(array1));

        System.out.println("Array1: " + array1);

        assertEquals(2.45f, array1.getFloat(0), 0.01);
        assertEquals(1.0f, array1.getFloat(1), 0.01);
    }

    @Test
    public void testPinnedFloor() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});


        Nd4j.getExecutioner().exec(new Floor(array1, array2));

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(1.0f, array2.getFloat(0), 0.01);
    }

    @Test
    public void testPinnedLog() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

         INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});


        Nd4j.getExecutioner().exec(new Log(array1, array2));

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(0.01f, array2.getFloat(0), 0.01);
    }

    @Test
    public void testPinnedPow() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        Nd4j.getExecutioner().exec(new Pow(array1, 3));

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(1.03f, array1.getFloat(0), 0.01);
    }

    @Test
    public void testPinnedSetRange() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});


        Nd4j.getExecutioner().exec(new SetRange(array1, 0.1f, 1.0f));

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(0.1f, array1.getFloat(0), 0.01);
    }

    @Test
    public void testPinnedSigmoid() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});


        Nd4j.getExecutioner().exec(new Sigmoid(array1, array2));

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(0.73f, array2.getFloat(0), 0.01);
    }

    @Test
    public void testPinnedSign() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{-1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});


        Nd4j.getExecutioner().exec(new Sign(array1, array2));

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(1.0f, array2.getFloat(0), 0.01);
    }

    @Test
    public void testPinnedSin() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});


        Nd4j.getExecutioner().exec(new Sin(array1, array2));

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(0.85f, array2.getFloat(0), 0.01);
    }

    @Test
    public void testPinnedSoftplus() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});


        Nd4j.getExecutioner().exec(new SoftPlus(array1, array2));

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(1.32f, array2.getFloat(0), 0.01);
    }

    @Test
    public void testPinnedSqrt() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{2.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});


        Nd4j.getExecutioner().exec(new Sqrt(array1, array2));

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(1.42f, array2.getFloat(0), 0.01);
    }

    @Test
    public void testPinnedTanh() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});


        Nd4j.getExecutioner().exec(new Tanh(array1, array2));

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(0.77f, array2.getFloat(0), 0.01);
    }

    @Test
    public void testPinnedAcos() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{0.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});


        Nd4j.getExecutioner().exec(new ACos(array1, array2));

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(1.56f, array2.getFloat(0), 0.01);
    }

    @Test
    public void testPinnedASin() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{0.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});


        Nd4j.getExecutioner().exec(new ASin(array1, array2));

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(0.01f, array2.getFloat(0), 0.01);
    }

    @Test
    public void testPinnedATan() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});


        Nd4j.getExecutioner().exec(new ATan(array1, array2));

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(0.79f, array2.getFloat(0), 0.01);
    }

    @Test
    public void testPinnedNegative() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});


        Nd4j.getExecutioner().exec(new Negative(array1, array2));

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(-1.01f, array2.getFloat(0), 0.01);
    }

    @Test
    public void testPinnedCosineF() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f}).dup('f');
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});


        Nd4j.getExecutioner().exec(new Cos(array1, array2));

        System.out.println("Array1: " + array1);
        System.out.println("Array2: " + array2);

        assertEquals(0.53f, array2.getFloat(0), 0.01);
    }

     @Test
    public void testSoftmaxFC()  throws Exception {
        INDArray array1 = Nd4j.ones(2048).dup('f');
        INDArray array2 = Nd4j.zeros(2048);

        Nd4j.getExecutioner().exec(new SoftMax(array1));

        Nd4j.getExecutioner().exec(new SoftMax(array2));

        System.out.println("Array1: " + Arrays.toString(array1.data().asFloat()));
        System.out.println("Array2: " + Arrays.toString(array2.data().asFloat()));

        //assertEquals(array1, array2);

        assertEquals(1.0, array1.sumNumber().doubleValue(), 0.0001);
        assertEquals(1.0, array2.sumNumber().doubleValue(), 0.0001);
    }

    @Test
    public void testIsMaxEqualValues(){
        //Assumption here: should only have a 1 for *first* maximum value, if multiple values are exactly equal

        //[1 1 1] -> [1 0 0]
        //Loop to double check against any threading weirdness...
        for( int i=0; i<10; i++ ) {
            assertEquals(Nd4j.create(new double[]{1, 0, 0}), Nd4j.getExecutioner().execAndReturn(new IsMax(Nd4j.ones(3))));
        }

        //[0 0 0 2 2 0] -> [0 0 0 1 0 0]
        assertEquals(Nd4j.create(new double[]{0, 0, 0, 1, 0, 0}), Nd4j.getExecutioner().execAndReturn(new IsMax(Nd4j.create(new double[]{0, 0, 0, 2, 2, 0}))));

        //[0 2]    [0 1]
        //[2 1] -> [0 0]
        INDArray orig = Nd4j.create(new double[][]{{0, 2}, {2, 1}});
        INDArray exp = Nd4j.create(new double[][]{{0, 1}, {0, 0}});
        INDArray outc = Nd4j.getExecutioner().execAndReturn(new IsMax(orig.dup('c')));
        INDArray outf = Nd4j.getExecutioner().execAndReturn(new IsMax(orig.dup('f')));


        System.out.println("Exp: " + Arrays.toString(exp.data().asFloat()));
        System.out.println("OutC: " + Arrays.toString(outc.data().asFloat()));
        System.out.println("OutF: " + Arrays.toString(outf.data().asFloat()));
        System.out.println("outC shapeInfo: " + outc.shapeInfoDataBuffer());
        System.out.println("outF shapeInfo: " + outf.shapeInfoDataBuffer());
        assertEquals(exp, outc);
        assertEquals(exp, outf);
    }

    @Test
    public void testIsMaxMinimized() throws Exception {
        INDArray orig = Nd4j.create(new double[][]{{0, 2}, {2, 1}});
        INDArray outf = Nd4j.getExecutioner().execAndReturn(new IsMax(orig.dup('f')));
        INDArray exp = Nd4j.create(new double[][]{{0, 1}, {0, 0}});


        assertEquals(exp, outf);
    }

    @Test
    public void testSoftmaxSmall()  throws Exception {
        INDArray array1 = Nd4j.zeros(15);
        array1.putScalar(0, 0.9f);

        Nd4j.getExecutioner().exec(new SoftMax(array1));

        System.out.println("Array1: " + Arrays.toString(array1.data().asFloat()));

        assertEquals(1.0, array1.sumNumber().doubleValue(), 0.0001);
        assertEquals(0.14f, array1.getFloat(0), 0.01f);
     }

    @Test
    public void testClassificationSoftmax() {
        INDArray input = Nd4j.zeros(150, 3);
        input.putScalar(0, 0.9);
        input.putScalar(3, 0.2);
        input.putScalar(152, 0.9);
        input.putScalar(157, 0.11);
        input.putScalar(310, 0.9);
        input.putScalar(317, 0.1);

        System.out.println("Data:" + input.data().length());

        SoftMax softMax = new SoftMax(input);
        Nd4j.getExecutioner().exec(softMax);
        assertEquals(0.5515296f,input.getFloat(0), 0.01f);
        assertEquals(0.5515296f,input.getFloat(152), 0.01f);
        assertEquals(0.5515296f,input.getFloat(310), 0.01f);

    }
}
