package org.nd4j.linalg.cpu.nativecpu;

import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.Eps;
import org.nd4j.linalg.cpu.nativecpu.CBLAS;
import org.nd4j.linalg.cpu.nativecpu.util.LibUtils;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;

import static org.junit.Assert.*;
/**
 * @author Adam Gibson
 */
public class LoopTest {
    static {
        LibUtils.loadLibrary("libnd4j");
    }
    @Test
    public void testLoop() {
        INDArray linspace = Nd4j.linspace(1,4,4);
        float sum = CBLAS.sasum(4,linspace.data().asNioFloat(),1);
        assertEquals(10,sum,1e-1);

    }

    @Test
    public void testPutSlice() {
        INDArray n = Nd4j.linspace(1,27,27).reshape(3, 3, 3);
        INDArray newSlice = Nd4j.zeros(3, 3);
        n.putSlice(0, newSlice);
        assertEquals(newSlice, n.slice(0));

        INDArray firstDimensionAs1 = newSlice.reshape(1, 3, 3);
        n.putSlice(0, firstDimensionAs1);


    }

    @Test
    public void testMultiDimSum() {
        double[] data = new double[]{22.,  26.,  30};
        INDArray assertion = Nd4j.create(data);
        for(int i = 0; i < data.length; i++) {
            assertEquals(data[i],assertion.getDouble(i),1e-1);
        }

        INDArray twoTwoByThree = Nd4j.linspace(1,12,12).reshape(2, 2, 3);
        INDArray multiSum = twoTwoByThree.sum(0, 1);
        assertEquals(assertion,multiSum);

    }

    @Test
    public void testArrCreationShape() {
        INDArray arr = Nd4j.linspace(1,4,4).reshape(2,2);
        for(int i = 0; i < 2; i++)
            assertEquals(2,arr.size(i));
        int[] stride = ArrayUtil.calcStrides(new int[]{2, 2});
        for(int i = 0; i < stride.length; i++) {
            assertEquals(stride[i],arr.stride(i));
        }

        assertArrayEquals(new int[]{2,2},arr.shape());
        assertArrayEquals(new int[]{2,1},arr.stride());
    }

    @Test
    public void testColumnSumDouble() {
        Nd4j.dtype = DataBuffer.Type.DOUBLE;
        INDArray twoByThree = Nd4j.linspace(1, 600, 600).reshape(150, 4);
        INDArray columnVar = twoByThree.sum(0);
        INDArray assertion = Nd4j.create(new float[]{44850.0f, 45000.0f, 45150.0f, 45300.0f});
        assertEquals(assertion, columnVar);

    }

    @Test
    public void testEps() {
        INDArray ones = Nd4j.ones(5);
        double sum = Nd4j.getExecutioner().exec(new Eps(ones, ones, ones, ones.length())).z().sumNumber().doubleValue();
        assertEquals(5, sum, 1e-1);
    }

    @Test
    public void testLogDouble() {
        Nd4j.dtype = DataBuffer.Type.DOUBLE;
        INDArray linspace = Nd4j.linspace(1, 6, 6);
        INDArray log = Transforms.log(linspace);
        INDArray assertion = Nd4j.create(new double[]{0, 0.6931471805599453, 1.0986122886681098, 1.3862943611198906, 1.6094379124341005, 1.791759469228055});
        assertEquals(assertion, log);
    }


    @Test
    public void testTile() {
        INDArray x = Nd4j.linspace(1,4,4).reshape(2, 2);
        INDArray repeated = x.repeat(new int[]{2});
        assertEquals(8,repeated.length());
        INDArray repeatAlongDimension = x.repeat(1,new int[]{2});
        INDArray assertionRepeat = Nd4j.create(new double[][]{
                {1, 1, 2, 2},
                {3, 3, 4, 4}
        });
        assertArrayEquals(new int[]{2,4},assertionRepeat.shape());
        assertEquals(assertionRepeat,repeatAlongDimension);
        System.out.println(repeatAlongDimension);
        INDArray ret = Nd4j.create(new double[]{0, 1, 2});
        INDArray tile = Nd4j.tile(ret, 2, 2);
        INDArray assertion = Nd4j.create(new double[][]{
                {0, 1, 2, 0, 1, 2}
                , {0, 1, 2, 0, 1, 2}
        });
        assertEquals(assertion,tile);
    }

    @Test
    public void testTensorDot() {
        INDArray oneThroughSixty = Nd4j.arange(60).reshape(3, 4, 5);
        INDArray oneThroughTwentyFour = Nd4j.arange(24).reshape(4, 3, 2);
        INDArray result = Nd4j.tensorMmul(oneThroughSixty, oneThroughTwentyFour, new int[][]{{1, 0}, {0, 1}});
        assertArrayEquals(new int[]{5, 2}, result.shape());
        INDArray assertion = Nd4j.create(new double[][]{
                {   4400 ,  4730},
                {  4532 ,  4874},
                {  4664  , 5018},
                {  4796 ,  5162},
                {  4928 , 5306}
        });
        assertEquals(assertion, result);

        INDArray w = Nd4j.valueArrayOf(new int[]{2, 1, 2, 2}, 0.5);
        INDArray col = Nd4j.create(new double[]{
                1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3,
                3, 1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4,
                4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4
        }, new int[]{1, 1, 2, 2, 4, 4});

        INDArray test = Nd4j.tensorMmul(col, w, new int[][]{{1, 2, 3}, {1, 2, 3}});
        INDArray assertion2 = Nd4j.create(new double[]{3., 3., 3., 3., 3., 3., 3., 3., 7., 7., 7., 7., 7., 7., 7., 7., 3., 3.
                , 3., 3., 3., 3., 3., 3., 7., 7., 7., 7., 7., 7., 7., 7.}, new int[]{1, 4, 4, 2}, new int[]{16, 8, 2, 1}, 0, 'f');
        assertion2.setOrder('f');
        assertEquals(assertion2,test);
    }

    @Test
    public void testMmulGet(){
        Nd4j.getRandom().setSeed(12345L);
        INDArray elevenByTwo = Nd4j.rand(new int[]{11, 2});
        INDArray twoByEight = Nd4j.rand(new int[]{2, 8});

        INDArray view = twoByEight.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2));
        INDArray viewCopy = view.dup();
        assertEquals(view,viewCopy);

        INDArray mmul1 = elevenByTwo.mmul(view);
        INDArray mmul2 = elevenByTwo.mmul(viewCopy);

        assertTrue(mmul1.equals(mmul2));
    }

}
