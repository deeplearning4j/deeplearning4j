package org.nd4j.linalg.cpu.nativecpu;

import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.Eps;
import org.nd4j.linalg.cpu.nativecpu.CBLAS;
import org.nd4j.linalg.cpu.nativecpu.util.LibUtils;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

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
        double sum = Nd4j.getBlasWrapper().asum(linspace);
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
    public void testStdev() {
        INDArray arr = Nd4j.create(new float[]{0.9296161f, 0.31637555f, 0.1839188f}, new int[]{1, 3}, 'c');
        double stdev = arr.stdNumber().doubleValue();
        double stdev2 = arr.std(1).getDouble(0);
        assertEquals(stdev,stdev2,0.0);

        double exp = 0.397842772f;
        assertEquals(exp,stdev,1e-7f);
    }

    @Test
    public void testDup() {
        Nd4j.getRandom().setSeed(12345L);
        INDArray twoByEight = Nd4j.linspace(1,16,16).reshape(2,8);

        INDArray view = twoByEight.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2));
        int eleStride = view.elementWiseStride();
        INDArray viewCopy = view.dup();
        assertEquals(view,viewCopy);

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
        INDArray twoByThree = Nd4j.linspace(1, 600, 600).reshape(150, 4);
        INDArray columnVar = twoByThree.sum(0);
        INDArray assertion = Nd4j.create(new float[]{44850.0f, 45000.0f, 45150.0f, 45300.0f});
        assertEquals(assertion, columnVar);

    }



    @Test
    public void testLength() {
        INDArray values = Nd4j.create(2, 2);
        INDArray values2 = Nd4j.create(2, 2);

        values.put(0, 0, 0);
        values2.put(0, 0, 2);
        values.put(1, 0, 0);
        values2.put(1, 0, 2);
        values.put(0, 1, 0);
        values2.put(0, 1, 0);
        values.put(1, 1, 2);
        values2.put(1, 1, 2);


        INDArray expected = Nd4j.repeat(Nd4j.scalar(2), 2).reshape(2,1);

        Accumulation accum = Nd4j.getOpFactory().createAccum("euclidean", values, values2);
        INDArray results = Nd4j.getExecutioner().exec(accum, 1);
        assertEquals(expected, results);

    }


    @Test
    public void testBroadCasting() {
        INDArray first = Nd4j.arange(0, 3).reshape(3, 1);
        INDArray ret = first.broadcast(3, 4);
        INDArray testRet = Nd4j.create(new double[][]{
                {0, 0, 0, 0},
                {1, 1, 1, 1},
                {2, 2, 2, 2}
        });
        assertEquals(testRet, ret);
        INDArray r = Nd4j.arange(0, 4).reshape(1, 4);
        INDArray r2 = r.broadcast(4, 4);
        INDArray testR2 = Nd4j.create(new double[][]{
                {0, 1, 2, 3},
                {0, 1, 2, 3},
                {0, 1, 2, 3},
                {0, 1, 2, 3}
        });
        assertEquals(testR2, r2);

    }


    @Test
    public void testSortRows() {
        int nRows = 10;
        int nCols = 5;
        java.util.Random r = new java.util.Random(12345);

        for( int i=0; i < nCols; i++) {
            INDArray in = Nd4j.rand(new int[]{nRows,nCols});

            List<Integer> order = new ArrayList<>(nRows);
            //in.row(order(i)) should end up as out.row(i) - ascending
            //in.row(order(i)) should end up as out.row(nRows-j-1) - descending
            for( int j=0; j<nRows; j++ ) order.add(j);
            Collections.shuffle(order, r);
            for( int j = 0; j<nRows; j++ )
                in.putScalar(new int[]{j,i},order.get(j));

            INDArray outAsc = Nd4j.sortRows(in, i, true);
            INDArray outDesc = Nd4j.sortRows(in, i, false);

            for( int j = 0; j<nRows; j++ ){
                assertTrue(outAsc.getDouble(j,i)==j);
                int origRowIdxAsc = order.indexOf(j);
                assertTrue(outAsc.getRow(j).equals(in.getRow(origRowIdxAsc)));

                assertTrue(outDesc.getDouble(j,i)==(nRows-j-1));
                int origRowIdxDesc = order.indexOf(nRows-j-1);
                assertTrue(outDesc.getRow(j).equals(in.getRow(origRowIdxDesc)));
            }
        }
    }

    @Test
    public void testSortColumns() {
        int nRows = 5;
        int nCols = 10;
        java.util.Random r = new java.util.Random(12345);

        for( int i=0; i<nRows; i++ ){
            INDArray in = Nd4j.rand(new int[]{nRows,nCols});

            List<Integer> order = new ArrayList<>(nRows);
            for( int j=0; j<nCols; j++ ) order.add(j);
            Collections.shuffle(order, r);
            for( int j=0; j<nCols; j++ ) in.putScalar(new int[]{i,j},order.get(j));

            INDArray outAsc = Nd4j.sortColumns(in, i, true);
            INDArray outDesc = Nd4j.sortColumns(in, i, false);

            for( int j = 0; j < nCols; j++ ){
                assertTrue(outAsc.getDouble(i,j)==j);
                int origColIdxAsc = order.indexOf(j);
                assertTrue(outAsc.getColumn(j).equals(in.getColumn(origColIdxAsc)));

                assertTrue(outDesc.getDouble(i,j)==(nCols-j-1));
                int origColIdxDesc = order.indexOf(nCols-j-1);
                assertTrue(outDesc.getColumn(j).equals(in.getColumn(origColIdxDesc)));
            }
        }
    }



    @Test
    public void testEps() {
        INDArray ones = Nd4j.ones(5);
        double sum = Nd4j.getExecutioner().exec(new Eps(ones, ones, ones, ones.length())).z().sumNumber().doubleValue();
        assertEquals(5, sum, 1e-1);
    }

    @Test
    public void testLogDouble() {
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
        Nd4j.dtype = DataBuffer.Type.DOUBLE;
        Nd4j.getRandom().setSeed(12345L);
        INDArray elevenByTwo = Nd4j.linspace(1,22,22).reshape(11,2);
        INDArray twoByEight = Nd4j.linspace(1,16,16).reshape(2,8);

        INDArray view = twoByEight.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2));
        INDArray assertion = Nd4j.create(new double[]{
                19.0,22.0,39.0,46.0,59.0,70.0,79.0,94.0,99.0,118.0,119.0,142.0,139.0,166.0,159.0,190.0,179.0,214.0,199.0,238.0,219.0,262.0,
        },new int[]{11,2});

        INDArray viewCopy = view.dup();
        assertEquals(view,viewCopy);

        INDArray mmul1 = elevenByTwo.mmul(view);
        INDArray mmul2 = elevenByTwo.mmul(viewCopy);
        assertEquals(assertion,mmul1);
        assertEquals(assertion,mmul2);
        assertTrue(mmul1.equals(mmul2));
    }

}
