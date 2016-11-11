package org.nd4j.linalg.cpu.nativecpu.ops;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.api.ops.impl.accum.distances.CosineSimilarity;
import org.nd4j.linalg.api.ops.impl.accum.distances.ManhattanDistance;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMin;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMin;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarAdd;
import org.nd4j.linalg.api.ops.impl.transforms.Exp;
import org.nd4j.linalg.api.ops.impl.transforms.IsMax;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMaxDerivative;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Properties;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@Ignore
public class NativeOpExecutionerTest {
    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void execBroadcastOp() throws Exception {
        INDArray array = Nd4j.ones(1024, 1024);
        INDArray arrayRow = Nd4j.linspace(1, 1024, 1024);

        float sum = (float) array.sumNumber().doubleValue();

        array.addiRowVector(arrayRow);

        long time1 = System.nanoTime();
        for (int x = 0; x < 100000; x++) {
            array.addiRowVector(arrayRow);
        }
        long time2 = System.nanoTime();
/*
        time1 = System.nanoTime();
        array.addiRowVector(arrayRow);
        time2 = System.nanoTime();
*/
        System.out.println("Execution time: " + ((time2 - time1) / 100000) );

        assertEquals(1002, array.getFloat(0), 0.1f);
        assertEquals(2003, array.getFloat(1), 0.1f);
    }

    @Test
    public void execReduceOp1() throws Exception {
        INDArray array = Nd4j.ones(1024, 1024);
        INDArray arrayRow1 = Nd4j.linspace(1, 1024, 1024);
        INDArray arrayRow2 = Nd4j.linspace(0, 1023, 1024);

        float sum = (float) array.sumNumber().doubleValue();
        long time1 = System.nanoTime();
        array.sum(0);
        long time2 = System.nanoTime();

        System.out.println("Execution time: " + (time2 - time1));
        System.out.println("Result: " + sum);
    }

    @Test
    public void execReduceOp2() throws Exception {
        INDArray array = Nd4j.ones(3, 1024);
        INDArray arrayRow = Nd4j.linspace(1, 1024, 1024);

        float sum = array.sumNumber().floatValue();
        long time1 = System.nanoTime();
        sum = array.sumNumber().floatValue();
        long time2 = System.nanoTime();

        System.out.println("Execution time: " + (time2 - time1));
    }

    @Test
    public void execTransformOp1() throws Exception {
        INDArray array1 = Nd4j.linspace(1, 20480, 20480);
        INDArray array2 = Nd4j.linspace(1, 20480, 20480);

        Nd4j.getExecutioner().exec(new Exp(array1, array2));

        long time1 = System.nanoTime();
        for (int x = 0; x < 10000; x++) {
            Nd4j.getExecutioner().exec(new Exp(array1, array2));
        }
        long time2 = System.nanoTime();

        System.out.println("Execution time: " + ((time2 - time1) / 10000));

       // System.out.println("Array1: " + array1);
       // System.out.println("Array2: " + array2);

        assertEquals(2.71f, array2.getFloat(0), 0.01);
    }

    @Test
    public void execPairwiseOp1() throws Exception {
        INDArray array1 = Nd4j.linspace(1, 20480, 20480);
        INDArray array2 = Nd4j.linspace(1, 20480, 20480);

        array1.sumNumber();
        array2.sumNumber();

        long time1 = System.nanoTime();
        for (int x = 0; x < 10000; x++) {
            array1.addiRowVector(array2);
        }
        long time2 = System.nanoTime();

        System.out.println("Execution time: " + ((time2 - time1) / 10000));

        // System.out.println("Array1: " + array1);
        // System.out.println("Array2: " + array2);

        assertEquals(10001f, array1.getFloat(0), 0.01);
    }


    @Test
    public void testScalarOp1() throws Exception {
        // simple way to stop test if we're not on CUDA backend here

        INDArray array1 = Nd4j.linspace(1, 20480, 20480);
        INDArray array2 = Nd4j.linspace(1, 20480, 20480);
        array2.addi(0.5f);

        long time1 = System.nanoTime();
        for (int x = 0; x < 10000; x++) {
            array2.addi(0.5f);
        }
        long time2 = System.nanoTime();

        System.out.println("Execution time: " + ((time2 - time1) / 10000));

        System.out.println("Divi result: " + array2.getFloat(0));
        assertEquals(5001.5, array2.getFloat(0), 0.01f);

    }

    @Test
    public void testSoftmax1D_1() throws Exception {
        INDArray input1T = Nd4j.create(new double[]{ -0.75, 0.58, 0.42, 1.03, -0.61, 0.19, -0.37, -0.40, -1.42, -0.04});
        INDArray input1 = Nd4j.create(new double[]{ -0.75, 0.58, 0.42, 1.03, -0.61, 0.19, -0.37, -0.40, -1.42, -0.04});
        INDArray input2 = Nd4j.zerosLike(input1);
        Nd4j.copy(input1, input2);
        INDArray output1 = Nd4j.create(1, 10);
        INDArray output1T = Nd4j.create(1, 10);

        System.out.println("FA --------------------");
        Nd4j.getExecutioner().exec(new SoftMax(input1, output1));
        Nd4j.getExecutioner().exec(new SoftMax(input1T, output1T));
        System.out.println("FB --------------------");

        System.out.println("Softmax = " + output1);
        INDArray output2 = Nd4j.create(1,10);
        Nd4j.getExecutioner().exec(new SoftMaxDerivative(input2, output2));
        System.out.println("Softmax Derivative = " + output2);

        INDArray assertion1 = Nd4j.create(new double[]{0.04, 0.16, 0.14, 0.26, 0.05, 0.11, 0.06, 0.06, 0.02, 0.09});

        assertArrayEquals(assertion1.data().asFloat(), output1.data().asFloat(), 0.01f);
        assertArrayEquals(assertion1.data().asFloat(), output1T.data().asFloat(), 0.01f);

    }

    @Test
    public void testNd4jDup() {
    /* set the dType */
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);

    /* create NDArray from a double[][] */
        int cnt = 0;
        double data[][] = new double[50][50];
        for (int x = 0; x < 50; x++) {
            for (int y = 0; y< 50; y++) {
                data[x][y] = cnt;
                cnt++;
            }
        }
        INDArray testNDArray = Nd4j.create(data);

    /* print the first row */
        System.out.println("A: " + testNDArray.getRow(0));

    /* set the dType again! */
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);

    /* print the first row */
        System.out.println("B: " + testNDArray.getRow(0));

    /* print the first row dup -- it should be different now! */
        System.out.println("C: " + testNDArray.getRow(0).dup());
    }

    @Test
    public void testPinnedManhattanDistance2() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        INDArray array1 = Nd4j.linspace(1, 1000, 1000);
        INDArray array2 = Nd4j.linspace(1, 900, 1000);

        double result = Nd4j.getExecutioner().execAndReturn(new ManhattanDistance(array1, array2)).getFinalResult().doubleValue();

        assertEquals(50000.0, result, 0.001f);
    }

    @Test
    public void testPinnedCosineSimilarity2() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        INDArray array1 = Nd4j.linspace(1, 1000, 1000);
        INDArray array2 = Nd4j.linspace(100, 200, 1000);

        double result = Nd4j.getExecutioner().execAndReturn(new CosineSimilarity(array1, array2)).getFinalResult().doubleValue();

        assertEquals(0.945f, result, 0.001f);
    }

    @Test
    public void testArgMax1() {
        INDArray array1 = Nd4j.create(new float[]{-1.0f, 2.0f});
        INDArray array2 = Nd4j.create(new float[]{2.0f});

        INDArray res = Nd4j.argMax(array1);
        System.out.println("Res length: " + res.length());
        assertEquals(1.0f, res.getFloat(0), 0.01f);

        System.out.println("--------------------");

        res = Nd4j.argMax(array2);
        System.out.println("Res length: " + res.length());
        assertEquals(0.0f, res.getFloat(0), 0.01f);
    }

    @Test
    public void testBroadcastWithPermute(){
        Nd4j.getRandom().setSeed(12345);
        int length = 4*4*5*2;
        INDArray arr = Nd4j.linspace(1,length,length).reshape('c',4,4,5,2).permute(2,3,1,0);
//        INDArray arr = Nd4j.linspace(1,length,length).reshape('f',4,4,5,2).permute(2,3,1,0);
        INDArray arrDup = arr.dup('c');

        INDArray row = Nd4j.rand(1,2);
        assertEquals(row.length(), arr.size(1));
        assertEquals(row.length(), arrDup.size(1));

        assertEquals(arr,arrDup);



        INDArray first =  Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(arr,    row, Nd4j.createUninitialized(arr.shape(), 'c'), 1));
        INDArray second = Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(arrDup, row, Nd4j.createUninitialized(arr.shape(), 'c'), 1));

        System.out.println("A1: " + Arrays.toString(arr.shapeInfoDataBuffer().asInt()));
        System.out.println("A2: " + Arrays.toString(first.shapeInfoDataBuffer().asInt()));
        System.out.println("B1: " + Arrays.toString(arrDup.shapeInfoDataBuffer().asInt()));
        System.out.println("B2: " + Arrays.toString(second.shapeInfoDataBuffer().asInt()));

        INDArray resultSameStrides = Nd4j.zeros(new int[]{4,4,5,2},'c').permute(2,3,1,0);
        assertArrayEquals(arr.stride(), resultSameStrides.stride());
     //   INDArray third = Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(arr, row, resultSameStrides, 1));

       // assertEquals(second, third);    //Original and result w/ same strides: passes
       // assertEquals(first,second);     //Original and result w/ different strides: fails
    }

    @Test
    public void testBroadcastEquality1() {
        INDArray array = Nd4j.zeros(new int[]{4, 5}, 'f');
        INDArray array2 = Nd4j.zeros(new int[]{4, 5}, 'f');
        INDArray row = Nd4j.create(new float[]{1, 2, 3, 4, 5});

        array.addiRowVector(row);

        System.out.println(array);

        System.out.println("-------");

        ScalarAdd add = new ScalarAdd(array2, row, array2, array2.length(), 0.0f);
        add.setDimension(0);
        Nd4j.getExecutioner().exec(add);

        System.out.println(array2);
        assertEquals(array, array2);
    }

    @Test
    public void testBroadcastEquality2() {
        INDArray array = Nd4j.zeros(new int[]{4, 5}, 'c');
        INDArray array2 = Nd4j.zeros(new int[]{4, 5}, 'c');
        INDArray column = Nd4j.create(new float[]{1, 2, 3, 4}).reshape(4,1);

        array.addiColumnVector(column);

        System.out.println(array);

        System.out.println("-------");

        ScalarAdd add = new ScalarAdd(array2, column, array2, array2.length(), 0.0f);
        add.setDimension(1);
        Nd4j.getExecutioner().exec(add);

        System.out.println(array2);
        assertEquals(array, array2);

    }


    @Test
    public void testIsMaxC1() throws Exception {
        INDArray array = Nd4j.zeros(new int[]{4, 5}, 'c');

        Nd4j.getExecutioner().exec(new IsMax(array, 1));
    }

    @Test
    public void testIsMaxC2() throws Exception {
        INDArray array = Nd4j.zeros(new int[]{4, 5}, 'c');

        Nd4j.getExecutioner().exec(new IsMax(array, 0));
    }

    @Test
    public void testIsMaxF1() throws Exception {
        INDArray array = Nd4j.zeros(new int[]{4, 5}, 'f');

        Nd4j.getExecutioner().exec(new IsMax(array, 1));
    }

    @Test
    public void testEnvironment() throws Exception {
        INDArray array = Nd4j.zeros(new int[]{4, 5}, 'f');
        Properties properties = Nd4j.getExecutioner().getEnvironmentInformation();

        System.out.println("Props: " + properties.toString());
    }


    @Test
    public void testIsView() {
        INDArray array = Nd4j.zeros(100, 100);

        assertFalse(array.isView());
    }

    @Test
    public void testIMaxIAMax(){
        INDArray arr = Nd4j.create(new double[]{-0.24, -0.26, -0.07, -0.01});

        double imax = Nd4j.getExecutioner().execAndReturn(new IMax(arr.dup())).getFinalResult();
        double iamax = Nd4j.getExecutioner().execAndReturn(new IAMax(arr.dup())).getFinalResult();
        System.out.println("IMAX: " + imax);
        System.out.println("IAMAX: " + iamax);

        assertEquals(3, imax, 0.0);
        assertEquals(1, iamax, 0.0);
    }


    @Test
    public void testIMinIAMin(){
        INDArray arr = Nd4j.create(new double[]{-0.24, -0.26, -0.07, -0.01});

        double imin = Nd4j.getExecutioner().execAndReturn(new IMin(arr.dup())).getFinalResult();
        double iamin = Nd4j.getExecutioner().execAndReturn(new IAMin(arr.dup())).getFinalResult();
        System.out.println("IMin: " + imin);
        System.out.println("IAMin: " + iamin);

        assertEquals(1, imin, 0.0);
        assertEquals(3, iamin, 0.0);
    }

    @Test
    public void testViewData1() {
        INDArray in = Nd4j.create(new double[][]{
                {1, 2},
                {3, 4}}, 'c');

        System.out.println("Input data: " + Arrays.toString(in.data().asDouble()));

        INDArray out = in.getRow(1);
        System.out.println("Out:        " + out);
        System.out.println("Out data:   " + Arrays.toString(out.data().asFloat()));


        assertTrue(out.isView());
        assertEquals(2, out.data().length());

        out.addi(2f);

        System.out.println("Out data:   " + Arrays.toString(out.data().asFloat()));
        System.out.println("Input data: " + Arrays.toString(in.data().asDouble()));
    }

    @Test
    public void testViewData2() {
        INDArray in = Nd4j.create(new double[][]{
                {1, 2},
                {3, 4}}, 'f');

        System.out.println("Input data: " + Arrays.toString(in.data().asDouble()));

        INDArray out = in.getRow(1);
        System.out.println("Out:        " + out);
        System.out.println("Out data:   " + Arrays.toString(out.data().asFloat()));


        assertTrue(out.isView());
        assertEquals(2, out.data().length());

        out.addi(2f);

        System.out.println("Out data:   " + Arrays.toString(out.data().asFloat()));
        System.out.println("Input data: " + Arrays.toString(in.data().asDouble()));
    }

    @Test
    public void testMmulC1() throws Exception {
        INDArray A = Nd4j.linspace(0, 11, 12).reshape('c', 4, 3);
        INDArray B = Nd4j.linspace(0, 11, 12).reshape('c', 3, 4);

        System.out.println("A: \n" + A);

        INDArray C = A.mmul(B);

        INDArray expC = Nd4j.create(new double[]{20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0, 92.0, 113.0, 134.0, 155.0, 128.0, 158.0, 188.0, 218.0}).reshape(4, 4);

        assertEquals(expC, C);
    }

    @Test
    public void testMmulF1() throws Exception {
        INDArray A = Nd4j.linspace(0, 11, 12).reshape('f', 4, 3);
        INDArray B = Nd4j.linspace(0, 11, 12).reshape('f', 3, 4);

        System.out.println("A: \n" + A);

        INDArray C = A.mmul(B);

        System.out.println("C: \n" + Arrays.toString(C.data().asFloat()));

        INDArray expF = Nd4j.create(new double[]{20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0, 92.0, 113.0, 134.0, 155.0, 128.0, 158.0, 188.0, 218.0}).reshape('f', 4, 4);

        assertEquals(expF, C);
    }
}