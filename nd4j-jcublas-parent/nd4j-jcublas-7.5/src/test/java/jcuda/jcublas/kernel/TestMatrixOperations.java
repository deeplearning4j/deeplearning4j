/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package jcuda.jcublas.kernel;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.io.DataInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.lang3.time.StopWatch;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.BroadcastOp;
import org.nd4j.linalg.api.ops.IndexAccumulation;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.broadcast.*;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMin;
import org.nd4j.linalg.api.ops.impl.transforms.Log;
import org.nd4j.linalg.api.ops.impl.transforms.LogSoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.Eps;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.executors.ExecutorServiceProvider;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;

import static org.junit.Assert.*;

public class TestMatrixOperations {


    @Test
    public void testDot() {
        INDArray four = Nd4j.linspace(1,4,4);
        double dot = Nd4j.getBlasWrapper().dot(four,four);
        assertEquals(30,dot,1e-1);
    }



    @Test
    public void testSums() {
        INDArray a = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray tad = a.tensorAlongDimension(1,0);
        INDArray tadOne = a.tensorAlongDimension(1,1);
        int ele = tad.elementWiseStride();
        int otherEle = tadOne.elementWiseStride();
        //assertEquals(Nd4j.create(new float[]{4, 6}), a.sum(0));
        assertEquals(Nd4j.create(new float[]{3, 7}), a.sum(1));
        assertEquals( 10, a.sumNumber().doubleValue(), 1e-1);


    }


    @Test
    public void testMeans() {
        INDArray a = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray mean1 = a.mean(1);
        assertEquals(Nd4j.create(new double[]{1.5, 3.5}), mean1);
        assertEquals( Nd4j.create(new double[]{2, 3}), a.mean(0));
        assertEquals(2.5, Nd4j.linspace(1, 4, 4).meanNumber().doubleValue(), 1e-1);
        assertEquals(2.5, a.meanNumber().doubleValue(), 1e-1);

    }

    @Test
    public void testTad() {
        INDArray arr = Nd4j.ones(2,10,10,10,10);
        for(int i = 0; i < 5; i++) {
            System.out.println(arr.tensorAlongDimension(i,1).offset());
        }
    }

    @Test
    public void testSumWithRow2(){
        //All sums in this method execute without exceptions.
        INDArray array3d = Nd4j.ones(2,10,10);
        array3d.sum(0);
        array3d.sum(1);
        array3d.sum(2);

        INDArray array4d = Nd4j.ones(2, 10, 10, 10);

        int tad = array4d.tensorAlongDimension(0,0).elementWiseStride();
        int tads = array4d.tensorssAlongDimension(0);
        for(int i = 10; i < array4d.tensorssAlongDimension(0); i++) {
            System.out.println(array4d.tensorAlongDimension(i,0).offset());
        }
        array4d.sum(0);
        array4d.sum(1);
        array4d.sum(2);
        array4d.sum(3);

        INDArray array5d = Nd4j.ones(2, 10, 10, 10, 10);
        array5d.sum(0);
        array5d.sum(1);
        array5d.sum(2);
        array5d.sum(3);
        array5d.sum(4);
    }



    @Test
    public void testEps() {
        INDArray ones = Nd4j.ones(5);
        INDArray eps = Nd4j.getExecutioner().exec(new Eps(ones, ones, ones, ones.length())).z();
        double sum = eps.sumNumber().doubleValue();
        assertEquals(5, sum, 1e-1);
    }


    @Test
    public void testMean() {
        INDArray mean2 = Nd4j.linspace(1, 5, 5);
        assertEquals(3,mean2.meanNumber().doubleValue(),1e-1);
    }



    @Test
    public void testSum2() {
        INDArray test = Nd4j.create(new float[]{1, 2, 3, 4}, new int[]{2, 2});
        INDArray sum = test.sum(1);
        INDArray assertion = Nd4j.create(new float[]{3, 7});
        assertEquals(assertion, sum);
        INDArray sum0 = Nd4j.create(new double[]{4, 6});
        assertEquals(sum0, test.sum(0));
    }


    @Test
    public void testRowSoftmax() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray arr = Nd4j.linspace(1, 6, 6);
        SoftMax softMax = new SoftMax(arr);
        opExecutioner.exec(softMax);
        assertEquals(1.0, softMax.z().sumNumber().doubleValue(), 1e-1);
    }

    @Test
    public void testRowLogSoftMax() {
        //For moderate input values, LogSoftMax op should be identical to log(softmax)
        // through is numerically more stable for
        int[][] shapes = new int[][]{{5,3},{5,100},{1,5},{1,100}};

        double eps = 1e-1;

        for( int[] shape : shapes) {
            INDArray orig = Nd4j.rand(shape);

            INDArray orig1 = orig.dup();
            INDArray orig2 = orig.dup();

            //First: standard log(softmax)
            Nd4j.getExecutioner().exec(new SoftMax(orig1), 1);
            Nd4j.getExecutioner().exec(new Log(orig1));

            //Second: LogSoftMax op
            Nd4j.getExecutioner().exec(new LogSoftMax(orig2),1);

            for( int i = 0; i < shape[0]; i++ ){
                for( int j = 0; j < shape[1]; j++ ){
                    double o1 = orig1.getDouble(i);
                    double o2 = orig2.getDouble(i);
                    if(Math.abs(o1-o2) > eps){
                        System.out.println();
                    }
                    assertEquals(o1,o2,eps);
                }
            }
        }
    }


    @Test
    public void testSum() {
        INDArray n = Nd4j.create(Nd4j.linspace(1, 8, 8).data(), new int[]{2, 2, 2});
        INDArray test = Nd4j.create(new float[]{3, 7, 11, 15}, new int[]{2, 2});

        INDArray sum = n.sum(-1);
        assertEquals(test, sum);
        INDArray sumZero = n.sum(0);
        INDArray assertion = Nd4j.create(new double[]{6,8,10,12},new int[]{2,2});
        assertEquals(assertion,sumZero);
        INDArray sumOne = n.sum(1);
        for(int i = 0; i < n.tensorssAlongDimension(1); i++) {
            System.out.println(n.tensorAlongDimension(i,1));
        }
        INDArray assertionTwo = Nd4j.create(new double[]{4,6,12,14},new int[]{2,2});
        assertEquals(assertionTwo,sumOne);
    }




    @Test
    public void testArgMax() {
        INDArray toArgMax = Nd4j.linspace(1,24,24).reshape(4, 3, 2);
        System.out.println(toArgMax.tensorssAlongDimension(0));
        int elementWise = toArgMax.tensorAlongDimension(0,0).elementWiseStride();
        for(int i = 0; i < toArgMax.tensorssAlongDimension(0); i++) {
            System.out.println(toArgMax.tensorAlongDimension(i,0));
        }
        INDArray tensor = toArgMax.tensorAlongDimension(0,0);
        System.out.println(toArgMax.max(0));
        System.out.println();
    }



    @Test
    public void testElementWiseOp() {
        Transforms.sigmoid(Nd4j.ones(5,5));
    }



    @Test
    public void testTensorAlongDimension() {
        int[] shape = new int[]{4,5,7};
        int length = ArrayUtil.prod(shape);
        INDArray arr = Nd4j.linspace(1, length, length).reshape(shape);


        int[] dim0s = {0,1,2,0,1,2};
        int[] dim1s = {1,0,0,2,2,1};

        double[] sums = {1350.,  1350.,  1582,  1582,  630,  630};

        for( int i = 0; i < dim0s.length; i++) {
            int firstDim = dim0s[i];
            int secondDim = dim1s[i];
            INDArray tad = arr.tensorAlongDimension(0, firstDim, secondDim);
            assertEquals("I " + i + " failed ",sums[i],tad.sumNumber().doubleValue(),1e-1);
        }
    }


    @Test
    public void testNorm2Double() {
        Nd4j.dtype = DataBuffer.Type.DOUBLE;
        INDArray n = Nd4j.create(new double[]{1, 2, 3, 4});
        double assertion = 5.47722557505;
        // double norm3 = n.norm2Number().doubleValue();
        // assertEquals(assertion, norm3, 1e-1);

        INDArray row = Nd4j.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
        INDArray row1 = row.getRow(1);
        double norm2 = row1.norm2Number().doubleValue();
        double assertion2 = 5.0f;
        assertEquals(assertion2, norm2, 1e-1);

    }


    @Test
    public void testNorm2() {
        INDArray n = Nd4j.create(new float[]{1, 2, 3, 4});
        float assertion = 5.47722557505f;
        float norm3 = n.norm2Number().floatValue();
        assertEquals(assertion, norm3, 1e-1);


        INDArray row = Nd4j.create(new float[]{1, 2, 3, 4}, new int[]{2, 2});
        INDArray row1 = row.getRow(1);
        float norm2 = row1.norm2Number().floatValue();
        float assertion2 = 5.0f;
        assertEquals(assertion2, norm2, 1e-1);
    }

    @Test
    public void testTADMMul(){
        Nd4j.getRandom().setSeed(12345);
        int[] shape = new int[]{4,5,7};
        INDArray arr = Nd4j.rand(shape);

        INDArray tad = arr.tensorAlongDimension(0, 1, 2);
        assertArrayEquals(tad.shape(), new int[]{7, 5});


        INDArray copy = Nd4j.zeros(7,5);
        for( int i = 0; i < 7; i++ ){
            for( int j = 0; j < 5; j++ ){
                copy.putScalar(new int[]{i,j},tad.getDouble(i,j));
            }
        }

//        System.out.println(tad);
//        System.out.println("\n");
//        System.out.println(copy);

        assertTrue(tad.equals(copy));

        INDArray first = Nd4j.rand(new int[]{2, 7});
        INDArray mmul = first.mmul(tad);
        INDArray mmulCopy = first.mmul(copy);

        assertTrue(mmul.equals(mmulCopy));

        INDArray mmul2 = tad.mmul(first);
        INDArray mmul2copy = copy.mmul(first);
        assertTrue(mmul2.equals(mmul2copy));
    }

    @Test
    public void testMeans3() {
        INDArray a = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray mean1 = a.mean(1);
        assertEquals(Nd4j.create(new double[]{1.5, 3.5}), mean1);
        assertEquals(Nd4j.create(new double[]{2, 3}), a.mean(0));
        assertEquals(2.5, Nd4j.linspace(1, 4, 4).meanNumber().doubleValue(), 1e-1);
        assertEquals( 2.5, a.meanNumber().doubleValue(), 1e-1);


        INDArray multipleMeans = Nd4j.linspace(1,24,24).reshape(2,2,3,2);
        INDArray mean = multipleMeans.mean(2,3);
        INDArray assertion = Nd4j.create(new double[][]{
                {3.5,9.5},
                {15.5,  21.5}
        });

        assertEquals(assertion,mean);
    }


    @Test
    public void testSums3() {
        INDArray multipleMeans = Nd4j.linspace(1,24,24).reshape(2,2,3,2);
        for(int i = 2; i < multipleMeans.tensorssAlongDimension(3); i++) {
            System.out.println(multipleMeans.tensorAlongDimension(i,3).offset());
        }
        INDArray mean = multipleMeans.sum(2, 3);
       /* INDArray sumAlongLast = multipleMeans.sum(3);
        for(int i = 0; i < sumAlongLast.tensorssAlongDimension(2); i++)
            System.out.println(sumAlongLast.tensorAlongDimension(i,2).offset());
*/
        INDArray assertion = Nd4j.create(new double[][]{
                {21,57},
                {93,  129}
        });

        assertEquals(assertion,mean);
    }


    @Test
    public void testMultiThreading() throws Exception {
        ExecutorService ex = ExecutorServiceProvider.getExecutorService();

        List<Future<?>> list = new ArrayList<>(100);
        for(int i = 0; i < 100; i++) {
            Future<?> future = ex.submit(new Runnable() {
                @Override
                public void run() {
                    INDArray dot = Nd4j.linspace(1, 8, 8);
                    System.out.println(Transforms.sigmoid(dot));
                }
            });
            list.add(future);
        }
        for(Future<?> future : list ){
            future.get(1, TimeUnit.MINUTES);
        }

    }

    @Test
    public void testMeanSumSimple(){
        System.out.println("3d");
        INDArray arr = Nd4j.ones(1,4,4);
        assertEquals(Nd4j.ones(1),arr.mean(1, 2));
        assertEquals(Nd4j.ones(1).muli(16), arr.sum(1,2));

        System.out.println("4d");
        INDArray arr4 = Nd4j.ones(1, 1, 4, 4);
        INDArray arr4m = arr4.mean(2, 3);
        INDArray arr4s = arr4.sum(2, 3);
        for( int i = 0; i < arr4m.length(); i++)
            assertEquals(arr4m.getDouble(i),1,0.0);
        for(int i = 0; i < arr4s.length(); i++)
            assertEquals(arr4s.getDouble(i),16,0.0);

        System.out.println("5d");
        INDArray arr5 = Nd4j.ones(1,1,4,4,4);
        INDArray arr5m = arr5.mean(2, 3);
        INDArray arr5s = arr5.sum(2,3);
        for(int i = 0; i < arr5s.length(); i++)
            assertEquals(16,arr5s.getDouble(i),0.0);

     /*   for( int i = 0; i < arr5m.length(); i++)
            assertEquals(1, arr5m.getDouble(i), 0.0);
*/
        System.out.println("6d");
        INDArray arr6 = Nd4j.ones(1,1,4,4,4,4);
        //  INDArray arr6m = arr6.mean(2, 3);
        INDArray arr6s = arr6.sum(2,3);
       /* for( int i = 0; i < arr6m.length(); i++)
            assertEquals(1,arr6m.getDouble(i),0.0);*/
        for(int i = 0; i < arr6s.length(); i++)
            assertEquals(16,arr6s.getDouble(i),0.0);
    }


    @Test
    public void testElementWiseStride() {
        int length = ArrayUtil.prod(2,2,3,2);
        INDArray ones = Nd4j.linspace(1,length,length).reshape(2,2,3,2);
        int[] dimensions = {3};
        System.out.println("Tads for 2,3 " + ones.tensorssAlongDimension(2,3) + " and 3 only " + ones.tensorssAlongDimension(3));
        for(int i = 0; i < ones.tensorssAlongDimension(dimensions); i++) {
            System.out.println(ones.tensorAlongDimension(i,dimensions));
        }
        System.out.println(ones.tensorAlongDimension(0, dimensions).elementWiseStride());
    }

    @Test
    public void testIMin() {
        INDArray arr = Nd4j.linspace(1, 10, 10);
        IMin imin = new IMin(arr);
        assertEquals(0, ((IndexAccumulation) Nd4j.getExecutioner().exec(imin)).getFinalResult());

        arr.muli(-1);
        imin = new IMin(arr);
        int minIdx = ((IndexAccumulation) Nd4j.getExecutioner().exec(imin)).getFinalResult();
        assertEquals(9, minIdx);
    }

    @Test
    @Ignore
    public void testNegativeNumbersSoftmax() throws Exception {
        Nd4j.MAX_ELEMENTS_PER_SLICE = Integer.MAX_VALUE;
        Nd4j.MAX_SLICES_TO_PRINT = Integer.MAX_VALUE;
        DataInputStream dis = new DataInputStream(new ClassPathResource("softmaxtest.nd").getInputStream());
        INDArray read = Nd4j.read(dis);
        dis.close();
        INDArray max1 = read.max(1);
        SoftMax softMax = new SoftMax(read);
        softMax.exec(1);
        INDArray z = softMax.z();
        INDArray zSums = z.sum(1);
        assertEquals(zSums.length(),zSums.sumNumber().doubleValue(),1e-1);
    }



    @Test
    public void testLength() {
        INDArray values = Nd4j.linspace(1,4,4).reshape(2,2);
        INDArray values2 = Nd4j.linspace(5,9,4).reshape(2,2);

        for(int i = 0; i < values.tensorssAlongDimension(1); i++) {
            System.out.println("X tad " + i  + " is " + values.tensorAlongDimension(i,1));
            System.out.println("Y tad " + i + " is " + values2.tensorAlongDimension(i,1));
        }

        INDArray expected = Nd4j.create(new double[]{5.89726867,  6.83942818});

        Accumulation accum = Nd4j.getOpFactory().createAccum("euclidean", values, values2);
        INDArray results = Nd4j.getExecutioner().exec(accum, 1);
        assertEquals(expected, results);

    }

    @Test
    public void testDivRowVector() {
        INDArray arr = Nd4j.linspace(1,4,4).reshape(2, 2);
        arr.diviRowVector(Nd4j.linspace(1, 2, 2));
        INDArray assertion = Nd4j.create(new double[][]{
                {1, 1}, {3, 2}
        });

        assertEquals(assertion,arr);
    }


    @Test
    public void testMulRowVector() {
        INDArray arr = Nd4j.linspace(1,4,4).reshape(2, 2);
        arr.muliRowVector(Nd4j.linspace(1, 2, 2));
        INDArray assertion = Nd4j.create(new double[][]{
                {1, 4}, {3, 8}
        });

        assertEquals(assertion,arr);
    }


    @Test
    public void testMMulColVectorRowVectorMixedOrder(){
        INDArray colVec = Nd4j.ones(5, 1);
        INDArray rowVec = Nd4j.ones(1, 5);
        INDArray out = rowVec.mmul(colVec);
        assertArrayEquals(out.shape(), new int[]{1, 1});
        assertTrue(out.equals(Nd4j.ones(1, 1).muli(5)));

        INDArray colVectorC = Nd4j.create(new int[]{5, 1}, 'c');
        INDArray rowVectorF = Nd4j.create(new int[]{1, 5}, 'f');
        for(int i = 0; i<colVectorC.length(); i++ )
            colVectorC.putScalar(i, 1.0);
        for(int i = 0; i < rowVectorF.length(); i++ )
            rowVectorF.putScalar(i, 1.0);
        assertTrue(colVec.equals(colVectorC));
        assertTrue(rowVec.equals(rowVectorF));

        INDArray outCF = rowVectorF.mmul(colVectorC);
        assertArrayEquals(outCF.shape(), new int[]{1, 1});
        assertTrue(outCF.equals(Nd4j.ones(1, 1).muli(5)));
    }

    @Test
    public void testNdVectorOpLinSpace() {
        int[] shape = {5,7,9,11,13};
        INDArray orig = Nd4j.linspace(1,ArrayUtil.prod(shape),ArrayUtil.prod(shape)).reshape(shape);
        int dimension = 0;
        System.out.println(orig.tensorssAlongDimension(dimension));
        for(int i = 0; i < 5; i++) {
            StringBuffer sb = new StringBuffer();
            INDArray tad = orig.tensorAlongDimension(i, dimension);
            for(int j = 0; j < tad.length(); j++) {
                sb.append(tad.get(NDArrayIndex.point(j)).offset());
                sb.append(",");
            }
            System.out.println(sb);
        }
        System.out.println();
        INDArray vector = Nd4j.linspace(1,shape[dimension],shape[dimension]);
        BroadcastOp op = new BroadcastAddOp(orig,vector,orig.dup(),dimension);
        Nd4j.getExecutioner().exec(op);
        for(int i = 0; i < 5; i++)
            System.out.println(op.z().tensorAlongDimension(i,dimension));
        int opNum = 0;
        //Compare expected vs. actual:
        for(int i = 0; i < orig.tensorssAlongDimension(dimension); i++) {
            INDArray tad = orig.tensorAlongDimension(i,dimension);
            INDArray zDim = op.z().tensorAlongDimension(i,dimension);
            INDArray assertion = tad.add(vector);
            assertEquals("Failed on tad with original tad " + tad + " at " + i,assertion,zDim);
        }
        NdIndexIterator iter = new NdIndexIterator(orig.shape());
        while (iter.hasNext()) {
            int[] next = iter.next();
            double origValue = orig.getDouble(next);
            double vectorValue = vector.getDouble(next[dimension]);   //current index in vector
            double exp;
            switch(opNum){
                case 0:
                    exp = origValue + vectorValue;
                    break;
                case 1:
                    exp = vectorValue;
                    break;
                case 2:
                    exp = origValue / vectorValue;
                    break;
                case 3:
                    exp = origValue * vectorValue;
                    break;
                case 4:
                    exp = vectorValue / origValue;
                    break;
                case 5:
                    exp = vectorValue - origValue;
                    break;
                case 6:
                    exp = origValue - vectorValue;
                    break;
                default:
                    throw new RuntimeException();
            }

            double actual = op.z().getDouble(next);
            double relError = Math.abs(exp - actual) / (Math.abs(exp) + Math.abs(actual));
            assertTrue("Failed on rank " + Arrays.toString(shape),relError < 1e-6);

        }
    }

    @Test
    public void testNdVectorOpLinSpaceDiv() {
        int[] shape = {5,7,9,11,13};
        INDArray orig = Nd4j.linspace(1,ArrayUtil.prod(shape),ArrayUtil.prod(shape)).reshape(shape);
        int dimension = 0;
        System.out.println(orig.tensorssAlongDimension(dimension));
        for(int i = 0; i < 5; i++) {
            StringBuffer sb = new StringBuffer();
            INDArray tad = orig.tensorAlongDimension(i, dimension);
            for(int j = 0; j < tad.length(); j++) {
                sb.append(tad.get(NDArrayIndex.point(j)).offset());
                sb.append(",");
            }
            System.out.println(sb);
        }
        System.out.println();
        INDArray vector = Nd4j.linspace(1,shape[dimension],shape[dimension]);
        BroadcastOp op = new BroadcastDivOp(orig,vector,orig.dup(),dimension);
        Nd4j.getExecutioner().exec(op);
        for(int i = 0; i < 5; i++)
            System.out.println(op.z().tensorAlongDimension(i,dimension));
        int opNum = 2;
        //Compare expected vs. actual:
        for(int i = 0; i < orig.tensorssAlongDimension(dimension); i++) {
            INDArray tad = orig.tensorAlongDimension(i,dimension);
            INDArray zDim = op.z().tensorAlongDimension(i,dimension);
            INDArray assertion = tad.div(vector);
            assertEquals("Failed on tad with original tad " + tad + " at " + i,assertion,zDim);
        }
        NdIndexIterator iter = new NdIndexIterator(orig.shape());
        while (iter.hasNext()) {
            int[] next = iter.next();
            double origValue = orig.getDouble(next);
            double vectorValue = vector.getDouble(next[dimension]);   //current index in vector
            double exp;
            switch(opNum){
                case 0:
                    exp = origValue + vectorValue;
                    break;
                case 1:
                    exp = vectorValue;
                    break;
                case 2:
                    exp = origValue / vectorValue;
                    break;
                case 3:
                    exp = origValue * vectorValue;
                    break;
                case 4:
                    exp = vectorValue / origValue;
                    break;
                case 5:
                    exp = vectorValue - origValue;
                    break;
                case 6:
                    exp = origValue - vectorValue;
                    break;
                default:
                    throw new RuntimeException();
            }

            double actual = op.z().getDouble(next);
            double relError = Math.abs(exp - actual) / (Math.abs(exp) + Math.abs(actual));
            assertTrue("Failed on rank " + Arrays.toString(shape),relError < 1e-6);

        }
    }

    @Test
    public void testFiveBySevenDimOne() {
        INDArray orig = Nd4j.linspace(1, 35, 35).reshape(5, 7);
        INDArray vector = Nd4j.linspace(1, 7, 7);
        int dimension = 1;
        System.out.println(orig.tensorssAlongDimension(dimension));
        for (int i = 0; i < 5; i++)
            System.out.println(orig.tensorAlongDimension(i, dimension));
        System.out.println();
        BroadcastOp op = new BroadcastAddOp(orig, vector, orig.dup(), dimension);
        Nd4j.getExecutioner().exec(op);
        //Compare expected vs. actual:
        for (int i = 0; i < orig.tensorssAlongDimension(dimension); i++) {
            INDArray tad = orig.tensorAlongDimension(i, dimension);
            INDArray zDim = op.z().tensorAlongDimension(i, dimension);
            INDArray assertion = tad.add(vector);
            assertEquals("Failed on tad with original tad " + tad + " at " + i, assertion, zDim);
        }


        NdIndexIterator iter = new NdIndexIterator(orig.shape());
        int[] shape = {5,7};
        int opNum = 0;
        while (iter.hasNext()) {
            int[] next = iter.next();
            double origValue = orig.getDouble(next);
            double vectorValue = vector.getDouble(next[dimension]);   //current index in vector
            double exp;
            switch (opNum) {
                case 0:
                    exp = origValue + vectorValue;
                    break;
                case 1:
                    exp = vectorValue;
                    break;
                case 2:
                    exp = origValue / vectorValue;
                    break;
                case 3:
                    exp = origValue * vectorValue;
                    break;
                case 4:
                    exp = vectorValue / origValue;
                    break;
                case 5:
                    exp = vectorValue - origValue;
                    break;
                case 6:
                    exp = origValue - vectorValue;
                    break;
                default:
                    throw new RuntimeException();
            }

            double actual = op.z().getDouble(next);
            double relError = Math.abs(exp - actual) / (Math.abs(exp) + Math.abs(actual));
            assertTrue("Failed on rank " + Arrays.toString(shape), relError < 1e-6);

        }
    }


    @Test
    public void testFiveBySevenRDiv() {
        INDArray orig = Nd4j.linspace(1, 35, 35).reshape(5, 7);
        INDArray vector = Nd4j.linspace(1, 5, 5);
        int dimension = 0;
        System.out.println(orig.tensorssAlongDimension(dimension));
        for (int i = 0; i < 5; i++)
            System.out.println(orig.tensorAlongDimension(i, dimension));
        System.out.println();
        BroadcastOp op = new BroadcastRDivOp(orig, vector, orig.dup(), dimension);
        Nd4j.getExecutioner().exec(op);
        //Compare expected vs. actual:
        for (int i = 0; i < orig.tensorssAlongDimension(dimension); i++) {
            INDArray tad = orig.tensorAlongDimension(i, dimension);
            INDArray zDim = op.z().tensorAlongDimension(i, dimension);
            INDArray assertion = tad.rdiv(vector);
            assertEquals("Failed on tad with original tad " + tad + " at " + i, assertion, zDim);
        }

    }


    @Test
    public void testFiveBySevenDiv() {
        INDArray orig = Nd4j.linspace(1, 35, 35).reshape(5, 7);
        INDArray vector = Nd4j.linspace(1, 5, 5);
        int dimension = 0;
        System.out.println(orig.tensorssAlongDimension(dimension));
        for (int i = 0; i < 5; i++)
            System.out.println(orig.tensorAlongDimension(i, dimension));
        System.out.println();
        BroadcastOp op = new BroadcastDivOp(orig, vector, orig.dup(), dimension);
        Nd4j.getExecutioner().exec(op);
        //Compare expected vs. actual:
        for (int i = 0; i < orig.tensorssAlongDimension(dimension); i++) {
            INDArray tad = orig.tensorAlongDimension(i, dimension);
            INDArray zDim = op.z().tensorAlongDimension(i, dimension);
            INDArray assertion = tad.div(vector);
            assertEquals("Failed on tad with original tad " + tad + " at " + i, assertion, zDim);
        }

    }

    @Test
    public void testFiveBySeven() {
        INDArray orig = Nd4j.linspace(1, 35, 35).reshape(5, 7);
        INDArray vector = Nd4j.linspace(1, 5, 5);
        int dimension = 0;
        System.out.println(orig.tensorssAlongDimension(dimension));
        for (int i = 0; i < 5; i++)
            System.out.println(orig.tensorAlongDimension(i, dimension));
        System.out.println();
        BroadcastOp op = new BroadcastAddOp(orig, vector, orig.dup(), dimension);
        Nd4j.getExecutioner().exec(op);
        //Compare expected vs. actual:
        for (int i = 0; i < orig.tensorssAlongDimension(dimension); i++) {
            INDArray tad = orig.tensorAlongDimension(i, dimension);
            INDArray zDim = op.z().tensorAlongDimension(i, dimension);
            INDArray assertion = tad.add(vector);
            assertEquals("Failed on tad with original tad " + tad + " at " + i, assertion, zDim);
        }


        NdIndexIterator iter = new NdIndexIterator(orig.shape());
        int[] shape = {5,7};
        int opNum = 0;
        while (iter.hasNext()) {
            int[] next = iter.next();
            double origValue = orig.getDouble(next);
            double vectorValue = vector.getDouble(next[dimension]);   //current index in vector
            double exp;
            switch (opNum) {
                case 0:
                    exp = origValue + vectorValue;
                    break;
                case 1:
                    exp = vectorValue;
                    break;
                case 2:
                    exp = origValue / vectorValue;
                    break;
                case 3:
                    exp = origValue * vectorValue;
                    break;
                case 4:
                    exp = vectorValue / origValue;
                    break;
                case 5:
                    exp = vectorValue - origValue;
                    break;
                case 6:
                    exp = origValue - vectorValue;
                    break;
                default:
                    throw new RuntimeException();
            }

            double actual = op.z().getDouble(next);
            double relError = Math.abs(exp - actual) / (Math.abs(exp) + Math.abs(actual));
            assertTrue("Failed on rank " + Arrays.toString(shape), relError < 1e-6);

        }
    }

    @Test
    public void testAddOnlyOneColumn() {
        INDArray arr = Nd4j.linspace(1,8,8).reshape(2,4);
        System.out.println(arr.tensorAlongDimension(1,0));
        arr.tensorAlongDimension(1,0).addi(Nd4j.ones(2));
        assertEquals(Nd4j.create(new double[]{3,7}),arr.tensorAlongDimension(1,0));
    }

    @Test
    public void testColumnVectorAdd() {
        INDArray vector = Nd4j.create(new double[]{0.8183500170707703,0.5002227425575256,0.810189425945282,0.09596852213144302,0.2189500331878662,0.2587190568447113,0.4681057631969452});
        INDArray matrix = Nd4j.create(new double[]{1.7479660511016846,0.8165982961654663,0.9941082000732422,0.30052879452705383,0.7866750359535217,0.8542637825012207,1.4326202869415283,1.471527099609375,1.249129295349121,1.4637593030929565,0.8436833620071411,1.1802568435668945,0.26710736751556396,0.5745501518249512,1.935403823852539,1.6568565368652344,2.4301915168762207,1.0641130208969116,1.4025475978851318,1.2411234378814697,1.5786868333816528,2.354153633117676,1.4680445194244385,1.9459636211395264,0.6315816640853882,1.1675891876220703,1.5114526748657227,1.6130852699279785,3.245872735977173,1.6715824604034424,2.4574174880981445,1.0882757902145386,1.560572624206543,0.8008333444595337,1.8960646390914917},new int[] {5,7});
        int dimension = 1;
        BroadcastAddOp op = new BroadcastAddOp(matrix,vector,matrix,dimension);
        INDArray assertion = matrix.dup();
        for(int i = 0; i < assertion.tensorssAlongDimension(dimension); i++) {
            assertion.tensorAlongDimension(i,dimension).addi(vector);
        }

        Nd4j.getExecutioner().exec(op);
        assertEquals(assertion,op.z());



    }



    @Test
    public void testManualAddRowVector() {
        INDArray seven = Nd4j.linspace(1,7,7);
        INDArray matrix  = Nd4j.linspace(1,14,14).reshape(2,7);
        INDArray firstRowAssertion = Nd4j.create(new double[]{2,4,6,8,10,12,14});
        INDArray secondRowAssertion = Nd4j.create(new double[]{9,11,13,15,17,19,21});
        matrix.tensorAlongDimension(0,1).addi(seven);
        assertEquals(firstRowAssertion,matrix.tensorAlongDimension(0,1));
        matrix.tensorAlongDimension(1,1).addi(seven);
        assertEquals(secondRowAssertion,matrix.tensorAlongDimension(1,1));

    }

    @Test
    public void testManualAddColumnVector() {
        INDArray seven = Nd4j.linspace(1,2,2);
        INDArray matrix  = Nd4j.linspace(1,14,14).reshape(2,7);
        INDArray firstRowAssertion = Nd4j.create(new double[]{2,10});
        INDArray secondRowAssertion = Nd4j.create(new double[]{3,11});
        matrix.tensorAlongDimension(0,0).addi(seven);
        assertEquals(firstRowAssertion,matrix.tensorAlongDimension(0,0));
        matrix.tensorAlongDimension(1,0).addi(seven);
        assertEquals(secondRowAssertion,matrix.tensorAlongDimension(1,0));

    }

    @Test
    public void testDimensionOneLengthSeven() {
        INDArray seven = Nd4j.linspace(1,7,7);
        int[] tensorShape = {5, 7, 9, 11, 13};
        int len = ArrayUtil.prod(tensorShape);
        int dimension = 1;
        INDArray arr = Nd4j.linspace(1,len,len).reshape(tensorShape);
        BroadcastAddOp op = new BroadcastAddOp(arr,seven,arr,dimension);
        INDArray dup = arr.dup();
        Nd4j.getExecutioner().exec(op);

        for(int i = 0; i < dup.tensorssAlongDimension(dimension); i++) {
            System.out.println("Adding vector " + seven + " to tad " + dup.tensorAlongDimension(i,dimension));
            System.out.println("Comparing against  vector " + seven + " to tad " + arr.tensorAlongDimension(i,dimension));
            dup.tensorAlongDimension(i,dimension).addi(seven);

        }

        assertEquals(dup,op.z());
    }


    @Test
    public void testNdVectorOp() {
        //Test 2d, 3d, ..., 6d vector ops

        Nd4j.getRandom().setSeed(12345);
        int[] maxShape = new int[]{5, 7, 9, 11, 13, 15};

        for(int opNum = 0; opNum < 6; opNum++) {
            for (int rank = 2; rank < maxShape.length; rank++) {
                int[] shape = Arrays.copyOfRange(maxShape, 0, rank);
                INDArray orig = Nd4j.rand(shape);

                for (int i = 0; i < rank; i++) {   //Test ops for each dimension
                    INDArray arr = orig.dup();
                    INDArray vector = i == 0 ? Nd4j.rand(1,shape[i]) : Nd4j.rand(shape[i],1);
                    System.out.println("Executed rank " + rank + " and dimension " + i + " with vector " + vector + " and array of shape " + Arrays.toString(arr.shape()));
                    BroadcastOp op;
                    switch(opNum){
                        case 0:
                            op = new BroadcastAddOp(arr, vector, arr.dup(), i);
                            break;
                        case 1:
                            op = new BroadcastCopyOp(arr, vector, arr, i);
                            break;
                        case 2:
                            op = new BroadcastDivOp(arr, vector, arr.dup(), i);
                            break;
                        case 3:
                            op = new BroadcastMulOp(arr, vector, arr.dup(), i);
                            break;
                        case 4:
                            op = new BroadcastRDivOp(arr, vector, arr.dup(), i);
                            break;
                        case 5:
                            op = new BroadcastRSubOp(arr, vector, arr.dup(), i);
                            break;
                        case 6:
                            op = new BroadcastSubOp(arr, vector, arr.dup(), i);
                            break;
                        default:
                            throw new RuntimeException();
                    }

                    StopWatch watch = new StopWatch();
                    watch.start();
                    System.out.println("About to execute op " + op.name());
                    Nd4j.getExecutioner().exec(op);
                    watch.stop();

                    System.out.println("After execution " + watch.getNanoTime() + " nanoseconds with " + op.x().tensorssAlongDimension(i));
                    INDArray assertion = arr.dup();
                    for(int j = 0; j < arr.tensorssAlongDimension(i); j++) {
                        switch(opNum) {
                            case 0:
                                assertion.tensorAlongDimension(j,i).addi(vector);
                                break;
                            case 1:
                                assertion.tensorAlongDimension(j,i).assign(vector);
                                break;
                            case 2:
                                assertion.tensorAlongDimension(j,i).divi(vector);
                                break;
                            case 3:
                                assertion.tensorAlongDimension(j,i).muli(vector);
                                break;
                            case 4:
                                assertion.tensorAlongDimension(j,i).rdivi(vector);
                                break;
                            case 5:
                                assertion.tensorAlongDimension(j,i).rsubi(vector);
                                break;
                            case 6:
                                assertion.tensorAlongDimension(j,i).subi(vector);
                                break;
                            default:
                                throw new RuntimeException();
                        }
                    }

                    assertEquals(assertion,op.z());
                }
            }
        }
    }



    @Test
    public void testCosineSim() {
        Nd4j.dtype = DataBuffer.Type.FLOAT;

        INDArray vec1 = Nd4j.create(new double[]{1, 2, 3, 4});
        INDArray vec2 = Nd4j.create(new double[]{1, 2, 3, 4});
        double sim = Transforms.cosineSim(vec1, vec2);
        assertEquals(1, sim, 1e-1);

        INDArray vec3 = Nd4j.create(new float[]{0.2f, 0.3f, 0.4f, 0.5f});
        INDArray vec4 = Nd4j.create(new float[]{0.6f, 0.7f, 0.8f, 0.9f});
        sim = Transforms.cosineSim(vec3, vec4);
        assertEquals(0.98, sim, 1e-1);

    }

    @Test
    public void testSumWithRow1(){
        //Works:
        INDArray array2d = Nd4j.ones(1,10);
        array2d.sum(0); //OK
        array2d.sum(1); //OK

        INDArray array3d = Nd4j.ones(1,10,10);
        array3d.sum(0); //OK
        array3d.sum(1); //OK
        array3d.sum(2); //java.lang.IllegalArgumentException: Illegal index 100 derived from 9 with offset of 10 and stride of 10

        INDArray array4d = Nd4j.ones(1,10,10,10);
        array4d.sum(0); //OK
        array4d.sum(1); //OK
        array4d.sum(2); //java.lang.IllegalArgumentException: Illegal index 1000 derived from 9 with offset of 910 and stride of 10
        array4d.sum(3); //java.lang.IllegalArgumentException: Illegal index 1000 derived from 9 with offset of 100 and stride of 100

        INDArray array5d = Nd4j.ones(1, 10, 10, 10, 10);
        array5d.sum(0); //OK
        array5d.sum(1); //OK
        array5d.sum(2); //java.lang.IllegalArgumentException: Illegal index 10000 derived from 9 with offset of 9910 and stride of 10
        array5d.sum(3); //java.lang.IllegalArgumentException: Illegal index 10000 derived from 9 with offset of 9100 and stride of 100
        array5d.sum(4); //java.lang.IllegalArgumentException: Illegal index 10000 derived from 9 with offset of 1000 and stride of 1000
    }

    @Test
    public void testToOffsetZero() {
        INDArray matrix  =  Nd4j.rand(3,5);
        INDArray rowOne = matrix.getRow(1);
        INDArray row1Copy = Shape.toOffsetZero(rowOne);
        assertEquals(rowOne,row1Copy);
        INDArray rows =  matrix.getRows(1, 2);
        INDArray rowsOffsetZero = Shape.toOffsetZero(rows);
        assertEquals(rows,rowsOffsetZero);

        INDArray tensor = Nd4j.rand(new int[]{3,3,3});
        INDArray getTensor = tensor.slice(1).slice(1);
        INDArray getTensorZero = Shape.toOffsetZero(getTensor);
        assertEquals(getTensor, getTensorZero);



    }

    @Test
    public void testDot2() {
        INDArray vec1 = Nd4j.create(new float[]{1, 2, 3, 4});
        INDArray vec2 = Nd4j.create(new float[]{1, 2, 3, 4});
        assertEquals(30, Nd4j.getBlasWrapper().dot(vec1, vec2), 1e-1);

        INDArray matrix = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray row = matrix.getRow(1);
        assertEquals(25, Nd4j.getBlasWrapper().dot(row, row), 1e-1);

    }


    @Test
    public void testMMul() {
        INDArray arr = Nd4j.create(new double[][]{
                {1, 2, 3}, {4, 5, 6}
        });

        INDArray assertion = Nd4j.create(new double[][]{
                {14, 32}, {32, 77}
        });

        INDArray test = arr.mmul(arr.transpose());
        assertEquals( assertion, test);

    }


    @Test
    public void testTADMMulLeadiusngOne(){
        Nd4j.getRandom().setSeed(12345);
        int[] shape = new int[]{1,5,7};
        INDArray arr = Nd4j.rand(shape);

        INDArray tad = arr.tensorAlongDimension(0, 1, 2);
        boolean order = Shape.cOrFortranOrder(tad.shape(), tad.stride(), tad.elementStride());
        assertArrayEquals(tad.shape(),new int[]{7,5});


        INDArray copy = Nd4j.zeros(7,5);
        for( int i = 0; i < 7; i++ ){
            for( int j = 0; j < 5; j++ ){
                copy.putScalar(new int[]{i,j},tad.getDouble(i,j));
            }
        }

        assertTrue(tad.equals(copy));

        INDArray first = Nd4j.rand(new int[]{2, 7});
        INDArray mmul = first.mmul(tad);
        INDArray mmulCopy = first.mmul(copy);

        assertTrue(mmul.equals(mmulCopy));

        INDArray mmul2 = tad.mmul(first);
        INDArray mmul2copy = copy.mmul(first);
        assertTrue(mmul2.equals(mmul2copy));
    }


    @Test
    public void testArgMax2() {
        INDArray toArgMax = Nd4j.linspace(1,24,24).reshape(4, 3, 2);
        for(int i = 0; i < toArgMax.tensorssAlongDimension(1); i++) {
            System.out.println(toArgMax.tensorAlongDimension(i,1));
        }
        INDArray  argMax = Nd4j.argMax(toArgMax, 1);
        INDArray argMaxZero = Nd4j.argMax(toArgMax,0);
        INDArray argMaxTwo = Nd4j.argMax(toArgMax,2);
        INDArray valueArray = Nd4j.valueArrayOf(new int[]{4, 2}, 2.0);
        INDArray valueArrayTwo = Nd4j.valueArrayOf(new int[]{3,2},3.0);
        INDArray valueArrayThree = Nd4j.valueArrayOf(new int[]{4,3},1.0);
        assertEquals(valueArray, argMax);
        assertEquals(valueArrayTwo, argMaxZero);
        assertEquals(valueArrayThree,argMaxTwo);
    }


    @Test
    public void testMeans2() {
        INDArray a = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray mean1 = a.mean(1);
        assertEquals( Nd4j.create(new double[]{1.5, 3.5}), mean1);
        assertEquals( Nd4j.create(new double[]{2, 3}), a.mean(0));
        assertEquals(2.5, Nd4j.linspace(1, 4, 4).meanNumber().doubleValue(), 1e-1);
        assertEquals(2.5, a.meanNumber().doubleValue(), 1e-1);

    }

    @Test
    public void testAddVectorWithOffset() throws Exception {
        INDArray oneThroughFour = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray row1 = oneThroughFour.getRow(1);
        row1.addi(1);
        INDArray result = Nd4j.create(new float[]{1, 2, 4, 5}, new int[]{2, 2});
        assertEquals(result, oneThroughFour);
    }


    @Test
    public void testColumnMmul() {
        DataBuffer data = Nd4j.linspace(1, 10, 18).data();
        INDArray x2 = Nd4j.create(data, new int[]{2,3,3});
        data = Nd4j.linspace(1, 12, 9).data();
        INDArray y2 = Nd4j.create(data, new int[]{3,3});
        INDArray z2 = Nd4j.create(3,2);
        z2.putColumn(0, y2.getColumn(0));
        z2.putColumn(1, y2.getColumn(1));
        INDArray nofOffset = Nd4j.create(3,3);
        nofOffset.assign(x2.slice(0));
        assertEquals(nofOffset,x2.slice(0));

        INDArray slice = x2.slice(0);
        INDArray zeroOffsetResult = slice.mmul(z2);
        INDArray offsetResult = nofOffset.mmul(z2);
        assertEquals(zeroOffsetResult,offsetResult);


        INDArray slice1 = x2.slice(1);
        INDArray noOffset2 = Nd4j.create(slice1.shape());
        noOffset2.assign(slice1);
        assertEquals(slice1,noOffset2);

        INDArray noOffsetResult = noOffset2.mmul(z2);
        INDArray slice1OffsetResult = slice1.mmul(z2);

        assertEquals(noOffsetResult,slice1OffsetResult);
    }



    @Test
    public void testSumLeadingTrailingZeros(){
        testSumHelper(1,5,5);
        testSumHelper(5,5,1);
        testSumHelper(1, 5, 1);

        testSumHelper(1,5,5,5);
        testSumHelper(5,5,5,1);
        testSumHelper(1,5,5,1);

        testSumHelper(1,5,5,5,5);
        testSumHelper(5,5,5,5,1);
        testSumHelper(1,5,5,5,1);

        testSumHelper(1,5,5,5,5,5);
        testSumHelper(5, 5, 5, 5, 5, 1);
        testSumHelper(1, 5, 5, 5, 5, 1);
    }

    private  void testSumHelper(int... shape) {
        INDArray array = Nd4j.ones(shape);
        for( int i = 0; i < shape.length; i++) {
            for(int j = 0; j < array.vectorsAlongDimension(i); j++) {
                INDArray vec = array.vectorAlongDimension(j,i);
            }
            array.sum(i);
        }
    }





}
