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

import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import jcuda.runtime.JCuda;
import org.apache.commons.lang3.time.StopWatch;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.Eps;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
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
    public void testSum() {
        INDArray n = Nd4j.create(Nd4j.linspace(1, 8, 8).data(), new int[]{2, 2, 2});
        int elementWiseStride = n.tensorAlongDimension(0,-1).elementWiseStride();
        for(int i = 0; i < n.tensorssAlongDimension(-1); i++) {
            System.out.println(n.tensorAlongDimension(i,-1).offset());
        }

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
        double norm3 = n.norm2Number().doubleValue();
        assertEquals(assertion, norm3, 1e-1);

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
    public void testMultipleThreads() throws InterruptedException {
        int numThreads = 10;
        final INDArray array = Nd4j.rand(300, 300);
        final INDArray expected = array.dup().mmul(array).mmul(array).div(array).div(array);
        final AtomicInteger correct = new AtomicInteger();
        final CountDownLatch latch = new CountDownLatch(numThreads);
        System.out.println("Running on " + ContextHolder.getInstance().deviceNum());
        ExecutorService executors = Executors.newCachedThreadPool();

        for(int x = 0; x< numThreads; x++) {
            executors.execute(new Runnable() {
                @Override
                public void run() {
                    try {
                        int total = 10;
                        int right = 0;
                        for(int x = 0; x< total; x++) {
                            StopWatch watch = new StopWatch();
                            watch.start();
                            INDArray actual = array.dup().mmul(array).mmul(array).div(array).div(array);
                            watch.stop();
                            if(expected.equals(actual)) right++;
                        }

                        if(total == right)
                            correct.incrementAndGet();
                    } finally {
                        latch.countDown();
                    }

                }
            });
        }

        latch.await();

        assertEquals(numThreads, correct.get());

    }

}
