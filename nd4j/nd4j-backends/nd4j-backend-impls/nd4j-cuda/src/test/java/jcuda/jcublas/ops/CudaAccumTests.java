/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package jcuda.jcublas.ops;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.Max;
import org.nd4j.linalg.api.ops.impl.accum.Mean;
import org.nd4j.linalg.api.ops.impl.accum.Min;
import org.nd4j.linalg.api.ops.impl.accum.Sum;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * @author raver119@gmail.com
 */
@Ignore
public class CudaAccumTests {

    @Before
    public void setUp() {
        CudaEnvironment.getInstance().getConfiguration()
                .setExecutionModel(Configuration.ExecutionModel.ASYNCHRONOUS)
                .setFirstMemory(AllocationStatus.DEVICE)
                .setMaximumBlockSize(128)
                .setMaximumGridSize(256)
                .enableDebug(false)
                .setVerbose(false);

        System.out.println("Init called");
    }

    @Test
    public void testBiggerSum() throws Exception {
        INDArray array = Nd4j.ones(128000, 512);

        array.sum(0);
    }

    /**
     * Sum call
     * @throws Exception
     */
    @Test
    public void testPinnedSum() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{2.01f, 2.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});

        Sum sum = new Sum(array1);
        Nd4j.getExecutioner().exec(sum, 1);

        Number resu = sum.getFinalResult();

        System.out.println("Result: " + resu);

        assertEquals(17.15f, resu.floatValue(), 0.01f);
    }

    @Test
    public void testPinnedSum2() throws Exception {
        // simple way to stop test if we're not on CUDA backend here

        INDArray array1 = Nd4j.linspace(1, 10000, 100000).reshape(100,1000);

        Sum sum = new Sum(array1);
        INDArray result;/* = Nd4j.getExecutioner().exec(sum, 0);

        assertEquals(495055.44f, result.getFloat(0), 0.01f);
*/
        result = Nd4j.getExecutioner().exec(sum, 1);
        result = Nd4j.getExecutioner().exec(sum, 1);
        assertEquals(50945.52f, result.getFloat(0), 0.01f);

    }

    @Test
    public void testPinnedSum3() throws Exception {
        // simple way to stop test if we're not on CUDA backend here

        INDArray array1 = Nd4j.linspace(1, 100000, 100000).reshape(100,1000);

        for (int x = 0; x < 100000; x++ ){
            assertEquals("Failed on iteration [" + x + "]", x+1, array1.getFloat(x), 0.01f);
        }
    }

    @Test
    public void testPinnedSumNumber() throws Exception {
        // simple way to stop test if we're not on CUDA backend here

        INDArray array1 = Nd4j.linspace(1, 10000, 10000);

        float sum = array1.sumNumber().floatValue();

        assertEquals(5.0005E7, sum, 1f);
    }

    @Test
    public void testPinnedSumNumber2() throws Exception {
        // simple way to stop test if we're not on CUDA backend here

        INDArray array1 = Nd4j.ones(128000);

        long time1 = System.currentTimeMillis();
        float sum = array1.sumNumber().floatValue();
        long time2 = System.currentTimeMillis();

        System.out.println("Execution time: " + (time2 - time1));

        assertEquals(128000f, sum, 0.01f);
    }

    @Test
    public void testPinnedSumNumber3() throws Exception {
        // simple way to stop test if we're not on CUDA backend here

        INDArray array1 = Nd4j.ones(12800000);

        float sum = array1.sumNumber().floatValue();

        assertEquals(12800000f, sum, 0.01f);
    }


    @Test
    public void testStdev0(){
        double[][] ind = {{5.1, 3.5, 1.4}, {4.9, 3.0, 1.4}, {4.7, 3.2, 1.3}};
        INDArray in = Nd4j.create(ind);
        INDArray stdev = in.std(0);

        INDArray exp = Nd4j.create(new double[]{0.2, 0.25166114784, 0.05773502692});

        System.out.println("Exp dtype: " + exp.data().dataType());
        System.out.println("Exp dtype: " + exp.data().dataType());

        System.out.println("Array: " + Arrays.toString(exp.data().asFloat()));
        assertEquals(exp,stdev);
    }

    @Test
    public void testStdev1(){
        double[][] ind = {{5.1, 3.5, 1.4}, {4.9, 3.0, 1.4}, {4.7, 3.2, 1.3}};
        INDArray in = Nd4j.create(ind);
        INDArray stdev = in.std(1);

        INDArray exp = Nd4j.create(new double[]{1.8556220880, 1.7521415468, 1.7039170559});

        assertEquals(exp,stdev);
    }


    @Test
    public void testStdevNum(){
        INDArray in = Nd4j.linspace(1, 1000, 10000);
        float stdev = in.stdNumber().floatValue();


        assertEquals(288.42972f, stdev, 0.001f);
    }

    /**
     * Mean call
     * @throws Exception
     */
    @Test
    public void testPinnedMean() throws Exception {
        // simple way to stop test if we're not on CUDA backend here
        assertEquals("JcublasLevel1", Nd4j.getBlasWrapper().level1().getClass().getSimpleName());

        INDArray array1 = Nd4j.create(new float[]{2.01f, 2.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});
        INDArray array2 = Nd4j.create(new float[]{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f});


        Mean mean = new Mean(array1);
        Nd4j.getExecutioner().exec(mean, 1);

        Number resu = mean.getFinalResult();


//        INDArray result = Nd4j.getExecutioner().exec(new Mean(array1), 1);

        System.out.println("Array1: " + array1);
        System.out.println("Result: " + resu);

        assertEquals(1.14f, resu.floatValue(), 0.01f);
    }

    @Test
    public void testSum2() {
        INDArray n = Nd4j.create(Nd4j.linspace(1, 8, 8).data(), new int[]{2, 2, 2});
        System.out.println("N result: " + n);
        INDArray test = Nd4j.create(new float[]{3, 7, 11, 15}, new int[]{2, 2});
        System.out.println("Test result: " + test);
        INDArray sum = n.sum(-1);

        System.out.println("Sum result: " + sum);
        assertEquals(test, sum);
    }

    @Test
    public void testMax() {
        INDArray n = Nd4j.linspace(1, 15, 15);

        float max = n.maxNumber().floatValue();

        assertEquals(15f, max, 0.001f);
    }

    @Test
    public void testSum3() {
        INDArray n = Nd4j.linspace(1, 1000, 128000).reshape(128, 1000);


        long time1 = System.currentTimeMillis();
        INDArray sum = n.sum(new int[]{0});
        long time2 = System.currentTimeMillis();

        System.out.println("Time elapsed: "+ (time2 - time1) );

        System.out.println("Sum: " + sum);
        System.out.println("Sum.Length: " + sum.length());
        System.out.println("elementWiseStride: " + n.elementWiseStride());
        System.out.println("elementStride: " + n.elementStride());

        assertEquals(63565.02f, sum.getFloat(0), 0.01f);
        assertEquals(63566.02f, sum.getFloat(1), 0.01f);
    }

    @Test
    public void testSum3_1() throws Exception {
        INDArray n = Nd4j.linspace(1, 128000, 128000).reshape(128, 1000);


        long time1 = System.currentTimeMillis();
        INDArray sum = n.sum(new int[]{0});
        long time2 = System.currentTimeMillis();

        System.out.println("Time elapsed: "+ (time2 - time1) );

        System.out.println("Sum: " + sum);
        System.out.println("Sum.Length: " + sum.length());
        System.out.println("elementWiseStride: " + n.elementWiseStride());
        System.out.println("elementStride: " + n.elementStride());

        assertEquals(8128128.0f, sum.getFloat(0), 0.01f);
        assertEquals(8128256.0f, sum.getFloat(1), 0.01f);
        assertEquals(8128512.0f, sum.getFloat(3), 0.01f);
        assertEquals(8128640.0f, sum.getFloat(4), 0.01f);
    }

    @Test
    public void testSum4() {
        INDArray n = Nd4j.linspace(1, 1000, 128000).reshape(128, 1000);


        long time1 = System.currentTimeMillis();
        INDArray sum = n.sum(new int[]{1});
        long time2 = System.currentTimeMillis();

        System.out.println("Execution time: " + (time2 - time1));

        System.out.println("elementWiseStride: " + n.elementWiseStride());
        System.out.println("elementStride: " + n.elementStride());

        assertEquals(4898.4707f, sum.getFloat(0), 0.01f);
        assertEquals(12703.209f, sum.getFloat(1), 0.01f);
    }

    @Test
    public void testSum5() {
        INDArray n = Nd4j.linspace(1, 1000, 128000).reshape(128, 1000);


        INDArray sum = n.sum(new int[]{1});
        INDArray sum2 = n.sum(new int[]{-1});
        INDArray sum3 = n.sum(new int[]{0});

        System.out.println("elementWiseStride: " + n.elementWiseStride());
        System.out.println("elementStride: " + n.elementStride());

        assertEquals(4898.4707f, sum.getFloat(0), 0.01f);
        assertEquals(12703.209f, sum.getFloat(1), 0.01f);
        assertEquals(sum, sum2);
        assertNotEquals(sum, sum3);
        assertEquals(63565.023f, sum3.getFloat(0), 0.01f);
        assertEquals(63570.008f, sum3.getFloat(5), 0.01f);
    }

    @Test
    public void testSum6() {
        INDArray n = Nd4j.linspace(1, 1000, 128000).reshape(128, 10, 10, 10);


        INDArray sum0 = n.sum(new int[]{0});
        INDArray sum1 = n.sum(new int[]{1});
        INDArray sum3 = n.sum(new int[]{3});
        INDArray sumN = n.sum(new int[]{-1});
        INDArray sum2 = n.sum(new int[]{2});

        System.out.println("elementWiseStride: " + n.elementWiseStride());
        System.out.println("elementStride: " + n.elementStride());

        assertEquals(63565.023f, sum0.getFloat(0), 0.01f);
        assertEquals(63570.008f, sum0.getFloat(5), 0.01f);

        assertEquals(45.12137f, sum1.getFloat(0), 0.01f);
        assertEquals(45.511604f, sum1.getFloat(5), 0.01f);

        assertEquals(10.351214f, sum3.getFloat(0), 0.01f);
        assertEquals(14.25359f, sum3.getFloat(5), 0.01f);

        assertEquals(14.25359f, sumN.getFloat(5), 0.01f);
        assertEquals(13.74628f, sum2.getFloat(3), 0.01f);
    }

    @Test
    public void testSum3Of4_2222() {
        int[] shape = {2, 2, 2, 2};
        int length = ArrayUtil.prod(shape);
        INDArray arrC = Nd4j.linspace(1, length, length).reshape(shape);
        INDArray arrF = Nd4j.create(arrC.shape()).reshape('f', arrC.shape()).assign(arrC);

        System.out.println("Arrf: " + arrF);
        System.out.println("Arrf: " + Arrays.toString(arrF.data().asFloat()));
        System.out.println("ArrF shapeInfo: " + arrF.shapeInfoDataBuffer());
        System.out.println("----------------------------");

        int[][] dimsToSum = new int[][]{{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
        double[][] expD = new double[][]{{64, 72}, {60, 76}, {52, 84}, {36, 100}};


        for (int i = 0; i < dimsToSum.length; i++) {
            int[] d = dimsToSum[i];

            INDArray outC = arrC.sum(d);
            INDArray outF = arrF.sum(d);
            INDArray exp = Nd4j.create(expD[i],outC.shape());

            assertEquals(exp, outC);
            assertEquals(exp, outF);

            System.out.println("PASSED:" + Arrays.toString(d) + "\t" + outC + "\t" + outF);
        }
    }

    @Test
    public void testDimensionMax() {
        INDArray linspace = Nd4j.linspace(1, 6, 6).reshape('f', 2, 3);
        int axis = 0;
        INDArray row = linspace.slice(axis);
        System.out.println("Linspace: " + linspace);
        System.out.println("Row: " + row);

        System.out.println("Row shapeInfo: " + row.shapeInfoDataBuffer());

        Max max = new Max(row);
        double max2 = Nd4j.getExecutioner().execAndReturn(max).getFinalResult().doubleValue();
        assertEquals(5.0, max2, 1e-1);

        Min min = new Min(row);
        double min2 = Nd4j.getExecutioner().execAndReturn(min).getFinalResult().doubleValue();
        assertEquals(1.0, min2, 1e-1);
    }

    @Test
    public void testNorm2() throws Exception {
        INDArray array1 = Nd4j.create(new float[]{2.01f, 2.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f, 1.01f});

        INDArray result = array1.norm2(1);

        System.out.println(result);
        assertEquals(4.62f,  result.getDouble(0), 0.001);
    }

    @Test
    public void testSumF() throws Exception {
        INDArray arrc = Nd4j.linspace(1,6,6).reshape('c',3,2);
        INDArray arrf = Nd4j.create(new double[6],new int[]{3,2},'f').assign(arrc);

        System.out.println("ArrC: " + arrc);
        System.out.println("ArrC buffer: " + Arrays.toString(arrc.data().asFloat()));
        System.out.println("ArrF: " + arrf);
        System.out.println("ArrF buffer: " + Arrays.toString(arrf.data().asFloat()));
        System.out.println("ArrF shape: " + arrf.shapeInfoDataBuffer());

        INDArray cSum = arrc.sum(0);

        INDArray fSum = arrf.sum(0);

        assertEquals(Nd4j.create(new float[]{9f,12f}),fSum);
    }

    @Test
    public void testMax1() throws Exception {
        INDArray array1 = Nd4j.linspace(1, 76800,76800).reshape(256, 300);

        long time1 = System.currentTimeMillis();
        INDArray array = array1.max(1);
        long time2 = System.currentTimeMillis();

        System.out.println("Time elapsed: "+ (time2 - time1) );

        assertEquals(256, array.length());

        for (int x = 0; x < 256; x++) {
            assertEquals((x + 1) * 300, array.getFloat(x), 0.01f);
        }
    }

    @Test
    public void testMax0() throws Exception {
        INDArray array1 = Nd4j.linspace(1, 76800,76800).reshape(256, 300);



        long time1 = System.currentTimeMillis();
        INDArray array = array1.max(0);
        long time2 = System.currentTimeMillis();

        System.out.println("Array1 shapeInfo: " + array1.shapeInfoDataBuffer());
        System.out.println("Result shapeInfo: " + array.shapeInfoDataBuffer());

        System.out.println("Time elapsed: "+ (time2 - time1) );

        assertEquals(300, array.length());

        for (int x = 0; x < 300; x++) {
            assertEquals("Failed on x: " + x, 76800 - (array1.columns() - x) + 1 , array.getFloat(x), 0.01f);
        }
    }


    @Test
    public void testMax1_2() throws Exception {
        INDArray array1 = Nd4j.linspace(1, 7680000,7680000).reshape(2560, 3000);
/*
        for (int x = 0; x < 7680000; x++) {
            assertEquals(x+1, array1.getFloat(x), 0.001f);
        }
*/
        long time1 = System.currentTimeMillis();
        INDArray array = array1.max(1);
        long time2 = System.currentTimeMillis();

        System.out.println("Time elapsed: "+ (time2 - time1) );

        assertEquals(2560, array.length());

        //System.out.println("Array: " + array);
        for (int x = 0; x < 2560; x++) {
            assertEquals("Failed on x:" + x,(x + 1) * 3000, array.getFloat(x), 0.01f);
        }
    }
}
