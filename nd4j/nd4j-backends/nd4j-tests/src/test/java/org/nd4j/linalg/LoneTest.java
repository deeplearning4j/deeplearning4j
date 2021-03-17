/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.lang3.RandomUtils;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Tanh;
import org.nd4j.linalg.api.ops.util.PrintVariable;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.common.primitives.Pair;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;


@Slf4j

public class LoneTest extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSoftmaxStability(Nd4jBackend backend) {
        INDArray input = Nd4j.create(new double[]{-0.75, 0.58, 0.42, 1.03, -0.61, 0.19, -0.37, -0.40, -1.42, -0.04}).reshape(1, -1).transpose();
//        System.out.println("Input transpose " + Shape.shapeToString(input.shapeInfo()));
        INDArray output = Nd4j.create(DataType.DOUBLE, 10, 1);
//        System.out.println("Element wise stride of output " + output.elementWiseStride());
        Nd4j.getExecutioner().exec(new SoftMax(input, output));
    }

    @Override
    public char ordering() {
        return 'c';
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFlattenedView(Nd4jBackend backend) {
        int rows = 8;
        int cols = 8;
        int dim2 = 4;
        int length = rows * cols;
        int length3d = rows * cols * dim2;

        INDArray first = Nd4j.linspace(1, length, length).reshape('c', rows, cols);
        INDArray second = Nd4j.create(DataType.DOUBLE, new long[]{rows, cols}, 'f').assign(first);
        INDArray third = Nd4j.linspace(1, length3d, length3d).reshape('c', rows, cols, dim2);
        first.addi(0.1);
        second.addi(0.2);
        third.addi(0.3);

        first = first.get(NDArrayIndex.interval(4, 8), NDArrayIndex.interval(0, 2, 8));
        for (int i = 0; i < first.tensorsAlongDimension(0); i++) {
//            System.out.println(first.tensorAlongDimension(i, 0));
            first.tensorAlongDimension(i, 0);
        }

        for (int i = 0; i < first.tensorsAlongDimension(1); i++) {
//            System.out.println(first.tensorAlongDimension(i, 1));
            first.tensorAlongDimension(i, 1);
        }
        second = second.get(NDArrayIndex.interval(3, 7), NDArrayIndex.all());
        third = third.permute(0, 2, 1);

        INDArray cAssertion = Nd4j.create(new double[]{33.10, 35.10, 37.10, 39.10, 41.10, 43.10, 45.10, 47.10, 49.10,
                51.10, 53.10, 55.10, 57.10, 59.10, 61.10, 63.10});
        INDArray fAssertion = Nd4j.create(new double[]{33.10, 41.10, 49.10, 57.10, 35.10, 43.10, 51.10, 59.10, 37.10,
                45.10, 53.10, 61.10, 39.10, 47.10, 55.10, 63.10});
        assertEquals(cAssertion, Nd4j.toFlattened('c', first));
        assertEquals(fAssertion, Nd4j.toFlattened('f', first));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIndexingColVec(Nd4jBackend backend) {
        int elements = 5;
        INDArray rowVector = Nd4j.linspace(1, elements, elements).reshape(1, elements);
        INDArray colVector = rowVector.transpose();
        int j;
        INDArray jj;
        for (int i = 0; i < elements; i++) {
            j = i + 1;
            assertEquals(i + 1,colVector.getRow(i).getInt(0));
            assertEquals(i + 1,rowVector.getColumn(i).getInt(0));
            assertEquals(i + 1,rowVector.get(NDArrayIndex.point(0), NDArrayIndex.interval(i, j)).getInt(0));
            assertEquals(i + 1,colVector.get(NDArrayIndex.interval(i, j), NDArrayIndex.point(0)).getInt(0));
//            System.out.println("Making sure index interval will not crash with begin/end vals...");
            jj = colVector.get(NDArrayIndex.interval(i, i + 1));
            jj = colVector.get(NDArrayIndex.interval(i, i + 1));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void concatScalarVectorIssue(Nd4jBackend backend) {
        //A bug was found when the first array that concat sees is a scalar and the rest vectors + scalars
        INDArray arr1 = Nd4j.create(1, 1);
        INDArray arr2 = Nd4j.create(1, 8);
        INDArray arr3 = Nd4j.create(1, 1);
        INDArray arr4 = Nd4j.concat(1, arr1, arr2, arr3);
        assertTrue(arr4.sumNumber().floatValue() <= Nd4j.EPS_THRESHOLD);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void reshapeTensorMmul(Nd4jBackend backend) {
        INDArray a = Nd4j.linspace(1, 2, 12).reshape(2, 3, 2);
        INDArray b = Nd4j.linspace(3, 4, 4).reshape(2, 2);
        int[][] axes = new int[2][];
        axes[0] = new int[]{0, 1};
        axes[1] = new int[]{0, 2};

        //this was throwing an exception
        INDArray c = Nd4j.tensorMmul(b, a, axes);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void maskWhenMerge(Nd4jBackend backend) {
        DataSet dsA = new DataSet(Nd4j.linspace(1, 15, 15).reshape(1, 3, 5), Nd4j.zeros(1, 3, 5));
        DataSet dsB = new DataSet(Nd4j.linspace(1, 9, 9).reshape(1, 3, 3), Nd4j.zeros(1, 3, 3));
        List<DataSet> dataSetList = new ArrayList<DataSet>();
        dataSetList.add(dsA);
        dataSetList.add(dsB);
        DataSet fullDataSet = DataSet.merge(dataSetList);
        assertTrue(fullDataSet.getFeaturesMaskArray() != null);

        DataSet fullDataSetCopy = fullDataSet.copy();
        assertTrue(fullDataSetCopy.getFeaturesMaskArray() != null);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRelu(Nd4jBackend backend) {
        INDArray aA = Nd4j.linspace(-3, 4, 8).reshape(2, 4);
        INDArray aD = Nd4j.linspace(-3, 4, 8).reshape(2, 4);
        INDArray b = Nd4j.getExecutioner().exec(new Tanh(aA));
        //Nd4j.getExecutioner().execAndReturn(new TanhDerivative(aD));
//        System.out.println(aA);
//        System.out.println(aD);
//        System.out.println(b);
    }

    @Test
    //broken at a threshold
    public void testArgMax(Nd4jBackend backend) {
        int max = 63;
        INDArray A = Nd4j.linspace(1, max, max).reshape(1, max);
        int currentArgMax = Nd4j.argMax(A).getInt(0);
        assertEquals(max - 1, currentArgMax);

        max = 64;
        A = Nd4j.linspace(1, max, max).reshape(1, max);
        currentArgMax = Nd4j.argMax(A).getInt(0);
//        System.out.println("Returned argMax is " + currentArgMax);
        assertEquals(max - 1, currentArgMax);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRPF(Nd4jBackend backend) {
        val array = Nd4j.createFromArray(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12).reshape(2, 2, 3);

        log.info("--------");

        val tad = array.tensorAlongDimension(1, 1, 2);
        Nd4j.exec(new PrintVariable(tad, false));
        log.info("TAD native shapeInfo: {}", tad.shapeInfoDataBuffer().asLong());
        log.info("TAD Java shapeInfo: {}", tad.shapeInfoJava());
        log.info("TAD:\n{}", tad);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcat3D_Vstack_C(Nd4jBackend backend) {
        val shape = new long[]{1, 1000, 20};

        List<INDArray> cArrays = new ArrayList<>();
        List<INDArray> fArrays = new ArrayList<>();

        for (int e = 0; e < 32; e++) {
            val arr = Nd4j.create(DataType.FLOAT, shape, 'c').assign(e);
            cArrays.add(arr);
            //            fArrays.add(cOrder.dup('f'));
        }

        Nd4j.getExecutioner().commit();

        val time1 = System.currentTimeMillis();
        val res = Nd4j.vstack(cArrays);
        val time2 = System.currentTimeMillis();

//        log.info("Time spent: {} ms", time2 - time1);

        for (int e = 0; e < 32; e++) {
            val tad = res.tensorAlongDimension(e, 1, 2);

            assertEquals((double) e, tad.meanNumber().doubleValue(), 1e-5,"Failed for TAD [" + e + "]");
            assertEquals((double) e, tad.getDouble(0), 1e-5);
        }
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetRow1(Nd4jBackend backend) {
        INDArray array = Nd4j.create(10000, 10000);

        //Thread.sleep(10000);

        int numTries = 1000;
        List<Long> times = new ArrayList<>();
        long time = 0;
        for (int i = 0; i < numTries; i++) {

            int idx = RandomUtils.nextInt(0, 10000);
            long time1 = System.nanoTime();
            array.getRow(idx);
            long time2 = System.nanoTime() - time1;

            times.add(time2);
            time += time2;
        }

        time /= numTries;

        Collections.sort(times);

//        log.info("p50: {}; avg: {};", times.get(times.size() / 2), time);
    }

    @Test()
    public void checkIllegalElementOps(Nd4jBackend backend) {
        assertThrows(Exception.class,() -> {
            INDArray A = Nd4j.linspace(1, 20, 20).reshape(4, 5);
            INDArray B = A.dup().reshape(2, 2, 5);

            //multiplication of arrays of different rank should throw exception
            INDArray C = A.mul(B);
        });

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void checkSliceofSlice(Nd4jBackend backend) {
        /*
            Issue 1: Slice of slice with c order and f order views are not equal

            Comment out assert and run then -> Issue 2: Index out of bound exception with certain shapes when accessing elements with getDouble() in f order
            (looks like problem is when rank-1==1) eg. 1,2,1 and 2,2,1
         */
        int[] ranksToCheck = new int[]{2, 3, 4, 5};
        for (int rank = 0; rank < ranksToCheck.length; rank++) {
//            log.info("\nRunning through rank " + ranksToCheck[rank]);
            List<Pair<INDArray, String>> allF = NDArrayCreationUtil.getTestMatricesWithVaryingShapes(ranksToCheck[rank], 'f', DataType.FLOAT);
            Iterator<Pair<INDArray, String>> iter = allF.iterator();
            while (iter.hasNext()) {
                Pair<INDArray, String> currentPair = iter.next();
                INDArray origArrayF = currentPair.getFirst();
                INDArray sameArrayC = origArrayF.dup('c');
//                log.info("\nLooping through slices for shape " + currentPair.getSecond());
//                log.info("\nOriginal array:\n" + origArrayF);
                origArrayF.toString();
                INDArray viewF = origArrayF.slice(0);
                INDArray viewC = sameArrayC.slice(0);
//                log.info("\nSlice 0, C order:\n" + viewC.toString());
//                log.info("\nSlice 0, F order:\n" + viewF.toString());
                viewC.toString();
                viewF.toString();
                for (int i = 0; i < viewF.slices(); i++) {
                    //assertEquals(viewF.slice(i),viewC.slice(i));
                    for (int j = 0; j < viewF.slice(i).length(); j++) {
                        //if (j>0) break;
//                        log.info("\nC order slice " + i + ", element 0 :" + viewC.slice(i).getDouble(j)); //C order is fine
//                        log.info("\nF order slice " + i + ", element 0 :" + viewF.slice(i).getDouble(j)); //throws index out of bound err on F order
                        viewC.slice(i).getDouble(j);
                        viewF.slice(i).getDouble(j);
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void checkWithReshape(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(1, 3);
        INDArray reshaped = arr.reshape('f', 3, 1);
        for (int i=0;i<reshaped.length();i++) {
//            log.info("C order element " + i + arr.getDouble(i));
//            log.info("F order element " + i + reshaped.getDouble(i));
            arr.getDouble(i);
            reshaped.getDouble(i);
        }
        for (int j=0;j<arr.slices();j++) {
            for (int k=0;k<arr.slice(j).length();k++) {
//                log.info("\nArr: slice " + j + " element " + k + " " + arr.slice(j).getDouble(k));
                arr.slice(j).getDouble(k);
            }
        }
        for (int j=0;j<reshaped.slices();j++) {
            for (int k=0;k<reshaped.slice(j).length();k++) {
//                log.info("\nReshaped: slice " + j + " element " + k + " " + reshaped.slice(j).getDouble(k));
                reshaped.slice(j).getDouble(k);
            }
        }
    }
}
