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
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.Eps;
import org.nd4j.linalg.api.ops.util.PrintVariable;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.executors.ExecutorServiceProvider;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.common.primitives.Pair;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * NDArrayTests for fortran ordering
 *
 * @author Adam Gibson
 */

@Slf4j
public class NDArrayTestsFortran extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarOps(Nd4jBackend backend) {
        INDArray n = Nd4j.create(Nd4j.ones(27).data(), new long[] {3, 3, 3});
        assertEquals(27d, n.length(), 1e-1);
        n.addi(Nd4j.scalar(1d));
        n.subi(Nd4j.scalar(1.0d));
        n.muli(Nd4j.scalar(1.0d));
        n.divi(Nd4j.scalar(1.0d));

        n = Nd4j.create(Nd4j.ones(27).data(), new long[] {3, 3, 3});
        assertEquals(27, n.sumNumber().doubleValue(), 1e-1);
        INDArray a = n.slice(2);
        assertEquals(true, Arrays.equals(new long[] {3, 3}, a.shape()));

    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testColumnMmul(Nd4jBackend backend) {
        DataBuffer data = Nd4j.linspace(1, 10, 18, DataType.FLOAT).data();
        INDArray x2 = Nd4j.create(data, new long[] {2, 3, 3});
        data = Nd4j.linspace(1, 12, 9, DataType.FLOAT).data();
        INDArray y2 = Nd4j.create(data, new long[] {3, 3});
        INDArray z2 = Nd4j.create(DataType.FLOAT, new long[] {3, 2}, 'f');
        z2.putColumn(0, y2.getColumn(0));
        z2.putColumn(1, y2.getColumn(1));
        INDArray nofOffset = Nd4j.create(DataType.FLOAT, new long[] {3, 3}, 'f');
        nofOffset.assign(x2.slice(0));
        assertEquals(nofOffset, x2.slice(0));

        INDArray slice = x2.slice(0);
        INDArray zeroOffsetResult = slice.mmul(z2);
        INDArray offsetResult = nofOffset.mmul(z2);
        assertEquals(zeroOffsetResult, offsetResult);


        INDArray slice1 = x2.slice(1);
        INDArray noOffset2 = Nd4j.create(DataType.FLOAT, slice1.shape());
        noOffset2.assign(slice1);
        assertEquals(slice1, noOffset2);

        INDArray noOffsetResult = noOffset2.mmul(z2);
        INDArray slice1OffsetResult = slice1.mmul(z2);

        assertEquals(noOffsetResult, slice1OffsetResult);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRowVectorGemm(Nd4jBackend backend) {
        INDArray linspace = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(1, -1).castTo(DataType.DOUBLE);
        INDArray other = Nd4j.linspace(1, 16, 16, DataType.DOUBLE).reshape(4, 4).castTo(DataType.DOUBLE);
        INDArray result = linspace.mmul(other);
        INDArray assertion = Nd4j.create(new double[] {30., 70., 110., 150.}, new int[]{1,4});
        assertEquals(assertion, result);
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRepmat(Nd4jBackend backend) {
        INDArray rowVector = Nd4j.create(1, 4);
        INDArray repmat = rowVector.repmat(4, 4);
        assertTrue(Arrays.equals(new long[] {4, 16}, repmat.shape()));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReadWrite() throws Exception {
        INDArray write = Nd4j.linspace(1, 4, 4, DataType.DOUBLE);
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(bos);
        Nd4j.write(write, dos);

        ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
        DataInputStream dis = new DataInputStream(bis);
        INDArray read = Nd4j.read(dis);
        assertEquals(write, read);

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReadWriteDouble() throws Exception {
        INDArray write = Nd4j.linspace(1, 4, 4, DataType.FLOAT);
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(bos);
        Nd4j.write(write, dos);

        ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
        DataInputStream dis = new DataInputStream(bis);
        INDArray read = Nd4j.read(dis);
        assertEquals(write, read);

    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultiThreading() throws Exception {
        ExecutorService ex = ExecutorServiceProvider.getExecutorService();

        List<Future<?>> list = new ArrayList<>(100);
        for (int i = 0; i < 100; i++) {
            Future<?> future = ex.submit(() -> {
                INDArray dot = Nd4j.linspace(1, 8, 8, DataType.DOUBLE);
//                    System.out.println(Transforms.sigmoid(dot));
                Transforms.sigmoid(dot);
            });
            list.add(future);
        }
        for (Future<?> future : list) {
            future.get(1, TimeUnit.MINUTES);
        }

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastingGenerated(Nd4jBackend backend) {
        int[][] broadcastShape = NDArrayCreationUtil.getRandomBroadCastShape(7, 6, 10);
        List<List<Pair<INDArray, String>>> broadCastList = new ArrayList<>(broadcastShape.length);
        for (int[] shape : broadcastShape) {
            List<Pair<INDArray, String>> arrShape = NDArrayCreationUtil.get6dPermutedWithShape(7, shape, DataType.DOUBLE);
            broadCastList.add(arrShape);
            broadCastList.add(NDArrayCreationUtil.get6dReshapedWithShape(7, shape, DataType.DOUBLE));
            broadCastList.add(NDArrayCreationUtil.getAll6dTestArraysWithShape(7, shape, DataType.DOUBLE));
        }

        for (List<Pair<INDArray, String>> b : broadCastList) {
            for (Pair<INDArray, String> val : b) {
                INDArray inputArrBroadcast = val.getFirst();
                val destShape = NDArrayCreationUtil.broadcastToShape(inputArrBroadcast.shape(), 7);
                INDArray output = inputArrBroadcast
                        .broadcast(NDArrayCreationUtil.broadcastToShape(inputArrBroadcast.shape(), 7));
                assertArrayEquals(destShape, output.shape());
            }
        }



    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadCasting(Nd4jBackend backend) {
        INDArray first = Nd4j.arange(0, 3).reshape(3, 1).castTo(DataType.DOUBLE);
        INDArray ret = first.broadcast(3, 4);
        INDArray testRet = Nd4j.create(new double[][] {{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
        assertEquals(testRet, ret);
        INDArray r = Nd4j.arange(0, 4).reshape(1, 4).castTo(DataType.DOUBLE);
        INDArray r2 = r.broadcast(4, 4);
        INDArray testR2 = Nd4j.create(new double[][] {{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}});
        assertEquals(testR2, r2);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testOneTensor(Nd4jBackend backend) {
        INDArray arr = Nd4j.ones(1, 1, 1, 1, 1, 1, 1);
        INDArray matrixToBroadcast = Nd4j.ones(1, 1);
        assertEquals(matrixToBroadcast.broadcast(arr.shape()), arr);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSortWithIndicesDescending(Nd4jBackend backend) {
        INDArray toSort = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        //indices,data
        INDArray[] sorted = Nd4j.sortWithIndices(toSort.dup(), 1, false);
        INDArray sorted2 = Nd4j.sort(toSort.dup(), 1, false);
        assertEquals(sorted[1], sorted2);
        INDArray shouldIndex = Nd4j.create(new double[] {1, 1, 0, 0}, new long[] {2, 2});
        assertEquals(shouldIndex, sorted[0],getFailureMessage());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSortDeadlock(Nd4jBackend backend) {
        val toSort = Nd4j.linspace(DataType.DOUBLE, 1, 32*768, 1).reshape(32, 768);

        val sorted = Nd4j.sort(toSort.dup(), 1, false);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSortWithIndices(Nd4jBackend backend) {
        INDArray toSort = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        //indices,data
        INDArray[] sorted = Nd4j.sortWithIndices(toSort.dup(), 1, true);
        INDArray sorted2 = Nd4j.sort(toSort.dup(), 1, true);
        assertEquals(sorted[1], sorted2);
        INDArray shouldIndex = Nd4j.create(new double[] {0, 0, 1, 1}, new long[] {2, 2});
        assertEquals(shouldIndex, sorted[0],getFailureMessage());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNd4jSortScalar(Nd4jBackend backend) {
        INDArray linspace = Nd4j.linspace(1, 8, 8, DataType.DOUBLE).reshape(1, -1);
        INDArray sorted = Nd4j.sort(linspace, 1, false);
//        System.out.println(sorted);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSwapAxesFortranOrder(Nd4jBackend backend) {
        INDArray n = Nd4j.create(Nd4j.linspace(1, 30, 30, DataType.DOUBLE).data(), new long[] {3, 5, 2}).castTo(DataType.DOUBLE);
        for (int i = 0; i < n.slices(); i++) {
            INDArray nSlice = n.slice(i);
            for (int j = 0; j < nSlice.slices(); j++) {
                INDArray sliceJ = nSlice.slice(j);
//                System.out.println(sliceJ);
            }
//            System.out.println(nSlice);
        }
        INDArray slice = n.swapAxes(2, 1);
        INDArray assertion = Nd4j.create(new double[] {1, 4, 7, 10, 13});
        INDArray test = slice.slice(0).slice(0);
        assertEquals(assertion, test);
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDimShuffle(Nd4jBackend backend) {
        INDArray n = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray twoOneTwo = n.dimShuffle(new Object[] {0, 'x', 1}, new int[] {0, 1}, new boolean[] {false, false});
        assertTrue(Arrays.equals(new long[] {2, 1, 2}, twoOneTwo.shape()));

        INDArray reverse = n.dimShuffle(new Object[] {1, 'x', 0}, new int[] {1, 0}, new boolean[] {false, false});
        assertTrue(Arrays.equals(new long[] {2, 1, 2}, reverse.shape()));

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetVsGetScalar(Nd4jBackend backend) {
        INDArray a = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        float element = a.getFloat(0, 1);
        double element2 = a.getDouble(0, 1);
        assertEquals(element, element2, 1e-1);
        INDArray a2 = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        float element23 = a2.getFloat(0, 1);
        double element22 = a2.getDouble(0, 1);
        assertEquals(element23, element22, 1e-1);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDivide(Nd4jBackend backend) {
        INDArray two = Nd4j.create(new float[] {2, 2, 2, 2});
        INDArray div = two.div(two);
        assertEquals( Nd4j.ones(DataType.FLOAT, 4), div,getFailureMessage());

        INDArray half = Nd4j.create(new float[] {0.5f, 0.5f, 0.5f, 0.5f}, new long[] {2, 2});
        INDArray divi = Nd4j.create(new float[] {0.3f, 0.6f, 0.9f, 0.1f}, new long[] {2, 2});
        INDArray assertion = Nd4j.create(new float[] {1.6666666f, 0.8333333f, 0.5555556f, 5}, new long[] {2, 2});
        INDArray result = half.div(divi);
        assertEquals( assertion, result,getFailureMessage());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSigmoid(Nd4jBackend backend) {
        INDArray n = Nd4j.create(new float[] {1, 2, 3, 4});
        INDArray assertion = Nd4j.create(new float[] {0.73105858f, 0.88079708f, 0.95257413f, 0.98201379f});
        INDArray sigmoid = Transforms.sigmoid(n, false);
        assertEquals( assertion, sigmoid,getFailureMessage());

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNeg(Nd4jBackend backend) {
        INDArray n = Nd4j.create(new float[] {1, 2, 3, 4});
        INDArray assertion = Nd4j.create(new float[] {-1, -2, -3, -4});
        INDArray neg = Transforms.neg(n);
        assertEquals(assertion, neg,getFailureMessage());

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCosineSim(Nd4jBackend backend) {
        INDArray vec1 = Nd4j.create(new double[] {1, 2, 3, 4});
        INDArray vec2 = Nd4j.create(new double[] {1, 2, 3, 4});
        double sim = Transforms.cosineSim(vec1, vec2);
        assertEquals(1, sim, 1e-1,getFailureMessage());

        INDArray vec3 = Nd4j.create(new float[] {0.2f, 0.3f, 0.4f, 0.5f});
        INDArray vec4 = Nd4j.create(new float[] {0.6f, 0.7f, 0.8f, 0.9f});
        sim = Transforms.cosineSim(vec3, vec4);
        assertEquals(0.98, sim, 1e-1,getFailureMessage());

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testExp(Nd4jBackend backend) {
        INDArray n = Nd4j.create(new double[] {1, 2, 3, 4});
        INDArray assertion = Nd4j.create(new double[] {2.71828183f, 7.3890561f, 20.08553692f, 54.59815003f});
        INDArray exped = Transforms.exp(n);
        assertEquals(assertion, exped);
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalar(Nd4jBackend backend) {
        INDArray a = Nd4j.scalar(1.0f);
        assertEquals(true, a.isScalar());

        INDArray n = Nd4j.create(new float[] {1.0f}, new long[0]);
        assertEquals(n, a);
        assertTrue(n.isScalar());
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWrap(Nd4jBackend backend) {
        int[] shape = {2, 4};
        INDArray d = Nd4j.linspace(1, 8, 8, DataType.DOUBLE).reshape(shape[0], shape[1]);
        INDArray n = d;
        assertEquals(d.rows(), n.rows());
        assertEquals(d.columns(), n.columns());

        INDArray vector = Nd4j.linspace(1, 3, 3, DataType.DOUBLE);
        INDArray testVector = vector;
        for (int i = 0; i < vector.length(); i++)
            assertEquals(vector.getDouble(i), testVector.getDouble(i), 1e-1);
        assertEquals(3, testVector.length());
        assertEquals(true, testVector.isVector());
        assertEquals(true, Shape.shapeEquals(new long[] {3}, testVector.shape()));

        INDArray row12 = Nd4j.linspace(1, 2, 2, DataType.DOUBLE).reshape(2, 1);
        INDArray row22 = Nd4j.linspace(3, 4, 2, DataType.DOUBLE).reshape(1, 2);

        assertEquals(row12.rows(), 2);
        assertEquals(row12.columns(), 1);
        assertEquals(row22.rows(), 1);
        assertEquals(row22.columns(), 2);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetRowFortran(Nd4jBackend backend) {
        INDArray n = Nd4j.create(Nd4j.linspace(1, 4, 4, DataType.FLOAT).data(), new long[] {2, 2});
        INDArray column = Nd4j.create(new float[] {1, 3});
        INDArray column2 = Nd4j.create(new float[] {2, 4});
        INDArray testColumn = n.getRow(0);
        INDArray testColumn1 = n.getRow(1);
        assertEquals(column, testColumn);
        assertEquals(column2, testColumn1);


    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetColumnFortran(Nd4jBackend backend) {
        INDArray n = Nd4j.create(Nd4j.linspace(1, 4, 4, DataType.DOUBLE).data(), new long[] {2, 2});
        INDArray column = Nd4j.create(new double[] {1, 2});
        INDArray column2 = Nd4j.create(new double[] {3, 4});
        INDArray testColumn = n.getColumn(0);
        INDArray testColumn1 = n.getColumn(1);
//        log.info("testColumn shape: {}", Arrays.toString(testColumn.shapeInfoDataBuffer().asInt()));
        assertEquals(column, testColumn);
        assertEquals(column2, testColumn1);

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetColumns(Nd4jBackend backend) {
        INDArray matrix = Nd4j.linspace(1, 6, 6, DataType.DOUBLE).reshape(2, 3).castTo(DataType.DOUBLE);
//        log.info("Original: {}", matrix);
        INDArray matrixGet = matrix.getColumns(1, 2);
        INDArray matrixAssertion = Nd4j.create(new double[][] {{3, 5}, {4, 6}});
//        log.info("order A: {}", Arrays.toString(matrixAssertion.shapeInfoDataBuffer().asInt()));
//        log.info("order B: {}", Arrays.toString(matrixGet.shapeInfoDataBuffer().asInt()));
//        log.info("data A: {}", Arrays.toString(matrixAssertion.data().asFloat()));
//        log.info("data B: {}", Arrays.toString(matrixGet.data().asFloat()));
        assertEquals(matrixAssertion, matrixGet);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVectorInit(Nd4jBackend backend) {
        DataBuffer data = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).data();
        INDArray arr = Nd4j.create(data, new long[] {1, 4});
        assertEquals(true, arr.isRowVector());
        INDArray arr2 = Nd4j.create(data, new long[] {1, 4});
        assertEquals(true, arr2.isRowVector());

        INDArray columnVector = Nd4j.create(data, new long[] {4, 1});
        assertEquals(true, columnVector.isColumnVector());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAssignOffset(Nd4jBackend backend) {
        INDArray arr = Nd4j.ones(5, 5);
        INDArray row = arr.slice(1);
        row.assign(1);
        assertEquals(Nd4j.ones(5), row);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testColumns(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(new long[] {3, 2}).castTo(DataType.DOUBLE);
        INDArray column = Nd4j.create(new double[] {1, 2, 3});
        arr.putColumn(0, column);

        INDArray firstColumn = arr.getColumn(0);

        assertEquals(column, firstColumn);


        INDArray column1 = Nd4j.create(new double[] {4, 5, 6});
        arr.putColumn(1, column1);
        INDArray testRow1 = arr.getColumn(1);
        assertEquals(column1, testRow1);


        INDArray evenArr = Nd4j.create(new double[] {1, 2, 3, 4}, new long[] {2, 2});
        INDArray put = Nd4j.create(new double[] {5, 6});
        evenArr.putColumn(1, put);
        INDArray testColumn = evenArr.getColumn(1);
        assertEquals(put, testColumn);


        INDArray n = Nd4j.create(Nd4j.linspace(1, 4, 4, DataType.DOUBLE).data(), new long[] {2, 2}).castTo(DataType.DOUBLE);
        INDArray column23 = n.getColumn(0);
        INDArray column12 = Nd4j.create(new double[] {1, 2});
        assertEquals(column23, column12);


        INDArray column0 = n.getColumn(1);
        INDArray column01 = Nd4j.create(new double[] {3, 4});
        assertEquals(column0, column01);


    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPutRow(Nd4jBackend backend) {
        INDArray d = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray n = d.dup();

        //works fine according to matlab, let's go with it..
        //reproduce with:  A = newShapeNoCopy(linspace(1,4,4),[2 2 ]);
        //A(1,2) % 1 index based
        float nFirst = 3;
        float dFirst = d.getFloat(0, 1);
        assertEquals(nFirst, dFirst, 1e-1);
        assertEquals(d, n);
        assertEquals(true, Arrays.equals(new long[] {2, 2}, n.shape()));

        INDArray newRow = Nd4j.linspace(5, 6, 2, DataType.DOUBLE);
        n.putRow(0, newRow);
        d.putRow(0, newRow);


        INDArray testRow = n.getRow(0);
        assertEquals(newRow.length(), testRow.length());
        assertEquals(true, Shape.shapeEquals(new long[] {1, 2}, testRow.shape()));


        INDArray nLast = Nd4j.create(Nd4j.linspace(1, 4, 4, DataType.DOUBLE).data(), new long[] {2, 2}).castTo(DataType.DOUBLE);
        INDArray row = nLast.getRow(1);
        INDArray row1 = Nd4j.create(new double[] {2, 4});
        assertEquals(row, row1);


        INDArray arr = Nd4j.create(new long[] {3, 2}).castTo(DataType.DOUBLE);
        INDArray evenRow = Nd4j.create(new double[] {1, 2});
        arr.putRow(0, evenRow);
        INDArray firstRow = arr.getRow(0);
        assertEquals(true, Shape.shapeEquals(new long[] {1, 2}, firstRow.shape()));
        INDArray testRowEven = arr.getRow(0);
        assertEquals(evenRow, testRowEven);


        INDArray row12 = Nd4j.create(new double[] {5, 6});
        arr.putRow(1, row12);
        assertEquals(true, Shape.shapeEquals(new long[] {1, 2}, arr.getRow(0).shape()));
        INDArray testRow1 = arr.getRow(1);
        assertEquals(row12, testRow1);


        INDArray multiSliceTest = Nd4j.create(Nd4j.linspace(1, 16, 16, DataType.DOUBLE).data(), new long[] {4, 2, 2}).castTo(DataType.DOUBLE);
        INDArray test = Nd4j.create(new double[] {2, 10});
        INDArray test2 = Nd4j.create(new double[] {6, 14});

        INDArray multiSliceRow1 = multiSliceTest.slice(1).getRow(0);
        INDArray multiSliceRow2 = multiSliceTest.slice(1).getRow(1);

        assertEquals(test, multiSliceRow1);
        assertEquals(test2, multiSliceRow2);
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testInplaceTranspose(Nd4jBackend backend) {
        INDArray test = Nd4j.rand(3, 4);
        INDArray orig = test.dup();
        INDArray transposei = test.transposei();

        for (int i = 0; i < orig.rows(); i++) {
            for (int j = 0; j < orig.columns(); j++) {
                assertEquals(orig.getDouble(i, j), transposei.getDouble(j, i), 1e-1);
            }
        }
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMmulF(Nd4jBackend backend) {

        DataBuffer data = Nd4j.linspace(1, 10, 10, DataType.DOUBLE).data();
        INDArray n = Nd4j.create(data, new long[] {1, 10});
        INDArray transposed = n.transpose();
        assertEquals(true, n.isRowVector());
        assertEquals(true, transposed.isColumnVector());


        INDArray innerProduct = n.mmul(transposed);

        INDArray scalar = Nd4j.scalar(385.0).reshape(1,1);
        assertEquals(scalar, innerProduct,getFailureMessage());
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRowsColumns(Nd4jBackend backend) {
        DataBuffer data = Nd4j.linspace(1, 6, 6, DataType.DOUBLE).data();
        INDArray rows = Nd4j.create(data, new long[] {2, 3});
        assertEquals(2, rows.rows());
        assertEquals(3, rows.columns());

        INDArray columnVector = Nd4j.create(data, new long[] {6, 1});
        assertEquals(6, columnVector.rows());
        assertEquals(1, columnVector.columns());
        INDArray rowVector = Nd4j.create(data, new long[] {1, 6});
        assertEquals(1, rowVector.rows());
        assertEquals(6, rowVector.columns());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTranspose(Nd4jBackend backend) {
        INDArray n = Nd4j.create(Nd4j.ones(100).castTo(DataType.DOUBLE).data(), new long[] {5, 5, 4});
        INDArray transpose = n.transpose();
        assertEquals(n.length(), transpose.length());
        assertEquals(true, Arrays.equals(new long[] {4, 5, 5}, transpose.shape()));

        INDArray rowVector = Nd4j.linspace(1, 10, 10, DataType.DOUBLE).reshape(1, -1);
        assertTrue(rowVector.isRowVector());
        INDArray columnVector = rowVector.transpose();
        assertTrue(columnVector.isColumnVector());


        INDArray linspaced = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray transposed = Nd4j.create(new double[] {1, 3, 2, 4}, new long[] {2, 2});
        assertEquals(transposed, linspaced.transpose());

        linspaced = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        //fortran ordered
        INDArray transposed2 = Nd4j.create(new double[] {1, 3, 2, 4}, new long[] {2, 2});
        transposed = linspaced.transpose();
        assertEquals(transposed, transposed2);


    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAddMatrix(Nd4jBackend backend) {
        INDArray five = Nd4j.ones(5);
        five.addi(five.dup());
        INDArray twos = Nd4j.valueArrayOf(5, 2);
        assertEquals(twos, five,getFailureMessage());

    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMMul(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(new double[][] {{1, 2, 3}, {4, 5, 6}});

        INDArray assertion = Nd4j.create(new double[][] {{14, 32}, {32, 77}});

        INDArray test = arr.mmul(arr.transpose());
        assertEquals(assertion, test,getFailureMessage());

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPutSlice(Nd4jBackend backend) {
        INDArray n = Nd4j.linspace(1, 27, 27, DataType.DOUBLE).reshape(3, 3, 3);
        INDArray newSlice = Nd4j.create(DataType.DOUBLE, 3, 3);
        Nd4j.exec(new PrintVariable(newSlice));
        log.info("Slice: {}", newSlice);
        n.putSlice(0, newSlice);
        assertEquals( newSlice, n.slice(0),getFailureMessage());

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRowVectorMultipleIndices(Nd4jBackend backend) {
        INDArray linear = Nd4j.create(DataType.DOUBLE, 1, 4);
        linear.putScalar(new long[] {0, 1}, 1);
        assertEquals(linear.getDouble(0, 1), 1, 1e-1,getFailureMessage());
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDim1(Nd4jBackend backend) {
        INDArray sum = Nd4j.linspace(1, 2, 2, DataType.DOUBLE).reshape(2, 1);
        INDArray same = sum.dup();
        assertEquals(same.sum(1), sum.reshape(2));
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEps(Nd4jBackend backend) {
        val ones = Nd4j.ones(5);
        val res = Nd4j.createUninitialized(DataType.BOOL, 5);
        assertTrue(Nd4j.getExecutioner().exec(new Eps(ones, ones, res)).all());
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLogDouble(Nd4jBackend backend) {
        INDArray linspace = Nd4j.linspace(1, 6, 6, DataType.DOUBLE).castTo(DataType.DOUBLE);
        INDArray log = Transforms.log(linspace);
        INDArray assertion = Nd4j.create(new double[] {0, 0.6931471805599453, 1.0986122886681098, 1.3862943611198906, 1.6094379124341005, 1.791759469228055});
        assertEquals(assertion, log);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVectorSum(Nd4jBackend backend) {
        INDArray lin = Nd4j.linspace(1, 4, 4, DataType.DOUBLE);
        assertEquals(10.0, lin.sumNumber().doubleValue(), 1e-1);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVectorSum2(Nd4jBackend backend) {
        INDArray lin = Nd4j.create(new double[] {1, 2, 3, 4});
        assertEquals(10.0, lin.sumNumber().doubleValue(), 1e-1);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVectorSum3(Nd4jBackend backend) {
        INDArray lin = Nd4j.create(new double[] {1, 2, 3, 4});
        INDArray lin2 = Nd4j.create(new double[] {1, 2, 3, 4});
        assertEquals(lin, lin2);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSmallSum(Nd4jBackend backend) {
        INDArray base = Nd4j.create(new double[] {5.843333333333335, 3.0540000000000007});
        base.addi(1e-12);
        INDArray assertion = Nd4j.create(new double[] {5.84333433, 3.054001});
        assertEquals(assertion, base);

    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPermute(Nd4jBackend backend) {
        INDArray n = Nd4j.create(Nd4j.linspace(1, 20, 20, DataType.DOUBLE).data(), new long[] {5, 4});
        INDArray transpose = n.transpose();
        INDArray permute = n.permute(1, 0);
        assertEquals(permute, transpose);
        assertEquals(transpose.length(), permute.length(), 1e-1);


        INDArray toPermute = Nd4j.create(Nd4j.linspace(0, 7, 8, DataType.DOUBLE).data(), new long[] {2, 2, 2});
        INDArray permuted = toPermute.dup().permute(2, 1, 0);
        boolean eq = toPermute.equals(permuted);
        assertNotEquals(toPermute, permuted);

        INDArray permuteOther = toPermute.permute(1, 2, 0);
        for (int i = 0; i < permuteOther.slices(); i++) {
            INDArray toPermutesliceI = toPermute.slice(i);
            INDArray permuteOtherSliceI = permuteOther.slice(i);
            permuteOtherSliceI.toString();
            assertNotEquals(toPermutesliceI, permuteOtherSliceI);
        }
        assertArrayEquals(permuteOther.shape(), toPermute.shape());
        assertNotEquals(toPermute, permuteOther);


    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAppendBias(Nd4jBackend backend) {
        INDArray rand = Nd4j.linspace(1, 25, 25, DataType.DOUBLE).reshape(1, -1).transpose();
        INDArray test = Nd4j.appendBias(rand);
        INDArray assertion = Nd4j.toFlattened(rand, Nd4j.scalar(DataType.DOUBLE, 1.0)).reshape(-1, 1);
        assertEquals(assertion, test);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRand(Nd4jBackend backend) {
        INDArray rand = Nd4j.randn(5, 5);
        Nd4j.getDistributions().createUniform(0.4, 4).sample(5);
        Nd4j.getDistributions().createNormal(1, 5).sample(10);
        //Nd4j.getDistributions().createBinomial(5, 1.0).sample(new long[]{5, 5});
        //Nd4j.getDistributions().createBinomial(1, Nd4j.ones(5, 5)).sample(rand.shape());
        Nd4j.getDistributions().createNormal(rand, 1).sample(rand.shape());
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIdentity(Nd4jBackend backend) {
        INDArray eye = Nd4j.eye(5);
        assertTrue(Arrays.equals(new long[] {5, 5}, eye.shape()));
        eye = Nd4j.eye(5);
        assertTrue(Arrays.equals(new long[] {5, 5}, eye.shape()));


    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testColumnVectorOpsFortran(Nd4jBackend backend) {
        INDArray twoByTwo = Nd4j.create(new float[] {1, 2, 3, 4}, new long[] {2, 2});
        INDArray toAdd = Nd4j.create(new float[] {1, 2}, new long[] {2, 1});
        twoByTwo.addiColumnVector(toAdd);
        INDArray assertion = Nd4j.create(new float[] {2, 4, 4, 6}, new long[] {2, 2});
        assertEquals(assertion, twoByTwo);
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRSubi(Nd4jBackend backend) {
        INDArray n2 = Nd4j.ones(2);
        INDArray n2Assertion = Nd4j.zeros(2);
        INDArray nRsubi = n2.rsubi(1);
        assertEquals(n2Assertion, nRsubi);
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAssign(Nd4jBackend backend) {
        INDArray vector = Nd4j.linspace(1, 5, 5, DataType.DOUBLE);
        vector.assign(1);
        assertEquals(Nd4j.ones(5).castTo(DataType.DOUBLE), vector);
        INDArray twos = Nd4j.ones(2, 2);
        INDArray rand = Nd4j.rand(2, 2);
        twos.assign(rand);
        assertEquals(rand, twos);

        INDArray tensor = Nd4j.rand(DataType.DOUBLE, 3, 3, 3);
        INDArray ones = Nd4j.ones(3, 3, 3).castTo(DataType.DOUBLE);
        assertTrue(Arrays.equals(tensor.shape(), ones.shape()));
        ones.assign(tensor);
        assertEquals(tensor, ones);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAddScalar(Nd4jBackend backend) {
        INDArray div = Nd4j.valueArrayOf(new long[] {1, 4}, 4.0);
        INDArray rdiv = div.add(1);
        INDArray answer = Nd4j.valueArrayOf(new long[] {1, 4}, 5.0);
        assertEquals(answer, rdiv);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRdivScalar(Nd4jBackend backend) {
        INDArray div = Nd4j.valueArrayOf(new long[] {1, 4}, 4.0);
        INDArray rdiv = div.rdiv(1);
        INDArray answer = Nd4j.valueArrayOf(new long[] {1, 4}, 0.25);
        assertEquals(rdiv, answer);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRDivi(Nd4jBackend backend) {
        INDArray n2 = Nd4j.valueArrayOf(new long[] {1, 2}, 4.0);
        INDArray n2Assertion = Nd4j.valueArrayOf(new long[] {1, 2}, 0.5);
        INDArray nRsubi = n2.rdivi(2);
        assertEquals(n2Assertion, nRsubi);
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNumVectorsAlongDimension(Nd4jBackend backend) {
        INDArray arr = Nd4j.linspace(1, 24, 24, DataType.DOUBLE).reshape(4, 3, 2);
        assertEquals(12, arr.vectorsAlongDimension(2));
    }



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadCast(Nd4jBackend backend) {
        INDArray n = Nd4j.linspace(1, 4, 4, DataType.DOUBLE);
        INDArray broadCasted = n.broadcast(5, 4);
        for (int i = 0; i < broadCasted.rows(); i++) {
            assertEquals(n, broadCasted.getRow(i));
        }

        INDArray broadCast2 = broadCasted.getRow(0).broadcast(5, 4);
        assertEquals(broadCasted, broadCast2);


        INDArray columnBroadcast = n.reshape(4,1).broadcast(4, 5);
        for (int i = 0; i < columnBroadcast.columns(); i++) {
            assertEquals(columnBroadcast.getColumn(i), n.reshape(4));
        }

        INDArray fourD = Nd4j.create(1, 2, 1, 1);
        INDArray broadCasted3 = fourD.broadcast(1, 2, 36, 36);
        assertTrue(Arrays.equals(new long[] {1, 2, 36, 36}, broadCasted3.shape()));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMatrix(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(new double[] {1, 2, 3, 4}, new long[] {2, 2});
        INDArray brr = Nd4j.create(new double[] {5, 6}, new long[] {2});
        INDArray row = arr.getRow(0);
        row.subi(brr);
        assertEquals(Nd4j.create(new double[] {-4, -3}), arr.getRow(0));

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPutRowGetRowOrdering(Nd4jBackend backend) {
        INDArray row1 = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray put = Nd4j.create(new double[] {5, 6});
        row1.putRow(1, put);

//        System.out.println(row1);
        row1.toString();

        INDArray row1Fortran = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray putFortran = Nd4j.create(new double[] {5, 6});
        row1Fortran.putRow(1, putFortran);
        assertEquals(row1, row1Fortran);
        INDArray row1CTest = row1.getRow(1);
        INDArray row1FortranTest = row1Fortran.getRow(1);
        assertEquals(row1CTest, row1FortranTest);



    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSumWithRow1(Nd4jBackend backend) {
        //Works:
        INDArray array2d = Nd4j.ones(1, 10);
        array2d.sum(0); //OK
        array2d.sum(1); //OK

        INDArray array3d = Nd4j.ones(1, 10, 10);
        array3d.sum(0); //OK
        array3d.sum(1); //OK
        array3d.sum(2); //java.lang.IllegalArgumentException: Illegal index 100 derived from 9 with offset of 10 and stride of 10

        INDArray array4d = Nd4j.ones(1, 10, 10, 10);
        INDArray sum40 = array4d.sum(0); //OK
        INDArray sum41 = array4d.sum(1); //OK
        INDArray sum42 = array4d.sum(2); //java.lang.IllegalArgumentException: Illegal index 1000 derived from 9 with offset of 910 and stride of 10
        INDArray sum43 = array4d.sum(3); //java.lang.IllegalArgumentException: Illegal index 1000 derived from 9 with offset of 100 and stride of 100

//        System.out.println("40: " + sum40.length());
//        System.out.println("41: " + sum41.length());
//        System.out.println("42: " + sum42.length());
//        System.out.println("43: " + sum43.length());

        INDArray array5d = Nd4j.ones(1, 10, 10, 10, 10);
        array5d.sum(0); //OK
        array5d.sum(1); //OK
        array5d.sum(2); //java.lang.IllegalArgumentException: Illegal index 10000 derived from 9 with offset of 9910 and stride of 10
        array5d.sum(3); //java.lang.IllegalArgumentException: Illegal index 10000 derived from 9 with offset of 9100 and stride of 100
        array5d.sum(4); //java.lang.IllegalArgumentException: Illegal index 10000 derived from 9 with offset of 1000 and stride of 1000
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSumWithRow2(Nd4jBackend backend) {
        //All sums in this method execute without exceptions.
        INDArray array3d = Nd4j.ones(2, 10, 10);
        array3d.sum(0);
        array3d.sum(1);
        array3d.sum(2);

        INDArray array4d = Nd4j.ones(2, 10, 10, 10);
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


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPutRowFortran(Nd4jBackend backend) {
        INDArray row1 = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2).castTo(DataType.DOUBLE);
        INDArray put = Nd4j.create(new double[] {5, 6});
        row1.putRow(1, put);

        INDArray row1Fortran = Nd4j.create(new double[][] {{1, 3}, {2, 4}});
        INDArray putFortran = Nd4j.create(new double[] {5, 6});
        row1Fortran.putRow(1, putFortran);
        assertEquals(row1, row1Fortran);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testElementWiseOps(Nd4jBackend backend) {
        INDArray n1 = Nd4j.scalar(1);
        INDArray n2 = Nd4j.scalar(2);
        INDArray nClone = n1.add(n2);
        assertEquals(Nd4j.scalar(3), nClone);
        INDArray n1PlusN2 = n1.add(n2);
        assertFalse(n1PlusN2.equals(n1),getFailureMessage());

        INDArray n3 = Nd4j.scalar(3.0);
        INDArray n4 = Nd4j.scalar(4.0);
        INDArray subbed = n4.sub(n3);
        INDArray mulled = n4.mul(n3);
        INDArray div = n4.div(n3);

        assertFalse(subbed.equals(n4));
        assertFalse(mulled.equals(n4));
        assertEquals(Nd4j.scalar(1.0), subbed);
        assertEquals(Nd4j.scalar(12.0), mulled);
        assertEquals(Nd4j.scalar(1.333333333333333333333), div);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRollAxis(Nd4jBackend backend) {
        INDArray toRoll = Nd4j.ones(3, 4, 5, 6);
        assertArrayEquals(new long[] {3, 6, 4, 5}, Nd4j.rollAxis(toRoll, 3, 1).shape());
        val shape = Nd4j.rollAxis(toRoll, 3).shape();
        assertArrayEquals(new long[] {6, 3, 4, 5}, shape);
    }

    @Test
    @Disabled
    public void testTensorDot(Nd4jBackend backend) {
        INDArray oneThroughSixty = Nd4j.arange(60).reshape('f', 3, 4, 5).castTo(DataType.DOUBLE);
        INDArray oneThroughTwentyFour = Nd4j.arange(24).reshape('f', 4, 3, 2).castTo(DataType.DOUBLE);
        INDArray result = Nd4j.tensorMmul(oneThroughSixty, oneThroughTwentyFour, new int[][] {{1, 0}, {0, 1}});
        assertArrayEquals(new long[] {5, 2}, result.shape());
        INDArray assertion = Nd4j.create(new double[][] {{440., 1232.}, {1232., 3752.}, {2024., 6272.}, {2816., 8792.},
                {3608., 11312.}});
        assertEquals(assertion, result);

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNegativeShape(Nd4jBackend backend) {
        INDArray linspace = Nd4j.linspace(1, 4, 4, DataType.DOUBLE);
        INDArray reshaped = linspace.reshape(-1, 2);
        assertArrayEquals(new long[] {2, 2}, reshaped.shape());

        INDArray linspace6 = Nd4j.linspace(1, 6, 6, DataType.DOUBLE);
        INDArray reshaped2 = linspace6.reshape(-1, 3);
        assertArrayEquals(new long[] {2, 3}, reshaped2.shape());

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetColumnGetRow(Nd4jBackend backend) {
        INDArray row = Nd4j.ones(1, 5);
        for (int i = 0; i < 5; i++) {
            INDArray col = row.getColumn(i);
            assertArrayEquals(col.shape(), new long[] {1});
        }

        INDArray col = Nd4j.ones(5, 1);
        for (int i = 0; i < 5; i++) {
            INDArray row2 = col.getRow(i);
            assertArrayEquals(new long[] {1}, row2.shape());
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDupAndDupWithOrder(Nd4jBackend backend) {
        List<Pair<INDArray, String>> testInputs = NDArrayCreationUtil.getAllTestMatricesWithShape(4, 5, 123, DataType.DOUBLE);
        int count = 0;
        for (Pair<INDArray, String> pair : testInputs) {
            String msg = pair.getSecond();
            INDArray in = pair.getFirst();
//            System.out.println("Count " + count);
            INDArray dup = in.dup();
            INDArray dupc = in.dup('c');
            INDArray dupf = in.dup('f');

            assertEquals(in, dup,msg);
            assertEquals(dup.ordering(), (char) Nd4j.order(),msg);
            assertEquals(dupc.ordering(), 'c',msg);
            assertEquals(dupf.ordering(), 'f',msg);
            assertEquals( in, dupc,msg);
            assertEquals(in, dupf,msg);
            count++;
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testToOffsetZeroCopy(Nd4jBackend backend) {
        List<Pair<INDArray, String>> testInputs = NDArrayCreationUtil.getAllTestMatricesWithShape(4, 5, 123, DataType.DOUBLE);

        int cnt = 0;
        for (Pair<INDArray, String> pair : testInputs) {
            String msg = pair.getSecond();
            INDArray in = pair.getFirst();
            INDArray dup = Shape.toOffsetZeroCopy(in);
            INDArray dupc = Shape.toOffsetZeroCopy(in, 'c');
            INDArray dupf = Shape.toOffsetZeroCopy(in, 'f');
            INDArray dupany = Shape.toOffsetZeroCopyAnyOrder(in);

            assertEquals( in, dup,msg + ": " + cnt);
            assertEquals(in, dupc,msg);
            assertEquals(in, dupf,msg);
            assertEquals(dupc.ordering(), 'c',msg);
            assertEquals(dupf.ordering(), 'f',msg);
            assertEquals( in, dupany,msg);

            assertEquals(dup.offset(), 0);
            assertEquals(dupc.offset(), 0);
            assertEquals(dupf.offset(), 0);
            assertEquals(dupany.offset(), 0);
            assertEquals(dup.length(), dup.data().length());
            assertEquals(dupc.length(), dupc.data().length());
            assertEquals(dupf.length(), dupf.data().length());
            assertEquals(dupany.length(), dupany.data().length());
            cnt++;
        }
    }


    @Override
    public char ordering() {
        return 'f';
    }
}
