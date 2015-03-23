/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.api.test;


import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.SliceOp;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.Eps;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.Shape;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

/**
 * NDArrayTests
 *
 * @author Adam Gibson
 */
public abstract class NDArrayTests {
    private static Logger log = LoggerFactory.getLogger(NDArrayTests.class);
    private INDArray n = Nd4j.create(Nd4j.linspace(1, 8, 8).data(), new int[]{2, 2, 2});


    @Before
    public void before() {
        Nd4j.factory().setOrder('c');
    }

    @After
    public void after() {
        Nd4j.factory().setOrder('c');
    }

    @Test
    public void testScalarOps() {
        INDArray n = Nd4j.create(Nd4j.ones(27).data(), new int[]{3, 3, 3});
        assertEquals(27d, n.length(), 1e-1);
        n.checkDimensions(n.addi(Nd4j.scalar(1d)));
        n.checkDimensions(n.subi(Nd4j.scalar(1.0d)));
        n.checkDimensions(n.muli(Nd4j.scalar(1.0d)));
        n.checkDimensions(n.divi(Nd4j.scalar(1.0d)));

        n = Nd4j.create(Nd4j.ones(27).data(), new int[]{3, 3, 3});
        assertEquals(27, n.sum(Integer.MAX_VALUE).getDouble(0), 1e-1);
        INDArray a = n.slice(2);
        assertEquals(true, Arrays.equals(new int[]{3, 3}, a.shape()));
        n.data().destroy();

    }

    @Test
    public void testReadWrite() throws Exception {
        Nd4j.dtype = DataBuffer.FLOAT;
        INDArray write = Nd4j.linspace(1,4,4);
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(bos);
        Nd4j.write(write,dos);

        ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
        DataInputStream dis = new DataInputStream(bis);
        INDArray read = Nd4j.read(dis);
        assertEquals(write,read);

    }

    @Test
    public void testConcatScalars() {
        INDArray first = Nd4j.arange(0,1).reshape(1,1);
        INDArray second = Nd4j.arange(0,1).reshape(1,1);
        INDArray firstRet = Nd4j.concat(0,first,second);
        assertTrue(firstRet.isColumnVector());
        INDArray secondRet = Nd4j.concat(1,first,second);
        assertTrue(secondRet.isRowVector());


    }

    @Test
    public void testReadWriteDouble() throws Exception {
        Nd4j.dtype = DataBuffer.DOUBLE;
        INDArray write = Nd4j.linspace(1,4,4);
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(bos);
        Nd4j.write(write,dos);

        ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
        DataInputStream dis = new DataInputStream(bis);
        INDArray read = Nd4j.read(dis);
        assertEquals(write,read);

    }

    @Test
    public void testTheReaper() {
        INDArray create = Nd4j.create(5);
        log.info("Testing creation");
    }


    @Test
    public void testSubiRowVector() {
        INDArray oneThroughFour = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray row1 = oneThroughFour.getRow(1);
        oneThroughFour.subiRowVector(row1);
        INDArray result = Nd4j.create(new float[]{-2, -2, 0, 0}, new int[]{2, 2});
        assertEquals(result, oneThroughFour);
        result.data().destroy();
        oneThroughFour.data().destroy();

    }

    @Test
    public void testBroadCasting() {
        INDArray first = Nd4j.arange(0,3).reshape(3,1);
        INDArray ret = first.broadcast(3,4);
        INDArray testRet = Nd4j.create(new double[][]{
                {0,0,0,0},
                {1,1,1,1},
                {2,2,2,2}
        });
        assertEquals(testRet,ret);
        INDArray r = Nd4j.arange(0,4).reshape(1,4);
        INDArray r2 = r.broadcast(4,4);
        INDArray testR2 = Nd4j.create(new double[][]{
                {0, 1, 2, 3},
                {0, 1, 2, 3},
                {0, 1, 2, 3},
                {0, 1, 2, 3}
        });
        assertEquals(testR2,r2);

    }




    @Test
    public void testSort() {
        INDArray toSort = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray ascending = Nd4j.sort(toSort.dup(), 1, true);
        //rows already already sorted
        assertEquals(toSort, ascending);

        INDArray columnSorted = Nd4j.create(new float[]{2, 1, 4, 3}, new int[]{2, 2});
        INDArray sorted = Nd4j.sort(toSort.dup(), 1, false);
        assertEquals(columnSorted, sorted);
        toSort.data().destroy();
        ascending.data().destroy();
    }

    @Test
    public void testVariance() {
        INDArray ones = Nd4j.ones(5);
        ones.var(0);
    }

    @Test
    public void testAddVectorWithOffset() {
        INDArray oneThroughFour = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray row1 = oneThroughFour.getRow(1);
        row1.addi(1);
        INDArray result = Nd4j.create(new float[]{1, 2, 4, 5}, new int[]{2, 2});
        assertEquals(result, oneThroughFour);
        oneThroughFour.data().destroy();


    }


    @Test
    public void testLinearViewGetAndPut() {
        INDArray test = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray linear = test.linearView();
        linear.putScalar(2, 6);
        linear.putScalar(3, 7);
        assertEquals(6, linear.getFloat(2), 1e-1);
        assertEquals(7, linear.getFloat(3), 1e-1);
        test.data().destroy();
    }


    @Test
    public void testGetIndices() {
        /*[[[1.0 ,13.0],[5.0 ,17.0],[9.0 ,21.0]],[[2.0 ,14.0],[6.0 ,18.0],[10.0 ,22.0]],[[3.0 ,15.0],[7.0 ,19.0],[11.0 ,23.0]],[[4.0 ,16.0],[8.0 ,20.0],[12.0 ,24.0]]]*/
        Nd4j.factory().setOrder('f');
        INDArray test = Nd4j.linspace(1, 24, 24).reshape(4, 3, 2);
        NDArrayIndex oneTwo = NDArrayIndex.interval(1, 2);
        NDArrayIndex twoToThree = NDArrayIndex.interval(1, 3);
        INDArray get = test.get(oneTwo, twoToThree);
        assertTrue(Arrays.equals(new int[]{1, 2, 2}, get.shape()));
        assertEquals(Nd4j.create(new float[]{6, 10, 18, 22}, new int[]{1, 2, 2}), get);

        INDArray anotherGet = Nd4j.create(new float[]{6, 7, 10, 11, 18, 19, 22, 23}, new int[]{2, 1, 2});
        INDArray test2 = test.get(NDArrayIndex.interval(1, 3), NDArrayIndex.interval(1, 2));
        assertEquals(5, test2.offset());
        //offset is off: should be 5
        assertTrue(Arrays.equals(new int[]{2, 1, 2}, test2.shape()));
        assertEquals(test2, anotherGet);

        INDArray linear = test2.slice(0).linearView();
        assertEquals(10, linear.getFloat(1), 1e-1);

        INDArray row = Nd4j.create(new float[]{7, 11});
        INDArray result = test2.slice(1);
        assertEquals(row, result);
        row.data().destroy();
        result.data().destroy();

    }


    @Test
    public void testSwapAxesFortranOrder() {
        Nd4j.factory().setOrder('f');

        INDArray n = Nd4j.create(Nd4j.linspace(1, 30, 30).data(), new int[]{3, 5, 2});
        INDArray slice = n.swapAxes(2, 1);
        INDArray assertion = Nd4j.create(new double[]{1, 4, 7, 10, 13});
        INDArray test = slice.slice(0).slice(0);
        assertEquals(assertion, test);
    }


    @Test
    public void testGetIndicesVector() {
        INDArray line = Nd4j.linspace(1, 4, 4);
        INDArray test = Nd4j.create(new float[]{2, 3});
        INDArray result = line.get(NDArrayIndex.interval(1, 3));
        assertEquals(test, result);
    }

    @Test
    public void testGetIndices2d() {
        Nd4j.factory().setOrder('f');
        Nd4j.dtype = DataBuffer.FLOAT;
        INDArray twoByTwo = Nd4j.linspace(1, 6, 6).reshape(3, 2);
        INDArray firstRow = twoByTwo.getRow(0);
        INDArray secondRow = twoByTwo.getRow(1);
        INDArray firstAndSecondRow = twoByTwo.getRows(new int[]{1, 2});
        INDArray firstRowViaIndexing = twoByTwo.get(NDArrayIndex.interval(0, 1));
        assertEquals(firstRow, firstRowViaIndexing);
        INDArray secondRowViaIndexing = twoByTwo.get(NDArrayIndex.interval(1, 2));
        assertEquals(secondRow, secondRowViaIndexing);
        INDArray individualElement = twoByTwo.get(NDArrayIndex.interval(1, 2), NDArrayIndex.interval(1, 2));
        individualElement.toString();
        assertEquals(Nd4j.create(new float[]{5}), individualElement);

        INDArray firstAndSecondRowTest = twoByTwo.get(NDArrayIndex.interval(1, 3));
        assertEquals(firstAndSecondRow, firstAndSecondRowTest);
        twoByTwo.data().destroy();

    }


    @Test
    public void testDup() {
        INDArray orig = Nd4j.linspace(1, 4, 4);
        INDArray dup = orig.dup();
        assertEquals(orig, dup);

        INDArray matrix = Nd4j.create(new float[]{1, 2, 3, 4}, new int[]{2, 2});
        INDArray dup2 = matrix.dup();
        assertEquals(matrix, dup2);

        INDArray row1 = matrix.getRow(1);
        INDArray dupRow = row1.dup();
        assertEquals(row1, dupRow);


        INDArray columnSorted = Nd4j.create(new float[]{2, 1, 4, 3}, new int[]{2, 2});
        INDArray dup3 = columnSorted.dup();
        assertEquals(columnSorted, dup3);


    }

    @Test
    public void testSortWithIndicesDescending() {
        INDArray toSort = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        //indices,data
        INDArray[] sorted = Nd4j.sortWithIndices(toSort.dup(), 1, false);
        INDArray sorted2 = Nd4j.sort(toSort.dup(), 1, false);
        assertEquals(sorted[1], sorted2);
        INDArray shouldIndex = Nd4j.create(new float[]{1, 0, 1, 0}, new int[]{2, 2});
        assertEquals(shouldIndex, sorted[0]);


    }


    @Test
    public void testSortWithIndices() {
        INDArray toSort = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        //indices,data
        INDArray[] sorted = Nd4j.sortWithIndices(toSort.dup(), 1, true);
        INDArray sorted2 = Nd4j.sort(toSort.dup(), 1, true);
        assertEquals(sorted[1], sorted2);
        INDArray shouldIndex = Nd4j.create(new float[]{0, 1, 0, 1}, new int[]{2, 2});
        assertEquals(shouldIndex, sorted[0]);


    }

    @Test
    public void testDimShuffle() {
        INDArray n = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray twoOneTwo = n.dimShuffle(new Object[]{0, 'x', 1}, new int[]{0, 1}, new boolean[]{false, false});
        assertTrue(Arrays.equals(new int[]{2, 1, 2}, twoOneTwo.shape()));

        INDArray reverse = n.dimShuffle(new Object[]{1, 'x', 0}, new int[]{1, 0}, new boolean[]{false, false});
        assertTrue(Arrays.equals(new int[]{2, 1, 2}, reverse.shape()));

    }

    @Test
    public void testGetVsGetScalar() {
        INDArray a = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        float element = a.getFloat(0, 1);
        double element2 = a.getDouble(0, 1);
        assertEquals(element, element2, 1e-1);
        Nd4j.factory().setOrder('f');
        INDArray a2 = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        float element23 = a2.getFloat(0, 1);
        double element22 = a2.getDouble(0, 1);
        assertEquals(element23, element22, 1e-1);

    }

    @Test
    public void testDivide() {
        INDArray two = Nd4j.create(new float[]{2, 2, 2, 2});
        INDArray div = two.div(two);
        assertEquals(Nd4j.ones(4), div);

        INDArray half = Nd4j.create(new float[]{0.5f, 0.5f, 0.5f, 0.5f}, new int[]{2, 2});
        INDArray divi = Nd4j.create(new float[]{0.3f, 0.6f, 0.9f, 0.1f}, new int[]{2, 2});
        INDArray assertion = Nd4j.create(new float[]{1.6666666f, 0.8333333f, 0.5555556f, 5}, new int[]{2, 2});
        INDArray result = half.div(divi);
        assertEquals(assertion, result);
    }


    @Test
    public void testSigmoid() {
        INDArray n = Nd4j.create(new float[]{1, 2, 3, 4});
        INDArray assertion = Nd4j.create(new float[]{0.73105858f, 0.88079708f, 0.95257413f, 0.98201379f});
        INDArray sigmoid = Transforms.sigmoid(n, false);
        assertEquals(assertion, sigmoid);
    }

    @Test
    public void testNeg() {
        INDArray n = Nd4j.create(new float[]{1, 2, 3, 4});
        INDArray assertion = Nd4j.create(new float[]{-1, -2, -3, -4});
        INDArray neg = Transforms.neg(n);
        assertEquals(assertion, neg);

    }

    @Test
    public void testNorm2Double() {
        Nd4j.dtype = DataBuffer.DOUBLE;
        INDArray n = Nd4j.create(new double[]{1, 2, 3, 4});
        double assertion = 5.47722557505;
        INDArray norm3 = n.norm2(Integer.MAX_VALUE);
        assertEquals(assertion, norm3.getDouble(0), 1e-1);

        INDArray row = Nd4j.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
        INDArray row1 = row.getRow(1);
        double norm2 = row1.norm2(Integer.MAX_VALUE).getDouble(0);
        double assertion2 = 5.0f;
        assertEquals(assertion2, norm2, 1e-1);

    }


    @Test
    public void testNorm2() {
        Nd4j.dtype = DataBuffer.FLOAT;
        INDArray n = Nd4j.create(new float[]{1, 2, 3, 4});
        float assertion = 5.47722557505f;
        INDArray norm3 = n.norm2(Integer.MAX_VALUE);
        assertEquals(assertion, norm3.getFloat(0), 1e-1);

        INDArray row = Nd4j.create(new float[]{1, 2, 3, 4}, new int[]{2, 2});
        INDArray row1 = row.getRow(1);
        float norm2 = row1.norm2(Integer.MAX_VALUE).getFloat(0);
        float assertion2 = 5.0f;
        assertEquals(assertion2, norm2, 1e-1);

    }


    @Test
    public void testDiag() {
        INDArray diag2 = Nd4j.linspace(1, 8, 8);
        INDArray diag = Nd4j.diag(diag2);
        for (int i = 0; i < diag2.length(); i++) {
            assertEquals(i + 1, diag.getFloat(i, i), 1e-1);
        }


        INDArray diag3 = diag2.reshape(2, 4);
        INDArray diagMatrix = Nd4j.diag(diag3);
        log.info("diag " + Nd4j.diag(diag3));


    }

    @Test
    public void testCosineSim() {
        Nd4j.dtype = DataBuffer.FLOAT;

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
    public void testScal() {
        double assertion = 2;
        INDArray answer = Nd4j.create(new double[]{2, 4, 6, 8});
        assertEquals(answer, Nd4j.getBlasWrapper().scal(assertion, answer));

        INDArray row = Nd4j.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
        INDArray row1 = row.getRow(1);
        double assertion2 = 5.0;
        INDArray answer2 = Nd4j.create(new double[]{15, 20});
        assertEquals(answer2, Nd4j.getBlasWrapper().scal(assertion2, row1));

    }

    @Test
    public void testExp() {
        INDArray n = Nd4j.create(new double[]{1, 2, 3, 4});
        INDArray assertion = Nd4j.create(new double[]{2.71828183f, 7.3890561f, 20.08553692f, 54.59815003f});
        INDArray exped = Transforms.exp(n);
        assertEquals(assertion, exped);
    }


    @Test
    public void testSlices() {
        INDArray arr = Nd4j.create(Nd4j.linspace(1, 24, 24).data(), new int[]{4, 3, 2});
        for (int i = 0; i < arr.slices(); i++) {
            assertEquals(2, arr.slice(i).slice(1).slices());
        }

    }


    @Test
    public void testScalar() {
        INDArray a = Nd4j.scalar(1.0);
        assertEquals(true, a.isScalar());

        INDArray n = Nd4j.create(new float[]{1.0f}, new int[]{1, 1});
        assertEquals(n, a);
        assertTrue(n.isScalar());
    }

    @Test
    public void testWrap() {
        int[] shape = {2, 4};
        INDArray d = Nd4j.linspace(1, 8, 8).reshape(shape[0], shape[1]);
        INDArray n = d;
        assertEquals(d.rows(), n.rows());
        assertEquals(d.columns(), n.columns());

        INDArray vector = Nd4j.linspace(1, 3, 3);
        INDArray testVector = vector;
        for (int i = 0; i < vector.length(); i++)
            assertEquals(vector.getDouble(i), testVector.getDouble(i), 1e-1);
        assertEquals(3, testVector.length());
        assertEquals(true, testVector.isVector());
        assertEquals(true, Shape.shapeEquals(new int[]{3}, testVector.shape()));

        INDArray row12 = Nd4j.linspace(1, 2, 2).reshape(2, 1);
        INDArray row22 = Nd4j.linspace(3, 4, 2).reshape(1, 2);

        assertEquals(row12.rows(), 2);
        assertEquals(row12.columns(), 1);
        assertEquals(row22.rows(), 1);
        assertEquals(row22.columns(), 2);


        d.data().destroy();
        vector.data().destroy();

    }

    @Test
    public void testGetRowFortran() {
        Nd4j.factory().setOrder('f');
        INDArray n = Nd4j.create(Nd4j.linspace(1, 4, 4).data(), new int[]{2, 2});
        INDArray column = Nd4j.create(new float[]{1, 3});
        INDArray column2 = Nd4j.create(new float[]{2, 4});
        INDArray testColumn = n.getRow(0);
        INDArray testColumn1 = n.getRow(1);
        assertEquals(column, testColumn);
        assertEquals(column2, testColumn1);
        Nd4j.factory().setOrder('c');
        n.data().destroy();


    }

    @Test
    public void testGetColumnFortran() {
        Nd4j.factory().setOrder('f');
        INDArray n = Nd4j.create(Nd4j.linspace(1, 4, 4).data(), new int[]{2, 2});
        INDArray column = Nd4j.create(new float[]{1, 2});
        INDArray column2 = Nd4j.create(new float[]{3, 4});
        INDArray testColumn = n.getColumn(0);
        INDArray testColumn1 = n.getColumn(1);
        assertEquals(column, testColumn);
        assertEquals(column2, testColumn1);
        Nd4j.factory().setOrder('c');

    }


    @Test
    public void testVectorInit() {
        DataBuffer data = Nd4j.linspace(1, 4, 4).data();
        INDArray arr = Nd4j.create(data, new int[]{4});
        assertEquals(true, arr.isRowVector());
        INDArray arr2 = Nd4j.create(data, new int[]{1, 4});
        assertEquals(true, arr2.isRowVector());

        INDArray columnVector = Nd4j.create(data, new int[]{4, 1});
        assertEquals(true, columnVector.isColumnVector());
    }


    @Test
    public void testColumns() {
        INDArray arr = Nd4j.create(new int[]{3, 2});
        INDArray column2 = arr.getColumn(0);
        assertEquals(true, Shape.shapeEquals(new int[]{3, 1}, column2.shape()));
        INDArray column = Nd4j.create(new double[]{1, 2, 3}, new int[]{3});
        arr.putColumn(0, column);

        INDArray firstColumn = arr.getColumn(0);

        assertEquals(column, firstColumn);


        INDArray column1 = Nd4j.create(new double[]{4, 5, 6}, new int[]{3});
        arr.putColumn(1, column1);
        assertEquals(true, Shape.shapeEquals(new int[]{3, 1}, arr.getColumn(1).shape()));
        INDArray testRow1 = arr.getColumn(1);
        assertEquals(column1, testRow1);


        INDArray evenArr = Nd4j.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
        INDArray put = Nd4j.create(new double[]{5, 6}, new int[]{2});
        evenArr.putColumn(1, put);
        INDArray testColumn = evenArr.getColumn(1);
        assertEquals(put, testColumn);


        INDArray n = Nd4j.create(Nd4j.linspace(1, 4, 4).data(), new int[]{2, 2});
        INDArray column23 = n.getColumn(0);
        INDArray column12 = Nd4j.create(new double[]{1, 3}, new int[]{2});
        assertEquals(column23, column12);


        INDArray column0 = n.getColumn(1);
        INDArray column01 = Nd4j.create(new double[]{2, 4}, new int[]{2});
        assertEquals(column0, column01);


    }


    @Test
    public void testPutRow() {
        INDArray d = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray n = d.dup();

        //works fine according to matlab, let's go with it..
        //reproduce with:  A = reshape(linspace(1,4,4),[2 2 ]);
        //A(1,2) % 1 index based
        float nFirst = 2;
        float dFirst = d.getFloat(0, 1);
        assertEquals(nFirst, dFirst, 1e-1);
        assertEquals(d.data(), n.data());
        assertEquals(true, Arrays.equals(new int[]{2, 2}, n.shape()));

        INDArray newRow = Nd4j.linspace(5, 6, 2);
        n.putRow(0, newRow);
        d.putRow(0, newRow);


        INDArray testRow = n.getRow(0);
        assertEquals(newRow.length(), testRow.length());
        assertEquals(true, Shape.shapeEquals(new int[]{2}, testRow.shape()));


        INDArray nLast = Nd4j.create(Nd4j.linspace(1, 4, 4).data(), new int[]{2, 2});
        INDArray row = nLast.getRow(1);
        INDArray row1 = Nd4j.create(new double[]{3, 4}, new int[]{2});
        assertEquals(row, row1);


        INDArray arr = Nd4j.create(new int[]{3, 2});
        INDArray evenRow = Nd4j.create(new double[]{1, 2}, new int[]{2});
        arr.putRow(0, evenRow);
        INDArray firstRow = arr.getRow(0);
        assertEquals(true, Shape.shapeEquals(new int[]{2}, firstRow.shape()));
        INDArray testRowEven = arr.getRow(0);
        assertEquals(evenRow, testRowEven);


        INDArray row12 = Nd4j.create(new double[]{5, 6}, new int[]{2});
        arr.putRow(1, row12);
        assertEquals(true, Shape.shapeEquals(new int[]{2}, arr.getRow(0).shape()));
        INDArray testRow1 = arr.getRow(1);
        assertEquals(row12, testRow1);


        INDArray multiSliceTest = Nd4j.create(Nd4j.linspace(1, 16, 16).data(), new int[]{4, 2, 2});
        INDArray test = Nd4j.create(new double[]{7, 8}, new int[]{2});
        INDArray test2 = Nd4j.create(new double[]{9, 10}, new int[]{2});

        assertEquals(test, multiSliceTest.slice(1).getRow(1));
        assertEquals(test2, multiSliceTest.slice(1).getRow(2));

    }

    @Test
    public void testOrdering() {
        //c ordering first
        Nd4j.factory().setOrder('c');
        Nd4j.factory().setDType(DataBuffer.FLOAT);

        INDArray data = Nd4j.create(new float[]{1, 2, 3, 4}, new int[]{2, 2});
        assertEquals(2.0, data.getDouble(0, 1), 1e-1);
        Nd4j.factory().setOrder('f');

        INDArray data2 = Nd4j.create(new float[]{1, 2, 3, 4}, new int[]{2, 2});
        assertNotEquals(data2.getDouble(0, 1), data.getDouble(0, 1), 1e-1);
        Nd4j.factory().setOrder('c');

    }


    @Test
    public void testSum() {
        INDArray n = Nd4j.create(Nd4j.linspace(1, 8, 8).data(), new int[]{2, 2, 2});
        INDArray test = Nd4j.create(new float[]{3, 7, 11, 15}, new int[]{2, 2});
        INDArray sum = n.sum(n.shape().length - 1);
        assertEquals(test, sum);


    }


    @Test
    public void testInplaceTransposeC() {
        Nd4j.factory().setOrder('c');
        INDArray test = Nd4j.rand(34, 484);
        INDArray transposei = test.transposei();

        for (int i = 0; i < test.rows(); i++) {
            for (int j = 0; j < test.columns(); j++) {
                assertEquals(test.getDouble(i, j), transposei.getDouble(j, i), 1e-1);
            }
        }

    }


    @Test
    public void testInplaceTranspose() {
        Nd4j.factory().setOrder('f');
        INDArray test = Nd4j.rand(34, 484);
        INDArray transposei = test.transposei();

        for (int i = 0; i < test.rows(); i++) {
            for (int j = 0; j < test.columns(); j++) {
                assertEquals(test.getDouble(i, j), transposei.getDouble(j, i), 1e-1);
            }
        }

    }


    @Test
    public void testTransposeMmul() {

        //note that transpose() and transposei() are equivalent here
        INDArray a = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        INDArray aT = a.transposei();

        double[][] result = new double[][]{
                {1, 4}, {2, 5}, {3, 6}
        };
        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[i].length; j++) {
                assertEquals(result[i][j], aT.getDouble(i, j), 1e-1);
            }
        }

        INDArray testMMul = a.mmul(aT);
        double[][] result2 = new double[][]{
                {23, 44}, {30, 58}
        };

        for (int i = 0; i < result2.length; i++) {
            for (int j = 0; j < result2[i].length; j++) {
                assertEquals(result2[i][j], testMMul.getDouble(i, j), 1e-1);
            }
        }


    }

    @Test
    public void testMmulF() {
        Nd4j.factory().setOrder('f');

        DataBuffer data = Nd4j.linspace(1, 10, 10).data();
        INDArray n = Nd4j.create(data, new int[]{10});
        INDArray transposed = n.transpose();
        assertEquals(true, n.isRowVector());
        assertEquals(true, transposed.isColumnVector());


        INDArray innerProduct = n.mmul(transposed);

        INDArray scalar = Nd4j.scalar(385);
        assertEquals(scalar, innerProduct);

    }

    @Test
    public void testSum2() {
        INDArray test = Nd4j.create(new float[]{1, 2, 3, 4}, new int[]{2, 2});
        INDArray sum = test.sum(1);
        INDArray assertion = Nd4j.create(new float[]{3, 7});
        assertEquals(assertion, sum);
    }


    @Test
    public void testMmul() {

        Nd4j.factory().setOrder('c');
        Nd4j.dtype = DataBuffer.DOUBLE;
        DataBuffer data = Nd4j.linspace(1, 10, 10).data();
        INDArray n = Nd4j.create(data, new int[]{10});
        INDArray transposed = n.transpose();
        assertEquals(true, n.isRowVector());
        assertEquals(true, transposed.isColumnVector());

        INDArray d = Nd4j.create(n.rows(), n.columns());
        d.setData(n.data());


        INDArray innerProduct = n.mmul(transposed);

        INDArray scalar = Nd4j.scalar(385);
        assertEquals(scalar, innerProduct);

        INDArray outerProduct = transposed.mmul(n);
        assertEquals(true, Shape.shapeEquals(new int[]{10, 10}, outerProduct.shape()));


        INDArray testMatrix = Nd4j.create(data, new int[]{5, 2});
        INDArray row1 = testMatrix.getRow(0).transpose();
        INDArray row2 = testMatrix.getRow(1);
        INDArray row12 = Nd4j.linspace(1, 2, 2).reshape(2, 1);
        INDArray row22 = Nd4j.linspace(3, 4, 2).reshape(1, 2);

        INDArray row122 = row12;
        INDArray row222 = row22;
        INDArray rowResult2 = row122.mmul(row222);


        INDArray d3 = Nd4j.create(new double[]{1, 2}).reshape(2, 1);
        INDArray d4 = Nd4j.create(new double[]{3, 4});
        INDArray resultNDArray = d3.mmul(d4);
        INDArray result = Nd4j.create(new double[][]{{3, 4}, {6, 8}});

        assertEquals(result, resultNDArray);


        INDArray three = Nd4j.create(new double[]{3, 4}, new int[]{2});
        INDArray test = Nd4j.create(Nd4j.linspace(1, 30, 30).data(), new int[]{3, 5, 2});
        INDArray sliceRow = test.slice(0).getRow(1);
        assertEquals(three, sliceRow);

        INDArray twoSix = Nd4j.create(new double[]{2, 6}, new int[]{2, 1});
        INDArray threeTwoSix = three.mmul(twoSix);

        INDArray sliceRowTwoSix = sliceRow.mmul(twoSix);

        assertEquals(threeTwoSix, sliceRowTwoSix);


        INDArray vectorVector = Nd4j.create(new double[]{
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99, 108, 117, 126, 135, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 0, 11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 0, 13, 26, 39, 52, 65, 78, 91, 104, 117, 130, 143, 156, 169, 182, 195, 0, 14, 28, 42, 56, 70, 84, 98, 112, 126, 140, 154, 168, 182, 196, 210, 0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225
        }, new int[]{16, 16});


        INDArray n1 = Nd4j.create(Nd4j.linspace(0, 15, 16).data(), new int[]{16});
        INDArray k1 = n1.transpose();

        INDArray testVectorVector = k1.mmul(n1);
        assertEquals(vectorVector, testVectorVector);


    }

    @Test
    public void testRowsColumns() {
        DataBuffer data = Nd4j.linspace(1, 6, 6).data();
        INDArray rows = Nd4j.create(data, new int[]{2, 3});
        assertEquals(2, rows.rows());
        assertEquals(3, rows.columns());

        INDArray columnVector = Nd4j.create(data, new int[]{6, 1});
        assertEquals(6, columnVector.rows());
        assertEquals(1, columnVector.columns());
        INDArray rowVector = Nd4j.create(data, new int[]{6});
        assertEquals(1, rowVector.rows());
        assertEquals(6, rowVector.columns());
    }


    @Test
    public void testTranspose() {
        Nd4j.factory().setOrder('f');
        INDArray n = Nd4j.create(Nd4j.ones(100).data(), new int[]{5, 5, 4});
        INDArray transpose = n.transpose();
        assertEquals(n.length(), transpose.length());
        assertEquals(true, Arrays.equals(new int[]{4, 5, 5}, transpose.shape()));

        INDArray rowVector = Nd4j.linspace(1, 10, 10);
        assertTrue(rowVector.isRowVector());
        INDArray columnVector = rowVector.transpose();
        assertTrue(columnVector.isColumnVector());


        INDArray linspaced = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray transposed = Nd4j.create(new float[]{1, 3, 2, 4}, new int[]{2, 2});
        assertEquals(transposed, linspaced.transpose());

        Nd4j.factory().setOrder('f');
        linspaced = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        //fortran ordered
        INDArray transposed2 = Nd4j.create(new float[]{1, 3, 2, 4}, new int[]{2, 2});
        transposed = linspaced.transpose();
        assertEquals(transposed, transposed2);
        Nd4j.factory().setOrder('c');


    }

    @Test
    public void testCopyMatrix() {
        INDArray twoByThree = Nd4j.linspace(1, 784, 784).reshape(28, 28);
        INDArray copy = Nd4j.create(784, 784);
        Nd4j.getBlasWrapper().copy(twoByThree, copy);
    }

    @Test
    public void testAddMatrix() {
        Nd4j.dtype = DataBuffer.FLOAT;
        INDArray five = Nd4j.ones(5);
        five.addi(five);
        INDArray twos = Nd4j.valueArrayOf(5, 2);
        assertEquals(twos, five);

        INDArray twoByThree = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        Nd4j.getBlasWrapper().axpy(1, twoByThree, twoByThree);
    }


    @Test
    public void testDimensionWiseWithVector() {
        INDArray ret = Nd4j.linspace(1,2,2).reshape(1,2);
        assertTrue(ret.sum(0).isRowVector());
        assertTrue(ret.sum(1).isScalar());
        INDArray retColumn = Nd4j.linspace(1,2,2).reshape(2,1);
        assertTrue(retColumn.sum(1).isRowVector());
        assertTrue(retColumn.sum(0).isScalar());

        INDArray m2 = Nd4j.rand(1, 2);
        Nd4j.sum(m2, 0);


        Nd4j.sum(m2, 1);

        INDArray m3 = Nd4j.rand(2, 1);

        Nd4j.sum(m3, 0);
        Nd4j.sum(m3, 1).toString();

    }

    @Test
    public void testCreationWithOrder() {
        INDArray ret = Nd4j.create(new float[]{1,1,1,1},new int[]{1,4},'f');
        INDArray ret2 = Nd4j.create(new double[]{1,1,1,1},new int[]{1,4},'f');
        Nd4j.dtype = DataBuffer.DOUBLE;
        ret = Nd4j.create(new float[]{1,1,1,1},new int[]{1,4},'f');
        ret2 = Nd4j.create(new double[]{1,1,1,1},new int[]{1,4},'f');
        INDArray  b0 = Nd4j.arange(0, 12).reshape(3, 4);
        INDArray b4 =
                Nd4j.create(
                        b0.data().asDouble(),
                        new int[]{3,4},
        'f');
        b4.toString();
        Nd4j.dtype = DataBuffer.FLOAT;


    }


    @Test
    public void testPutSlice() {
        INDArray n = Nd4j.create(Nd4j.ones(27).data(), new int[]{3, 3, 3});
        INDArray newSlice = Nd4j.zeros(3, 3);
        n.putSlice(0, newSlice);
        assertEquals(newSlice, n.slice(0));


    }

    @Test
    public void testColumnMean() {
        INDArray twoByThree = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray columnMean = twoByThree.mean(0);
        INDArray assertion = Nd4j.create(new float[]{2, 3});
        assertEquals(assertion, columnMean);
    }


    @Test
    public void testColumnVar() {
        INDArray twoByThree = Nd4j.linspace(1, 600, 600).reshape(150, 4);
        INDArray columnStd = twoByThree.var(0);
        INDArray assertion = Nd4j.create(new float[]{30200f, 30200f, 30200f, 30200f});
        assertEquals(assertion, columnStd);

    }

    @Test
    public void testColumnStd() {
        INDArray twoByThree = Nd4j.linspace(1, 600, 600).reshape(150, 4);
        INDArray columnStd = twoByThree.std(0);
        INDArray assertion = Nd4j.create(new float[]{173.78147196982766f, 173.78147196982766f, 173.78147196982766f, 173.78147196982766f});
        assertEquals(assertion, columnStd);

    }

    @Test
    public void testEps() {
        INDArray ones = Nd4j.ones(5);
        double sum = Nd4j.getExecutioner().exec(new Eps(ones, ones, ones, ones.length())).z().sum(Integer.MAX_VALUE).getDouble(0);
        assertEquals(5, sum, 1e-1);
    }


    @Test
    public void testLogDouble() {
        Nd4j.dtype = DataBuffer.DOUBLE;
        INDArray log = Transforms.log(Nd4j.linspace(1, 6, 6));
        INDArray assertion = Nd4j.create(new double[]{0, 0.69314718, 1.09861229, 1.38629436, 1.60943791, 1.79175947});
        assertEquals(assertion, log);
    }

    @Test
    public void testIrisStatsDouble() throws IOException {
        Nd4j.dtype = DataBuffer.DOUBLE;
        ClassPathResource res = new ClassPathResource("/iris.txt");
        File file = res.getFile();
        INDArray data = Nd4j.readTxt(file.getAbsolutePath(), "\t");
        INDArray mean = Nd4j.create(new double[]{5.843333333333335, 3.0540000000000007, 3.7586666666666693, 1.1986666666666672});
        INDArray std = Nd4j.create(new double[]{0.8280661279778629, 0.4335943113621737, 1.7644204199522617, 0.7631607417008414});

        INDArray testSum = Nd4j.create(new double[]{876.4999990463257, 458.1000003814697, 563.7999982833862, 179.7999987155199});
        INDArray sum = data.sum(0);
        INDArray test = data.mean(0);
        INDArray testStd = data.std(0);
        assertEquals(sum, testSum);
        assertEquals(mean, test);
        assertEquals(std, testStd);

    }

    @Test
    public void testSmallSum() {
        INDArray base = Nd4j.create(new double[]{5.843333333333335, 3.0540000000000007});
        base.addi(1e-12);
        INDArray assertion = Nd4j.create(new double[]{5.84333433, 3.054001});
        assertEquals(assertion, base);

    }

    @Test
    public void testIrisStats() throws IOException {
        Nd4j.dtype = DataBuffer.FLOAT;
        ClassPathResource res = new ClassPathResource("/iris.txt");
        File file = res.getFile();
        INDArray data = Nd4j.readTxt(file.getAbsolutePath(), "\t");
        INDArray sum = data.sum(0);
        INDArray mean = Nd4j.create(new double[]{5.843333333333335, 3.0540000000000007, 3.7586666666666693, 1.1986666666666672});
        INDArray std = Nd4j.create(new double[]{0.8280661279778629, 0.4335943113621737, 1.7644204199522617, 0.7631607417008414});

        INDArray testSum = Nd4j.create(new double[]{876.4999990463257, 458.1000003814697, 563.7999982833862, 179.7999987155199});
        assertEquals(testSum, sum);

        INDArray testMean = data.mean(0);
        assertEquals(mean, testMean);

        INDArray testStd = data.std(0);
        assertEquals(std, testStd);
    }

    @Test
    public void testColumnVariance() {
        INDArray twoByThree = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray columnVar = twoByThree.var(0);
        INDArray assertion = Nd4j.create(new float[]{2f, 2f});
        assertEquals(assertion, columnVar);

    }

    @Test
    public void testColumnSumDouble() {
        Nd4j.dtype = DataBuffer.DOUBLE;
        INDArray twoByThree = Nd4j.linspace(1, 600, 600).reshape(150, 4);
        INDArray columnVar = twoByThree.sum(0);
        INDArray assertion = Nd4j.create(new float[]{44850.0f, 45000.0f, 45150.0f, 45300.0f});
        assertEquals(assertion, columnVar);

    }


    @Test
    public void testColumnSum() {
        INDArray twoByThree = Nd4j.linspace(1, 600, 600).reshape(150, 4);
        INDArray columnVar = twoByThree.sum(0);
        INDArray assertion = Nd4j.create(new float[]{44850.0f, 45000.0f, 45150.0f, 45300.0f});
        assertEquals(assertion, columnVar);

    }

    @Test
    public void testRowMean() {
        INDArray twoByThree = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray rowMean = twoByThree.mean(1);
        INDArray assertion = Nd4j.create(new float[]{1.5f, 3.5f});
        assertEquals(assertion, rowMean);


    }

    @Test
    public void testRowStd() {
        INDArray twoByThree = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray rowStd = twoByThree.std(1);
        INDArray assertion = Nd4j.create(new float[]{0.7071067811865476f, 0.7071067811865476f});
        assertEquals(assertion, rowStd);

    }


    @Test
    public void testPermute() {
        INDArray n = Nd4j.create(Nd4j.linspace(1, 20, 20).data(), new int[]{5, 4});
        INDArray transpose = n.transpose();
        INDArray permute = n.permute(1, 0);
        assertEquals(permute, transpose);
        assertEquals(transpose.length(), permute.length(), 1e-1);


        INDArray toPermute = Nd4j.create(Nd4j.linspace(0, 7, 8).data(), new int[]{2, 2, 2});
        INDArray permuted = toPermute.permute(2, 1, 0);
        INDArray assertion = Nd4j.create(new float[]{0, 4, 2, 6, 1, 5, 3, 7}, new int[]{2, 2, 2});
        assertEquals(permuted, assertion);

    }

    @Test
    public void testSlice() {
        assertEquals(8, n.length());
        assertEquals(true, Arrays.equals(new int[]{2, 2, 2}, n.shape()));
        INDArray slice = n.slice(0);
        assertEquals(true, Arrays.equals(new int[]{2, 2}, slice.shape()));

        INDArray slice1 = n.slice(1);
        assertNotEquals(slice, slice1);

        INDArray arr = Nd4j.create(Nd4j.linspace(1, 24, 24).data(), new int[]{4, 3, 2});
        INDArray slice0 = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6}, new int[]{3, 2});
        INDArray slice2 = Nd4j.create(new float[]{7, 8, 9, 10, 11, 12}, new int[]{3, 2});

        INDArray testSlice0 = arr.slice(0);
        INDArray testSlice1 = arr.slice(1);

        assertEquals(slice0, testSlice0);
        assertEquals(slice2, testSlice1);


    }

    @Test
    public void testSwapAxes() {
        INDArray n = Nd4j.create(Nd4j.linspace(0, 7, 8).data(), new int[]{2, 2, 2});
        INDArray assertion = n.permute(2, 1, 0);
        INDArray validate = Nd4j.create(new float[]{0, 4, 2, 6, 1, 5, 3, 7}, new int[]{2, 2, 2});
        assertEquals(validate, assertion);

        INDArray thirty = Nd4j.linspace(1, 30, 30).reshape(3, 5, 2);
        INDArray swapped = thirty.swapAxes(2, 1);
        INDArray slice = swapped.slice(0).slice(0);
        INDArray assertion2 = Nd4j.create(new double[]{1, 3, 5, 7, 9});
        assertEquals(assertion2, slice);


    }


    @Test
    public void testLinearIndex() {
        INDArray n = Nd4j.create(Nd4j.linspace(1, 8, 8).data(), new int[]{8});
        for (int i = 0; i < n.length(); i++) {
            int linearIndex = n.linearIndex(i);
            assertEquals(i, linearIndex);
            double d = n.getDouble(i);
            assertEquals(i + 1, d, 1e-1);
        }
    }

    @Test
    public void testSliceConstructor() {
        List<INDArray> testList = new ArrayList<>();
        for (int i = 0; i < 5; i++)
            testList.add(Nd4j.scalar(i + 1));

        INDArray test = Nd4j.create(testList, new int[]{testList.size()});
        INDArray expected = Nd4j.create(new float[]{1, 2, 3, 4, 5}, new int[]{5});
        assertEquals(expected, test);
        test.data().destroy();
        expected.data().dataType();
    }


    @Test
    public void testVectorDimension() {
        INDArray test = Nd4j.create(Nd4j.linspace(1, 4, 4).data(), new int[]{2, 2});
        final AtomicInteger count = new AtomicInteger(0);
        //row wise
        test.iterateOverDimension(1, new SliceOp() {


            /**
             * Operates on an ndarray slice
             *
             * @param nd the result to operate on
             */
            @Override
            public void operate(INDArray nd) {
                INDArray test = nd;
                if (count.get() == 0) {
                    INDArray firstDimension = Nd4j.create(new float[]{1, 2}, new int[]{2});
                    assertEquals(firstDimension, test);
                } else {
                    INDArray firstDimension = Nd4j.create(new float[]{3, 4}, new int[]{2});
                    assertEquals(firstDimension, test);

                }

                count.incrementAndGet();
            }

        }, false);


        count.set(0);

        //columnwise
        test.iterateOverDimension(0, new SliceOp() {


            /**
             * Operates on an ndarray slice
             *
             * @param nd the result to operate on
             */
            @Override
            public void operate(INDArray nd) {
                log.info("Operator " + nd);
                INDArray test = nd;
                if (count.get() == 0) {
                    INDArray firstDimension = Nd4j.create(new float[]{1, 3}, new int[]{2});
                    assertEquals(firstDimension, test);
                } else {
                    INDArray firstDimension = Nd4j.create(new float[]{2, 4}, new int[]{2});
                    assertEquals(firstDimension, test);
                    firstDimension.data().destroy();

                }

                count.incrementAndGet();
            }

        }, false);


        test.data().destroy();


    }


    @Test
    public void testDimension() {
        INDArray test = Nd4j.create(Nd4j.linspace(1, 4, 4).data(), new int[]{2, 2});
        //row
        INDArray slice0 = test.slice(0, 1);
        INDArray slice02 = test.slice(1, 1);

        INDArray assertSlice0 = Nd4j.create(new float[]{1, 2});
        INDArray assertSlice02 = Nd4j.create(new float[]{3, 4});
        assertEquals(assertSlice0, slice0);
        assertEquals(assertSlice02, slice02);

        //column
        INDArray assertSlice1 = Nd4j.create(new float[]{1, 3});
        INDArray assertSlice12 = Nd4j.create(new float[]{2, 4});


        INDArray slice1 = test.slice(0, 0);
        INDArray slice12 = test.slice(1, 0);


        assertEquals(assertSlice1, slice1);
        assertEquals(assertSlice12, slice12);


        INDArray arr = Nd4j.create(Nd4j.linspace(1, 24, 24).data(), new int[]{4, 3, 2});
        INDArray firstSliceFirstDimension = arr.slice(0, 1);
        INDArray secondSliceFirstDimension = arr.slice(1, 1);

        INDArray firstSliceFirstDimensionAssert = Nd4j.create(new float[]{1, 2, 7, 8, 13, 14, 19, 20});
        INDArray secondSliceFirstDimension2Test = firstSliceFirstDimensionAssert.add(1);
        assertEquals(secondSliceFirstDimension, secondSliceFirstDimension);


    }


    @Test
    public void testAppendBias() {
        INDArray rand = Nd4j.linspace(1, 25, 25).transpose();
        INDArray test = Nd4j.appendBias(rand);
        INDArray assertion = Nd4j.toFlattened(rand, Nd4j.scalar(1));
        assertEquals(assertion, test);
    }

    @Test
    public void testRand() {
        INDArray rand = Nd4j.randn(5, 5);
        Nd4j.getDistributions().createUniform(1, 5).sample(5);
        Nd4j.getDistributions().createNormal(1, 5).sample();
        Nd4j.getDistributions().createBinomial(5, 1.0).sample(new int[]{5, 5});
        Nd4j.getDistributions().createBinomial(1, Nd4j.ones(5, 5)).sample(rand.shape());
        Nd4j.getDistributions().createNormal(rand, 1).sample(rand.shape());
    }


    @Test
    public void testReshape() {
        INDArray arr = Nd4j.create(Nd4j.linspace(1, 24, 24).data(), new int[]{4, 3, 2});
        INDArray reshaped = arr.reshape(2, 3, 4);
        assertEquals(arr.length(), reshaped.length());
        assertEquals(true, Arrays.equals(new int[]{4, 3, 2}, arr.shape()));
        assertEquals(true, Arrays.equals(new int[]{2, 3, 4}, reshaped.shape()));

        INDArray n2 = Nd4j.create(Nd4j.linspace(1, 30, 30).data(), new int[]{3, 5, 2});
        INDArray swapped = n2.swapAxes(n2.shape().length - 1, 1);
        INDArray firstSlice2 = swapped.slice(0).slice(0);
        INDArray oneThreeFiveSevenNine = Nd4j.create(new float[]{1, 3, 5, 7, 9});
        assertEquals(firstSlice2, oneThreeFiveSevenNine);
        INDArray raveled = oneThreeFiveSevenNine.reshape(5, 1);
        INDArray raveledOneThreeFiveSevenNine = oneThreeFiveSevenNine.reshape(5, 1);
        assertEquals(raveled, raveledOneThreeFiveSevenNine);


        INDArray firstSlice3 = swapped.slice(0).slice(1);
        INDArray twoFourSixEightTen = Nd4j.create(new float[]{2, 4, 6, 8, 10});
        assertEquals(firstSlice2, oneThreeFiveSevenNine);
        INDArray raveled2 = twoFourSixEightTen.reshape(5, 1);
        INDArray raveled3 = firstSlice3.reshape(5, 1);
        assertEquals(raveled2, raveled3);


    }

    @Test
    public void testDot() {
        INDArray vec1 = Nd4j.create(new float[]{1, 2, 3, 4});
        INDArray vec2 = Nd4j.create(new float[]{1, 2, 3, 4});
        assertEquals(30, Nd4j.getBlasWrapper().dot(vec1, vec2), 1e-1);

        INDArray matrix = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray row = matrix.getRow(1);
        assertEquals(25, Nd4j.getBlasWrapper().dot(row, row), 1e-1);

    }


    @Test
    public void testIdentity() {
        INDArray eye = Nd4j.eye(5);
        assertTrue(Arrays.equals(new int[]{5, 5}, eye.shape()));
        Nd4j.factory().setOrder('f');
        eye = Nd4j.eye(5);
        assertTrue(Arrays.equals(new int[]{5, 5}, eye.shape()));


    }


    @Test
    public void testColumnVectorOpsFortran() {
        Nd4j.factory().setOrder('f');
        INDArray twoByTwo = Nd4j.create(new float[]{1, 2, 3, 4}, new int[]{2, 2});
        INDArray toAdd = Nd4j.create(new float[]{1, 2}, new int[]{2, 1});
        twoByTwo.addiColumnVector(toAdd);
        INDArray assertion = Nd4j.create(new float[]{2, 4, 4, 6}, new int[]{2, 2});
        assertEquals(assertion, twoByTwo);


    }

    @Test
    public void testGetNonContiguous() {
        INDArray create = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        NDArrayIndex[] indices = new NDArrayIndex[2];
        indices[0] = NDArrayIndex.interval(0, 1);
        indices[1] = new NDArrayIndex(0, 2);

        INDArray assertion = Nd4j.create(new double[]{1, 3});
        assertEquals(create.get(indices), assertion);
        INDArray assertion2 = Nd4j.create(new double[]{4, 7});
        create.put(indices, Nd4j.create(new double[]{4, 7}));
        assertEquals(assertion2, create.get(indices));

        INDArray multiRowAssign = Nd4j.create(new double[]{5, 6, 7, 8}, new int[]{2, 2});
        NDArrayIndex[] index2 = new NDArrayIndex[]{
                NDArrayIndex.interval(0, 2),
                new NDArrayIndex(0, 2)
        };

        create.put(index2, multiRowAssign);

        INDArray get = create.get(index2);
        assertEquals(multiRowAssign, get);

    }

    @Test
    public void testMeans() {
        INDArray a = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        assertEquals(Nd4j.create(new float[]{1.5f, 3.5f}), a.mean(1));
        assertEquals(Nd4j.create(new float[]{2, 3}), a.mean(0));
        assertEquals(2.5, a.mean(Integer.MAX_VALUE).getDouble(0), 1e-1);

    }


    @Test
    public void testSums() {
        INDArray a = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        assertEquals(Nd4j.create(new float[]{4, 6}), a.sum(0));
        assertEquals(Nd4j.create(new float[]{3, 7}), a.sum(1));
        assertEquals(10, a.sum(Integer.MAX_VALUE).getDouble(0), 1e-1);


    }


    @Test
    public void testCumSum() {
        INDArray n = Nd4j.create(new float[]{1, 2, 3, 4}, new int[]{4});
        INDArray cumSumAnswer = Nd4j.create(new float[]{1, 3, 6, 10}, new int[]{4});
        INDArray cumSumTest = n.cumsum(0);
        assertEquals(cumSumAnswer, cumSumTest);

        INDArray n2 = Nd4j.linspace(1, 24, 24).reshape(4, 3, 2);
        INDArray cumSumCorrect2 = Nd4j.create(new double[]{1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0, 45.0, 55.0, 66.0, 78.0, 91.0, 105.0, 120.0, 136.0, 153.0, 171.0, 190.0, 210.0, 231.0, 253.0, 276.0, 300.0}, new int[]{24});
        INDArray cumSumTest2 = n2.cumsum(n2.shape().length - 1);
        assertEquals(cumSumCorrect2, cumSumTest2);

        INDArray axis0assertion = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 21, 24, 27, 30, 33, 36, 40, 44, 48, 52, 56, 60}, n2.shape());
        INDArray axis0Test = n2.cumsum(0);
        assertEquals(axis0assertion, axis0Test);

    }


    @Test
    public void testRSubi() {
        INDArray n2 = Nd4j.ones(2);
        INDArray n2Assertion = Nd4j.zeros(2);
        INDArray nRsubi = n2.rsubi(1);
        assertEquals(n2Assertion, nRsubi);
    }


    @Test
    public void testConcat() {
        INDArray A = Nd4j.linspace(1, 8, 8).reshape(2, 2, 2);
        INDArray B = Nd4j.linspace(1, 12, 12).reshape(3, 2, 2);
        INDArray concat = Nd4j.concat(0, A, B);
        assertTrue(Arrays.equals(new int[]{5, 2, 2}, concat.shape()));

    }

    @Test
    public void testConcatHorizontally() {
        INDArray rowVector = Nd4j.ones(5);
        INDArray other = Nd4j.ones(5);
        INDArray concat = Nd4j.hstack(other, rowVector);
        assertEquals(rowVector.rows(), concat.rows());
        assertEquals(rowVector.columns() * 2, concat.columns());

    }


    @Test
    public void testConcatVertically() {
        INDArray rowVector = Nd4j.ones(5);
        INDArray other = Nd4j.ones(5);
        INDArray concat = Nd4j.vstack(other, rowVector);
        assertEquals(rowVector.rows() * 2, concat.rows());
        assertEquals(rowVector.columns(), concat.columns());

    }

    @Test
    public void testAddScalar() {
        INDArray div = Nd4j.valueArrayOf(new int[]{4}, 4);
        float[] value = div.data().asFloat();
        div.toString();
        INDArray rdiv = div.add(1);
        INDArray answer = Nd4j.valueArrayOf(new int[]{4}, 5);
        assertEquals(rdiv, answer);
    }

    @Test
    public void testRdivScalar() {
        INDArray div = Nd4j.valueArrayOf(2, 4);
        INDArray rdiv = div.rdiv(1);
        INDArray answer = Nd4j.valueArrayOf(new int[]{4}, 0.25);
        assertEquals(rdiv, answer);
    }

    @Test
    public void testRDivi() {
        INDArray n2 = Nd4j.valueArrayOf(new int[]{2}, 4);
        INDArray n2Assertion = Nd4j.valueArrayOf(new int[]{2}, 0.5);
        INDArray nRsubi = n2.rdivi(2);
        assertEquals(n2Assertion, nRsubi);
    }


    @Test
    public void testVectorAlongDimension() {
        INDArray arr = Nd4j.linspace(1, 24, 24).reshape(4, 3, 2);
        INDArray assertion = Nd4j.create(new float[]{1, 2}, new int[]{2});
        assertEquals(Nd4j.create(new float[]{3, 4}, new int[]{2}), arr.vectorAlongDimension(1, 2));
        assertEquals(assertion, arr.vectorAlongDimension(0, 2));
        assertEquals(arr.vectorAlongDimension(0, 1), Nd4j.create(new float[]{1, 3, 5}));

        INDArray testColumn2Assertion = Nd4j.create(new float[]{7, 9, 11});
        INDArray testColumn2 = arr.vectorAlongDimension(1, 1);

        assertEquals(testColumn2Assertion, testColumn2);


        INDArray testColumn3Assertion = Nd4j.create(new float[]{13, 15, 17});
        INDArray testColumn3 = arr.vectorAlongDimension(2, 1);
        assertEquals(testColumn3Assertion, testColumn3);


        INDArray v1 = Nd4j.linspace(1, 4, 4).reshape(new int[]{2, 2});
        INDArray testColumnV1 = v1.vectorAlongDimension(0, 0);
        INDArray testColumnV1Assertion = Nd4j.create(new float[]{1, 3});
        assertEquals(testColumnV1Assertion, testColumnV1);

        INDArray testRowV1 = v1.vectorAlongDimension(1, 0);
        INDArray testRowV1Assertion = Nd4j.create(new float[]{2, 4});
        assertEquals(testRowV1Assertion, testRowV1);


        INDArray lastAxis = arr.vectorAlongDimension(0, 2);
        assertEquals(assertion, lastAxis);


    }

    @Test
    public void testSquareMatrix() {
        INDArray n = Nd4j.create(Nd4j.linspace(1, 8, 8).data(), new int[]{2, 2, 2});
        INDArray eightFirstTest = n.vectorAlongDimension(0, 2);
        INDArray eightFirstAssertion = Nd4j.create(new float[]{1, 2}, new int[]{2});
        assertEquals(eightFirstAssertion, eightFirstTest);

        INDArray eightFirstTestSecond = n.vectorAlongDimension(1, 2);
        INDArray eightFirstTestSecondAssertion = Nd4j.create(new float[]{3, 4});
        assertEquals(eightFirstTestSecondAssertion, eightFirstTestSecond);

    }

    @Test
    public void testNumVectorsAlongDimension() {
        INDArray arr = Nd4j.linspace(1, 24, 24).reshape(4, 3, 2);
        assertEquals(12, arr.vectorsAlongDimension(2));
    }


    @Test
    public void testGetScalar() {
        INDArray n = Nd4j.create(new float[]{1, 2, 3, 4}, new int[]{4});
        assertTrue(n.isVector());
        for (int i = 0; i < n.length(); i++) {
            INDArray scalar = Nd4j.scalar((float) i + 1);
            assertEquals(scalar, n.getScalar(i));
        }


    }

    @Test
    public void testGetScalarFortran() {
        Nd4j.factory().setOrder('f');
        n = Nd4j.create(new float[]{1, 2, 3, 4}, new int[]{4});
        for (int i = 0; i < n.length(); i++) {
            INDArray scalar = Nd4j.scalar((float) i + 1);
            assertEquals(scalar, n.getScalar(i));
        }


        INDArray twoByTwo = Nd4j.create(new float[][]{{1, 2}, {3, 4}});
        INDArray column = twoByTwo.getColumn(0);
        assertEquals(Nd4j.create(new float[]{1, 3}), column);
        assertEquals(1, column.getFloat(0), 1e-1);
        assertEquals(3, column.getFloat(1), 1e-1);
        assertEquals(Nd4j.scalar(1), column.getScalar(0));
        assertEquals(Nd4j.scalar(3), column.getScalar(1));

    }

    @Test
    public void testGetMulti() {
        assertEquals(8, n.length());
        assertEquals(true, Arrays.equals(ArrayUtil.of(2, 2, 2), n.shape()));
        double val = n.getDouble(1, 1, 1);
        assertEquals(8.0, val, 1e-6);
    }


    @Test
    public void testGetRowOrdering() {
        INDArray row1 = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        Nd4j.factory().setOrder('f');
        INDArray row1Fortran = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        assertNotEquals(row1.getFloat(0, 1), row1Fortran.getFloat(0, 1), 1e-1);
        Nd4j.factory().setOrder('c');
    }

    @Test
    public void testBroadCast() {
        INDArray n = Nd4j.linspace(1, 4, 4);
        INDArray broadCasted = n.broadcast(5, 4);
        for (int i = 0; i < broadCasted.rows(); i++) {
            assertEquals(n, broadCasted.getRow(i));
        }

        INDArray broadCast2 = broadCasted.getRow(0).broadcast(5, 4);
        assertEquals(broadCasted, broadCast2);


        INDArray columnBroadcast = n.transpose().broadcast(4, 5);
        for (int i = 0; i < columnBroadcast.columns(); i++) {
            assertEquals(columnBroadcast.getColumn(i), n.transpose());
        }

        INDArray fourD = Nd4j.create(1, 2, 1, 1);
        INDArray broadCasted3 = fourD.broadcast(1, 1, 36, 36);
        assertTrue(Arrays.equals(new int[]{1, 2, 36, 36}, broadCasted3.shape()));
    }


    @Test
    public void testPutRowGetRowOrdering() {
        Nd4j.dtype = DataBuffer.DOUBLE;
        INDArray row1 = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray put = Nd4j.create(new double[]{5, 6});
        row1.putRow(1, put);

        Nd4j.factory().setOrder('f');

        INDArray row1Fortran = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray putFortran = Nd4j.create(new double[]{5, 6});
        row1Fortran.putRow(1, putFortran);
        assertNotEquals(row1, row1Fortran);
        INDArray row1CTest = row1.getRow(1);
        INDArray row1FortranTest = row1Fortran.getRow(1);
        assertEquals(row1CTest, row1FortranTest);

        Nd4j.factory().setOrder('c');


    }


    @Test
    public void testPutRowFortran() {
        INDArray row1 = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray put = Nd4j.create(new double[]{5, 6});
        row1.putRow(1, put);

        Nd4j.factory().setOrder('f');

        INDArray row1Fortran = Nd4j.create(new double[][]{{1, 2}, {3, 4}});
        INDArray putFortran = Nd4j.create(new double[]{5, 6});
        row1Fortran.putRow(1, putFortran);
        assertEquals(row1, row1Fortran);

        Nd4j.factory().setOrder('c');


    }


    @Test
    public void testElementWiseOps() {
        INDArray n1 = Nd4j.scalar(1);
        INDArray n2 = Nd4j.scalar(2);
        INDArray nClone = n1.add(n2);
        assertEquals(Nd4j.scalar(3), nClone);
        assertFalse(n1.add(n2).equals(n1));

        INDArray n3 = Nd4j.scalar(3);
        INDArray n4 = Nd4j.scalar(4);
        INDArray subbed = n4.sub(n3);
        INDArray mulled = n4.mul(n3);
        INDArray div = n4.div(n3);

        assertFalse(subbed.equals(n4));
        assertFalse(mulled.equals(n4));
        assertEquals(Nd4j.scalar(1), subbed);
        assertEquals(Nd4j.scalar(12), mulled);
        assertEquals(Nd4j.scalar(1.333333333333333333333), div);
    }


    @Test
    public void testSlicing() {
        INDArray arr = n.slice(1, 1);
        // assertEquals(1,arr.shape().length());
        INDArray n2 = Nd4j.create(Nd4j.linspace(1, 16, 16).data(), new int[]{2, 2, 2, 2});
        log.info("N2 shape " + n2.slice(1, 1).slice(1));

    }


    @Test
    public void testEndsForSlices() {
        INDArray arr = Nd4j.create(Nd4j.linspace(1, 24, 24).data(), new int[]{4, 3, 2});
        int[] endsForSlices = arr.endsForSlices();
        assertEquals(true, Arrays.equals(new int[]{5, 11, 17, 23}, endsForSlices));
    }


    @Test
    public void testFlatten() {
        INDArray arr = Nd4j.create(Nd4j.linspace(1, 4, 4).data(), new int[]{2, 2});
        INDArray flattened = arr.ravel();
        assertEquals(arr.length(), flattened.length());
        assertEquals(true, Shape.shapeEquals(new int[]{1, arr.length()}, flattened.shape()));
        for (int i = 0; i < arr.length(); i++) {
            assertEquals(i + 1, flattened.getFloat(i), 1e-1);
        }
        assertTrue(flattened.isVector());


        INDArray n = Nd4j.create(Nd4j.ones(27).data(), new int[]{3, 3, 3});
        INDArray nFlattened = n.ravel();
        assertTrue(nFlattened.isVector());

        INDArray n1 = Nd4j.linspace(1, 24, 24);
        assertEquals(n1, Nd4j.linspace(1, 24, 24).reshape(4, 3, 2).ravel());


    }

    @Test
    public void testVectorDimensionMulti() {
        INDArray arr = Nd4j.create(Nd4j.linspace(1, 24, 24).data(), new int[]{4, 3, 2});
        final AtomicInteger count = new AtomicInteger(0);

        arr.iterateOverDimension(arr.shape().length - 1, new SliceOp() {


            /**
             * Operates on an ndarray slice
             *
             * @param nd the result to operate on
             */
            @Override
            public void operate(INDArray nd) {
                INDArray test = nd;
                if (count.get() == 0) {
                    INDArray answer = Nd4j.create(new float[]{1, 2}, new int[]{2});
                    assertEquals(answer, test);
                } else if (count.get() == 1) {
                    INDArray answer = Nd4j.create(new float[]{3, 4}, new int[]{2});
                    assertEquals(answer, test);
                } else if (count.get() == 2) {
                    INDArray answer = Nd4j.create(new float[]{5, 6}, new int[]{2});
                    assertEquals(answer, test);
                } else if (count.get() == 3) {
                    INDArray answer = Nd4j.create(new float[]{7, 8}, new int[]{2});
                    assertEquals(answer, test);
                    answer.data().destroy();
                } else if (count.get() == 4) {
                    INDArray answer = Nd4j.create(new float[]{9, 10}, new int[]{2});
                    assertEquals(answer, test);
                    answer.data().destroy();
                } else if (count.get() == 5) {
                    INDArray answer = Nd4j.create(new float[]{11, 12}, new int[]{2});
                    assertEquals(answer, test);
                    answer.data().destroy();
                }


                count.incrementAndGet();
            }
        }, false);
    }


}
