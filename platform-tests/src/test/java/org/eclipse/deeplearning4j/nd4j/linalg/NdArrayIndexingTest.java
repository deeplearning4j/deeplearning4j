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

package org.eclipse.deeplearning4j.nd4j.linalg;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.SpecifiedIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for NDArray indexing, get, put, row/column access extracted from Nd4jTestsC.
 */
@Slf4j
@NativeTag
@Tag(TagNames.FILE_IO)
public class NdArrayIndexingTest extends BaseNd4jTestWithBackends {

    @Override
    public long getTimeoutMilliseconds() {
        return 90000;
    }

    @BeforeEach
    public void before() throws Exception {
        Nd4j.getRandom().setSeed(123);
        Nd4j.getExecutioner().enableDebugMode(false);
        Nd4j.getExecutioner().enableVerboseMode(false);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetRowEdgeCase(Nd4jBackend backend) {
        INDArray orig = Nd4j.linspace(1, 300, 300, DataType.DOUBLE).reshape('c', 100, 3);
        INDArray col = orig.getColumn(0).reshape(100, 1);

        for (int i = 0; i < 100; i++) {
            INDArray row = col.getRow(i);
            INDArray rowDup = row.dup();
            double d = orig.getDouble(i, 0);
            double d2 = col.getDouble(i);
            double dRowDup = rowDup.getDouble(0);
            double dRow = row.getDouble(0);

            String s = String.valueOf(i);
            assertEquals(d, d2, 0.0, s);
            assertEquals(d, dRowDup, 0.0, s);   //Fails
            assertEquals(d, dRow, 0.0, s);      //Fails
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetColumns(Nd4jBackend backend) {
        INDArray matrix = Nd4j.linspace(1, 6, 6, DataType.DOUBLE).reshape(2, 3);
        INDArray matrixGet = matrix.getColumns(1, 2);
        INDArray matrixAssertion = Nd4j.create(new double[][] {{2, 3}, {5, 6}});
        assertEquals(matrixAssertion, matrixGet);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLinearViewGetAndPut(Nd4jBackend backend) {
        INDArray test = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray linear = test.reshape(-1);
        linear.putScalar(2, 6);
        linear.putScalar(3, 7);
        assertEquals(6, linear.getFloat(2), 1e-1, getFailureMessage(backend));
        assertEquals(7, linear.getFloat(3), 1e-1, getFailureMessage(backend));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetFromRowVector(Nd4jBackend backend) {
        INDArray matrix = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray rowGet = matrix.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, 2));
        assertArrayEquals(new long[] {2}, rowGet.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testColumns(Nd4jBackend backend) {
        INDArray arr = Nd4j.create(new long[] {3, 2}).castTo(DataType.DOUBLE);
        INDArray column2 = arr.getColumn(0);
        INDArray column = Nd4j.create(new double[] {1, 2, 3}, new long[] {3});
        arr.putColumn(0, column);

        INDArray firstColumn = arr.getColumn(0);

        assertEquals(column, firstColumn);


        INDArray column1 = Nd4j.create(new double[] {4, 5, 6}, new long[] {3});
        arr.putColumn(1, column1);
        INDArray testRow1 = arr.getColumn(1);
        assertEquals(column1, testRow1);


        INDArray evenArr = Nd4j.create(new double[] {1, 2, 3, 4}, new long[] {2, 2});
        INDArray put = Nd4j.create(new double[] {5, 6}, new long[] {2});
        evenArr.putColumn(1, put);
        INDArray testColumn = evenArr.getColumn(1);
        assertEquals(put, testColumn);


        INDArray n = Nd4j.create(Nd4j.linspace(1, 4, 4, DataType.DOUBLE).data(), new long[] {2, 2});
        INDArray column23 = n.getColumn(0);
        INDArray column12 = Nd4j.create(new double[] {1, 3}, new long[] {2});
        assertEquals(column23, column12);


        INDArray column0 = n.getColumn(1);
        INDArray column01 = Nd4j.create(new double[] {2, 4}, new long[] {2});
        assertEquals(column0, column01);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPutRow(Nd4jBackend backend) {
        INDArray d = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray slice1 = d.slice(1);
        INDArray n = d.dup();

        //works fine according to matlab, let's go with it..
        //reproduce with:  A = newShapeNoCopy(linspace(1,4,4),[2 2 ]);
        //A(1,2) % 1 index based
        float nFirst = 2;
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


        INDArray nLast = Nd4j.create(Nd4j.linspace(1, 4, 4, DataType.DOUBLE).data(), new long[] {2, 2});
        INDArray row = nLast.getRow(1);
        INDArray row1 = Nd4j.create(new double[] {3, 4});
        assertEquals(row, row1);


        INDArray arr = Nd4j.create(new long[] {3, 2});
        INDArray evenRow = Nd4j.create(new double[] {1, 2});
        arr.putRow(0, evenRow);
        INDArray firstRow = arr.getRow(0);
        assertEquals(true, Shape.shapeEquals(new long[] {2}, firstRow.shape()));
        INDArray testRowEven = arr.getRow(0);
        assertEquals(evenRow, testRowEven);


        INDArray row12 = Nd4j.create(new double[] {5, 6}, new long[] {2});
        arr.putRow(1, row12);
        assertEquals(true, Shape.shapeEquals(new long[] {2}, arr.getRow(0).shape()));
        INDArray testRow1 = arr.getRow(1);
        assertEquals(row12, testRow1);


        INDArray multiSliceTest = Nd4j.create(Nd4j.linspace(1, 16, 16, DataType.DOUBLE).data(), new long[] {4, 2, 2});
        INDArray test = Nd4j.create(new double[] {5, 6}, new long[] {2});
        INDArray test2 = Nd4j.create(new double[] {7, 8}, new long[] {2});

        INDArray multiSliceRow1 = multiSliceTest.slice(1).getRow(0);
        INDArray multiSliceRow2 = multiSliceTest.slice(1).getRow(1);

        assertEquals(test, multiSliceRow1);
        assertEquals(test2, multiSliceRow2);


        INDArray threeByThree = Nd4j.create(3, 3);
        INDArray threeByThreeRow1AndTwo = threeByThree.get(NDArrayIndex.interval(1, 2), NDArrayIndex.all());
        threeByThreeRow1AndTwo.putRow(1, Nd4j.ones(3));
        assertEquals(Nd4j.ones(3), threeByThreeRow1AndTwo.getRow(0));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetIntervalEdgeCase(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int[] shape = {3, 2, 4};
        INDArray arr3d = Nd4j.rand(shape).castTo(DataType.DOUBLE);

        INDArray get0 = arr3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 1));
        INDArray getPoint0 = arr3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0));
        get0 = get0.reshape(getPoint0.shape());
        INDArray tad0 = arr3d.tensorAlongDimension(0, 1, 0);

        assertTrue(get0.equals(getPoint0)); //OK
        assertTrue(getPoint0.equals(tad0)); //OK

        INDArray get1 = arr3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(1, 2));
        INDArray getPoint1 = arr3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(1));
        get1 = get1.reshape(getPoint1.shape());
        INDArray tad1 = arr3d.tensorAlongDimension(1, 1, 0);

        assertTrue(getPoint1.equals(tad1)); //OK
        assertTrue(get1.equals(getPoint1)); //Fails
        assertTrue(get1.equals(tad1));

        INDArray get2 = arr3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(2, 3));
        INDArray getPoint2 = arr3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(2));
        get2 = get2.reshape(getPoint2.shape());
        INDArray tad2 = arr3d.tensorAlongDimension(2, 1, 0);

        assertTrue(getPoint2.equals(tad2)); //OK
        assertTrue(get2.equals(getPoint2)); //Fails
        assertTrue(get2.equals(tad2));

        INDArray get3 = arr3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(3, 4));
        INDArray getPoint3 = arr3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(3));
        get3 = get3.reshape(getPoint3.shape());
        INDArray tad3 = arr3d.tensorAlongDimension(3, 1, 0);

        assertTrue(getPoint3.equals(tad3)); //OK
        assertTrue(get3.equals(getPoint3)); //Fails
        assertTrue(get3.equals(tad3));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetIntervalEdgeCase2(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int[] shape = {3, 2, 4};
        INDArray arr3d = Nd4j.rand(shape).castTo(DataType.DOUBLE);

        for (int x = 0; x < 4; x++) {
            INDArray getInterval = arr3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(x, x + 1)); //3d
            INDArray getPoint = arr3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(x)); //2d
            INDArray tad = arr3d.tensorAlongDimension(x, 1, 0); //2d

            assertEquals(getPoint, tad);
            assertArrayEquals(getInterval.shape(), new long[] {3, 2, 1});
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 2; j++) {
                    assertEquals(getInterval.getDouble(i, j, 0), getPoint.getDouble(i, j), 1e-1);
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetRow(Nd4jBackend backend) {
        INDArray arr = Nd4j.ones(10, 4).castTo(DataType.DOUBLE);
        for (int i = 0; i < 10; i++) {
            INDArray row = arr.getRow(i);
            assertArrayEquals(row.shape(), new long[] {4});
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetPermuteReshapeSub(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        INDArray first = Nd4j.rand(new long[] {10, 4}).castTo(DataType.DOUBLE);

        //Reshape, as per RnnOutputLayer etc on labels
        INDArray orig3d = Nd4j.rand(new long[] {2, 4, 15});
        INDArray subset3d = orig3d.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(5, 10));
        INDArray permuted = subset3d.permute(0, 2, 1);
        val newShape = new long[]{subset3d.size(0) * subset3d.size(2), subset3d.size(1)};
        INDArray second = permuted.reshape(newShape);

        assertArrayEquals(first.shape(), second.shape());
        assertEquals(first.length(), second.length());
        assertArrayEquals(first.stride(), second.stride());

        first.sub(second); //Exception
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPutAtIntervalIndexWithStride(Nd4jBackend backend) {
        INDArray n1 = Nd4j.create(3, 3).assign(0.0).castTo(DataType.DOUBLE);
        INDArrayIndex[] indices = {NDArrayIndex.interval(0, 2, 3), NDArrayIndex.all()};
        n1.put(indices, 1);
        INDArray expected = Nd4j.create(new double[][] {{1d, 1d, 1d}, {0d, 0d, 0d}, {1d, 1d, 1d}});
        assertEquals(expected, n1);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVPull1(Nd4jBackend backend) {
        int indexes[] = new int[] {0, 2, 4};
        INDArray array = Nd4j.linspace(1, 25, 25, DataType.DOUBLE).reshape(5, 5);
        INDArray assertion = Nd4j.createUninitialized(new long[] {3, 5}, 'f');
        for (int i = 0; i < 3; i++) {
            assertion.putRow(i, array.getRow(indexes[i]));
        }

        INDArray result = Nd4j.pullRows(array, 1, indexes, 'f');

        assertEquals(3, result.rows());
        assertEquals(5, result.columns());
        assertEquals(assertion, result);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPullRowsValidation1(Nd4jBackend backend) {
        assertThrows(IllegalStateException.class, () -> {
            Nd4j.pullRows(Nd4j.create(10, 10), 2, new int[] {0, 1, 2});
        });
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPullRowsValidation2(Nd4jBackend backend) {
        assertThrows(IllegalStateException.class, () -> {
            Nd4j.pullRows(Nd4j.create(10, 10), 1, new int[] {0, -1, 2});
        });
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPullRowsValidation3(Nd4jBackend backend) {
        assertThrows(IllegalStateException.class, () -> {
            Nd4j.pullRows(Nd4j.create(10, 10), 1, new int[] {0, 1, 10});
        });
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPullRowsValidation4(Nd4jBackend backend) {
        assertThrows(IllegalStateException.class, () -> {
            Nd4j.pullRows(Nd4j.create(3, 10), 1, new int[] {0, 1, 2, 3});
        });
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPullRowsValidation5(Nd4jBackend backend) {
        assertThrows(IllegalStateException.class, () -> {
            Nd4j.pullRows(Nd4j.create(3, 10), 1, new int[] {0, 1, 2}, 'e');
        });
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testVPull2(Nd4jBackend backend) {
        val indexes = new int[] {0, 2, 4};
        INDArray array = Nd4j.linspace(1, 25, 25, DataType.DOUBLE).reshape(5, 5);
        INDArray assertion = Nd4j.createUninitialized(new long[] {3, 5}, 'c');
        for (int i = 0; i < 3; i++) {
            assertion.putRow(i, array.getRow(indexes[i]));
        }

        INDArray result = Nd4j.pullRows(array, 1, indexes, 'c');

        assertEquals(3, result.rows());
        assertEquals(5, result.columns());
        assertEquals(assertion, result);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGet() {
        //https://github.com/eclipse/deeplearning4j/issues/6133
        INDArray m = Nd4j.linspace(0, 99, 100, DataType.DOUBLE).reshape('c', 10, 10);
        INDArray exp = Nd4j.create(new double[]{5, 15, 25, 35, 45, 55, 65, 75, 85, 95}, new int[]{10});
        INDArray col = m.getColumn(5);

        for (int i = 0; i < 10; i++) {
            col.slice(i);
        }

        //First element: index 5
        //Last element: index 95
        //Column is a view; data() returns the full underlying buffer (100 elements)
        assertEquals(5, m.getDouble(5), 1e-6);
        assertEquals(95, m.getDouble(95), 1e-6);
        assertEquals(100, col.data().length());

        assertEquals(exp, col);
        assertEquals(exp.toString(), col.toString());
        assertArrayEquals(exp.toDoubleVector(), col.toDoubleVector(), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWhere1() {
        INDArray arr = Nd4j.create(new boolean[][]{{false, true, false}, {false, false, true}, {false, false, true}});
        INDArray[] exp = new INDArray[]{
                Nd4j.createFromArray(new long[]{0, 1, 2}),
                Nd4j.createFromArray(new long[]{1, 2, 2})};

        INDArray[] act = Nd4j.where(arr, null, null);

        assertArrayEquals(exp, act);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWhere2() {
        INDArray arr = Nd4j.create(DataType.BOOL, 3, 3, 3);
        arr.putScalar(0, 1, 0, 1.0);
        arr.putScalar(1, 2, 1, 1.0);
        arr.putScalar(2, 2, 1, 1.0);
        INDArray[] exp = new INDArray[]{
                Nd4j.createFromArray(new long[]{0, 1, 2}),
                Nd4j.createFromArray(new long[]{1, 2, 2}),
                Nd4j.createFromArray(new long[]{0, 1, 1})
        };

        INDArray[] act = Nd4j.where(arr, null, null);

        assertArrayEquals(exp, act);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWhere3() {
        INDArray arr = Nd4j.create(new boolean[][]{{false, true, false}, {false, false, true}, {false, false, true}});
        INDArray x = Nd4j.valueArrayOf(3, 3, 1.0);
        INDArray y = Nd4j.valueArrayOf(3, 3, 2.0);
        INDArray exp = Nd4j.create(new double[][]{
                {1, 2, 1},
                {1, 1, 2},
                {1, 1, 2}});

        INDArray[] act = Nd4j.where(arr, x, y);
        assertEquals(1, act.length);

        assertEquals(exp, act[0]);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWhereEmpty() {
        INDArray inArray = Nd4j.zeros(2, 3);
        inArray.putScalar(0, 0, 10.0f);
        inArray.putScalar(1, 2, 10.0f);

        INDArray mask1 = inArray.match(1, Conditions.greaterThanOrEqual(1));

        assertEquals(1, mask1.castTo(DataType.INT).maxNumber().intValue()); // ! Not Empty Match

        INDArray[] matchIndexes = Nd4j.where(mask1, null, null);

        assertArrayEquals(new int[] {0, 1}, matchIndexes[0].toIntVector());
        assertArrayEquals(new int[] {0, 2}, matchIndexes[1].toIntVector());

        INDArray mask2 = inArray.match(11, Conditions.greaterThanOrEqual(11));

        assertEquals(0, mask2.castTo(DataType.INT).maxNumber().intValue());

        INDArray[] matchIndexes2 = Nd4j.where(mask2, null, null);
        for (int i = 0; i < matchIndexes2.length; i++) {
            assertTrue(matchIndexes2[i].isEmpty());
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPutSpecifiedIndex() {
        long[][] ss = new long[][]{{3, 4}, {3, 4, 5}, {3, 4, 5, 6}};
        long[][] st = new long[][]{{4, 4}, {4, 4, 5}, {4, 4, 5, 6}};
        long[][] ds = new long[][]{{1, 4}, {1, 4, 5}, {1, 4, 5, 6}};

        for (int test = 0; test < ss.length; test++) {
            long[] shapeSource = ss[test];
            long[] shapeTarget = st[test];
            long[] diffShape = ds[test];

            final INDArray source = Nd4j.ones(shapeSource);
            final INDArray target = Nd4j.zeros(shapeTarget);

            final INDArrayIndex[] targetIndexes = new INDArrayIndex[shapeTarget.length];
            Arrays.fill(targetIndexes, NDArrayIndex.all());
            int[] arr = new int[(int) shapeSource[0]];
            for (int i = 0; i < arr.length; i++) {
                arr[i] = i;
            }
            targetIndexes[0] = new SpecifiedIndex(arr);

            target.put(targetIndexes, source);
            final INDArray expected = Nd4j.concat(0, Nd4j.ones(shapeSource), Nd4j.zeros(diffShape));
            assertEquals(expected, target, "Expected array to be set!");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPutSpecifiedIndices2d() {
        INDArray arr = Nd4j.create(3, 4);
        INDArray toPut = Nd4j.create(new double[]{1, 2, 3, 4}, new int[]{2, 2}, 'c');
        INDArrayIndex[] indices = new INDArrayIndex[]{
                NDArrayIndex.indices(0, 2),
                NDArrayIndex.indices(1, 3)};

        INDArray exp = Nd4j.create(new double[][]{
                {0, 1, 0, 2},
                {0, 0, 0, 0},
                {0, 3, 0, 4}});

        arr.put(indices, toPut);
        assertEquals(exp, arr);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPutSpecifiedIndices3d() {
        INDArray arr = Nd4j.create(2, 3, 4);
        INDArray toPut = Nd4j.create(new double[]{1, 2, 3, 4}, new int[]{1, 2, 2}, 'c');
        INDArrayIndex[] indices = new INDArrayIndex[]{
                NDArrayIndex.point(1),
                NDArrayIndex.indices(0, 2),
                NDArrayIndex.indices(1, 3)};

        INDArray exp = Nd4j.create(2, 3, 4);
        exp.putScalar(1, 0, 1, 1);
        exp.putScalar(1, 0, 3, 2);
        exp.putScalar(1, 2, 1, 3);
        exp.putScalar(1, 2, 3, 4);

        arr.put(indices, toPut);
        assertEquals(exp, arr);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSpecifiedIndexArraySize1(Nd4jBackend backend) {
        long[] shape = {2, 2, 2, 2};
        INDArray in = Nd4j.create(shape);
        INDArrayIndex[] idx1 = new INDArrayIndex[]{NDArrayIndex.all(), new SpecifiedIndex(0), NDArrayIndex.all(), NDArrayIndex.all()};

        INDArray arr = in.get(idx1);
        long[] expShape = new long[]{2, 1, 2, 2};
        assertArrayEquals(expShape, arr.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRowsEdgeCaseView() {
        INDArray arr = Nd4j.linspace(0, 9, 10, DataType.DOUBLE).reshape('f', 5, 2).dup('c');    //0,1,2... along columns
        INDArray view = arr.getColumn(0);
        assertEquals(Nd4j.createFromArray(0.0, 1.0, 2.0, 3.0, 4.0), view);
        int[] idxs = new int[]{0, 2, 3, 4};

        INDArray out = Nd4j.pullRows(view.reshape(5, 1), 1, idxs);
        INDArray exp = Nd4j.createFromArray(new double[]{0, 2, 3, 4}).reshape(4, 1);

        assertEquals(exp, out);   //Failing here
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPullRowsFailure(Nd4jBackend backend) {
        assertThrows(IllegalArgumentException.class, () -> {
            val idxs = new int[]{0, 2, 3, 4};
            val out = Nd4j.pullRows(Nd4j.createFromArray(0.0, 1.0, 2.0, 3.0, 4.0), 0, idxs);
        });
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetColumnRowVector() {
        INDArray arr = Nd4j.create(1, 4);
        INDArray col = arr.getColumn(0);
        assertArrayEquals(new long[]{1}, col.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPutRowValidation(Nd4jBackend backend) {
        assertThrows(IllegalArgumentException.class, () -> {
            val matrix = Nd4j.create(5, 10);
            val row = Nd4j.create(25);

            matrix.putRow(1, row);
        });
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPutColumnValidation(Nd4jBackend backend) {
        assertThrows(IllegalArgumentException.class, () -> {
            val matrix = Nd4j.create(5, 10);
            val column = Nd4j.create(25);

            matrix.putColumn(1, column);
        });
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetWhereINDArray(Nd4jBackend backend) {
        INDArray input = Nd4j.create(new double[] { 1, -3, 4, 8, -2, 5 });
        INDArray comp = Nd4j.create(new double[]{2, -3, 1, 1, -2, 1 });
        INDArray expected = Nd4j.create(new double[] { 4, 8, 5 });
        INDArray actual = input.getWhere(comp, Conditions.greaterThan(1));

        assertEquals(expected, actual);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetWhereNumber(Nd4jBackend backend) {
        INDArray input = Nd4j.create(new double[] { 1, -3, 4, 8, -2, 5 });
        INDArray expected = Nd4j.create(new double[] { 8, 5 });
        INDArray actual = input.getWhere(4, Conditions.greaterThan(1));

        assertEquals(expected, actual);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPut3dPointPointAllSyncsCorrectly(Nd4jBackend backend) {
        // Regression test: put() with point/point/all indices on a 3D array must
        // correctly update the underlying data so that minNumber()/maxNumber() and
        // subsequent operations (e.g. neural network forward pass) see the new values.
        // On CUDA, put() delegates to get(view).assign() which runs a CUDA kernel.
        // If host/device sync is broken, minNumber()/maxNumber() may return stale values.

        int batch = 1, seqLen = 10, hidden = 8;
        INDArray arr = Nd4j.zeros(DataType.FLOAT, batch, seqLen, hidden);

        // Put large values at positions 2 and 5 (simulating vision embedding injection)
        INDArray bigRow1 = Nd4j.ones(DataType.FLOAT, hidden).mul(50.0f);
        INDArray bigRow2 = Nd4j.ones(DataType.FLOAT, hidden).mul(-40.0f);

        // put large values via view assign
        INDArray view2 = arr.get(NDArrayIndex.point(0), NDArrayIndex.point(2), NDArrayIndex.all());
        view2.assign(bigRow1);
        Nd4j.getExecutioner().commit();

        arr.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.point(5), NDArrayIndex.all()}, bigRow2);
        Nd4j.getExecutioner().commit();

        // The real assertions
        double max = arr.maxNumber().doubleValue();
        double min = arr.minNumber().doubleValue();
        assertEquals(50.0, max, 1e-3, "maxNumber() should reflect put() values. Got " + max);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
