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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for NDArray sort and shuffle operations extracted from Nd4jTestsC.
 */
@Slf4j
@NativeTag
@Tag(TagNames.FILE_IO)
public class NdArraySortTest extends BaseNd4jTestWithBackends {

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
    public void testSort(Nd4jBackend backend) {
        INDArray toSort = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        INDArray ascending = Nd4j.sort(toSort.dup(), 1, true);
        //rows are already sorted
        assertEquals(toSort, ascending);

        INDArray columnSorted = Nd4j.create(new double[] {2, 1, 4, 3}, new long[] {2, 2});
        INDArray sorted = Nd4j.sort(toSort.dup(), 1, false);
        assertEquals(columnSorted, sorted);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSortRows(Nd4jBackend backend) {
        int nRows = 10;
        int nCols = 5;
        Random r = new Random(12345);

        for (int i = 0; i < nCols; i++) {
            INDArray in = Nd4j.linspace(1, nRows * nCols, nRows * nCols, DataType.DOUBLE).reshape(nRows, nCols);

            List<Integer> order = new ArrayList<>(nRows);
            //in.row(order(i)) should end up as out.row(i) - ascending
            //in.row(order(i)) should end up as out.row(nRows-j-1) - descending
            for (int j = 0; j < nRows; j++)
                order.add(j);
            Collections.shuffle(order, r);
            for (int j = 0; j < nRows; j++)
                in.putScalar(new long[] {j, i}, order.get(j));

            INDArray outAsc = Nd4j.sortRows(in, i, true);
            INDArray outDesc = Nd4j.sortRows(in, i, false);

            for (int j = 0; j < nRows; j++) {
                assertEquals(outAsc.getDouble(j, i), j, 1e-1);
                int origRowIdxAsc = order.indexOf(j);
                assertTrue(outAsc.getRow(j).equals(in.getRow(origRowIdxAsc)));

                assertEquals((nRows - j - 1), outDesc.getDouble(j, i), 0.001f);
                int origRowIdxDesc = order.indexOf(nRows - j - 1);
                assertTrue(outDesc.getRow(j).equals(in.getRow(origRowIdxDesc)));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSortColumns(Nd4jBackend backend) {
        int nRows = 5;
        int nCols = 10;
        Random r = new Random(12345);

        for (int i = 0; i < nRows; i++) {
            INDArray in = Nd4j.rand(new long[] {nRows, nCols});

            List<Integer> order = new ArrayList<>(nRows);
            for (int j = 0; j < nCols; j++)
                order.add(j);
            Collections.shuffle(order, r);
            for (int j = 0; j < nCols; j++)
                in.putScalar(new long[] {i, j}, order.get(j));

            INDArray outAsc = Nd4j.sortColumns(in, i, true);
            INDArray outDesc = Nd4j.sortColumns(in, i, false);

            for (int j = 0; j < nCols; j++) {
                assertTrue(outAsc.getDouble(i, j) == j);
                int origColIdxAsc = order.indexOf(j);
                assertTrue(outAsc.getColumn(j).equals(in.getColumn(origColIdxAsc)));

                assertTrue(outDesc.getDouble(i, j) == (nCols - j - 1));
                int origColIdxDesc = order.indexOf(nCols - j - 1);
                assertTrue(outDesc.getColumn(j).equals(in.getColumn(origColIdxDesc)));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSortWithIndicesDescending(Nd4jBackend backend) {
        INDArray toSort = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        //indices,data
        INDArray[] sorted = Nd4j.sortWithIndices(toSort.dup(), 1, false);
        INDArray sorted2 = Nd4j.sort(toSort.dup(), 1, false);
        assertEquals(sorted[1], sorted2);
        INDArray shouldIndex = Nd4j.create(new double[] {1, 0, 1, 0}, new long[] {2, 2});
        assertEquals(shouldIndex, sorted[0]);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNativeSortView1(Nd4jBackend backend) {
        INDArray matrix = Nd4j.create(10, 10);
        INDArray exp = Nd4j.linspace(0, 9, 10, DataType.DOUBLE);
        int cnt = 0;
        for (long i = matrix.rows() - 1; i >= 0; i--) {
            matrix.getRow((int) i).assign(cnt);
            cnt++;
        }

        Nd4j.sort(matrix.getColumn(0), true);

        assertEquals(exp, matrix.getColumn(0));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNativeSort1(Nd4jBackend backend) {
        INDArray array = Nd4j.create(new double[] {9, 2, 1, 7, 6, 5, 4, 3, 8, 0});
        INDArray exp1 = Nd4j.create(new double[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        INDArray exp2 = Nd4j.create(new double[] {9, 8, 7, 6, 5, 4, 3, 2, 1, 0});

        INDArray res = Nd4j.sort(array, true);

        assertEquals(exp1, res);

        res = Nd4j.sort(res, false);

        assertEquals(exp2, res);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNativeSort2(Nd4jBackend backend) {
        INDArray array = Nd4j.rand(1, 10000).castTo(DataType.DOUBLE);

        INDArray res = Nd4j.sort(array, true);
        INDArray exp = res.dup();

        res = Nd4j.sort(res, false);
        res = Nd4j.sort(res, true);

        assertEquals(exp, res);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Tag(TagNames.NEEDS_VERIFY)
    public void testNativeSort3(Nd4jBackend backend) {
        int length = isIntegrationTests() ? 1048576 : 16484;
        INDArray array = Nd4j.linspace(1, length, length, DataType.DOUBLE).reshape(1, -1);
        INDArray exp = array.dup();
        Nd4j.shuffle(array, 0);

        long time1 = System.currentTimeMillis();
        INDArray res = Nd4j.sort(array, true);
        long time2 = System.currentTimeMillis();
        log.info("Time spent: {} ms", time2 - time1);

        assertEquals(exp, res);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNativeSort3_1(Nd4jBackend backend) {
        INDArray array = Nd4j.linspace(1, 2017152, 2017152, DataType.DOUBLE).reshape(1, -1);
        INDArray exp = array.dup();
        Transforms.reverse(array, false);

        long time1 = System.currentTimeMillis();
        INDArray res = Nd4j.sort(array, true);
        long time2 = System.currentTimeMillis();
        log.info("Time spent: {} ms", time2 - time1);

        assertEquals(exp, res);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Tag(TagNames.NEEDS_VERIFY)
    public void testNativeSortAlongDimension1(Nd4jBackend backend) {
        INDArray array = Nd4j.create(1000, 1000);
        INDArray exp1 = Nd4j.linspace(1, 1000, 1000, DataType.DOUBLE);
        INDArray dps = exp1.dup();
        Nd4j.shuffle(dps, 0);

        assertNotEquals(exp1, dps);

        for (int r = 0; r < array.rows(); r++) {
            array.getRow(r).assign(dps);
        }

        long time1 = System.currentTimeMillis();
        INDArray res = Nd4j.sort(array, 1, true);
        long time2 = System.currentTimeMillis();

        log.info("Time spent: {} ms", time2 - time1);

        val e = exp1.toDoubleVector();
        for (int r = 0; r < array.rows(); r++) {
            val d = res.getRow(r).dup();

            assertArrayEquals(e, d.toDoubleVector(), 1e-5);
            assertEquals(exp1, d, "Failed at " + r);
        }
    }

    protected boolean checkIfUnique(INDArray array, int iteration) {
        var jarray = array.data().asInt();
        var set = new HashSet<Integer>();

        for (val v : jarray) {
            if (set.contains(Integer.valueOf(v)))
                throw new IllegalStateException("Duplicate value found: [" + v + "] on iteration " + iteration);

            set.add(Integer.valueOf(v));
        }

        return true;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void shuffleTest(Nd4jBackend backend) {
        for (int e = 0; e < 5; e++) {
            val array = Nd4j.linspace(1, 1011, 1011, DataType.INT);

            checkIfUnique(array, e);
            Nd4j.getExecutioner().commit();

            Nd4j.shuffle(array, 0);
            Nd4j.getExecutioner().commit();

            checkIfUnique(array, e);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Tag(TagNames.NEEDS_VERIFY)
    public void testNativeSortAlongDimension3(Nd4jBackend backend) {
        INDArray array = Nd4j.create(2000, 2000);
        INDArray exp1 = Nd4j.linspace(1, 2000, 2000, DataType.DOUBLE);
        INDArray dps = exp1.dup();

        Nd4j.getExecutioner().commit();
        Nd4j.shuffle(dps, 0);

        assertNotEquals(exp1, dps);

        for (int r = 0; r < array.rows(); r++) {
            array.getRow(r).assign(dps);
        }

        val arow = array.getRow(0).toFloatVector();

        long time1 = System.currentTimeMillis();
        INDArray res = Nd4j.sort(array, 1, true);
        long time2 = System.currentTimeMillis();

        log.info("Time spent: {} ms", time2 - time1);

        val jexp = exp1.toFloatVector();
        for (int r = 0; r < array.rows(); r++) {
            val jrow = res.getRow(r).toFloatVector();
            assertArrayEquals(jexp, jrow, 1e-5f, "Failed at " + r);
            assertEquals(exp1, res.getRow(r), "Failed at " + r);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Tag(TagNames.NEEDS_VERIFY)
    public void testNativeSortAlongDimension2(Nd4jBackend backend) {
        INDArray array = Nd4j.create(100, 10);
        INDArray exp1 = Nd4j.create(new double[] {9, 8, 7, 6, 5, 4, 3, 2, 1, 0});

        for (int r = 0; r < array.rows(); r++) {
            array.getRow(r).assign(Nd4j.create(new double[] {3, 8, 2, 7, 5, 6, 4, 9, 1, 0}));
        }

        INDArray res = Nd4j.sort(array, 1, false);

        for (int r = 0; r < array.rows(); r++) {
            assertEquals(exp1, res.getRow(r).dup(), "Failed at " + r);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSmallSort() {
        INDArray arr = Nd4j.createFromArray(0.5, 0.4, 0.1, 0.2);
        INDArray expected = Nd4j.createFromArray(0.1, 0.2, 0.4, 0.5);
        INDArray sorted = Nd4j.sort(arr, true);
        assertEquals(expected, sorted);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
