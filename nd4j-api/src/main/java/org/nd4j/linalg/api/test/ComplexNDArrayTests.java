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

import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.SliceOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.Shape;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

/**
 * Tests for a complex ndarray
 *
 * @author Adam Gibson
 */
public abstract class ComplexNDArrayTests {

    private static Logger log = LoggerFactory.getLogger(ComplexNDArrayTests.class);

    @Before
    public void before() {
        Nd4j.factory().setOrder('c');
    }

    @Test
    public void testConstruction() {

        IComplexNDArray arr2 = Nd4j.createComplex(new int[]{3, 2});
        assertEquals(3, arr2.rows());
        assertEquals(arr2.rows(), arr2.rows());
        assertEquals(2, arr2.columns());
        assertEquals(arr2.columns(), arr2.columns());
        assertTrue(arr2.isMatrix());


        IComplexNDArray arr = Nd4j.createComplex(new double[]{0, 1}, new int[]{1});
        //only each complex double: one element
        assertEquals(1, arr.length());
        //both real and imaginary components
        assertEquals(2, arr.data().length());
        IComplexNumber n1 = (IComplexNumber) arr.getScalar(0).element();
        assertEquals(0, n1.realComponent().doubleValue(), 1e-1);


        IComplexDouble[] two = new IComplexDouble[2];
        two[0] = Nd4j.createDouble(1, 0);
        two[1] = Nd4j.createDouble(2, 0);
        double[] testArr = {1, 0, 2, 0};
        IComplexNDArray assertComplexDouble = Nd4j.createComplex(testArr, new int[]{2});
        IComplexNDArray testComplexDouble = Nd4j.createComplex(two, new int[]{2});
        assertEquals(assertComplexDouble, testComplexDouble);

    }

    @Test
    public void testSortFortran() {

        IComplexNDArray matrix = Nd4j.complexLinSpace(1, 4, 4).reshape(2, 2);
        IComplexNDArray sorted = Nd4j.sort(matrix.dup(), 1, true);
        assertEquals(matrix, sorted);

        IComplexNDArray reversed = Nd4j.createComplex(
                new float[]{2, 0, 1, 0, 4, 0, 3, 0}
                , new int[]{2, 2});

        IComplexNDArray sortedReversed = Nd4j.sort(matrix.dup(), 1, false);
        assertEquals(reversed, sortedReversed);

    }

    @Test
    public void testSort() {
        IComplexNDArray matrix = Nd4j.complexLinSpace(1, 4, 4).reshape(2, 2);
        IComplexNDArray sorted = Nd4j.sort(matrix.dup(), 1, true);
        assertEquals(matrix, sorted);

        IComplexNDArray reversed = Nd4j.createComplex(
                new float[]{2, 0, 1, 0, 4, 0, 3, 0}
                , new int[]{2, 2});

        IComplexNDArray sortedReversed = Nd4j.sort(matrix, 1, false);
        assertEquals(reversed, sortedReversed);

    }


    @Test
    public void testSortWithIndicesDescending() {
        IComplexNDArray toSort = Nd4j.complexLinSpace(1, 4, 4).reshape(2, 2);
        //indices,data
        INDArray[] sorted = Nd4j.sortWithIndices(toSort.dup(), 1, false);
        INDArray sorted2 = Nd4j.sort(toSort.dup(), 1, false);
        assertEquals(sorted[1], sorted2);
        INDArray shouldIndex = Nd4j.create(new float[]{1, 0, 1, 0}, new int[]{2, 2});
        assertEquals(shouldIndex, sorted[0]);


    }


    @Test
    public void testSortWithIndices() {
        IComplexNDArray toSort = Nd4j.complexLinSpace(1, 4, 4).reshape(2, 2);
        //indices,data
        INDArray[] sorted = Nd4j.sortWithIndices(toSort.dup(), 1, true);
        INDArray sorted2 = Nd4j.sort(toSort.dup(), 1, true);
        assertEquals(sorted[1], sorted2);
        INDArray shouldIndex = Nd4j.create(new float[]{0, 1, 0, 1}, new int[]{2, 2});
        assertEquals(shouldIndex, sorted[0]);


    }

    @Test
    public void testDimShuffle() {
        IComplexNDArray n = Nd4j.complexLinSpace(1, 4, 4).reshape(2, 2);
        IComplexNDArray twoOneTwo = n.dimShuffle(new Object[]{0, 'x', 1}, new int[]{0, 1}, new boolean[]{false, false});
        assertTrue(Arrays.equals(new int[]{2, 1, 2}, twoOneTwo.shape()));

        IComplexNDArray reverse = n.dimShuffle(new Object[]{1, 'x', 0}, new int[]{1, 0}, new boolean[]{false, false});
        assertTrue(Arrays.equals(new int[]{2, 1, 2}, reverse.shape()));

    }

    @Test
    public void testPutComplex() {
        INDArray fourTwoTwo = Nd4j.linspace(1, 16, 16).reshape(4, 2, 2);
        IComplexNDArray test = Nd4j.createComplex(4, 2, 2);


        for (int i = 0; i < test.vectorsAlongDimension(0); i++) {
            INDArray vector = fourTwoTwo.vectorAlongDimension(i, 0);
            IComplexNDArray complexVector = test.vectorAlongDimension(i, 0);
            for (int j = 0; j < complexVector.length(); j++) {
                complexVector.putReal(j, vector.getFloat(j));
            }
        }

        for (int i = 0; i < test.vectorsAlongDimension(0); i++) {
            INDArray vector = fourTwoTwo.vectorAlongDimension(i, 0);
            IComplexNDArray complexVector = test.vectorAlongDimension(i, 0);
            assertEquals(vector, complexVector.real());
        }

    }

    @Test
    public void testColumnWithReshape() {
        IComplexNDArray ones = Nd4j.complexOnes(4).reshape(2, 2);
        IComplexNDArray column = Nd4j.createComplex(new float[]{2, 0, 6, 0});
        ones.putColumn(1, column);
        assertEquals(column, ones.getColumn(1));
    }


    @Test
    public void testCreateFromNDArray() {
        INDArray arr = Nd4j.create(new double[][]{{1, 2}, {3, 4}});
        IComplexNDArray complex = Nd4j.createComplex(arr);
        for (int i = 0; i < arr.rows(); i++) {
            for (int j = 0; j < arr.columns(); j++) {
                double d = arr.getFloat(i, j);
                IComplexNumber complexD = complex.getComplex(i, j);
                assertEquals(Nd4j.createDouble(d, 0), complexD);
            }
        }

        Nd4j.factory().setOrder('f');
        INDArray fortran = Nd4j.create(new double[][]{{1, 2}, {3, 4}});
        assertEquals(arr, fortran);

        IComplexNDArray fortranComplex = Nd4j.createComplex(fortran);
        for (int i = 0; i < fortran.rows(); i++) {
            for (int j = 0; j < fortran.columns(); j++) {
                double d = fortran.getFloat(i, j);
                IComplexNumber complexD = fortranComplex.getComplex(i, j);
                assertEquals(Nd4j.createDouble(d, 0), complexD);
            }
        }

        Nd4j.factory().setOrder('c');

    }


    @Test
    public void testSum() {
        IComplexNDArray n = Nd4j.createComplex(Nd4j.create(Nd4j.linspace(1, 8, 8).data(), new int[]{2, 2, 2}));
        assertEquals(Nd4j.createDouble(36, 0), n.sum(Integer.MAX_VALUE).element());
    }


    @Test
    public void testCreateComplexFromReal() {
        INDArray n = Nd4j.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, new int[]{2, 4});
        IComplexNDArray nComplex = Nd4j.createComplex(n);
        for (int i = 0; i < n.vectorsAlongDimension(0); i++) {
            INDArray vec = n.vectorAlongDimension(i, 0);
            IComplexNDArray vecComplex = nComplex.vectorAlongDimension(i, 0);
            assertEquals(vec.length(), vecComplex.length());
            for (int j = 0; j < vec.length(); j++) {
                IComplexNumber currComplex = vecComplex.getComplex(j);
                double curr = vec.getFloat(j);
                assertEquals(curr, currComplex.realComponent().doubleValue(), 1e-1);
            }
            assertEquals(vec, vecComplex.getReal());
        }
    }


    @Test
    public void testVectorAlongDimension() {
        INDArray n = Nd4j.linspace(1, 8, 8).reshape(2, 4);
        IComplexNDArray nComplex = Nd4j.createComplex(Nd4j.linspace(1, 8, 8)).reshape(2, 4);
        assertEquals(n.vectorsAlongDimension(0), nComplex.vectorsAlongDimension(0));

        for (int i = 0; i < n.vectorsAlongDimension(0); i++) {
            INDArray vec = n.vectorAlongDimension(i, 0);
            IComplexNDArray vecComplex = nComplex.vectorAlongDimension(i, 0);
            assertEquals(vec.length(), vecComplex.length());
            for (int j = 0; j < vec.length(); j++) {
                IComplexNumber currComplex = vecComplex.getComplex(j);
                double curr = vec.getFloat(j);
                assertEquals(curr, currComplex.realComponent().doubleValue(), 1e-1);
            }
            assertEquals(vec, vecComplex.getReal());
        }


    }

    @Test
    public void testVectorGet() {
        IComplexNDArray arr = Nd4j.createComplex(Nd4j.create(Nd4j.linspace(1, 8, 8).data(), new int[]{8}));
        for (int i = 0; i < arr.length(); i++) {
            IComplexNumber curr = arr.getComplex(i);
            assertEquals(Nd4j.createDouble(i + 1, 0), curr);
        }

        IComplexNDArray matrix = Nd4j.createComplex(Nd4j.create(Nd4j.linspace(1, 8, 8).data(), new int[]{2, 4}));
        IComplexNDArray row = matrix.getRow(1);
        IComplexNDArray column = matrix.getColumn(1);

        IComplexNDArray validate = Nd4j.createComplex(Nd4j.create(new double[]{5, 6, 7, 8}, new int[]{4}));
        IComplexNumber d = row.getComplex(3);
        assertEquals(Nd4j.createDouble(8, 0), d);
        assertEquals(row, validate);

        IComplexNumber d2 = column.getComplex(1);

        assertEquals(Nd4j.createDouble(6, 0), d2);


    }

    @Test
    public void testLinearView() {
        IComplexNDArray n = Nd4j.complexLinSpace(1, 4, 4).reshape(2, 2);
        IComplexNDArray row = n.getRow(1);
        IComplexNDArray linear = row.linearView();
        assertEquals(row, linear);
    }

    @Test
    public void testSwapAxesFortranOrder() {
        Nd4j.factory().setOrder('f');

        IComplexNDArray n = Nd4j.createComplex(Nd4j.linspace(1, 30, 30)).reshape(new int[]{3, 5, 2});
        IComplexNDArray slice = n.swapAxes(2, 1);
        IComplexNDArray assertion = Nd4j.createComplex(new double[]{1, 0, 4, 0, 7, 0, 10, 0, 13, 0});
        IComplexNDArray test = slice.slice(0).slice(0);
        assertEquals(assertion, test);
    }


    @Test
    public void testSwapAxes() {
        IComplexNDArray n = Nd4j.createComplex(Nd4j.create(new double[]{1, 2, 3}, new int[]{3, 1}));
        IComplexNDArray swapped = n.swapAxes(1, 0);
        assertEquals(n.transpose(), swapped);
        //vector despite being transposed should have same linear index
        assertEquals(swapped.getScalar(0), n.getScalar(0));
        assertEquals(swapped.getScalar(1), n.getScalar(1));
        assertEquals(swapped.getScalar(2), n.getScalar(2));

        IComplexNDArray n2 = Nd4j.createComplex(Nd4j.create(Nd4j.linspace(0, 7, 8).data(), new int[]{2, 2, 2}));
        IComplexNDArray assertion = n2.permute(new int[]{2, 1, 0});
        IComplexNDArray validate = Nd4j.createComplex(Nd4j.create(new double[]{0, 4, 2, 6, 1, 5, 3, 7}, new int[]{2, 2, 2}));
        assertEquals(validate, assertion);


        IComplexNDArray v1 = Nd4j.createComplex(Nd4j.create(Nd4j.linspace(1, 8, 8).data(), new int[]{8, 1}));
        IComplexNDArray swap = v1.swapAxes(1, 0);
        IComplexNDArray transposed = v1.transpose();
        assertEquals(swap, transposed);


        transposed.put(1, Nd4j.scalar(9));
        swap.put(1, Nd4j.scalar(9));
        assertEquals(transposed, swap);
        assertEquals(transposed.getScalar(1).element(), swap.getScalar(1).element());


        IComplexNDArray row = n2.slice(0).getRow(1);
        row.put(1, Nd4j.scalar(9));

        IComplexNumber n3 = (IComplexNumber) row.getScalar(1).element();

        assertEquals(9, n3.realComponent().doubleValue(), 1e-1);


    }


    @Test
    public void testSlice() {
        Nd4j.MAX_ELEMENTS_PER_SLICE = -1;
        Nd4j.MAX_SLICES_TO_PRINT = -1;
        INDArray arr = Nd4j.create(Nd4j.linspace(1, 24, 24).data(), new int[]{4, 3, 2});
        IComplexNDArray arr2 = Nd4j.createComplex(arr);
        assertEquals(arr, arr2.getReal());

        INDArray firstSlice = arr.slice(0);
        INDArray firstSliceTest = arr2.slice(0).getReal();
        assertEquals(firstSlice, firstSliceTest);


        INDArray secondSlice = arr.slice(1);
        INDArray secondSliceTest = arr2.slice(1).getReal();
        assertEquals(secondSlice, secondSliceTest);


        INDArray slice0 = Nd4j.create(new double[]{1, 2, 3, 4, 5, 6}, new int[]{3, 2});
        INDArray slice2 = Nd4j.create(new double[]{7, 8, 9, 10, 11, 12}, new int[]{3, 2});


        IComplexNDArray testSliceComplex = arr2.slice(0);
        IComplexNDArray testSliceComplex2 = arr2.slice(1);

        INDArray testSlice0 = testSliceComplex.getReal();
        INDArray testSlice1 = testSliceComplex2.getReal();

        assertEquals(slice0, testSlice0);
        assertEquals(slice2, testSlice1);


        INDArray n2 = Nd4j.create(Nd4j.linspace(1, 30, 30).data(), new int[]{3, 5, 2});
        INDArray swapped = n2.swapAxes(n2.shape().length - 1, 1);
        INDArray firstSlice2 = swapped.slice(0).slice(0);
        IComplexNDArray testSlice = Nd4j.createComplex(firstSlice2);
        IComplexNDArray testNoOffset = Nd4j.createComplex(new double[]{1, 0, 3, 0, 5, 0, 7, 0, 9, 0}, new int[]{5});
        assertEquals(testSlice, testNoOffset);


    }

    @Test
    public void testSliceConstructor() {
        List<IComplexNDArray> testList = new ArrayList<>();
        for (int i = 0; i < 5; i++)
            testList.add(Nd4j.complexScalar(i + 1));

        IComplexNDArray test = Nd4j.createComplex(testList, new int[]{testList.size()});
        IComplexNDArray expected = Nd4j.createComplex(Nd4j.create(new double[]{1, 2, 3, 4, 5}, new int[]{5}));
        assertEquals(expected, test);
    }


    @Test
    public void testVectorInit() {
        DataBuffer data = Nd4j.linspace(1, 4, 4).data();
        IComplexNDArray arr = Nd4j.createComplex(data, new int[]{4});
        assertEquals(true, arr.isRowVector());
        IComplexNDArray arr2 = Nd4j.createComplex(data, new int[]{1, 4});
        assertEquals(true, arr2.isRowVector());

        IComplexNDArray columnVector = Nd4j.createComplex(data, new int[]{4, 1});
        assertEquals(true, columnVector.isColumnVector());
    }


    @Test
    public void testIterateOverAllRows() {
        IComplexNDArray c = Nd4j.createComplex(Nd4j.create(Nd4j.linspace(0, 29, 30).data(), new int[]{3, 5, 2}));

        final AtomicInteger i = new AtomicInteger(0);
        final Set<IComplexNDArray> set = new HashSet<>();

        c.iterateOverAllRows(new SliceOp() {


            /**
             * Operates on an ndarray slice
             *
             * @param nd the result to operate on
             */
            @Override
            public void operate(INDArray nd) {
                IComplexNDArray result = (IComplexNDArray) nd;
                int curr = i.get();
                i.incrementAndGet();
                IComplexNDArray test = Nd4j.createComplex(new double[]{curr * 2, 0, curr * 2 + 1, 0}, new int[]{2});
                assertEquals(result, test);
                assertEquals(true, !set.contains(test));
                set.add(result);

                result.put(0, Nd4j.scalar((curr + 1) * 3));
                result.put(1, Nd4j.scalar((curr + 2) * 3));
                IComplexNumber n = (IComplexNumber) result.getScalar(0).element();
                IComplexNumber n2 = (IComplexNumber) result.getScalar(1).element();

                assertEquals((curr + 1) * 3, n.realComponent().doubleValue(), 1e-1);
                assertEquals((curr + 2) * 3, n2.realComponent().doubleValue(), 1e-1);

            }
        });

        IComplexNDArray permuted = c.permute(new int[]{2, 1, 0});
        set.clear();
        i.set(0);

        permuted.iterateOverAllRows(new SliceOp() {


            /**
             * Operates on an ndarray slice
             *
             * @param nd the result to operate on
             */
            @Override
            public void operate(INDArray nd) {
                IComplexNDArray result = (IComplexNDArray) nd;
                int curr = i.get();
                i.incrementAndGet();

                result.put(0, Nd4j.scalar((curr + 1) * 3));
                result.put(1, Nd4j.scalar((curr + 2) * 3));

                IComplexNumber n = (IComplexNumber) result.getScalar(0).element();
                IComplexNumber n2 = (IComplexNumber) result.getScalar(1).element();


                assertEquals((curr + 1) * 3, n.realComponent().doubleValue(), 1e-1);
                assertEquals((curr + 2) * 3, n2.realComponent().doubleValue(), 1e-1);
            }
        });

        IComplexNDArray swapped = c.swapAxes(2, 1);
        i.set(0);

        swapped.iterateOverAllRows(new SliceOp() {

            /**
             * Operates on an ndarray slice
             *
             * @param nd the result to operate on
             */
            @Override
            public void operate(INDArray nd) {
                IComplexNDArray result = (IComplexNDArray) nd;
                int curr = i.get();
                i.incrementAndGet();


                result.put(0, Nd4j.scalar((curr + 1) * 3));
                result.put(1, Nd4j.scalar((curr + 2) * 3));


                IComplexNumber n = (IComplexNumber) result.getScalar(0).element();
                IComplexNumber n2 = (IComplexNumber) result.getScalar(1).element();


                assertEquals((curr + 1) * 3, n.realComponent().doubleValue(), 1e-1);
                assertEquals((curr + 2) * 3, n2.realComponent().doubleValue(), 1e-1);

            }
        });


    }


    @Test
    public void testMmulOffset() {
        IComplexNDArray three = Nd4j.createComplex(Nd4j.create(new double[]{3, 4}, new int[]{2}));
        IComplexNDArray test = Nd4j.createComplex(Nd4j.create(Nd4j.linspace(1, 30, 30).data(), new int[]{3, 5, 2}));
        IComplexNDArray sliceRow = test.slice(0).getRow(1);
        assertEquals(three, sliceRow);

        IComplexNDArray twoSix = Nd4j.createComplex(Nd4j.create(new double[]{2, 6}, new int[]{2, 1}));
        IComplexNDArray threeTwoSix = three.mmul(twoSix);


        IComplexNDArray sliceRowTwoSix = sliceRow.mmul(twoSix);
        verifyElements(three, sliceRow);
        assertEquals(threeTwoSix, sliceRowTwoSix);

    }


    @Test
    public void testTwoByTwoMmul() {
        Nd4j.factory().setOrder('f');
        IComplexNDArray oneThroughFour = Nd4j.createComplex(Nd4j.linspace(1, 4, 4).reshape(2, 2));
        IComplexNDArray fiveThroughEight = Nd4j.createComplex(Nd4j.linspace(5, 8, 4).reshape(2, 2));

        IComplexNDArray solution = Nd4j.createComplex(Nd4j.create(new double[][]{{23, 31}, {34, 46}}));
        IComplexNDArray test = oneThroughFour.mmul(fiveThroughEight);
        assertEquals(solution, test);

    }


    @Test
    public void testMmul() {
        Nd4j.factory().setOrder('f');
        DataBuffer data = Nd4j.linspace(1, 10, 10).data();
        IComplexNDArray n = Nd4j.createComplex((Nd4j.create(data, new int[]{10})));
        IComplexNDArray transposed = n.transpose();
        assertEquals(true, n.isRowVector());
        assertEquals(true, transposed.isColumnVector());

        IComplexNDArray innerProduct = n.mmul(transposed);
        INDArray scalar = Nd4j.scalar(385);
        assertEquals(scalar, innerProduct.getReal());

        IComplexNDArray outerProduct = transposed.mmul(n);
        assertEquals(true, Shape.shapeEquals(new int[]{10, 10}, outerProduct.shape()));


        IComplexNDArray vectorVector = Nd4j.createComplex(Nd4j.create(new double[]{
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99, 108, 117, 126, 135, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 0, 11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 0, 13, 26, 39, 52, 65, 78, 91, 104, 117, 130, 143, 156, 169, 182, 195, 0, 14, 28, 42, 56, 70, 84, 98, 112, 126, 140, 154, 168, 182, 196, 210, 0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225
        }, new int[]{16, 16}));

        IComplexNDArray n1 = Nd4j.createComplex(Nd4j.create(Nd4j.linspace(0, 15, 16).data(), new int[]{16}));
        IComplexNDArray k1 = n1.transpose();

        IComplexNDArray testVectorVector = k1.mmul(n1);
        assertEquals(vectorVector, testVectorVector);


        IComplexNDArray M2 = Nd4j.createComplex(new double[]{1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.30901699437494745, -0.9510565162951535, -0.8090169943749473, -0.5877852522924732, -0.8090169943749478, 0.5877852522924727, 0.30901699437494723, 0.9510565162951536, 1.0, 0.0, -0.8090169943749473, -0.5877852522924732, 0.30901699437494723, 0.9510565162951536, 0.30901699437494856, -0.9510565162951532, -0.8090169943749477, 0.5877852522924728, 1.0, 0.0, -0.8090169943749478, 0.5877852522924727, 0.30901699437494856, -0.9510565162951532, 0.309016994374947, 0.9510565162951538, -0.809016994374946, -0.587785252292475, 1.0, 0.0, 0.30901699437494723, 0.9510565162951536, -0.8090169943749477, 0.5877852522924728, -0.809016994374946, -0.587785252292475, 0.3090169943749482, -0.9510565162951533}, new int[]{5, 5});
        INDArray n2 = Nd4j.create(Nd4j.linspace(1, 30, 30).data(), new int[]{3, 5, 2});
        INDArray swapped = n2.swapAxes(n2.shape().length - 1, 1);
        INDArray firstSlice = swapped.slice(0).slice(0);
        IComplexNDArray testSlice = Nd4j.createComplex(firstSlice);
        IComplexNDArray testNoOffset = Nd4j.createComplex(new double[]{1, 0, 4, 0, 7, 0, 10, 0, 13, 0}, new int[]{5});
        assertEquals(testSlice, testNoOffset);


        IComplexNDArray testSliceM2 = testSlice.mmul(M2);
        IComplexNDArray testNofOffsetM2 = testNoOffset.mmul(M2);
        assertEquals(testSliceM2, testNofOffsetM2);


    }

    @Test
    public void testTranspose() {
        IComplexNDArray ndArray = Nd4j.createComplex(new double[]{1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 6.999999999999999, 0.0, 8.0, 0.0, 9.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new int[]{16, 1});
        IComplexNDArray transposed2 = ndArray.transpose();
        assertEquals(16, transposed2.columns());

    }


    @Test
    public void testConjugate() {
        IComplexNDArray negative = Nd4j.createComplex(new double[]{1, -1, 2, -1}, new int[]{2});
        IComplexNDArray positive = Nd4j.createComplex(new double[]{1, 1, 2, 1}, new int[]{2});
        assertEquals(negative, positive.conj());

    }


    @Test
    public void testGetRow() {
        IComplexNDArray arr = Nd4j.createComplex(new int[]{3, 2});
        IComplexNDArray row = Nd4j.createComplex(new double[]{1, 0, 2, 0}, new int[]{2});
        arr.putRow(0, row);
        IComplexNDArray firstRow = arr.getRow(0);
        assertEquals(true, Shape.shapeEquals(new int[]{2}, firstRow.shape()));
        IComplexNDArray testRow = arr.getRow(0);
        assertEquals(row, testRow);


        IComplexNDArray row1 = Nd4j.createComplex(new double[]{3, 0, 4, 0}, new int[]{2});
        arr.putRow(1, row1);
        assertEquals(true, Shape.shapeEquals(new int[]{2}, arr.getRow(0).shape()));
        IComplexNDArray testRow1 = arr.getRow(1);
        assertEquals(row1, testRow1);


        INDArray fourTwoTwo = Nd4j.linspace(1, 16, 16).reshape(4, 2, 2);

        IComplexNDArray multiRow = Nd4j.createComplex(fourTwoTwo);
        IComplexNDArray test = Nd4j.createComplex(Nd4j.create(new double[]{7, 8}, new int[]{1, 2}));
        IComplexNDArray multiRowSlice1 = multiRow.slice(0);
        IComplexNDArray multiRowSlice = multiRow.slice(1);
        IComplexNDArray testMultiRow = multiRowSlice.getRow(1);

        assertEquals(test, testMultiRow);


    }

    @Test
    public void testMultiDimensionalCreation() {
        INDArray fourTwoTwo = Nd4j.linspace(1, 16, 16).reshape(4, 2, 2);

        IComplexNDArray multiRow = Nd4j.createComplex(fourTwoTwo);
        multiRow.toString();
        assertEquals(fourTwoTwo, multiRow.getReal());


    }


    @Test
    public void testLinearIndex() {
        IComplexNDArray n = Nd4j.createComplex(Nd4j.create(Nd4j.linspace(1, 8, 8).data(), new int[]{8}));
        for (int i = 0; i < n.length(); i++) {
            int linearIndex = n.linearIndex(i);
            assertEquals(i * 2, linearIndex);
            IComplexDouble d = (IComplexDouble) n.getScalar(i).element();
            double curr = d.realComponent();
            assertEquals(i + 1, curr, 1e-1);
        }
    }


    @Test
    public void testNdArrayConstructor() {
        IComplexNDArray result = Nd4j.createComplex(Nd4j.create(new double[]{2, 6}, new int[]{1, 2}));
        result.toString();
    }

    @Test
    public void testGetColumn() {
        IComplexNDArray arr = Nd4j.createComplex(Nd4j.create(Nd4j.linspace(1, 8, 8).data(), new int[]{2, 4}));
        IComplexNDArray column2 = arr.getColumn(1);
        IComplexNDArray result = Nd4j.createComplex(Nd4j.create(new double[]{2, 6}, new int[]{1, 2}));

        assertEquals(result, column2);
        assertEquals(true, Shape.shapeEquals(new int[]{2}, column2.shape()));
        IComplexNDArray column = Nd4j.createComplex(new double[]{11, 0, 12, 0}, new int[]{2});
        arr.putColumn(1, column);

        IComplexNDArray firstColumn = arr.getColumn(1);

        assertEquals(column, firstColumn);


        IComplexNDArray column1 = Nd4j.createComplex(new double[]{5, 0, 6, 0}, new int[]{2});
        arr.putColumn(1, column1);
        assertEquals(true, Shape.shapeEquals(new int[]{2}, arr.getColumn(1).shape()));
        IComplexNDArray testC = arr.getColumn(1);
        assertEquals(column1, testC);


        IComplexNDArray multiSlice = Nd4j.createComplex(Nd4j.create(Nd4j.linspace(1, 32, 32).data(), new int[]{4, 4, 2}));
        IComplexNDArray testColumn = Nd4j.createComplex(Nd4j.create(new double[]{10, 12, 14, 16}, new int[]{4}));
        IComplexNDArray sliceColumn = multiSlice.slice(1).getColumn(1);
        assertEquals(sliceColumn, testColumn);

        IComplexNDArray testColumn2 = Nd4j.createComplex(Nd4j.create(new double[]{17, 19, 21, 23}, new int[]{4}));
        IComplexNDArray testSlice2 = multiSlice.slice(2).getColumn(0);
        assertEquals(testColumn2, testSlice2);

        IComplexNDArray testColumn3 = Nd4j.createComplex(Nd4j.create(new double[]{18, 20, 22, 24}, new int[]{4}));
        IComplexNDArray testSlice3 = multiSlice.slice(2).getColumn(1);
        assertEquals(testColumn3, testSlice3);

    }


    @Test
    public void testPutAndGet() {
        IComplexNDArray arr = Nd4j.createComplex(Nd4j.create(new double[]{1, 2, 3, 4}, new int[]{2, 2}));
        assertEquals(4, arr.length());
        assertEquals(8, arr.data().length());
        arr.put(1, 1, Nd4j.scalar(5.0));

        IComplexNumber n1 = arr.getComplex(1, 1);
        IComplexNumber n2 = arr.getComplex(1, 1);

        assertEquals(5.0, n1.realComponent().doubleValue(), 1e-1);
        assertEquals(0.0, n2.imaginaryComponent().doubleValue(), 1e-1);

    }

    @Test
    public void testGetReal() {
        DataBuffer data = Nd4j.linspace(1, 8, 8).data();
        int[] shape = new int[]{8};
        IComplexNDArray arr = Nd4j.createComplex(shape);
        for (int i = 0; i < arr.length(); i++)
            arr.put(i, Nd4j.scalar(data.getFloat(i)));
        INDArray arr2 = Nd4j.create(data, shape);
        assertEquals(arr2, arr.getReal());

        INDArray ones = Nd4j.ones(10);
        IComplexNDArray n2 = Nd4j.complexOnes(10);
        assertEquals(ones, n2.getReal());

    }


    @Test
    public void testBasicOperations() {
        IComplexNDArray arr = Nd4j.createComplex(new double[]{0, 1, 2, 1, 1, 2, 3, 4}, new int[]{2, 2});
        IComplexDouble scalar = (IComplexDouble) arr.sum(Integer.MAX_VALUE).element();
        double sum = scalar.realComponent();
        assertEquals(6, sum, 1e-1);
        arr.addi(1);
        scalar = (IComplexDouble) arr.sum(Integer.MAX_VALUE).element();
        sum = scalar.realComponent();
        assertEquals(10, sum, 1e-1);
        arr.subi(Nd4j.createDouble(1, 0));
        scalar = (IComplexDouble) arr.sum(Integer.MAX_VALUE).element();

        sum = scalar.realComponent();
        assertEquals(6, sum, 1e-1);
    }


    @Test
    public void testElementWiseOps() {
        IComplexNDArray n1 = Nd4j.complexScalar(1);
        IComplexNDArray n2 = Nd4j.complexScalar(2);
        assertEquals(Nd4j.complexScalar(3), n1.add(n2));
        assertFalse(n1.add(n2).equals(n1));

        IComplexNDArray n3 = Nd4j.complexScalar(3);
        IComplexNDArray n4 = Nd4j.complexScalar(4);
        IComplexNDArray subbed = n4.sub(n3);
        IComplexNDArray mulled = n4.mul(n3);
        IComplexNDArray div = n4.div(n3);

        assertFalse(subbed.equals(n4));
        assertFalse(mulled.equals(n4));
        assertEquals(Nd4j.complexScalar(1), subbed);
        assertEquals(Nd4j.complexScalar(12), mulled);
        assertEquals(Nd4j.complexScalar(1.3333333333333333), div);


        IComplexNDArray multiDimensionElementWise = Nd4j.createComplex(Nd4j.create(Nd4j.linspace(1, 24, 24).data(), new int[]{4, 3, 2}));
        IComplexDouble sum2 = (IComplexDouble) multiDimensionElementWise.sum(Integer.MAX_VALUE).element();
        assertEquals(sum2, Nd4j.createDouble(300, 0));
        IComplexNDArray added = multiDimensionElementWise.add(Nd4j.complexScalar(1));
        IComplexDouble sum3 = (IComplexDouble) added.sum(Integer.MAX_VALUE).element();
        assertEquals(sum3, Nd4j.createDouble(324, 0));


    }


    @Test
    public void testVectorDimension() {
        IComplexNDArray test = Nd4j.createComplex(new double[]{1, 0, 2, 0, 3, 0, 4, 0}, new int[]{2, 2});
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
                log.info("Operator " + nd);
                IComplexNDArray test = (IComplexNDArray) nd;
                if (count.get() == 0) {
                    IComplexNDArray firstDimension = Nd4j.createComplex(new double[]{1, 0, 2, 0}, new int[]{2, 1});
                    assertEquals(firstDimension, test);
                } else {
                    IComplexNDArray firstDimension = Nd4j.createComplex(new double[]{3, 0, 4, 0}, new int[]{2});
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
                IComplexNDArray test = (IComplexNDArray) nd;
                if (count.get() == 0) {
                    IComplexNDArray firstDimension = Nd4j.createComplex(new double[]{1, 0, 3, 0}, new int[]{2});
                    assertEquals(firstDimension, test);
                } else {
                    IComplexNDArray firstDimension = Nd4j.createComplex(new double[]{2, 0, 4, 0}, new int[]{2});
                    assertEquals(firstDimension, test);

                }

                count.incrementAndGet();
            }

        }, false);


    }

    @Test
    public void testFlatten() {
        IComplexNDArray arr = Nd4j.createComplex(Nd4j.create(Nd4j.linspace(1, 4, 4).data(), new int[]{2, 2}));
        IComplexNDArray flattened = arr.ravel();
        assertEquals(arr.length(), flattened.length());
        assertTrue(Shape.shapeEquals(new int[]{1, 4}, flattened.shape()));
        for (int i = 0; i < arr.length(); i++) {
            IComplexNumber get = (IComplexNumber) flattened.getScalar(i).element();
            assertEquals(i + 1, get.realComponent().doubleValue(), 1e-1);
        }
    }


    @Test
    public void testMatrixGet() {

        IComplexNDArray arr = Nd4j.createComplex((Nd4j.linspace(1, 4, 4))).reshape(2, 2);
        IComplexNumber n1 = arr.getComplex(0, 0);
        IComplexNumber n2 = arr.getComplex(0, 1);
        IComplexNumber n3 = arr.getComplex(1, 0);
        IComplexNumber n4 = arr.getComplex(1, 1);

        assertEquals(1, n1.realComponent().doubleValue(), 1e-1);
        assertEquals(2, n2.realComponent().doubleValue(), 1e-1);
        assertEquals(3, n3.realComponent().doubleValue(), 1e-1);
        assertEquals(4, n4.realComponent().doubleValue(), 1e-1);
    }

    @Test
    public void testEndsForSlices() {
        IComplexNDArray arr = Nd4j.createComplex(Nd4j.create(Nd4j.linspace(1, 24, 24).data(), new int[]{4, 3, 2}));
        int[] endsForSlices = arr.endsForSlices();
        assertEquals(true, Arrays.equals(new int[]{0, 12, 24, 36}, endsForSlices));
    }


    @Test
    public void testWrap() {
        IComplexNDArray c = Nd4j.createComplex(Nd4j.linspace(1, 4, 4).reshape(2, 2));
        IComplexNDArray wrapped = c;
        assertEquals(true, Arrays.equals(new int[]{2, 2}, wrapped.shape()));

        IComplexNDArray vec = Nd4j.createComplex(Nd4j.linspace(1, 4, 4));
        IComplexNDArray wrappedVector = vec;
        assertEquals(true, wrappedVector.isVector());
        assertEquals(true, Shape.shapeEquals(new int[]{4}, wrappedVector.shape()));

    }


    @Test
    public void testVectorDimensionMulti() {
        IComplexNDArray arr = Nd4j.createComplex(Nd4j.create(Nd4j.linspace(1, 24, 24).data(), new int[]{4, 3, 2}));
        final AtomicInteger count = new AtomicInteger(0);

        arr.iterateOverDimension(0, new SliceOp() {


            /**
             * Operates on an ndarray slice
             *
             * @param nd the result to operate on
             */
            @Override
            public void operate(INDArray nd) {
                IComplexNDArray test = (IComplexNDArray) nd;
                if (count.get() == 0) {
                    IComplexNDArray answer = Nd4j.createComplex(new double[]{1, 0, 7, 0, 13, 0, 19, 0}, new int[]{4});
                    assertEquals(answer, test);
                } else if (count.get() == 1) {
                    IComplexNDArray answer = Nd4j.createComplex(new double[]{2, 0, 8, 0, 14, 0, 20, 0}, new int[]{4});
                    assertEquals(answer, test);
                } else if (count.get() == 2) {
                    IComplexNDArray answer = Nd4j.createComplex(new double[]{3, 0, 9, 0, 15, 0, 21, 0}, new int[]{4});
                    assertEquals(answer, test);
                } else if (count.get() == 3) {
                    IComplexNDArray answer = Nd4j.createComplex(new double[]{4, 0, 10, 0, 16, 0, 22, 0}, new int[]{4});
                    assertEquals(answer, test);
                } else if (count.get() == 4) {
                    IComplexNDArray answer = Nd4j.createComplex(new double[]{5, 0, 11, 0, 17, 0, 23, 0}, new int[]{4});
                    assertEquals(answer, test);
                } else if (count.get() == 5) {
                    IComplexNDArray answer = Nd4j.createComplex(new double[]{6, 0, 12, 0, 18, 0, 24, 0}, new int[]{4});
                    assertEquals(answer, test);
                }


                count.incrementAndGet();
            }
        }, false);


        IComplexNDArray ret = Nd4j.createComplex(new double[]{1, 0, 2, 0, 3, 0, 4, 0}, new int[]{2, 2});
        final IComplexNDArray firstRow = Nd4j.createComplex(new double[]{1, 0, 2, 0}, new int[]{2});
        final IComplexNDArray secondRow = Nd4j.createComplex(new double[]{3, 0, 4, 0}, new int[]{2});
        count.set(0);
        ret.iterateOverDimension(1, new SliceOp() {

            /**
             * Operates on an ndarray slice
             *
             * @param nd the result to operate on
             */
            @Override
            public void operate(INDArray nd) {
                IComplexNDArray c = (IComplexNDArray) nd;
                if (count.get() == 0) {
                    assertEquals(firstRow, c);
                } else if (count.get() == 1)
                    assertEquals(secondRow, c);
                count.incrementAndGet();
            }
        }, false);
    }


    protected void verifyElements(IComplexNDArray d, IComplexNDArray d2) {
        for (int i = 0; i < d.rows(); i++) {
            for (int j = 0; j < d.columns(); j++) {
                IComplexNumber test1 = d.getComplex(i, j);
                IComplexNumber test2 = d2.getComplex(i, j);
                assertEquals(test1.realComponent().doubleValue(), test2.realComponent().doubleValue(), 1e-6);
                assertEquals(test1.imaginaryComponent().doubleValue(), test2.imaginaryComponent().doubleValue(), 1e-6);

            }
        }
    }

}
