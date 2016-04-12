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

package org.nd4j.linalg;

import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import static org.junit.Assert.*;


import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.VectorFFT;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.ComplexUtil;
import org.nd4j.linalg.api.shape.Shape;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;



/**
 * Tests for a complex ndarray
 *
 * @author Adam Gibson
 */
@Ignore
@RunWith(Parameterized.class)
public  class ComplexNDArrayTestsC extends BaseComplexNDArrayTests  {

    private static Logger log = LoggerFactory.getLogger(ComplexNDArrayTestsC.class);

    public ComplexNDArrayTestsC() {
    }


    public ComplexNDArrayTestsC(Nd4jBackend backend) {
        super(backend);
    }

    @Before
    public void before() {
        super.before();
    }

    @After
    public void after() {
        super.after();
    }

    @Test
    public void testConstruction() {

        IComplexNDArray arr2 = Nd4j.createComplex(new int[]{3, 2});
        assertEquals(3, arr2.rows());
        assertEquals(arr2.rows(), arr2.rows());
        assertEquals(2, arr2.columns());
        assertEquals(arr2.columns(), arr2.columns());
        assertTrue(arr2.isMatrix());


        IComplexNDArray arr = Nd4j.createComplex(new double[]{0, 1}, new int[]{1,1});
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
        IComplexNDArray assertComplexDouble = Nd4j.createComplex(testArr, new int[]{1,2});
        IComplexNDArray testComplexDouble = Nd4j.createComplex(two, new int[]{1,2});
        assertEquals(assertComplexDouble, testComplexDouble);

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
    public void testAssignOffset() {
        IComplexNDArray arr = Nd4j.complexOnes(5, 5);
        IComplexNDArray row = arr.slice(1);
        row.assign(1);
        assertEquals(Nd4j.complexOnes(5),row);
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
    public void testPutSlice() {

    }



    @Test
    public void testSum() {
        IComplexNDArray n = Nd4j.createComplex(Nd4j.create(Nd4j.linspace(1, 8, 8).data(), new int[]{2, 2, 2}));
        assertEquals(Nd4j.createDouble(36, 0), n.sumComplex());
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
    public void testVectorOffsetRavel() {
        IComplexNDArray arr = Nd4j.complexLinSpace(1,20,20).reshape(4, 5);
        for(int i = 0; i < arr.slices(); i++) {
            assertEquals(arr.slice(i),arr.slice(i).ravel());
        }
    }


    @Test
    public void testSliceVsVectorAlongDimension() {
        IComplexNDArray arr = Nd4j.complexLinSpace(1,20,20).reshape(4,5);
        assertEquals(arr.slices(),arr.vectorsAlongDimension(1));
        for(int i = 0; i < arr.slices(); i++) {
            assertEquals(arr.vectorAlongDimension(i,1),arr.slice(i));
            assertEquals(arr.vectorAlongDimension(i,1).ravel(),arr.slice(i).ravel());
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
        IComplexNDArray arr = Nd4j.createComplex(Nd4j.create(Nd4j.linspace(1, 8, 8).data(), new int[]{1, 8}));
        for (int i = 0; i < arr.length(); i++) {
            IComplexNumber curr = arr.getComplex(i);
            assertEquals(Nd4j.createDouble(i + 1, 0), curr);
        }

        IComplexNDArray matrix = Nd4j.createComplex(Nd4j.create(Nd4j.linspace(1, 8, 8).data(), new int[]{2, 4}));
        IComplexNDArray row = matrix.getRow(1);
        IComplexNDArray column = matrix.getColumn(1);

        IComplexNDArray validate = Nd4j.createComplex(Nd4j.create(new double[]{5, 6, 7, 8}, new int[]{1,4}));
        IComplexNumber d = row.getComplex(3);
        assertEquals(Nd4j.createDouble(8, 0), d);
        assertEquals(row, validate);

        IComplexNumber d2 = column.getComplex(1);

        assertEquals(Nd4j.createDouble(6, 0), d2);


    }


    @Test
    public void testTensorStrides() {
        INDArray arr = Nd4j.createComplex(106, 1, 3, 3);
        //(144, 144, 48, 16)
        int[] assertion = ArrayUtil.of(18, 18, 6, 2);
        int[] arrShape = arr.stride();
        assertTrue(Arrays.equals(assertion, arrShape));
        Nd4j.factory().setOrder('f');
        arr = Nd4j.createComplex(106,1,3,3);
        //(16, 1696, 1696, 5088)
        assertion = ArrayUtil.of(2, 212, 212, 636);
        arrShape = arr.stride();
        assertTrue(Arrays.equals(assertion, arrShape));

    }



    @Test
    public void testLinearView() {
        IComplexNDArray n = Nd4j.complexLinSpace(1, 4, 4).reshape(2, 2);
        IComplexNDArray row = n.getRow(1);
        IComplexNDArray linear = row.linearView();
        assertEquals(row, linear);

        IComplexNDArray large = Nd4j.complexLinSpace(1, 1000, 1000).reshape(2, 500);
        IComplexNDArray largeLinear = large.linearView();
        for(int i = 0; i < largeLinear.length(); i++)
            assertEquals(i + 1,largeLinear.getReal(i),1e-1);

        IComplexNDArray largeTensor = large.reshape(1000,1,1,1);
        for(int i = 0; i < largeLinear.length(); i++)
            assertEquals(i + 1,largeTensor.getReal(i),1e-1);


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
    public void testSliceOffset() {
        IComplexNDArray test = Nd4j.complexLinSpace(1, 10, 10).reshape(2,5);
        IComplexNDArray testSlice0 = Nd4j.complexLinSpace(1, 5, 5);
        IComplexNDArray testSlice1 = Nd4j.complexLinSpace(6, 10, 5);
        assertEquals(testSlice0,test.slice(0));
        assertEquals(testSlice1,test.slice(1));

        IComplexNDArray sliceOfSlice0 = test.slice(0).slice(0);
        assertEquals(sliceOfSlice0.getComplex(0),Nd4j.createComplexNumber(1,0));
        assertEquals( test.slice(1).slice(0).getComplex(0),Nd4j.createComplexNumber(6,0));
        assertEquals(test.slice(1).getComplex(1),Nd4j.createComplexNumber(7,0));


    }


    @Test
    public void testSlice() {
        IComplexNDArray slices = Nd4j.createComplex(2,3);
        slices.put(0,0,1);
        slices.put(0,1,2);
        slices.put(0,2,3);
        slices.put(1,1,4);
        IComplexNDArray assertion = Nd4j.createComplex(new IComplexNumber[]{
                Nd4j.createComplexNumber(1,0),
                Nd4j.createComplexNumber(2,0),
                Nd4j.createComplexNumber(3,0),

        });
        assertEquals(assertion,slices.slice(0));



        INDArray arr = Nd4j.create(Nd4j.linspace(1, 24, 24).data(), new int[]{4, 3, 2});
        IComplexNDArray arr2 = Nd4j.createComplex(arr);
        assertEquals(arr, arr2.getReal());

        INDArray firstSlice = arr.slice(0);
        INDArray firstSliceTest = arr2.slice(0).getReal();
        assertEquals(firstSlice, firstSliceTest);


        INDArray secondSlice = arr.slice(1);
        INDArray secondSliceTest = arr2.slice(1).getReal();
        assertEquals(secondSlice, secondSliceTest);


        INDArray slice0 = Nd4j.create(new double[]{1,2,3,4,5,6}, new int[]{3, 2});
        INDArray slice2 = Nd4j.create(new double[]{7,8,9,10,11,12}, new int[]{3, 2});


        IComplexNDArray testSliceComplex = arr2.slice(0);
        IComplexNDArray testSliceComplex2 = arr2.slice(1);

        INDArray testSlice0 = testSliceComplex.getReal();
        INDArray testSlice1 = testSliceComplex2.getReal();

        assertEquals(slice0, testSlice0);
        assertEquals(slice2, testSlice1);

        //weird slice striding issues here. try to avoid hacks related to if() the problem is not complex related
        INDArray n2 = Nd4j.create(Nd4j.linspace(1, 30, 30).data(), new int[]{3, 5, 2});
        INDArray swapped = n2.swapAxes(n2.shape().length - 1, 1);
        INDArray firstSlice2 = swapped.slice(0).slice(0);
        //problem ends  here. Something with slicing?
        IComplexNDArray testSlice = Nd4j.createComplex(firstSlice2);
        IComplexNDArray testNoOffset = Nd4j.createComplex(new double[]{1, 0, 3, 0, 5, 0, 7, 0, 9, 0}, new int[]{1,5});
        assertEquals(testSlice, testNoOffset);


    }

    @Test
    public void testSliceConstructor() {
        List<IComplexNDArray> testList = new ArrayList<>();
        for (int i = 0; i < 5; i++)
            testList.add(Nd4j.complexScalar(i + 1));

        IComplexNDArray test = Nd4j.createComplex(testList, new int[]{1,testList.size()});
        IComplexNDArray expected = Nd4j.createComplex(Nd4j.create(new double[]{1, 2, 3, 4, 5}, new int[]{1,5}));
        assertEquals(expected, test);
    }


    @Test
    public void testVectorInit() {
        DataBuffer data = Nd4j.linspace(1, 4, 4).data();
        IComplexNDArray arr = Nd4j.createComplex(data, new int[]{1,4});
        assertEquals(true, arr.isRowVector());
        IComplexNDArray arr2 = Nd4j.createComplex(data, new int[]{1, 4});
        assertEquals(true, arr2.isRowVector());

        IComplexNDArray columnVector = Nd4j.createComplex(data, new int[]{4, 1});
        assertEquals(true, columnVector.isColumnVector());
    }




    @Test
    public void testMmulOffset() {
        IComplexNDArray three = Nd4j.createComplex(Nd4j.create(new double[]{3, 4}, new int[]{1,2}));
        IComplexNDArray test = Nd4j.createComplex(Nd4j.create(Nd4j.linspace(1, 30, 30).data(), new int[]{3, 5, 2}));
        IComplexNDArray sliceRow = test.slice(0).getRow(1);
        assertEquals(getFailureMessage(),three, sliceRow);

        IComplexNDArray twoSix = Nd4j.createComplex(Nd4j.create(new double[]{2, 6}, new int[]{2, 1}));
        IComplexNDArray threeTwoSix = three.mmul(twoSix);


        IComplexNDArray sliceRowTwoSix = sliceRow.mmul(twoSix);
        verifyElements(three, sliceRow);
        assertEquals(getFailureMessage(),threeTwoSix, sliceRowTwoSix);

    }


    @Test
    public void testIterateOverAllRows() {
        Nd4j.EPS_THRESHOLD = 1e-1;
        IComplexNDArray ones = Nd4j.complexOnes(5, 5);
        VectorFFT fft = new VectorFFT(ones);
        IComplexNDArray assertion = Nd4j.createComplex(5, 5);
        for(int i = 0; i < assertion.rows(); i++)
            assertion.getRow(i).putScalar(0,Nd4j.createComplexNumber(5,0));
        Nd4j.getExecutioner().iterateOverAllRows(fft);
        assertEquals(getFailureMessage(),assertion,ones);
    }



    @Test
    public void testRowVectorGemm() {
        IComplexNDArray linspace = Nd4j.complexLinSpace(1,4,4);
        IComplexNDArray other = Nd4j.complexLinSpace(1,16,16).reshape(4, 4);
        IComplexNDArray result = linspace.mmul(other);
        IComplexNDArray assertion = Nd4j.createComplex(ComplexUtil.complexNumbersFor(new double[]{90,100,110,120}));
        assertEquals(assertion,result);
    }



    @Test
    public void testRealConversion() {
        IComplexNDArray arr = Nd4j.createComplex(1,5);
        INDArray arr1 = Nd4j.create(1,5);
        assertEquals(arr,Nd4j.createComplex(arr1));
        IComplexNDArray arr3 = Nd4j.complexLinSpace(1,6,6).reshape(2,3);
        INDArray linspace = Nd4j.linspace(1,6,6).reshape(2,3);
        assertEquals(arr3,Nd4j.createComplex(linspace));
    }



    @Test
    public void testTranspose() {
        IComplexNDArray ndArray = Nd4j.createComplex(new double[]{1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 6.999999999999999, 0.0, 8.0, 0.0, 9.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new int[]{16, 1});
        IComplexNDArray transposed2 = ndArray.transpose();
        assertEquals(16, transposed2.columns());

    }


    @Test
    public void testConjugate() {
        IComplexNDArray negative = Nd4j.createComplex(new double[]{1, -1, 2, -1}, new int[]{1,2});
        IComplexNDArray positive = Nd4j.createComplex(new double[]{1, 1, 2, 1}, new int[]{1,2});
        assertEquals(negative, positive.conj());

    }


    @Test
    public void testGetRow() {
        IComplexNDArray arr = Nd4j.createComplex(new int[]{3, 2});
        IComplexNDArray row = Nd4j.createComplex(new double[]{1, 0, 2, 0}, new int[]{1,2});
        arr.putRow(0, row);
        IComplexNDArray firstRow = arr.getRow(0);
        assertEquals(true, Shape.shapeEquals(new int[]{1,2}, firstRow.shape()));
        IComplexNDArray testRow = arr.getRow(0);
        assertEquals(row, testRow);


        IComplexNDArray row1 = Nd4j.createComplex(new double[]{3, 0, 4, 0}, new int[]{1,2});
        arr.putRow(1, row1);
        assertEquals(true, Shape.shapeEquals(new int[]{2}, arr.getRow(0).shape()));
        IComplexNDArray testRow1 = arr.getRow(1);
        assertEquals(row1, testRow1);


        INDArray fourTwoTwo = Nd4j.linspace(1, 16, 16).reshape(4, 2, 2);

        IComplexNDArray multiRow = Nd4j.createComplex(fourTwoTwo);
        IComplexNDArray test = Nd4j.createComplex(Nd4j.create(new double[]{7, 8}, new int[]{1, 2}));
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
    public void testGetComplex() {
        IComplexNDArray arr = Nd4j.createComplex(Nd4j.create(Nd4j.createBuffer(new double[]{
                1,2,3,4,5
        })));

        IComplexNumber num = arr.getComplex(4);
        assertEquals(Nd4j.createDouble(5, 0),num);

        IComplexNDArray matrix = Nd4j.complexLinSpace(1,10,10).reshape(2,5);
        IComplexNDArray slice = matrix.slice(0);
        IComplexNDArray assertion = Nd4j.complexLinSpace(1,5,5);
        assertEquals(assertion,slice);
        IComplexNDArray assert2 = Nd4j.complexLinSpace(6,10,5);
        assertEquals(assert2,matrix.slice(1));

    }



    @Test
    public void testGetColumn() {
        IComplexNDArray arr = Nd4j.createComplex(Nd4j.create(Nd4j.linspace(1, 8, 8).data(), new int[]{2, 4}));
        IComplexNDArray column2 = arr.getColumn(1);
        IComplexNDArray result = Nd4j.createComplex(Nd4j.create(new double[]{2, 6}, new int[]{1, 2}));

        assertEquals(result, column2);
        assertEquals(true, Shape.shapeEquals(new int[]{2,1}, column2.shape()));
        IComplexNDArray column = Nd4j.createComplex(new double[]{11, 0, 12, 0}, new int[]{1,2});
        arr.putColumn(1, column);

        IComplexNDArray firstColumn = arr.getColumn(1);

        assertEquals(column, firstColumn);


        IComplexNDArray column1 = Nd4j.createComplex(new double[]{5, 0, 6, 0}, new int[]{1,2});
        arr.putColumn(1, column1);
        assertEquals(true, Shape.shapeEquals(new int[]{2,1}, arr.getColumn(1).shape()));
        IComplexNDArray testC = arr.getColumn(1);
        assertEquals(column1, testC);


        IComplexNDArray multiSlice = Nd4j.createComplex(Nd4j.create(Nd4j.linspace(1, 32, 32).data(), new int[]{4, 4, 2}));
        IComplexNDArray testColumn = Nd4j.createComplex(Nd4j.create(new double[]{10, 12, 14, 16}, new int[]{1,4}));
        IComplexNDArray sliceColumn = multiSlice.slice(1).getColumn(1);
        assertEquals(sliceColumn, testColumn);

        IComplexNDArray testColumn2 = Nd4j.createComplex(Nd4j.create(new double[]{17, 19, 21, 23}, new int[]{1,4}));
        IComplexNDArray testSlice2 = multiSlice.slice(2);
        IComplexNDArray testSlice2ColumnZero = testSlice2.getColumn(0);
        assertEquals(testColumn2, testSlice2ColumnZero);

        IComplexNDArray testColumn3 = Nd4j.createComplex(Nd4j.create(new double[]{18, 20, 22, 24}, new int[]{1,4}));
        IComplexNDArray testSlice3 = multiSlice.slice(2).getColumn(1);
        assertEquals(testColumn3, testSlice3);

    }




    @Test
    public void testGetIndexing() {
        Nd4j.MAX_SLICES_TO_PRINT = Integer.MAX_VALUE;
        Nd4j.MAX_ELEMENTS_PER_SLICE = Integer.MAX_VALUE;
        IComplexNDArray tenByTen = Nd4j.complexLinSpace(1,100,100).reshape(10,10);
        IComplexNDArray thirtyToSixty = (IComplexNDArray) Transforms.round(Nd4j.complexLinSpace(31,60,30)).reshape(3,10);
        IComplexNDArray test = tenByTen.get(NDArrayIndex.interval(3, 6), NDArrayIndex.interval(0, tenByTen.columns()));
        assertEquals(thirtyToSixty,test);
    }

    @Test
    public void testPutAndGet() {
        IComplexNDArray multiRow = Nd4j.createComplex(2,2);
        multiRow.putScalar(0,0,Nd4j.createComplexNumber(1,0));
        multiRow.putScalar(0,1,Nd4j.createComplexNumber(2,0));
        multiRow.putScalar(1,0,Nd4j.createComplexNumber(3,0));
        multiRow.putScalar(1, 1, Nd4j.createComplexNumber(4, 0));
        assertEquals(Nd4j.createComplexNumber(1, 0), multiRow.getComplex(0, 0));
        assertEquals(Nd4j.createComplexNumber(2, 0), multiRow.getComplex(0, 1));
        assertEquals(Nd4j.createComplexNumber(3, 0), multiRow.getComplex(1,0));
        assertEquals(Nd4j.createComplexNumber(4, 0), multiRow.getComplex(1, 1));

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
        int[] shape = new int[]{1,8};
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
    public void testBroadcast() {
        IComplexNDArray arr = Nd4j.complexLinSpace(1,5,5);
        IComplexNDArray arrs = arr.broadcast(new int[]{5,5});
        IComplexNDArray arrs3 = Nd4j.createComplex(5,5);
        assertTrue(Arrays.equals(arrs.shape(),arrs3.shape()));
        for(int i = 0; i < arrs.slices(); i++)
            arrs3.putSlice(i,arr);
        assertEquals(arrs3,arrs);
    }

    @Test
    public void testBasicOperations() {
        IComplexNDArray arr = Nd4j.createComplex(new double[]{0, 1, 2, 1, 1, 2, 3, 4}, new int[]{2, 2});
        IComplexNumber scalar =  arr.sumComplex();
        double sum = scalar.realComponent().doubleValue();
        assertEquals(6, sum, 1e-1);
        arr.addi(1);
        scalar = arr.sumComplex();
        sum = scalar.realComponent().doubleValue();
        assertEquals(10, sum, 1e-1);
        arr.subi(Nd4j.createDouble(1, 0));
        scalar = arr.sumComplex().asDouble();

        sum = scalar.realComponent().doubleValue();
        assertEquals(6, sum, 1e-1);
    }

    @Test
    public void testComplexCalculation() {
        IComplexNDArray arr = Nd4j.createComplex(
                new IComplexNumber[][]{{Nd4j.createComplexNumber(1, 1), Nd4j.createComplexNumber(2, 1)},
                {Nd4j.createComplexNumber(3, 2), Nd4j.createComplexNumber(4, 2)}});

        IComplexNumber scalar =  arr.sumComplex();
        double sum = scalar.realComponent().doubleValue();
        assertEquals(10, sum, 1e-1);

        double sumImag = scalar.imaginaryComponent().doubleValue();
        assertEquals(6, sumImag, 1e-1);

        IComplexNDArray res = arr.add(Nd4j.createComplexNumber(1, 1));
        scalar = res.sumComplex();
        sum = scalar.realComponent().doubleValue();
        assertEquals(14, sum, 1e-1);
        sumImag = scalar.imaginaryComponent().doubleValue();
        assertEquals(10, sumImag, 1e-1);

        //original array should keep as it is
        sum = arr.sumComplex().realComponent().doubleValue();
        assertEquals(10, sum, 1e-1);
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
        IComplexNumber sum2 = multiDimensionElementWise.sumComplex();
        assertEquals(sum2, Nd4j.createDouble(300, 0));
        IComplexNDArray added = multiDimensionElementWise.add(Nd4j.complexScalar(1));
        IComplexNumber sum3 =  added.sumComplex();
        assertEquals(sum3, Nd4j.createDouble(324, 0));


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




    @Override
    public char ordering() {
        return 'c';
    }
}
