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
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataBuffer.Type;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.util.ComplexUtil;
import org.nd4j.linalg.api.shape.Shape;


import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Tests for a complex ndarray
 *
 * @author Adam Gibson
 */
@RunWith(Parameterized.class)
@Ignore
public  class ComplexNDArrayTestsFortran extends BaseComplexNDArrayTests  {


    public ComplexNDArrayTestsFortran(Nd4jBackend backend) {
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
    public void testLeadingOnes() {
        IComplexNDArray complexRand = Nd4j.complexRand(100,1,28,28);
        assertArrayEquals(new int[]{100,1,28,28},complexRand.shape());
        IComplexNDArray arr = complexRand.linearView();
        for(int i = 0; i < arr.length(); i++) {
            arr.putScalar(i,arr.getComplex(i));
        }

        IComplexNDArray complexRand2 = Nd4j.complexRand(28,28,1);
        assertArrayEquals(new int[]{28,28,1},complexRand2.shape());
        IComplexNDArray arr2 = complexRand.linearView();
        for(int i = 0; i < arr2.length(); i++) {
            arr2.putScalar(i,arr2.getComplex(i));
        }

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
    public void testAssignOffset() {
        IComplexNDArray arr = Nd4j.complexOnes(5, 5);
        IComplexNDArray row = arr.slice(1);
        row.assign(1);
        assertEquals(Nd4j.complexOnes(5), row);

        IComplexNDArray row2 = arr.slice(2);
        row2.assign(Nd4j.complexValueOf(5, 2));
        assertEquals(Nd4j.complexValueOf(5, 2),row2);
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
    public void testVectorGet() {
        IComplexNDArray arr = Nd4j.createComplex(Nd4j.create(Nd4j.linspace(1, 8, 8).data(), new int[]{1,8}));
        for (int i = 0; i < arr.length(); i++) {
            IComplexNumber curr = arr.getComplex(i);
            assertEquals(Nd4j.createDouble(i + 1, 0), curr);
        }

        IComplexNDArray matrix = Nd4j.createComplex(Nd4j.create(Nd4j.linspace(1, 8, 8).data(), new int[]{2, 4}));
        IComplexNDArray row = matrix.getRow(1);
        IComplexNDArray column = matrix.getColumn(1);

        IComplexNDArray validate = Nd4j.createComplex(Nd4j.create(new double[]{2,4,6,8}, new int[]{1,4}));
        IComplexNumber d = row.getComplex(3);
        assertEquals(Nd4j.createDouble(8, 0), d);
        assertEquals(row, validate);

        IComplexNumber d2 = column.getComplex(1);

        assertEquals(Nd4j.createDouble(4, 0), d2);


    }





    @Test
    public void testCreateFromNDArray() {

        Nd4j.dtype = Type.DOUBLE;
        INDArray fortran = Nd4j.create(new double[][]{{1, 2}, {3, 4}});

        IComplexNDArray fortranComplex = Nd4j.createComplex(fortran);
        for (int i = 0; i < fortran.rows(); i++) {
            for (int j = 0; j < fortran.columns(); j++) {
                double d = fortran.getFloat(i, j);
                IComplexNumber complexD = fortranComplex.getComplex(i, j);
                assertEquals(Nd4j.createDouble(d, 0), complexD);
            }
        }

    }



    @Test
    public void testSwapAxesFortranOrder() {

        IComplexNDArray n = Nd4j.createComplex(Nd4j.linspace(1, 30, 30)).reshape(3, 5, 2);
        IComplexNDArray slice = n.swapAxes(2, 1);
        IComplexNDArray assertion = Nd4j.createComplex(new double[]{1, 0, 4, 0, 7, 0, 10, 0, 13, 0});
        IComplexNDArray test = slice.slice(0).slice(0);
        assertEquals(assertion, test);
    }




    @Test
    public void testSliceOffset() {
        Nd4j.EPS_THRESHOLD = 1e-1;
        IComplexNDArray test = Nd4j.complexLinSpace(1, 10, 10).reshape(2,5);
        IComplexNDArray testSlice0 = Nd4j.createComplex(new IComplexNumber[]{
                Nd4j.createComplexNumber(1, 0),
                Nd4j.createComplexNumber(3, 0),
                Nd4j.createComplexNumber(5, 0),
                Nd4j.createComplexNumber(7, 0),
                Nd4j.createComplexNumber(9, 0),

        });

        IComplexNDArray testSlice1 = Nd4j.createComplex(new IComplexNumber[]{
                Nd4j.createComplexNumber(2, 0),
                Nd4j.createComplexNumber(4, 0),
                Nd4j.createComplexNumber(6, 0),
                Nd4j.createComplexNumber(8, 0),
                Nd4j.createComplexNumber(10, 0),

        });

        assertEquals(getFailureMessage(),testSlice0,test.slice(0));
        assertEquals(getFailureMessage(),testSlice1,test.slice(1));

        IComplexNDArray sliceOfSlice0 = test.slice(0).slice(0);
        assertEquals(sliceOfSlice0.getComplex(0),Nd4j.createComplexNumber(1,0));
        assertEquals(getFailureMessage(),test.slice(1).slice(0).getComplex(0),Nd4j.createComplexNumber(2, 0));
        assertEquals(getFailureMessage(),test.slice(1).getComplex(1),Nd4j.createComplexNumber(4, 0));


    }




    @Test
    public void testSliceMatrix() {
        IComplexNDArray arr = Nd4j.complexLinSpace(1, 8, 8).reshape(2, 4);
        assertEquals(Nd4j.createComplex(new IComplexNumber[]{
                Nd4j.createComplexNumber(1, 0),
                Nd4j.createComplexNumber(3, 0),
                Nd4j.createComplexNumber(5, 0),
                Nd4j.createComplexNumber(7, 0)
        }),arr.slice(0));

        assertEquals(Nd4j.createComplex(new IComplexNumber[]{
                Nd4j.createComplexNumber(2, 0),
                Nd4j.createComplexNumber(4, 0),
                Nd4j.createComplexNumber(6, 0),
                Nd4j.createComplexNumber(8, 0)
        }),arr.slice(1));
    }


    @Test
    public void testSliceConstructor() {
        List<IComplexNDArray> testList = new ArrayList<>();
        for (int i = 0; i < 5; i++)
            testList.add(Nd4j.complexScalar(i + 1));

        IComplexNDArray test = Nd4j.createComplex(testList, new int[]{testList.size()});
        IComplexNDArray expected = Nd4j.createComplex(Nd4j.create(new double[]{1, 2, 3, 4, 5}, new int[]{1,5}));
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
    public void testRealConversion() {
        IComplexNDArray arr = Nd4j.createComplex(1,5);
        INDArray arr1 = Nd4j.create(1, 5);
        assertEquals(arr,Nd4j.createComplex(arr1));
        IComplexNDArray arr3 = Nd4j.complexLinSpace(1,6,6).reshape(2, 3);
        INDArray linspace = Nd4j.linspace(1,6,6).reshape(2, 3);
        assertEquals(getFailureMessage(),arr3,Nd4j.createComplex(linspace));
    }


    @Test
    public void testTranspose() {
        IComplexNDArray ndArray = Nd4j.createComplex(new double[]{1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 6.999999999999999, 0.0, 8.0, 0.0, 9.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, new int[]{16, 1});
        IComplexNDArray transposed2 = ndArray.transpose();
        assertEquals(getFailureMessage(),16, transposed2.columns());


    }


    @Test
    public void testConjugate() {
        IComplexNDArray negative = Nd4j.createComplex(new double[]{1, -1, 2, -1}, new int[]{1,2});
        IComplexNDArray positive = Nd4j.createComplex(new double[]{1, 1, 2, 1}, new int[]{1,2});
        assertEquals(getFailureMessage(),negative, positive.conj());

    }



    @Test
    public void testMultiDimensionalCreation() {
        INDArray fourTwoTwo = Nd4j.linspace(1, 16, 16).reshape(4, 2, 2);
        IComplexNDArray multiRow = Nd4j.createComplex(fourTwoTwo);
        assertEquals(fourTwoTwo, multiRow.getReal());
    }

    @Test
    public void testMmul() {
        IComplexNDArray n = Nd4j.createComplex(Nd4j.linspace(1, 10, 10));
        IComplexNDArray transposed = n.transpose();
        assertEquals(true, n.isRowVector());
        assertEquals(true, transposed.isColumnVector());



        INDArray innerProduct = n.mmul(transposed);

        INDArray scalar = Nd4j.scalar(Nd4j.createComplexNumber(385, 0));
        assertEquals(getFailureMessage(),scalar, innerProduct);

        INDArray outerProduct = transposed.mmul(n);
        assertEquals(true, Shape.shapeEquals(new int[]{10, 10}, outerProduct.shape()));

        IComplexNDArray d3 = Nd4j.createComplex(ComplexUtil.complexNumbersFor(new double[]{1, 2})).reshape(2, 1);
        IComplexNDArray d4 = Nd4j.createComplex(ComplexUtil.complexNumbersFor(new double[]{3, 4}));
        INDArray resultNDArray = d3.mmul(d4);
        INDArray result = Nd4j.createComplex(new IComplexNumber[][]{
                {Nd4j.createComplexNumber(3, 0), Nd4j.createComplexNumber(4, 0)}
                , {Nd4j.createComplexNumber(6, 0), Nd4j.createComplexNumber(8, 0)}});

        assertEquals(getFailureMessage(),result, resultNDArray);




    }

    @Test
    public void testMmulOffset() {
        IComplexNDArray arr = Nd4j.complexLinSpace(1, 4, 4).reshape(2, 2);
        IComplexNDArray row1 = arr.getRow(1);
        IComplexNDArray arrT = row1.transpose();
        assertEquals(row1,arrT);
    }

    @Test
    public void testMmulColumnVector() {


        IComplexNDArray three = Nd4j.createComplex(ComplexUtil.complexNumbersFor(new double[]{4, 19}), new int[]{1,2});
        IComplexNDArray test = Nd4j.complexLinSpace(1, 30, 30).reshape(3, 5, 2);
        IComplexNDArray sliceRow = test.slice(0).getRow(1);
        assertEquals(three, sliceRow);


        IComplexNDArray twoSix = Nd4j.createComplex(ComplexUtil.complexNumbersFor(new double[]{2, 6}), new int[]{2, 1});
        IComplexNDArray threeTwoSix = three.mmul(twoSix);

        IComplexNDArray sliceRowTwoSix = sliceRow.mmul(twoSix);
        assertEquals(getFailureMessage(),threeTwoSix, sliceRowTwoSix);

    }




    @Test
    public void testPermute() {
        IComplexNDArray arr = Nd4j.complexLinSpace(1,8,8).reshape(2,2,2);
        IComplexNDArray permute = arr.permute(2, 1, 0);
        IComplexNDArray assertion = Nd4j.createComplex(2, 2, 2);
        IComplexNDArray assertionSlice1 = Nd4j.createComplex(Nd4j.create(new double[][]{{1,2},{3,4}}));
        IComplexNDArray assertionSlice2 = Nd4j.createComplex(Nd4j.create(new double[][]{{5,6},{7,8}}));
        assertion.putSlice(0,assertionSlice1);
        assertion.putSlice(1,assertionSlice2);
        assertEquals(assertion,permute);

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

        assertEquals(getFailureMessage(),5.0, n1.realComponent().doubleValue(), 1e-1);
        assertEquals(getFailureMessage(),0.0, n2.imaginaryComponent().doubleValue(), 1e-1);




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
        assertEquals(getFailureMessage(),ones, n2.getReal());

    }





    @Test
    public void testBroadcast() {
        IComplexNDArray arr = Nd4j.complexLinSpace(1, 5, 5);
        IComplexNDArray arrs = arr.broadcast(new int[]{5,5});
        IComplexNDArray arrs3 = Nd4j.createComplex(5,5);
        assertTrue(Arrays.equals(arrs.shape(), arrs3.shape()));
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
        scalar =  arr.sumComplex();
        sum = scalar.realComponent().doubleValue();
        assertEquals(10, sum, 1e-1);
        arr.subi(Nd4j.createDouble(1, 0));
        scalar = arr.sumComplex();

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
        IComplexDouble sum3 = (IComplexDouble) added.sumComplex();
        assertEquals(sum3, Nd4j.createDouble(324, 0));


    }






    @Test
    public void testMatrixGet() {

        IComplexNDArray arr = Nd4j.createComplex((Nd4j.linspace(1, 4, 4))).reshape(2, 2);
        IComplexNumber n1 = arr.getComplex(0, 0);
        IComplexNumber n2 = arr.getComplex(0, 1);
        IComplexNumber n3 = arr.getComplex(1, 0);
        IComplexNumber n4 = arr.getComplex(1, 1);

        assertEquals(1, n1.realComponent().doubleValue(), 1e-1);
        assertEquals(3, n2.realComponent().doubleValue(), 1e-1);
        assertEquals(2, n3.realComponent().doubleValue(), 1e-1);
        assertEquals(4, n4.realComponent().doubleValue(), 1e-1);
    }




    @Test
    public void testTwoByTwoMmul() {
        IComplexNDArray oneThroughFour = Nd4j.createComplex(Nd4j.linspace(1, 4, 4).reshape(2, 2));
        IComplexNDArray fiveThroughEight = Nd4j.createComplex(Nd4j.linspace(5, 8, 4).reshape(2, 2));

        IComplexNDArray solution = Nd4j.createComplex(Nd4j.create(new double[][]{{23, 31}, {34, 46}}));
        IComplexNDArray test = oneThroughFour.mmul(fiveThroughEight);
        assertEquals(getFailureMessage(),solution, test);

    }





    @Override
    public char ordering() {
        return 'f';
    }
}
