package org.deeplearning4j.nn.linalg;

import static org.junit.Assert.*;

import org.deeplearning4j.util.ArrayUtil;
import org.deeplearning4j.util.ComplexNDArrayUtil;
import org.jblas.ComplexDouble;
import org.jblas.ComplexDoubleMatrix;
import org.jblas.DoubleMatrix;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Tests for a complex ndarray
 */
public class ComplexNDArrayTests {

    private static Logger log = LoggerFactory.getLogger(ComplexNDArrayTests.class);

    @Test
    public void testConstruction() {

        ComplexNDArray arr2 = new ComplexNDArray(new int[]{3,2});
        assertEquals(3,arr2.rows());
        assertEquals(arr2.rows(),arr2.rows);
        assertEquals(2,arr2.columns());
        assertEquals(arr2.columns(),arr2.columns);
        assertTrue(arr2.isMatrix());



        ComplexNDArray arr = new ComplexNDArray(new double[]{0,1},new int[]{1});
        //only each complex double: one element
        assertEquals(1,arr.length);
        //both real and imaginary components
        assertEquals(2,arr.data.length);
        assertEquals(0,arr.get(0).real(),1e-1);




    }

    @Test
    public void testVectorGet() {
        ComplexNDArray arr = new ComplexNDArray(new NDArray(DoubleMatrix.linspace(1,8,8).data,new int[]{8}));
        for(int i = 0; i < arr.length; i++) {
            assertEquals(new ComplexDouble(i + 1),arr.get(i));
        }

        ComplexNDArray matrix = new ComplexNDArray(new NDArray(DoubleMatrix.linspace(1,8,8).data,new int[]{2,4}));
        ComplexNDArray row = matrix.getRow(1);
        ComplexNDArray column = matrix.getColumn(1);

        ComplexNDArray validate = new ComplexNDArray(new NDArray(new double[]{5,6,7,8},new int[]{4}));
        ComplexDouble d = row.get(3);
        assertEquals(new ComplexDouble(8), d);
        assertEquals(row,validate);

        assertEquals(new ComplexDouble(6),column.get(1));





    }

    @Test
    public void testSwapAxes() {
        ComplexNDArray n = new ComplexNDArray(new NDArray(new double[]{1,2,3},new int[]{3,1}));
        ComplexNDArray swapped = n.swapAxes(1,0);
        assertEquals(n.transpose(),swapped);
        //vector despite being transposed should have same linear index
        assertEquals(swapped.get(0),n.get(0));
        assertEquals(swapped.get(1),n.get(1));
        assertEquals(swapped.get(2),n.get(2));

        ComplexNDArray n2 = new ComplexNDArray(new NDArray(DoubleMatrix.linspace(0,7,8).data,new int[]{2,2,2}));
        ComplexNDArray assertion = n2.permute(new int[]{2,1,0});
        ComplexNDArray validate = new ComplexNDArray(new NDArray(new double[]{0,4,2,6,1,5,3,7},new int[]{2,2,2}));
        assertEquals(validate,assertion);


        ComplexNDArray v1 = new ComplexNDArray(new NDArray(DoubleMatrix.linspace(1,8,8).data,new int[]{8,1}));
        ComplexNDArray swap = v1.swapAxes(1,0);
        ComplexNDArray transposed = v1.transpose();
        assertEquals(swap, transposed);


        transposed.put(1,9);
        swap.put(1,9);
        assertEquals(transposed,swap);
        assertEquals(transposed.get(1),swap.get(1));


        ComplexNDArray row = n2.slice(0).getRow(1);
        row.put(1,9);
        assertEquals(9,row.get(1).real(),1e-1);






    }


    @Test
    public void testSlice() {
        NDArray arr = new NDArray(DoubleMatrix.linspace(1,24,24).data,new int[]{4,3,2});
        ComplexNDArray arr2 = new ComplexNDArray(arr);
        assertEquals(arr,arr2.getReal());

        NDArray firstSlice = arr.slice(0);
        NDArray firstSliceTest = arr2.slice(0).getReal();
        assertEquals(firstSlice,firstSliceTest);


        NDArray secondSlice = arr.slice(1);
        NDArray secondSliceTest = arr2.slice(1).getReal();
        assertEquals(secondSlice,secondSliceTest);


        NDArray slice0 = new NDArray(new double[]{1,2,3,4,5,6},new int[]{3,2});
        NDArray slice2 = new NDArray(new double[]{7,8,9,10,11,12},new int[]{3,2});


        ComplexNDArray testSliceComplex = arr2.slice(0);
        ComplexNDArray testSliceComplex2 = arr2.slice(1);

        NDArray testSlice0 = testSliceComplex.getReal();
        NDArray testSlice1 = testSliceComplex2.getReal();

        assertEquals(slice0,testSlice0);
        assertEquals(slice2,testSlice1);


    }

    @Test
    public void testSliceConstructor() {
        List<ComplexNDArray> testList = new ArrayList<>();
        for(int i = 0; i < 5; i++)
            testList.add(ComplexNDArray.scalar(i + 1));

        ComplexNDArray test = new ComplexNDArray(testList,new int[]{testList.size()});
        ComplexNDArray expected = new ComplexNDArray(new NDArray(new double[]{1,2,3,4,5},new int[]{5}));
        assertEquals(expected,test);
    }


    @Test
    public void testVectorInit() {
        double[] data = DoubleMatrix.linspace(1,4,4).data;
        ComplexNDArray arr = new ComplexNDArray(data,new int[]{4});
        assertEquals(true,arr.isRowVector());
        ComplexNDArray arr2 = new ComplexNDArray(data,new int[]{1,4});
        assertEquals(true,arr2.isRowVector());

        ComplexNDArray columnVector = new ComplexNDArray(data,new int[]{4,1});
        assertEquals(true,columnVector.isColumnVector());
    }



    @Test
    public void testIterateOverAllRows() {
        ComplexNDArray c = new ComplexNDArray(new NDArray(DoubleMatrix.linspace(0,29,30).data,new int[]{3,5,2}));

        final AtomicInteger i = new AtomicInteger(0);
        final Set<ComplexNDArray> set = new HashSet<>();

        c.iterateOverAllRows(new SliceOp() {
            @Override
            public void operate(DimensionSlice nd) {
                ComplexNDArray result = (ComplexNDArray) nd.getResult();
                int curr = i.get();
                i.incrementAndGet();
                ComplexNDArray test = new ComplexNDArray(new double[]{curr * 2,0,curr * 2 + 1,0},new int[]{2});
                assertEquals(result,test);
                assertEquals(true,!set.contains(test));
                set.add(result);

                result.put(0,(curr + 1) * 3);
                result.put(1,(curr + 2) * 3);
                assertEquals((curr + 1) * 3,result.get(0).real(),1e-1);
                assertEquals((curr + 2) * 3,result.get(1).real(),1e-1);
            }
        });

        ComplexNDArray permuted = c.permute(new int[]{2,1,0});
        set.clear();
        i.set(0);

        permuted.iterateOverAllRows(new SliceOp() {
            @Override
            public void operate(DimensionSlice nd) {
                ComplexNDArray result = (ComplexNDArray) nd.getResult();
                int curr = i.get();
                i.incrementAndGet();

                result.put(0,(curr + 1) * 3);
                result.put(1,(curr + 2) * 3);
                assertEquals((curr + 1) * 3,result.get(0).real(),1e-1);
                assertEquals((curr + 2) * 3,result.get(1).real(),1e-1);
            }
        });

        ComplexNDArray swapped = c.swapAxes(2,1);
        i.set(0);

        swapped.iterateOverAllRows(new SliceOp() {
            @Override
            public void operate(DimensionSlice nd) {
                ComplexNDArray result = (ComplexNDArray) nd.getResult();
                int curr = i.get();
                i.incrementAndGet();

                result.put(0,(curr + 1) * 3);
                result.put(1,(curr + 2) * 3);
                assertEquals((curr + 1) * 3,result.get(0).real(),1e-1);
                assertEquals((curr + 2) * 3,result.get(1).real(),1e-1);
            }
        });





    }


    @Test
    public void testMmul() {
        double[] data = DoubleMatrix.linspace(1,10,10).data;
        ComplexNDArray n = new ComplexNDArray((new NDArray(data,new int[]{10})));
        ComplexNDArray transposed = n.transpose();
        assertEquals(true,n.isRowVector());
        assertEquals(true,transposed.isColumnVector());

        ComplexNDArray innerProduct = n.mmul(transposed);
        NDArray scalar = NDArray.scalar(385);
        assertEquals(scalar,innerProduct.getReal());

        ComplexNDArray outerProduct = transposed.mmul(n);
        assertEquals(true, Shape.shapeEquals(new int[]{10,10},outerProduct.shape()));


        ComplexNDArray three = new ComplexNDArray(new NDArray(new double[]{3,4},new int[]{2}));
        ComplexNDArray test = new ComplexNDArray(new NDArray(DoubleMatrix.linspace(1,30,30).data,new int[]{3,5,2}));
        ComplexNDArray sliceRow = test.slice(0).getRow(1);
        assertEquals(three,sliceRow);

        ComplexNDArray twoSix = new ComplexNDArray(new NDArray(new double[]{2,6},new int[]{2,1}));
        ComplexNDArray threeTwoSix = three.mmul(twoSix);

        ComplexNDArray sliceRowTwoSix = sliceRow.mmul(twoSix);

        assertEquals(threeTwoSix,sliceRowTwoSix);



        ComplexNDArray anotherOffsetTest = new ComplexNDArray(new double[]{
                3.0,0.0,-1.0,-2.4492935982947064E-16,7.0,0.0,-1.0,-4.898587196589413E-16,11.0,0.0,-1.0,
                -7.347880794884119E-16,15.0,0.0,-1.0,-9.797174393178826E-16,19.0,0.0,-1.0,-1.2246467991473533E-15,23.0,0.0,-1.0,
                -1.4695761589768238E-15,27.0,0.0,-1.0,-1.7145055188062944E-15,31.0,0.0,-0.9999999999999982,-1.959434878635765E-15,35.0,0.0,
                -1.0,-2.204364238465236E-15,39.0,0.0,-1.0,-2.4492935982947065E-15,43.0,0.0,-1.0,-2.6942229581241772E-15,47.0,0.0,-1.0000000000000036,
                -2.9391523179536483E-15,51.0,0.0,-0.9999999999999964,-3.1840816777831178E-15,55.0,0.0,-1.0,-3.429011037612589E-15,59.0,0.0,-0.9999999999999964,
                -3.67394039744206E-15},new int[]{3,2,5},new int[]{20,2,4});

        ComplexNDArray rowToTest = anotherOffsetTest.slice(0).slice(0);
        ComplexNDArray noOffsetRow = new ComplexNDArray(new double[]{3,0,7,0,11,0,15,0,19,0},new int[]{5});
        assertEquals(rowToTest,noOffsetRow);

        ComplexNDArray rowOther = new ComplexNDArray(new NDArray(DoubleMatrix.linspace(1,8,8).data,new int[]{5,1}));
        ComplexNDArray noOffsetTimesrowOther = noOffsetRow.mmul(rowOther);
        ComplexNDArray rowToTestTimesrowOther = rowToTest.mmul(rowOther);
        assertEquals(noOffsetTimesrowOther,rowToTestTimesrowOther);




    }



    @Test
    public void testLinearData() {
        double[] d = {1,0,2,0};
        ComplexNDArray c = new ComplexNDArray(d,new int[]{2});
        assertTrue(Arrays.equals(d,c.data()));

        ComplexNDArray needsToBeFlattened = new ComplexNDArray(new NDArray(new double[]{1,2,3,4},new int[]{2,2}));
        double[] d2 = {1,0,2,0,3,0,4,0};
        assertTrue(Arrays.equals(d2,needsToBeFlattened.data()));

        ComplexNDArray anotherOffsetTest = new ComplexNDArray(new double[]{
                3.0,0.0,-1.0,-2.4492935982947064E-16,7.0,0.0,-1.0,-4.898587196589413E-16,11.0,0.0,-1.0,-7.347880794884119E-16,15.0,0.0,-1.0,-9.797174393178826E-16,19.0,0.0,-1.0,-1.2246467991473533E-15,23.0,0.0,-1.0,-1.4695761589768238E-15,27.0,0.0,-1.0,-1.7145055188062944E-15,31.0,0.0,-0.9999999999999982,-1.959434878635765E-15,35.0,0.0,-1.0,-2.204364238465236E-15,39.0,0.0,-1.0,-2.4492935982947065E-15,43.0,0.0,-1.0,-2.6942229581241772E-15,47.0,0.0,-1.0000000000000036,-2.9391523179536483E-15,51.0,0.0,-0.9999999999999964,-3.1840816777831178E-15,55.0,0.0,-1.0,-3.429011037612589E-15,59.0,0.0,-0.9999999999999964,-3.67394039744206E-15},new int[]{3,2,5},new int[]{20,2,4});


        ComplexNDArray rowToTest = anotherOffsetTest.slice(0).slice(0);
        ComplexNDArray noOffsetRow = new ComplexNDArray(new double[]{3,0,-1,0,7,0,-1,0,11,0},new int[]{5});
        assertEquals(rowToTest,noOffsetRow);

    }

    @Test
    public void testGetRow() {
        ComplexNDArray arr = new ComplexNDArray(new int[]{3,2});
        ComplexNDArray row = new ComplexNDArray(new double[]{1,0,2,0},new int[]{2});
        arr.putRow(0,row);
        ComplexNDArray firstRow = arr.getRow(0);
        assertEquals(true, Shape.shapeEquals(new int[]{2},firstRow.shape()));
        ComplexNDArray testRow = arr.getRow(0);
        assertEquals(row,testRow);


        ComplexNDArray row1 = new ComplexNDArray(new double[]{3,0,4,0},new int[]{2});
        arr.putRow(1,row1);
        assertEquals(true, Shape.shapeEquals(new int[]{2}, arr.getRow(0).shape()));
        ComplexNDArray testRow1 = arr.getRow(1);
        assertEquals(row1,testRow1);

        ComplexNDArray multiRow = new ComplexNDArray(new NDArray(DoubleMatrix.linspace(1,16,16).data,new int[]{4,2,2}));
        ComplexNDArray test = new ComplexNDArray(new NDArray(new double[]{7,8},new int[]{1,2}));
        ComplexNDArray multiRowSlice1 = multiRow.slice(0);
        ComplexNDArray multiRowSlice = multiRow.slice(1);
        ComplexNDArray testMultiRow = multiRowSlice.getRow(1);

        assertEquals(test,testMultiRow);



    }

    @Test
    public void testLinearIndex() {
        ComplexNDArray n = new ComplexNDArray(new NDArray(DoubleMatrix.linspace(1,8,8).data,new int[]{8}));
        for(int i = 0; i < n.length; i++) {
            int linearIndex = n.linearIndex(i);
            assertEquals(i * 2,linearIndex);
            double curr = n.get(i).real();
            assertEquals(i + 1,curr,1e-1);
        }
    }


    @Test
    public void testNdArrayConstructor() {
        ComplexNDArray result = new ComplexNDArray(new NDArray(new double[]{2,6},new int[]{1,2}));
        result.toString();
    }

    @Test
    public void testGetColumn() {
        ComplexNDArray arr = new ComplexNDArray(new NDArray(DoubleMatrix.linspace(1,8,8).data,new int[]{2,4}));
        ComplexNDArray column2 = arr.getColumn(1);
        ComplexNDArray result = new ComplexNDArray(new NDArray(new double[]{2,6},new int[]{1,2}));

        assertEquals(result, column2);
        assertEquals(true,Shape.shapeEquals(new int[]{2}, column2.shape()));
        ComplexNDArray column = new ComplexNDArray(new double[]{11,0,12,0},new int[]{2});
        arr.putColumn(1,column);

        ComplexNDArray firstColumn = arr.getColumn(1);

        assertEquals(column,firstColumn);


        ComplexNDArray column1 = new ComplexNDArray(new double[]{5,0,6,0},new int[]{2});
        arr.putColumn(1,column1);
        assertEquals(true, Shape.shapeEquals(new int[]{2}, arr.getColumn(1).shape()));
        ComplexNDArray testC = arr.getColumn(1);
        assertEquals(column1,testC);


        ComplexNDArray multiSlice = new ComplexNDArray(new NDArray(DoubleMatrix.linspace(1,32,32).data,new int[]{4,4,2}));
        ComplexNDArray testColumn = new ComplexNDArray(new NDArray(new double[]{10,12,14,16},new int[]{4}));
        ComplexNDArray sliceColumn = multiSlice.slice(1).getColumn(1);
        assertEquals(sliceColumn,testColumn);

        ComplexNDArray testColumn2 = new ComplexNDArray(new NDArray(new double[]{17,19,21,23},new int[]{4}));
        ComplexNDArray testSlice2 = multiSlice.slice(2).getColumn(0);
        assertEquals(testColumn2,testSlice2);

        ComplexNDArray testColumn3 = new ComplexNDArray(new NDArray(new double[]{18,20,22,24},new int[]{4}));
        ComplexNDArray testSlice3 = multiSlice.slice(2).getColumn(1);
        assertEquals(testColumn3,testSlice3);

    }






    @Test
    public void testPutAndGet() {
        ComplexNDArray arr = new ComplexNDArray(new NDArray(new double[]{1,2,3,4},new int[]{2,2}));
        assertEquals(4,arr.length);
        assertEquals(8,arr.data.length);
        arr.put(1,1,5.0);
        assertEquals(5.0,arr.get(1,1).real(),1e-1);
        assertEquals(0.0,arr.get(1,1).imag(),1e-1);

    }

    @Test
    public void testGetReal() {
        double[] data = DoubleMatrix.linspace(1,8,8).data;
        int[] shape = new int[]{1,8};
        ComplexNDArray arr = new ComplexNDArray(shape);
        for(int i = 0;i  < arr.length; i++)
            arr.put(i,data[i]);
        NDArray arr2 = new NDArray(data,shape);
        assertEquals(arr2,arr.getReal());
    }




    @Test
    public void testBasicOperations() {
        ComplexNDArray arr = new ComplexNDArray(new double[]{0,1,2,1,1,2,3,4},new int[]{2,2});
        double sum = arr.sum().real();
        assertEquals(4,sum,1e-1);
        arr.addi(1);
        sum = arr.sum().real();
        assertEquals(8,sum,1e-1);
        arr.subi(1);
        sum = arr.sum().real();
        assertEquals(4,sum,1e-1);
    }



    @Test
    public void testElementWiseOps() {
        ComplexNDArray n1 = ComplexNDArray.scalar(1);
        ComplexNDArray n2 = ComplexNDArray.scalar(2);
        assertEquals(ComplexNDArray.scalar(3),n1.add(n2));
        assertFalse(n1.add(n2).equals(n1));

        ComplexNDArray n3 = ComplexNDArray.scalar(3);
        ComplexNDArray n4 = ComplexNDArray.scalar(4);
        ComplexNDArray subbed = n4.sub(n3);
        ComplexNDArray mulled = n4.mul(n3);
        ComplexNDArray div = n4.div(n3);

        assertFalse(subbed.equals(n4));
        assertFalse(mulled.equals(n4));
        assertEquals(ComplexNDArray.scalar(1),subbed);
        assertEquals(ComplexNDArray.scalar(12),mulled);
        assertEquals(ComplexNDArray.scalar(new ComplexDouble(1.3333333333333333)),div);
    }


    @Test
    public void testVectorDimension() {
        ComplexNDArray test = new ComplexNDArray(new double[]{1,0,2,0,3,0,4,0},new int[]{2,2});
        final AtomicInteger count = new AtomicInteger(0);
        //row wise
        test.iterateOverDimension(1,new SliceOp() {
            @Override
            public void operate(DimensionSlice nd) {
                log.info("Operator " + nd);
                ComplexNDArray test = (ComplexNDArray) nd.getResult();
                if(count.get() == 0) {
                    ComplexNDArray firstDimension = new ComplexNDArray(new double[]{1,0,2,0},new int[]{2,1});
                    assertEquals(firstDimension,test);
                }
                else {
                    ComplexNDArray firstDimension = new ComplexNDArray(new double[]{3,0,4,0},new int[]{2});
                    assertEquals(firstDimension,test);

                }

                count.incrementAndGet();
            }

        },false);



        count.set(0);

        //columnwise
        test.iterateOverDimension(0,new SliceOp() {
            @Override
            public void operate(DimensionSlice nd) {
                log.info("Operator " + nd);
                ComplexNDArray test = (ComplexNDArray) nd.getResult();
                if(count.get() == 0) {
                    ComplexNDArray firstDimension = new ComplexNDArray(new double[]{1,0,3,0},new int[]{2});
                    assertEquals(firstDimension,test);
                }
                else {
                    ComplexNDArray firstDimension = new ComplexNDArray(new double[]{2,0,4,0},new int[]{2});
                    assertEquals(firstDimension,test);

                }

                count.incrementAndGet();
            }

        },false);




    }

    @Test
    public void testFlatten() {
        ComplexNDArray arr = new ComplexNDArray(new NDArray(DoubleMatrix.linspace(1,4,4).data,new int[]{2,2}));
        ComplexNDArray flattened = arr.flatten();
        assertEquals(arr.length,flattened.length);
        assertTrue(Shape.shapeEquals(new int[]{1, 4}, flattened.shape()));
        for(int i = 0; i < arr.length; i++) {
            assertEquals(i + 1,flattened.get(i).real(),1e-1);
        }
    }


    @Test
    public void testMatrixGet() {
        ComplexNDArray arr = new ComplexNDArray(new NDArray(DoubleMatrix.linspace(1,4,4).data,new int[]{2,2}));
        assertEquals(1,arr.get(0,0).real(),1e-1);
        assertEquals(2,arr.get(0,1).real(),1e-1);
        assertEquals(3,arr.get(1,0).real(),1e-1);
        assertEquals(4,arr.get(1,1).real(),1e-1);
    }

    @Test
    public void testEndsForSlices() {
        ComplexNDArray arr = new ComplexNDArray(new NDArray(DoubleMatrix.linspace(1,24,24).data,new int[]{4,3,2}));
        int[] endsForSlices = arr.endsForSlices();
        assertEquals(true, Arrays.equals(new int[]{0, 12, 24, 36}, endsForSlices));
    }


    @Test
    public void testWrap() {
        ComplexDoubleMatrix c = new ComplexDoubleMatrix(DoubleMatrix.linspace(1,4,4).reshape(2,2));
        ComplexNDArray wrapped = ComplexNDArray.wrap(c);
        assertEquals(true,Arrays.equals(new int[]{2,2},wrapped.shape()));

        ComplexDoubleMatrix vec = new ComplexDoubleMatrix(DoubleMatrix.linspace(1,4,4));
        ComplexNDArray wrappedVector = ComplexNDArray.wrap(vec);
        assertEquals(true,wrappedVector.isVector());
        assertEquals(true,Shape.shapeEquals(new int[]{4},wrappedVector.shape()));

    }



    @Test
    public void testVectorDimensionMulti() {
        ComplexNDArray arr = new ComplexNDArray(new NDArray(DoubleMatrix.linspace(1,24,24).data,new int[]{4,3,2}));
        final AtomicInteger count = new AtomicInteger(0);

        arr.iterateOverDimension(0,new SliceOp() {
            @Override
            public void operate(DimensionSlice nd) {
                ComplexNDArray test =(ComplexNDArray) nd.getResult();
                if(count.get() == 0) {
                    ComplexNDArray answer = new ComplexNDArray(new double[]{1,0,7,0,13,0,19,0},new int[]{4});
                    assertEquals(answer,test);
                }
                else if(count.get() == 1) {
                    ComplexNDArray answer = new ComplexNDArray(new double[]{2,0,8,0,14,0,20,0},new int[]{4});
                    assertEquals(answer,test);
                }
                else if(count.get() == 2) {
                    ComplexNDArray answer = new ComplexNDArray(new double[]{3,0,9,0,15,0,21,0},new int[]{4});
                    assertEquals(answer,test);
                }
                else if(count.get() == 3) {
                    ComplexNDArray answer = new ComplexNDArray(new double[]{4,0,10,0,16,0,22,0},new int[]{4});
                    assertEquals(answer,test);
                }
                else if(count.get() == 4) {
                    ComplexNDArray answer = new ComplexNDArray(new double[]{5,0,11,0,17,0,23,0},new int[]{4});
                    assertEquals(answer,test);
                }
                else if(count.get() == 5) {
                    ComplexNDArray answer = new ComplexNDArray(new double[]{6,0,12,0,18,0,24,0},new int[]{4});
                    assertEquals(answer,test);
                }


                count.incrementAndGet();
            }
        },false);



        ComplexNDArray ret = new ComplexNDArray(new double[]{1,0,2,0,3,0,4,0},new int[]{2,2});
        final ComplexNDArray firstRow = new ComplexNDArray(new double[]{1,0,2,0},new int[]{2});
        final ComplexNDArray secondRow = new ComplexNDArray(new double[]{3,0,4,0},new int[]{2});
        count.set(0);
        ret.iterateOverDimension(1,new SliceOp() {
            @Override
            public void operate(DimensionSlice nd) {
                ComplexNDArray c = (ComplexNDArray) nd.getResult();
                if(count.get() == 0) {
                    assertEquals(firstRow,c);
                }
                else if(count.get() == 1)
                    assertEquals(secondRow,c);
                count.incrementAndGet();
            }
        },false);
    }



}
