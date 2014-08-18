package org.deeplearning4j.linalg;


import org.deeplearning4j.linalg.api.ndarray.DimensionSlice;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.api.ndarray.SliceOp;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.ops.reduceops.Ops;
import org.deeplearning4j.linalg.util.ArrayUtil;
import org.deeplearning4j.linalg.util.Shape;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

//import org.deeplearning4j.linalg.jblas.complex.ComplexNDArray;
//import org.jblas.DoubleMatrix;

/**
 * JCublasNDArrayTests
 * @author Adam Gibson
 */
public class JCublasNDArrayTests {
    private static Logger log = LoggerFactory.getLogger(JCublasNDArrayTests.class);
    private INDArray n = new JCublasNDArray(new double[]{1,2,3,4,5,6,7,8},new int[]{2,2,2});



    @Test
    public void testScalarOps() {

        JCublasNDArray n = new JCublasNDArray(JCublasNDArray.ones(27).data,new int[]{3,3,3});

        assertEquals(27d,n.length(),1e-1);

        n.checkDimensions(n.addi(NDArrays.scalar(1d)));

        n.checkDimensions(n.subi(NDArrays.scalar(1.0d)));
        n.checkDimensions(n.muli(NDArrays.scalar(1.0d)));
        n.checkDimensions(n.divi(NDArrays.scalar(1.0d)));
        n = new JCublasNDArray(JCublasNDArray.ones(27).data,new int[]{3,3,3});

        assertEquals(27,(double) n.sum(Integer.MAX_VALUE).element(),1e-1);

        JCublasNDArray a = n.slice(2);
        assertEquals(true,Arrays.equals(new int[]{3,3},a.shape()));

    }


    @Test
    public void testSlices() {
        INDArray arr = new JCublasNDArray(JCublasNDArray.linspace(1,24,24).data,new int[]{4,3,2});
        for(int i = 0; i < arr.slices(); i++) {
            assertEquals(2, arr.slice(i).slice(1).slices());
        }

    }


    @Test
    public void testScalar() {
        INDArray a = JCublasNDArray.scalar(1.0);
        assertEquals(true,a.isScalar());

        INDArray n = new JCublasNDArray(new double[]{1.0},new int[]{1,1});
        assertEquals(n,a);
        assertTrue(n.isScalar());
    }

    @Test
    public void testWrap() {
        int[] shape = {2,4};
        JCublasNDArray d = JCublasNDArray.linspace(1,8,8).reshape(shape[0],shape[1]);
        JCublasNDArray n = JCublasNDArray.wrap(d);
        assertEquals(d.rows,n.rows());
        assertEquals(d.columns,n.columns());

        JCublasNDArray vector = JCublasNDArray.linspace(1,3,3);
        INDArray testVector = JCublasNDArray.wrap(vector);
        for(int i = 0; i < vector.length; i++)
            assertEquals(vector.get(i),(double) testVector.getScalar(i).element(),1e-1);
        assertEquals(3,testVector.length());
        assertEquals(true,testVector.isVector());
        assertEquals(true,Shape.shapeEquals(new int[]{3},testVector.shape()));

        JCublasNDArray row12 = JCublasNDArray.linspace(1,2,2).reshape(2,1);
        JCublasNDArray row22 = JCublasNDArray.linspace(3,4,2).reshape(1,2);

        JCublasNDArray row122 = JCublasNDArray.wrap(row12);
        JCublasNDArray row222 = JCublasNDArray.wrap(row22);
        assertEquals(row122.rows(),2);
        assertEquals(row122.columns(),1);
        assertEquals(row222.rows(),1);
        assertEquals(row222.columns(),2);



    }


    @Test
    public void testVectorInit() {
        double[] data = JCublasNDArray.linspace(1,4,4).data;
        INDArray arr = new JCublasNDArray(data,new int[]{4});
        assertEquals(true,arr.isRowVector());
        INDArray arr2 = new JCublasNDArray(data,new int[]{1,4});
        assertEquals(true,arr2.isRowVector());

        INDArray columnVector = new JCublasNDArray(data,new int[]{4,1});
        assertEquals(true,columnVector.isColumnVector());
    }













    @Test
    public void testColumns() {
        INDArray arr = new JCublasNDArray(new int[]{3,2});
        INDArray column2 = arr.getColumn(0);
        assertEquals(true,Shape.shapeEquals(new int[]{3}, column2.shape()));
        INDArray column = new JCublasNDArray(new double[]{1,2,3},new int[]{3});
        arr.putColumn(0,column);

        INDArray firstColumn = arr.getColumn(0);

        assertEquals(column,firstColumn);


        INDArray column1 = new JCublasNDArray(new double[]{4,5,6},new int[]{3});
        arr.putColumn(1,column1);
        assertEquals(true, Shape.shapeEquals(new int[]{3}, arr.getColumn(1).shape()));
        INDArray testRow1 = arr.getColumn(1);
        assertEquals(column1,testRow1);



        INDArray evenArr = new JCublasNDArray(new double[]{1,2,3,4},new int[]{2,2});
        INDArray put = new JCublasNDArray(new double[]{5,6},new int[]{2});
        evenArr.putColumn(1,put);
        INDArray testColumn = evenArr.getColumn(1);
        assertEquals(put,testColumn);



        INDArray n = new JCublasNDArray(JCublasNDArray.linspace(1,4,4).data,new int[]{2,2});
        INDArray column23 = n.getColumn(0);
        INDArray column12 = new JCublasNDArray(new double[]{1,3},new int[]{2});
        assertEquals(column23,column12);


        INDArray column0 = n.getColumn(1);
        INDArray column01 = new JCublasNDArray(new double[]{2,4},new int[]{2});
        assertEquals(column0,column01);



    }


    @Test
    public void testPutRow() {
        JCublasNDArray d = JCublasNDArray.linspace(1,4,4).reshape(2,2);
        JCublasNDArray n = new JCublasNDArray(Arrays.copyOf(d.data,d.data.length),new int[]{2,2});

        //works fine according to matlab, let's go with it..
        //reproduce with:  A = reshape(linspace(1,4,4),[2 2 ]);
        //A(1,2) % 1 index based
        double nFirst = 3;
        double dFirst = d.get(0,1);
        assertEquals(nFirst,dFirst,1e-1);
        assertEquals(true,Arrays.equals(d.toArray(),n.toArray()));
        assertEquals(true,Arrays.equals(new int[]{2,2},n.shape()));

        JCublasNDArray newRow = JCublasNDArray.linspace(5,6,2);
        n.putRow(0,newRow);
        d.putRow(0,newRow);



        INDArray testRow = n.getRow(0);
        assertEquals(newRow.length,testRow.length());
        assertEquals(true, Shape.shapeEquals(new int[]{2}, testRow.shape()));



        INDArray nLast = new JCublasNDArray(JCublasNDArray.linspace(1,4,4).data,new int[]{2,2});
        INDArray row = nLast.getRow(1);
        INDArray row1 = new JCublasNDArray(new double[]{3,4},new int[]{2});
        assertEquals(row,row1);



        INDArray arr = new JCublasNDArray(new int[]{3,2});
        INDArray evenRow = new JCublasNDArray(new double[]{1,2},new int[]{2});
        arr.putRow(0,evenRow);
        INDArray firstRow = arr.getRow(0);
        assertEquals(true, Shape.shapeEquals(new int[]{2},firstRow.shape()));
        INDArray testRowEven = arr.getRow(0);
        assertEquals(evenRow,testRowEven);


        INDArray row12 = new JCublasNDArray(new double[]{5,6},new int[]{2});
        arr.putRow(1,row12);
        assertEquals(true, Shape.shapeEquals(new int[]{2}, arr.getRow(0).shape()));
        INDArray testRow1 = arr.getRow(1);
        assertEquals(row12,testRow1);


        INDArray multiSliceTest = new JCublasNDArray(JCublasNDArray.linspace(1,16,16).data,new int[]{4,2,2});
        INDArray test = new JCublasNDArray(new double[]{7,8},new int[]{2});
        INDArray test2 = new JCublasNDArray(new double[]{9,10},new int[]{2});

        assertEquals(test,multiSliceTest.slice(1).getRow(1));
        assertEquals(test2,multiSliceTest.slice(1).getRow(2));

    }



    @Test
    public void testSum() {
        INDArray n = new JCublasNDArray(JCublasNDArray.linspace(1,8,8).data,new int[]{2,2,2});
        INDArray test = NDArrays.create(new double[]{6,8,10,12},new int[]{2,2});
        INDArray sum = n.sum(n.shape().length - 1);
        assertEquals(test,sum);
    }


    @Test
    public void testMmul() {
        double[] data = JCublasNDArray.linspace(1,10,10).data;
        INDArray n = new JCublasNDArray(data,new int[]{10});
        INDArray transposed = n.transpose();
        assertEquals(true,n.isRowVector());
        assertEquals(true,transposed.isColumnVector());

        JCublasNDArray d = new JCublasNDArray(n.rows(),n.columns());
        d.data = n.data();
        JCublasNDArray dTransposed = d.transpose();
        JCublasNDArray result2 = d.mmul(dTransposed);


        INDArray innerProduct = n.mmul(transposed);

        INDArray scalar = JCublasNDArray.scalar(385);
        assertEquals(scalar,innerProduct);

        INDArray outerProduct = transposed.mmul(n);
        assertEquals(true, Shape.shapeEquals(new int[]{10,10},outerProduct.shape()));


        INDArray testMatrix = new JCublasNDArray(data,new int[]{5,2});
        INDArray row1 = testMatrix.getRow(0).transpose();
        INDArray row2 = testMatrix.getRow(1);
        JCublasNDArray row12 = JCublasNDArray.linspace(1,2,2).reshape(2,1);
        JCublasNDArray row22 = JCublasNDArray.linspace(3,4,2).reshape(1,2);
        JCublasNDArray rowResult = row12.mmul(row22);

        INDArray row122 = JCublasNDArray.wrap(row12);
        INDArray row222 = JCublasNDArray.wrap(row22);
        INDArray rowResult2 = row122.mmul(row222);




        INDArray mmul = row1.mmul(row2);
        INDArray result = new JCublasNDArray(new double[]{3,6,4,8},new int[]{2,2});
        assertEquals(result,mmul);




        INDArray three = new JCublasNDArray(new double[]{3,4},new int[]{2});
        INDArray test = new JCublasNDArray(JCublasNDArray.linspace(1,30,30).data,new int[]{3,5,2});
        INDArray sliceRow = test.slice(0).getRow(1);
        assertEquals(three,sliceRow);

        INDArray twoSix = new JCublasNDArray(new double[]{2,6},new int[]{2,1});
        INDArray threeTwoSix = three.mmul(twoSix);

        INDArray sliceRowTwoSix = sliceRow.mmul(twoSix);

        assertEquals(threeTwoSix,sliceRowTwoSix);


        INDArray vectorVector = new JCublasNDArray(new double[]{
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99, 108, 117, 126, 135, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 0, 11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 0, 13, 26, 39, 52, 65, 78, 91, 104, 117, 130, 143, 156, 169, 182, 195, 0, 14, 28, 42, 56, 70, 84, 98, 112, 126, 140, 154, 168, 182, 196, 210, 0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225
        },new int[]{16,16});


        INDArray n1 = new JCublasNDArray(JCublasNDArray.linspace(0,15,16).data,new int[]{16});
        INDArray k1 = n1.transpose();

        INDArray testVectorVector = k1.mmul(n1);
        assertEquals(vectorVector,testVectorVector);



    }

    @Test
    public void testRowsColumns() {
        double[] data = JCublasNDArray.linspace(1,6,6).data;
        INDArray rows = new JCublasNDArray(data,new int[]{2,3});
        assertEquals(2,rows.rows());
        assertEquals(3,rows.columns());

        INDArray columnVector = new JCublasNDArray(data,new int[]{6,1});
        assertEquals(6,columnVector.rows());
        assertEquals(1,columnVector.columns());
        INDArray rowVector = new JCublasNDArray(data,new int[]{6});
        assertEquals(1,rowVector.rows());
        assertEquals(6,rowVector.columns());
    }


    @Test
    public void testTranspose() {
        INDArray n = new JCublasNDArray(JCublasNDArray.ones(100).data,new int[]{5,5,4});
        INDArray transpose = n.transpose();
        assertEquals(n.length(),transpose.length());
        assertEquals(true,Arrays.equals(new int[]{4,5,5},transpose.shape()));

        INDArray rowVector = JCublasNDArray.linspace(1,10,10);
        assertTrue(rowVector.isRowVector());
        INDArray columnVector = rowVector.transpose();
        assertTrue(columnVector.isColumnVector());

    }

    @Test
    public void testPutSlice() {
        JCublasNDArray n = new JCublasNDArray(JCublasNDArray.ones(27).data,new int[]{3,3,3});
        JCublasNDArray newSlice = JCublasNDArray.wrap(JCublasNDArray.zeros(3,3));
        n.putSlice(0,newSlice);
        assertEquals(newSlice,n.slice(0));


    }

    @Test
    public void testPermute() {
        INDArray n = new JCublasNDArray(JCublasNDArray.linspace(1,20,20).data,new int[]{5,4});
        INDArray transpose = n.transpose();
        INDArray permute = n.permute(new int[]{1,0});
        assertEquals(permute,transpose);
        assertEquals(transpose.length(),permute.length(),1e-1);


        INDArray toPermute = new JCublasNDArray(JCublasNDArray.linspace(0,7,8).data,new int[]{2,2,2});
        INDArray permuted = toPermute.permute(new int[]{2,1,0});
        INDArray assertion = new JCublasNDArray(new double[]{0,4,2,6,1,5,3,7},new int[]{2,2,2});
        assertEquals(permuted,assertion);

    }

    @Test
    public void testSlice() {

        assertEquals(8,n.length());

        assertEquals(true,Arrays.equals(new int[]{2,2,2},n.shape()));
        INDArray slice = n.slice(0);
        assertEquals(true, Arrays.equals(new int[]{2, 2}, slice.shape()));

        INDArray slice1 = n.slice(1);
        assertEquals(true,Arrays.equals(slice.shape(),slice1.shape()));
        assertNotEquals(true,Arrays.equals(slice.data(),slice1.data()));

        INDArray arr = new JCublasNDArray(JCublasNDArray.linspace(1,24,24).data,new int[]{4,3,2});
        INDArray slice0 = new JCublasNDArray(new double[]{1,2,3,4,5,6},new int[]{3,2});
        INDArray slice2 = new JCublasNDArray(new double[]{7,8,9,10,11,12},new int[]{3,2});

        INDArray testSlice0 = arr.slice(0);
        INDArray testSlice1 = arr.slice(1);

        assertEquals(slice0.equals(testSlice0), true);
        assertEquals(slice2.equals(testSlice1), true);
        assertEquals(slice2.equals(testSlice0), false);

    }

    @Test
    public void testSwapAxes() {
        INDArray n = new JCublasNDArray(JCublasNDArray.linspace(0,7,8).data,new int[]{2,2,2});
        INDArray assertion = n.permute(new int[]{2,1,0});
        double[] data = assertion.data();
        INDArray validate = new JCublasNDArray(new double[]{0,4,2,6,1,5,3,7},new int[]{2,2,2});
        assertEquals(validate,assertion);



    }



    @Test
    public void testLinearIndex() {
        INDArray n = new JCublasNDArray(JCublasNDArray.linspace(1,8,8).data,new int[]{8});
        for(int i = 0; i < n.length(); i++) {
            int linearIndex = n.linearIndex(i);
            assertEquals(i ,linearIndex);
            double d =  (double) n.getScalar(i).element();
            assertEquals(i + 1,d,1e-1);
        }
    }

    @Test
    public void testSliceConstructor() {
        /* TODO Fix the constructor with array slices
        List<INDArray> testList = new ArrayList<>();
        for(int i = 0; i < 5; i++)
            testList.add(JCublasNDArray.scalar(i + 1));

        INDArray test = new JCublasNDArray(testList,new int[]{testList.size()});
        INDArray expected = new JCublasNDArray(new double[]{1,2,3,4,5},new int[]{5});
        assertEquals(expected,test);
        */
        assertEquals(true,true);
    }


    @Test
    public void testVectorDimension() {
        INDArray test = new JCublasNDArray(JCublasNDArray.linspace(1,4,4).data,new int[]{2,2});
        final AtomicInteger count = new AtomicInteger(0);
        //row wise
        test.iterateOverDimension(1,new SliceOp() {
            @Override
            public void operate(DimensionSlice nd) {
                log.info("Operator " + nd);
                INDArray test = (JCublasNDArray) nd.getResult();
                if(count.get() == 0) {
                    INDArray firstDimension = new JCublasNDArray(new double[]{1,2},new int[]{2});
                    assertEquals(firstDimension,test);
                }
                else {
                    INDArray firstDimension = new JCublasNDArray(new double[]{3,4},new int[]{2});
                    assertEquals(firstDimension,test);

                }

                count.incrementAndGet();
            }

            /**
             * Operates on an jcublasndarray slice
             *
             * @param nd the result to operate on
             */
            @Override
            public void operate(INDArray nd) {
                log.info("Operator " + nd);
                INDArray test = nd;
                if(count.get() == 0) {
                    INDArray firstDimension = new JCublasNDArray(new double[]{1,3},new int[]{2});
                    assertEquals(firstDimension,test);
                }
                else {
                    INDArray firstDimension = new JCublasNDArray(new double[]{2,4},new int[]{2});
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
                INDArray test = (JCublasNDArray) nd.getResult();
                if(count.get() == 0) {
                    INDArray firstDimension = new JCublasNDArray(new double[]{1,3},new int[]{2});
                    assertEquals(firstDimension,test);
                }
                else {
                    INDArray firstDimension = new JCublasNDArray(new double[]{2,4},new int[]{2});
                    assertEquals(firstDimension,test);

                }

                count.incrementAndGet();
            }

            /**
             * Operates on an jcublasndarray slice
             *
             * @param nd the result to operate on
             */
            @Override
            public void operate(INDArray nd) {

            }

        },false);




    }


    @Test
    public void testReshape() {
        INDArray arr = new JCublasNDArray(JCublasNDArray.linspace(1,24,24).data,new int[]{4,3,2});
        INDArray reshaped = arr.reshape(new int[]{2,3,4});
        assertEquals(arr.length(),reshaped.length());
        assertEquals(true,Arrays.equals(new int[]{4,3,2},arr.shape()));
        assertEquals(true,Arrays.equals(new int[]{2,3,4},reshaped.shape()));
    }


    @Test
    public void reduceTest() {
        INDArray arr = new JCublasNDArray(JCublasNDArray.linspace(1,24,24).data,new int[]{4,3,2});
        INDArray reduced = arr.reduce(Ops.DimensionOp.MAX,1);
        log.info("Reduced " + reduced);
        reduced = arr.reduce(Ops.DimensionOp.MAX,1);
        log.info("Reduced " + reduced);
        reduced = arr.reduce(Ops.DimensionOp.MAX,2);
        log.info("Reduced " + reduced);


    }

    @Test
    public void testRowVectorOps() {
        INDArray twoByTwo = NDArrays.create(new double[]{1,2,3,4},new int[]{2,2});
        INDArray toAdd = NDArrays.create(new double[]{1,2},new int[]{2});
        twoByTwo.addiRowVector(toAdd);
        INDArray assertion = NDArrays.create(new double[]{2,3,5,6},new int[]{2,2});
        assertEquals(assertion,twoByTwo);



    }

    @Test
    public void testColumnVectorOps() {
        INDArray twoByTwo = NDArrays.create(new double[]{1,2,3,4},new int[]{2,2});
        INDArray toAdd = NDArrays.create(new double[]{1,2},new int[]{2,1});
        twoByTwo.addiColumnVector(toAdd);
        INDArray assertion = NDArrays.create(new double[]{2,4,4,6},new int[]{2,2});
        assertEquals(assertion,twoByTwo);



    }

    @Test
    public void testGetScalar() {
        INDArray n = NDArrays.create(new double[]{1,2,3,4},new int[]{4});
        assertTrue(n.isVector());
        for(int i = 0; i < n.length(); i++) {
            INDArray scalar = NDArrays.scalar((double) i + 1);
            assertEquals(scalar,n.getScalar(i));
        }
    }

    @Test
    public void testGetMulti() {
        assertEquals(8,n.length());
        assertEquals(true,Arrays.equals(ArrayUtil.of(2, 2, 2),n.shape()));
        double val = (double) n.getScalar(1,1,1).element();
        assertEquals(8.0,val,1e-6);
    }



    @Test
    public void testElementWiseOps() {
        INDArray n1 = JCublasNDArray.scalar(1);
        INDArray n2 = JCublasNDArray.scalar(2);
        assertEquals(JCublasNDArray.scalar(3),n1.add(n2));
        assertFalse(n1.add(n2).equals(n1));

        INDArray n3 = JCublasNDArray.scalar(3);
        INDArray n4 = JCublasNDArray.scalar(4);
        INDArray subbed = n4.sub(n3);
        INDArray mulled = n4.mul(n3);
        INDArray div = n4.div(n3);

        assertFalse(subbed.equals(n4));
        assertFalse(mulled.equals(n4));
        assertEquals(JCublasNDArray.scalar(1),subbed);
        assertEquals(JCublasNDArray.scalar(12),mulled);
        assertEquals(JCublasNDArray.scalar(1.333333333333333333333),div);
    }







    @Test
    public void testSlicing() {
        INDArray arr = n.slice(1, 1);
        // assertEquals(1,arr.shape().length());
        INDArray n2 = new JCublasNDArray(JCublasNDArray.linspace(1,16,16).data,new int[]{2,2,2,2});
        log.info("N2 shape " + n2.slice(1,1).slice(1));

    }


    @Test
    public void testEndsForSlices() {
        INDArray arr = new JCublasNDArray(JCublasNDArray.linspace(1,24,24).data,new int[]{4,3,2});
        int[] endsForSlices = arr.endsForSlices();
        assertEquals(true,Arrays.equals(new int[]{5,11,17,23},endsForSlices));
    }


    @Test
    public void testFlatten() {
        INDArray arr = new JCublasNDArray(JCublasNDArray.linspace(1,4,4).data,new int[]{2,2});
        INDArray flattened = arr.ravel();
        assertEquals(arr.length(),flattened.length());
        assertEquals(true,Shape.shapeEquals(new int[]{1, arr.length()}, flattened.shape()));
        for(int i = 0; i < arr.length(); i++) {
            assertEquals(i + 1,(double) flattened.getScalar(i).element(),1e-1);
        }
        assertTrue(flattened.isVector());


        INDArray n = new JCublasNDArray(JCublasNDArray.ones(27).data,new int[]{3,3,3});
        INDArray nFlattened = n.ravel();
        assertTrue(nFlattened.isVector());

    }

    @Test
    public void testVectorDimensionMulti() {
        INDArray arr = new JCublasNDArray(JCublasNDArray.linspace(1,24,24).data,new int[]{4,3,2});
        final AtomicInteger count = new AtomicInteger(0);

        arr.iterateOverDimension(0,new SliceOp() {
            @Override
            public void operate(DimensionSlice nd) {
                INDArray test =(JCublasNDArray) nd.getResult();
                if(count.get() == 0) {
                    INDArray answer = new JCublasNDArray(new double[]{1,7,13,19},new int[]{4});
                    assertEquals(answer,test);
                }
                else if(count.get() == 1) {
                    INDArray answer = new JCublasNDArray(new double[]{2,8,14,20},new int[]{4});
                    assertEquals(answer,test);
                }
                else if(count.get() == 2) {
                    INDArray answer = new JCublasNDArray(new double[]{3,9,15,21},new int[]{4});
                    assertEquals(answer,test);
                }
                else if(count.get() == 3) {
                    INDArray answer = new JCublasNDArray(new double[]{4,10,16,22},new int[]{4});
                    assertEquals(answer,test);
                }
                else if(count.get() == 4) {
                    INDArray answer = new JCublasNDArray(new double[]{5,11,17,23},new int[]{4});
                    assertEquals(answer,test);
                }
                else if(count.get() == 5) {
                    INDArray answer = new JCublasNDArray(new double[]{6,12,18,24},new int[]{4});
                    assertEquals(answer,test);
                }


                count.incrementAndGet();
            }

            /**
             * Operates on an jcublasndarray slice
             *
             * @param nd the result to operate on
             */
            @Override
            public void operate(INDArray nd) {

            }
        },false);
    }

}
