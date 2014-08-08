package org.deeplearning4j.nn.linalg;

import static org.junit.Assert.*;

import org.deeplearning4j.util.ArrayUtil;
import org.deeplearning4j.util.NDArrayUtil;
import org.jblas.DoubleMatrix;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * NDArrayTests
 * @author Adam Gibson
 */
public class NDArrayTests {
    private static Logger log = LoggerFactory.getLogger(NDArrayTests.class);
    private NDArray n = new NDArray(DoubleMatrix.linspace(1,8,8).data,new int[]{2,2,2});



    @Test
    public void testScalarOps() {
        NDArray n = new NDArray(DoubleMatrix.ones(27).data,new int[]{3,3,3});
        assertEquals(27,n.length);
        n.checkDimensions(n.addi(1));
        n.checkDimensions(n.subi(1.0));
        n.checkDimensions(n.muli(1.0));
        n.checkDimensions(n.divi(1.0));

        n = new NDArray(DoubleMatrix.ones(27).data,new int[]{3,3,3});
        assertEquals(27,n.sum(),1e-1);
        NDArray a = n.slice(2);
        assertEquals(true,Arrays.equals(new int[]{3,3},a.shape()));

    }


    @Test
    public void testSlices() {
        NDArray arr = new NDArray(DoubleMatrix.linspace(1,24,24).data,new int[]{4,3,2});
        for(int i = 0; i < arr.slices(); i++) {
            assertEquals(2, arr.slice(i).slice(1).slices());
        }

    }


    @Test
    public void testScalar() {
        NDArray a = NDArray.scalar(1.0);
        assertEquals(true,a.isScalar());

        NDArray n = new NDArray(new double[]{1.0},new int[]{1,1});
        assertEquals(n,a);
        assertTrue(n.isScalar());
    }

    @Test
    public void testWrap() {
        int[] shape = {2,4};
        DoubleMatrix d = DoubleMatrix.linspace(1,8,8).reshape(shape[0],shape[1]);
        NDArray n = NDArray.wrap(d);
        assertEquals(d.rows,n.rows());
        assertEquals(d.columns,n.columns());

        DoubleMatrix vector = DoubleMatrix.linspace(1,3,3);
        NDArray testVector = NDArray.wrap(vector);
        for(int i = 0; i < vector.length; i++)
            assertEquals(vector.get(i),testVector.get(i),1e-1);
        assertEquals(3,testVector.length);
        assertEquals(true,testVector.isVector());
        assertEquals(true,Shape.shapeEquals(new int[]{3},testVector.shape()));

        DoubleMatrix row12 = DoubleMatrix.linspace(1,2,2).reshape(2,1);
        DoubleMatrix row22 = DoubleMatrix.linspace(3,4,2).reshape(1,2);

        NDArray row122 = NDArray.wrap(row12);
        NDArray row222 = NDArray.wrap(row22);
        assertEquals(row122.rows(),2);
        assertEquals(row122.columns(),1);
        assertEquals(row222.rows(),1);
        assertEquals(row222.columns(),2);



    }


    @Test
    public void testVectorInit() {
        double[] data = DoubleMatrix.linspace(1,4,4).data;
        NDArray arr = new NDArray(data,new int[]{4});
        assertEquals(true,arr.isRowVector());
        NDArray arr2 = new NDArray(data,new int[]{1,4});
        assertEquals(true,arr2.isRowVector());

        NDArray columnVector = new NDArray(data,new int[]{4,1});
        assertEquals(true,columnVector.isColumnVector());
    }



    @Test
    public void testToArray() {
        DoubleMatrix d = DoubleMatrix.linspace(1,4,4).reshape(2,2);
        NDArray n = new NDArray(Arrays.copyOf(d.data,d.data.length),new int[]{2,2});
        n.toArray();
    }









    @Test
    public void testColumns() {
        NDArray arr = new NDArray(new int[]{3,2});
        NDArray column2 = arr.getColumn(0);
        assertEquals(true,Shape.shapeEquals(new int[]{3}, column2.shape()));
        NDArray column = new NDArray(new double[]{1,2,3},new int[]{3});
        arr.putColumn(0,column);

        NDArray firstColumn = arr.getColumn(0);

        assertEquals(column,firstColumn);


        NDArray column1 = new NDArray(new double[]{4,5,6},new int[]{3});
        arr.putColumn(1,column1);
        assertEquals(true, Shape.shapeEquals(new int[]{3}, arr.getColumn(1).shape()));
        NDArray testRow1 = arr.getColumn(1);
        assertEquals(column1,testRow1);



        NDArray evenArr = new NDArray(new double[]{1,2,3,4},new int[]{2,2});
        NDArray put = new NDArray(new double[]{5,6},new int[]{2});
        evenArr.putColumn(1,put);
        NDArray testColumn = evenArr.getColumn(1);
        assertEquals(put,testColumn);



        NDArray n = new NDArray(DoubleMatrix.linspace(1,4,4).data,new int[]{2,2});
        NDArray column23 = n.getColumn(0);
        NDArray column12 = new NDArray(new double[]{1,3},new int[]{2});
        assertEquals(column23,column12);


        NDArray column0 = n.getColumn(1);
        NDArray column01 = new NDArray(new double[]{2,4},new int[]{2});
        assertEquals(column0,column01);



    }


    @Test
    public void testPutRow() {
        DoubleMatrix d = DoubleMatrix.linspace(1,4,4).reshape(2,2);
        NDArray n = new NDArray(Arrays.copyOf(d.data,d.data.length),new int[]{2,2});

        //works fine according to matlab, let's go with it..
        //reproduce with:  A = reshape(linspace(1,4,4),[2 2 ]);
        //A(1,2) % 1 index based
        double nFirst = 3;
        double dFirst = d.get(0,1);
        assertEquals(nFirst,dFirst,1e-1);
        assertEquals(true,Arrays.equals(d.toArray(),n.toArray()));
        assertEquals(true,Arrays.equals(new int[]{2,2},n.shape()));

        DoubleMatrix newRow = DoubleMatrix.linspace(5,6,2);
        n.putRow(0,newRow);
        d.putRow(0,newRow);



        NDArray testRow = n.getRow(0);
        assertEquals(newRow.length,testRow.length);
        assertEquals(true,Shape.shapeEquals(new int[]{2},testRow.shape()));



        NDArray nLast = new NDArray(DoubleMatrix.linspace(1,4,4).data,new int[]{2,2});
        NDArray row = nLast.getRow(1);
        NDArray row1 = new NDArray(new double[]{3,4},new int[]{2});
        assertEquals(row,row1);



        NDArray arr = new NDArray(new int[]{3,2});
        NDArray evenRow = new NDArray(new double[]{1,2},new int[]{2});
        arr.putRow(0,evenRow);
        NDArray firstRow = arr.getRow(0);
        assertEquals(true, Shape.shapeEquals(new int[]{2},firstRow.shape()));
        NDArray testRowEven = arr.getRow(0);
        assertEquals(evenRow,testRowEven);


        NDArray row12 = new NDArray(new double[]{5,6},new int[]{2});
        arr.putRow(1,row12);
        assertEquals(true, Shape.shapeEquals(new int[]{2}, arr.getRow(0).shape()));
        NDArray testRow1 = arr.getRow(1);
        assertEquals(row12,testRow1);


        NDArray multiSliceTest = new NDArray(DoubleMatrix.linspace(1,16,16).data,new int[]{4,2,2});
        NDArray test = new NDArray(new double[]{7,8},new int[]{2});
        NDArray test2 = new NDArray(new double[]{9,10},new int[]{2});

        assertEquals(test,multiSliceTest.slice(1).getRow(1));
        assertEquals(test2,multiSliceTest.slice(1).getRow(2));

    }





    @Test
    public void testMmul() {
        double[] data = DoubleMatrix.linspace(1,10,10).data;
        NDArray n = new NDArray(data,new int[]{10});
        NDArray transposed = n.transpose();
        assertEquals(true,n.isRowVector());
        assertEquals(true,transposed.isColumnVector());

        DoubleMatrix d = new DoubleMatrix(n.rows(),n.columns());
        d.data = n.data;
        DoubleMatrix dTransposed = d.transpose();
        DoubleMatrix result2 = d.mmul(dTransposed);


        NDArray innerProduct = n.mmul(transposed);

        NDArray scalar = NDArray.scalar(385);
        assertEquals(scalar,innerProduct);

        NDArray outerProduct = transposed.mmul(n);
        assertEquals(true, Shape.shapeEquals(new int[]{10,10},outerProduct.shape()));


        NDArray testMatrix = new NDArray(data,new int[]{5,2});
        NDArray row1 = testMatrix.getRow(0).transpose();
        NDArray row2 = testMatrix.getRow(1);
        DoubleMatrix row12 = DoubleMatrix.linspace(1,2,2).reshape(2,1);
        DoubleMatrix row22 = DoubleMatrix.linspace(3,4,2).reshape(1,2);
        DoubleMatrix rowResult = row12.mmul(row22);

        NDArray row122 = NDArray.wrap(row12);
        NDArray row222 = NDArray.wrap(row22);
        NDArray rowResult2 = row122.mmul(row222);




        NDArray mmul = row1.mmul(row2);
        NDArray result = new NDArray(new double[]{3,6,4,8},new int[]{2,2});
        assertEquals(result,mmul);




        NDArray three = new NDArray(new double[]{3,4},new int[]{2});
        NDArray test = new NDArray(DoubleMatrix.linspace(1,30,30).data,new int[]{3,5,2});
        NDArray sliceRow = test.slice(0).getRow(1);
        assertEquals(three,sliceRow);

        NDArray twoSix = new NDArray(new double[]{2,6},new int[]{2,1});
        NDArray threeTwoSix = three.mmul(twoSix);

        NDArray sliceRowTwoSix = sliceRow.mmul(twoSix);

        assertEquals(threeTwoSix,sliceRowTwoSix);




    }

    @Test
    public void testRowsColumns() {
        double[] data = DoubleMatrix.linspace(1,6,6).data;
        NDArray rows = new NDArray(data,new int[]{2,3});
        assertEquals(2,rows.rows());
        assertEquals(3,rows.columns());

        NDArray columnVector = new NDArray(data,new int[]{6,1});
        assertEquals(6,columnVector.rows());
        assertEquals(1,columnVector.columns());
        NDArray rowVector = new NDArray(data,new int[]{6});
        assertEquals(1,rowVector.rows());
        assertEquals(6,rowVector.columns());
    }


    @Test
    public void testTranspose() {
        NDArray n = new NDArray(DoubleMatrix.ones(100).data,new int[]{5,5,4});
        NDArray transpose = n.transpose();
        assertEquals(n.length,transpose.length);
        assertEquals(true,Arrays.equals(new int[]{4,5,5},transpose.shape()));

        NDArray rowVector = NDArray.linspace(1,10,10);
        assertTrue(rowVector.isRowVector());
        NDArray columnVector = rowVector.transpose();
        assertTrue(columnVector.isColumnVector());

    }

    @Test
    public void testPutSlice() {
        NDArray n = new NDArray(DoubleMatrix.ones(27).data,new int[]{3,3,3});
        NDArray newSlice = NDArray.wrap(DoubleMatrix.zeros(3,3));
        n.putSlice(0,newSlice);
        assertEquals(newSlice,n.slice(0));


    }

    @Test
    public void testPermute() {
        NDArray n = new NDArray(DoubleMatrix.linspace(1,20,20).data,new int[]{5,4});
        NDArray transpose = n.transpose();
        NDArray permute = n.permute(new int[]{1,0});
        assertEquals(permute,transpose);
        assertEquals(transpose.length,permute.length,1e-1);


        NDArray toPermute = new NDArray(DoubleMatrix.linspace(0,7,8).data,new int[]{2,2,2});
        NDArray permuted = toPermute.permute(new int[]{2,1,0});
        NDArray assertion = new NDArray(new double[]{0,4,2,6,1,5,3,7},new int[]{2,2,2});
        assertEquals(permuted,assertion);

    }

    @Test
    public void testSlice() {
        assertEquals(8,n.length);
        assertEquals(true,Arrays.equals(new int[]{2,2,2},n.shape()));
        NDArray slice = n.slice(0);
        assertEquals(true, Arrays.equals(new int[]{2, 2}, slice.shape()));

        NDArray slice1 = n.slice(1);
        assertEquals(true,Arrays.equals(slice.shape(),slice1.shape()));
        assertNotEquals(true,Arrays.equals(slice.toArray(),slice1.toArray()));

        NDArray arr = new NDArray(DoubleMatrix.linspace(1,24,24).data,new int[]{4,3,2});
        NDArray slice0 = new NDArray(new double[]{1,2,3,4,5,6},new int[]{3,2});
        NDArray slice2 = new NDArray(new double[]{7,8,9,10,11,12},new int[]{3,2});

        NDArray testSlice0 = arr.slice(0);
        NDArray testSlice1 = arr.slice(1);

        assertEquals(slice0,testSlice0);
        assertEquals(slice2,testSlice1);







    }

    @Test
    public void testSwapAxes() {
        NDArray n = new NDArray(DoubleMatrix.linspace(0,7,8).data,new int[]{2,2,2});
        NDArray assertion = n.permute(new int[]{2,1,0});
        double[] data = assertion.data();
        NDArray validate = new NDArray(new double[]{0,4,2,6,1,5,3,7},new int[]{2,2,2});
        assertEquals(validate,assertion);



    }



    @Test
    public void testSliceConstructor() {
        List<NDArray> testList = new ArrayList<>();
        for(int i = 0; i < 5; i++)
            testList.add(NDArray.scalar(i + 1));

        NDArray test = new NDArray(testList,new int[]{testList.size()});
        NDArray expected = new NDArray(new double[]{1,2,3,4,5},new int[]{5});
        assertEquals(expected,test);
    }


    @Test
    public void testVectorDimension() {
        NDArray test = new NDArray(DoubleMatrix.linspace(1,4,4).data,new int[]{2,2});
        final AtomicInteger count = new AtomicInteger(0);
        //row wise
        test.iterateOverDimension(1,new SliceOp() {
            @Override
            public void operate(DimensionSlice nd) {
                log.info("Operator " + nd);
                NDArray test = (NDArray) nd.getResult();
                if(count.get() == 0) {
                    NDArray firstDimension = new NDArray(new double[]{1,2},new int[]{2});
                    assertEquals(firstDimension,test);
                }
                else {
                    NDArray firstDimension = new NDArray(new double[]{3,4},new int[]{2});
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
                NDArray test = (NDArray) nd.getResult();
                if(count.get() == 0) {
                    NDArray firstDimension = new NDArray(new double[]{1,3},new int[]{2});
                    assertEquals(firstDimension,test);
                }
                else {
                    NDArray firstDimension = new NDArray(new double[]{2,4},new int[]{2});
                    assertEquals(firstDimension,test);

                }

                count.incrementAndGet();
            }

        },false);




    }


    @Test
    public void testReshape() {
        NDArray arr = new NDArray(DoubleMatrix.linspace(1,24,24).data,new int[]{4,3,2});
        NDArray reshaped = arr.reshape(new int[]{2,3,4});
        assertEquals(arr.length,reshaped.length);
        assertEquals(true,Arrays.equals(new int[]{4,3,2},arr.shape()));
        assertEquals(true,Arrays.equals(new int[]{2,3,4},reshaped.shape()));
    }


    @Test
    public void reduceTest() {
        NDArray arr = new NDArray(DoubleMatrix.linspace(1,24,24).data,new int[]{4,3,2});
        NDArray reduced = arr.reduce(NDArrayUtil.DimensionOp.MAX,1);
        log.info("Reduced " + reduced);
        reduced = arr.reduce(NDArrayUtil.DimensionOp.MAX,1);
        log.info("Reduced " + reduced);
        reduced = arr.reduce(NDArrayUtil.DimensionOp.MAX,2);
        log.info("Reduced " + reduced);


    }


    @Test
    public void testGetMulti() {
        assertEquals(8,n.length);
        assertEquals(true,Arrays.equals(ArrayUtil.of(2,2,2),n.shape()));
        double val = n.getMulti(1,1,1);
        assertEquals(8.0,val,1e-6);
    }


    @Test
    public void testShapeFor() {
        //real shape
        Integer[] realShape = {3,2,1};
        //test query
        Integer[] test = {1,2,3};
        Integer[] testShape =  NDArray.shapeForObject(realShape, test);
        assertEquals(true,Arrays.equals(realShape,testShape));
        Object[] nextTest = {':',':',3};
        assertEquals(true,Arrays.equals(realShape,NDArray.shapeForObject(realShape, nextTest)));
        Object[] nextTest2 = {':',':',':'};
        assertEquals(true,Arrays.equals(realShape,NDArray.shapeForObject(realShape, nextTest2)));

        //the subset query is the shape of the solution
        Integer[] subSetTest = {1,1,1};
        assertEquals(true,Arrays.equals(subSetTest,NDArray.shapeForObject(realShape, subSetTest)));

        Integer[] singular = {1,1};
        assertEquals(true,Arrays.equals(singular,NDArray.shapeForObject(realShape,singular)));


        Integer[] zeroTest = {0,1,1};
        Integer[] zeroTestResult = {1,1};
        assertEquals(true,Arrays.equals(zeroTestResult,NDArray.shapeForObject(realShape,zeroTest)));



    }

    @Test
    public void testElementWiseOps() {
        NDArray n1 = NDArray.scalar(1);
        NDArray n2 = NDArray.scalar(2);
        assertEquals(NDArray.scalar(3),n1.add(n2));
        assertFalse(n1.add(n2).equals(n1));

        NDArray n3 = NDArray.scalar(3);
        NDArray n4 = NDArray.scalar(4);
        NDArray subbed = n4.sub(n3);
        NDArray mulled = n4.mul(n3);
        NDArray div = n4.div(n3);

        assertFalse(subbed.equals(n4));
        assertFalse(mulled.equals(n4));
        assertEquals(NDArray.scalar(1),subbed);
        assertEquals(NDArray.scalar(12),mulled);
        assertEquals(NDArray.scalar(1),div);
    }




    @Test
    public void testGetVaried() {

        int[] shape = NDArray.shapeFor(n.shape(),new Object[]{':',':',0},true);
        assertEquals(true,Arrays.equals(new int[]{2,2},shape));

        NDArray n3 = n.slice(0,0);
        assertEquals(true,Arrays.equals(new int[]{2,2},n3.shape()));

    }


    @Test
    public void testSlicing() {
        NDArray arr = n.slice(1, 1);
        // assertEquals(1,arr.shape().length);
        NDArray n2 = new NDArray(DoubleMatrix.linspace(1,16,16).data,new int[]{2,2,2,2});
        log.info("N2 shape " + n2.slice(1,1).slice(1));

    }


    @Test
    public void testEndsForSlices() {
        NDArray arr = new NDArray(DoubleMatrix.linspace(1,24,24).data,new int[]{4,3,2});
        int[] endsForSlices = arr.endsForSlices();
        assertEquals(true,Arrays.equals(new int[]{5,11,17,23},endsForSlices));
    }


    @Test
    public void testFlatten() {
        NDArray arr = new NDArray(DoubleMatrix.linspace(1,4,4).data,new int[]{2,2});
        NDArray flattened = arr.flatten();
        assertEquals(arr.length,flattened.length);
        assertEquals(true,Shape.shapeEquals(new int[]{1, arr.length}, flattened.shape()));
        for(int i = 0; i < arr.length; i++) {
            assertEquals(i + 1,flattened.get(i),1e-1);
        }
        assertTrue(flattened.isVector());


        NDArray n = new NDArray(DoubleMatrix.ones(27).data,new int[]{3,3,3});
        NDArray nFlattened = n.flatten();
        assertTrue(nFlattened.isVector());

    }

    @Test
    public void testVectorDimensionMulti() {
        NDArray arr = new NDArray(DoubleMatrix.linspace(1,24,24).data,new int[]{4,3,2});
        final AtomicInteger count = new AtomicInteger(0);

        arr.iterateOverDimension(0,new SliceOp() {
            @Override
            public void operate(DimensionSlice nd) {
                NDArray test =(NDArray) nd.getResult();
                if(count.get() == 0) {
                    NDArray answer = new NDArray(new double[]{1,7,13,19},new int[]{4});
                    assertEquals(answer,test);
                }
                else if(count.get() == 1) {
                    NDArray answer = new NDArray(new double[]{2,8,14,20},new int[]{4});
                    assertEquals(answer,test);
                }
                else if(count.get() == 2) {
                    NDArray answer = new NDArray(new double[]{3,9,15,21},new int[]{4});
                    assertEquals(answer,test);
                }
                else if(count.get() == 3) {
                    NDArray answer = new NDArray(new double[]{4,10,16,22},new int[]{4});
                    assertEquals(answer,test);
                }
                else if(count.get() == 4) {
                    NDArray answer = new NDArray(new double[]{5,11,17,23},new int[]{4});
                    assertEquals(answer,test);
                }
                else if(count.get() == 5) {
                    NDArray answer = new NDArray(new double[]{6,12,18,24},new int[]{4});
                    assertEquals(answer,test);
                }


                count.incrementAndGet();
            }
        },false);
    }

}
