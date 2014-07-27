package org.deeplearning4j.nn.linalg;

import static org.junit.Assert.*;

import org.deeplearning4j.util.ArrayUtil;
import org.deeplearning4j.util.NDArrayUtil;
import org.jblas.DoubleMatrix;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * NDArrayTests
 * @author Adam Gibson
 */
public class NDArrayTests {
    private static Logger log = LoggerFactory.getLogger(NDArrayTests.class);
    private NDArray n = new NDArray(DoubleMatrix.linspace(1,8,8).data,new int[]{2,2,2});



    @Test
    public void testBasicOps() {
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
        assertEquals(true,Arrays.equals(new int[]{1},a.shape()));
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

        DoubleMatrix newRow = DoubleMatrix.linspace(1,2,2);
        n.putRow(0,newRow);
        d.putRow(0,newRow);



        NDArray testRow = n.getRow(0);
        assertEquals(newRow.length,testRow.length);
        assertEquals(true,Arrays.equals(new int[]{2},testRow.shape()));



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


    }


    @Test
    public void testMatrixMultiply() {
        DoubleMatrix d = DoubleMatrix.linspace(1,2,4).reshape(2,2);
        NDArray n = new NDArray(Arrays.copyOf(d.data,d.data.length),new int[]{2,2});

        DoubleMatrix d2 = DoubleMatrix.linspace(1,2,4).reshape(2,2);
        NDArray n2 = new NDArray(Arrays.copyOf(d.data,d.data.length),new int[]{2,2});
        n.mmul(n2);

    }



    @Test
    public void testTranspose() {
        NDArray n = new NDArray(DoubleMatrix.ones(100).data,new int[]{5,5,4});
        NDArray transpose = n.transpose();
        assertEquals(n.length,transpose.length);
        assertEquals(true,Arrays.equals(new int[]{4,5,5},transpose.shape()));

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
        NDArray n = new NDArray(DoubleMatrix.rand(20).data,new int[]{5,4});
        NDArray transpose = n.transpose();
        NDArray permute = n.permute(new int[]{1,0});
        assertEquals(permute,transpose);
        assertEquals(transpose.length,permute.length,1e-1);
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

        });



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

        });




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
        });
    }

}
