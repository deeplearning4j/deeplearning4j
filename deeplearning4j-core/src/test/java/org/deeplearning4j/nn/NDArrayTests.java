package org.deeplearning4j.nn;

import static org.junit.Assert.*;

import org.deeplearning4j.util.ArrayUtil;
import org.jblas.DoubleMatrix;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * NDArrayTests
 * @author Adam Gibson
 */
public class NDArrayTests {
    private static Logger log = LoggerFactory.getLogger(NDArrayTests.class);
    private  NDArray n = new NDArray(DoubleMatrix.linspace(1,8,8).data,new int[]{2,2,2});
    ;
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
    public void testRow() {
        int[] shape = {2,4};
        DoubleMatrix d = DoubleMatrix.linspace(1,8,8).reshape(shape[0],shape[1]);
        NDArray n = NDArray.wrap(d);
        assertEquals(n,d);
        assertEquals(true,Arrays.equals(shape,n.shape()));
        DoubleMatrix r1 = d.getRow(0);
        NDArray r2 = n.getRow(0);

        assertEquals(r2,r1);


        for(int i = 0; i < n.rows(); i++) {
            for(int j = 0; j < n.columns(); j++) {
                double val = d.get(i,j);
                double val2 = n.get(i,j);
                assertEquals(val,val2,1e-1);
            }
        }


    }

    @Test
    public void testColumn() {
        int[] shape = {2,4};
        DoubleMatrix d = DoubleMatrix.linspace(1,8,8).reshape(shape[0],shape[1]);
        NDArray n = new NDArray(Arrays.copyOf(d.data,d.data.length),shape);
        assertEquals(true,Arrays.equals(d.data,n.data));
        assertEquals(true,Arrays.equals(shape,n.shape()));
        DoubleMatrix r1 = d.getColumn(0);
        NDArray r2 = n.getColumn(0);

         //note that when comparing NDArrays and DoubleMatrix, you need to use the NDArray equals() method
        assertEquals(r2,r1);

        DoubleMatrix r12 = d.getColumn(1);
        NDArray r22 = n.getColumn(1);
        assertEquals(r22,r12);



    }

    @Test
    public void testToArray() {
        DoubleMatrix d = DoubleMatrix.linspace(1,4,4).reshape(2,2);
        NDArray n = new NDArray(Arrays.copyOf(d.data,d.data.length),new int[]{2,2});
        n.toArray();
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

        DoubleMatrix newRow = DoubleMatrix.rand(1,2);
        n.putRow(0,newRow);
        d.putRow(0,newRow);



        NDArray testRow = n.getRow(0);
        assertEquals(newRow.length,testRow.length);
        assertEquals(true,Arrays.equals(new int[]{2},testRow.shape()));

        assertEquals(newRow,d.getRow(0));


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
    public void testSliceWiseAggregateStats() {
        DoubleMatrix d = DoubleMatrix.linspace(1,4,4).reshape(2,2);
        DoubleMatrix columnMaxs = d.columnMaxs();
        NDArray n = NDArray.wrap(d);
        NDArray nColumnMaxes = n.columnMaxs();
        assertEquals(nColumnMaxes,columnMaxs);
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
    public void testGetMulti() {
        assertEquals(8,n.length);
        assertEquals(true,Arrays.equals(new int[]{2,2,2},n.shape()));
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

}
