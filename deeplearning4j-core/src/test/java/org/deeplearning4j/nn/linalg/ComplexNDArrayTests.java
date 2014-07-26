package org.deeplearning4j.nn.linalg;

import static org.junit.Assert.*;

import org.jblas.DoubleMatrix;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Tests for a complex ndarray
 */
public class ComplexNDArrayTests {

    private static Logger log = LoggerFactory.getLogger(ComplexNDArrayTests.class);

    @Test
    public void testConstruction() {
        ComplexNDArray arr = new ComplexNDArray(new double[]{0,1},new int[]{1});
        //only each complex double: one element
        assertEquals(1,arr.length);
        //both real and imaginary components
        assertEquals(2,arr.data.length);
        assertEquals(0,arr.get(0).real(),1e-1);


    }


    @Test
    public void testSlice() {
        NDArray arr = new NDArray(DoubleMatrix.linspace(1,24,24).data,new int[]{4,3,2});
        ComplexNDArray arr2 = new ComplexNDArray(arr);
        assertEquals(arr,arr2.getReal());

        NDArray firstSlice = arr.slice(0);
        NDArray firstSliceTest = arr2.slice(0).getReal();
        assertEquals(firstSlice,firstSliceTest);

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


        ComplexNDArray row1 = new ComplexNDArray(new double[]{1,0,3,0},new int[]{2});
        arr.putRow(1,row1);
        assertEquals(true, Shape.shapeEquals(new int[]{2}, arr.getRow(0).shape()));
        ComplexNDArray testRow1 = arr.getRow(1);
        assertEquals(row1,testRow1);

    }

    @Test
    public void testGetColumn() {
        ComplexNDArray arr = new ComplexNDArray(new int[]{3,2});
        ComplexNDArray column2 = arr.getColumn(0);
        assertEquals(true,Shape.shapeEquals(new int[]{3}, column2.shape()));
        ComplexNDArray column = new ComplexNDArray(new double[]{1,0,2,0,3,0},new int[]{3});
        arr.putColumn(0,column);

        ComplexNDArray firstColumn = arr.getColumn(0);

        assertEquals(column,firstColumn);


        ComplexNDArray column1 = new ComplexNDArray(new double[]{4,0,5,0,6,0},new int[]{3});
        arr.putColumn(1,column1);
        assertEquals(true, Shape.shapeEquals(new int[]{3}, arr.getColumn(1).shape()));
        ComplexNDArray testC = arr.getColumn(1);
        assertEquals(column1,testC);



    }


    @Test
    public void testPutAndGet() {
        ComplexNDArray arr = new ComplexNDArray(new double[]{0,1,2,1,1,2,3,4},new int[]{2,2});
        assertEquals(4,arr.length);
        assertEquals(8,arr.data.length);
        arr.put(1,1,1.0);
        assertEquals(1.0,arr.get(1,1).real(),1e-1);
        assertEquals(4.0,arr.get(1,1).imag(),1e-1);

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
    public void testVectorDimension() {
        ComplexNDArray test = new ComplexNDArray(DoubleMatrix.linspace(1,24,24).data,new int[]{4,3,2});
        ComplexNDArray dimension = (ComplexNDArray) test.vectorForDimensionAndOffset(1,1).getResult();
        log.info("Dimension " + dimension);
    }


    @Test
    public void testBasicOperations() {
        ComplexNDArray arr = new ComplexNDArray(new double[]{0,1,2,1,1,2,3,4},new int[]{2,2});
        double sum = arr.sum().real();
        assertEquals(4,sum,1e-1);
        log.info("Sum " + sum);
        arr.addi(1);
        sum = arr.sum().real();
        assertEquals(6,sum,1e-1);
        log.info("Sum " + sum);
        arr.subi(1);
        sum = arr.sum().real();
        assertEquals(2,sum,1e-1);
    }




}
