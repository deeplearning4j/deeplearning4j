package org.deeplearning4j.nn.linalg;

import static org.junit.Assert.*;

import org.deeplearning4j.util.ComplexNDArrayUtil;
import org.jblas.ComplexDoubleMatrix;
import org.jblas.DoubleMatrix;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

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
                    ComplexNDArray firstDimension = new ComplexNDArray(new double[]{1,0,2,0},new int[]{2});
                    assertEquals(firstDimension,test);
                }
                else {
                    ComplexNDArray firstDimension = new ComplexNDArray(new double[]{3,0,4,0},new int[]{2});
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

        });




    }


    @Test
    public void testEndsForSlices() {
        ComplexNDArray arr = new ComplexNDArray(new NDArray(DoubleMatrix.linspace(1,24,24).data,new int[]{4,3,2}));
        int[] endsForSlices = arr.endsForSlices();
        assertEquals(true, Arrays.equals(new int[]{0, 12, 24, 36}, endsForSlices));
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
        });



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
        });
    }



}
