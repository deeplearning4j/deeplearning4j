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
    public void testPutAndGet() {
        ComplexNDArray arr = new ComplexNDArray(new double[]{0,1,2,1,1,2,3,4},new int[]{2,2});
        assertEquals(4,arr.length);
        assertEquals(8,arr.data.length);
        arr.put(1,1,1.0);
        assertEquals(1.0,arr.get(1,1).real(),1e-1);
        assertEquals(4.0,arr.get(1,1).imag(),1e-1);

    }

    @Test
    public void testVectorDimension() {
        ComplexNDArray test = new ComplexNDArray(DoubleMatrix.linspace(1,24,24).data,new int[]{4,3,2});
        ComplexNDArray dimension = test.vectorForDimensionAndOffset(1,1);
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
