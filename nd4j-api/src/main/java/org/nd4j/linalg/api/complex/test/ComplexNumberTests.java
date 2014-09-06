package org.nd4j.linalg.api.complex.test;

import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Tests for complex numbers
 *
 * @author Adam Gibson
 */
public abstract class ComplexNumberTests {
    @Test
    public void testScalar() {
        IComplexDouble test = Nd4j.createDouble(1,1);
        test.addi(1);
        assertEquals(2,test.realComponent().doubleValue(),1e-1);
        assertEquals(2,test.imaginaryComponent(),1e-1);
        test.subi(1);
        assertEquals(1,test.realComponent().doubleValue(),1e-1);
        assertEquals(1,test.imaginaryComponent(),1e-1);
        test.muli(2);
        assertEquals(2,test.realComponent().doubleValue(),1e-1);
        assertEquals(2,test.imaginaryComponent(),1e-1);
        test.divi(2);
        assertEquals(1,test.realComponent().doubleValue(),1e-1);
        assertEquals(1,test.imaginaryComponent(),1e-1);


    }


    @Test
    public void testScalarFloat() {
        IComplexFloat test = Nd4j.createFloat(1, 1);
        test.addi(1);
        assertEquals(2,test.realComponent().doubleValue(),1e-1);
        assertEquals(2,test.imaginaryComponent(),1e-1);
        test.subi(1);
        assertEquals(1,test.realComponent().doubleValue(),1e-1);
        assertEquals(1,test.imaginaryComponent(),1e-1);
        test.muli(2);
        assertEquals(2,test.realComponent().doubleValue(),1e-1);
        assertEquals(2,test.imaginaryComponent(),1e-1);
        test.divi(2);
        assertEquals(1,test.realComponent().doubleValue(),1e-1);
        assertEquals(1,test.imaginaryComponent(),1e-1);


    }

}
