package org.nd4j.linalg.jblas.complex;

import static org.junit.Assert.*;
import static org.junit.Assert.assertEquals;

import org.jblas.ComplexDouble;
import org.junit.Test;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by agibsonccc on 9/5/14.
 */
public class ComplexNumberTests extends org.nd4j.linalg.api.complex.test.ComplexNumberTests {


    @Test
    public void testComplexComplexOperations() {
        IComplexDouble d =  Nd4j.createDouble(2, 3);
        IComplexDouble d2 = Nd4j.createDouble(4,5);
        IComplexDouble d3 = d.mul(d2).asDouble();

        org.jblas.ComplexDouble d4 = new ComplexDouble(2,3);
        org.jblas.ComplexDouble d5 = new org.jblas.ComplexDouble(4,5);
        ComplexDouble d6 = d4.mul(d5);
        assertEquals(d3.realComponent().doubleValue(),d6.real(),1e-1);
        assertEquals(d3.imaginaryComponent().doubleValue(), d6.imag(), 1e-1);


        IComplexDouble d7 = d.mul(d2).asDouble();
        org.jblas.ComplexDouble d8 = d4.mul(d5);
        assertComponents(d7,d8);


        IComplexDouble d9 = d.add(d2).asDouble();
        org.jblas.ComplexDouble d10 = d4.add(d5);
        assertComponents(d9,d10);

        IComplexDouble d11 = d.sub(d2).asDouble();
        org.jblas.ComplexDouble d12 = d4.sub(d5);
        assertComponents(d11,d12);


    }


    private void assertComponents(IComplexNumber d3,ComplexDouble d6) {
        assertEquals(d3.realComponent().doubleValue(),d6.real(),1e-1);
        assertEquals(d3.imaginaryComponent().doubleValue(), d6.imag(), 1e-1);

    }

}
