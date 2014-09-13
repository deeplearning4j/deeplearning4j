package org.nd4j.linalg.jcublas.complex;

import org.junit.Test;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;



/**
 * Tests for a complex ndarray
 *
 * @author Adam Gibson
 */
public class JCublasComplexNDArrayTests extends org.nd4j.linalg.api.test.ComplexNDArrayTests {

    private static Logger log = LoggerFactory.getLogger(JCublasComplexNDArrayTests.class);

    @Test
    public void testAlloc() {
        JCublasComplexNDArray n = (JCublasComplexNDArray) Nd4j.createComplex(new float[]{1,2,3,4});
        n.allocTest();
        n.free();

        float[] data = new float[4];
        n.alloc();
        n.getData(data);
        n.free();

    }






}
