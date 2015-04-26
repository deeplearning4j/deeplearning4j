package org.nd4j.linalg.jblas.util;

import org.junit.Test;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ComplexNDArrayUtil;

import static org.junit.Assert.*;

/**
 * Created by agibsoncccc on 4/25/15.
 */
public class ComplexNDArrayUtilTest {

    @Test
    public void testTruncate() {
        IComplexNDArray truncate = Nd4j.createComplex(5,5);
        IComplexNDArray truncated = ComplexNDArrayUtil.truncate(truncate, 3, 0);
        IComplexNDArray assertion = Nd4j.createComplex(3,5);
        assertEquals(assertion,truncated);
    }

}
