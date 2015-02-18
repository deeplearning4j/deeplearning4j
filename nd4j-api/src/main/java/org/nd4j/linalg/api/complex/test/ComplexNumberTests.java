/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.api.complex.test;

import org.junit.Test;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * Tests for complex numbers
 *
 * @author Adam Gibson
 */
public abstract class ComplexNumberTests {
    @Test
    public void testScalar() {
        IComplexDouble test = Nd4j.createDouble(1, 1);
        test.addi(1);
        assertEquals(2, test.realComponent().doubleValue(), 1e-1);
        assertEquals(2, test.imaginaryComponent(), 1e-1);
        test.subi(1);
        assertEquals(1, test.realComponent().doubleValue(), 1e-1);
        assertEquals(1, test.imaginaryComponent(), 1e-1);
        test.muli(2);
        assertEquals(2, test.realComponent().doubleValue(), 1e-1);
        assertEquals(2, test.imaginaryComponent(), 1e-1);
        test.divi(2);
        assertEquals(1, test.realComponent().doubleValue(), 1e-1);
        assertEquals(1, test.imaginaryComponent(), 1e-1);


    }


    @Test
    public void testScalarFloat() {
        IComplexFloat test = Nd4j.createFloat(1, 1);
        test.addi(1);
        assertEquals(2, test.realComponent().doubleValue(), 1e-1);
        assertEquals(2, test.imaginaryComponent(), 1e-1);
        test.subi(1);
        assertEquals(1, test.realComponent().doubleValue(), 1e-1);
        assertEquals(1, test.imaginaryComponent(), 1e-1);
        test.muli(2);
        assertEquals(2, test.realComponent().doubleValue(), 1e-1);
        assertEquals(2, test.imaginaryComponent(), 1e-1);
        test.divi(2);
        assertEquals(1, test.realComponent().doubleValue(), 1e-1);
        assertEquals(1, test.imaginaryComponent(), 1e-1);


    }

    @Test
    public void testComplexComplexOperations() {
        IComplexDouble d = Nd4j.createDouble(2, 3);
        IComplexDouble d2 = Nd4j.createDouble(4, 5);
        IComplexDouble d3 = d.mul(d2).asDouble();


    }


}
