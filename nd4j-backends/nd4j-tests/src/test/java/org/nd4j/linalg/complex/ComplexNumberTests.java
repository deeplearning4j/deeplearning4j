/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */


package org.nd4j.linalg.complex;

import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertEquals;

/**
 * Tests for complex numbers
 *
 * @author Adam Gibson
 */
@Ignore
@RunWith(Parameterized.class)
public class ComplexNumberTests extends BaseNd4jTest {
    public ComplexNumberTests(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testScalar() {
        IComplexDouble test = Nd4j.createDouble(1, 1);
        test.addi(1);
        assertEquals(2, test.realComponent().doubleValue(), 1e-1);
        assertEquals(1, test.imaginaryComponent(), 1e-1);
        test.subi(1);
        assertEquals(1, test.realComponent().doubleValue(), 1e-1);
        assertEquals(getFailureMessage(), 1, test.imaginaryComponent(), 1e-1);
        test.muli(2);
        assertEquals(2, test.realComponent().doubleValue(), 1e-1);
        assertEquals(2, test.imaginaryComponent(), 1e-1);
        test.divi(2);
        assertEquals(1, test.realComponent().doubleValue(), 1e-1);
        assertEquals(1, test.imaginaryComponent(), 1e-1);
        test.addi(Nd4j.createDouble(1, 1));
        assertEquals(2, test.realComponent().doubleValue(), 1e-1);
        assertEquals(2, test.imaginaryComponent(), 1e-1);
        test.rdivi(1);
        assertEquals(0.5d, test.realComponent().doubleValue(), 1e-1);
        assertEquals(2.0d, test.imaginaryComponent(), 1e-1);
    }


    @Test
    public void testScalarFloat() {
        IComplexFloat test = Nd4j.createFloat(1, 1);
        test.addi(1);
        assertEquals(2, test.realComponent().floatValue(), 1e-1);
        assertEquals(1, test.imaginaryComponent(), 1e-1);
        test.subi(1);
        assertEquals(1, test.realComponent().floatValue(), 1e-1);
        assertEquals(getFailureMessage(), 1, test.imaginaryComponent(), 1e-1);
        test.muli(2);
        assertEquals(2, test.realComponent().floatValue(), 1e-1);
        assertEquals(2, test.imaginaryComponent(), 1e-1);
        test.divi(2);
        assertEquals(1, test.realComponent().floatValue(), 1e-1);
        assertEquals(1, test.imaginaryComponent(), 1e-1);
        test.addi(Nd4j.createDouble(1, 1));
        assertEquals(2, test.realComponent().floatValue(), 1e-1);
        assertEquals(2, test.imaginaryComponent(), 1e-1);
        test.rdivi(1);
        assertEquals(0.25d, test.realComponent().floatValue(), 1e-1);
        assertEquals(-0.25d, test.imaginaryComponent(), 1e-1);
    }

    @Test
    public void testExponentFloat() {
        IComplexFloat test = Nd4j.createFloat(1, 1);
        assertEquals(test.realComponent(), 1.468694, 1e-3);
        assertEquals(test.imaginaryComponent(), 2.2873552, 1e-3);
    }

    @Test
    public void testExponentDouble() {
        IComplexDouble test = Nd4j.createDouble(1, 1);
        assertEquals(test.realComponent(), 1.4686939399158851, 1e-3);
        assertEquals(test.imaginaryComponent(), 2.2873552871788423, 1e-3);
    }

    @Test
    public void testPowerDouble() {
        IComplexDouble test = Nd4j.createDouble(1, 1);
        IComplexDouble test2 = Nd4j.createDouble(1, 1);
        IComplexNumber result = test.pow(test2);
        assertEquals(result.realComponent(), 0.273957253830121);
        assertEquals(result.imaginaryComponent(), 0.5837007587586147);
    }

    @Test
    public void testPowerFloat() {
        IComplexDouble test = Nd4j.createDouble(1, 1);
        IComplexDouble test2 = Nd4j.createDouble(1, 1);
        IComplexNumber result = test.pow(test2);
        assertEquals(result.realComponent(), 0.2739572);
        assertEquals(result.imaginaryComponent(), 0.583700);
    }

    @Test
    public void testLogarithmFloat() {
        IComplexDouble test = Nd4j.createDouble(1, 1);
        IComplexDouble test2 = Nd4j.createDouble(1, 1);
        IComplexNumber result = test.pow(test2);
        assertEquals(result.realComponent(), 0.3465736);
        assertEquals(result.imaginaryComponent(), 0.7853982);
    }

    @Test
    public void testLogarithmDouble() {
        IComplexDouble test = Nd4j.createDouble(1, 1);
        IComplexDouble test2 = Nd4j.createDouble(1, 1);
        IComplexNumber result = test.pow(test2);
        assertEquals(result.realComponent(), 0.3465735902799727);
        assertEquals(result.imaginaryComponent(), 0.7853981633974483);
    }


    @Override
    public char ordering() {
        return 'f';
    }
}
