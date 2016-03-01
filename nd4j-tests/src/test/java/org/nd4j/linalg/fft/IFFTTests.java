/*
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

package org.nd4j.linalg.fft;

import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;


import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Tests for ifft
 *
 * @author Adam Gibson
 */
@Ignore
@RunWith(Parameterized.class)
public  class IFFTTests extends BaseNd4jTest {

    public IFFTTests(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testIfft() {
        Nd4j.EPS_THRESHOLD = 1e-1;
        double[] ffted = {10.2, 5., -3.0, -1.};
        double[] orig = {3.5999999999999996, 2, 6.5999999999999996, 3};
        IComplexNDArray c = Nd4j.createComplex(orig, new int[]{1,2});
        IComplexNDArray assertion = Nd4j.createComplex(ffted, new int[]{1,2});

        assertEquals(getFailureMessage(),assertion, Nd4j.getFFt().fft(c.dup(), 2));
        IComplexNDArray iffted =  Nd4j.getFFt().ifft(Nd4j.getFFt().fft(c.dup(), 2),2);
        assertEquals(getFailureMessage(),iffted, c);


    }



    @Override
    public char ordering() {
        return 'f';
    }
}
