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

package org.nd4j.linalg.factory;

import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

import static org.junit.Assert.assertEquals;


/**
 * Created by agibsonccc on 9/11/14.
 */
public abstract class BlasWrapperTests {

    @Test
    public void axpyTest() {
        Nd4j.dtype = DataBuffer.DOUBLE;
        INDArray a = Nd4j.getBlasWrapper().axpy(1.0, Nd4j.ones(3), Nd4j.ones(3));
        INDArray a2 = Nd4j.create(new double[]{2, 2, 2});
        assertEquals(a2, a);

        INDArray matrix = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray row = matrix.getRow(1);
        INDArray result = Nd4j.create(new double[]{1, 2});
        Nd4j.getBlasWrapper().axpy(1.0, row, result);
        assertEquals(Nd4j.create(new double[]{3, 6}), result);


    }

    @Test
    public void testAxpyFortran() {
        Nd4j.factory().setOrder('f');
        INDArray threeByFour = Nd4j.linspace(1, 12, 12).reshape(3, 4);
        INDArray row = threeByFour.getRow(1);
        Nd4j.getBlasWrapper().axpy(2.0, row, row);
    }


    @Test
    public void testIaMax() {
        INDArray test = Nd4j.create(new float[]{1, 2, 3, 4});
        test.toString();
        int max = Nd4j.getBlasWrapper().iamax(test);
        assertEquals(3, max);


        INDArray rows = Nd4j.create(new float[]{1, 3, 2, 4}, new int[]{2, 2});
        for (int i = 0; i < rows.rows(); i++) {
            INDArray row = rows.getRow(i);
            int max2 = Nd4j.getBlasWrapper().iamax(row);
            assertEquals(1, max2);
        }

    }

}
