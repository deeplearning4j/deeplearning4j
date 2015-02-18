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

package org.nd4j.linalg.api.activation;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertEquals;

/**
 * Test for softmax function
 *
 * @author Adam Gibson
 */
public abstract class SoftMaxTest {

    private static Logger log = LoggerFactory.getLogger(SoftMaxTest.class);

    @Test
    public void testSoftMax() {
        Nd4j.factory().setOrder('f');
        INDArray test = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        INDArray softMaxColumns = Activations.softmax().apply(test);
        INDArray softMaxRows = Activations.softMaxRows().apply(test);
        INDArray columns = softMaxColumns.sum(0);
        INDArray rows = softMaxRows.sum(1);
        //softmax along columns: should be 1 in every cell ( note that there are 3 columns)
        assertEquals(3, columns.sum(Integer.MAX_VALUE).getFloat(0), 1e-1);
        //softmax along rows: should be 1 in every cell (note that there are 2 rows
        assertEquals(2, rows.sum(Integer.MAX_VALUE).getFloat(0), 1e-1);

    }

    @Test
    public void testSoftMaxCOrder() {
        Nd4j.factory().setOrder('c');
        INDArray test = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        INDArray softMaxColumns = Activations.softmax().apply(test);
        INDArray softMaxRows = Activations.softMaxRows().apply(test);

        INDArray columns = softMaxColumns.sum(0);
        INDArray rows = softMaxRows.sum(1);
        //softmax along columns: should be 1 in every cell ( note that there are 3 columns)
        assertEquals(3, columns.sum(Integer.MAX_VALUE).getFloat(0), 1e-1);
        //softmax along rows: should be 1 in every cell (note that there are 2 rows
        assertEquals(2, rows.sum(Integer.MAX_VALUE).getFloat(0), 1e-1);

    }

}
