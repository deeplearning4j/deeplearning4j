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

package org.nd4j.linalg.jblas;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * NDArray creation tests
 *
 * @author Adam Gibson
 */
public class TestNDArrayCreation {

    private static Logger log = LoggerFactory.getLogger(TestNDArrayCreation.class);

    @Test
    public void testCreation() {
        INDArray arr = Nd4j.create(1, 1);
        assertTrue(arr.isScalar());

        INDArray arr2 = Nd4j.scalar(0d, 0);
        assertEquals(arr, arr2);
        arr = Nd4j.create(1, 1);

    }


}
