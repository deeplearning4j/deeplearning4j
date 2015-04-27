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

package org.nd4j.linalg.learning;


import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class AdaGradTest {

    private static Logger log = LoggerFactory.getLogger(AdaGradTest.class);


    @Test
    public void testAdaGrad1() {
        int rows = 1;
        int cols = 1;


        AdaGrad grad = new AdaGrad(rows, cols, 1e-3);
        INDArray W = Nd4j.ones(rows, cols);

        log.info("Learning rates for 1 " + grad.getGradient(W));


    }

    @Test
    public void testAdaGrad() {
        int rows = 10;
        int cols = 2;


        AdaGrad grad = new AdaGrad(rows, cols, 0.1);
        INDArray W = Nd4j.zeros(rows, cols);

    }
}
