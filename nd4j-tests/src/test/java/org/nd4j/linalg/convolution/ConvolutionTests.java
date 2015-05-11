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

package org.nd4j.linalg.convolution;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 9/6/14.
 */
public  class ConvolutionTests extends BaseNd4jTest {
    public ConvolutionTests(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    public ConvolutionTests(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void convNTest() {
        Nd4j.EPS_THRESHOLD = 1e-1;
        INDArray arr = Nd4j.linspace(1, 8, 8);
        INDArray kernel = Nd4j.linspace(1, 3, 3);
            INDArray answer = Nd4j.create(new double[]{1.0000012});
        INDArray test = Convolution.convn(arr, kernel, Convolution.Type.VALID);
        assertEquals(answer, test);
    }



    @Test
    public void testConv2d() {
        INDArray arr = Nd4j.linspace(1, 8, 8).reshape(2,4);
        INDArray kernel = Nd4j.linspace(1, 6, 6).reshape(2,3);
        INDArray answer = Nd4j.create(new double[]{56,98});
        INDArray test = Convolution.convn(arr, kernel, Convolution.Type.VALID);
        assertEquals(answer, test);
    }




}
