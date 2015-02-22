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

package org.nd4j.linalg.api.activation.test;

import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by agibsonccc on 2/21/15.
 */
public abstract class BaseActivationFunctionTests {

    @Test
    public void testExp() {
        INDArray exp = Nd4j.linspace(1, 5, 5);
        INDArray answer =  Nd4j.create(new double[]{ 2.71828183,    7.3890561 ,   20.08553692,   54.59815003,
                148.4131591});
        INDArray test = Activations.exp().apply(exp);
        assertEquals(answer,test);
    }


}
