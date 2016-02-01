/*
 *
 *  * Copyright 2016 Skymind,Inc.
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
 */
package org.arbiter.deeplearning4j.multilayernetwork;

import org.apache.commons.lang.ArrayUtils;
import org.arbiter.deeplearning4j.layers.DenseLayerSpace;
import org.arbiter.optimize.parameter.FixedValue;
import org.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.arbiter.optimize.parameter.discrete.DiscreteParameterSpace;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class TestLayerSpace {

    @Test
    public void testBasic1(){

        DenseLayer expected = new DenseLayer.Builder()
                .nOut(13).activation("relu").build();

        DenseLayerSpace space = new DenseLayerSpace.Builder()
                .nOut(13).activation("relu").build();

        DenseLayer actual = space.randomLayer();

        assertEquals(expected,actual);
    }

    @Test
    public void testBasic2(){

        String[] actFns = new String[]{"softsign","relu","leakyrelu"};

        for( int i=0; i<20; i++ ) {

            new DenseLayer.Builder().build();

            DenseLayerSpace ls = new DenseLayerSpace.Builder()
                    .nOut(20)
                    .learningRate(new ContinuousParameterSpace(0.3,0.4))
                    .l2(new ContinuousParameterSpace(0.01,0.1))
                    .activation(new DiscreteParameterSpace<>(actFns))
                    .build();

            DenseLayer l = ls.randomLayer();

            assertEquals(20, l.getNOut());
            double lr = l.getLearningRate();
            double l2 = l.getL2();
            String activation = l.getActivationFunction();

            System.out.println(lr + "\t" + l2 + "\t" + activation);

            assertTrue(lr >= 0.3 && lr <= 0.4);
            assertTrue(l2 >= 0.01 && l2 <= 0.1 );
            assertTrue(ArrayUtils.contains(actFns, activation));
        }
    }

}
