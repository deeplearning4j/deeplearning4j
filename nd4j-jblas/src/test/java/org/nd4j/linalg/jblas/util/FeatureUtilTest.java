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

package org.nd4j.linalg.jblas.util;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 10/31/14.
 */
public class FeatureUtilTest {

    private static Logger log = LoggerFactory.getLogger(FeatureUtil.class);


    @Test
    public void testMinMax() {
        Nd4j.factory().setOrder('f');
        INDArray test = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray assertion = Nd4j.create(Nd4j.createBuffer(new double[]{0, 1, 0, 1}), new int[]{2, 2});
        FeatureUtil.scaleMinMax(0, 1, test);
        assertEquals(assertion, test);

        INDArray twoThree = Nd4j.create(new double[][]{{1, 2, 3}, {4, 5, 6}});
        INDArray assertion2 = Nd4j.create(Nd4j.createBuffer(new double[]{0, 1, 0, 1, 0, 1}), new int[]{2, 3});
        FeatureUtil.scaleMinMax(0, 1, twoThree);
        assertEquals(assertion2, twoThree);


    }


}
