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
 */

package org.deeplearning4j.nn.weights;


import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.junit.Test;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by agibsonccc on 9/13/14.
 */
public class WeightInitTests {

    @Test
    public void testSi() {
        WeightInitUtil.initWeights(1,2, WeightInit.VI, Nd4j.getDistributions().createNormal(0, 1));
    }

}
