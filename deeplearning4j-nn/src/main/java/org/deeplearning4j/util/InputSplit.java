/*-
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

package org.deeplearning4j.util;

import org.deeplearning4j.berkeley.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class InputSplit {

    private InputSplit() {}

    public static void splitInputs(INDArray inputs, INDArray outcomes, List<Pair<INDArray, INDArray>> train,
                    List<Pair<INDArray, INDArray>> test, double split) {
        List<Pair<INDArray, INDArray>> list = new ArrayList<>();
        for (int i = 0; i < inputs.rows(); i++) {
            list.add(new Pair<>(inputs.getRow(i), outcomes.getRow(i)));
        }

        splitInputs(list, train, test, split);
    }

    public static void splitInputs(List<Pair<INDArray, INDArray>> pairs, List<Pair<INDArray, INDArray>> train,
                    List<Pair<INDArray, INDArray>> test, double split) {
        Random rand = new Random();

        for (Pair<INDArray, INDArray> pair : pairs)
            if (rand.nextDouble() <= split)
                train.add(pair);
            else
                test.add(pair);


    }

}
