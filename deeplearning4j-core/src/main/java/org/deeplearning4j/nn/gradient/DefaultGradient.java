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

package org.deeplearning4j.nn.gradient;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

/**
 * Default gradient implementation. Basically lookup table
 * for ndarrays
 *
 * @author Adam Gibson
 */

public class DefaultGradient implements Gradient {
    private Map<String,INDArray> gradients = new LinkedHashMap<>();



    @Override
    public Map<String, INDArray> gradientForVariable() {
        return gradients;
    }

    @Override
    public INDArray gradient(List<String> order) {
        List<INDArray> ret = new ArrayList<>();
        for(String s : order) {
            if(!gradientForVariable().containsKey(s))
               continue;
            ret.add(gradientForVariable().get(s));
        }
        return Nd4j.toFlattened('f',ret);
    }

    @Override
    public INDArray gradient() {
        return Nd4j.toFlattened('f',gradients.values());
    }

    @Override
    public void clear() {
        gradients.clear();
    }

    @Override
    public INDArray getGradientFor(String variable) {
        return gradients.get(variable);
    }

    @Override
    public INDArray setGradientFor(String variable, INDArray newGradient) {
        return gradients.put(variable, newGradient);
    }


    @Override
    public String toString() {
        return "DefaultGradient{" +
                "gradients=" + gradients +
                '}';
    }
}
