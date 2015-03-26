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

package org.arbiter.nn.gradient;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

/**
 * Generic gradient
 *
 * @author Adam Gibson
 */
public interface Gradient extends Serializable {

    /**
     * Gradient look up table
     * @return the gradient look up table
     */
    Map<String,INDArray> gradientForVariable();
    /**
     * The full gradient as one flat vector
     * @return
     */
    INDArray gradient(List<String> order);

    /**
     * The full gradient as one flat vector
     * @return
     */
    INDArray gradient();

    /**
     * Clear residual parameters (useful for returning a gradient and then clearing old objects)
     */
    void clear();

    /**
     * The gradient for the given variable
     * @param variable the variable to get the gradient for
     * @return the gradient for the given variable or null
     */
    INDArray getGradientFor(String variable);


}
