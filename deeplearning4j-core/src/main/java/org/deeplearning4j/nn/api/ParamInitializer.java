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

package org.deeplearning4j.nn.api;

import org.canova.api.conf.Configuration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

/**
 * Param initializer for a layer
 *
 * @author Adam Gibson
 */
public interface ParamInitializer {

    int numParams( NeuralNetConfiguration conf, boolean backprop );

    /**
     * Initialize the parameters
     * @param paramsMap the map (initially empty) that will contain a view of the 'paramsView' array
     * @param conf the configuration
     * @param paramsView a view of the full network (backprop) parameters
     */
    void init(Map<String, INDArray> paramsMap, NeuralNetConfiguration conf, INDArray paramsView);

    /**
     * Return a map of gradients (in their standard non-flattened representation), taken from the flattened (row vector) gradientView array.
     * The idea is that operates in exactly the same way as the the paramsView does in {@link #init(Map, NeuralNetConfiguration, INDArray)};
     * thus the position in the view (and, the array orders) must match those of the parameters
     * @param conf            Configuration
     * @param gradientView    The flattened gradients array, as a view of the larger array
     * @return                A map containing an array by parameter type, that is a view of the full network gradients array
     */
    Map<String,INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView);

}
