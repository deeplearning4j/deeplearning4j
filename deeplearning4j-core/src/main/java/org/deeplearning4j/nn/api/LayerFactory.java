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

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;

/**
 *
 * Common interface for creating neural network layers.
 *
 *
 * @author Adam Gibson
 */
public interface LayerFactory {

    /**
     *
     * Create a layer based on the based in configuration
     * @param conf the configuration to create the layer based on
     * @param iterationListeners the list of iterations listners
     * @param index the layer number
     * @param layerParamsView An array where the parameters are stored, in flattened order
     * @param initializeParams if true: initialize the parameter values (in layerParamsView) using the parameter initializer.
     *                         If false: make no changes to the values in layerParamsView
     * @return the created layer
     */
    <E extends Layer> E create(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners, int index,
                               INDArray layerParamsView, boolean initializeParams);


    /**
     * Get the param initializer used for initializing layers
     * @return the param initializer
     */
    ParamInitializer initializer();



}
