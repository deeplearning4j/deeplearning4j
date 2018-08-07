/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.api;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Map;

/**
 * Param initializer for a layer
 *
 * @author Adam Gibson
 */
public interface ParamInitializer {

    long numParams(NeuralNetConfiguration conf);

    long numParams(org.deeplearning4j.nn.conf.layers.Layer layer);

    /**
     * Get a list of all parameter keys given the layer configuration
     *
     * @param layer Layer
     * @return All parameter keys
     */
    List<String> paramKeys(org.deeplearning4j.nn.conf.layers.Layer layer);

    /**
     * Weight parameter keys given the layer configuration
     *
     * @param layer Layer
     * @return Weight parameter keys
     */
    List<String> weightKeys(org.deeplearning4j.nn.conf.layers.Layer layer);

    /**
     * Bias parameter keys given the layer configuration
     *
     * @param layer Layer
     * @return Bias parameter keys
     */
    List<String> biasKeys(org.deeplearning4j.nn.conf.layers.Layer layer);

    /**
     * Is the specified parameter a weight?
     *
     * @param layer Layer
     * @param key Key to check
     * @return True if parameter is a weight
     */
    boolean isWeightParam(Layer layer, String key);

    /**
     * Is the specified parameter a bias?
     *
     * @param layer Layer
     * @param key Key to check
     * @return True if parameter is a bias
     */
    boolean isBiasParam(Layer layer, String key);

    /**
     * Initialize the parameters
     *
     * @param conf             the configuration
     * @param paramsView       a view of the full network (backprop) parameters
     * @param initializeParams if true: initialize the parameters according to the configuration. If false: don't modify the
     *                         values in the paramsView array (but do select out the appropriate subset, reshape etc as required)
     * @return Map of parameters keyed by type (view of the 'paramsView' array)
     */
    Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams);

    /**
     * Return a map of gradients (in their standard non-flattened representation), taken from the flattened (row vector) gradientView array.
     * The idea is that operates in exactly the same way as the the paramsView does in {@link #init(Map, NeuralNetConfiguration, INDArray)};
     * thus the position in the view (and, the array orders) must match those of the parameters
     *
     * @param conf         Configuration
     * @param gradientView The flattened gradients array, as a view of the larger array
     * @return A map containing an array by parameter type, that is a view of the full network gradients array
     */
    Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView);

}
