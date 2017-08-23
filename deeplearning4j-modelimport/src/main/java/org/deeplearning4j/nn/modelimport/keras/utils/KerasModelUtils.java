/*-
 *
 *  * Copyright 2017 Skymind,Inc.
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
package org.deeplearning4j.nn.modelimport.keras.utils;


import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Utility functionality to import keras models
 *
 * @author Max Pumperla
 */
public class KerasModelUtils {

    /**
     * Helper function to import weights from nested Map into existing model. Depends critically
     * on matched layer and parameter names. In general this seems to be straightforward for most
     * Keras models and layersOrdered, but there may be edge cases.
     *
     * @param model DL4J Model interface
     * @return DL4J Model interface
     * @throws InvalidKerasConfigurationException
     */
    public static Model copyWeightsToModel(Model model, Map<String, KerasLayer> layers)
            throws InvalidKerasConfigurationException {
        /* Get list if layers from model. */
        Layer[] layersFromModel;
        if (model instanceof MultiLayerNetwork)
            layersFromModel = ((MultiLayerNetwork) model).getLayers();
        else
            layersFromModel = ((ComputationGraph) model).getLayers();

        /* Iterate over layers in model, setting weights when relevant. */
        Set<String> layerNames = new HashSet<>(layers.keySet());
        for (org.deeplearning4j.nn.api.Layer layer : layersFromModel) {
            String layerName = layer.conf().getLayer().getLayerName();
            if (!layers.containsKey(layerName))
                throw new InvalidKerasConfigurationException(
                        "No weights found for layer in model (named " + layerName + ")");
            layers.get(layerName).copyWeightsToLayer(layer);
            layerNames.remove(layerName);
        }

        for (String layerName : layerNames) {
            if (layers.get(layerName).getNumParams() > 0)
                throw new InvalidKerasConfigurationException(
                        "Attemping to copy weights for layer not in model (named " + layerName + ")");
        }
        return model;
    }
}
