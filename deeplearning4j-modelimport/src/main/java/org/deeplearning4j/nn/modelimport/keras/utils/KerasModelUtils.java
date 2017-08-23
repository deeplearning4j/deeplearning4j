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


import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.config.KerasModelConfiguration;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Utility functionality to import keras models
 *
 * @author Max Pumperla
 */
@Slf4j
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

    /**
     *
     * @param modelConfig parsed model configuration for keras model
     * @param config basic model configuration (KerasModelConfiguration)
     * @return Major Keras version (1 or 2)
     * @throws InvalidKerasConfigurationException
     */
    public static int determineKerasMajorVersion(Map<String, Object> modelConfig, KerasModelConfiguration config)
            throws InvalidKerasConfigurationException {
        int kerasMajorVersion;
        if (!modelConfig.containsKey(config.getFieldKerasVersion())) {
            log.warn("Could not read keras version used (no "
                    + config.getFieldKerasVersion() + " field found) \n"
                    + "assuming keras version is 1.0.7 or earlier."
            );
            kerasMajorVersion = 1;
        } else {
            String kerasVersionString = (String) modelConfig.get(config.getFieldKerasVersion());
            if (Character.isDigit(kerasVersionString.charAt(0))) {
                kerasMajorVersion = Character.getNumericValue(kerasVersionString.charAt(0));
            } else {
                throw new InvalidKerasConfigurationException(
                        "Keras version was not readable (" +  config.getFieldKerasVersion() + " provided)"
                );
            }
        }
        return kerasMajorVersion;
    }

    public static String findParameterName(String parameter, String[] fragmentList) {
        Matcher layerNameMatcher =
                Pattern.compile(fragmentList[fragmentList.length - 1]).matcher(parameter);
        if (!(layerNameMatcher.find()))
            log.warn("Unable to match layer parameter name " + parameter + " for stored weights.");
        String parameterNameFound = layerNameMatcher.replaceFirst("");

        /* Usually layer name is separated from parameter name by an underscore. */
        Matcher paramNameMatcher = Pattern.compile("^_(.+)$").matcher(parameterNameFound);
        if (paramNameMatcher.find())
            parameterNameFound = paramNameMatcher.group(1);

        /* TensorFlow backend often appends ":" followed by one or more digits to parameter names. */
        Matcher tfSuffixMatcher = Pattern.compile(":\\d+?$").matcher(parameterNameFound);
        if (tfSuffixMatcher.find())
            parameterNameFound = tfSuffixMatcher.replaceFirst("");

        /* TensorFlow backend also may append "_" followed by one or more digits to parameter names.*/
        Matcher tfParamNbMatcher = Pattern.compile("_\\d+$").matcher(parameterNameFound);
        if (tfParamNbMatcher.find())
            parameterNameFound = tfParamNbMatcher.replaceFirst("");

        return parameterNameFound;
    }

}
