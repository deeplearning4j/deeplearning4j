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
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.weights.WeightInit;

import java.util.HashMap;
import java.util.Map;

/**
 * Utility functionality for Keras weight initializers
 *
 * @author Max Pumperla
 */
@Slf4j
public class KerasInitilizationUtils {

    /**
     * Map Keras to DL4J weight initialization functions.
     *
     * @param kerasInit String containing Keras initialization function name
     * @return DL4J weight initialization enum
     * @see WeightInit
     */
    public static WeightInit mapWeightInitialization(String kerasInit, KerasLayerConfiguration conf)
            throws UnsupportedKerasConfigurationException {
        /* INIT_IDENTITY INIT_ORTHOGONAL INIT_LECUN_UNIFORM, INIT_NORMAL
         * INIT_VARIANCE_SCALING, INIT_CONSTANT, INIT_ONES missing.
         * Remaining dl4j distributions: DISTRIBUTION, SIZE, NORMALIZED,VI
         */
        WeightInit init = WeightInit.XAVIER;
        if (kerasInit != null) {
            if (kerasInit.equals(conf.getINIT_GLOROT_NORMAL())) {
                init = WeightInit.XAVIER;
            } else if (kerasInit.equals(conf.getINIT_GLOROT_UNIFORM())) {
                init = WeightInit.XAVIER_UNIFORM;
            } else if (kerasInit.equals(conf.getINIT_LECUN_NORMAL())) {
                init = WeightInit.NORMAL;
            } else if (kerasInit.equals(conf.getINIT_UNIFORM()) ||
                    kerasInit.equals(conf.getINIT_RANDOM_UNIFORM()) ||
                    kerasInit.equals(conf.getINIT_RANDOM_UNIFORM_ALIAS())) {
                init = WeightInit.UNIFORM;
            }  else if (kerasInit.equals(conf.getINIT_HE_NORMAL())) {
                init = WeightInit.RELU;
            } else if (kerasInit.equals(conf.getINIT_HE_UNIFORM())) {
                init = WeightInit.RELU_UNIFORM;
            } else if (kerasInit.equals(conf.getINIT_ZERO()) ||
                    kerasInit.equals(conf.getINIT_ZEROS()) ||
                    kerasInit.equals(conf.getINIT_ZEROS_ALIAS())) {
                init = WeightInit.ZERO;
            } else if (kerasInit.equals(conf.getINIT_VARIANCE_SCALING())) {
                // TODO: This is incorrect, but we need it in tests for now
                init = WeightInit.XAVIER_UNIFORM;
            } else {
                throw new UnsupportedKerasConfigurationException("Unknown keras weight initializer " + kerasInit);
            }
        }
        return init;
    }

    /**
     * Get weight initialization from Keras layer configuration.
     *
     * @param layerConfig           dictionary containing Keras layer configuration
     * @param enforceTrainingConfig
     * @return
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public static WeightInit getWeightInitFromConfig(Map<String, Object> layerConfig, String initField,
                                                 boolean enforceTrainingConfig,
                                                 KerasLayerConfiguration conf,
                                                 int kerasMajorVersion)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        if (!innerConfig.containsKey(initField))
            throw new InvalidKerasConfigurationException("Keras layer is missing " + initField + " field");
        String kerasInit = "glorot_normal";
        if (kerasMajorVersion != 2)
            kerasInit = (String) innerConfig.get(initField);
        else {
            HashMap initMap = (HashMap) innerConfig.get(initField);
            if (initMap.containsKey("class_name")) {
                kerasInit = (String) initMap.get("class_name");
            }
        }
        WeightInit init;
        try {
            init = mapWeightInitialization(kerasInit, conf);
        } catch (UnsupportedKerasConfigurationException e) {
            if (enforceTrainingConfig)
                throw e;
            else {
                init = WeightInit.XAVIER;
                log.warn("Unknown weight initializer " + kerasInit + " (Using XAVIER instead).");
            }
        }
        return init;
    }

}
