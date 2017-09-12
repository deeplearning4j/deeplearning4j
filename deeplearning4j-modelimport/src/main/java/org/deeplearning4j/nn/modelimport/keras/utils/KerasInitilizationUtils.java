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
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.primitives.Pair;

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
    public static Pair<WeightInit, Distribution> mapWeightInitialization(String kerasInit,
                                                                         KerasLayerConfiguration conf,
                                                                         Map<String, Object> initConfig)
            throws UnsupportedKerasConfigurationException {

        WeightInit init = WeightInit.XAVIER;
        Distribution dist = null;
        if (kerasInit != null) {
            if (kerasInit.equals(conf.getINIT_GLOROT_NORMAL())) {
                init = WeightInit.XAVIER;
            } else if (kerasInit.equals(conf.getINIT_GLOROT_UNIFORM())) {
                init = WeightInit.XAVIER_UNIFORM;
            } else if (kerasInit.equals(conf.getINIT_LECUN_NORMAL())) {
                init = WeightInit.LECUN_NORMAL;
            } else if (kerasInit.equals(conf.getINIT_LECUN_UNIFORM())) {
                init = WeightInit.LECUN_UNIFORM;
            } else if (kerasInit.equals(conf.getINIT_HE_NORMAL())) {
                init = WeightInit.RELU;
            } else if (kerasInit.equals(conf.getINIT_HE_UNIFORM())) {
                init = WeightInit.RELU_UNIFORM;
            } else if (kerasInit.equals(conf.getINIT_ONE()) ||
                    kerasInit.equals(conf.getINIT_ONES()) ||
                    kerasInit.equals(conf.getINIT_ONES_ALIAS())) {
                init = WeightInit.ONES;
            } else if (kerasInit.equals(conf.getINIT_ZERO()) ||
                kerasInit.equals(conf.getINIT_ZEROS()) ||
                kerasInit.equals(conf.getINIT_ZEROS_ALIAS())) {
            init = WeightInit.ZERO;
            } else if (kerasInit.equals(conf.getINIT_CONSTANT()) ||
                    kerasInit.equals(conf.getINIT_CONSTANT_ALIAS())) {
                // FIXME: CONSTANT
                // keras.initializers.Constant(value=0)
                init = WeightInit.ZERO;
            } else if (kerasInit.equals(conf.getINIT_UNIFORM()) ||
                    kerasInit.equals(conf.getINIT_RANDOM_UNIFORM()) ||
                    kerasInit.equals(conf.getINIT_RANDOM_UNIFORM_ALIAS())) {
                // FIXME: read minval and maxval from config
                // keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None) keras1: scale
                init = WeightInit.UNIFORM;
            } else if (kerasInit.equals(conf.getINIT_RANDOM_NORMAL()) ||
                    kerasInit.equals(conf.getINIT_RANDOM_NORMAL_ALIAS())) {
                // FIXME: read mean and stddev from config
                // keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
                init = WeightInit.DISTRIBUTION;
            } else if (kerasInit.equals(conf.getINIT_ORTHOGONAL()) ||
                    kerasInit.equals(conf.getINIT_ORTHOGONAL_ALIAS())) {
                // TODO keras.initializers.Orthogonal(gain=1.0, seed=None)
                init = WeightInit.DISTRIBUTION;
            } else if (kerasInit.equals(conf.getINIT_TRUNCATED_NORMAL()) ||
                    kerasInit.equals(conf.getINIT_TRUNCATED_NORMAL_ALIAS())) {
                // FIXME: read mean and stddev from config
                // keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None) keras1: no mean, always 0, stddev is scale
                init = WeightInit.DISTRIBUTION;
            } else if (kerasInit.equals(conf.getINIT_IDENTITY()) ||
                    kerasInit.equals(conf.getINIT_IDENTITY_ALIAS())) {
                // TODO: takes gain/scale parameter
                // keras.initializers.Identity(gain=1.0) keras1: scale
                init = WeightInit.IDENTITY;
            } else if (kerasInit.equals(conf.getINIT_VARIANCE_SCALING())) {
                //  keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
                //  With distribution="normal", samples are drawn from a truncated normal distribution centered on zero, with  stddev = sqrt(scale / n) where n is:
                //  number of input units in the weight tensor, if mode = "fan_in"
                //  number of output units, if mode = "fan_out"
                //  average of the numbers of input and output units, if mode = "fan_avg"
                //  With distribution="uniform", samples are drawn from a uniform distribution within [-limit, limit], with  limit = sqrt(3 * scale / n).

                init = WeightInit.XAVIER_UNIFORM;
            } else {
                throw new UnsupportedKerasConfigurationException("Unknown keras weight initializer " + kerasInit);
            }
        }
        return new Pair<>(init, dist);
    }

    /**
     * Get weight initialization from Keras layer configuration.
     *
     * @param layerConfig           dictionary containing Keras layer configuration
     * @param enforceTrainingConfig whether to enforce loading configuration for further training
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
        Map<String, Object> initMap;
        if (kerasMajorVersion != 2) {
            kerasInit = (String) innerConfig.get(initField);
            initMap = innerConfig;
        } else {
            initMap = (HashMap) innerConfig.get(initField);
            if (initMap.containsKey("class_name")) {
                kerasInit = (String) initMap.get("class_name");
            }
        }
        Pair<WeightInit, Distribution> init;
        try {
            init = mapWeightInitialization(kerasInit, conf, initMap);
        } catch (UnsupportedKerasConfigurationException e) {
            if (enforceTrainingConfig)
                throw e;
            else {
                init = new Pair<>(WeightInit.XAVIER, null);
                log.warn("Unknown weight initializer " + kerasInit + " (Using XAVIER instead).");
            }
        }
        return init.getFirst();
    }

}
