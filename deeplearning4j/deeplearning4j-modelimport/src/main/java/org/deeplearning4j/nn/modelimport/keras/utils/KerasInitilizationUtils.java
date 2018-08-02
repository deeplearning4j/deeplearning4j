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

package org.deeplearning4j.nn.modelimport.keras.utils;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.distribution.*;
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
                                                                         Map<String, Object> initConfig,
                                                                         int kerasMajorVersion)
            throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {

        // TODO: Identity and VarianceScaling need "scale" factor
        WeightInit init = null;
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
            } else if (kerasInit.equals(conf.getINIT_UNIFORM()) ||
                    kerasInit.equals(conf.getINIT_RANDOM_UNIFORM()) ||
                    kerasInit.equals(conf.getINIT_RANDOM_UNIFORM_ALIAS())) {
                if (kerasMajorVersion == 2) {
                    double minVal = (double) initConfig.get(conf.getLAYER_FIELD_INIT_MINVAL());
                    double maxVal = (double) initConfig.get(conf.getLAYER_FIELD_INIT_MAXVAL());
                    dist = new UniformDistribution(minVal, maxVal);
                } else {
                    double scale = 0.05;
                    if (initConfig.containsKey(conf.getLAYER_FIELD_INIT_SCALE()))
                        scale = (double) initConfig.get(conf.getLAYER_FIELD_INIT_SCALE());
                    dist = new UniformDistribution(-scale, scale);
                }
                init = WeightInit.DISTRIBUTION;
            } else if (kerasInit.equals(conf.getINIT_NORMAL()) ||
                    kerasInit.equals(conf.getINIT_RANDOM_NORMAL()) ||
                    kerasInit.equals(conf.getINIT_RANDOM_NORMAL_ALIAS())) {
                if (kerasMajorVersion == 2) {
                    double mean = (double) initConfig.get(conf.getLAYER_FIELD_INIT_MEAN());
                    double stdDev = (double) initConfig.get(conf.getLAYER_FIELD_INIT_STDDEV());
                    dist = new NormalDistribution(mean, stdDev);
                } else {
                    double scale = 0.05;
                    if (initConfig.containsKey(conf.getLAYER_FIELD_INIT_SCALE()))
                        scale = (double) initConfig.get(conf.getLAYER_FIELD_INIT_SCALE());
                    dist = new NormalDistribution(0, scale);
                }
                init = WeightInit.DISTRIBUTION;
            } else if (kerasInit.equals(conf.getINIT_CONSTANT()) ||
                    kerasInit.equals(conf.getINIT_CONSTANT_ALIAS())) {
                double value = (double) initConfig.get(conf.getLAYER_FIELD_INIT_VALUE());
                dist = new ConstantDistribution(value);
                init = WeightInit.DISTRIBUTION;
            } else if (kerasInit.equals(conf.getINIT_ORTHOGONAL()) ||
                    kerasInit.equals(conf.getINIT_ORTHOGONAL_ALIAS())) {
                if (kerasMajorVersion == 2) {
                    double gain;
                    try {
                        gain = (double) initConfig.get(conf.getLAYER_FIELD_INIT_GAIN());
                    } catch (Exception e) {
                        gain = (int) initConfig.get(conf.getLAYER_FIELD_INIT_GAIN());
                    }
                    dist = new OrthogonalDistribution(gain);
                } else {
                    double scale = 1.1;
                    if (initConfig.containsKey(conf.getLAYER_FIELD_INIT_SCALE()))
                        scale = (double) initConfig.get(conf.getLAYER_FIELD_INIT_SCALE());
                    dist = new OrthogonalDistribution(scale);
                }
                init = WeightInit.DISTRIBUTION;
            } else if (kerasInit.equals(conf.getINIT_TRUNCATED_NORMAL()) ||
                    kerasInit.equals(conf.getINIT_TRUNCATED_NORMAL_ALIAS())) {
                double mean = (double) initConfig.get(conf.getLAYER_FIELD_INIT_MEAN());
                double stdDev = (double) initConfig.get(conf.getLAYER_FIELD_INIT_STDDEV());
                dist = new TruncatedNormalDistribution(mean, stdDev);
                init = WeightInit.DISTRIBUTION;
            } else if (kerasInit.equals(conf.getINIT_IDENTITY()) ||
                    kerasInit.equals(conf.getINIT_IDENTITY_ALIAS())) {
                if (kerasMajorVersion == 2) {
                    double gain = (double) initConfig.get(conf.getLAYER_FIELD_INIT_GAIN());
                    if (gain != 1.)
                        log.warn("Scaled identity weight init not supported, setting gain=1");
                } else {
                    double scale = 1.;
                    if (initConfig.containsKey(conf.getLAYER_FIELD_INIT_SCALE()))
                        scale = (double) initConfig.get(conf.getLAYER_FIELD_INIT_SCALE());
                    if (scale != 1.)
                        log.warn("Scaled identity weight init not supported, setting scale=1");
                }
                init = WeightInit.IDENTITY;
            } else if (kerasInit.equals(conf.getINIT_VARIANCE_SCALING())) {
                double scale;
                try {
                    scale = (double) initConfig.get(conf.getLAYER_FIELD_INIT_SCALE());
                } catch (Exception e) {
                    scale = (int) initConfig.get(conf.getLAYER_FIELD_INIT_SCALE());
                }
                if (scale != 1.)
                    log.warn("Scaled identity weight init not supported, setting scale=1");
                String mode = (String) initConfig.get(conf.getLAYER_FIELD_INIT_MODE());
                String distribution = (String) initConfig.get(conf.getLAYER_FIELD_INIT_DISTRIBUTION());
                switch (mode) {
                    case "fan_in":
                        if (distribution.equals("normal")) {
                            init = WeightInit.VAR_SCALING_NORMAL_FAN_IN;
                        } else {
                            init = WeightInit.VAR_SCALING_UNIFORM_FAN_IN;
                        }
                        break;
                    case "fan_out":
                        if (distribution.equals("normal")) {
                            init = WeightInit.VAR_SCALING_NORMAL_FAN_OUT;
                        } else {
                            init = WeightInit.VAR_SCALING_UNIFORM_FAN_OUT;
                        }
                        break;
                    case "fan_avg":
                        if (distribution.equals("normal")) {
                            init = WeightInit.VAR_SCALING_NORMAL_FAN_AVG;
                        } else {
                            init = WeightInit.VAR_SCALING_UNIFORM_FAN_AVG;
                        }
                        break;
                    default:
                        throw new InvalidKerasConfigurationException("Initialization argument 'mode' has to be either " +
                                "fan_in, fan_out or fan_avg");
                }
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
     * @return Pair of DL4J weight initialization and distribution
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public static Pair<WeightInit, Distribution> getWeightInitFromConfig(Map<String, Object> layerConfig, String initField,
                                                                         boolean enforceTrainingConfig,
                                                                         KerasLayerConfiguration conf,
                                                                         int kerasMajorVersion)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        if (!innerConfig.containsKey(initField))
            throw new InvalidKerasConfigurationException("Keras layer is missing " + initField + " field");
        String kerasInit;
        Map<String, Object> initMap;
        if (kerasMajorVersion != 2) {
            kerasInit = (String) innerConfig.get(initField);
            initMap = innerConfig;
        } else {
            @SuppressWarnings("unchecked")
            Map<String, Object> fullInitMap = (HashMap) innerConfig.get(initField);
            initMap = (HashMap) fullInitMap.get("config");
            if (fullInitMap.containsKey("class_name")) {
                kerasInit = (String) fullInitMap.get("class_name");
            } else {
                throw new UnsupportedKerasConfigurationException("Incomplete initialization class");
            }
        }
        Pair<WeightInit, Distribution> init;
        try {
            init = mapWeightInitialization(kerasInit, conf, initMap, kerasMajorVersion);
        } catch (UnsupportedKerasConfigurationException e) {
            if (enforceTrainingConfig)
                throw e;
            else {
                init = new Pair<>(WeightInit.XAVIER, null);
                log.warn("Unknown weight initializer " + kerasInit + " (Using XAVIER instead).");
            }
        }
        return init;
    }

}
