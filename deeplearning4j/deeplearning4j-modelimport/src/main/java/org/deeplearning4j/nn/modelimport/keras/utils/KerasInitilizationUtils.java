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
import org.deeplearning4j.nn.weights.*;

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
    public static IWeightInit mapWeightInitialization(String kerasInit,
                                                      KerasLayerConfiguration conf,
                                                      Map<String, Object> initConfig,
                                                      int kerasMajorVersion)
            throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {


        // TODO: Identity and VarianceScaling need "scale" factor
        if (kerasInit != null) {
            if (kerasInit.equals(conf.getINIT_GLOROT_NORMAL()) ||
                    kerasInit.equals(conf.getINIT_GLOROT_NORMAL_ALIAS())) {
                return WeightInit.XAVIER.getWeightInitFunction();
            } else if (kerasInit.equals(conf.getINIT_GLOROT_UNIFORM()) ||
                    kerasInit.equals(conf.getINIT_GLOROT_UNIFORM_ALIAS())) {
                return WeightInit.XAVIER_UNIFORM.getWeightInitFunction();
            } else if (kerasInit.equals(conf.getINIT_LECUN_NORMAL()) ||
                    kerasInit.equals(conf.getINIT_LECUN_NORMAL_ALIAS())) {
                return WeightInit.LECUN_NORMAL.getWeightInitFunction();
            } else if (kerasInit.equals(conf.getINIT_LECUN_UNIFORM()) ||
                    kerasInit.equals(conf.getINIT_LECUN_UNIFORM_ALIAS())) {
                return WeightInit.LECUN_UNIFORM.getWeightInitFunction();
            } else if (kerasInit.equals(conf.getINIT_HE_NORMAL()) ||
                    kerasInit.equals(conf.getINIT_HE_NORMAL_ALIAS())) {
                return WeightInit.RELU.getWeightInitFunction();
            } else if (kerasInit.equals(conf.getINIT_HE_UNIFORM()) ||
                    kerasInit.equals(conf.getINIT_HE_UNIFORM_ALIAS())) {
                return WeightInit.RELU_UNIFORM.getWeightInitFunction();
            } else if (kerasInit.equals(conf.getINIT_ONE()) ||
                    kerasInit.equals(conf.getINIT_ONES()) ||
                    kerasInit.equals(conf.getINIT_ONES_ALIAS())) {
                return WeightInit.ONES.getWeightInitFunction();
            } else if (kerasInit.equals(conf.getINIT_ZERO()) ||
                    kerasInit.equals(conf.getINIT_ZEROS()) ||
                    kerasInit.equals(conf.getINIT_ZEROS_ALIAS())) {
                return WeightInit.ZERO.getWeightInitFunction();
            } else if (kerasInit.equals(conf.getINIT_UNIFORM()) ||
                    kerasInit.equals(conf.getINIT_RANDOM_UNIFORM()) ||
                    kerasInit.equals(conf.getINIT_RANDOM_UNIFORM_ALIAS())) {
                if (kerasMajorVersion == 2) {
                    double minVal = (double) initConfig.get(conf.getLAYER_FIELD_INIT_MINVAL());
                    double maxVal = (double) initConfig.get(conf.getLAYER_FIELD_INIT_MAXVAL());
                    return new WeightInitDistribution(new UniformDistribution(minVal, maxVal));
                } else {
                    double scale = 0.05;
                    if (initConfig.containsKey(conf.getLAYER_FIELD_INIT_SCALE()))
                        scale = (double) initConfig.get(conf.getLAYER_FIELD_INIT_SCALE());
                    return new WeightInitDistribution(new UniformDistribution(-scale, scale));
                }
            } else if (kerasInit.equals(conf.getINIT_NORMAL()) ||
                    kerasInit.equals(conf.getINIT_RANDOM_NORMAL()) ||
                    kerasInit.equals(conf.getINIT_RANDOM_NORMAL_ALIAS())) {
                if (kerasMajorVersion == 2) {
                    double mean = (double) initConfig.get(conf.getLAYER_FIELD_INIT_MEAN());
                    double stdDev = (double) initConfig.get(conf.getLAYER_FIELD_INIT_STDDEV());
                    return new WeightInitDistribution(new NormalDistribution(mean, stdDev));
                } else {
                    double scale = 0.05;
                    if (initConfig.containsKey(conf.getLAYER_FIELD_INIT_SCALE()))
                        scale = (double) initConfig.get(conf.getLAYER_FIELD_INIT_SCALE());
                    return new WeightInitDistribution(new NormalDistribution(0, scale));
                }
            } else if (kerasInit.equals(conf.getINIT_CONSTANT()) ||
                    kerasInit.equals(conf.getINIT_CONSTANT_ALIAS())) {
                double value = (double) initConfig.get(conf.getLAYER_FIELD_INIT_VALUE());
                return new WeightInitDistribution(new ConstantDistribution(value));
            } else if (kerasInit.equals(conf.getINIT_ORTHOGONAL()) ||
                    kerasInit.equals(conf.getINIT_ORTHOGONAL_ALIAS())) {
                if (kerasMajorVersion == 2) {
                    double gain;
                    try {
                        gain = (double) initConfig.get(conf.getLAYER_FIELD_INIT_GAIN());
                    } catch (Exception e) {
                        gain = (int) initConfig.get(conf.getLAYER_FIELD_INIT_GAIN());
                    }
                    return new WeightInitDistribution(new OrthogonalDistribution(gain));
                } else {
                    double scale = 1.1;
                    if (initConfig.containsKey(conf.getLAYER_FIELD_INIT_SCALE()))
                        scale = (double) initConfig.get(conf.getLAYER_FIELD_INIT_SCALE());
                    return new WeightInitDistribution(new OrthogonalDistribution(scale));
                }
            } else if (kerasInit.equals(conf.getINIT_TRUNCATED_NORMAL()) ||
                    kerasInit.equals(conf.getINIT_TRUNCATED_NORMAL_ALIAS())) {
                double mean = (double) initConfig.get(conf.getLAYER_FIELD_INIT_MEAN());
                double stdDev = (double) initConfig.get(conf.getLAYER_FIELD_INIT_STDDEV());
                return new WeightInitDistribution(new TruncatedNormalDistribution(mean, stdDev));
            } else if (kerasInit.equals(conf.getINIT_IDENTITY()) ||
                    kerasInit.equals(conf.getINIT_IDENTITY_ALIAS())) {
                if (kerasMajorVersion == 2) {
                    double gain = (double) initConfig.get(conf.getLAYER_FIELD_INIT_GAIN());
                    if (gain != 1.0)
                    if (gain != 1.0) {
                        return new WeightInitIdentity(gain);
                    } else {
                        return new WeightInitIdentity();
                    }
                } else {
                    double scale = 1.;
                    if (initConfig.containsKey(conf.getLAYER_FIELD_INIT_SCALE()))
                        scale = (double) initConfig.get(conf.getLAYER_FIELD_INIT_SCALE());
                    if (scale != 1.0) {
                        return new WeightInitIdentity(scale);
                    } else {
                        return new WeightInitIdentity();
                    }
                }
            } else if (kerasInit.equals(conf.getINIT_VARIANCE_SCALING())) {
                double scale;
                try {
                    scale = (double) initConfig.get(conf.getLAYER_FIELD_INIT_SCALE());
                } catch (Exception e) {
                    scale = (int) initConfig.get(conf.getLAYER_FIELD_INIT_SCALE());
                }
                String mode = (String) initConfig.get(conf.getLAYER_FIELD_INIT_MODE());
                String distribution = (String) initConfig.get(conf.getLAYER_FIELD_INIT_DISTRIBUTION());
                switch (mode) {
                    case "fan_in":
                        if (distribution.equals("normal")) {
                            return new WeightInitVarScalingNormalFanIn(scale);
                        } else {
                            return new WeightInitVarScalingUniformFanIn(scale);
                        }
                    case "fan_out":
                        if (distribution.equals("normal")) {
                            return new WeightInitVarScalingNormalFanOut(scale);
                        } else {
                            return new WeightInitVarScalingUniformFanOut(scale);
                        }
                    case "fan_avg":
                        if (distribution.equals("normal")) {
                            return new WeightInitVarScalingNormalFanAvg(scale);
                        } else {
                            return new WeightInitVarScalingUniformFanAvg(scale);
                        }
                    default:
                        throw new InvalidKerasConfigurationException("Initialization argument 'mode' has to be either " +
                                "fan_in, fan_out or fan_avg");
                }
            } else {
                throw new UnsupportedKerasConfigurationException("Unknown keras weight initializer " + kerasInit);
            }
        }
        throw new IllegalStateException("Error getting Keras weight initialization");
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
    public static IWeightInit getWeightInitFromConfig(Map<String, Object> layerConfig, String initField,
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
        IWeightInit init;
        try {
            init = mapWeightInitialization(kerasInit, conf, initMap, kerasMajorVersion);
        } catch (UnsupportedKerasConfigurationException e) {
            if (enforceTrainingConfig)
                throw e;
            else {
                init = new WeightInitXavier();
                log.warn("Unknown weight initializer " + kerasInit + " (Using XAVIER instead).");
            }
        }
        return init;
    }

}
