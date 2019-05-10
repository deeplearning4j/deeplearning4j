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

import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;

import java.util.Map;

public class KerasRegularizerUtils {

    /**
     * Get weight regularization from Keras weight regularization configuration.
     *
     * @param layerConfig     Map containing Keras weight regularization configuration
     * @param conf            Keras layer configuration
     * @param configField     regularization config field to use
     * @param regularizerType type of regularization as string (e.g. "l2")
     * @return L1 or L2 regularization strength (0.0 if none)
     */
    public static double getWeightRegularizerFromConfig(Map<String, Object> layerConfig,
                                                        KerasLayerConfiguration conf,
                                                        String configField,
                                                        String regularizerType)
            throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        if (innerConfig.containsKey(configField)) {
            Map<String, Object> regularizerConfig = (Map<String, Object>) innerConfig.get(configField);
            if (regularizerConfig != null) {
                if (regularizerConfig.containsKey(regularizerType)) {
                    return (double) regularizerConfig.get(regularizerType);
                }
                if (regularizerConfig.containsKey(conf.getLAYER_FIELD_CLASS_NAME()) &&
                        regularizerConfig.get(conf.getLAYER_FIELD_CLASS_NAME()).equals("L1L2")) {
                    Map<String, Object> innerRegularizerConfig =
                            KerasLayerUtils.getInnerLayerConfigFromConfig(regularizerConfig, conf);
                    try {
                        return (double) innerRegularizerConfig.get(regularizerType);
                    } catch (Exception e) {
                        return (double) (int) innerRegularizerConfig.get(regularizerType);
                    }


                }
            }
        }
        return 0.0;
    }
}
