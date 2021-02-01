/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.modelimport.keras.utils;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.HashMap;
import java.util.Map;


/**
 * Utility functionality for keras loss functions
 *
 * @author Max Pumperla
 */
@Slf4j
public class KerasLossUtils {
    static final Map<String, ILossFunction> customLoss = new HashMap<>();

    /**
     * Register a custom loss function
     *
     * @param lossName   name of the lambda layer in the serialized Keras model
     * @param lossFunction SameDiffLambdaLayer instance to map to Keras Lambda layer
     */
    public static void registerCustomLoss(String lossName, ILossFunction lossFunction) {
        customLoss.put(lossName, lossFunction);
    }

    /**
     * Clear all lambda layers
     *
     */
    public static void clearCustomLoss() {
        customLoss.clear();
    }

    /**
     * Map Keras to DL4J loss functions.
     *
     * @param kerasLoss String containing Keras loss function name
     * @return String containing DL4J loss function
     */
    public static ILossFunction mapLossFunction(String kerasLoss, KerasLayerConfiguration conf)
            throws UnsupportedKerasConfigurationException {
        LossFunctions.LossFunction dl4jLoss;
        if (kerasLoss.equals(conf.getKERAS_LOSS_MEAN_SQUARED_ERROR()) ||
                kerasLoss.equals(conf.getKERAS_LOSS_MSE())) {
            dl4jLoss = LossFunctions.LossFunction.SQUARED_LOSS;
        } else if (kerasLoss.equals(conf.getKERAS_LOSS_MEAN_ABSOLUTE_ERROR()) ||
                kerasLoss.equals(conf.getKERAS_LOSS_MAE())) {
            dl4jLoss = LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR;
        } else if (kerasLoss.equals(conf.getKERAS_LOSS_MEAN_ABSOLUTE_PERCENTAGE_ERROR()) ||
                kerasLoss.equals(conf.getKERAS_LOSS_MAPE())) {
            dl4jLoss = LossFunctions.LossFunction.MEAN_ABSOLUTE_PERCENTAGE_ERROR;
        } else if (kerasLoss.equals(conf.getKERAS_LOSS_MEAN_SQUARED_LOGARITHMIC_ERROR()) ||
                kerasLoss.equals(conf.getKERAS_LOSS_MSLE())) {
            dl4jLoss = LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR;
        } else if (kerasLoss.equals(conf.getKERAS_LOSS_SQUARED_HINGE())) {
            dl4jLoss = LossFunctions.LossFunction.SQUARED_HINGE;
        } else if (kerasLoss.equals(conf.getKERAS_LOSS_HINGE())) {
            dl4jLoss = LossFunctions.LossFunction.HINGE;
        } else if (kerasLoss.equals(conf.getKERAS_LOSS_SPARSE_CATEGORICAL_CROSSENTROPY())) {
            dl4jLoss = LossFunctions.LossFunction.SPARSE_MCXENT;
        } else if (kerasLoss.equals(conf.getKERAS_LOSS_BINARY_CROSSENTROPY())) {
            dl4jLoss = LossFunctions.LossFunction.XENT;
        } else if (kerasLoss.equals(conf.getKERAS_LOSS_CATEGORICAL_CROSSENTROPY())) {
            dl4jLoss = LossFunctions.LossFunction.MCXENT;
        } else if (kerasLoss.equals(conf.getKERAS_LOSS_KULLBACK_LEIBLER_DIVERGENCE()) ||
                kerasLoss.equals(conf.getKERAS_LOSS_KLD())) {
            dl4jLoss = LossFunctions.LossFunction.KL_DIVERGENCE;
        } else if (kerasLoss.equals(conf.getKERAS_LOSS_POISSON())) {
            dl4jLoss = LossFunctions.LossFunction.POISSON;
        } else if (kerasLoss.equals(conf.getKERAS_LOSS_COSINE_PROXIMITY())) {
            dl4jLoss = LossFunctions.LossFunction.COSINE_PROXIMITY;
        } else {
            ILossFunction lossClass = customLoss.get(kerasLoss);
            if(lossClass != null){
                return lossClass;
            }else{
                throw new UnsupportedKerasConfigurationException("Unknown Keras loss function " + kerasLoss);
            }
        }
        return dl4jLoss.getILossFunction();
    }
}
