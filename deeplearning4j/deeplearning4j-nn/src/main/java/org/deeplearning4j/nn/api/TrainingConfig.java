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

import org.deeplearning4j.nn.conf.GradientNormalization;
import org.nd4j.linalg.learning.config.IUpdater;

/**
 * Simple interface for the training configuration (updater, L1/L2 values, etc) for trainable layers/vertices.
 *
 * @author Alex Black
 */
public interface TrainingConfig {

    /**
     * @return Name of the layer
     */
    String getLayerName();

    /**
     * @return True if the layer is set up for layerwise pretraining
     */
    boolean isPretrain();

    /**
     * Get the L1 coefficient for the given parameter.
     * Different parameters may have different L1 values, even for a single .l1(x) configuration.
     * For example, biases generally aren't L1 regularized, even if weights are
     *
     * @param paramName Parameter name
     * @return L1 value for that parameter
     */
    double getL1ByParam(String paramName);

    /**
     * Get the L2 coefficient for the given parameter.
     * Different parameters may have different L2 values, even for a single .l2(x) configuration.
     * For example, biases generally aren't L1 regularized, even if weights are
     *
     * @param paramName Parameter name
     * @return L2 value for that parameter
     */
    double getL2ByParam(String paramName);

    /**
     * Is the specified parameter a layerwise pretraining only parameter?<br>
     * For example, visible bias params in an autoencoder (or, decoder params in a variational autoencoder) aren't
     * used during supervised backprop.<br>
     * Layers (like DenseLayer, etc) with no pretrainable parameters will return false for all (valid) inputs.
     *
     * @param paramName Parameter name/key
     * @return True if the parameter is for layerwise pretraining only, false otherwise
     */
    boolean isPretrainParam(String paramName);

    /**
     * Get the updater for the given parameter. Typically the same updater will be used for all updaters, but this
     * is not necessarily the case
     *
     * @param paramName Parameter name
     * @return IUpdater for the parameter
     */
    IUpdater getUpdaterByParam(String paramName);

    /**
     * @return The gradient normalization configuration
     */
    GradientNormalization getGradientNormalization();

    /**
     * @return The gradient normalization threshold
     */
    double getGradientNormalizationThreshold();

    /**
     * @param pretrain Whether the layer is currently undergoing layerwise unsupervised training, or multi-layer backprop
     */
    void setPretrain(boolean pretrain);

}
