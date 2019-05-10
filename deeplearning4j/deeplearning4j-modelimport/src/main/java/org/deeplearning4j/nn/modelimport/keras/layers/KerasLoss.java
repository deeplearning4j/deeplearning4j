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

package org.deeplearning4j.nn.modelimport.keras.layers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.CnnLossLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.conf.layers.RnnLossLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;

import static org.deeplearning4j.nn.modelimport.keras.utils.KerasLossUtils.mapLossFunction;

/**
 * Builds a DL4J LossLayer from a Keras training loss function.
 *
 * @author dave@skymind.io
 */
@Slf4j
@Data
@EqualsAndHashCode(callSuper = false)
public class KerasLoss extends KerasLayer {

    private final String KERAS_CLASS_NAME_LOSS = "Loss";
    private LossFunctions.LossFunction loss;


    /**
     * Constructor from layer name and input shape.
     *
     * @param layerName        layer name
     * @param inboundLayerName name of inbound layer
     * @param kerasLoss        name of Keras loss function
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasLoss(String layerName, String inboundLayerName, String kerasLoss)
            throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        this(layerName, inboundLayerName, kerasLoss, true);
    }

    /**
     * Constructor from layer name and input shape.
     *
     * @param layerName             layer name
     * @param inboundLayerName      name of inbound layer
     * @param kerasLoss             name of Keras loss function
     * @param enforceTrainingConfig whether to enforce training-related configuration options
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasLoss(String layerName, String inboundLayerName, String kerasLoss, boolean enforceTrainingConfig)
            throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        this.className = KERAS_CLASS_NAME_LOSS;
        this.layerName = layerName;
        this.inputShape = null;
        this.dimOrder = DimOrder.NONE;
        this.inboundLayerNames = new ArrayList<>();
        this.inboundLayerNames.add(inboundLayerName);
        try {
            loss = mapLossFunction(kerasLoss, conf);
        } catch (UnsupportedKerasConfigurationException e) {
            if (enforceTrainingConfig)
                throw e;
            log.warn("Unsupported Keras loss function. Replacing with MSE.");
            loss = LossFunctions.LossFunction.SQUARED_LOSS;
        }
    }

    /**
     * Get DL4J LossLayer.
     *
     * @return LossLayer
     */
    public FeedForwardLayer getLossLayer(InputType type) throws UnsupportedKerasConfigurationException {
        if (type instanceof InputType.InputTypeFeedForward) {
            this.layer = new LossLayer.Builder(loss).name(this.layerName).build();
        }
        else if (type instanceof  InputType.InputTypeRecurrent) {
            this.layer = new RnnLossLayer.Builder(loss).name(this.layerName).build();
        }
        else if (type instanceof InputType.InputTypeConvolutional) {
            this.layer = new CnnLossLayer.Builder(loss).name(this.layerName).build();
        } else {
            throw new UnsupportedKerasConfigurationException("Unsupported output layer type"
                    + "got : " + type.toString());
        }
        return (FeedForwardLayer) this.layer;
    }

    /**
     * Get layer output type.
     *
     * @param inputType Array of InputTypes
     * @return output type as InputType
     * @throws InvalidKerasConfigurationException Invalid Keras config
     */
    @Override
    public InputType getOutputType(InputType... inputType) throws InvalidKerasConfigurationException,
    UnsupportedKerasConfigurationException {
        if (inputType.length > 1)
            throw new InvalidKerasConfigurationException(
                    "Keras Loss layer accepts only one input (received " + inputType.length + ")");
        return this.getLossLayer(inputType[0]).getOutputType(-1, inputType[0]);
    }
}
