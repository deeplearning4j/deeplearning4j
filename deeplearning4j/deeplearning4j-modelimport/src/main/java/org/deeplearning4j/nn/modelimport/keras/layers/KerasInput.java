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
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;

import java.util.ArrayList;
import java.util.Map;



/**
 * Imports an Input layer from Keras. Used to set InputType of DL4J model.
 *
 * @author dave@skymind.io
 */
@Slf4j
@Data
@EqualsAndHashCode(callSuper = false)
public class KerasInput extends KerasLayer {

    private final int NO_TRUNCATED_BPTT = 0;

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration.
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasInput(Map<String, Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig           dictionary containing Keras layer configuration
     * @param enforceTrainingConfig whether to enforce training-related configuration options
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasInput(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        if (this.inputShape.length > 3)
            throw new UnsupportedKerasConfigurationException(
                    "Inputs with " + this.inputShape.length + " dimensions not supported");
    }

    /**
     * Constructor from layer name and input shape.
     *
     * @param layerName  layer name
     * @param inputShape input shape as array
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasInput(String layerName, int[] inputShape) throws
            UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        this(layerName, inputShape, true);
    }

    /**
     * Constructor from layer name and input shape.
     *
     * @param layerName             layer name
     * @param inputShape            input shape as array
     * @param enforceTrainingConfig whether to enforce training-related configuration options
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasInput(String layerName, int[] inputShape, boolean enforceTrainingConfig)
            throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        this.className = conf.getLAYER_CLASS_NAME_INPUT();
        this.layerName = layerName;
        this.inputShape = inputShape;
        this.inboundLayerNames = new ArrayList<>();
        this.layer = null;
        this.vertex = null;
        if (this.inputShape.length > 3)
            throw new UnsupportedKerasConfigurationException(
                    "Inputs with " + this.inputShape.length + " dimensions not supported");
    }

    /**
     * Get layer output type.
     *
     * @param inputType Array of InputTypes
     * @return output type as InputType
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    @Override
    public InputType getOutputType(InputType... inputType)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        if (inputType.length > 0)
            log.warn("Keras Input layer does not accept inputs (received " + inputType.length + "). Ignoring.");
        InputType myInputType;
        switch (this.inputShape.length) {
            case 1:
                myInputType = new InputType.InputTypeFeedForward(this.inputShape[0]);
                break;
            case 2:
                myInputType = new InputType.InputTypeRecurrent(this.inputShape[1], this.inputShape[0]);
                break;
            case 3:
                switch (this.dimOrder) {
                    case TENSORFLOW:
                        /* TensorFlow convolutional input: # rows, # cols, # channels */
                        myInputType = new InputType.InputTypeConvolutional(this.inputShape[0], this.inputShape[1],
                                this.inputShape[2]);
                        break;
                    case THEANO:
                        /* Theano convolutional input:     # channels, # rows, # cols */
                        myInputType = new InputType.InputTypeConvolutional(this.inputShape[1], this.inputShape[2],
                                this.inputShape[0]);
                        break;
                    default:
                        this.dimOrder = DimOrder.THEANO;
                        myInputType = new InputType.InputTypeConvolutional(this.inputShape[1], this.inputShape[2],
                                this.inputShape[0]);
                        log.warn("Couldn't determine dim ordering / data format from model file. Older Keras " +
                                "versions may come without specified backend, in which case we assume the model was " +
                                "built with theano." );
                }
                break;
            default:
                throw new UnsupportedKerasConfigurationException(
                        "Inputs with " + this.inputShape.length + " dimensions not supported");
        }
        return myInputType;
    }

    /**
     * Returns value of truncated BPTT, if any found.
     *
     * @return value of truncated BPTT
     */
    public int getTruncatedBptt() {
        if (this.inputShape.length == 2 && this.inputShape[0] > 0)
            return this.inputShape[0];
        return NO_TRUNCATED_BPTT;
    }
}
