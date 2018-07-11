/*
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
package org.deeplearning4j.nn.modelimport.keras.layers.local;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.LocallyConnected2D;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasConvolution;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasConstraintUtils;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.primitives.Pair;

import java.util.Map;

import static org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasConvolutionUtils.*;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasActivationUtils.getActivationFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasInitilizationUtils.getWeightInitFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils.getHasBiasFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils.getNOutFromConfig;


/**
 * Imports a 2D locally connected layer from Keras.
 *
 * @author Max Pumperla
 */
@Slf4j
@Data
@EqualsAndHashCode(callSuper = false)
public class KerasLocallyConnected2D extends KerasConvolution {

    /**
     * Pass-through constructor from KerasLayer
     *
     * @param kerasVersion major keras version
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasLocallyConnected2D(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
        super(kerasVersion);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasLocallyConnected2D(Map<String, Object> layerConfig)
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
    public KerasLocallyConnected2D(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);

        hasBias = getHasBiasFromConfig(layerConfig, conf);
        numTrainableParams = hasBias ? 2 : 1;
        int[] dilationRate = getDilationRate(layerConfig, 2, conf, false);

        Pair<WeightInit, Distribution> init = getWeightInitFromConfig(layerConfig, conf.getLAYER_FIELD_INIT(),
                enforceTrainingConfig, conf, kerasMajorVersion);
        WeightInit weightInit = init.getFirst();
        Distribution distribution = init.getSecond();

        LayerConstraint biasConstraint = KerasConstraintUtils.getConstraintsFromConfig(
                layerConfig, conf.getLAYER_FIELD_B_CONSTRAINT(), conf, kerasMajorVersion);
        LayerConstraint weightConstraint = KerasConstraintUtils.getConstraintsFromConfig(
                layerConfig, conf.getLAYER_FIELD_W_CONSTRAINT(), conf, kerasMajorVersion);

        LocallyConnected2D.Builder builder = new LocallyConnected2D.Builder().name(this.layerName)
                .nOut(getNOutFromConfig(layerConfig, conf)).dropOut(this.dropout)
                .activation(getActivationFromConfig(layerConfig, conf))
                .weightInit(weightInit)
                .l1(this.weightL1Regularization).l2(this.weightL2Regularization)
                .convolutionMode(getConvolutionModeFromConfig(layerConfig, conf))
                .kernelSize(getKernelSizeFromConfig(layerConfig, 2, conf, kerasMajorVersion))
                .hasBias(hasBias)
                .stride(getStrideFromConfig(layerConfig, 2, conf));
        int[] padding = getPaddingFromBorderModeConfig(layerConfig, 2, conf, kerasMajorVersion);
        if (distribution != null)
            builder.dist(distribution);
        if (hasBias)
            builder.biasInit(0.0);
        if (padding != null)
            builder.padding(padding);
        if (dilationRate != null)
            builder.dilation(dilationRate);
        if (biasConstraint != null)
            builder.constrainBias(biasConstraint);
        if (weightConstraint != null)
            builder.constrainWeights(weightConstraint);
        this.layer = builder.build();
    }

    /**
     * Get DL4J ConvolutionLayer.
     *
     * @return ConvolutionLayer
     */
    public ConvolutionLayer getConvolution2DLayer() {
        return (ConvolutionLayer) this.layer;
    }

    /**
     * Get layer output type.
     *
     * @param inputType Array of InputTypes
     * @return output type as InputType
     * @throws InvalidKerasConfigurationException Invalid Keras config
     */
    @Override
    public InputType getOutputType(InputType... inputType) throws InvalidKerasConfigurationException {
        if (inputType.length > 1)
            throw new InvalidKerasConfigurationException(
                    "Keras Convolution layer accepts only one input (received " + inputType.length + ")");
        InputPreProcessor preprocessor = getInputPreprocessor(inputType[0]);
        if (preprocessor != null) {
            return this.getConvolution2DLayer().getOutputType(-1, preprocessor.getOutputType(inputType[0]));
        }
        return this.getConvolution2DLayer().getOutputType(-1, inputType[0]);
    }

}
