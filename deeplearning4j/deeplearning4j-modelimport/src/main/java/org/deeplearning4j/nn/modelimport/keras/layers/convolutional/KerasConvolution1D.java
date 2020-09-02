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

package org.deeplearning4j.nn.modelimport.keras.layers.convolutional;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution1DLayer;
import org.deeplearning4j.nn.conf.layers.InputTypeUtil;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasConstraintUtils;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.Map;

import static org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasConvolutionUtils.*;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasActivationUtils.getIActivationFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasInitilizationUtils.getWeightInitFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils.*;

/**
 * Imports a 1D Convolution layer from Keras.
 *
 * @author Max Pumperla
 */
@Slf4j
@Data
@EqualsAndHashCode(callSuper = false)
public class KerasConvolution1D extends KerasConvolution {

    /**
     * Pass-through constructor from KerasLayer
     * @param kerasVersion major keras version
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasConvolution1D(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
        super(kerasVersion);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasConvolution1D(Map<String, Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig               dictionary containing Keras layer configuration
     * @param enforceTrainingConfig     whether to enforce training-related configuration options
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasConvolution1D(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        hasBias = getHasBiasFromConfig(layerConfig, conf);
        numTrainableParams = hasBias ? 2 : 1;
        int[] dilationRate = getDilationRate(layerConfig, 1, conf, false);
        LayerConstraint biasConstraint = KerasConstraintUtils.getConstraintsFromConfig(
                layerConfig, conf.getLAYER_FIELD_B_CONSTRAINT(), conf, kerasMajorVersion);
        LayerConstraint weightConstraint = KerasConstraintUtils.getConstraintsFromConfig(
                layerConfig, conf.getLAYER_FIELD_W_CONSTRAINT(), conf, kerasMajorVersion);

        IWeightInit init = getWeightInitFromConfig(layerConfig, conf.getLAYER_FIELD_INIT(),
                enforceTrainingConfig, conf, kerasMajorVersion);
        Convolution1DLayer.Builder builder = new Convolution1DLayer.Builder().name(this.layerName)
                .nOut(getNOutFromConfig(layerConfig, conf)).dropOut(this.dropout)
                .activation(getIActivationFromConfig(layerConfig, conf))
                .weightInit(init)
                .l1(this.weightL1Regularization).l2(this.weightL2Regularization)
                .convolutionMode(getConvolutionModeFromConfig(layerConfig, conf))
                .kernelSize(getKernelSizeFromConfig(layerConfig, 1,  conf, kerasMajorVersion)[0])
                .hasBias(hasBias)
                .stride(getStrideFromConfig(layerConfig, 1, conf)[0]).rnnDataFormat(dimOrder == DimOrder.TENSORFLOW ? RNNFormat.NCW: RNNFormat.NWC);
        int[] padding = getPaddingFromBorderModeConfig(layerConfig, 1, conf, kerasMajorVersion);
        if (hasBias)
            builder.biasInit(0.0);
        if (padding != null)
            builder.padding(padding[0]);
        if (dilationRate != null)
            builder.dilation(dilationRate[0]);
        if (biasConstraint != null)
            builder.constrainBias(biasConstraint);
        if (weightConstraint != null)
            builder.constrainWeights(weightConstraint);
        if(inputShape != null) {
            if(dimOrder == DimOrder.THEANO) {
                builder.nIn(inputShape[0]);
            }
            else {
                builder.nIn(inputShape[1]);
            }
        }
        this.layer = builder.build();
    }

    /**
     * Get DL4J ConvolutionLayer.
     *
     * @return  ConvolutionLayer
     */
    public Convolution1DLayer getConvolution1DLayer() {
        return (Convolution1DLayer) this.layer;
    }


    /**
     * Get layer output type.
     *
     * @param inputType Array of InputTypes
     * @return output type as InputType
     * @throws InvalidKerasConfigurationException
     */
    @Override
    public InputType getOutputType(InputType... inputType) throws InvalidKerasConfigurationException {
        if (inputType.length > 1)
            throw new InvalidKerasConfigurationException(
                    "Keras Convolution layer accepts only one input (received " + inputType.length + ")");
        InputPreProcessor preprocessor = getInputPreprocessor(inputType[0]);
        if (preprocessor != null) {
            return this.getConvolution1DLayer().getOutputType(-1, preprocessor.getOutputType(inputType[0]));
        }
        return this.getConvolution1DLayer().getOutputType(-1, inputType[0]);
    }


    /**
     * Gets appropriate DL4J InputPreProcessor for given InputTypes.
     *
     * @param inputType Array of InputTypes
     * @return DL4J InputPreProcessor
     * @throws InvalidKerasConfigurationException Invalid Keras configuration exception
     * @see org.deeplearning4j.nn.conf.InputPreProcessor
     */
    @Override
    public InputPreProcessor getInputPreprocessor(InputType... inputType) throws InvalidKerasConfigurationException {
        if (inputType.length > 1)
            throw new InvalidKerasConfigurationException(
                    "Keras Conv1D layer accepts only one input (received " + inputType.length + ")");
        return InputTypeUtil.getPreprocessorForInputTypeRnnLayers(inputType[0], RNNFormat.NCW,layerName);
    }


    /**
     * Set weights for layer.
     *
     * @param weights   Map from parameter name to INDArray.
     */
    @Override
    public void setWeights(Map<String, INDArray> weights) throws InvalidKerasConfigurationException {
        this.weights = new HashMap<>();
        if (weights.containsKey(conf.getKERAS_PARAM_NAME_W())) {
            INDArray kerasParamValue = weights.get(conf.getKERAS_PARAM_NAME_W());
            INDArray paramValue;
            switch (this.getDimOrder()) {
                case TENSORFLOW:
                    paramValue = kerasParamValue.permute(2, 1, 0);
                    paramValue = paramValue.reshape(
                            paramValue.size(0), paramValue.size(1),
                            paramValue.size(2), 1);
                    break;

                case THEANO:
                    //Convert from keras [k,nIn,nOut] to DL4J conv2d [nOut, nIn, k, 1]
                    long k = kerasParamValue.size(0);
                    long nIn = kerasParamValue.size(1);
                    long nOut = kerasParamValue.size(2);
                    paramValue = kerasParamValue.permute(2, 1, 0).dup('c').reshape(nOut, nIn, k, 1);
                    break;
                default:
                    throw new InvalidKerasConfigurationException("Unknown keras backend " + this.getDimOrder());
            }

            this.weights.put(ConvolutionParamInitializer.WEIGHT_KEY, paramValue);
        } else
            throw new InvalidKerasConfigurationException(
                    "Parameter " + conf.getKERAS_PARAM_NAME_W() + " does not exist in weights");

        if (hasBias) {
            if (weights.containsKey(conf.getKERAS_PARAM_NAME_B()))
                this.weights.put(ConvolutionParamInitializer.BIAS_KEY, weights.get(conf.getKERAS_PARAM_NAME_B()));
            else
                throw new InvalidKerasConfigurationException(
                        "Parameter " + conf.getKERAS_PARAM_NAME_B() + " does not exist in weights");
        }
        removeDefaultWeights(weights, conf);
    }
}
