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
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.SeparableConvolution2D;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasConstraintUtils;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasRegularizerUtils;
import org.deeplearning4j.nn.params.SeparableConvolutionParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.HashMap;
import java.util.Map;

import static org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasConvolutionUtils.*;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasActivationUtils.getIActivationFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasInitilizationUtils.getWeightInitFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils.getHasBiasFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils.getNOutFromConfig;


/**
 * Keras separable convolution 2D layer support
 *
 * @author Max Pumperla
 */
@Slf4j
@Data
@EqualsAndHashCode(callSuper = false)
public class KerasSeparableConvolution2D extends KerasConvolution {


    /**
     * Pass-through constructor from KerasLayer
     *
     * @param kerasVersion major keras version
     * @throws UnsupportedKerasConfigurationException Unsupported Keras configuration
     */
    public KerasSeparableConvolution2D(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
        super(kerasVersion);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @throws InvalidKerasConfigurationException     Invalid Keras configuration
     * @throws UnsupportedKerasConfigurationException Unsupported Keras configuration
     */
    public KerasSeparableConvolution2D(Map<String, Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig           dictionary containing Keras layer configuration
     * @param enforceTrainingConfig whether to enforce training-related configuration options
     * @throws InvalidKerasConfigurationException     Invalid Keras configuration
     * @throws UnsupportedKerasConfigurationException Unsupported Keras configuration
     */
    public KerasSeparableConvolution2D(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);

        hasBias = getHasBiasFromConfig(layerConfig, conf);
        numTrainableParams = hasBias ? 3 : 2;
        int[] dilationRate = getDilationRate(layerConfig, 2, conf, false);

        int depthMultiplier = getDepthMultiplier(layerConfig, conf);

        Pair<WeightInit, Distribution> depthWiseInit = getWeightInitFromConfig(layerConfig,
                conf.getLAYER_FIELD_DEPTH_WISE_INIT(), enforceTrainingConfig, conf, kerasMajorVersion);
        WeightInit depthWeightInit = depthWiseInit.getFirst();
        Distribution depthDistribution = depthWiseInit.getSecond();

        Pair<WeightInit, Distribution> pointWiseInit = getWeightInitFromConfig(layerConfig,
                conf.getLAYER_FIELD_POINT_WISE_INIT(), enforceTrainingConfig, conf, kerasMajorVersion);
        WeightInit pointWeightInit = pointWiseInit.getFirst();
        Distribution pointDistribution = pointWiseInit.getSecond();

        if (depthWeightInit != pointWeightInit || depthDistribution != pointDistribution)
            if (enforceTrainingConfig)
                throw new UnsupportedKerasConfigurationException(
                        "Specifying different initialization for depth- and point-wise weights not supported.");
            else
                log.warn("Specifying different initialization for depth- and point-wise  weights not supported.");

        this.weightL1Regularization = KerasRegularizerUtils.getWeightRegularizerFromConfig(
                layerConfig, conf, conf.getLAYER_FIELD_DEPTH_WISE_REGULARIZER(), conf.getREGULARIZATION_TYPE_L1());
        this.weightL2Regularization = KerasRegularizerUtils.getWeightRegularizerFromConfig(
                layerConfig, conf, conf.getLAYER_FIELD_DEPTH_WISE_REGULARIZER(), conf.getREGULARIZATION_TYPE_L2());


        LayerConstraint biasConstraint = KerasConstraintUtils.getConstraintsFromConfig(
                layerConfig, conf.getLAYER_FIELD_B_CONSTRAINT(), conf, kerasMajorVersion);
        LayerConstraint depthWiseWeightConstraint = KerasConstraintUtils.getConstraintsFromConfig(
                layerConfig, conf.getLAYER_FIELD_DEPTH_WISE_CONSTRAINT(), conf, kerasMajorVersion);
        LayerConstraint pointWiseWeightConstraint = KerasConstraintUtils.getConstraintsFromConfig(
                layerConfig, conf.getLAYER_FIELD_POINT_WISE_CONSTRAINT(), conf, kerasMajorVersion);

        SeparableConvolution2D.Builder builder = new SeparableConvolution2D.Builder().name(this.layerName)
                .nOut(getNOutFromConfig(layerConfig, conf)).dropOut(this.dropout)
                .activation(getIActivationFromConfig(layerConfig, conf))
                .weightInit(depthWeightInit)
                .depthMultiplier(depthMultiplier)
                .l1(this.weightL1Regularization).l2(this.weightL2Regularization)
                .convolutionMode(getConvolutionModeFromConfig(layerConfig, conf))
                .kernelSize(getKernelSizeFromConfig(layerConfig, 2, conf, kerasMajorVersion))
                .hasBias(hasBias)
                .stride(getStrideFromConfig(layerConfig, 2, conf));
        int[] padding = getPaddingFromBorderModeConfig(layerConfig, 2, conf, kerasMajorVersion);
        if (depthDistribution != null)
            builder.dist(depthDistribution);
        if (hasBias)
            builder.biasInit(0.0);
        if (padding != null)
            builder.padding(padding);
        if (dilationRate != null)
            builder.dilation(dilationRate);
        if (biasConstraint != null)
            builder.constrainBias(biasConstraint);
        if (depthWiseWeightConstraint != null)
            builder.constrainWeights(depthWiseWeightConstraint);
        if (pointWiseWeightConstraint != null)
            builder.constrainPointWise(pointWiseWeightConstraint);
        this.layer = builder.build();
    }

    /**
     * Set weights for layer.
     *
     * @param weights Map of weights
     */
    @Override
    public void setWeights(Map<String, INDArray> weights) throws InvalidKerasConfigurationException {
        this.weights = new HashMap<>();

        INDArray dW;
        if (weights.containsKey(conf.getLAYER_PARAM_NAME_DEPTH_WISE_KERNEL())) {
            dW = weights.get(conf.getLAYER_PARAM_NAME_DEPTH_WISE_KERNEL());
            dW = dW.permute(3, 2, 0, 1);
        } else
            throw new InvalidKerasConfigurationException(
                    "Keras SeparableConvolution2D layer does not contain parameter "
                            + conf.getLAYER_PARAM_NAME_DEPTH_WISE_KERNEL());

        this.weights.put(SeparableConvolutionParamInitializer.DEPTH_WISE_WEIGHT_KEY, dW);

        INDArray pW;
        if (weights.containsKey(conf.getLAYER_PARAM_NAME_POINT_WISE_KERNEL())) {
            pW = weights.get(conf.getLAYER_PARAM_NAME_POINT_WISE_KERNEL());
            pW = pW.permute(3, 2, 0, 1);
        }
        else
            throw new InvalidKerasConfigurationException(
                    "Keras SeparableConvolution2D layer does not contain parameter "
                            + conf.getLAYER_PARAM_NAME_POINT_WISE_KERNEL());

        this.weights.put(SeparableConvolutionParamInitializer.POINT_WISE_WEIGHT_KEY, pW);

        if (hasBias) {
            INDArray bias;
            if (kerasMajorVersion == 2 && weights.containsKey("bias"))
                bias = weights.get("bias");
            else if (kerasMajorVersion == 1 && weights.containsKey("b"))
                bias = weights.get("b");
            else
                throw new InvalidKerasConfigurationException(
                        "Keras SeparableConvolution2D layer does not contain bias parameter");
            this.weights.put(SeparableConvolutionParamInitializer.BIAS_KEY, bias);

        }

    }

    /**
     * Get DL4J SeparableConvolution2D.
     *
     * @return SeparableConvolution2D
     */
    public SeparableConvolution2D getSeparableConvolution2DLayer() {
        return (SeparableConvolution2D) this.layer;
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
                    "Keras separable convolution 2D layer accepts only one input (received " + inputType.length + ")");
        return this.getSeparableConvolution2DLayer().getOutputType(-1, inputType[0]);
    }

}