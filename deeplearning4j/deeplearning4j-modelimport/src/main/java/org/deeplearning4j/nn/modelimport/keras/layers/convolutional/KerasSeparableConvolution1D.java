/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.modelimport.keras.layers.convolutional;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasActivationUtils;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.SeparableConvolution1D;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasConstraintUtils;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasRegularizerUtils;
import org.deeplearning4j.nn.modelimport.keras.utils.*;
import org.deeplearning4j.nn.params.SeparableConvolutionParamInitializer;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.Map;

import static org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasConvolutionUtils.*;

/**
 * Keras SeparableConv1D layer importer.
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@Slf4j
@Data
@EqualsAndHashCode(callSuper = false)
public class KerasSeparableConvolution1D extends KerasConvolution {

    /**
     * Pass-through constructor from KerasLayer
     *
     * @param kerasVersion major keras version
     * @throws UnsupportedKerasConfigurationException Unsupported Keras configuration
     */
    public KerasSeparableConvolution1D(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
        super(kerasVersion);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @throws InvalidKerasConfigurationException     Invalid Keras configuration
     * @throws UnsupportedKerasConfigurationException Unsupported Keras configuration
     */
    public KerasSeparableConvolution1D(Map<String, Object> layerConfig)
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
    public KerasSeparableConvolution1D(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);

        hasBias = KerasLayerUtils.getHasBiasFromConfig(layerConfig, conf);
        numTrainableParams = hasBias ? 3 : 2;
        long[] dilationRate = getDilationRateLong(layerConfig, 1, conf, false);

        int depthMultiplier = getDepthMultiplier(layerConfig, conf);

        IWeightInit depthWiseInit = KerasInitilizationUtils.getWeightInitFromConfig(layerConfig,
                conf.getLAYER_FIELD_DEPTH_WISE_INIT(), enforceTrainingConfig, conf, kerasMajorVersion);

        IWeightInit pointWiseInit = KerasInitilizationUtils.getWeightInitFromConfig(layerConfig,
                conf.getLAYER_FIELD_POINT_WISE_INIT(), enforceTrainingConfig, conf, kerasMajorVersion);

        if (!depthWiseInit.getClass().equals(pointWiseInit.getClass()))
            if (enforceTrainingConfig)
                throw new UnsupportedKerasConfigurationException(
                        "Specifying different initialization for depth- and point-wise weights not supported.");
            else
                log.warn("Specifying different initialization for depth- and point-wise weights not supported.");

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

        SeparableConvolution1D.Builder builder = new SeparableConvolution1D.Builder()
                .name(this.layerName)
                .nOut(KerasLayerUtils.getNOutFromConfig(layerConfig, conf))
                .dropOut(this.dropout)
                .activation(KerasActivationUtils.getIActivationFromConfig(layerConfig, conf))
                .weightInit(depthWiseInit)
                .depthMultiplier(depthMultiplier)
                .l1(this.weightL1Regularization)
                .l2(this.weightL2Regularization)
                .convolutionMode(getConvolutionModeFromConfig(layerConfig, conf))
                .kernelSize((int) getKernelSizeFromConfigLong(layerConfig, 1, conf, kerasMajorVersion)[0])
                .hasBias(hasBias)
                .stride((int) getStrideFromConfigLong(layerConfig, 1, conf)[0]);

        long[] padding = getPaddingFromBorderModeConfigLong(layerConfig, 1, conf, kerasMajorVersion);
        if (padding != null)
            builder.padding((int) padding[0]);
        if (dilationRate != null)
            builder.dilation((int) dilationRate[0]);
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
            // Keras: [kernel_size, input_channels, depth_multiplier]
            // DL4J: [depth_multiplier, input_channels, kernel_size, 1]
            dW = dW.permute(2, 1, 0);
            dW = dW.reshape(dW.size(0), dW.size(1), dW.size(2), 1);
        } else
            throw new InvalidKerasConfigurationException(
                    "Keras SeparableConvolution1D layer does not contain parameter "
                            + conf.getLAYER_PARAM_NAME_DEPTH_WISE_KERNEL());

        this.weights.put(SeparableConvolutionParamInitializer.DEPTH_WISE_WEIGHT_KEY, dW);

        INDArray pW;
        if (weights.containsKey(conf.getLAYER_PARAM_NAME_POINT_WISE_KERNEL())) {
            pW = weights.get(conf.getLAYER_PARAM_NAME_POINT_WISE_KERNEL());
            // Keras: [1, input_channels * depth_multiplier, output_channels]
            // DL4J: [output_channels, input_channels * depth_multiplier, 1, 1]
            pW = pW.permute(2, 1, 0);
            pW = pW.reshape(pW.size(0), pW.size(1), 1, 1);
        } else
            throw new InvalidKerasConfigurationException(
                    "Keras SeparableConvolution1D layer does not contain parameter "
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
                        "Keras SeparableConvolution1D layer does not contain bias parameter");
            this.weights.put(SeparableConvolutionParamInitializer.BIAS_KEY, bias);
        }
    }

    /**
     * Get DL4J SeparableConvolution1D layer.
     *
     * @return SeparableConvolution1D
     */
    public SeparableConvolution1D getSeparableConvolution1DLayer() {
        return (SeparableConvolution1D) this.layer;
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
                    "Keras separable convolution 1D layer accepts only one input (received " + inputType.length + ")");
        return this.getSeparableConvolution1DLayer().getOutputType(-1, inputType[0]);
    }
}
