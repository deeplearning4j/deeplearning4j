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
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasActivationUtils;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasConstraintUtils;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasInitilizationUtils;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;


@Slf4j
@Data
@EqualsAndHashCode(callSuper = false)
public class KerasConvolution2D extends KerasConvolution {

    /**
     * Return TF conv2d weights without permuting.
     *
     * TensorFlow stores 2D convolution weights as [kH, kW, iC, oC] which is YXIO — the same
     * format DL4J uses internally for all data formats (NCHW and NHWC).  The base-class
     * implementation permutes to OIYX [oC, iC, kH, kW] which is no longer correct after the
     * switch to YXIO-always in ConvolutionParamInitializer.
     *
     * @param kerasParamValue raw weight array loaded from the HDF5 file
     * @return weight array in YXIO [kH, kW, iC, oC] format ready for DL4J
     */
    @Override
    public INDArray getConvParameterValues(INDArray kerasParamValue) throws InvalidKerasConfigurationException {
        if (this.getDimOrder() == KerasLayer.DimOrder.TENSORFLOW && kerasParamValue.rank() != 5) {
            // TF 2D weights are already YXIO [kH, kW, iC, oC]; return as-is.
            return kerasParamValue;
        }
        // Theano and 3D TF handled by base class
        return super.getConvParameterValues(kerasParamValue);
    }

    /**
     * Pass-through constructor from KerasLayer
     *
     * @param kerasVersion major keras version
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasConvolution2D(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
        super(kerasVersion);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasConvolution2D(Map<String, Object> layerConfig)
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
    public KerasConvolution2D(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);

        hasBias = KerasLayerUtils.getHasBiasFromConfig(layerConfig, conf);
        numTrainableParams = hasBias ? 2 : 1;
        long[] dilationRate = KerasConvolutionUtils.getDilationRateLong(layerConfig, 2, conf, false);

        IWeightInit init = KerasInitilizationUtils.getWeightInitFromConfig(layerConfig, conf.getLAYER_FIELD_INIT(),
                enforceTrainingConfig, conf, kerasMajorVersion);

        LayerConstraint biasConstraint = KerasConstraintUtils.getConstraintsFromConfig(
                layerConfig, conf.getLAYER_FIELD_B_CONSTRAINT(), conf, kerasMajorVersion);
        LayerConstraint weightConstraint = KerasConstraintUtils.getConstraintsFromConfig(
                layerConfig, conf.getLAYER_FIELD_W_CONSTRAINT(), conf, kerasMajorVersion);

        ConvolutionLayer.Builder builder = new ConvolutionLayer.Builder().name(this.layerName)
                .nOut(KerasLayerUtils.getNOutFromConfig(layerConfig, conf)).dropOut(this.dropout)
                .activation(KerasActivationUtils.getIActivationFromConfig(layerConfig, conf))
                .weightInit(init)
                .dataFormat(dimOrder == KerasLayer.DimOrder.TENSORFLOW ? CNN2DFormat.NHWC : CNN2DFormat.NCHW)
                .l1(this.weightL1Regularization).l2(this.weightL2Regularization)
                .l1Bias(this.biasL1Regularization).l2Bias(this.biasL2Regularization)
                .convolutionMode(KerasConvolutionUtils.getConvolutionModeFromConfig(layerConfig, conf))
                .kernelSize(KerasConvolutionUtils.getKernelSizeFromConfigLong(layerConfig, 2, conf, kerasMajorVersion))
                .hasBias(hasBias)
                .stride(KerasConvolutionUtils.getStrideFromConfigLong(layerConfig, 2, conf));
        long[] padding = KerasConvolutionUtils.getPaddingFromBorderModeConfigLong(layerConfig, 2, conf, kerasMajorVersion);
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
        ConvolutionLayer convolutionLayer = (ConvolutionLayer) layer;
        convolutionLayer.setDefaultValueOverriden(true);

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
