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
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Deconvolution1D;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasActivationUtils;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasInitilizationUtils;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.Map;

import static org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasConvolutionUtils.*;

/**
 * Imports a Keras Conv1DTranspose (Deconvolution1D) layer.
 *
 * Conv1DTranspose performs transposed 1D convolution, also known as
 * deconvolution or fractionally-strided convolution. It's commonly used
 * for upsampling in autoencoders and generative models.
 *
 * @author Adam Gibson
 */
@Slf4j
@Data
@EqualsAndHashCode(callSuper = false)
public class KerasDeconvolution1D extends KerasLayer {

    private boolean hasBias;
    private int numTrainableParams;

    /**
     * Pass-through constructor from KerasLayer
     *
     * @param kerasVersion major keras version
     * @throws UnsupportedKerasConfigurationException Unsupported Keras configuration
     */
    public KerasDeconvolution1D(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
        super(kerasVersion);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @throws InvalidKerasConfigurationException     Invalid Keras configuration
     * @throws UnsupportedKerasConfigurationException Unsupported Keras configuration
     */
    public KerasDeconvolution1D(Map<String, Object> layerConfig)
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
    public KerasDeconvolution1D(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);

        hasBias = KerasLayerUtils.getHasBiasFromConfig(layerConfig, conf);
        numTrainableParams = hasBias ? 2 : 1;

        // Get kernel size (1D)
        long[] kernelSize = getKernelSizeFromConfigLong(layerConfig, 1, conf, kerasMajorVersion);
        long kernel = kernelSize[0];

        // Get strides
        long[] strides = getStrideFromConfigLong(layerConfig, 1, conf);
        long stride = strides[0];

        // Get dilation rate
        long[] dilationRate = getDilationRateLong(layerConfig, 1, conf, false);
        long dilation = dilationRate != null ? dilationRate[0] : 1;

        // Get number of filters (output channels)
        long nOut = KerasLayerUtils.getNOutFromConfig(layerConfig, conf);

        // Get convolution mode from padding
        ConvolutionMode convolutionMode = getConvolutionModeFromConfig(layerConfig, conf);

        // Get padding
        long[] padding = getPaddingFromBorderModeConfigLong(layerConfig, 1, conf, kerasMajorVersion);
        long pad = padding != null ? padding[0] : 0;

        // Get initializer
        IWeightInit weightInit = KerasInitilizationUtils.getWeightInitFromConfig(layerConfig,
                conf.getLAYER_FIELD_INIT(), enforceTrainingConfig, conf, kerasMajorVersion);

        // Build Deconvolution1D layer
        Deconvolution1D.Builder builder = new Deconvolution1D.Builder()
                .name(this.layerName)
                .nOut(nOut)
                .kernelSize(kernel)
                .stride(stride)
                .padding(pad)
                .dilation(dilation)
                .hasBias(hasBias)
                .convolutionMode(convolutionMode)
                .activation(KerasActivationUtils.getIActivationFromConfig(layerConfig, conf));

        // Set weight initializer
        if (weightInit != null) {
            builder.weightInit("weight", weightInit);
        }

        this.layer = builder.build();
    }

    /**
     * Get DL4J Deconvolution1D layer.
     *
     * @return Deconvolution1D
     */
    public Deconvolution1D getDeconvolution1DLayer() {
        return (Deconvolution1D) this.layer;
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
                    "Keras Deconvolution1D layer accepts only one input (received " + inputType.length + ")");
        return this.getDeconvolution1DLayer().getOutputType(-1, inputType[0]);
    }

    /**
     * Returns number of trainable parameters in layer.
     *
     * @return number of trainable parameters
     */
    @Override
    public int getNumParams() {
        return numTrainableParams;
    }

    /**
     * Set weights for layer.
     *
     * @param weights Map of weights
     */
    @Override
    public void setWeights(Map<String, INDArray> weights) throws InvalidKerasConfigurationException {
        this.weights = new HashMap<>();

        // Kernel weights - check for 'kernel' (Keras 2+) or 'W' (Keras 1)
        INDArray kernel;
        if (weights.containsKey("kernel")) {
            kernel = weights.get("kernel");
            // Keras format for Conv1DTranspose: [kernel_size, filters, input_dim]
            // DL4J format: [kernel_size, 1, nOut, nIn]
            long[] shape = kernel.shape();
            kernel = kernel.reshape(shape[0], 1, shape[1], shape[2]);
        } else if (weights.containsKey(conf.getKERAS_PARAM_NAME_W())) {
            kernel = weights.get(conf.getKERAS_PARAM_NAME_W());
            long[] shape = kernel.shape();
            kernel = kernel.reshape(shape[0], 1, shape[1], shape[2]);
        } else {
            throw new InvalidKerasConfigurationException(
                    "Keras Deconvolution1D layer does not contain kernel parameter");
        }
        this.weights.put("weight", kernel);

        // Bias
        if (hasBias) {
            INDArray bias;
            if (kerasMajorVersion >= 2 && weights.containsKey("bias")) {
                bias = weights.get("bias");
            } else if (kerasMajorVersion == 1 && weights.containsKey("b")) {
                bias = weights.get("b");
            } else {
                throw new InvalidKerasConfigurationException(
                        "Keras Deconvolution1D layer does not contain bias parameter");
            }
            this.weights.put("bias", bias);
        }
    }
}
