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

package org.deeplearning4j.nn.modelimport.keras.layers.local;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasConvolution;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasActivationUtils;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.LocallyConnected2D;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasConstraintUtils;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasInitilizationUtils;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.Map;

import static org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasConvolutionUtils.*;


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

        hasBias = KerasLayerUtils.getHasBiasFromConfig(layerConfig, conf);
        numTrainableParams = hasBias ? 2 : 1;
        long[] dilationRate = getDilationRateLong(layerConfig, 2, conf, false);

        IWeightInit init = KerasInitilizationUtils.getWeightInitFromConfig(layerConfig, conf.getLAYER_FIELD_INIT(),
                enforceTrainingConfig, conf, kerasMajorVersion);
        // TODO: take care of bias init

        LayerConstraint biasConstraint = KerasConstraintUtils.getConstraintsFromConfig(
                layerConfig, conf.getLAYER_FIELD_B_CONSTRAINT(), conf, kerasMajorVersion);
        LayerConstraint weightConstraint = KerasConstraintUtils.getConstraintsFromConfig(
                layerConfig, conf.getLAYER_FIELD_W_CONSTRAINT(), conf, kerasMajorVersion);

        LocallyConnected2D.Builder builder = new LocallyConnected2D.Builder().name(this.layerName)
                .nOut(KerasLayerUtils.getNOutFromConfig(layerConfig, conf)).dropOut(this.dropout)
                .activation(KerasActivationUtils.getActivationFromConfig(layerConfig, conf))
                .weightInit(conf.getKERAS_PARAM_NAME_W(), init)
                .l1(this.weightL1Regularization).l2(this.weightL2Regularization)
                .convolutionMode(getConvolutionModeFromConfig(layerConfig, conf))
                .kernelSize(getKernelSizeFromConfigLong(layerConfig, 2, conf, kerasMajorVersion))
                .hasBias(hasBias)
                .stride(getStrideFromConfigLong(layerConfig, 2, conf));
        long[] padding = getPaddingFromBorderModeConfigLong(layerConfig, 2, conf, kerasMajorVersion);
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
     * Get DL4J LocallyConnected2D layer.
     *
     * @return Locally connected 2D layer.
     */
    public LocallyConnected2D getLocallyConnected2DLayer() {
        return (LocallyConnected2D) this.layer;
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
        InputType.InputTypeConvolutional convType = (InputType.InputTypeConvolutional) inputType[0];

        // Override input/output shape and input channels dynamically. This works since getOutputType will always
        // be called when initializing the model.
        ((LocallyConnected2D) this.layer).setInputSize(new long[] {convType.getHeight(),convType.getWidth()});
        ((LocallyConnected2D) this.layer).setNIn(convType.getChannels());
        ((LocallyConnected2D) this.layer).computeOutputSize();

        InputPreProcessor preprocessor = getInputPreprocessor(inputType[0]);
        if (preprocessor != null) {
            return this.getLocallyConnected2DLayer().getOutputType(-1, preprocessor.getOutputType(inputType[0]));
        }
        return this.getLocallyConnected2DLayer().getOutputType(-1, inputType[0]);
    }


    /**
     * Set weights for 2D locally connected layer.
     *
     * @param weights Map from parameter name to INDArray.
     */
    @Override
    public void setWeights(Map<String, INDArray> weights) throws InvalidKerasConfigurationException {
        this.weights = new HashMap<>();
        if (weights.containsKey(conf.getKERAS_PARAM_NAME_W())) {
            INDArray kerasParamValue = weights.get(conf.getKERAS_PARAM_NAME_W());
            this.weights.put(ConvolutionParamInitializer.WEIGHT_KEY, kerasParamValue);
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
        KerasLayerUtils.removeDefaultWeights(weights, conf);
    }
}
