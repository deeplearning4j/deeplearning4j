/*-
 *
 *  * Copyright 2017 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
package org.deeplearning4j.nn.modelimport.keras.layers.convolutional;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils.removeDefaultWeights;

/**
 * Keras Convolution base layer
 *
 * @author Max Pumperla
 */

@Slf4j
@Data
@EqualsAndHashCode(callSuper = false)
abstract public class KerasConvolution extends KerasLayer {

    protected int numTrainableParams;
    protected boolean hasBias;

    /**
     * Pass-through constructor from KerasLayer
     *
     * @param kerasVersion major keras version
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasConvolution(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
        super(kerasVersion);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasConvolution(Map<String, Object> layerConfig)
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
    public KerasConvolution(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);

    }

    /**
     * Returns number of trainable parameters in layer.
     *
     * @return number of trainable parameters (2)
     */
    @Override
    public int getNumParams() {
        return numTrainableParams;
    }

    /**
     * Set weights for layer.
     *
     * @param weights Map from parameter name to INDArray.
     */
    @Override
    public void setWeights(Map<String, INDArray> weights) throws InvalidKerasConfigurationException {
        this.weights = new HashMap<>();
        if (weights.containsKey(conf.getKERAS_PARAM_NAME_W())) {
            INDArray kerasParamValue = weights.get(conf.getKERAS_PARAM_NAME_W());
            INDArray paramValue = getConvParameterValues(kerasParamValue);
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

    /**
     * Return processed parameter values obtained from Keras convolutional layers.
     *
     * @param kerasParamValue INDArray containing raw Keras weights to be processed
     * @return Processed weights, according to which backend was used.
     * @throws InvalidKerasConfigurationException Invalid Keras configuration exception.
     */
    public INDArray getConvParameterValues(INDArray kerasParamValue) throws InvalidKerasConfigurationException {
        INDArray paramValue;
        switch (this.getDimOrder()) {
            case TENSORFLOW:
                /* TensorFlow convolutional weights: # rows, # cols, # inputs, # outputs */
                paramValue = kerasParamValue.permute(3, 2, 0, 1);
                break;
            case THEANO:
                /* Theano convolutional weights match DL4J: # outputs, # inputs, # rows, # cols
                 * Theano's default behavior is to rotate filters by 180 degree before application.
                 */
                paramValue = kerasParamValue.dup();
                for (int i = 0; i < paramValue.tensorssAlongDimension(2, 3); i++) {
                    //dup required since we only want data from the view not the whole array
                    INDArray copyFilter = paramValue.tensorAlongDimension(i, 2, 3).dup();
                    double[] flattenedFilter = copyFilter.ravel().data().asDouble();
                    ArrayUtils.reverse(flattenedFilter);
                    INDArray newFilter = Nd4j.create(flattenedFilter, copyFilter.shape());
                    //manipulating weights in place to save memory
                    INDArray inPlaceFilter = paramValue.tensorAlongDimension(i, 2, 3);
                    inPlaceFilter.muli(0).addi(newFilter);
                }
                break;
            default:
                throw new InvalidKerasConfigurationException("Unknown keras backend " + this.getDimOrder());
        }
        return paramValue;
    }
}
