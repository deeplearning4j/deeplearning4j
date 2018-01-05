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
package org.deeplearning4j.nn.modelimport.keras.layers.wrappers;

import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.InputTypeUtil;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.layers.recurrent.KerasLstm;
import org.deeplearning4j.nn.modelimport.keras.layers.recurrent.KerasSimpleRnn;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

/**
 * Builds a DL4J Bidirectional layer from a Keras Bidirectional layer wrapper
 *
 * @author Max Pumperla
 */
public class KerasBidirectional extends KerasLayer {

    private KerasLayer kerasRnnlayer;
    private String rnnClass;

    /**
     * Pass-through constructor from KerasLayer
     * @param kerasVersion major keras version
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasBidirectional(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
        super(kerasVersion);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasBidirectional(Map<String, Object> layerConfig)
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
    public KerasBidirectional(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);

        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        if (!innerConfig.containsKey("merge_mode")) {
            throw new InvalidKerasConfigurationException("Field 'merge_mode' not found in configuration of " +
                    "Bidirectional layer.");
        }
        if (!innerConfig.containsKey("layer")) {
            throw new InvalidKerasConfigurationException("Field 'layer' not found in configuration of" +
                    "Bidirectional layer, i.e. no layer to be wrapped found.");
        }
        Map<String, Object> innerRnnConfig = (Map<String, Object>) innerConfig.get("layer");
        if (!innerRnnConfig.containsKey("class_name")) {
            throw new InvalidKerasConfigurationException("No 'class_name' specified within Bidirectional layer" +
                    "configuration.");
        }

        Bidirectional.Mode mode;
        String mergeModeString = (String) innerConfig.get("merge_mode");
        switch (mergeModeString) {
            case "sum":
                mode = Bidirectional.Mode.ADD;
                break;
            case "concat":
                mode = Bidirectional.Mode.CONCAT;
                break;
            case "mul":
                mode = Bidirectional.Mode.MUL;
                break;
            case "ave":
                mode = Bidirectional.Mode.AVERAGE;
                break;
            default:
                // Note that this is only for "None" mode, which we currently can't do.
                throw new UnsupportedKerasConfigurationException("Merge mode " + mergeModeString + " not supported.");
        }

        innerRnnConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasMajorVersion);

        Layer rnnLayer;
        rnnClass = (String) innerRnnConfig.get("class_name");
        if (rnnClass.equals("LSTM")) {
            kerasRnnlayer =  new KerasLstm(innerRnnConfig, enforceTrainingConfig);
            rnnLayer = ( (KerasLstm) kerasRnnlayer).getLSTMLayer();
        } else if (rnnClass.equals("SimpleRNN")) {
            kerasRnnlayer = new KerasSimpleRnn(innerRnnConfig, enforceTrainingConfig);
            rnnLayer = ((KerasSimpleRnn) kerasRnnlayer).getSimpleRnnLayer();
        } else {
            throw new UnsupportedKerasConfigurationException("Currently only two types of recurrent Keras layers are" +
                    "supported, 'LSTM' and 'SimpleRNN'. You tried to load a layer of class:" + rnnClass );
        }

       this.layer = new Bidirectional(mode, rnnLayer);
    }

    /**
     * Return the underlying recurrent layer of this bidirectional layer
     *
     * @return Layer, recurrent layer
     */
    public Layer getUnderlyingRecurrentLayer() { return kerasRnnlayer.getLayer();}

    /**
     * Get DL4J Bidirectional layer.
     *
     * @return  Bidirectional Layer
     */
    public Bidirectional getBidirectionalLayer() {
        return (Bidirectional) this.layer;
    }

    /**
     * Get layer output type.
     *
     * @param  inputType    Array of InputTypes
     * @return              output type as InputType
     * @throws InvalidKerasConfigurationException
     */
    @Override
    public InputType getOutputType(InputType... inputType) throws InvalidKerasConfigurationException {
        if (inputType.length > 1)
            throw new InvalidKerasConfigurationException(
                    "Keras Bidirectional layer accepts only one input (received " + inputType.length + ")");
        InputPreProcessor preProcessor = getInputPreprocessor(inputType);
        if (preProcessor != null)
            return  preProcessor.getOutputType(inputType[0]);
        else
            return this.getBidirectionalLayer().getOutputType(-1, inputType[0]);
    }

    /**
     * Returns number of trainable parameters in layer.
     *
     * @return number of trainable parameters
     */
    @Override
    public int getNumParams() { return  kerasRnnlayer.getNumParams(); }

    /**
     * Gets appropriate DL4J InputPreProcessor for given InputTypes.
     *
     * @param  inputType    Array of InputTypes
     * @return              DL4J InputPreProcessor
     * @throws InvalidKerasConfigurationException Invalid Keras configuration exception
     * @see org.deeplearning4j.nn.conf.InputPreProcessor
     */
    @Override
    public InputPreProcessor getInputPreprocessor(InputType... inputType) throws InvalidKerasConfigurationException {
        if (inputType.length > 1)
            throw new InvalidKerasConfigurationException(
                    "Keras Bidirectional layer accepts only one input (received " + inputType.length + ")");
        InputPreProcessor preProcessor = InputTypeUtil.getPreprocessorForInputTypeRnnLayers(inputType[0], layerName);
        return preProcessor;
    }

    /**
     * Set weights for Bidirectional layer.
     *
     * @param weights Map of weights
     */
    @Override
    public void setWeights(Map<String, INDArray> weights) throws InvalidKerasConfigurationException {
        kerasRnnlayer.setWeights(weights);
    }

}
