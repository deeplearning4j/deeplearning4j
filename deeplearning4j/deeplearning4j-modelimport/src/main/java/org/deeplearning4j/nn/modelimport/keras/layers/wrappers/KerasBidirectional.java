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

package org.deeplearning4j.nn.modelimport.keras.layers.wrappers;

import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.InputTypeUtil;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.layers.recurrent.KerasLstm;
import org.deeplearning4j.nn.modelimport.keras.layers.recurrent.KerasSimpleRnn;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.Map;

/**
 * Builds a DL4J Bidirectional layer from a Keras Bidirectional layer wrapper
 *
 * @author Max Pumperla
 */
public class KerasBidirectional extends KerasLayer {

    private KerasLayer kerasRnnlayer;

    /**
     * Pass-through constructor from KerasLayer
     *
     * @param kerasVersion major keras version
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasBidirectional(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
        super(kerasVersion);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasBidirectional(Map<String, Object> layerConfig)
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
        @SuppressWarnings("unchecked")
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

        String rnnClass = (String) innerRnnConfig.get("class_name");
        switch (rnnClass) {
            case "LSTM":
                kerasRnnlayer = new KerasLstm(innerRnnConfig, enforceTrainingConfig);
                try {
                    LSTM rnnLayer = (LSTM) ((KerasLstm) kerasRnnlayer).getLSTMLayer();
                    layer = new Bidirectional(mode, rnnLayer);
                    layer.setLayerName(layerName);
                } catch (Exception e) {
                    LastTimeStep rnnLayer = (LastTimeStep) ((KerasLstm) kerasRnnlayer).getLSTMLayer();
                    this.layer = new Bidirectional(mode, rnnLayer);
                    layer.setLayerName(layerName);
                }
                break;
            case "SimpleRNN":
                kerasRnnlayer = new KerasSimpleRnn(innerRnnConfig, enforceTrainingConfig);
                SimpleRnn rnnLayer = (SimpleRnn) ((KerasSimpleRnn) kerasRnnlayer).getSimpleRnnLayer();
                this.layer = new Bidirectional(mode, rnnLayer);
                layer.setLayerName(layerName);
                break;
            default:
                throw new UnsupportedKerasConfigurationException("Currently only two types of recurrent Keras layers are" +
                        "supported, 'LSTM' and 'SimpleRNN'. You tried to load a layer of class:" + rnnClass);
        }

    }

    /**
     * Return the underlying recurrent layer of this bidirectional layer
     *
     * @return Layer, recurrent layer
     */
    public Layer getUnderlyingRecurrentLayer() {
        return kerasRnnlayer.getLayer();
    }

    /**
     * Get DL4J Bidirectional layer.
     *
     * @return Bidirectional Layer
     */
    public Bidirectional getBidirectionalLayer() {
        return (Bidirectional) this.layer;
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
                    "Keras Bidirectional layer accepts only one input (received " + inputType.length + ")");
        InputPreProcessor preProcessor = getInputPreprocessor(inputType);
        if (preProcessor != null)
            return preProcessor.getOutputType(inputType[0]);
        else
            return this.getBidirectionalLayer().getOutputType(-1, inputType[0]);
    }

    /**
     * Returns number of trainable parameters in layer.
     *
     * @return number of trainable parameters
     */
    @Override
    public int getNumParams() {
        return 2 * kerasRnnlayer.getNumParams();
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
                    "Keras Bidirectional layer accepts only one input (received " + inputType.length + ")");
        return InputTypeUtil.getPreprocessorForInputTypeRnnLayers(inputType[0], layerName);
    }

    /**
     * Set weights for Bidirectional layer.
     *
     * @param weights Map of weights
     */
    @Override
    public void setWeights(Map<String, INDArray> weights) throws InvalidKerasConfigurationException {

        Map<String, INDArray> forwardWeights = getUnderlyingWeights(weights, "forward");
        Map<String, INDArray> backwardWeights = getUnderlyingWeights(weights, "backward");

        this.weights = new HashMap<>();

        for (String key : forwardWeights.keySet())
            this.weights.put("f" + key, forwardWeights.get(key));
        for (String key : backwardWeights.keySet())
            this.weights.put("b" + key, backwardWeights.get(key));
    }


    private Map<String, INDArray> getUnderlyingWeights(Map<String, INDArray> weights, String direction)
            throws InvalidKerasConfigurationException {
        int keras1SubstringLength;
        if (kerasRnnlayer instanceof KerasLstm)
            keras1SubstringLength = 3;
        else if (kerasRnnlayer instanceof KerasSimpleRnn)
            keras1SubstringLength = 1;
        else throw new InvalidKerasConfigurationException("Unsupported layer type " + kerasRnnlayer.getClassName());

        Map newWeights = new HashMap<String, INDArray>();
        for (String key : weights.keySet()) {
            if (key.contains(direction)) {
                String newKey;
                if (kerasMajorVersion == 2) {
                    String[] subKeys = key.split("_");
                    if (key.contains("recurrent"))
                        newKey = subKeys[subKeys.length - 2] + "_" + subKeys[subKeys.length - 1];
                    else
                        newKey = subKeys[subKeys.length - 1];
                } else {
                    newKey = key.substring(key.length() - keras1SubstringLength);
                }
                newWeights.put(newKey, weights.get(key));
            }
        }
        if (!newWeights.isEmpty()) {
            weights = newWeights;
        }

        kerasRnnlayer.setWeights(weights);
        return kerasRnnlayer.getWeights();
    }

}
