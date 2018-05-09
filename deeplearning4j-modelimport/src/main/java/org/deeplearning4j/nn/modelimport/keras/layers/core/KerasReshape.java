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
package org.deeplearning4j.nn.modelimport.keras.layers.core;


import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.preprocessors.ReshapePreprocessor;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.List;
import java.util.Map;

/**
 * Imports Reshape layer from Keras
 *
 * @author Max Pumperla
 */
public class KerasReshape extends KerasLayer {

    private int[] targetShape;


    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasReshape(Map<String, Object> layerConfig)
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
    public KerasReshape(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        String targetShape = "target_shape";
        if (innerConfig.containsKey(targetShape)) {
            @SuppressWarnings("unchecked")
            List<Integer> targetShapeList = (List<Integer>) innerConfig.get(targetShape);
            this.targetShape = ArrayUtil.toArray(targetShapeList);
        }

    }

    /**
     * Whether this Keras layer maps to a DL4J InputPreProcessor.
     *
     * @return true
     */
    @Override
    public boolean isInputPreProcessor() {
        return true;
    }

    /**
     * Gets appropriate DL4J InputPreProcessor for given InputTypes.
     *
     * @param inputType Array of InputTypes
     * @return DL4J InputPreProcessor
     * @throws InvalidKerasConfigurationException Invalid Keras config
     * @see org.deeplearning4j.nn.conf.InputPreProcessor
     */
    @Override
    public InputPreProcessor getInputPreprocessor(InputType... inputType) throws InvalidKerasConfigurationException {
        if (inputType.length > 1)
            throw new InvalidKerasConfigurationException(
                    "Keras Reshape layer accepts only one input (received " + inputType.length + ")");
        InputPreProcessor preprocessor = null;
        if (inputType[0] instanceof InputType.InputTypeConvolutional) {
            InputType.InputTypeConvolutional it = (InputType.InputTypeConvolutional) inputType[0];
            int[] inputShape = new int[] {it.getChannels(), it.getHeight(), it.getWidth()};
            switch (this.getDimOrder()) {
                case THEANO: // Theano is channels first
                    if (this.kerasMajorVersion == 1) {
                        targetShape = new int[] {targetShape[1], targetShape[0], targetShape[2]};
                    }
                    preprocessor = new ReshapePreprocessor(inputShape, targetShape);
                    break;
                case NONE: // TF is now the default, channels last
                case TENSORFLOW:
                    if (inputShape[0] != targetShape[0]) {
                        targetShape = new int[] {targetShape[2], targetShape[0], targetShape[1]};
                    }
                    preprocessor = new ReshapePreprocessor(inputShape, targetShape);
            }
        } else if (inputType[0] instanceof InputType.InputTypeRecurrent) {
            InputType.InputTypeRecurrent it = (InputType.InputTypeRecurrent) inputType[0];
            int[] inputShape = new int[]{it.getSize(), it.getTimeSeriesLength()};
            preprocessor = new ReshapePreprocessor(inputShape, this.targetShape);
        } else if (inputType[0] instanceof InputType.InputTypeFeedForward) {
            InputType.InputTypeFeedForward it = (InputType.InputTypeFeedForward) inputType[0];
            int[] inputShape = new int[]{it.getSize()};
            preprocessor = new ReshapePreprocessor(inputShape, this.targetShape);
        }
        return preprocessor;
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
                    "Keras Reshape layer accepts only one input (received " + inputType.length + ")");
        ReshapePreprocessor reshape = (ReshapePreprocessor) getInputPreprocessor(inputType);
        return reshape.getOutputType(inputType[0]);
    }
}
