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
import org.deeplearning4j.nn.modelimport.keras.preprocessors.PermutePreprocessor;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Imports Permute layer from Keras
 *
 * @author Max Pumperla
 */
public class KerasPermute extends KerasLayer {

    private int[] permutationIndices;


    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasPermute(Map<String, Object> layerConfig)
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
    public KerasPermute(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        String permutationInfo = "dims";
        if (innerConfig.containsKey(permutationInfo)) {
            @SuppressWarnings("unchecked")
            List<Integer> targetShapeList = (List<Integer>) innerConfig.get(permutationInfo);
            this.permutationIndices = ArrayUtil.toArray(targetShapeList);
        }

    }

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
     * @see InputPreProcessor
     */
    @Override
    public InputPreProcessor getInputPreprocessor(InputType... inputType) throws
            InvalidKerasConfigurationException {
        if (inputType.length > 1)
            throw new InvalidKerasConfigurationException(
                    "Keras Permute layer accepts only one input (received " + inputType.length + ")");
        InputPreProcessor preprocessor = null;
        if (inputType[0] instanceof InputType.InputTypeConvolutional) {
            switch (this.getDimOrder()) {
                case THEANO:
                    if (Arrays.equals(permutationIndices, new int[]{1, 3, 2})) // channels first, swapping H and W.
                        preprocessor = new PermutePreprocessor(permutationIndices);
                    else
                        throw new InvalidKerasConfigurationException("Attempting to permute dimensions other than" +
                                "spatial dimensions (height and width), got " + Arrays.toString(permutationIndices));
                    break;
                case NONE: // TF by default
                case TENSORFLOW:
                    if (Arrays.equals(permutationIndices, new int[]{2, 1, 3})) // channels last, swapping H and W
                        preprocessor = new PermutePreprocessor(new int[]{1, 3, 2}); // DL4J is channels first
                    else
                        throw new InvalidKerasConfigurationException("Attempting to permute dimensions other than" +
                                "spatial dimensions (height and width) in Permute layer, got "
                                + Arrays.toString(permutationIndices));
            }
        } else if (inputType[0] instanceof InputType.InputTypeRecurrent) {
            if (Arrays.equals(permutationIndices, new int[] {2, 1}))
                preprocessor = new PermutePreprocessor(permutationIndices);
            else
                throw new InvalidKerasConfigurationException("For RNN type input data, permutation dims have to be" +
                        "(2, 1) in Permute layer, got " + Arrays.toString(permutationIndices));
        } else if (inputType[0] instanceof InputType.InputTypeFeedForward) {
            preprocessor = null;
        } else {
            throw new InvalidKerasConfigurationException("Input type not supported: " + inputType[0]);
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
                    "Keras Permute layer accepts only one input (received " + inputType.length + ")");
        PermutePreprocessor reshape = (PermutePreprocessor) getInputPreprocessor(inputType);
        return reshape.getOutputType(inputType[0]);
    }
}
