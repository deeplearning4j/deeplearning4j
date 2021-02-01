/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.modelimport.keras.layers.core;


import lombok.val;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.preprocessors.ReshapePreprocessor;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;

import java.util.List;
import java.util.Map;

/**
 * Imports Reshape layer from Keras
 *
 * @author Max Pumperla
 */
public class KerasReshape extends KerasLayer {

    private long[] targetShape;

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

    private long[] listToLongArray(List<Integer> list) {
        long[] retVal = new long[list.size()];
        for (int i = 0; i < list.size(); ++i) {
            retVal[i] = list.get(i);
        }
        return retVal;
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
            this.targetShape = listToLongArray(targetShapeList);
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
            val inputShape = new long[]{it.getChannels(), it.getHeight(), it.getWidth()};
            val dimOrder = getDimOrder();
            if (dimOrder == DimOrder.THEANO || dimOrder == DimOrder.NONE && kerasMajorVersion == 1) {
                if (targetShape.length == 2) { // edge caseKeras
                    targetShape = new long[]{targetShape[1], targetShape[0]};
                } else {
                    targetShape = new long[]{targetShape[1], targetShape[0], targetShape[2]};
                }
                preprocessor = new ReshapePreprocessor(inputShape, targetShape, false, CNN2DFormat.NCHW);
            } else { // (dimOrder == DimOrder.TENSORFLOW || dimOrder == DimOrder.NONE && kerasMajorVersion == 2)
                preprocessor = new ReshapePreprocessor(inputShape, targetShape, false, CNN2DFormat.NHWC);
            }

        } else if (inputType[0] instanceof InputType.InputTypeConvolutional3D) {
            InputType.InputTypeConvolutional3D it = (InputType.InputTypeConvolutional3D) inputType[0];
            val inputShape = new long[] { it.getDepth(), it.getHeight(), it.getWidth(), it.getChannels() };
            val dimOrder = getDimOrder();
            if (dimOrder == DimOrder.THEANO || dimOrder == DimOrder.NONE && kerasMajorVersion == 1) {
                if (targetShape.length == 3) { // Keras edge case
                    targetShape = new long[] { targetShape[1], targetShape[0], targetShape[2] };
                } else {
                    targetShape = new long[] { targetShape[2], targetShape[1], targetShape[0], targetShape[3] };
                }
                preprocessor = new ReshapePreprocessor(inputShape, targetShape, false, null);
            } else {
                if (inputShape[0] != targetShape[0])
                    targetShape = new long[] { targetShape[3], targetShape[0], targetShape[1], targetShape[2] };
                preprocessor = new ReshapePreprocessor(inputShape, targetShape, false, null);
            }
        }  else if (inputType[0] instanceof InputType.InputTypeRecurrent) {
            InputType.InputTypeRecurrent it = (InputType.InputTypeRecurrent) inputType[0];
            val inputShape = new long[]{it.getSize(), it.getTimeSeriesLength()};
            preprocessor = new ReshapePreprocessor(inputShape, this.targetShape, false, null);
        } else if (inputType[0] instanceof InputType.InputTypeFeedForward) {
            InputType.InputTypeFeedForward it = (InputType.InputTypeFeedForward) inputType[0];
            val inputShape = new long[]{it.getSize()};
            if (targetShape.length == 3) {
                targetShape = targetShapeForDimOrder(inputShape, targetShape);
            }
            preprocessor = new ReshapePreprocessor(inputShape, this.targetShape, false, null);
        }
        return preprocessor;
    }

    public long[] targetShapeForDimOrder(long[] inputShape, long[] targetShape) {
        if (dimOrder == DimOrder.THEANO || dimOrder == DimOrder.NONE && kerasMajorVersion == 1) {
            if (dimOrder == DimOrder.NONE) {
                targetShape = new long[]{targetShape[2], targetShape[0], targetShape[1]};
            } else {
                targetShape = new long[]{targetShape[1], targetShape[2], targetShape[0]};
            }
        } else {
            if (inputShape[0] != targetShape[0]) {
                targetShape = new long[]{targetShape[0], targetShape[1], targetShape[2]};
            }
        }
        return targetShape;
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
