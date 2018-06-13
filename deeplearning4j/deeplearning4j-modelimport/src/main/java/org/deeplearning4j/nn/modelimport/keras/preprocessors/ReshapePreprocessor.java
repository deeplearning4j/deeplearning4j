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
package org.deeplearning4j.nn.modelimport.keras.preprocessors;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.extern.slf4j.Slf4j;

import lombok.val;
import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.preprocessor.BaseInputPreProcessor;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;

import static org.nd4j.linalg.util.ArrayUtil.prodLong;

/**
 * Generic reshape preprocessor
 *
 * @author Max Pumperla
 */
@Data
@Slf4j
@EqualsAndHashCode(callSuper = false)
public class ReshapePreprocessor extends BaseInputPreProcessor {

    private long[] inputShape;
    private long[] targetShape;
    private boolean hasMiniBatchDimension = false;
    private int miniBatchSize;

    public ReshapePreprocessor(long[] inputShape, long[] targetShape) {
        this.inputShape = inputShape;
        this.targetShape = targetShape;
    }

    private static int prod(int[] array) {
        int prod = 1;
        for (int i : array) {
            prod *= i;
        }
        return prod;
    }

    private static long[] prependMiniBatchSize(long[] shape, long miniBatchSize) {
        int shapeLength = shape.length;
        val miniBatchShape = new long[shapeLength + 1];
        for (int i = 0; i < miniBatchShape.length; i++) {
            if (i == 0)
                miniBatchShape[i] = miniBatchSize;
            else
                miniBatchShape[i] = shape[i - 1];
        }
        return miniBatchShape;
    }

    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        // the target shape read from a keras config does not have mini-batch size
        // included. We prepend it here dynamically.
        if (targetShape.length + 1 == input.shape().length) {
            targetShape = prependMiniBatchSize(targetShape, miniBatchSize);
            inputShape = prependMiniBatchSize(inputShape, miniBatchSize);
            this.hasMiniBatchDimension = true;
            this.miniBatchSize = miniBatchSize;
        }
        if (this.miniBatchSize != miniBatchSize) {
            targetShape = prependMiniBatchSize(ArrayUtils.subarray(targetShape, 1, targetShape.length), miniBatchSize);
            inputShape = prependMiniBatchSize(ArrayUtils.subarray(inputShape, 1, targetShape.length), miniBatchSize);
            this.miniBatchSize = miniBatchSize;
        }
        if (prodLong(input.shape()) == prodLong((targetShape))) {
            if(input.ordering() != 'c' || !Shape.hasDefaultStridesForShape(input)){
                input = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, 'c');
            }
            val shp =  inputShape;
            val outShp = targetShape;
            return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, input.reshape(this.targetShape));
        } else {
            throw new IllegalStateException("Input shape " + Arrays.toString(input.shape())
                    + " and output shape" + Arrays.toString(inputShape) + " do not match");
        }
    }

    @Override
    public INDArray backprop(INDArray output, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        if (!Arrays.equals(targetShape, output.shape())) {
            throw new IllegalStateException("Unexpected output shape" + Arrays.toString(output.shape())
                    + " (expected to be " + Arrays.toString(targetShape) + ")");
        }
        if (prodLong(output.shape()) == prodLong((targetShape))) {
            if(output.ordering() != 'c' || !Shape.hasDefaultStridesForShape(output)){
                output = workspaceMgr.dup(ArrayType.ACTIVATIONS, output, 'c');
            }
            return workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, output.reshape(this.inputShape));
        } else {
            throw new IllegalStateException("Output shape" + Arrays.toString(output.shape())
                    + " and input shape" + Arrays.toString(targetShape) + " do not match");
        }
    }

    @Override
    public InputType getOutputType(InputType inputType) throws InvalidInputTypeException {

        val shape = hasMiniBatchDimension ? targetShape : prependMiniBatchSize(targetShape, 0);
        switch (shape.length) {
            case 2:
                return InputType.feedForward(shape[1]);
            case 3:
                return InputType.recurrent(shape[2], shape[1]);
            case 4:
                if (inputShape.length == 1)
                    return InputType.convolutional(shape[1], shape[2], shape[3]);
                else
                    return InputType.convolutional(shape[2], shape[3], shape[1]);
            default:
                throw new UnsupportedOperationException(
                        "Cannot infer input type for reshape array " + Arrays.toString(shape));
        }
    }
}