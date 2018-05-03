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
import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.preprocessor.BaseInputPreProcessor;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * Permute preprocessor
 *
 * @author Max Pumperla
 */
@Data
@Slf4j
@EqualsAndHashCode(callSuper = false)
public class PermutePreprocessor extends BaseInputPreProcessor {

    private int[] permutationIndices;
    private int[] inverseIndices = null;
    private int[] outShape;
    private boolean hasLeadingDimension = false;

    public PermutePreprocessor(int[] permutationIndices) {
        this.permutationIndices = permutationIndices;
    }


    private static int[] prependZero(int[] shape) {
        int shapeLength = shape.length;
        int[] augmentedShape = new int[shapeLength + 1];
        for (int i = 0; i < augmentedShape.length; i++) {
            if (i == 0)
                augmentedShape[i] = 0;
            else
                augmentedShape[i] = shape[i - 1];
        }
        return augmentedShape;
    }

    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        if (permutationIndices.length + 1 == input.shape().length) {
            permutationIndices = prependZero(permutationIndices);
            this.hasLeadingDimension = true;
        }
        if (input.ordering() != 'c' || !Shape.hasDefaultStridesForShape(input)) {
            input = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, 'c');
        }
        INDArray output = workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, input.permute(this.permutationIndices));
        outShape = output.shape();
        return output;
    }

    @Override
    public INDArray backprop(INDArray output, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        if (!Arrays.equals(permutationIndices, output.shape())) {
            throw new IllegalStateException("Unexpected output shape" + Arrays.toString(output.shape())
                    + " (expected to be " + Arrays.toString(permutationIndices) + ")");
        }
        if (output.ordering() != 'c' || !Shape.hasDefaultStridesForShape(output)) {
            output = workspaceMgr.dup(ArrayType.ACTIVATIONS, output, 'c');
        }
        if (inverseIndices == null) {
            INDArray inIndices = Nd4j.create(permutationIndices);
            INDArray outIndices = Nd4j.zerosLike(inIndices);
            DynamicCustomOp invert = DynamicCustomOp.builder("invert_permutation")
                    .addInputs(inIndices)
                    .addOutputs(outIndices).build();
        }

        return workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, output.permute(inverseIndices));

    }

    @Override
    public InputType getOutputType(InputType inputType) throws InvalidInputTypeException {
        switch (outShape.length) {
            case 2:
                return InputType.feedForward(outShape[1]);
            case 3:
                return InputType.recurrent(outShape[1]);
            case 4:
                return InputType.convolutional(outShape[1], outShape[2], outShape[3]);
            default:
                throw new UnsupportedOperationException(
                        "Cannot infer input type for reshape array " + Arrays.toString(outShape));
        }
    }
}