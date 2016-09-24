/*
 *
 *  * Copyright 2015 Skymind,Inc.
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

package org.deeplearning4j.nn.conf.preprocessor;

import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;

/**Reshape post processor.<br>
 * Used to reshape activations on forward pass.<br>
 * Also (optionally, if fromShape != null) used to reshape, weights*deltas
 * during backward pass. Otherwise, no changes are made during backward pass
 *
 * @author Adam Gibson
 */
@Data @EqualsAndHashCode(callSuper=false)
@Deprecated
public class ReshapePreProcessor extends BaseInputPreProcessor {
    private int[] fromShape;	//Epsilons: To this shape in backward pass
    private int[] toShape;		//Activations: To this shape in forward pass
    private boolean dynamic=true;

    /**
     * @param fromShape May be null. If null: no change/op during backward pass.
     * @param toShape The shape that activations are reshaped to
     * @param dynamic Infer the number of examples or not
     * Otherwise fromShape is the shape that epsilons (weights*deltas or equiv.)
     *  are reshaped to by backprop(...)
     */
    @JsonCreator
    public ReshapePreProcessor(@JsonProperty("fromShape") int[] fromShape,
                               @JsonProperty("toShape") int[] toShape,
                               @JsonProperty("dynamic") boolean dynamic){
        this.fromShape = fromShape;
        this.toShape = toShape;
        this.dynamic = dynamic;
    }

    public ReshapePreProcessor(int... toShape) {
        this(null, toShape, true);
    }
    public ReshapePreProcessor(int[] fromShape, int[] toShape) {
        this(fromShape, toShape, true);
    }

    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize) {
        if (dynamic) fromShape[0] = input.shape()[0];
        if (input.shape().length == toShape.length) return input;
        return input.reshape(toShape);
    }

    @Override
    public INDArray backprop(INDArray output, int miniBatchSize) {
        if( fromShape == null || output.shape().length == fromShape.length)
            return output;	//no-op
        if(output.length() != ArrayUtil.prod(fromShape))
            throw new IllegalStateException("Illegal shape");
        return output.reshape(fromShape);
    }

    @Override
    public ReshapePreProcessor clone() {
        ReshapePreProcessor clone = (ReshapePreProcessor) super.clone();
        if(clone.fromShape != null) clone.fromShape = clone.fromShape.clone();
        if(clone.toShape != null) clone.toShape = clone.toShape.clone();
        return clone;
    }

    @Override
    public InputType getOutputType(InputType inputType) {
        switch (toShape.length){
            case 2:
            case 3:
                return InputType.feedForward(toShape[1]);
            case 4:
                return InputType.convolutional(toShape[3],toShape[2],toShape[1]);
            default:
                throw new IllegalStateException();
        }
    }
}
