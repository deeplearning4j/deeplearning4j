/*-
 *
 *  * Copyright 2016 Skymind,Inc.
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

package org.deeplearning4j.nn.conf.graph;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;
import java.util.Collection;

/**
 * Adds the ability to reshape and flatten the tensor in the computation graph.<br>
 * NOTE: This class should only be used if you know exactly what you are doing with reshaping activations.
 * Use preprocessors such as {@link org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor} and
 * {@link org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor} for most cases.
 *
 * @author Justin Long (crockpotveggies)
 */
@Data
@EqualsAndHashCode(callSuper = false)
public class ReshapeVertex extends BaseGraphVertex {
    public static final char DEFAULT_RESHAPE_ORDER = 'c';

    protected char reshapeOrder = 'c';
    protected int[] newShape;
    protected int[] maskShape;

    public ReshapeVertex(int... newShape){
        this(DEFAULT_RESHAPE_ORDER, newShape, null);
    }

    public ReshapeVertex(@JsonProperty("reshapeOrder") char reshapeOrder, @JsonProperty("newShape") int[] newShape,
                         @JsonProperty("maskShape") int[] maskShape) {
        this.reshapeOrder = reshapeOrder;
        this.newShape = newShape;
        this.maskShape = maskShape;
    }

    @Override
    public ReshapeVertex clone() {
        return new ReshapeVertex(newShape);
    }

    @Override
    public int minInputs() {
        return 1;
    }

    @Override
    public int maxInputs() {
        return 1;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf,
                             Collection<IterationListener> iterationListeners,
                             String name, int idx, int numInputs, INDArray layerParamsView,
                             boolean initializeParams) {
        return new org.deeplearning4j.nn.graph.vertex.impl.ReshapeVertex(name, idx, numInputs, reshapeOrder, newShape, maskShape);
    }

    @Override
    public InputType[] getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        //Infer output shape from specified shape:
        InputType ret;
        switch (newShape.length) {
            case 2:
                ret = InputType.feedForward(newShape[1]);
                break;
            case 3:
                ret = InputType.recurrent(newShape[1]);
                break;
            case 4:
                ret = InputType.convolutional(newShape[2], newShape[3], newShape[1]); //[mb,d,h,w] for activations
                break;
            default:
                throw new UnsupportedOperationException(
                                "Cannot infer input type for reshape array " + Arrays.toString(newShape));
        }
        return new InputType[]{ret};
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType... inputTypes) {
        //Assume it's a reshape-with-copy op. In this case: memory use is accounted for in activations
        InputType outputType = getOutputType(-1, inputTypes)[0];
        return new LayerMemoryReport.Builder(null, ReshapeVertex.class, inputTypes[0], outputType).standardMemory(0, 0) //No params
                        .workingMemory(0, 0, 0, 0).cacheMemory(0, 0) //No caching
                        .build();
    }
}
