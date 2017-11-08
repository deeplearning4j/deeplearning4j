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

package org.deeplearning4j.nn.graph.vertex.impl;

import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

/**
 * A ScaleVertex is used to scale the size of activations of a single layer<br>
 * For example, ResNet activations can be scaled in repeating blocks to keep variance
 * under control.
 *
 * @author Justin Long (@crockpotveggies)
 */
public class ScaleVertex extends BaseGraphVertex {

    private double scaleFactor;

    public ScaleVertex(String name, int vertexIndex, int numInputs, double scaleFactor) {
        super(name, vertexIndex, numInputs);
        this.scaleFactor = scaleFactor;
    }

    @Override
    public Activations activate(boolean training) {
        if (input == null || input.anyActivationsNull())
            throw new IllegalStateException("Cannot do forward pass: inputs not set (ScaleVertex " + vertexName
                            + " idx " + getIndex() + ")");

        if (input.size() > 1)
            throw new IllegalArgumentException(
                            "ScaleVertex (name " + vertexName + " idx " + getIndex() + ") only supports 1 input.");

        INDArray prod = input.get(0).dup();
        prod.muli(scaleFactor);

        Pair<INDArray, MaskState> masks = feedForwardMaskArrays(new INDArray[]{input.getMask(0)}, MaskState.Active, getInputMiniBatchSize());
        return ActivationsFactory.getInstance().create(prod, masks.getFirst(), masks.getSecond());
    }

    @Override
    public Gradients backpropGradient(Gradients gradient) {
        if (gradient == null || gradient.get(0) == null)
            throw new IllegalStateException("Cannot do backward pass: activation gradients not available (null) " + layerId());

        INDArray epsilon = gradient.get(0);
        epsilon.muli(scaleFactor);

        return gradient;
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if (backpropGradientsViewArray != null)
            throw new RuntimeException(
                            "Vertex does not have gradients; gradients view array cannot be set here (ScaleVertex "
                                            + vertexName + " idx " + getIndex() + ")");
    }

    @Override
    public String toString() {
        return "ScaleVertex(id=" + this.getIndex() + ",name=\"" + this.getName() + "\",scaleFactor="
                        + scaleFactor + ")";
    }

    protected Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState,
                    int minibatchSize) {
        //No op
        if (maskArrays == null || maskArrays.length == 0) {
            return null;
        }

        return new Pair<>(maskArrays[0], currentMaskState);
    }
}
