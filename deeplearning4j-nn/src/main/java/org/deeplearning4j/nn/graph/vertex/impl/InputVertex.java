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

import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * An InputVertex simply holds activations and passes them through to other layers when required.
 *
 * @author Alex Black
 */
public class InputVertex extends BaseGraphVertex {


    public InputVertex(String name, int vertexIndex, int numInputs) {
        super(name, vertexIndex, numInputs);
    }

    @Override
    public int numInputs(){
        return 1;
    }

    @Override
    public boolean isOutputVertex() {
        return false;
    }

    @Override
    public Activations activate(boolean training) {
        return input;
    }

    @Override
    public Gradients backpropGradient(Gradients gradient) {
        throw new UnsupportedOperationException("Cannot do backward pass for InputVertex");
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if (backpropGradientsViewArray != null)
            throw new RuntimeException("Vertex does not have gradients; gradients view array cannot be set here");
    }

    @Override
    public String toString() {
        return "InputVertex(id=" + getIndex() + ",name=\"" + vertexName + "\")";
    }
}
