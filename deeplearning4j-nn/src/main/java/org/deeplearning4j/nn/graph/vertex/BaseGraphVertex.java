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

package org.deeplearning4j.nn.graph.vertex;

import lombok.Data;
import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

/** BaseGraphVertex defines a set of common functionality for GraphVertex instances.
 */
@Data
public abstract class BaseGraphVertex extends AbstractLayer {

    protected ComputationGraph graph;

    protected String vertexName;

    /** The index of this vertex */
    protected int vertexIndex;

    protected INDArray[] inputs;

    //Set outputVertex to true when Layer is an OutputLayer, OR For use in specialized situations like reinforcement learning
    // For RL situations, this Layer insn't an OutputLayer, but is the last layer in a graph, that gets its error/epsilon
    // passed in externally
    @Setter @Getter
    protected boolean outputVertex;

    protected BaseGraphVertex(ComputationGraph graph, String name, int vertexIndex, int numInputs) {
        this.graph = graph;
        this.vertexName = name;
        this.vertexIndex = vertexIndex;

        this.inputs = new INDArray[numInputs];
    }

    @Override
    public int getIndex() {
        return vertexIndex;
    }

    @Override
    public String getName(){
        return vertexName;
    }

    @Override
    public int numInputs() {
        return inputs.length;
    }

    @Override
    public void setInput(int inputNumber, INDArray input) {
        if (inputNumber >= numInputs()) {
            throw new IllegalArgumentException("Invalid input number: got " + inputNumber + ", inputs must be in range" +
                    "0 to " + (numInputs()-1) + " inclusive");
        }
        inputs[inputNumber] = input;
    }

    @Override
    public void clear() {
        for (int i = 0; i < inputs.length; i++) {
            inputs[i] = null;
        }
    }

    @Override
    public abstract String toString();

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public void clearNoiseWeightParams() {

    }

    protected abstract Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState,
                                                                              int minibatchSize);

    @Override
    public INDArray getInput(int inputNumber){
        //TODO have both input and inputs fields...
        return this.inputs[inputNumber];
    }

    public boolean canDoForward(){
        for( int i=0; i<numInputs(); i++ ){
            if(getInput(i) == null){
                return false;
            }
        }
        return true;
    }

    public boolean canDoBackward(){
        return true;    //TODO remove
    }
}
