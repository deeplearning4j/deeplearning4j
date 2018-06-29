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
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.impl.LayerVertex;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.util.Collections;
import java.util.Map;

/** BaseGraphVertex defines a set of common functionality for GraphVertex instances.
 */
@Data
public abstract class BaseGraphVertex implements GraphVertex {

    protected ComputationGraph graph;

    protected String vertexName;

    /** The index of this vertex */
    protected int vertexIndex;

    /**A representation of the vertices that are inputs to this vertex (inputs during forward pass)
     * Specifically, if inputVertices[X].getVertexIndex() = Y, and inputVertices[X].getVertexEdgeNumber() = Z
     * then the Zth output of vertex Y is the Xth input to this vertex
     */
    protected VertexIndices[] inputVertices;

    /**A representation of the vertices that this vertex is connected to (outputs duing forward pass)
     * Specifically, if outputVertices[X].getVertexIndex() = Y, and outputVertices[X].getVertexEdgeNumber() = Z
     * then the output of this vertex (there is only one output) is connected to the Zth input of vertex Y
     */
    protected VertexIndices[] outputVertices;

    protected INDArray[] inputs;
    protected INDArray epsilon;

    //Set outputVertex to true when Layer is an OutputLayer, OR For use in specialized situations like reinforcement learning
    // For RL situations, this Layer insn't an OutputLayer, but is the last layer in a graph, that gets its error/epsilon
    // passed in externally
    @Setter @Getter
    protected boolean outputVertex;

    protected BaseGraphVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
                    VertexIndices[] outputVertices) {
        this.graph = graph;
        this.vertexName = name;
        this.vertexIndex = vertexIndex;
        this.inputVertices = inputVertices;
        this.outputVertices = outputVertices;

        this.inputs = new INDArray[(inputVertices != null ? inputVertices.length : 0)];
    }

    @Override
    public String getVertexName() {
        return vertexName;
    }

    @Override
    public int getVertexIndex() {
        return vertexIndex;
    }

    @Override
    public int getNumInputArrays() {
        return (inputVertices == null ? 0 : inputVertices.length);
    }

    @Override
    public int getNumOutputConnections() {
        return (outputVertices == null ? 0 : outputVertices.length);
    }

    /**A representation of the vertices that are inputs to this vertex (inputs duing forward pass)<br>
     * Specifically, if inputVertices[X].getVertexIndex() = Y, and inputVertices[X].getVertexEdgeNumber() = Z
     * then the Zth output of vertex Y is the Xth input to this vertex
     */
    @Override
    public VertexIndices[] getInputVertices() {
        return inputVertices;
    }

    @Override
    public void setInputVertices(VertexIndices[] inputVertices) {
        this.inputVertices = inputVertices;
        this.inputs = new INDArray[(inputVertices != null ? inputVertices.length : 0)];
    }

    /**A representation of the vertices that this vertex is connected to (outputs duing forward pass)
     * Specifically, if outputVertices[X].getVertexIndex() = Y, and outputVertices[X].getVertexEdgeNumber() = Z
     * then the Xth output of this vertex is connected to the Zth input of vertex Y
     */
    @Override
    public VertexIndices[] getOutputVertices() {
        return outputVertices;
    }

    @Override
    public void setOutputVertices(VertexIndices[] outputVertices) {
        this.outputVertices = outputVertices;
    }

    @Override
    public boolean isInputVertex() {
        return false;
    }

    @Override
    public void setInput(int inputNumber, INDArray input, LayerWorkspaceMgr workspaceMgr) {
        if (inputNumber >= getNumInputArrays()) {
            throw new IllegalArgumentException("Invalid input number");
        }
        inputs[inputNumber] = input;
    }

    @Override
    public void setEpsilon(INDArray epsilon) {
        this.epsilon = epsilon;
    }

    @Override
    public void clear() {
        for (int i = 0; i < inputs.length; i++) {
            inputs[i] = null;
        }
        epsilon = null;
        if(getLayer() != null){
            getLayer().clear();
        }
    }

    @Override
    public boolean canDoForward() {
        for (INDArray input : inputs) {
            if (input == null) {
                return false;
            }
        }
        return true;
    }

    @Override
    public boolean canDoBackward() {
        for (INDArray input : inputs) {
            if (input == null) {
                return false;
            }
        }
        return epsilon != null;
    }

    @Override
    public INDArray getEpsilon() {
        return epsilon;
    }

    @Override
    public abstract String toString();

    @Override
    public void setLayerAsFrozen() {
        if (!(this instanceof LayerVertex)) {
            throw new IllegalArgumentException("Cannot set non layer vertices as frozen");
        }
    }

    @Override
    public void clearVertex() {
        clear();
    }

    @Override
    public Map<String, INDArray> paramTable(boolean backpropOnly) {
        return Collections.emptyMap();
    }
}
