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

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.api.layers.RecurrentLayer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.deeplearning4j.nn.layers.FrozenLayer;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;

/**
 * LayerVertex is a GraphVertex with a neural network Layer (and, optionally an {@link InputPreProcessor}) in it
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class LayerVertex extends BaseGraphVertex {

    private Layer layer;
    private final InputPreProcessor layerPreProcessor;
    private boolean setLayerInput;

    /**
     * Create a network input vertex:
     */
    public LayerVertex(ComputationGraph graph, String name, int vertexIndex, Layer layer,
                    InputPreProcessor layerPreProcessor, boolean outputVertex) {
        this(graph, name, vertexIndex, null, null, layer, layerPreProcessor, outputVertex);
    }

    public LayerVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
                    VertexIndices[] outputVertices, Layer layer, InputPreProcessor layerPreProcessor,
                    boolean outputVertex) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
        this.graph = graph;
        this.vertexName = name;
        this.vertexIndex = vertexIndex;
        this.inputVertices = inputVertices;
        this.outputVertices = outputVertices;
        this.layer = layer;
        this.layerPreProcessor = layerPreProcessor;
        this.outputVertex = outputVertex;

        this.inputs = new INDArray[(inputVertices != null ? inputVertices.length : 0)];
    }

    @Override
    public boolean hasLayer() {
        return true;
    }

    public void setLayerAsFrozen() {
        if (this.layer instanceof FrozenLayer)
            return;

        this.layer = new FrozenLayer(this.layer);
        this.layer.conf().getLayer().setLayerName(vertexName);
    }

    @Override
    public boolean isOutputVertex() {
        return outputVertex || layer instanceof BaseOutputLayer;
    }

    @Override
    public Layer getLayer() {
        return layer;
    }

    @Override
    public INDArray doForward(boolean training) {
        if (!canDoForward())
            throw new IllegalStateException("Cannot do forward pass: all inputs not set");
        return layer.activate(training);
    }

    protected void applyPreprocessorAndSetInput(){
        //Apply preprocessor
        INDArray currInput = inputs[0];
        if (layerPreProcessor != null) {
            if (Nd4j.getWorkspaceManager().checkIfWorkspaceExistsAndActive(ComputationGraph.workspaceExternal)
                    && Nd4j.getMemoryManager().getCurrentWorkspace() != Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(ComputationGraph.workspaceExternal)) {
                //WS single, or FF as part of backprop
                //NOTE: we *could* leverage instead (less memory, worse performance), but most preprocessors will only
                //allocate 1 array (i.e., the new output), so this is usually preferable in practice
                try (MemoryWorkspace wsB = Nd4j.getWorkspaceManager()
                        .getWorkspaceForCurrentThread(ComputationGraph.workspaceExternal).notifyScopeBorrowed()) {
                    currInput = layerPreProcessor.preProcess(currInput, graph.batchSize());
                }
            } else {
                currInput = layerPreProcessor.preProcess(currInput, graph.batchSize());
            }
        }
        layer.setInput(currInput);
        setLayerInput = true;
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt) {
        if (!canDoBackward()) {
            if(inputs == null || inputs[0] == null){
                throw new IllegalStateException("Cannot do backward pass: inputs not set. Layer " + vertexName
                        + " (idx " + vertexIndex + ") numInputs " + getNumInputArrays());
            } else {
                throw new IllegalStateException("Cannot do backward pass: all epsilons not set. Layer " + vertexName
                        + " (idx " + vertexIndex + ") numInputs " + getNumInputArrays() + "; numOutputs "
                        + getNumOutputConnections());
            }
        }

        //Edge case: output layer - never did forward pass hence layer.setInput was never called...
        if(!setLayerInput){
            applyPreprocessorAndSetInput();
        }

        Pair<Gradient, INDArray> pair;
        if (tbptt && layer instanceof RecurrentLayer) {
            //Truncated BPTT for recurrent layers
            pair = ((RecurrentLayer) layer).tbpttBackpropGradient(epsilon,
                            graph.getConfiguration().getTbpttBackLength());
        } else {
            //Normal backprop
            pair = layer.backpropGradient(epsilon); //epsTotal may be null for OutputLayers
        }

        if (layerPreProcessor != null) {
            INDArray eps = pair.getSecond();
            eps = layerPreProcessor.backprop(eps, graph.batchSize());
            pair.setSecond(eps);
        }

        //Layers always have single activations input -> always have single epsilon output during backprop
        return new Pair<>(pair.getFirst(), new INDArray[] {pair.getSecond()});
    }

    @Override
    public void setInput(int inputNumber, INDArray input) {
        if (inputNumber > 0)
            throw new IllegalArgumentException(
                            "Invalid input number: LayerVertex instances have only 1 input (got inputNumber = "
                                            + inputNumber + ")");
        inputs[inputNumber] = input;
        setLayerInput = false;
        applyPreprocessorAndSetInput();
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        layer.setBackpropGradientsViewArray(backpropGradientsViewArray);
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState,
                    int minibatchSize) {
        if (maskArrays == null || maskArrays.length == 0) {
            return new Pair<>(null, currentMaskState);
        }

        if (layerPreProcessor != null) {
            Pair<INDArray, MaskState> pair =
                            layerPreProcessor.feedForwardMaskArray(maskArrays[0], currentMaskState, minibatchSize);
            if (pair == null) {
                maskArrays[0] = null;
                currentMaskState = null;
            } else {
                maskArrays[0] = pair.getFirst();
                currentMaskState = pair.getSecond();
            }
        }

        return layer.feedForwardMaskArray(maskArrays[0], currentMaskState, minibatchSize);
    }


    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("LayerVertex(id=").append(vertexIndex).append(",name=\"").append(vertexName).append("\",inputs=")
                        .append(Arrays.toString(inputVertices)).append(",outputs=")
                        .append(Arrays.toString(outputVertices)).append(")");
        return sb.toString();
    }

    @Override
    public boolean canDoBackward() {
        if (!isOutputVertex()) {
            //inputs to frozen layer go unchecked, so could be null
            if (getLayer() instanceof FrozenLayer) {
                return true;
            } else {
                return super.canDoBackward();
            }
        }

        for (INDArray input : inputs) {
            if (input == null) {
                return false;
            }
        }

        if (!(layer instanceof IOutputLayer)) {
            if (epsilon == null) {
                return false;
            }
        }

        return true;
    }

    public double computeScore(double l1, double l2, boolean training){
        if(!(layer instanceof IOutputLayer)){
            throw new UnsupportedOperationException("Cannot compute score: layer is not an output layer (layer class: "
                    + layer.getClass().getSimpleName());
        }
        //Edge case: output layer - never did forward pass hence layer.setInput was never called...
        if(!setLayerInput){
            applyPreprocessorAndSetInput();
        }

        IOutputLayer ol = (IOutputLayer)layer;
        return ol.computeScore(l1, l2, training);
    }

    public INDArray computeScoreForExamples(double l1, double l2){
        if(!(layer instanceof IOutputLayer)){
            throw new UnsupportedOperationException("Cannot compute score: layer is not an output layer (layer class: "
                    + layer.getClass().getSimpleName());
        }
        //Edge case: output layer - never did forward pass hence layer.setInput was never called...
        if(!setLayerInput){
            applyPreprocessorAndSetInput();
        }

        IOutputLayer ol = (IOutputLayer)layer;
        return ol.computeScoreForExamples(l1, l2);
    }

    @Override
    public void migrateInput(){
        layer.migrateInput();
    }
}
