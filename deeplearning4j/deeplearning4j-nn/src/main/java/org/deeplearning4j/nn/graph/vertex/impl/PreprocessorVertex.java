/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.graph.vertex.impl;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

/** PreprocessorVertex is a simple adaptor class that allows a {@link InputPreProcessor} to be used in a ComputationGraph
 * GraphVertex, without it being associated with a layer.
 * @author Alex Black
 */
public class PreprocessorVertex extends BaseGraphVertex {

    private InputPreProcessor preProcessor;

    public PreprocessorVertex(ComputationGraph graph, String name, int vertexIndex, InputPreProcessor preProcessor) {
        this(graph, name, vertexIndex, null, null, preProcessor);
    }

    public PreprocessorVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
                    VertexIndices[] outputVertices, InputPreProcessor preProcessor) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
        this.preProcessor = preProcessor;
    }

    @Override
    public boolean hasLayer() {
        return false;
    }

    @Override
    public Layer getLayer() {
        return null;
    }

    @Override
    public INDArray doForward(boolean training, LayerWorkspaceMgr workspaceMgr) {
        return preProcessor.preProcess(inputs[0], graph.batchSize(), workspaceMgr);
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        return new Pair<>(null, new INDArray[] {preProcessor.backprop(epsilon, graph.batchSize(), workspaceMgr)});
    }

    @Override
    public String toString() {
        return "PreprocessorVertex(id=" + this.getVertexIndex() + ",name=\"" + this.getVertexName() + "\",preProcessor="
                        + preProcessor.toString() + ")";
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if (backpropGradientsViewArray != null)
            throw new RuntimeException("Vertex does not have gradients; gradients view array cannot be set here");
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState,
                    int minibatchSize) {
        //No op
        if (maskArrays == null || maskArrays.length == 0) {
            return null;
        }

        return preProcessor.feedForwardMaskArray(maskArrays[0], currentMaskState, minibatchSize);
    }
}
