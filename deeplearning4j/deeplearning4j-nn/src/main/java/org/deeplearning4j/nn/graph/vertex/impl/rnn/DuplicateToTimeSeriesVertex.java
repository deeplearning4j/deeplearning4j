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

package org.deeplearning4j.nn.graph.vertex.impl.rnn;

import lombok.val;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

/**DuplicateToTimeSeriesVertex is a vertex that goes from 2d activations to a 3d time series activations, by means of
 * duplication. That is, given a 2d input with shape [numExamples,nIn] duplicate each row to give output of
 * [numExamples,nIn,timeSeriesLength], where the activations are the same for all time steps.<br>
 * This method is used for example in sequence to sequence models.<br>
 * <b>Note</b>: The length of the output time series (number of time steps) is determined by means of referencing one of the
 * inputs in the ComputationGraph. That is: Because the length of the time series may differ at runtime, we generally want the number
 * of time steps to match some other input; here, we are specifying the length of the output time series to be the same as
 * one of the input time series<br>
 * @author Alex Black
 */
public class DuplicateToTimeSeriesVertex extends BaseGraphVertex {

    private String inputName;
    private int inputVertexIndex;

    public DuplicateToTimeSeriesVertex(ComputationGraph graph, String name, int vertexIndex, String inputVertexName) {
        this(graph, name, vertexIndex, null, null, inputVertexName);
    }

    public DuplicateToTimeSeriesVertex(ComputationGraph graph, String name, int vertexIndex,
                    VertexIndices[] inputVertices, VertexIndices[] outputVertices, String inputName) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
        this.inputName = inputName;
        this.inputVertexIndex = graph.getConfiguration().getNetworkInputs().indexOf(inputName);
        if (inputVertexIndex == -1)
            throw new IllegalArgumentException("Invalid input name: \"" + inputName + "\" not found in list "
                            + "of network inputs (" + graph.getConfiguration().getNetworkInputs() + ")");
    }

    @Override
    public boolean hasLayer() {
        return false;
    }

    @Override
    public boolean isOutputVertex() {
        return false;
    }

    @Override
    public Layer getLayer() {
        return null;
    }

    @Override
    public INDArray doForward(boolean training, LayerWorkspaceMgr workspaceMgr) {

        //First: work out the time series length
        val tsLength = graph.getInput(inputVertexIndex).size(2);
        val outShape = new long[] {inputs[0].size(0), inputs[0].size(1), tsLength};

        INDArray out = workspaceMgr.create(ArrayType.ACTIVATIONS, outShape, 'f');
        for (int i = 0; i < tsLength; i++) {
            out.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(i)}, inputs[0]);
        }
        return out;
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        //Because we duplicated for each time step: simply need to sum along time for errors/epsilons
        INDArray ret = epsilon.sum(workspaceMgr.create(ArrayType.ACTIVATION_GRAD, epsilon.size(0), epsilon.size(1)), 2);
        return new Pair<>(null, new INDArray[] {ret});
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if (backpropGradientsViewArray != null)
            throw new RuntimeException("Vertex does not have gradients; gradients view array cannot be set here");
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState,
                    int minibatchSize) {
        //Present for all time steps, or as per the corresponding input mask (if present)
        INDArray[] allMasks = graph.getInputMaskArrays();
        if (allMasks == null || allMasks[inputVertexIndex] == null) {
            //No mask
            return null;
        }
        return new Pair<>(allMasks[inputVertexIndex], MaskState.Active);
    }

    @Override
    public String toString() {
        return "DuplicateToTimeSeriesVertex(inputName=" + inputName + ")";
    }
}
