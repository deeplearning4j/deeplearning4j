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

import lombok.val;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.Or;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.common.primitives.Pair;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.util.Arrays;

/** A MergeVertex is used to combine the activations of two or more layers/GraphVertex by means of concatenation/merging.<br>
 * Exactly how this is done depends on the type of input.<br>
 * For 2d (feed forward layer) inputs: MergeVertex([numExamples,layerSize1],[numExamples,layerSize2]) -> [numExamples,layerSize1 + layerSize2]<br>
 * For 3d (time series) inputs: MergeVertex([numExamples,layerSize1,timeSeriesLength],[numExamples,layerSize2,timeSeriesLength])
 *      -> [numExamples,layerSize1 + layerSize2,timeSeriesLength]<br>
 * For 4d (convolutional) inputs: MergeVertex([numExamples,depth1,width,height],[numExamples,depth2,width,height])
 *      -> [numExamples,depth1 + depth2,width,height]<br>
 * @author Alex Black
 */
public class MergeVertex extends BaseGraphVertex {

    private long[][] forwardPassShapes;
    private int fwdPassRank;
    private int mergeAxis;

    public MergeVertex(ComputationGraph graph, String name, int vertexIndex, DataType dataType, int mergeAxis) {
        this(graph, name, vertexIndex, null, null, dataType, mergeAxis);
    }

    public MergeVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
                    VertexIndices[] outputVertices, DataType dataType, int mergeAxis) {
        super(graph, name, vertexIndex, inputVertices, outputVertices, dataType);
        this.mergeAxis = mergeAxis;
    }

    @Override
    public String toString() {
        return "MergeVertex(id=" + this.getVertexIndex() + ",name=\"" + this.getVertexName() + "\")";
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
        if (!canDoForward())
            throw new IllegalStateException("Cannot do forward pass: inputs not set");

        if (inputs.length == 1) {
            //No-op case
            val shape = inputs[0].shape();
            forwardPassShapes = new long[][] {Arrays.copyOf(shape, shape.length)};
            return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, inputs[0]);
        }

        INDArray[] in = new INDArray[inputs.length];
        for( int i=0; i<in.length; i++ ){
            in[i] = inputs[i].castTo(dataType); //No-op if correct type
        }

        forwardPassShapes = new long[in.length][0];
        val nExamples = in[0].size(0);
        fwdPassRank = in[0].rank();
        for (int i = 0; i < in.length; i++) {
            val currShape = in[i].shape();
            if (fwdPassRank != currShape.length) {
                throw new IllegalStateException(
                                "Cannot merge activations with different ranks: first activations have rank "
                                                + fwdPassRank + ", activations[" + i + "] have rank " + currShape.length
                                                + " (shape=" + Arrays.toString(currShape) + ")");
            }
            forwardPassShapes[i] = Arrays.copyOf(currShape, currShape.length);
            if (currShape[0] != nExamples) {
                throw new IllegalStateException(
                                "Cannot merge activations with different number of examples (activations[0] shape: "
                                                + Arrays.toString(in[0].shape()) + ", activations[" + i
                                                + "] shape: " + Arrays.toString(in[i].shape()));
            }
        }

        try(MemoryWorkspace ws = workspaceMgr.notifyScopeBorrowed(ArrayType.ACTIVATIONS)){
            INDArray out = Nd4j.concat(mergeAxis, in);
            return out;
        }
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        if (!canDoBackward())
            throw new IllegalStateException("Cannot do backward pass: errors not set");

        if (forwardPassShapes.length == 1) {
            //No op case
            return new Pair<>(null, new INDArray[] {workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, epsilon)});
        }

        //Split the epsilons in the opposite way that the activations were merged
        INDArray[] out = new INDArray[forwardPassShapes.length];
        for (int i = 0; i < out.length; i++)
            out[i] = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, epsilon.dataType(), forwardPassShapes[i]);

        int cumulative = 0;
        switch (fwdPassRank) {
            case 2:
                //Standard
                for (int i = 0; i < forwardPassShapes.length; i++) {
                    out[i].assign(epsilon.get(NDArrayIndex.all(), //All rows
                                    NDArrayIndex.interval(cumulative, cumulative + forwardPassShapes[i][1]))); //subset of columns
                    cumulative += forwardPassShapes[i][1];
                }
                break;
            case 3:
                for (int i = 0; i < forwardPassShapes.length; i++) {
                    out[i].assign(epsilon.get(indices(3, mergeAxis, cumulative, cumulative + forwardPassShapes[i][mergeAxis]))); //All time steps

                    cumulative += forwardPassShapes[i][mergeAxis];
                }
                break;
            case 4:
                for (int i = 0; i < forwardPassShapes.length; i++) {
                    out[i].assign(epsilon.get(indices(4, mergeAxis, cumulative, cumulative + forwardPassShapes[i][mergeAxis]))); //height

                    cumulative += forwardPassShapes[i][mergeAxis];
                }
                break;
            default:
                throw new RuntimeException("Invalid rank during forward pass (not 2, 3, 4)"); //Should never happen
        }

        return new Pair<>(null, out);
    }

    private INDArrayIndex[] indices(int num, int axis, long from, long to){
        INDArrayIndex[] out = new INDArrayIndex[num];
        for( int i=0; i<num; i++ ){
            if(i == axis){
                out[i] = NDArrayIndex.interval(from, to);
            } else {
                out[i] = NDArrayIndex.all();
            }
        }
        return out;
    }


    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if (backpropGradientsViewArray != null)
            throw new RuntimeException("Vertex does not have gradients; gradients view array cannot be set here");
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState,
                    int minibatchSize) {
        if (maskArrays == null) {
            return new Pair<>(null, currentMaskState);
        }

        //Most common case: all or none.
        //If there's only *some* mask arrays: assume the others (missing) are equivalent to all 1s
        //And for handling multiple masks: best strategy seems to be an OR operation
        //i.e., output is 1 if any of the input are 1s
        //Which means: if any masks are missing, output null (equivalent to no mask)
        //Otherwise do an element-wise OR operation

        for (INDArray arr : maskArrays) {
            if (arr == null) {
                return new Pair<>(null, currentMaskState);
            }
        }

        //At this point: all present. Do OR operation
        if (maskArrays.length == 1) {
            return new Pair<>(maskArrays[0], currentMaskState);
        } else {
            INDArray ret;
            if(maskArrays[0].dataType() == DataType.BOOL){
                ret = maskArrays[0].dup(maskArrays[0].ordering());
            } else {
                ret = maskArrays[0].castTo(DataType.BOOL);
            }
            Nd4j.getExecutioner().exec(new Or(ret, maskArrays[1].castTo(DataType.BOOL), ret));
            for (int i = 2; i < maskArrays.length; i++) {
                Nd4j.getExecutioner().exec(new Or(maskArrays[i].castTo(DataType.BOOL), ret, ret));
            }
            return new Pair<>(ret, currentMaskState);
        }
    }
}
