/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.graph.vertex.impl;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastTo;
import org.nd4j.linalg.api.ops.impl.transforms.bool.MatchConditionTransform;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.SubOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.Or;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.common.primitives.Pair;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.util.Arrays;

/** An ElementWiseVertex is used to combine the activations of two or more layer in an element-wise manner<br>
 * For example, the activations may be combined by addition, subtraction or multiplication or by selecting the maximum.
 * Addition, Average, Product and Max may use an arbitrary number of input arrays. Note that in the case of subtraction, only two inputs may be used.
 * In all cases, the shape of the input arrays must be identical.
 * @author Alex Black
 */
public class ElementWiseVertex extends BaseGraphVertex {

    public enum Op {
        Add, Subtract, Product, Average, Max
    }

    private Op op;
    private int nInForwardPass;

    public ElementWiseVertex(ComputationGraph graph, String name, int vertexIndex, Op op, DataType dataType) {
        this(graph, name, vertexIndex, null, null, op, dataType);
    }

    public ElementWiseVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
                    VertexIndices[] outputVertices, Op op, DataType dataType) {
        super(graph, name, vertexIndex, inputVertices, outputVertices, dataType);
        this.op = op;
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

        nInForwardPass = inputs.length;
        if (inputs.length == 1)
            return workspaceMgr.dup(ArrayType.ACTIVATIONS, inputs[0]);

        boolean isBc = false;
        for(int i = 1; i < inputs.length; i++) {
            if(!inputs[0].equalShapes(inputs[i])) {
                isBc = true;
                break;
            }
        }

        long[] outShape;
        if(!isBc) {
            outShape = inputs[0].shape();
        } else {
            outShape = Shape.broadcastOutputShape(inputs[0].shape(), inputs[1].shape());
            for( int i = 2; i < inputs.length; i++) {
                outShape = Shape.broadcastOutputShape(outShape, inputs[i].shape());
            }
        }

        switch (op) {
            case Add:
                INDArray sum =  workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, dataType, outShape);
                if(isBc && !Arrays.equals(outShape, inputs[0].shape())) {
                    Nd4j.exec(new BroadcastTo(inputs[0], outShape, sum));
                } else {
                    sum.assign(inputs[0]);
                }

                for (int i = 1; i < inputs.length; i++) {
                    sum.addi(inputs[i].castTo(dataType));
                }
                return sum;
            case Average:
                INDArray average =  workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, dataType, outShape);
                if(isBc && !Arrays.equals(outShape, inputs[0].shape())){
                    Nd4j.exec(new BroadcastTo(inputs[0], outShape, average));
                } else {
                    average.assign(inputs[0]);
                }
                for (int i = 1; i < inputs.length; i++) {
                    average.addi(inputs[i].castTo(dataType));
                }
                return average.divi(inputs.length);
            case Subtract:
                if (inputs.length != 2)
                    throw new IllegalArgumentException("ElementWise subtraction only supports 2 inputs");
                return Nd4j.exec(new SubOp(inputs, new INDArray[]{workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, inputs[0].dataType(), outShape)}))[0];
            case Product:
                INDArray product =  workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, dataType, outShape);

                if(isBc && !Arrays.equals(outShape, inputs[0].shape())) {
                    Nd4j.exec(new BroadcastTo(inputs[0], outShape, product));
                } else {
                    product.assign(inputs[0]);
                }

                for (int i = 1; i < inputs.length; i++) {
                    product.muli(inputs[i].castTo(dataType));
                }
                return product;
            case Max:
                boolean isBroadcast = false;
                for(int i=1; i<inputs.length; i++) {
                    isBroadcast |= !inputs[0].equalShapes(inputs[i]);
                    if(isBroadcast)
                        break;
                }
                if(!isBroadcast) {
                    INDArray max = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, inputs[0].dataType(), inputs[0].shape(), inputs[0].ordering());
                    CustomOp op = DynamicCustomOp.builder("mergemax")
                            .addInputs(inputs)
                            .addOutputs(max)
                            .callInplace(false)
                            .build();
                    Nd4j.getExecutioner().exec(op);
                    return max;
                } else {
                    //AB 20190729 mergemax doesn't support broadcast at this point
                    if(inputs.length == 1) {
                        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, inputs[0]);
                    } else {
                        INDArray max = Transforms.max(inputs[0], inputs[1], true);
                        for( int i = 2; i < inputs.length; i++) {
                            max = Transforms.max(max, inputs[i], false);
                        }
                        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, max);
                    }
                }

            default:
                throw new UnsupportedOperationException("Unknown op: " + this.op);
        }
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        if (!canDoBackward())
            throw new IllegalStateException("Cannot do backward pass: errors not set");

        if (nInForwardPass == 1)
            return new Pair<>(null, new INDArray[] {workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, epsilon)});

        boolean broadcastCase = false;
        for( int i=1; i<nInForwardPass; i++ ){
            broadcastCase |= !inputs[0].equalShapes(inputs[i]);
        }

        switch (op) {
            case Add:
                //If x=sum_i a_i then dL/da_i = dL/dx * dx/da_i = dL/dx
                INDArray[] out = new INDArray[nInForwardPass];
                for (int i = 0; i < nInForwardPass; i++) {
                    if(!broadcastCase) {
                        //Standard case
                        out[i] = workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, epsilon);
                    } else {
                        //For broadcast case, we need to sum along the broadcast dimensions
                        //So if [mb,3]+[mb,1] -> input 0 backprops epsilon, input 1 backprops epsilon.sum(1,keepDim=true)
                        if(inputs[i].equalShapes(epsilon)){
                            out[i] = workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, epsilon);
                        } else {
                            int[] bcDim = Shape.getBroadcastDimensions(inputs[i].shape(), epsilon.shape());
                            try(MemoryWorkspace ws = workspaceMgr.notifyScopeBorrowed(ArrayType.ACTIVATION_GRAD)){
                                out[i] = epsilon.sum(true, bcDim);
                            }
                        }
                    }
                }
                return new Pair<>(null, out);
            case Average:
                INDArray[] outAverage = new INDArray[nInForwardPass];
                try(MemoryWorkspace ws = workspaceMgr.notifyScopeBorrowed(ArrayType.ACTIVATION_GRAD)){
                    for (int i = 0; i < nInForwardPass; i++) {
                        if(inputs[i].equalShapes(epsilon)){
                            outAverage[i] = epsilon.div(nInForwardPass);
                        } else {
                            int[] bcDim = Shape.getBroadcastDimensions(inputs[i].shape(), epsilon.shape());
                            outAverage[i] = epsilon.div(nInForwardPass).sum(true, bcDim);
                        }
                    }
                }
                return new Pair<>(null, outAverage);
            case Subtract:
                INDArray[] out2 = new INDArray[2];
                if(!broadcastCase){
                    out2[0] = workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, epsilon);
                    out2[1] = workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, epsilon).negi();
                } else {
                    if(inputs[0].equalShapes(epsilon)){
                        //Second input is smaller/broadcast
                        out2[0] = workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, epsilon);
                        int[] bcDim = Shape.getBroadcastDimensions(inputs[1].shape(), epsilon.shape());
                        try(MemoryWorkspace ws = workspaceMgr.notifyScopeBorrowed(ArrayType.ACTIVATION_GRAD)) {
                            out2[1] = epsilon.sum(true, bcDim).negi();
                        }
                    } else {
                        //First input is smaller/broadcast
                        int[] bcDim = Shape.getBroadcastDimensions(inputs[0].shape(), epsilon.shape());
                        try(MemoryWorkspace ws = workspaceMgr.notifyScopeBorrowed(ArrayType.ACTIVATION_GRAD)) {
                            out2[0] = epsilon.sum(true, bcDim);
                        }
                        out2[1] = workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, epsilon).negi();
                    }
                }
                return new Pair<>(null, out2);
            case Product:
                INDArray[] out_product = new INDArray[nInForwardPass];
                INDArray[] inBc = inputs;
                if(broadcastCase){
                    inBc = new INDArray[inputs.length];
                    for( int i=0; i<inputs.length; i++ ){
                        if(inputs[i].equalShapes(epsilon)){
                            inBc[i] = inputs[i];
                        } else {
                            inBc[i] = epsilon.ulike();
                            Nd4j.exec(new BroadcastTo(inputs[i], epsilon.shape(), inBc[i]));
                        }
                    }
                }

                for (int i = 0; i < nInForwardPass; i++) {
                    out_product[i] = workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, epsilon);
                    for (int j = 0; j < nInForwardPass; ++j) {
                        if (i != j)
                            out_product[i].muli(inBc[j]);
                    }

                    if(!inputs[i].equalShapes(epsilon)){
                        int[] bcDim = Shape.getBroadcastDimensions(inputs[i].shape(), epsilon.shape());
                        try(MemoryWorkspace ws = workspaceMgr.notifyScopeBorrowed(ArrayType.ACTIVATION_GRAD)) {
                            out_product[i] = out_product[i].sum(true, bcDim);
                        }
                    }
                }
                return new Pair<>(null, out_product);
            case Max:
                INDArray[] outMax = new INDArray[nInForwardPass];
                INDArray maxIndices = workspaceMgr.createUninitialized(ArrayType.BP_WORKING_MEM, DataType.INT, epsilon.shape(), epsilon.ordering());

                INDArray[] bcIn = inputs;
                if(broadcastCase){
                    //Broadcast to right shape...
                    bcIn = new INDArray[inputs.length];
                    for( int i=0; i<inputs.length; i++ ){
                        if(inputs[i].equalShapes(epsilon)){
                            bcIn[i] = inputs[i];
                        } else {
                            bcIn[i] = epsilon.ulike();
                            Nd4j.exec(new BroadcastTo(inputs[i], epsilon.shape(), bcIn[i]));
                        }
                    }
                }

                CustomOp op = DynamicCustomOp.builder("mergemaxindex")
                        .addInputs(bcIn)
                        .addOutputs(maxIndices)
                        .callInplace(false)
                        .build();
                Nd4j.getExecutioner().exec(op);
                for (int i = 0; i < nInForwardPass; i++) {
                    //gradient is epsilon where the max index is the same as i and zero elsewhere
                    outMax[i] = workspaceMgr.create(ArrayType.BP_WORKING_MEM, DataType.BOOL, maxIndices.shape());    //workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, maxIndices);
                    //generate a mask with 1s and 0s in the right places and muli with epsilon
                    MatchConditionTransform nd4jop = new MatchConditionTransform(maxIndices, outMax[i], Conditions.equals(i));
                    Nd4j.getExecutioner().exec(nd4jop);
                    if(broadcastCase && !epsilon.equalShapes(inputs[i])){
                        //Broadcast  for ths input
                        outMax[i] = outMax[i].castTo(epsilon.dataType()).mul(epsilon);
                        int[] bcDim = Shape.getBroadcastDimensions(inputs[i].shape(), epsilon.shape());
                        try(MemoryWorkspace ws = workspaceMgr.notifyScopeBorrowed(ArrayType.ACTIVATION_GRAD)) {
                            outMax[i] = outMax[i].sum(true, bcDim);
                        }
                    } else {
                        //Standard case
                        outMax[i] = workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, outMax[i].castTo(epsilon.dataType()).muli(epsilon));
                    }
                }
                return new Pair<>(null, outMax);
            default:
                throw new UnsupportedOperationException("Unknown op: " + this.op);
        }
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
        //Which means: if any masks are missing, output null (equivalent to no mask, or all steps present)
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
            INDArray ret = Nd4j.createUninitialized(DataType.BOOL, maskArrays[0].shape());  //maskArrays[0].dup(maskArrays[0].ordering());
            Nd4j.getExecutioner().exec(new Or(maskArrays[0].castTo(DataType.BOOL), maskArrays[1].castTo(DataType.BOOL), ret));
            for (int i = 2; i < maskArrays.length; i++) {
                Nd4j.getExecutioner().exec(new Or(maskArrays[i].castTo(DataType.BOOL), ret, ret));
            }
            return new Pair<>(ret.castTo(Nd4j.defaultFloatingPointType()), currentMaskState);
        }
    }

    @Override
    public String toString() {
        return "ElementWiseVertex(id=" + this.getVertexIndex() + ",name=\"" + this.getVertexName() + "\",op=" + op
                        + ")";
    }
}
