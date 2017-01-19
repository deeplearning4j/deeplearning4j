/*
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

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDivOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * NormalizeVertex performs L2 normalization on a single input.
 *
 * @author Justin Long (crockpotveggies)
 * @author Alex Black (AlexDBlack)
 */
public class NormalizeVertex extends BaseGraphVertex {

    private int[] dimension;
    private double eps;

    public NormalizeVertex(ComputationGraph graph, String name, int vertexIndex, int[] dimension, double eps){
        this(graph,name,vertexIndex,null,null,dimension,eps);
    }

    public NormalizeVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
                           VertexIndices[] outputVertices, int[] dimension, double eps) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
        this.dimension = dimension;
        this.eps = eps;
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
    public INDArray doForward(boolean training) {
        if(!canDoForward()) throw new IllegalStateException("Cannot do forward pass: inputs not set (NormalizeVertex "+vertexName+" idx "+vertexIndex+")");

        // L2 norm along all dimensions except 0, unless user-specified
        // x / |x|2
        INDArray x = inputs[0];
        int[] dimensions;
        if(dimension.length<1) {
            dimensions = new int[x.rank()-1];
            for( int i=1; i<x.rank(); i++ ){
                dimensions[i-1] = i;
            }
        } else {
            dimensions = dimension;
        }

        x.diviColumnVector(x.norm2(dimensions));
        Transforms.max(x, eps, false); // in case of div by 0

        return x;

    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt) {
        if(!canDoBackward()) throw new IllegalStateException("Cannot do backward pass: errors not set (NormalizeVertex "+vertexName+" idx "+vertexIndex+")");

        INDArray x = inputs[0];
        int[] dimensions;
        if(dimension.length<1) {
            dimensions = new int[x.rank()-1];
            for( int i=1; i<x.rank(); i++ ){
                dimensions[i-1] = i;
            }
        } else {
            dimensions = dimension;
        }

        INDArray norm = x.norm2(dimensions);
        INDArray norm3 = Transforms.pow(norm, 3.0, true);

        INDArray dLdx;
        if (x.rank() == 2) {
            // 2D case
            dLdx = epsilon.divColumnVector(norm)
                .sub(x.divColumnVector(norm3).muliColumnVector(epsilon.mul(x).sum(1)));
        } else {
            //RNN and CNN case - Broadcast along dimension 0
            INDArray dx = epsilon.mul(x).sum(1);

            Nd4j.getExecutioner().exec(new BroadcastDivOp(x,norm3,x,0));
            Nd4j.getExecutioner().exec(new BroadcastMulOp(x,dx,x,0));
            dLdx = Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(epsilon,norm,epsilon,0));
            dLdx = Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(dLdx,x,dLdx,0));
        }
        Transforms.max(dLdx, eps); // in case of div by 0

        return new Pair<>(null, new INDArray[]{dLdx});
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if(backpropGradientsViewArray != null) throw new RuntimeException("Vertex does not have gradients; gradients view array cannot be set here");
    }

    @Override
    public String toString(){
        return "NormalizeVertex(id=" + this.getVertexIndex() + ",name=\"" + this.getVertexName() + ",dim=\""+dimension+"\")";
    }
}