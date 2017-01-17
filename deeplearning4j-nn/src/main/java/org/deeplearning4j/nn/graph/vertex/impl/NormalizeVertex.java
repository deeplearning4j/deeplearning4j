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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * NormalizeVertex performs L2 normalization on a single input.
 *
 * @author Justin Long (crockpotveggies)
 * @author Alex Black (AlexDBlack)
 */
public class NormalizeVertex extends BaseGraphVertex {

    private int dimension;

    public NormalizeVertex(ComputationGraph graph, String name, int vertexIndex, int dimension){
        this(graph,name,vertexIndex,null,null,dimension);
    }

    public NormalizeVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
                           VertexIndices[] outputVertices, int dimension) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
        this.dimension = dimension;
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
        if(dimension<0) {
            dimensions = new int[x.rank()-1];
            for( int i=1; i<x.rank(); i++ ){
                dimensions[i-1] = i;
            }
        } else {
            dimensions = new int[]{dimension};
        }

        return x.diviColumnVector(x.norm2(dimensions));

    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt) {
        if(!canDoBackward()) throw new IllegalStateException("Cannot do backward pass: errors not set (NormalizeVertex "+vertexName+" idx "+vertexIndex+")");

        INDArray x = inputs[0];
        int[] dimensions;
        if(dimension<0) {
            dimensions = new int[x.rank()-1];
            for( int i=1; i<x.rank(); i++ ){
                dimensions[i-1] = i;
            }
        } else {
            dimensions = new int[]{dimension};
        }

        INDArray norm = x.norm2(dimensions);
        INDArray xSq = Transforms.pow(x, 2, true);
        INDArray normSq = Transforms.pow(norm, 2, true);
        INDArray norm3 = Transforms.pow(norm, 3);
        INDArray dadx;
        if (x.rank() == 2){
            //2d case (MLPs etc)
            dadx = xSq.rsubColumnVector(normSq).diviColumnVector(norm3);
        } else {
            //RNN and CNN case - Broadcast along dimension 0
            dadx = xSq.rsubColumnVector(normSq);
            dadx = Nd4j.getExecutioner().execAndReturn(new BroadcastDivOp(norm3, dadx, norm3, dimensions));
        }
        INDArray dLdx = epsilon.mmul(dadx.transpose());

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
