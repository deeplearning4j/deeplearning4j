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

/** An ElementWiseVertex is used to combine the activations of two or more layer in an element-wise manner<br>
 * For example, the activations may be combined by addition, subtraction or multiplication.
 * Addition may use an arbitrary number of input arrays. Note that in the case of subtraction, only two inputs may be used.
 * In all cases, the shape of the input arrays must be identical.
 * @author Alex Black
 */
public class ElementWiseVertex extends BaseGraphVertex {

    public enum Op {Add, Subtract, Product};

    private Op op;
    private int nInForwardPass;

    public ElementWiseVertex(ComputationGraph graph, String name, int vertexIndex, Op op){
        this(graph,name,vertexIndex,null,null,op);
    }

    public ElementWiseVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices, VertexIndices[] outputVertices, Op op) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
        this.op = op;
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
        if(!canDoForward()) throw new IllegalStateException("Cannot do forward pass: inputs not set");

        nInForwardPass = inputs.length;
        if(inputs.length == 1) return inputs[0];

        switch(op){
            case Add:
                INDArray sum = inputs[0].dup();
                for( int i=1; i<inputs.length; i++){
                    sum.addi(inputs[i]);
                }
                return sum;
            case Subtract:
                if(inputs.length != 2) throw new IllegalArgumentException("ElementWise subtraction only supports 2 inputs");
                return inputs[0].sub(inputs[1]);
            case Product:
                throw new UnsupportedOperationException("ElementWise product: Not yet implemented");
            default:
                throw new UnsupportedOperationException("Unknown op: " + op);
        }
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt) {
        if(!canDoBackward()) throw new IllegalStateException("Cannot do backward pass: errors not set");

        if(nInForwardPass == 1) return new Pair<>(null,epsilons);

        switch(op){
            case Add:
                //If x=sum_i a_i then dL/da_i = dL/dx * dx/da_i = dL/dx
                INDArray[] out = new INDArray[nInForwardPass];
                out[0] = epsilons[0];
                for( int i=1; i<nInForwardPass; i++ ) out[i] = out[0].dup();
                return new Pair<>(null,out);
            case Subtract:
                INDArray[] out2 = new INDArray[2];
                out2[0] = epsilons[0];
                out2[1] = epsilons[0].mul(-1);
                return new Pair<>(null,out2);
            case Product:
                throw new UnsupportedOperationException("Not yet implemented");
            default:
                throw new UnsupportedOperationException("Unknown op: " + op);
        }
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if(backpropGradientsViewArray != null) throw new RuntimeException("Vertex does not have gradients; gradients view array cannot be set here");
    }

    @Override
    public String toString() {
        return "ElementWiseVertex(id=" + this.getVertexIndex() + ",name=\"" + this.getVertexName() + "\",op=" + op + ")";
    }
}
