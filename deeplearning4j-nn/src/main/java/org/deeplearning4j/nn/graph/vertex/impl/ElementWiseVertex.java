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

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.api.gradients.GradientsFactory;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.transforms.MatchConditionTransform;
import org.nd4j.linalg.api.ops.impl.transforms.Or;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.primitives.Pair;

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

    public ElementWiseVertex(ComputationGraph graph, String name, int vertexIndex, int numInputs, Op op) {
        super(graph, name, vertexIndex, numInputs);
        this.op = op;
    }

    @Override
    public Activations activate(boolean training) {
        if (!canDoForward())
            throw new IllegalStateException("Cannot do forward pass: inputs not set");

        nInForwardPass = input.size();
        if (input.size() == 1)
            return input;

        INDArray ret;
        switch (op) {
            case Add:
                INDArray sum = input.get(0).dup();
                for (int i = 1; i < input.size(); i++) {
                    sum.addi(input.get(i));
                }
                ret = sum;
            case Average:
                INDArray average = input.get(0).dup(input.get(0).ordering());
                for (int i = 1; i < input.size(); i++) {
                    average.addi(input.get(i));
                }
                ret = average.divi(input.size());
                break;
            case Subtract:
                if (input.size() != 2)
                    throw new IllegalArgumentException("ElementWise subtraction only supports 2 inputs");
                ret =  input.get(0).sub(input.get(1));
                break;
            case Product:
                INDArray product = input.get(0).dup();
                for (int i = 1; i < input.size(); i++) {
                    product.muli(input.get(i));
                }
                ret = product;
            case Max:
                INDArray max =  Nd4j.createUninitialized(input.get(0).shape(), input.get(0).ordering());
                CustomOp op = DynamicCustomOp.builder("mergemax")
                        .addInputs(input.getAsArray())
                        .addOutputs(max)
                        .callInplace(false)
                        .build();
                Nd4j.getExecutioner().exec(op);
                ret = max;
                break;
            default:
                throw new UnsupportedOperationException("Unknown op: " + this.op);
        }

        Pair<INDArray, MaskState> masks = feedForwardMaskArrays(new INDArray[]{input.getMask(0)}, MaskState.Active, input.get(0).size(0));    //TODO
        return ActivationsFactory.getInstance().create(ret, masks.getFirst(), masks.getSecond());
    }


    @Override
    public Gradients backpropGradient(Gradients gradients) {
        INDArray epsilon = gradients.get(0);
        if (!canDoBackward())
            throw new IllegalStateException("Cannot do backward pass: errors not set");

        if (nInForwardPass == 1)
            return gradients;

        INDArray[] out = new INDArray[nInForwardPass];
        switch (op) {
            case Add:
                //If x=sum_i a_i then dL/da_i = dL/dx * dx/da_i = dL/dx
                for (int i = 0; i < nInForwardPass; i++)
                    out[i] = epsilon.dup();
                break;
            case Average:
                for (int i = 0; i < nInForwardPass; i++)
                    out[i] = epsilon.div(nInForwardPass);
                break;
            case Subtract:
                out = new INDArray[2];
                out[0] = epsilon;
                out[1] = epsilon.neg();
                break;
            case Product:
                out = new INDArray[nInForwardPass];
                for (int i = 0; i < nInForwardPass; i++) {
                    out[i] = epsilon.dup();
                    for (int j = 0; j < nInForwardPass; ++j) {
                        if (i != j)
                            out[i].muli(input.get(j));
                    }
                }
                break;
            case Max:
                INDArray maxIndices = Nd4j.createUninitialized(epsilon.shape(), epsilon.ordering());
                CustomOp op = DynamicCustomOp.builder("mergemaxindex")
                        .addInputs(inputs)
                        .addOutputs(maxIndices)
                        .callInplace(false)
                        .build();
                Nd4j.getExecutioner().exec(op);
                for (int i = 0; i < nInForwardPass; i++) {
                    //gradient is epsilon where the max index is the same as i and zero elsewhere
                    out[i] = maxIndices.dup();
                    //generate a mask with 1s and 0s in the right places and muli with epsilon
                    MatchConditionTransform nd4jop = new MatchConditionTransform(out[i], out[i], Conditions.equals(i));
                    Nd4j.getExecutioner().exec(nd4jop);
                    out[i].muli(epsilon);
                }
                break;
            default:
                throw new UnsupportedOperationException("Unknown op: " + this.op);
        }
        return GradientsFactory.getInstance().create(null, out);
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
            INDArray ret = maskArrays[0].dup(maskArrays[0].ordering());
            Nd4j.getExecutioner().exec(new Or(maskArrays[0], maskArrays[1], ret));
            for (int i = 2; i < maskArrays.length; i++) {
                Nd4j.getExecutioner().exec(new Or(maskArrays[i], ret, ret));
            }
            return new Pair<>(ret, currentMaskState);
        }
    }

    @Override
    public String toString() {
        return "ElementWiseVertex(id=" + this.getIndex() + ",name=\"" + this.getName() + "\",op=" + op
                        + ")";
    }
}
