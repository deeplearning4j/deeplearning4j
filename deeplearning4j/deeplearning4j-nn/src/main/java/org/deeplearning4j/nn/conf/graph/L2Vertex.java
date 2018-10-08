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

package org.deeplearning4j.nn.conf.graph;


import lombok.val;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * L2Vertex calculates the L2 (Euclidean) least squares error of two inputs, on a per-example basis.
 * It outputs a single value for each input - i.e., for input [minibatch,X] it outputs shape [minibatch,1]
 * where each value {@code out[i,0] = l2Distance(in1[i,...], in2[i,...])}<br>
 * Note however than epsilon value (1e-8 by default) will be added to inputs to avoid the gradient being undefined
 * for all zero inputs
 *
 * For example, in Triplet Embedding you can input an anchor and a pos/neg class and use two parallel
 * L2 vertices to calculate two real numbers which can be fed into a LossLayer to calculate TripletLoss.
 *
 * @author Justin Long (crockpotveggies)
 */
public class L2Vertex extends GraphVertex {
    protected double eps;

    /**
     * Constructor with default epsilon value of 1e-8
     */
    public L2Vertex() {
        this.eps = 1e-8;
    }

    /**
     * @param eps Epsilon value to add to inputs (to avoid all zeros input and hence undefined gradients)
     */
    public L2Vertex(double eps) {
        this.eps = eps;
    }

    @Override
    public L2Vertex clone() {
        return new L2Vertex();
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof L2Vertex;
    }

    @Override
    public int numParams(boolean backprop) {
        return 0;
    }

    @Override
    public int minVertexInputs() {
        return 2;
    }

    @Override
    public int maxVertexInputs() {
        return 2;
    }

    @Override
    public int hashCode() {
        return 433682566;
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx,
                    INDArray paramsView, boolean initializeParams) {
        return new org.deeplearning4j.nn.graph.vertex.impl.L2Vertex(graph, name, idx, null, null, eps);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        return InputType.feedForward(1);
    }

    @Override
    public MemoryReport getMemoryReport(InputType... inputTypes) {
        InputType outputType = getOutputType(-1, inputTypes);

        //Inference: only calculation is for output activations; no working memory
        //Working memory for training:
        //1 for each example (fwd pass) + output size (1 per ex) + input size + output size... in addition to the returned eps arrays
        //output size == input size here
        val trainWorkingSizePerEx = 3 + 2 * inputTypes[0].arrayElementsPerExample();

        return new LayerMemoryReport.Builder(null, L2Vertex.class, inputTypes[0], outputType).standardMemory(0, 0) //No params
                        .workingMemory(0, 0, 0, trainWorkingSizePerEx).cacheMemory(0, 0) //No caching
                        .build();
    }
}
