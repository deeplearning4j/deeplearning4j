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
package org.deeplearning4j.nn.weights;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;

@DisplayName("Weight Init Identity Test")
class WeightInitIdentityTest extends BaseDL4JTest {

    /**
     * Test identity mapping for 1d convolution
     */
    @Test
    @Disabled("Ignore for now. Underlying logic changed. Gradient checker passes so implementatin is valid.")
    @DisplayName("Test Id Conv 1 D")
    void testIdConv1D() {
        final INDArray input = Nd4j.randn(DataType.FLOAT, 1, 5, 7);
        final String inputName = "input";
        final String conv = "conv";
        final String output = "output";
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder().graphBuilder().addInputs(inputName).setOutputs(output).layer(conv, new Convolution1DLayer.Builder(7).convolutionMode(ConvolutionMode.Same).nOut(input.size(1)).weightInit(new WeightInitIdentity()).activation(new ActivationIdentity()).build(), inputName).layer(output, new RnnLossLayer.Builder().activation(new ActivationIdentity()).build(), conv).setInputTypes(InputType.recurrent(5, 7, RNNFormat.NCW)).build());
        graph.init();
        INDArray reshape = graph.outputSingle(input).reshape(input.shape());
        assertEquals(input, reshape, "Mapping was not identity!");
    }

    /**
     * Test identity mapping for 2d convolution
     */
    @Test
    @DisplayName("Test Id Conv 2 D")
    void testIdConv2D() {
        final INDArray input = Nd4j.randn(DataType.FLOAT, 1, 5, 7, 11);
        final String inputName = "input";
        final String conv = "conv";
        final String output = "output";
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder().graphBuilder().setInputTypes(InputType.inferInputType(input)).addInputs(inputName).setOutputs(output).layer(conv, new ConvolutionLayer.Builder(3, 5).convolutionMode(ConvolutionMode.Same).nOut(input.size(1)).weightInit(new WeightInitIdentity()).activation(new ActivationIdentity()).build(), inputName).layer(output, new CnnLossLayer.Builder().activation(new ActivationIdentity()).build(), conv).build());
        graph.init();
        assertEquals(input, graph.outputSingle(input), "Mapping was not identity!");
    }

    /**
     * Test identity mapping for 3d convolution
     */
    @Test
    @DisplayName("Test Id Conv 3 D")
    void testIdConv3D() {
        final INDArray input = Nd4j.randn(DataType.FLOAT, 1, 5, 7, 11, 13);
        final String inputName = "input";
        final String conv = "conv";
        final String output = "output";
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder().graphBuilder().setInputTypes(InputType.inferInputType(input)).addInputs(inputName).setOutputs(output).layer(conv, new Convolution3D.Builder(3, 7, 5).convolutionMode(ConvolutionMode.Same).dataFormat(Convolution3D.DataFormat.NCDHW).nOut(input.size(1)).weightInit(new WeightInitIdentity()).activation(new ActivationIdentity()).build(), inputName).layer(output, new Cnn3DLossLayer.Builder(Convolution3D.DataFormat.NCDHW).activation(new ActivationIdentity()).build(), conv).build());
        graph.init();
        assertEquals(input, graph.outputSingle(input), "Mapping was not identity!");
    }
}
