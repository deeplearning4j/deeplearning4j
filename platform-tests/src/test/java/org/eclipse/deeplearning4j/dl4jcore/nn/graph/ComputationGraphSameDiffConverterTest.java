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

package org.eclipse.deeplearning4j.dl4jcore.nn.graph;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.ComputationGraphSameDiffConverter;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.impl.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for ComputationGraphSameDiffConverter
 */
@NativeTag
@Tag(TagNames.DL4J_OLD_API)
@DisplayName("ComputationGraph SameDiff Converter Test")
public class ComputationGraphSameDiffConverterTest extends BaseDL4JTest {

    @Test
    @DisplayName("Test Simple Dense Graph To SameDiff")
    public void testSimpleDenseGraphToSameDiff() {
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(0.001))
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                .addInputs("input")
                .addLayer("dense1", new DenseLayer.Builder()
                        .nIn(10)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build(), "input")
                .addLayer("dense2", new DenseLayer.Builder()
                        .nIn(20)
                        .nOut(10)
                        .activation(Activation.RELU)
                        .build(), "dense1")
                .addLayer("output", new OutputLayer.Builder()
                        .nIn(10)
                        .nOut(5)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(new LossMCXENT())
                        .build(), "dense2")
                .setOutputs("output")
                .build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        SameDiff sd = ComputationGraphSameDiffConverter.toSameDiff(graph);

        assertNotNull(sd);
        assertTrue(sd.hasVariable("input"));
        assertTrue(sd.getVariables().keySet().stream().anyMatch(name -> name.contains("output")));
    }

    @Test
    @DisplayName("Test Dense Graph Forward Pass")
    public void testDenseGraphForwardPass() {
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(0.001))
                .weightInit(WeightInit.XAVIER)
                .inferenceWorkspaceMode(org.deeplearning4j.nn.conf.WorkspaceMode.NONE)  // Disable workspaces for inference
                .graphBuilder()
                .addInputs("input")
                .addLayer("dense1", new DenseLayer.Builder()
                        .nIn(4)
                        .nOut(8)
                        .activation(Activation.RELU)
                        .build(), "input")
                .addLayer("output", new OutputLayer.Builder()
                        .nIn(8)
                        .nOut(3)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(new LossMCXENT())
                        .build(), "dense1")
                .setOutputs("output")
                .build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        // Get output first before converting (to avoid any internal state changes)
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 4);
        INDArray originalOutput = graph.outputSingle(input);

        // Now convert to SameDiff
        SameDiff sd = ComputationGraphSameDiffConverter.toSameDiff(graph);

        assertNotNull(sd);
        assertNotNull(originalOutput);
        assertEquals(2, originalOutput.rows());
        assertEquals(3, originalOutput.columns());
    }

    @Test
    @DisplayName("Test CNN Graph To SameDiff")
    public void testCNNGraphToSameDiff() {
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(0.001))
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.convolutional(28, 28, 1))
                .addLayer("conv1", new ConvolutionLayer.Builder()
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .nOut(16)
                        .activation(Activation.RELU)
                        .build(), "input")
                .addLayer("pool1", new SubsamplingLayer.Builder()
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build(), "conv1")
                .addLayer("conv2", new ConvolutionLayer.Builder()
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build(), "pool1")
                .addLayer("globalPool", new GlobalPoolingLayer.Builder()
                        .poolingType(PoolingType.AVG)
                        .build(), "conv2")
                .addLayer("output", new OutputLayer.Builder()
                        .nOut(10)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(new LossMCXENT())
                        .build(), "globalPool")
                .setOutputs("output")
                .build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        SameDiff sd = ComputationGraphSameDiffConverter.toSameDiff(graph);

        assertNotNull(sd);
        assertTrue(sd.hasVariable("input"));
    }

    @Test
    @DisplayName("Test Batch Norm Graph To SameDiff")
    public void testBatchNormGraphToSameDiff() {
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(0.001))
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                .addInputs("input")
                .addLayer("dense1", new DenseLayer.Builder()
                        .nIn(10)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build(), "input")
                .addLayer("bn1", new BatchNormalization.Builder()
                        .nOut(20)
                        .build(), "dense1")
                .addLayer("output", new OutputLayer.Builder()
                        .nIn(20)
                        .nOut(5)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(new LossMCXENT())
                        .build(), "bn1")
                .setOutputs("output")
                .build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        SameDiff sd = ComputationGraphSameDiffConverter.toSameDiff(graph);

        assertNotNull(sd);
        boolean hasBnParams = sd.getVariables().keySet().stream()
                .anyMatch(name -> name.contains("bn1"));
        assertTrue(hasBnParams, "Should have batch normalization parameters");
    }

    @Test
    @DisplayName("Test Multi Input Graph To SameDiff")
    public void testMultiInputGraphToSameDiff() {
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(0.001))
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                .addInputs("input1", "input2")
                .addLayer("dense1", new DenseLayer.Builder()
                        .nIn(10)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build(), "input1")
                .addLayer("dense2", new DenseLayer.Builder()
                        .nIn(10)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build(), "input2")
                .addVertex("merge", new org.deeplearning4j.nn.conf.graph.MergeVertex(), "dense1", "dense2")
                .addLayer("output", new OutputLayer.Builder()
                        .nIn(40)
                        .nOut(5)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(new LossMCXENT())
                        .build(), "merge")
                .setOutputs("output")
                .build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        SameDiff sd = ComputationGraphSameDiffConverter.toSameDiff(graph);

        assertNotNull(sd);
        assertTrue(sd.hasVariable("input1"));
        assertTrue(sd.hasVariable("input2"));
    }

    @Test
    @DisplayName("Test Loss Function Mappings")
    public void testLossFunctionMappings() {
        assertEquals("mse", ComputationGraphSameDiffConverter.getLossFunctionName(new LossMSE()));
        assertEquals("mcxent", ComputationGraphSameDiffConverter.getLossFunctionName(new LossMCXENT()));
        assertEquals("binary_xent", ComputationGraphSameDiffConverter.getLossFunctionName(new LossBinaryXENT()));
        assertEquals("hinge", ComputationGraphSameDiffConverter.getLossFunctionName(new LossHinge()));
        assertEquals("squared_hinge", ComputationGraphSameDiffConverter.getLossFunctionName(new LossSquaredHinge()));
        assertEquals("mae", ComputationGraphSameDiffConverter.getLossFunctionName(new LossMAE()));
        assertEquals("kld", ComputationGraphSameDiffConverter.getLossFunctionName(new LossKLD()));
        assertEquals("negativeloglikelihood", ComputationGraphSameDiffConverter.getLossFunctionName(new LossNegativeLogLikelihood()));
    }

    @Test
    @DisplayName("Test Activation Mappings")
    public void testActivationMappings() {
        assertEquals("relu", ComputationGraphSameDiffConverter.getActivationName(Activation.RELU));
        assertEquals("sigmoid", ComputationGraphSameDiffConverter.getActivationName(Activation.SIGMOID));
        assertEquals("tanh", ComputationGraphSameDiffConverter.getActivationName(Activation.TANH));
        assertEquals("softmax", ComputationGraphSameDiffConverter.getActivationName(Activation.SOFTMAX));
        assertEquals("identity", ComputationGraphSameDiffConverter.getActivationName(Activation.IDENTITY));
        assertEquals("leakyrelu", ComputationGraphSameDiffConverter.getActivationName(Activation.LEAKYRELU));
        assertEquals("elu", ComputationGraphSameDiffConverter.getActivationName(Activation.ELU));
        assertEquals("selu", ComputationGraphSameDiffConverter.getActivationName(Activation.SELU));
        assertEquals("softplus", ComputationGraphSameDiffConverter.getActivationName(Activation.SOFTPLUS));
        assertEquals("softsign", ComputationGraphSameDiffConverter.getActivationName(Activation.SOFTSIGN));
        assertEquals("hardtanh", ComputationGraphSameDiffConverter.getActivationName(Activation.HARDTANH));
        assertEquals("relu6", ComputationGraphSameDiffConverter.getActivationName(Activation.RELU6));
        assertEquals("swish", ComputationGraphSameDiffConverter.getActivationName(Activation.SWISH));
        assertEquals("gelu", ComputationGraphSameDiffConverter.getActivationName(Activation.GELU));
    }

    @Test
    @DisplayName("Test Graph With Dropout")
    public void testGraphWithDropout() {
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(0.001))
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                .addInputs("input")
                .addLayer("dense1", new DenseLayer.Builder()
                        .nIn(10)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .dropOut(0.5)
                        .build(), "input")
                .addLayer("output", new OutputLayer.Builder()
                        .nIn(20)
                        .nOut(5)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(new LossMCXENT())
                        .build(), "dense1")
                .setOutputs("output")
                .build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        SameDiff sd = ComputationGraphSameDiffConverter.toSameDiff(graph);

        assertNotNull(sd);
    }

    @Test
    @DisplayName("Test Graph With No Bias")
    public void testGraphWithNoBias() {
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(0.001))
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                .addInputs("input")
                .addLayer("dense1", new DenseLayer.Builder()
                        .nIn(10)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .hasBias(false)
                        .build(), "input")
                .addLayer("output", new OutputLayer.Builder()
                        .nIn(20)
                        .nOut(5)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(new LossMCXENT())
                        .build(), "dense1")
                .setOutputs("output")
                .build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        SameDiff sd = ComputationGraphSameDiffConverter.toSameDiff(graph);

        assertNotNull(sd);

        boolean hasDense1Bias = sd.getVariables().keySet().stream()
                .anyMatch(name -> name.equals("dense1_b") || name.equals("dense1_bias"));
        assertFalse(hasDense1Bias, "Should not have bias for dense1 layer");
    }

    @Test
    @DisplayName("Test Residual Connection")
    public void testResidualConnection() {
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(0.001))
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                .addInputs("input")
                .addLayer("dense1", new DenseLayer.Builder()
                        .nIn(10)
                        .nOut(10)
                        .activation(Activation.RELU)
                        .build(), "input")
                .addLayer("dense2", new DenseLayer.Builder()
                        .nIn(10)
                        .nOut(10)
                        .activation(Activation.RELU)
                        .build(), "dense1")
                .addVertex("add", new org.deeplearning4j.nn.conf.graph.ElementWiseVertex(
                        org.deeplearning4j.nn.conf.graph.ElementWiseVertex.Op.Add), "input", "dense2")
                .addLayer("output", new OutputLayer.Builder()
                        .nIn(10)
                        .nOut(5)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(new LossMCXENT())
                        .build(), "add")
                .setOutputs("output")
                .build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        SameDiff sd = ComputationGraphSameDiffConverter.toSameDiff(graph);

        assertNotNull(sd);
        assertTrue(sd.hasVariable("input"));
    }
}
