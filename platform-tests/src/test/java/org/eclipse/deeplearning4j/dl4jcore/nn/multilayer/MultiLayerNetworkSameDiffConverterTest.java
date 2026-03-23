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
package org.eclipse.deeplearning4j.dl4jcore.nn.multilayer;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.multilayer.MultiLayerNetworkSameDiffConverter;
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
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.impl.*;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for MultiLayerNetworkSameDiffConverter.
 * Verifies both structural correctness and forward-pass numerical equivalence.
 */
@NativeTag
@Tag(TagNames.DL4J_OLD_API)
@DisplayName("MultiLayerNetwork SameDiff Converter Test")
public class MultiLayerNetworkSameDiffConverterTest extends BaseDL4JTest {

    private static final double TOLERANCE = 1e-3;

    /**
     * Helper to compare DL4J forward pass output with SameDiff forward pass output.
     */
    private void assertForwardPassEquivalent(MultiLayerNetwork network, INDArray input) {
        // DL4J forward pass
        INDArray dl4jOutput = network.output(input);

        // SameDiff conversion and forward pass
        SameDiff sd = MultiLayerNetworkSameDiffConverter.toSameDiff(network);
        Map<String, INDArray> sdOutputs = sd.output(Map.of("input", input), sd.outputs());
        INDArray sdOutput = sdOutputs.values().iterator().next();

        assertNotNull(dl4jOutput, "DL4J output should not be null");
        assertNotNull(sdOutput, "SameDiff output should not be null");
        assertArrayEquals(dl4jOutput.shape(), sdOutput.shape(),
            "Output shapes should match. DL4J: " + java.util.Arrays.toString(dl4jOutput.shape()) +
            " SameDiff: " + java.util.Arrays.toString(sdOutput.shape()));

        double maxDiff = dl4jOutput.sub(sdOutput).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOLERANCE,
            "Outputs should be numerically equivalent. Max diff: " + maxDiff);
    }

    @Test
    @DisplayName("Test Dense Network Forward Pass Equivalence")
    public void testDenseNetworkForwardPassEquivalence() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .dataType(DataType.FLOAT)
                .inferenceWorkspaceMode(WorkspaceMode.NONE)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(10)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(20)
                        .nOut(15)
                        .activation(Activation.TANH)
                        .build())
                .layer(new OutputLayer.Builder()
                        .nIn(15)
                        .nOut(5)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build())
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        INDArray input = Nd4j.rand(DataType.FLOAT, 3, 10);
        assertForwardPassEquivalent(network, input);
    }

    @Test
    @DisplayName("Test Dense Network Without Bias")
    public void testDenseNetworkWithoutBias() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .dataType(DataType.FLOAT)
                .inferenceWorkspaceMode(WorkspaceMode.NONE)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(10)
                        .nOut(20)
                        .hasBias(false)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder()
                        .nIn(20)
                        .nOut(5)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build())
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 10);
        assertForwardPassEquivalent(network, input);

        // Verify first layer has no bias in SameDiff
        SameDiff sd = MultiLayerNetworkSameDiffConverter.toSameDiff(network);
        boolean hasFirstLayerBias = sd.getVariables().keySet().stream()
                .anyMatch(n -> n.startsWith("layer_0_b") || n.startsWith("0_b"));
        assertFalse(hasFirstLayerBias, "First layer should NOT have bias");
    }

    @Test
    @DisplayName("Test Multiple Activations")
    public void testMultipleActivations() {
        Activation[] activations = {
            Activation.RELU, Activation.SIGMOID, Activation.TANH,
            Activation.ELU, Activation.SOFTPLUS, Activation.SOFTSIGN,
            Activation.SWISH, Activation.GELU, Activation.HARDSIGMOID
        };

        for (Activation act : activations) {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(42)
                    .weightInit(WeightInit.XAVIER)
                    .dataType(DataType.FLOAT)
                    .inferenceWorkspaceMode(WorkspaceMode.NONE)
                    .list()
                    .layer(new DenseLayer.Builder()
                            .nIn(5)
                            .nOut(5)
                            .activation(act)
                            .build())
                    .layer(new OutputLayer.Builder()
                            .nIn(5)
                            .nOut(3)
                            .activation(Activation.SOFTMAX)
                            .lossFunction(LossFunctions.LossFunction.MCXENT)
                            .build())
                    .build();

            MultiLayerNetwork network = new MultiLayerNetwork(conf);
            network.init();

            INDArray input = Nd4j.rand(DataType.FLOAT, 2, 5);
            assertForwardPassEquivalent(network, input);
        }
    }

    @Test
    @DisplayName("Test Conv2D Network Forward Pass")
    public void testConv2DNetworkForwardPass() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .dataType(DataType.FLOAT)
                .inferenceWorkspaceMode(WorkspaceMode.NONE)
                .list()
                .layer(new ConvolutionLayer.Builder()
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .nIn(1)
                        .nOut(8)
                        .activation(Activation.RELU)
                        .convolutionMode(ConvolutionMode.Same)
                        .build())
                .layer(new SubsamplingLayer.Builder()
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(new GlobalPoolingLayer.Builder()
                        .poolingType(PoolingType.AVG)
                        .build())
                .layer(new OutputLayer.Builder()
                        .nIn(8)
                        .nOut(3)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build())
                .setInputType(InputType.convolutional(16, 16, 1))
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 1, 16, 16);
        assertForwardPassEquivalent(network, input);
    }

    @Test
    @DisplayName("Test BatchNorm Network Forward Pass")
    public void testBatchNormNetworkForwardPass() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .dataType(DataType.FLOAT)
                .inferenceWorkspaceMode(WorkspaceMode.NONE)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(10)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new BatchNormalization.Builder()
                        .nOut(20)
                        .build())
                .layer(new ActivationLayer.Builder()
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder()
                        .nIn(20)
                        .nOut(5)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build())
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        // Verify conversion succeeds with BatchNorm parameters
        SameDiff sd = MultiLayerNetworkSameDiffConverter.toSameDiff(network);
        assertNotNull(sd);
        boolean hasBnParams = sd.getVariables().keySet().stream()
                .anyMatch(name -> name.contains("gamma") || name.contains("beta"));
        assertTrue(hasBnParams, "Should have batch normalization parameters");
    }

    @Test
    @DisplayName("Test Embedding Layer")
    public void testEmbeddingLayer() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .dataType(DataType.FLOAT)
                .inferenceWorkspaceMode(WorkspaceMode.NONE)
                .list()
                .layer(new EmbeddingLayer.Builder()
                        .nIn(100)
                        .nOut(32)
                        .build())
                .layer(new OutputLayer.Builder()
                        .nIn(32)
                        .nOut(5)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build())
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        SameDiff sd = MultiLayerNetworkSameDiffConverter.toSameDiff(network);
        assertNotNull(sd);
        assertTrue(sd.getVariables().keySet().stream().anyMatch(n -> n.endsWith("_W")),
            "Should have embedding weight matrix");

        // Verify forward pass with INT indices (1D for gather)
        INDArray intInput = Nd4j.createFromArray(1, 5, 10);
        // DL4J embedding expects 2D [batch, 1] float input
        INDArray dl4jInput = intInput.castTo(DataType.FLOAT).reshape(3, 1);
        INDArray dl4jOutput = network.output(dl4jInput);
        Map<String, INDArray> sdOutputs = sd.output(Map.of("input", intInput), sd.outputs());
        INDArray sdOutput = sdOutputs.values().iterator().next();
        assertArrayEquals(dl4jOutput.shape(), sdOutput.shape(), "Embedding output shapes should match");
    }

    @Test
    @DisplayName("Test Uninitialized Network Throws Exception")
    public void testUninitializedNetworkThrowsException() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(10)
                        .nOut(5)
                        .build())
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);

        assertThrows(RuntimeException.class, () -> {
            MultiLayerNetworkSameDiffConverter.toSameDiff(network);
        }, "Should throw exception for uninitialized network");
    }

    @Test
    @DisplayName("Test Loss Function Name Mapping")
    public void testLossFunctionNameMapping() {
        assertEquals("mse", MultiLayerNetworkSameDiffConverter.getLossFunctionName(new LossMSE()));
        assertEquals("mcxent", MultiLayerNetworkSameDiffConverter.getLossFunctionName(new LossMCXENT()));
        assertEquals("binary_xent", MultiLayerNetworkSameDiffConverter.getLossFunctionName(new LossBinaryXENT()));
        assertEquals("mae", MultiLayerNetworkSameDiffConverter.getLossFunctionName(new LossMAE()));
        assertEquals("hinge", MultiLayerNetworkSameDiffConverter.getLossFunctionName(new LossHinge()));
        assertEquals("squared_hinge", MultiLayerNetworkSameDiffConverter.getLossFunctionName(new LossSquaredHinge()));
        assertEquals("kld", MultiLayerNetworkSameDiffConverter.getLossFunctionName(new LossKLD()));
        assertEquals("negativeloglikelihood", MultiLayerNetworkSameDiffConverter.getLossFunctionName(new LossNegativeLogLikelihood()));
    }

    @Test
    @DisplayName("Test Activation Name Mapping")
    public void testActivationNameMapping() {
        assertEquals("relu", MultiLayerNetworkSameDiffConverter.getActivationName(Activation.RELU));
        assertEquals("sigmoid", MultiLayerNetworkSameDiffConverter.getActivationName(Activation.SIGMOID));
        assertEquals("tanh", MultiLayerNetworkSameDiffConverter.getActivationName(Activation.TANH));
        assertEquals("softmax", MultiLayerNetworkSameDiffConverter.getActivationName(Activation.SOFTMAX));
        assertEquals("identity", MultiLayerNetworkSameDiffConverter.getActivationName(Activation.IDENTITY));
        assertEquals("elu", MultiLayerNetworkSameDiffConverter.getActivationName(Activation.ELU));
        assertEquals("selu", MultiLayerNetworkSameDiffConverter.getActivationName(Activation.SELU));
        assertEquals("softplus", MultiLayerNetworkSameDiffConverter.getActivationName(Activation.SOFTPLUS));
        assertEquals("softsign", MultiLayerNetworkSameDiffConverter.getActivationName(Activation.SOFTSIGN));
        assertEquals("swish", MultiLayerNetworkSameDiffConverter.getActivationName(Activation.SWISH));
        assertEquals("gelu", MultiLayerNetworkSameDiffConverter.getActivationName(Activation.GELU));
    }

    @Test
    @DisplayName("Test Unsupported Layer Throws")
    public void testUnsupportedLayerThrows() {
        // VariationalAutoencoder is not supported
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .dataType(DataType.FLOAT)
                .list()
                .layer(new org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder.Builder()
                        .nIn(10)
                        .nOut(5)
                        .encoderLayerSizes(8)
                        .decoderLayerSizes(8)
                        .build())
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        assertThrows(UnsupportedOperationException.class, () -> {
            MultiLayerNetworkSameDiffConverter.toSameDiff(network);
        }, "Should throw for unsupported layer type");
    }

    @Test
    @DisplayName("Test Variable Naming Convention")
    public void testVariableNamingConvention() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .dataType(DataType.FLOAT)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(10)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder()
                        .nIn(20)
                        .nOut(5)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build())
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        SameDiff sd = MultiLayerNetworkSameDiffConverter.toSameDiff(network);

        // Should have input placeholder
        assertTrue(sd.hasVariable("input"), "Should have 'input' placeholder");

        // Should have weight variables with _W suffix
        long weightCount = sd.getVariables().keySet().stream()
                .filter(n -> n.endsWith("_W"))
                .count();
        assertEquals(2, weightCount, "Should have 2 weight variables (one per layer)");

        // Should have bias variables with _b suffix
        long biasCount = sd.getVariables().keySet().stream()
                .filter(n -> n.endsWith("_b"))
                .count();
        assertEquals(2, biasCount, "Should have 2 bias variables (one per layer)");
    }

    @Test
    @DisplayName("Test Deep Dense Network")
    public void testDeepDenseNetwork() {
        // Test a deeper network with various activations
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .weightInit(WeightInit.XAVIER)
                .dataType(DataType.FLOAT)
                .inferenceWorkspaceMode(WorkspaceMode.NONE)
                .list()
                .layer(new DenseLayer.Builder().nIn(20).nOut(64).activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nIn(64).nOut(32).activation(Activation.SIGMOID).build())
                .layer(new DenseLayer.Builder().nIn(32).nOut(16).activation(Activation.TANH).build())
                .layer(new DenseLayer.Builder().nIn(16).nOut(8).activation(Activation.ELU).build())
                .layer(new OutputLayer.Builder().nIn(8).nOut(4).activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        INDArray input = Nd4j.rand(DataType.FLOAT, 5, 20);
        assertForwardPassEquivalent(network, input);
    }

    @Test
    @DisplayName("Test Conv2D Valid Padding Forward Pass")
    public void testConv2DValidPaddingForwardPass() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .dataType(DataType.FLOAT)
                .inferenceWorkspaceMode(WorkspaceMode.NONE)
                .list()
                .layer(new ConvolutionLayer.Builder()
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .nIn(1)
                        .nOut(4)
                        .activation(Activation.RELU)
                        .convolutionMode(ConvolutionMode.Truncate)
                        .build())
                .layer(new GlobalPoolingLayer.Builder()
                        .poolingType(PoolingType.AVG)
                        .build())
                .layer(new OutputLayer.Builder()
                        .nIn(4)
                        .nOut(2)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build())
                .setInputType(InputType.convolutional(8, 8, 1))
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 1, 8, 8);
        assertForwardPassEquivalent(network, input);
    }

    @Test
    @DisplayName("Test Dropout Layer Passthrough")
    public void testDropoutLayerPassthrough() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .dataType(DataType.FLOAT)
                .inferenceWorkspaceMode(WorkspaceMode.NONE)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(10)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DropoutLayer.Builder()
                        .dropOut(0.5)
                        .build())
                .layer(new OutputLayer.Builder()
                        .nIn(20)
                        .nOut(5)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build())
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        // SameDiff conversion should succeed (dropout = identity in inference)
        SameDiff sd = MultiLayerNetworkSameDiffConverter.toSameDiff(network);
        assertNotNull(sd);
    }

    @Test
    @DisplayName("Test Global Pooling Types")
    public void testGlobalPoolingTypes() {
        for (PoolingType poolType : new PoolingType[]{PoolingType.MAX, PoolingType.AVG, PoolingType.SUM}) {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .weightInit(WeightInit.XAVIER)
                    .dataType(DataType.FLOAT)
                    .inferenceWorkspaceMode(WorkspaceMode.NONE)
                    .list()
                    .layer(new ConvolutionLayer.Builder()
                            .kernelSize(3, 3)
                            .stride(1, 1)
                            .nIn(1)
                            .nOut(4)
                            .activation(Activation.RELU)
                            .convolutionMode(ConvolutionMode.Same)
                            .build())
                    .layer(new GlobalPoolingLayer.Builder()
                            .poolingType(poolType)
                            .build())
                    .layer(new OutputLayer.Builder()
                            .nIn(4)
                            .nOut(2)
                            .activation(Activation.SOFTMAX)
                            .lossFunction(LossFunctions.LossFunction.MCXENT)
                            .build())
                    .setInputType(InputType.convolutional(8, 8, 1))
                    .build();

            MultiLayerNetwork network = new MultiLayerNetwork(conf);
            network.init();

            INDArray input = Nd4j.rand(DataType.FLOAT, 2, 1, 8, 8);
            assertForwardPassEquivalent(network, input);
        }
    }

    @Test
    @DisplayName("Test Output Layer Without Bias Forward Pass")
    public void testOutputLayerWithoutBiasForwardPass() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .dataType(DataType.FLOAT)
                .inferenceWorkspaceMode(WorkspaceMode.NONE)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(10)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder()
                        .nIn(20)
                        .nOut(5)
                        .hasBias(false)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build())
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 10);
        assertForwardPassEquivalent(network, input);
    }
}
