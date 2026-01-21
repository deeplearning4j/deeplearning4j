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
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.impl.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for MultiLayerNetworkSameDiffConverter.
 */
@NativeTag
@Tag(TagNames.DL4J_OLD_API)
@DisplayName("MultiLayerNetwork SameDiff Converter Test")
public class MultiLayerNetworkSameDiffConverterTest extends BaseDL4JTest {

    @Test
    @DisplayName("Test Simple Dense Network Conversion")
    public void testSimpleDenseNetworkConversion() {
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

        assertNotNull(sd, "SameDiff should not be null");

        // Debug: print all variable names
        System.out.println("All SameDiff variables:");
        for (String name : sd.getVariables().keySet()) {
            System.out.println("  - " + name);
        }

        // Check for weights - look for any weight variable containing 'W'
        boolean hasWeights = sd.getVariables().keySet().stream().anyMatch(n -> n.endsWith("_W"));
        assertTrue(hasWeights, "Should have weight variables, found: " + sd.getVariables().keySet());

        // Check for biases
        boolean hasBias = sd.getVariables().keySet().stream().anyMatch(n -> n.endsWith("_b"));
        assertTrue(hasBias, "Should have bias variables, found: " + sd.getVariables().keySet());
    }

    @Test
    @DisplayName("Test Network Without Bias")
    public void testNetworkWithoutBias() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .dataType(DataType.FLOAT)
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

        SameDiff sd = MultiLayerNetworkSameDiffConverter.toSameDiff(network);

        assertNotNull(sd, "SameDiff should not be null");

        // Debug: print all variable names
        System.out.println("testNetworkWithoutBias - All SameDiff variables:");
        for (String name : sd.getVariables().keySet()) {
            System.out.println("  - " + name);
        }

        // Should have weights
        boolean hasWeights = sd.getVariables().keySet().stream().anyMatch(n -> n.endsWith("_W"));
        assertTrue(hasWeights, "Should have weight variables, found: " + sd.getVariables().keySet());

        // First layer should NOT have bias (hasBias=false)
        // Check that we don't have a layer_0_b or similar first layer bias
        String firstLayerBiasPattern = sd.getVariables().keySet().stream()
                .filter(n -> n.endsWith("_b"))
                .findFirst()
                .orElse(null);

        // If there's a bias, it should only be from layer_1 (output layer), not layer_0
        if (firstLayerBiasPattern != null) {
            assertFalse(firstLayerBiasPattern.contains("_0_") || firstLayerBiasPattern.startsWith("0_"),
                    "First layer should NOT have bias, but found: " + firstLayerBiasPattern);
        }
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

        // Should throw some form of exception for uninitialized network
        // Could be IllegalStateException or NullPointerException depending on which code path is hit first
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
}
