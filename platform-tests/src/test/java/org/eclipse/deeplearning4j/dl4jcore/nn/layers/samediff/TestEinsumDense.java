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
package org.eclipse.deeplearning4j.dl4jcore.nn.layers.samediff;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.EinsumDense;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for EinsumDense layer.
 *
 * @author Adam Gibson
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
@Tag(TagNames.DL4J_OLD_API)
@DisplayName("EinsumDense Layer Tests")
public class TestEinsumDense extends BaseDL4JTest {

    @Test
    @DisplayName("Test EinsumDense Basic Configuration")
    public void testEinsumDenseBasic() {
        // Standard dense-like equation: ab,bc->ac
        // Input: [batch, inputFeatures], Weight: [inputFeatures, outputFeatures] -> Output: [batch, outputFeatures]

        int nIn = 4;
        int nOut = 3;

        EinsumDense einsumDense = new EinsumDense.Builder()
                .name("einsum_dense")
                .equation("ab,bc->ac")
                .kernelShape(nIn, nOut)
                .outputShape(nOut)
                .hasBias(true)
                .build();

        assertNotNull(einsumDense);
        assertEquals("ab,bc->ac", einsumDense.getEquation());
        assertArrayEquals(new long[]{nIn, nOut}, einsumDense.getKernelShape());
    }

    @Test
    @DisplayName("Test EinsumDense Forward Pass")
    public void testEinsumDenseForward() {
        int batchSize = 2;
        int nIn = 4;
        int nOut = 3;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .dataType(DataType.DOUBLE)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.01))
                .list()
                .layer(new EinsumDense.Builder()
                        .equation("ab,bc->ac")
                        .kernelShape(nIn, nOut)
                        .outputShape(nOut)
                        .hasBias(true)
                        .build())
                .layer(new OutputLayer.Builder()
                        .nIn(nOut)
                        .nOut(2)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        INDArray input = Nd4j.rand(DataType.DOUBLE, batchSize, nIn);
        INDArray output = net.output(input);

        assertNotNull(output);
        assertArrayEquals(new long[]{batchSize, 2}, output.shape());

        // Output should be valid probabilities (sum to 1 for each sample)
        for (int i = 0; i < batchSize; i++) {
            double sum = output.getRow(i).sumNumber().doubleValue();
            assertEquals(1.0, sum, 1e-5, "Softmax output should sum to 1");
        }
    }

    @Test
    @DisplayName("Test EinsumDense Without Bias")
    public void testEinsumDenseNoBias() {
        int nIn = 4;
        int nOut = 3;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .dataType(DataType.DOUBLE)
                .list()
                .layer(new EinsumDense.Builder()
                        .equation("ab,bc->ac")
                        .kernelShape(nIn, nOut)
                        .outputShape(nOut)
                        .hasBias(false)
                        .build())
                .layer(new OutputLayer.Builder()
                        .nIn(nOut)
                        .nOut(2)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        // Should only have weight parameter, no bias
        Map<String, INDArray> params = net.getLayer(0).paramTable();
        assertTrue(params.containsKey("W"));
        assertFalse(params.containsKey("b") && params.get("b") != null);

        INDArray input = Nd4j.rand(DataType.DOUBLE, 2, nIn);
        INDArray output = net.output(input);
        assertNotNull(output);
    }

    @Test
    @DisplayName("Test EinsumDense 3D Input: abc,cd->abd")
    public void testEinsumDense3DInput() {
        // For sequence data: abc,cd->abd
        // Input: [batch, seq, features], Weight: [features, outputFeatures] -> Output: [batch, seq, outputFeatures]

        int batchSize = 2;
        int seqLength = 5;
        int nIn = 4;
        int nOut = 3;

        EinsumDense einsumDense = new EinsumDense.Builder()
                .equation("abc,cd->abd")
                .kernelShape(nIn, nOut)
                .outputShape(seqLength, nOut)
                .hasBias(true)
                .build();

        assertNotNull(einsumDense);
        assertEquals("abc,cd->abd", einsumDense.getEquation());
    }

    @Test
    @DisplayName("Test EinsumDense Parameter Shapes")
    public void testEinsumDenseParameterShapes() {
        int nIn = 8;
        int nOut = 4;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .dataType(DataType.DOUBLE)
                .list()
                .layer(new EinsumDense.Builder()
                        .equation("ab,bc->ac")
                        .kernelShape(nIn, nOut)
                        .outputShape(nOut)
                        .hasBias(true)
                        .build())
                .layer(new OutputLayer.Builder()
                        .nIn(nOut)
                        .nOut(2)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        Map<String, INDArray> params = net.getLayer(0).paramTable();

        // Check weight shape
        INDArray weights = params.get("W");
        assertNotNull(weights);
        assertArrayEquals(new long[]{nIn, nOut}, weights.shape());
    }

    @Test
    @DisplayName("Test EinsumDense With Different Workspace Modes")
    public void testEinsumDenseWorkspaceModes() {
        for (WorkspaceMode wsm : new WorkspaceMode[]{WorkspaceMode.ENABLED, WorkspaceMode.NONE}) {
            int nIn = 4;
            int nOut = 3;

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .dataType(DataType.DOUBLE)
                    .trainingWorkspaceMode(wsm)
                    .inferenceWorkspaceMode(wsm)
                    .list()
                    .layer(new EinsumDense.Builder()
                            .equation("ab,bc->ac")
                            .kernelShape(nIn, nOut)
                            .outputShape(nOut)
                            .hasBias(true)
                            .build())
                    .layer(new OutputLayer.Builder()
                            .nIn(nOut)
                            .nOut(2)
                            .activation(Activation.SOFTMAX)
                            .lossFunction(LossFunctions.LossFunction.MCXENT)
                            .build())
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            INDArray input = Nd4j.rand(DataType.DOUBLE, 2, nIn);

            // Test forward pass only (Einsum gradient not yet implemented)
            INDArray output = net.output(input);
            assertNotNull(output);
            assertArrayEquals(new long[]{2, 2}, output.shape());

            // Note: Training (backpropagation) is not yet supported for Einsum
            // because Einsum.doDiff() is not implemented

            log.info("EinsumDense test passed with workspace mode: {}", wsm);
        }
    }

    @Test
    @DisplayName("Test EinsumDense Builder Validation")
    public void testEinsumDenseBuilderValidation() {
        // Missing equation should fail
        assertThrows(IllegalStateException.class, () -> {
            new EinsumDense.Builder()
                    .kernelShape(4, 3)
                    .outputShape(3)
                    .build();
        });

        // Missing kernel shape should fail
        assertThrows(IllegalStateException.class, () -> {
            new EinsumDense.Builder()
                    .equation("ab,bc->ac")
                    .outputShape(3)
                    .build();
        });

        // Missing output shape should fail
        assertThrows(IllegalStateException.class, () -> {
            new EinsumDense.Builder()
                    .equation("ab,bc->ac")
                    .kernelShape(4, 3)
                    .build();
        });
    }

    @Test
    @DisplayName("Test EinsumDense Multiple Batches")
    public void testEinsumDenseMultipleBatches() {
        int nIn = 4;
        int nOut = 3;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .dataType(DataType.DOUBLE)
                .list()
                .layer(new EinsumDense.Builder()
                        .equation("ab,bc->ac")
                        .kernelShape(nIn, nOut)
                        .outputShape(nOut)
                        .hasBias(true)
                        .build())
                .layer(new OutputLayer.Builder()
                        .nIn(nOut)
                        .nOut(2)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        // Test with different batch sizes
        for (int batchSize : new int[]{1, 2, 8, 32}) {
            INDArray input = Nd4j.rand(DataType.DOUBLE, batchSize, nIn);
            INDArray output = net.output(input);

            assertNotNull(output);
            assertArrayEquals(new long[]{batchSize, 2}, output.shape());
        }
    }
}
