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

package org.eclipse.deeplearning4j.dl4jcore.nn.modelimport.keras;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.ComputationGraphSameDiffConverter;
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
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.impl.LossMSE;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for KerasModelImport → SameDiff convenience methods.
 * Since we cannot easily create real Keras HDF5 files in a unit test,
 * these tests verify the converter chain by building DL4J models directly
 * and comparing the converter output with the convenience method pattern.
 */
@NativeTag
@Tag(TagNames.DL4J_OLD_API)
@DisplayName("Keras Model Import SameDiff Test")
public class KerasModelImportSameDiffTest extends BaseDL4JTest {

    private static final double TOLERANCE = 1e-3;

    @Test
    @DisplayName("Sequential model → SameDiff produces equivalent output")
    public void testSequentialToSameDiffEquivalence() {
        // Build a simple sequential model (equivalent to what Keras sequential import would produce)
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .dataType(DataType.FLOAT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .list()
                .layer(new DenseLayer.Builder().nIn(4).nOut(8).activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nIn(8).nOut(4).activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder().nIn(4).nOut(3).activation(Activation.SOFTMAX)
                        .lossFunction(new LossMSE()).build())
                .build();

        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
        mln.init();

        // Forward pass through DL4J
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 4);
        INDArray dl4jOutput = mln.output(input);

        // Convert to SameDiff (this is the same path importKerasSequentialModelToSameDiff uses)
        SameDiff sd = MultiLayerNetworkSameDiffConverter.toSameDiff(mln);

        // Verify structure
        assertNotNull(sd);
        assertTrue(sd.outputs().size() > 0, "SameDiff should have outputs defined");
        assertNotNull(sd.getVariable("input"), "SameDiff should have 'input' placeholder");

        // Forward pass through SameDiff
        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        placeholders.put("input", input);
        Map<String, INDArray> sdOutputs = sd.output(placeholders, sd.outputs());
        INDArray sdOutput = sdOutputs.values().iterator().next();

        // Compare outputs
        assertArrayEquals(dl4jOutput.shape(), sdOutput.shape(), "Output shapes should match");
        assertTrue(dl4jOutput.equalsWithEps(sdOutput, TOLERANCE),
                "DL4J and SameDiff outputs should be numerically equivalent");
    }

    @Test
    @DisplayName("Functional API model → SameDiff produces equivalent output")
    public void testFunctionalToSameDiffEquivalence() {
        // Build a simple computation graph (equivalent to what Keras functional API import would produce)
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .dataType(DataType.FLOAT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .graphBuilder()
                .addInputs("input")
                .addLayer("dense1", new DenseLayer.Builder().nIn(4).nOut(8)
                        .activation(Activation.RELU).build(), "input")
                .addLayer("dense2", new DenseLayer.Builder().nIn(8).nOut(4)
                        .activation(Activation.RELU).build(), "dense1")
                .addLayer("output", new OutputLayer.Builder().nIn(4).nOut(3)
                        .activation(Activation.SOFTMAX).lossFunction(new LossMSE()).build(), "dense2")
                .setOutputs("output")
                .build();

        ComputationGraph cg = new ComputationGraph(conf);
        cg.init();

        // Forward pass through DL4J
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 4);
        INDArray[] dl4jOutputs = cg.output(input);
        INDArray dl4jOutput = dl4jOutputs[0];

        // Convert to SameDiff (this is the same path importKerasModelToSameDiff uses)
        SameDiff sd = ComputationGraphSameDiffConverter.toSameDiff(cg);

        // Verify structure
        assertNotNull(sd);
        assertTrue(sd.outputs().size() > 0, "SameDiff should have outputs defined");
        assertNotNull(sd.getVariable("input"), "SameDiff should have 'input' placeholder");

        // Forward pass through SameDiff
        List<String> networkInputs = cg.getConfiguration().getNetworkInputs();
        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        placeholders.put(networkInputs.get(0), input);
        Map<String, INDArray> sdOutputs = sd.output(placeholders, sd.outputs());
        INDArray sdOutput = sdOutputs.values().iterator().next();

        // Compare outputs
        assertArrayEquals(dl4jOutput.shape(), sdOutput.shape(), "Output shapes should match");
        assertTrue(dl4jOutput.equalsWithEps(sdOutput, TOLERANCE),
                "DL4J and SameDiff outputs should be numerically equivalent");
    }

    @Test
    @DisplayName("SameDiff from sequential model has correct variable structure")
    public void testSequentialSameDiffStructure() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .dataType(DataType.FLOAT)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new DenseLayer.Builder().nIn(4).nOut(8).activation(Activation.RELU)
                        .name("dense1").build())
                .layer(new OutputLayer.Builder().nIn(8).nOut(3).activation(Activation.SOFTMAX)
                        .lossFunction(new LossMSE()).name("output").build())
                .build();

        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
        mln.init();

        SameDiff sd = MultiLayerNetworkSameDiffConverter.toSameDiff(mln);

        // Check weight variables exist
        assertNotNull(sd.getVariable("dense1_W"), "Should have dense1 weight variable");
        assertNotNull(sd.getVariable("dense1_b"), "Should have dense1 bias variable");
        assertNotNull(sd.getVariable("output_W"), "Should have output weight variable");
        assertNotNull(sd.getVariable("output_b"), "Should have output bias variable");

        // Check outputs are set
        assertFalse(sd.outputs().isEmpty(), "SameDiff should have outputs set");
    }
}
