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

package org.deeplearning4j.nn.conf;


import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.SubsetVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.conf.misc.TestGraphVertex;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class ComputationGraphConfigurationTest extends BaseDL4JTest {

    @Test
    public void testJSONBasic() {
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1)).updater(new NoOp())
                .graphBuilder().addInputs("input")
                .addLayer("firstLayer",
                        new DenseLayer.Builder().nIn(4).nOut(5).activation(Activation.TANH).build(),
                        "input")
                .addLayer("outputLayer",
                        new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT)
                                .activation(Activation.SOFTMAX).nIn(5).nOut(3).build(),
                        "firstLayer")
                .setOutputs("outputLayer").build();

        String json = conf.toJson();
        ComputationGraphConfiguration conf2 = ComputationGraphConfiguration.fromJson(json);

        assertEquals(json, conf2.toJson());
        assertEquals(conf, conf2);
    }

    @Test
    public void testJSONBasic2() {
        ComputationGraphConfiguration conf =
                new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .graphBuilder().addInputs("input")
                        .addLayer("cnn1",
                                new ConvolutionLayer.Builder(2, 2).stride(2, 2).nIn(1).nOut(5)
                                        .build(),
                                "input")
                        .addLayer("cnn2",
                                new ConvolutionLayer.Builder(2, 2).stride(2, 2).nIn(1).nOut(5)
                                        .build(),
                                "input")
                        .addLayer("max1",
                                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                        .kernelSize(2, 2).build(),
                                "cnn1", "cnn2")
                        .addLayer("dnn1", new DenseLayer.Builder().nOut(7).build(), "max1")
                        .addLayer("max2", new SubsamplingLayer.Builder().build(), "max1")
                        .addLayer("output", new OutputLayer.Builder().nIn(7).nOut(10).activation(Activation.SOFTMAX).build(), "dnn1",
                                "max2")
                        .setOutputs("output")
                        .inputPreProcessor("cnn1", new FeedForwardToCnnPreProcessor(32, 32, 3))
                        .inputPreProcessor("cnn2", new FeedForwardToCnnPreProcessor(32, 32, 3))
                        .inputPreProcessor("dnn1", new CnnToFeedForwardPreProcessor(8, 8, 5))
                        .build();

        String json = conf.toJson();
        ComputationGraphConfiguration conf2 = ComputationGraphConfiguration.fromJson(json);

        assertEquals(json, conf2.toJson());
        assertEquals(conf, conf2);
    }

    @Test
    public void testJSONWithGraphNodes() {

        ComputationGraphConfiguration conf =
                new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .graphBuilder().addInputs("input1", "input2")
                        .addLayer("cnn1",
                                new ConvolutionLayer.Builder(2, 2).stride(2, 2).nIn(1).nOut(5)
                                        .build(),
                                "input1")
                        .addLayer("cnn2",
                                new ConvolutionLayer.Builder(2, 2).stride(2, 2).nIn(1).nOut(5)
                                        .build(),
                                "input2")
                        .addVertex("merge1", new MergeVertex(), "cnn1", "cnn2")
                        .addVertex("subset1", new SubsetVertex(0, 1), "merge1")
                        .addLayer("dense1", new DenseLayer.Builder().nIn(20).nOut(5).build(), "subset1")
                        .addLayer("dense2", new DenseLayer.Builder().nIn(20).nOut(5).build(), "subset1")
                        .addVertex("add", new ElementWiseVertex(ElementWiseVertex.Op.Add), "dense1",
                                "dense2")
                        .addLayer("out", new OutputLayer.Builder().nIn(1).nOut(1).lossFunction(LossFunctions.LossFunction.MSE).build(), "add")
                        .setOutputs("out").build();

        String json = conf.toJson();
        System.out.println(json);

        ComputationGraphConfiguration conf2 = ComputationGraphConfiguration.fromJson(json);

        assertEquals(json, conf2.toJson());
        assertEquals(conf, conf2);
    }

    @Test
    public void testInvalidConfigurations() {

        //Test no inputs for a layer:
        try {
            new NeuralNetConfiguration.Builder().graphBuilder().addInputs("input1")
                    .addLayer("dense1", new DenseLayer.Builder().nIn(2).nOut(2).build(), "input1")
                    .addLayer("out", new OutputLayer.Builder().nIn(2).nOut(2).build()).setOutputs("out")
                    .build();
            fail("No exception thrown for invalid configuration");
        } catch (IllegalStateException e) {
            //OK - exception is good
            //e.printStackTrace();
        }

        //Test no network inputs
        try {
            new NeuralNetConfiguration.Builder().graphBuilder()
                    .addLayer("dense1", new DenseLayer.Builder().nIn(2).nOut(2).build(), "input1")
                    .addLayer("out", new OutputLayer.Builder().nIn(2).nOut(2).build(), "dense1")
                    .setOutputs("out").build();
            fail("No exception thrown for invalid configuration");
        } catch (IllegalStateException e) {
            //OK - exception is good
            //e.printStackTrace();
        }

        //Test no network outputs
        try {
            new NeuralNetConfiguration.Builder().graphBuilder().addInputs("input1")
                    .addLayer("dense1", new DenseLayer.Builder().nIn(2).nOut(2).build(), "input1")
                    .addLayer("out", new OutputLayer.Builder().nIn(2).nOut(2).build(), "dense1").build();
            fail("No exception thrown for invalid configuration");
        } catch (IllegalStateException e) {
            //OK - exception is good
            //e.printStackTrace();
        }

        //Test: invalid input
        try {
            new NeuralNetConfiguration.Builder().graphBuilder().addInputs("input1")
                    .addLayer("dense1", new DenseLayer.Builder().nIn(2).nOut(2).build(), "input1")
                    .addLayer("out", new OutputLayer.Builder().nIn(2).nOut(2).build(), "thisDoesntExist")
                    .setOutputs("out").build();
            fail("No exception thrown for invalid configuration");
        } catch (IllegalStateException e) {
            //OK - exception is good
            //e.printStackTrace();
        }

        //Test: graph with cycles
        try {
            ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("input1")
                    .addLayer("dense1", new DenseLayer.Builder().nIn(2).nOut(2).build(), "input1", "dense3")
                    .addLayer("dense2", new DenseLayer.Builder().nIn(2).nOut(2).build(), "dense1")
                    .addLayer("dense3", new DenseLayer.Builder().nIn(2).nOut(2).build(), "dense2")
                    .addLayer("out", new OutputLayer.Builder().nIn(2).nOut(2).lossFunction(LossFunctions.LossFunction.MSE).build(), "dense1")
                    .setOutputs("out").build();
            //Cycle detection happens in ComputationGraph.init()
            ComputationGraph graph = new ComputationGraph(conf);
            graph.init();

            fail("No exception thrown for invalid configuration");
        } catch (IllegalStateException e) {
            //OK - exception is good
            //e.printStackTrace();
        }
    }


    @Test
    public void testConfigurationWithRuntimeJSONSubtypes() {
        //Idea: suppose someone wants to use a ComputationGraph with a custom GraphVertex
        // (i.e., one not built into DL4J). Check that this works for JSON serialization
        // using runtime/reflection subtype mechanism in ComputationGraphConfiguration.fromJson()
        //Check a standard GraphVertex implementation, plus a static inner graph vertex

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("in")
                .addVertex("test", new TestGraphVertex(3, 7), "in")
                .addVertex("test2", new StaticInnerGraphVertex(4, 5), "in").setOutputs("test", "test2").build();

        String json = conf.toJson();
        System.out.println(json);

        ComputationGraphConfiguration conf2 = ComputationGraphConfiguration.fromJson(json);

        assertEquals(conf, conf2);
        assertEquals(json, conf2.toJson());

        TestGraphVertex tgv = (TestGraphVertex) conf2.getVertices().get("test");
        assertEquals(3, tgv.getFirstVal());
        assertEquals(7, tgv.getSecondVal());

        StaticInnerGraphVertex sigv = (StaticInnerGraphVertex) conf.getVertices().get("test2");
        assertEquals(4, sigv.getFirstVal());
        assertEquals(5, sigv.getSecondVal());
    }

    @Test
    public void testOutputOrderDoesntChangeWhenCloning() {
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("in")
                .addLayer("out1", new OutputLayer.Builder().nIn(1).nOut(1).build(), "in")
                .addLayer("out2", new OutputLayer.Builder().nIn(1).nOut(1).build(), "in")
                .addLayer("out3", new OutputLayer.Builder().nIn(1).nOut(1).build(), "in")
                .validateOutputLayerConfig(false)
                .setOutputs("out1", "out2", "out3").build();

        ComputationGraphConfiguration cloned = conf.clone();

        String json = conf.toJson();
        String jsonCloned = cloned.toJson();

        assertEquals(json, jsonCloned);
    }

    @Test
    public void testAllowDisconnectedLayers() {
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("in")
                .addLayer("bidirectional",
                        new Bidirectional(new LSTM.Builder().activation(Activation.TANH).nOut(10).build()),
                        "in")
                .addLayer("out", new RnnOutputLayer.Builder().nOut(6)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .build(), "bidirectional")
                .addLayer("disconnected_layer",
                        new Bidirectional(new LSTM.Builder().activation(Activation.TANH).nOut(10).build()),
                        "in")
                .setOutputs("out")
                .setInputTypes(new InputType.InputTypeRecurrent(10, 12))
                .allowDisconnected(true)
                .build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();
    }

    @Test
    public void testBidirectionalGraphSummary() {
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("in")
                .addLayer("bidirectional",
                        new Bidirectional(new LSTM.Builder().activation(Activation.TANH).nOut(10).build()),
                        "in")
                .addLayer("out", new RnnOutputLayer.Builder().nOut(6)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .build(), "bidirectional")
                .setOutputs("out")
                .setInputTypes(new InputType.InputTypeRecurrent(10, 12))
                .build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();
        graph.summary();
    }

    @AllArgsConstructor
    @NoArgsConstructor
    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class StaticInnerGraphVertex extends GraphVertex {

        private int firstVal;
        private int secondVal;

        @Override
        public GraphVertex clone() {
            return new TestGraphVertex(firstVal, secondVal);
        }

        @Override
        public int numParams(boolean backprop) {
            return 0;
        }

        @Override
        public int minVertexInputs() {
            return 1;
        }

        @Override
        public int maxVertexInputs() {
            return 1;
        }

        @Override
        public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx,
                                                                          INDArray paramsView, boolean initializeParams) {
            throw new UnsupportedOperationException("Not supported");
        }

        @Override
        public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
            throw new UnsupportedOperationException();
        }

        @Override
        public MemoryReport getMemoryReport(InputType... inputTypes) {
            throw new UnsupportedOperationException();
        }
    }


    @Test
    public void testInvalidOutputLayer(){
        /*
        Test case (invalid configs)
        1. nOut=1 + softmax
        2. mcxent + tanh
        3. xent + softmax
        4. xent + relu
        5. mcxent + sigmoid
         */

        LossFunctions.LossFunction[] lf = new LossFunctions.LossFunction[]{
                LossFunctions.LossFunction.MCXENT, LossFunctions.LossFunction.MCXENT, LossFunctions.LossFunction.XENT,
                LossFunctions.LossFunction.XENT, LossFunctions.LossFunction.MCXENT};
        int[] nOut = new int[]{1, 3, 3, 3, 3};
        Activation[] activations = new Activation[]{Activation.SOFTMAX, Activation.TANH, Activation.SOFTMAX, Activation.RELU, Activation.SIGMOID};
        for( int i=0; i<lf.length; i++ ){
            for(boolean lossLayer : new boolean[]{false, true}) {
                for (boolean validate : new boolean[]{true, false}) {
                    String s = "nOut=" + nOut[i] + ",lossFn=" + lf[i] + ",lossLayer=" + lossLayer + ",validate=" + validate;
                    if(nOut[i] == 1 && lossLayer)
                        continue;   //nOuts are not availabel in loss layer, can't expect it to detect this case
                    try {
                        new NeuralNetConfiguration.Builder()
                                .graphBuilder()
                                .addInputs("in")
                                .layer("0", new DenseLayer.Builder().nIn(10).nOut(10).build(), "in")
                                .layer("1",
                                        !lossLayer ? new OutputLayer.Builder().nIn(10).nOut(nOut[i]).activation(activations[i]).lossFunction(lf[i]).build()
                                                : new LossLayer.Builder().activation(activations[i]).lossFunction(lf[i]).build(), "0")
                                .setOutputs("1")
                                .validateOutputLayerConfig(validate)
                                .build();
                        if (validate) {
                            fail("Expected exception: " + s);
                        }
                    } catch (DL4JInvalidConfigException e) {
                        if (validate) {
                            assertTrue(s, e.getMessage().toLowerCase().contains("invalid output"));
                        } else {
                            fail("Validation should not be enabled");
                        }
                    }
                }
            }
        }
    }
}
