/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.gradientcheck;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.graph.*;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.graph.rnn.ReverseTimeSeriesVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.*;

@Slf4j
public class GradientCheckTestsComputationGraph extends BaseDL4JTest {

    public static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-9;

    static {
        Nd4j.setDataType(DataType.DOUBLE);
    }

    @Override
    public long getTimeoutMilliseconds() {
        return 999999999L;
    }

    @Test
    public void testBasicIris() {
        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                .dataType(DataType.DOUBLE)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .dist(new NormalDistribution(0, 1)).updater(new NoOp())
                .graphBuilder().addInputs("input")
                .addLayer("firstLayer",
                        new DenseLayer.Builder().nIn(4).nOut(5).activation(Activation.TANH).build(),
                        "input")
                .addLayer("outputLayer",
                        new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT)
                                .activation(Activation.SOFTMAX).nIn(5).nOut(3).build(),
                        "firstLayer")
                .setOutputs("outputLayer").build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        Nd4j.getRandom().setSeed(12345);
        long nParams = graph.numParams();
        INDArray newParams = Nd4j.rand(new long[]{1, nParams});
        graph.setParams(newParams);

        DataSet ds = new IrisDataSetIterator(150, 150).next();
        INDArray min = ds.getFeatures().min(0);
        INDArray max = ds.getFeatures().max(0);
        ds.getFeatures().subiRowVector(min).diviRowVector(max.sub(min));
        INDArray input = ds.getFeatures();
        INDArray labels = ds.getLabels();

        if (PRINT_RESULTS) {
            System.out.println("testBasicIris()");
//            for (int j = 0; j < graph.getNumLayers(); j++)
//                System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(graph).inputs(new INDArray[]{input})
                .labels(new INDArray[]{labels}));

        String msg = "testBasicIris()";
        assertTrue(msg, gradOK);
        TestUtils.testModelSerialization(graph);
    }

    @Test
    public void testBasicIrisWithMerging() {
        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                .dataType(DataType.DOUBLE)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .dist(new NormalDistribution(0, 1)).updater(new NoOp())
                .graphBuilder().addInputs("input")
                .addLayer("l1", new DenseLayer.Builder().nIn(4).nOut(5).activation(Activation.TANH).build(),
                        "input")
                .addLayer("l2", new DenseLayer.Builder().nIn(4).nOut(5).activation(Activation.TANH).build(),
                        "input")
                .addVertex("merge", new MergeVertex(), "l1", "l2")
                .addLayer("outputLayer",
                        new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT)
                                .activation(Activation.SOFTMAX).nIn(5 + 5).nOut(3).build(),
                        "merge")
                .setOutputs("outputLayer").build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        int numParams = (4 * 5 + 5) + (4 * 5 + 5) + (10 * 3 + 3);
        assertEquals(numParams, graph.numParams());

        Nd4j.getRandom().setSeed(12345);
        long nParams = graph.numParams();
        INDArray newParams = Nd4j.rand(new long[]{1, nParams});
        graph.setParams(newParams);

        DataSet ds = new IrisDataSetIterator(150, 150).next();
        INDArray min = ds.getFeatures().min(0);
        INDArray max = ds.getFeatures().max(0);
        ds.getFeatures().subiRowVector(min).diviRowVector(max.sub(min));
        INDArray input = ds.getFeatures();
        INDArray labels = ds.getLabels();

        if (PRINT_RESULTS) {
            System.out.println("testBasicIrisWithMerging()");
//            for (int j = 0; j < graph.getNumLayers(); j++)
//                System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(graph).inputs(new INDArray[]{input})
                .labels(new INDArray[]{labels}));

        String msg = "testBasicIrisWithMerging()";
        assertTrue(msg, gradOK);
        TestUtils.testModelSerialization(graph);
    }

    @Test
    public void testBasicIrisWithElementWiseNode() {

        ElementWiseVertex.Op[] ops = new ElementWiseVertex.Op[] {ElementWiseVertex.Op.Add,
                ElementWiseVertex.Op.Subtract, ElementWiseVertex.Op.Product, ElementWiseVertex.Op.Average, ElementWiseVertex.Op.Max};

        for (ElementWiseVertex.Op op : ops) {

            Nd4j.getRandom().setSeed(12345);
            ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                    .dataType(DataType.DOUBLE)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .dist(new NormalDistribution(0, 1))
                    .updater(new NoOp()).graphBuilder().addInputs("input")
                    .addLayer("l1", new DenseLayer.Builder().nIn(4).nOut(5).activation(Activation.TANH).build(),
                            "input")
                    .addLayer("l2", new DenseLayer.Builder().nIn(4).nOut(5).activation(Activation.SIGMOID)
                            .build(), "input")
                    .addVertex("elementwise", new ElementWiseVertex(op), "l1", "l2")
                    .addLayer("outputLayer",
                            new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT)
                                    .activation(Activation.SOFTMAX).nIn(5).nOut(3).build(),
                            "elementwise")
                    .setOutputs("outputLayer").build();

            ComputationGraph graph = new ComputationGraph(conf);
            graph.init();

            int numParams = (4 * 5 + 5) + (4 * 5 + 5) + (5 * 3 + 3);
            assertEquals(numParams, graph.numParams());

            Nd4j.getRandom().setSeed(12345);
            long nParams = graph.numParams();
            INDArray newParams = Nd4j.rand(new long[]{1, nParams});
            graph.setParams(newParams);

            DataSet ds = new IrisDataSetIterator(150, 150).next();
            INDArray min = ds.getFeatures().min(0);
            INDArray max = ds.getFeatures().max(0);
            ds.getFeatures().subiRowVector(min).diviRowVector(max.sub(min));
            INDArray input = ds.getFeatures();
            INDArray labels = ds.getLabels();

            if (PRINT_RESULTS) {
                System.out.println("testBasicIrisWithElementWiseVertex(op=" + op + ")");
//                for (int j = 0; j < graph.getNumLayers(); j++)
//                    System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(graph).inputs(new INDArray[]{input})
                    .labels(new INDArray[]{labels}));

            String msg = "testBasicIrisWithElementWiseVertex(op=" + op + ")";
            assertTrue(msg, gradOK);
            TestUtils.testModelSerialization(graph);
        }
    }

    @Test
    public void testBasicIrisWithElementWiseNodeInputSizeGreaterThanTwo() {

        ElementWiseVertex.Op[] ops =
                new ElementWiseVertex.Op[] {ElementWiseVertex.Op.Add, ElementWiseVertex.Op.Product, ElementWiseVertex.Op.Average, ElementWiseVertex.Op.Max};

        for (ElementWiseVertex.Op op : ops) {

            Nd4j.getRandom().setSeed(12345);
            ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                    .dataType(DataType.DOUBLE)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .dist(new NormalDistribution(0, 1))
                    .updater(new NoOp()).graphBuilder().addInputs("input")
                    .addLayer("l1", new DenseLayer.Builder().nIn(4).nOut(5).activation(Activation.TANH).build(),
                            "input")
                    .addLayer("l2", new DenseLayer.Builder().nIn(4).nOut(5).activation(Activation.SIGMOID)
                            .build(), "input")
                    .addLayer("l3", new DenseLayer.Builder().nIn(4).nOut(5).activation(Activation.RELU).build(),
                            "input")
                    .addVertex("elementwise", new ElementWiseVertex(op), "l1", "l2", "l3")
                    .addLayer("outputLayer",
                            new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT)
                                    .activation(Activation.SOFTMAX).nIn(5).nOut(3).build(),
                            "elementwise")
                    .setOutputs("outputLayer").build();

            ComputationGraph graph = new ComputationGraph(conf);
            graph.init();

            int numParams = (4 * 5 + 5) + (4 * 5 + 5) + (4 * 5 + 5) + (5 * 3 + 3);
            assertEquals(numParams, graph.numParams());

            Nd4j.getRandom().setSeed(12345);
            long nParams = graph.numParams();
            INDArray newParams = Nd4j.rand(new long[]{1, nParams});
            graph.setParams(newParams);

            DataSet ds = new IrisDataSetIterator(150, 150).next();
            INDArray min = ds.getFeatures().min(0);
            INDArray max = ds.getFeatures().max(0);
            ds.getFeatures().subiRowVector(min).diviRowVector(max.sub(min));
            INDArray input = ds.getFeatures();
            INDArray labels = ds.getLabels();

            if (PRINT_RESULTS) {
                System.out.println("testBasicIrisWithElementWiseVertex(op=" + op + ")");
//                for (int j = 0; j < graph.getNumLayers(); j++)
//                    System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(graph).inputs(new INDArray[]{input})
                    .labels(new INDArray[]{labels}));

            String msg = "testBasicIrisWithElementWiseVertex(op=" + op + ")";
            assertTrue(msg, gradOK);
            TestUtils.testModelSerialization(graph);
        }
    }

    @Test
    public void testElementWiseVertexBroadcast(){

        ElementWiseVertex.Op[] ops =
                new ElementWiseVertex.Op[] {ElementWiseVertex.Op.Add, ElementWiseVertex.Op.Average,
                        ElementWiseVertex.Op.Subtract, ElementWiseVertex.Op.Max, ElementWiseVertex.Op.Product};

        for(boolean firstSmaller : new boolean[]{false, true}) {
            for (ElementWiseVertex.Op op : ops) {
                ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                        .updater(new NoOp())
                        .dataType(DataType.DOUBLE)
                        .activation(Activation.TANH)
                        .seed(12345)
                        .graphBuilder()
                        .addInputs("in")
                        .setOutputs("out")
                        .layer("l1", new DenseLayer.Builder().nIn(3).nOut(firstSmaller ? 1 : 3).build(), "in")   //[mb,3]
                        .layer("l2", new DenseLayer.Builder().nIn(3).nOut(firstSmaller ? 3 : 1).build(), "in")   //[mb,1]
                        .addVertex("ew", new ElementWiseVertex(op), "l1", "l2")
                        .layer("out", new OutputLayer.Builder().nIn(3).nOut(2).lossFunction(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).build(), "ew")
                        .build();

                ComputationGraph graph = new ComputationGraph(conf);
                graph.init();

                for (int mb : new int[]{1, 5}) {
                    String msg = (firstSmaller ? "first smaller, " : "second smaller, ") + "mb=" + mb + ", op=" + op;

                    log.info("Test: {}", msg);

                    INDArray in = Nd4j.rand(DataType.FLOAT, mb, 3);

                    INDArray out = graph.outputSingle(in);
                    assertArrayEquals(new long[]{mb, 2}, out.shape());

                    INDArray labels = TestUtils.randomOneHot(mb, 2);

                    graph.fit(new DataSet(in, labels));

                    boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(graph).inputs(new INDArray[]{in})
                            .labels(new INDArray[]{labels}));
                    assertTrue(msg, gradOK);
                    TestUtils.testModelSerialization(graph);
                }
            }
        }
    }

    @Test
    public void testCnnDepthMerge() {

        for(CNN2DFormat format : CNN2DFormat.values()) {

            String msg = "testCnnDepthMerge - " + format;

            Nd4j.getRandom().setSeed(12345);
            ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                    .dataType(DataType.DOUBLE)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .dist(new NormalDistribution(0, 0.1))
                    .updater(new NoOp()).graphBuilder().addInputs("input")
                    .addLayer("l1", new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1).padding(0, 0)
                            .dataFormat(format)
                            .nIn(2).nOut(2).activation(Activation.TANH).build(), "input")
                    .addLayer("l2", new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1)
                            .padding(0, 0).dataFormat(format)
                            .nIn(2).nOut(2).activation(Activation.TANH).build(), "input")
                    .addVertex("merge", new MergeVertex(), "l1", "l2")
                    .addLayer("outputLayer",
                            new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT)
                                    .activation(Activation.SOFTMAX).nIn(5 * 5 * (2 + 2)).nOut(3)
                                    .build(),
                            "merge")
                    .setOutputs("outputLayer")
                    .setInputTypes(InputType.convolutional(6, 6, 2, format))
                    .build();

            ComputationGraph graph = new ComputationGraph(conf);
            graph.init();

            Random r = new Random(12345);
            INDArray input = Nd4j.rand(DataType.DOUBLE, format == CNN2DFormat.NCHW ? new long[]{5,2,6,6} : new long[]{5,6,6,2});
            INDArray labels = Nd4j.zeros(5, 3);
            for (int i = 0; i < 5; i++)
                labels.putScalar(new int[]{i, r.nextInt(3)}, 1.0);

            if (PRINT_RESULTS) {
                System.out.println(msg);
//            for (int j = 0; j < graph.getNumLayers(); j++)
//                System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(graph).inputs(new INDArray[]{input})
                    .labels(new INDArray[]{labels}));

            assertTrue(msg, gradOK);
            TestUtils.testModelSerialization(graph);
        }
    }

    @Test
    public void testRNNWithMerging() {
        for(RNNFormat format : RNNFormat.values()) {

            String msg = "testRNNWithMerging - " + format;
            int timeSeriesLength = 4;
            int batchSize = 2;
            int inputChannels = 3;
            int outSize = 3;
            Nd4j.getRandom().setSeed(12345);
            ComputationGraphConfiguration conf =
                    new NeuralNetConfiguration.Builder().seed(12345)
                            .dataType(DataType.DOUBLE)
                            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                            .dist(new UniformDistribution(0.2, 0.6))
                            .updater(new NoOp()).graphBuilder().addInputs("input")
                            .setOutputs("out")
                            .addLayer("rnn1",
                                    new SimpleRnn.Builder().nOut(3)
                                            .activation(Activation.TANH).build(),
                                    "input")
                            .addLayer("rnn2",
                                    new SimpleRnn.Builder().nOut(3)
                                            .activation(Activation.TANH).build(),
                                    "rnn1")
                            .addLayer("dense1",
                                    new DenseLayer.Builder().nOut(3)
                                            .activation(Activation.SIGMOID).build(),
                                    "rnn1")
                            .addLayer("rnn3",
                                    new SimpleRnn.Builder().nOut(3)
                                            .activation(Activation.TANH).build(),
                                    "dense1")
                            .addVertex("merge", new MergeVertex(), "rnn2", "rnn3")
                            .addLayer("out", new RnnOutputLayer.Builder().nOut(outSize)

                                            .activation(Activation.SOFTMAX)
                                            .lossFunction(LossFunctions.LossFunction.MCXENT).build(),
                                    "merge")
                            .setInputTypes(InputType.recurrent(inputChannels,timeSeriesLength, format))
                            .build();

            ComputationGraph graph = new ComputationGraph(conf);
            graph.init();
            System.out.println("Configuration for " + format + " " + conf);

            INDArray input = Nd4j.rand(DataType.DOUBLE, format == RNNFormat.NCW ? new long[]{batchSize, inputChannels, timeSeriesLength} : new long[]{batchSize,timeSeriesLength,inputChannels});
            INDArray labels = TestUtils.randomOneHotTimeSeries(format, batchSize, outSize, timeSeriesLength, new Random(12345));

            if (PRINT_RESULTS) {
                System.out.println(msg);
//            for (int j = 0; j < graph.getNumLayers(); j++)
//                System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(graph).inputs(new INDArray[]{input})
                    .labels(new INDArray[]{labels}));

            assertTrue(msg, gradOK);
            TestUtils.testModelSerialization(graph);

        }
    }

    @Test
    public void testLSTMWithSubset() {
        Nd4j.getRandom().setSeed(1234);
        int batchSize = 2;
        int timeSeriesLength = 4;
        int inLength = 3;
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(1234)
                .dataType(DataType.DOUBLE)
                .weightInit(new NormalDistribution(0, 1))
                .updater(new NoOp()).graphBuilder().addInputs("input").setOutputs("out")
                .addLayer("lstm1", new LSTM.Builder().nOut(6).activation(Activation.TANH).build(),
                        "input")
                .addVertex("subset", new SubsetVertex(0, 2), "lstm1")
                .addLayer("out", new RnnOutputLayer.Builder().nOut(2).activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build(), "subset")
                .setInputTypes(InputType.recurrent(inLength,timeSeriesLength,RNNFormat.NCW))
                .build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        INDArray input = Nd4j.rand(new int[] {batchSize, inLength, timeSeriesLength});
        INDArray labels = TestUtils.randomOneHotTimeSeries(batchSize, 2, timeSeriesLength);

        if (PRINT_RESULTS) {
            System.out.println("testLSTMWithSubset()");
//            for (int j = 0; j < graph.getNumLayers(); j++)
//                System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(graph).inputs(new INDArray[]{input})
                .labels(new INDArray[]{labels}));

        String msg = "testLSTMWithSubset()";
        assertTrue(msg, gradOK);
        TestUtils.testModelSerialization(graph);
    }

    @Test
    public void testLSTMWithLastTimeStepVertex() {

        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                .dataType(DataType.DOUBLE)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .dist(new NormalDistribution(0, 1))
                .updater(new NoOp()).graphBuilder().addInputs("input").setOutputs("out")
                .addLayer("lstm1", new LSTM.Builder().nIn(3).nOut(4).activation(Activation.TANH).build(),
                        "input")
                .addVertex("lastTS", new LastTimeStepVertex("input"), "lstm1")
                .addLayer("out", new OutputLayer.Builder().nIn(4).nOut(2).activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build(), "lastTS")
                .build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        Random r = new Random(12345);
        INDArray input = Nd4j.rand(new int[] {2, 3, 4});
        INDArray labels = TestUtils.randomOneHot(2, 2); //Here: labels are 2d (due to LastTimeStepVertex)

        if (PRINT_RESULTS) {
            System.out.println("testLSTMWithLastTimeStepVertex()");
//            for (int j = 0; j < graph.getNumLayers(); j++)
//                System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
        }

        //First: test with no input mask array
        boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(graph).inputs(new INDArray[]{input})
                .labels(new INDArray[]{labels}));

        String msg = "testLSTMWithLastTimeStepVertex()";
        assertTrue(msg, gradOK);

        //Second: test with input mask arrays.
        INDArray inMask = Nd4j.zeros(3, 4);
        inMask.putRow(0, Nd4j.create(new double[] {1, 1, 0, 0}));
        inMask.putRow(1, Nd4j.create(new double[] {1, 1, 1, 0}));
        inMask.putRow(2, Nd4j.create(new double[] {1, 1, 1, 1}));
        gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(graph).inputs(new INDArray[]{input})
                .labels(new INDArray[]{labels}).inputMask(new INDArray[]{inMask}));

        assertTrue(msg, gradOK);
        TestUtils.testModelSerialization(graph);
    }

    @Test
    public void testLSTMWithDuplicateToTimeSeries() {

        int batchSize = 2;
        int outSize = 2;
        int timeSeriesLength = 4;
        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf =
                new NeuralNetConfiguration.Builder().seed(12345)
                        .dataType(DataType.DOUBLE)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .dist(new NormalDistribution(0, 1))
                        .updater(new NoOp()).graphBuilder()
                        .addInputs("input1", "input2").setOutputs("out")
                        .addLayer("lstm1",
                                new LSTM.Builder().nIn(3).nOut(3)
                                        .activation(Activation.TANH).build(),
                                "input1")
                        .addLayer("lstm2",
                                new LSTM.Builder().nIn(2).nOut(4)
                                        .activation(Activation.SOFTSIGN).build(),
                                "input2")
                        .addVertex("lastTS", new LastTimeStepVertex("input2"), "lstm2")
                        .addVertex("duplicate", new DuplicateToTimeSeriesVertex("input2"), "lastTS")
                        .addLayer("out", new RnnOutputLayer.Builder().nIn(3+4).nOut(2)
                                        .activation(Activation.SOFTMAX)
                                        .lossFunction(LossFunctions.LossFunction.MCXENT).build(),
                                "lstm1", "duplicate")
                        .setInputTypes(InputType.recurrent(3,timeSeriesLength,RNNFormat.NCW),InputType.recurrent(2,timeSeriesLength,RNNFormat.NCW))
                        .build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        Random r = new Random(12345);
        INDArray input1 = Nd4j.rand(new int[] {batchSize, 3, 4});
        INDArray input2 = Nd4j.rand(new int[] {batchSize, 2, 4});
        INDArray labels = TestUtils.randomOneHotTimeSeries(batchSize, outSize, timeSeriesLength);

        if (PRINT_RESULTS) {
            System.out.println("testLSTMWithDuplicateToTimeSeries()");
//            for (int j = 0; j < graph.getNumLayers(); j++)
//                System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(graph).inputs(new INDArray[]{input1, input2})
                .labels(new INDArray[]{labels}));

        String msg = "testLSTMWithDuplicateToTimeSeries()";
        assertTrue(msg, gradOK);
        TestUtils.testModelSerialization(graph);
    }

    @Test
    public void testLSTMWithReverseTimeSeriesVertex() {
        int timeSeriesLength = 4;
        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf =
                new NeuralNetConfiguration.Builder().seed(12345)
                        .dataType(DataType.DOUBLE)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .dist(new NormalDistribution(0, 1))
                        .updater(new NoOp()).graphBuilder()
                        .addInputs("input").setOutputs("out")
                        .addLayer("lstm_a",
                                new LSTM.Builder().nIn(2).nOut(3)
                                        .activation(Activation.TANH).build(),
                                "input")
                        .addVertex("input_rev", new ReverseTimeSeriesVertex("input"), "input")
                        .addLayer("lstm_b",
                                new LSTM.Builder().nIn(2).nOut(3)
                                        .activation(Activation.TANH).build(),
                                "input_rev")
                        .addVertex("lstm_b_rev", new ReverseTimeSeriesVertex("input"), "lstm_b")
                        .addLayer("out", new RnnOutputLayer.Builder().nIn(3 + 3).nOut(2)
                                        .activation(Activation.SOFTMAX)
                                        .lossFunction(LossFunctions.LossFunction.MCXENT).build(),
                                "lstm_a", "lstm_b_rev")
                        .setInputTypes(InputType.recurrent(2,timeSeriesLength,RNNFormat.NCW))
                        .build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        Random r = new Random(12345);
        INDArray input  = Nd4j.rand(new int[] {2, 2, 4});
        INDArray labels = TestUtils.randomOneHotTimeSeries(2, 2, 4);

        if (PRINT_RESULTS) {
            System.out.println("testLSTMWithReverseTimeSeriesVertex()");
//            for (int j = 0; j < graph.getNumLayers(); j++)
//                System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(graph).inputs(new INDArray[]{input})
                .labels(new INDArray[]{labels}));

        String msg = "testLSTMWithDuplicateToTimeSeries()";
        assertTrue(msg, gradOK);

        //Second: test with input mask arrays.
        INDArray inMask = Nd4j.zeros(3, 5);
        inMask.putRow(0, Nd4j.create(new double[] {1, 1, 1, 0, 0}));
        inMask.putRow(1, Nd4j.create(new double[] {1, 1, 0, 1, 0}));
        inMask.putRow(2, Nd4j.create(new double[] {1, 1, 1, 1, 1}));
        graph.setLayerMaskArrays(new INDArray[] {inMask}, null);
        gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(graph).inputs(new INDArray[]{input})
                .labels(new INDArray[]{labels}));

        assertTrue(msg, gradOK);
        TestUtils.testModelSerialization(graph);
    }

    @Test
    public void testMultipleInputsLayer() {

        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                .dataType(DataType.DOUBLE)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .dist(new NormalDistribution(0, 1))
                .updater(new NoOp()).activation(Activation.TANH).graphBuilder().addInputs("i0", "i1", "i2")
                .addLayer("d0", new DenseLayer.Builder().nIn(2).nOut(2).build(), "i0")
                .addLayer("d1", new DenseLayer.Builder().nIn(2).nOut(2).build(), "i1")
                .addLayer("d2", new DenseLayer.Builder().nIn(2).nOut(2).build(), "i2")
                .addLayer("d3", new DenseLayer.Builder().nIn(6).nOut(2).build(), "d0", "d1", "d2")
                .addLayer("out", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(2)
                        .nOut(2).build(), "d3")
                .setOutputs("out").build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        int[] minibatchSizes = {1, 3};
        for (int mb : minibatchSizes) {
            INDArray[] inputs = new INDArray[3];
            for (int i = 0; i < 3; i++) {
                inputs[i] = Nd4j.rand(mb, 2);
            }
            INDArray out = Nd4j.rand(mb, 2);


            String msg = "testMultipleInputsLayer() - minibatchSize = " + mb;
            if (PRINT_RESULTS) {
                System.out.println(msg);
//                for (int j = 0; j < graph.getNumLayers(); j++)
//                    System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(graph).inputs(inputs)
                    .labels(new INDArray[]{out}));

            assertTrue(msg, gradOK);
            TestUtils.testModelSerialization(graph);
        }
    }

    @Test
    public void testMultipleOutputsLayer() {
        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                .dataType(DataType.DOUBLE)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .dist(new NormalDistribution(0, 1))
                .updater(new NoOp()).activation(Activation.TANH).graphBuilder().addInputs("i0")
                .addLayer("d0", new DenseLayer.Builder().nIn(2).nOut(2).build(), "i0")
                .addLayer("d1", new DenseLayer.Builder().nIn(2).nOut(2).build(), "d0")
                .addLayer("d2", new DenseLayer.Builder().nIn(2).nOut(2).build(), "d0")
                .addLayer("d3", new DenseLayer.Builder().nIn(2).nOut(2).build(), "d0")
                .addLayer("out", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(6)
                        .nOut(2).build(), "d1", "d2", "d3")
                .setOutputs("out").build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        int[] minibatchSizes = {1, 3};
        for (int mb : minibatchSizes) {
            INDArray input = Nd4j.rand(mb, 2);
            INDArray out = Nd4j.rand(mb, 2);


            String msg = "testMultipleOutputsLayer() - minibatchSize = " + mb;
            if (PRINT_RESULTS) {
                System.out.println(msg);
//                for (int j = 0; j < graph.getNumLayers(); j++)
//                    System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(graph).inputs(new INDArray[]{input})
                    .labels(new INDArray[]{out}));

            assertTrue(msg, gradOK);
            TestUtils.testModelSerialization(graph);
        }
    }

    @Test
    public void testMultipleOutputsMergeVertex() {
        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                .dataType(DataType.DOUBLE)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .dist(new NormalDistribution(0, 1))
                .updater(new NoOp()).activation(Activation.TANH).graphBuilder().addInputs("i0", "i1", "i2")
                .addLayer("d0", new DenseLayer.Builder().nIn(2).nOut(2).build(), "i0")
                .addLayer("d1", new DenseLayer.Builder().nIn(2).nOut(2).build(), "i1")
                .addLayer("d2", new DenseLayer.Builder().nIn(2).nOut(2).build(), "i2")
                .addVertex("m", new MergeVertex(), "d0", "d1", "d2")
                .addLayer("D0", new DenseLayer.Builder().nIn(6).nOut(2).build(), "m")
                .addLayer("D1", new DenseLayer.Builder().nIn(6).nOut(2).build(), "m")
                .addLayer("D2", new DenseLayer.Builder().nIn(6).nOut(2).build(), "m")
                .addLayer("out", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(6)
                        .nOut(2).build(), "D0", "D1", "D2")
                .setOutputs("out").build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        int[] minibatchSizes = {1, 3};
        for (int mb : minibatchSizes) {
            INDArray[] input = new INDArray[3];
            for (int i = 0; i < 3; i++) {
                input[i] = Nd4j.rand(mb, 2);
            }
            INDArray out = Nd4j.rand(mb, 2);


            String msg = "testMultipleOutputsMergeVertex() - minibatchSize = " + mb;
            if (PRINT_RESULTS) {
                System.out.println(msg);
//                for (int j = 0; j < graph.getNumLayers(); j++)
//                    System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(graph).inputs(input)
                    .labels(new INDArray[]{out}));

            assertTrue(msg, gradOK);
            TestUtils.testModelSerialization(graph);
        }
    }

    @Test
    public void testMultipleOutputsMergeCnn() {
        int inH = 7;
        int inW = 7;

        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                .dataType(DataType.DOUBLE)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .dist(new NormalDistribution(0, 1))
                .updater(new NoOp()).activation(Activation.TANH).graphBuilder().addInputs("input")
                .addLayer("l0", new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1).padding(0, 0)
                        .nIn(2).nOut(2).activation(Activation.TANH).build(), "input")
                .addLayer("l1", new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1).padding(0, 0)
                        .nIn(2).nOut(2).activation(Activation.TANH).build(), "l0")
                .addLayer("l2", new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1).padding(0, 0)
                        .nIn(2).nOut(2).activation(Activation.TANH).build(), "l0")
                .addVertex("m", new MergeVertex(), "l1", "l2")
                .addLayer("l3", new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1).padding(0, 0)
                        .nIn(4).nOut(2).activation(Activation.TANH).build(), "m")
                .addLayer("l4", new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1).padding(0, 0)
                        .nIn(4).nOut(2).activation(Activation.TANH).build(), "m")
                .addLayer("out", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY).nOut(2)
                        .build(), "l3", "l4")
                .setOutputs("out").setInputTypes(InputType.convolutional(inH, inW, 2))
                .build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        int[] minibatchSizes = {1, 3};
        for (int mb : minibatchSizes) {
            INDArray input = Nd4j.rand(new int[] {mb, 2, inH, inW}).muli(4); //Order: examples, channels, height, width
            INDArray out = Nd4j.rand(mb, 2);

            String msg = "testMultipleOutputsMergeVertex() - minibatchSize = " + mb;
            if (PRINT_RESULTS) {
                System.out.println(msg);
//                for (int j = 0; j < graph.getNumLayers(); j++)
//                    System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(graph).inputs(new INDArray[]{input})
                    .labels(new INDArray[]{out}));

            assertTrue(msg, gradOK);
            TestUtils.testModelSerialization(graph);
        }
    }


    @Test
    public void testBasicIrisTripletStackingL2Loss() {
        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf =
                new NeuralNetConfiguration.Builder().seed(12345)
                        .dataType(DataType.DOUBLE)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .dist(new NormalDistribution(0, 1))
                        .updater(new NoOp()).graphBuilder()
                        .addInputs("input1", "input2", "input3")
                        .addVertex("stack1", new StackVertex(), "input1", "input2", "input3")
                        .addLayer("l1", new DenseLayer.Builder().nIn(4).nOut(5)
                                .activation(Activation.TANH).build(), "stack1")
                        .addVertex("unstack0", new UnstackVertex(0, 3), "l1")
                        .addVertex("unstack1", new UnstackVertex(1, 3), "l1")
                        .addVertex("unstack2", new UnstackVertex(2, 3), "l1")
                        .addVertex("l2-1", new L2Vertex(), "unstack1", "unstack0") // x - x-
                        .addVertex("l2-2", new L2Vertex(), "unstack1", "unstack2") // x - x+
                        .addLayer("lossLayer",
                                new LossLayer.Builder()
                                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                                        .activation(Activation.SOFTMAX).build(),
                                "l2-1", "l2-2")
                        .setOutputs("lossLayer").build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        int numParams = (4 * 5 + 5);
        assertEquals(numParams, graph.numParams());

        Nd4j.getRandom().setSeed(12345);
        long nParams = graph.numParams();
        INDArray newParams = Nd4j.rand(new long[]{1, nParams});
        graph.setParams(newParams);

        INDArray pos = Nd4j.rand(150, 4);
        INDArray anc = Nd4j.rand(150, 4);
        INDArray neg = Nd4j.rand(150, 4);

        INDArray labels = Nd4j.zeros(150, 2);
        Random r = new Random(12345);
        for (int i = 0; i < 150; i++) {
            labels.putScalar(i, r.nextInt(2), 1.0);
        }


        Map<String, INDArray> out = graph.feedForward(new INDArray[] {pos, anc, neg}, true);

//        for (String s : out.keySet()) {
//            System.out.println(s + "\t" + Arrays.toString(out.get(s).shape()));
//        }

        if (PRINT_RESULTS) {
            System.out.println("testBasicIrisTripletStackingL2Loss()");
//            for (int j = 0; j < graph.getNumLayers(); j++)
//                System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(graph).inputs(new INDArray[]{pos, anc, neg})
                .labels(new INDArray[]{labels}));

        String msg = "testBasicIrisTripletStackingL2Loss()";
        assertTrue(msg, gradOK);
        TestUtils.testModelSerialization(graph);
    }


    @Test
    public void testBasicCenterLoss() {
        Nd4j.getRandom().setSeed(12345);
        int numLabels = 2;

        boolean[] trainFirst = new boolean[] {false, true};

        for (boolean train : trainFirst) {
            for (double lambda : new double[] {0.0, 0.5, 2.0}) {

                ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                        .dataType(DataType.DOUBLE)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .dist(new GaussianDistribution(0, 1))
                        .updater(new NoOp()).graphBuilder().addInputs("input1")
                        .addLayer("l1", new DenseLayer.Builder().nIn(4).nOut(5).activation(Activation.TANH)
                                .build(), "input1")
                        .addLayer("cl", new CenterLossOutputLayer.Builder()
                                .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(5).nOut(numLabels)
                                .alpha(1.0).lambda(lambda).gradientCheck(true)
                                .activation(Activation.SOFTMAX).build(), "l1")
                        .setOutputs("cl").build();

                ComputationGraph graph = new ComputationGraph(conf);
                graph.init();

                INDArray example = Nd4j.rand(150, 4);

                INDArray labels = Nd4j.zeros(150, numLabels);
                Random r = new Random(12345);
                for (int i = 0; i < 150; i++) {
                    labels.putScalar(i, r.nextInt(numLabels), 1.0);
                }

                if (train) {
                    for (int i = 0; i < 10; i++) {
                        INDArray f = Nd4j.rand(10, 4);
                        INDArray l = Nd4j.zeros(10, numLabels);
                        for (int j = 0; j < 10; j++) {
                            l.putScalar(j, r.nextInt(numLabels), 1.0);
                        }
                        graph.fit(new INDArray[] {f}, new INDArray[] {l});
                    }
                }

                String msg = "testBasicCenterLoss() - lambda = " + lambda + ", trainFirst = " + train;
                if (PRINT_RESULTS) {
                    System.out.println(msg);
//                    for (int j = 0; j < graph.getNumLayers(); j++)
//                        System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
                }

                boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(graph).inputs(new INDArray[]{example})
                        .labels(new INDArray[]{labels}));

                assertTrue(msg, gradOK);
                TestUtils.testModelSerialization(graph);
            }
        }
    }

    @Test
    public void testCnnPoolCenterLoss() {
        Nd4j.getRandom().setSeed(12345);
        int numLabels = 2;

        boolean[] trainFirst = new boolean[] {false, true};

        int inputH = 5;
        int inputW = 4;
        int inputDepth = 3;

        for (boolean train : trainFirst) {
            for (double lambda : new double[] {0.0, 0.5, 2.0}) {

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .dataType(DataType.DOUBLE)
                        .updater(new NoOp())
                        .dist(new NormalDistribution(0, 1.0)).seed(12345L).list()
                        .layer(0, new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1).nOut(3).build())
                        .layer(1, new GlobalPoolingLayer.Builder().poolingType(PoolingType.AVG).build())
                        .layer(2, new CenterLossOutputLayer.Builder()
                                .lossFunction(LossFunctions.LossFunction.MCXENT).nOut(numLabels)
                                .alpha(1.0).lambda(lambda).gradientCheck(true)
                                .activation(Activation.SOFTMAX).build())

                        .setInputType(InputType.convolutional(inputH, inputW, inputDepth)).build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                INDArray example = Nd4j.rand(new int[] {150, inputDepth, inputH, inputW});

                INDArray labels = Nd4j.zeros(150, numLabels);
                Random r = new Random(12345);
                for (int i = 0; i < 150; i++) {
                    labels.putScalar(i, r.nextInt(numLabels), 1.0);
                }

                if (train) {
                    for (int i = 0; i < 10; i++) {
                        INDArray f = Nd4j.rand(new int[] {10, inputDepth, inputH, inputW});
                        INDArray l = Nd4j.zeros(10, numLabels);
                        for (int j = 0; j < 10; j++) {
                            l.putScalar(j, r.nextInt(numLabels), 1.0);
                        }
                        net.fit(f, l);
                    }
                }

                String msg = "testBasicCenterLoss() - trainFirst = " + train;
                if (PRINT_RESULTS) {
                    System.out.println(msg);
//                    for (int j = 0; j < net.getnLayers(); j++)
//                        System.out.println("Layer " + j + " # params: " + net.getLayer(j).numParams());
                }

                boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, example, labels);

                assertTrue(msg, gradOK);
                TestUtils.testModelSerialization(net);
            }
        }
    }

    @Test
    public void testBasicL2() {
        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                .dataType(DataType.DOUBLE)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .dist(new NormalDistribution(0, 1))
                .activation(Activation.TANH).updater(new NoOp()).graphBuilder()
                .addInputs("in1", "in2").addLayer("d0", new DenseLayer.Builder().nIn(2).nOut(2).build(), "in1")
                .addLayer("d1", new DenseLayer.Builder().nIn(2).nOut(2).build(), "in2")
                .addVertex("l2", new L2Vertex(), "d0", "d1")
                .addLayer("out", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.L2).nIn(1)
                        .nOut(1).activation(Activation.IDENTITY).build(), "l2")
                .setOutputs("out").build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();


        Nd4j.getRandom().setSeed(12345);
        long nParams = graph.numParams();
        INDArray newParams = Nd4j.rand(new long[]{1, nParams});
        graph.setParams(newParams);

        int[] mbSizes = new int[] {1, 3, 10};
        for (int minibatch : mbSizes) {

            INDArray in1 = Nd4j.rand(DataType.DOUBLE, minibatch, 2);
            INDArray in2 = Nd4j.rand(DataType.DOUBLE, minibatch, 2);

            INDArray labels = Nd4j.rand(DataType.DOUBLE, minibatch, 1);

            String testName = "testBasicL2() - minibatch = " + minibatch;

            if (PRINT_RESULTS) {
                System.out.println(testName);
//                for (int j = 0; j < graph.getNumLayers(); j++)
//                    System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(graph).inputs(new INDArray[]{in1, in2})
                    .labels(new INDArray[]{labels}));

            assertTrue(testName, gradOK);
            TestUtils.testModelSerialization(graph);
        }
    }


    @Test
    public void testBasicStackUnstack() {

        int layerSizes = 2;

        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                .dataType(DataType.DOUBLE)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .dist(new NormalDistribution(0, 1))
                .activation(Activation.TANH).updater(new NoOp()).graphBuilder()
                .addInputs("in1", "in2")
                .addLayer("d0", new DenseLayer.Builder().nIn(layerSizes).nOut(layerSizes).build(), "in1")
                .addLayer("d1", new DenseLayer.Builder().nIn(layerSizes).nOut(layerSizes).build(), "in2")
                .addVertex("stack", new StackVertex(), "d0", "d1")
                .addLayer("d2", new DenseLayer.Builder().nIn(layerSizes).nOut(layerSizes).build(), "stack")
                .addVertex("u1", new UnstackVertex(0, 2), "d2").addVertex("u2", new UnstackVertex(1, 2), "d2")
                .addLayer("out1", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.L2)
                        .nIn(layerSizes).nOut(layerSizes).activation(Activation.IDENTITY).build(), "u1")
                .addLayer("out2", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.L2)
                        .nIn(layerSizes).nOut(2).activation(Activation.IDENTITY).build(), "u2")
                .setOutputs("out1", "out2").build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();


        Nd4j.getRandom().setSeed(12345);
        long nParams = graph.numParams();
        INDArray newParams = Nd4j.rand(new long[]{1, nParams});
        graph.setParams(newParams);

        int[] mbSizes = new int[] {1, 3, 10};
        for (int minibatch : mbSizes) {

            INDArray in1 = Nd4j.rand(minibatch, layerSizes);
            INDArray in2 = Nd4j.rand(minibatch, layerSizes);

            INDArray labels1 = Nd4j.rand(minibatch, 2);
            INDArray labels2 = Nd4j.rand(minibatch, 2);

            String testName = "testBasicStackUnstack() - minibatch = " + minibatch;

            if (PRINT_RESULTS) {
                System.out.println(testName);
//                for (int j = 0; j < graph.getNumLayers(); j++)
//                    System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(graph).inputs(new INDArray[]{in1, in2})
                    .labels(new INDArray[]{labels1, labels2}));

            assertTrue(testName, gradOK);
            TestUtils.testModelSerialization(graph);
        }
    }

    @Test
    public void testBasicStackUnstackDebug() {
        Nd4j.getRandom().setSeed(12345);

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                .dataType(DataType.DOUBLE)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .dist(new NormalDistribution(0, 1))
                .activation(Activation.TANH).updater(new NoOp()).graphBuilder()
                .addInputs("in1", "in2").addLayer("d0", new DenseLayer.Builder().nIn(2).nOut(2).build(), "in1")
                .addLayer("d1", new DenseLayer.Builder().nIn(2).nOut(2).build(), "in2")
                .addVertex("stack", new StackVertex(), "d0", "d1")
                .addVertex("u0", new UnstackVertex(0, 2), "stack")
                .addVertex("u1", new UnstackVertex(1, 2), "stack")
                .addLayer("out1",
                        new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.L2).nIn(2)
                                .nOut(2).activation(Activation.IDENTITY).build(),
                        "u0")
                .addLayer("out2",
                        new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.L2).nIn(2)
                                .nOut(2).activation(Activation.IDENTITY).build(),
                        "u1")
                .setOutputs("out1", "out2").build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();


        Nd4j.getRandom().setSeed(12345);
        long nParams = graph.numParams();
        INDArray newParams = Nd4j.rand(new long[]{1, nParams});
        graph.setParams(newParams);

        int[] mbSizes = new int[] {1, 3, 10};
        for (int minibatch : mbSizes) {

            INDArray in1 = Nd4j.rand(minibatch, 2);
            INDArray in2 = Nd4j.rand(minibatch, 2);

            INDArray labels1 = Nd4j.rand(minibatch, 2);
            INDArray labels2 = Nd4j.rand(minibatch, 2);

            String testName = "testBasicStackUnstack() - minibatch = " + minibatch;

            if (PRINT_RESULTS) {
                System.out.println(testName);
//                for (int j = 0; j < graph.getNumLayers(); j++)
//                    System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(graph).inputs(new INDArray[]{in1, in2})
                    .labels(new INDArray[]{labels1, labels2}));

            assertTrue(testName, gradOK);
            TestUtils.testModelSerialization(graph);
        }
    }

    @Test
    public void testBasicStackUnstackVariableLengthTS() {

        int layerSizes = 2;

        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                .dataType(DataType.DOUBLE)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .dist(new NormalDistribution(0, 1))
                .activation(Activation.TANH).updater(new NoOp()).graphBuilder()
                .addInputs("in1", "in2")
                .addLayer("d0", new SimpleRnn.Builder().nIn(layerSizes).nOut(layerSizes).build(), "in1")
                .addLayer("d1", new SimpleRnn.Builder().nIn(layerSizes).nOut(layerSizes).build(), "in2")
                .addVertex("stack", new StackVertex(), "d0", "d1")
                .addLayer("d2", new SimpleRnn.Builder().nIn(layerSizes).nOut(layerSizes).build(), "stack")
                .addVertex("u1", new UnstackVertex(0, 2), "d2").addVertex("u2", new UnstackVertex(1, 2), "d2")
                .addLayer("p1", new GlobalPoolingLayer.Builder(PoolingType.AVG).build(), "u1")
                .addLayer("p2", new GlobalPoolingLayer.Builder(PoolingType.AVG).build(), "u2")
                .addLayer("out1", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.L2)
                        .nIn(layerSizes).nOut(layerSizes).activation(Activation.IDENTITY).build(), "p1")
                .addLayer("out2", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.L2)
                        .nIn(layerSizes).nOut(2).activation(Activation.IDENTITY).build(), "p2")
                .setOutputs("out1", "out2").build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();


        Nd4j.getRandom().setSeed(12345);
        long nParams = graph.numParams();
        INDArray newParams = Nd4j.rand(new long[]{1, nParams});
        graph.setParams(newParams);

        int[] mbSizes = new int[] {1, 2, 3};
        for (int minibatch : mbSizes) {

            INDArray in1 = Nd4j.rand(new int[] {minibatch, layerSizes, 4});
            INDArray in2 = Nd4j.rand(new int[] {minibatch, layerSizes, 5});
            INDArray inMask1 = Nd4j.zeros(minibatch, 4);
            inMask1.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 3)).assign(1);
            INDArray inMask2 = Nd4j.zeros(minibatch, 5);
            inMask2.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 4)).assign(1);

            INDArray labels1 = Nd4j.rand(new int[] {minibatch, 2});
            INDArray labels2 = Nd4j.rand(new int[] {minibatch, 2});

            String testName = "testBasicStackUnstackVariableLengthTS() - minibatch = " + minibatch;

            if (PRINT_RESULTS) {
                System.out.println(testName);
//                for (int j = 0; j < graph.getNumLayers(); j++)
//                    System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
            }

            graph.setLayerMaskArrays(new INDArray[] {inMask1, inMask2}, null);

            boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(graph).inputs(new INDArray[]{in1, in2})
                    .labels(new INDArray[]{labels1, labels2}).inputMask(new INDArray[]{inMask1, inMask2}));

            assertTrue(testName, gradOK);
            TestUtils.testModelSerialization(graph);
        }
    }

    @Test
    public void testBasicTwoOutputs() {
        Nd4j.getRandom().setSeed(12345);

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                .dataType(DataType.DOUBLE)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .dist(new NormalDistribution(0, 1))
                .activation(Activation.TANH).updater(new NoOp()).graphBuilder()
                .addInputs("in1", "in2").addLayer("d0", new DenseLayer.Builder().nIn(2).nOut(2).build(), "in1")
                .addLayer("d1", new DenseLayer.Builder().nIn(2).nOut(2).build(), "in2")
                .addLayer("out1",
                        new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.L2).nIn(2)
                                .nOut(2).activation(Activation.IDENTITY).build(),
                        "d0")
                .addLayer("out2",
                        new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.L2).nIn(2)
                                .nOut(2).activation(Activation.IDENTITY).build(),
                        "d1")
                .setOutputs("out1", "out2").build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        System.out.println("Num layers: " + graph.getNumLayers());
        System.out.println("Num params: " + graph.numParams());


        Nd4j.getRandom().setSeed(12345);
        long nParams = graph.numParams();
        INDArray newParams = Nd4j.rand(new long[]{1, nParams});
        graph.setParams(newParams);

        int[] mbSizes = new int[] {1, 3, 10};
        for (int minibatch : mbSizes) {

            INDArray in1 = Nd4j.rand(minibatch, 2);
            INDArray in2 = Nd4j.rand(minibatch, 2);
            INDArray labels1 = Nd4j.rand(minibatch, 2);
            INDArray labels2 = Nd4j.rand(minibatch, 2);

            String testName = "testBasicStackUnstack() - minibatch = " + minibatch;

            if (PRINT_RESULTS) {
                System.out.println(testName);
//                for (int j = 0; j < graph.getNumLayers(); j++)
//                    System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(graph).inputs(new INDArray[]{in1, in2})
                    .labels(new INDArray[]{labels1, labels2}));
            assertTrue(testName, gradOK);
            TestUtils.testModelSerialization(graph);
        }
    }




    @Test
    public void testL2NormalizeVertex2d() {
        Nd4j.getRandom().setSeed(12345);
        int[][] definitions = {null,new int[]{1}};
        for(int[] definition : definitions) {
            log.info("Testing definition {}",definition);
            ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                    .dataType(DataType.DOUBLE)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .activation(Activation.TANH).updater(new NoOp()).graphBuilder()
                    .addInputs("in1").addLayer("d1", new DenseLayer.Builder().nIn(2).nOut(3).build(), "in1")
                    .addVertex("norm", new L2NormalizeVertex(definition,L2NormalizeVertex.DEFAULT_EPS), "d1")
                    .addLayer("out1",
                            new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.L2).nIn(3)
                                    .nOut(2).activation(Activation.IDENTITY).build(),
                            "norm")
                    .setOutputs("out1").build();

            ComputationGraph graph = new ComputationGraph(conf);
            graph.init();

            int[] mbSizes = new int[] {1, 3, 10};
            for (int minibatch : mbSizes) {

                INDArray in1 = Nd4j.rand(minibatch, 2);

                INDArray labels1 = Nd4j.rand(minibatch, 2);

                String testName = "testL2NormalizeVertex2d() - minibatch = " + minibatch;

                if (PRINT_RESULTS) {
                    System.out.println(testName);
//                for (int j = 0; j < graph.getNumLayers(); j++)
//                    System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
                }

                boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(graph).inputs(new INDArray[]{in1})
                        .labels(new INDArray[]{labels1}));

                assertTrue(testName, gradOK);
                TestUtils.testModelSerialization(graph);
            }
        }

    }

    @Test
    public void testL2NormalizeVertex4d() {
        Nd4j.getRandom().setSeed(12345);

        int h = 4;
        int w = 4;
        int dIn = 2;

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                .dataType(DataType.DOUBLE)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .dist(new NormalDistribution(0, 1))
                .activation(Activation.TANH).updater(new NoOp()).graphBuilder()
                .addInputs("in1")
                .addLayer("d1", new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1).nOut(2).build(),
                        "in1")
                .addVertex("norm", new L2NormalizeVertex(), "d1")
                .addLayer("out1",
                        new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.L2).nOut(2)
                                .activation(Activation.IDENTITY).build(),
                        "norm")
                .setOutputs("out1").setInputTypes(InputType.convolutional(h, w, dIn)).build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        int[] mbSizes = new int[] {1, 3, 10};
        for (int minibatch : mbSizes) {

            INDArray in1 = Nd4j.rand(new int[] {minibatch, dIn, h, w});

            INDArray labels1 = Nd4j.rand(minibatch, 2);

            String testName = "testL2NormalizeVertex4d() - minibatch = " + minibatch;

            if (PRINT_RESULTS) {
                System.out.println(testName);
//                for (int j = 0; j < graph.getNumLayers(); j++)
//                    System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(graph).inputs(new INDArray[]{in1})
                    .labels(new INDArray[]{labels1}));

            assertTrue(testName, gradOK);
            TestUtils.testModelSerialization(graph);
        }
    }

    @Test
    public void testGraphEmbeddingLayerSimple() {
        Random r = new Random(12345);
        int nExamples = 5;
        INDArray input = Nd4j.zeros(nExamples, 1);
        INDArray labels = Nd4j.zeros(nExamples, 3);
        for (int i = 0; i < nExamples; i++) {
            input.putScalar(i, r.nextInt(4));
            labels.putScalar(new int[] {i, r.nextInt(3)}, 1.0);
        }

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().l2(0.2).l1(0.1)
                .dataType(DataType.DOUBLE)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).seed(12345L)
                .updater(new NoOp()).graphBuilder().addInputs("in")
                .addLayer("0", new EmbeddingLayer.Builder().nIn(4).nOut(3).weightInit(WeightInit.XAVIER)
                        .activation(Activation.TANH).build(), "in")
                .addLayer("1", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(3).nOut(3)
                        .activation(Activation.SOFTMAX).build(), "0")
                .setOutputs("1").build();

        ComputationGraph cg = new ComputationGraph(conf);
        cg.init();

        if (PRINT_RESULTS) {
            System.out.println("testGraphEmbeddingLayerSimple");
//            for (int j = 0; j < cg.getNumLayers(); j++)
//                System.out.println("Layer " + j + " # params: " + cg.getLayer(j).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(cg).inputs(new INDArray[]{input})
                .labels(new INDArray[]{labels}));

        String msg = "testGraphEmbeddingLayerSimple";
        assertTrue(msg, gradOK);
        TestUtils.testModelSerialization(cg);
    }
}
