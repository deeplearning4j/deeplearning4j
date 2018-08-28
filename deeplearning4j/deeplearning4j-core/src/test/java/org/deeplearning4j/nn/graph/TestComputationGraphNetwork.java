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

package org.deeplearning4j.nn.graph;

import lombok.extern.slf4j.Slf4j;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.SingletonMultiDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.exception.DL4JException;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.graph.*;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.conf.preprocessor.*;
import org.deeplearning4j.nn.conf.weightnoise.DropConnect;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.util.GraphIndices;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.multilayer.MultiLayerTest;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.*;

import static org.junit.Assert.*;

@Slf4j
public class TestComputationGraphNetwork extends BaseDL4JTest {

    private static ComputationGraphConfiguration getIrisGraphConfiguration() {
        return new NeuralNetConfiguration.Builder().seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).graphBuilder()
                .addInputs("input")
                .addLayer("firstLayer", new DenseLayer.Builder().nIn(4).nOut(5).build(), "input")
                .addLayer("outputLayer", new OutputLayer.Builder().nIn(5).nOut(3).build(), "firstLayer")
                .setOutputs("outputLayer").build();
    }

    private static MultiLayerConfiguration getIrisMLNConfiguration() {
        return new NeuralNetConfiguration.Builder().seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).list()
                .layer(0, new DenseLayer.Builder().nIn(4).nOut(5).build())
                .layer(1, new OutputLayer.Builder().nIn(5).nOut(3).build()).build();
    }

    private static int getNumParams() {
        //Number of parameters for both iris models
        return (4 * 5 + 5) + (5 * 3 + 3);
    }

    private static OpExecutioner.ProfilingMode origMode;

    @BeforeClass
    public static void beforeClass(){
        origMode = Nd4j.getExecutioner().getProfilingMode();
    }

    @Before
    public void before(){
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.SCOPE_PANIC);
    }

    @AfterClass
    public static void afterClass(){
        Nd4j.getExecutioner().setProfilingMode(origMode);
    }


    @Test
    public void testFeedForwardToLayer() {

        ComputationGraphConfiguration configuration = getIrisGraphConfiguration();

        ComputationGraph graph = new ComputationGraph(configuration);
        graph.init();

        MultiLayerConfiguration mlc = getIrisMLNConfiguration();
        MultiLayerNetwork net = new MultiLayerNetwork(mlc);
        net.init();


        DataSetIterator iris = new IrisDataSetIterator(150, 150);
        DataSet ds = iris.next();

        graph.setInput(0, ds.getFeatures());
        net.setParams(graph.params());
        Map<String, INDArray> activations = graph.feedForward(false);

        List<INDArray> feedForward = net.feedForward(ds.getFeatures());
        assertEquals(activations.size(),feedForward.size());
        assertEquals(activations.get("outputLayer"),feedForward.get(feedForward.size() - 1));

        Map<String,INDArray> graphForward = graph.feedForward(ds.getFeatures(),0,false);
        List<INDArray> networkForward =  net.feedForwardToLayer(0,ds.getFeatures(),false);
        assertEquals(graphForward.get("firstLayer"),networkForward.get(1));
    }

    @Test
    public void testConfigurationBasic() {

        ComputationGraphConfiguration configuration = getIrisGraphConfiguration();

        ComputationGraph graph = new ComputationGraph(configuration);
        graph.init();

        //Get topological sort order
        int[] order = graph.topologicalSortOrder();
        int[] expOrder = new int[]{0, 1, 2};
        assertArrayEquals(expOrder, order); //Only one valid order: 0 (input) -> 1 (firstlayer) -> 2 (outputlayer)

        INDArray params = graph.params();
        assertNotNull(params);

        int nParams = getNumParams();
        assertEquals(nParams, params.length());

        INDArray arr = Nd4j.linspace(0, nParams, nParams);
        assertEquals(nParams, arr.length());

        graph.setParams(arr);
        params = graph.params();
        assertEquals(arr, params);

        //Number of inputs and outputs:
        assertEquals(1, graph.getNumInputArrays());
        assertEquals(1, graph.getNumOutputArrays());
    }

    @Test
    public void testForwardBasicIris() {

        ComputationGraphConfiguration configuration = getIrisGraphConfiguration();
        ComputationGraph graph = new ComputationGraph(configuration);
        graph.init();

        MultiLayerConfiguration mlc = getIrisMLNConfiguration();
        MultiLayerNetwork net = new MultiLayerNetwork(mlc);
        net.init();

        DataSetIterator iris = new IrisDataSetIterator(150, 150);
        DataSet ds = iris.next();

        graph.setInput(0, ds.getFeatures());
        Map<String, INDArray> activations = graph.feedForward(false);
        assertEquals(3, activations.size()); //2 layers + 1 input node
        assertTrue(activations.containsKey("input"));
        assertTrue(activations.containsKey("firstLayer"));
        assertTrue(activations.containsKey("outputLayer"));

        //Now: set parameters of both networks to be identical. Then feedforward, and check we get the same outputs
        Nd4j.getRandom().setSeed(12345);
        int nParams = getNumParams();
        INDArray params = Nd4j.rand(1, nParams);
        graph.setParams(params.dup());
        net.setParams(params.dup());

        List<INDArray> mlnAct = net.feedForward(ds.getFeatures(), false);
        activations = graph.feedForward(ds.getFeatures(), false);

        assertEquals(mlnAct.get(0), activations.get("input"));
        assertEquals(mlnAct.get(1), activations.get("firstLayer"));
        assertEquals(mlnAct.get(2), activations.get("outputLayer"));
    }

    @Test
    public void testBackwardIrisBasic() {
        ComputationGraphConfiguration configuration = getIrisGraphConfiguration();
        ComputationGraph graph = new ComputationGraph(configuration);
        graph.init();

        MultiLayerConfiguration mlc = getIrisMLNConfiguration();
        MultiLayerNetwork net = new MultiLayerNetwork(mlc);
        net.init();

        DataSetIterator iris = new IrisDataSetIterator(150, 150);
        DataSet ds = iris.next();

        //Now: set parameters of both networks to be identical. Then feedforward, and check we get the same outputs
        Nd4j.getRandom().setSeed(12345);
        int nParams = (4 * 5 + 5) + (5 * 3 + 3);
        INDArray params = Nd4j.rand(1, nParams);
        graph.setParams(params.dup());
        net.setParams(params.dup());

        INDArray input = ds.getFeatures();
        INDArray labels = ds.getLabels();
        graph.setInput(0, input.dup());
        graph.setLabel(0, labels.dup());

        net.setInput(input.dup());
        net.setLabels(labels.dup());

        //Compute gradients
        net.computeGradientAndScore();
        Pair<Gradient, Double> netGradScore = net.gradientAndScore();

        graph.computeGradientAndScore();
        Pair<Gradient, Double> graphGradScore = graph.gradientAndScore();

        assertEquals(netGradScore.getSecond(), graphGradScore.getSecond(), 1e-3);

        //Compare gradients
        Gradient netGrad = netGradScore.getFirst();
        Gradient graphGrad = graphGradScore.getFirst();

        assertNotNull(graphGrad);
        assertEquals(netGrad.gradientForVariable().size(), graphGrad.gradientForVariable().size());

        assertEquals(netGrad.getGradientFor("0_W"), graphGrad.getGradientFor("firstLayer_W"));
        assertEquals(netGrad.getGradientFor("0_b"), graphGrad.getGradientFor("firstLayer_b"));
        assertEquals(netGrad.getGradientFor("1_W"), graphGrad.getGradientFor("outputLayer_W"));
        assertEquals(netGrad.getGradientFor("1_b"), graphGrad.getGradientFor("outputLayer_b"));
    }

    @Test
    public void testIrisFit() {

        ComputationGraphConfiguration configuration = getIrisGraphConfiguration();
        ComputationGraph graph = new ComputationGraph(configuration);
        graph.init();

        MultiLayerConfiguration mlnConfig = getIrisMLNConfiguration();
        MultiLayerNetwork net = new MultiLayerNetwork(mlnConfig);
        net.init();

        Nd4j.getRandom().setSeed(12345);
        int nParams = getNumParams();
        INDArray params = Nd4j.rand(1, nParams);

        graph.setParams(params.dup());
        net.setParams(params.dup());


        DataSetIterator iris = new IrisDataSetIterator(75, 150);

        net.fit(iris);
        iris.reset();

        graph.fit(iris);

        //Check that parameters are equal for both models after fitting:
        INDArray paramsMLN = net.params();
        INDArray paramsGraph = graph.params();

        assertNotEquals(params, paramsGraph);
        assertEquals(paramsMLN, paramsGraph);
    }

    @Test
    public void testIrisFitMultiDataSetIterator() throws Exception {

        RecordReader rr = new CSVRecordReader(0, ',');
        rr.initialize(new FileSplit(new ClassPathResource("iris.txt").getTempFileFromArchive()));

        MultiDataSetIterator iter = new RecordReaderMultiDataSetIterator.Builder(10).addReader("iris", rr)
                .addInput("iris", 0, 3).addOutputOneHot("iris", 4, 3).build();

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .updater(new Sgd(0.1))
                .graphBuilder().addInputs("in")
                .addLayer("dense", new DenseLayer.Builder().nIn(4).nOut(2).build(), "in").addLayer("out",
                        new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(2).nOut(3)
                                .build(),
                        "dense")
                .setOutputs("out").pretrain(false).backprop(true).build();

        ComputationGraph cg = new ComputationGraph(config);
        cg.init();

        cg.fit(iter);


        rr.reset();
        iter = new RecordReaderMultiDataSetIterator.Builder(10).addReader("iris", rr).addInput("iris", 0, 3)
                .addOutputOneHot("iris", 4, 3).build();
        while (iter.hasNext()) {
            cg.fit(iter.next());
        }
    }

    @Test
    public void testCloning() {
        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf = getIrisGraphConfiguration();
        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        ComputationGraph g2 = graph.clone();

        DataSetIterator iris = new IrisDataSetIterator(150, 150);
        INDArray in = iris.next().getFeatures();
        Map<String, INDArray> activations = graph.feedForward(in, false);
        Map<String, INDArray> activations2 = g2.feedForward(in, false);
        assertEquals(activations, activations2);
    }

    @Test
    public void testScoringDataSet() {
        ComputationGraphConfiguration configuration = getIrisGraphConfiguration();
        ComputationGraph graph = new ComputationGraph(configuration);
        graph.init();

        MultiLayerConfiguration mlc = getIrisMLNConfiguration();
        MultiLayerNetwork net = new MultiLayerNetwork(mlc);
        net.init();

        DataSetIterator iris = new IrisDataSetIterator(150, 150);
        DataSet ds = iris.next();

        //Now: set parameters of both networks to be identical. Then feedforward, and check we get the same score
        Nd4j.getRandom().setSeed(12345);
        int nParams = getNumParams();
        INDArray params = Nd4j.rand(1, nParams);
        graph.setParams(params.dup());
        net.setParams(params.dup());

        double scoreMLN = net.score(ds, false);
        double scoreCG = graph.score(ds, false);

        assertEquals(scoreMLN, scoreCG, 1e-4);
    }

    @Test
    public void testPreprocessorAddition() {
        //Also check that nIns are set automatically
        //First: check FF -> RNN
        ComputationGraphConfiguration conf1 = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("in")
                .setInputTypes(InputType.feedForward(5))
                .addLayer("rnn", new GravesLSTM.Builder().nOut(5).build(), "in")
                .addLayer("out", new RnnOutputLayer.Builder().nOut(5).build(), "rnn").setOutputs("out").build();

        assertEquals(5, ((FeedForwardLayer) ((LayerVertex) conf1.getVertices().get("rnn")).getLayerConf().getLayer())
                .getNIn());
        assertEquals(5, ((FeedForwardLayer) ((LayerVertex) conf1.getVertices().get("out")).getLayerConf().getLayer())
                .getNIn());

        LayerVertex lv1 = (LayerVertex) conf1.getVertices().get("rnn");
        assertTrue(lv1.getPreProcessor() instanceof FeedForwardToRnnPreProcessor);
        LayerVertex lv2 = (LayerVertex) conf1.getVertices().get("out");
        assertNull(lv2.getPreProcessor());

        //Check RNN -> FF -> RNN
        ComputationGraphConfiguration conf2 = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("in")
                .setInputTypes(InputType.recurrent(5))
                .addLayer("ff", new DenseLayer.Builder().nOut(5).build(), "in")
                .addLayer("out", new RnnOutputLayer.Builder().nOut(5).build(), "ff").setOutputs("out").build();

        assertEquals(5, ((FeedForwardLayer) ((LayerVertex) conf2.getVertices().get("ff")).getLayerConf().getLayer())
                .getNIn());
        assertEquals(5, ((FeedForwardLayer) ((LayerVertex) conf2.getVertices().get("out")).getLayerConf().getLayer())
                .getNIn());

        lv1 = (LayerVertex) conf2.getVertices().get("ff");
        assertTrue(lv1.getPreProcessor() instanceof RnnToFeedForwardPreProcessor);
        lv2 = (LayerVertex) conf2.getVertices().get("out");
        assertTrue(lv2.getPreProcessor() instanceof FeedForwardToRnnPreProcessor);

        //CNN -> Dense
        ComputationGraphConfiguration conf3 = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("in")
                .setInputTypes(InputType.convolutional(28, 28, 1))
                .addLayer("cnn", new ConvolutionLayer.Builder().kernelSize(2, 2).padding(0, 0).stride(2, 2)
                        .nOut(3).build(), "in") //(28-2+0)/2+1 = 14
                .addLayer("pool",
                        new SubsamplingLayer.Builder().kernelSize(2, 2).padding(0, 0).stride(2, 2)
                                .build(),
                        "cnn") //(14-2+0)/2+1=7
                .addLayer("dense", new DenseLayer.Builder().nOut(10).build(), "pool")
                .addLayer("out", new OutputLayer.Builder().nIn(10).nOut(5).build(), "dense").setOutputs("out")
                .build();
        //Check preprocessors:
        lv1 = (LayerVertex) conf3.getVertices().get("cnn");
        assertNull(lv1.getPreProcessor()); //Shouldn't be adding preprocessor here

        lv2 = (LayerVertex) conf3.getVertices().get("pool");
        assertNull(lv2.getPreProcessor());
        LayerVertex lv3 = (LayerVertex) conf3.getVertices().get("dense");
        assertTrue(lv3.getPreProcessor() instanceof CnnToFeedForwardPreProcessor);
        CnnToFeedForwardPreProcessor proc = (CnnToFeedForwardPreProcessor) lv3.getPreProcessor();
        assertEquals(3, proc.getNumChannels());
        assertEquals(7, proc.getInputHeight());
        assertEquals(7, proc.getInputWidth());
        LayerVertex lv4 = (LayerVertex) conf3.getVertices().get("out");
        assertNull(lv4.getPreProcessor());
        //Check nIns:
        assertEquals(7 * 7 * 3, ((FeedForwardLayer) lv3.getLayerConf().getLayer()).getNIn());

        //CNN->Dense, RNN->Dense, Dense->RNN
        ComputationGraphConfiguration conf4 =
                new NeuralNetConfiguration.Builder().graphBuilder().addInputs("inCNN", "inRNN")
                        .setInputTypes(InputType.convolutional(28, 28, 1), InputType.recurrent(5))
                        .addLayer("cnn", new ConvolutionLayer.Builder().kernelSize(2, 2).padding(0, 0)
                                .stride(2, 2).nOut(3).build(), "inCNN") //(28-2+0)/2+1 = 14
                        .addLayer("pool",
                                new SubsamplingLayer.Builder().kernelSize(2, 2).padding(0, 0)
                                        .stride(2, 2).build(),
                                "cnn") //(14-2+0)/2+1=7
                        .addLayer("dense", new DenseLayer.Builder().nOut(10).build(), "pool")
                        .addLayer("dense2", new DenseLayer.Builder().nOut(10).build(), "inRNN")
                        .addVertex("merge", new MergeVertex(), "dense", "dense2")
                        .addLayer("out", new RnnOutputLayer.Builder().nOut(5).build(), "merge")
                        .setOutputs("out").build();

        //Check preprocessors:
        lv1 = (LayerVertex) conf4.getVertices().get("cnn");
        assertNull(lv1.getPreProcessor()); //Expect no preprocessor: cnn data -> cnn layer

        lv2 = (LayerVertex) conf4.getVertices().get("pool");
        assertNull(lv2.getPreProcessor());
        lv3 = (LayerVertex) conf4.getVertices().get("dense");
        assertTrue(lv3.getPreProcessor() instanceof CnnToFeedForwardPreProcessor);
        proc = (CnnToFeedForwardPreProcessor) lv3.getPreProcessor();
        assertEquals(3, proc.getNumChannels());
        assertEquals(7, proc.getInputHeight());
        assertEquals(7, proc.getInputWidth());
        lv4 = (LayerVertex) conf4.getVertices().get("dense2");
        assertTrue(lv4.getPreProcessor() instanceof RnnToFeedForwardPreProcessor);
        LayerVertex lv5 = (LayerVertex) conf4.getVertices().get("out");
        assertTrue(lv5.getPreProcessor() instanceof FeedForwardToRnnPreProcessor);
        //Check nIns:
        assertEquals(7 * 7 * 3, ((FeedForwardLayer) lv3.getLayerConf().getLayer()).getNIn());
        assertEquals(5, ((FeedForwardLayer) lv4.getLayerConf().getLayer()).getNIn());
        assertEquals(20, ((FeedForwardLayer) lv5.getLayerConf().getLayer()).getNIn()); //10+10 out of the merge vertex -> 20 in to output layer vertex


        //Input to 2 CNN layers:
        ComputationGraphConfiguration conf5 =
                new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .graphBuilder().addInputs("input")
                        .setInputTypes(InputType.convolutional(28, 28, 1))
                        .addLayer("cnn_1",
                                new ConvolutionLayer.Builder(2, 2).stride(2, 2).nIn(1).nOut(3)
                                        .build(),
                                "input")
                        .addLayer("cnn_2",
                                new ConvolutionLayer.Builder(4, 4).stride(2, 2).padding(1, 1)
                                        .nIn(1).nOut(3).build(),
                                "input")
                        .addLayer("max_1",
                                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                        .kernelSize(2, 2).build(),
                                "cnn_1", "cnn_2")
                        .addLayer("output", new OutputLayer.Builder().nOut(10).build(), "max_1") //.nIn(7 * 7 * 6)
                        .setOutputs("output").pretrain(false).backprop(true).build();
        lv1 = (LayerVertex) conf5.getVertices().get("cnn_1");
        assertNull(lv1.getPreProcessor()); //Expect no preprocessor: cnn data -> cnn layer

        lv2 = (LayerVertex) conf5.getVertices().get("cnn_2");
        assertNull(lv2.getPreProcessor()); //Expect no preprocessor: cnn data -> cnn layer

        assertNull(((LayerVertex) conf5.getVertices().get("max_1")).getPreProcessor());

        lv3 = (LayerVertex) conf5.getVertices().get("output");
        assertTrue(lv3.getPreProcessor() instanceof CnnToFeedForwardPreProcessor);
        CnnToFeedForwardPreProcessor cnnff = (CnnToFeedForwardPreProcessor) lv3.getPreProcessor();
        assertEquals(6, cnnff.getNumChannels());
        assertEquals(7, cnnff.getInputHeight());
        assertEquals(7, cnnff.getInputWidth());

        ComputationGraph graph = new ComputationGraph(conf1);
        graph.init();
        System.out.println(graph.summary());
        System.out.println(graph.summary(InputType.feedForward(5)));

        graph = new ComputationGraph(conf2);
        graph.init();
        System.out.println(graph.summary());
        System.out.println(graph.summary(InputType.recurrent(5)));

        graph = new ComputationGraph(conf3);
        graph.init();
        System.out.println(graph.summary());
        System.out.println(graph.summary(InputType.convolutional(28, 28, 1)));

        graph = new ComputationGraph(conf4);
        graph.init();
        System.out.println(graph.summary());
        System.out.println(graph.summary(InputType.convolutional(28, 28, 1), InputType.recurrent(5)));

        graph = new ComputationGraph(conf5);
        graph.init();
        System.out.println(graph.summary());
        System.out.println(graph.summary(InputType.convolutional(28, 28, 1)));
    }

    @Test
    public void testCompGraphUnderscores() {
        //Problem: underscores in names could be problematic for ComputationGraphUpdater, HistogramIterationListener

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).graphBuilder()
                .addInputs("input")
                .addLayer("first_layer", new DenseLayer.Builder().nIn(4).nOut(5).build(), "input")
                .addLayer("output_layer", new OutputLayer.Builder().nIn(5).nOut(3).build(), "first_layer")
                .setOutputs("output_layer").pretrain(false).backprop(true).build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        DataSetIterator iris = new IrisDataSetIterator(10, 150);
        while (iris.hasNext()) {
            net.fit(iris.next());
        }
    }

    @Test
    public void testPreTraining() {
        ComputationGraphConfiguration conf =
                new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Sgd(1e-6))
                        .l2(2e-4).graphBuilder().addInputs("in")
                        .addLayer("layer0",
                                new VariationalAutoencoder.Builder().nIn(4).nOut(3)
                                        .weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new UniformDistribution(0,
                                                1))
                                        .activation(Activation.TANH)
                                        .lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE)
                                        .build(),
                                "in")
                        .addLayer("layer1",
                                new VariationalAutoencoder.Builder().nIn(4).nOut(3)
                                        .weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new UniformDistribution(0,
                                                1))
                                        .activation(Activation.TANH)
                                        .lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE)
                                        .build(),
                                "in")
                        .addLayer("layer2",
                                new VariationalAutoencoder.Builder().nIn(3).nOut(3)
                                        .weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new UniformDistribution(0,
                                                1))
                                        .activation(Activation.TANH)
                                        .lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE)
                                        .build(),
                                "layer1")
                        .addLayer("out", new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).nIn(3 + 3).nOut(3)
                                        .weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new UniformDistribution(0, 1))
                                        .activation(Activation.SOFTMAX).build(),
                                "layer0", "layer2")
                        .setOutputs("out").pretrain(true).backprop(false).build();


        ComputationGraph net = new ComputationGraph(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        DataSetIterator iter = new IrisDataSetIterator(10, 150);
        net.pretrain(iter);
    }

    @Test
    public void testScoreExamples() {
        Nd4j.getRandom().setSeed(12345);
        int nIn = 5;
        int nOut = 6;
        ComputationGraphConfiguration conf =
                new NeuralNetConfiguration.Builder().seed(12345).l1(0.01).l2(0.01)
                        .updater(new Sgd(0.1))
                        .activation(Activation.TANH).weightInit(WeightInit.XAVIER)
                        .graphBuilder().addInputs("in")
                        .addLayer("0", new DenseLayer.Builder().nIn(nIn).nOut(20).build(), "in")
                        .addLayer("1", new DenseLayer.Builder().nIn(20).nOut(30).build(), "0")
                        .addLayer("2", new OutputLayer.Builder()
                                .lossFunction(LossFunctions.LossFunction.MSE).nIn(30).nOut(nOut)
                                .build(), "1")
                        .setOutputs("2").build();

        ComputationGraphConfiguration confNoReg =
                new NeuralNetConfiguration.Builder().seed(12345).updater(new Sgd(0.1)).activation(Activation.TANH)
                        .weightInit(WeightInit.XAVIER).graphBuilder().addInputs("in")
                        .addLayer("0", new DenseLayer.Builder().nIn(nIn).nOut(20).build(), "in")
                        .addLayer("1", new DenseLayer.Builder().nIn(20).nOut(30).build(), "0")
                        .addLayer("2", new OutputLayer.Builder()
                                .lossFunction(LossFunctions.LossFunction.MSE).nIn(30).nOut(nOut)
                                .build(), "1")
                        .setOutputs("2").build();


        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        ComputationGraph netNoReg = new ComputationGraph(confNoReg);
        netNoReg.init();
        netNoReg.setParams(net.params().dup());

        //Score single example, and compare to scoreExamples:
        INDArray input = Nd4j.rand(3, nIn);
        INDArray output = Nd4j.rand(3, nOut);
        DataSet ds = new DataSet(input, output);

        INDArray scoresWithRegularization = net.scoreExamples(ds, true);
        INDArray scoresNoRegularization = net.scoreExamples(ds, false);

        assertArrayEquals(new long[]{3, 1}, scoresWithRegularization.shape());
        assertArrayEquals(new long[]{3, 1}, scoresNoRegularization.shape());

        for (int i = 0; i < 3; i++) {
            DataSet singleEx = new DataSet(input.getRow(i), output.getRow(i));
            double score = net.score(singleEx);
            double scoreNoReg = netNoReg.score(singleEx);

            double scoreUsingScoreExamples = scoresWithRegularization.getDouble(i);
            double scoreUsingScoreExamplesNoReg = scoresNoRegularization.getDouble(i);
            assertEquals(score, scoreUsingScoreExamples, 1e-4);
            assertEquals(scoreNoReg, scoreUsingScoreExamplesNoReg, 1e-4);
            assertTrue(scoreUsingScoreExamples > scoreUsingScoreExamplesNoReg); //Regularization term increases score

            //            System.out.println(score + "\t" + scoreUsingScoreExamples + "\t|\t" + scoreNoReg + "\t" + scoreUsingScoreExamplesNoReg);
        }
    }


    @Test
    public void testExternalErrors() {
        //Simple test: same network, but in one case: one less layer (the OutputLayer), where the epsilons are passed in externally
        // instead. Should get identical results

        for(WorkspaceMode ws : WorkspaceMode.values()) {
            log.info("Workspace mode: " + ws);

            Nd4j.getRandom().setSeed(12345);
            INDArray inData = Nd4j.rand(3, 10);
            INDArray outData = Nd4j.rand(3, 10);

            Nd4j.getRandom().setSeed(12345);
            ComputationGraphConfiguration standard = new NeuralNetConfiguration.Builder().updater(new Sgd(0.1))
                    .trainingWorkspaceMode(ws).inferenceWorkspaceMode(ws)
                    .seed(12345).graphBuilder().addInputs("in")
                    .addLayer("l0", new DenseLayer.Builder().nIn(10).nOut(10).build(), "in")
                    .addLayer("out", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(10)
                            .nOut(10).build(), "l0")
                    .setOutputs("out").pretrain(false).backprop(true).build();
            ComputationGraph s = new ComputationGraph(standard);
            s.init();


            Nd4j.getRandom().setSeed(12345);
            ComputationGraphConfiguration external = new NeuralNetConfiguration.Builder().updater(new Sgd(0.1))
                    .trainingWorkspaceMode(ws).inferenceWorkspaceMode(ws)
                    .seed(12345).graphBuilder().addInputs("in")
                    .addLayer("l0", new DenseLayer.Builder().nIn(10).nOut(10).build(), "in").setOutputs("l0")
                    .pretrain(false).backprop(true).build();

            ComputationGraph e = new ComputationGraph(external);
            e.init();

            s.setInputs(inData);
            s.setLabels(outData);
            s.computeGradientAndScore();
            Gradient sGrad = s.gradient();

            s.feedForward(new INDArray[]{inData}, true, false); //FF without clearing inputs as we need them later

            org.deeplearning4j.nn.layers.OutputLayer ol = (org.deeplearning4j.nn.layers.OutputLayer) s.getLayer(1);
            ol.setLabels(outData);
            Pair<Gradient, INDArray> olPairStd = ol.backpropGradient(null, LayerWorkspaceMgr.noWorkspaces());

            INDArray olEpsilon = olPairStd.getSecond();

            e.feedForward(new INDArray[]{inData}, true, false); //FF without clearing inputs as we need them later
            Gradient extErrorGrad = e.backpropGradient(olEpsilon);

            int nParamsDense = 10 * 10 + 10;
            assertEquals(sGrad.gradient().get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nParamsDense)),
                    extErrorGrad.gradient());

            Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        }
    }

    @Test
    public void testExternalErrors2(){
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.SCOPE_PANIC);
        int nIn = 4;
        int nOut = 3;

        for(WorkspaceMode ws : WorkspaceMode.values()) {
            System.out.println("***** WORKSPACE: " + ws);

            ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                    .updater(new Adam(0.01))
                    .trainingWorkspaceMode(ws)
                    .inferenceWorkspaceMode(ws)
                    .graphBuilder()
                    .addInputs("features")
                    .addVertex("rnn2ffn", new PreprocessorVertex(new RnnToFeedForwardPreProcessor()), "features")
                    .addLayer("predict", new DenseLayer.Builder().nIn(nIn).nOut(nOut).activation(Activation.RELU).build(), "rnn2ffn")
                    .addVertex("ffn2rnn", new PreprocessorVertex(new FeedForwardToRnnPreProcessor()), "predict")
                    .addLayer("output", new ActivationLayer.Builder().activation(Activation.IDENTITY).build(), "ffn2rnn")
                    .setOutputs("output")
                    .build();

            ComputationGraph graph = new ComputationGraph(conf);
            graph.init();

            final int minibatch = 5;
            final int seqLen = 6;

            INDArray param = Nd4j.create(new double[]{0.54, 0.31, 0.98, -0.30, -0.66, -0.19, -0.29, -0.62, 0.13, -0.32, 0.01, -0.03, 0.00, 0.00, 0.00});
            graph.setParams(param);

            INDArray input = Nd4j.rand(new int[]{minibatch, nIn, seqLen}, 12);
            INDArray expected = Nd4j.ones(minibatch, nOut, seqLen);

            INDArray output = graph.outputSingle(false, false, input);
            INDArray error = output.sub(expected);

            for (org.deeplearning4j.nn.api.Layer l : graph.getLayers()) {
                assertNotNull(l.input());
                assertFalse(l.input().isAttached());
            }

            // Compute Gradient
            Gradient gradient = graph.backpropGradient(error);
            graph.getUpdater().update(gradient, 0, 0, minibatch, LayerWorkspaceMgr.noWorkspaces());

            Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        }

        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.DISABLED);
    }

    @Test
    public void testExternalErrorsInvalid(){

        int nIn = 2;
        int nOut = 4;
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("in")
                .addLayer("0", new DenseLayer.Builder().nIn(nIn).nOut(4).activation(Activation.RELU).build(), "in")
                .addLayer("1", new DenseLayer.Builder().nIn(4).nOut(4).activation(Activation.RELU).build(), "0")
                .addLayer("out", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nOut(nOut).build(), "1")
                .setOutputs("out")
                .setInputTypes(InputType.feedForward(nIn))
                .build();

        ComputationGraph cg = new ComputationGraph(conf);
        cg.init();

        INDArray actionInput = Nd4j.randn(3, nIn);
        INDArray[] input = new INDArray[]{actionInput};

        cg.setInputs(input);
        cg.feedForward(input, true, false);

        INDArray externalError = Nd4j.rand(3, nIn);
        try {
            cg.backpropGradient(externalError);
            fail("Expected exception");
        } catch (IllegalStateException e){
            assertTrue(e.getMessage().contains("output layer"));
        }
    }

    @Test
    public void testGradientUpdate() {
        DataSetIterator iter = new IrisDataSetIterator(1, 1);

        Gradient expectedGradient = new DefaultGradient();
        expectedGradient.setGradientFor("first_W", Nd4j.ones(4, 5));
        expectedGradient.setGradientFor("first_b", Nd4j.ones(1, 5));
        expectedGradient.setGradientFor("output_W", Nd4j.ones(5, 3));
        expectedGradient.setGradientFor("output_b", Nd4j.ones(1, 3));

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).graphBuilder()
                .addInputs("input").addLayer("first", new DenseLayer.Builder().nIn(4).nOut(5).build(), "input")
                .addLayer("output", new OutputLayer.Builder().nIn(5).nOut(3).build(), "first")
                .setOutputs("output").pretrain(false).backprop(true).build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();
        net.fit(iter.next());
        Gradient actualGradient = net.gradient;
        assertNotEquals(expectedGradient.getGradientFor("first_W"), actualGradient.getGradientFor("first_W"));

        net.update(expectedGradient);
        actualGradient = net.gradient;
        assertEquals(expectedGradient.getGradientFor("first_W"), actualGradient.getGradientFor("first_W"));

        // Update params with set
        net.setParam("first_W", Nd4j.ones(4, 5));
        net.setParam("first_b", Nd4j.ones(1, 5));
        net.setParam("output_W", Nd4j.ones(5, 3));
        net.setParam("output_b", Nd4j.ones(1, 3));
        INDArray actualParams = net.params();

        // Confirm params
        assertEquals(Nd4j.ones(1, 43), actualParams);

        net.update(expectedGradient);
        actualParams = net.params();
        assertEquals(Nd4j.ones(1, 43).addi(1), actualParams);
    }


    @Test
    public void testCnnFlatInputType1() {

        //First: check conv input type. Expect: no preprocessor, nIn set appropriately
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("in")
                .setInputTypes(InputType.convolutional(10, 8, 3))
                .addLayer("layer",
                        new ConvolutionLayer.Builder().kernelSize(2, 2).padding(0, 0).stride(1, 1)
                                .build(),
                        "in")
                .addLayer("out", new OutputLayer.Builder().nOut(10).build(), "layer").setOutputs("out")
                .pretrain(false).backprop(true).build();

        LayerVertex lv = (LayerVertex) conf.getVertices().get("layer");
        FeedForwardLayer l = ((FeedForwardLayer) (lv).getLayerConf().getLayer());
        assertEquals(3, l.getNIn());
        assertNull(lv.getPreProcessor());

        //Check the equivalent config, but with flat conv data input instead
        //In this case, the only difference should be the addition of a preprocessor
        //First: check conv input type. Expect: no preprocessor, nIn set appropriately
        conf = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("in")
                .setInputTypes(InputType.convolutionalFlat(10, 8, 3))
                .addLayer("layer",
                        new ConvolutionLayer.Builder().kernelSize(2, 2).padding(0, 0).stride(1, 1)
                                .build(),
                        "in")
                .addLayer("out", new OutputLayer.Builder().nOut(10).build(), "layer").setOutputs("out")
                .pretrain(false).backprop(true).build();

        lv = (LayerVertex) conf.getVertices().get("layer");
        l = ((FeedForwardLayer) (lv).getLayerConf().getLayer());
        assertEquals(3, l.getNIn());
        assertNotNull(lv.getPreProcessor());
        InputPreProcessor preProcessor = lv.getPreProcessor();
        assertTrue(preProcessor instanceof FeedForwardToCnnPreProcessor);
        FeedForwardToCnnPreProcessor preproc = (FeedForwardToCnnPreProcessor) preProcessor;
        assertEquals(10, preproc.getInputHeight());
        assertEquals(8, preproc.getInputWidth());
        assertEquals(3, preproc.getNumChannels());


        //Finally, check configuration with a subsampling layer
        conf = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("in")
                .setInputTypes(InputType.convolutionalFlat(10, 8, 3))
                .addLayer("l0", new SubsamplingLayer.Builder().kernelSize(2, 2).stride(1, 1).padding(0, 0)
                        .build(), "in")
                .addLayer("layer",
                        new ConvolutionLayer.Builder().kernelSize(2, 2).padding(0, 0).stride(1, 1)
                                .build(),
                        "l0")
                .addLayer("out", new OutputLayer.Builder().nOut(10).build(), "layer").setOutputs("out")
                .pretrain(false).backprop(true).build();

        //Check subsampling layer:
        lv = (LayerVertex) conf.getVertices().get("l0");
        SubsamplingLayer sl = ((SubsamplingLayer) (lv).getLayerConf().getLayer());
        assertNotNull(lv.getPreProcessor());
        preProcessor = lv.getPreProcessor();
        assertTrue(preProcessor instanceof FeedForwardToCnnPreProcessor);
        preproc = (FeedForwardToCnnPreProcessor) preProcessor;
        assertEquals(10, preproc.getInputHeight());
        assertEquals(8, preproc.getInputWidth());
        assertEquals(3, preproc.getNumChannels());
        //Check dense layer
        lv = (LayerVertex) conf.getVertices().get("layer");
        l = ((FeedForwardLayer) (lv).getLayerConf().getLayer());
        assertEquals(3, l.getNIn());
        assertNull(lv.getPreProcessor());

    }

    @Test
    public void testCGEvaluation() {

        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration configuration = getIrisGraphConfiguration();
        ComputationGraph graph = new ComputationGraph(configuration);
        graph.init();

        Nd4j.getRandom().setSeed(12345);
        MultiLayerConfiguration mlnConfig = getIrisMLNConfiguration();
        MultiLayerNetwork net = new MultiLayerNetwork(mlnConfig);
        net.init();

        DataSetIterator iris = new IrisDataSetIterator(75, 150);

        net.fit(iris);
        iris.reset();
        graph.fit(iris);

        iris.reset();
        Evaluation evalExpected = net.evaluate(iris);
        iris.reset();
        Evaluation evalActual = graph.evaluate(iris);

        assertEquals(evalExpected.accuracy(), evalActual.accuracy(), 0e-4);
    }

    @Test
    public void testOptimizationAlgorithmsSearchBasic() {
        DataSetIterator iter = new IrisDataSetIterator(1, 1);

        OptimizationAlgorithm[] oas = new OptimizationAlgorithm[]{OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT,
                OptimizationAlgorithm.LINE_GRADIENT_DESCENT, OptimizationAlgorithm.CONJUGATE_GRADIENT,
                OptimizationAlgorithm.LBFGS};

        for (OptimizationAlgorithm oa : oas) {
            System.out.println(oa);
            ComputationGraphConfiguration conf =
                    new NeuralNetConfiguration.Builder().optimizationAlgo(oa).graphBuilder()
                            .addInputs("input")
                            .addLayer("first", new DenseLayer.Builder().nIn(4).nOut(5).build(), "input")
                            .addLayer("output", new OutputLayer.Builder().nIn(5).nOut(3).build(),
                                    "first")
                            .setOutputs("output").pretrain(false).backprop(true).build();

            ComputationGraph net = new ComputationGraph(conf);
            net.init();
            net.fit(iter.next());

        }
    }

    @Test
    public void testIterationCountAndPersistence() throws IOException {
        Nd4j.getRandom().setSeed(123);
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).seed(123)
                .graphBuilder().addInputs("in")
                .addLayer("0", new DenseLayer.Builder().nIn(4).nOut(3).weightInit(WeightInit.XAVIER)
                        .activation(Activation.TANH).build(), "in")
                .addLayer("1", new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(3).nOut(3)
                                .build(),
                        "0")
                .setOutputs("1").backprop(true).pretrain(false).build();


        ComputationGraph network = new ComputationGraph(conf);
        network.init();

        DataSetIterator iter = new IrisDataSetIterator(50, 150);

        assertEquals(0, network.getConfiguration().getIterationCount());
        network.fit(iter);
        assertEquals(3, network.getConfiguration().getIterationCount());
        iter.reset();
        network.fit(iter);
        assertEquals(6, network.getConfiguration().getIterationCount());
        iter.reset();
        network.fit(iter.next());
        assertEquals(7, network.getConfiguration().getIterationCount());

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ModelSerializer.writeModel(network, baos, true);
        byte[] asBytes = baos.toByteArray();

        ByteArrayInputStream bais = new ByteArrayInputStream(asBytes);
        ComputationGraph net = ModelSerializer.restoreComputationGraph(bais, true);
        assertEquals(7, net.getConfiguration().getIterationCount());
    }

    @Test
    public void printSummary() {
        NeuralNetConfiguration.Builder overallConf = new NeuralNetConfiguration.Builder().updater(new Sgd(0.1))
                .activation(Activation.IDENTITY);

        ComputationGraphConfiguration conf = overallConf.graphBuilder().addInputs("inCentre", "inRight")
                .addLayer("denseCentre0", new DenseLayer.Builder().nIn(10).nOut(9).build(), "inCentre")
                .addLayer("denseCentre1", new DenseLayer.Builder().nIn(9).nOut(8).build(), "denseCentre0")
                .addLayer("denseCentre2", new DenseLayer.Builder().nIn(8).nOut(7).build(), "denseCentre1")
                .addLayer("denseCentre3", new DenseLayer.Builder().nIn(7).nOut(7).build(), "denseCentre2")
                .addLayer("outCentre",
                        new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(7).nOut(4).build(),
                        "denseCentre3")
                .addVertex("subsetLeft", new SubsetVertex(0, 3), "denseCentre1")
                .addLayer("denseLeft0", new DenseLayer.Builder().nIn(4).nOut(5).build(), "subsetLeft")
                .addLayer("outLeft",
                        new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(5).nOut(6).build(),
                        "denseLeft0")
                .addLayer("denseRight", new DenseLayer.Builder().nIn(7).nOut(7).build(), "denseCentre2")
                .addLayer("denseRight0", new DenseLayer.Builder().nIn(2).nOut(3).build(), "inRight")
                .addVertex("mergeRight", new MergeVertex(), "denseRight", "denseRight0")
                .addLayer("denseRight1", new DenseLayer.Builder().nIn(10).nOut(5).build(), "mergeRight")
                .addLayer("outRight",
                        new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(5).nOut(5).build(),
                        "denseRight1")
                .setOutputs("outLeft", "outCentre", "outRight").build();

        ComputationGraph modelToTune = new ComputationGraph(conf);
        modelToTune.init();
        System.out.println(modelToTune.summary());

        ComputationGraph modelNow =
                new TransferLearning.GraphBuilder(modelToTune).setFeatureExtractor("denseCentre2").build();
        System.out.println(modelNow.summary());
        System.out.println(modelNow.summary(InputType.feedForward(10),InputType.feedForward(2)));
    }

    @Test
    public void testFeedForwardIncludeNonLayerVertices() {

        ComputationGraphConfiguration c = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("in")
                .addLayer("0", new DenseLayer.Builder().nIn(5).nOut(5).build(), "in")
                .addLayer("1", new DenseLayer.Builder().nIn(5).nOut(5).build(), "in")
                .addVertex("merge", new MergeVertex(), "0", "1")
                .addLayer("out", new OutputLayer.Builder().nIn(10).nOut(5).build(), "merge").setOutputs("out")
                .build();

        ComputationGraph cg = new ComputationGraph(c);
        cg.init();

        cg.setInputs(Nd4j.ones(5));

        Map<String, INDArray> layersOnly = cg.feedForward(true, false, false);
        Map<String, INDArray> alsoVertices = cg.feedForward(true, false, true);

        assertEquals(4, layersOnly.size()); //3 layers + 1 input
        assertEquals(5, alsoVertices.size()); //3 layers + 1 input + merge vertex

        assertFalse(layersOnly.containsKey("merge"));
        assertTrue(alsoVertices.containsKey("merge"));
    }


    @Test
    public void testSetOutputsMultipleCalls() {

        //Users generally shouldn't do this, but multiple setOutputs calls should *replace* not *add* outputs

        ComputationGraphConfiguration c = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("in")
                .addLayer("out", new OutputLayer.Builder().nIn(10).nOut(5).build(), "in").setOutputs("out")
                .setOutputs("out").build();

        List<String> l = c.getNetworkOutputs();
        assertEquals(1, l.size());
    }

    @Test
    public void testDropoutValidation() {
        //At one point: this threw an exception due to incorrect validation
        for (boolean dropConnect : new boolean[]{false, true}) {
            new NeuralNetConfiguration.Builder().weightNoise(new DropConnect(0.5))
                    .graphBuilder().setInputTypes(InputType.feedForward(1)).addInputs("input1")
                    .addLayer("output",
                            new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(1).nOut(1)
                                    .activation(Activation.SIGMOID).build(),
                            "input1")
                    .setOutputs("output").pretrain(false).backprop(true).backpropType(BackpropType.Standard)
                    .build();
        }
    }

    @Test
    public void testNoParamLayersL1L2() {

        //Don't care about this being valid
        ComputationGraphConfiguration c =
                new NeuralNetConfiguration.Builder().l1(0.5).l2(0.6).graphBuilder()
                        .addInputs("in")
                        .addLayer("sub1", new SubsamplingLayer.Builder(2, 2).build(), "in")
                        .addLayer("sub2", new Subsampling1DLayer.Builder(2).build(), "sub1")
                        .addLayer("act", new ActivationLayer.Builder().activation(Activation.TANH)
                                .build(), "sub2")
                        .addLayer("pad", new ZeroPaddingLayer.Builder(2, 3).build(), "act")
                        .addLayer("lrn", new LocalResponseNormalization.Builder().build(), "pad")
                        .addLayer("pool", new GlobalPoolingLayer.Builder(PoolingType.AVG).build(),
                                "act")
                        .addLayer("drop", new DropoutLayer.Builder(0.5).build(), "pool")
                        .addLayer("dense", new DenseLayer.Builder().nIn(1).nOut(1).build(), "drop")
                        .addLayer("loss", new LossLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                .build(), "dense")
                        .allowDisconnected(true)
                        .setOutputs("loss").build();

        ComputationGraph g = new ComputationGraph(c);
        g.init();

        g.calcL2();
        g.calcL1();
    }

    @Test(expected = DL4JException.class)
    public void testErrorNoOutputLayer() {

        ComputationGraphConfiguration c = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("in")
                .addLayer("dense", new DenseLayer.Builder().nIn(10).nOut(10).build(), "in").setOutputs("dense")
                .build();

        ComputationGraph cg = new ComputationGraph(c);
        cg.init();

        INDArray f = Nd4j.create(1, 10);
        INDArray l = Nd4j.create(1, 10);

        cg.setInputs(f);
        cg.setLabels(l);

        cg.computeGradientAndScore();
    }


    @Test
    public void testMergeVertexAddition() {

        //When a vertex supports only one input, and gets multiple inputs - we should automatically add a merge
        //vertex

        NeuralNetConfiguration nnc = new NeuralNetConfiguration();
        nnc.setLayer(new DenseLayer.Builder().build());
        GraphVertex[] singleInputVertices = new GraphVertex[]{new L2NormalizeVertex(), new LayerVertex(nnc, null),
                new PoolHelperVertex(), new PreprocessorVertex(), new ReshapeVertex(new int[]{1, 1}),
                new ScaleVertex(1.0), new ShiftVertex(1.0), new SubsetVertex(1, 1), new UnstackVertex(0, 2),
                new DuplicateToTimeSeriesVertex("in1"), new LastTimeStepVertex("in1")};

        for (GraphVertex gv : singleInputVertices) {
            ComputationGraphConfiguration c = new NeuralNetConfiguration.Builder().graphBuilder()
                    .addInputs("in1", "in2").addVertex("gv", gv, "in1", "in2").setOutputs("gv").build();

            boolean foundMerge = false;
            for (GraphVertex g : c.getVertices().values()) {
                if (g instanceof MergeVertex) {
                    foundMerge = true;
                    break;
                }
            }

            if (!foundMerge) {
                fail("Network did not add merge vertex for vertex " + gv.getClass());
            }
        }
    }


    @Test
    public void testVertexAsOutput() {
        //Simple sanity check: vertex is the last output...

        int minibatch = 10;
        int height = 24;
        int width = 24;
        int depth = 3;

        INDArray img = Nd4j.ones(minibatch, depth, height, width);
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("input")
                .addLayer("L1", new ConvolutionLayer.Builder(new int[]{1, 1}, new int[]{1, 1}, new int[]{0, 0}).nIn(depth).nOut(depth)
                        .build(), "input")
                .addVertex("L2", new ReshapeVertex(minibatch, 1, 36, 48), "L1")
                .setOutputs("L2")
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        INDArray[] out = net.output(img);

        assertNotNull(out);
        assertEquals(1, out.length);
        assertNotNull(out[0]);

        assertArrayEquals(new long[]{minibatch, 1, 36, 48}, out[0].shape());
    }

    @Test
    public void testEpochCounter() throws Exception {

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("in")
                .addLayer("out", new OutputLayer.Builder().nIn(4).nOut(3).build(), "in")
                .setOutputs("out")
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        assertEquals(0, net.getConfiguration().getEpochCount());


        DataSetIterator iter = new IrisDataSetIterator(150, 150);

        for( int i=0; i<4; i++ ){
            assertEquals(i, net.getConfiguration().getEpochCount());
            net.fit(iter);
            assertEquals(i+1, net.getConfiguration().getEpochCount());
        }

        assertEquals(4, net.getConfiguration().getEpochCount());

        ByteArrayOutputStream baos = new ByteArrayOutputStream();

        ModelSerializer.writeModel(net, baos, true);
        byte[] bytes = baos.toByteArray();

        ByteArrayInputStream bais = new ByteArrayInputStream(bytes);

        ComputationGraph restored = ModelSerializer.restoreComputationGraph(bais, true);
        assertEquals(4, restored.getConfiguration().getEpochCount());
    }

    @Test
    public void testSummary() {
        int V_WIDTH = 130;
        int V_HEIGHT = 130;
        int V_NFRAMES = 150;
        ComputationGraphConfiguration confForArchitecture =
                new NeuralNetConfiguration.Builder().seed(12345).l2(0.001) //l2 regularization on all layers
                        .updater(new AdaGrad(0.4)).graphBuilder()
                        .addInputs("in")
                        .addLayer("layer0", new ConvolutionLayer.Builder(10, 10).nIn(3) //3 channels: RGB
                                .nOut(30).stride(4, 4).activation(Activation.RELU).weightInit(
                                        WeightInit.RELU).build(),"in") //Output: (130-10+0)/4+1 = 31 -> 31*31*30
                        .addLayer("layer1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(3, 3).stride(2, 2).build(),"layer0") //(31-3+0)/2+1 = 15
                        .addLayer("layer2", new ConvolutionLayer.Builder(3, 3).nIn(30).nOut(10).stride(2, 2)
                                .activation(Activation.RELU).weightInit(WeightInit.RELU)
                                .updater(Updater.ADAGRAD).build(), "layer1") //Output: (15-3+0)/2+1 = 7 -> 7*7*10 = 490
                        .addLayer("layer3", new DenseLayer.Builder().activation(Activation.RELU).nIn(490).nOut(50)
                                .weightInit(WeightInit.RELU).gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                                .gradientNormalizationThreshold(10).build(), "layer2")
                        .addLayer("layer4", new GravesLSTM.Builder().activation(Activation.SOFTSIGN).nIn(50)
                                .nOut(50).weightInit(WeightInit.XAVIER).updater(Updater.ADAGRAD)
                                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                                .gradientNormalizationThreshold(10)
                                .build(), "layer3")
                        .addLayer("layer5", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                .activation(Activation.SOFTMAX).nIn(50).nOut(4) //4 possible shapes: circle, square, arc, line
                                .weightInit(WeightInit.XAVIER)
                                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                                .gradientNormalizationThreshold(10).build(), "layer4")
                        .setOutputs("layer5")
                        .inputPreProcessor("layer0", new RnnToCnnPreProcessor(V_HEIGHT, V_WIDTH, 3))
                        .inputPreProcessor("layer3", new CnnToFeedForwardPreProcessor(7, 7, 10))
                        .inputPreProcessor("layer4", new FeedForwardToRnnPreProcessor()).pretrain(false)
                        .backprop(true).backpropType(BackpropType.TruncatedBPTT)
                        .tBPTTForwardLength(V_NFRAMES / 5).tBPTTBackwardLength(V_NFRAMES / 5).build();
        ComputationGraph modelExpectedArch = new ComputationGraph(confForArchitecture);
        modelExpectedArch.init();
        ComputationGraph modelMow = new TransferLearning.GraphBuilder(modelExpectedArch).setFeatureExtractor("layer2").build();
        System.out.println(modelExpectedArch.summary());
        System.out.println(modelMow.summary());
        System.out.println(modelExpectedArch.summary(InputType.recurrent(V_HEIGHT* V_WIDTH* 3)));
    }

    @Test
    public void testInputClearance() throws Exception {
        //Activations should be cleared - if not, it's possible for out of (workspace) scope arrays to be around
        // which can cause a crash
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .convolutionMode(ConvolutionMode.Same)
                .graphBuilder()
                .addInputs("in")
                .addLayer("0", new ConvolutionLayer.Builder().kernelSize(2,2).stride(1,1).nIn(1).nOut(1).build(), "in")
                .addLayer("1", new SubsamplingLayer.Builder().kernelSize(2,2).stride(1,1).build(), "0")
                .addLayer("2", new DenseLayer.Builder().nOut(10).build(), "1")
                .addLayer("3", new OutputLayer.Builder().nOut(10).build(), "2")
                .setOutputs("3")
                .setInputTypes(InputType.convolutional(28,28,1))
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        INDArray content = Nd4j.create(1,1,28,28);

        //Check output:
        net.output(content);
        for(org.deeplearning4j.nn.api.Layer l : net.getLayers()){
            assertNull(l.input());
        }

        //Check feedForward:
        net.feedForward(content, false);
        for(org.deeplearning4j.nn.api.Layer l : net.getLayers()){
            assertNull(l.input());
        }
    }


    @Test
    public void testDisconnectedVertex(){

        for(boolean allowDisconnected : new boolean[]{false, true}) {
            try {
                ComputationGraphConfiguration.GraphBuilder b = new NeuralNetConfiguration.Builder()
                        .graphBuilder()
                        .addInputs("in")
                        .addLayer("0", new DenseLayer.Builder().activation(Activation.SIGMOID).nOut(8).build(), "in")
                        .addLayer("1", new DenseLayer.Builder().activation(Activation.SIGMOID).nOut(8).build(), "in") //Disconnected
                        .addLayer("O", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nOut(10).build(), "0")
                        .setOutputs("O")
                        .setInputTypes(InputType.feedForward(8));

                if(allowDisconnected){
                    b.allowDisconnected(true).build();  //No exception
                } else {
                    b.build();  //Expect exception here
                    fail("Expected exception for disconnected vertex");
                }


            } catch (Exception e) {
                //e.printStackTrace();
                if(allowDisconnected){
                    fail("No exception expected");
                } else {
                    String msg = e.getMessage().toLowerCase();
                    assertTrue(msg.contains("disconnected"));
                }
            }
        }
    }

    @Test
    public void testLayerSize(){
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()

                .graphBuilder()
                .addInputs("in")
                .layer("0", new ConvolutionLayer.Builder().kernelSize(2,2).nOut(6).build(), "in")
                .layer("1", new SubsamplingLayer.Builder().kernelSize(2,2).build(), "0")
                .layer("2", new DenseLayer.Builder().nOut(30).build(), "1")
                .layer("3", new OutputLayer.Builder().nOut(13).build(), "2")
                .setOutputs("3")
                .setInputTypes(InputType.convolutional(28,28,3))
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        assertEquals(6, net.layerSize(0));
        assertEquals(0, net.layerSize(1));
        assertEquals(30, net.layerSize(2));
        assertEquals(13, net.layerSize(3));

        assertEquals(3, net.layerInputSize(0));
        assertEquals(0, net.layerInputSize(1));
        assertEquals(((FeedForwardLayer)net.getLayer(2).conf().getLayer()).getNIn(), net.layerInputSize(2));
        assertEquals(30, net.layerInputSize(3));

        assertEquals(6, net.layerSize("0"));
        assertEquals(0, net.layerSize("1"));
        assertEquals(30, net.layerSize("2"));
        assertEquals(13, net.layerSize("3"));

        assertEquals(3, net.layerInputSize("0"));
        assertEquals(0, net.layerInputSize("1"));
        assertEquals(((FeedForwardLayer)net.getLayer(2).conf().getLayer()).getNIn(), net.layerInputSize("2"));
        assertEquals(30, net.layerInputSize("3"));
    }

    @Test
    public void testZeroParamNet() throws Exception {

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("in")
                .layer("0", new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).build(), "in")
                .layer("1", new LossLayer.Builder().activation(Activation.SIGMOID).lossFunction(LossFunctions.LossFunction.MSE).build(), "0")
                .setOutputs("1")
                .setInputTypes(InputType.convolutionalFlat(28,28,1))
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        DataSet ds = new MnistDataSetIterator(16, true, 12345).next();

        INDArray out = net.outputSingle(ds.getFeatures());

        INDArray labelTemp = Nd4j.create(out.shape());
        ds.setLabels(labelTemp);

        net.fit(ds);

        ComputationGraph net2 = TestUtils.testModelSerialization(net);
        INDArray out2 = net2.outputSingle(ds.getFeatures());
        assertEquals(out, out2);
    }

    @Test
    public void scaleVertexGraphTest() {
        final double scaleFactor = 2;
        final double[] inputArr = new double[]{-2, -1, 0, 1, 2};//IntStream.rangeClosed(-2, 2).mapToDouble(i -> i).toArray();
        final double[] expected = new double[inputArr.length];  //DoubleStream.of(inputArr).map(i -> i * scaleFactor).toArray();
        for( int i=0; i<expected.length; i++ ){
            expected[i] = inputArr[i] * scaleFactor;
        }

        final INDArray input = getInputArray4d(inputArr); // Replacing this line with the line below is enough to make test pass
        //final INDArray input = Nd4j.create(new double[][]{inputArr});

        final String inputName = "input";
        final String outputName = "output";
        final String scaleName = "scale";
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                //.inferenceWorkspaceMode(WorkspaceMode.NONE)
                .graphBuilder()
                .addInputs(inputName)
                .setOutputs(outputName)
                .setInputTypes(InputType.inferInputType(input))
                .addVertex(scaleName, new ScaleVertex(scaleFactor), inputName)
                .addLayer(outputName, new OutputLayer.Builder()
                        .activation(new ActivationIdentity())
                        .nOut(input.length())
                        .biasInit(0)
                        .build(), scaleName)
                .build());
        graph.init();

        //graph.fit(new DataSet(input, Nd4j.ones(input.length()))); // Does not help
        //graph.feedForward(new INDArray[] {input}, false); // Uncommenting this line is enough to make test pass

        //Hack output layer to be identity mapping
        graph.getOutputLayer(0).setParam("W", Nd4j.eye(input.length()));
        graph.getOutputLayer(0).setParam("b", Nd4j.zeros(input.length()));
        assertEquals("Incorrect output", Nd4j.create(expected), graph.outputSingle(input));
    }

    private static INDArray getInputArray4d(double[] inputArr) {
        final INDArray input = Nd4j.create(1, 1, inputArr.length, 1);
        for (int i = 0; i < input.length(); i++) {
            input.putScalar(new int[]{0, 0, i, 0}, inputArr[i]);
        }
        return input;
    }



    @Test
    public void testGraphOutputIterators(){

        DataSet all = new IrisDataSetIterator(150,150).next();
        DataSetIterator iter = new IrisDataSetIterator(5,150);

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .graphBuilder()
                .addInputs("in")
                .layer("layer", new OutputLayer.Builder().nIn(4).nOut(3).build(), "in")
                .setOutputs("layer")
                .build();
        ComputationGraph cg = new ComputationGraph(conf);
        cg.init();


        INDArray outAll = cg.outputSingle(all.getFeatures());
        INDArray outIter = cg.outputSingle(iter);

        assertEquals(outAll, outIter);
    }


    @Test
    public void testComputationGraphConfgurationActivationTypes(){

        //Test for a simple net:

        ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("in1", "in2")
                .layer("0", new DenseLayer.Builder().nOut(10).build(), "in1")
                .layer("1", new DenseLayer.Builder().nOut(9).build(), "in1", "in2")
                .layer("2", new DenseLayer.Builder().nOut(8).build(), "in2")
                .layer("3", new DenseLayer.Builder().nOut(7).build(), "0")
                .layer("4", new DenseLayer.Builder().nOut(6).build(), "1", "2")
                .setInputTypes(InputType.feedForward(5), InputType.feedForward(6))
                .allowNoOutput(true);

        ComputationGraphConfiguration conf = builder.build();

        Map<String,InputType> actBuilder = builder.getLayerActivationTypes();
        Map<String,InputType> actConf = conf.getLayerActivationTypes(InputType.feedForward(5), InputType.feedForward(6));

        Map<String, InputType> exp = new HashMap<>();
        exp.put("in1", InputType.feedForward(5));
        exp.put("in2", InputType.feedForward(6));
        exp.put("0", InputType.feedForward(10));
        exp.put("1", InputType.feedForward(9));
        exp.put("1-merge", InputType.feedForward(5+6));
        exp.put("2", InputType.feedForward(8));
        exp.put("3", InputType.feedForward(7));
        exp.put("4", InputType.feedForward(6));
        exp.put("4-merge", InputType.feedForward(9+8));

        assertEquals(exp, actBuilder);
        assertEquals(exp, actConf);
    }




    @Test
    public void testTopoSortSaving(){

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("in1", "in2")
                .addLayer("l0", new DenseLayer.Builder().nIn(10).nOut(10).build(), "in1")
                .addLayer("l1", new DenseLayer.Builder().nIn(20).nOut(10).build(), "in1", "in2")
                .addLayer("l2", new DenseLayer.Builder().nIn(10).nOut(10).build(), "in2")
                .addLayer("l3", new DenseLayer.Builder().nIn(10).nOut(10).build(), "l0")
                .addLayer("l4", new DenseLayer.Builder().nIn(10).nOut(10).build(), "l1")
                .addLayer("l5", new DenseLayer.Builder().nIn(10).nOut(10).build(), "l2")
                .addLayer("l6", new OutputLayer.Builder().nIn(20).nOut(10).build(), "l3", "l5")
                .addLayer("l7", new OutputLayer.Builder().nIn(10).nOut(10).build(), "l4")
                .setOutputs("l6", "l7")
                .build();

        INDArray[] in = new INDArray[]{
                Nd4j.rand(3, 10),
                Nd4j.rand(3, 10)};

        ComputationGraph cg = new ComputationGraph(conf);
        cg.init();

        GraphIndices indices = cg.calculateIndices();

        int[] order = cg.topologicalSortOrder();
        List<String> strOrder = cg.getConfiguration().getTopologicalOrderStr();
        INDArray[] out1 = cg.output(in);

        //Check it's the same after loading:
        ComputationGraph cg2 = TestUtils.testModelSerialization(cg);
        int[] order2 = cg2.topologicalSortOrder();
        List<String> strOrder2 = cg.getConfiguration().getTopologicalOrderStr();
        assertArrayEquals(order, order2);
        assertEquals(strOrder, strOrder2);

        INDArray[] out2 = cg2.output(in);
        assertArrayEquals(out1, out2);

        //Delete the topological order, ensure it gets recreated properly:
        ComputationGraphConfiguration conf3 = cg2.getConfiguration().clone();
        conf3.setTopologicalOrder(null);
        conf3.setTopologicalOrderStr(null);
        ComputationGraph cg3 = new ComputationGraph(conf3);
        cg3.init();
        cg3.setParams(cg2.params());

        int[] order3 = cg3.topologicalSortOrder();
        List<String> strOrder3 = cg.getConfiguration().getTopologicalOrderStr();
        INDArray[] out3 = cg3.output(in);
        assertArrayEquals(order, order3);
        assertEquals(strOrder, strOrder3);
        assertArrayEquals(out1, out3);


        //Now, change the order, and ensure the net is the same... note that we can do [l0, l1, l2] in any order
        List<List<String>> someValidOrders = new ArrayList<>();
        someValidOrders.add(Arrays.asList("in1", "in2", "l0", "l1-merge", "l1", "l2", "l3", "l4", "l5", "l6-merge", "l6", "l7"));
        someValidOrders.add(Arrays.asList("in1", "in2", "l1-merge", "l1", "l0", "l2", "l3", "l4", "l5", "l6-merge", "l6", "l7"));
        someValidOrders.add(Arrays.asList("in1", "in2", "l2", "l1-merge", "l1", "l0", "l3", "l4", "l5", "l6-merge", "l6", "l7"));
        someValidOrders.add(Arrays.asList("in1", "in2", "l2", "l5", "l0", "l1-merge", "l1", "l3", "l4", "l7", "l6-merge", "l6"));

        for(List<String> l : someValidOrders){
            assertEquals(strOrder.size(), l.size());
        }

        for( int i=0; i<someValidOrders.size(); i++ ){
            List<String> l = someValidOrders.get(i);
            int[] arr = new int[l.size()];
            int j=0;
            for(String s : l){
                arr[j++] = indices.getNameToIdx().get(s);
            }

            ComputationGraphConfiguration conf2 = conf.clone();
            conf2.setTopologicalOrderStr(l);
            conf2.setTopologicalOrder(arr);

            ComputationGraph g = new ComputationGraph(conf2);
            g.init();
            g.setParamTable(cg.paramTable());
            int[] origOrder = g.topologicalSortOrder();

            INDArray[] out4 = g.output(in);
            assertArrayEquals(out1, out4);

            ComputationGraph g2 = TestUtils.testModelSerialization(g);
            int[] loadedOrder = g2.topologicalSortOrder();

            assertArrayEquals(origOrder, loadedOrder);

            INDArray[] out5 = g2.output(in);
            assertArrayEquals(out1, out5);
        }
    }

    @Test
    public void testPretrainFitMethods(){

        //The fit methods should *not* do layerwise pretraining:

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()

                .graphBuilder()
                .addInputs("in")
                .layer("0", new VariationalAutoencoder.Builder()
                        .nIn(10).nOut(10).encoderLayerSizes(10).decoderLayerSizes(10).build(), "in")
                .layer("1", new OutputLayer.Builder().nIn(10).nOut(10).build(), "0")
                .setOutputs("1")
                .pretrain(true).backprop(true)
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        Set<Class<?>> exp = new HashSet<>();
        exp.add(ComputationGraph.class);

        MultiLayerTest.CheckModelsListener listener = new MultiLayerTest.CheckModelsListener();
        net.setListeners(listener);

        INDArray f = Nd4j.create(1,10);
        INDArray l = Nd4j.create(1,10);
        DataSet ds = new DataSet(f,l);
        MultiDataSet mds = new org.nd4j.linalg.dataset.MultiDataSet(f,l);

        DataSetIterator iter = new ExistingDataSetIterator(Collections.singletonList(ds));

        net.fit(ds);
        assertEquals(exp, listener.getModelClasses());

        net.fit(iter);
        assertEquals(exp, listener.getModelClasses());

        net.fit(new INDArray[]{f}, new INDArray[]{l});
        assertEquals(exp, listener.getModelClasses());

        net.fit(new INDArray[]{f}, new INDArray[]{l}, null, null);
        assertEquals(exp, listener.getModelClasses());

        net.fit(mds);
        assertEquals(exp, listener.getModelClasses());

        net.fit(new SingletonMultiDataSetIterator(mds));
        assertEquals(exp, listener.getModelClasses());
    }

    @Test
    public void testAllowInputModification(){
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()

                .graphBuilder()
                .addInputs("in1", "in2")
                .layer("0", new DenseLayer.Builder().nOut(10).build(), "in1")   //Modification should not be allowed on input
                .layer("1", new DenseLayer.Builder().nOut(10).build(), "in2")   //Modification should not be allowed on input
                .layer("2", new DenseLayer.Builder().nOut(10).build(), "0")     //Modification SHOULD be allowed
                .layer("3", new DenseLayer.Builder().nOut(10).build(), "1")     //First in topo sort for using this input - not allowed
                .layer("4", new DenseLayer.Builder().nOut(10).build(), "1")     //Second in topo sort - not allowed
                .layer("5", new DenseLayer.Builder().nOut(10).build(), "1")     //Last in topo sort - allowed
                .layer("6", new DenseLayer.Builder().nOut(10).build(), "2", "3", "4", "5")   //Input from merge vertex - allowed
                .setOutputs("6")
                .setInputTypes(InputType.feedForward(10), InputType.feedForward(10))
                .build();

        ComputationGraph cg = new ComputationGraph(conf);
        cg.init();

        Map<String,Boolean> exp = new HashMap<>();
        exp.put("0", false);
        exp.put("1", false);
        exp.put("2", true);
        exp.put("3", false);
        exp.put("4", false);
        exp.put("5", true);
        exp.put("6", true);


        for(String s : exp.keySet()){
            boolean allowed = ((org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer)cg.getLayer(s)).isInputModificationAllowed();
//            System.out.println(s + "\t" + allowed);
            assertEquals(s, exp.get(s), allowed);
        }
    }
}
