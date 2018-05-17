package org.deeplearning4j.gradientcheck;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.graph.*;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.graph.rnn.ReverseTimeSeriesVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class GradientCheckTestsComputationGraph extends BaseDL4JTest {

    public static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-10;

    static {
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void testBasicIris() {
        Nd4j.getRandom().setSeed(12345);
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
                        .setOutputs("outputLayer").pretrain(false).backprop(true).build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        Nd4j.getRandom().setSeed(12345);
        int nParams = graph.numParams();
        INDArray newParams = Nd4j.rand(1, nParams);
        graph.setParams(newParams);

        DataSet ds = new IrisDataSetIterator(150, 150).next();
        INDArray min = ds.getFeatureMatrix().min(0);
        INDArray max = ds.getFeatureMatrix().max(0);
        ds.getFeatureMatrix().subiRowVector(min).diviRowVector(max.sub(min));
        INDArray input = ds.getFeatureMatrix();
        INDArray labels = ds.getLabels();

        if (PRINT_RESULTS) {
            System.out.println("testBasicIris()");
            for (int j = 0; j < graph.getNumLayers(); j++)
                System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(graph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[] {input},
                        new INDArray[] {labels});

        String msg = "testBasicIris()";
        assertTrue(msg, gradOK);
        TestUtils.testModelSerialization(graph);
    }

    @Test
    public void testBasicIrisWithMerging() {
        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1)).updater(new NoOp())
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
                        .setOutputs("outputLayer").pretrain(false).backprop(true).build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        int numParams = (4 * 5 + 5) + (4 * 5 + 5) + (10 * 3 + 3);
        assertEquals(numParams, graph.numParams());

        Nd4j.getRandom().setSeed(12345);
        int nParams = graph.numParams();
        INDArray newParams = Nd4j.rand(1, nParams);
        graph.setParams(newParams);

        DataSet ds = new IrisDataSetIterator(150, 150).next();
        INDArray min = ds.getFeatureMatrix().min(0);
        INDArray max = ds.getFeatureMatrix().max(0);
        ds.getFeatureMatrix().subiRowVector(min).diviRowVector(max.sub(min));
        INDArray input = ds.getFeatureMatrix();
        INDArray labels = ds.getLabels();

        if (PRINT_RESULTS) {
            System.out.println("testBasicIrisWithMerging()");
            for (int j = 0; j < graph.getNumLayers(); j++)
                System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(graph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[] {input},
                        new INDArray[] {labels});

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
                            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                            .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
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
                            .setOutputs("outputLayer").pretrain(false).backprop(true).build();

            ComputationGraph graph = new ComputationGraph(conf);
            graph.init();

            int numParams = (4 * 5 + 5) + (4 * 5 + 5) + (5 * 3 + 3);
            assertEquals(numParams, graph.numParams());

            Nd4j.getRandom().setSeed(12345);
            int nParams = graph.numParams();
            INDArray newParams = Nd4j.rand(1, nParams);
            graph.setParams(newParams);

            DataSet ds = new IrisDataSetIterator(150, 150).next();
            INDArray min = ds.getFeatureMatrix().min(0);
            INDArray max = ds.getFeatureMatrix().max(0);
            ds.getFeatureMatrix().subiRowVector(min).diviRowVector(max.sub(min));
            INDArray input = ds.getFeatureMatrix();
            INDArray labels = ds.getLabels();

            if (PRINT_RESULTS) {
                System.out.println("testBasicIrisWithElementWiseVertex(op=" + op + ")");
                for (int j = 0; j < graph.getNumLayers(); j++)
                    System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(graph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[] {input},
                            new INDArray[] {labels});

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
                            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                            .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
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
                            .setOutputs("outputLayer").pretrain(false).backprop(true).build();

            ComputationGraph graph = new ComputationGraph(conf);
            graph.init();

            int numParams = (4 * 5 + 5) + (4 * 5 + 5) + (4 * 5 + 5) + (5 * 3 + 3);
            assertEquals(numParams, graph.numParams());

            Nd4j.getRandom().setSeed(12345);
            int nParams = graph.numParams();
            INDArray newParams = Nd4j.rand(1, nParams);
            graph.setParams(newParams);

            DataSet ds = new IrisDataSetIterator(150, 150).next();
            INDArray min = ds.getFeatureMatrix().min(0);
            INDArray max = ds.getFeatureMatrix().max(0);
            ds.getFeatureMatrix().subiRowVector(min).diviRowVector(max.sub(min));
            INDArray input = ds.getFeatureMatrix();
            INDArray labels = ds.getLabels();

            if (PRINT_RESULTS) {
                System.out.println("testBasicIrisWithElementWiseVertex(op=" + op + ")");
                for (int j = 0; j < graph.getNumLayers(); j++)
                    System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(graph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[] {input},
                            new INDArray[] {labels});

            String msg = "testBasicIrisWithElementWiseVertex(op=" + op + ")";
            assertTrue(msg, gradOK);
            TestUtils.testModelSerialization(graph);
        }
    }

    @Test
    public void testCnnDepthMerge() {

        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 0.1))
                        .updater(new NoOp()).graphBuilder().addInputs("input")
                        .addLayer("l1", new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1).padding(0, 0)
                                        .nIn(2).nOut(2).activation(Activation.TANH).build(), "input")
                        .addLayer("l2", new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1).padding(0, 0)
                                        .nIn(2).nOut(2).activation(Activation.TANH).build(), "input")
                        .addVertex("merge", new MergeVertex(), "l1", "l2")
                        .addLayer("outputLayer",
                                        new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT)
                                                        .activation(Activation.SOFTMAX).nIn(5 * 5 * (2 + 2)).nOut(3)
                                                        .build(),
                                        "merge")
                        .setOutputs("outputLayer")
                        .inputPreProcessor("outputLayer", new CnnToFeedForwardPreProcessor(5, 5, 4)).pretrain(false)
                        .backprop(true).build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        Random r = new Random(12345);
        INDArray input = Nd4j.rand(new int[] {5, 2, 6, 6}); //Order: examples, channels, height, width
        INDArray labels = Nd4j.zeros(5, 3);
        for (int i = 0; i < 5; i++)
            labels.putScalar(new int[] {i, r.nextInt(3)}, 1.0);

        if (PRINT_RESULTS) {
            System.out.println("testCnnDepthMerge()");
            for (int j = 0; j < graph.getNumLayers(); j++)
                System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(graph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[] {input},
                        new INDArray[] {labels});

        String msg = "testCnnDepthMerge()";
        assertTrue(msg, gradOK);
        TestUtils.testModelSerialization(graph);
    }

    @Test
    public void testLSTMWithMerging() {

        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf =
                        new NeuralNetConfiguration.Builder().seed(12345)
                                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                        .weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(0.2, 0.6))
                                        .updater(new NoOp()).graphBuilder().addInputs("input")
                                        .setOutputs("out")
                                        .addLayer("lstm1",
                                                        new GravesLSTM.Builder().nIn(3).nOut(4)
                                                                        .activation(Activation.TANH).build(),
                                                        "input")
                                        .addLayer("lstm2",
                                                        new GravesLSTM.Builder().nIn(4).nOut(4)
                                                                        .activation(Activation.TANH).build(),
                                                        "lstm1")
                                        .addLayer("dense1",
                                                        new DenseLayer.Builder().nIn(4).nOut(4)
                                                                        .activation(Activation.SIGMOID).build(),
                                                        "lstm1")
                                        .addLayer("lstm3",
                                                        new GravesLSTM.Builder().nIn(4).nOut(4)
                                                                        .activation(Activation.TANH).build(),
                                                        "dense1")
                                        .addVertex("merge", new MergeVertex(), "lstm2", "lstm3")
                                        .addLayer("out", new RnnOutputLayer.Builder().nIn(8).nOut(3)
                                                        .activation(Activation.SOFTMAX)
                                                        .lossFunction(LossFunctions.LossFunction.MCXENT).build(),
                                                        "merge")
                                        .inputPreProcessor("dense1", new RnnToFeedForwardPreProcessor())
                                        .inputPreProcessor("lstm3", new FeedForwardToRnnPreProcessor()).pretrain(false)
                                        .backprop(true).build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        Random r = new Random(12345);
        INDArray input = Nd4j.rand(new int[] {3, 3, 5});
        INDArray labels = Nd4j.zeros(3, 3, 5);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 5; j++) {
                labels.putScalar(new int[] {i, r.nextInt(3), j}, 1.0);
            }
        }

        if (PRINT_RESULTS) {
            System.out.println("testLSTMWithMerging()");
            for (int j = 0; j < graph.getNumLayers(); j++)
                System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(graph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[] {input},
                        new INDArray[] {labels});

        String msg = "testLSTMWithMerging()";
        assertTrue(msg, gradOK);
        TestUtils.testModelSerialization(graph);
    }

    @Test
    public void testLSTMWithSubset() {
        Nd4j.getRandom().setSeed(1234);
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(1234)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                        .updater(new NoOp()).graphBuilder().addInputs("input").setOutputs("out")
                        .addLayer("lstm1", new GravesLSTM.Builder().nIn(3).nOut(8).activation(Activation.TANH).build(),
                                        "input")
                        .addVertex("subset", new SubsetVertex(0, 3), "lstm1")
                        .addLayer("out", new RnnOutputLayer.Builder().nIn(4).nOut(3).activation(Activation.SOFTMAX)
                                        .lossFunction(LossFunctions.LossFunction.MCXENT).build(), "subset")
                        .pretrain(false).backprop(true).build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        Random r = new Random(12345);
        INDArray input = Nd4j.rand(new int[] {3, 3, 5});
        INDArray labels = Nd4j.zeros(3, 3, 5);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 5; j++) {
                labels.putScalar(new int[] {i, r.nextInt(3), j}, 1.0);
            }
        }

        if (PRINT_RESULTS) {
            System.out.println("testLSTMWithSubset()");
            for (int j = 0; j < graph.getNumLayers(); j++)
                System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(graph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[] {input},
                        new INDArray[] {labels});

        String msg = "testLSTMWithSubset()";
        assertTrue(msg, gradOK);
        TestUtils.testModelSerialization(graph);
    }

    @Test
    public void testLSTMWithLastTimeStepVertex() {

        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                        .updater(new NoOp()).graphBuilder().addInputs("input").setOutputs("out")
                        .addLayer("lstm1", new GravesLSTM.Builder().nIn(3).nOut(4).activation(Activation.TANH).build(),
                                        "input")
                        .addVertex("lastTS", new LastTimeStepVertex("input"), "lstm1")
                        .addLayer("out", new OutputLayer.Builder().nIn(4).nOut(3).activation(Activation.SOFTMAX)
                                        .lossFunction(LossFunctions.LossFunction.MCXENT).build(), "lastTS")
                        .pretrain(false).backprop(true).build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        Random r = new Random(12345);
        INDArray input = Nd4j.rand(new int[] {3, 3, 5});
        INDArray labels = Nd4j.zeros(3, 3); //Here: labels are 2d (due to LastTimeStepVertex)
        for (int i = 0; i < 3; i++) {
            labels.putScalar(new int[] {i, r.nextInt(3)}, 1.0);
        }

        if (PRINT_RESULTS) {
            System.out.println("testLSTMWithLastTimeStepVertex()");
            for (int j = 0; j < graph.getNumLayers(); j++)
                System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
        }

        //First: test with no input mask array
        boolean gradOK = GradientCheckUtil.checkGradients(graph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[] {input},
                        new INDArray[] {labels});

        String msg = "testLSTMWithLastTimeStepVertex()";
        assertTrue(msg, gradOK);

        //Second: test with input mask arrays.
        INDArray inMask = Nd4j.zeros(3, 5);
        inMask.putRow(0, Nd4j.create(new double[] {1, 1, 1, 0, 0}));
        inMask.putRow(1, Nd4j.create(new double[] {1, 1, 1, 1, 0}));
        inMask.putRow(2, Nd4j.create(new double[] {1, 1, 1, 1, 1}));
        graph.setLayerMaskArrays(new INDArray[] {inMask}, null);
        gradOK = GradientCheckUtil.checkGradients(graph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR, DEFAULT_MIN_ABS_ERROR,
                        PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[] {input}, new INDArray[] {labels});

        assertTrue(msg, gradOK);
        TestUtils.testModelSerialization(graph);
    }

    @Test
    public void testLSTMWithDuplicateToTimeSeries() {

        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf =
                        new NeuralNetConfiguration.Builder().seed(12345)
                                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                                        .updater(new NoOp()).graphBuilder()
                                        .addInputs("input1", "input2").setOutputs("out")
                                        .addLayer("lstm1",
                                                        new GravesLSTM.Builder().nIn(3).nOut(4)
                                                                        .activation(Activation.TANH).build(),
                                                        "input1")
                                        .addLayer("lstm2",
                                                        new GravesLSTM.Builder().nIn(4).nOut(5)
                                                                        .activation(Activation.SOFTSIGN).build(),
                                                        "input2")
                                        .addVertex("lastTS", new LastTimeStepVertex("input2"), "lstm2")
                                        .addVertex("duplicate", new DuplicateToTimeSeriesVertex("input2"), "lastTS")
                                        .addLayer("out", new RnnOutputLayer.Builder().nIn(5 + 4).nOut(3)
                                                        .activation(Activation.SOFTMAX)
                                                        .lossFunction(LossFunctions.LossFunction.MCXENT).build(),
                                                        "lstm1", "duplicate")
                                        .pretrain(false).backprop(true).build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        Random r = new Random(12345);
        INDArray input1 = Nd4j.rand(new int[] {3, 3, 5});
        INDArray input2 = Nd4j.rand(new int[] {3, 4, 5});
        INDArray labels = Nd4j.zeros(3, 3, 5);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 5; j++) {
                labels.putScalar(new int[] {i, r.nextInt(3), j}, 1.0);
            }
        }

        if (PRINT_RESULTS) {
            System.out.println("testLSTMWithDuplicateToTimeSeries()");
            for (int j = 0; j < graph.getNumLayers(); j++)
                System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(graph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[] {input1, input2},
                        new INDArray[] {labels});

        String msg = "testLSTMWithDuplicateToTimeSeries()";
        assertTrue(msg, gradOK);
        TestUtils.testModelSerialization(graph);
    }

    @Test
    public void testLSTMWithReverseTimeSeriesVertex() {

        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf =
                new NeuralNetConfiguration.Builder().seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                        .updater(new NoOp()).graphBuilder()
                        .addInputs("input").setOutputs("out")
                        .addLayer("lstm_a",
                                new GravesLSTM.Builder().nIn(3).nOut(4)
                                        .activation(Activation.TANH).build(),
                                "input")
                        .addVertex("input_rev", new ReverseTimeSeriesVertex("input"), "input")
                        .addLayer("lstm_b",
                                new GravesLSTM.Builder().nIn(3).nOut(4)
                                        .activation(Activation.TANH).build(),
                                "input_rev")
                        .addVertex("lstm_b_rev", new ReverseTimeSeriesVertex("input"), "lstm_b")
                        .addLayer("out", new RnnOutputLayer.Builder().nIn(4 + 4).nOut(3)
                                        .activation(Activation.SOFTMAX)
                                        .lossFunction(LossFunctions.LossFunction.MCXENT).build(),
                                "lstm_a", "lstm_b_rev")
                        .pretrain(false).backprop(true).build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        Random r = new Random(12345);
        INDArray input  = Nd4j.rand(new int[] {3, 3, 5});
        INDArray labels = Nd4j.zeros(3, 3, 5);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 5; j++) {
                labels.putScalar(new int[] {i, r.nextInt(3), j}, 1.0);
            }
        }

        if (PRINT_RESULTS) {
            System.out.println("testLSTMWithReverseTimeSeriesVertex()");
            for (int j = 0; j < graph.getNumLayers(); j++)
                System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(graph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[] {input},
                new INDArray[] {labels});

        String msg = "testLSTMWithDuplicateToTimeSeries()";
        assertTrue(msg, gradOK);

        //Second: test with input mask arrays.
        INDArray inMask = Nd4j.zeros(3, 5);
        inMask.putRow(0, Nd4j.create(new double[] {1, 1, 1, 0, 0}));
        inMask.putRow(1, Nd4j.create(new double[] {1, 1, 0, 1, 0}));
        inMask.putRow(2, Nd4j.create(new double[] {1, 1, 1, 1, 1}));
        graph.setLayerMaskArrays(new INDArray[] {inMask}, null);
        gradOK = GradientCheckUtil.checkGradients(graph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR, DEFAULT_MIN_ABS_ERROR,
                PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[] {input}, new INDArray[] {labels});

        assertTrue(msg, gradOK);
        TestUtils.testModelSerialization(graph);
    }

    @Test
    public void testMultipleInputsLayer() {

        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                        .updater(new NoOp()).activation(Activation.TANH).graphBuilder().addInputs("i0", "i1", "i2")
                        .addLayer("d0", new DenseLayer.Builder().nIn(2).nOut(2).build(), "i0")
                        .addLayer("d1", new DenseLayer.Builder().nIn(2).nOut(2).build(), "i1")
                        .addLayer("d2", new DenseLayer.Builder().nIn(2).nOut(2).build(), "i2")
                        .addLayer("d3", new DenseLayer.Builder().nIn(6).nOut(2).build(), "d0", "d1", "d2")
                        .addLayer("out", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(2)
                                        .nOut(2).build(), "d3")
                        .setOutputs("out").pretrain(false).backprop(true).build();

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
                for (int j = 0; j < graph.getNumLayers(); j++)
                    System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(graph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, inputs,
                            new INDArray[] {out});

            assertTrue(msg, gradOK);
            TestUtils.testModelSerialization(graph);
        }
    }

    @Test
    public void testMultipleOutputsLayer() {
        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                        .updater(new NoOp()).activation(Activation.TANH).graphBuilder().addInputs("i0")
                        .addLayer("d0", new DenseLayer.Builder().nIn(2).nOut(2).build(), "i0")
                        .addLayer("d1", new DenseLayer.Builder().nIn(2).nOut(2).build(), "d0")
                        .addLayer("d2", new DenseLayer.Builder().nIn(2).nOut(2).build(), "d0")
                        .addLayer("d3", new DenseLayer.Builder().nIn(2).nOut(2).build(), "d0")
                        .addLayer("out", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(6)
                                        .nOut(2).build(), "d1", "d2", "d3")
                        .setOutputs("out").pretrain(false).backprop(true).build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        int[] minibatchSizes = {1, 3};
        for (int mb : minibatchSizes) {
            INDArray input = Nd4j.rand(mb, 2);
            INDArray out = Nd4j.rand(mb, 2);


            String msg = "testMultipleOutputsLayer() - minibatchSize = " + mb;
            if (PRINT_RESULTS) {
                System.out.println(msg);
                for (int j = 0; j < graph.getNumLayers(); j++)
                    System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(graph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[] {input},
                            new INDArray[] {out});

            assertTrue(msg, gradOK);
            TestUtils.testModelSerialization(graph);
        }
    }

    @Test
    public void testMultipleOutputsMergeVertex() {
        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
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
                        .setOutputs("out").pretrain(false).backprop(true).build();

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
                for (int j = 0; j < graph.getNumLayers(); j++)
                    System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(graph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, new INDArray[] {out});

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
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
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
                        .setOutputs("out").setInputTypes(InputType.convolutional(inH, inW, 2)).pretrain(false)
                        .backprop(true).build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        int[] minibatchSizes = {1, 3};
        for (int mb : minibatchSizes) {
            INDArray input = Nd4j.rand(new int[] {mb, 2, inH, inW}).muli(4); //Order: examples, channels, height, width
            INDArray out = Nd4j.rand(mb, 2);

            String msg = "testMultipleOutputsMergeVertex() - minibatchSize = " + mb;
            if (PRINT_RESULTS) {
                System.out.println(msg);
                for (int j = 0; j < graph.getNumLayers(); j++)
                    System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(graph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[] {input},
                            new INDArray[] {out});

            assertTrue(msg, gradOK);
            TestUtils.testModelSerialization(graph);
        }
    }


    @Test
    public void testBasicIrisTripletStackingL2Loss() {
        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf =
                        new NeuralNetConfiguration.Builder().seed(12345)
                                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
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
                                        .setOutputs("lossLayer").pretrain(false).backprop(true).build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        int numParams = (4 * 5 + 5);
        assertEquals(numParams, graph.numParams());

        Nd4j.getRandom().setSeed(12345);
        int nParams = graph.numParams();
        INDArray newParams = Nd4j.rand(1, nParams);
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

        for (String s : out.keySet()) {
            System.out.println(s + "\t" + Arrays.toString(out.get(s).shape()));
        }

        if (PRINT_RESULTS) {
            System.out.println("testBasicIrisTripletStackingL2Loss()");
            for (int j = 0; j < graph.getNumLayers(); j++)
                System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(graph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[] {pos, anc, neg},
                        new INDArray[] {labels});

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
                                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                .weightInit(WeightInit.DISTRIBUTION).dist(new GaussianDistribution(0, 1))
                                .updater(new NoOp()).graphBuilder().addInputs("input1")
                                .addLayer("l1", new DenseLayer.Builder().nIn(4).nOut(5).activation(Activation.TANH)
                                                .build(), "input1")
                                .addLayer("cl", new CenterLossOutputLayer.Builder()
                                                .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(5).nOut(numLabels)
                                                .alpha(1.0).lambda(lambda).gradientCheck(true)
                                                .activation(Activation.SOFTMAX).build(), "l1")
                                .setOutputs("cl").pretrain(false).backprop(true).build();

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
                    for (int j = 0; j < graph.getNumLayers(); j++)
                        System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
                }

                boolean gradOK = GradientCheckUtil.checkGradients(graph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                                DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[] {example},
                                new INDArray[] {labels});

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
                                .updater(new NoOp()).weightInit(WeightInit.DISTRIBUTION)
                                .dist(new NormalDistribution(0, 1.0)).seed(12345L).list()
                                .layer(0, new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1).nOut(3).build())
                                .layer(1, new GlobalPoolingLayer.Builder().poolingType(PoolingType.AVG).build())
                                .layer(2, new CenterLossOutputLayer.Builder()
                                                .lossFunction(LossFunctions.LossFunction.MCXENT).nOut(numLabels)
                                                .alpha(1.0).lambda(lambda).gradientCheck(true)
                                                .activation(Activation.SOFTMAX).build())
                                .pretrain(false).backprop(true)
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
                    for (int j = 0; j < net.getnLayers(); j++)
                        System.out.println("Layer " + j + " # params: " + net.getLayer(j).numParams());
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
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                        .activation(Activation.TANH).updater(new NoOp()).graphBuilder()
                        .addInputs("in1", "in2").addLayer("d0", new DenseLayer.Builder().nIn(2).nOut(2).build(), "in1")
                        .addLayer("d1", new DenseLayer.Builder().nIn(2).nOut(2).build(), "in2")
                        .addVertex("l2", new L2Vertex(), "d0", "d1")
                        .addLayer("out", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.L2).nIn(1)
                                        .nOut(1).activation(Activation.IDENTITY).build(), "l2")
                        .setOutputs("out").pretrain(false).backprop(true).build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();


        Nd4j.getRandom().setSeed(12345);
        int nParams = graph.numParams();
        INDArray newParams = Nd4j.rand(1, nParams);
        graph.setParams(newParams);

        int[] mbSizes = new int[] {1, 3, 10};
        for (int minibatch : mbSizes) {

            INDArray in1 = Nd4j.rand(minibatch, 2);
            INDArray in2 = Nd4j.rand(minibatch, 2);

            INDArray labels = Nd4j.rand(minibatch, 1);

            String testName = "testBasicL2() - minibatch = " + minibatch;

            if (PRINT_RESULTS) {
                System.out.println(testName);
                for (int j = 0; j < graph.getNumLayers(); j++)
                    System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(graph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[] {in1, in2},
                            new INDArray[] {labels});

            assertTrue(testName, gradOK);
            TestUtils.testModelSerialization(graph);
        }
    }


    @Test
    public void testBasicStackUnstack() {

        int layerSizes = 2;

        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
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
                        .setOutputs("out1", "out2").pretrain(false).backprop(true).build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();


        Nd4j.getRandom().setSeed(12345);
        int nParams = graph.numParams();
        INDArray newParams = Nd4j.rand(1, nParams);
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
                for (int j = 0; j < graph.getNumLayers(); j++)
                    System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(graph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[] {in1, in2},
                            new INDArray[] {labels1, labels2});

            assertTrue(testName, gradOK);
            TestUtils.testModelSerialization(graph);
        }
    }

    @Test
    public void testBasicStackUnstackDebug() {
        Nd4j.getRandom().setSeed(12345);

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
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
                        .setOutputs("out1", "out2").pretrain(false).backprop(true).build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();


        Nd4j.getRandom().setSeed(12345);
        int nParams = graph.numParams();
        INDArray newParams = Nd4j.rand(1, nParams);
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
                for (int j = 0; j < graph.getNumLayers(); j++)
                    System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(graph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[] {in1, in2},
                            new INDArray[] {labels1, labels2});

            assertTrue(testName, gradOK);
            TestUtils.testModelSerialization(graph);
        }
    }

    @Test
    public void testBasicStackUnstackVariableLengthTS() {

        int layerSizes = 2;

        Nd4j.getRandom().setSeed(12345);
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                        .activation(Activation.TANH).updater(new NoOp()).graphBuilder()
                        .addInputs("in1", "in2")
                        .addLayer("d0", new GravesLSTM.Builder().nIn(layerSizes).nOut(layerSizes).build(), "in1")
                        .addLayer("d1", new GravesLSTM.Builder().nIn(layerSizes).nOut(layerSizes).build(), "in2")
                        .addVertex("stack", new StackVertex(), "d0", "d1")
                        .addLayer("d2", new GravesLSTM.Builder().nIn(layerSizes).nOut(layerSizes).build(), "stack")
                        .addVertex("u1", new UnstackVertex(0, 2), "d2").addVertex("u2", new UnstackVertex(1, 2), "d2")
                        .addLayer("p1", new GlobalPoolingLayer.Builder(PoolingType.AVG).build(), "u1")
                        .addLayer("p2", new GlobalPoolingLayer.Builder(PoolingType.AVG).build(), "u2")
                        .addLayer("out1", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.L2)
                                        .nIn(layerSizes).nOut(layerSizes).activation(Activation.IDENTITY).build(), "p1")
                        .addLayer("out2", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.L2)
                                        .nIn(layerSizes).nOut(2).activation(Activation.IDENTITY).build(), "p2")
                        .setOutputs("out1", "out2").pretrain(false).backprop(true).build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();


        Nd4j.getRandom().setSeed(12345);
        int nParams = graph.numParams();
        INDArray newParams = Nd4j.rand(1, nParams);
        graph.setParams(newParams);

        int[] mbSizes = new int[] {1, 3, 10};
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
                for (int j = 0; j < graph.getNumLayers(); j++)
                    System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
            }

            graph.setLayerMaskArrays(new INDArray[] {inMask1, inMask2}, null);

            boolean gradOK = GradientCheckUtil.checkGradients(graph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[] {in1, in2},
                            new INDArray[] {labels1, labels2}, new INDArray[] {inMask1, inMask2}, null);

            assertTrue(testName, gradOK);
            TestUtils.testModelSerialization(graph);
        }
    }

    @Test
    public void testBasicTwoOutputs() {
        Nd4j.getRandom().setSeed(12345);

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
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
                        .setOutputs("out1", "out2").pretrain(false).backprop(true).build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        System.out.println("Num layers: " + graph.getNumLayers());
        System.out.println("Num params: " + graph.numParams());


        Nd4j.getRandom().setSeed(12345);
        int nParams = graph.numParams();
        INDArray newParams = Nd4j.rand(1, nParams);
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
                for (int j = 0; j < graph.getNumLayers(); j++)
                    System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(graph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[] {in1, in2},
                            new INDArray[] {labels1, labels2});
            assertTrue(testName, gradOK);
            TestUtils.testModelSerialization(graph);
        }
    }

    @Test
    public void testL2NormalizeVertex2d() {
        Nd4j.getRandom().setSeed(12345);

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                        .activation(Activation.TANH).updater(new NoOp()).graphBuilder()
                        .addInputs("in1").addLayer("d1", new DenseLayer.Builder().nIn(2).nOut(3).build(), "in1")
                        .addVertex("norm", new L2NormalizeVertex(), "d1")
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
                for (int j = 0; j < graph.getNumLayers(); j++)
                    System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(graph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[] {in1},
                            new INDArray[] {labels1});

            assertTrue(testName, gradOK);
            TestUtils.testModelSerialization(graph);
        }
    }

    @Test
    public void testL2NormalizeVertex4d() {
        Nd4j.getRandom().setSeed(12345);

        int h = 4;
        int w = 4;
        int dIn = 2;

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
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
                for (int j = 0; j < graph.getNumLayers(); j++)
                    System.out.println("Layer " + j + " # params: " + graph.getLayer(j).numParams());
            }

            boolean gradOK = GradientCheckUtil.checkGradients(graph, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[] {in1},
                            new INDArray[] {labels1});

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
            for (int j = 0; j < cg.getNumLayers(); j++)
                System.out.println("Layer " + j + " # params: " + cg.getLayer(j).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(cg, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR, DEFAULT_MIN_ABS_ERROR,
                        PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[] {input}, new INDArray[] {labels});

        String msg = "testGraphEmbeddingLayerSimple";
        assertTrue(msg, gradOK);
        TestUtils.testModelSerialization(cg);
    }
}
