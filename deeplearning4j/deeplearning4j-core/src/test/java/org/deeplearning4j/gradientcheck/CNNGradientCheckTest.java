package org.deeplearning4j.gradientcheck;

import lombok.val;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping2D;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.util.Arrays;

import static org.deeplearning4j.nn.conf.ConvolutionMode.Same;
import static org.deeplearning4j.nn.conf.ConvolutionMode.Truncate;
import static org.junit.Assert.*;

/**
 * Created by nyghtowl on 9/1/15.
 */
public class CNNGradientCheckTest extends BaseDL4JTest {
    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;

    static {
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void testGradientCNNMLN() {
        //Parameterized test, testing combinations of:
        // (a) activation function
        // (b) Whether to test at random initialization, or after some learning (i.e., 'characteristic mode of operation')
        // (c) Loss function (with specified output activations)
        Activation[] activFns = {Activation.SIGMOID, Activation.TANH};
        boolean[] characteristic = {false, true}; //If true: run some backprop steps first

        LossFunctions.LossFunction[] lossFunctions =
                {LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD, LossFunctions.LossFunction.MSE};
        Activation[] outputActivations = {Activation.SOFTMAX, Activation.TANH}; //i.e., lossFunctions[i] used with outputActivations[i] here

        DataSet ds = new IrisDataSetIterator(150, 150).next();
        ds.normalizeZeroMeanZeroUnitVariance();
        INDArray input = ds.getFeatureMatrix();
        INDArray labels = ds.getLabels();

        for (Activation afn : activFns) {
            for (boolean doLearningFirst : characteristic) {
                for (int i = 0; i < lossFunctions.length; i++) {
                    LossFunctions.LossFunction lf = lossFunctions[i];
                    Activation outputActivation = outputActivations[i];

                    MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                            .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT).updater(new NoOp())
                            .weightInit(WeightInit.XAVIER).seed(12345L).list()
                            .layer(0, new ConvolutionLayer.Builder(1, 1).nOut(6).activation(afn).build())
                            .layer(1, new OutputLayer.Builder(lf).activation(outputActivation).nOut(3).build())
                            .setInputType(InputType.convolutionalFlat(1, 4, 1)).pretrain(false).backprop(true);

                    MultiLayerConfiguration conf = builder.build();

                    MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                    mln.init();
                    String name = new Object() {
                    }.getClass().getEnclosingMethod().getName();

                    if (doLearningFirst) {
                        //Run a number of iterations of learning
                        mln.setInput(ds.getFeatures());
                        mln.setLabels(ds.getLabels());
                        mln.computeGradientAndScore();
                        double scoreBefore = mln.score();
                        for (int j = 0; j < 10; j++)
                            mln.fit(ds);
                        mln.computeGradientAndScore();
                        double scoreAfter = mln.score();
                        //Can't test in 'characteristic mode of operation' if not learning
                        String msg = name + " - score did not (sufficiently) decrease during learning - activationFn="
                                + afn + ", lossFn=" + lf + ", outputActivation=" + outputActivation
                                + ", doLearningFirst= " + doLearningFirst + " (before=" + scoreBefore
                                + ", scoreAfter=" + scoreAfter + ")";
                        assertTrue(msg, scoreAfter < 0.8 * scoreBefore);
                    }

                    if (PRINT_RESULTS) {
                        System.out.println(name + " - activationFn=" + afn + ", lossFn=" + lf + ", outputActivation="
                                + outputActivation + ", doLearningFirst=" + doLearningFirst);
                        for (int j = 0; j < mln.getnLayers(); j++)
                            System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
                    }

                    boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

                    assertTrue(gradOK);
                    TestUtils.testModelSerialization(mln);
                }
            }
        }
    }


    @Test
    public void testGradientCNNL1L2MLN() {
        //Parameterized test, testing combinations of:
        // (a) activation function
        // (b) Whether to test at random initialization, or after some learning (i.e., 'characteristic mode of operation')
        // (c) Loss function (with specified output activations)
        Activation[] activFns = {Activation.SIGMOID, Activation.TANH};
        boolean[] characteristic = {false, true}; //If true: run some backprop steps first

        LossFunctions.LossFunction[] lossFunctions =
                {LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD, LossFunctions.LossFunction.MSE};
        Activation[] outputActivations = {Activation.SOFTMAX, Activation.TANH}; //i.e., lossFunctions[i] used with outputActivations[i] here

        DataSet ds = new IrisDataSetIterator(150, 150).next();
        ds.normalizeZeroMeanZeroUnitVariance();
        INDArray input = ds.getFeatureMatrix();
        INDArray labels = ds.getLabels();

        //use l2vals[i] with l1vals[i]
        double[] l2vals = {0.4, 0.0, 0.4, 0.4};
        double[] l1vals = {0.0, 0.0, 0.5, 0.0};
        double[] biasL2 = {0.0, 0.0, 0.0, 0.2};
        double[] biasL1 = {0.0, 0.0, 0.6, 0.0};

        for (Activation afn : activFns) {
            for (boolean doLearningFirst : characteristic) {
                for (int i = 0; i < lossFunctions.length; i++) {
                    for (int k = 0; k < l2vals.length; k++) {
                        LossFunctions.LossFunction lf = lossFunctions[i];
                        Activation outputActivation = outputActivations[i];
                        double l2 = l2vals[k];
                        double l1 = l1vals[k];

                        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                                .l2(l2).l1(l1).l2Bias(biasL2[k]).l1Bias(biasL1[k])
                                .optimizationAlgo(
                                        OptimizationAlgorithm.CONJUGATE_GRADIENT)
                                .seed(12345L).list()
                                .layer(0, new ConvolutionLayer.Builder(new int[]{1, 1}).nIn(1).nOut(6)
                                        .weightInit(WeightInit.XAVIER).activation(afn)
                                        .updater(new NoOp()).build())
                                .layer(1, new OutputLayer.Builder(lf).activation(outputActivation).nOut(3)
                                        .weightInit(WeightInit.XAVIER).updater(new NoOp()).build())
                                .pretrain(false).backprop(true)
                                .setInputType(InputType.convolutionalFlat(1, 4, 1));

                        MultiLayerConfiguration conf = builder.build();

                        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                        mln.init();
                        String testName = new Object() {
                        }.getClass().getEnclosingMethod().getName();

                        if (doLearningFirst) {
                            //Run a number of iterations of learning
                            mln.setInput(ds.getFeatures());
                            mln.setLabels(ds.getLabels());
                            mln.computeGradientAndScore();
                            double scoreBefore = mln.score();
                            for (int j = 0; j < 10; j++)
                                mln.fit(ds);
                            mln.computeGradientAndScore();
                            double scoreAfter = mln.score();
                            //Can't test in 'characteristic mode of operation' if not learning
                            String msg = testName
                                    + "- score did not (sufficiently) decrease during learning - activationFn="
                                    + afn + ", lossFn=" + lf + ", outputActivation=" + outputActivation
                                    + ", doLearningFirst=" + doLearningFirst + " (before=" + scoreBefore
                                    + ", scoreAfter=" + scoreAfter + ")";
                            assertTrue(msg, scoreAfter < 0.8 * scoreBefore);
                        }

                        if (PRINT_RESULTS) {
                            System.out.println(testName + "- activationFn=" + afn + ", lossFn=" + lf
                                    + ", outputActivation=" + outputActivation + ", doLearningFirst="
                                    + doLearningFirst);
                            for (int j = 0; j < mln.getnLayers(); j++)
                                System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
                        }

                        boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                                DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

                        assertTrue(gradOK);
                        TestUtils.testModelSerialization(mln);
                    }
                }
            }
        }
    }

    @Ignore
    @Test
    public void testCnnWithSpaceToDepth() {
        Nd4j.getRandom().setSeed(12345);
        int nOut = 4;
        int minibatchSize = 2;

        int width = 5;
        int height = 5;
        int inputDepth = 1;

        int[] kernel = {2, 2};
        int blocks = 2;

        String[] activations = {"sigmoid"};
        SubsamplingLayer.PoolingType[] poolingTypes =
                new SubsamplingLayer.PoolingType[]{SubsamplingLayer.PoolingType.MAX,
                        SubsamplingLayer.PoolingType.AVG, SubsamplingLayer.PoolingType.PNORM};

        for (String afn : activations) {
            for (SubsamplingLayer.PoolingType poolingType : poolingTypes) {
                INDArray input = Nd4j.rand(minibatchSize, width * height * inputDepth);
                INDArray labels = Nd4j.zeros(minibatchSize, nOut);
                for (int i = 0; i < minibatchSize; i++) {
                    labels.putScalar(new int[]{i, i % nOut}, 1.0);
                }

                MultiLayerConfiguration conf =
                        new NeuralNetConfiguration.Builder()
                                .updater(new NoOp()).weightInit(WeightInit.DISTRIBUTION)
                                .dist(new NormalDistribution(0, 1))
                                .list().layer(new ConvolutionLayer.Builder(kernel).nIn(inputDepth).hasBias(false)
                                .nOut(1).build()) //output: (5-2+0)/1+1 = 4
                                .layer(new SpaceToDepthLayer.Builder(blocks, SpaceToDepthLayer.DataFormat.NCHW)
                                        .build()) // (mb,1,4,4) -> (mb,4,2,2)
                                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .activation(Activation.SOFTMAX).nIn(2 * 2 * 4)
                                        .nOut(nOut).build())
                                .setInputType(InputType.convolutionalFlat(height, width, inputDepth))
                                .build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                String msg = "PoolingType=" + poolingType + ", minibatch=" + minibatchSize + ", activationFn="
                        + afn;

                if (PRINT_RESULTS) {
                    System.out.println(msg);
                    for (int j = 0; j < net.getnLayers(); j++)
                        System.out.println("Layer " + j + " # params: " + net.getLayer(j).numParams());
                }
                boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

                assertTrue(msg, gradOK);

                TestUtils.testModelSerialization(net);
            }
        }
    }

    @Test
    public void testCnnWithSpaceToBatch() {
        Nd4j.getRandom().setSeed(12345);
        int nOut = 4;

        int[] minibatchSizes = {2, 4};
        int width = 5;
        int height = 5;
        int inputDepth = 1;

        int[] kernel = {2, 2};
        int[] blocks = {1, 1};

        String[] activations = {"sigmoid", "tanh"};
        SubsamplingLayer.PoolingType[] poolingTypes =
                new SubsamplingLayer.PoolingType[]{SubsamplingLayer.PoolingType.MAX,
                        SubsamplingLayer.PoolingType.AVG, SubsamplingLayer.PoolingType.PNORM};

        for (String afn : activations) {
            for (SubsamplingLayer.PoolingType poolingType : poolingTypes) {
                for (int minibatchSize : minibatchSizes) {
                    INDArray input = Nd4j.rand(minibatchSize, width * height * inputDepth);
                    INDArray labels = Nd4j.zeros(minibatchSize, nOut);
                    for (int i = 0; i < minibatchSize; i++) {
                        labels.putScalar(new int[]{i, i % nOut}, 1.0);
                    }

                    MultiLayerConfiguration conf =
                            new NeuralNetConfiguration.Builder()
                                    .updater(new NoOp()).weightInit(WeightInit.DISTRIBUTION)
                                    .dist(new NormalDistribution(0, 1))
                                    .list().layer(new ConvolutionLayer.Builder(kernel).nIn(inputDepth)
                                    .nOut(3).build())//output: (5-2+0)/1+1 = 4
                                    .layer(new SpaceToBatchLayer.Builder(blocks).build()) //trivial space to batch
                                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                            .activation(Activation.SOFTMAX).nIn(4 * 4 * 3)
                                            .nOut(nOut).build())
                                    .setInputType(InputType.convolutionalFlat(height, width, inputDepth))
                                    .build();

                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();

                    String msg = "PoolingType=" + poolingType + ", minibatch=" + minibatchSize + ", activationFn="
                            + afn;

                    if (PRINT_RESULTS) {
                        System.out.println(msg);
                        for (int j = 0; j < net.getnLayers(); j++)
                            System.out.println("Layer " + j + " # params: " + net.getLayer(j).numParams());
                    }
                    boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

                    assertTrue(msg, gradOK);

                    TestUtils.testModelSerialization(net);
                }
            }
        }
    }


    @Test
    public void testCnnWithUpsampling() {
        Nd4j.getRandom().setSeed(12345);
        int nOut = 4;

        int[] minibatchSizes = {1, 3};
        int width = 5;
        int height = 5;
        int inputDepth = 1;

        int[] kernel = {2, 2};
        int[] stride = {1, 1};
        int[] padding = {0, 0};
        int size = 2;

        String[] activations = {"sigmoid", "tanh"};
        SubsamplingLayer.PoolingType[] poolingTypes =
                new SubsamplingLayer.PoolingType[]{SubsamplingLayer.PoolingType.MAX,
                        SubsamplingLayer.PoolingType.AVG, SubsamplingLayer.PoolingType.PNORM};

        for (String afn : activations) {
            for (SubsamplingLayer.PoolingType poolingType : poolingTypes) {
                for (int minibatchSize : minibatchSizes) {
                    INDArray input = Nd4j.rand(minibatchSize, width * height * inputDepth);
                    INDArray labels = Nd4j.zeros(minibatchSize, nOut);
                    for (int i = 0; i < minibatchSize; i++) {
                        labels.putScalar(new int[]{i, i % nOut}, 1.0);
                    }

                    MultiLayerConfiguration conf =
                            new NeuralNetConfiguration.Builder()
                                    .updater(new NoOp()).weightInit(WeightInit.DISTRIBUTION)
                                    .dist(new NormalDistribution(0, 1))
                                    .list().layer(new ConvolutionLayer.Builder(kernel,
                                    stride, padding).nIn(inputDepth)
                                    .nOut(3).build())//output: (5-2+0)/1+1 = 4
                                    .layer(new Upsampling2D.Builder().size(size).build()) //output: 4*2 =8 -> 8x8x3
                                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                            .activation(Activation.SOFTMAX).nIn(8 * 8 * 3)
                                            .nOut(4).build())
                                    .setInputType(InputType.convolutionalFlat(height, width,
                                            inputDepth))
                                    .build();

                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();

                    String msg = "PoolingType=" + poolingType + ", minibatch=" + minibatchSize + ", activationFn="
                            + afn;

                    if (PRINT_RESULTS) {
                        System.out.println(msg);
                        for (int j = 0; j < net.getnLayers(); j++)
                            System.out.println("Layer " + j + " # params: " + net.getLayer(j).numParams());
                    }

                    boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

                    assertTrue(msg, gradOK);

                    TestUtils.testModelSerialization(net);
                }
            }
        }
    }


    @Test
    public void testCnnWithSubsampling() {
        Nd4j.getRandom().setSeed(12345);
        int nOut = 4;

        int[] minibatchSizes = {1, 3};
        int width = 5;
        int height = 5;
        int inputDepth = 1;

        int[] kernel = {2, 2};
        int[] stride = {1, 1};
        int[] padding = {0, 0};
        int pnorm = 2;

        Activation[] activations = {Activation.SIGMOID, Activation.TANH};
        SubsamplingLayer.PoolingType[] poolingTypes =
                new SubsamplingLayer.PoolingType[]{SubsamplingLayer.PoolingType.MAX,
                        SubsamplingLayer.PoolingType.AVG, SubsamplingLayer.PoolingType.PNORM};

        for (Activation afn : activations) {
            for (SubsamplingLayer.PoolingType poolingType : poolingTypes) {
                for (int minibatchSize : minibatchSizes) {
                    INDArray input = Nd4j.rand(minibatchSize, width * height * inputDepth);
                    INDArray labels = Nd4j.zeros(minibatchSize, nOut);
                    for (int i = 0; i < minibatchSize; i++) {
                        labels.putScalar(new int[]{i, i % nOut}, 1.0);
                    }

                    MultiLayerConfiguration conf =
                            new NeuralNetConfiguration.Builder().updater(new NoOp())
                                    .weightInit(WeightInit.DISTRIBUTION)
                                    .dist(new NormalDistribution(0, 1))
                                    .list().layer(0,
                                    new ConvolutionLayer.Builder(kernel,
                                            stride, padding).nIn(inputDepth)
                                            .nOut(3).build())//output: (5-2+0)/1+1 = 4
                                    .layer(1, new SubsamplingLayer.Builder(poolingType)
                                            .kernelSize(kernel).stride(stride).padding(padding)
                                            .pnorm(pnorm).build()) //output: (4-2+0)/1+1 =3 -> 3x3x3
                                    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                            .activation(Activation.SOFTMAX).nIn(3 * 3 * 3)
                                            .nOut(4).build())
                                    .setInputType(InputType.convolutionalFlat(height, width,
                                            inputDepth))
                                    .build();

                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();

                    String msg = "PoolingType=" + poolingType + ", minibatch=" + minibatchSize + ", activationFn="
                            + afn;

                    if (PRINT_RESULTS) {
                        System.out.println(msg);
                        for (int j = 0; j < net.getnLayers(); j++)
                            System.out.println("Layer " + j + " # params: " + net.getLayer(j).numParams());
                    }

                    boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

                    assertTrue(msg, gradOK);

                    TestUtils.testModelSerialization(net);
                }
            }
        }
    }

    @Test
    public void testCnnWithSubsamplingV2() {
        Nd4j.getRandom().setSeed(12345);
        int nOut = 4;

        int[] minibatchSizes = {1, 3};
        int width = 5;
        int height = 5;
        int inputDepth = 1;

        int[] kernel = {2, 2};
        int[] stride = {1, 1};
        int[] padding = {0, 0};
        int pNorm = 3;

        Activation[] activations = {Activation.SIGMOID, Activation.TANH};
        SubsamplingLayer.PoolingType[] poolingTypes =
                new SubsamplingLayer.PoolingType[]{SubsamplingLayer.PoolingType.MAX,
                        SubsamplingLayer.PoolingType.AVG, SubsamplingLayer.PoolingType.PNORM};

        for (Activation afn : activations) {
            for (SubsamplingLayer.PoolingType poolingType : poolingTypes) {
                for (int minibatchSize : minibatchSizes) {
                    INDArray input = Nd4j.rand(minibatchSize, width * height * inputDepth);
                    INDArray labels = Nd4j.zeros(minibatchSize, nOut);
                    for (int i = 0; i < minibatchSize; i++) {
                        labels.putScalar(new int[]{i, i % nOut}, 1.0);
                    }

                    MultiLayerConfiguration conf =
                            new NeuralNetConfiguration.Builder().updater(new NoOp()).weightInit(WeightInit.DISTRIBUTION)
                                    .dist(new NormalDistribution(0, 1))
                                    .list().layer(0,
                                    new ConvolutionLayer.Builder(kernel,
                                            stride, padding).nIn(inputDepth)
                                            .nOut(3).build())//output: (5-2+0)/1+1 = 4
                                    .layer(1, new SubsamplingLayer.Builder(poolingType)
                                            .kernelSize(kernel).stride(stride).padding(padding)
                                            .pnorm(pNorm).build()) //output: (4-2+0)/1+1 =3 -> 3x3x3
                                    .layer(2, new ConvolutionLayer.Builder(kernel, stride, padding)
                                            .nIn(3).nOut(2).build()) //Output: (3-2+0)/1+1 = 2
                                    .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                            .activation(Activation.SOFTMAX).nIn(2 * 2 * 2)
                                            .nOut(4).build())
                                    .setInputType(InputType.convolutionalFlat(height, width,
                                            inputDepth))
                                    .build();

                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();

                    String msg = "PoolingType=" + poolingType + ", minibatch=" + minibatchSize + ", activationFn="
                            + afn;
                    System.out.println(msg);

                    boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

                    assertTrue(msg, gradOK);

                    TestUtils.testModelSerialization(net);
                }
            }
        }
    }

    @Test
    public void testCnnLocallyConnected2D() {
        int nOut = 3;

        int[] minibatchSizes = {2};
        int width = 5;
        int height = 5;
        int[] inputDepths = {1, 2, 4};

        Activation[] activations = {Activation.SIGMOID, Activation.TANH};
        SubsamplingLayer.PoolingType[] poolingTypes = new SubsamplingLayer.PoolingType[]{
                SubsamplingLayer.PoolingType.MAX, SubsamplingLayer.PoolingType.AVG};

        Nd4j.getRandom().setSeed(12345);

        for (int inputDepth : inputDepths) {
            for (Activation afn : activations) {
                for (SubsamplingLayer.PoolingType poolingType : poolingTypes) {
                    for (int minibatchSize : minibatchSizes) {
                        INDArray input = Nd4j.rand(minibatchSize, width * height * inputDepth);
                        INDArray labels = Nd4j.zeros(minibatchSize, nOut);
                        for (int i = 0; i < minibatchSize; i++) {
                            labels.putScalar(new int[]{i, i % nOut}, 1.0);
                        }

                        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).updater(new NoOp())
                                .activation(afn)
                                .list()
                                .layer(0, new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1)
                                        .padding(0, 0).nIn(inputDepth).nOut(2).build())//output: (5-2+0)/1+1 = 4
                                .layer(1, new SDLocallyConnected2D.Builder().nIn(2).nOut(7).kernelSize(2, 2)
                                        .setInputSize(4, 4).convolutionMode(ConvolutionMode.Strict).hasBias(false)
                                        .stride(1, 1).padding(0, 0).build()) //(4-2+0)/1+1 = 3
                                .layer(2, new ConvolutionLayer.Builder().nIn(7).nOut(2).kernelSize(2, 2)
                                        .stride(1, 1).padding(0, 0).build()) //(3-2+0)/1+1 = 2
                                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .activation(Activation.SOFTMAX).nIn(2 * 2 * 2).nOut(nOut)
                                        .build())
                                .setInputType(InputType.convolutionalFlat(height, width, inputDepth)).build();

                        assertEquals(ConvolutionMode.Truncate,
                                ((ConvolutionLayer) conf.getConf(0).getLayer()).getConvolutionMode());

                        MultiLayerNetwork net = new MultiLayerNetwork(conf);
                        net.init();

                        for (int i = 0; i < 4; i++) {
                            System.out.println("nParams, layer " + i + ": " + net.getLayer(i).numParams());
                        }

                        String msg = "PoolingType=" + poolingType + ", minibatch=" + minibatchSize + ", activationFn="
                                + afn;
                        System.out.println(msg);

                        boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                                DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

                        assertTrue(msg, gradOK);

                        TestUtils.testModelSerialization(net);
                    }
                }
            }
        }
    }

    @Test
    public void testCnnMultiLayer() {
        int nOut = 2;

        int[] minibatchSizes = {1, 2, 5};
        int width = 5;
        int height = 5;
        int[] inputDepths = {1, 2, 4};

        Activation[] activations = {Activation.SIGMOID, Activation.TANH};
        SubsamplingLayer.PoolingType[] poolingTypes = new SubsamplingLayer.PoolingType[]{
                SubsamplingLayer.PoolingType.MAX, SubsamplingLayer.PoolingType.AVG};

        Nd4j.getRandom().setSeed(12345);

        for (int inputDepth : inputDepths) {
            for (Activation afn : activations) {
                for (SubsamplingLayer.PoolingType poolingType : poolingTypes) {
                    for (int minibatchSize : minibatchSizes) {
                        INDArray input = Nd4j.rand(minibatchSize, width * height * inputDepth);
                        INDArray labels = Nd4j.zeros(minibatchSize, nOut);
                        for (int i = 0; i < minibatchSize; i++) {
                            labels.putScalar(new int[]{i, i % nOut}, 1.0);
                        }

                        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).updater(new NoOp())
                                .activation(afn)
                                .list()
                                .layer(0, new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1)
                                        .padding(0, 0).nIn(inputDepth).nOut(2).build())//output: (5-2+0)/1+1 = 4
                                .layer(1, new ConvolutionLayer.Builder().nIn(2).nOut(2).kernelSize(2, 2)
                                        .stride(1, 1).padding(0, 0).build()) //(4-2+0)/1+1 = 3
                                .layer(2, new ConvolutionLayer.Builder().nIn(2).nOut(2).kernelSize(2, 2)
                                        .stride(1, 1).padding(0, 0).build()) //(3-2+0)/1+1 = 2
                                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .activation(Activation.SOFTMAX).nIn(2 * 2 * 2).nOut(nOut)
                                        .build())
                                .setInputType(InputType.convolutionalFlat(height, width, inputDepth)).build();

                        assertEquals(ConvolutionMode.Truncate,
                                ((ConvolutionLayer) conf.getConf(0).getLayer()).getConvolutionMode());

                        MultiLayerNetwork net = new MultiLayerNetwork(conf);
                        net.init();

                        for (int i = 0; i < 4; i++) {
                            System.out.println("nParams, layer " + i + ": " + net.getLayer(i).numParams());
                        }

                        String msg = "PoolingType=" + poolingType + ", minibatch=" + minibatchSize + ", activationFn="
                                + afn;
                        System.out.println(msg);

                        boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                                DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

                        assertTrue(msg, gradOK);

                        TestUtils.testModelSerialization(net);
                    }
                }
            }
        }
    }


    @Test
    public void testCnnSamePaddingMode() {
        int nOut = 2;

        int[] minibatchSizes = {1, 3};
        int width = 5;
        int[] heights = new int[]{4, 5, 6}; //Same padding mode: insensitive to exact input size...
        int[] kernelSizes = new int[]{2, 3};
        int[] inputDepths = {1, 2, 4};

        Nd4j.getRandom().setSeed(12345);

        for (int inputDepth : inputDepths) {
            for (int minibatchSize : minibatchSizes) {
                for (int height : heights) {
                    for (int k : kernelSizes) {

                        INDArray input = Nd4j.rand(minibatchSize, width * height * inputDepth);
                        INDArray labels = Nd4j.zeros(minibatchSize, nOut);
                        for (int i = 0; i < minibatchSize; i++) {
                            labels.putScalar(new int[]{i, i % nOut}, 1.0);
                        }

                        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                                .updater(new NoOp())
                                .activation(Activation.TANH).convolutionMode(Same).list()
                                .layer(0, new ConvolutionLayer.Builder().name("layer 0").kernelSize(k, k)
                                        .stride(1, 1).padding(0, 0).nIn(inputDepth).nOut(2).build())
                                .layer(1, new SubsamplingLayer.Builder()
                                        .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(k, k)
                                        .stride(1, 1).padding(0, 0).build())
                                .layer(2, new ConvolutionLayer.Builder().nIn(2).nOut(2).kernelSize(k, k)
                                        .stride(1, 1).padding(0, 0).build())
                                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .activation(Activation.SOFTMAX).nOut(nOut).build())
                                .setInputType(InputType.convolutionalFlat(height, width, inputDepth)).build();

                        MultiLayerNetwork net = new MultiLayerNetwork(conf);
                        net.init();

                        for (int i = 0; i < net.getLayers().length; i++) {
                            System.out.println("nParams, layer " + i + ": " + net.getLayer(i).numParams());
                        }

                        String msg = "Minibatch=" + minibatchSize + ", inDepth=" + inputDepth + ", height=" + height
                                + ", kernelSize=" + k;
                        System.out.println(msg);

                        boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                                DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

                        assertTrue(msg, gradOK);

                        TestUtils.testModelSerialization(net);
                    }
                }
            }
        }
    }

    @Test
    public void testCnnSamePaddingModeStrided() {
        int nOut = 2;

        int[] minibatchSizes = {1, 3};
        int width = 16;
        int height = 16;
        int[] kernelSizes = new int[]{2, 3};
        int[] strides = {1, 2, 3};
        int[] inputDepths = {1, 3};

        Nd4j.getRandom().setSeed(12345);

        for (int inputDepth : inputDepths) {
            for (int minibatchSize : minibatchSizes) {
                for (int stride : strides) {
                    for (int k : kernelSizes) {
                        for (boolean convFirst : new boolean[]{true, false}) {

                            INDArray input = Nd4j.rand(minibatchSize, width * height * inputDepth);
                            INDArray labels = Nd4j.zeros(minibatchSize, nOut);
                            for (int i = 0; i < minibatchSize; i++) {
                                labels.putScalar(new int[]{i, i % nOut}, 1.0);
                            }

                            Layer convLayer = new ConvolutionLayer.Builder().name("layer 0").kernelSize(k, k)
                                    .stride(stride, stride).padding(0, 0).nIn(inputDepth).nOut(2).build();

                            Layer poolLayer = new SubsamplingLayer.Builder()
                                    .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(k, k)
                                    .stride(stride, stride).padding(0, 0).build();

                            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                                    .updater(new NoOp())
                                    .activation(Activation.TANH).convolutionMode(Same).list()
                                    .layer(0, convFirst ? convLayer : poolLayer)
                                    .layer(1, convFirst ? poolLayer : convLayer)
                                    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                            .activation(Activation.SOFTMAX).nOut(nOut).build())
                                    .setInputType(InputType.convolutionalFlat(height, width, inputDepth))
                                    .build();

                            MultiLayerNetwork net = new MultiLayerNetwork(conf);
                            net.init();

                            for (int i = 0; i < net.getLayers().length; i++) {
                                System.out.println("nParams, layer " + i + ": " + net.getLayer(i).numParams());
                            }

                            String msg = "Minibatch=" + minibatchSize + ", inDepth=" + inputDepth + ", height=" + height
                                    + ", kernelSize=" + k + ", stride = " + stride + ", convLayer first = "
                                    + convFirst;
                            System.out.println(msg);

                            boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                                    DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input,
                                    labels);

                            assertTrue(msg, gradOK);

                            TestUtils.testModelSerialization(net);
                        }
                    }
                }
            }
        }
    }


    @Test
    public void testCnnZeroPaddingLayer() {
        Nd4j.getRandom().setSeed(12345);
        int nOut = 4;

        int[] minibatchSizes = {1, 3};
        int width = 6;
        int height = 6;
        int[] inputDepths = {1, 3};

        int[] kernel = {2, 2};
        int[] stride = {1, 1};
        int[] padding = {0, 0};

        int[][] zeroPadLayer = new int[][]{{0, 0, 0, 0}, {1, 1, 0, 0}, {2, 2, 2, 2}};

        for (int inputDepth : inputDepths) {
            for (int minibatchSize : minibatchSizes) {
                INDArray input = Nd4j.rand(new int[]{minibatchSize, inputDepth, height, width});
                INDArray labels = Nd4j.zeros(minibatchSize, nOut);
                for (int i = 0; i < minibatchSize; i++) {
                    labels.putScalar(new int[]{i, i % nOut}, 1.0);
                }
                for (int[] zeroPad : zeroPadLayer) {

                    MultiLayerConfiguration conf =
                            new NeuralNetConfiguration.Builder().updater(new NoOp()).weightInit(WeightInit.DISTRIBUTION)
                                    .dist(new NormalDistribution(0, 1)).list()
                                    .layer(0, new ConvolutionLayer.Builder(kernel, stride, padding)
                                            .nIn(inputDepth).nOut(3).build())//output: (6-2+0)/1+1 = 5
                                    .layer(1, new ZeroPaddingLayer.Builder(zeroPad).build()).layer(2,
                                    new ConvolutionLayer.Builder(kernel, stride,
                                            padding).nIn(3).nOut(3).build())//output: (6-2+0)/1+1 = 5
                                    .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                            .activation(Activation.SOFTMAX).nOut(4).build())
                                    .setInputType(InputType.convolutional(height, width, inputDepth))
                                    .build();

                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();

                    //Check zero padding activation shape
                    org.deeplearning4j.nn.layers.convolution.ZeroPaddingLayer zpl =
                            (org.deeplearning4j.nn.layers.convolution.ZeroPaddingLayer) net.getLayer(1);
                    val expShape = new long[]{minibatchSize, inputDepth, height + zeroPad[0] + zeroPad[1],
                            width + zeroPad[2] + zeroPad[3]};
                    INDArray out = zpl.activate(input, false, LayerWorkspaceMgr.noWorkspaces());
                    assertArrayEquals(expShape, out.shape());

                    String msg = "minibatch=" + minibatchSize + ", channels=" + inputDepth + ", zeroPad = "
                            + Arrays.toString(zeroPad);

                    if (PRINT_RESULTS) {
                        System.out.println(msg);
                        for (int j = 0; j < net.getnLayers(); j++)
                            System.out.println("Layer " + j + " # params: " + net.getLayer(j).numParams());
                    }

                    boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

                    assertTrue(msg, gradOK);

                    TestUtils.testModelSerialization(net);
                }
            }
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void testDeconvolution2DUnsupportedSameModeLayer() {
        /*
         * When setting border mode same on layer directly, we can catch
         * it in the builder
         */
        Deconvolution2D.Builder deconv = new Deconvolution2D.Builder();
        deconv.convolutionMode(Same);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testDeconvolution2DUnsupportedSameModeNetwork() {
        /*
         * When convolution mode Same is set for the network and a deconvolution layer is added
         * then only layer activation will fail. Suboptimal, but I don't think we want special
         * logic for NNC in this case.
         */
        NeuralNetConfiguration.ListBuilder b = new NeuralNetConfiguration.Builder().seed(12345)
                .updater(new NoOp())
                .activation(Activation.SIGMOID)
                .convolutionMode(Same)
                .list()
                .layer(new Deconvolution2D.Builder().name("deconvolution")
                        .nIn(3).nOut(2).build());

        MultiLayerConfiguration conf = b.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX).nOut(2).build())
                .setInputType(InputType.convolutionalFlat(7, 7, 3)).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.getLayer(0).activate(Nd4j.rand(10, 7 * 7 * 3), false, LayerWorkspaceMgr.noWorkspaces());
    }

    @Test
    public void testDeconvolution2D() {
        int nOut = 2;

        int[] minibatchSizes = new int[]{1, 3, 1, 3, 1, 3, 1, 3};
        int[] kernelSizes = new int[]{1, 1, 3, 3, 1, 1, 3, 3};
        int[] strides = {1, 1, 1, 1, 2, 2, 2, 2};
        int[] dilation = {1, 2, 2, 1, 1, 1, 2, 2};
        Activation[] activations = new Activation[]{Activation.SIGMOID, Activation.TANH, Activation.TANH, Activation.TANH, Activation.TANH,  Activation.SIGMOID, Activation.SIGMOID, Activation.SIGMOID};
        ConvolutionMode[] cModes = new ConvolutionMode[]{Truncate, Truncate, Truncate, Truncate, Truncate, Truncate, Truncate, Truncate};
        int width = 7;
        int height = 7;
        int inputDepth = 3;

        Nd4j.getRandom().setSeed(12345);

        for (int i = 0; i < minibatchSizes.length; i++) {
            int minibatchSize = minibatchSizes[i];
            int k = kernelSizes[i];
            int s = strides[i];
            int d = dilation[i];
            ConvolutionMode cm = cModes[i];
            Activation act = activations[i];


            int w = d * width;
            int h = d * height;

            INDArray input = Nd4j.rand(minibatchSize, w * h * inputDepth);
            INDArray labels = Nd4j.zeros(minibatchSize, nOut);
            for (int j = 0; j < minibatchSize; j++) {
                labels.putScalar(new int[]{j, j % nOut}, 1.0);
            }

            NeuralNetConfiguration.ListBuilder b = new NeuralNetConfiguration.Builder().seed(12345)
                    .updater(new NoOp())
                    .activation(act)
                    .list()
                    .layer(new Deconvolution2D.Builder().name("deconvolution_2D_layer")
                            .kernelSize(k, k)
                            .stride(s, s)
                            .dilation(d, d)
                            .convolutionMode(cm)
                            .nIn(inputDepth).nOut(nOut).build());

            MultiLayerConfiguration conf = b.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                    .activation(Activation.SOFTMAX).nOut(nOut).build())
                    .setInputType(InputType.convolutionalFlat(h, w, inputDepth)).build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            for (int j = 0; j < net.getLayers().length; j++) {
                System.out.println("nParams, layer " + j + ": " + net.getLayer(j).numParams());
            }

            String msg = " - mb=" + minibatchSize + ", k="
                    + k + ", s=" + s + ", d=" + d + ", cm=" + cm;
            System.out.println(msg);

            boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                    DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

            assertTrue(msg, gradOK);

            TestUtils.testModelSerialization(net);
        }
    }

    @Test
    public void testSeparableConv2D() {
        int nOut = 2;

        int[] minibatchSizes = new int[] {1, 3};
        int width = 8;
        int height = 8;
        int inputDepth = 3;
        int[] kernelSizes = new int[]{2, 3};
        int[] strides = {1, 2};
        int[] dilation = {1, 2};
        ConvolutionMode[] cModes = new ConvolutionMode[]{ConvolutionMode.Truncate};

        Nd4j.getRandom().setSeed(12345);

        int[] ks = new int[]{1,3,1,3,1,3,1,3};
        int[] ss = new int[]{1,1,2,2,1,1,2,2};
        int[] ds = new int[]{1,1,1,1,2,2,2,2};
        ConvolutionMode[] cms = new ConvolutionMode[]{Truncate, Truncate, Truncate, Truncate, Truncate, Truncate, Truncate, Truncate};
        int[] mb = new int[]{1,1,3,3,3,1,3,3};

        for( int t=0; t<ks.length; t++ ){

            int k = ks[t];
            int s = ss[t];
            int d = ds[t];
            ConvolutionMode cm = cms[t];
            int minibatchSize = mb[t];

            //Use larger input with larger dilation values (to avoid invalid config)
            int w = d * width;
            int h = d * height;

            INDArray input = Nd4j.rand(minibatchSize, w * h * inputDepth);
            INDArray labels = Nd4j.zeros(minibatchSize, nOut);
            for (int i = 0; i < minibatchSize; i++) {
                labels.putScalar(new int[]{i, i % nOut}, 1.0);
            }

            NeuralNetConfiguration.ListBuilder b = new NeuralNetConfiguration.Builder().seed(12345)
                    .updater(new NoOp())
                    .activation(Activation.TANH)
                    .convolutionMode(cm)
                    .list()
                    .layer(new SeparableConvolution2D.Builder().name("Separable conv 2D layer")
                            .kernelSize(k, k)
                            .stride(s, s)
                            .dilation(d, d)
                            .depthMultiplier(3)
                            .nIn(inputDepth).nOut(2).build());

            MultiLayerConfiguration conf = b.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                    .activation(Activation.SOFTMAX).nOut(nOut).build())
                    .setInputType(InputType.convolutionalFlat(h, w, inputDepth)).build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            for (int i = 0; i < net.getLayers().length; i++) {
                System.out.println("nParams, layer " + i + ": " + net.getLayer(i).numParams());
            }

            String msg = " - mb=" + minibatchSize + ", k="
                    + k + ", s=" + s + ", d=" + d + ", cm=" + cm;
            System.out.println(msg);

            boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                    DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

            assertTrue(msg, gradOK);

            TestUtils.testModelSerialization(net);
        }
    }

    @Test
    public void testCnnDilated() {
        int nOut = 2;

        int minibatchSize = 3;
        int width = 8;
        int height = 8;
        int inputDepth = 3;

        Nd4j.getRandom().setSeed(12345);

        boolean[] sub = new boolean[]{true,false,true,false,true,false,true,false};
        int[] stride = new int[]{1,1,2,2,1,1,2,2};
        int[] kernel = new int[]{2,2,2,2,3,3,3,3};
        int[] ds = new int[]{2,3,3,2,2,3,3,2};
        ConvolutionMode[] cms = new ConvolutionMode[]{Same, Same, Same, Truncate, Truncate, Truncate, Same, Truncate};


        for(int t=0; t<sub.length; t++ ){

            boolean subsampling = sub[t];
            int s = stride[t];
            int k = kernel[t];
            int d = ds[t];
            ConvolutionMode cm = cms[t];

            //Use larger input with larger dilation values (to avoid invalid config)
            int w = d * width;
            int h = d * height;

            INDArray input = Nd4j.rand(minibatchSize, w * h * inputDepth);
            INDArray labels = Nd4j.zeros(minibatchSize, nOut);
            for (int i = 0; i < minibatchSize; i++) {
                labels.putScalar(new int[]{i, i % nOut}, 1.0);
            }

            NeuralNetConfiguration.ListBuilder b = new NeuralNetConfiguration.Builder().seed(12345)
                    .updater(new NoOp())
                    .activation(Activation.TANH).convolutionMode(cm).list()
                    .layer(new ConvolutionLayer.Builder().name("layer 0")
                            .kernelSize(k, k)
                            .stride(s, s)
                            .dilation(d, d)
                            .nIn(inputDepth).nOut(2).build());
            if (subsampling) {
                b.layer(new SubsamplingLayer.Builder()
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(k, k)
                        .stride(s, s)
                        .dilation(d, d)
                        .build());
            } else {
                b.layer(new ConvolutionLayer.Builder().nIn(2).nOut(2)
                        .kernelSize(k, k)
                        .stride(s, s)
                        .dilation(d, d)
                        .build());
            }

            MultiLayerConfiguration conf = b.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                    .activation(Activation.SOFTMAX).nOut(nOut).build())
                    .setInputType(InputType.convolutionalFlat(h, w, inputDepth)).build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            for (int i = 0; i < net.getLayers().length; i++) {
                System.out.println("nParams, layer " + i + ": " + net.getLayer(i).numParams());
            }

            String msg = (subsampling ? "subsampling" : "conv") + " - mb=" + minibatchSize + ", k="
                    + k + ", s=" + s + ", d=" + d + ", cm=" + cm;
            System.out.println(msg);

            boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                    DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

            assertTrue(msg, gradOK);

            TestUtils.testModelSerialization(net);
        }
    }


    @Test
    public void testCropping2DLayer() {
        Nd4j.getRandom().setSeed(12345);
        int nOut = 4;

        int[] minibatchSizes = {1, 3};
        int width = 12;
        int height = 11;
        int[] inputDepths = {1, 3};

        int[] kernel = {2, 2};
        int[] stride = {1, 1};
        int[] padding = {0, 0};

        int[][] cropTestCases = new int[][]{{0, 0, 0, 0}, {1, 1, 0, 0}, {2, 2, 2, 2}, {1,2,3,4}};

        for (int inputDepth : inputDepths) {
            for (int minibatchSize : minibatchSizes) {
                INDArray input = Nd4j.rand(new int[]{minibatchSize, inputDepth, height, width});
                INDArray labels = Nd4j.zeros(minibatchSize, nOut);
                for (int i = 0; i < minibatchSize; i++) {
                    labels.putScalar(new int[]{i, i % nOut}, 1.0);
                }
                for (int[] crop : cropTestCases) {

                    MultiLayerConfiguration conf =
                            new NeuralNetConfiguration.Builder()
                                    .updater(new NoOp())
                                    .convolutionMode(ConvolutionMode.Same)
                                    .weightInit(new NormalDistribution(0, 1)).list()
                                    .layer(new ConvolutionLayer.Builder(kernel, stride, padding)
                                            .nIn(inputDepth).nOut(3).build())//output: (6-2+0)/1+1 = 5
                                    .layer(new Cropping2D(crop))
                                    .layer(new ConvolutionLayer.Builder(kernel, stride,padding).nIn(3).nOut(3).build())
                                    .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                            .activation(Activation.SOFTMAX).nOut(4).build())
                                    .setInputType(InputType.convolutional(height, width, inputDepth))
                                    .build();

                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();

                    //Check cropping activation shape
                    org.deeplearning4j.nn.layers.convolution.Cropping2DLayer cl =
                            (org.deeplearning4j.nn.layers.convolution.Cropping2DLayer) net.getLayer(1);
                    val expShape = new long[]{minibatchSize, inputDepth, height - crop[0] - crop[1],
                            width - crop[2] - crop[3]};
                    INDArray out = cl.activate(input, false, LayerWorkspaceMgr.noWorkspaces());
                    assertArrayEquals(expShape, out.shape());

                    String msg = "minibatch=" + minibatchSize + ", channels=" + inputDepth + ", zeroPad = "
                            + Arrays.toString(crop);

                    if (PRINT_RESULTS) {
                        System.out.println(msg);
                        for (int j = 0; j < net.getnLayers(); j++)
                            System.out.println("Layer " + j + " # params: " + net.getLayer(j).numParams());
                    }

                    boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

                    assertTrue(msg, gradOK);

                    TestUtils.testModelSerialization(net);
                }
            }
        }
    }
}
