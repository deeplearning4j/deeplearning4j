package org.deeplearning4j.gradientcheck;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Random;

import static org.junit.Assert.assertTrue;

/**
 *
 */
public class BNGradientCheckTest extends BaseDL4JTest {
    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-5;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-5;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-9;

    static {
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void testGradient2dSimple() {
        DataNormalization scaler = new NormalizerMinMaxScaler();
        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        scaler.fit(iter);
        iter.setPreProcessor(scaler);
        DataSet ds = iter.next();
        INDArray input = ds.getFeatureMatrix();
        INDArray labels = ds.getLabels();

        MultiLayerConfiguration.Builder builder =
                        new NeuralNetConfiguration.Builder().updater(new NoOp())
                                .seed(12345L).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 1)).list()
                                        .layer(0, new DenseLayer.Builder().nIn(4).nOut(3)
                                                        .activation(Activation.IDENTITY).build())
                                        .layer(1, new BatchNormalization.Builder().nOut(3).build())
                                        .layer(2, new ActivationLayer.Builder().activation(Activation.TANH).build())
                                        .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                                        .activation(Activation.SOFTMAX).nIn(3).nOut(3).build())
                                        .pretrain(false).backprop(true);

        MultiLayerNetwork mln = new MultiLayerNetwork(builder.build());
        mln.init();

        if (PRINT_RESULTS) {
            for (int j = 0; j < mln.getnLayers(); j++)
                System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

        assertTrue(gradOK);
        TestUtils.testModelSerialization(mln);
    }

    @Test
    public void testGradientCnnSimple() {
        Nd4j.getRandom().setSeed(12345);
        int minibatch = 10;
        int depth = 1;
        int hw = 4;
        int nOut = 4;
        INDArray input = Nd4j.rand(new int[] {minibatch, depth, hw, hw});
        INDArray labels = Nd4j.zeros(minibatch, nOut);
        Random r = new Random(12345);
        for (int i = 0; i < minibatch; i++) {
            labels.putScalar(i, r.nextInt(nOut), 1.0);
        }

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                        .updater(new NoOp()).seed(12345L).weightInit(WeightInit.DISTRIBUTION)
                        .dist(new NormalDistribution(0, 2)).list()
                        .layer(0, new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1).nIn(depth).nOut(2)
                                        .activation(Activation.IDENTITY).build())
                        .layer(1, new BatchNormalization.Builder().build())
                        .layer(2, new ActivationLayer.Builder().activation(Activation.TANH).build())
                        .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .activation(Activation.SOFTMAX).nOut(nOut).build())
                        .setInputType(InputType.convolutional(hw, hw, depth)).pretrain(false).backprop(true);

        MultiLayerNetwork mln = new MultiLayerNetwork(builder.build());
        mln.init();

        if (PRINT_RESULTS) {
            for (int j = 0; j < mln.getnLayers(); j++)
                System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

        assertTrue(gradOK);
    }

    @Test
    public void testGradientBNWithCNNandSubsampling() {
        //Parameterized test, testing combinations of:
        // (a) activation function
        // (b) Whether to test at random initialization, or after some learning (i.e., 'characteristic mode of operation')
        // (c) Loss function (with specified output activations)
        // (d) l1 and l2 values
        Activation[] activFns = {Activation.SIGMOID, Activation.TANH, Activation.IDENTITY};
        boolean[] characteristic = {false, true}; //If true: run some backprop steps first

        LossFunctions.LossFunction[] lossFunctions =
                        {LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD, LossFunctions.LossFunction.MSE};
        Activation[] outputActivations = {Activation.SOFTMAX, Activation.TANH}; //i.e., lossFunctions[i] used with outputActivations[i] here

        double[] l2vals = {0.0, 0.1, 0.1};
        double[] l1vals = {0.0, 0.0, 0.2}; //i.e., use l2vals[j] with l1vals[j]

        Nd4j.getRandom().setSeed(12345);
        int minibatch = 10;
        int depth = 2;
        int hw = 5;
        int nOut = 3;
        INDArray input = Nd4j.rand(new int[] {minibatch, depth, hw, hw});
        INDArray labels = Nd4j.zeros(minibatch, nOut);
        Random r = new Random(12345);
        for (int i = 0; i < minibatch; i++) {
            labels.putScalar(i, r.nextInt(nOut), 1.0);
        }

        DataSet ds = new DataSet(input, labels);

        for (Activation afn : activFns) {
            for (boolean doLearningFirst : characteristic) {
                for (int i = 0; i < lossFunctions.length; i++) {
                    for (int j = 0; j < l2vals.length; j++) {
                        LossFunctions.LossFunction lf = lossFunctions[i];
                        Activation outputActivation = outputActivations[i];

                        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().seed(12345)
                                        .l2(l2vals[j])
                                        .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                                        .updater(new NoOp()).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new UniformDistribution(-2, 2)).seed(12345L).list()
                                        .layer(0, new ConvolutionLayer.Builder(2, 2).stride(1, 1).nOut(3)
                                                        .activation(afn).build())
                                        .layer(1, new BatchNormalization.Builder().build())
                                        .layer(2, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                                        .kernelSize(2, 2).stride(1, 1).build())
                                        .layer(3, new BatchNormalization())
                                        .layer(4, new ActivationLayer.Builder().activation(afn).build())
                                        .layer(5, new OutputLayer.Builder(lf).activation(outputActivation).nOut(nOut)
                                                        .build())
                                        .setInputType(InputType.convolutional(hw, hw, depth)).pretrain(false)
                                        .backprop(true);

                        MultiLayerConfiguration conf = builder.build();

                        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                        mln.init();
                        String name = new Object() {}.getClass().getEnclosingMethod().getName();

                        if (doLearningFirst) {
                            //Run a number of iterations of learning
                            mln.setInput(ds.getFeatures());
                            mln.setLabels(ds.getLabels());
                            mln.computeGradientAndScore();
                            double scoreBefore = mln.score();
                            for (int k = 0; k < 20; k++)
                                mln.fit(ds);
                            mln.computeGradientAndScore();
                            double scoreAfter = mln.score();
                            //Can't test in 'characteristic mode of operation' if not learning
                            String msg = name
                                            + " - score did not (sufficiently) decrease during learning - activationFn="
                                            + afn + ", lossFn=" + lf + ", outputActivation=" + outputActivation
                                            + ", doLearningFirst= " + doLearningFirst + " (before=" + scoreBefore
                                            + ", scoreAfter=" + scoreAfter + ")";
                            assertTrue(msg, scoreAfter < 0.9 * scoreBefore);
                        }

                        if (PRINT_RESULTS) {
                            System.out.println(name + " - activationFn=" + afn + ", lossFn=" + lf
                                            + ", outputActivation=" + outputActivation + ", doLearningFirst="
                                            + doLearningFirst + ", l1=" + l1vals[j] + ", l2=" + l2vals[j]);
                            for (int k = 0; k < mln.getnLayers(); k++)
                                System.out.println("Layer " + k + " # params: " + mln.getLayer(k).numParams());
                        }

                        boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

                        assertTrue(gradOK);
                    }
                }
            }
        }
    }


    @Test
    public void testGradientDense() {
        //Parameterized test, testing combinations of:
        // (a) activation function
        // (b) Whether to test at random initialization, or after some learning (i.e., 'characteristic mode of operation')
        // (c) Loss function (with specified output activations)
        // (d) l1 and l2 values
        Activation[] activFns = {Activation.SIGMOID, Activation.TANH, Activation.IDENTITY};
        boolean[] characteristic = {false, true}; //If true: run some backprop steps first

        LossFunctions.LossFunction[] lossFunctions =
                        {LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD, LossFunctions.LossFunction.MSE};
        Activation[] outputActivations = {Activation.SOFTMAX, Activation.TANH}; //i.e., lossFunctions[i] used with outputActivations[i] here

        double[] l2vals = {0.0, 0.1, 0.1};
        double[] l1vals = {0.0, 0.0, 0.2}; //i.e., use l2vals[j] with l1vals[j]

        Nd4j.getRandom().setSeed(12345);
        int minibatch = 10;
        int nIn = 5;
        int nOut = 3;
        INDArray input = Nd4j.rand(new int[] {minibatch, nIn});
        INDArray labels = Nd4j.zeros(minibatch, nOut);
        Random r = new Random(12345);
        for (int i = 0; i < minibatch; i++) {
            labels.putScalar(i, r.nextInt(nOut), 1.0);
        }

        DataSet ds = new DataSet(input, labels);

        for (Activation afn : activFns) {
            for (boolean doLearningFirst : characteristic) {
                for (int i = 0; i < lossFunctions.length; i++) {
                    for (int j = 0; j < l2vals.length; j++) {
                        LossFunctions.LossFunction lf = lossFunctions[i];
                        Activation outputActivation = outputActivations[i];

                        MultiLayerConfiguration.Builder builder =
                                        new NeuralNetConfiguration.Builder()
                                                        .l2(l2vals[j])
                                                        .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                                                        .updater(new NoOp()).weightInit(WeightInit.DISTRIBUTION)
                                                        .dist(new UniformDistribution(-2, 2)).seed(12345L).list()
                                                        .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(4)
                                                                        .activation(afn).build())
                                                        .layer(1, new BatchNormalization.Builder().build())
                                                        .layer(2, new DenseLayer.Builder().nIn(4).nOut(4).build())
                                                        .layer(3, new BatchNormalization())
                                                        .layer(4, new OutputLayer.Builder(lf)
                                                                        .activation(outputActivation).nOut(nOut)
                                                                        .build())
                                                        .pretrain(false).backprop(true);

                        MultiLayerConfiguration conf = builder.build();

                        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                        mln.init();

                        String name = new Object() {}.getClass().getEnclosingMethod().getName();

                        if (doLearningFirst) {
                            //Run a number of iterations of learning
                            mln.setInput(ds.getFeatures());
                            mln.setLabels(ds.getLabels());
                            mln.computeGradientAndScore();
                            double scoreBefore = mln.score();
                            for (int k = 0; k < 10; k++)
                                mln.fit(ds);
                            mln.computeGradientAndScore();
                            double scoreAfter = mln.score();
                            //Can't test in 'characteristic mode of operation' if not learning
                            String msg = name
                                            + " - score did not (sufficiently) decrease during learning - activationFn="
                                            + afn + ", lossFn=" + lf + ", outputActivation=" + outputActivation
                                            + ", doLearningFirst= " + doLearningFirst + " (before=" + scoreBefore
                                            + ", scoreAfter=" + scoreAfter + ")";
                            assertTrue(msg, scoreAfter < 0.8 * scoreBefore);
                        }

                        if (PRINT_RESULTS) {
                            System.out.println(name + " - activationFn=" + afn + ", lossFn=" + lf
                                            + ", outputActivation=" + outputActivation + ", doLearningFirst="
                                            + doLearningFirst + ", l1=" + l1vals[j] + ", l2=" + l2vals[j]);
                            for (int k = 0; k < mln.getnLayers(); k++)
                                System.out.println("Layer " + k + " # params: " + mln.getLayer(k).numParams());
                        }

                        boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

                        assertTrue(gradOK);
                    }
                }
            }
        }
    }

    @Test
    public void testGradient2dFixedGammaBeta() {
        DataNormalization scaler = new NormalizerMinMaxScaler();
        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        scaler.fit(iter);
        iter.setPreProcessor(scaler);
        DataSet ds = iter.next();
        INDArray input = ds.getFeatureMatrix();
        INDArray labels = ds.getLabels();

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().updater(new NoOp())
                        .seed(12345L).weightInit(WeightInit.DISTRIBUTION)
                        .dist(new NormalDistribution(0, 1)).list()
                        .layer(0, new DenseLayer.Builder().nIn(4).nOut(3).activation(Activation.IDENTITY).build())
                        .layer(1, new BatchNormalization.Builder().lockGammaBeta(true).gamma(2.0).beta(0.5).nOut(3)
                                        .build())
                        .layer(2, new ActivationLayer.Builder().activation(Activation.TANH).build())
                        .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .activation(Activation.SOFTMAX).nIn(3).nOut(3).build())
                        .pretrain(false).backprop(true);

        MultiLayerNetwork mln = new MultiLayerNetwork(builder.build());
        mln.init();

        if (PRINT_RESULTS) {
            for (int j = 0; j < mln.getnLayers(); j++)
                System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

        assertTrue(gradOK);
    }

    @Test
    public void testGradientCnnFixedGammaBeta() {
        Nd4j.getRandom().setSeed(12345);
        int minibatch = 10;
        int depth = 1;
        int hw = 4;
        int nOut = 4;
        INDArray input = Nd4j.rand(new int[] {minibatch, depth, hw, hw});
        INDArray labels = Nd4j.zeros(minibatch, nOut);
        Random r = new Random(12345);
        for (int i = 0; i < minibatch; i++) {
            labels.putScalar(i, r.nextInt(nOut), 1.0);
        }

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().updater(new NoOp())
                        .seed(12345L).weightInit(WeightInit.DISTRIBUTION)
                        .dist(new NormalDistribution(0, 2)).list()
                        .layer(0, new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1).nIn(depth).nOut(2)
                                        .activation(Activation.IDENTITY).build())
                        .layer(1, new BatchNormalization.Builder().lockGammaBeta(true).gamma(2.0).beta(0.5).build())
                        .layer(2, new ActivationLayer.Builder().activation(Activation.TANH).build())
                        .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .activation(Activation.SOFTMAX).nOut(nOut).build())
                        .setInputType(InputType.convolutional(hw, hw, depth)).pretrain(false).backprop(true);

        MultiLayerNetwork mln = new MultiLayerNetwork(builder.build());
        mln.init();

        if (PRINT_RESULTS) {
            for (int j = 0; j < mln.getnLayers(); j++)
                System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

        assertTrue(gradOK);
    }

    @Test
    public void testBatchNormCompGraphSimple() {

        int numClasses = 2;
        int height = 3;
        int width = 3;
        int channels = 1;
        long seed = 123;

        int minibatchSize = 3;

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).updater(new NoOp())
                        .weightInit(WeightInit.XAVIER).graphBuilder().addInputs("in")
                        .setInputTypes(InputType.convolutional(height, width, channels))
                        .addLayer("bn", new BatchNormalization.Builder().build(), "in")
                        .addLayer("out", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT)
                                        .activation(Activation.SOFTMAX).nOut(numClasses).build(), "bn")
                        .setOutputs("out").backprop(true).pretrain(false).build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        Random r = new Random(12345);
        INDArray input = Nd4j.rand(new int[] {minibatchSize, channels, height, width}); //Order: examples, channels, height, width
        INDArray labels = Nd4j.zeros(minibatchSize, numClasses);
        for (int i = 0; i < minibatchSize; i++) {
            labels.putScalar(new int[] {i, r.nextInt(numClasses)}, 1.0);
        }

        boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, new INDArray[] {input},
                        new INDArray[] {labels});

        assertTrue(gradOK);
    }


    @Test
    public void testGradientBNWithCNNandSubsamplingCompGraph() {
        //Parameterized test, testing combinations of:
        // (a) activation function
        // (b) Whether to test at random initialization, or after some learning (i.e., 'characteristic mode of operation')
        // (c) Loss function (with specified output activations)
        // (d) l1 and l2 values
        Activation[] activFns = {Activation.SIGMOID, Activation.TANH, Activation.IDENTITY};
        boolean[] characteristic = {false, true}; //If true: run some backprop steps first

        LossFunctions.LossFunction[] lossFunctions =
                        {LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD, LossFunctions.LossFunction.MSE};
        Activation[] outputActivations = {Activation.SOFTMAX, Activation.TANH}; //i.e., lossFunctions[i] used with outputActivations[i] here

        double[] l2vals = {0.0, 0.1, 0.1};
        double[] l1vals = {0.0, 0.0, 0.2}; //i.e., use l2vals[j] with l1vals[j]

        Nd4j.getRandom().setSeed(12345);
        int minibatch = 10;
        int depth = 2;
        int hw = 5;
        int nOut = 3;
        INDArray input = Nd4j.rand(new int[] {minibatch, depth, hw, hw});
        INDArray labels = Nd4j.zeros(minibatch, nOut);
        Random r = new Random(12345);
        for (int i = 0; i < minibatch; i++) {
            labels.putScalar(i, r.nextInt(nOut), 1.0);
        }

        DataSet ds = new DataSet(input, labels);

        for (Activation afn : activFns) {
            for (boolean doLearningFirst : characteristic) {
                for (int i = 0; i < lossFunctions.length; i++) {
                    for (int j = 0; j < l2vals.length; j++) {
                        LossFunctions.LossFunction lf = lossFunctions[i];
                        Activation outputActivation = outputActivations[i];

                        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                                        .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                                        .updater(new NoOp()).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new UniformDistribution(-2, 2)).seed(12345L).graphBuilder()
                                        .addInputs("in")
                                        .addLayer("0", new ConvolutionLayer.Builder(2, 2).stride(1, 1).nOut(3)
                                                        .activation(afn).build(), "in")
                                        .addLayer("1", new BatchNormalization.Builder().build(), "0")
                                        .addLayer("2", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                                        .kernelSize(2, 2).stride(1, 1).build(), "1")
                                        .addLayer("3", new BatchNormalization(), "2")
                                        .addLayer("4", new ActivationLayer.Builder().activation(afn).build(), "3")
                                        .addLayer("5", new OutputLayer.Builder(lf).activation(outputActivation)
                                                        .nOut(nOut).build(), "4")
                                        .setOutputs("5").setInputTypes(InputType.convolutional(hw, hw, depth))
                                        .pretrain(false).backprop(true).build();

                        ComputationGraph net = new ComputationGraph(conf);
                        net.init();
                        String name = new Object() {}.getClass().getEnclosingMethod().getName();

                        if (doLearningFirst) {
                            //Run a number of iterations of learning
                            net.setInput(0, ds.getFeatures());
                            net.setLabels(ds.getLabels());
                            net.computeGradientAndScore();
                            double scoreBefore = net.score();
                            for (int k = 0; k < 20; k++)
                                net.fit(ds);
                            net.computeGradientAndScore();
                            double scoreAfter = net.score();
                            //Can't test in 'characteristic mode of operation' if not learning
                            String msg = name
                                            + " - score did not (sufficiently) decrease during learning - activationFn="
                                            + afn + ", lossFn=" + lf + ", outputActivation=" + outputActivation
                                            + ", doLearningFirst= " + doLearningFirst + " (before=" + scoreBefore
                                            + ", scoreAfter=" + scoreAfter + ")";
                            assertTrue(msg, scoreAfter < 0.9 * scoreBefore);
                        }

                        if (PRINT_RESULTS) {
                            System.out.println(name + " - activationFn=" + afn + ", lossFn=" + lf
                                            + ", outputActivation=" + outputActivation + ", doLearningFirst="
                                            + doLearningFirst + ", l1=" + l1vals[j] + ", l2=" + l2vals[j]);
                            for (int k = 0; k < net.getNumLayers(); k++)
                                System.out.println("Layer " + k + " # params: " + net.getLayer(k).numParams());
                        }

                        boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE,
                                        new INDArray[] {input}, new INDArray[] {labels});

                        assertTrue(gradOK);
                    }
                }
            }
        }
    }

}
