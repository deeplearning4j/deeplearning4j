package org.deeplearning4j.gradientcheck;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.*;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.variational.*;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.activations.impl.ActivationTanH;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossMAE;
import org.nd4j.linalg.lossfunctions.impl.LossMSE;

import java.util.Arrays;
import java.util.Random;

import static junit.framework.TestCase.fail;
import static org.junit.Assert.assertTrue;

/**
 * @author Alex Black
 */
public class VaeGradientCheckTests {

    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;

    static {
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void testVaeAsMLP() {

        //Post pre-training: a VAE can be used as a MLP, by taking the mean value from p(z|x) as the output
        //This gradient check tests this part

        String[] activFns = {"identity", "tanh"}; //activation functions such as relu and hardtanh: may randomly fail due to discontinuities

        LossFunction[] lossFunctions = {LossFunction.MCXENT, LossFunction.MSE};
        String[] outputActivations = {"softmax", "tanh"}; //i.e., lossFunctions[i] used with outputActivations[i] here

        //use l2vals[i] with l1vals[i]
        double[] l2vals = {0.4, 0.0, 0.4, 0.4};
        double[] l1vals = {0.0, 0.0, 0.5, 0.0};
        double[] biasL2 = {0.0, 0.0, 0.0, 0.2};
        double[] biasL1 = {0.0, 0.0, 0.6, 0.0};

        int[][] encoderLayerSizes = new int[][] {{5}, {5, 6}};
        int[][] decoderLayerSizes = new int[][] {{6}, {7, 8}};

        Nd4j.getRandom().setSeed(12345);
        for (int minibatch : new int[] {1, 5}) {
            INDArray input = Nd4j.rand(minibatch, 4);
            INDArray labels = Nd4j.create(minibatch, 3);
            for (int i = 0; i < minibatch; i++) {
                labels.putScalar(i, i % 3, 1.0);
            }

            for (int ls = 0; ls < encoderLayerSizes.length; ls++) {
                int[] encoderSizes = encoderLayerSizes[ls];
                int[] decoderSizes = decoderLayerSizes[ls];

                for (String afn : activFns) {
                    for (int i = 0; i < lossFunctions.length; i++) {
                        for (int k = 0; k < l2vals.length; k++) {
                            LossFunction lf = lossFunctions[i];
                            String outputActivation = outputActivations[i];
                            double l2 = l2vals[k];
                            double l1 = l1vals[k];

                            MultiLayerConfiguration conf =
                                            new NeuralNetConfiguration.Builder().regularization(true).l2(l2).l1(l1)
                                                            .l2Bias(biasL2[k]).l1Bias(biasL1[k])
                                                            .optimizationAlgo(
                                                                            OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                                            .learningRate(1.0).seed(12345L).list()
                                                            .layer(0, new VariationalAutoencoder.Builder().nIn(4)
                                                                            .nOut(3).encoderLayerSizes(encoderSizes)
                                                                            .decoderLayerSizes(decoderSizes)
                                                                            .weightInit(WeightInit.DISTRIBUTION)
                                                                            .dist(new NormalDistribution(0, 1))
                                                                            .activation(afn).updater(
                                                                                            Updater.SGD)
                                                                            .build())
                                                            .layer(1, new OutputLayer.Builder(lf)
                                                                            .activation(outputActivation).nIn(3).nOut(3)
                                                                            .weightInit(WeightInit.DISTRIBUTION)
                                                                            .dist(new NormalDistribution(0, 1))
                                                                            .updater(Updater.SGD).build())
                                                            .pretrain(false).backprop(true).build();

                            MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                            mln.init();

                            String msg = "testVaeAsMLP() - activationFn=" + afn + ", lossFn=" + lf
                                            + ", outputActivation=" + outputActivation + ", encLayerSizes = "
                                            + Arrays.toString(encoderSizes) + ", decLayerSizes = "
                                            + Arrays.toString(decoderSizes) + ", l2=" + l2 + ", l1=" + l1;
                            if (PRINT_RESULTS) {
                                System.out.println(msg);
                                for (int j = 0; j < mln.getnLayers(); j++)
                                    System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
                            }

                            boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input,
                                            labels);
                            assertTrue(msg, gradOK);
                        }
                    }
                }
            }
        }
    }


    @Test
    public void testVaePretrain() {

        String[] activFns = {"identity", "identity", "tanh", "tanh"}; //activation functions such as relu and hardtanh: may randomly fail due to discontinuities
        String[] pzxAfns = {"identity", "tanh", "identity", "tanh"};
        String[] pxzAfns = {"tanh", "identity", "tanh", "identity"};

        //use l2vals[i] with l1vals[i]
        double[] l2vals = {0.4, 0.0, 0.4, 0.4};
        double[] l1vals = {0.0, 0.0, 0.5, 0.0};
        double[] biasL2 = {0.0, 0.0, 0.0, 0.2};
        double[] biasL1 = {0.0, 0.0, 0.6, 0.0};

        int[][] encoderLayerSizes = new int[][] {{5}, {5, 6}};
        int[][] decoderLayerSizes = new int[][] {{6}, {7, 8}};

        Nd4j.getRandom().setSeed(12345);
        for (int minibatch : new int[] {1, 5}) {
            INDArray features = Nd4j.rand(minibatch, 4);

            for (int ls = 0; ls < encoderLayerSizes.length; ls++) {
                int[] encoderSizes = encoderLayerSizes[ls];
                int[] decoderSizes = decoderLayerSizes[ls];

                for (int j = 0; j < activFns.length; j++) {
                    String afn = activFns[j];
                    String pzxAfn = pzxAfns[j];
                    String pxzAfn = pxzAfns[j];
                    double l2 = l2vals[j]; //Ideally we'd do the cartesian product of l1/l2 and the activation functions, but that takes too long...
                    double l1 = l1vals[j];

                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().regularization(true).l2(l2)
                                    .l1(l1).l2Bias(biasL2[j]).l1Bias(biasL1[j])
                                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                    .learningRate(1.0).seed(12345L).weightInit(WeightInit.XAVIER).list()
                                    .layer(0, new VariationalAutoencoder.Builder().nIn(4).nOut(3)
                                                    .encoderLayerSizes(encoderSizes).decoderLayerSizes(decoderSizes)
                                                    .pzxActivationFunction(pzxAfn)
                                                    .reconstructionDistribution(
                                                                    new GaussianReconstructionDistribution(pxzAfn))
                                                    .activation(afn).updater(Updater.SGD).build())
                                    .pretrain(true).backprop(false).build();

                    MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                    mln.init();
                    mln.initGradientsView();

                    org.deeplearning4j.nn.api.Layer layer = mln.getLayer(0);

                    String msg = "testVaePretrain() - activationFn=" + afn + ", p(z|x) afn = " + pzxAfn
                                    + ", p(x|z) afn = " + pxzAfn + ", encLayerSizes = " + Arrays.toString(encoderSizes)
                                    + ", decLayerSizes = " + Arrays.toString(decoderSizes) + ", l2=" + l2 + ", l1="
                                    + l1;
                    if (PRINT_RESULTS) {
                        System.out.println(msg);
                        for (int l = 0; l < mln.getnLayers(); l++)
                            System.out.println("Layer " + l + " # params: " + mln.getLayer(l).numParams());
                    }

                    boolean gradOK = GradientCheckUtil.checkGradientsPretrainLayer(layer, DEFAULT_EPS,
                                    DEFAULT_MAX_REL_ERROR, DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS,
                                    RETURN_ON_FIRST_FAILURE, features, 12345);

                    assertTrue(msg, gradOK);
                }
            }
        }
    }

    @Test
    public void testVaePretrainReconstructionDistributions() {

        int inOutSize = 6;

        ReconstructionDistribution[] reconstructionDistributions =
                        new ReconstructionDistribution[] {new GaussianReconstructionDistribution(Activation.IDENTITY),
                                        new GaussianReconstructionDistribution(Activation.TANH),
                                        new BernoulliReconstructionDistribution(Activation.SIGMOID),
                                        new CompositeReconstructionDistribution.Builder()
                                                        .addDistribution(2,
                                                                        new GaussianReconstructionDistribution(
                                                                                        Activation.IDENTITY))
                                                        .addDistribution(2, new BernoulliReconstructionDistribution())
                                                        .addDistribution(2,
                                                                        new GaussianReconstructionDistribution(
                                                                                        Activation.TANH))
                                                        .build(),
                                        new ExponentialReconstructionDistribution("identity"),
                                        new ExponentialReconstructionDistribution("tanh"),
                                        new LossFunctionWrapper(new ActivationTanH(), new LossMSE()),
                                        new LossFunctionWrapper(new ActivationIdentity(), new LossMAE())};

        Nd4j.getRandom().setSeed(12345);
        for (int minibatch : new int[] {1, 5}) {
            for (int i = 0; i < reconstructionDistributions.length; i++) {

                INDArray data;
                switch (i) {
                    case 0: //Gaussian + identity
                    case 1: //Gaussian + tanh
                        data = Nd4j.rand(minibatch, inOutSize);
                        break;
                    case 2: //Bernoulli
                        data = Nd4j.create(minibatch, inOutSize);
                        Nd4j.getExecutioner().exec(new BernoulliDistribution(data, 0.5), Nd4j.getRandom());
                        break;
                    case 3: //Composite
                        data = Nd4j.create(minibatch, inOutSize);
                        data.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2)).assign(Nd4j.rand(minibatch, 2));
                        Nd4j.getExecutioner()
                                        .exec(new BernoulliDistribution(
                                                        data.get(NDArrayIndex.all(), NDArrayIndex.interval(2, 4)), 0.5),
                                                        Nd4j.getRandom());
                        data.get(NDArrayIndex.all(), NDArrayIndex.interval(4, 6)).assign(Nd4j.rand(minibatch, 2));
                        break;
                    case 4:
                    case 5:
                        data = Nd4j.rand(minibatch, inOutSize);
                        break;
                    case 6:
                    case 7:
                        data = Nd4j.randn(minibatch, inOutSize);
                        break;
                    default:
                        throw new RuntimeException();
                }

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().regularization(true).l2(0.2).l1(0.3)
                                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).learningRate(1.0)
                                .seed(12345L).weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                                .list().layer(0,
                                                new VariationalAutoencoder.Builder().nIn(inOutSize).nOut(3)
                                                                .encoderLayerSizes(5).decoderLayerSizes(6)
                                                                .pzxActivationFunction(Activation.TANH)
                                                                .reconstructionDistribution(
                                                                                reconstructionDistributions[i])
                                                                .activation(Activation.TANH).updater(Updater.SGD)
                                                                .build())
                                .pretrain(true).backprop(false).build();

                MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                mln.init();
                mln.initGradientsView();

                org.deeplearning4j.nn.api.Layer layer = mln.getLayer(0);

                String msg = "testVaePretrainReconstructionDistributions() - " + reconstructionDistributions[i];
                if (PRINT_RESULTS) {
                    System.out.println(msg);
                    for (int j = 0; j < mln.getnLayers(); j++)
                        System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
                }

                boolean gradOK = GradientCheckUtil.checkGradientsPretrainLayer(layer, DEFAULT_EPS,
                                DEFAULT_MAX_REL_ERROR, DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE,
                                data, 12345);

                assertTrue(msg, gradOK);
            }
        }
    }

    @Test
    public void testVaePretrainMultipleSamples() {

        Nd4j.getRandom().setSeed(12345);
        for (int minibatch : new int[] {1, 5}) {
            for (int numSamples : new int[] {1, 10}) {
                //            for (int numSamples : new int[]{10}) {
                INDArray features = Nd4j.rand(minibatch, 4);

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().regularization(true).l2(0.2).l1(0.3)
                                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).learningRate(1.0)
                                .seed(12345L).weightInit(WeightInit.XAVIER).list()
                                .layer(0, new VariationalAutoencoder.Builder().nIn(4).nOut(3).encoderLayerSizes(5, 6)
                                                .decoderLayerSizes(7, 8).pzxActivationFunction(Activation.TANH)
                                                .reconstructionDistribution(
                                                                new GaussianReconstructionDistribution(Activation.TANH))
                                                .numSamples(numSamples).activation(Activation.TANH).updater(Updater.SGD)
                                                .build())
                                .pretrain(true).backprop(false).build();

                MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                mln.init();
                mln.initGradientsView();

                org.deeplearning4j.nn.api.Layer layer = mln.getLayer(0);

                String msg = "testVaePretrainMultipleSamples() - numSamples = " + numSamples;
                if (PRINT_RESULTS) {
                    System.out.println(msg);
                    for (int j = 0; j < mln.getnLayers(); j++)
                        System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
                }

                boolean gradOK = GradientCheckUtil.checkGradientsPretrainLayer(layer, DEFAULT_EPS,
                                DEFAULT_MAX_REL_ERROR, DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE,
                                features, 12345);

                assertTrue(msg, gradOK);
            }
        }
    }
}
