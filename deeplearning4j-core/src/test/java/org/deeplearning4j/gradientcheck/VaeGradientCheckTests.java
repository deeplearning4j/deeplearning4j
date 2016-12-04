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
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.GaussianReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.ReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

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
        //Force Nd4j initialization, then set data type to double:
        Nd4j.zeros(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void testVaeAsMLP() {

        //Post pre-training: a VAE can be used as a MLP, by taking the mean value from p(z|x) as the output
        //This gradient check tests this part

        String[] activFns = {"identity", "tanh"};    //activation functions such as relu and hardtanh: may randomly fail due to discontinuities

        LossFunction[] lossFunctions = {LossFunction.MCXENT, LossFunction.MSE};
        String[] outputActivations = {"softmax", "tanh"};    //i.e., lossFunctions[i] used with outputActivations[i] here

        double[] l2vals = {0.0, 0.4, 0.0, 0.4};
        double[] l1vals = {0.0, 0.0, 0.5, 0.5};    //i.e., use l2vals[i] with l1vals[i]

        int[][] encoderLayerSizes = new int[][]{{5}, {5,6}};
        int[][] decoderLayerSizes = new int[][]{{6}, {7,8}};

        Nd4j.getRandom().setSeed(12345);
        for(int minibatch : new int[]{1, 5}) {
            INDArray input = Nd4j.rand(minibatch, 4);
            INDArray labels = Nd4j.create(minibatch, 3);
            for( int i= 0; i<minibatch; i++ ){
                labels.putScalar(i, i%3, 1.0);
            }

            for( int ls=0; ls < encoderLayerSizes.length; ls++ ) {
                int[] encoderSizes = encoderLayerSizes[ls];
                int[] decoderSizes = decoderLayerSizes[ls];

                for (String afn : activFns) {
                    for (int i = 0; i < lossFunctions.length; i++) {
                        for (int k = 0; k < l2vals.length; k++) {
                            LossFunction lf = lossFunctions[i];
                            String outputActivation = outputActivations[i];
                            double l2 = l2vals[k];
                            double l1 = l1vals[k];

                            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                                    .regularization(true)
                                    .l2(l2).l1(l1)
                                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                    .learningRate(1.0)
                                    .seed(12345L)
                                    .list()
                                    .layer(0, new VariationalAutoencoder.Builder()
                                            .nIn(4).nOut(3)
                                            .encoderLayerSizes(encoderSizes)
                                            .decoderLayerSizes(decoderSizes)
                                            .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                                            .activation(afn)
                                            .updater(Updater.SGD)
                                            .build())
                                    .layer(1, new OutputLayer.Builder(lf)
                                            .activation(outputActivation)
                                            .nIn(3).nOut(3)
                                            .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                                            .updater(Updater.SGD)
                                            .build())
                                    .pretrain(false).backprop(true)
                                    .build();

                            MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                            mln.init();

                            String msg = "testVaeAsMLP() - activationFn=" + afn + ", lossFn=" + lf + ", outputActivation=" + outputActivation
                                    + ", encLayerSizes = " + Arrays.toString(encoderSizes) + ", decLayerSizes = " + Arrays.toString(decoderSizes)
                                    + ", l2=" + l2 + ", l1=" + l1;
                            if (PRINT_RESULTS) {
                                System.out.println(msg);
                                for (int j = 0; j < mln.getnLayers(); j++)
                                    System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
                            }

                            boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR, DEFAULT_MIN_ABS_ERROR,
                                    PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);
                            assertTrue(msg, gradOK);
                        }
                    }
                }
            }
        }
    }


    @Test
    public void testVaePretrain() {

        String[] activFns = {"identity", "tanh"};    //activation functions such as relu and hardtanh: may randomly fail due to discontinuities

        double[] l2vals = {0.0, 0.4, 0.0, 0.4};
        double[] l1vals = {0.0, 0.0, 0.5, 0.5};    //i.e., use l2vals[i] with l1vals[i]

        int[][] encoderLayerSizes = new int[][]{{5}, {5,6}};
        int[][] decoderLayerSizes = new int[][]{{6}, {7,8}};

        Nd4j.getRandom().setSeed(12345);
        for(int minibatch : new int[]{1, 5}) {
            INDArray features = Nd4j.rand(minibatch, 4);

            for (int ls = 0; ls < encoderLayerSizes.length; ls++) {
                int[] encoderSizes = encoderLayerSizes[ls];
                int[] decoderSizes = decoderLayerSizes[ls];

                for (String afn : activFns) {
                    for (String pzxAfn : activFns) {
                        for (String pxzAfn : activFns) {
                            for (int k = 0; k < l2vals.length; k++) {
                                double l2 = l2vals[k];
                                double l1 = l1vals[k];

                                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                                        .regularization(true)
                                        .l2(l2).l1(l1)
                                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                        .learningRate(1.0)
                                        .seed(12345L)
//                                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                                        .weightInit(WeightInit.XAVIER)
                                        .list()
                                        .layer(0, new VariationalAutoencoder.Builder()
                                                .nIn(4).nOut(3)
                                                .encoderLayerSizes(encoderSizes)
                                                .decoderLayerSizes(decoderSizes)
                                                .pzxActivationFunction(pzxAfn)
                                                .reconstructionDistribution(new GaussianReconstructionDistribution(pxzAfn))
                                                .activation(afn)
                                                .updater(Updater.SGD)
                                                .build())
                                        .pretrain(true).backprop(false)
                                        .build();

                                MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                                mln.init();
                                mln.initGradientsView();

                                org.deeplearning4j.nn.api.Layer layer = mln.getLayer(0);

                                String msg = "testVaePretrain() - activationFn=" + afn + ", p(z|x) afn = " + pzxAfn + ", p(x|z) afn = " + pxzAfn +
                                        ", encLayerSizes = " + Arrays.toString(encoderSizes) + ", decLayerSizes = " + Arrays.toString(decoderSizes)
                                        + ", l2=" + l2 + ", l1=" + l1;
                                if (PRINT_RESULTS) {
                                    System.out.println(msg);
                                    for (int j = 0; j < mln.getnLayers(); j++)
                                        System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
                                }

                                boolean gradOK = GradientCheckUtil.checkGradientsPretrainLayer(layer, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR, DEFAULT_MIN_ABS_ERROR,
                                        PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, features, 12345);

                                assertTrue(msg, gradOK);
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    public void testVaePretrainReconstructionDistributions() {

        int[][] encoderLayerSizes = new int[][]{{5}, {5,6}};
        int[][] decoderLayerSizes = new int[][]{{6}, {7,8}};

        ReconstructionDistribution[] reconstructionDistributions = new ReconstructionDistribution[]{
                new GaussianReconstructionDistribution("identity"), new GaussianReconstructionDistribution("tanh"),
                new BernoulliReconstructionDistribution("sigmoid")};

        Nd4j.getRandom().setSeed(12345);
        for(int minibatch : new int[]{1, 5}) {
            INDArray features = Nd4j.rand(minibatch, 4);

            for (int ls = 0; ls < encoderLayerSizes.length; ls++) {
                int[] encoderSizes = encoderLayerSizes[ls];
                int[] decoderSizes = decoderLayerSizes[ls];

                for(ReconstructionDistribution rd : reconstructionDistributions) {

                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .regularization(true)
                            .l2(0.2).l1(0.3)
                            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                            .learningRate(1.0)
                            .seed(12345L)
                            .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                            .list()
                            .layer(0, new VariationalAutoencoder.Builder()
                                    .nIn(4).nOut(3)
                                    .encoderLayerSizes(encoderSizes)
                                    .decoderLayerSizes(decoderSizes)
                                    .pzxActivationFunction("tanh")
                                    .reconstructionDistribution(rd)
                                    .activation("tanh")
                                    .updater(Updater.SGD)
                                    .build())
                            .pretrain(true).backprop(false)
                            .build();

                    MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                    mln.init();
                    mln.initGradientsView();

                    org.deeplearning4j.nn.api.Layer layer = mln.getLayer(0);

                    String msg = "testVaePretrainReconstructionDistributions() - " + rd + ", encLayerSizes = " + Arrays.toString(encoderSizes)
                            + ", decLayerSizes = " + Arrays.toString(decoderSizes);
                    if (PRINT_RESULTS) {
                        System.out.println(msg);
                        for (int j = 0; j < mln.getnLayers(); j++)
                            System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
                    }

                    boolean gradOK = GradientCheckUtil.checkGradientsPretrainLayer(layer, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR, DEFAULT_MIN_ABS_ERROR,
                            PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, features, 12345);

                    assertTrue(msg, gradOK);
                }
            }
        }
    }
}
