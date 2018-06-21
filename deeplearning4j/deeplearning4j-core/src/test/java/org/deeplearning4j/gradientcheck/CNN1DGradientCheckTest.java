package org.deeplearning4j.gradientcheck;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping1D;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class CNN1DGradientCheckTest extends BaseDL4JTest {
    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;

    static {
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void testCnn1DWithCropping1D() {
        Nd4j.getRandom().setSeed(1337);

        int[] minibatchSizes = {1, 3};
        int length = 7;
        int convNIn = 2;
        int convNOut1 = 3;
        int convNOut2 = 4;
        int finalNOut = 4;


        int[] kernels = {1, 2, 4};
        int stride = 1;

        int padding = 0;
        int cropping = 1;
        int croppedLength = length - 2 * cropping;

        Activation[] activations = {Activation.SIGMOID};
        SubsamplingLayer.PoolingType[] poolingTypes =
                new SubsamplingLayer.PoolingType[] {SubsamplingLayer.PoolingType.MAX,
                        SubsamplingLayer.PoolingType.AVG, SubsamplingLayer.PoolingType.PNORM};

        for (Activation afn : activations) {
            for (SubsamplingLayer.PoolingType poolingType : poolingTypes) {
                for (int minibatchSize : minibatchSizes) {
                    for (int kernel : kernels) {
                        INDArray input = Nd4j.rand(new int[] {minibatchSize, convNIn, length});
                        INDArray labels = Nd4j.zeros(minibatchSize, finalNOut, croppedLength);
                        for (int i = 0; i < minibatchSize; i++) {
                            for (int j = 0; j < croppedLength; j++) {
                                labels.putScalar(new int[] {i, i % finalNOut, j}, 1.0);
                            }
                        }

                        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                                .updater(new NoOp()).weightInit(WeightInit.DISTRIBUTION)
                                .dist(new NormalDistribution(0, 1)).convolutionMode(ConvolutionMode.Same).list()
                                .layer(new Convolution1DLayer.Builder().activation(afn).kernelSize(kernel)
                                        .stride(stride).padding(padding).nIn(convNIn).nOut(convNOut1)
                                        .build())
                                .layer(new Cropping1D.Builder(cropping).build())
                                .layer(new Convolution1DLayer.Builder().activation(afn).kernelSize(kernel)
                                        .stride(stride).padding(padding).nIn(convNOut1).nOut(convNOut2)
                                        .build())
                                .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .activation(Activation.SOFTMAX).nOut(finalNOut).build())
                                .setInputType(InputType.recurrent(convNIn, length)).build();

                        String json = conf.toJson();
                        MultiLayerConfiguration c2 = MultiLayerConfiguration.fromJson(json);
                        assertEquals(conf, c2);

                        MultiLayerNetwork net = new MultiLayerNetwork(conf);
                        net.init();

                        String msg = "PoolingType=" + poolingType + ", minibatch=" + minibatchSize + ", activationFn="
                                + afn + ", kernel = " + kernel;

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


    @Test
    public void testCnn1DWithZeroPadding1D() {
        Nd4j.getRandom().setSeed(1337);

        int[] minibatchSizes = {1, 3};
        int length = 7;
        int convNIn = 2;
        int convNOut1 = 3;
        int convNOut2 = 4;
        int finalNOut = 4;


        int[] kernels = {1, 2, 4};
        int stride = 1;
        int pnorm = 2;

        int padding = 0;
        int zeroPadding = 2;
        int paddedLength = length + 2 * zeroPadding;

        Activation[] activations = {Activation.SIGMOID};
        SubsamplingLayer.PoolingType[] poolingTypes =
                new SubsamplingLayer.PoolingType[] {SubsamplingLayer.PoolingType.MAX,
                        SubsamplingLayer.PoolingType.AVG, SubsamplingLayer.PoolingType.PNORM};

        for (Activation afn : activations) {
            for (SubsamplingLayer.PoolingType poolingType : poolingTypes) {
                for (int minibatchSize : minibatchSizes) {
                    for (int kernel : kernels) {
                        INDArray input = Nd4j.rand(new int[] {minibatchSize, convNIn, length});
                        INDArray labels = Nd4j.zeros(minibatchSize, finalNOut, paddedLength);
                        for (int i = 0; i < minibatchSize; i++) {
                            for (int j = 0; j < paddedLength; j++) {
                                labels.putScalar(new int[] {i, i % finalNOut, j}, 1.0);
                            }
                        }

                        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                                .updater(new NoOp()).weightInit(WeightInit.DISTRIBUTION)
                                .dist(new NormalDistribution(0, 1)).convolutionMode(ConvolutionMode.Same).list()
                                .layer(new Convolution1DLayer.Builder().activation(afn).kernelSize(kernel)
                                        .stride(stride).padding(padding).nIn(convNIn).nOut(convNOut1)
                                        .build())
                                .layer(new ZeroPadding1DLayer.Builder(zeroPadding).build())
                                .layer(new Convolution1DLayer.Builder().activation(afn).kernelSize(kernel)
                                        .stride(stride).padding(padding).nIn(convNOut1).nOut(convNOut2)
                                        .build())
                                .layer(new ZeroPadding1DLayer.Builder(0).build())
                                .layer(new Subsampling1DLayer.Builder(poolingType).kernelSize(kernel)
                                        .stride(stride).padding(padding).pnorm(pnorm).build())
                                .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .activation(Activation.SOFTMAX).nOut(finalNOut).build())
                                .setInputType(InputType.recurrent(convNIn, length)).build();

                        String json = conf.toJson();
                        MultiLayerConfiguration c2 = MultiLayerConfiguration.fromJson(json);
                        assertEquals(conf, c2);

                        MultiLayerNetwork net = new MultiLayerNetwork(conf);
                        net.init();

                        String msg = "PoolingType=" + poolingType + ", minibatch=" + minibatchSize + ", activationFn="
                                + afn + ", kernel = " + kernel;

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


    @Test
    public void testCnn1DWithSubsampling1D() {
        Nd4j.getRandom().setSeed(12345);

        int[] minibatchSizes = {1, 3};
        int length = 7;
        int convNIn = 2;
        int convNOut1 = 3;
        int convNOut2 = 4;
        int finalNOut = 4;

        int[] kernels = {1, 2, 4};
        int stride = 1;
        int padding = 0;
        int pnorm = 2;

        Activation[] activations = {Activation.SIGMOID, Activation.TANH};
        SubsamplingLayer.PoolingType[] poolingTypes =
                        new SubsamplingLayer.PoolingType[] {SubsamplingLayer.PoolingType.MAX,
                                        SubsamplingLayer.PoolingType.AVG, SubsamplingLayer.PoolingType.PNORM};

        for (Activation afn : activations) {
            for (SubsamplingLayer.PoolingType poolingType : poolingTypes) {
                for (int minibatchSize : minibatchSizes) {
                    for (int kernel : kernels) {
                        INDArray input = Nd4j.rand(new int[] {minibatchSize, convNIn, length});
                        INDArray labels = Nd4j.zeros(minibatchSize, finalNOut, length);
                        for (int i = 0; i < minibatchSize; i++) {
                            for (int j = 0; j < length; j++) {
                                labels.putScalar(new int[] {i, i % finalNOut, j}, 1.0);
                            }
                        }

                        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                                        .updater(new NoOp()).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 1)).convolutionMode(ConvolutionMode.Same).list()
                                        .layer(0, new Convolution1DLayer.Builder().activation(afn).kernelSize(kernel)
                                                        .stride(stride).padding(padding).nIn(convNIn).nOut(convNOut1)
                                                        .build())
                                        .layer(1, new Convolution1DLayer.Builder().activation(afn).kernelSize(kernel)
                                                        .stride(stride).padding(padding).nIn(convNOut1).nOut(convNOut2)
                                                        .build())
                                        .layer(2, new Subsampling1DLayer.Builder(poolingType).kernelSize(kernel)
                                                        .stride(stride).padding(padding).pnorm(pnorm).build())
                                        .layer(3, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                                        .activation(Activation.SOFTMAX).nOut(finalNOut).build())
                                        .setInputType(InputType.recurrent(convNIn, length)).build();

                        String json = conf.toJson();
                        MultiLayerConfiguration c2 = MultiLayerConfiguration.fromJson(json);
                        assertEquals(conf, c2);

                        MultiLayerNetwork net = new MultiLayerNetwork(conf);
                        net.init();

                        String msg = "PoolingType=" + poolingType + ", minibatch=" + minibatchSize + ", activationFn="
                                        + afn + ", kernel = " + kernel;

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
}
