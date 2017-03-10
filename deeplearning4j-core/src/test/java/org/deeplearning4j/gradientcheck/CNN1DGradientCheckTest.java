package org.deeplearning4j.gradientcheck;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * Created by nyghtowl on 9/1/15.
 */
public class CNN1DGradientCheckTest {
    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;
    public static final int finalNOut = 4;

    static {
        //Force Nd4j initialization, then set data type to double:
        Nd4j.zeros(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void testCnn1DWithSubsampling1D(){
        Nd4j.getRandom().setSeed(12345);

        int[] minibatchSizes = {3,1};
        int length = 7;
        int convNIn = 2;
        int convNOut = 3;

        int kernel = 2;
        int stride = 1;
        int padding = 0;
        int pnorm = 2;

        Activation[] activations = {Activation.SIGMOID, Activation.TANH};
        SubsamplingLayer.PoolingType[] poolingTypes = new SubsamplingLayer.PoolingType[]{SubsamplingLayer.PoolingType.MAX, SubsamplingLayer.PoolingType.AVG, SubsamplingLayer.PoolingType.PNORM};

        for(Activation afn : activations) {
            for (SubsamplingLayer.PoolingType poolingType : poolingTypes) {
                for (int minibatchSize : minibatchSizes) {
                    INDArray input = Nd4j.rand(minibatchSize, convNIn*length).reshape(minibatchSize, convNIn, length);
                    INDArray labels = Nd4j.zeros(minibatchSize, finalNOut*length).reshape(minibatchSize, finalNOut, length);
                    for (int i = 0; i < minibatchSize; i++) {
                        for (int j = 0; j < length; j++) {
                            labels.putScalar(new int[]{i, i % finalNOut, j}, 1.0);
                        }
                    }

                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .regularization(false)
                            .learningRate(1.0)
                            .updater(Updater.SGD)
                            .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,1))
                            .list()
                            .layer(0, new Convolution1DLayer.Builder()
                                    .kernelSize(kernel)
                                    .stride(stride)
                                    .padding(padding)
                                    .nIn(convNIn)
                                    .nOut(convNOut)
                                    .convolutionMode(ConvolutionMode.Same)
                                    .activation(afn)
                                    .build())
//                            .layer(1, new Subsampling1DLayer.Builder(poolingType)
//                                    .kernelSize(kernel)
//                                    .stride(stride)
//                                    .padding(padding)
//                                    .convolutionMode(ConvolutionMode.Same)
//                                    .pnorm(pnorm)
//                                    .build())
                            .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
                                    .nOut(finalNOut)
                                    .build())
                            .setInputType(InputType.recurrent(convNIn))
                            .build();

                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();

                    String msg = "PoolingType=" + poolingType + ", minibatch=" + minibatchSize + ", activationFn=" + afn;

                    if (PRINT_RESULTS) {
                        System.out.println(msg);
                        for (int j = 0; j < net.getnLayers(); j++)
                            System.out.println("Layer " + j + " # params: " + net.getLayer(j).numParams());
                    }

                    boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR, DEFAULT_MIN_ABS_ERROR,
                            PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

                    assertTrue(msg, gradOK);
                }
            }
        }
    }

    @Test
    public void testCnn2DWithSubsampling2D(){
        Nd4j.getRandom().setSeed(12345);

        int[] minibatchSizes = {3,1};
        int length = 7;
        int convNIn = 2;
        int convNOut = 3;

        int kernel = 2;
        int stride = 1;
        int padding = 0;
        int pnorm = 2;

        String[] activations = {"sigmoid","tanh"};
        SubsamplingLayer.PoolingType[] poolingTypes = new SubsamplingLayer.PoolingType[]{SubsamplingLayer.PoolingType.MAX, SubsamplingLayer.PoolingType.AVG, SubsamplingLayer.PoolingType.PNORM};

        for(String afn : activations) {
            for (SubsamplingLayer.PoolingType poolingType : poolingTypes) {
                for (int minibatchSize : minibatchSizes) {
                    INDArray input = Nd4j.rand(minibatchSize, convNIn*length*1).reshape(minibatchSize, convNIn, length, 1);
                    INDArray labels = Nd4j.zeros(minibatchSize, finalNOut).reshape(minibatchSize, finalNOut);
                    for (int i = 0; i < minibatchSize; i++) {
                            labels.putScalar(new int[]{i, i % convNOut}, 1.0);
//                        for (int j = 0; j < length; j++) {
//                            labels.putScalar(new int[]{i, i % convNOut, j}, 1.0);
//                        }
                    }

                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .regularization(false)
                            .learningRate(1.0)
                            .updater(Updater.SGD)
                            .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,1))
                            .list()
                            .layer(0, new ConvolutionLayer.Builder()
                                    .kernelSize(kernel, 1)
                                    .stride(stride, 1)
                                    .padding(padding, 1)
                                    .nIn(convNIn)
                                    .nOut(convNOut)
                                    .convolutionMode(ConvolutionMode.Same)
                                    .build())
//                            .layer(1, new Subsampling1DLayer.Builder(poolingType)
//                                    .kernelSize(kernel)
//                                    .stride(stride)
//                                    .padding(padding)
//                                    .convolutionMode(ConvolutionMode.Same)
//                                    .pnorm(pnorm)
//                                    .build())
//                            .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                              .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                    .activation(Activation.SOFTMAX)
                                    .nOut(finalNOut)
                                    .build())
//                            .setInputType(InputType.recurrent(convNIn))
                            .setInputType(InputType.convolutional(length, 1, convNIn))
                            .build();

                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();

                    String msg = "PoolingType=" + poolingType + ", minibatch=" + minibatchSize + ", activationFn=" + afn;

                    if (PRINT_RESULTS) {
                        System.out.println(msg);
                        for (int j = 0; j < net.getnLayers(); j++)
                            System.out.println("Layer " + j + " # params: " + net.getLayer(j).numParams());
                    }

                    boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR, DEFAULT_MIN_ABS_ERROR,
                            PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

                    assertTrue(msg, gradOK);
                }
            }
        }
    }
}
