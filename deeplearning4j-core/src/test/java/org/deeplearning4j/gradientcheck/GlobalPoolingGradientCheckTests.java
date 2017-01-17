package org.deeplearning4j.gradientcheck;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Random;

import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 17/01/2017.
 */
public class GlobalPoolingGradientCheckTests {

    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;

    @Test
    public void testLSTMGlobalPoolingBasicMultiLayer() {
        //Basic test of GravesLSTM layer
        Nd4j.getRandom().setSeed(12345L);

        int timeSeriesLength = 10;
        int nIn = 5;
        int layerSize = 4;
        int nOut = 2;

        int[] minibatchSizes = new int[]{1,3};
        PoolingType[] poolingTypes = new PoolingType[]{PoolingType.AVG, PoolingType.SUM, PoolingType.MAX};
//        PoolingType[] poolingTypes = new PoolingType[]{PoolingType.AVG, PoolingType.SUM};

        for( int miniBatchSize : minibatchSizes) {
            for (PoolingType pt : poolingTypes) {

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .regularization(false)
                        .updater(Updater.NONE)
                        .seed(12345L)
                        .list()
                        .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(layerSize).activation(Activation.TANH)
                                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1.0)).build())
                        .layer(1, new GlobalPoolingLayer.Builder().poolingType(pt).build())
                        .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(layerSize).nOut(nOut)
                                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1.0)).build())
                        .pretrain(false).backprop(true)
                        .build();

                MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                mln.init();

                Random r = new Random(12345L);
                INDArray input = Nd4j.zeros(miniBatchSize, nIn, timeSeriesLength);
                for (int i = 0; i < miniBatchSize; i++) {
                    for (int j = 0; j < nIn; j++) {
                        for (int k = 0; k < timeSeriesLength; k++) {
                            input.putScalar(new int[]{i, j, k}, r.nextDouble() - 0.5);
                        }
                    }
                }

                INDArray labels = Nd4j.zeros(miniBatchSize, nOut);
                for (int i = 0; i < miniBatchSize; i++) {
                    int idx = r.nextInt(nOut);
                    labels.putScalar(i, idx, 1.0);
                }

                if (PRINT_RESULTS) {
                    System.out.println("testLSTMGlobalPoolingBasicMultiLayer() - " + pt + ", minibatch = " + miniBatchSize);
                    for (int j = 0; j < mln.getnLayers(); j++)
                        System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
                }

                boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR, DEFAULT_MIN_ABS_ERROR,
                        PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

                assertTrue(gradOK);

            }
        }
    }

}
