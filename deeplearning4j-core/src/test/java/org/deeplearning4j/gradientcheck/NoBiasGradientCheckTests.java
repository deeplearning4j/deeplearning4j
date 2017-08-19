package org.deeplearning4j.gradientcheck;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.*;
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
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Alex Black 14 Aug 2015
 */
public class NoBiasGradientCheckTests {

    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;

    static {
        Nd4j.zeros(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void testGradientNoBiasDenseOutput() {

        int nIn = 5;
        int nOut = 3;
        int layerSize = 6;

        for (int minibatch : new int[]{1, 4}) {
            INDArray input = Nd4j.rand(minibatch, nIn);
            INDArray labels = Nd4j.zeros(minibatch, nOut);
            for (int i = 0; i < minibatch; i++) {
                labels.putScalar(i, i % nOut, 1.0);
            }

            for (boolean denseNoBias : new boolean[]{false, true}) {
                for (boolean outNoBias : new boolean[]{false, true}) {

                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().regularization(false)
                            .updater(Updater.NONE)
                            .seed(12345L)
                            .list()
                            .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(layerSize)
                                    .weightInit(WeightInit.DISTRIBUTION)
                                    .dist(new NormalDistribution(0, 1))
                                    .activation(Activation.TANH)
                                    .noBias(false)  //Layer 0: Always have a bias
                                    .build())
                            .layer(1, new DenseLayer.Builder().nIn(layerSize).nOut(layerSize)
                                    .weightInit(WeightInit.DISTRIBUTION)
                                    .dist(new NormalDistribution(0, 1))
                                    .activation(Activation.TANH)
                                    .noBias(denseNoBias)
                                    .build())
                            .layer(2, new OutputLayer.Builder(LossFunction.MCXENT)
                                    .activation(Activation.SOFTMAX).nIn(layerSize).nOut(nOut)
                                    .weightInit(WeightInit.DISTRIBUTION)
                                    .dist(new NormalDistribution(0, 1))
                                    .noBias(outNoBias)
                                    .build())
                            .build();

                    MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                    mln.init();

                    if (denseNoBias) {
                        assertEquals(layerSize * layerSize, mln.getLayer(1).numParams());
                    } else {
                        assertEquals(layerSize * layerSize + layerSize, mln.getLayer(1).numParams());
                    }

                    if (outNoBias) {
                        assertEquals(layerSize * nOut, mln.getLayer(2).numParams());
                    } else {
                        assertEquals(layerSize * nOut + nOut, mln.getLayer(2).numParams());
                    }

                    String msg = "testGradientNoBiasDenseOutput() denseNoBias = " + denseNoBias + ", outNoBias = " + outNoBias + ")";

                    if (PRINT_RESULTS) {
                        System.out.println(msg);
                    }

                    boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);
                    assertTrue(msg, gradOK);
                }
            }
        }
    }


    @Test
    public void testGradientNoBiasEmbedding() {

        int nIn = 5;
        int nOut = 3;
        int layerSize = 6;

        for (int minibatch : new int[]{1, 4}) {
            INDArray input = Nd4j.zeros(minibatch, 1);
            for( int i=0; i<minibatch; i++ ){
                input.putScalar(i, 0, i%layerSize);
            }
            INDArray labels = Nd4j.zeros(minibatch, nOut);
            for (int i = 0; i < minibatch; i++) {
                labels.putScalar(i, i % nOut, 1.0);
            }

            for (boolean embeddingNoBias : new boolean[]{false, true}) {

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().regularization(false)
                        .updater(Updater.NONE)
                        .seed(12345L)
                        .list()
                        .layer(0, new EmbeddingLayer.Builder().nIn(nIn).nOut(layerSize)
                                .weightInit(WeightInit.DISTRIBUTION)
                                .dist(new NormalDistribution(0, 1))
                                .activation(Activation.TANH)
                                .noBias(embeddingNoBias)
                                .build())
                        .layer(1, new OutputLayer.Builder(LossFunction.MCXENT)
                                .activation(Activation.SOFTMAX).nIn(layerSize).nOut(nOut)
                                .weightInit(WeightInit.DISTRIBUTION)
                                .dist(new NormalDistribution(0, 1))
                                .build())
                        .build();

                MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                mln.init();

                if (embeddingNoBias) {
                    assertEquals(nIn * layerSize, mln.getLayer(0).numParams());
                } else {
                    assertEquals(nIn * layerSize + layerSize, mln.getLayer(0).numParams());
                }

                String msg = "testGradientNoBiasEmbedding() denseNoBias = " + embeddingNoBias + ")";

                if (PRINT_RESULTS) {
                    System.out.println(msg);
                }

                boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);
                assertTrue(msg, gradOK);
            }
        }
    }
}
