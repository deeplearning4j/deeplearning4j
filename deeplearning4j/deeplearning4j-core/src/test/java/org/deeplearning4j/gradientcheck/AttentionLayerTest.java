package org.deeplearning4j.gradientcheck;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Random;

import static org.junit.Assert.assertTrue;

public class AttentionLayerTest extends BaseDL4JTest {
    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;

    @Test
    public void testSelfAttentionLayer() {
        int nIn = 3;
        int nOut = 5;
        int tsLength = 4;
        int layerSize = 8;

        Random r = new Random(12345);
        for (int mb : new int[]{1, 2, 3}) {
            for (boolean inputMask : new boolean[]{false, true}) {
                for (boolean projectInput : new boolean[]{false, true}) {
                    INDArray in = Nd4j.rand(new int[]{mb, nIn, tsLength});
                    INDArray labels = Nd4j.create(mb, nOut);
                    for (int i = 0; i < mb; i++) {
                        labels.putScalar(i, r.nextInt(nOut), 1.0);
                    }
                    String maskType = (inputMask ? "inputMask" : "none");

                    INDArray inMask = null;
                    if (inputMask) {
                        inMask = Nd4j.ones(mb, tsLength);
                        for (int i = 0; i < mb; i++) {
                            int firstMaskedStep = tsLength - 1 - i;
                            if (firstMaskedStep == 0) {
                                firstMaskedStep = tsLength;
                            }
                            for (int j = firstMaskedStep; j < tsLength; j++) {
                                inMask.putScalar(i, j, 0.0);
                            }
                        }
                    }

                    String name = "testSelfAttentionLayer() - mb=" + mb + ", tsLength = " + tsLength + ", maskType=" + maskType + ", projectInput = " + projectInput;
                    System.out.println("Starting test: " + name);


                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .activation(Activation.TANH)
                            .updater(new NoOp())
                            .weightInit(WeightInit.XAVIER)
                            .list()
                            .layer(new LSTM.Builder().nOut(layerSize).build())
                            .layer( projectInput ?
                                            new SelfAttentionLayer.Builder().nOut(8).nHeads(2).projectInput(true).build()
                                            : new SelfAttentionLayer.Builder().nHeads(1).projectInput(false).build()
                                    )
                            .layer(new GlobalPoolingLayer.Builder().poolingType(PoolingType.MAX).build())
                            .layer(new OutputLayer.Builder().nOut(nOut).activation(Activation.SOFTMAX)
                                    .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                            .setInputType(InputType.recurrent(nIn))
                            .build();

                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();

                    boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, in, labels, inMask, null);
                    assertTrue(name, gradOK);
                }
            }
        }
    }

    @Test
    public void testLearnedSelfAttentionLayer() {
        int nIn = 3;
        int nOut = 5;
        int tsLength = 4;
        int layerSize = 8;
        int numQueries = 6;

        Random r = new Random(12345);
        for (boolean inputMask : new boolean[]{false, true}) {
            for (int mb : new int[]{3, 2, 1}) {
                for (boolean projectInput : new boolean[]{false, true}) {
                    INDArray in = Nd4j.rand(new int[]{mb, nIn, tsLength});
                    INDArray labels = Nd4j.create(mb, nOut);
                    for (int i = 0; i < mb; i++) {
                        labels.putScalar(i, r.nextInt(nOut), 1.0);
                    }
                    String maskType = (inputMask ? "inputMask" : "none");

                    INDArray inMask = null;
                    if (inputMask) {
                        inMask = Nd4j.ones(mb, tsLength);
                        for (int i = 0; i < mb; i++) {
                            int firstMaskedStep = tsLength - 1 - i;
                            if (firstMaskedStep == 0) {
                                firstMaskedStep = tsLength;
                            }
                            for (int j = firstMaskedStep; j < tsLength; j++) {
                                inMask.putScalar(i, j, 0.0);
                            }
                        }
                    }

                    String name = "testLearnedSelfAttentionLayer() - mb=" + mb + ", tsLength = " + tsLength + ", maskType=" + maskType + ", projectInput = " + projectInput;
                    System.out.println("Starting test: " + name);


                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .activation(Activation.TANH)
                            .updater(new NoOp())
                            .weightInit(WeightInit.XAVIER)
                            .list()
                            .layer(new LSTM.Builder().nOut(layerSize).build())
                            .layer( projectInput ?
                                    new LearnedSelfAttentionLayer.Builder().nOut(8).nHeads(2).numQueries(numQueries).projectInput(true).build()
                                    : new LearnedSelfAttentionLayer.Builder().nHeads(1).numQueries(numQueries).projectInput(false).build()
                            )
                            .layer(new GlobalPoolingLayer.Builder().poolingType(PoolingType.MAX).build())
                            .layer(new OutputLayer.Builder().nOut(nOut).activation(Activation.SOFTMAX)
                                    .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                            .setInputType(InputType.recurrent(nIn))
                            .build();

                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();

                    boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, in, labels, inMask, null);
                    assertTrue(name, gradOK);
                }
            }
        }
    }

    @Test
    public void testRecurrentAttentionLayer() {
        int nIn = 9;
        int nOut = 5;
        int tsLength = 2;
        int layerSize = 8;


        Random r = new Random(12345);
        for (int mb : new int[]{3, 2, 1}) {
            for (boolean inputMask : new boolean[]{false, true}) {
                INDArray in = Nd4j.rand(new int[]{mb, nIn, tsLength});
                INDArray labels = Nd4j.create(mb, nOut);
                for (int i = 0; i < mb; i++) {
                    labels.putScalar(i, r.nextInt(nOut), 1.0);
                }
                String maskType = (inputMask ? "inputMask" : "none");

                INDArray inMask = null;
                if (inputMask) {
                    inMask = Nd4j.ones(mb, tsLength);
                    for (int i = 0; i < mb; i++) {
                        int firstMaskedStep = tsLength - 1 - i;
                        if (firstMaskedStep == 0) {
                            firstMaskedStep = tsLength;
                        }
                        for (int j = firstMaskedStep; j < tsLength; j++) {
                            inMask.putScalar(i, j, 0.0);
                        }
                    }
                }

                String name = "testRecurrentAttentionLayer() - mb=" + mb + ", tsLength = " + tsLength + ", maskType=" + maskType;
                System.out.println("Starting test: " + name);


                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .activation(Activation.IDENTITY)
                        .updater(new NoOp())
                        .weightInit(WeightInit.XAVIER)
                        .list()
                        .layer(new LSTM.Builder().nOut(layerSize).build())
                        .layer(new RecurrentAttentionLayer.Builder().nIn(layerSize).nOut(layerSize).nHeads(1).projectInput(false).hasBias(false).build())
                        .layer(new GlobalPoolingLayer.Builder().poolingType(PoolingType.AVG).build())
                        .layer(new OutputLayer.Builder().nOut(nOut).activation(Activation.SOFTMAX)
                                .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                        .setInputType(InputType.recurrent(nIn))
                        .build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                //System.out.println("Original");
                boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, in, labels, inMask, null, false, -1, null
                        //Sets.newHashSet(  /*"1_b", "1_W",* "1_WR", "1_WQR", "1_WQ", "1_bQ",*/ "2_b", "2_W" ,"0_W", "0_RW", "0_b"/**/)
                );
                assertTrue(name, gradOK);
            }
        }
    }
}
