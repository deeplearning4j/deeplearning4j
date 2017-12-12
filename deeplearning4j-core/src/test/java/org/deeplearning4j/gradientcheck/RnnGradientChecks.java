package org.deeplearning4j.gradientcheck;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.NoOp;

import java.util.Random;

import static org.junit.Assert.assertTrue;

public class RnnGradientChecks {

    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;

    static {
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void testBidirectionalWrapper(){

        int nIn = 3;
        int nOut = 5;
        int tsLength = 4;

        Bidirectional.Mode[] modes = new Bidirectional.Mode[]{Bidirectional.Mode.CONCAT, Bidirectional.Mode.ADD,
                Bidirectional.Mode.AVERAGE, Bidirectional.Mode.MUL};

        Random r = new Random(12345);
        for( int mb : new int[]{1, 3}) {
            for(boolean inputMask : new boolean[]{false, true}){

                INDArray in = Nd4j.rand(new int[]{mb, nIn, tsLength });
                INDArray labels = Nd4j.create(mb, nOut, tsLength);
                for( int i=0; i<mb; i++ ){
                    for( int j=0; j<tsLength; j++ ){
                        labels.putScalar(i,r.nextInt(nOut),j, 1.0);
                    }
                }
                String maskType = (inputMask ? "inputMask" : "none");

                INDArray inMask = null;
                if(inputMask){
                    inMask = Nd4j.ones(mb, tsLength);
                    for( int i=0; i<mb; i++ ){
                        int firstMaskedStep = tsLength - 1 - i;
                        if(firstMaskedStep == 0){
                            firstMaskedStep = tsLength;
                        }
                        for(int j=firstMaskedStep; j<tsLength; j++ ){
                            inMask.putScalar(i, j, 1.0);
                        }
                    }
                }

                for (Bidirectional.Mode m : modes) {
                    String name = "mb=" + mb + ", maskType=" + maskType + ", mode=" + m;

                    System.out.println("Starting test: " + name);

                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .updater(new NoOp())
                            .weightInit(WeightInit.XAVIER)
                            .list()
                            .layer(new LSTM.Builder().nIn(nIn).nOut(3).build())
                            .layer(new Bidirectional(m, new LSTM.Builder().nIn(3).nOut(3).build()))
                            .layer(new RnnOutputLayer.Builder().nOut(nOut).build())
                            .build();

                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();


                    boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, in, labels, inMask, null);
                    assertTrue(gradOK);
                }
            }
        }

    }

}
