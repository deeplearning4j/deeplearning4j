package org.deeplearning4j.nn.layers.recurrent;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import static org.junit.Assert.assertTrue;

public class TestRecurrentWeightInit {

    @Test
    public void testRWInit() {

        for (boolean rwInit : new boolean[]{false, true}) {
            for (int i = 0; i < 3; i++) {

                NeuralNetConfiguration.ListBuilder b = new NeuralNetConfiguration.Builder()
                        .weightInit(new UniformDistribution(0, 1))
                        .list();

                if(rwInit) {
                    switch (i) {
                        case 0:
                            b.layer(new LSTM.Builder().nIn(10).nOut(10)
                                    .weightInitRecurrent(new UniformDistribution(2, 3))
                                    .build());
                            break;
                        case 1:
                            b.layer(new GravesLSTM.Builder().nIn(10).nOut(10)
                                    .weightInitRecurrent(new UniformDistribution(2, 3))
                                    .build());
                            break;
                        case 2:
                            b.layer(new SimpleRnn.Builder().nIn(10).nOut(10)
                                    .weightInitRecurrent(new UniformDistribution(2, 3)).build());
                            break;
                        default:
                            throw new RuntimeException();
                    }
                } else {
                    switch (i) {
                        case 0:
                            b.layer(new LSTM.Builder().nIn(10).nOut(10).build());
                            break;
                        case 1:
                            b.layer(new GravesLSTM.Builder().nIn(10).nOut(10).build());
                            break;
                        case 2:
                            b.layer(new SimpleRnn.Builder().nIn(10).nOut(10).build());
                            break;
                        default:
                            throw new RuntimeException();
                    }
                }

                MultiLayerNetwork net = new MultiLayerNetwork(b.build());
                net.init();

                INDArray rw = net.getParam("0_RW");
                double min = rw.minNumber().doubleValue();
                double max = rw.maxNumber().doubleValue();
                if(rwInit){
                    assertTrue(String.valueOf(min), min >= 2.0);
                    assertTrue(String.valueOf(max), max <= 3.0);
                } else {
                    assertTrue(String.valueOf(min), min >= 0.0);
                    assertTrue(String.valueOf(max), max <= 1.0);
                }
            }
        }
    }

}
