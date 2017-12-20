package org.deeplearning4j.samediff;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.samediff.testlayers.SameDiffDense;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class SameDiffTest {

    @Test
    public void testSameDiffDenseBasic(){

        int nIn = 3;
        int nOut = 4;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new SameDiffDense.Builder().nIn(nIn).nOut(nOut).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        Map<String,INDArray> pt1 = net.getLayer(0).paramTable();
        assertNotNull(pt1);
        assertEquals(2, pt1.size());
        assertNotNull(pt1.get(DefaultParamInitializer.WEIGHT_KEY));
        assertNotNull(pt1.get(DefaultParamInitializer.BIAS_KEY));

        assertArrayEquals(new int[]{nIn, nOut}, pt1.get(DefaultParamInitializer.WEIGHT_KEY).shape());
        assertArrayEquals(new int[]{1, nOut}, pt1.get(DefaultParamInitializer.BIAS_KEY).shape());
    }

    @Test
    public void testSameDiffDenseForward(){

        int nIn = 3;
        int nOut = 4;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new SameDiffDense.Builder().nIn(nIn).nOut(nOut).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        Map<String,INDArray> pt1 = net.paramTable();
        assertNotNull(pt1);

        System.out.println(pt1);

//        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
//                .list()
//                .layer(new DenseLayer.Builder().activation(Activation.SIGMOID).nIn(nIn).nOut(nOut).build())
//                .build();
//
//        MultiLayerNetwork net2 = new MultiLayerNetwork(conf2);
//        net2.init();



    }

}
