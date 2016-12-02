package org.deeplearning4j.nn.layers.variational;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.variational.*;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 26/11/2016.
 */
public class TestVAE {

    @Test
    public void testInitialization() {

        MultiLayerConfiguration mlc = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, new org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder.Builder()
                        .nIn(10).nOut(5).encoderLayerSizes(12).decoderLayerSizes(13).build())
                .build();

        NeuralNetConfiguration c = mlc.getConf(0);
        org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder vae = (org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder) c.getLayer();

        int allParams = vae.initializer().numParams(c);

        //                  Encoder         Encoder -> p(z|x)       Decoder         //p(x|z)
        int expNumParams = (10 * 12 + 12) + (12 * (2*5) + (2*5)) + (5 * 13 + 13) + (13 * (2*10) + (2*10));
        assertEquals(expNumParams, allParams);

        MultiLayerNetwork net = new MultiLayerNetwork(mlc);
        net.init();

        System.out.println("Exp num params: " + expNumParams);
        assertEquals(expNumParams, net.getLayer(0).params().length());
        Map<String,INDArray> paramTable = net.getLayer(0).paramTable();
        int count = 0;
        for(INDArray arr : paramTable.values()){
            count += arr.length();
        }
        assertEquals(expNumParams, count);

        assertEquals(expNumParams, net.getLayer(0).numParams());
    }

    @Test
    public void testForwardPass() {

        int[][] encLayerSizes = new int[][]{ {12}, {12,13}, {12,13,14}};
        for( int i=0; i<encLayerSizes.length; i++ ) {

            MultiLayerConfiguration mlc = new NeuralNetConfiguration.Builder()
                    .list()
                    .layer(0, new org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder.Builder()
                            .nIn(10).nOut(5).encoderLayerSizes(encLayerSizes[i]).decoderLayerSizes(13).build())
                    .build();

            NeuralNetConfiguration c = mlc.getConf(0);
            org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder vae = (org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder) c.getLayer();

            MultiLayerNetwork net = new MultiLayerNetwork(mlc);
            net.init();

            INDArray in = Nd4j.rand(1, 10);

//        net.output(in);
            List<INDArray> out = net.feedForward(in);
            assertArrayEquals(new int[]{1, 10}, out.get(0).shape());
            assertArrayEquals(new int[]{1, 5}, out.get(1).shape());
        }
    }

    @Test
    public void testPretrainSimple(){

        int inputSize = 3;

        MultiLayerConfiguration mlc = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, new org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder.Builder()
                        .nIn(inputSize).nOut(4).encoderLayerSizes(5).decoderLayerSizes(6).build())
                .pretrain(true).backprop(false)
                .build();

        NeuralNetConfiguration c = mlc.getConf(0);
        org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder vae = (org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder) c.getLayer();

        int allParams = vae.initializer().numParams(c);

        MultiLayerNetwork net = new MultiLayerNetwork(mlc);
        net.init();
        net.initGradientsView();        //TODO this should happen automatically

        Map<String,INDArray> paramTable = net.getLayer(0).paramTable();
        Map<String,INDArray> gradTable = ((org.deeplearning4j.nn.layers.variational.VariationalAutoencoder)net.getLayer(0)).getGradientViews();

        assertEquals(paramTable.keySet(), gradTable.keySet());
        for(String s : paramTable.keySet()){
            assertEquals(paramTable.get(s).length(), gradTable.get(s).length());
            assertArrayEquals(paramTable.get(s).shape(), gradTable.get(s).shape());
        }

        System.out.println("Num params: " + net.numParams());

        INDArray data = Nd4j.rand(1, inputSize);


        net.fit(data);



    }

}
