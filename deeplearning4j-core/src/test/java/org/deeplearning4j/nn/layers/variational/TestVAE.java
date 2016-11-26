package org.deeplearning4j.nn.layers.variational;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.variational.*;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;

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

        int backpropParams = vae.initializer().numParams(c, true);
        int allParams = vae.initializer().numParams(c,false);

        MultiLayerNetwork net = new MultiLayerNetwork(mlc);
        net.init();



    }

}
