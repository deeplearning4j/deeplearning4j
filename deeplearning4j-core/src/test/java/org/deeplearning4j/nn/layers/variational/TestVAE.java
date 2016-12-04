package org.deeplearning4j.nn.layers.variational;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.variational.*;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

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


    @Test
    public void testParamGradientOrderAndViews() {
        Nd4j.getRandom().setSeed(12345);
        MultiLayerConfiguration mlc = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, new org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder.Builder()
                        .nIn(10).nOut(5).encoderLayerSizes(12,13).decoderLayerSizes(14,15).build())
                .build();

        NeuralNetConfiguration c = mlc.getConf(0);
        org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder vae = (org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder) c.getLayer();

        MultiLayerNetwork net = new MultiLayerNetwork(mlc);
        net.init();

        net.initGradientsView();

        org.deeplearning4j.nn.layers.variational.VariationalAutoencoder layer = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) net.getLayer(0);

        Map<String,INDArray> layerParams = layer.paramTable();
        Map<String,INDArray> layerGradViews = layer.getGradientViews();

        layer.setInput(Nd4j.rand(3,10));
        layer.computeGradientAndScore();;
        Gradient g = layer.gradient();
        Map<String,INDArray> grads = g.gradientForVariable();

        assertEquals(layerParams.size(), layerGradViews.size());
        assertEquals(layerParams.size(), grads.size());

        //Iteration order should be consistent due to linked hashmaps
        Iterator<String> pIter = layerParams.keySet().iterator();
        Iterator<String> gvIter = layerGradViews.keySet().iterator();
        Iterator<String> gIter = grads.keySet().iterator();

        while(pIter.hasNext()){
            String p = pIter.next();
            String gv = gvIter.next();
            String gr = gIter.next();

//            System.out.println(p + "\t" + gv + "\t" + gr);

            assertEquals(p, gv);
            assertEquals(p, gr);

            INDArray pArr = layerParams.get(p);
            INDArray gvArr = layerGradViews.get(p);
            INDArray gArr = grads.get(p);

            assertArrayEquals(pArr.shape(), gvArr.shape());
            assertTrue(gvArr == gArr);  //Should be the exact same object due to view mechanics
        }
    }


    @Test
    public void testPretrainParamsDuringBackprop(){
        //Idea: pretrain-specific parameters shouldn't change during backprop

        Nd4j.getRandom().setSeed(12345);
        MultiLayerConfiguration mlc = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .list()
                .layer(0, new org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder.Builder()
                        .nIn(10).nOut(5).encoderLayerSizes(12,13).decoderLayerSizes(14,15).build())
                .layer(1, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(5).nOut(6).activation("tanh").build())
                .pretrain(true).backprop(true)
                .build();

        NeuralNetConfiguration c = mlc.getConf(0);
        org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder vae = (org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder) c.getLayer();

        MultiLayerNetwork net = new MultiLayerNetwork(mlc);
        net.init();

        net.initGradientsView();

        org.deeplearning4j.nn.layers.variational.VariationalAutoencoder layer = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) net.getLayer(0);

        INDArray input = Nd4j.rand(3, 10);
//        layer.fit(input);
        net.pretrain(input);

        //Get a snapshot of the pretrain params after fitting:
        Map<String,INDArray> layerParams = layer.paramTable();
        Map<String,INDArray> pretrainParamsBefore = new HashMap<>();
        for(String s : layerParams.keySet()){
            if(layer.isPretrainParam(s)){
                pretrainParamsBefore.put(s, layerParams.get(s).dup());
            }
        }


        INDArray features = Nd4j.rand(3, 10);
        INDArray labels = Nd4j.rand(3, 6);

        net.getLayerWiseConfigurations().setPretrain(false);

        for( int i=0; i<3; i++ ){
            net.fit(features, labels);
        }

        Map<String,INDArray> layerParamsAfter = layer.paramTable();

        for(String s : pretrainParamsBefore.keySet()){
            INDArray before = pretrainParamsBefore.get(s);
            INDArray after = layerParamsAfter.get(s);
            assertEquals(before, after);
        }
    }
}
