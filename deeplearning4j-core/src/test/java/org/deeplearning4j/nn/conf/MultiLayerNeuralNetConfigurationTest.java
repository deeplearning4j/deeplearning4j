/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.conf;

import static org.junit.Assert.*;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.conf.preprocessor.ReshapePreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.util.Arrays;
import java.util.Collections;
import java.util.Properties;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class MultiLayerNeuralNetConfigurationTest {

    @Test
    public void testJson() throws Exception {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0,new RBM.Builder().dist(new NormalDistribution(1, 1e-1)).build())
                .inputPreProcessor(0, new ReshapePreProcessor())
                .build();

        String json = conf.toJson();
        MultiLayerConfiguration from = MultiLayerConfiguration.fromJson(json);
        assertEquals(conf.getConf(0),from.getConf(0));

        Properties props = new Properties();
        props.put("json",json);
        String key = props.getProperty("json");
        assertEquals(json,key);
        File f = new File("props");
        f.deleteOnExit();
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(f));
        props.store(bos,"");
        bos.flush();
        bos.close();
        BufferedInputStream bis = new BufferedInputStream(new FileInputStream(f));
        Properties props2 = new Properties();
        props2.load(bis);
        bis.close();
        assertEquals(props2.getProperty("json"),props.getProperty("json"));
        String json2 = props2.getProperty("json");
        MultiLayerConfiguration conf3 = MultiLayerConfiguration.fromJson(json2);
        assertEquals(conf.getConf(0),conf3.getConf(0));

    }

    @Test
    public void testConvnetJson() {
        final int numRows = 75;
        final int numColumns = 75;
        int nChannels = 3;
        int outputNum = 6;
        int batchSize = 500;
        int iterations = 10;
        int seed = 123;

        //setup the network
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations).regularization(true)
                .l1(1e-1).l2(2e-4).useDropConnect(true)
                .miniBatch(true)
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nOut(5).dropOut(0.5)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())

                .layer(1, new SubsamplingLayer
                        .Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .build())
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .nOut(10).dropOut(0.5)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(3, new SubsamplingLayer
                        .Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .build())
                .layer(4, new DenseLayer.Builder().nOut(100).activation("relu")
                        .build())

                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);

        new ConvolutionLayerSetup(builder,numRows,numColumns,nChannels);
        MultiLayerConfiguration conf = builder.build();
        String json = conf.toJson();
        MultiLayerConfiguration conf2 = MultiLayerConfiguration.fromJson(json);
        assertEquals(conf, conf2);
    }


    @Test
    public void testYaml() throws Exception {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, new RBM.Builder().dist(new NormalDistribution(1, 1e-1)).build())
                .inputPreProcessor(0, new ReshapePreProcessor())
                .build();
        String json = conf.toYaml();
        MultiLayerConfiguration from = MultiLayerConfiguration.fromYaml(json);
        assertEquals(conf.getConf(0),from.getConf(0));

        Properties props = new Properties();
        props.put("json",json);
        String key = props.getProperty("json");
        assertEquals(json,key);
        File f = new File("props");
        f.deleteOnExit();
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(f));
        props.store(bos,"");
        bos.flush();
        bos.close();
        BufferedInputStream bis = new BufferedInputStream(new FileInputStream(f));
        Properties props2 = new Properties();
        props2.load(bis);
        bis.close();
        assertEquals(props2.getProperty("json"),props.getProperty("json"));
        String yaml = props2.getProperty("json");
        MultiLayerConfiguration conf3 = MultiLayerConfiguration.fromYaml(yaml);
        assertEquals(conf.getConf(0),conf3.getConf(0));

    }

    @Test
    public void testClone() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, new RBM.Builder().build())
                .layer(1, new OutputLayer.Builder().build())
                .inputPreProcessor(1, new ReshapePreProcessor(new int[] {1,2}, new int[] {3,4}))
                .build();

        MultiLayerConfiguration conf2 = conf.clone();

        assertEquals(conf, conf2);
        assertNotSame(conf, conf2);
        assertNotSame(conf.getConfs(), conf2.getConfs());
        for(int i = 0; i < conf.getConfs().size(); i++) {
            assertNotSame(conf.getConf(i), conf2.getConf(i));
        }
        assertNotSame(conf.getInputPreProcessors(), conf2.getInputPreProcessors());
        for(Integer layer : conf.getInputPreProcessors().keySet()) {
            assertNotSame(conf.getInputPreProcess(layer), conf2.getInputPreProcess(layer));
        }
    }

    @Test
    public void testRandomWeightInit() {
        MultiLayerNetwork model1 = new MultiLayerNetwork(getConf());
        model1.init();

        Nd4j.getRandom().setSeed(12345L);
        MultiLayerNetwork model2 = new MultiLayerNetwork(getConf());
        model2.init();

        float[] p1 = model1.params().data().asFloat();
        float[] p2 = model2.params().data().asFloat();
        System.out.println(Arrays.toString(p1));
        System.out.println(Arrays.toString(p2));

        org.junit.Assert.assertArrayEquals(p1, p2, 0.0f);
    }

    @Test
    public void testIterationListener(){
        MultiLayerNetwork model1 = new MultiLayerNetwork(getConf());
        model1.init();
        model1.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(1)));

        MultiLayerNetwork model2 = new MultiLayerNetwork(getConf());
        model2.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(1)));
        model2.init();

        Layer[] l1 = model1.getLayers();
        for( int i = 0; i < l1.length; i++ )
            assertTrue(l1[i].getListeners() != null && l1[i].getListeners().size() == 1);

        Layer[] l2 = model2.getLayers();
        for( int i = 0; i < l2.length; i++ )
            assertTrue(l2[i].getListeners() != null && l2[i].getListeners().size() == 1);
    }


    private static MultiLayerConfiguration getConf(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345l)
                .list()
                .layer(0, new RBM.Builder()
                        .nIn(2).nOut(2)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                        .build())
                .layer(1, new OutputLayer.Builder()
                        .nIn(2).nOut(1)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
                        .build())
                .build();
        return conf;
    }

    @Test
    public void testInvalidConfig(){

        try {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .list()
                    .pretrain(false).backprop(true)
                    .build();
            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();
            fail("No exception thrown for invalid configuration");
        } catch(IllegalStateException e){
            //OK
            e.printStackTrace();
        } catch(Throwable e){
            e.printStackTrace();
            fail("Unexpected exception thrown for invalid config");
        }

        try {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .list()
                    .layer(1, new DenseLayer.Builder().nIn(3).nOut(4).build())
                    .layer(2, new OutputLayer.Builder().nIn(4).nOut(5).build())
                    .pretrain(false).backprop(true)
                    .build();
            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();
            fail("No exception thrown for invalid configuration");
        } catch(IllegalStateException e){
            //OK
            e.printStackTrace();
        } catch(Throwable e){
            e.printStackTrace();
            fail("Unexpected exception thrown for invalid config");
        }

        try {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(3).nOut(4).build())
                    .layer(2, new OutputLayer.Builder().nIn(4).nOut(5).build())
                    .pretrain(false).backprop(true)
                    .build();
            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();
            fail("No exception thrown for invalid configuration");
        } catch(IllegalStateException e){
            //OK
            e.printStackTrace();
        } catch(Throwable e){
            e.printStackTrace();
            fail("Unexpected exception thrown for invalid config");
        }
    }

    @Test
    public void testListOverloads(){

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(3).nOut(4).build())
                .layer(1, new OutputLayer.Builder().nIn(4).nOut(5).build())
                .pretrain(false).backprop(true)
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        DenseLayer dl = (DenseLayer)conf.getConf(0).getLayer();
        assertEquals(3,dl.getNIn());
        assertEquals(4,dl.getNOut());
        OutputLayer ol = (OutputLayer)conf.getConf(1).getLayer();
        assertEquals(4,ol.getNIn());
        assertEquals(5,ol.getNOut());

        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(3).nOut(4).build())
                .layer(1, new OutputLayer.Builder().nIn(4).nOut(5).build())
                .pretrain(false).backprop(true)
                .build();
        MultiLayerNetwork net2 = new MultiLayerNetwork(conf2);
        net2.init();

        MultiLayerConfiguration conf3 = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .list(
                    new DenseLayer.Builder().nIn(3).nOut(4).build(),
                    new OutputLayer.Builder().nIn(4).nOut(5).build())
                .pretrain(false).backprop(true)
                .build();
        MultiLayerNetwork net3 = new MultiLayerNetwork(conf3);
        net3.init();


        assertEquals(conf, conf2);
        assertEquals(conf, conf3);
    }

}
