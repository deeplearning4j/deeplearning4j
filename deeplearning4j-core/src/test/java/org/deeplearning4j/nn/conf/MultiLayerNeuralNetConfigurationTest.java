/*-
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

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.weightnoise.DropConnect;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.util.Arrays;
import java.util.Properties;

import static org.junit.Assert.*;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class MultiLayerNeuralNetConfigurationTest extends BaseDL4JTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void testJson() throws Exception {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new DenseLayer.Builder().dist(new NormalDistribution(1, 1e-1)).build())
                        .inputPreProcessor(0, new CnnToFeedForwardPreProcessor()).build();

        String json = conf.toJson();
        MultiLayerConfiguration from = MultiLayerConfiguration.fromJson(json);
        assertEquals(conf.getConf(0), from.getConf(0));

        Properties props = new Properties();
        props.put("json", json);
        String key = props.getProperty("json");
        assertEquals(json, key);
        File f = testDir.newFile("props");
        f.deleteOnExit();
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(f));
        props.store(bos, "");
        bos.flush();
        bos.close();
        BufferedInputStream bis = new BufferedInputStream(new FileInputStream(f));
        Properties props2 = new Properties();
        props2.load(bis);
        bis.close();
        assertEquals(props2.getProperty("json"), props.getProperty("json"));
        String json2 = props2.getProperty("json");
        MultiLayerConfiguration conf3 = MultiLayerConfiguration.fromJson(json2);
        assertEquals(conf.getConf(0), conf3.getConf(0));

    }

    @Test
    public void testConvnetJson() {
        final int numRows = 76;
        final int numColumns = 76;
        int nChannels = 3;
        int outputNum = 6;
        int seed = 123;

        //setup the network
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().seed(seed)
                        .l1(1e-1).l2(2e-4).weightNoise(new DropConnect(0.5)).miniBatch(true)
                        .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT).list()
                        .layer(0, new ConvolutionLayer.Builder(5, 5).nOut(5).dropOut(0.5).weightInit(WeightInit.XAVIER)
                                        .activation(Activation.RELU).build())
                        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2, 2})
                                        .build())
                        .layer(2, new ConvolutionLayer.Builder(3, 3).nOut(10).dropOut(0.5).weightInit(WeightInit.XAVIER)
                                        .activation(Activation.RELU).build())
                        .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2, 2})
                                        .build())
                        .layer(4, new DenseLayer.Builder().nOut(100).activation(Activation.RELU).build())
                        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                        .nOut(outputNum).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX)
                                        .build())
                        .backprop(true).pretrain(false)
                        .setInputType(InputType.convolutional(numRows, numColumns, nChannels));

        MultiLayerConfiguration conf = builder.build();
        String json = conf.toJson();
        MultiLayerConfiguration conf2 = MultiLayerConfiguration.fromJson(json);
        assertEquals(conf, conf2);
    }

    @Test
    public void testUpsamplingConvnetJson() {
        final int numRows = 76;
        final int numColumns = 76;
        int nChannels = 3;
        int outputNum = 6;
        int seed = 123;

        //setup the network
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().seed(seed)
                .l1(1e-1).l2(2e-4).dropOut(0.5).miniBatch(true)
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT).list()
                .layer(new ConvolutionLayer.Builder(5, 5).nOut(5).dropOut(0.5).weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU).build())
                .layer(new Upsampling2D.Builder().size(2).build())
                .layer(2, new ConvolutionLayer.Builder(3, 3).nOut(10).dropOut(0.5).weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU).build())
                .layer(new Upsampling2D.Builder().size(2).build())
                .layer(4, new DenseLayer.Builder().nOut(100).activation(Activation.RELU).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX)
                        .build())
                .backprop(true).pretrain(false)
                .setInputType(InputType.convolutional(numRows, numColumns, nChannels));

        MultiLayerConfiguration conf = builder.build();
        String json = conf.toJson();
        MultiLayerConfiguration conf2 = MultiLayerConfiguration.fromJson(json);
        assertEquals(conf, conf2);
    }

    @Test
    public void testGlobalPoolingJson() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new NoOp())
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1.0)).seed(12345L).list()
                        .layer(0, new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1).nOut(5).build())
                        .layer(1, new GlobalPoolingLayer.Builder().poolingType(PoolingType.PNORM).pnorm(3).build())
                        .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .activation(Activation.SOFTMAX).nOut(3).build())
                        .pretrain(false).backprop(true).setInputType(InputType.convolutional(32, 32, 1)).build();

        String str = conf.toJson();
        MultiLayerConfiguration fromJson = conf.fromJson(str);

        assertEquals(conf, fromJson);
    }


    @Test
    public void testYaml() throws Exception {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new DenseLayer.Builder().dist(new NormalDistribution(1, 1e-1)).build())
                        .inputPreProcessor(0, new CnnToFeedForwardPreProcessor()).build();
        String json = conf.toYaml();
        MultiLayerConfiguration from = MultiLayerConfiguration.fromYaml(json);
        assertEquals(conf.getConf(0), from.getConf(0));

        Properties props = new Properties();
        props.put("json", json);
        String key = props.getProperty("json");
        assertEquals(json, key);
        File f = testDir.newFile("props");
        f.deleteOnExit();
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(f));
        props.store(bos, "");
        bos.flush();
        bos.close();
        BufferedInputStream bis = new BufferedInputStream(new FileInputStream(f));
        Properties props2 = new Properties();
        props2.load(bis);
        bis.close();
        assertEquals(props2.getProperty("json"), props.getProperty("json"));
        String yaml = props2.getProperty("json");
        MultiLayerConfiguration conf3 = MultiLayerConfiguration.fromYaml(yaml);
        assertEquals(conf.getConf(0), conf3.getConf(0));

    }

    @Test
    public void testClone() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list().layer(0, new DenseLayer.Builder().build())
                        .layer(1, new OutputLayer.Builder().build())
                        .inputPreProcessor(1, new CnnToFeedForwardPreProcessor()).build();

        MultiLayerConfiguration conf2 = conf.clone();

        assertEquals(conf, conf2);
        assertNotSame(conf, conf2);
        assertNotSame(conf.getConfs(), conf2.getConfs());
        for (int i = 0; i < conf.getConfs().size(); i++) {
            assertNotSame(conf.getConf(i), conf2.getConf(i));
        }
        assertNotSame(conf.getInputPreProcessors(), conf2.getInputPreProcessors());
        for (Integer layer : conf.getInputPreProcessors().keySet()) {
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
    public void testTrainingListener() {
        MultiLayerNetwork model1 = new MultiLayerNetwork(getConf());
        model1.init();
        model1.addListeners( new ScoreIterationListener(1));

        MultiLayerNetwork model2 = new MultiLayerNetwork(getConf());
        model2.addListeners( new ScoreIterationListener(1));
        model2.init();

        Layer[] l1 = model1.getLayers();
        for (int i = 0; i < l1.length; i++)
            assertTrue(l1[i].getListeners() != null && l1[i].getListeners().size() == 1);

        Layer[] l2 = model2.getLayers();
        for (int i = 0; i < l2.length; i++)
            assertTrue(l2[i].getListeners() != null && l2[i].getListeners().size() == 1);
    }


    private static MultiLayerConfiguration getConf() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345l).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 1)).build())
                        .layer(1, new OutputLayer.Builder().nIn(2).nOut(1).weightInit(WeightInit.DISTRIBUTION)
                                        .dist(new NormalDistribution(0, 1)).build())
                        .build();
        return conf;
    }

    @Test
    public void testInvalidConfig() {

        try {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).list().pretrain(false)
                            .backprop(true).build();
            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();
            fail("No exception thrown for invalid configuration");
        } catch (IllegalStateException e) {
            //OK
            e.printStackTrace();
        } catch (Throwable e) {
            e.printStackTrace();
            fail("Unexpected exception thrown for invalid config");
        }

        try {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).list()
                            .layer(1, new DenseLayer.Builder().nIn(3).nOut(4).build())
                            .layer(2, new OutputLayer.Builder().nIn(4).nOut(5).build()).pretrain(false).backprop(true)
                            .build();
            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();
            fail("No exception thrown for invalid configuration");
        } catch (IllegalStateException e) {
            //OK
            e.printStackTrace();
        } catch (Throwable e) {
            e.printStackTrace();
            fail("Unexpected exception thrown for invalid config");
        }

        try {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).list()
                            .layer(0, new DenseLayer.Builder().nIn(3).nOut(4).build())
                            .layer(2, new OutputLayer.Builder().nIn(4).nOut(5).build()).pretrain(false).backprop(true)
                            .build();
            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();
            fail("No exception thrown for invalid configuration");
        } catch (IllegalStateException e) {
            //OK
            e.printStackTrace();
        } catch (Throwable e) {
            e.printStackTrace();
            fail("Unexpected exception thrown for invalid config");
        }
    }

    @Test
    public void testListOverloads() {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).list()
                        .layer(0, new DenseLayer.Builder().nIn(3).nOut(4).build())
                        .layer(1, new OutputLayer.Builder().nIn(4).nOut(5).build()).pretrain(false).backprop(true)
                        .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        DenseLayer dl = (DenseLayer) conf.getConf(0).getLayer();
        assertEquals(3, dl.getNIn());
        assertEquals(4, dl.getNOut());
        OutputLayer ol = (OutputLayer) conf.getConf(1).getLayer();
        assertEquals(4, ol.getNIn());
        assertEquals(5, ol.getNOut());

        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder().seed(12345).list()
                        .layer(0, new DenseLayer.Builder().nIn(3).nOut(4).build())
                        .layer(1, new OutputLayer.Builder().nIn(4).nOut(5).build()).pretrain(false).backprop(true)
                        .build();
        MultiLayerNetwork net2 = new MultiLayerNetwork(conf2);
        net2.init();

        MultiLayerConfiguration conf3 = new NeuralNetConfiguration.Builder().seed(12345)
                        .list(new DenseLayer.Builder().nIn(3).nOut(4).build(),
                                        new OutputLayer.Builder().nIn(4).nOut(5).build())
                        .pretrain(false).backprop(true).build();
        MultiLayerNetwork net3 = new MultiLayerNetwork(conf3);
        net3.init();


        assertEquals(conf, conf2);
        assertEquals(conf, conf3);
    }

    @Test
    public void testPreBackFineValidation() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();

        assertFalse(conf.isPretrain());
        assertTrue(conf.isBackprop());

        conf = new NeuralNetConfiguration.Builder().list().layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).pretrain(true).backprop(false)
                        .build();

        assertTrue(conf.isPretrain());
        assertFalse(conf.isBackprop());
    }


    @Test
    public void testBiasLr() {
        //setup the network
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).updater(new Adam(1e-2))
                        .biasUpdater(new Adam(0.5)).list()
                        .layer(0, new ConvolutionLayer.Builder(5, 5).nOut(5).weightInit(WeightInit.XAVIER)
                                        .activation(Activation.RELU).build())
                        .layer(1, new DenseLayer.Builder().nOut(100).activation(Activation.RELU).build())
                        .layer(2, new DenseLayer.Builder().nOut(100).activation(Activation.RELU).build())
                        .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(10)
                                        .weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).build())
                        .setInputType(InputType.convolutional(28, 28, 1)).build();

        org.deeplearning4j.nn.conf.layers.BaseLayer l0 = (BaseLayer) conf.getConf(0).getLayer();
        org.deeplearning4j.nn.conf.layers.BaseLayer l1 = (BaseLayer) conf.getConf(1).getLayer();
        org.deeplearning4j.nn.conf.layers.BaseLayer l2 = (BaseLayer) conf.getConf(2).getLayer();
        org.deeplearning4j.nn.conf.layers.BaseLayer l3 = (BaseLayer) conf.getConf(3).getLayer();

        assertEquals(0.5, ((Adam)l0.getUpdaterByParam("b")).getLearningRate(), 1e-6);
        assertEquals(1e-2, ((Adam)l0.getUpdaterByParam("W")).getLearningRate(), 1e-6);

        assertEquals(0.5, ((Adam)l1.getUpdaterByParam("b")).getLearningRate(), 1e-6);
        assertEquals(1e-2, ((Adam)l1.getUpdaterByParam("W")).getLearningRate(), 1e-6);

        assertEquals(0.5, ((Adam)l2.getUpdaterByParam("b")).getLearningRate(), 1e-6);
        assertEquals(1e-2, ((Adam)l2.getUpdaterByParam("W")).getLearningRate(), 1e-6);

        assertEquals(0.5, ((Adam)l3.getUpdaterByParam("b")).getLearningRate(), 1e-6);
        assertEquals(1e-2, ((Adam)l3.getUpdaterByParam("W")).getLearningRate(), 1e-6);
    }

}
