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
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.ReshapePreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

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
                .layer(new RBM.Builder().dist(new NormalDistribution(1, 1e-1)).build())
                .list(2).inputPreProcessor(1, new ReshapePreProcessor())
                .build();

        String json = conf.toJson();
        MultiLayerConfiguration from = MultiLayerConfiguration.fromJson(json);
        assertEquals(conf.getConf(1),from.getConf(1));

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
    public void testYaml() throws Exception {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM.Builder().dist(new NormalDistribution(1, 1e-1)).build())
                .list(2).inputPreProcessor(1, new ReshapePreProcessor())
                .build();
        String json = conf.toYaml();
        MultiLayerConfiguration from = MultiLayerConfiguration.fromYaml(json);
        assertEquals(conf.getConf(1),from.getConf(1));

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
                .list(2)
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

        org.junit.Assert.assertArrayEquals(p1,p2,0.0f);
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
                .list(2)
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

}
