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

import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.conf.rng.DefaultRandom;
import org.deeplearning4j.nn.layers.convolution.preprocessor.ConvolutionPostProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.SerializationUtils;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.Arrays;
import java.util.Properties;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class MultiLayerNeuralNetConfigurationTest {

    @Test
    public void testJson() throws Exception {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM()).dist(new NormalDistribution(1,1e-1))
                .list(4).preProcessor(0,new ConvolutionPostProcessor())
                .hiddenLayerSizes(3, 2, 2).build();
        String json = conf.toJson();
        MultiLayerConfiguration from = MultiLayerConfiguration.fromJson(json);
        assertEquals(conf,from);

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

        MultiLayerConfiguration conf3 = MultiLayerConfiguration.fromJson(props2.getProperty("json"));
        assertEquals(conf,conf3);

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




    private static MultiLayerConfiguration getConf(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM())
                .nIn(2).nOut(1)
                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,1))
                .rng(new DefaultRandom(12345L)) //RNG with specified seed
                .list(2)
                .hiddenLayerSizes(5)
                .override(1, new ClassifierOverride())
                .build();
        return conf;
    }

}
