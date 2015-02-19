/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.deeplearning4j.nn.conf;

import static org.junit.Assert.*;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.layers.convolution.preprocessor.ConvolutionPostProcessor;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.junit.Test;

import java.io.*;
import java.util.Properties;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class MultiLayerNeuralNetConfigurationTest {

    @Test
    public void testJson() throws Exception {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layerFactory(LayerFactories.getFactory(RBM.class)).dist(Distributions.normal(new MersenneTwister(123),1e-1))
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


}
