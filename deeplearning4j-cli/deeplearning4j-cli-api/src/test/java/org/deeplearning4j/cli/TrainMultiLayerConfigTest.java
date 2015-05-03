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

package org.deeplearning4j.cli;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.cli.api.flags.Model;
import org.deeplearning4j.cli.subcommands.Train;
import org.deeplearning4j.nn.layers.feedforward.rbm.RBM;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.layers.convolution.preprocessor.ConvolutionPostProcessor;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;

import static org.junit.Assert.*;

/**
 * @author sonali
 */
public class TrainMultiLayerConfigTest {
    @Test
    public void testMultiLayerConfig() throws Exception {
        Model testModelFlag = new Model();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layerFactory(LayerFactories.getFactory(RBM.class)).dist(new NormalDistribution(1e-1,0))
                .list(4).preProcessor(0,new ConvolutionPostProcessor())
                .hiddenLayerSizes(3, 2, 2).build();
        String json = conf.toJson();

        FileUtils.writeStringToFile(new File("model_multi.json"), json);

        MultiLayerConfiguration from = testModelFlag.value("model_multi.json");
        assertEquals(conf,from);

        String[] cmd = {
                "--input", "iris.txt", "--model", "model_multi.json", "--output", "model_results.txt"
        };

        Train train = new Train(cmd);
    }

}
