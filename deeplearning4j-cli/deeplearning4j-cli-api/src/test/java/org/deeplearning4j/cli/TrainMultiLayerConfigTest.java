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
import org.deeplearning4j.cli.driver.CommandLineInterfaceDriver;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;

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
                .iterations(100)
                .learningRate(1e-1f).momentum(0.9).regularization(true).l2(2e-4)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(0, new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
                        .nIn(4).nOut(3)
                        .activation("tanh")
                        .weightInit(WeightInit.XAVIER)
                        .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(3).nOut(3)
                        .activation("softmax")
                        .weightInit(WeightInit.XAVIER)
                        .build()).pretrain(true).backprop(false)
                .build();
        String json = conf.toJson();

        FileUtils.writeStringToFile(new File("model_multi.json"), json);

        MultiLayerConfiguration from = testModelFlag.value("model_multi.json");
        assertEquals(conf, from);
        File parent = new File(System.getProperty("java.io.tmpdir"),"data");
        FileUtils.copyFile(new ClassPathResource("data/irisSvmLight.txt").getFile(),new File(parent,"irisSvmLight.txt"));


        CommandLineInterfaceDriver driver = new CommandLineInterfaceDriver();

        String[] cmd = {
                "train","-conf",
                new ClassPathResource("confs/cli_train_unit_test_conf.txt").getFile().getAbsolutePath(),
                "-input", new ClassPathResource("iris.txt").getFile().getAbsolutePath()
                , "-model", "model_multi.json"
                , "-output", "model_results.txt"
                ,"-verbose"
        };

        driver.doMain(cmd);
        FileUtils.deleteDirectory(parent);

    }

}
