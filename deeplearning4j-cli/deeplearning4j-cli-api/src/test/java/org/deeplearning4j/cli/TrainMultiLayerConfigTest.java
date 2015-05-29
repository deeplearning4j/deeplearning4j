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
import org.deeplearning4j.cli.subcommands.Train;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.layers.convolution.preprocessor.ConvolutionPostProcessor;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.core.io.ClassPathResource;

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
                .layer(new RBM()).nIn(4).nOut(3)
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN).hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .iterations(100).weightInit(WeightInit.DISTRIBUTION).dist(new UniformDistribution(0, 1))
                .activationFunction("tanh").k(1).batchSize(10)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f).momentum(0.9).regularization(true).l2(2e-4)
                .optimizationAlgo(OptimizationAlgorithm.LBFGS).constrainGradientToUnitNorm(true)
                .list(2).hiddenLayerSizes(3)
                .override(1, new ClassifierOverride(1))
                .build();
        String json = conf.toJson();

        FileUtils.writeStringToFile(new File("model_multi.json"), json);

        MultiLayerConfiguration from = testModelFlag.value("model_multi.json");
        assertEquals(conf, from);


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

    }

}
