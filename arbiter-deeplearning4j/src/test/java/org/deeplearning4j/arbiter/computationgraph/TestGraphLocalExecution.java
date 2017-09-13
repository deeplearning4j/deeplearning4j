/*-
 *
 *  * Copyright 2016 Skymind,Inc.
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
package org.deeplearning4j.arbiter.computationgraph;

import org.deeplearning4j.arbiter.ComputationGraphSpace;
import org.deeplearning4j.arbiter.conf.updater.AdamSpace;
import org.deeplearning4j.arbiter.conf.updater.SgdSpace;
import org.deeplearning4j.arbiter.evaluator.multilayer.ClassificationEvaluator;
import org.deeplearning4j.arbiter.layers.DenseLayerSpace;
import org.deeplearning4j.arbiter.layers.OutputLayerSpace;
import org.deeplearning4j.arbiter.multilayernetwork.MnistDataSetIteratorFactory;
import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.data.DataSetIteratorFactoryProvider;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxTimeCondition;
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.discrete.DiscreteParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.saver.local.FileModelSaver;
import org.deeplearning4j.arbiter.scoring.ScoreFunctions;
import org.deeplearning4j.arbiter.task.ComputationGraphTaskCreator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculatorCG;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;


public class TestGraphLocalExecution {

    @Test
    @Ignore
    public void testLocalExecution() throws Exception {
        Map<String, Object> commands = new HashMap<>();
        commands.put(DataSetIteratorFactoryProvider.FACTORY_KEY, MnistDataSetIteratorFactory.class.getCanonicalName());

        //Define: network config (hyperparameter space)
        ComputationGraphSpace mls = new ComputationGraphSpace.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new SgdSpace(new ContinuousParameterSpace(0.0001, 0.1)))
                .l2(new ContinuousParameterSpace(0.0001, 0.01)).iterations(100).addInputs("in")
                .setInputTypes(InputType.feedForward(4))
                .addLayer("layer0",
                        new DenseLayerSpace.Builder().nIn(784).nOut(new IntegerParameterSpace(2, 10))
                                .activation(new DiscreteParameterSpace<>(Activation.RELU,Activation.TANH))
                                .build(),
                        "in")
                .addLayer("out", new OutputLayerSpace.Builder().nOut(10).activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build(), "layer0")
                .setOutputs("out").numEpochs(3).pretrain(false).backprop(true).build();

        //Define configuration:
        CandidateGenerator candidateGenerator = new RandomSearchGenerator(mls, commands);
        DataProvider dataProvider = new DataSetIteratorFactoryProvider();

        String modelSavePath = new File(System.getProperty("java.io.tmpdir"), "ArbiterDL4JTest\\").getAbsolutePath();

        File f = new File(modelSavePath);
        if (f.exists())
            f.delete();
        f.mkdir();
        f.deleteOnExit();
        if (!f.exists())
            throw new RuntimeException();

        OptimizationConfiguration configuration = new OptimizationConfiguration.Builder()
                        .candidateGenerator(candidateGenerator).dataProvider(dataProvider)
                        .modelSaver(new FileModelSaver(modelSavePath)).scoreFunction(ScoreFunctions.testSetLoss(true))
                        .terminationConditions(new MaxTimeCondition(2, TimeUnit.MINUTES),
                                        new MaxCandidatesCondition(100))
                        .build();

        IOptimizationRunner runner = new LocalOptimizationRunner(configuration,
                        new ComputationGraphTaskCreator(new ClassificationEvaluator()));

        runner.execute();

        System.out.println("----- COMPLETE -----");
    }

    @Test
    @Ignore
    public void testLocalExecutionEarlyStopping() throws Exception {
        EarlyStoppingConfiguration<ComputationGraph> esConf = new EarlyStoppingConfiguration.Builder<ComputationGraph>()
                        .epochTerminationConditions(new MaxEpochsTerminationCondition(100))
                        .scoreCalculator(new DataSetLossCalculatorCG(new MnistDataSetIterator(128, 1280), true))
                        .modelSaver(new InMemoryModelSaver()).build();
        Map<String, Object> commands = new HashMap<>();
        commands.put(DataSetIteratorFactoryProvider.FACTORY_KEY, MnistDataSetIteratorFactory.class.getCanonicalName());

        //Define: network config (hyperparameter space)
        ComputationGraphSpace cgs = new ComputationGraphSpace.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new AdamSpace(new ContinuousParameterSpace(0.0001, 0.1)))
                        .l2(new ContinuousParameterSpace(0.0001, 0.01)).iterations(1).addInputs("in")
                        .setInputTypes(InputType.feedForward(784))
                        .addLayer("first",
                                        new DenseLayerSpace.Builder().nIn(784).nOut(new IntegerParameterSpace(2, 10))
                                                        .activation(new DiscreteParameterSpace<>(Activation.RELU,
                                                                        Activation.TANH))
                                                        .build(),
                                        "in") //1-2 identical layers (except nIn)
                        .addLayer("out", new OutputLayerSpace.Builder().nOut(10).activation(Activation.SOFTMAX)
                                        .lossFunction(LossFunctions.LossFunction.MCXENT).build(), "first")
                        .setOutputs("out").earlyStoppingConfiguration(esConf).pretrain(false).backprop(true).build();

        //Define configuration:

        CandidateGenerator candidateGenerator = new RandomSearchGenerator(cgs, commands);
        DataProvider dataProvider = new DataSetIteratorFactoryProvider();


        String modelSavePath = new File(System.getProperty("java.io.tmpdir"), "ArbiterDL4JTest2CG\\").getAbsolutePath();

        File f = new File(modelSavePath);
        if (f.exists())
            f.delete();
        f.mkdir();
        f.deleteOnExit();
        if (!f.exists())
            throw new RuntimeException();

        OptimizationConfiguration configuration = new OptimizationConfiguration.Builder()
                        .candidateGenerator(candidateGenerator).dataProvider(dataProvider)
                        .modelSaver(new FileModelSaver(modelSavePath)).scoreFunction(ScoreFunctions.testSetLoss(true))
                        .terminationConditions(new MaxTimeCondition(2, TimeUnit.MINUTES),
                                        new MaxCandidatesCondition(100))
                        .build();


        IOptimizationRunner runner = new LocalOptimizationRunner(configuration,
                        new ComputationGraphTaskCreator(new ClassificationEvaluator()));
        runner.execute();


        System.out.println("----- COMPLETE -----");
    }
}
