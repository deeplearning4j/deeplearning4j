/*
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
package org.arbiter.deeplearning4j.computationgraph;

import org.arbiter.deeplearning4j.ComputationGraphSpace;
import org.arbiter.deeplearning4j.GraphConfiguration;
import org.arbiter.deeplearning4j.evaluator.graph.GraphClassificationDataSetEvaluator;
import org.arbiter.deeplearning4j.layers.DenseLayerSpace;
import org.arbiter.deeplearning4j.layers.OutputLayerSpace;
import org.arbiter.deeplearning4j.multilayernetwork.TestDL4JLocalExecution;
import org.arbiter.deeplearning4j.saver.local.graph.LocalComputationGraphSaver;
import org.arbiter.deeplearning4j.scoring.ScoreFunctions;
import org.arbiter.deeplearning4j.task.ComputationGraphTaskCreator;
import org.arbiter.optimize.api.CandidateGenerator;
import org.arbiter.optimize.api.data.DataProvider;
import org.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.arbiter.optimize.api.termination.MaxTimeCondition;
import org.arbiter.optimize.config.OptimizationConfiguration;
import org.arbiter.optimize.executor.CandidateExecutor;
import org.arbiter.optimize.executor.local.LocalCandidateExecutor;
import org.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.arbiter.optimize.parameter.discrete.DiscreteParameterSpace;
import org.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.arbiter.optimize.candidategenerator.RandomSearchGenerator;
import org.arbiter.optimize.runner.OptimizationRunner;
import org.arbiter.optimize.ui.ArbiterUIServer;
import org.arbiter.optimize.ui.listener.UIOptimizationRunnerStatusListener;
import org.arbiter.util.WebUtils;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculatorCG;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.fail;

public class TestGraphLocalExecution {

    private static Logger log = LoggerFactory.getLogger(TestGraphLocalExecution.class);

    @Test
    @Ignore
    public void testLocalExecution() throws Exception {

        //Define: network config (hyperparameter space)
        ComputationGraphSpace mls = new ComputationGraphSpace.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(new ContinuousParameterSpace(0.0001, 0.1))
                .regularization(true)
                .l2(new ContinuousParameterSpace(0.0001, 0.01))
                .iterations(100)
                .addInputs("in").setInputTypes(InputType.feedForward(4))
                .addLayer("layer0", new DenseLayerSpace.Builder().nIn(4).nOut(new IntegerParameterSpace(2,10))
                        .activation(new DiscreteParameterSpace<>("relu", "tanh"))
                        .build(), "in")
                .addLayer("out", new OutputLayerSpace.Builder().nOut(3).activation("softmax")
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build(), "layer0")
                .setOutputs("out")
                .numEpochs(3)
                .pretrain(false).backprop(true).build();

        //Define configuration:
        CandidateGenerator<GraphConfiguration> candidateGenerator = new RandomSearchGenerator<>(mls);
        DataProvider<DataSetIterator> dataProvider = new TestDL4JLocalExecution.IrisDataSetProvider();


//        String modelSavePath = FilenameUtils.concat(System.getProperty("java.io.tmpdir"),"ArbiterDL4JTest/");
        String modelSavePath = new File(System.getProperty("java.io.tmpdir"),"ArbiterDL4JTest\\").getAbsolutePath();

        File f = new File(modelSavePath);
        if(f.exists()) f.delete();
        f.mkdir();
        f.deleteOnExit();
        if(!f.exists()) throw new RuntimeException();

        OptimizationConfiguration<GraphConfiguration,ComputationGraph,DataSetIterator,Evaluation> configuration
                = new OptimizationConfiguration.Builder<GraphConfiguration,ComputationGraph,DataSetIterator,Evaluation>()
                .candidateGenerator(candidateGenerator)
                .dataProvider(dataProvider)
                .modelSaver(new LocalComputationGraphSaver<Evaluation>(modelSavePath))
                .scoreFunction(ScoreFunctions.testSetLossGraphDataSet(true))
                .terminationConditions(new MaxTimeCondition(2, TimeUnit.MINUTES),
                        new MaxCandidatesCondition(100))
                .build();

        CandidateExecutor<GraphConfiguration,ComputationGraph,DataSetIterator,Evaluation> executor =
                new LocalCandidateExecutor<>(new ComputationGraphTaskCreator<>(new GraphClassificationDataSetEvaluator()),true,1);

        OptimizationRunner<GraphConfiguration,ComputationGraph,DataSetIterator,Evaluation> runner
                = new OptimizationRunner<>(configuration, executor);

        ArbiterUIServer server = new ArbiterUIServer();
        String[] str = new String[]{"server", "dropwizard.yml"};
        server.run(str);
        WebUtils.tryOpenBrowser("http://localhost:8080/arbiter", log);    //TODO don't hardcode
        runner.addListeners(new UIOptimizationRunnerStatusListener(server));

        runner.execute();


        System.out.println("----- COMPLETE -----");
    }

    @Test
    @Ignore
    public void testLocalExecutionEarlyStopping() throws Exception {

        EarlyStoppingConfiguration<ComputationGraph> esConf = new EarlyStoppingConfiguration.Builder<ComputationGraph>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(100))
                .scoreCalculator(new DataSetLossCalculatorCG(new IrisDataSetIterator(150,150),true))
                .modelSaver(new InMemoryModelSaver<ComputationGraph>())
                .build();

        //Define: network config (hyperparameter space)
        ComputationGraphSpace cgs = new ComputationGraphSpace.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(new ContinuousParameterSpace(0.0001, 0.1))
                .regularization(true)
                .l2(new ContinuousParameterSpace(0.0001, 0.01))
                .iterations(1)
                .addInputs("in").setInputTypes(InputType.feedForward(4))
                .addLayer("first", new DenseLayerSpace.Builder().nIn(4).nOut(new IntegerParameterSpace(2, 10))
                        .activation(new DiscreteParameterSpace<String>("relu", "tanh"))
                        .build(), "in")   //1-2 identical layers (except nIn)
                .addLayer("out", new OutputLayerSpace.Builder().nOut(3).activation("softmax")
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build(), "first")
                .setOutputs("out")
                .earlyStoppingConfiguration(esConf)
                .pretrain(false).backprop(true).build();

        //Define configuration:

        CandidateGenerator<GraphConfiguration> candidateGenerator = new RandomSearchGenerator<>(cgs);
        DataProvider<DataSetIterator> dataProvider = new TestDL4JLocalExecution.IrisDataSetProvider();


        String modelSavePath = new File(System.getProperty("java.io.tmpdir"),"ArbiterDL4JTest2CG\\").getAbsolutePath();

        File f = new File(modelSavePath);
        if(f.exists()) f.delete();
        f.mkdir();
        f.deleteOnExit();
        if(!f.exists()) throw new RuntimeException();

        OptimizationConfiguration<GraphConfiguration,ComputationGraph,DataSetIterator,Evaluation> configuration
                = new OptimizationConfiguration.Builder<GraphConfiguration,ComputationGraph,DataSetIterator,Evaluation>()
                .candidateGenerator(candidateGenerator)
                .dataProvider(dataProvider)
                .modelSaver(new LocalComputationGraphSaver<Evaluation>(modelSavePath))
                .scoreFunction(ScoreFunctions.testSetLossGraphDataSet(true))
                .terminationConditions(new MaxTimeCondition(2, TimeUnit.MINUTES),
                        new MaxCandidatesCondition(100))
                .build();

        CandidateExecutor<GraphConfiguration,ComputationGraph,DataSetIterator,Evaluation> executor =
                new LocalCandidateExecutor<>(new ComputationGraphTaskCreator<>(new GraphClassificationDataSetEvaluator()),true,1);

        OptimizationRunner<GraphConfiguration,ComputationGraph,DataSetIterator,Evaluation> runner
                = new OptimizationRunner<>(configuration, executor);

        ArbiterUIServer server = new ArbiterUIServer();
        String[] str = new String[]{"server", "dropwizard.yml"};
        server.run(str);
        WebUtils.tryOpenBrowser("http://localhost:8080/arbiter", log);    //TODO don't hardcode
        runner.addListeners(new UIOptimizationRunnerStatusListener(server));

        runner.execute();


        System.out.println("----- COMPLETE -----");

    }
}
