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
package org.arbiter.deeplearning4j.multilayernetwork;

import org.arbiter.deeplearning4j.DL4JConfiguration;
import org.arbiter.deeplearning4j.MultiLayerSpace;
import org.arbiter.deeplearning4j.evaluator.multilayer.ClassificationEvaluator;
import org.arbiter.deeplearning4j.layers.DenseLayerSpace;
import org.arbiter.deeplearning4j.layers.OutputLayerSpace;
import org.arbiter.deeplearning4j.saver.local.multilayer.LocalMultiLayerNetworkSaver;
import org.arbiter.deeplearning4j.scoring.multilayer.TestSetLossScoreFunction;
import org.arbiter.deeplearning4j.task.MultiLayerNetworkTaskCreator;
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
import org.arbiter.optimize.randomsearch.RandomSearchGenerator;
import org.arbiter.optimize.runner.OptimizationRunner;
import org.arbiter.optimize.ui.ArbiterUIServer;
import org.arbiter.optimize.ui.listener.UIOptimizationRunnerStatusListener;
import org.arbiter.util.WebUtils;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Map;
import java.util.concurrent.TimeUnit;

public class TestDL4JLocalExecution {

    private static Logger log = LoggerFactory.getLogger(TestDL4JLocalExecution.class);

    @Test
    @org.junit.Ignore
    public void testLocalExecution() throws Exception {

        //Define: network config (hyperparameter space)
        MultiLayerSpace mls = new MultiLayerSpace.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(new ContinuousParameterSpace(0.0001, 0.1))
                .regularization(true)
                .l2(new ContinuousParameterSpace(0.0001, 0.01))
                .iterations(100)
                .addLayer(new DenseLayerSpace.Builder().nIn(4).nOut(new IntegerParameterSpace(2,10))
                            .activation(new DiscreteParameterSpace<>("relu","tanh"))
                            .build(),new IntegerParameterSpace(1,2),true)   //1-2 identical layers (except nIn)
                .addLayer(new OutputLayerSpace.Builder().nOut(3).activation("softmax")
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .numEpochs(3)
                .pretrain(false).backprop(true).build();

        //Define configuration:

        CandidateGenerator<DL4JConfiguration> candidateGenerator = new RandomSearchGenerator<>(mls);
        DataProvider<DataSetIterator> dataProvider = new IrisDataSetProvider();


//        String modelSavePath = FilenameUtils.concat(System.getProperty("java.io.tmpdir"),"ArbiterDL4JTest/");
        String modelSavePath = new File(System.getProperty("java.io.tmpdir"),"ArbiterDL4JTest\\").getAbsolutePath();

        File f = new File(modelSavePath);
        if(f.exists()) f.delete();
        f.mkdir();
        if(!f.exists()) throw new RuntimeException();

        OptimizationConfiguration<DL4JConfiguration,MultiLayerNetwork,DataSetIterator,Evaluation> configuration
                = new OptimizationConfiguration.Builder<DL4JConfiguration,MultiLayerNetwork,DataSetIterator,Evaluation>()
                .candidateGenerator(candidateGenerator)
                .dataProvider(dataProvider)
                .modelSaver(new LocalMultiLayerNetworkSaver<Evaluation>(modelSavePath))
                .scoreFunction(new TestSetLossScoreFunction())
                .terminationConditions(new MaxTimeCondition(2, TimeUnit.MINUTES),
                        new MaxCandidatesCondition(100))
                .build();

        CandidateExecutor<DL4JConfiguration,MultiLayerNetwork,DataSetIterator,Evaluation> executor =
                new LocalCandidateExecutor<>(new MultiLayerNetworkTaskCreator<>(new ClassificationEvaluator()),1);

        OptimizationRunner<DL4JConfiguration,MultiLayerNetwork,DataSetIterator,Evaluation> runner
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

        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(100))
                .scoreCalculator(new DataSetLossCalculator(new IrisDataSetIterator(150,150),true))
                .modelSaver(new InMemoryModelSaver<MultiLayerNetwork>())
                .build();

        //Define: network config (hyperparameter space)
        MultiLayerSpace mls = new MultiLayerSpace.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(new ContinuousParameterSpace(0.0001, 0.1))
                .regularization(true)
                .l2(new ContinuousParameterSpace(0.0001, 0.01))
                .iterations(1)
                .addLayer(new DenseLayerSpace.Builder().nIn(4).nOut(new IntegerParameterSpace(2, 10))
                        .activation(new DiscreteParameterSpace<String>("relu", "tanh"))
                        .build(), new IntegerParameterSpace(1, 2), true)   //1-2 identical layers (except nIn)
                .addLayer(new OutputLayerSpace.Builder().nOut(3).activation("softmax")
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .earlyStoppingConfiguration(esConf)
                .pretrain(false).backprop(true).build();

        //Define configuration:

        CandidateGenerator<DL4JConfiguration> candidateGenerator = new RandomSearchGenerator<>(mls);
        DataProvider<DataSetIterator> dataProvider = new IrisDataSetProvider();


        String modelSavePath = new File(System.getProperty("java.io.tmpdir"),"ArbiterDL4JTest2\\").getAbsolutePath();

        File f = new File(modelSavePath);
        if(f.exists()) f.delete();
        f.mkdir();
        if(!f.exists()) throw new RuntimeException();

        OptimizationConfiguration<DL4JConfiguration,MultiLayerNetwork,DataSetIterator,Evaluation> configuration
                = new OptimizationConfiguration.Builder<DL4JConfiguration,MultiLayerNetwork,DataSetIterator,Evaluation>()
                .candidateGenerator(candidateGenerator)
                .dataProvider(dataProvider)
                .modelSaver(new LocalMultiLayerNetworkSaver<Evaluation>(modelSavePath))
                .scoreFunction(new TestSetLossScoreFunction())
                .terminationConditions(new MaxTimeCondition(2, TimeUnit.MINUTES),
                        new MaxCandidatesCondition(100))
                .build();

        CandidateExecutor<DL4JConfiguration,MultiLayerNetwork,DataSetIterator,Evaluation> executor =
                new LocalCandidateExecutor<>(new MultiLayerNetworkTaskCreator<>(new ClassificationEvaluator()),1);

        OptimizationRunner<DL4JConfiguration,MultiLayerNetwork,DataSetIterator,Evaluation> runner
                = new OptimizationRunner<>(configuration, executor);

       /* ArbiterUIServer server = new ArbiterUIServer();
        String[] str = new String[]{"server", "dropwizard.yml"};
        server.run(str);
        WebUtils.tryOpenBrowser("http://localhost:8080/arbiter", log);    //TODO don't hardcode
        runner.addListeners(new UIOptimizationRunnerStatusListener(server));

       */ runner.execute();


        System.out.println("----- COMPLETE -----");

    }


    private static class IrisDataSetProvider implements DataProvider<DataSetIterator>{

        @Override
        public DataSetIterator trainData(Map<String, Object> dataParameters) {
            if(dataParameters == null || dataParameters.isEmpty()) return new IrisDataSetIterator(150,150);
            if(dataParameters.containsKey("batchsize")){
                int b = (Integer)dataParameters.get("batchsize");
                return new IrisDataSetIterator(b,150);
            }
            return new IrisDataSetIterator(150,150);
        }

        @Override
        public DataSetIterator testData(Map<String, Object> dataParameters) {
            return trainData(dataParameters);
        }

        @Override
        public String toString(){
            return "IrisDataSetProvider()";
        }
    }
}
