package org.arbiter.deeplearning4j;

import org.apache.commons.io.FilenameUtils;
import org.apache.commons.math3.distribution.UniformIntegerDistribution;
import org.arbiter.deeplearning4j.saver.local.LocalMultiLayerNetworkSaver;
import org.arbiter.deeplearning4j.scoring.TestSetLossScoreFunction;
import org.arbiter.deeplearning4j.task.DL4JTaskCreator;
import org.arbiter.optimize.api.CandidateGenerator;
import org.arbiter.optimize.api.data.DataProvider;
import org.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.arbiter.optimize.api.termination.MaxTimeCondition;
import org.arbiter.optimize.config.OptimizationConfiguration;
import org.arbiter.optimize.executor.CandidateExecutor;
import org.arbiter.optimize.executor.local.LocalCandidateExecutor;
import org.arbiter.optimize.parameter.FixedValue;
import org.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.arbiter.optimize.parameter.discrete.DiscreteParameterSpace;
import org.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.arbiter.optimize.randomsearch.RandomSearchGenerator;
import org.arbiter.optimize.runner.OptimizationRunner;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;

import java.io.File;
import java.util.concurrent.TimeUnit;

public class TestDL4JLocalExecution {

    @Test
    public void testLocalExecution(){

        //Define: network config (hyperparameter space)
        LayerSpace ls1 = new LayerSpace.Builder()
                .layer(DenseLayer.class)
                .numLayersDistribution(new UniformIntegerDistribution(1, 2))     //1 or 2 layers
                .add("nIn", new FixedValue<Integer>(4))
                .add("nOut", new IntegerParameterSpace(2, 10))
                .add("activation", new DiscreteParameterSpace<String>("relu", "tanh"))
                .build();

        LayerSpace ls2 = new LayerSpace.Builder()
                .layer(OutputLayer.class)
                .add("nOut", new FixedValue<Integer>(3))
                .add("activation", new FixedValue<Object>("softmax"))
                .build();

        MultiLayerSpace mls = new MultiLayerSpace.Builder()
                .add("pretrain", new FixedValue<>(false))
                .add("backprop", new FixedValue<>(true))
                .add("learningRate", new ContinuousParameterSpace(0.0001, 0.1))  //TODO: logarithmic
                .add("regularization", new FixedValue<>(true))
                .add("l2", new ContinuousParameterSpace(0.0001, 0.1))
                .add("optimizationAlgo", new FixedValue<>(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT))
                .add("iterations",new FixedValue<>(1))
                .addLayer(ls1)
                .addLayer(ls2)
                .build();

        //Define configuration:

        CandidateGenerator<MultiLayerConfiguration> candidateGenerator = new RandomSearchGenerator<>(mls);
        DataProvider<DataSetIterator> dataProvider = new IrisDSP();


//        String modelSavePath = FilenameUtils.concat(System.getProperty("java.io.tmpdir"),"ArbiterDL4JTest/");
        String modelSavePath = new File(System.getProperty("java.io.tmpdir"),"ArbiterDL4JTest\\").getAbsolutePath();

        File f = new File(modelSavePath);
        if(f.exists()) f.delete();
        f.mkdir();

        OptimizationConfiguration<MultiLayerConfiguration,MultiLayerNetwork,DataSetIterator> configuration
                = new OptimizationConfiguration.Builder<MultiLayerConfiguration,MultiLayerNetwork,DataSetIterator>()
                .candidateGenerator(candidateGenerator)
                .dataProvider(dataProvider)
                .modelSaver(new LocalMultiLayerNetworkSaver(modelSavePath))
                .scoreFunction(new TestSetLossScoreFunction())
                .terminationConditions(new MaxTimeCondition(2, TimeUnit.MINUTES),
                        new MaxCandidatesCondition(10))
                .maxConcurrentJobs(5)
                .build();

        CandidateExecutor<MultiLayerConfiguration,MultiLayerNetwork,DataSetIterator> executor =
                new LocalCandidateExecutor<>(new DL4JTaskCreator(),1);

        OptimizationRunner<MultiLayerConfiguration,MultiLayerNetwork,DataSetIterator> runner
                = new OptimizationRunner<>(configuration, executor);

        runner.execute();


        System.out.println("----- COMPLETE -----");


    }


    private static class IrisDSP implements DataProvider<DataSetIterator>{
        @Override
        public DataSetIterator trainData() {
            return new IrisDataSetIterator(150,150);
        }

        @Override
        public DataSetIterator testData() {
            return new IrisDataSetIterator(150,150);
        }
    }
}
