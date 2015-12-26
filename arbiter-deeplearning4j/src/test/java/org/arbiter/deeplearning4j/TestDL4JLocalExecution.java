package org.arbiter.deeplearning4j;

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
import org.arbiter.optimize.ui.ArbiterUIServer;
import org.arbiter.optimize.ui.listener.UIOptimizationRunnerStatusListener;
import org.arbiter.util.WebUtils;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
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
    public void testLocalExecution() throws Exception {

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
                .add("activation", new FixedValue<>("softmax"))
                .add("lossFunction", new FixedValue<>(LossFunctions.LossFunction.MCXENT))
                .build();

        MultiLayerSpaceOld mls = new MultiLayerSpaceOld.Builder()
                .add("optimizationAlgo", new FixedValue<>(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT))
                .add("pretrain", new FixedValue<>(false))
                .add("backprop", new FixedValue<>(true))
                .add("learningRate", new ContinuousParameterSpace(0.0001, 0.1))  //TODO: logarithmic
                .add("regularization", new FixedValue<>(true))
                .add("l2", new ContinuousParameterSpace(0.0001, 0.01))
                .add("optimizationAlgo", new FixedValue<>(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT))
                .add("iterations",new FixedValue<>(100))
                .addLayer(ls1)
                .addLayer(ls2)
                .build();

        //Define configuration:

        CandidateGenerator<MultiLayerConfiguration> candidateGenerator = new RandomSearchGenerator<>(mls);
        DataProvider<DataSetIterator> dataProvider = new IrisDataSetProvider();


//        String modelSavePath = FilenameUtils.concat(System.getProperty("java.io.tmpdir"),"ArbiterDL4JTest/");
        String modelSavePath = new File(System.getProperty("java.io.tmpdir"),"ArbiterDL4JTest\\").getAbsolutePath();

        File f = new File(modelSavePath);
        if(f.exists()) f.delete();
        f.mkdir();
        if(!f.exists()) throw new RuntimeException();

        OptimizationConfiguration<MultiLayerConfiguration,MultiLayerNetwork,DataSetIterator,Evaluation> configuration
                = new OptimizationConfiguration.Builder<MultiLayerConfiguration,MultiLayerNetwork,DataSetIterator,Evaluation>()
                .candidateGenerator(candidateGenerator)
                .dataProvider(dataProvider)
                .modelSaver(new LocalMultiLayerNetworkSaver(modelSavePath))
                .scoreFunction(new TestSetLossScoreFunction())
                .terminationConditions(new MaxTimeCondition(2, TimeUnit.MINUTES),
                        new MaxCandidatesCondition(100))
                .build();

        CandidateExecutor<MultiLayerConfiguration,MultiLayerNetwork,DataSetIterator,Evaluation> executor =
                new LocalCandidateExecutor<>(new DL4JTaskCreator(new DL4JClassificationEvaluator()),1);

        OptimizationRunner<MultiLayerConfiguration,MultiLayerNetwork,DataSetIterator,Evaluation> runner
                = new OptimizationRunner<>(configuration, executor);

        ArbiterUIServer server = new ArbiterUIServer();
        String[] str = new String[]{"server", "dropwizard.yml"};
        server.run(str);
        WebUtils.tryOpenBrowser("http://localhost:8080/arbiter", log);    //TODO don't hardcode
        runner.addListeners(new UIOptimizationRunnerStatusListener(server));

        runner.execute();


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
