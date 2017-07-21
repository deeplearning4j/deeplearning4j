package org.deeplearning4j.arbiter.optimize;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.arbiter.ComputationGraphSpace;
import org.deeplearning4j.arbiter.DL4JConfiguration;
import org.deeplearning4j.arbiter.GraphConfiguration;
import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.layers.ConvolutionLayerSpace;
import org.deeplearning4j.arbiter.layers.DenseLayerSpace;
import org.deeplearning4j.arbiter.layers.OutputLayerSpace;
import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxTimeCondition;
import org.deeplearning4j.arbiter.optimize.candidategenerator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.discrete.DiscreteParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.listener.StatusListener;
import org.deeplearning4j.arbiter.saver.local.graph.LocalComputationGraphSaver;
import org.deeplearning4j.arbiter.saver.local.multilayer.LocalMultiLayerNetworkSaver;
import org.deeplearning4j.arbiter.scoring.graph.GraphTestSetLossScoreFunctionDataSet;
import org.deeplearning4j.arbiter.scoring.impl.TestSetLossScoreFunction;
import org.deeplearning4j.arbiter.task.ComputationGraphTaskCreator;
import org.deeplearning4j.arbiter.task.MultiLayerNetworkTaskCreator;
import org.deeplearning4j.arbiter.ui.listener.ArbiterStatusListener;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Created by Alex on 19/07/2017.
 */
public class TestBasic {

    @Test
    public void testBasicUiOnly() throws Exception {

        UIServer.getInstance();

        Thread.sleep(1000000);
    }


    @Test
    @Ignore
    public void testBasicMnist() throws Exception {

        MultiLayerSpace mls = new MultiLayerSpace.Builder()
                .learningRate(new ContinuousParameterSpace(0.0001, 0.2))
                .l2(new ContinuousParameterSpace(0.0001, 0.05))
                .dropOut(new ContinuousParameterSpace(0.2, 0.7))
                .addLayer(
                        new ConvolutionLayerSpace.Builder().nIn(1)
                                .nOut(new IntegerParameterSpace(5, 30))
                                .kernelSize(new DiscreteParameterSpace<>(new int[] {3, 3},
                                        new int[] {4, 4}, new int[] {5, 5}))
                                .stride(new DiscreteParameterSpace<>(new int[] {1, 1},
                                        new int[] {2, 2}))
                                .activation(new DiscreteParameterSpace<>(Activation.RELU,
                                        Activation.SOFTPLUS, Activation.LEAKYRELU))
                                .build())
                .addLayer(new DenseLayerSpace.Builder().nOut(new IntegerParameterSpace(32, 128))
                        .activation(new DiscreteParameterSpace<>(Activation.RELU, Activation.TANH))
                        .build(), new IntegerParameterSpace(0, 1), true) //0 to 1 layers
                .addLayer(new OutputLayerSpace.Builder().nOut(10).activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .setInputType(InputType.convolutionalFlat(28, 28, 1))
                .build();
        Map<String, Object> commands = new HashMap<>();
//        commands.put(DataSetIteratorFactoryProvider.FACTORY_KEY, MnistDataSetIteratorFactory.class.getCanonicalName());

        //Define configuration:
        CandidateGenerator<DL4JConfiguration> candidateGenerator = new RandomSearchGenerator<>(mls, commands);
        DataProvider<Object> dataProvider = new MnistDataSetProvider();


        String modelSavePath = new File(System.getProperty("java.io.tmpdir"), "ArbiterUiTestBasicMnist\\").getAbsolutePath();

        File f = new File(modelSavePath);
        if (f.exists())
            f.delete();
        f.mkdir();
        if (!f.exists())
            throw new RuntimeException();

        OptimizationConfiguration<DL4JConfiguration, MultiLayerNetwork, Object, Evaluation> configuration =
                new OptimizationConfiguration.Builder<DL4JConfiguration, MultiLayerNetwork, Object, Evaluation>()
                        .candidateGenerator(candidateGenerator).dataProvider(dataProvider)
                        .modelSaver(new LocalMultiLayerNetworkSaver(modelSavePath))
                        .scoreFunction(new TestSetLossScoreFunction(true))
                        .terminationConditions(new MaxTimeCondition(120, TimeUnit.MINUTES),
                                new MaxCandidatesCondition(100))
                        .build();

        IOptimizationRunner<DL4JConfiguration, MultiLayerNetwork, Evaluation> runner =
                new LocalOptimizationRunner<>(configuration, new MultiLayerNetworkTaskCreator<Evaluation>());

        StatsStorage ss = new InMemoryStatsStorage();
        StatusListener sl = new ArbiterStatusListener(ss);
        runner.addListeners(sl);

        UIServer.getInstance().attach(ss);

        runner.execute();
        Thread.sleep(100000);
    }


    @Test
    @Ignore
    public void testBasicMnistCompGraph() throws Exception {

        ComputationGraphSpace cgs = new ComputationGraphSpace.Builder()
                .learningRate(new ContinuousParameterSpace(0.0001, 0.2))
                .l2(new ContinuousParameterSpace(0.0001, 0.05))
                .dropOut(new ContinuousParameterSpace(0.2, 0.7))
                .addInputs("in")
                .addLayer("0",
                        new ConvolutionLayerSpace.Builder().nIn(1)
                                .nOut(new IntegerParameterSpace(5, 30))
                                .kernelSize(new DiscreteParameterSpace<>(new int[] {3, 3},
                                        new int[] {4, 4}, new int[] {5, 5}))
                                .stride(new DiscreteParameterSpace<>(new int[] {1, 1},
                                        new int[] {2, 2}))
                                .activation(new DiscreteParameterSpace<>(Activation.RELU,
                                        Activation.SOFTPLUS, Activation.LEAKYRELU))
                                .build(), "in")
                .addLayer("1", new DenseLayerSpace.Builder().nOut(new IntegerParameterSpace(32, 128))
                        .activation(new DiscreteParameterSpace<>(Activation.RELU, Activation.TANH))
                        .build(), "0")
                .addLayer("out", new OutputLayerSpace.Builder().nOut(10).activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build(), "1")
                .setOutputs("out")
                .setInputTypes(InputType.convolutionalFlat(28, 28, 1))
                .build();

        //Define configuration:
        CandidateGenerator<GraphConfiguration> candidateGenerator = new RandomSearchGenerator<>(cgs, Collections.EMPTY_MAP);
        DataProvider<Object> dataProvider = new MnistDataSetProvider();


        String modelSavePath = new File(System.getProperty("java.io.tmpdir"), "ArbiterUiTestBasicMnistCG\\").getAbsolutePath();

        File f = new File(modelSavePath);
        if (f.exists())
            f.delete();
        f.mkdir();
        if (!f.exists())
            throw new RuntimeException();

        OptimizationConfiguration<GraphConfiguration, ComputationGraph, Object, Evaluation> configuration =
                new OptimizationConfiguration.Builder<GraphConfiguration, ComputationGraph, Object, Evaluation>()
                        .candidateGenerator(candidateGenerator).dataProvider(dataProvider)
                        .modelSaver(new LocalComputationGraphSaver(modelSavePath))
                        .scoreFunction(new GraphTestSetLossScoreFunctionDataSet(true))
                        .terminationConditions(new MaxTimeCondition(120, TimeUnit.MINUTES),
                                new MaxCandidatesCondition(100))
                        .build();

        IOptimizationRunner<GraphConfiguration, ComputationGraph, Evaluation> runner =
                new LocalOptimizationRunner<>(configuration, new ComputationGraphTaskCreator<>());

        StatsStorage ss = new InMemoryStatsStorage();
        StatusListener sl = new ArbiterStatusListener(ss);
        runner.addListeners(sl);

        UIServer.getInstance().attach(ss);

        runner.execute();
        Thread.sleep(100000);
    }



    private static class MnistDataSetProvider implements DataProvider<Object> {

        @Override
        public DataSetIterator trainData(Map<String, Object> dataParameters) {
            try {
                if (dataParameters == null || dataParameters.isEmpty()) {
                    return new MnistDataSetIterator(64, 10000, false, true, true, 123);
                }
                if (dataParameters.containsKey("batchsize")) {
                    int b = (Integer) dataParameters.get("batchsize");
                    return new MnistDataSetIterator(b, 10000, false, true, true, 123);
                }
                return new MnistDataSetIterator(64, 10000, false, true, true, 123);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        @Override
        public DataSetIterator testData(Map<String, Object> dataParameters) {
            return trainData(dataParameters);
        }

        @Override
        public String toString() {
            return "MnistDataSetProvider()";
        }
    }
}
