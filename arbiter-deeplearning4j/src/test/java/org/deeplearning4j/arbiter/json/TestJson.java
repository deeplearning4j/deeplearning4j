package org.deeplearning4j.arbiter.json;

import org.deeplearning4j.arbiter.ComputationGraphSpace;
import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.conf.updater.AdaMaxSpace;
import org.deeplearning4j.arbiter.conf.updater.AdamSpace;
import org.deeplearning4j.arbiter.conf.updater.SgdSpace;
import org.deeplearning4j.arbiter.layers.DenseLayerSpace;
import org.deeplearning4j.arbiter.layers.OutputLayerSpace;
import org.deeplearning4j.arbiter.multilayernetwork.MnistDataSetIteratorFactory;
import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.data.DataSetIteratorFactoryProvider;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxTimeCondition;
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.discrete.DiscreteParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.arbiter.optimize.serde.jackson.JsonMapper;
import org.deeplearning4j.arbiter.scoring.RegressionValue;
import org.deeplearning4j.arbiter.scoring.ScoreFunctions;
import org.deeplearning4j.arbiter.scoring.impl.TestSetLossScoreFunction;
import org.deeplearning4j.arbiter.util.TestDataFactoryProviderMnist;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculatorCG;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 14/02/2017.
 */
public class TestJson {

    @Test
    public void testMultiLayerSpaceJson() {
        MultiLayerSpace mls = new MultiLayerSpace.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new SgdSpace(new ContinuousParameterSpace(0.0001, 0.2)))
                        .l2(new ContinuousParameterSpace(0.0001, 0.05))
                        .addLayer(new DenseLayerSpace.Builder().nIn(1).nOut(new IntegerParameterSpace(5, 30))
                                        .activation(new DiscreteParameterSpace<>(Activation.RELU, Activation.SOFTPLUS,
                                                        Activation.LEAKYRELU))
                                        .build(), new IntegerParameterSpace(1, 2), true) //1-2 identical layers
                        .addLayer(new DenseLayerSpace.Builder().nIn(4).nOut(new IntegerParameterSpace(2, 10))
                                        .activation(new DiscreteParameterSpace<>(Activation.RELU, Activation.TANH))
                                        .build(), new IntegerParameterSpace(0, 1), true) //0 to 1 layers
                        .addLayer(new OutputLayerSpace.Builder().nOut(10).activation(Activation.SOFTMAX)
                                        .iLossFunction(LossFunctions.LossFunction.MCXENT.getILossFunction()).build())
                        .setInputType(InputType.convolutional(28, 28, 1)).pretrain(false).backprop(true).build();

        String asJson = mls.toJson();
        //        System.out.println(asJson);

        MultiLayerSpace fromJson = MultiLayerSpace.fromJson(asJson);

        assertEquals(mls, fromJson);
    }



    @Test
    public void testOptimizationFromJson() {
        EarlyStoppingConfiguration<ComputationGraph> esConf =
                        new EarlyStoppingConfiguration.Builder<ComputationGraph>()
                                        .epochTerminationConditions(new MaxEpochsTerminationCondition(100))
                                        .scoreCalculator(new DataSetLossCalculatorCG(new IrisDataSetIterator(150, 150),
                                                        true))
                                        .modelSaver(new InMemoryModelSaver<ComputationGraph>()).build();

        //Define: network config (hyperparameter space)
        ComputationGraphSpace cgs = new ComputationGraphSpace.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new AdaMaxSpace(new ContinuousParameterSpace(0.0001, 0.1)))
                        .l2(new ContinuousParameterSpace(0.0001, 0.01)).addInputs("in")
                        .setInputTypes(InputType.feedForward(4))
                        .addLayer("first",
                                        new DenseLayerSpace.Builder().nIn(4).nOut(new IntegerParameterSpace(2, 10))
                                                        .activation(new DiscreteParameterSpace<>(Activation.RELU,
                                                                        Activation.TANH))
                                                        .build(),
                                        "in") //1-2 identical layers (except nIn)
                        .addLayer("out", new OutputLayerSpace.Builder().nOut(3).activation(Activation.SOFTMAX)
                                        .lossFunction(LossFunctions.LossFunction.MCXENT).build(), "first")
                        .setOutputs("out").earlyStoppingConfiguration(esConf).pretrain(false).backprop(true).build();

        //Define configuration:
        Map<String, Object> commands = new HashMap<>();
        commands.put(DataSetIteratorFactoryProvider.FACTORY_KEY, TestDataFactoryProviderMnist.class.getCanonicalName());

        CandidateGenerator candidateGenerator = new RandomSearchGenerator(cgs, commands);
        DataProvider dataProvider = new DataSetIteratorFactoryProvider();


        OptimizationConfiguration configuration =
                        new OptimizationConfiguration.Builder().candidateGenerator(candidateGenerator)
                                        .dataProvider(dataProvider).scoreFunction(new TestSetLossScoreFunction())
                                        .terminationConditions(new MaxTimeCondition(2, TimeUnit.MINUTES),
                                                        new MaxCandidatesCondition(100))
                                        .build();

        String json = configuration.toJson();
        OptimizationConfiguration loadConf = OptimizationConfiguration.fromJson(json);
        assertEquals(configuration, loadConf);
    }

    @Test
    public void testComputationGraphSpaceJson() {
        ParameterSpace<Integer> p = new IntegerParameterSpace(10, 100);
        ComputationGraphSpace cgs =
                        new ComputationGraphSpace.Builder()
                                        .updater(new AdamSpace(new DiscreteParameterSpace<>(0.1, 0.5, 1.0)))
                                        .seed(12345).addInputs("in")
                                        .addLayer("0", new DenseLayerSpace.Builder()
                                                        .nIn(new IntegerParameterSpace(1, 100)).nOut(p).build(), "in")
                                        .addLayer("1", new DenseLayerSpace.Builder().nIn(p).nOut(10).build(), "0")
                                        .addLayer("2", new OutputLayerSpace.Builder().iLossFunction(
                                                        LossFunctions.LossFunction.MCXENT.getILossFunction()).nIn(10)
                                                        .nOut(5).build(), "1")
                                        .setOutputs("2").backprop(true).pretrain(false).build();

        String asJson = cgs.toJson();
        ComputationGraphSpace fromJson = ComputationGraphSpace.fromJson(asJson);

        assertEquals(cgs, fromJson);
    }

    @Test
    public void testScoreFunctionJson() throws Exception {

        ScoreFunction[] scoreFunctions = new ScoreFunction[]{
                ScoreFunctions.testSetAccuracy(), ScoreFunctions.testSetF1(),
                ScoreFunctions.testSetLoss(true), ScoreFunctions.testSetRegression(RegressionValue.MAE),
                ScoreFunctions.testSetRegression(RegressionValue.RMSE)};

        for(ScoreFunction sc : scoreFunctions){
            String json = JsonMapper.getMapper().writeValueAsString(sc);
            ScoreFunction fromJson = JsonMapper.getMapper().readValue(json, ScoreFunction.class);

            assertEquals(sc, fromJson);
        }
    }
}
