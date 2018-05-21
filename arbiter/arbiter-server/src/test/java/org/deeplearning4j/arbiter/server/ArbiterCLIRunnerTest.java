package org.deeplearning4j.arbiter.server;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.conf.updater.SgdSpace;
import org.deeplearning4j.arbiter.layers.DenseLayerSpace;
import org.deeplearning4j.arbiter.layers.OutputLayerSpace;
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
import org.deeplearning4j.arbiter.saver.local.FileModelSaver;
import org.deeplearning4j.arbiter.scoring.impl.TestSetLossScoreFunction;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 3/12/17.
 */
@Slf4j
public class ArbiterCLIRunnerTest {


    @Test
    public void testCliRunner() throws Exception {
        ArbiterCliRunner cliRunner = new ArbiterCliRunner();

        //Define: network config (hyperparameter space)
        MultiLayerSpace mls = new MultiLayerSpace.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new SgdSpace(new ContinuousParameterSpace(0.0001, 0.1)))
                .l2(new ContinuousParameterSpace(0.0001, 0.01))
                .addLayer(new DenseLayerSpace.Builder().nIn(784).nOut(new IntegerParameterSpace(2,10))
                        .activation(new DiscreteParameterSpace<>(Activation.RELU, Activation.TANH))
                        .build(),new IntegerParameterSpace(1,2),true)   //1-2 identical layers (except nIn)
                .addLayer(new OutputLayerSpace.Builder().nOut(10).activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .numEpochs(3)
                .pretrain(false).backprop(true).build();
         assertEquals(mls,MultiLayerSpace.fromJson(mls.toJson()));
        //Define configuration:
        Map<String,Object> commands = new HashMap<>();
        commands.put(DataSetIteratorFactoryProvider.FACTORY_KEY,TestDataFactoryProviderMnist.class.getCanonicalName());

        CandidateGenerator candidateGenerator = new RandomSearchGenerator(mls,commands);
        DataProvider dataProvider = new DataSetIteratorFactoryProvider();


//        String modelSavePath = FilenameUtils.concat(System.getProperty("java.io.tmpdir"),"ArbiterDL4JTest/");
        String modelSavePath = new File(System.getProperty("java.io.tmpdir"),"ArbiterDL4JTest/").getAbsolutePath();
        File dir = new File(modelSavePath);
        if(!dir.exists())
            dir.mkdirs();
        String configPath = System.getProperty("java.io.tmpdir") + File.separator + UUID.randomUUID().toString() + ".json";
        OptimizationConfiguration configuration
                = new OptimizationConfiguration.Builder()
                .candidateGenerator(candidateGenerator)
                .dataProvider(dataProvider)
                .modelSaver(new FileModelSaver(modelSavePath))
                .scoreFunction(new TestSetLossScoreFunction())
                .terminationConditions(new MaxTimeCondition(30, TimeUnit.SECONDS),
                        new MaxCandidatesCondition(5))
                .build();
        assertEquals(configuration,OptimizationConfiguration.fromJson(configuration.toJson()));

        FileUtils.writeStringToFile(new File(configPath),configuration.toJson());
        System.out.println(configuration.toJson());

        log.info("Starting test");
        cliRunner.runMain(
                "--dataSetIteratorClass",
                TestDataFactoryProviderMnist.class.getCanonicalName(),
                "--neuralNetType",
                ArbiterCliRunner.MULTI_LAYER_NETWORK,
                "--optimizationConfigPath",
                configPath
        );
    }



}
