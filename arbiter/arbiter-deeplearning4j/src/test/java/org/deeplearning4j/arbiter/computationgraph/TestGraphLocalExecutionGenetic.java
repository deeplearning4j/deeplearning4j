/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.arbiter.computationgraph;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.arbiter.ComputationGraphSpace;
import org.deeplearning4j.arbiter.conf.updater.AdamSpace;
import org.deeplearning4j.arbiter.conf.updater.SgdSpace;
import org.deeplearning4j.arbiter.evaluator.multilayer.ClassificationEvaluator;
import org.deeplearning4j.arbiter.layers.DenseLayerSpace;
import org.deeplearning4j.arbiter.layers.OutputLayerSpace;
import org.deeplearning4j.arbiter.multilayernetwork.TestDL4JLocalExecution;
import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.data.DataSetIteratorFactoryProvider;
import org.deeplearning4j.arbiter.optimize.api.data.DataSource;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxTimeCondition;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.generator.GeneticSearchCandidateGenerator;
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.discrete.DiscreteParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.saver.local.FileModelSaver;
import org.deeplearning4j.arbiter.scoring.ScoreFunctions;
import org.deeplearning4j.arbiter.scoring.impl.TestSetLossScoreFunction;
import org.deeplearning4j.arbiter.task.ComputationGraphTaskCreator;
import org.deeplearning4j.arbiter.util.TestDataFactoryProviderMnist;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MultiDataSetIteratorAdapter;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculatorCG;
import org.deeplearning4j.earlystopping.scorecalc.ScoreCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.function.Supplier;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

@Slf4j
public class TestGraphLocalExecutionGenetic extends BaseDL4JTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Override
    public long getTimeoutMilliseconds() {
        return 45000L;
    }

    @Test
    public void testLocalExecutionDataSources() throws Exception {
        for (int dataApproach = 0; dataApproach < 3; dataApproach++) {
            log.info("////////////////// Starting Test: {} ///////////////////", dataApproach);

            //Define: network config (hyperparameter space)
            ComputationGraphSpace mls = new ComputationGraphSpace.Builder()
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(new SgdSpace(new ContinuousParameterSpace(0.0001, 0.1)))
                    .l2(new ContinuousParameterSpace(0.0001, 0.01))
                    .addInputs("in")
                    .addLayer("0",
                            new DenseLayerSpace.Builder().nIn(784).nOut(new IntegerParameterSpace(5, 32))
                                    .activation(new DiscreteParameterSpace<>(Activation.RELU, Activation.TANH, Activation.LEAKYRELU))
                                    .build(), "in") //1-2 identical layers (except nIn)
                    .addLayer("1", new OutputLayerSpace.Builder().nOut(10).activation(Activation.SOFTMAX)
                            .lossFunction(LossFunctions.LossFunction.MCXENT).build(), "0")
                    .setOutputs("1")
                    .setInputTypes(InputType.feedForward(784))
                    .numEpochs(3).build();

            DataProvider dp = null;
            Class<? extends DataSource> ds = null;
            Properties dsP = null;
            CandidateGenerator candidateGenerator;

            TestSetLossScoreFunction scoreFunction = new TestSetLossScoreFunction();

            if (dataApproach == 0) {
                ds = TestDL4JLocalExecution.MnistDataSource.class;
                dsP = new Properties();
                dsP.setProperty("minibatch", "2");

                candidateGenerator = new GeneticSearchCandidateGenerator.Builder(mls, scoreFunction)
                        .populationModel(new PopulationModel.Builder().populationSize(5).build())
                        .build();
            } else if (dataApproach == 1) {
                //DataProvider approach
                dp = new TestDL4JLocalExecution.MnistDataProvider();

                candidateGenerator = new GeneticSearchCandidateGenerator.Builder(mls, scoreFunction)
                        .populationModel(new PopulationModel.Builder().populationSize(5).build())
                        .build();
            } else {
                //Factory approach
                Map<String, Object> commands = new HashMap<>();
                commands.put(DataSetIteratorFactoryProvider.FACTORY_KEY, TestDataFactoryProviderMnist.class.getCanonicalName());

                candidateGenerator = new GeneticSearchCandidateGenerator.Builder(mls, scoreFunction)
                        .dataParameters(commands)
                        .populationModel(new PopulationModel.Builder().populationSize(5).build())
                        .build();
                dp = new DataSetIteratorFactoryProvider();
            }

            File f = testDir.newFolder();
            File modelSave = new File(f, "modelSaveDir");

            OptimizationConfiguration configuration = new OptimizationConfiguration.Builder()
                    .candidateGenerator(candidateGenerator)
                    .dataProvider(dp)
                    .dataSource(ds, dsP)
                    .modelSaver(new FileModelSaver(modelSave))
                    .scoreFunction(new TestSetLossScoreFunction())
                    .terminationConditions(new MaxTimeCondition(5, TimeUnit.SECONDS),
                            new MaxCandidatesCondition(10))
                    .build();

            IOptimizationRunner runner = new LocalOptimizationRunner(configuration, new ComputationGraphTaskCreator(new ClassificationEvaluator()));

            runner.execute();

            List<ResultReference> results = runner.getResults();
            assertTrue(results.size() > 0);

//            System.out.println("----- COMPLETE - " + results.size() + " results -----");
        }
    }

    public static class TestMdsDataProvider implements DataProvider {
        private int numEpochs;
        private int batchSize;

        public TestMdsDataProvider(@JsonProperty("numEpochs") int numEpochs, @JsonProperty("batchSize") int batchSize) {
            this.numEpochs = numEpochs;
            this.batchSize = batchSize;
        }

        private TestMdsDataProvider() {
        }


        @Override
        public Object trainData(Map<String, Object> dataParameters) {
            try {
                DataSetIterator underlying = new MnistDataSetIterator(batchSize, Math.min(60000, 10 * batchSize), false, true, true, 12345);
                return new MultiDataSetIteratorAdapter(new MultipleEpochsIterator(numEpochs, underlying));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        @Override
        public Object testData(Map<String, Object> dataParameters) {
            try {
                DataSetIterator underlying = new MnistDataSetIterator(batchSize, Math.min(10000, 5 * batchSize), false, false, false, 12345);
                return new MultiDataSetIteratorAdapter(underlying);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        @Override
        public Class<?> getDataType() {
            return MultiDataSetIterator.class;
        }
    }

    private static class ScoreProvider implements Supplier<ScoreCalculator>, Serializable {
        @Override
        public ScoreCalculator get() {
            try {
                return new DataSetLossCalculatorCG(new MnistDataSetIterator(128, 1280), true);
            } catch (Exception e){
                throw new RuntimeException(e);
            }
        }
    }
}
