package org.arbiter.deeplearning4j.task;

import lombok.AllArgsConstructor;
import org.arbiter.optimize.api.Candidate;
import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.api.TaskCreator;
import org.arbiter.optimize.api.data.DataProvider;
import org.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.util.concurrent.Callable;

public class DL4JTaskCreator implements TaskCreator<MultiLayerConfiguration,MultiLayerNetwork,DataSetIterator>{


    @Override
    public Callable<OptimizationResult<MultiLayerConfiguration, MultiLayerNetwork>>
            create(Candidate<MultiLayerConfiguration> candidate, DataProvider<DataSetIterator> dataProvider,
                   ScoreFunction<MultiLayerNetwork,DataSetIterator> scoreFunction) {

        return new DL4JLearningTask(candidate,dataProvider,scoreFunction);

    }


    @AllArgsConstructor
    private static class DL4JLearningTask implements Callable<OptimizationResult<MultiLayerConfiguration,MultiLayerNetwork>> {

        private Candidate<MultiLayerConfiguration> candidate;
        private DataProvider<DataSetIterator> dataProvider;
        private ScoreFunction<MultiLayerNetwork,DataSetIterator> scoreFunction;


        @Override
        public OptimizationResult<MultiLayerConfiguration, MultiLayerNetwork> call() throws Exception {
            MultiLayerNetwork net = new MultiLayerNetwork(candidate.getValue());
            net.init();

            DataSetIterator dataSetIterator = dataProvider.testData(candidate.getDataParameters());
            net.fit(dataSetIterator);

            //TODO: This only fits for a single epoch. Need additional functionality here:
            // (a) Early stopping (better)
            // (b) Specify number of epochs (less good, but perhaps worth supporting)

            return new OptimizationResult<>(candidate,net,scoreFunction.score(net,dataProvider,candidate.getDataParameters()),
                    candidate.getIndex());
        }
    }
}
