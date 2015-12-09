package org.arbiter.deeplearning4j.task;

import lombok.AllArgsConstructor;
import org.arbiter.optimize.api.Candidate;
import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.api.TaskCreator;
import org.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.util.concurrent.Callable;

public class DL4JTaskCreator implements TaskCreator<MultiLayerConfiguration,MultiLayerNetwork>{


    @Override
    public Callable<OptimizationResult<MultiLayerConfiguration, MultiLayerNetwork>> create(Candidate<MultiLayerConfiguration> candidate, DataProvider<?> dataProvider) {

        return new DL4JLearningTask(candidate.getValue(),dataProvider);

    }


    @AllArgsConstructor
    private static class DL4JLearningTask implements Callable<OptimizationResult<MultiLayerConfiguration,MultiLayerNetwork>> {

        private MultiLayerConfiguration conf;
        private DataProvider<?> dataProvider;


        @Override
        public OptimizationResult<MultiLayerConfiguration, MultiLayerNetwork> call() throws Exception {
            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            DataSetIterator dataSetIterator = (DataSetIterator)dataProvider.testData();

            net.fit(dataSetIterator);

            return new OptimizationResult<>(conf,net,0.0);  //TODO: scoring
        }
    }
}
