package org.arbiter.deeplearning4j.task;

import lombok.AllArgsConstructor;
import org.arbiter.optimize.api.Candidate;
import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.api.TaskCreator;
import org.arbiter.optimize.api.data.DataProvider;
import org.arbiter.optimize.api.evaluation.ModelEvaluator;
import org.arbiter.optimize.api.score.ScoreFunction;
import org.arbiter.optimize.runner.Status;
import org.arbiter.optimize.runner.listener.candidate.UICandidateStatusListener;
import org.arbiter.optimize.ui.components.RenderableComponentString;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.util.concurrent.Callable;

@AllArgsConstructor
public class DL4JTaskCreator<A> implements TaskCreator<MultiLayerConfiguration,MultiLayerNetwork,DataSetIterator,A>{

    private ModelEvaluator<MultiLayerNetwork,DataSetIterator,A> modelEvaluator;

    @Override
    public Callable<OptimizationResult<MultiLayerConfiguration, MultiLayerNetwork, A>>
            create(Candidate<MultiLayerConfiguration> candidate, DataProvider<DataSetIterator> dataProvider,
                   ScoreFunction<MultiLayerNetwork,DataSetIterator> scoreFunction,
                   UICandidateStatusListener statusListener) {

        return new DL4JLearningTask(candidate,dataProvider,scoreFunction,modelEvaluator,statusListener);

    }


    @AllArgsConstructor
    private static class DL4JLearningTask<A> implements Callable<OptimizationResult<MultiLayerConfiguration,MultiLayerNetwork,A>> {

        private Candidate<MultiLayerConfiguration> candidate;
        private DataProvider<DataSetIterator> dataProvider;
        private ScoreFunction<MultiLayerNetwork,DataSetIterator> scoreFunction;
        private ModelEvaluator<MultiLayerNetwork,DataSetIterator,A> modelEvaluator;
        private UICandidateStatusListener listener;


        @Override
        public OptimizationResult<MultiLayerConfiguration, MultiLayerNetwork,A> call() throws Exception {
            MultiLayerNetwork net = new MultiLayerNetwork(candidate.getValue());
            net.init();

            if(listener != null){
                listener.reportStatus(Status.Running,new RenderableComponentString("Running (todo)"));
            }

            DataSetIterator dataSetIterator = dataProvider.testData(candidate.getDataParameters());
            net.fit(dataSetIterator);

            //TODO: This only fits for a single epoch. Need additional functionality here:
            // (a) Early stopping (better)
            // (b) Specify number of epochs (less good, but perhaps worth supporting)

            A additionalEvaluation = (modelEvaluator != null ? modelEvaluator.evaluateModel(net,dataProvider) : null);

            if(listener != null){
                listener.reportStatus(Status.Complete,new RenderableComponentString("Complete (todo)"));
            }

            return new OptimizationResult<>(candidate,net,scoreFunction.score(net,dataProvider,candidate.getDataParameters()),
                    candidate.getIndex(), additionalEvaluation);
        }
    }
}
