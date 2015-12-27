package org.arbiter.deeplearning4j.task;

import lombok.AllArgsConstructor;
import org.arbiter.deeplearning4j.listener.UIStatusReportingListener;
import org.arbiter.optimize.api.Candidate;
import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.api.TaskCreator;
import org.arbiter.optimize.api.data.DataProvider;
import org.arbiter.optimize.api.evaluation.ModelEvaluator;
import org.arbiter.optimize.api.score.ScoreFunction;
import org.arbiter.optimize.runner.Status;
import org.arbiter.optimize.runner.listener.candidate.UICandidateStatusListener;
import org.arbiter.optimize.ui.components.RenderableComponent;
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


    private static class DL4JLearningTask<A> implements Callable<OptimizationResult<MultiLayerConfiguration,MultiLayerNetwork,A>> {

        private Candidate<MultiLayerConfiguration> candidate;
        private DataProvider<DataSetIterator> dataProvider;
        private ScoreFunction<MultiLayerNetwork,DataSetIterator> scoreFunction;
        private ModelEvaluator<MultiLayerNetwork,DataSetIterator,A> modelEvaluator;
        private UICandidateStatusListener listener;

        private UIStatusReportingListener dl4jListener;

        public DL4JLearningTask(Candidate<MultiLayerConfiguration> candidate, DataProvider<DataSetIterator> dataProvider, ScoreFunction<MultiLayerNetwork, DataSetIterator> scoreFunction, ModelEvaluator<MultiLayerNetwork, DataSetIterator, A> modelEvaluator, UICandidateStatusListener listener) {
            this.candidate = candidate;
            this.dataProvider = dataProvider;
            this.scoreFunction = scoreFunction;
            this.modelEvaluator = modelEvaluator;
            this.listener = listener;

            dl4jListener = new UIStatusReportingListener(listener);
        }


        @Override
        public OptimizationResult<MultiLayerConfiguration, MultiLayerNetwork,A> call() throws Exception {
            MultiLayerNetwork net = new MultiLayerNetwork(candidate.getValue());
            net.init();
            net.setListeners(dl4jListener);

//            reportStatus(net, Status.Running);

            DataSetIterator dataSetIterator = dataProvider.testData(candidate.getDataParameters());
            net.fit(dataSetIterator);

            //TODO: This only fits for a single epoch. Need additional functionality here:
            // (a) Early stopping (better)
            // (b) Specify number of epochs (less good, but perhaps worth supporting)

            A additionalEvaluation = (modelEvaluator != null ? modelEvaluator.evaluateModel(net,dataProvider) : null);

//            reportStatus(net,Status.Complete);

            OptimizationResult<MultiLayerConfiguration,MultiLayerNetwork,A> result = new OptimizationResult<>(candidate,net,scoreFunction.score(net,dataProvider,candidate.getDataParameters()),
                    candidate.getIndex(), additionalEvaluation);

            //Do a final status update
            dl4jListener.postReport(Status.Complete);   //TODO don't hardcode; don't do this if early stopping is used

            return result;
        }

//        private void reportStatus(MultiLayerNetwork network, Status status){
//            if(listener == null) return;
//            //Status to report:
//            //(a) configuration
//            //(b) score vs. iteration
//            //(c) score vs. epoch
//
//            RenderableComponent config = new RenderableComponentString(network.getLayerWiseConfigurations().toString());
//
//            listener.reportStatus(status,config);
//
//
//        }
    }
}
