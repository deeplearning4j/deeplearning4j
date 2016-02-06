/*
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
package org.arbiter.deeplearning4j.task;

import lombok.AllArgsConstructor;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.arbiter.deeplearning4j.DL4JConfiguration;
import org.arbiter.deeplearning4j.listener.BaseUIStatusReportingListener;
import org.arbiter.deeplearning4j.listener.UIStatusReportingListener;
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
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.util.concurrent.Callable;

@AllArgsConstructor
public class MultiLayerNetworkTaskCreator<A> implements TaskCreator<DL4JConfiguration,MultiLayerNetwork,DataSetIterator,A>{

    private ModelEvaluator<MultiLayerNetwork,DataSetIterator,A> modelEvaluator;

    @Override
    public Callable<OptimizationResult<DL4JConfiguration, MultiLayerNetwork, A>> create(
                Candidate<DL4JConfiguration> candidate, DataProvider<DataSetIterator> dataProvider,
                   ScoreFunction<MultiLayerNetwork,DataSetIterator> scoreFunction,
                   UICandidateStatusListener statusListener) {

        return new DL4JLearningTask<>(candidate,dataProvider,scoreFunction,modelEvaluator,statusListener);

    }


    private static class DL4JLearningTask<A> implements Callable<OptimizationResult<DL4JConfiguration,MultiLayerNetwork,A>> {

        private Candidate<DL4JConfiguration> candidate;
        private DataProvider<DataSetIterator> dataProvider;
        private ScoreFunction<MultiLayerNetwork,DataSetIterator> scoreFunction;
        private ModelEvaluator<MultiLayerNetwork,DataSetIterator,A> modelEvaluator;

        private BaseUIStatusReportingListener<MultiLayerNetwork> dl4jListener;

        public DL4JLearningTask(Candidate<DL4JConfiguration> candidate, DataProvider<DataSetIterator> dataProvider, ScoreFunction<MultiLayerNetwork, DataSetIterator> scoreFunction, ModelEvaluator<MultiLayerNetwork, DataSetIterator, A> modelEvaluator, UICandidateStatusListener listener) {
            this.candidate = candidate;
            this.dataProvider = dataProvider;
            this.scoreFunction = scoreFunction;
            this.modelEvaluator = modelEvaluator;

            dl4jListener = new UIStatusReportingListener(listener);
        }


        @Override
        public OptimizationResult<DL4JConfiguration, MultiLayerNetwork,A> call() throws Exception {
            //Create network
            MultiLayerNetwork net = new MultiLayerNetwork(candidate.getValue().getMultiLayerConfiguration());
            net.init();
            net.setListeners(dl4jListener);

            //Early stopping or fixed number of epochs:
            DataSetIterator dataSetIterator = dataProvider.testData(candidate.getDataParameters());


            EarlyStoppingConfiguration<MultiLayerNetwork> esConfig = candidate.getValue().getEarlyStoppingConfiguration();
            EarlyStoppingResult<MultiLayerNetwork> esResult = null;
            if(esConfig != null){
                EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConfig,net,dataSetIterator,dl4jListener);
                try{
                    esResult = trainer.fit();
                    net = esResult.getBestModel();  //Can return null if failed OR if
                } catch(Exception e){
                    dl4jListener.postReport(Status.Failed, null,
                            new RenderableComponentString("Unexpected exception during model training\n"),
                            new RenderableComponentString(ExceptionUtils.getFullStackTrace(e)));
                    throw e;
                }

                switch(esResult.getTerminationReason()){
                    case Error:
                        dl4jListener.postReport(Status.Failed, esResult);
                        break;
                    case IterationTerminationCondition:
                    case EpochTerminationCondition:
                        dl4jListener.postReport(Status.Complete, esResult);
                        break;
                }

            } else {
                //Fixed number of epochs
                int nEpochs = candidate.getValue().getNumEpochs();
                for( int i=0; i<nEpochs; i++){
                    net.fit(dataSetIterator);
                    dataSetIterator.reset();
                }
                //Do a final status update
                dl4jListener.postReport(Status.Complete,null);
            }

            A additionalEvaluation = null;
            if( esConfig != null && esResult.getTerminationReason() != EarlyStoppingResult.TerminationReason.Error ) {
                try {
                    additionalEvaluation = (modelEvaluator != null ? modelEvaluator.evaluateModel(net, dataProvider) : null);
                } catch (Exception e) {
                    dl4jListener.postReport(Status.Failed, esResult,
                            new RenderableComponentString("Failed during additional evaluation stage\n"),
                            new RenderableComponentString(ExceptionUtils.getFullStackTrace(e)));
                }
            }

            Double score = null;
            if(net == null){
                dl4jListener.postReport(Status.Complete, esResult,
                        new RenderableComponentString("No best model available; cannot calculate model score"));
            } else {
                try {
                    score = scoreFunction.score(net, dataProvider, candidate.getDataParameters());
                } catch (Exception e) {
                    dl4jListener.postReport(Status.Failed, esResult,
                            new RenderableComponentString("Failed during score calculation stage\n"),
                            new RenderableComponentString(ExceptionUtils.getFullStackTrace(e)));
                }
            }

            return new OptimizationResult<>(candidate, net, score, candidate.getIndex(), additionalEvaluation);
        }
    }
}
