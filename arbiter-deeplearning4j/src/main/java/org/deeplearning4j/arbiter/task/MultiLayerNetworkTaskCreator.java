/*-
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
package org.deeplearning4j.arbiter.task;

import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.deeplearning4j.arbiter.DL4JConfiguration;
import org.deeplearning4j.arbiter.listener.BaseUIStatusReportingListener;
import org.deeplearning4j.arbiter.listener.UIStatusReportingListener;
import org.deeplearning4j.arbiter.optimize.api.Candidate;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.TaskCreator;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.evaluation.ModelEvaluator;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.runner.Status;
import org.deeplearning4j.arbiter.optimize.runner.listener.candidate.UICandidateStatusListener;

import org.deeplearning4j.arbiter.optimize.ui.ArbiterUIServer;
import org.deeplearning4j.arbiter.scoring.util.ScoreUtil;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.components.text.ComponentText;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.concurrent.Callable;

/**
 * Task creator for MultiLayerNetworks
 *
 * @param <A> Additional evaluation type
 * @author Alex Black
 */
@AllArgsConstructor
@NoArgsConstructor
public class MultiLayerNetworkTaskCreator<A> implements TaskCreator<DL4JConfiguration, MultiLayerNetwork, Object, A> {

    private ModelEvaluator<MultiLayerNetwork, Object, A> modelEvaluator;

    @Override
    public Callable<OptimizationResult<DL4JConfiguration, MultiLayerNetwork, A>> create(
            Candidate<DL4JConfiguration> candidate, DataProvider<Object> dataProvider,
            ScoreFunction<MultiLayerNetwork, Object> scoreFunction,
            UICandidateStatusListener statusListener) {

        return new DL4JLearningTask<>(candidate, dataProvider, scoreFunction, modelEvaluator, statusListener);

    }


    private static class DL4JLearningTask<A> implements Callable<OptimizationResult<DL4JConfiguration, MultiLayerNetwork, A>> {

        private Candidate<DL4JConfiguration> candidate;
        private DataProvider<Object> dataProvider;
        private ScoreFunction<MultiLayerNetwork, Object> scoreFunction;
        private ModelEvaluator<MultiLayerNetwork, Object, A> modelEvaluator;

        private BaseUIStatusReportingListener<MultiLayerNetwork> dl4jListener;

        public DL4JLearningTask(Candidate<DL4JConfiguration> candidate,
                                DataProvider<Object> dataProvider,
                                ScoreFunction<MultiLayerNetwork, Object> scoreFunction,
                                ModelEvaluator<MultiLayerNetwork, Object, A> modelEvaluator,
                                UICandidateStatusListener listener) {
            this.candidate = candidate;
            this.dataProvider = dataProvider;
            this.scoreFunction = scoreFunction;
            this.modelEvaluator = modelEvaluator;

            dl4jListener = (ArbiterUIServer.isRunning() ? new UIStatusReportingListener(listener) : null);
        }


        @Override
        public OptimizationResult<DL4JConfiguration, MultiLayerNetwork, A> call() throws Exception {
            //Create network
            MultiLayerNetwork net = new MultiLayerNetwork(candidate.getValue().getMultiLayerConfiguration());
            net.init();
            net.setListeners(dl4jListener);

            //Early stopping or fixed number of epochs:
            DataSetIterator dataSetIterator = ScoreUtil.getIterator(dataProvider.trainData(candidate.getDataParameters()));


            EarlyStoppingConfiguration<MultiLayerNetwork> esConfig = candidate.getValue().getEarlyStoppingConfiguration();
            EarlyStoppingResult<MultiLayerNetwork> esResult = null;
            if (esConfig != null) {
                EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConfig, net, dataSetIterator, dl4jListener);
                try {
                    esResult = trainer.fit();
                    net = esResult.getBestModel();  //Can return null if failed OR if
                } catch (Exception e) {
                    if(dl4jListener != null) {
                        dl4jListener.postReport(Status.Failed, null,
                                new ComponentText("Unexpected exception during model training\n", null),
                                new ComponentText(ExceptionUtils.getStackTrace(e), null));
                    }
                    throw e;
                }

                switch (esResult.getTerminationReason()) {
                    case Error:
                        if(dl4jListener != null) {
                            dl4jListener.postReport(Status.Failed, esResult);
                        }
                        break;
                    case IterationTerminationCondition:
                    case EpochTerminationCondition:
                        if(dl4jListener != null) {
                            dl4jListener.postReport(Status.Complete, esResult);
                        }
                        break;
                }

            } else {
                //Fixed number of epochs
                int nEpochs = candidate.getValue().getNumEpochs();
                for (int i = 0; i < nEpochs; i++) {
                    net.fit(dataSetIterator);
                    dataSetIterator.reset();
                }
                //Do a final status update
                if(dl4jListener != null) {
                    dl4jListener.postReport(Status.Complete, null);
                }
            }

            A additionalEvaluation = null;
            if (esConfig != null && esResult.getTerminationReason() != EarlyStoppingResult.TerminationReason.Error) {
                try {
                    additionalEvaluation = (modelEvaluator != null ? modelEvaluator.evaluateModel(net, dataProvider) : null);
                } catch (Exception e) {
                    if(dl4jListener != null) {
                        dl4jListener.postReport(Status.Failed, esResult,
                                new ComponentText("Failed during additional evaluation stage\n", null),
                                new ComponentText(ExceptionUtils.getStackTrace(e), null));
                    }
                }
            }

            Double score = null;
            if (net == null) {
                if(dl4jListener != null) {
                    dl4jListener.postReport(Status.Complete, esResult,
                            new ComponentText("No best model available; cannot calculate model score", null));
                }
            } else {
                try {
                    score = scoreFunction.score(net, dataProvider, candidate.getDataParameters());
                } catch (Exception e) {
                    if(dl4jListener != null) {
                        dl4jListener.postReport(Status.Failed, esResult,
                                new ComponentText("Failed during score calculation stage\n", null),
                                new ComponentText(ExceptionUtils.getStackTrace(e), null));
                    }
                }
            }

            return new OptimizationResult<>(candidate, net, score, candidate.getIndex(), additionalEvaluation);
        }
    }
}
