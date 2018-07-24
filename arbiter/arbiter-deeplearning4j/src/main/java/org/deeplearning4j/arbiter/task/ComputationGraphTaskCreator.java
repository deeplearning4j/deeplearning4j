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

package org.deeplearning4j.arbiter.task;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.deeplearning4j.arbiter.GraphConfiguration;
import org.deeplearning4j.arbiter.listener.DL4JArbiterStatusReportingListener;
import org.deeplearning4j.arbiter.optimize.api.Candidate;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.TaskCreator;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.data.DataSource;
import org.deeplearning4j.arbiter.optimize.api.evaluation.ModelEvaluator;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultSaver;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.runner.CandidateInfo;
import org.deeplearning4j.arbiter.optimize.runner.CandidateStatus;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.listener.StatusListener;
import org.deeplearning4j.arbiter.scoring.util.ScoreUtil;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.function.BiConsumer;
import org.nd4j.linalg.function.BiFunction;

import java.io.IOException;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.Callable;

/**
 * Task creator for ComputationGraph
 *
 * @author Alex Black
 */
@AllArgsConstructor
@NoArgsConstructor
@Slf4j
public class ComputationGraphTaskCreator implements TaskCreator {

    private ModelEvaluator modelEvaluator;
    @Getter
    @Setter
    private TaskListener taskListener;

    public ComputationGraphTaskCreator(ModelEvaluator modelEvaluator){
        this(modelEvaluator, null);
    }

    @Override
    public Callable<OptimizationResult> create(Candidate candidate, DataProvider dataProvider,
                                               ScoreFunction scoreFunction, List<StatusListener> statusListener,
                                               IOptimizationRunner runner) {

        return new GraphLearningTask(candidate, dataProvider, scoreFunction, modelEvaluator, statusListener,
                taskListener, runner);
    }

    @Override
    public Callable<OptimizationResult> create(Candidate candidate, Class<? extends DataSource> dataSource, Properties dataSourceProperties,
                                               ScoreFunction scoreFunction, List<StatusListener> statusListeners, IOptimizationRunner runner) {
        return new GraphLearningTask(candidate, dataSource, dataSourceProperties, scoreFunction, modelEvaluator, statusListeners,
                taskListener, runner);
    }

    @AllArgsConstructor
    private static class GraphLearningTask implements Callable<OptimizationResult> {

        private Candidate candidate;
        private DataProvider dataProvider;
        private Class<? extends DataSource> dataSource;
        private Properties dataSourceProperties;
        private ScoreFunction scoreFunction;
        private ModelEvaluator modelEvaluator;
        private List<StatusListener> listeners;
        private TaskListener taskListener;
        private IOptimizationRunner runner;

        private long startTime;

        public GraphLearningTask(Candidate candidate, DataProvider dataProvider, ScoreFunction scoreFunction,
                                 ModelEvaluator modelEvaluator, List<StatusListener> listeners,
                                 TaskListener taskListener, IOptimizationRunner runner) {
            this.candidate = candidate;
            this.dataProvider = dataProvider;
            this.scoreFunction = scoreFunction;
            this.modelEvaluator = modelEvaluator;
            this.listeners = listeners;
            this.taskListener = taskListener;
            this.runner = runner;
        }

        public GraphLearningTask(Candidate candidate, Class<? extends DataSource> dataSource, Properties dataSourceProperties,
                                 ScoreFunction scoreFunction, ModelEvaluator modelEvaluator, List<StatusListener> listeners,
                                 TaskListener taskListener, IOptimizationRunner runner) {
            this.candidate = candidate;
            this.dataSource = dataSource;
            this.dataSourceProperties = dataSourceProperties;
            this.scoreFunction = scoreFunction;
            this.modelEvaluator = modelEvaluator;
            this.listeners = listeners;
            this.taskListener = taskListener;
            this.runner = runner;
        }


        @Override
        public OptimizationResult call() throws Exception {

            try {
                OptimizationResult result = callHelper();
                if(listeners != null && !listeners.isEmpty()){
                    CandidateInfo ci = new CandidateInfo(candidate.getIndex(), CandidateStatus.Complete, result.getScore(),
                            startTime, startTime, System.currentTimeMillis(), candidate.getFlatParameters(), null);
                    for(StatusListener sl : listeners){
                        try{
                            sl.onCandidateStatusChange(ci, runner, result);
                        } catch (Exception e){
                            log.error("Error in status listener for candidate {}", candidate.getIndex(), e);
                        }
                    }
                }
                return result;
            } catch (Throwable e) {
                String stackTrace = ExceptionUtils.getStackTrace(e);
                log.warn("Execution failed for task {}", candidate.getIndex(), e);

                CandidateInfo ci = new CandidateInfo(candidate.getIndex(), CandidateStatus.Failed, null, startTime,
                        null, null, candidate.getFlatParameters(), stackTrace);
                return new OptimizationResult(candidate, null, candidate.getIndex(), null, ci, null);
            }  finally {
                //Destroy workspaces to free memory
                Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
                System.gc();
                try {
                    //Sleep for a few seconds - workspace destruction and memory deallocation happens quickly but doesn't
                    // happen instantly; if we didn't have this, we may run into a situation where the next thread/task
                    // tries to allocate before WS memory is fully deallocated, resulting in an OOM in memory constrained
                    // environments
                    Thread.sleep(2000L);
                } catch (Exception e){ }
            }
        }

        private OptimizationResult callHelper() throws Exception {
            startTime = System.currentTimeMillis();
            CandidateInfo ci = new CandidateInfo(candidate.getIndex(), CandidateStatus.Running, null, startTime, startTime,
                    null, candidate.getFlatParameters(), null);

            //Create network
            ComputationGraph net = new ComputationGraph(((GraphConfiguration) candidate.getValue()).getConfiguration());
            net.init();

            if(taskListener != null){
                net = taskListener.preProcess(net, candidate);
            }

            if (listeners != null) {
                net.addListeners(new DL4JArbiterStatusReportingListener(listeners, ci));
            }

            //For DataSetIterator: wraps in a MultiDataSetIterator, hence method can be used for both
            MultiDataSetIterator iterator;
            if(dataSource != null){
                DataSource dsInstance = dataSource.newInstance();
                if(dataSourceProperties != null)
                    dsInstance.configure(dataSourceProperties);
                iterator = ScoreUtil.getMultiIterator(dsInstance.trainData());
            } else {
                iterator = ScoreUtil.getMultiIterator(dataProvider.trainData(candidate.getDataParameters()));
            }


            EarlyStoppingConfiguration<ComputationGraph> esConfig =
                    ((GraphConfiguration) candidate.getValue()).getEarlyStoppingConfiguration();
            EarlyStoppingResult<ComputationGraph> esResult = null;
            if (esConfig != null) {
                EarlyStoppingGraphTrainer trainer = new EarlyStoppingGraphTrainer(esConfig, net, iterator, null);
                esResult = trainer.fit();
                net = esResult.getBestModel(); //Can return null if failed OR if

                switch (esResult.getTerminationReason()) {
                    case Error:
                        ci.setCandidateStatus(CandidateStatus.Failed);
                        ci.setExceptionStackTrace(esResult.getTerminationDetails());
                        break;
                    case IterationTerminationCondition:
                    case EpochTerminationCondition:
                        ci.setCandidateStatus(CandidateStatus.Complete);
                        break;
                }

            } else {
                //Fixed number of epochs
                int nEpochs = ((GraphConfiguration) candidate.getValue()).getNumEpochs();
                for (int i = 0; i < nEpochs; i++) {
                    net.fit(iterator);
                }
                ci.setCandidateStatus(CandidateStatus.Complete);
            }
            Nd4j.getExecutioner().commit();

            Object additionalEvaluation = null;
            if (esConfig != null && esResult.getTerminationReason() != EarlyStoppingResult.TerminationReason.Error) {
                additionalEvaluation =
                        (modelEvaluator != null ? modelEvaluator.evaluateModel(net, dataProvider) : null);
            }

            Double score = null;
            if (net != null) {
                score = scoreFunction.score(net, dataProvider, candidate.getDataParameters());
                ci.setScore(score);
            }

            if(taskListener != null){
                taskListener.postProcess(net, candidate);
            }

            OptimizationResult result = new OptimizationResult(candidate, score, candidate.getIndex(), additionalEvaluation, ci, null);

            //Save the model:
            ResultSaver saver = runner.getConfiguration().getResultSaver();
            ResultReference resultReference = null;
            if (saver != null) {
                try {
                    resultReference = saver.saveModel(result, net);
                } catch (IOException e) {
                    //TODO: Do we want ta warn or fail on IOException?
                    log.warn("Error saving model (id={}): IOException thrown. ", result.getIndex(), e);
                }
            }
            result.setResultReference(resultReference);
            return result;
        }
    }
}
