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

package org.deeplearning4j.earlystopping.trainer;

import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener;
import org.deeplearning4j.earlystopping.scorecalc.ScoreCalculator;
import org.deeplearning4j.earlystopping.termination.EpochTerminationCondition;
import org.deeplearning4j.earlystopping.termination.IterationTerminationCondition;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;

/**Base/abstract class for conducting early stopping training locally (single machine).<br>
 * Can be used to train a {@link MultiLayerNetwork} or a {@link ComputationGraph} via early stopping
 * @author Alex Black
 */
public abstract class BaseEarlyStoppingTrainer<T extends Model> implements IEarlyStoppingTrainer<T> {

    private static Logger log = LoggerFactory.getLogger(BaseEarlyStoppingTrainer.class);

    protected T model;

    protected final EarlyStoppingConfiguration<T> esConfig;
    private final DataSetIterator train;
    private final MultiDataSetIterator trainMulti;
    private final Iterator<?> iterator;
    private EarlyStoppingListener<T> listener;

    private double bestModelScore = Double.MAX_VALUE;
    private int bestModelEpoch = -1;

    protected BaseEarlyStoppingTrainer(EarlyStoppingConfiguration<T> earlyStoppingConfiguration, T model,
                                       DataSetIterator train, MultiDataSetIterator trainMulti, EarlyStoppingListener<T> listener) {
        this.esConfig = earlyStoppingConfiguration;
        this.model = model;
        this.train = train;
        this.trainMulti = trainMulti;
        this.iterator = (train != null ? train : trainMulti);
        this.listener = listener;
    }

    protected abstract void fit(DataSet ds);

    protected abstract void fit(MultiDataSet mds);

    @Override
    public EarlyStoppingResult<T> fit() {
        esConfig.validate();
        log.info("Starting early stopping training");
        if (esConfig.getScoreCalculator() == null)
            log.warn("No score calculator provided for early stopping. Score will be reported as 0.0 to epoch termination conditions");

        //Initialize termination conditions:
        if (esConfig.getIterationTerminationConditions() != null) {
            for (IterationTerminationCondition c : esConfig.getIterationTerminationConditions()) {
                c.initialize();
            }
        }
        if (esConfig.getEpochTerminationConditions() != null) {
            for (EpochTerminationCondition c : esConfig.getEpochTerminationConditions()) {
                c.initialize();
            }
        }

        if (listener != null) {
            listener.onStart(esConfig, model);
        }

        Map<Integer, Double> scoreVsEpoch = new LinkedHashMap<>();

        Preconditions.checkNotNull(esConfig.getScoreCalculator(), "Score calculator cannot be null");
        if(esConfig.getScoreCalculator().minimizeScore()){
            bestModelScore = Double.MAX_VALUE;
        } else {
            bestModelScore = -Double.MAX_VALUE;
        }

        int epochCount = 0;
        while (true) {
            reset();
            double lastScore;
            boolean terminate = false;
            IterationTerminationCondition terminationReason = null;
            int iterCount = 0;
            triggerEpochListeners(true, model, epochCount);
            while (iterator.hasNext()) {
                try {
                    if (train != null) {
                        fit((DataSet) iterator.next());
                    } else
                        fit(trainMulti.next());
                } catch (Exception e) {
                    log.warn("Early stopping training terminated due to exception at epoch {}, iteration {}",
                            epochCount, iterCount, e);
                    //Load best model to return
                    T bestModel;
                    try {
                        bestModel = esConfig.getModelSaver().getBestModel();
                    } catch (IOException e2) {
                        throw new RuntimeException(e2);
                    }
                    return new EarlyStoppingResult<>(EarlyStoppingResult.TerminationReason.Error, e.toString(),
                            scoreVsEpoch, bestModelEpoch, bestModelScore, epochCount, bestModel);
                }

                //Check per-iteration termination conditions
                lastScore = model.score();
                for (IterationTerminationCondition c : esConfig.getIterationTerminationConditions()) {
                    if (c.terminate(lastScore)) {
                        terminate = true;
                        terminationReason = c;
                        break;
                    }
                }
                if (terminate) {
                    break;
                }

                iterCount++;
            }

            if(!iterator.hasNext()){
                //End of epoch (if iterator does have next - means terminated)
                triggerEpochListeners(false, model, epochCount);
            }

            if (terminate) {
                //Handle termination condition:
                log.info("Hit per iteration epoch termination condition at epoch {}, iteration {}. Reason: {}",
                        epochCount, iterCount, terminationReason);

                if (esConfig.isSaveLastModel()) {
                    //Save last model:
                    try {
                        esConfig.getModelSaver().saveLatestModel(model, 0.0);
                    } catch (IOException e) {
                        //best model not saved, let's just use default
                        if(e instanceof FileNotFoundException) {

                        }
                        else
                            throw new RuntimeException("Error saving most recent model", e);
                    }
                }

                T bestModel;
                try {
                    bestModel = esConfig.getModelSaver().getBestModel();
                } catch (IOException e2) {
                    throw new RuntimeException(e2);
                }


                EarlyStoppingResult<T> result = new EarlyStoppingResult<>(
                        EarlyStoppingResult.TerminationReason.IterationTerminationCondition,
                        terminationReason.toString(), scoreVsEpoch, bestModelEpoch, bestModelScore, epochCount,
                        bestModel);
                if (listener != null) {
                    listener.onCompletion(result);
                }
                return result;
            }

            log.info("Completed training epoch {}", epochCount);


            if ((epochCount == 0 && esConfig.getEvaluateEveryNEpochs() == 1)
                    || epochCount % esConfig.getEvaluateEveryNEpochs() == 0) {
                //Calculate score at this epoch:
                ScoreCalculator sc = esConfig.getScoreCalculator();
                double score = esConfig.getScoreCalculator().calculateScore(model);
                scoreVsEpoch.put(epochCount, score);

                boolean invalidScore = Double.isNaN(score) || Double.isInfinite(score);
                if(invalidScore){
                    log.warn("Score is not finite for epoch {}: score = {}", epochCount, score);
                }

                if ((sc.minimizeScore() && score < bestModelScore) || (!sc.minimizeScore() && score > bestModelScore) || (bestModelEpoch == -1 && invalidScore)) {
                    //Save best model:
                    if (bestModelEpoch == -1) {
                        //First calculated/reported score
                        log.info("Score at epoch {}: {}", epochCount, score);
                    } else {
                        log.info("New best model: score = {}, epoch = {} (previous: score = {}, epoch = {})", score,
                                epochCount, bestModelScore, bestModelEpoch);
                    }
                    bestModelScore = score;
                    bestModelEpoch = epochCount;

                    try {
                        esConfig.getModelSaver().saveBestModel(model, score);
                    } catch (IOException e) {
                        throw new RuntimeException("Error saving best model", e);
                    }
                } else {
                    log.info("Score at epoch {}: {}", epochCount, score);
                }

                if (esConfig.isSaveLastModel()) {
                    //Save last model:
                    try {
                        esConfig.getModelSaver().saveLatestModel(model, score);
                    } catch (IOException e) {
                        throw new RuntimeException("Error saving most recent model", e);
                    }
                }

                if (listener != null) {
                    listener.onEpoch(epochCount, score, esConfig, model);
                }

                //Check per-epoch termination conditions:
                boolean epochTerminate = false;
                EpochTerminationCondition termReason = null;
                for (EpochTerminationCondition c : esConfig.getEpochTerminationConditions()) {
                    if (c.terminate(epochCount, score)) {
                        epochTerminate = true;
                        termReason = c;
                        break;
                    }
                }
                if (epochTerminate) {
                    log.info("Hit epoch termination condition at epoch {}. Details: {}", epochCount,
                            termReason.toString());
                    T bestModel;
                    try {
                        bestModel = esConfig.getModelSaver().getBestModel();
                    } catch (IOException e2) {
                        //Best model does not exist. Just save the current model
                        if(esConfig.isSaveLastModel()) {
                            try {
                                esConfig.getModelSaver().saveBestModel(model,0.0);
                                bestModel = model;
                            } catch (IOException e) {
                                log.error("Unable to save model.",e);
                                throw new RuntimeException(e);
                            }
                        }
                        else {
                            log.error("Error with earlystopping",e2);
                            throw new RuntimeException(e2);
                        }

                    }


                    EarlyStoppingResult<T> result = new EarlyStoppingResult<>(
                            EarlyStoppingResult.TerminationReason.EpochTerminationCondition,
                            termReason.toString(), scoreVsEpoch, bestModelEpoch, bestModelScore, epochCount + 1,
                            bestModel);
                    if (listener != null) {
                        listener.onCompletion(result);
                    }

                    return result;
                }
            }
            epochCount++;

        }
    }

    @Override
    public void setListener(EarlyStoppingListener<T> listener) {
        this.listener = listener;
    }

    //Trigger epoch listener methods manually - these won't be triggered due to not calling fit(DataSetIterator) etc
    protected void triggerEpochListeners(boolean epochStart, Model model, int epochNum){
        Collection<TrainingListener> listeners;
        if(model instanceof MultiLayerNetwork){
            MultiLayerNetwork n = ((MultiLayerNetwork) model);
            listeners = n.getListeners();
            n.setEpochCount(epochNum);
        } else if(model instanceof ComputationGraph){
            ComputationGraph cg = ((ComputationGraph) model);
            listeners = cg.getListeners();
            cg.getConfiguration().setEpochCount(epochNum);
        } else {
            return;
        }

        if(listeners != null && !listeners.isEmpty()){
            for (TrainingListener l : listeners) {
                if (epochStart) {
                    l.onEpochStart(model);
                } else {
                    l.onEpochEnd(model);
                }
            }
        }
    }

    protected void reset() {
        if (train != null) {
            train.reset();
        }
        if (trainMulti != null) {
            trainMulti.reset();
        }
    }


}
