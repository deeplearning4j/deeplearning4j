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

package org.deeplearning4j.spark.earlystopping;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener;
import org.deeplearning4j.earlystopping.scorecalc.ScoreCalculator;
import org.deeplearning4j.earlystopping.termination.EpochTerminationCondition;
import org.deeplearning4j.earlystopping.termination.IterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.IEarlyStoppingTrainer;
import org.deeplearning4j.nn.api.Model;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Base/abstract class for conducting early stopping training via Spark, on a {@link org.deeplearning4j.nn.multilayer.MultiLayerNetwork}
 * or a {@link org.deeplearning4j.nn.graph.ComputationGraph}
 * @author Alex Black
 */
public abstract class BaseSparkEarlyStoppingTrainer<T extends Model> implements IEarlyStoppingTrainer<T> {

    private static Logger log = LoggerFactory.getLogger(BaseSparkEarlyStoppingTrainer.class);

    private JavaSparkContext sc;
    private final EarlyStoppingConfiguration<T> esConfig;
    private T net;
    private final JavaRDD<DataSet> train;
    private final JavaRDD<MultiDataSet> trainMulti;
    private EarlyStoppingListener<T> listener;

    private double bestModelScore = Double.MAX_VALUE;
    private int bestModelEpoch = -1;

    protected BaseSparkEarlyStoppingTrainer(JavaSparkContext sc, EarlyStoppingConfiguration<T> esConfig, T net,
                    JavaRDD<DataSet> train, JavaRDD<MultiDataSet> trainMulti, EarlyStoppingListener<T> listener) {
        if ((esConfig.getEpochTerminationConditions() == null || esConfig.getEpochTerminationConditions().isEmpty())
                        && (esConfig.getIterationTerminationConditions() == null
                                        || esConfig.getIterationTerminationConditions().isEmpty())) {
            throw new IllegalArgumentException(
                            "Cannot conduct early stopping without a termination condition (both Iteration "
                                            + "and Epoch termination conditions are null/empty)");
        }

        this.sc = sc;
        this.esConfig = esConfig;
        this.net = net;
        this.train = train;
        this.trainMulti = trainMulti;
        this.listener = listener;
    }

    protected abstract void fit(JavaRDD<DataSet> data);

    protected abstract void fitMulti(JavaRDD<MultiDataSet> data);

    protected abstract double getScore();

    @Override
    public EarlyStoppingResult<T> fit() {
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

        if (listener != null)
            listener.onStart(esConfig, net);

        Map<Integer, Double> scoreVsEpoch = new LinkedHashMap<>();

        int epochCount = 0;
        while (true) { //Iterate (do epochs) until termination condition hit
            double lastScore;
            boolean terminate = false;
            IterationTerminationCondition terminationReason = null;

            if (train != null)
                fit(train);
            else
                fitMulti(trainMulti);

            //TODO revisit per iteration termination conditions, ensuring they are evaluated *per averaging* not per epoch
            //Check per-iteration termination conditions
            lastScore = getScore();
            for (IterationTerminationCondition c : esConfig.getIterationTerminationConditions()) {
                if (c.terminate(lastScore)) {
                    terminate = true;
                    terminationReason = c;
                    break;
                }
            }

            if (terminate) {
                //Handle termination condition:
                log.info("Hit per iteration epoch termination condition at epoch {}, iteration {}. Reason: {}",
                                epochCount, epochCount, terminationReason);

                if (esConfig.isSaveLastModel()) {
                    //Save last model:
                    try {
                        esConfig.getModelSaver().saveLatestModel(net, 0.0);
                    } catch (IOException e) {
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
                if (listener != null)
                    listener.onCompletion(result);
                return result;
            }



            log.info("Completed training epoch {}", epochCount);


            if ((epochCount == 0 && esConfig.getEvaluateEveryNEpochs() == 1)
                            || epochCount % esConfig.getEvaluateEveryNEpochs() == 0) {
                //Calculate score at this epoch:
                ScoreCalculator sc = esConfig.getScoreCalculator();
                double score = (sc == null ? 0.0 : esConfig.getScoreCalculator().calculateScore(net));
                scoreVsEpoch.put(epochCount - 1, score);

                if (sc != null && score < bestModelScore) {
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
                        esConfig.getModelSaver().saveBestModel(net, score);
                    } catch (IOException e) {
                        throw new RuntimeException("Error saving best model", e);
                    }
                }

                if (esConfig.isSaveLastModel()) {
                    //Save last model:
                    try {
                        esConfig.getModelSaver().saveLatestModel(net, score);
                    } catch (IOException e) {
                        throw new RuntimeException("Error saving most recent model", e);
                    }
                }

                if (listener != null)
                    listener.onEpoch(epochCount, score, esConfig, net);

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
                        throw new RuntimeException(e2);
                    }
                    EarlyStoppingResult<T> result = new EarlyStoppingResult<>(
                                    EarlyStoppingResult.TerminationReason.EpochTerminationCondition,
                                    termReason.toString(), scoreVsEpoch, bestModelEpoch, bestModelScore, epochCount + 1,
                                    bestModel);
                    if (listener != null)
                        listener.onCompletion(result);
                    return result;
                }

                epochCount++;
            }
        }
    }

    @Override
    public void setListener(EarlyStoppingListener<T> listener) {
        this.listener = listener;
    }
}
