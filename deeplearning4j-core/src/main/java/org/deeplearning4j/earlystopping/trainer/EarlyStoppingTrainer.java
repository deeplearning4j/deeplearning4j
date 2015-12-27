/*
 *
 *  * Copyright 2015 Skymind,Inc.
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

package org.deeplearning4j.earlystopping.trainer;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener;
import org.deeplearning4j.earlystopping.scorecalc.ScoreCalculator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.earlystopping.termination.EpochTerminationCondition;
import org.deeplearning4j.earlystopping.termination.IterationTerminationCondition;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

/**Class for conducting early stopping training locally (single machine)
 */
public class EarlyStoppingTrainer implements IEarlyStoppingTrainer {

    private static Logger log = LoggerFactory.getLogger(EarlyStoppingTrainer.class);

    private final EarlyStoppingConfiguration esConfig;
    private MultiLayerNetwork net;
    private final DataSetIterator train;
    private EarlyStoppingListener listener;

    private double bestModelScore = Double.MAX_VALUE;
    private int bestModelEpoch = -1;


    public EarlyStoppingTrainer(EarlyStoppingConfiguration earlyStoppingConfiguration, MultiLayerConfiguration configuration,
                                DataSetIterator train) {
        this(earlyStoppingConfiguration,new MultiLayerNetwork(configuration),train);
        net.init();
    }

    public EarlyStoppingTrainer(EarlyStoppingConfiguration esConfig, MultiLayerNetwork net,
                                DataSetIterator train) {
        this(esConfig,net,train,null);
    }

    public EarlyStoppingTrainer(EarlyStoppingConfiguration esConfig, MultiLayerNetwork net,
                                DataSetIterator train, EarlyStoppingListener listener) {
        if((esConfig.getEpochTerminationConditions() == null || esConfig.getEpochTerminationConditions().size() == 0)
                && (esConfig.getIterationTerminationConditions() == null || esConfig.getIterationTerminationConditions().size() == 0)){
            throw new IllegalArgumentException("Cannot conduct early stopping without a termination condition (both Iteration "
                + "and Epoch termination conditions are null/empty)");
        }
        this.esConfig = esConfig;
        this.net = net;
        this.train = train;
        this.listener = listener;
    }

    @Override
    public EarlyStoppingResult fit() {
        log.info("Starting early stopping training");
        if(esConfig.getScoreCalculator() == null) log.warn("No score calculator provided for early stopping. Score will be reported as 0.0 to epoch termination conditions");

        //Initialize termination conditions:
        if(esConfig.getIterationTerminationConditions() != null){
            for( IterationTerminationCondition c : esConfig.getIterationTerminationConditions()){
                c.initialize();
            }
        }
        if(esConfig.getEpochTerminationConditions() != null){
            for( EpochTerminationCondition c : esConfig.getEpochTerminationConditions()){
                c.initialize();
            }
        }

        if(listener != null) listener.onStart(esConfig,net);

        Map<Integer,Double> scoreVsEpoch = new LinkedHashMap<>();

        int epochCount = 0;
        while (true) {
            train.reset();
            double lastScore;
            boolean terminate = false;
            IterationTerminationCondition terminationReason = null;
            int iterCount = 0;
            while (train.hasNext()) {
                DataSet ds = train.next();

                try {
                    net.fit(ds);
                } catch(Exception e){
                    log.warn("Early stopping training terminated due to exception at epoch {}, iteration {}",
                            epochCount,iterCount,e);
                    //Load best model to return
                    MultiLayerNetwork bestModel;
                    try{
                        bestModel = esConfig.getModelSaver().getBestModel();
                    }catch(IOException e2){
                        throw new RuntimeException(e2);
                    }
                    return new EarlyStoppingResult(
                            EarlyStoppingResult.TerminationReason.Error,
                            e.toString(),
                            scoreVsEpoch,
                            bestModelEpoch,
                            bestModelScore,
                            epochCount,
                            bestModel);
                }

                //Check per-iteration termination conditions
                lastScore = net.score();
                for (IterationTerminationCondition c : esConfig.getIterationTerminationConditions()) {
                    if (c.terminate(lastScore)) {
                        terminate = true;
                        terminationReason = c;
                        break;
                    }
                }
                if(terminate) break;

                iterCount++;
            }
            if(terminate){
                //Handle termination condition:
                log.info("Hit per iteration epoch termination condition at epoch {}, iteration {}. Reason: {}",
                        epochCount, iterCount, terminationReason);

                if(esConfig.isSaveLastModel()) {
                    //Save last model:
                    try {
                        esConfig.getModelSaver().saveLatestModel(net, 0.0);
                    } catch (IOException e) {
                        throw new RuntimeException("Error saving most recent model", e);
                    }
                }

                MultiLayerNetwork bestModel;
                try{
                    bestModel = esConfig.getModelSaver().getBestModel();
                }catch(IOException e2){
                    throw new RuntimeException(e2);
                }


                EarlyStoppingResult result = new EarlyStoppingResult(
                        EarlyStoppingResult.TerminationReason.IterationTerminationCondition,
                        terminationReason.toString(),
                        scoreVsEpoch,
                        bestModelEpoch,
                        bestModelScore,
                        epochCount,
                        bestModel);
                if(listener != null) listener.onCompletion(result);
                return result;
            }

            log.info("Completed training epoch {}",epochCount);


            if( (epochCount==0 && esConfig.getEvaluateEveryNEpochs()==1) || epochCount % esConfig.getEvaluateEveryNEpochs() == 0 ){
                //Calculate score at this epoch:
                ScoreCalculator sc = esConfig.getScoreCalculator();
                double score = (sc == null ? 0.0 : esConfig.getScoreCalculator().calculateScore(net));
                scoreVsEpoch.put(epochCount-1,score);

                if (sc != null && score < bestModelScore) {
                    //Save best model:
                    if (bestModelEpoch == -1) {
                        //First calculated/reported score
                        log.info("Score at epoch {}: {}", epochCount, score);
                    } else {
                        log.info("New best model: score = {}, epoch = {} (previous: score = {}, epoch = {})",
                                score, epochCount, bestModelScore, bestModelEpoch);
                    }
                    bestModelScore = score;
                    bestModelEpoch = epochCount;

                    try{
                        esConfig.getModelSaver().saveBestModel(net,score);
                    }catch(IOException e){
                        throw new RuntimeException("Error saving best model",e);
                    }
                }

                if(esConfig.isSaveLastModel()) {
                    //Save last model:
                    try {
                        esConfig.getModelSaver().saveLatestModel(net, score);
                    } catch (IOException e) {
                        throw new RuntimeException("Error saving most recent model", e);
                    }
                }

                if(listener != null) listener.onEpoch(epochCount,score,esConfig,net);

                //Check per-epoch termination conditions:
                boolean epochTerminate = false;
                EpochTerminationCondition termReason = null;
                for(EpochTerminationCondition c : esConfig.getEpochTerminationConditions()){
                    if(c.terminate(epochCount,score)){
                        epochTerminate = true;
                        termReason = c;
                        break;
                    }
                }
                if(epochTerminate){
                    log.info("Hit epoch termination condition at epoch {}. Details: {}", epochCount, termReason.toString());
                    MultiLayerNetwork bestModel;
                    try{
                        bestModel = esConfig.getModelSaver().getBestModel();
                    }catch(IOException e2){
                        throw new RuntimeException(e2);
                    }
                    EarlyStoppingResult result = new EarlyStoppingResult(
                            EarlyStoppingResult.TerminationReason.EpochTerminationCondition,
                            termReason.toString(),
                            scoreVsEpoch,
                            bestModelEpoch,
                            bestModelScore,
                            epochCount+1,
                            bestModel);
                    if(listener != null) listener.onCompletion(result);

                    return result;
                }

                epochCount++;
            }
        }
    }

    @Override
    public void setListener(EarlyStoppingListener listener) {
        this.listener = listener;
    }
}
