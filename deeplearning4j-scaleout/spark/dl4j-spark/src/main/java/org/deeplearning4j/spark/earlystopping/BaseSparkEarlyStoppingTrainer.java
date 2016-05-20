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

package org.deeplearning4j.spark.earlystopping;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
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
import java.lang.reflect.Array;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Base/abstract class for conducting early stopping training via Spark, on a {@link org.deeplearning4j.nn.multilayer.MultiLayerNetwork}
 * or a {@link org.deeplearning4j.nn.graph.ComputationGraph}
 * @author Alex Black
 */
public abstract class BaseSparkEarlyStoppingTrainer<T extends Model> implements IEarlyStoppingTrainer<T> {

    private static Logger log = LoggerFactory.getLogger(BaseSparkEarlyStoppingTrainer.class);

    private SparkContext sc;
    private final EarlyStoppingConfiguration<T> esConfig;
    private T net;
    private final JavaRDD<DataSet> train;
    private final JavaRDD<MultiDataSet> trainMulti;
    protected final int examplesPerFit;
    protected final int totalExamples;
    protected final int numPartitions;
    private EarlyStoppingListener<T> listener;

    private double bestModelScore = Double.MAX_VALUE;
    private int bestModelEpoch = -1;

    protected BaseSparkEarlyStoppingTrainer(SparkContext sc, EarlyStoppingConfiguration<T> esConfig, T net, JavaRDD<DataSet> train,
                                            JavaRDD<MultiDataSet> trainMulti, int examplesPerFit, int totalExamples, int numPartitions,
                                            EarlyStoppingListener<T> listener) {
        if((esConfig.getEpochTerminationConditions() == null || esConfig.getEpochTerminationConditions().size() == 0)
                && (esConfig.getIterationTerminationConditions() == null || esConfig.getIterationTerminationConditions().size() == 0)){
            throw new IllegalArgumentException("Cannot conduct early stopping without a termination condition (both Iteration "
                + "and Epoch termination conditions are null/empty)");
        }

        // repartition if size is different
        if(numPartitions != 0 && numPartitions != train.partitions().size()){
            log.info("Repartitioning training set to {}", numPartitions);
            this.train = train.repartition(numPartitions);
        } else {
            this.train = train;
        }

        this.sc = sc;
        this.esConfig = esConfig;
        this.net = net;
        this.trainMulti = trainMulti;
        this.examplesPerFit = examplesPerFit;
        this.totalExamples = totalExamples;
        this.numPartitions = numPartitions;
        this.listener = listener;
    }

    protected abstract void fit(JavaRDD<DataSet> data );

    protected abstract void fitMulti(JavaRDD<MultiDataSet> data);

    protected abstract double getScore();

    @Override
    public EarlyStoppingResult<T> fit() {
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

        if(listener != null)
            listener.onStart(esConfig,net);

        Map<Integer,Double> scoreVsEpoch = new LinkedHashMap<>();

        if(train != null) train.cache();
        else trainMulti.cache();

        int epochCount = 0;
        while (true) {  //Iterate (do epochs) until termination condition hit
            double lastScore;
            boolean terminate = false;
            IterationTerminationCondition terminationReason = null;
            int iterCount = 0;

            //Create random split of RDD:
            int nSplits;
            if(totalExamples%examplesPerFit==0){
                nSplits = (totalExamples / examplesPerFit);
            } else {
                nSplits = (totalExamples / examplesPerFit) + 1;
            }

            JavaRDD<DataSet>[] subsets = null;
            JavaRDD<MultiDataSet>[] subsetsMulti = null;
            if(train != null){
                if(nSplits == 1){
                    subsets = (JavaRDD<DataSet>[])Array.newInstance(JavaRDD.class,1);   //new Object[]{train};
                    subsets[0] = train;
                } else {
                    double[] splitWeights = new double[nSplits];
                    for( int i=0; i<nSplits; i++ ) splitWeights[i] = 1.0 / nSplits;
                    subsets = train.randomSplit(splitWeights);
                }
            } else {
                if(nSplits == 1){
                    subsetsMulti = (JavaRDD<MultiDataSet>[])Array.newInstance(JavaRDD.class,1);   //new Object[]{train};
                    subsetsMulti[0] = trainMulti;
                } else {
                    double[] splitWeights = new double[nSplits];
                    for( int i=0; i<nSplits; i++ ) splitWeights[i] = 1.0 / nSplits;
                    subsetsMulti = trainMulti.randomSplit(splitWeights);
                }
            }

            int nSubsets = (subsets != null ? subsets.length : subsetsMulti.length);

            for(int i = 0; i<nSubsets; i++) {
                log.info("Initiating distributed training of subset {} of {}",(i+1),nSubsets);
                try{
                    if(subsets != null) fit(subsets[i]);
                    else fitMulti(subsetsMulti[i]);
                }catch(Exception e){
                    log.warn("Early stopping training terminated due to exception at epoch {}, iteration {}",
                            epochCount,iterCount,e);
                    //Load best model to return
                    T bestModel;
                    try{
                        bestModel = esConfig.getModelSaver().getBestModel();
                    }catch(IOException e2){
                        throw new RuntimeException(e2);
                    }
                    return new EarlyStoppingResult<T>(
                            EarlyStoppingResult.TerminationReason.Error,
                            e.toString(),
                            scoreVsEpoch,
                            bestModelEpoch,
                            bestModelScore,
                            epochCount,
                            bestModel);
                }

                //Check per-iteration termination conditions
                lastScore = getScore();
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

                T bestModel;
                try{
                    bestModel = esConfig.getModelSaver().getBestModel();
                }catch(IOException e2){
                    throw new RuntimeException(e2);
                }
                EarlyStoppingResult<T> result = new EarlyStoppingResult<>(
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
                    T bestModel;
                    try{
                        bestModel = esConfig.getModelSaver().getBestModel();
                    }catch(IOException e2){
                        throw new RuntimeException(e2);
                    }
                    EarlyStoppingResult<T> result = new EarlyStoppingResult<>(
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
    public void setListener(EarlyStoppingListener<T> listener) {
        this.listener = listener;
    }
}
