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

package org.deeplearning4j.nn.earlystopping.runner;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.nn.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.nn.earlystopping.termination.EpochTerminationCondition;
import org.deeplearning4j.nn.earlystopping.termination.IterationTerminationCondition;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Class for conducting early stopping training locally (single machine)
 */
public class EarlyStoppingTrainer implements IEarlyStoppingTrainer {

    private static Logger log = LoggerFactory.getLogger(EarlyStoppingTrainer.class);

    private final EarlyStoppingConfiguration esConfig;
    private final MultiLayerConfiguration configuration;
    private MultiLayerNetwork net;
    private final DataSetIterator train;
    private final DataSetIterator test;

    private double bestModelScore = Double.MAX_VALUE;
    private int bestModelEpoch = -1;


    public EarlyStoppingTrainer(EarlyStoppingConfiguration earlyStoppingConfiguration, MultiLayerConfiguration configuration,
                                DataSetIterator train, DataSetIterator test) {
        this.esConfig = earlyStoppingConfiguration;
        this.configuration = configuration;
        this.train = train;
        this.test = test;
    }

    public EarlyStoppingTrainer(EarlyStoppingConfiguration earlyStoppingConfiguration, MultiLayerNetwork net,
                                DataSetIterator train, DataSetIterator test) {
        this.esConfig = earlyStoppingConfiguration;
        this.net = net;
        this.configuration = net.getLayerWiseConfigurations();
        this.train = train;
        this.test = test;
    }

    @Override
    public EarlyStoppingResult fit() {
        log.info("Starting early stopping training");

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

        if( net == null) net = new MultiLayerNetwork(configuration);
        net.init();
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
                    return new EarlyStoppingResult(
                            EarlyStoppingResult.TerminationReason.Error,
                            e.toString(),
                            scoreVsEpoch,
                            bestModelEpoch,
                            bestModelScore,
                            net);
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
                log.info("Hit per iteration epoch condition at epoch {}, iteration {}. Reason: {}",
                        epochCount, iterCount, terminationReason);

                return new EarlyStoppingResult(
                        EarlyStoppingResult.TerminationReason.IterationTerminationCondition,
                        terminationReason.toString(),
                        scoreVsEpoch,
                        bestModelEpoch,
                        bestModelScore,
                        net);
            }

            log.info("Completed training epoch {}",epochCount);
            epochCount++;


            if( epochCount % esConfig.getEvaluateEveryNEpochs() == 0 ){
                //Check per epoch termination conditions:
                //First: calculate various values required for termination condition
                double testSetScore = 0.0;  //TODO
                scoreVsEpoch.put(epochCount-1,testSetScore);


                if (testSetScore < bestModelScore) {
                    //Save best model:
                    if (bestModelEpoch == -1) {
                        log.info("Score at epoch 0: {}", testSetScore);
                    } else {
                        log.info("New best model: score = {}, epoch = {} (previous: score = {}, epoch = {}",
                                testSetScore, epochCount, bestModelScore, bestModelEpoch);
                    }
                    bestModelScore = testSetScore;
                    bestModelEpoch = epochCount;

                    try{
                        esConfig.getModelSaver().saveBestModel(net,testSetScore);
                    }catch(IOException e){
                        throw new RuntimeException("Error saving best model",e);
                    }
                }

                if(esConfig.isSaveLastModel()) {
                    //Save last model:
                    try {
                        esConfig.getModelSaver().saveLatestModel(net, testSetScore);
                    } catch (IOException e) {
                        throw new RuntimeException("Error saving most frequent model", e);
                    }
                }
            }
        }


    }
}
