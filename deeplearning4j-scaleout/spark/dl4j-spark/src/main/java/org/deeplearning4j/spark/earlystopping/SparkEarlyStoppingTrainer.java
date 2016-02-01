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

package org.deeplearning4j.spark.earlystopping;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener;
import org.deeplearning4j.earlystopping.scorecalc.ScoreCalculator;
import org.deeplearning4j.earlystopping.termination.EpochTerminationCondition;
import org.deeplearning4j.earlystopping.termination.IterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.IEarlyStoppingTrainer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.reflect.Array;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Class for conducting early stopping training via Spark on a {@link MultiLayerNetwork}
 *
 * @author Alex Black
 */
public class SparkEarlyStoppingTrainer extends BaseSparkEarlyStoppingTrainer<MultiLayerNetwork> {

    private SparkDl4jMultiLayer sparkNet;

    public SparkEarlyStoppingTrainer(SparkContext sc, EarlyStoppingConfiguration<MultiLayerNetwork> esConfig, MultiLayerNetwork net,
                                     JavaRDD<DataSet> train, int examplesPerFit, int totalExamples, int numPartitions) {
        this(sc, esConfig, net, train, examplesPerFit, totalExamples, numPartitions, null);
    }

    public SparkEarlyStoppingTrainer(SparkContext sc, EarlyStoppingConfiguration<MultiLayerNetwork> esConfig, MultiLayerNetwork net,
                                     JavaRDD<DataSet> train, int examplesPerFit, int totalExamples, int numPartitions, EarlyStoppingListener<MultiLayerNetwork> listener) {
        super(sc, esConfig, net, train, null, examplesPerFit, totalExamples, numPartitions, listener);
        sparkNet = new SparkDl4jMultiLayer(sc, net);
    }


    @Override
    protected void fit(JavaRDD<DataSet> data) {
        sparkNet.fitDataSet(data, Integer.MAX_VALUE, 0, this.numPartitions);   //With examplesPerFit = Integer.MAX_VALUE -> fit all
    }

    @Override
    protected void fitMulti(JavaRDD<MultiDataSet> data) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    protected double getScore() {
        return sparkNet.getScore();
    }
}
