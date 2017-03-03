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

package org.deeplearning4j.spark.earlystopping;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.impl.graph.dataset.DataSetToMultiDataSetFn;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;

/**
 * Class for conducting early stopping training via Spark on a ComputationGraph
 *
 * @author Alex Black
 */
public class SparkEarlyStoppingGraphTrainer extends BaseSparkEarlyStoppingTrainer<ComputationGraph> {

    private SparkComputationGraph sparkNet;

    public SparkEarlyStoppingGraphTrainer(SparkContext sc, TrainingMaster trainingMaster,
                    EarlyStoppingConfiguration<ComputationGraph> esConfig, ComputationGraph net,
                    JavaRDD<MultiDataSet> train, int examplesPerFit, int totalExamples) {
        this(new JavaSparkContext(sc), trainingMaster, esConfig, net, train, null);
    }

    public SparkEarlyStoppingGraphTrainer(JavaSparkContext sc, TrainingMaster trainingMaster,
                    EarlyStoppingConfiguration<ComputationGraph> esConfig, ComputationGraph net,
                    JavaRDD<MultiDataSet> train, int examplesPerFit, int totalExamples) {
        this(sc, trainingMaster, esConfig, net, train, null);
    }

    public SparkEarlyStoppingGraphTrainer(SparkContext sc, TrainingMaster trainingMaster,
                    EarlyStoppingConfiguration<ComputationGraph> esConfig, ComputationGraph net,
                    JavaRDD<MultiDataSet> train) {
        this(new JavaSparkContext(sc), trainingMaster, esConfig, net, train, null);
    }

    public SparkEarlyStoppingGraphTrainer(JavaSparkContext sc, TrainingMaster trainingMaster,
                    EarlyStoppingConfiguration<ComputationGraph> esConfig, ComputationGraph net,
                    JavaRDD<MultiDataSet> train) {
        this(sc, trainingMaster, esConfig, net, train, null);
    }

    public SparkEarlyStoppingGraphTrainer(JavaSparkContext sc, TrainingMaster trainingMaster,
                    EarlyStoppingConfiguration<ComputationGraph> esConfig, ComputationGraph net,
                    JavaRDD<MultiDataSet> train, EarlyStoppingListener<ComputationGraph> listener) {
        super(sc, esConfig, net, null, train, listener);
        this.sparkNet = new SparkComputationGraph(sc, net, trainingMaster);
    }


    @Override
    protected void fit(JavaRDD<DataSet> data) {
        fitMulti(data.map(new DataSetToMultiDataSetFn()));
    }

    @Override
    protected void fitMulti(JavaRDD<MultiDataSet> data) {
        sparkNet.fitMultiDataSet(data);
    }

    @Override
    protected double getScore() {
        return sparkNet.getScore();
    }
}
