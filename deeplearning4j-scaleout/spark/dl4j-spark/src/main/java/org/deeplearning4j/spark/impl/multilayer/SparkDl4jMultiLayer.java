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
 *  *    WÏITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.spark.impl.multilayer;

import lombok.NonNull;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.rdd.RDD;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.impl.common.reduce.IntDoubleReduceFunction;
import org.deeplearning4j.spark.impl.multilayer.evaluation.EvaluateFlatMapFunction;
import org.deeplearning4j.spark.impl.multilayer.evaluation.EvaluationReduceFunction;
import org.deeplearning4j.spark.impl.multilayer.scoring.ScoreExamplesFunction;
import org.deeplearning4j.spark.impl.multilayer.scoring.ScoreExamplesWithKeyFunction;
import org.deeplearning4j.spark.impl.multilayer.scoring.ScoreFlatMapFunction;
import org.deeplearning4j.spark.util.MLLibUtil;
import org.deeplearning4j.spark.util.SparkUtils;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.heartbeat.Heartbeat;
import org.nd4j.linalg.heartbeat.reports.Environment;
import org.nd4j.linalg.heartbeat.reports.Event;
import org.nd4j.linalg.heartbeat.reports.Task;
import org.nd4j.linalg.heartbeat.utils.EnvironmentUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;

import java.io.IOException;
import java.io.OutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Master class for spark
 *
 * @author Adam Gibson, Alex Black
 */
public class SparkDl4jMultiLayer implements Serializable {
    private static final Logger log = LoggerFactory.getLogger(SparkDl4jMultiLayer.class);

    public static final int DEFAULT_EVAL_SCORE_BATCH_SIZE = 64;
    private transient JavaSparkContext sc;
    private TrainingMaster trainingMaster;
    private MultiLayerConfiguration conf;
    private MultiLayerNetwork network;
    private double lastScore;

    private List<IterationListener> listeners = new ArrayList<>();

    /**
     * Instantiate a multi layer spark instance
     * with the given context and network.
     * This is the prediction constructor
     *
     * @param sparkContext the spark context to use
     * @param network      the network to use
     */
    public SparkDl4jMultiLayer(SparkContext sparkContext, MultiLayerNetwork network, TrainingMaster<?,?> trainingMaster) {
        this(new JavaSparkContext(sparkContext), network, trainingMaster);
    }

    /**
     * Training constructor. Instantiate with a configuration
     *
     * @param sparkContext the spark context to use
     * @param conf         the configuration of the network
     */
    public SparkDl4jMultiLayer(SparkContext sparkContext, MultiLayerConfiguration conf, TrainingMaster<?,?> trainingMaster) {
        this(new JavaSparkContext(sparkContext), initNetwork(conf), trainingMaster);
    }

    /**
     * Training constructor. Instantiate with a configuration
     *
     * @param sc   the spark context to use
     * @param conf the configuration of the network
     */
    public SparkDl4jMultiLayer(JavaSparkContext sc, MultiLayerConfiguration conf, TrainingMaster<?,?> trainingMaster) {
        this(sc.sc(), conf, trainingMaster);
    }

    public SparkDl4jMultiLayer(JavaSparkContext javaSparkContext, MultiLayerNetwork network, TrainingMaster<?,?> trainingMaster) {
        sc = javaSparkContext;
        this.conf = network.getLayerWiseConfigurations().clone();
        this.network = network;
        if (!network.isInitCalled()) network.init();
        this.trainingMaster = trainingMaster;

        //Check if kryo configuration is correct:
        SparkUtils.checkKryoConfiguration(javaSparkContext, log);
    }

    private static MultiLayerNetwork initNetwork(MultiLayerConfiguration conf) {
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        return net;
    }

    public JavaSparkContext getSparkContext() {
        return sc;
    }

    /**
     * @return The MultiLayerNetwork underlying the SparkDl4jMultiLayer
     */
    public MultiLayerNetwork getNetwork() {
        return network;
    }

    /**
     * @return The TrainingMaster for this network
     */
    public TrainingMaster getTrainingMaster(){
        return trainingMaster;
    }

    /**
     * Set the network that underlies this SparkDl4jMultiLayer instacne
     *
     * @param network network to set
     */
    public void setNetwork(MultiLayerNetwork network) {
        this.network = network;
    }

    /**
     * Set whether training statistics should be collected for debugging purposes. Statistics collection is disabled by default
     *
     * @param collectTrainingStats If true: collect training statistics. If false: don't collect.
     */
    public void setCollectTrainingStats(boolean collectTrainingStats) {
        trainingMaster.setCollectTrainingStats(collectTrainingStats);
    }

    /**
     * Get the training statistics, after collection of stats has been enabled using {@link #setCollectTrainingStats(boolean)}
     *
     * @return Training statistics
     */
    public SparkTrainingStats getSparkTrainingStats() {
        return trainingMaster.getTrainingStats();
    }

    /**
     * Predict the given feature matrix
     *
     * @param features the given feature matrix
     * @return the predictions
     */
    public Matrix predict(Matrix features) {
        return MLLibUtil.toMatrix(network.output(MLLibUtil.toMatrix(features)));
    }


    /**
     * Predict the given vector
     *
     * @param point the vector to predict
     * @return the predicted vector
     */
    public Vector predict(Vector point) {
        return MLLibUtil.toVector(network.output(MLLibUtil.toVector(point)));
    }

    /**
     * Fit the DataSet RDD. Equivalent to fit(trainingData.toJavaRDD())
     *
     * @param trainingData the training data RDD to fitDataSet
     * @return the MultiLayerNetwork after training
     */
    public MultiLayerNetwork fit(RDD<DataSet> trainingData) {
        return fit(trainingData.toJavaRDD());
    }

    /**
     * Fit the DataSet RDD
     *
     * @param trainingData the training data RDD to fitDataSet
     * @return the MultiLayerNetwork after training
     */
    public MultiLayerNetwork fit(JavaRDD<DataSet> trainingData) {
        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner)Nd4j.getExecutioner()).flushQueue();

        trainingMaster.executeTraining(this, trainingData);
        return network;
    }

    /**
     * Fit the SparkDl4jMultiLayer network using a directory of serialized DataSet objects
     * The assumption here is that the directory contains a number of {@link DataSet} objects, each serialized using
     * {@link DataSet#save(OutputStream)}
     *
     * @param path Path to the directory containing the serialized DataSet objcets
     * @return The MultiLayerNetwork after training
     */
    public MultiLayerNetwork fit(String path) {
        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner)Nd4j.getExecutioner()).flushQueue();

        JavaRDD<String> paths;
        try{
            paths = SparkUtils.listPaths(sc, path);
        }catch(IOException e){
            throw new RuntimeException("Error listing paths in directory",e);
        }

        return fitPaths(paths);
    }

    /**
     * @deprecated Use {@link #fit(String)}
     */
    @Deprecated
    public MultiLayerNetwork fit(String path, int minPartitions) {
        return fit(path);
    }

    /**
     * Fit the network using a list of paths for serialized DataSet objects.
     *
     * @param paths    List of paths
     * @return trained network
     */
    public MultiLayerNetwork fitPaths(JavaRDD<String> paths){
        trainingMaster.executeTrainingPaths(this, paths);
        return network;
    }

    /**
     * Fit a MultiLayerNetwork using Spark MLLib LabeledPoint instances.
     * This will convert the labeled points to the internal DL4J data format and train the model on that
     *
     * @param rdd the rdd to fitDataSet
     * @return the multi layer network that was fitDataSet
     */
    public MultiLayerNetwork fitLabeledPoint(JavaRDD<LabeledPoint> rdd) {
        int nLayers = network.getLayerWiseConfigurations().getConfs().size();
        FeedForwardLayer ffl = (FeedForwardLayer) network.getLayerWiseConfigurations().getConf(nLayers - 1).getLayer();
        JavaRDD<DataSet> ds = MLLibUtil.fromLabeledPoint(sc, rdd, ffl.getNOut());
        return fit(ds);
    }

    /**
     * This method allows you to specify IterationListeners for this model.
     * <p>
     * PLEASE NOTE:
     * 1. These iteration listeners should be configured to use remote UiServer
     * 2. Remote UiServer should be accessible via network from Spark master node.
     *
     * @param listeners
     */
    public void setListeners(@NonNull Collection<IterationListener> listeners) {
        this.listeners.clear();
        this.listeners.addAll(listeners);
        if (trainingMaster != null) trainingMaster.setListeners(this.listeners);
    }

    protected void invokeListeners(MultiLayerNetwork network, int iteration) {
        for (IterationListener listener : listeners) {
            try {
                listener.iterationDone(network, iteration);
            } catch (Exception e) {
                log.error("Exception caught at IterationListener invocation" + e.getMessage());
                e.printStackTrace();
            }
        }
    }

    /**
     * Gets the last (average) minibatch score from calling fit. This is the average score across all executors for the
     * last minibatch executed in each worker
     */
    public double getScore() {
        return lastScore;
    }

    public void setScore(double lastScore) {
        this.lastScore = lastScore;
    }

    /**
     * Overload of {@link #calculateScore(JavaRDD, boolean)} for {@code RDD<DataSet>} instead of {@code JavaRDD<DataSet>}
     */
    public double calculateScore(RDD<DataSet> data, boolean average) {
        return calculateScore(data.toJavaRDD(), average);
    }

    /**
     * Calculate the score for all examples in the provided {@code JavaRDD<DataSet>}, either by summing
     * or averaging over the entire data set. To calculate a score for each example individually, use {@link #scoreExamples(JavaPairRDD, boolean)}
     * or one of the similar methods. Uses default minibatch size in each worker, {@link SparkDl4jMultiLayer#DEFAULT_EVAL_SCORE_BATCH_SIZE}
     *
     * @param data    Data to score
     * @param average Whether to sum the scores, or average them
     */
    public double calculateScore(JavaRDD<DataSet> data, boolean average) {
        return calculateScore(data, average, DEFAULT_EVAL_SCORE_BATCH_SIZE);
    }

    /**
     * Calculate the score for all examples in the provided {@code JavaRDD<DataSet>}, either by summing
     * or averaging over the entire data set. To calculate a score for each example individually, use {@link #scoreExamples(JavaPairRDD, boolean)}
     * or one of the similar methods
     *
     * @param data          Data to score
     * @param average       Whether to sum the scores, or average them
     * @param minibatchSize The number of examples to use in each minibatch when scoring. If more examples are in a partition than
     *                      this, multiple scoring operations will be done (to avoid using too much memory by doing the whole partition
     *                      in one go)
     */
    public double calculateScore(JavaRDD<DataSet> data, boolean average, int minibatchSize) {
        JavaRDD<Tuple2<Integer, Double>> rdd = data.mapPartitions(new ScoreFlatMapFunction(conf.toJson(), sc.broadcast(network.params(false)), minibatchSize));

        //Reduce to a single tuple, with example count + sum of scores
        Tuple2<Integer, Double> countAndSumScores = rdd.reduce(new IntDoubleReduceFunction());
        if (average) {
            return countAndSumScores._2() / countAndSumScores._1();
        } else {
            return countAndSumScores._2();
        }
    }

    /**
     * {@code RDD<DataSet>} overload of {@link #scoreExamples(JavaPairRDD, boolean)}
     */
    public JavaDoubleRDD scoreExamples(RDD<DataSet> data, boolean includeRegularizationTerms) {
        return scoreExamples(data.toJavaRDD(), includeRegularizationTerms);
    }

    /**
     * Score the examples individually, using the default batch size {@link #DEFAULT_EVAL_SCORE_BATCH_SIZE}. Unlike {@link #calculateScore(JavaRDD, boolean)},
     * this method returns a score for each example separately. If scoring is needed for specific examples use either
     * {@link #scoreExamples(JavaPairRDD, boolean)} or {@link #scoreExamples(JavaPairRDD, boolean, int)} which can have
     * a key for each example.
     *
     * @param data                       Data to score
     * @param includeRegularizationTerms If  true: include the l1/l2 regularization terms with the score (if any)
     * @return A JavaDoubleRDD containing the scores of each example
     * @see MultiLayerNetwork#scoreExamples(DataSet, boolean)
     */
    public JavaDoubleRDD scoreExamples(JavaRDD<DataSet> data, boolean includeRegularizationTerms) {
        return scoreExamples(data, includeRegularizationTerms, DEFAULT_EVAL_SCORE_BATCH_SIZE);
    }

    /**
     * {@code RDD<DataSet>} overload of {@link #scoreExamples(JavaRDD, boolean, int)}
     */
    public JavaDoubleRDD scoreExamples(RDD<DataSet> data, boolean includeRegularizationTerms, int batchSize) {
        return scoreExamples(data.toJavaRDD(), includeRegularizationTerms, batchSize);
    }

    /**
     * Score the examples individually, using a specified batch size. Unlike {@link #calculateScore(JavaRDD, boolean)},
     * this method returns a score for each example separately. If scoring is needed for specific examples use either
     * {@link #scoreExamples(JavaPairRDD, boolean)} or {@link #scoreExamples(JavaPairRDD, boolean, int)} which can have
     * a key for each example.
     *
     * @param data                       Data to score
     * @param includeRegularizationTerms If  true: include the l1/l2 regularization terms with the score (if any)
     * @param batchSize                  Batch size to use when doing scoring
     * @return A JavaDoubleRDD containing the scores of each example
     * @see MultiLayerNetwork#scoreExamples(DataSet, boolean)
     */
    public JavaDoubleRDD scoreExamples(JavaRDD<DataSet> data, boolean includeRegularizationTerms, int batchSize) {
        return data.mapPartitionsToDouble(new ScoreExamplesFunction(sc.broadcast(network.params()), sc.broadcast(conf.toJson()),
                includeRegularizationTerms, batchSize));
    }

    /**
     * Score the examples individually, using the default batch size {@link #DEFAULT_EVAL_SCORE_BATCH_SIZE}. Unlike {@link #calculateScore(JavaRDD, boolean)},
     * this method returns a score for each example separately<br>
     * Note: The provided JavaPairRDD has a key that is associated with each example and returned score.<br>
     * <b>Note:</b> The DataSet objects passed in must have exactly one example in them (otherwise: can't have a 1:1 association
     * between keys and data sets to score)
     *
     * @param data                       Data to score
     * @param includeRegularizationTerms If  true: include the l1/l2 regularization terms with the score (if any)
     * @param <K>                        Key type
     * @return A {@code JavaPairRDD<K,Double>} containing the scores of each example
     * @see MultiLayerNetwork#scoreExamples(DataSet, boolean)
     */
    public <K> JavaPairRDD<K, Double> scoreExamples(JavaPairRDD<K, DataSet> data, boolean includeRegularizationTerms) {
        return scoreExamples(data, includeRegularizationTerms, DEFAULT_EVAL_SCORE_BATCH_SIZE);
    }

    /**
     * Score the examples individually, using a specified batch size. Unlike {@link #calculateScore(JavaRDD, boolean)},
     * this method returns a score for each example separately<br>
     * Note: The provided JavaPairRDD has a key that is associated with each example and returned score.<br>
     * <b>Note:</b> The DataSet objects passed in must have exactly one example in them (otherwise: can't have a 1:1 association
     * between keys and data sets to score)
     *
     * @param data                       Data to score
     * @param includeRegularizationTerms If  true: include the l1/l2 regularization terms with the score (if any)
     * @param <K>                        Key type
     * @return A {@code JavaPairRDD<K,Double>} containing the scores of each example
     * @see MultiLayerNetwork#scoreExamples(DataSet, boolean)
     */
    public <K> JavaPairRDD<K, Double> scoreExamples(JavaPairRDD<K, DataSet> data, boolean includeRegularizationTerms, int batchSize) {
        return data.mapPartitionsToPair(new ScoreExamplesWithKeyFunction<K>(sc.broadcast(network.params()), sc.broadcast(conf.toJson()),
                includeRegularizationTerms, batchSize));
    }

    /**
     * {@code RDD<DataSet>} overload of {@link #evaluate(JavaRDD)}
     */
    public Evaluation evaluate(RDD<DataSet> data) {
        return evaluate(data.toJavaRDD());
    }

    /**
     * Evaluate the network (classification performance) in a distributed manner on the provided data
     *
     * @param data Data to evaluate on
     * @return Evaluation object; results of evaluation on all examples in the data set
     */
    public Evaluation evaluate(JavaRDD<DataSet> data) {
        return evaluate(data, null);
    }

    /**
     * {@code RDD<DataSet>} overload of {@link #evaluate(JavaRDD, List)}
     */
    public Evaluation evaluate(RDD<DataSet> data, List<String> labelsList) {
        return evaluate(data.toJavaRDD(), labelsList);
    }

    /**
     * Evaluate the network (classification performance) in a distributed manner, using default batch size and a provided
     * list of labels
     *
     * @param data       Data to evaluate on
     * @param labelsList List of labels used for evaluation
     * @return Evaluation object; results of evaluation on all examples in the data set
     */
    public Evaluation evaluate(JavaRDD<DataSet> data, List<String> labelsList) {
        return evaluate(data, labelsList, DEFAULT_EVAL_SCORE_BATCH_SIZE);
    }

    private void update(int mr, long mg) {
        Environment env = EnvironmentUtils.buildEnvironment();
        env.setNumCores(mr);
        env.setAvailableMemory(mg);
        Task task = ModelSerializer.taskByModel(network);
        Heartbeat.getInstance().reportEvent(Event.SPARK, env, task);
    }

    /**
     * Evaluate the network (classification performance) in a distributed manner, using specified batch size and a provided
     * list of labels
     *
     * @param data          Data to evaluate on
     * @param labelsList    List of labels used for evaluation
     * @param evalBatchSize Batch size to use when conducting evaluations
     * @return Evaluation object; results of evaluation on all examples in the data set
     */
    public Evaluation evaluate(JavaRDD<DataSet> data, List<String> labelsList, int evalBatchSize) {
        Broadcast<List<String>> listBroadcast = (labelsList == null ? null : sc.broadcast(labelsList));
        JavaRDD<Evaluation> evaluations = data.mapPartitions(new EvaluateFlatMapFunction(sc.broadcast(conf.toJson()),
                sc.broadcast(network.params()), evalBatchSize, listBroadcast));
        return evaluations.reduce(new EvaluationReduceFunction());
    }

}
