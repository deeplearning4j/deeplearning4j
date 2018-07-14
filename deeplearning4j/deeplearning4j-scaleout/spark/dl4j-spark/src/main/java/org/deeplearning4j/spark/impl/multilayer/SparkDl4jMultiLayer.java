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

package org.deeplearning4j.spark.impl.multilayer;

import lombok.extern.slf4j.Slf4j;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.rdd.RDD;
import org.deeplearning4j.api.loader.DataSetLoader;
import org.deeplearning4j.api.loader.impl.SerializedDataSetLoader;
import org.deeplearning4j.eval.*;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.impl.SparkListenable;
import org.deeplearning4j.spark.impl.common.reduce.IntDoubleReduceFunction;
import org.deeplearning4j.spark.impl.multilayer.evaluation.IEvaluateAggregateFunction;
import org.deeplearning4j.spark.impl.multilayer.evaluation.IEvaluateFlatMapFunction;
import org.deeplearning4j.spark.impl.multilayer.evaluation.IEvaluationReduceFunction;
import org.deeplearning4j.spark.impl.multilayer.scoring.*;
import org.deeplearning4j.spark.util.MLLibUtil;
import org.deeplearning4j.spark.util.SparkUtils;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.heartbeat.Heartbeat;
import org.nd4j.linalg.heartbeat.reports.Environment;
import org.nd4j.linalg.heartbeat.reports.Event;
import org.nd4j.linalg.heartbeat.reports.Task;
import org.nd4j.linalg.heartbeat.utils.EnvironmentUtils;
import scala.Tuple2;

import java.io.IOException;
import java.io.OutputStream;
import java.util.List;

/**
 * Master class for spark
 *
 * @author Adam Gibson, Alex Black
 */
@Slf4j
public class SparkDl4jMultiLayer extends SparkListenable {
    public static final int DEFAULT_EVAL_SCORE_BATCH_SIZE = 64;
    public static final int DEFAULT_ROC_THRESHOLD_STEPS = 32;
    private transient JavaSparkContext sc;
    private MultiLayerConfiguration conf;
    private MultiLayerNetwork network;
    private double lastScore;

    /**
     * Instantiate a multi layer spark instance
     * with the given context and network.
     * This is the prediction constructor
     *
     * @param sparkContext the spark context to use
     * @param network      the network to use
     */
    public SparkDl4jMultiLayer(SparkContext sparkContext, MultiLayerNetwork network,
                    TrainingMaster<?, ?> trainingMaster) {
        this(new JavaSparkContext(sparkContext), network, trainingMaster);
    }

    /**
     * Training constructor. Instantiate with a configuration
     *
     * @param sparkContext the spark context to use
     * @param conf         the configuration of the network
     */
    public SparkDl4jMultiLayer(SparkContext sparkContext, MultiLayerConfiguration conf,
                    TrainingMaster<?, ?> trainingMaster) {
        this(new JavaSparkContext(sparkContext), initNetwork(conf), trainingMaster);
    }

    /**
     * Training constructor. Instantiate with a configuration
     *
     * @param sc   the spark context to use
     * @param conf the configuration of the network
     */
    public SparkDl4jMultiLayer(JavaSparkContext sc, MultiLayerConfiguration conf, TrainingMaster<?, ?> trainingMaster) {
        this(sc.sc(), conf, trainingMaster);
    }

    public SparkDl4jMultiLayer(JavaSparkContext javaSparkContext, MultiLayerNetwork network,
                    TrainingMaster<?, ?> trainingMaster) {
        sc = javaSparkContext;
        this.conf = network.getLayerWiseConfigurations().clone();
        this.network = network;
        if (!network.isInitCalled())
            network.init();
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
    public TrainingMaster getTrainingMaster() {
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
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        trainingMaster.executeTraining(this, trainingData);
        network.incrementEpochCount();
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
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        JavaRDD<String> paths;
        try {
            paths = SparkUtils.listPaths(sc, path);
        } catch (IOException e) {
            throw new RuntimeException("Error listing paths in directory", e);
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
    public MultiLayerNetwork fitPaths(JavaRDD<String> paths) {
        return fitPaths(paths, new SerializedDataSetLoader());
    }

    public MultiLayerNetwork fitPaths(JavaRDD<String> paths, DataSetLoader loader) {
        trainingMaster.executeTrainingPaths(this, null, paths, loader, null);
        network.incrementEpochCount();
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
     * Fits a MultiLayerNetwork using Spark MLLib LabeledPoint instances
     * This will convert labeled points that have continuous labels used for regression to the internal
     * DL4J data format and train the model on that
     * @param rdd the javaRDD containing the labeled points
     * @return a MultiLayerNetwork
     */
    public MultiLayerNetwork fitContinuousLabeledPoint(JavaRDD<LabeledPoint> rdd) {
        return fit(MLLibUtil.fromContinuousLabeledPoint(sc, rdd));
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
        JavaRDD<Tuple2<Integer, Double>> rdd = data.mapPartitions(
                        new ScoreFlatMapFunction(conf.toJson(), sc.broadcast(network.params(false)), minibatchSize));

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
     * {@code RDD<DataSet>}
     * overload of {@link #scoreExamples(JavaRDD, boolean, int)}
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
        return data.mapPartitionsToDouble(new ScoreExamplesFunction(sc.broadcast(network.params()),
                        sc.broadcast(conf.toJson()), includeRegularizationTerms, batchSize));
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
    public <K> JavaPairRDD<K, Double> scoreExamples(JavaPairRDD<K, DataSet> data, boolean includeRegularizationTerms,
                    int batchSize) {
        return data.mapPartitionsToPair(new ScoreExamplesWithKeyFunction<K>(sc.broadcast(network.params()),
                        sc.broadcast(conf.toJson()), includeRegularizationTerms, batchSize));
    }

    /**
     * Feed-forward the specified data, with the given keys. i.e., get the network output/predictions for the specified data
     *
     * @param featuresData Features data to feed through the network
     * @param batchSize    Batch size to use when doing feed forward operations
     * @param <K>          Type of data for key - may be anything
     * @return Network output given the input, by key
     */
    public <K> JavaPairRDD<K, INDArray> feedForwardWithKey(JavaPairRDD<K, INDArray> featuresData, int batchSize) {
        return feedForwardWithMaskAndKey(featuresData.mapToPair(new SingleToPairFunction<K>()), batchSize);
    }

    /**
     * Feed-forward the specified data (and optionally mask array), with the given keys. i.e., get the network
     * output/predictions for the specified data
     *
     * @param featuresDataAndMask Features data to feed through the network. The Tuple2 is of the network input (features),
     *                            and optionally the feature mask arrays
     * @param batchSize           Batch size to use when doing feed forward operations
     * @param <K>                 Type of data for key - may be anything
     * @return Network output given the input (and optionally mask), by key
     */
    public <K> JavaPairRDD<K, INDArray> feedForwardWithMaskAndKey(JavaPairRDD<K, Tuple2<INDArray,INDArray>> featuresDataAndMask, int batchSize) {
        return featuresDataAndMask
                .mapPartitionsToPair(new FeedForwardWithKeyFunction<K>(sc.broadcast(network.params()),
                        sc.broadcast(conf.toJson()), batchSize));
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
     * Evaluate the network (regression performance) in a distributed manner on the provided data
     *
     * @param data Data to evaluate
     * @return     {@link RegressionEvaluation} instance with regression performance
     */
    public RegressionEvaluation evaluateRegression(JavaRDD<DataSet> data) {
        return evaluateRegression(data, DEFAULT_EVAL_SCORE_BATCH_SIZE);
    }

    /**
     * Evaluate the network (regression performance) in a distributed manner on the provided data
     *
     * @param data Data to evaluate
     * @param minibatchSize Minibatch size to use when doing performing evaluation
     * @return     {@link RegressionEvaluation} instance with regression performance
     */
    public RegressionEvaluation evaluateRegression(JavaRDD<DataSet> data, int minibatchSize) {
        long nOut = ((FeedForwardLayer) network.getOutputLayer().conf().getLayer()).getNOut();
        return doEvaluation(data, new RegressionEvaluation(nOut), minibatchSize);
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

    /**
     * Perform ROC analysis/evaluation on the given DataSet in a distributed manner, using the default number of
     * threshold steps ({@link #DEFAULT_ROC_THRESHOLD_STEPS}) and the default minibatch size ({@link #DEFAULT_EVAL_SCORE_BATCH_SIZE})
     *
     * @param data                    Test set data (to evaluate on)
     * @return ROC for the entire data set
     */
    public ROC evaluateROC(JavaRDD<DataSet> data) {
        return evaluateROC(data, DEFAULT_ROC_THRESHOLD_STEPS, DEFAULT_EVAL_SCORE_BATCH_SIZE);
    }

    /**
     * Perform ROC analysis/evaluation on the given DataSet in a distributed manner
     *
     * @param data                    Test set data (to evaluate on)
     * @param thresholdSteps          Number of threshold steps for ROC - see {@link ROC}
     * @param evaluationMinibatchSize Minibatch size to use when performing ROC evaluation
     * @return ROC for the entire data set
     */
    public ROC evaluateROC(JavaRDD<DataSet> data, int thresholdSteps, int evaluationMinibatchSize) {
        return doEvaluation(data, new ROC(thresholdSteps), evaluationMinibatchSize);
    }

    /**
     * Perform ROC analysis/evaluation (for the multi-class case, using {@link ROCMultiClass} on the given DataSet in a distributed manner
     *
     * @param data                    Test set data (to evaluate on)
     * @return ROC for the entire data set
     */
    public ROCMultiClass evaluateROCMultiClass(JavaRDD<DataSet> data) {
        return evaluateROCMultiClass(data, DEFAULT_ROC_THRESHOLD_STEPS, DEFAULT_EVAL_SCORE_BATCH_SIZE);
    }

    /**
     * Perform ROC analysis/evaluation (for the multi-class case, using {@link ROCMultiClass} on the given DataSet in a distributed manner
     *
     * @param data                    Test set data (to evaluate on)
     * @param thresholdSteps          Number of threshold steps for ROC - see {@link ROC}
     * @param evaluationMinibatchSize Minibatch size to use when performing ROC evaluation
     * @return ROCMultiClass for the entire data set
     */
    public ROCMultiClass evaluateROCMultiClass(JavaRDD<DataSet> data, int thresholdSteps, int evaluationMinibatchSize) {
        return doEvaluation(data, new ROCMultiClass(thresholdSteps), evaluationMinibatchSize);
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
        Evaluation e = new Evaluation();
        e = doEvaluation(data, e, evalBatchSize);
        if (labelsList != null) {
            e.setLabelsList(labelsList);
        }
        return e;
    }

    /**
     * Perform distributed evaluation of any type of {@link IEvaluation}. For example, {@link Evaluation}, {@link RegressionEvaluation},
     * {@link ROC}, {@link ROCMultiClass} etc.
     *
     * @param data            Data to evaluate on
     * @param emptyEvaluation Empty evaluation instance. This is the starting point (serialized/duplicated, then merged)
     * @param evalBatchSize   Evaluation batch size
     * @param <T>             Type of evaluation instance to return
     * @return                IEvaluation instance
     */
    @SuppressWarnings("unchecked")
    public <T extends IEvaluation> T doEvaluation(JavaRDD<DataSet> data, T emptyEvaluation, int evalBatchSize) {
        return doEvaluation(data, evalBatchSize, emptyEvaluation)[0];
    }

    /**
     * Perform distributed evaluation of any type of {@link IEvaluation} - or multiple IEvaluation instances.
     * Distributed equivalent of {@link MultiLayerNetwork#doEvaluation(DataSetIterator, IEvaluation[])}
     *
     * @param data             Data to evaluate on
     * @param emptyEvaluations Empty evaluation instances. Starting point (serialized/duplicated, then merged)
     * @param evalBatchSize    Evaluation batch size
     * @param <T>              Type of evaluation instance to return
     * @return IEvaluation instances
     */
    @SuppressWarnings("unchecked")
    public <T extends IEvaluation> T[] doEvaluation(JavaRDD<DataSet> data, int evalBatchSize, T... emptyEvaluations) {
        IEvaluateFlatMapFunction<T> evalFn = new IEvaluateFlatMapFunction<>(false, sc.broadcast(conf.toJson()),
                        sc.broadcast(network.params()), evalBatchSize, emptyEvaluations);
        JavaRDD<T[]> evaluations = data.mapPartitions(evalFn);
        return evaluations.treeAggregate(null, new IEvaluateAggregateFunction<T>(), new IEvaluationReduceFunction<T>());
    }
}
