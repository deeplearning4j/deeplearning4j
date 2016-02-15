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

package org.deeplearning4j.spark.impl.computationgraph;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.DoubleFunction;
import org.apache.spark.broadcast.Broadcast;
import org.canova.api.records.reader.RecordReader;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.deeplearning4j.spark.canova.RecordReaderFunction;
import org.deeplearning4j.spark.impl.common.Adder;
import org.deeplearning4j.spark.impl.common.gradient.GradientAdder;
import org.deeplearning4j.spark.impl.common.misc.*;
import org.deeplearning4j.spark.impl.common.updater.UpdaterAggregatorCombinerCG;
import org.deeplearning4j.spark.impl.common.updater.UpdaterElementCombinerCG;
import org.deeplearning4j.spark.impl.computationgraph.dataset.DataSetToMultiDataSetFn;
import org.deeplearning4j.spark.impl.computationgraph.dataset.PairDataSetToMultiDataSetFn;
import org.deeplearning4j.spark.impl.computationgraph.gradientaccum.GradientAccumFlatMapCG;
import org.deeplearning4j.spark.impl.computationgraph.scoring.ScoreExamplesFunction;
import org.deeplearning4j.spark.impl.computationgraph.scoring.ScoreExamplesWithKeyFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple3;

import java.io.Serializable;
import java.util.List;

/**Main class for training ComputationGraph networks using Spark
 *
 * @author Alex Black
 */
public class SparkComputationGraph implements Serializable {

    public static final int DEFAULT_EVAL_SCORE_BATCH_SIZE = 50;
    private transient JavaSparkContext sc;
    private ComputationGraphConfiguration conf;
    private ComputationGraph network;
    private Broadcast<INDArray> params;
    private Broadcast<ComputationGraphUpdater> updater;
    private boolean averageEachIteration = false;
    public final static String AVERAGE_EACH_ITERATION = "org.deeplearning4j.spark.iteration.average";
    public final static String ACCUM_GRADIENT = "org.deeplearning4j.spark.iteration.accumgrad";
    public final static String DIVIDE_ACCUM_GRADIENT = "org.deeplearning4j.spark.iteration.dividegrad";

    private double lastScore;

    private static final Logger log = LoggerFactory.getLogger(SparkComputationGraph.class);

    /**
     * Instantiate a ComputationGraph instance with the given context and network.
     * @param sparkContext  the spark context to use
     * @param network the network to use
     */
    public SparkComputationGraph(SparkContext sparkContext, ComputationGraph network) {
        this(new JavaSparkContext(sparkContext),network);
    }

    public SparkComputationGraph(JavaSparkContext javaSparkContext, ComputationGraph network){
        sc = javaSparkContext;
        this.conf = network.getConfiguration().clone();
        this.network = network;
        this.network.init();
        this.updater = sc.broadcast(network.getUpdater());
        this.averageEachIteration = sc.getConf().getBoolean(AVERAGE_EACH_ITERATION, false);
    }


    public SparkComputationGraph(SparkContext sparkContext, ComputationGraphConfiguration conf) {
        this(new JavaSparkContext(sparkContext),conf);
    }

    public SparkComputationGraph(JavaSparkContext sparkContext, ComputationGraphConfiguration conf){
        sc = sparkContext;
        this.conf = conf.clone();
        this.network = new ComputationGraph(conf);
        this.network.init();
        this.averageEachIteration = sparkContext.sc().conf().getBoolean(AVERAGE_EACH_ITERATION, false);
        this.updater = sc.broadcast(network.getUpdater());
    }

    /**Train a ComputationGraph network based on data loaded from a text file + {@link RecordReader}.
     * This method splits the data into approximately {@code examplesPerFit} sized splits, and trains on each split.
     * one after the other. See {@link #fitDataSet(JavaRDD, int, int, int)} for further details.<br>
     * <b>NOTE: This method can only be used with ComputationGraph instances that have a single input and single output</b>
     * @param path the path to the text file
     * @param labelIndex the label index
     * @param recordReader the record reader to parse results
     * @param examplesPerFit Number of examples to fit on at each iteration
     * @param totalExamples total number of examples
     * @param numPartitions Number of partitions. Usually set to number of executors
     * @return {@link MultiLayerNetwork}
     */
    public ComputationGraph fit(String path,int labelIndex, RecordReader recordReader, int examplesPerFit, int totalExamples, int numPartitions ) {
        if(network.getNumInputArrays() != 1 || network.getNumOutputArrays() != 1){
            throw new UnsupportedOperationException("Cannot train ComputationGraph with multiple inputs/outputs from text file + record reader");
        }
        JavaRDD<MultiDataSet> points = loadFromTextFile(path, labelIndex, recordReader).map(new DataSetToMultiDataSetFn());
        return fitMultiDataSet(points, examplesPerFit, totalExamples, numPartitions);
    }

    private JavaRDD<DataSet> loadFromTextFile(String path, int labelIndex, RecordReader recordReader ){
        JavaRDD<String> lines = sc.textFile(path);
        int nOut = ((FeedForwardLayer)network.getOutputLayer(0)).getNOut();
        return lines.map(new RecordReaderFunction(recordReader, labelIndex, nOut));
    }

    public ComputationGraph getNetwork() {
        return network;
    }

    public void setNetwork(ComputationGraph network) {
        this.network = network;
    }

    /**Fit the data, splitting into smaller data subsets if necessary. This allows large {@code JavaRDD<DataSet>}s)
     * to be trained as a set of smaller steps instead of all together.<br>
     * Using this method, training progresses as follows:<br>
     * train on {@code examplesPerFit} examples -> average parameters -> train on {@code examplesPerFit} -> average
     * parameters etc until entire data set has been processed<br>
     * <em>Note</em>: The actual number of splits for the input data is based on rounding up.<br>
     * Suppose {@code examplesPerFit}=1000, with {@code rdd.count()}=1200. Then, we round up to 2000 examples, and the
     * network will then be fit in two steps (as 2000/1000=2), with 1200/2=600 examples at each step. These 600 examples
     * will then be distributed approximately equally (no guarantees) amongst each executor/core for training.
     *
     * @param rdd Data to train on
     * @param examplesPerFit Number of examples to learn on (between averaging) across all executors. For example, if set to
     *                       1000 and rdd.count() == 10k, then we do 10 sets of learning, each on 1000 examples.
     *                       To use all examples, set maxExamplesPerFit to Integer.MAX_VALUE
     * @param totalExamples total number of examples in the data RDD
     * @param numPartitions number of partitions to divide the data in to
     * @return Trained network
     */
    public ComputationGraph fitMultiDataSet(JavaRDD<MultiDataSet> rdd, int examplesPerFit, int totalExamples, int numPartitions ){
        int nSplits;
        if(examplesPerFit == Integer.MAX_VALUE || examplesPerFit >= totalExamples ) nSplits = 1;
        else {
            if(totalExamples%examplesPerFit==0){
                nSplits = (totalExamples / examplesPerFit);
            } else {
                nSplits = (totalExamples/ examplesPerFit) + 1;
            }
        }

        if(nSplits == 1){
            fitDataSet(rdd);
        } else {
            double[] splitWeights = new double[nSplits];
            for( int i=0; i<nSplits; i++ ) splitWeights[i] = 1.0 / nSplits;
            JavaRDD<MultiDataSet>[] subsets = rdd.randomSplit(splitWeights);
            for( int i=0; i<subsets.length; i++ ){
                log.info("Initiating distributed training of subset {} of {}", (i + 1), subsets.length);
                JavaRDD<MultiDataSet> next = subsets[i].repartition(numPartitions);
                fitDataSet(next);
            }
        }
        return network;
    }

    /**DataSet version of {@link #fitMultiDataSet(JavaRDD, int, int, int)}.
     * Handles conversion from DataSet to MultiDataSet internally.
     */
    public ComputationGraph fitDataSet(JavaRDD<DataSet> rdd, int examplesPerFit, int totalExamples, int numPartitions ){
        if(network.getNumInputArrays() != 1 || network.getNumOutputArrays() != 1){
            throw new UnsupportedOperationException("Cannot train ComputationGraph with multiple inputs/outputs from DataSet");
        }
        JavaRDD<MultiDataSet> mds = rdd.map(new DataSetToMultiDataSetFn());
        return fitMultiDataSet(mds, examplesPerFit, totalExamples, numPartitions);
    }

    /**
     * Fit the dataset rdd
     * @param rdd the rdd to fitDataSet
     * @return the ComputationGraph after parameter averaging
     */
    public ComputationGraph fitDataSet(JavaRDD<MultiDataSet> rdd) {
        int iterations = network.getLayer(0).conf().getNumIterations();
        log.info("Running distributed training: (averaging each iteration = " + averageEachIteration + "), (iterations = " +
                iterations + "), (num partions = " + rdd.partitions().size() + ")");
        if(!averageEachIteration) {
            //Do multiple iterations and average once at the end
            runIteration(rdd);
        } else {
            //Temporarily set numIterations = 1. Control numIterations externally here so we can average between iterations
            for(GraphVertex gv : conf.getVertices().values()) {
                if(gv instanceof LayerVertex){
                    ((LayerVertex)gv).getLayerConf().setNumIterations(1);   //TODO - do this more elegantly...
                }
            }

            //Run learning, and average at each iteration
            for(int i = 0; i < iterations; i++) {
                runIteration(rdd);
            }

            //Reset number of iterations in config
            if(iterations > 1 ){
                for(GraphVertex gv : conf.getVertices().values()) {
                    if(gv instanceof LayerVertex){
                        ((LayerVertex)gv).getLayerConf().setNumIterations(iterations);   //TODO - do this more elegantly...
                    }
                }
            }
        }

        return network;
    }


    protected void runIteration(JavaRDD<MultiDataSet> rdd) {

        log.info("Broadcasting initial parameters of length " + network.numParams(false));
        INDArray valToBroadcast = network.params(false);
        this.params = sc.broadcast(valToBroadcast);
        ComputationGraphUpdater updater = network.getUpdater();
        if(updater == null) {
            network.setUpdater(new ComputationGraphUpdater(network));
            log.warn("Unable to propagate null updater");
            updater = network.getUpdater();
        }
        this.updater = sc.broadcast(updater);

        int paramsLength = network.numParams(true);
        boolean accumGrad = sc.getConf().getBoolean(ACCUM_GRADIENT, false);

        if(accumGrad) {
            //Learning via averaging gradients
            JavaRDD<Tuple3<Gradient,ComputationGraphUpdater,Double>> results = rdd.mapPartitions(new GradientAccumFlatMapCG(conf.toJson(),
                    this.params, this.updater),true).cache();

            JavaRDD<Gradient> resultsGradient = results.map(new GradientFromTupleFunctionCG());
            log.info("Ran iterative reduce... averaging gradients now.");

            GradientAdder a = new GradientAdder(paramsLength);
            resultsGradient.foreach(a);
            INDArray accumulatedGradient = a.getAccumulator().value();
            boolean divideGrad = sc.getConf().getBoolean(DIVIDE_ACCUM_GRADIENT,false);
            if(divideGrad)
                accumulatedGradient.divi(results.partitions().size());
            log.info("Accumulated parameters");
            log.info("Summed gradients.");
            network.setParams(network.params(false).addi(accumulatedGradient));
            log.info("Set parameters");

            log.info("Processing updaters");
            JavaRDD<ComputationGraphUpdater> resultsUpdater = results.map(new UpdaterFromGradientTupleFunctionCG());
            JavaDoubleRDD scores = results.mapToDouble(new DoubleFunction<Tuple3<Gradient, ComputationGraphUpdater, Double>>() {
                @Override
                public double call(Tuple3<Gradient, ComputationGraphUpdater, Double> t3) throws Exception {
                    return t3._3();
                }
            });

            lastScore = scores.mean();

            ComputationGraphUpdater.Aggregator aggregator = resultsUpdater.aggregate(
                    null,
                    new UpdaterElementCombinerCG(),
                    new UpdaterAggregatorCombinerCG()
            );
            ComputationGraphUpdater combinedUpdater = aggregator.getUpdater();
            network.setUpdater(combinedUpdater);
            log.info("Set updater");
        }
        else {
            //Standard parameter averaging
            JavaRDD<Tuple3<INDArray,ComputationGraphUpdater,Double>> results = rdd.mapPartitions(new IterativeReduceFlatMapCG(conf.toJson(),
                    this.params, this.updater),true).cache();

            JavaRDD<INDArray> resultsParams = results.map(new INDArrayFromTupleFunctionCG());
            log.info("Running iterative reduce and averaging parameters");

            Adder a = new Adder(paramsLength,sc.accumulator(0));
            resultsParams.foreach(a);

            INDArray newParams = a.getAccumulator().value();
            newParams.divi(a.getCounter().value());

            network.setParams(newParams);
            log.info("Accumulated and set parameters");

            JavaRDD<ComputationGraphUpdater> resultsUpdater = results.map(new UpdaterFromTupleFunctionCG());
            JavaDoubleRDD scores = results.mapToDouble(new DoubleFunction<Tuple3<INDArray, ComputationGraphUpdater, Double>>() {
                @Override
                public double call(Tuple3<INDArray, ComputationGraphUpdater, Double> t3) throws Exception {
                    return t3._3();
                }
            });

            lastScore = scores.mean();

            ComputationGraphUpdater.Aggregator aggregator = resultsUpdater.aggregate(
                    null,
                    new UpdaterElementCombinerCG(),
                    new UpdaterAggregatorCombinerCG()
            );
            ComputationGraphUpdater combinedUpdater = aggregator.getUpdater();
            network.setUpdater(combinedUpdater);

            log.info("Processed and set updater");
        }
    }


    /** Gets the last (average) minibatch score from calling fit */
    public double getScore(){
        return lastScore;
    }

    public double calculateScoreDataSet(JavaRDD<DataSet> data, boolean average){
        long n = data.count();
        JavaRDD<Double> scores = data.mapPartitions(new ScoreFlatMapFunctionCGDataSet(conf.toJson(), sc.broadcast(network.params(false))));
        List<Double> scoresList = scores.collect();
        double sum = 0.0;
        for(Double d : scoresList) sum += d;
        if(average) return sum / n;
        return sum;
    }

    public double calculateScore(JavaRDD<MultiDataSet> data, boolean average){
        long n = data.count();
        JavaRDD<Double> scores = data.mapPartitions(new ScoreFlatMapFunctionCGMultiDataSet(conf.toJson(), sc.broadcast(network.params(false))));
        List<Double> scoresList = scores.collect();
        double sum = 0.0;
        for(Double d : scoresList) sum += d;
        if(average) return sum / n;
        return sum;
    }

    /** DataSet version of {@link #scoreExamples(JavaRDD, boolean)}
     */
    public JavaDoubleRDD scoreExamplesDataSet(JavaRDD<DataSet> data, boolean includeRegularizationTerms) {
        return scoreExamples(data.map(new DataSetToMultiDataSetFn()),includeRegularizationTerms);
    }

    /**DataSet version of {@link #scoreExamples(JavaPairRDD, boolean, int)}
     */
    public JavaDoubleRDD scoreExamplesDataSet(JavaRDD<DataSet> data, boolean includeRegularizationTerms, int batchSize) {
        return scoreExamples(data.map(new DataSetToMultiDataSetFn()), includeRegularizationTerms, batchSize);
    }

    /**DataSet version of {@link #scoreExamples(JavaPairRDD, boolean)}
     */
    public <K> JavaPairRDD<K,Double> scoreExamplesDataSet(JavaPairRDD<K,DataSet> data, boolean includeRegularizationTerms){
        return scoreExamples(data.mapToPair(new PairDataSetToMultiDataSetFn<K>()),includeRegularizationTerms,DEFAULT_EVAL_SCORE_BATCH_SIZE);
    }

    /**DataSet version of {@link #scoreExamples(JavaPairRDD, boolean,int)}
     */
    public <K> JavaPairRDD<K,Double> scoreExamplesDataSet(JavaPairRDD<K,DataSet> data, boolean includeRegularizationTerms, int batchSize){
        return scoreExamples(data.mapToPair(new PairDataSetToMultiDataSetFn<K>()),includeRegularizationTerms,batchSize);
    }

    /** Score the examples individually, using the default batch size {@link #DEFAULT_EVAL_SCORE_BATCH_SIZE}. Unlike {@link #calculateScore(JavaRDD, boolean)},
     * this method returns a score for each example separately. If scoring is needed for specific examples use either
     * {@link #scoreExamples(JavaPairRDD, boolean)} or {@link #scoreExamples(JavaPairRDD, boolean, int)} which can have
     * a key for each example.
     * @param data Data to score
     * @param includeRegularizationTerms If true: include the l1/l2 regularization terms with the score (if any)
     * @return A JavaDoubleRDD containing the scores of each example
     * @see ComputationGraph#scoreExamples(MultiDataSet, boolean)
     */
    public JavaDoubleRDD scoreExamples(JavaRDD<MultiDataSet> data, boolean includeRegularizationTerms) {
        return scoreExamples(data,includeRegularizationTerms,DEFAULT_EVAL_SCORE_BATCH_SIZE);
    }

    /** Score the examples individually, using a specified batch size. Unlike {@link #calculateScore(JavaRDD, boolean)},
     * this method returns a score for each example separately. If scoring is needed for specific examples use either
     * {@link #scoreExamples(JavaPairRDD, boolean)} or {@link #scoreExamples(JavaPairRDD, boolean, int)} which can have
     * a key for each example.
     * @param data Data to score
     * @param includeRegularizationTerms If true: include the l1/l2 regularization terms with the score (if any)
     * @param batchSize Batch size to use when doing scoring
     * @return A JavaDoubleRDD containing the scores of each example
     * @see ComputationGraph#scoreExamples(MultiDataSet, boolean)
     */
    public JavaDoubleRDD scoreExamples(JavaRDD<MultiDataSet> data, boolean includeRegularizationTerms, int batchSize) {
        return data.mapPartitionsToDouble(new ScoreExamplesFunction(sc.broadcast(network.params()), sc.broadcast(conf.toJson()),
                includeRegularizationTerms, batchSize));
    }

    /** Score the examples individually, using the default batch size {@link #DEFAULT_EVAL_SCORE_BATCH_SIZE}. Unlike {@link #calculateScore(JavaRDD, boolean)},
     * this method returns a score for each example separately<br>
     * Note: The provided JavaPairRDD has a key that is associated with each example and returned score.<br>
     * <b>Note:</b> The DataSet objects passed in must have exactly one example in them (otherwise: can't have a 1:1 association
     * between keys and data sets to score)
     * @param data Data to score
     * @param includeRegularizationTerms If true: include the l1/l2 regularization terms with the score (if any)
     * @param <K> Key type
     * @return A {@code JavaPairRDD<K,Double>} containing the scores of each example
     * @see MultiLayerNetwork#scoreExamples(DataSet, boolean)
     */
    public <K> JavaPairRDD<K,Double> scoreExamples(JavaPairRDD<K,MultiDataSet> data, boolean includeRegularizationTerms){
        return scoreExamples(data,includeRegularizationTerms,DEFAULT_EVAL_SCORE_BATCH_SIZE);
    }

    /** Score the examples individually, using a specified batch size. Unlike {@link #calculateScore(JavaRDD, boolean)},
     * this method returns a score for each example separately<br>
     * Note: The provided JavaPairRDD has a key that is associated with each example and returned score.<br>
     * <b>Note:</b> The DataSet objects passed in must have exactly one example in them (otherwise: can't have a 1:1 association
     * between keys and data sets to score)
     * @param data Data to score
     * @param includeRegularizationTerms If true: include the l1/l2 regularization terms with the score (if any)
     * @param <K> Key type
     * @return A {@code JavaPairRDD<K,Double>} containing the scores of each example
     * @see MultiLayerNetwork#scoreExamples(DataSet, boolean)
     */
    public <K> JavaPairRDD<K,Double> scoreExamples(JavaPairRDD<K,MultiDataSet> data, boolean includeRegularizationTerms, int batchSize ){
        return data.mapPartitionsToPair(new ScoreExamplesWithKeyFunction<K>(sc.broadcast(network.params()), sc.broadcast(conf.toJson()),
                includeRegularizationTerms, batchSize));
    }

}
