package org.deeplearning4j.spark.parameterserver.training;

import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.api.TrainingHook;
import org.deeplearning4j.spark.api.TrainingWorker;
import org.deeplearning4j.spark.api.WorkerConfiguration;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.api.worker.NetBroadcastTuple;
import org.deeplearning4j.spark.parameterserver.conf.SharedTrainingConfiguration;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;

/**
 * @author raver119@gmail.com
 */
public class SharedTrainingWorker implements TrainingWorker<SharedTrainingResult> {

    private final Broadcast<NetBroadcastTuple> broadcastModel;
    private final Broadcast<SharedTrainingConfiguration> broadcastConfiguration;

    public SharedTrainingWorker(Broadcast<NetBroadcastTuple> broadcastModel, Broadcast<SharedTrainingConfiguration> broadcastConfiguration) {
        // our initial model is stored here.
        this.broadcastModel = broadcastModel;
        this.broadcastConfiguration = broadcastConfiguration;
    }

    @Override
    public void removeHook(TrainingHook trainingHook) {

    }

    @Override
    public void addHook(TrainingHook trainingHook) {

    }

    @Override
    public MultiLayerNetwork getInitialModel() {
        // This method will be called ONLY once, in master thread
        return null;
    }

    @Override
    public ComputationGraph getInitialModelGraph() {
        // This method will be called ONLY once, in master thread
        return null;
    }

    @Override
    public SharedTrainingResult processMinibatch(DataSet dataSet, MultiLayerNetwork network, boolean isLast) {
        /*
            We're not really going to use this method for training.
            Partitions will be mapped to ParallelWorker threads dynamically, wrt thread/device affinity.
            So plan is simple: we're going to use individual partitions to feed main worker
         */
        throw new UnsupportedOperationException();
    }

    @Override
    public SharedTrainingResult processMinibatch(DataSet dataSet, ComputationGraph graph, boolean isLast) {
        throw new UnsupportedOperationException();
    }

    @Override
    public SharedTrainingResult processMinibatch(MultiDataSet dataSet, ComputationGraph graph, boolean isLast) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Pair<SharedTrainingResult, SparkTrainingStats> processMinibatchWithStats(DataSet dataSet, MultiLayerNetwork network, boolean isLast) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Pair<SharedTrainingResult, SparkTrainingStats> processMinibatchWithStats(DataSet dataSet, ComputationGraph graph, boolean isLast) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Pair<SharedTrainingResult, SparkTrainingStats> processMinibatchWithStats(MultiDataSet dataSet, ComputationGraph graph, boolean isLast) {
        throw new UnsupportedOperationException();
    }

    @Override
    public SharedTrainingResult getFinalResult(MultiLayerNetwork network) {
        return null;
    }

    @Override
    public SharedTrainingResult getFinalResult(ComputationGraph graph) {
        return null;
    }

    @Override
    public SharedTrainingResult getFinalResultNoData() {
        return null;
    }

    @Override
    public Pair<SharedTrainingResult, SparkTrainingStats> getFinalResultNoDataWithStats() {
        return null;
    }

    @Override
    public Pair<SharedTrainingResult, SparkTrainingStats> getFinalResultWithStats(MultiLayerNetwork network) {
        return null;
    }

    @Override
    public Pair<SharedTrainingResult, SparkTrainingStats> getFinalResultWithStats(ComputationGraph graph) {
        return null;
    }

    @Override
    public WorkerConfiguration getDataConfiguration() {
        return null;
    }
}
