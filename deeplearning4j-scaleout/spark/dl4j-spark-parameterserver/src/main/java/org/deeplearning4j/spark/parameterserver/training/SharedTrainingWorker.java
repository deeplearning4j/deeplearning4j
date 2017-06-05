package org.deeplearning4j.spark.parameterserver.training;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.api.TrainingHook;
import org.deeplearning4j.spark.api.TrainingWorker;
import org.deeplearning4j.spark.api.WorkerConfiguration;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;

/**
 * @author raver119@gmail.com
 */
public class SharedTrainingWorker implements TrainingWorker<SharedTrainingResult> {
    @Override
    public void removeHook(TrainingHook trainingHook) {

    }

    @Override
    public void addHook(TrainingHook trainingHook) {

    }

    @Override
    public MultiLayerNetwork getInitialModel() {
        return null;
    }

    @Override
    public ComputationGraph getInitialModelGraph() {
        return null;
    }

    @Override
    public SharedTrainingResult processMinibatch(DataSet dataSet, MultiLayerNetwork network, boolean isLast) {
        return null;
    }

    @Override
    public SharedTrainingResult processMinibatch(DataSet dataSet, ComputationGraph graph, boolean isLast) {
        return null;
    }

    @Override
    public SharedTrainingResult processMinibatch(MultiDataSet dataSet, ComputationGraph graph, boolean isLast) {
        return null;
    }

    @Override
    public Pair<SharedTrainingResult, SparkTrainingStats> processMinibatchWithStats(DataSet dataSet, MultiLayerNetwork network, boolean isLast) {
        return null;
    }

    @Override
    public Pair<SharedTrainingResult, SparkTrainingStats> processMinibatchWithStats(DataSet dataSet, ComputationGraph graph, boolean isLast) {
        return null;
    }

    @Override
    public Pair<SharedTrainingResult, SparkTrainingStats> processMinibatchWithStats(MultiDataSet dataSet, ComputationGraph graph, boolean isLast) {
        return null;
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
