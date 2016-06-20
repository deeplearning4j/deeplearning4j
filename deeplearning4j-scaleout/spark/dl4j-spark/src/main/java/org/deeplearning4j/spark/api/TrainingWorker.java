package org.deeplearning4j.spark.api;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.io.Serializable;

/**
 * Created by Alex on 14/06/2016.
 */
public interface TrainingWorker<R extends TrainingResult> extends Serializable {

    MultiLayerNetwork getInitialModel();

    ComputationGraph getInitialModelGraph();

    R processMinibatch(DataSet dataSet, MultiLayerNetwork network, boolean isLast);

    R processMinibatch(DataSet dataSet, ComputationGraph graph, boolean isLast);

    R processMinibatch(MultiDataSet dataSet, ComputationGraph graph, boolean isLast);

    Pair<R, SparkTrainingStats> processMinibatchWithStats(DataSet dataSet, MultiLayerNetwork network, boolean isLast);

    Pair<R, SparkTrainingStats> processMinibatchWithStats(DataSet dataSet, ComputationGraph graph, boolean isLast);

    Pair<R, SparkTrainingStats> processMinibatchWithStats(MultiDataSet dataSet, ComputationGraph graph, boolean isLast);

    R getFinalResult(MultiLayerNetwork network);

    R getFinalResult(ComputationGraph graph);

    R getFinalResultNoData();

    Pair<R, SparkTrainingStats> getFinalResultNoDataWithStats();

    Pair<R, SparkTrainingStats> getFinalResultWithStats(MultiLayerNetwork network);

    Pair<R, SparkTrainingStats> getFinalResultWithStats(ComputationGraph graph);

    WorkerConfiguration getDataConfiguration();
}
