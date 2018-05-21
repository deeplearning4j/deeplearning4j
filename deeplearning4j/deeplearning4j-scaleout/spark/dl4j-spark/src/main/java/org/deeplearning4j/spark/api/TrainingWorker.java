package org.deeplearning4j.spark.api;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.primitives.Pair;

import java.io.Serializable;

/**
 * TrainingWorker is a small serializable class that can be passed (in serialized form) to each Spark executor
 * for actually conducting training. The results are then passed back to the {@link TrainingMaster} for processing.<br>
 * <p>
 * TrainingWorker implementations provide a layer of abstraction for network learning tha should allow for more flexibility/
 * control over how learning is conducted (including for example asynchronous communication)
 *
 * @author Alex Black
 */
public interface TrainingWorker<R extends TrainingResult> extends Serializable {

    /**
     * Remove a training hook from the worker
     * @param trainingHook the training hook to remove
     */
    void removeHook(TrainingHook trainingHook);

    /**
     * Add a training hook to be used
     * during training of the worker
     * @param trainingHook the training hook to add
     */
    void addHook(TrainingHook trainingHook);

    /**
     * Get the initial model when training a MultiLayerNetwork/SparkDl4jMultiLayer
     *
     * @return Initial model for this worker
     */
    MultiLayerNetwork getInitialModel();

    /**
     * Get the initial model when training a ComputationGraph/SparkComputationGraph
     *
     * @return Initial model for this worker
     */
    ComputationGraph getInitialModelGraph();

    /**
     * Process (fit) a minibatch for a MultiLayerNetwork
     *
     * @param dataSet Data set to train on
     * @param network Network to train
     * @param isLast  If true: last data set currently available. If false: more data sets will be processed for this executor
     * @return Null, or a training result if training should be terminated immediately.
     */
    R processMinibatch(DataSet dataSet, MultiLayerNetwork network, boolean isLast);

    /**
     * Process (fit) a minibatch for a ComputationGraph
     *
     * @param dataSet Data set to train on
     * @param graph   Network to train
     * @param isLast  If true: last data set currently available. If false: more data sets will be processed for this executor
     * @return Null, or a training result if training should be terminated immediately.
     */
    R processMinibatch(DataSet dataSet, ComputationGraph graph, boolean isLast);

    /**
     * Process (fit) a minibatch for a ComputationGraph using a MultiDataSet
     *
     * @param dataSet Data set to train on
     * @param graph   Network to train
     * @param isLast  If true: last data set currently available. If false: more data sets will be processed for this executor
     * @return Null, or a training result if training should be terminated immediately.
     */
    R processMinibatch(MultiDataSet dataSet, ComputationGraph graph, boolean isLast);

    /**
     * As per {@link #processMinibatch(DataSet, MultiLayerNetwork, boolean)} but used when {@link SparkTrainingStats} are being collecte
     */
    Pair<R, SparkTrainingStats> processMinibatchWithStats(DataSet dataSet, MultiLayerNetwork network, boolean isLast);

    /**
     * As per {@link #processMinibatch(DataSet, ComputationGraph, boolean)} but used when {@link SparkTrainingStats} are being collected
     */
    Pair<R, SparkTrainingStats> processMinibatchWithStats(DataSet dataSet, ComputationGraph graph, boolean isLast);

    /**
     * As per {@link #processMinibatch(MultiDataSet, ComputationGraph, boolean)} but used when {@link SparkTrainingStats} are being collected
     */
    Pair<R, SparkTrainingStats> processMinibatchWithStats(MultiDataSet dataSet, ComputationGraph graph, boolean isLast);

    /**
     * Get the final result to be returned to the driver
     *
     * @param network Current state of the network
     * @return Result to return to the driver
     */
    R getFinalResult(MultiLayerNetwork network);

    /**
     * Get the final result to be returned to the driver
     *
     * @param graph Current state of the network
     * @return Result to return to the driver
     */
    R getFinalResult(ComputationGraph graph);

    /**
     * Get the final result to be returned to the driver, if no data was available for this executor
     *
     * @return Result to return to the driver
     */
    R getFinalResultNoData();

    /**
     * As per {@link #getFinalResultNoData()} but used when {@link SparkTrainingStats} are being collected
     */
    Pair<R, SparkTrainingStats> getFinalResultNoDataWithStats();

    /**
     * As per {@link #getFinalResult(MultiLayerNetwork)} but used when {@link SparkTrainingStats} are being collected
     */
    Pair<R, SparkTrainingStats> getFinalResultWithStats(MultiLayerNetwork network);

    /**
     * As per {@link #getFinalResult(ComputationGraph)} but used when {@link SparkTrainingStats} are being collected
     */
    Pair<R, SparkTrainingStats> getFinalResultWithStats(ComputationGraph graph);

    /**
     * Get the {@link WorkerConfiguration} that contains information such as minibatch sizes, etc
     *
     * @return Worker configuration
     */
    WorkerConfiguration getDataConfiguration();
}
