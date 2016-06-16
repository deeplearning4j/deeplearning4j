package org.deeplearning4j.spark.api;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.nd4j.linalg.dataset.api.DataSet;

import java.io.Serializable;

/**
 * Created by Alex on 14/06/2016.
 */
public interface TrainingWorker<R extends TrainingResult> extends Serializable {

    MultiLayerNetwork getInitialModel();

    R processMinibatch(DataSet dataSet, MultiLayerNetwork network, boolean isLast);

    Pair<R, SparkTrainingStats> processMinibatchWithStats(DataSet dataSet, MultiLayerNetwork network, boolean isLast);

    R getFinalResult(MultiLayerNetwork network);

    Pair<R, SparkTrainingStats> getFinalResultWithStats(MultiLayerNetwork network);

    WorkerConfiguration getDataConfiguration();
}
