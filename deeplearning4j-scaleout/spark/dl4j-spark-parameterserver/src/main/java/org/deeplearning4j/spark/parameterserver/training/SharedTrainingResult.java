package org.deeplearning4j.spark.parameterserver.training;

import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StorageMetaData;
import org.deeplearning4j.spark.api.TrainingResult;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.impl.paramavg.BaseTrainingResult;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingResult;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;

/**
 * @author raver119@gmail.com
 */
public class SharedTrainingResult extends BaseTrainingResult implements TrainingResult {

    @Override
    public void setStats(SparkTrainingStats sparkTrainingStats) {

    }
}
