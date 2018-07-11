package org.deeplearning4j.spark.parameterserver.training;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StorageMetaData;
import org.deeplearning4j.spark.api.TrainingResult;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.impl.paramavg.BaseTrainingResult;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.Collection;
import java.util.Map;

/**
 * @author raver119@gmail.com
 */
@Data
@AllArgsConstructor
@Builder
@NoArgsConstructor
public class SharedTrainingResult extends BaseTrainingResult implements TrainingResult, Serializable {
    private INDArray updaterStateArray;
    private double scoreSum;
    private int aggregationsCount;
    private SparkTrainingStats sparkTrainingStats;
    private Collection<StorageMetaData> listenerMetaData;
    private Collection<Persistable> listenerStaticInfo;
    private Collection<Persistable> listenerUpdates;
    private Map<String,Integer> minibatchesPerExecutor;


    @Override
    public void setStats(SparkTrainingStats sparkTrainingStats) {
        setSparkTrainingStats(sparkTrainingStats);
    }
}
