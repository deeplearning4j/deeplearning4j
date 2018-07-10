package org.deeplearning4j.spark.parameterserver.accumulation;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StorageMetaData;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.Collection;
import java.util.Map;

/**
 * @author raver119@gmail.com
 */
@AllArgsConstructor
@Data
@NoArgsConstructor
@Builder
public class SharedTrainingAccumulationTuple implements Serializable {
    private INDArray updaterStateArray;
    private double scoreSum;
    private int aggregationsCount;
    private SparkTrainingStats sparkTrainingStats;
    private Collection<StorageMetaData> listenerMetaData;
    private Collection<Persistable> listenerStaticInfo;
    private Collection<Persistable> listenerUpdates;
    private Map<String,Integer> minibatchesPerExecutor;
}
