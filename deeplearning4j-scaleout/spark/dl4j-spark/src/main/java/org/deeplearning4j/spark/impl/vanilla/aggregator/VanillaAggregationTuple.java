package org.deeplearning4j.spark.impl.vanilla.aggregator;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.updater.aggregate.UpdaterAggregator;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * Created by Alex on 15/06/2016.
 */
@AllArgsConstructor @Data
public class VanillaAggregationTuple implements Serializable {
    private final INDArray parameters;
    private final UpdaterAggregator updaterAggregator;
    private final double scoreSum;
}
