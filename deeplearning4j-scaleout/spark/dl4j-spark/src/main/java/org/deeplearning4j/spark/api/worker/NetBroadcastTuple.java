package org.deeplearning4j.spark.api.worker;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * Created by Alex on 14/06/2016.
 */
@AllArgsConstructor @Data
public class NetBroadcastTuple implements Serializable {

    private final MultiLayerConfiguration configuration;
    private final INDArray parameters;
    private final Updater updater;

}
