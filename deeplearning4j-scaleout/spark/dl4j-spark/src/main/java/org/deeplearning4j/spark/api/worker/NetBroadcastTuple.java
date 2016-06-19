package org.deeplearning4j.spark.api.worker;

import lombok.Data;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * Created by Alex on 14/06/2016.
 */
@Data
public class NetBroadcastTuple implements Serializable {

    private final MultiLayerConfiguration configuration;
    private final ComputationGraphConfiguration graphConfiguration;
    private final INDArray parameters;
    private final Updater updater;
    private final ComputationGraphUpdater graphUpdater;

    public NetBroadcastTuple(MultiLayerConfiguration configuration, INDArray parameters, Updater updater) {
        this(configuration, null, parameters, updater, null);
    }

    public NetBroadcastTuple(ComputationGraphConfiguration graphConfiguration, INDArray parameters, ComputationGraphUpdater graphUpdater) {
        this(null, graphConfiguration, parameters, null, graphUpdater);

    }

    public NetBroadcastTuple(MultiLayerConfiguration configuration, ComputationGraphConfiguration graphConfiguration, INDArray parameters, Updater updater, ComputationGraphUpdater graphUpdater) {
        this.configuration = configuration;
        this.graphConfiguration = graphConfiguration;
        this.parameters = parameters;
        this.updater = updater;
        this.graphUpdater = graphUpdater;
    }
}
