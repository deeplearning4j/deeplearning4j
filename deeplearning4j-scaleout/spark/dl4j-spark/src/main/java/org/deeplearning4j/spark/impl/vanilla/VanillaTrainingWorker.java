package org.deeplearning4j.spark.impl.vanilla;

import lombok.AllArgsConstructor;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.api.TrainingWorker;
import org.deeplearning4j.spark.api.WorkerConfiguration;
import org.deeplearning4j.spark.api.worker.NetBroadcastTuple;
import org.nd4j.linalg.dataset.api.DataSet;

/**
 * Created by Alex on 14/06/2016.
 */
@AllArgsConstructor
public class VanillaTrainingWorker implements TrainingWorker<VanillaTrainingResult> {

    private final Broadcast<NetBroadcastTuple> broadcast;

    @Override
    public MultiLayerNetwork getInitialModel() {
        return null;
    }

    @Override
    public VanillaTrainingResult processMinibatch(DataSet dataSet, MultiLayerNetwork network, boolean isLast) {
        return null;
    }

    @Override
    public VanillaTrainingResult getFinalResult(MultiLayerNetwork network) {
        return null;
    }

    @Override
    public WorkerConfiguration getDataConfiguration() {
        return null;
    }
}
