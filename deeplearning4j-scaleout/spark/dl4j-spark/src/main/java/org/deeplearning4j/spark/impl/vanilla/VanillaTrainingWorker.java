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
    private final boolean saveUpdater;
    private final WorkerConfiguration configuration;

    @Override
    public MultiLayerNetwork getInitialModel() {
        NetBroadcastTuple tuple = broadcast.getValue();
        MultiLayerNetwork net = new MultiLayerNetwork(tuple.getConfiguration());
        //Can't have shared parameter array across executors for parameter averaging, hence the 'true' for clone parameters array arg
        net.init(tuple.getParameters(), true);

        if(tuple.getUpdater() != null){
            net.setUpdater(tuple.getUpdater().clone()); //Again: can'h have shared updaters
        }

        return net;
    }

    @Override
    public VanillaTrainingResult processMinibatch(DataSet dataSet, MultiLayerNetwork network, boolean isLast) {
        network.fit(dataSet);

        if(isLast) return getFinalResult(network);

        return null;
    }

    @Override
    public VanillaTrainingResult getFinalResult(MultiLayerNetwork network) {
        //TODO: don't want to use java serialization for updater, in case worker is using cuda and master is using native, etc
        return new VanillaTrainingResult(network.params(), (saveUpdater ? network.getUpdater() : null), network.score());
    }

    @Override
    public WorkerConfiguration getDataConfiguration() {
        return configuration;
    }
}
