package org.deeplearning4j.spark.impl.vanilla;

import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.api.TrainingWorker;
import org.deeplearning4j.spark.api.WorkerConfiguration;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.api.worker.NetBroadcastTuple;
import org.deeplearning4j.spark.impl.vanilla.stats.VanillaTrainingWorkerStats;
import org.nd4j.linalg.dataset.api.DataSet;

/**
 * Created by Alex on 14/06/2016.
 */
public class VanillaTrainingWorker implements TrainingWorker<VanillaTrainingResult> {

    private final Broadcast<NetBroadcastTuple> broadcast;
    private final boolean saveUpdater;
    private final WorkerConfiguration configuration;
    private VanillaTrainingWorkerStats.VanillaTrainingWorkerStatsHelper stats = null;

    public VanillaTrainingWorker(Broadcast<NetBroadcastTuple> broadcast, boolean saveUpdater, WorkerConfiguration configuration) {
        this.broadcast = broadcast;
        this.saveUpdater = saveUpdater;
        this.configuration = configuration;
    }

    @Override
    public MultiLayerNetwork getInitialModel() {
        if(configuration.isCollectTrainingStats()) stats = new VanillaTrainingWorkerStats.VanillaTrainingWorkerStatsHelper();

        if(configuration.isCollectTrainingStats()) stats.logBroadcastGetValueStart();
        NetBroadcastTuple tuple = broadcast.getValue();
        if(configuration.isCollectTrainingStats()) stats.logBroadcastGetValueEnd();

        MultiLayerNetwork net = new MultiLayerNetwork(tuple.getConfiguration());
        //Can't have shared parameter array across executors for parameter averaging, hence the 'true' for clone parameters array arg
        net.init(tuple.getParameters(), true);

        if(tuple.getUpdater() != null){
            net.setUpdater(tuple.getUpdater().clone()); //Again: can't have shared updaters
        }

        if(configuration.isCollectTrainingStats()) stats.logInitEnd();

        return net;
    }

    @Override
    public VanillaTrainingResult processMinibatch(DataSet dataSet, MultiLayerNetwork network, boolean isLast) {

        if(configuration.isCollectTrainingStats()) stats.logFitStart();
        network.fit(dataSet);
        if(configuration.isCollectTrainingStats()) stats.logFitEnd();

        if(isLast) return getFinalResult(network);

        return null;
    }

    @Override
    public Pair<VanillaTrainingResult, SparkTrainingStats> processMinibatchWithStats(DataSet dataSet, MultiLayerNetwork network, boolean isLast) {
        VanillaTrainingResult result = processMinibatch(dataSet,network,isLast);
        if(result == null) return null;

        SparkTrainingStats statsToReturn = (stats != null ? stats.build() : null);
        return new Pair<>(result, statsToReturn);
    }

    @Override
    public VanillaTrainingResult getFinalResult(MultiLayerNetwork network) {
        //TODO: don't want to use java serialization for updater, in case worker is using cuda and master is using native, etc
        return new VanillaTrainingResult(network.params(), (saveUpdater ? network.getUpdater() : null), network.score());
    }

    @Override
    public Pair<VanillaTrainingResult,SparkTrainingStats> getFinalResultWithStats(MultiLayerNetwork network) {
        VanillaTrainingResult result = getFinalResult(network);
        if(result == null) return null;

        SparkTrainingStats statsToReturn = (stats != null ? stats.build() : null);
        return new Pair<>(getFinalResult(network),statsToReturn);
    }

    @Override
    public WorkerConfiguration getDataConfiguration() {
        return configuration;
    }



}
