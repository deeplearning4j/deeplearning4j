package org.deeplearning4j.spark.impl.paramavg;

import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.util.ComputationGraphUtil;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.api.TrainingWorker;
import org.deeplearning4j.spark.api.WorkerConfiguration;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.api.worker.NetBroadcastTuple;
import org.deeplearning4j.spark.impl.paramavg.stats.ParameterAveragingTrainingWorkerStats;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;

/**
 * Created by Alex on 14/06/2016.
 */
public class ParameterAveragingTrainingWorker implements TrainingWorker<ParameterAveragingTrainingResult> {

    private final Broadcast<NetBroadcastTuple> broadcast;
    private final boolean saveUpdater;
    private final WorkerConfiguration configuration;
    private ParameterAveragingTrainingWorkerStats.ParameterAveragingTrainingWorkerStatsHelper stats = null;

    public ParameterAveragingTrainingWorker(Broadcast<NetBroadcastTuple> broadcast, boolean saveUpdater, WorkerConfiguration configuration) {
        this.broadcast = broadcast;
        this.saveUpdater = saveUpdater;
        this.configuration = configuration;
    }

    @Override
    public MultiLayerNetwork getInitialModel() {
        if(configuration.isCollectTrainingStats()) stats = new ParameterAveragingTrainingWorkerStats.ParameterAveragingTrainingWorkerStatsHelper();

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
    public ComputationGraph getInitialModelGraph() {
        if(configuration.isCollectTrainingStats()) stats = new ParameterAveragingTrainingWorkerStats.ParameterAveragingTrainingWorkerStatsHelper();

        if(configuration.isCollectTrainingStats()) stats.logBroadcastGetValueStart();
        NetBroadcastTuple tuple = broadcast.getValue();
        if(configuration.isCollectTrainingStats()) stats.logBroadcastGetValueEnd();

        ComputationGraph net = new ComputationGraph(tuple.getGraphConfiguration());
        //Can't have shared parameter array across executors for parameter averaging, hence the 'true' for clone parameters array arg
        net.init(tuple.getParameters(), true);

        if(tuple.getGraphUpdater() != null){
            net.setUpdater(tuple.getGraphUpdater().clone()); //Again: can't have shared updaters
        }

        if(configuration.isCollectTrainingStats()) stats.logInitEnd();

        return net;
    }

    @Override
    public ParameterAveragingTrainingResult processMinibatch(DataSet dataSet, MultiLayerNetwork network, boolean isLast) {
        if(configuration.isCollectTrainingStats()) stats.logFitStart();
        network.fit(dataSet);
        if(configuration.isCollectTrainingStats()) stats.logFitEnd();

        if(isLast) return getFinalResult(network);

        return null;
    }

    @Override
    public ParameterAveragingTrainingResult processMinibatch(DataSet dataSet, ComputationGraph graph, boolean isLast){
        return processMinibatch(ComputationGraphUtil.toMultiDataSet(dataSet), graph, isLast);
    }

    @Override
    public ParameterAveragingTrainingResult processMinibatch(MultiDataSet dataSet, ComputationGraph graph, boolean isLast){
        if(configuration.isCollectTrainingStats()) stats.logFitStart();
        graph.fit(dataSet);
        if(configuration.isCollectTrainingStats()) stats.logFitEnd();

        if(isLast) return getFinalResult(graph);

        return null;
    }



    @Override
    public Pair<ParameterAveragingTrainingResult, SparkTrainingStats> processMinibatchWithStats(DataSet dataSet, MultiLayerNetwork network, boolean isLast) {
        ParameterAveragingTrainingResult result = processMinibatch(dataSet,network,isLast);
        if(result == null) return null;

        SparkTrainingStats statsToReturn = (stats != null ? stats.build() : null);
        return new Pair<>(result, statsToReturn);
    }

    @Override
    public Pair<ParameterAveragingTrainingResult, SparkTrainingStats> processMinibatchWithStats(DataSet dataSet, ComputationGraph graph, boolean isLast) {
        return processMinibatchWithStats(ComputationGraphUtil.toMultiDataSet(dataSet), graph, isLast);
    }

    @Override
    public Pair<ParameterAveragingTrainingResult, SparkTrainingStats> processMinibatchWithStats(MultiDataSet dataSet, ComputationGraph graph, boolean isLast) {
        ParameterAveragingTrainingResult result = processMinibatch(dataSet,graph,isLast);
        if(result == null) return null;

        SparkTrainingStats statsToReturn = (stats != null ? stats.build() : null);
        return new Pair<>(result, statsToReturn);
    }

    @Override
    public ParameterAveragingTrainingResult getFinalResult(MultiLayerNetwork network) {
        //TODO: don't want to use java serialization for updater, in case worker is using cuda and master is using native, etc
        return new ParameterAveragingTrainingResult(network.params(), (saveUpdater ? network.getUpdater() : null), network.score());
    }

    @Override
    public ParameterAveragingTrainingResult getFinalResult(ComputationGraph network) {
        //TODO: don't want to use java serialization for updater, in case worker is using cuda and master is using native, etc
        return new ParameterAveragingTrainingResult(network.params(), (saveUpdater ? network.getUpdater() : null), network.score());
    }

    @Override
    public Pair<ParameterAveragingTrainingResult,SparkTrainingStats> getFinalResultWithStats(MultiLayerNetwork network) {
        ParameterAveragingTrainingResult result = getFinalResult(network);
        if(result == null) return null;

        SparkTrainingStats statsToReturn = (stats != null ? stats.build() : null);
        return new Pair<>(result,statsToReturn);
    }

    @Override
    public Pair<ParameterAveragingTrainingResult, SparkTrainingStats> getFinalResultWithStats(ComputationGraph graph){
        ParameterAveragingTrainingResult result = getFinalResult(graph);
        if(result == null) return null;

        SparkTrainingStats statsToReturn = (stats != null ? stats.build() : null);
        return new Pair<>(result,statsToReturn);
    }

    @Override
    public WorkerConfiguration getDataConfiguration() {
        return configuration;
    }



}
