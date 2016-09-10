package org.deeplearning4j.spark.impl.paramavg;

import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.util.ComputationGraphUtil;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.MultiLayerUpdater;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.deeplearning4j.spark.api.TrainingWorker;
import org.deeplearning4j.spark.api.WorkerConfiguration;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.api.worker.NetBroadcastTuple;
import org.deeplearning4j.spark.impl.paramavg.stats.ParameterAveragingTrainingWorkerStats;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;

/**
 * ParameterAveragingTrainingWorker implements standard parameter averaging every m iterations.
 *
 * @author Alex Black
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
        net.init(tuple.getParameters().unsafeDuplication(), false);

        if(tuple.getUpdaterState() != null){
            net.setUpdater(new MultiLayerUpdater(net, tuple.getUpdaterState().unsafeDuplication()));  //Can't have shared updater state
        }

        if(configuration.isCollectTrainingStats()) stats.logInitEnd();

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner)Nd4j.getExecutioner()).flushQueueBlocking();

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
        net.init(tuple.getParameters().unsafeDuplication(), false);

        if(tuple.getUpdaterState() != null){
            net.setUpdater(new ComputationGraphUpdater(net, tuple.getUpdaterState().unsafeDuplication())); //Again: can't have shared updater state
        }

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner)Nd4j.getExecutioner()).flushQueueBlocking();

        if(configuration.isCollectTrainingStats()) stats.logInitEnd();

        return net;
    }

    @Override
    public ParameterAveragingTrainingResult processMinibatch(DataSet dataSet, MultiLayerNetwork network, boolean isLast) {
        if(configuration.isCollectTrainingStats()) stats.logFitStart();
        network.fit(dataSet);
        if(configuration.isCollectTrainingStats()) stats.logFitEnd(dataSet.numExamples());

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner)Nd4j.getExecutioner()).flushQueueBlocking();

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
        if(configuration.isCollectTrainingStats()) stats.logFitEnd(dataSet.getFeatures(0).size(0));

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner)Nd4j.getExecutioner()).flushQueueBlocking();

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
        INDArray updaterState = null;
        if(saveUpdater){
            Updater u = network.getUpdater();
            if(u != null) updaterState = u.getStateViewArray();
        }

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner)Nd4j.getExecutioner()).flushQueueBlocking();

        return new ParameterAveragingTrainingResult(network.params(), updaterState, network.score());
    }

    @Override
    public ParameterAveragingTrainingResult getFinalResult(ComputationGraph network) {
        INDArray updaterState = null;
        if(saveUpdater){
            ComputationGraphUpdater u = network.getUpdater();
            if(u != null) updaterState = u.getStateViewArray();
        }

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner)Nd4j.getExecutioner()).flushQueueBlocking();

        return new ParameterAveragingTrainingResult(network.params(), updaterState, network.score());
    }

    @Override
    public ParameterAveragingTrainingResult getFinalResultNoData(){
        return new ParameterAveragingTrainingResult(null, null, 0.0, null);
    }

    @Override
    public Pair<ParameterAveragingTrainingResult, SparkTrainingStats> getFinalResultNoDataWithStats(){
        return new Pair<>(new ParameterAveragingTrainingResult(null, null, 0.0, null),null);
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
