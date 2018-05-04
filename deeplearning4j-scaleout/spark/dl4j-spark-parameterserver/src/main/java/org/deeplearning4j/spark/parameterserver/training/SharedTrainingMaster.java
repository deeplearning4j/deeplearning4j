package org.deeplearning4j.spark.parameterserver.training;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaRDDLike;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.input.PortableDataStream;
import org.apache.spark.storage.StorageLevel;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.api.storage.StorageMetaData;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.spark.api.*;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.api.worker.NetBroadcastTuple;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.BaseTrainingMaster;
import org.deeplearning4j.spark.impl.paramavg.stats.ParameterAveragingTrainingMasterStats;
import org.deeplearning4j.spark.parameterserver.accumulation.SharedTrainingAccumulationFunction;
import org.deeplearning4j.spark.parameterserver.accumulation.SharedTrainingAccumulationTuple;
import org.deeplearning4j.spark.parameterserver.accumulation.SharedTrainingAggregateFunction;
import org.deeplearning4j.spark.parameterserver.conf.SharedTrainingConfiguration;
import org.deeplearning4j.spark.parameterserver.functions.*;
import org.deeplearning4j.spark.parameterserver.networking.SilentTrainingDriver;
import org.deeplearning4j.spark.util.SparkUtils;
import org.deeplearning4j.util.UIDProvider;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.distributed.VoidParameterServer;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.enums.ExecutionMode;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.enums.TransportType;
import org.nd4j.parameterserver.distributed.transport.MulticastTransport;
import org.nd4j.parameterserver.distributed.transport.RoutedTransport;
import org.nd4j.parameterserver.distributed.transport.Transport;
import org.nd4j.parameterserver.distributed.util.NetworkOrganizer;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class SharedTrainingMaster extends BaseTrainingMaster<SharedTrainingResult, SharedTrainingWorker>
                implements TrainingMaster<SharedTrainingResult, SharedTrainingWorker> {
    protected List<TrainingHook> trainingHooks;
    protected VoidConfiguration voidConfiguration;

    protected Integer numWorkers;
    protected Integer numWorkersPerNode;
    protected RDDTrainingApproach rddTrainingApproach;
    protected StorageLevel storageLevel;

    protected boolean collectTrainingStats;
    protected int rddDataSetNumExamples;
    protected long debugLongerIterations = 0L;

    // TODO: this option should be abstracted, if we decide to generalize this trainingmaster
    protected double threshold;
    protected double thresholdStep;
    protected double minThreshold;
    protected double stepTrigger = 0.05;
    protected int stepDelay = 50;
    protected int shakeFrequency;

    protected Repartition repartition;
    protected RepartitionStrategy repartitionStrategy;

    protected ParameterAveragingTrainingMasterStats.ParameterAveragingTrainingMasterStatsHelper stats;

    protected Random rng;

    protected AtomicBoolean isFirstRun;

    // better ignore
    protected transient Broadcast<NetBroadcastTuple> broadcastModel;
    protected transient Broadcast<SharedTrainingConfiguration> broadcastConfiguration;
    protected transient Transport transport;
    protected transient SilentTrainingDriver trainingDriver;

    protected SharedTrainingMaster() {
        // just a stub for ser/de
    }

    public SharedTrainingMaster(@NonNull VoidConfiguration voidConfiguration, Integer numWorkers,
                    RDDTrainingApproach rddTrainingApproach, StorageLevel storageLevel, boolean collectTrainingStats,
                    RepartitionStrategy repartitionStrategy, Repartition repartition, double threshold,
                    double minThreshold, double thresholdStep, double stepTrigger, int stepDelay, int shakeFrequency,
                    int batchSizePerWorker, long debugLongerIterations, int numWorkersPerNode) {
        this.voidConfiguration = voidConfiguration;
        this.numWorkers = numWorkers;
        this.threshold = threshold;
        this.minThreshold = minThreshold;
        this.thresholdStep = thresholdStep;
        this.stepTrigger = stepTrigger;
        this.stepDelay = stepDelay;
        this.rddTrainingApproach = rddTrainingApproach;
        this.repartitionStrategy = repartitionStrategy;
        this.repartition = repartition;
        this.storageLevel = storageLevel;
        this.collectTrainingStats = collectTrainingStats;
        this.isFirstRun = new AtomicBoolean(false);
        this.batchSizePerWorker = batchSizePerWorker;
        this.rddDataSetNumExamples = batchSizePerWorker;
        this.debugLongerIterations = debugLongerIterations;
        this.numWorkersPerNode = numWorkersPerNode;


        if (collectTrainingStats)
            stats = new ParameterAveragingTrainingMasterStats.ParameterAveragingTrainingMasterStatsHelper();


        String jvmuid = UIDProvider.getJVMUID();
        this.trainingMasterUID =
                        System.currentTimeMillis() + "_" + (jvmuid.length() <= 8 ? jvmuid : jvmuid.substring(0, 8));
    }

    @Override
    public void removeHook(TrainingHook trainingHook) {
        if (trainingHooks != null)
            trainingHooks.remove(trainingHook);
    }

    @Override
    public void addHook(@NonNull TrainingHook trainingHook) {
        if (trainingHooks == null)
            trainingHooks = new ArrayList<>();

        trainingHooks.add(trainingHook);
    }

    @Override
    public String toJson() {
        ObjectMapper om = getJsonMapper();

        try {
            return om.writeValueAsString(this);
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Error producing JSON representation for ParameterAveragingTrainingMaster", e);
        }
    }

    @Override
    public String toYaml() {
        ObjectMapper om = getYamlMapper();

        try {
            return om.writeValueAsString(this);
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Error producing YAML representation for ParameterAveragingTrainingMaster", e);
        }
    }

    /**
     * Create a SharedTrainingMaster instance by deserializing a JSON string that has been serialized with
     * {@link #toJson()}
     *
     * @param jsonStr SharedTrainingMaster configuration serialized as JSON
     */
    public static SharedTrainingMaster fromJson(String jsonStr) {
        ObjectMapper om = getJsonMapper();
        try {
            return om.readValue(jsonStr, SharedTrainingMaster.class);
        } catch (IOException e) {
            throw new RuntimeException("Could not parse JSON", e);
        }
    }

    /**
     * Create a SharedTrainingMaster instance by deserializing a YAML string that has been serialized with
     * {@link #toYaml()}
     *
     * @param yamlStr SharedTrainingMaster configuration serialized as YAML
     */
    public static SharedTrainingMaster fromYaml(String yamlStr) {
        ObjectMapper om = getYamlMapper();
        try {
            return om.readValue(yamlStr, SharedTrainingMaster.class);
        } catch (IOException e) {
            throw new RuntimeException("Could not parse YAML", e);
        }
    }

    @Override
    public SharedTrainingWorker getWorkerInstance(SparkDl4jMultiLayer network) {
        /*
            Here we're going create our worker, which will be passed into corresponding FlatMapFunction
         */
        NetBroadcastTuple tuple = new NetBroadcastTuple(network.getNetwork().getLayerWiseConfigurations(),
                        network.getNetwork().params(), network.getNetwork().getUpdater().getStateViewArray());

        SharedTrainingConfiguration configuration = SharedTrainingConfiguration.builder().threshold(threshold)
                        .minThreshold(minThreshold).shakeFrequency(shakeFrequency).thresholdStep(thresholdStep)
                        .stepTrigger(stepTrigger).stepDelay(stepDelay).voidConfiguration(voidConfiguration)
                        .debugLongerIterations(debugLongerIterations).numberOfWorkersPerNode(numWorkersPerNode).build();

        if (collectTrainingStats)
            stats.logBroadcastStart();

        if (broadcastModel == null)
            broadcastModel = network.getSparkContext().broadcast(tuple);

        if (broadcastConfiguration == null)
            broadcastConfiguration = network.getSparkContext().broadcast(configuration);

        if (collectTrainingStats)
            stats.logBroadcastEnd();

        SharedTrainingWorker worker = new SharedTrainingWorker(broadcastModel, broadcastConfiguration);

        return worker;
    }

    @Override
    public SharedTrainingWorker getWorkerInstance(SparkComputationGraph graph) {
        NetBroadcastTuple tuple = new NetBroadcastTuple(graph.getNetwork().getConfiguration(),
                        graph.getNetwork().params(), graph.getNetwork().getUpdater().getStateViewArray());

        SharedTrainingConfiguration configuration = SharedTrainingConfiguration.builder().threshold(threshold)
                        .minThreshold(minThreshold).shakeFrequency(shakeFrequency).thresholdStep(thresholdStep)
                        .voidConfiguration(voidConfiguration).debugLongerIterations(debugLongerIterations)
                        .numberOfWorkersPerNode(numWorkersPerNode).build();

        if (collectTrainingStats)
            stats.logBroadcastStart();

        if (broadcastModel == null)
            broadcastModel = graph.getSparkContext().broadcast(tuple);

        if (broadcastConfiguration == null)
            broadcastConfiguration = graph.getSparkContext().broadcast(configuration);

        if (collectTrainingStats)
            stats.logBroadcastEnd();

        SharedTrainingWorker worker = new SharedTrainingWorker(broadcastModel, broadcastConfiguration);

        return worker;
    }

    protected int numObjectsEachWorker(int numExamplesEachRddObject) {
        return batchSizePerWorker / numExamplesEachRddObject;
    }

    protected int getNumDataSetObjectsPerSplit(int numExamplesEachRddObject) {
        int dataSetObjectsPerSplit;
        if (numExamplesEachRddObject == 1) {
            dataSetObjectsPerSplit = numWorkers * batchSizePerWorker; // * averagingFrequency;
        } else {
            int numDataSetObjsReqEachWorker = numObjectsEachWorker(numExamplesEachRddObject);
            if (numDataSetObjsReqEachWorker < 1) {
                //In this case: more examples in a DataSet object than we actually require
                //For example, 100 examples in DataSet, with batchSizePerWorker=50 and averagingFrequency=1
                numDataSetObjsReqEachWorker = 1;
            }

            dataSetObjectsPerSplit = numDataSetObjsReqEachWorker * numWorkers;
        }
        return dataSetObjectsPerSplit;
    }

    protected <T> JavaRDD<T>[] getSplitRDDs(JavaRDD<T> trainingData, int totalDataSetObjectCount,
                    int examplesPerDataSetObject) {
        int dataSetObjectsPerSplit = getNumDataSetObjectsPerSplit(examplesPerDataSetObject);

        if (collectTrainingStats)
            stats.logSplitStart();

        JavaRDD<T>[] splits = SparkUtils.balancedRandomSplit(totalDataSetObjectCount, dataSetObjectsPerSplit,
                        trainingData, rng.nextLong());

        if (collectTrainingStats)
            stats.logSplitEnd();

        return splits;
    }

    protected <T, Repr extends JavaRDDLike<T, Repr>> long getTotalDataSetObjectCount(
                    JavaRDDLike<T, Repr> trainingData) {
        if (collectTrainingStats)
            stats.logCountStart();

        long totalDataSetObjectCount = trainingData.count();

        if (collectTrainingStats)
            stats.logCountEnd();

        return totalDataSetObjectCount;
    }

    protected void executeTrainingDirect(SparkDl4jMultiLayer network, JavaRDD<DataSet> trainingData) {
        if (collectTrainingStats)
            stats.logFitStart();

        //For "vanilla" parameter averaging training, we need to split the full data set into batches of size N, such that we can process the specified
        // number of minibatches between averagings
        //But to do that, wee need to know: (a) the number of examples, and (b) the number of workers
        if (storageLevel != null)
            trainingData.persist(storageLevel);

        long totalDataSetObjectCount = getTotalDataSetObjectCount(trainingData);

        // since this is real distributed training, we don't need to split data
        doIteration(network, trainingData, 1, 1);

        if (collectTrainingStats)
            stats.logFitEnd((int) totalDataSetObjectCount);
    }

    protected void executeTrainingDirectMDS(SparkComputationGraph network, JavaRDD<MultiDataSet> trainingData) {
        if (collectTrainingStats)
            stats.logFitStart();

        //For "vanilla" parameter averaging training, we need to split the full data set into batches of size N, such that we can process the specified
        // number of minibatches between averagings
        //But to do that, wee need to know: (a) the number of examples, and (b) the number of workers
        if (storageLevel != null)
            trainingData.persist(storageLevel);

        long totalDataSetObjectCount = getTotalDataSetObjectCount(trainingData);

        // since this is real distributed training, we don't need to split data
        doIterationMDS(network, trainingData, 1, 1);

        if (collectTrainingStats)
            stats.logFitEnd((int) totalDataSetObjectCount);
    }

    protected void executeTrainingDirect(SparkComputationGraph network, JavaRDD<DataSet> trainingData) {
        if (collectTrainingStats)
            stats.logFitStart();

        //For "vanilla" parameter averaging training, we need to split the full data set into batches of size N, such that we can process the specified
        // number of minibatches between averagings
        //But to do that, wee need to know: (a) the number of examples, and (b) the number of workers
        if (storageLevel != null)
            trainingData.persist(storageLevel);

        long totalDataSetObjectCount = getTotalDataSetObjectCount(trainingData);

        // since this is real distributed training, we don't need to split data
        doIteration(network, trainingData, 1, 1);

        if (collectTrainingStats)
            stats.logFitEnd((int) totalDataSetObjectCount);
    }


    protected void executeTrainingPathsHelper(SparkDl4jMultiLayer network, JavaRDD<String> trainingDataPaths,
                    int dataSetObjectsNumExamples) {
        if (numWorkers == null)
            numWorkers = network.getSparkContext().defaultParallelism();

        if (collectTrainingStats)
            stats.logFitStart();

        if (storageLevelStreams != null)
            trainingDataPaths.persist(storageLevelStreams);

        long totalDataSetObjectCount = getTotalDataSetObjectCount(trainingDataPaths);

        doIterationPaths(network, null, trainingDataPaths, 1, 1, dataSetObjectsNumExamples);

        if (collectTrainingStats)
            stats.logFitEnd((int) totalDataSetObjectCount);
    }

    protected void executeTrainingPathsHelper(SparkComputationGraph network, JavaRDD<String> trainingDataPaths,
                    int dataSetObjectsNumExamples) {
        if (numWorkers == null)
            numWorkers = network.getSparkContext().defaultParallelism();

        if (collectTrainingStats)
            stats.logFitStart();

        if (storageLevelStreams != null)
            trainingDataPaths.persist(storageLevelStreams);

        long totalDataSetObjectCount = getTotalDataSetObjectCount(trainingDataPaths);

        doIterationPaths(null, network, trainingDataPaths, 1, 1, dataSetObjectsNumExamples);

        if (collectTrainingStats)
            stats.logFitEnd((int) totalDataSetObjectCount);
    }

    protected void executeTrainingPathsMDSHelper(SparkComputationGraph network, JavaRDD<String> trainingMultiDataPaths,
                    int dataSetObjectsNumExamples) {
        if (numWorkers == null)
            numWorkers = network.getSparkContext().defaultParallelism();

        if (collectTrainingStats)
            stats.logFitStart();
        if (storageLevelStreams != null)
            trainingMultiDataPaths.persist(storageLevelStreams);

        long totalDataSetObjectCount = getTotalDataSetObjectCount(trainingMultiDataPaths);

        int splitNum = 1;

        doIterationPathsMDS(network, trainingMultiDataPaths, splitNum++, 1, dataSetObjectsNumExamples);


        if (collectTrainingStats)
            stats.logFitEnd((int) totalDataSetObjectCount);
    }

    protected void prepareNetworkAndStuff(SparkDl4jMultiLayer network, SparkComputationGraph graph) {
        if (network == null && graph == null)
            throw new IllegalStateException("Both MLN & CG are undefined");

        // first of all, we're instantiating ParameterServer shard here\
        if (numWorkers == null)
            numWorkers = network != null ? network.getSparkContext().defaultParallelism()
                            : graph.getSparkContext().defaultParallelism();

        // set current box as controller, if field is unset - switch to next stop
        if (voidConfiguration.getControllerAddress() == null) {
            try {
                String sparkIp = InetAddress.getByName(System.getenv("SPARK_PUBLIC_DNS")).getHostAddress();
                voidConfiguration.setControllerAddress(sparkIp);
            } catch (UnknownHostException e) {
            }
        }

        // next step - is to get ip address that matches specific network mask
        if (voidConfiguration.getControllerAddress() == null && voidConfiguration.getNetworkMask() != null) {
            NetworkOrganizer organizer = new NetworkOrganizer(voidConfiguration.getNetworkMask());
            voidConfiguration.setControllerAddress(organizer.getMatchingAddress());
        }

        if (voidConfiguration.getControllerAddress() == null)
            voidConfiguration.setControllerAddress(System.getenv("DL4J_VOID_IP"));

        if (voidConfiguration.getControllerAddress() == null)
            throw new DL4JInvalidConfigException(
                            "Can't get Spark Master local address. Please specify it manually using VoidConfiguration.setControllerAddress(String) method or VoidConfiguration.setNetworkMask(String) method");

        // we're forcing proper defaults
        log.info("Setting controller address to {}:{}", voidConfiguration.getControllerAddress(),
                        voidConfiguration.getUnicastPort());
        voidConfiguration.setShardAddresses(voidConfiguration.getControllerAddress());
        voidConfiguration.setNumberOfShards(1);

        Transport transport = voidConfiguration.getTransportType() == TransportType.ROUTED ? new RoutedTransport()
                        : voidConfiguration.getTransportType() == TransportType.BROADCAST ? new MulticastTransport()
                                        : this.transport;

        if (transport == null)
            throw new DL4JInvalidConfigException("No Transport implementation was defined for this training session!");

        if (network != null)
            network.getNetwork().init();
        else
            graph.getNetwork().init();

        // this instance will be SilentWorker - it'll accept and apply messages, but won't contribute to training. And we init it only once
        if (isFirstRun.compareAndSet(false, true)) {
            trainingDriver = new SilentTrainingDriver(
                            network != null ? network.getNetwork().params() : graph.getNetwork().params(),
                            network != null ? network.getNetwork().getOptimizer().getStepFunction()
                                            : graph.getNetwork().getOptimizer().getStepFunction());
            VoidParameterServer.getInstance().init(voidConfiguration, transport, trainingDriver);
        }
    }

    protected void finalizeTraining() {
        /*
            Here we basically want to do few things:
            1) update statistics, if any
            2) finalize updates of silent worker
            3) pull back gradients, maybe?
         */

        // applying non-applied updates, if any :)
        if (trainingDriver != null)
            trainingDriver.finishTraining(0L, 0L);
    }

    @Override
    public void executeTraining(SparkDl4jMultiLayer network, JavaRDD<DataSet> trainingData) {
        /*
            This method (and other similar methods) is basically one of our entry points, here we'll spawn our training process:
            1) broadcast everything needed: initial model params, updaters state, conf. Useful for uptraining
            2) shuffle, if needed
            3) repartition, if needed
            4) EXECUTE SILENT WORKER
            5) invoke training function via mapPartitions
            6) wait till finished
            7) do something with final model, i.e. export it somewhere :)
         */

        prepareNetworkAndStuff(network, null);

        // at this moment we have coordinator server up (master works as coordinator)
        if (rddTrainingApproach == RDDTrainingApproach.Direct) {
            executeTrainingDirect(network, trainingData);
        } else if (rddTrainingApproach == RDDTrainingApproach.Export) {
            //Export data if required (or, use cached export)
            JavaRDD<String> paths = exportIfRequired(network.getSparkContext(), trainingData);
            executeTrainingPathsHelper(network, paths, batchSizePerWorker);
        } else
            throw new DL4JInvalidConfigException(
                            "Unknown RDDtrainingApproach [" + rddTrainingApproach + "] was specified!");
    }

    @Override
    public void executeTraining(SparkDl4jMultiLayer network, JavaPairRDD<String, PortableDataStream> trainingData) {
        prepareNetworkAndStuff(network, null);

        doIterationPDS(network, null, trainingData.values(), 1, 1);
    }

    @Override
    public void executeTrainingPaths(SparkDl4jMultiLayer network, JavaRDD<String> trainingDataPaths) {
        prepareNetworkAndStuff(network, null);

        executeTrainingPathsHelper(network, trainingDataPaths, batchSizePerWorker);
    }

    @Override
    public void executeTraining(SparkComputationGraph graph, JavaRDD<DataSet> trainingData) {
        prepareNetworkAndStuff(null, graph);

        // at this moment we have coordinator server up (master works as coordinator)
        if (rddTrainingApproach == RDDTrainingApproach.Direct) {
            executeTrainingDirect(graph, trainingData);
        } else if (rddTrainingApproach == RDDTrainingApproach.Export) {
            //Export data if required (or, use cached export)
            JavaRDD<String> paths = exportIfRequired(graph.getSparkContext(), trainingData);
            executeTrainingPathsHelper(graph, paths, batchSizePerWorker);
        } else
            throw new DL4JInvalidConfigException(
                            "Unknown RDDtrainingApproach [" + rddTrainingApproach + "] was specified!");
    }

    @Override
    public void executeTraining(SparkComputationGraph network, JavaPairRDD<String, PortableDataStream> trainingData) {
        prepareNetworkAndStuff(null, network);

        doIterationPDS(null, network, trainingData.values(), 1, 1);
    }

    @Override
    public void executeTrainingPaths(SparkComputationGraph network, JavaRDD<String> trainingDataPaths) {
        prepareNetworkAndStuff(null, network);

        executeTrainingPathsHelper(network, trainingDataPaths, batchSizePerWorker);
    }

    @Override
    public void executeTrainingPathsMDS(SparkComputationGraph network, JavaRDD<String> trainingMultiDataSetPaths) {
        prepareNetworkAndStuff(null, network);

        executeTrainingPathsMDSHelper(network, trainingMultiDataSetPaths, batchSizePerWorker);
    }

    @Override
    public void executeTrainingMDS(SparkComputationGraph graph, JavaRDD<MultiDataSet> trainingData) {
        prepareNetworkAndStuff(null, graph);

        // at this moment we have coordinator server up (master works as coordinator)
        if (rddTrainingApproach == RDDTrainingApproach.Direct) {
            executeTrainingDirectMDS(graph, trainingData);
        } else if (rddTrainingApproach == RDDTrainingApproach.Export) {
            //Export data if required (or, use cached export)
            JavaRDD<String> paths = exportIfRequiredMDS(graph.getSparkContext(), trainingData);
            executeTrainingPathsMDSHelper(graph, paths, batchSizePerWorker);
        } else
            throw new DL4JInvalidConfigException(
                            "Unknown RDDtrainingApproach [" + rddTrainingApproach + "] was specified!");
    }

    @Override
    public void executeTrainingMDS(SparkComputationGraph network,
                    JavaPairRDD<String, PortableDataStream> trainingData) {
        prepareNetworkAndStuff(null, network);

        doIterationMultiPDS(network, trainingData.values(), 1, 1);
    }

    @Override
    public void setCollectTrainingStats(boolean collectTrainingStats) {
        this.collectTrainingStats = collectTrainingStats;
    }

    @Override
    public boolean getIsCollectTrainingStats() {
        return collectTrainingStats;
    }

    @Override
    public SparkTrainingStats getTrainingStats() {
        return null;
    }

    @Override
    public void setListeners(Collection<TrainingListener> listeners) {
        // optional stuff actually
    }

    @Override
    public void setListeners(StatsStorageRouter router, Collection<TrainingListener> listeners) {
        // optional stuff actually
    }


    protected void processResults(SparkDl4jMultiLayer network, SparkComputationGraph graph,
                    JavaRDD<SharedTrainingResult> results) {
        if (network == null && graph == null)
            throw new IllegalStateException("Both MLN & CG are null");


        finalizeTraining();

        if (collectTrainingStats)
            stats.logAggregateStartTime();

        SharedTrainingAccumulationTuple finalResult = results.treeAggregate(null, new SharedTrainingAggregateFunction(),
                        new SharedTrainingAccumulationFunction(), 4);
        SparkTrainingStats aggregatedStats = finalResult.getSparkTrainingStats();
        if (collectTrainingStats)
            stats.logAggregationEndTime();


        if (collectTrainingStats)
            stats.logProcessParamsUpdaterStart();

        if (finalResult.getUpdaterStateArray() != null) {

            if (finalResult.getAggregationsCount() > 1) {
                finalResult.getUpdaterStateArray().divi(finalResult.getAggregationsCount());
            }

            if (network != null) {
                if (network.getNetwork().getUpdater() != null
                                && network.getNetwork().getUpdater().getStateViewArray() != null)
                    network.getNetwork().getUpdater().getStateViewArray().assign(finalResult.getUpdaterStateArray());
            } else {
                if (graph.getNetwork().getUpdater() != null
                                && graph.getNetwork().getUpdater().getStateViewArray() != null)
                    graph.getNetwork().getUpdater().getStateViewArray().assign(finalResult.getUpdaterStateArray());
            }
        }


        double score = finalResult.getScoreSum() / Math.max(1, finalResult.getAggregationsCount());

        if (network != null) {
            network.getNetwork().setScore(score);
        } else {
            graph.getNetwork().setScore(score);
        }

        if (collectTrainingStats)
            stats.logProcessParamsUpdaterEnd();


        if (collectTrainingStats) {
            stats.logProcessParamsUpdaterEnd();
            stats.addWorkerStats(aggregatedStats);
        }

        if (statsStorage != null) {
            Collection<StorageMetaData> meta = finalResult.getListenerMetaData();
            if (meta != null && !meta.isEmpty()) {
                statsStorage.putStorageMetaData(meta);
            }

            Collection<Persistable> staticInfo = finalResult.getListenerStaticInfo();
            if (staticInfo != null && !staticInfo.isEmpty()) {
                statsStorage.putStaticInfo(staticInfo);
            }

            Collection<Persistable> updates = finalResult.getListenerUpdates();
            if (updates != null && !updates.isEmpty()) {
                statsStorage.putUpdate(updates);
            }
        }

        Nd4j.getExecutioner().commit();
    }

    protected void doIteration(SparkDl4jMultiLayer network, JavaRDD<DataSet> split, int splitNum, int numSplits) {
        log.info("Starting training of split {} of {}. workerMiniBatchSize={}, updatesThreshold={}, Configured for {} workers",
                        splitNum, numSplits, batchSizePerWorker, threshold, numWorkers);

        if (collectTrainingStats)
            stats.logMapPartitionsStart();

        JavaRDD<DataSet> splitData = split;

        if (collectTrainingStats)
            stats.logRepartitionStart();

        splitData = SparkUtils.repartition(splitData, repartition, repartitionStrategy,
                        numObjectsEachWorker(rddDataSetNumExamples), numWorkers);
        int nPartitions = splitData.partitions().size();

        if (collectTrainingStats && repartition != Repartition.Never)
            stats.logRepartitionEnd();


        FlatMapFunction<Iterator<DataSet>, SharedTrainingResult> function =
                        new SharedFlatMapDataSet<>(getWorkerInstance(network));

        JavaRDD<SharedTrainingResult> result = splitData.mapPartitions(function);

        processResults(network, null, result);

        if (collectTrainingStats)
            stats.logMapPartitionsEnd(nPartitions);
    }

    protected void doIterationMDS(SparkComputationGraph network, JavaRDD<MultiDataSet> split, int splitNum,
                    int numSplits) {
        log.info("Starting training of split {} of {}. workerMiniBatchSize={}, updatesThreshold={}, Configured for {} workers",
                        splitNum, numSplits, batchSizePerWorker, threshold, numWorkers);

        if (collectTrainingStats)
            stats.logMapPartitionsStart();

        JavaRDD<MultiDataSet> splitData = split;

        if (collectTrainingStats)
            stats.logRepartitionStart();

        splitData = SparkUtils.repartition(splitData, repartition, repartitionStrategy,
                        numObjectsEachWorker(rddDataSetNumExamples), numWorkers);
        int nPartitions = splitData.partitions().size();

        if (collectTrainingStats && repartition != Repartition.Never)
            stats.logRepartitionEnd();


        FlatMapFunction<Iterator<MultiDataSet>, SharedTrainingResult> function =
                        new SharedFlatMapMultiDataSet<>(getWorkerInstance(network));

        JavaRDD<SharedTrainingResult> result = splitData.mapPartitions(function);

        processResults(null, network, result);

        if (collectTrainingStats)
            stats.logMapPartitionsEnd(nPartitions);
    }

    protected void doIteration(SparkComputationGraph network, JavaRDD<DataSet> split, int splitNum, int numSplits) {
        log.info("Starting training of split {} of {}. workerMiniBatchSize={}, updatesThreshold={}, Configured for {} workers",
                        splitNum, numSplits, batchSizePerWorker, threshold, numWorkers);

        if (collectTrainingStats)
            stats.logMapPartitionsStart();

        JavaRDD<DataSet> splitData = split;

        if (collectTrainingStats)
            stats.logRepartitionStart();

        splitData = SparkUtils.repartition(splitData, repartition, repartitionStrategy,
                        numObjectsEachWorker(rddDataSetNumExamples), numWorkers);
        int nPartitions = splitData.partitions().size();

        if (collectTrainingStats && repartition != Repartition.Never)
            stats.logRepartitionEnd();


        FlatMapFunction<Iterator<DataSet>, SharedTrainingResult> function =
                        new SharedFlatMapDataSet<>(getWorkerInstance(network));

        JavaRDD<SharedTrainingResult> result = splitData.mapPartitions(function);

        processResults(null, network, result);

        if (collectTrainingStats)
            stats.logMapPartitionsEnd(nPartitions);
    }

    protected void doIterationPathsMDS(SparkComputationGraph graph, JavaRDD<String> split, int splitNum, int numSplits,
                    int dataSetObjectNumExamples) {

        log.info("Starting training of split {} of {}. workerMiniBatchSize={}, updatesThreshold={}, Configured for {} workers",
                        splitNum, numSplits, batchSizePerWorker, threshold, numWorkers);

        if (collectTrainingStats)
            stats.logMapPartitionsStart();

        JavaRDD<String> splitData = split;

        if (collectTrainingStats)
            stats.logRepartitionStart();

        // FIXME: do we still want that?
        splitData = SparkUtils.repartition(splitData, repartition, repartitionStrategy,
                        numObjectsEachWorker(dataSetObjectNumExamples), numWorkers);

        int nPartitions = splitData.partitions().size();
        if (collectTrainingStats && repartition != Repartition.Never)
            stats.logRepartitionEnd();

        FlatMapFunction<Iterator<String>, SharedTrainingResult> function =
                        new SharedFlatMapPathsMDS<>(getWorkerInstance(graph));


        JavaRDD<SharedTrainingResult> result = splitData.mapPartitions(function);

        processResults(null, graph, result);

        if (collectTrainingStats)
            stats.logMapPartitionsEnd(nPartitions);
    }


    protected void doIterationPaths(SparkDl4jMultiLayer network, SparkComputationGraph graph, JavaRDD<String> split,
                    int splitNum, int numSplits, int dataSetObjectNumExamples) {
        if (network == null && graph == null)
            throw new DL4JInvalidConfigException("Both MLN & CompGraph are NULL");

        log.info("Starting training of split {} of {}. workerMiniBatchSize={}, updatesThreshold={}, Configured for {} workers",
                        splitNum, numSplits, batchSizePerWorker, threshold, numWorkers);

        if (collectTrainingStats)
            stats.logMapPartitionsStart();

        JavaRDD<String> splitData = split;

        if (collectTrainingStats)
            stats.logRepartitionStart();

        // FIXME: do we still want that?
        splitData = SparkUtils.repartition(splitData, repartition, repartitionStrategy,
                        numObjectsEachWorker(dataSetObjectNumExamples), numWorkers);

        int nPartitions = splitData.partitions().size();
        if (collectTrainingStats && repartition != Repartition.Never)
            stats.logRepartitionEnd();

        FlatMapFunction<Iterator<String>, SharedTrainingResult> function = new SharedFlatMapPaths<>(
                        network != null ? getWorkerInstance(network) : getWorkerInstance(graph));


        JavaRDD<SharedTrainingResult> result = splitData.mapPartitions(function);

        processResults(network, graph, result);

        if (collectTrainingStats)
            stats.logMapPartitionsEnd(nPartitions);
    }

    protected void doIterationPDS(SparkDl4jMultiLayer network, SparkComputationGraph graph,
                    JavaRDD<PortableDataStream> split, int splitNum, int numSplits) {
        log.info("Starting training of split {} of {}. workerMiniBatchSize={}, updatesThreshold={}, Configured for {} workers",
                        splitNum, numSplits, batchSizePerWorker, threshold, numWorkers);

        if (collectTrainingStats)
            stats.logMapPartitionsStart();

        JavaRDD<PortableDataStream> splitData = split;
        if (collectTrainingStats)
            stats.logRepartitionStart();

        splitData = SparkUtils.repartition(splitData, repartition, repartitionStrategy,
                        numObjectsEachWorker(rddDataSetNumExamples), numWorkers);

        int nPartitions = splitData.partitions().size();

        if (collectTrainingStats && repartition != Repartition.Never)
            stats.logRepartitionEnd();

        FlatMapFunction<Iterator<PortableDataStream>, SharedTrainingResult> function =
                        new SharedFlatMapPDS<>(getWorkerInstance(network));


        JavaRDD<SharedTrainingResult> result = splitData.mapPartitions(function);

        processResults(network, graph, result);

        if (collectTrainingStats)
            stats.logMapPartitionsEnd(nPartitions);
    }


    protected void doIterationMultiPDS(SparkComputationGraph graph, JavaRDD<PortableDataStream> split, int splitNum,
                    int numSplits) {
        log.info("Starting training of split {} of {}. workerMiniBatchSize={}, updatesThreshold={}, Configured for {} workers",
                        splitNum, numSplits, batchSizePerWorker, threshold, numWorkers);

        if (collectTrainingStats)
            stats.logMapPartitionsStart();

        JavaRDD<PortableDataStream> splitData = split;
        if (collectTrainingStats)
            stats.logRepartitionStart();

        splitData = SparkUtils.repartition(splitData, repartition, repartitionStrategy,
                        numObjectsEachWorker(rddDataSetNumExamples), numWorkers);

        int nPartitions = splitData.partitions().size();

        if (collectTrainingStats && repartition != Repartition.Never)
            stats.logRepartitionEnd();

        FlatMapFunction<Iterator<PortableDataStream>, SharedTrainingResult> function =
                        new SharedFlatMapMultiPDS<>(getWorkerInstance(graph));


        JavaRDD<SharedTrainingResult> result = splitData.mapPartitions(function);

        processResults(null, graph, result);

        if (collectTrainingStats)
            stats.logMapPartitionsEnd(nPartitions);
    }


    public static class Builder {
        protected double threshold = 1e-3;
        protected double thresholdStep = 1e-5;
        protected double minThreshold = 1e-5;
        protected double stepTrigger = 0.05;
        protected int stepDelay = 50;
        protected int shakeFrequency = 0;
        protected Repartition repartition = Repartition.Always;
        protected RepartitionStrategy repartitionStrategy = RepartitionStrategy.Balanced;
        protected StorageLevel storageLevel = StorageLevel.MEMORY_ONLY_SER();
        protected StorageLevel storageLevelStreams = StorageLevel.MEMORY_ONLY();
        protected VoidConfiguration voidConfiguration;
        protected RDDTrainingApproach rddTrainingApproach = RDDTrainingApproach.Export;
        protected long rngSeed;
        protected String exportDirectory = null;
        protected Integer numWorkers;
        protected boolean collectTrainingStats;
        protected Transport transport;
        protected int batchSize;
        protected long debugLongerIterations = 0L;
        protected int numWorkersPerNode = -1;


        public Builder(int rddDataSetNumExamples) {
            this(1e-3, rddDataSetNumExamples);
        }

        public Builder(@NonNull VoidConfiguration voidConfiguration, int rddDataSetNumExamples) {
            this(voidConfiguration, 1e-3, rddDataSetNumExamples);
        }

        public Builder(double threshold, int rddDataSetNumExamples) {
            this(VoidConfiguration.builder().executionMode(ExecutionMode.MANAGED).forcedRole(NodeRole.SHARD)

                            // we're setting controller to Spark Master, if it's null - that's ok for now.
                            .controllerAddress(System.getenv("SPARK_PUBLIC_DNS")).build(), null, threshold,
                            rddDataSetNumExamples);
        }

        public Builder(@NonNull VoidConfiguration voidConfiguration, double threshold, int rddDataSetNumExamples) {
            this(voidConfiguration, null, threshold, rddDataSetNumExamples);
        }

        /**
         *
         * @param voidConfiguration ParameterServer configuration POJO
         * @param numWorkers
         * @param threshold Update sharing threshold
         * @param rddDataSetNumExamples
         */
        public Builder(@NonNull VoidConfiguration voidConfiguration, Integer numWorkers, double threshold,
                        int rddDataSetNumExamples) {
            this.threshold = threshold;
            this.voidConfiguration = voidConfiguration;

            // we're enforcing managed mode in all cases here
            this.voidConfiguration.setExecutionMode(ExecutionMode.MANAGED);
        }

        /**
         * Enable/disable collection of training statistics
         * @param reallyConnect
         * @return
         */
        public Builder collectTrainingStats(boolean reallyConnect) {
            this.collectTrainingStats = reallyConnect;
            return this;
        }

        /**
         * This parameter defines when repartition is applied (if applied)
         * @param repartition
         * @return
         */
        public Builder repartitionData(Repartition repartition) {
            this.repartition = repartition;
            return this;
        }

        /**
         * Used in conjunction with {@link #repartitionData(Repartition)} (which defines <i>when</i> repartitioning should be
         * conducted), repartitionStrategy defines <i>how</i> the repartitioning should be done. See {@link RepartitionStrategy}
         * for details
         *
         * @param repartitionStrategy Repartitioning strategy to use
         */
        public Builder repartitionStrategy(RepartitionStrategy repartitionStrategy) {
            this.repartitionStrategy = repartitionStrategy;
            return this;
        }

        /**
         * Set the storage level for {@code RDD<DataSet>}s.<br>
         * Default: StorageLevel.MEMORY_ONLY_SER() - i.e., store in memory, in serialized form<br>
         * To use no RDD persistence, use {@code null}<br>
         * <p>
         * <b>Note</b>: Spark's StorageLevel.MEMORY_ONLY() and StorageLevel.MEMORY_AND_DISK() can be problematic when
         * it comes to off-heap data (which DL4J/ND4J uses extensively). Spark does not account for off-heap memory
         * when deciding if/when to drop blocks to ensure enough free memory; consequently, for DataSet RDDs that are
         * larger than the total amount of (off-heap) memory, this can lead to OOM issues. Put another way: Spark counts
         * the on-heap size of DataSet and INDArray objects only (which is negligible) resulting in a significant
         * underestimate of the true DataSet object sizes. More DataSets are thus kept in memory than we can really afford.
         *
         * @param storageLevel Storage level to use for DataSet RDDs
         */
        public Builder storageLevel(StorageLevel storageLevel) {
            this.storageLevel = storageLevel;
            return this;
        }

        /**
         * The approach to use when training on a {@code RDD<DataSet>} or {@code RDD<MultiDataSet>}.
         * Default: {@link RDDTrainingApproach#Export}, which exports data to a temporary directory first
         *
         * @param rddTrainingApproach Training approach to use when training from a {@code RDD<DataSet>} or {@code RDD<MultiDataSet>}
         */
        public Builder rddTrainingApproach(RDDTrainingApproach rddTrainingApproach) {
            this.rddTrainingApproach = rddTrainingApproach;
            return this;
        }

        /**
         * When {@link #rddTrainingApproach(RDDTrainingApproach)} is set to {@link RDDTrainingApproach#Export} (as it is by default)
         * the data is exported to a temporary directory first.
         * <p>
         * Default: null. -> use {hadoop.tmp.dir}/dl4j/. In this case, data is exported to {hadoop.tmp.dir}/dl4j/SOME_UNIQUE_ID/<br>
         * If you specify a directory, the directory {exportDirectory}/SOME_UNIQUE_ID/ will be used instead.
         *
         * @param exportDirectory Base directory to export data
         */
        public Builder exportDirectory(String exportDirectory) {
            this.exportDirectory = exportDirectory;
            return this;
        }

        /**
         * Random number generator seed, used mainly for enforcing repeatable splitting on RDDs
         * Default: no seed set (i.e., random seed)
         *
         * @param rngSeed RNG seed
         * @return
         */
        public Builder rngSeed(long rngSeed) {
            this.rngSeed = rngSeed;
            return this;
        }

        /**
         * Threshold for updates encoding
         *
         * Default value: 1e-3
         * @param threshold
         * @return
         */
        public Builder updatesThreshold(double threshold) {
            this.threshold = threshold;
            return this;
        }

        /**
         * Once update with given threshold become too sparse, threshold will be decreased by thresholdStep, but not below minimum threshold
         *
         * Default value: 1e-5
         * @param threshold
         * @return
         */
        public Builder minUpdatesThreshold(double threshold) {
            this.minThreshold = threshold;
            return this;
        }

        /**
         * Step size for threshold decay
         *
         * Default value: 1e-5
         * @param step
         * @return
         */
        public Builder thresholdStep(double step) {
            if (step < 0.0)
                throw new DL4JInvalidConfigException("shakeFrequency should be non-negative value");

            this.thresholdStep = step;
            return this;
        }

        /**
         * Target sparsity/dense level, when threshold step will happen. i.e. 5 value = 5% of original updates size.
         *
         * Default value: 0.05
         * @param step
         * @return
         */
        public Builder stepTrigger(double step) {
            if (step < 0.0 || step > 100.0)
                throw new DL4JInvalidConfigException("stepTrigger value should be in range of 0..100");

            return this;
        }

        /**
         * Wait at least X iterations between applying threshold decay
         *
         * Default value: 50
         * @param step
         * @return
         */
        public Builder stepDelay(int step) {
            this.stepDelay = step;
            return this;
        }

        /**
         * During NN training, each X iterations, executors will send encoded dense updates with lower threshold.
         * Please note: If you'll set this value too low (i.e. 1) - it might lead to worse performance
         *
         * Default value: 0 (disabled)
         * @param frequency
         * @return
         */
        public Builder shakeFrequency(int frequency) {
            if (frequency < 0)
                throw new DL4JInvalidConfigException("shakeFrequency should be non-negative value");

            if (frequency == 1)
                log.warn("shakeFrequency of 1 means that all updates will be sparse, and might lead to worse performance");

            this.shakeFrequency = frequency;
            return this;
        }

        /**
         * Batch size value,  used for repartition purposes
         *
         * @param batchSize
         * @return
         */
        public Builder batchSizePerWorker(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        /**
         * This method allows to configure number of trainer threads per cluster node.
         *
         *
         * Default value: -1, which defines automated number of workers selection, based on hardware present in system
         *
         * @param numWorkers
         * @return
         */
        public Builder workersPerNode(int numWorkers) {
            if (numWorkers < 1)
                numWorkers = -1;

            this.numWorkersPerNode = numWorkers;
            return this;
        }

        /**
         * This method allows you to artificially extend iteration time using Thread.sleep() for a given time.
         *
         * PLEASE NOTE: Never use that option in production environment. It's suited for debugging purposes only.
         *
         * @param timeMs
         * @return
         */
        @Deprecated
        public Builder debugLongerIterations(long timeMs) {
            if (timeMs < 0)
                timeMs = 0L;
            this.debugLongerIterations = timeMs;
            return this;
        }

        /**
         * Optional method: Transport implementation to be used as TransportType.CUSTOM for VoidParameterAveraging method
         *
         * @param transport
         * @return
         */
        public Builder transport(Transport transport) {
            this.transport = transport;
            return this;
        }

        public SharedTrainingMaster build() {
            SharedTrainingMaster master = new SharedTrainingMaster(voidConfiguration, numWorkers, rddTrainingApproach,
                            storageLevel, collectTrainingStats, repartitionStrategy, repartition, threshold,
                            minThreshold, thresholdStep, stepTrigger, stepDelay, shakeFrequency, batchSize,
                            debugLongerIterations, numWorkersPerNode);
            if (transport != null)
                master.transport = this.transport;

            return master;
        }
    }
}
