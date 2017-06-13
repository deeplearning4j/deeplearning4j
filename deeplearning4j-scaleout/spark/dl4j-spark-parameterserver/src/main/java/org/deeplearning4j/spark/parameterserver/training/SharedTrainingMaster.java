package org.deeplearning4j.spark.parameterserver.training;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaRDDLike;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.input.PortableDataStream;
import org.apache.spark.storage.StorageLevel;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.spark.api.*;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.api.worker.ExecuteWorkerFlatMap;
import org.deeplearning4j.spark.api.worker.NetBroadcastTuple;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingResult;
import org.deeplearning4j.spark.impl.paramavg.stats.ParameterAveragingTrainingMasterStats;
import org.deeplearning4j.spark.parameterserver.conf.SharedTrainingConfiguration;
import org.deeplearning4j.spark.parameterserver.functions.SharedFlatMapDataSet;
import org.deeplearning4j.spark.parameterserver.networking.SilentTrainingDriver;
import org.deeplearning4j.spark.util.SparkUtils;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.parameterserver.distributed.VoidParameterServer;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.enums.ExecutionMode;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.enums.TransportType;
import org.nd4j.parameterserver.distributed.transport.MulticastTransport;
import org.nd4j.parameterserver.distributed.transport.RoutedTransport;
import org.nd4j.parameterserver.distributed.transport.Transport;
import org.nd4j.shade.jackson.annotation.JsonAutoDetect;
import org.nd4j.shade.jackson.annotation.PropertyAccessor;
import org.nd4j.shade.jackson.core.JsonFactory;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.MapperFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;
import org.nd4j.shade.jackson.dataformat.yaml.YAMLFactory;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class SharedTrainingMaster implements TrainingMaster<SharedTrainingResult, SharedTrainingWorker> {
    protected List<TrainingHook> trainingHooks;
    protected VoidConfiguration voidConfiguration;

    protected Integer numWorkers;
    protected RDDTrainingApproach rddTrainingApproach;
    protected StorageLevel storageLevel;

    protected boolean collectTrainingStats;
    protected int rddDataSetNumExamples;
    protected int batchSizePerWorker;

    // TODO: this option should be abstracted, if we decide to generalize this trainingmaster
    protected double threshold;

    protected Repartition repartition;
    protected RepartitionStrategy repartitionStrategy;

    protected ParameterAveragingTrainingMasterStats.ParameterAveragingTrainingMasterStatsHelper stats;

    protected Random rng;

    protected AtomicBoolean isFirstRun;


    // safe to ignore
    private static ObjectMapper jsonMapper;
    private static ObjectMapper yamlMapper;


    // better ignore
    protected transient Broadcast<NetBroadcastTuple> broadcastModel;
    protected transient Broadcast<SharedTrainingConfiguration> broadcastConfiguration;
    protected transient Transport transport;

    protected SharedTrainingMaster() {
        // just a stub for ser/de
    }

    public SharedTrainingMaster(@NonNull VoidConfiguration voidConfiguration, Integer numWorkers, RDDTrainingApproach rddTrainingApproach, StorageLevel storageLevel, boolean collectTrainingStats, RepartitionStrategy repartitionStrategy, Repartition repartition, double threshold) {
        this.voidConfiguration = voidConfiguration;
        this.repartition = repartition;
        this.repartitionStrategy = repartitionStrategy;
        this.numWorkers = numWorkers;
        this.rddTrainingApproach = rddTrainingApproach;
        this.repartitionStrategy = repartitionStrategy;
        this.repartition = repartition;
        this.storageLevel = storageLevel;
        this.collectTrainingStats = collectTrainingStats;
        this.threshold = threshold;
        this.isFirstRun = new AtomicBoolean(false);

        if (collectTrainingStats)
            stats = new ParameterAveragingTrainingMasterStats.ParameterAveragingTrainingMasterStatsHelper();
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

        SharedTrainingConfiguration configuration = SharedTrainingConfiguration.builder()
                .threshold(threshold)
                .voidConfiguration(voidConfiguration)
                .build();

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

        SharedTrainingConfiguration configuration = SharedTrainingConfiguration.builder()
                .threshold(threshold)
                .voidConfiguration(voidConfiguration)

                .build();

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
        return 0;//batchSizePerWorker * averagingFrequency / numExamplesEachRddObject;
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

    protected <T, Repr extends JavaRDDLike<T, Repr>> long getTotalDataSetObjectCount(JavaRDDLike<T, Repr> trainingData) {
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
        // first of all, we're instantiating ParameterServer shard here
        Transport transport = voidConfiguration.getTransportType() == TransportType.ROUTED ? new RoutedTransport() : voidConfiguration.getTransportType() == TransportType.BROADCAST ? new MulticastTransport() : this.transport;

        if (transport == null)
            throw new DL4JInvalidConfigException("No Transport implementation was defined for this training session!");

        // TODO: Alex, any better ideas here? Plus, if we're not on SGD updater, we might want to eventually dump updater to Master?
        if (!network.getNetwork().isInitCalled())
            network.getNetwork().init();

        // this instance will be SilentWorker - it'll accept and apply messages, but won't contribute to training. And we init it only once
        if (isFirstRun.compareAndSet(false, true))
            VoidParameterServer.getInstance().init(voidConfiguration, transport, new SilentTrainingDriver(network.getNetwork().params(), network.getNetwork().getOptimizer().getStepFunction()));

        if (numWorkers == null)
            numWorkers = network.getSparkContext().defaultParallelism();

        // at this moment we have coordinator server up (master works as coordinator)
        if (rddTrainingApproach == RDDTrainingApproach.Direct) {
            executeTrainingDirect(network, trainingData);
        } else if (rddTrainingApproach == RDDTrainingApproach.Export){
            //Export data if required (or, use cached export)
        } else
            throw new DL4JInvalidConfigException("Unknown RDDtrainingApproach [" + rddTrainingApproach + "] was specified!");
    }

    @Override
    public void executeTraining(SparkDl4jMultiLayer network, JavaPairRDD<String, PortableDataStream> trainingData) {

    }

    @Override
    public void executeTrainingPaths(SparkDl4jMultiLayer network, JavaRDD<String> trainingDataPaths) {

    }

    @Override
    public void executeTraining(SparkComputationGraph graph, JavaRDD<DataSet> trainingData) {

    }

    @Override
    public void executeTraining(SparkComputationGraph network, JavaPairRDD<String, PortableDataStream> trainingData) {

    }

    @Override
    public void executeTrainingPaths(SparkComputationGraph network, JavaRDD<String> trainingDataPaths) {

    }

    @Override
    public void executeTrainingPathsMDS(SparkComputationGraph network, JavaRDD<String> trainingMultiDataSetPaths) {

    }

    @Override
    public void executeTrainingMDS(SparkComputationGraph graph, JavaRDD<MultiDataSet> trainingData) {

    }

    @Override
    public void executeTrainingMDS(SparkComputationGraph network, JavaPairRDD<String, PortableDataStream> trainingData) {

    }

    @Override
    public void setCollectTrainingStats(boolean collectTrainingStats) {
        // simple enabler/disabler
    }

    @Override
    public boolean getIsCollectTrainingStats() {
        return false;
    }

    @Override
    public SparkTrainingStats getTrainingStats() {
        return null;
    }

    @Override
    public void setListeners(Collection<IterationListener> listeners) {
        // optional stuff actually
    }

    @Override
    public void setListeners(StatsStorageRouter router, Collection<IterationListener> listeners) {
        // optional stuff actually
    }

    @Override
    public boolean deleteTempFiles(JavaSparkContext sc) {
        return false;
    }

    @Override
    public boolean deleteTempFiles(SparkContext sc) {
        return false;
    }

    private static synchronized ObjectMapper getJsonMapper() {
        if (jsonMapper == null) {
            jsonMapper = getNewMapper(new JsonFactory());
        }
        return jsonMapper;
    }

    private static synchronized ObjectMapper getYamlMapper() {
        if (yamlMapper == null) {
            yamlMapper = getNewMapper(new YAMLFactory());
        }
        return yamlMapper;
    }

    private static ObjectMapper getNewMapper(JsonFactory jsonFactory) {
        ObjectMapper om = new ObjectMapper(jsonFactory);
        om.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        om.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        om.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, true);
        om.enable(SerializationFeature.INDENT_OUTPUT);
        om.setVisibility(PropertyAccessor.ALL, JsonAutoDetect.Visibility.NONE);
        om.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
        return om;
    }

    protected void doIteration(SparkDl4jMultiLayer network, JavaRDD<DataSet> split, int splitNum, int numSplits) {
        // TODO: here we'll basically call for mapPartition

        log.info("Starting training of split {} of {}. workerMiniBatchSize={}, averagingFreq={}, Configured for {} workers",
                splitNum, numSplits, batchSizePerWorker, 0, numWorkers);

        if (collectTrainingStats)
            stats.logMapPartitionsStart();

        JavaRDD<DataSet> splitData = split;

        if (collectTrainingStats)
            stats.logRepartitionStart();

       // splitData = SparkUtils.repartition(splitData, repartition, repartitionStrategy, numObjectsEachWorker(rddDataSetNumExamples), numWorkers);
        int nPartitions = splitData.partitions().size();

        if (collectTrainingStats && repartition != Repartition.Never)
            stats.logRepartitionEnd();


        FlatMapFunction<Iterator<DataSet>, SharedTrainingResult> function = new SharedFlatMapDataSet<>(getWorkerInstance(network));

        JavaRDD<SharedTrainingResult> result = splitData.mapPartitions(function);

        // meh, just to invoke previous function
        long cnt = result.count();

        log.info("Results count: {}", cnt);

        // TODO: implement something here
//        processResults(network, null, result, splitNum, numSplits);

        if (collectTrainingStats)
            stats.logMapPartitionsEnd(nPartitions);
    }


    public static class Builder {
        protected double threshold = 1e-3;
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

        public Builder(int rddDataSetNumExamples) {
            this(1e-3, rddDataSetNumExamples);
        }

        public Builder(@NonNull VoidConfiguration voidConfiguration, int rddDataSetNumExamples) {
            this(voidConfiguration,1e-3, rddDataSetNumExamples);
        }

        public Builder(double threshold, int rddDataSetNumExamples) {
            this(VoidConfiguration.builder()
                    .executionMode(ExecutionMode.MANAGED)
                    .forcedRole(NodeRole.SHARD)

                    // we're setting controller to Spark Master
                    .controllerAddress(System.getenv("SPARK_PUBLIC_DNS"))
                    .build(), null, threshold, rddDataSetNumExamples);
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
        public Builder(@NonNull VoidConfiguration voidConfiguration, Integer numWorkers, double threshold, int rddDataSetNumExamples) {
            this.threshold = threshold;
            this.voidConfiguration = voidConfiguration;
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
         *
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
            SharedTrainingMaster master = new SharedTrainingMaster(voidConfiguration, numWorkers, rddTrainingApproach, storageLevel, true, repartitionStrategy, repartition, threshold);
            if (transport != null)
                master.transport = this.transport;

            return master;
        }
    }
}
