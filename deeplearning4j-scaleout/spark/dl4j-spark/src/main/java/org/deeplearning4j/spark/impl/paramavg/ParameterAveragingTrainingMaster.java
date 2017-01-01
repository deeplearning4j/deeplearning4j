package org.deeplearning4j.spark.impl.paramavg;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.api.storage.*;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.spark.impl.listeners.VanillaStatsStorageRouterProvider;
import org.nd4j.shade.jackson.annotation.JsonAutoDetect;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.PropertyAccessor;
import org.nd4j.shade.jackson.core.JsonFactory;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.MapperFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;
import org.nd4j.shade.jackson.dataformat.yaml.YAMLFactory;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.input.PortableDataStream;
import org.apache.spark.storage.StorageLevel;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.spark.api.*;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.api.worker.*;
import org.deeplearning4j.spark.data.BatchAndExportDataSetsFunction;
import org.deeplearning4j.spark.data.BatchAndExportMultiDataSetsFunction;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.impl.graph.dataset.DataSetToMultiDataSetFn;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.aggregator.ParameterAveragingAggregationTuple;
import org.deeplearning4j.spark.impl.paramavg.aggregator.ParameterAveragingElementAddFunction;
import org.deeplearning4j.spark.impl.paramavg.aggregator.ParameterAveragingElementCombineFunction;
import org.deeplearning4j.spark.impl.paramavg.stats.ParameterAveragingTrainingMasterStats;
import org.deeplearning4j.spark.util.SparkUtils;
import org.deeplearning4j.util.UIDProvider;
import org.deeplearning4j.spark.util.serde.StorageLevelDeserializer;
import org.deeplearning4j.spark.util.serde.StorageLevelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.*;

/**
 * ParameterAveragingTrainingMaster: A {@link TrainingMaster} implementation for training networks on Spark.
 * This is standard parameter averaging with a configurable averaging period.
 *
 * @author Alex Black
 */
@Data
@JsonIgnoreProperties({"stats", "listeners", "iterationCount", "rng", "lastExportedRDDId", "lastRDDExportPath", "trainingMasterUID"})
@EqualsAndHashCode(exclude = {"stats", "listeners", "iterationCount", "rng", "lastExportedRDDId", "lastRDDExportPath", "trainingMasterUID"})
@Slf4j
public class ParameterAveragingTrainingMaster implements TrainingMaster<ParameterAveragingTrainingResult, ParameterAveragingTrainingWorker> {

    private static final int COALESCE_THRESHOLD = 3;
    private static ObjectMapper jsonMapper;
    private static ObjectMapper yamlMapper;

    private boolean saveUpdater;
    private Integer numWorkers;
    private int rddDataSetNumExamples;
    private int batchSizePerWorker;
    private int averagingFrequency;
    private int prefetchNumBatches;
    private boolean collectTrainingStats;
    private ParameterAveragingTrainingMasterStats.ParameterAveragingTrainingMasterStatsHelper stats;
    private int iterationCount = 0;
    private Repartition repartition;
    private RepartitionStrategy repartitionStrategy;
    @JsonSerialize(using = StorageLevelSerializer.class)
    @JsonDeserialize(using = StorageLevelDeserializer.class)
    private StorageLevel storageLevel;
    @JsonSerialize(using = StorageLevelSerializer.class)
    @JsonDeserialize(using = StorageLevelDeserializer.class)
    private StorageLevel storageLevelStreams = StorageLevel.MEMORY_ONLY();
    private RDDTrainingApproach rddTrainingApproach = RDDTrainingApproach.Export;
    private String exportDirectory = null;
    private Random rng;

    private Collection<TrainingHook> trainingHookList;
    private int lastExportedRDDId = Integer.MIN_VALUE;
    private String lastRDDExportPath;
    private final String trainingMasterUID;

    //Listeners etc
    private Collection<IterationListener> listeners;
    private StatsStorageRouter statsStorage;

    private ParameterAveragingTrainingMaster() {
        // no-arg constructor for Jackson

        String jvmuid = UIDProvider.getJVMUID();
        this.trainingMasterUID = System.currentTimeMillis() + "_" + (jvmuid.length() <= 8 ? jvmuid : jvmuid.substring(0, 8));
        this.rng = new Random();
    }

    private ParameterAveragingTrainingMaster(Builder builder) {
        this.saveUpdater = builder.saveUpdater;
        this.numWorkers = builder.numWorkers;
        this.rddDataSetNumExamples = builder.rddDataSetNumExamples;
        this.batchSizePerWorker = builder.batchSizePerWorker;
        this.averagingFrequency = builder.averagingFrequency;
        this.prefetchNumBatches = builder.prefetchNumBatches;
        this.repartition = builder.repartition;
        this.repartitionStrategy = builder.repartitionStrategy;
        this.storageLevel = builder.storageLevel;
        this.storageLevelStreams = builder.storageLevelStreams;
        this.rddTrainingApproach = builder.rddTrainingApproach;
        this.exportDirectory = builder.exportDirectory;
        this.trainingHookList = builder.trainingHooks;

        if (builder.rngSeed == null) {
            this.rng = new Random();
        } else {
            this.rng = new Random(builder.rngSeed);
        }

        String jvmuid = UIDProvider.getJVMUID();
        this.trainingMasterUID = System.currentTimeMillis() + "_" + (jvmuid.length() <= 8 ? jvmuid : jvmuid.substring(0, 8));
    }

    public ParameterAveragingTrainingMaster(boolean saveUpdater, Integer numWorkers, int rddDataSetNumExamples, int batchSizePerWorker,
                                            int averagingFrequency, int prefetchNumBatches) {
        this(saveUpdater, numWorkers, rddDataSetNumExamples, batchSizePerWorker, averagingFrequency, prefetchNumBatches,
                Repartition.Always, RepartitionStrategy.Balanced, false);
    }

    /**
     * @param saveUpdater           If true: save (and average) the updater state when doing parameter averaging
     * @param numWorkers            Number of workers (executors * threads per executor) for the cluster
     * @param rddDataSetNumExamples Number of examples in each DataSet object in the {@code RDD<DataSet>}
     * @param batchSizePerWorker    Number of examples to use per worker per fit
     * @param averagingFrequency    Frequency (in number of minibatches) with which to average parameters
     * @param prefetchNumBatches    Number of batches to asynchronously prefetch (0: disable)
     * @param collectTrainingStats  If true: collect training statistics for debugging/optimization purposes
     */
    public ParameterAveragingTrainingMaster(boolean saveUpdater, Integer numWorkers, int rddDataSetNumExamples, int batchSizePerWorker,
                                            int averagingFrequency, int prefetchNumBatches, Repartition repartition,
                                            RepartitionStrategy repartitionStrategy, boolean collectTrainingStats) {
        this(saveUpdater, numWorkers, rddDataSetNumExamples, batchSizePerWorker, averagingFrequency, prefetchNumBatches,
                repartition, repartitionStrategy, StorageLevel.MEMORY_ONLY_SER(), collectTrainingStats);
    }

    public ParameterAveragingTrainingMaster(boolean saveUpdater, Integer numWorkers, int rddDataSetNumExamples, int batchSizePerWorker,
                                            int averagingFrequency, int prefetchNumBatches, Repartition repartition,
                                            RepartitionStrategy repartitionStrategy, StorageLevel storageLevel, boolean collectTrainingStats) {
        if (numWorkers <= 0)
            throw new IllegalArgumentException("Invalid number of workers: " + numWorkers + " (must be >= 1)");
        if (rddDataSetNumExamples <= 0)
            throw new IllegalArgumentException("Invalid rdd data set size: " + rddDataSetNumExamples + " (must be >= 1)");

        this.saveUpdater = saveUpdater;
        this.numWorkers = numWorkers;
        this.rddDataSetNumExamples = rddDataSetNumExamples;
        this.batchSizePerWorker = batchSizePerWorker;
        this.averagingFrequency = averagingFrequency;
        this.prefetchNumBatches = prefetchNumBatches;
        this.collectTrainingStats = collectTrainingStats;
        this.repartition = repartition;
        this.repartitionStrategy = repartitionStrategy;
        this.storageLevel = storageLevel;
        if (collectTrainingStats)
            stats = new ParameterAveragingTrainingMasterStats.ParameterAveragingTrainingMasterStatsHelper();

        String jvmuid = UIDProvider.getJVMUID();
        this.trainingMasterUID = System.currentTimeMillis() + "_" + (jvmuid.length() <= 8 ? jvmuid : jvmuid.substring(0, 8));
        this.rng = new Random();
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

    /**
     * Remove a training hook from the worker
     *
     * @param trainingHook the training hook to remove
     */
    @Override
    public void removeHook(TrainingHook trainingHook) {
        if(trainingHookList == null) return;
        trainingHookList.remove(trainingHook);
    }

    /**
     * Add a hook for the master for pre and post training
     *
     * @param trainingHook the training hook to add
     */
    @Override
    public void addHook(TrainingHook trainingHook) {
        if(trainingHookList == null){
            trainingHookList = new ArrayList<>();
        }
        trainingHookList.add(trainingHook);
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
     * Create a ParameterAveragingTrainingMaster instance by deserializing a JSON string that has been serialized with
     * {@link #toJson()}
     *
     * @param jsonStr ParameterAveragingTrainingMaster configuration serialized as JSON
     */
    public static ParameterAveragingTrainingMaster fromJson(String jsonStr) {
        ObjectMapper om = getJsonMapper();
        try {
            return om.readValue(jsonStr, ParameterAveragingTrainingMaster.class);
        } catch (IOException e) {
            throw new RuntimeException("Could not parse JSON", e);
        }
    }

    /**
     * Create a ParameterAveragingTrainingMaster instance by deserializing a YAML string that has been serialized with
     * {@link #toYaml()}
     *
     * @param yamlStr ParameterAveragingTrainingMaster configuration serialized as YAML
     */
    public static ParameterAveragingTrainingMaster fromYaml(String yamlStr) {
        ObjectMapper om = getYamlMapper();
        try {
            return om.readValue(yamlStr, ParameterAveragingTrainingMaster.class);
        } catch (IOException e) {
            throw new RuntimeException("Could not parse YAML", e);
        }
    }


    @Override
    public ParameterAveragingTrainingWorker getWorkerInstance(SparkDl4jMultiLayer network) {
        NetBroadcastTuple tuple = new NetBroadcastTuple(network.getNetwork().getLayerWiseConfigurations(),
                network.getNetwork().params(),
                network.getNetwork().getUpdater().getStateViewArray());

        if (collectTrainingStats) stats.logBroadcastStart();
        Broadcast<NetBroadcastTuple> broadcast = network.getSparkContext().broadcast(tuple);
        if (collectTrainingStats) stats.logBroadcastEnd();

        WorkerConfiguration configuration = new WorkerConfiguration(false, rddDataSetNumExamples, batchSizePerWorker, averagingFrequency, prefetchNumBatches, collectTrainingStats);
        return new ParameterAveragingTrainingWorker(broadcast, saveUpdater, configuration, trainingHookList, listeners, getRouterProvider());
    }

    @Override
    public ParameterAveragingTrainingWorker getWorkerInstance(SparkComputationGraph graph) {
        NetBroadcastTuple tuple = new NetBroadcastTuple(graph.getNetwork().getConfiguration(),
                graph.getNetwork().params(),
                graph.getNetwork().getUpdater().getStateViewArray());

        if (collectTrainingStats) stats.logBroadcastStart();
        Broadcast<NetBroadcastTuple> broadcast = graph.getSparkContext().broadcast(tuple);
        if (collectTrainingStats) stats.logBroadcastEnd();

        WorkerConfiguration configuration = new WorkerConfiguration(true, rddDataSetNumExamples, batchSizePerWorker, averagingFrequency, prefetchNumBatches, collectTrainingStats);
        return new ParameterAveragingTrainingWorker(broadcast, saveUpdater, configuration, trainingHookList, listeners, getRouterProvider());
    }

    private int numObjectsEachWorker(int numExamplesEachRddObject) {
        return batchSizePerWorker * averagingFrequency / numExamplesEachRddObject;
    }

    private int getNumDataSetObjectsPerSplit(int numExamplesEachRddObject) {
        int dataSetObjectsPerSplit;
        if (numExamplesEachRddObject == 1) {
            dataSetObjectsPerSplit = numWorkers * batchSizePerWorker * averagingFrequency;
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

    @Override
    public void executeTraining(SparkDl4jMultiLayer network, JavaRDD<DataSet> trainingData) {
        if (numWorkers == null) numWorkers = network.getSparkContext().defaultParallelism();

        if (rddTrainingApproach == RDDTrainingApproach.Direct) {
            executeTrainingDirect(network, trainingData);
        } else {
            //Export data if required (or, use cached export)
            JavaRDD<String> paths = exportIfRequired(network.getSparkContext(), trainingData);
            executeTrainingPathsHelper(network, paths, batchSizePerWorker);     //Originally (pre-export): had rddDataSetNumExamples per DataSet. Now we have batchSizePerWorker per exported DataSet
        }
    }

    private void executeTrainingDirect(SparkDl4jMultiLayer network, JavaRDD<DataSet> trainingData) {
        if (collectTrainingStats) stats.logFitStart();
        //For "vanilla" parameter averaging training, we need to split the full data set into batches of size N, such that we can process the specified
        // number of minibatches between averagings
        //But to do that, wee need to know: (a) the number of examples, and (b) the number of workers
        if (storageLevel != null) trainingData.persist(storageLevel);

        if (collectTrainingStats) stats.logCountStart();
        long totalDataSetObjectCount = trainingData.count();
        if (collectTrainingStats) stats.logCountEnd();
        int dataSetObjectsPerSplit = getNumDataSetObjectsPerSplit(rddDataSetNumExamples);

        if (collectTrainingStats) stats.logSplitStart();
        JavaRDD<DataSet>[] splits = SparkUtils.balancedRandomSplit((int) totalDataSetObjectCount, dataSetObjectsPerSplit, trainingData, rng.nextLong());
        if (collectTrainingStats) stats.logSplitEnd();

        int splitNum = 1;
        for (JavaRDD<DataSet> split : splits) {
            doIteration(network, split, splitNum++, splits.length);
        }

        if (collectTrainingStats) stats.logFitEnd((int) totalDataSetObjectCount);
    }

    @Override
    @Deprecated
    public void executeTraining(SparkDl4jMultiLayer network, JavaPairRDD<String, PortableDataStream> trainingData) {
        if (numWorkers == null) numWorkers = network.getSparkContext().defaultParallelism();

        if (collectTrainingStats) stats.logFitStart();

        int origNumPartitions = trainingData.partitions().size();
        if (origNumPartitions >= COALESCE_THRESHOLD * numWorkers) {
            log.info("Coalescing PortableDataStreams from {} to {} partitions", origNumPartitions, numWorkers);
            trainingData = trainingData.coalesce(numWorkers);
        }
        if (storageLevelStreams != null) trainingData.persist(storageLevelStreams);

        if (collectTrainingStats) stats.logCountStart();
        long totalDataSetObjectCount = trainingData.count();
        if (collectTrainingStats) stats.logCountEnd();
        int dataSetObjectsPerSplit = getNumDataSetObjectsPerSplit(rddDataSetNumExamples);
        if (collectTrainingStats) stats.logSplitStart();
        JavaPairRDD<String, PortableDataStream>[] splits = SparkUtils.balancedRandomSplit((int) totalDataSetObjectCount, dataSetObjectsPerSplit, trainingData, rng.nextLong());
        if (collectTrainingStats) stats.logSplitEnd();

        int splitNum = 1;
        for (JavaPairRDD<String, PortableDataStream> split : splits) {
            JavaRDD<PortableDataStream> streams = split.values();
            doIterationPDS(network, null, streams, splitNum++, splits.length);
        }

        if (collectTrainingStats) stats.logFitEnd((int) totalDataSetObjectCount);
    }

    @Override
    public void executeTrainingPaths(SparkDl4jMultiLayer network, JavaRDD<String> trainingDataPaths) {
        executeTrainingPathsHelper(network, trainingDataPaths, rddDataSetNumExamples);
    }

    private void executeTrainingPathsHelper(SparkDl4jMultiLayer network, JavaRDD<String> trainingDataPaths, int dataSetObjectsNumExamples) {
        if (numWorkers == null) numWorkers = network.getSparkContext().defaultParallelism();

        if (collectTrainingStats) stats.logFitStart();
        if (storageLevelStreams != null) trainingDataPaths.persist(storageLevelStreams);

        if (collectTrainingStats) stats.logCountStart();
        long totalDataSetObjectCount = trainingDataPaths.count();
        if (collectTrainingStats) stats.logCountEnd();

        int dataSetObjectsPerSplit = getNumDataSetObjectsPerSplit(dataSetObjectsNumExamples);
        if (collectTrainingStats) stats.logSplitStart();
        JavaRDD<String>[] splits = SparkUtils.balancedRandomSplit((int) totalDataSetObjectCount, dataSetObjectsPerSplit, trainingDataPaths, rng.nextLong());
        if (collectTrainingStats) stats.logSplitEnd();


        int splitNum = 1;
        for (JavaRDD<String> split : splits) {
            doIterationPaths(network, null, split, splitNum++, splits.length, dataSetObjectsNumExamples);
        }

        if (collectTrainingStats) stats.logFitEnd((int) totalDataSetObjectCount);
    }

    @Override
    public void executeTraining(SparkComputationGraph graph, JavaRDD<DataSet> trainingData) {
        if (numWorkers == null) numWorkers = graph.getSparkContext().defaultParallelism();

        JavaRDD<MultiDataSet> mdsTrainingData = trainingData.map(new DataSetToMultiDataSetFn());

        executeTrainingMDS(graph, mdsTrainingData);
    }

    @Override
    public void executeTrainingMDS(SparkComputationGraph graph, JavaRDD<MultiDataSet> trainingData) {
        if (numWorkers == null) numWorkers = graph.getSparkContext().defaultParallelism();

        if (rddTrainingApproach == RDDTrainingApproach.Direct) {
            executeTrainingDirect(graph, trainingData);
        } else {
            //Export data if required (or, use cached export)
            JavaRDD<String> paths = exportIfRequiredMDS(graph.getSparkContext(), trainingData);
            executeTrainingPathsMDSHelper(graph, paths, batchSizePerWorker);
        }
    }

    private void executeTrainingDirect(SparkComputationGraph graph, JavaRDD<MultiDataSet> trainingData) {
        if (collectTrainingStats) stats.logFitStart();
        //For "vanilla" parameter averaging training, we need to split the full data set into batches of size N, such that we can process the specified
        // number of minibatches between averaging
        //But to do that, we need to know: (a) the number of examples, and (b) the number of workers
        if (storageLevel != null) trainingData.persist(storageLevel);

        if (collectTrainingStats) stats.logCountStart();
        long totalDataSetObjectCount = trainingData.count();
        if (collectTrainingStats) stats.logCountEnd();
        int dataSetObjectsPerSplit = getNumDataSetObjectsPerSplit(rddDataSetNumExamples);

        if (collectTrainingStats) stats.logSplitStart();
        JavaRDD<MultiDataSet>[] splits = SparkUtils.balancedRandomSplit((int) totalDataSetObjectCount, dataSetObjectsPerSplit, trainingData, rng.nextLong());
        if (collectTrainingStats) stats.logSplitEnd();

        int splitNum = 1;
        for (JavaRDD<MultiDataSet> split : splits) {
            doIteration(graph, split, splitNum++, splits.length);
        }

        if (collectTrainingStats) stats.logFitEnd((int) totalDataSetObjectCount);
    }

    @Override
    public void executeTraining(SparkComputationGraph graph, JavaPairRDD<String, PortableDataStream> trainingData) {
        if (numWorkers == null) numWorkers = graph.getSparkContext().defaultParallelism();

        if (collectTrainingStats) stats.logFitStart();
        //For "vanilla" parameter averaging training, we need to split the full data set into batches of size N, such that we can process the specified
        // number of minibatches between averagings
        //But to do that, we need to know: (a) the number of examples, and (b) the number of workers

        int origNumPartitions = trainingData.partitions().size();
        if (origNumPartitions >= COALESCE_THRESHOLD * numWorkers) {
            log.info("Coalesing streams from {} to {} partitions", origNumPartitions, numWorkers);
            trainingData = trainingData.coalesce(numWorkers);
        }
        if (storageLevelStreams != null) trainingData.persist(storageLevelStreams);

        if (collectTrainingStats) stats.logCountStart();
        long totalDataSetObjectCount = trainingData.count();
        if (collectTrainingStats) stats.logCountEnd();
        int dataSetObjectsPerSplit = getNumDataSetObjectsPerSplit(rddDataSetNumExamples);

        if (collectTrainingStats) stats.logSplitStart();
        JavaPairRDD<String, PortableDataStream>[] splits = SparkUtils.balancedRandomSplit((int) totalDataSetObjectCount, dataSetObjectsPerSplit, trainingData, rng.nextLong());
        if (collectTrainingStats) stats.logSplitEnd();

        int splitNum = 1;
        for (JavaPairRDD<String, PortableDataStream> split : splits) {
            JavaRDD<PortableDataStream> streams = split.values();
            doIterationPDS(null, graph, streams, splitNum++, splits.length);
        }

        if (collectTrainingStats) stats.logFitEnd((int) totalDataSetObjectCount);
    }

    @Override
    public void executeTrainingMDS(SparkComputationGraph graph, JavaPairRDD<String, PortableDataStream> trainingData) {
        if (numWorkers == null) numWorkers = graph.getSparkContext().defaultParallelism();

        if (collectTrainingStats) stats.logFitStart();
        if (storageLevelStreams != null) trainingData.persist(storageLevelStreams);

        if (collectTrainingStats) stats.logCountStart();
        long totalDataSetObjectCount = trainingData.count();
        if (collectTrainingStats) stats.logCountEnd();
        int dataSetObjectsPerSplit = getNumDataSetObjectsPerSplit(rddDataSetNumExamples);

        if (collectTrainingStats) stats.logSplitStart();
        JavaPairRDD<String, PortableDataStream>[] splits = SparkUtils.balancedRandomSplit((int) totalDataSetObjectCount, dataSetObjectsPerSplit, trainingData, rng.nextLong());
        if (collectTrainingStats) stats.logSplitEnd();

        int splitNum = 1;
        for (JavaPairRDD<String, PortableDataStream> split : splits) {
            JavaRDD<PortableDataStream> streams = split.values();
            if (collectTrainingStats) stats.logRepartitionStart();
            streams = SparkUtils.repartition(streams, repartition, repartitionStrategy, numObjectsEachWorker(rddDataSetNumExamples), numWorkers);
            if (collectTrainingStats && repartition != Repartition.Never) stats.logRepartitionEnd();

            doIterationPDS_MDS(graph, streams, splitNum++, splits.length);
        }

        if (collectTrainingStats) stats.logFitEnd((int) totalDataSetObjectCount);
    }

    @Override
    public void executeTrainingPaths(SparkComputationGraph network, JavaRDD<String> trainingDataPaths) {
        if (numWorkers == null) numWorkers = network.getSparkContext().defaultParallelism();

        if (collectTrainingStats) stats.logFitStart();
        if (storageLevelStreams != null) trainingDataPaths.persist(storageLevelStreams);

        if (collectTrainingStats) stats.logCountStart();
        long totalDataSetObjectCount = trainingDataPaths.count();
        if (collectTrainingStats) stats.logCountEnd();

        int dataSetObjectsPerSplit = getNumDataSetObjectsPerSplit(rddDataSetNumExamples);
        if (collectTrainingStats) stats.logSplitStart();
        JavaRDD<String>[] splits = SparkUtils.balancedRandomSplit((int) totalDataSetObjectCount, dataSetObjectsPerSplit, trainingDataPaths, rng.nextLong());
        if (collectTrainingStats) stats.logSplitEnd();


        int splitNum = 1;
        for (JavaRDD<String> split : splits) {
            doIterationPaths(null, network, split, splitNum++, splits.length, rddDataSetNumExamples);
        }

        if (collectTrainingStats) stats.logFitEnd((int) totalDataSetObjectCount);
    }

    @Override
    public void executeTrainingPathsMDS(SparkComputationGraph network, JavaRDD<String> trainingMultiDataPaths) {
        executeTrainingPathsMDSHelper(network, trainingMultiDataPaths, rddDataSetNumExamples);
    }

    private void executeTrainingPathsMDSHelper(SparkComputationGraph network, JavaRDD<String> trainingMultiDataPaths, int dataSetObjectsNumExamples) {
        if (numWorkers == null) numWorkers = network.getSparkContext().defaultParallelism();

        if (collectTrainingStats) stats.logFitStart();
        if (storageLevelStreams != null) trainingMultiDataPaths.persist(storageLevelStreams);

        if (collectTrainingStats) stats.logCountStart();
        long totalDataSetObjectCount = trainingMultiDataPaths.count();
        if (collectTrainingStats) stats.logCountEnd();

        int dataSetObjectsPerSplit = getNumDataSetObjectsPerSplit(dataSetObjectsNumExamples);
        if (collectTrainingStats) stats.logSplitStart();
        JavaRDD<String>[] splits = SparkUtils.balancedRandomSplit((int) totalDataSetObjectCount, dataSetObjectsPerSplit, trainingMultiDataPaths, rng.nextLong());
        if (collectTrainingStats) stats.logSplitEnd();


        int splitNum = 1;
        for (JavaRDD<String> split : splits) {
            doIterationPathsMDS(network, split, splitNum++, splits.length, dataSetObjectsNumExamples);
        }

        if (collectTrainingStats) stats.logFitEnd((int) totalDataSetObjectCount);
    }

    @Override
    public void setCollectTrainingStats(boolean collectTrainingStats) {
        this.collectTrainingStats = collectTrainingStats;
        if (collectTrainingStats) {
            if (this.stats == null)
                this.stats = new ParameterAveragingTrainingMasterStats.ParameterAveragingTrainingMasterStatsHelper();
        } else {
            this.stats = null;
        }
    }

    @Override
    public boolean getIsCollectTrainingStats() {
        return collectTrainingStats;
    }

    @Override
    public SparkTrainingStats getTrainingStats() {
        if (stats != null) return stats.build();
        return null;
    }

    @Override
    public void setListeners(Collection<IterationListener> listeners) {
        setListeners(null, listeners);
    }

    @Override
    public void setListeners(StatsStorageRouter statsStorage, Collection<IterationListener> listeners) {
        this.statsStorage = statsStorage;
        this.listeners = listeners;
    }

    @Override
    public boolean deleteTempFiles(JavaSparkContext sc) {
        return lastRDDExportPath == null || deleteTempDir(sc, lastRDDExportPath);
    }

    @Override
    public boolean deleteTempFiles(SparkContext sc) {
        return deleteTempFiles(new JavaSparkContext(sc));
    }


    private void doIteration(SparkDl4jMultiLayer network, JavaRDD<DataSet> split, int splitNum, int numSplits) {
        log.info("Starting training of split {} of {}. workerMiniBatchSize={}, averagingFreq={}, Configured for {} workers",
                splitNum, numSplits, batchSizePerWorker, averagingFrequency, numWorkers);
        if (collectTrainingStats) stats.logMapPartitionsStart();

        JavaRDD<DataSet> splitData = split;
        if (collectTrainingStats) stats.logRepartitionStart();
        splitData = SparkUtils.repartition(splitData, repartition, repartitionStrategy, numObjectsEachWorker(rddDataSetNumExamples), numWorkers);
        int nPartitions = splitData.partitions().size();
        if (collectTrainingStats && repartition != Repartition.Never) stats.logRepartitionEnd();


        FlatMapFunction<Iterator<DataSet>, ParameterAveragingTrainingResult> function = new ExecuteWorkerFlatMap<>(getWorkerInstance(network));
        JavaRDD<ParameterAveragingTrainingResult> result = splitData.mapPartitions(function);
        processResults(network, null, result, splitNum, numSplits);

        if (collectTrainingStats) stats.logMapPartitionsEnd(nPartitions);
    }

    private void doIterationPDS(SparkDl4jMultiLayer network, SparkComputationGraph graph, JavaRDD<PortableDataStream> split, int splitNum, int numSplits) {
        log.info("Starting training of split {} of {}. workerMiniBatchSize={}, averagingFreq={}, Configured for {} workers",
                splitNum, numSplits, batchSizePerWorker, averagingFrequency, numWorkers);
        if (collectTrainingStats) stats.logMapPartitionsStart();

        JavaRDD<PortableDataStream> splitData = split;
        if (collectTrainingStats) stats.logRepartitionStart();
        splitData = SparkUtils.repartition(splitData, repartition, repartitionStrategy, numObjectsEachWorker(rddDataSetNumExamples), numWorkers);
        int nPartitions = splitData.partitions().size();
        if (collectTrainingStats && repartition != Repartition.Never) stats.logRepartitionEnd();

        FlatMapFunction<Iterator<PortableDataStream>, ParameterAveragingTrainingResult> function;
        if (network != null) function = new ExecuteWorkerPDSFlatMap<>(getWorkerInstance(network));
        else function = new ExecuteWorkerPDSFlatMap<>(getWorkerInstance(graph));

        JavaRDD<ParameterAveragingTrainingResult> result = splitData.mapPartitions(function);
        processResults(network, graph, result, splitNum, numSplits);

        if (collectTrainingStats) stats.logMapPartitionsEnd(nPartitions);
    }

    private void doIterationPaths(SparkDl4jMultiLayer network, SparkComputationGraph graph, JavaRDD<String> split,
                                  int splitNum, int numSplits, int dataSetObjectNumExamples) {
        log.info("Starting training of split {} of {}. workerMiniBatchSize={}, averagingFreq={}, Configured for {} workers",
                splitNum, numSplits, batchSizePerWorker, averagingFrequency, numWorkers);
        if (collectTrainingStats) stats.logMapPartitionsStart();

        JavaRDD<String> splitData = split;
        if (collectTrainingStats) stats.logRepartitionStart();
        splitData = SparkUtils.repartition(splitData, repartition, repartitionStrategy, numObjectsEachWorker(dataSetObjectNumExamples), numWorkers);
        int nPartitions = splitData.partitions().size();
        if (collectTrainingStats && repartition != Repartition.Never) stats.logRepartitionEnd();

        FlatMapFunction<Iterator<String>, ParameterAveragingTrainingResult> function;
        if (network != null) function = new ExecuteWorkerPathFlatMap<>(getWorkerInstance(network));
        else function = new ExecuteWorkerPathFlatMap<>(getWorkerInstance(graph));

        JavaRDD<ParameterAveragingTrainingResult> result = splitData.mapPartitions(function);
        processResults(network, graph, result, splitNum, numSplits);

        if (collectTrainingStats) stats.logMapPartitionsEnd(nPartitions);
    }

    private void doIterationPathsMDS(SparkComputationGraph graph, JavaRDD<String> split, int splitNum, int numSplits, int dataSetObjectNumExamples) {
        log.info("Starting training of split {} of {}. workerMiniBatchSize={}, averagingFreq={}, Configured for {} workers",
                splitNum, numSplits, batchSizePerWorker, averagingFrequency, numWorkers);
        if (collectTrainingStats) stats.logMapPartitionsStart();

        JavaRDD<String> splitData = split;
        if (collectTrainingStats) stats.logRepartitionStart();
        splitData = SparkUtils.repartition(splitData, repartition, repartitionStrategy, numObjectsEachWorker(dataSetObjectNumExamples), numWorkers);
        int nPartitions = splitData.partitions().size();
        if (collectTrainingStats && repartition != Repartition.Never) stats.logRepartitionEnd();


        FlatMapFunction<Iterator<String>, ParameterAveragingTrainingResult> function
                = new ExecuteWorkerPathMDSFlatMap<>(getWorkerInstance(graph));

        JavaRDD<ParameterAveragingTrainingResult> result = splitData.mapPartitions(function);
        processResults(null, graph, result, splitNum, numSplits);

        if (collectTrainingStats) stats.logMapPartitionsEnd(nPartitions);
    }

    private void doIteration(SparkComputationGraph graph, JavaRDD<MultiDataSet> split, int splitNum, int numSplits) {
        log.info("Starting training of split {} of {}. workerMiniBatchSize={}, averagingFreq={}, Configured for {} workers",
                splitNum, numSplits, batchSizePerWorker, averagingFrequency, numWorkers);
        if (collectTrainingStats) stats.logMapPartitionsStart();

        JavaRDD<MultiDataSet> splitData = split;

        splitData = SparkUtils.repartition(splitData, repartition, repartitionStrategy, numObjectsEachWorker(rddDataSetNumExamples), numWorkers);
        int nPartitions = split.partitions().size();

        FlatMapFunction<Iterator<MultiDataSet>, ParameterAveragingTrainingResult> function = new ExecuteWorkerMultiDataSetFlatMap<>(getWorkerInstance(graph));
        JavaRDD<ParameterAveragingTrainingResult> result = splitData.mapPartitions(function);
        processResults(null, graph, result, splitNum, numSplits);

        if (collectTrainingStats) stats.logMapPartitionsEnd(nPartitions);
    }

    private void doIterationPDS_MDS(SparkComputationGraph graph, JavaRDD<PortableDataStream> split, int splitNum, int numSplits) {
        log.info("Starting training of split {} of {}. workerMiniBatchSize={}, averagingFreq={}, Configured for {} workers",
                splitNum, numSplits, batchSizePerWorker, averagingFrequency, numWorkers);
        if (collectTrainingStats) stats.logMapPartitionsStart();

        JavaRDD<PortableDataStream> splitData = split;
        if (collectTrainingStats) stats.logRepartitionStart();
        splitData = SparkUtils.repartition(splitData, repartition, repartitionStrategy, numObjectsEachWorker(rddDataSetNumExamples), numWorkers);
        int nPartitions = splitData.partitions().size();
        if (collectTrainingStats && repartition != Repartition.Never) stats.logRepartitionEnd();

        FlatMapFunction<Iterator<PortableDataStream>, ParameterAveragingTrainingResult> function = new ExecuteWorkerPDSMDSFlatMap<>(getWorkerInstance(graph));

        JavaRDD<ParameterAveragingTrainingResult> result = splitData.mapPartitions(function);
        processResults(null, graph, result, splitNum, numSplits);

        if (collectTrainingStats) stats.logMapPartitionsEnd(nPartitions);
    }


    private void processResults(SparkDl4jMultiLayer network, SparkComputationGraph graph, JavaRDD<ParameterAveragingTrainingResult> results, int splitNum, int totalSplits) {
        //Need to do parameter averaging, and where necessary also do averaging of the updaters
        //Let's do all of this in ONE step, such that we don't have extra synchronization costs

        if (collectTrainingStats) stats.logAggregateStartTime();
        ParameterAveragingAggregationTuple tuple = results.aggregate(null,
                new ParameterAveragingElementAddFunction(),
                new ParameterAveragingElementCombineFunction());
        INDArray params = tuple.getParametersSum();
        int aggCount = tuple.getAggregationsCount();
        SparkTrainingStats aggregatedStats = tuple.getSparkTrainingStats();
        if (collectTrainingStats) stats.logAggregationEndTime();


        if (collectTrainingStats) stats.logProcessParamsUpdaterStart();
        if(params != null){
            params.divi(aggCount);
            INDArray updaterState = tuple.getUpdaterStateSum();
            if (updaterState != null) updaterState.divi(aggCount);   //May be null if all SGD updaters, for example

            if (network != null) {
                MultiLayerNetwork net = network.getNetwork();
                net.setParameters(params);
                if (updaterState != null) net.getUpdater().setStateViewArray(null, updaterState, false);

                network.setScore(tuple.getScoreSum() / tuple.getAggregationsCount());
            } else {
                ComputationGraph g = graph.getNetwork();
                g.setParams(params);
                if (updaterState != null) g.getUpdater().setStateViewArray(updaterState);

                graph.setScore(tuple.getScoreSum() / tuple.getAggregationsCount());
            }
        } else {
            log.info("Skipping imbalanced split with no data for all executors");
        }



        if (collectTrainingStats) {
            stats.logProcessParamsUpdaterEnd();
            stats.addWorkerStats(aggregatedStats);
        }

        if(statsStorage != null) {
            Collection<StorageMetaData> meta = tuple.getListenerMetaData();
            if(meta != null && meta.size() > 0){
                statsStorage.putStorageMetaData(meta);
            }

            Collection<Persistable> staticInfo = tuple.getListenerStaticInfo();
            if(staticInfo != null && staticInfo.size() > 0){
                statsStorage.putStaticInfo(staticInfo);
            }

            Collection<Persistable> updates = tuple.getListenerUpdates();
            if(updates != null && updates.size() > 0){
                statsStorage.putUpdate(updates);
            }
        }

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueueBlocking();



        log.info("Completed training of split {} of {}", splitNum, totalSplits);

        if(params != null) {
            //Params may be null for edge case (empty RDD)
            if (network != null) {
                MultiLayerConfiguration conf = network.getNetwork().getLayerWiseConfigurations();
                int numUpdates = network.getNetwork().conf().getNumIterations() * averagingFrequency;
                conf.setIterationCount(conf.getIterationCount() + numUpdates);
            } else {
                ComputationGraphConfiguration conf = graph.getNetwork().getConfiguration();
                int numUpdates = graph.getNetwork().conf().getNumIterations() * averagingFrequency;
                conf.setIterationCount(conf.getIterationCount() + numUpdates);
            }
        }
    }


    private JavaRDD<String> exportIfRequired(JavaSparkContext sc, JavaRDD<DataSet> trainingData) {
        if (collectTrainingStats) stats.logExportStart();

        //Two possibilities here:
        // 1. We've seen this RDD before (i.e., multiple epochs training case)
        // 2. We have not seen this RDD before
        //    (a) And we havent got any stored data -> simply export
        //    (b) And we previously exported some data from a different RDD -> delete the last data
        int currentRDDUid = trainingData.id();       //Id is a "A unique ID for this RDD (within its SparkContext)."

        String baseDir;
        if (lastExportedRDDId == Integer.MIN_VALUE) {
            //Haven't seen a RDD<DataSet> yet in this training master -> export data
            baseDir = export(trainingData);
        } else {
            if (lastExportedRDDId == currentRDDUid) {
                //Use the already-exported data again for another epoch
                baseDir = getBaseDirForRDD(trainingData);
            } else {
                //The new RDD is different to the last one
                // Clean up the data for the last one, and export
                deleteTempDir(sc, lastRDDExportPath);
                baseDir = export(trainingData);
            }
        }

        if (collectTrainingStats) stats.logExportEnd();

        return sc.textFile(baseDir + "paths/");
    }

    private JavaRDD<String> exportIfRequiredMDS(JavaSparkContext sc, JavaRDD<MultiDataSet> trainingData) {
        if (collectTrainingStats) stats.logExportStart();

        //Two possibilities here:
        // 1. We've seen this RDD before (i.e., multiple epochs training case)
        // 2. We have not seen this RDD before
        //    (a) And we havent got any stored data -> simply export
        //    (b) And we previously exported some data from a different RDD -> delete the last data
        int currentRDDUid = trainingData.id();       //Id is a "A unique ID for this RDD (within its SparkContext)."

        String baseDir;
        if (lastExportedRDDId == Integer.MIN_VALUE) {
            //Haven't seen a RDD<DataSet> yet in this training master -> export data
            baseDir = exportMDS(trainingData);
        } else {
            if (lastExportedRDDId == currentRDDUid) {
                //Use the already-exported data again for another epoch
                baseDir = getBaseDirForRDD(trainingData);
            } else {
                //The new RDD is different to the last one
                // Clean up the data for the last one, and export
                deleteTempDir(sc, lastRDDExportPath);
                baseDir = exportMDS(trainingData);
            }
        }

        if (collectTrainingStats) stats.logExportEnd();

        return sc.textFile(baseDir + "paths/");
    }

    private String export(JavaRDD<DataSet> trainingData) {
        String baseDir = getBaseDirForRDD(trainingData);
        String dataDir = baseDir + "data/";
        String pathsDir = baseDir + "paths/";

        log.info("Initiating RDD<DataSet> export at {}", baseDir);
        JavaRDD<String> paths = trainingData.mapPartitionsWithIndex(new BatchAndExportDataSetsFunction(batchSizePerWorker, dataDir), true);
        paths.saveAsTextFile(pathsDir);
        log.info("RDD<DataSet> export complete at {}", baseDir);

        lastExportedRDDId = trainingData.id();
        lastRDDExportPath = baseDir;
        return baseDir;
    }

    private String exportMDS(JavaRDD<MultiDataSet> trainingData) {
        String baseDir = getBaseDirForRDD(trainingData);
        String dataDir = baseDir + "data/";
        String pathsDir = baseDir + "paths/";

        log.info("Initiating RDD<MultiDataSet> export at {}", baseDir);
        JavaRDD<String> paths = trainingData.mapPartitionsWithIndex(new BatchAndExportMultiDataSetsFunction(batchSizePerWorker, dataDir), true);
        paths.saveAsTextFile(pathsDir);
        log.info("RDD<MultiDataSet> export complete at {}", baseDir);

        lastExportedRDDId = trainingData.id();
        lastRDDExportPath = baseDir;
        return baseDir;
    }

    private String getBaseDirForRDD(JavaRDD<?> rdd) {
        if (exportDirectory == null) {
            exportDirectory = getDefaultExportDirectory(rdd.context());
        }

        return exportDirectory + (exportDirectory.endsWith("/") ? "" : "/") + trainingMasterUID + "/" + rdd.id() + "/";
    }

    private boolean deleteTempDir(JavaSparkContext sc, String tempDirPath) {
        log.info("Attempting to delete temporary directory: {}", tempDirPath);

        Configuration hadoopConfiguration = sc.hadoopConfiguration();
        FileSystem fileSystem;
        try {
            fileSystem = FileSystem.get(new URI(tempDirPath), hadoopConfiguration);
        } catch (URISyntaxException | IOException e) {
            throw new RuntimeException(e);
        }

        try {
            fileSystem.delete(new Path(tempDirPath), true);
            log.info("Deleted temporary directory: {}", tempDirPath);
            return true;
        } catch (IOException e) {
            log.warn("Could not delete temporary directory: {}", tempDirPath, e);
            return false;
        }
    }

    private String getDefaultExportDirectory(SparkContext sc) {
        String hadoopTmpDir = sc.hadoopConfiguration().get("hadoop.tmp.dir");
        if (!hadoopTmpDir.endsWith("/") && !hadoopTmpDir.endsWith("\\")) hadoopTmpDir = hadoopTmpDir + "/";
        return hadoopTmpDir + "dl4j/";
    }


    private StatsStorageRouterProvider getRouterProvider(){
        if(statsStorage == null) return null;   //Not needed
        return new VanillaStatsStorageRouterProvider();
    }


    public static class Builder {
        private boolean saveUpdater;
        private Integer numWorkers;
        private int rddDataSetNumExamples;
        private int batchSizePerWorker = 16;
        private int averagingFrequency = 5;
        private int prefetchNumBatches = 0;
        private Repartition repartition = Repartition.Always;
        private RepartitionStrategy repartitionStrategy = RepartitionStrategy.Balanced;
        private StorageLevel storageLevel = StorageLevel.MEMORY_ONLY_SER();
        private StorageLevel storageLevelStreams = StorageLevel.MEMORY_ONLY();
        private RDDTrainingApproach rddTrainingApproach = RDDTrainingApproach.Export;
        private String exportDirectory = null;
        private Long rngSeed;
        private Collection<TrainingHook> trainingHooks;


        /**
         * Adds training hooks to the master.
         * The training master will setup the workers
         * with the desired hooks for training.
         * This can allow for tings like parameter servers
         * and async updates as well as collecting statistics.
         *
         * @param trainingHooks the training hooks to ad
         * @return
         */
        public Builder trainingHooks(Collection<TrainingHook> trainingHooks) {
            this.trainingHooks = trainingHooks;
            return this;
        }

        /**
         * Adds training hooks to the master.
         * The training master will setup the workers
         * with the desired hooks for training.
         * This can allow for tings like parameter servers
         * and async updates as well as collecting statistics.
         * @param hooks the training hooks to ad
         * @return
         */
        public Builder trainingHooks(TrainingHook... hooks) {
            this.trainingHooks = Arrays.asList(hooks);
            return this;
        }

        /**
         * Same as {@link #Builder(Integer, int)} but automatically set number of workers based on JavaSparkContext.defaultParallelism()
         *
         * @param rddDataSetNumExamples Number of examples in each DataSet object in the {@code RDD<DataSet>}
         */
        public Builder(int rddDataSetNumExamples) {
            this(null, rddDataSetNumExamples);
        }

        /**
         * Create a builder, where the following number of workers (Spark executors * number of threads per executor) are
         * being used.<br>
         * Note: this should match the configuration of the cluster.<br>
         * <p>
         * It is also necessary to specify how many examples are in each DataSet that appears in the {@code RDD<DataSet>}
         * or {@code JavaRDD<DataSet>} used for training.<br>
         * Two most common cases here:<br>
         * (a) Preprocessed data pipelines will often load binary DataSet objects with N > 1 examples in each; in this case,
         * rddDataSetNumExamples should be set to N <br>
         * (b) "In line" data pipelines (for example, CSV String -> record reader -> DataSet just before training) will
         * typically have exactly 1 example in each DataSet object. In this case, rddDataSetNumExamples should be set to 1
         *
         * @param numWorkers            Number of Spark execution threads in the cluster. May be null. If null: number of workers will
         *                              be obtained from JavaSparkContext.defaultParallelism(), which should provide the number of cores
         *                              in the cluster.
         * @param rddDataSetNumExamples Number of examples in each DataSet object in the {@code RDD<DataSet>}
         */
        public Builder(Integer numWorkers, int rddDataSetNumExamples) {
            if (numWorkers != null && numWorkers <= 0)
                throw new IllegalArgumentException("Invalid number of workers: " + numWorkers + " (must be >= 1)");
            if (rddDataSetNumExamples <= 0)
                throw new IllegalArgumentException("Invalid rdd data set size: " + rddDataSetNumExamples + " (must be >= 1)");
            this.numWorkers = numWorkers;
            this.rddDataSetNumExamples = rddDataSetNumExamples;
        }

        /**
         * Batch size (in number of examples) per worker, for each fit(DataSet) call.
         *
         * @param batchSizePerWorker Size of each minibatch to use for each worker
         * @return
         */
        public Builder batchSizePerWorker(int batchSizePerWorker) {
            this.batchSizePerWorker = batchSizePerWorker;
            return this;
        }

        /**
         * Frequency with which to average worker parameters.<br>
         * <b>Note</b>: Too high or too low can be bad for different reasons.<br>
         * - Too low (such as 1) can result in a lot of network traffic<br>
         * - Too high (>> 20 or so) can result in accuracy issues or problems with network convergence
         *
         * @param averagingFrequency Frequency (in number of minibatches of size 'batchSizePerWorker') to average parameters
         */
        public Builder averagingFrequency(int averagingFrequency) {
            if (averagingFrequency <= 0)
                throw new IllegalArgumentException("Ivalid input: averaging frequency must be >= 1");
            this.averagingFrequency = averagingFrequency;
            return this;
        }

        /**
         * Set the number of minibatches to asynchronously prefetch in the worker.
         * <p>
         * Default: 0 (no prefetching)
         *
         * @param prefetchNumBatches Number of minibatches (DataSets of size batchSizePerWorker) to fetch
         */
        public Builder workerPrefetchNumBatches(int prefetchNumBatches) {
            this.prefetchNumBatches = prefetchNumBatches;
            return this;
        }

        /**
         * Set whether the updater (i.e., historical state for momentum, adagrad, etc should be saved).
         * <b>NOTE</b>: This can <b>double</b> (or more) the amount of network traffic in each direction, but might
         * improve network training performance (and can be more stable for certain updaters such as adagrad).<br>
         * <p>
         * This is <b>enabled</b> by default.
         *
         * @param saveUpdater If true: retain the updater state (default). If false, don't retain (updaters will be
         *                    reinitalized in each worker after averaging).
         */
        public Builder saveUpdater(boolean saveUpdater) {
            this.saveUpdater = saveUpdater;
            return this;
        }

        /**
         * Set if/when repartitioning should be conducted for the training data.<br>
         * Default value: always repartition (if required to guarantee correct number of partitions and correct number
         * of examples in each partition).
         *
         * @param repartition Setting for repartitioning
         */
        public Builder repartionData(Repartition repartition) {
            this.repartition = repartition;
            return this;
        }

        /**
         * Used in conjunction with {@link #repartionData(Repartition)} (which defines <i>when</i> repartitioning should be
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
         * Set the storage level RDDs used when fitting data from Streams: either PortableDataStreams (sc.binaryFiles via
         * {@link SparkDl4jMultiLayer#fit(String)} and {@link SparkComputationGraph#fit(String)}) or String paths
         * (via {@link SparkDl4jMultiLayer#fitPaths(JavaRDD)}, {@link SparkComputationGraph#fitPaths(JavaRDD)} and
         * {@link SparkComputationGraph#fitPathsMultiDataSet(JavaRDD)}).<br>
         * <p>
         * Default storage level is StorageLevel.MEMORY_ONLY() which should be appropriate in most cases.
         *
         * @param storageLevelStreams Storage level to use
         */
        public Builder storageLevelStreams(StorageLevel storageLevelStreams) {
            this.storageLevelStreams = storageLevelStreams;
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

        public ParameterAveragingTrainingMaster build() {
            return new ParameterAveragingTrainingMaster(this);
        }
    }


}
