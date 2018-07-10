/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.spark.impl.paramavg;

import lombok.Data;
import lombok.EqualsAndHashCode;
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
import org.deeplearning4j.api.storage.StatsStorageRouterProvider;
import org.deeplearning4j.api.storage.StorageMetaData;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.spark.api.*;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.api.worker.*;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.impl.graph.dataset.DataSetToMultiDataSetFn;
import org.deeplearning4j.spark.impl.listeners.VanillaStatsStorageRouterProvider;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.aggregator.ParameterAveragingAggregationTuple;
import org.deeplearning4j.spark.impl.paramavg.aggregator.ParameterAveragingElementAddFunction;
import org.deeplearning4j.spark.impl.paramavg.aggregator.ParameterAveragingElementCombineFunction;
import org.deeplearning4j.spark.impl.paramavg.stats.ParameterAveragingTrainingMasterStats;
import org.deeplearning4j.spark.util.SparkUtils;
import org.deeplearning4j.util.UIDProvider;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.util.*;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * ParameterAveragingTrainingMaster: A {@link TrainingMaster}
 * implementation for training networks on Spark.
 * This is standard parameter averaging with a
 * configurable averaging period.
 *
 * @author Alex Black
 */
@Data
@JsonIgnoreProperties({"stats", "listeners", "iterationCount", "rng", "lastExportedRDDId", "lastRDDExportPath",
                "trainingMasterUID"})
@EqualsAndHashCode(exclude = {"stats", "listeners", "iterationCount", "rng", "lastExportedRDDId", "lastRDDExportPath",
                "trainingMasterUID"})
@Slf4j
public class ParameterAveragingTrainingMaster
                extends BaseTrainingMaster<ParameterAveragingTrainingResult, ParameterAveragingTrainingWorker>
                implements TrainingMaster<ParameterAveragingTrainingResult, ParameterAveragingTrainingWorker> {

    protected static final int COALESCE_THRESHOLD = 3;


    protected boolean saveUpdater;
    protected Integer numWorkers;
    protected int rddDataSetNumExamples;

    protected int averagingFrequency;
    protected int aggregationDepth;
    protected int prefetchNumBatches;
    protected int iterationCount = 0;

    protected Collection<TrainingHook> trainingHookList;

    protected ParameterAveragingTrainingMaster() {
        // no-arg constructor for Jackson

        String jvmuid = UIDProvider.getJVMUID();
        this.trainingMasterUID =
                        System.currentTimeMillis() + "_" + (jvmuid.length() <= 8 ? jvmuid : jvmuid.substring(0, 8));
        this.rng = new Random();
    }

    protected ParameterAveragingTrainingMaster(Builder builder) {
        this.saveUpdater = builder.saveUpdater;
        this.numWorkers = builder.numWorkers;
        this.rddDataSetNumExamples = builder.rddDataSetNumExamples;
        this.batchSizePerWorker = builder.batchSizePerWorker;
        this.averagingFrequency = builder.averagingFrequency;
        this.aggregationDepth = builder.aggregationDepth;
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
        this.trainingMasterUID =
                        System.currentTimeMillis() + "_" + (jvmuid.length() <= 8 ? jvmuid : jvmuid.substring(0, 8));
    }

    public ParameterAveragingTrainingMaster(boolean saveUpdater, Integer numWorkers, int rddDataSetNumExamples,
                    int batchSizePerWorker, int averagingFrequency, int prefetchNumBatches) {
        this(saveUpdater, numWorkers, rddDataSetNumExamples, batchSizePerWorker, averagingFrequency, 2,
                        prefetchNumBatches, Repartition.Always, RepartitionStrategy.Balanced, false);
    }

    /**
     * @param saveUpdater           If true: save (and average) the updater state when doing parameter averaging
     * @param numWorkers            Number of workers (executors * threads per executor) for the cluster
     * @param rddDataSetNumExamples Number of examples in each DataSet object in the {@code RDD<DataSet>}
     * @param batchSizePerWorker    Number of examples to use per worker per fit
     * @param averagingFrequency    Frequency (in number of minibatches) with which to average parameters
     * @param aggregationDepth      Number of aggregation levels used in parameter aggregation
     * @param prefetchNumBatches    Number of batches to asynchronously prefetch (0: disable)
     * @param repartition           Set if/when repartitioning should be conducted for the training data
     * @param repartitionStrategy   Repartitioning strategy to use. See {@link RepartitionStrategy}
     * @param collectTrainingStats  If true: collect training statistics for debugging/optimization purposes
     */
    public ParameterAveragingTrainingMaster(boolean saveUpdater, Integer numWorkers, int rddDataSetNumExamples,
                    int batchSizePerWorker, int averagingFrequency, int aggregationDepth, int prefetchNumBatches,
                    Repartition repartition, RepartitionStrategy repartitionStrategy, boolean collectTrainingStats) {
        this(saveUpdater, numWorkers, rddDataSetNumExamples, batchSizePerWorker, averagingFrequency, aggregationDepth,
                        prefetchNumBatches, repartition, repartitionStrategy, StorageLevel.MEMORY_ONLY_SER(),
                        collectTrainingStats);
    }

    public ParameterAveragingTrainingMaster(boolean saveUpdater, Integer numWorkers, int rddDataSetNumExamples,
                    int batchSizePerWorker, int averagingFrequency, int aggregationDepth, int prefetchNumBatches,
                    Repartition repartition, RepartitionStrategy repartitionStrategy, StorageLevel storageLevel,
                    boolean collectTrainingStats) {
        checkArgument(numWorkers > 0, "Invalid number of workers: " + numWorkers + " (must be >= 1)");
        checkArgument(rddDataSetNumExamples > 0,
                        "Invalid rdd data set size: " + rddDataSetNumExamples + " (must be >= 1)");
        checkArgument(averagingFrequency > 0, "Invalid input: averaging frequency must be >= 1");
        checkArgument(aggregationDepth > 0, "Invalid input: tree aggregation channels must be >= 1");

        this.saveUpdater = saveUpdater;
        this.numWorkers = numWorkers;
        this.rddDataSetNumExamples = rddDataSetNumExamples;
        this.batchSizePerWorker = batchSizePerWorker;
        this.averagingFrequency = averagingFrequency;
        this.aggregationDepth = aggregationDepth;
        this.prefetchNumBatches = prefetchNumBatches;
        this.collectTrainingStats = collectTrainingStats;
        this.repartition = repartition;
        this.repartitionStrategy = repartitionStrategy;
        this.storageLevel = storageLevel;
        if (collectTrainingStats)
            stats = new ParameterAveragingTrainingMasterStats.ParameterAveragingTrainingMasterStatsHelper();

        String jvmuid = UIDProvider.getJVMUID();
        this.trainingMasterUID =
                        System.currentTimeMillis() + "_" + (jvmuid.length() <= 8 ? jvmuid : jvmuid.substring(0, 8));
        this.rng = new Random();
    }



    /**
     * Remove a training hook from the worker
     *
     * @param trainingHook the training hook to remove
     */
    @Override
    public void removeHook(TrainingHook trainingHook) {
        if (trainingHookList == null)
            return;
        trainingHookList.remove(trainingHook);
    }

    /**
     * Add a hook for the master for pre and post training
     *
     * @param trainingHook the training hook to add
     */
    @Override
    public void addHook(TrainingHook trainingHook) {
        if (trainingHookList == null) {
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
                        network.getNetwork().params(), network.getNetwork().getUpdater().getStateViewArray());

        if (collectTrainingStats)
            stats.logBroadcastStart();
        Broadcast<NetBroadcastTuple> broadcast = network.getSparkContext().broadcast(tuple);
        if (collectTrainingStats)
            stats.logBroadcastEnd();

        WorkerConfiguration configuration = new WorkerConfiguration(false, rddDataSetNumExamples, batchSizePerWorker,
                        averagingFrequency, prefetchNumBatches, collectTrainingStats);
        return new ParameterAveragingTrainingWorker(broadcast, saveUpdater, configuration, trainingHookList, listeners,
                        getRouterProvider());
    }

    @Override
    public ParameterAveragingTrainingWorker getWorkerInstance(SparkComputationGraph graph) {
        NetBroadcastTuple tuple = new NetBroadcastTuple(graph.getNetwork().getConfiguration(),
                        graph.getNetwork().params(), graph.getNetwork().getUpdater().getStateViewArray());

        if (collectTrainingStats)
            stats.logBroadcastStart();
        Broadcast<NetBroadcastTuple> broadcast = graph.getSparkContext().broadcast(tuple);
        if (collectTrainingStats)
            stats.logBroadcastEnd();

        WorkerConfiguration configuration = new WorkerConfiguration(true, rddDataSetNumExamples, batchSizePerWorker,
                        averagingFrequency, prefetchNumBatches, collectTrainingStats);
        return new ParameterAveragingTrainingWorker(broadcast, saveUpdater, configuration, trainingHookList, listeners,
                        getRouterProvider());
    }

    protected int numObjectsEachWorker(int numExamplesEachRddObject) {
        return batchSizePerWorker * averagingFrequency / numExamplesEachRddObject;
    }

    protected int getNumDataSetObjectsPerSplit(int numExamplesEachRddObject) {
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
        if (numWorkers == null)
            numWorkers = network.getSparkContext().defaultParallelism();

        if (rddTrainingApproach == RDDTrainingApproach.Direct) {
            executeTrainingDirect(network, trainingData);
        } else {
            //Export data if required (or, use cached export)
            JavaRDD<String> paths = exportIfRequired(network.getSparkContext(), trainingData);
            executeTrainingPathsHelper(network, paths, batchSizePerWorker); //Originally (pre-export): had rddDataSetNumExamples per DataSet. Now we have batchSizePerWorker per exported DataSet
        }
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

    protected <T, Repr> JavaPairRDD<T, Repr>[] getSplitRDDs(JavaPairRDD<T, Repr> trainingData,
                    int totalDataSetObjectCount) {
        int dataSetObjectsPerSplit = getNumDataSetObjectsPerSplit(rddDataSetNumExamples);

        if (collectTrainingStats)
            stats.logSplitStart();
        JavaPairRDD<T, Repr>[] splits = SparkUtils.balancedRandomSplit(totalDataSetObjectCount, dataSetObjectsPerSplit,
                        trainingData, rng.nextLong());
        if (collectTrainingStats)
            stats.logSplitEnd();
        return splits;
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

    protected void executeTrainingDirect(SparkDl4jMultiLayer network, JavaRDD<DataSet> trainingData) {
        if (collectTrainingStats)
            stats.logFitStart();
        //For "vanilla" parameter averaging training, we need to split the full data set into batches of size N, such that we can process the specified
        // number of minibatches between averagings
        //But to do that, wee need to know: (a) the number of examples, and (b) the number of workers
        if (storageLevel != null)
            trainingData.persist(storageLevel);

        long totalDataSetObjectCount = getTotalDataSetObjectCount(trainingData);
        JavaRDD<DataSet>[] splits = getSplitRDDs(trainingData, (int) totalDataSetObjectCount, rddDataSetNumExamples);

        int splitNum = 1;
        for (JavaRDD<DataSet> split : splits) {
            doIteration(network, split, splitNum++, splits.length);
        }

        if (collectTrainingStats)
            stats.logFitEnd((int) totalDataSetObjectCount);
    }

    /**
     * @deprecated Due to poor performance
     */
    @Override
    @Deprecated
    public void executeTraining(SparkDl4jMultiLayer network, JavaPairRDD<String, PortableDataStream> trainingData) {
        if (numWorkers == null)
            numWorkers = network.getSparkContext().defaultParallelism();

        if (collectTrainingStats)
            stats.logFitStart();

        int origNumPartitions = trainingData.partitions().size();
        if (origNumPartitions >= COALESCE_THRESHOLD * numWorkers) {
            log.info("Coalescing PortableDataStreams from {} to {} partitions", origNumPartitions, numWorkers);
            trainingData = trainingData.coalesce(numWorkers);
        }
        if (storageLevelStreams != null)
            trainingData.persist(storageLevelStreams);

        long totalDataSetObjectCount = getTotalDataSetObjectCount(trainingData);
        JavaPairRDD<String, PortableDataStream>[] splits = getSplitRDDs(trainingData, (int) totalDataSetObjectCount);

        int splitNum = 1;
        for (JavaPairRDD<String, PortableDataStream> split : splits) {
            JavaRDD<PortableDataStream> streams = split.values();
            doIterationPDS(network, null, streams, splitNum++, splits.length);
        }

        if (collectTrainingStats)
            stats.logFitEnd((int) totalDataSetObjectCount);
    }

    @Override
    public void executeTrainingPaths(SparkDl4jMultiLayer network, JavaRDD<String> trainingDataPaths) {
        executeTrainingPathsHelper(network, trainingDataPaths, rddDataSetNumExamples);
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
        JavaRDD<String>[] splits =
                        getSplitRDDs(trainingDataPaths, (int) totalDataSetObjectCount, dataSetObjectsNumExamples);

        int splitNum = 1;
        for (JavaRDD<String> split : splits) {
            doIterationPaths(network, null, split, splitNum++, splits.length, dataSetObjectsNumExamples);
        }

        if (collectTrainingStats)
            stats.logFitEnd((int) totalDataSetObjectCount);
    }

    @Override
    public void executeTraining(SparkComputationGraph graph, JavaRDD<DataSet> trainingData) {
        if (numWorkers == null)
            numWorkers = graph.getSparkContext().defaultParallelism();

        JavaRDD<MultiDataSet> mdsTrainingData = trainingData.map(new DataSetToMultiDataSetFn());

        executeTrainingMDS(graph, mdsTrainingData);
    }

    @Override
    public void executeTrainingMDS(SparkComputationGraph graph, JavaRDD<MultiDataSet> trainingData) {
        if (numWorkers == null)
            numWorkers = graph.getSparkContext().defaultParallelism();

        if (rddTrainingApproach == RDDTrainingApproach.Direct) {
            executeTrainingDirect(graph, trainingData);
        } else {
            //Export data if required (or, use cached export)
            JavaRDD<String> paths = exportIfRequiredMDS(graph.getSparkContext(), trainingData);
            executeTrainingPathsMDSHelper(graph, paths, batchSizePerWorker);
        }
    }

    protected void executeTrainingDirect(SparkComputationGraph graph, JavaRDD<MultiDataSet> trainingData) {
        if (collectTrainingStats)
            stats.logFitStart();
        //For "vanilla" parameter averaging training, we need to split the full data set into batches of size N, such that we can process the specified
        // number of minibatches between averaging
        //But to do that, we need to know: (a) the number of examples, and (b) the number of workers
        if (storageLevel != null)
            trainingData.persist(storageLevel);

        long totalDataSetObjectCount = getTotalDataSetObjectCount(trainingData);

        JavaRDD<MultiDataSet>[] splits =
                        getSplitRDDs(trainingData, (int) totalDataSetObjectCount, rddDataSetNumExamples);

        int splitNum = 1;
        for (JavaRDD<MultiDataSet> split : splits) {
            doIteration(graph, split, splitNum++, splits.length);
        }

        if (collectTrainingStats)
            stats.logFitEnd((int) totalDataSetObjectCount);
    }

    @Override
    public void executeTraining(SparkComputationGraph graph, JavaPairRDD<String, PortableDataStream> trainingData) {
        if (numWorkers == null)
            numWorkers = graph.getSparkContext().defaultParallelism();

        if (collectTrainingStats)
            stats.logFitStart();
        //For "vanilla" parameter averaging training, we need to split the full data set into batches of size N, such that we can process the specified
        // number of minibatches between averagings
        //But to do that, we need to know: (a) the number of examples, and (b) the number of workers

        int origNumPartitions = trainingData.partitions().size();
        if (origNumPartitions >= COALESCE_THRESHOLD * numWorkers) {
            log.info("Coalescing streams from {} to {} partitions", origNumPartitions, numWorkers);
            trainingData = trainingData.coalesce(numWorkers);
        }
        if (storageLevelStreams != null)
            trainingData.persist(storageLevelStreams);

        long totalDataSetObjectCount = getTotalDataSetObjectCount(trainingData);
        JavaPairRDD<String, PortableDataStream>[] splits = getSplitRDDs(trainingData, (int) totalDataSetObjectCount);

        int splitNum = 1;
        for (JavaPairRDD<String, PortableDataStream> split : splits) {
            JavaRDD<PortableDataStream> streams = split.values();
            doIterationPDS(null, graph, streams, splitNum++, splits.length);
        }

        if (collectTrainingStats)
            stats.logFitEnd((int) totalDataSetObjectCount);
    }

    @Override
    public void executeTrainingMDS(SparkComputationGraph graph, JavaPairRDD<String, PortableDataStream> trainingData) {
        if (numWorkers == null)
            numWorkers = graph.getSparkContext().defaultParallelism();

        if (collectTrainingStats)
            stats.logFitStart();

        if (storageLevelStreams != null)
            trainingData.persist(storageLevelStreams);
        long totalDataSetObjectCount = getTotalDataSetObjectCount(trainingData);
        JavaPairRDD<String, PortableDataStream>[] splits = getSplitRDDs(trainingData, (int) totalDataSetObjectCount);

        int splitNum = 1;
        for (JavaPairRDD<String, PortableDataStream> split : splits) {
            JavaRDD<PortableDataStream> streams = split.values();
            if (collectTrainingStats)
                stats.logRepartitionStart();
            streams = SparkUtils.repartition(streams, repartition, repartitionStrategy,
                            numObjectsEachWorker(rddDataSetNumExamples), numWorkers);
            if (collectTrainingStats && repartition != Repartition.Never)
                stats.logRepartitionEnd();

            doIterationPDS_MDS(graph, streams, splitNum++, splits.length);
        }

        if (collectTrainingStats)
            stats.logFitEnd((int) totalDataSetObjectCount);
    }

    @Override
    public void executeTrainingPaths(SparkComputationGraph network, JavaRDD<String> trainingDataPaths) {
        if (numWorkers == null)
            numWorkers = network.getSparkContext().defaultParallelism();

        if (collectTrainingStats)
            stats.logFitStart();
        if (storageLevelStreams != null)
            trainingDataPaths.persist(storageLevelStreams);
        long totalDataSetObjectCount = getTotalDataSetObjectCount(trainingDataPaths);
        JavaRDD<String>[] splits =
                        getSplitRDDs(trainingDataPaths, (int) totalDataSetObjectCount, rddDataSetNumExamples);

        int splitNum = 1;
        for (JavaRDD<String> split : splits) {
            doIterationPaths(null, network, split, splitNum++, splits.length, rddDataSetNumExamples);
        }

        if (collectTrainingStats)
            stats.logFitEnd((int) totalDataSetObjectCount);
    }

    @Override
    public void executeTrainingPathsMDS(SparkComputationGraph network, JavaRDD<String> trainingMultiDataPaths) {
        executeTrainingPathsMDSHelper(network, trainingMultiDataPaths, rddDataSetNumExamples);
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

        JavaRDD<String>[] splits =
                        getSplitRDDs(trainingMultiDataPaths, (int) totalDataSetObjectCount, dataSetObjectsNumExamples);

        int splitNum = 1;
        for (JavaRDD<String> split : splits) {
            doIterationPathsMDS(network, split, splitNum++, splits.length, dataSetObjectsNumExamples);
        }

        if (collectTrainingStats)
            stats.logFitEnd((int) totalDataSetObjectCount);
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
        if (stats != null)
            return stats.build();
        return null;
    }

    @Override
    public void setListeners(Collection<TrainingListener> listeners) {
        setListeners(null, listeners);
    }

    @Override
    public void setListeners(StatsStorageRouter statsStorage, Collection<TrainingListener> listeners) {
        this.statsStorage = statsStorage;
        this.listeners = listeners;
    }



    protected void doIteration(SparkDl4jMultiLayer network, JavaRDD<DataSet> split, int splitNum, int numSplits) {
        log.info("Starting training of split {} of {}. workerMiniBatchSize={}, averagingFreq={}, Configured for {} workers",
                        splitNum, numSplits, batchSizePerWorker, averagingFrequency, numWorkers);
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


        FlatMapFunction<Iterator<DataSet>, ParameterAveragingTrainingResult> function =
                        new ExecuteWorkerFlatMap<>(getWorkerInstance(network));
        JavaRDD<ParameterAveragingTrainingResult> result = splitData.mapPartitions(function);
        processResults(network, null, result, splitNum, numSplits);

        if (collectTrainingStats)
            stats.logMapPartitionsEnd(nPartitions);
    }

    protected void doIterationPDS(SparkDl4jMultiLayer network, SparkComputationGraph graph,
                    JavaRDD<PortableDataStream> split, int splitNum, int numSplits) {
        log.info("Starting training of split {} of {}. workerMiniBatchSize={}, averagingFreq={}, Configured for {} workers",
                        splitNum, numSplits, batchSizePerWorker, averagingFrequency, numWorkers);
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

        FlatMapFunction<Iterator<PortableDataStream>, ParameterAveragingTrainingResult> function;
        if (network != null)
            function = new ExecuteWorkerPDSFlatMap<>(getWorkerInstance(network));
        else
            function = new ExecuteWorkerPDSFlatMap<>(getWorkerInstance(graph));

        JavaRDD<ParameterAveragingTrainingResult> result = splitData.mapPartitions(function);
        processResults(network, graph, result, splitNum, numSplits);

        if (collectTrainingStats)
            stats.logMapPartitionsEnd(nPartitions);
    }

    protected void doIterationPaths(SparkDl4jMultiLayer network, SparkComputationGraph graph, JavaRDD<String> split,
                    int splitNum, int numSplits, int dataSetObjectNumExamples) {
        log.info("Starting training of split {} of {}. workerMiniBatchSize={}, averagingFreq={}, Configured for {} workers",
                        splitNum, numSplits, batchSizePerWorker, averagingFrequency, numWorkers);
        if (collectTrainingStats)
            stats.logMapPartitionsStart();

        JavaRDD<String> splitData = split;
        if (collectTrainingStats)
            stats.logRepartitionStart();
        splitData = SparkUtils.repartition(splitData, repartition, repartitionStrategy,
                        numObjectsEachWorker(dataSetObjectNumExamples), numWorkers);
        int nPartitions = splitData.partitions().size();
        if (collectTrainingStats && repartition != Repartition.Never)
            stats.logRepartitionEnd();

        FlatMapFunction<Iterator<String>, ParameterAveragingTrainingResult> function;
        if (network != null)
            function = new ExecuteWorkerPathFlatMap<>(getWorkerInstance(network));
        else
            function = new ExecuteWorkerPathFlatMap<>(getWorkerInstance(graph));

        JavaRDD<ParameterAveragingTrainingResult> result = splitData.mapPartitions(function);
        processResults(network, graph, result, splitNum, numSplits);

        if (collectTrainingStats)
            stats.logMapPartitionsEnd(nPartitions);
    }

    protected void doIterationPathsMDS(SparkComputationGraph graph, JavaRDD<String> split, int splitNum, int numSplits,
                    int dataSetObjectNumExamples) {
        log.info("Starting training of split {} of {}. workerMiniBatchSize={}, averagingFreq={}, Configured for {} workers",
                        splitNum, numSplits, batchSizePerWorker, averagingFrequency, numWorkers);
        if (collectTrainingStats)
            stats.logMapPartitionsStart();

        JavaRDD<String> splitData = split;
        if (collectTrainingStats)
            stats.logRepartitionStart();
        splitData = SparkUtils.repartition(splitData, repartition, repartitionStrategy,
                        numObjectsEachWorker(dataSetObjectNumExamples), numWorkers);
        int nPartitions = splitData.partitions().size();
        if (collectTrainingStats && repartition != Repartition.Never)
            stats.logRepartitionEnd();


        FlatMapFunction<Iterator<String>, ParameterAveragingTrainingResult> function =
                        new ExecuteWorkerPathMDSFlatMap<>(getWorkerInstance(graph));

        JavaRDD<ParameterAveragingTrainingResult> result = splitData.mapPartitions(function);
        processResults(null, graph, result, splitNum, numSplits);

        if (collectTrainingStats)
            stats.logMapPartitionsEnd(nPartitions);
    }

    protected void doIteration(SparkComputationGraph graph, JavaRDD<MultiDataSet> split, int splitNum, int numSplits) {
        log.info("Starting training of split {} of {}. workerMiniBatchSize={}, averagingFreq={}, Configured for {} workers",
                        splitNum, numSplits, batchSizePerWorker, averagingFrequency, numWorkers);
        if (collectTrainingStats)
            stats.logMapPartitionsStart();

        JavaRDD<MultiDataSet> splitData = split;

        splitData = SparkUtils.repartition(splitData, repartition, repartitionStrategy,
                        numObjectsEachWorker(rddDataSetNumExamples), numWorkers);
        int nPartitions = split.partitions().size();

        FlatMapFunction<Iterator<MultiDataSet>, ParameterAveragingTrainingResult> function =
                        new ExecuteWorkerMultiDataSetFlatMap<>(getWorkerInstance(graph));
        JavaRDD<ParameterAveragingTrainingResult> result = splitData.mapPartitions(function);
        processResults(null, graph, result, splitNum, numSplits);

        if (collectTrainingStats)
            stats.logMapPartitionsEnd(nPartitions);
    }

    protected void doIterationPDS_MDS(SparkComputationGraph graph, JavaRDD<PortableDataStream> split, int splitNum,
                    int numSplits) {
        log.info("Starting training of split {} of {}. workerMiniBatchSize={}, averagingFreq={}, Configured for {} workers",
                        splitNum, numSplits, batchSizePerWorker, averagingFrequency, numWorkers);
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

        FlatMapFunction<Iterator<PortableDataStream>, ParameterAveragingTrainingResult> function =
                        new ExecuteWorkerPDSMDSFlatMap<>(getWorkerInstance(graph));

        JavaRDD<ParameterAveragingTrainingResult> result = splitData.mapPartitions(function);
        processResults(null, graph, result, splitNum, numSplits);

        if (collectTrainingStats)
            stats.logMapPartitionsEnd(nPartitions);
    }


    protected void processResults(SparkDl4jMultiLayer network, SparkComputationGraph graph,
                    JavaRDD<ParameterAveragingTrainingResult> results, int splitNum, int totalSplits) {
        //Need to do parameter averaging, and where necessary also do averaging of the updaters
        //Let's do all of this in ONE step, such that we don't have extra synchronization costs

        if (collectTrainingStats)
            stats.logAggregateStartTime();
        ParameterAveragingAggregationTuple tuple =
                        results.treeAggregate(null, new ParameterAveragingElementAddFunction(),
                                        new ParameterAveragingElementCombineFunction(), this.aggregationDepth);
        INDArray params = tuple.getParametersSum();
        int aggCount = tuple.getAggregationsCount();
        SparkTrainingStats aggregatedStats = tuple.getSparkTrainingStats();
        if (collectTrainingStats)
            stats.logAggregationEndTime();


        if (collectTrainingStats)
            stats.logProcessParamsUpdaterStart();
        if (params != null) {
            params.divi(aggCount);
            INDArray updaterState = tuple.getUpdaterStateSum();
            if (updaterState != null)
                updaterState.divi(aggCount); //May be null if all SGD updaters, for example

            if (network != null) {
                MultiLayerNetwork net = network.getNetwork();
                net.setParameters(params);
                if (updaterState != null)
                    net.getUpdater().setStateViewArray(null, updaterState, false);

                network.setScore(tuple.getScoreSum() / tuple.getAggregationsCount());
            } else {
                ComputationGraph g = graph.getNetwork();
                g.setParams(params);
                if (updaterState != null)
                    g.getUpdater().setStateViewArray(updaterState);

                graph.setScore(tuple.getScoreSum() / tuple.getAggregationsCount());
            }
        } else {
            log.info("Skipping imbalanced split with no data for all executors");
        }



        if (collectTrainingStats) {
            stats.logProcessParamsUpdaterEnd();
            stats.addWorkerStats(aggregatedStats);
        }

        if (statsStorage != null) {
            Collection<StorageMetaData> meta = tuple.getListenerMetaData();
            if (meta != null && !meta.isEmpty()) {
                statsStorage.putStorageMetaData(meta);
            }

            Collection<Persistable> staticInfo = tuple.getListenerStaticInfo();
            if (staticInfo != null && !staticInfo.isEmpty()) {
                statsStorage.putStaticInfo(staticInfo);
            }

            Collection<Persistable> updates = tuple.getListenerUpdates();
            if (updates != null && !updates.isEmpty()) {
                statsStorage.putUpdate(updates);
            }
        }

        Nd4j.getExecutioner().commit();

        log.info("Completed training of split {} of {}", splitNum, totalSplits);

        if (params != null) {
            //Params may be null for edge case (empty RDD)
            if (network != null) {
                MultiLayerConfiguration conf = network.getNetwork().getLayerWiseConfigurations();
                int numUpdates = averagingFrequency;
                conf.setIterationCount(conf.getIterationCount() + numUpdates);
            } else {
                ComputationGraphConfiguration conf = graph.getNetwork().getConfiguration();
                int numUpdates = averagingFrequency;
                conf.setIterationCount(conf.getIterationCount() + numUpdates);
            }
        }
    }



    protected StatsStorageRouterProvider getRouterProvider() {
        if (statsStorage == null)
            return null; //Not needed
        return new VanillaStatsStorageRouterProvider();
    }


    public static class Builder {
        protected boolean saveUpdater;
        protected Integer numWorkers;
        protected int rddDataSetNumExamples;
        protected int batchSizePerWorker = 16;
        protected int averagingFrequency = 5;
        protected int aggregationDepth = 2;
        protected int prefetchNumBatches = 0;
        protected Repartition repartition = Repartition.Always;
        protected RepartitionStrategy repartitionStrategy = RepartitionStrategy.Balanced;
        protected StorageLevel storageLevel = StorageLevel.MEMORY_ONLY_SER();
        protected StorageLevel storageLevelStreams = StorageLevel.MEMORY_ONLY();
        protected RDDTrainingApproach rddTrainingApproach = RDDTrainingApproach.Export;
        protected String exportDirectory = null;
        protected Long rngSeed;
        protected Collection<TrainingHook> trainingHooks;


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
            checkArgument(numWorkers == null || numWorkers > 0,
                            "Invalid number of workers: " + numWorkers + " (must be >= 1)");
            checkArgument(rddDataSetNumExamples > 0,
                            "Invalid rdd data set size: " + rddDataSetNumExamples + " (must be >= 1)");
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
            checkArgument(averagingFrequency > 0, "Invalid input: averaging frequency must be >= 1");
            this.averagingFrequency = averagingFrequency;
            return this;
        }

        /**
         * The number of levels in the aggregation tree for parameter synchronization. (default: 2)
         * <b>Note</b>: For large models trained with many partitions, increasing this number
         * will reduce the load on the driver and help prevent it from becoming a bottleneck.<br>
         *
         * @param aggregationDepth RDD tree aggregation channels when averaging parameter updates.
         */
        public Builder aggregationDepth(int aggregationDepth) {
            checkArgument(aggregationDepth > 0, "Invalid input: tree aggregation channels must be >= 1");
            this.aggregationDepth = aggregationDepth;
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
