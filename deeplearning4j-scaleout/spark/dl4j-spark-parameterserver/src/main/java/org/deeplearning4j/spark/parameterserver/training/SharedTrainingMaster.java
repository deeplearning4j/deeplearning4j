package org.deeplearning4j.spark.parameterserver.training;

import lombok.NonNull;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.input.PortableDataStream;
import org.apache.spark.storage.StorageLevel;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.spark.api.Repartition;
import org.deeplearning4j.spark.api.RepartitionStrategy;
import org.deeplearning4j.spark.api.TrainingHook;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
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
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
public class SharedTrainingMaster implements TrainingMaster<SharedTrainingResult, SharedTrainingWorker> {
    protected List<TrainingHook> trainingHooks;
    protected VoidConfiguration voidConfiguration;

    private static ObjectMapper jsonMapper;
    private static ObjectMapper yamlMapper;


    public SharedTrainingMaster(@NonNull VoidConfiguration voidConfiguration) {
        this.voidConfiguration = voidConfiguration;
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
        return null;
    }

    @Override
    public SharedTrainingWorker getWorkerInstance(SparkComputationGraph graph) {
        return null;
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
    }


    public static class Builder {
        protected double threshold;
        private Repartition repartition = Repartition.Always;
        private RepartitionStrategy repartitionStrategy = RepartitionStrategy.Balanced;
        private StorageLevel storageLevel = StorageLevel.MEMORY_ONLY_SER();
        private StorageLevel storageLevelStreams = StorageLevel.MEMORY_ONLY();
        private VoidConfiguration voidConfiguration;

        public Builder(int rddDataSetNumExamples) {
            this(1e-3, rddDataSetNumExamples);
        }

        public Builder(@NonNull VoidConfiguration voidConfiguration, int rddDataSetNumExamples) {
            this(voidConfiguration,1e-3, rddDataSetNumExamples);
        }

        public Builder(double threshold, int rddDataSetNumExamples) {
            // TODO: we need proper default for VoidConfiguration here
            this(new VoidConfiguration(), null, threshold, rddDataSetNumExamples);
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

        public SharedTrainingMaster build() {


            return null;
        }
    }
}
