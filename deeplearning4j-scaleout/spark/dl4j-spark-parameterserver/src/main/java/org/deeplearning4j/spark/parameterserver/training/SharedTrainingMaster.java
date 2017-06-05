package org.deeplearning4j.spark.parameterserver.training;

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
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.util.Collection;

/**
 * @author raver119@gmail.com
 */
public class SharedTrainingMaster implements TrainingMaster<SharedTrainingResult, SharedTrainingWorker> {

    public SharedTrainingMaster() {

    }

    @Override
    public void removeHook(TrainingHook trainingHook) {

    }

    @Override
    public void addHook(TrainingHook trainingHook) {

    }

    @Override
    public String toJson() {
        return null;
    }

    @Override
    public String toYaml() {
        return null;
    }

    @Override
    public SharedTrainingWorker getWorkerInstance(SparkDl4jMultiLayer network) {
        return null;
    }

    @Override
    public SharedTrainingWorker getWorkerInstance(SparkComputationGraph graph) {
        return null;
    }

    @Override
    public void executeTraining(SparkDl4jMultiLayer network, JavaRDD<DataSet> trainingData) {

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

    }

    @Override
    public void setListeners(StatsStorageRouter router, Collection<IterationListener> listeners) {

    }

    @Override
    public boolean deleteTempFiles(JavaSparkContext sc) {
        return false;
    }

    @Override
    public boolean deleteTempFiles(SparkContext sc) {
        return false;
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

        public Builder(int rddDataSetNumExamples) {
            this(1e-3, rddDataSetNumExamples);
        }

        public Builder(double threshold, int rddDataSetNumExamples) {
            this(null, threshold, rddDataSetNumExamples);
        }

        /**
         *
         * @param numWorkers
         * @param threshold Update sharing threshold
         * @param rddDataSetNumExamples
         */
        public Builder(Integer numWorkers, double threshold, int rddDataSetNumExamples) {
            this.threshold = threshold;
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
