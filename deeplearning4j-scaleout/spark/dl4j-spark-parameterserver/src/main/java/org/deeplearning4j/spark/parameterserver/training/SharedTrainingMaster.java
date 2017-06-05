package org.deeplearning4j.spark.parameterserver.training;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.input.PortableDataStream;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.spark.api.TrainingHook;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.util.Collection;

/**
 * @author raver119@gmail.com
 */
public class SharedTrainingMaster implements TrainingMaster<SharedTrainingResult, SharedTrainingWorker> {
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
}
