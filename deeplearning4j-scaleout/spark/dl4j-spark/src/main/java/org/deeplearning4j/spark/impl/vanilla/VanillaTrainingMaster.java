package org.deeplearning4j.spark.impl.vanilla;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.api.worker.NetBroadcastTuple;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.dataset.api.DataSet;

/**
 * Created by Alex on 14/06/2016.
 */
public class VanillaTrainingMaster implements TrainingMaster<VanillaTrainingResult, VanillaTrainingWorker> {
    @Override
    public VanillaTrainingWorker getWorkerInstance(SparkDl4jMultiLayer network) {
        NetBroadcastTuple tuple = new NetBroadcastTuple(network.getNetwork().getLayerWiseConfigurations(),
                network.getNetwork().params(),
                network.getNetwork().getUpdater());

        Broadcast<NetBroadcastTuple> broadcast = network.getSparkContext().broadcast(tuple);
        return new VanillaTrainingWorker(broadcast);
    }

    @Override
    public JavaRDD<VanillaTrainingResult> executeTraining(SparkDl4jMultiLayer network, JavaRDD<DataSet> trainingData) {
        return null;
    }

    @Override
    public void processResults(SparkDl4jMultiLayer network, JavaRDD<VanillaTrainingResult> results) {

    }
}
