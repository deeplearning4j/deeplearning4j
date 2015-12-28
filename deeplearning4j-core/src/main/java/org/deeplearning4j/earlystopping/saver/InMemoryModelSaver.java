package org.deeplearning4j.earlystopping.saver;

import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.io.IOException;

/** Save the best (and latest) models for early stopping training to memory for later retrieval */
public class InMemoryModelSaver implements EarlyStoppingModelSaver {

    private transient MultiLayerNetwork bestModel;
    private transient MultiLayerNetwork latestModel;

    @Override
    public void saveBestModel(MultiLayerNetwork net, double score) throws IOException {
        bestModel = net.clone();
    }

    @Override
    public void saveLatestModel(MultiLayerNetwork net, double score) throws IOException {
        latestModel = net.clone();
    }

    @Override
    public MultiLayerNetwork getBestModel() throws IOException {
        return bestModel;
    }

    @Override
    public MultiLayerNetwork getLatestModel() throws IOException {
        return latestModel;
    }

    @Override
    public String toString(){
        return "InMemoryModelSaver()";
    }
}
