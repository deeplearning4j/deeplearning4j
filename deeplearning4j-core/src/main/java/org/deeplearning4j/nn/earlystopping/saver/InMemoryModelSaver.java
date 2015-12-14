package org.deeplearning4j.nn.earlystopping.saver;

import org.deeplearning4j.nn.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.io.IOException;

public class InMemoryModelSaver implements EarlyStoppingModelSaver {

    private MultiLayerNetwork bestModel;
    private MultiLayerNetwork latestModel;

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
}
