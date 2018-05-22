package org.deeplearning4j.ui.stats;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

/**
 * Created by Alex on 07/04/2017.
 */
public class TestTransferStatsCollection {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void test() throws IOException {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new DenseLayer.Builder().nIn(10).nOut(10).build())
                        .layer(1, new OutputLayer.Builder().nIn(10).nOut(10).build()).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();


        MultiLayerNetwork net2 =
                        new TransferLearning.Builder(net)
                                        .fineTuneConfiguration(
                                                        new FineTuneConfiguration.Builder().updater(new Sgd(0.01)).build())
                                        .setFeatureExtractor(0).build();

        File f = testDir.newFile("dl4jTestTransferStatsCollection.bin");
        f.delete();
        net2.setListeners(new StatsListener(new FileStatsStorage(f)));

        //Previosuly: failed on frozen layers
        net2.fit(new DataSet(Nd4j.rand(8, 10), Nd4j.rand(8, 10)));

        f.deleteOnExit();
    }
}
