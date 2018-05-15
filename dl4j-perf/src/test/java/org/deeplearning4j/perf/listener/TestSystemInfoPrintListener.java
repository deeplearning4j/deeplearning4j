package org.deeplearning4j.perf.listener;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;

import static org.junit.Assert.assertEquals;

public class TestSystemInfoPrintListener {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void testListener() throws Exception {
        SystemInfoPrintListener systemInfoPrintListener = SystemInfoPrintListener.builder()
                .printOnEpochStart(true).printOnEpochEnd(true)
                .build();

        File tmpFile = testDir.newFile("tmpfile-log.txt");
        assertEquals(0, tmpFile.length() );

        SystemInfoFilePrintListener systemInfoFilePrintListener = SystemInfoFilePrintListener.builder()
                .printOnEpochStart(true).printOnEpochEnd(true).printFileTarget(tmpFile)
                .build();
        tmpFile.deleteOnExit();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new OutputLayer.Builder().nIn(4).nOut(3).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(systemInfoFilePrintListener);

        DataSetIterator iter = new IrisDataSetIterator(10, 150);

        net.fit(iter, 3);

        System.out.println(FileUtils.readFileToString(tmpFile));

    }

}
