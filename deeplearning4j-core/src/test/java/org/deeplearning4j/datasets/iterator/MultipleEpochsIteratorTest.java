package org.deeplearning4j.datasets.iterator;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.records.reader.impl.FileRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;


public class MultipleEpochsIteratorTest {

    @Test
    public void testNextAndReset() throws Exception{
        int epochs = 3;

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
        DataSetIterator iter = new RecordReaderDataSetIterator(rr, 150);
        MultipleEpochsIterator multiIter = new MultipleEpochsIterator(epochs, iter);

        assertTrue(multiIter.hasNext());
        int actualEpochs = 0;
        while(multiIter.hasNext()){
            DataSet path = multiIter.next();
            assertFalse(path == null);
            actualEpochs++;
        }
        assertEquals(epochs, actualEpochs, 0.0);
    }

    @Test
    public void testLoadFullDataSet() throws Exception {
        int epochs = 3;

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
        DataSetIterator iter = new RecordReaderDataSetIterator(rr, 150);
        DataSet ds = iter.next(50);
        MultipleEpochsIterator multiIter = new MultipleEpochsIterator(epochs, ds);

        assertTrue(multiIter.hasNext());

        int actualEpochs = 0;
        while (multiIter.hasNext()) {
            DataSet path = multiIter.next();
            assertEquals(path.numExamples(), 50, 0.0);
            assertFalse(path == null);
            actualEpochs++;
        }
        assertEquals(epochs, actualEpochs, 0.0);
    }

    @Test
    public void testLoadBatchDataSet() throws Exception{
        int epochs = 2;

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
        DataSetIterator iter = new RecordReaderDataSetIterator(rr, 150);
        DataSet ds = iter.next(20);
        MultipleEpochsIterator multiIter = new MultipleEpochsIterator(epochs, ds);

        int actualTotalPasses = 0;
        while(multiIter.hasNext()){
            DataSet path = multiIter.next(10);
            assertEquals(path.numExamples(), 10, 0.0);
            assertFalse(path == null);
            actualTotalPasses++;
        }

        assertEquals(epochs*2, actualTotalPasses, 0.0);
    }

    @Test
    public void testCifarDataSetIteratorReset() {
        int epochs = 3;
        Nd4j.getRandom().setSeed(12345);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .regularization(false)
                .learningRate(1.0)
                .weightInit(WeightInit.XAVIER)
                .seed(12345L)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(400).nOut(50).activation("relu").build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation("softmax").nIn(50).nOut(10).build())
                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        MultipleEpochsIterator ds = new MultipleEpochsIterator(epochs, new CifarDataSetIterator(10,10, new int[]{20,20,1}));
        net.fit(ds);
    }
}
