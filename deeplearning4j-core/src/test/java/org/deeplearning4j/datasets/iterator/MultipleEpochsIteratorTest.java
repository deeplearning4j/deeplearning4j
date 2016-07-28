package org.deeplearning4j.datasets.iterator;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;

import static org.junit.Assert.*;


public class MultipleEpochsIteratorTest {

    @Test
    public void testNextAndReset() throws Exception{
        int epochs = 3;

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
        DataSetIterator iter = new RecordReaderDataSetIterator(rr, 150);
        MultipleEpochsIterator multiIter = new MultipleEpochsIterator(epochs, iter);

        assertTrue(multiIter.hasNext());
        while(multiIter.hasNext()){
            DataSet path = multiIter.next();
            assertFalse(path == null);
        }
        assertEquals(epochs, multiIter.epochs);
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
        while (multiIter.hasNext()) {
            DataSet path = multiIter.next();
            assertEquals(path.numExamples(), 50, 0.0);
            assertFalse(path == null);
        }
        assertEquals(epochs, multiIter.epochs);
    }

    @Test
    public void testLoadBatchDataSet() throws Exception{
        int epochs = 2;

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
        DataSetIterator iter = new RecordReaderDataSetIterator(rr, 150);
        DataSet ds = iter.next(20);
        MultipleEpochsIterator multiIter = new MultipleEpochsIterator(epochs, ds);

        while(multiIter.hasNext()){
            DataSet path = multiIter.next(10);
            assertEquals(path.numExamples(), 10, 0.0);
            assertFalse(path == null);
        }

        assertEquals(epochs, multiIter.epochs);
    }

    @Test
    public void testCifarDataSetIteratorReset() {
        int epochs = 2;
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
        net.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(1)));

        MultipleEpochsIterator ds = new MultipleEpochsIterator(epochs, new CifarDataSetIterator(10,20, new int[]{20,20,1}));
        net.fit(ds);
        assertEquals(epochs, ds.epochs);
        assertEquals(2, ds.batch);
    }
}
