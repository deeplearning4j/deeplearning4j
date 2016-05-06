package org.deeplearning4j.datasets.iterator;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.records.reader.impl.FileRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

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

}
