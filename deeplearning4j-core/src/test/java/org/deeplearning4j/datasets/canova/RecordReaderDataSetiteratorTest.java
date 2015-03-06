package org.deeplearning4j.datasets.canova;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.springframework.core.io.ClassPathResource;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 3/6/15.
 */
public class RecordReaderDataSetiteratorTest {

    @Test
    public void testRecordReader() throws Exception {
        RecordReader recordReader = new CSVRecordReader();
        FileSplit csv = new FileSplit(new ClassPathResource("csv-example.csv").getFile());
        recordReader.initialize(csv);
        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader,34);
        DataSet next = iter.next();
        assertEquals(34,next.numExamples());


    }

}
