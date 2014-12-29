package org.deeplearning4j.datasets.iterator;

import org.nd4j.linalg.dataset.DataSet;
import org.junit.Test;
import org.springframework.core.io.ClassPathResource;
import static org.junit.Assert.assertEquals;

/**
 * CSV Test
 * @author Adam Gibson
 */
public class CSVDataSetIteratorTest {
    @Test
    public void testCSV() throws Exception {
       DataSetIterator iter =
           new CSVDataSetIterator(10, 10, new ClassPathResource("csv-example.csv").getFile(), 1, 1);
       DataSet next = iter.next();
       assertEquals("Unexpected number of samples", 10, next.numExamples());
       assertEquals("Unexpected dimension of the samples", 479, next.numInputs());
    }
}
