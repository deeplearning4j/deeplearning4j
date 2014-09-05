package org.deeplearning4j.datasets.iterator;

import org.nd4j.linalg.dataset.DataSet;
import org.junit.Test;
import org.springframework.core.io.ClassPathResource;

/**
 * CSV Test
 * @author Adam Gibson
 */
public class CSVDataSetIteratorTest {
    @Test
    public void testCSV() throws Exception {
       DataSetIterator iter = new CSVDataSetIterator(10,10,new ClassPathResource("csv-example.csv").getFile(),1,1);
        DataSet next = iter.next();
    }


}
