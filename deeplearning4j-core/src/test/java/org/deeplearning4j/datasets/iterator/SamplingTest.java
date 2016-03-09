package org.deeplearning4j.datasets.iterator;

import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;

/**
 * @author Adam Gibson
 */
public class SamplingTest {

    @Test
    public void testSample() throws Exception {
        DataSetIterator iter = new TestMnistIterator();
        //batch size and total
        DataSetIterator sampling = new SamplingDataSetIterator(iter.next(),10,10);
        assertEquals(sampling.next().numExamples(),10);
    }

}
