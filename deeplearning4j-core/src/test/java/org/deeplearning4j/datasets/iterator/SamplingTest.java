package org.deeplearning4j.datasets.iterator;

import static org.junit.Assert.*;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.junit.Test;

/**
 * @author Adam Gibson
 */
public class SamplingTest {

    @Test
    public void testSample() throws Exception {
        DataSetIterator iter = new MnistDataSetIterator(10,10);
        //batch size and total
        DataSetIterator sampling = new SamplingDataSetIterator(iter.next(),10,10);
        assertEquals(sampling.next().numExamples(),10);
    }

}
