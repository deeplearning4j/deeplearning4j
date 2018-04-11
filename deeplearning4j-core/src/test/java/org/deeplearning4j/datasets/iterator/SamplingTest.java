package org.deeplearning4j.datasets.iterator;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.junit.Test;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import static org.junit.Assert.assertEquals;

/**
 * @author Adam Gibson
 */
public class SamplingTest extends BaseDL4JTest {

    @Test
    public void testSample() throws Exception {
        DataSetIterator iter = new MnistDataSetIterator(10, 10);
        //batch size and total
        DataSetIterator sampling = new SamplingDataSetIterator(iter.next(), 10, 10);
        assertEquals(sampling.next().numExamples(), 10);
    }

}
