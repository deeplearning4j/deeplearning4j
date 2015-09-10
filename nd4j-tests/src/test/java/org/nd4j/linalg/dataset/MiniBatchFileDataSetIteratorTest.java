package org.nd4j.linalg.dataset;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import static org.junit.Assert.*;

/**
 * Created by agibsonccc on 9/10/15.
 */
public class MiniBatchFileDataSetIteratorTest extends BaseNd4jTest {
    @Test
    public void testMiniBatches() throws Exception {
        DataSet load = new IrisDataSetIterator(150,150).next();
        DataSetIterator iter = new MiniBatchFileDataSetIterator(load,10);
        while(iter.hasNext())
            assertEquals(10,iter.next().numExamples());
    }

    @Override
    public char ordering() {
        return 'f';
    }
}

