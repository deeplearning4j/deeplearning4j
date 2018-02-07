package org.deeplearning4j.base;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.MatchCondition;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.io.File;

import static org.junit.Assert.assertEquals;

/**
 * @author Justin Long (crockpotveggies)
 */
public class MnistFetcherTest extends BaseDL4JTest {

    @Test
    public void testMnist() throws Exception {
        DataSetIterator iter = new MnistDataSetIterator(32, 60000, false, true, false, -1);
        int count = 0;
        while(iter.hasNext()){
            DataSet ds = iter.next();
            INDArray arr = ds.getFeatures().sum(1);
            int countMatch = Nd4j.getExecutioner().execAndReturn(new MatchCondition(arr, Conditions.equals(0))).z().getInt(0);
            assertEquals(0, countMatch);
            count++;
        }
        assertEquals(60000/32, count);

        count = 0;
        iter = new MnistDataSetIterator(32, false, 12345);
        while(iter.hasNext()){
            DataSet ds = iter.next();
            INDArray arr = ds.getFeatures().sum(1);
            int countMatch = Nd4j.getExecutioner().execAndReturn(new MatchCondition(arr, Conditions.equals(0))).z().getInt(0);
            assertEquals(0, countMatch);
            count++;
        }
        assertEquals((int)Math.ceil(10000/32.0), count);
    }

    @Test
    public void testMnistDataFetcher() throws Exception {
        MnistFetcher mnistFetcher = new MnistFetcher();
        File mnistDir = mnistFetcher.downloadAndUntar();

        assert (mnistDir.isDirectory());
    }
}
