package org.deeplearning4j.datasets.iterator;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by susaneraly on 6/8/17.
 */
public class EarlyTerminationMultiDataSetIteratorTest {

    @Test
    public void testNextAndReset() throws Exception {
        int minibatchSize = 5;
        int numExamples = 20;
        int terminateAfter = 2;

        int count = 0;
        List<org.nd4j.linalg.dataset.api.MultiDataSet> someMDS = new ArrayList<>();
        while (count < numExamples) {
            INDArray[] feat = new INDArray[1];
            feat[0] = Nd4j.rand(1, 7);
            INDArray[] labs = new INDArray[2];
            labs[0] = Nd4j.rand(1, 2);
            labs[1] = Nd4j.rand(1, 2);
            someMDS.add(new MultiDataSet(feat,labs));
        }

        MultiDataSetIterator iter = new IteratorMultiDataSetIterator(someMDS.iterator(),minibatchSize);
        EarlyTerminationMultiDataSetIterator earlyEndIter = new EarlyTerminationMultiDataSetIterator(iter,terminateAfter);
        /*
        DataSetIterator iter = new MnistDataSetIterator(minibatchSize,numExamples);
        EarlyTerminationDataSetIterator earlyEndIter = new EarlyTerminationDataSetIterator(iter,terminateAfter);

        assertTrue(earlyEndIter.hasNext());
        int batchesSeen = 0;
        List<DataSet> seenData = new ArrayList<>();
        while (earlyEndIter.hasNext()) {
            DataSet path = earlyEndIter.next();
            assertFalse(path == null);
            seenData.add(path);
            batchesSeen++;
        }
        assertEquals(batchesSeen,terminateAfter);

        earlyEndIter.reset();
        batchesSeen = 0;
        while (earlyEndIter.hasNext()) {
            DataSet path = earlyEndIter.next();
            assertEquals(seenData.get(batchesSeen).getFeatures(),path.getFeatures());
            assertEquals(seenData.get(batchesSeen).getLabels(),path.getLabels());
            batchesSeen++;
        }
        */
    }

    @Test
    public void testNextNum() throws IOException {
        /*
        int minibatchSize = 10;
        int numExamples = 105;
        int terminateAfter = 2;

        DataSetIterator iter = new MnistDataSetIterator(minibatchSize,numExamples);
        EarlyTerminationDataSetIterator earlyEndIter = new EarlyTerminationDataSetIterator(iter,terminateAfter);

        earlyEndIter.next(10);
        earlyEndIter.next(10);
        assertEquals(earlyEndIter.hasNext(),false);

        earlyEndIter.reset();
        assertEquals(earlyEndIter.hasNext(),true);
        */
    }
}
