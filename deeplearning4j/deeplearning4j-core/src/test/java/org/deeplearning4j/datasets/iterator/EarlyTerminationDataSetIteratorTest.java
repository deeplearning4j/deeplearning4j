package org.deeplearning4j.datasets.iterator;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Created by susaneraly on 6/8/17.
 */
public class EarlyTerminationDataSetIteratorTest extends BaseDL4JTest {

    int minibatchSize = 10;
    int numExamples = 105;
    @Rule
    public final ExpectedException exception = ExpectedException.none();

    @Test
    public void testNextAndReset() throws Exception {

        int terminateAfter = 2;

        DataSetIterator iter = new MnistDataSetIterator(minibatchSize, numExamples);
        EarlyTerminationDataSetIterator earlyEndIter = new EarlyTerminationDataSetIterator(iter, terminateAfter);

        assertTrue(earlyEndIter.hasNext());
        int batchesSeen = 0;
        List<DataSet> seenData = new ArrayList<>();
        while (earlyEndIter.hasNext()) {
            DataSet path = earlyEndIter.next();
            assertFalse(path == null);
            seenData.add(path);
            batchesSeen++;
        }
        assertEquals(batchesSeen, terminateAfter);

        //check data is repeated after reset
        earlyEndIter.reset();
        batchesSeen = 0;
        while (earlyEndIter.hasNext()) {
            DataSet path = earlyEndIter.next();
            assertEquals(seenData.get(batchesSeen).getFeatures(), path.getFeatures());
            assertEquals(seenData.get(batchesSeen).getLabels(), path.getLabels());
            batchesSeen++;
        }
    }

    @Test
    public void testNextNum() throws IOException {
        int terminateAfter = 1;

        DataSetIterator iter = new MnistDataSetIterator(minibatchSize, numExamples);
        EarlyTerminationDataSetIterator earlyEndIter = new EarlyTerminationDataSetIterator(iter, terminateAfter);

        earlyEndIter.next(10);
        assertEquals(earlyEndIter.hasNext(), false);

        earlyEndIter.reset();
        assertEquals(earlyEndIter.hasNext(), true);

    }

    @Test
    public void testCallstoNextNotAllowed() throws IOException {
        int terminateAfter = 1;

        DataSetIterator iter = new MnistDataSetIterator(minibatchSize, numExamples);
        EarlyTerminationDataSetIterator earlyEndIter = new EarlyTerminationDataSetIterator(iter, terminateAfter);

        earlyEndIter.next(10);
        iter.reset();
        exception.expect(RuntimeException.class);
        earlyEndIter.next(10);
    }
}
