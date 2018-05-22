package org.deeplearning4j.datasets.iterator;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.*;

/**
 *
 * @author Alex Black
 */
@Ignore
public class TestAsyncIterator extends BaseDL4JTest {

    @Test
    public void testBasic() {

        //Basic test. Make sure it returns the right number of elements,
        // hasNext() works, etc

        int size = 13;

        DataSetIterator baseIter = new TestIterator(size, 0);

        //async iterator with queue size of 1
        DataSetIterator async = new AsyncDataSetIterator(baseIter, 1);

        for (int i = 0; i < size; i++) {
            assertTrue(async.hasNext());
            DataSet ds = async.next();
            assertEquals(ds.getFeatureMatrix().getDouble(0), i, 0.0);
            assertEquals(ds.getLabels().getDouble(0), i, 0.0);
        }

        assertFalse(async.hasNext());
        async.reset();
        assertEquals(baseIter.cursor(), 0);
        assertTrue(async.hasNext());
        ((AsyncDataSetIterator) async).shutdown();

        //async iterator with queue size of 5
        baseIter = new TestIterator(size, 5);
        async = new AsyncDataSetIterator(baseIter, 5);

        for (int i = 0; i < size; i++) {
            assertTrue(async.hasNext());
            DataSet ds = async.next();
            assertEquals(ds.getFeatureMatrix().getDouble(0), i, 0.0);
            assertEquals(ds.getLabels().getDouble(0), i, 0.0);
        }
        assertFalse(async.hasNext());
        async.reset();
        assertEquals(baseIter.cursor(), 0);
        assertTrue(async.hasNext());
        ((AsyncDataSetIterator) async).shutdown();

        //async iterator with queue size of 100
        baseIter = new TestIterator(size, 100);
        async = new AsyncDataSetIterator(baseIter, 100);

        for (int i = 0; i < size; i++) {
            assertTrue(async.hasNext());
            DataSet ds = async.next();
            while (ds == null)
                ds = async.next();
            assertEquals(ds.getFeatureMatrix().getDouble(0), i, 0.0);
            assertEquals(ds.getLabels().getDouble(0), i, 0.0);
        }

        assertFalse(async.hasNext());
        async.reset();
        assertEquals(baseIter.cursor(), 0);
        assertTrue(async.hasNext());
        ((AsyncDataSetIterator) async).shutdown();

        //Test iteration where performance is limited by baseIterator.next() speed
        baseIter = new TestIterator(size, 1000);
        async = new AsyncDataSetIterator(baseIter, 5);
        for (int i = 0; i < size; i++) {
            assertTrue(async.hasNext());
            DataSet ds = async.next();
            assertEquals(ds.getFeatureMatrix().getDouble(0), i, 0.0);
            assertEquals(ds.getLabels().getDouble(0), i, 0.0);
        }
        assertFalse(async.hasNext());
        async.reset();
        assertEquals(baseIter.cursor(), 0);
        assertTrue(async.hasNext());
        ((AsyncDataSetIterator) async).shutdown();
    }

    @Test
    public void testInitializeNoNextIter() {

        DataSetIterator iter = new IrisDataSetIterator(10, 150);
        while (iter.hasNext())
            iter.next();

        DataSetIterator async = new AsyncDataSetIterator(iter, 2);

        assertFalse(iter.hasNext());
        assertFalse(async.hasNext());
        try {
            iter.next();
            fail("Should have thrown NoSuchElementException");
        } catch (Exception e) {
            //OK
        }

        async.reset();
        int count = 0;
        while (async.hasNext()) {
            async.next();
            count++;
        }
        assertEquals(150 / 10, count);
    }

    @Test
    public void testResetWhileBlocking() {
        int size = 6;
        //Test reset while blocking on baseIterator.next()
        DataSetIterator baseIter = new TestIterator(size, 1000);
        AsyncDataSetIterator async = new AsyncDataSetIterator(baseIter);
        async.next();
        //Should be waiting on baseIter.next()
        async.reset();
        for (int i = 0; i < 6; i++) {
            assertTrue(async.hasNext());
            DataSet ds = async.next();
            assertEquals(ds.getFeatureMatrix().getDouble(0), i, 0.0);
            assertEquals(ds.getLabels().getDouble(0), i, 0.0);
        }
        assertFalse(async.hasNext());
        async.shutdown();

        //Test reset while blocking on blockingQueue.put()
        baseIter = new TestIterator(size, 0);
        async = new AsyncDataSetIterator(baseIter);
        async.next();
        async.next();
        //Should be waiting on blocingQueue
        async.reset();
        for (int i = 0; i < 6; i++) {
            assertTrue(async.hasNext());
            DataSet ds = async.next();
            assertEquals(ds.getFeatureMatrix().getDouble(0), i, 0.0);
            assertEquals(ds.getLabels().getDouble(0), i, 0.0);
        }
        assertFalse(async.hasNext());
        async.shutdown();
    }


    private static class TestIterator implements DataSetIterator {

        private int size;
        private int cursor;
        private long delayMSOnNext;

        private TestIterator(int size, long delayMSOnNext) {
            this.size = size;
            this.cursor = 0;
            this.delayMSOnNext = delayMSOnNext;
        }

        @Override
        public DataSet next(int num) {
            throw new UnsupportedOperationException();
        }

        @Override
        public int totalExamples() {
            return size;
        }

        @Override
        public int inputColumns() {
            return 1;
        }

        @Override
        public int totalOutcomes() {
            return 1;
        }

        @Override
        public boolean resetSupported() {
            return true;
        }

        @Override
        public boolean asyncSupported() {
            return false;
        }

        @Override
        public void reset() {
            cursor = 0;
        }

        @Override
        public int batch() {
            return 1;
        }

        @Override
        public int cursor() {
            return cursor;
        }

        @Override
        public int numExamples() {
            return size;
        }

        @Override
        public void setPreProcessor(DataSetPreProcessor preProcessor) {
            throw new UnsupportedOperationException();
        }

        @Override
        public DataSetPreProcessor getPreProcessor() {
            throw new UnsupportedOperationException();
        }

        @Override
        public List<String> getLabels() {
            return null;
        }

        @Override
        public boolean hasNext() {
            return cursor < size;
        }

        @Override
        public DataSet next() {
            if (delayMSOnNext > 0) {
                try {
                    Thread.sleep(delayMSOnNext);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
            INDArray features = Nd4j.scalar(cursor);
            INDArray labels = Nd4j.scalar(cursor);
            cursor++;
            return new DataSet(features, labels);
        }

        @Override
        public void remove() {}
    }

}
