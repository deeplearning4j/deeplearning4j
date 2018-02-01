package org.deeplearning4j.datasets.iterator;

import org.apache.commons.lang3.RandomUtils;
import org.deeplearning4j.BaseDL4JTest;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.primitives.Pair;

import java.util.Iterator;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by raver on 16.06.2016.
 */
public class AbstractDataSetIteratorTest extends BaseDL4JTest {
    @Test
    public void next() throws Exception {
        int numFeatures = 128;
        int batchSize = 10;
        int numRows = 1000;
        AtomicInteger cnt = new AtomicInteger(0);
        FloatsDataSetIterator iterator = new FloatsDataSetIterator(floatIterable(numRows, numFeatures), batchSize);

        assertTrue(iterator.hasNext());

        while (iterator.hasNext()) {
            DataSet dataSet = iterator.next();

            INDArray features = dataSet.getFeatures();

            assertEquals(batchSize, features.rows());
            assertEquals(numFeatures, features.columns());
            cnt.incrementAndGet();
        }

        assertEquals(numRows / batchSize, cnt.get());
    }


    protected static Iterable<Pair<float[], float[]>> floatIterable(final int totalRows, final int numColumns) {
        return new Iterable<Pair<float[], float[]>>() {
            @Override
            public Iterator<Pair<float[], float[]>> iterator() {
                return new Iterator<Pair<float[], float[]>>() {
                    private AtomicInteger cnt = new AtomicInteger(0);

                    @Override
                    public boolean hasNext() {
                        return cnt.incrementAndGet() <= totalRows;
                    }

                    @Override
                    public Pair<float[], float[]> next() {
                        float features[] = new float[numColumns];
                        float labels[] = new float[numColumns];
                        for (int i = 0; i < numColumns; i++) {
                            features[i] = (float) i;
                            labels[i] = RandomUtils.nextFloat(0, 5);
                        }
                        return Pair.makePair(features, labels);
                    }

                    @Override
                    public void remove() {
                        // no-op
                    }
                };
            }
        };
    }
}
