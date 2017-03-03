package org.deeplearning4j.parallelism.main;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;

/**
 * Created by agibsonccc on 12/29/16.
 */
public class MnistDataSetIteratorProviderFactory implements DataSetIteratorProviderFactory {
    /**
     * Create an {@link DataSetIterator}
     *
     * @return
     */
    @Override
    public DataSetIterator create() {
        try {
            return new MnistDataSetIterator(100, 1000);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
