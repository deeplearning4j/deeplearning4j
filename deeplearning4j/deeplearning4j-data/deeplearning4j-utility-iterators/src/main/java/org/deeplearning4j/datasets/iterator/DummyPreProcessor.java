package org.deeplearning4j.datasets.iterator;

import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

/**
 * This is special dummy preProcessor, that does nothing.
 *
 * @author raver119@gmail.com
 */
public class DummyPreProcessor implements DataSetPreProcessor {
    /**
     * Pre process a dataset
     *
     * @param toPreProcess the data set to pre process
     */
    @Override
    public void preProcess(DataSet toPreProcess) {
        // no-op
    }
}
