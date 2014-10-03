package org.deeplearning4j.datasets.iterator;

import org.nd4j.linalg.dataset.api.DataSet;

/**
 * Pre process a dataset
 */
public interface DataSetPreProcessor {

    /**
     * Pre process a dataset
     * @param toPreProcess the data set to pre process
     */
    void preProcess(DataSet toPreProcess);


}
