package org.deeplearning4j.datasets.iterator.callbacks;


import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;

/**
 * @author raver119@gmail.com
 */
public interface DataSetCallback {

    void call(DataSet dataSet);

    void call(MultiDataSet multiDataSet);

    void reset();
}
