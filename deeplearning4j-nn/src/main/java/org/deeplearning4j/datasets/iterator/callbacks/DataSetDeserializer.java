package org.deeplearning4j.datasets.iterator.callbacks;

import org.nd4j.linalg.dataset.DataSet;

import java.io.File;

/**
 * @author raver119@gmail.com
 */
public class DataSetDeserializer implements FileCallback {

    @Override
    public <T> T call(File file) {
        DataSet dataSet = new DataSet();
        dataSet.load(file);
        return (T) dataSet;
    }
}
