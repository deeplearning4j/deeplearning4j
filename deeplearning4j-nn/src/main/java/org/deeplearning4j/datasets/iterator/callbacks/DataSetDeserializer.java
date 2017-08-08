package org.deeplearning4j.datasets.iterator.callbacks;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;

/**
 * This callback does DataSet deserialization of a given file.
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class DataSetDeserializer implements FileCallback {
    @Override
    public <T> T call(File file) {
        DataSet dataSet = new DataSet();
        dataSet.load(file);
        return (T) dataSet;
    }
}
