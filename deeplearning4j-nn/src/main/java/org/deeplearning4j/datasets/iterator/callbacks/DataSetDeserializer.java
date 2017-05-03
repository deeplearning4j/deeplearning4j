package org.deeplearning4j.datasets.iterator.callbacks;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class DataSetDeserializer implements FileCallback {
    private AtomicLong counter = new AtomicLong(0);

    @Override
    public <T> T call(File file) {
        long time1 = System.nanoTime();
        DataSet dataSet = new DataSet();
        dataSet.load(file);
        long time2 = System.nanoTime();

        if (counter.getAndIncrement() % 5 == 0)
            log.info("Real time: [{}] ns", time2 - time1);

        return (T) dataSet;
    }
}
