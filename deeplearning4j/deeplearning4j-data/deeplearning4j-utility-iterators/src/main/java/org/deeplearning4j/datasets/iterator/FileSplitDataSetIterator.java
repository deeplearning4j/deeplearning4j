package org.deeplearning4j.datasets.iterator;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.callbacks.FileCallback;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Simple iterator working with list of files.
 * File -> DataSet conversion will be handled via provided FileCallback implementation
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class FileSplitDataSetIterator implements DataSetIterator {
    private DataSetPreProcessor preProcessor;

    private List<File> files;
    private int numFiles;
    private AtomicInteger counter = new AtomicInteger(0);
    private FileCallback callback;

    public FileSplitDataSetIterator(@NonNull List<File> files, @NonNull FileCallback callback) {
        this.files = files;
        this.numFiles = files.size();
        this.callback = callback;
    }


    @Override
    public DataSet next(int num) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int inputColumns() {
        return 0;
    }

    @Override
    public int totalOutcomes() {
        return 0;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        counter.set(0);
    }

    @Override
    public int batch() {
        return 0;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return preProcessor;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public boolean hasNext() {
        return counter.get() < numFiles;
    }

    @Override
    public DataSet next() {
        //        long time1 = System.nanoTime();
        DataSet ds = callback.call(files.get(counter.getAndIncrement()));

        if (preProcessor != null && ds != null)
            preProcessor.preProcess(ds);

        //        long time2 = System.nanoTime();

        //        if (counter.get() % 5 == 0)
        //            log.info("Device: [{}]; Time: [{}] ns;", Nd4j.getAffinityManager().getDeviceForCurrentThread(), time2 - time1);

        return ds;
    }

    @Override
    public void remove() {
        // no-op
    }
}
