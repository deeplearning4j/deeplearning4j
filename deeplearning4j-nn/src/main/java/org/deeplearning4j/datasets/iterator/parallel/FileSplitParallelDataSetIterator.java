package org.deeplearning4j.datasets.iterator.parallel;

import lombok.NonNull;
import org.deeplearning4j.datasets.iterator.callbacks.FileCallback;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;

/**
 * @author raver119@gmail.com
 */
public class FileSplitParallelDataSetIterator extends BaseParallelDataSetIterator {


    public FileSplitParallelDataSetIterator(@NonNull File rootFolder, @NonNull FileCallback callback) {
        this(rootFolder, callback, Nd4j.getAffinityManager().getNumberOfDevices());
    }

    public FileSplitParallelDataSetIterator(@NonNull File rootFolder, @NonNull FileCallback callback, int numThreads) {
        super(numThreads);


    }

    @Override
    public boolean hasNextFor(int consumer) {
        return false;
    }

    @Override
    public DataSet nextFor(int consumer) {
        return null;
    }

    @Override
    protected void reset(int consumer) {

    }
}
