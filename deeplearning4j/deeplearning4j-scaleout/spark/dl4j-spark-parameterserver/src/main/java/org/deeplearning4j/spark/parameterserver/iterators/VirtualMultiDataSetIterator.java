package org.deeplearning4j.spark.parameterserver.iterators;

import lombok.NonNull;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.ParallelMultiDataSetIterator;

import java.util.Iterator;
import java.util.List;

/**
 * This MultiDataSetIterator implementation does accumulation of MultiDataSets from different Spark executors, wrt Thread/Device Affinity
 *
 * @author raver119@gmail.com
 */
public class VirtualMultiDataSetIterator implements ParallelMultiDataSetIterator {

    protected final List<Iterator<MultiDataSet>> iterators;

    public VirtualMultiDataSetIterator(@NonNull List<Iterator<MultiDataSet>> iterators) {
        this.iterators = iterators;
    }

    @Override
    public MultiDataSet next(int num) {
        return null;
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {

    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public boolean resetSupported() {
        return false;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {

    }

    @Override
    public boolean hasNext() {
        return false;
    }

    @Override
    public MultiDataSet next() {
        return null;
    }

    @Override
    public void remove() {

    }

    @Override
    public void attachThread(int producer) {

    }

    @Override
    public boolean hasNextFor() {
        return false;
    }

    @Override
    public boolean hasNextFor(int consumer) {
        return false;
    }

    @Override
    public MultiDataSet nextFor(int consumer) {
        return null;
    }

    @Override
    public MultiDataSet nextFor() {
        return null;
    }
}
