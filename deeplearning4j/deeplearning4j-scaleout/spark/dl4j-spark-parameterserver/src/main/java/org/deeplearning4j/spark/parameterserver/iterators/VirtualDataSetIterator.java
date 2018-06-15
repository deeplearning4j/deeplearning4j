package org.deeplearning4j.spark.parameterserver.iterators;

import lombok.NonNull;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * This DataSetIterator implementation does accumulation of DataSets from different Spark executors, wrt Thread/Device Affinity
 *
 *
 * @author raver119@gmail.com
 */
public class VirtualDataSetIterator implements DataSetIterator {

    /**
     * Basic idea here is simple: this DataSetIterator will take in multiple lazy Iterator<DataSet>,
     * and will push them is round-robin manner to ParallelWrapper workers
     */

    protected final List<Iterator<DataSet>> iterators;
    protected final AtomicInteger position;

    public VirtualDataSetIterator(@NonNull List<Iterator<DataSet>> iterators) {
        this.iterators = iterators;
        this.position = new AtomicInteger(0);
    }

    /*
    
    // TODO: to be implemented
    
    @Override
    public void attachThread(int producer) {
        throw new UnsupportedOperationException();
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
    public DataSet nextFor(int consumer) {
        return null;
    }
    
    @Override
    public DataSet nextFor() {
        return null;
    }
    
    */
    @Override
    public boolean resetSupported() {
        // we're NOT supporting reset() here
        return false;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {

    }

    @Override
    public boolean hasNext() {
        // just checking if that's not the last iterator, or if that's the last one - check if it has something
        return position.get() < iterators.size() - 1
                        || (position.get() < iterators.size() && iterators.get(position.get()).hasNext());
    }

    @Override
    public DataSet next() {
        // TODO: this solution isn't ideal, it assumes non-empty iterators all the time. Would be nice to do something here
        if (!iterators.get(position.get()).hasNext())
            position.getAndIncrement();

        return iterators.get(position.get()).next();
    }

    @Override
    public void remove() {
        // no-op
    }

    @Override
    public void reset() {
        throw new UnsupportedOperationException();
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        // we probably don't need this thing here
        return null;
    }

    @Override
    public int batch() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int inputColumns() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int totalOutcomes() {
        throw new UnsupportedOperationException();
    }

    @Override
    public DataSet next(int num) {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<String> getLabels() {
        return null;
    }
}
