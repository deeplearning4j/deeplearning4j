package org.deeplearning4j.scaleout.job;

import org.deeplearning4j.datasets.iterator.DataSetIterator;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class DataSetIteratorJobIterator implements JobIterator {
    protected DataSetIterator iter;

    public DataSetIteratorJobIterator(DataSetIterator iter) {
        this.iter = iter;
    }

    @Override
    public Job next(String workerId) {
        return new Job(iter.next(),workerId);
    }

    @Override
    public Job next() {
        return new Job(iter.next(),"");
    }

    @Override
    public boolean hasNext() {
        return iter.hasNext();
    }

    @Override
    public void reset() {
        iter.reset();
    }
}
