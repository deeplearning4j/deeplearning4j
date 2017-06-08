package org.deeplearning4j.datasets.iterator;

import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

/**
 * Builds an iterator that terminates once the number of minibatches returned with .next() is equal to a specified number
 * Note that this essentially restricts the data to this specified number of minibatches.
 */
public class EarlyTerminationMultiDataSetIterator implements MultiDataSetIterator {

    private MultiDataSetIterator underlyingIterator;
    private int terminationPoint;
    private int minibatchCount = 0;

    /**
     * Constructor takes the iterator to wrap and the number of minibatches after which the call to hasNext()
     * will return false
     * @param underlyingIterator, iterator to wrap
     * @param terminationPoint, minibatches after which hasNext() will return false
     */
    public EarlyTerminationMultiDataSetIterator(MultiDataSetIterator underlyingIterator, int terminationPoint) {
        this.underlyingIterator = underlyingIterator;
        this.terminationPoint = terminationPoint;
    }

    @Override
    public MultiDataSet next(int num) {
        minibatchCount += num;
        return underlyingIterator.next(num);
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        underlyingIterator.setPreProcessor(preProcessor);
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return underlyingIterator.getPreProcessor();
    }

    @Override
    public boolean resetSupported() {
        return underlyingIterator.resetSupported();
    }

    @Override
    public boolean asyncSupported() {
        return underlyingIterator.asyncSupported();
    }

    @Override
    public void reset() {
        underlyingIterator.reset();
    }

    @Override
    public boolean hasNext() {
        return underlyingIterator.hasNext() && minibatchCount < terminationPoint;
    }

    @Override
    public MultiDataSet next() {
        minibatchCount++;
        return underlyingIterator.next();
    }

    @Override
    public void remove() {
        underlyingIterator.remove();
    }
}
