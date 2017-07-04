package org.deeplearning4j.datasets.iterator;

import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

/**
 * Builds an iterator that terminates once the number of minibatches returned with .next() is equal to a specified number
 * Note that a call to .next(num) is counted as a call to return a minibatch regardless of the value of num
 * This essentially restricts the data to this specified number of minibatches.
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
        if (terminationPoint <= 0)
            throw new IllegalArgumentException(
                            "Termination point (the number of calls to .next() or .next(num)) has to be > 0");
        this.underlyingIterator = underlyingIterator;
        this.terminationPoint = terminationPoint;
    }

    @Override
    public MultiDataSet next(int num) {
        if (minibatchCount < terminationPoint) {
            minibatchCount++;
            return underlyingIterator.next(num);
        } else {
            throw new RuntimeException("Calls to next have exceeded termination point.");
        }
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
        minibatchCount = 0;
        underlyingIterator.reset();
    }

    @Override
    public boolean hasNext() {
        return underlyingIterator.hasNext() && minibatchCount < terminationPoint;
    }

    @Override
    public MultiDataSet next() {
        if (minibatchCount < terminationPoint) {
            minibatchCount++;
            return underlyingIterator.next();
        } else {
            throw new RuntimeException("Calls to next have exceeded the allotted number of minibatches.");
        }
    }

    @Override
    public void remove() {
        underlyingIterator.remove();
    }
}
