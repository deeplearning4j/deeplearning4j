package org.deeplearning4j.datasets.iterator;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.List;

/**
 * Builds an iterator that terminates once the number of minibatches returned with .next() is equal to a specified number
 * Note that this essentially restricts the data to this specified number of minibatches.
 */
public class EarlyTerminationDataSetIterator implements DataSetIterator {

    private DataSetIterator underlyingIterator;
    private int terminationPoint;
    private int minibatchCount = 0;

    /**
     * Constructor takes the iterator to wrap and the number of minibatches after which the call to hasNext()
     * will return false
     * @param underlyingIterator, iterator to wrap
     * @param terminationPoint, minibatches after which hasNext() will return false
     */
    public EarlyTerminationDataSetIterator(DataSetIterator underlyingIterator, int terminationPoint) {
        this.underlyingIterator = underlyingIterator;
        this.terminationPoint = terminationPoint;
    }

    @Override
    public DataSet next(int num) {
        minibatchCount += num;
        return underlyingIterator.next(num);
    }

    @Override
    public int totalExamples() {
        return underlyingIterator.totalExamples();
    }

    @Override
    public int inputColumns() {
        return underlyingIterator.inputColumns();
    }

    @Override
    public int totalOutcomes() {
        return underlyingIterator.totalOutcomes();
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
    public int batch() {
        return underlyingIterator.batch();
    }

    @Override
    public int cursor() {
        return underlyingIterator.cursor();
    }

    @Override
    public int numExamples() {
        return underlyingIterator.numExamples();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        underlyingIterator.setPreProcessor(preProcessor);
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return underlyingIterator.getPreProcessor();
    }

    @Override
    public List<String> getLabels() {
        return underlyingIterator.getLabels();
    }

    @Override
    public boolean hasNext() {
        return underlyingIterator.hasNext() && minibatchCount < terminationPoint;
    }

    @Override
    public DataSet next() {
        minibatchCount++;
        return underlyingIterator.next();
    }

    @Override
    public void remove() {
        underlyingIterator.remove();
    }
}
